"""Spectrum dataset creation and dataloading utilities"""

import warnings
from collections.abc import Callable, Iterable
from os import PathLike

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Sequence, Value
from pyteomics.mgf import read
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from metabo_depthcharge.datasets._common import (
    hf_cache_file,
    hf_persist,
    hf_silent,
    hf_tempcache,
    infer_value,
)
from metabo_depthcharge.spec.adducts import encode_adduct
from metabo_depthcharge.spec.metadata_parsers import METADATA_FIELDS, METADATA_PARSERS
from metabo_depthcharge.spec.preprocessing import Spectrum
from metabo_depthcharge.spec.subformulae import ELEMENT_DIM, assign_peak_subformulae


def _read_mgf(path: str | PathLike) -> Iterable[Spectrum]:
    """Yield :class:`Spectrum` objects from an MGF file.

    Parameters
    ----------
    path : str or PathLike
        Path to an MGF-format peak list.

    Yields
    ------
    Spectrum
        One spectrum per ``BEGIN IONS`` block. Header keys land in
        :attr:`Spectrum.metadata` lowercased (pyteomics convention).

    Notes
    -----
    Standard MGF fields (``pepmass``, ``charge``, ``rtinseconds``) are parsed
    by pyteomics into their native types; other custom headers pass through
    as strings and are cast by HF Datasets to the dtype declared via
    :meth:`SpectrumDataset.from_mgf`'s ``columns`` and ``metadata`` arguments.
    """
    for s in read(str(path), use_index=False):
        yield Spectrum(
            mz=np.asarray(s["m/z array"], dtype=np.float64),
            intensity=np.asarray(s["intensity array"], dtype=np.float32),
            metadata=dict(s["params"]),
        )


def _sniff_mgf_columns(path: str | PathLike, n: int = 1000) -> dict[str, Value]:
    """Infer a column schema from the first ``n`` spectra of an MGF.

    Walks up to ``n`` spectra, taking the union of their ``params`` keys
    and inferring a :class:`Value` dtype from the first observed value of
    each key. Non-scalar values (tuples, lists, ...) fall back to
    ``Value("string")`` and are str-coerced at build time by
    :meth:`SpectrumDataset.from_mgf`.
    """
    inferred: dict[str, Value] = {}
    for i, s in enumerate(_read_mgf(path)):
        if i >= n:
            break
        for k, v in s.metadata.items():
            if k not in inferred and v is not None:
                inferred[k] = infer_value(v)
    return inferred


def _parse_precursor_mz(v) -> float:
    """Parse a raw precursor m/z value to float; returns 0.0 for missing/unparseable."""
    if v is None:
        return 0.0
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, tuple | list):  # pyteomics pepmass → (mz, charge)
        return float(v[0]) if v else 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


class SpectrumDataset(torch.utils.data.Dataset):
    """Spectrum dataset backed by an Arrow table, via HF Datasets.

    Build once via :meth:`from_mgf` or :meth:`from_list`, persist to disk,
    and reload with :meth:`from_disk`. Direct instantiation is not supported.

    Example
    -------
    .. code-block:: python

        # build and save
        ds = SpectrumDataset.from_mgf(
            "spectra.mgf",
            metadata=["adduct"],
            save_to="spectra/"
        )

        # load from previously-built
        ds = SpectrumDataset.from_disk("spectra/")

        # attach subformulae
        ds = ds.add_subformulae("gt", source="formulae.tsv", save_to="spectra/")

        # Usage like torch Dataset:
        len(ds)       # number of spectra
        ds[0]         # dict with "mz", "intensity", "precursor_mz", ... as tensors
        ds[0]["mz"]   # 1-D float32 tensor of m/z values

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=32,
            collate_fn=SpectrumDataset.collate
        )
    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            f"Use {type(self).__name__}.from_mgf(), .from_list(), or .from_disk() "
            "to construct a SpectrumDataset."
        )

    @classmethod
    def from_mgf(
        cls,
        path: str | PathLike,
        *,
        metadata: list[str] | None = None,
        columns: dict[str, Value] | None = None,
        precursor_mz_field: str = "pepmass",
        processor: Callable[[Spectrum], Spectrum] | None = None,
        transform: Callable[[Spectrum], Spectrum] | None = None,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "SpectrumDataset":
        """Parse an MGF file and materialize a preprocessed Arrow dataset.

        Parameters
        ----------
        path : str or PathLike
            Path to an MGF-format peak list.
        metadata : list[str], optional
            Any of: {``adduct``, ``collision_energy``, ``instrument_type``}.
            Metadata fields to be parsed and encoded for the neural network.
            ``None`` (default) means no metadata parsing.
        columns : dict[str, datasets.Value], optional
            Passthrough columns in dict format:
            MGF header key → HF :class:`Value` dtype. ``None`` (default) auto-sniffs
            the schema from the first 1000 spectra (non-scalars fall back to ``Value("string")``).
            Keys also listed in ``metadata`` are dropped from the sniffed schema.
        precursor_mz_field : str
            MGF field (lowercased) that holds the precursor m/z.
            Defaults to ``"pepmass"``.
            Some files may carry a `precursor_mz` field instead.
        processor : Callable[[Spectrum], Spectrum], optional
            Build-time preprocessor baked into the Arrow output.
            ``None`` (default) skips preprocessing.
            Recommended for deterministic expensive processing not to be repeated at each iteration.
            See the :doc:`spec module </api/spec>` for example preprocessors.
        transform : Callable[[Spectrum], Spectrum], optional
            Same as `processor`, but applied at iteration time.
            Recommended for lightweight or stochastic processing.
            ``None`` (default) skips iteration-time transforms.
        save_to : str or PathLike, optional
            If given, persist the built dataset to this path (loadable with
            :meth:`from_disk`). ``None`` (default) builds in memory.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults to
            ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset or resolves to a small ``tmpfs``.

        Returns
        -------
        SpectrumDataset
        """
        path = str(path)
        if columns is None:
            columns = _sniff_mgf_columns(path)
        for k in metadata or []:
            columns.pop(k, None)
        columns.pop("precursor_mz", None)
        columns.pop(precursor_mz_field, None)

        return cls._from_generator(
            lambda: _read_mgf(path),
            metadata=metadata,
            columns=columns,
            precursor_mz_field=precursor_mz_field,
            processor=processor,
            transform=transform,
            save_to=save_to,
            tmp_dir=tmp_dir,
        )

    @classmethod
    def from_list(
        cls,
        spectra: list[np.ndarray],
        *,
        precursor_mz: list[float] | np.ndarray | None = None,
        processor: Callable[[Spectrum], Spectrum] | None = None,
        transform: Callable[[Spectrum], Spectrum] | None = None,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "SpectrumDataset":
        r"""Build a dataset from a list of peak arrays.
        Note that passing extensive metadata/columns is not supported by this method; use from_mgf for that.

        Parameters
        ----------
        spectra : list of np.ndarray
            One array per spectrum, shape ``(2, N_peaks)``: ``arr[0]`` is
            the m/z values and ``arr[1]`` is the intensities.
        precursor_mz : list of float or np.ndarray, optional
            Precursor m/z per spectrum. Stored as a float32 column.
            Defaults to ``0.0`` for every row when not provided.
        processor : Callable[[Spectrum], Spectrum], optional
            Build-time preprocessor baked into the Arrow output.
            ``None`` (default) skips preprocessing.
            Recommended for deterministic expensive processing not to be repeated at each iteration.
            See the :doc:`spec module </api/spec>` for example preprocessors.
        transform : Callable[[Spectrum], Spectrum], optional
            Same as `processor`, but applied at iteration time.
            Recommended for lightweight or stochastic processing.
            ``None`` (default) skips iteration-time transforms.
        save_to : str or PathLike, optional
            If given, persist the built dataset to this path (loadable with
            :meth:`from_disk`). ``None`` (default) builds in memory.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults to
            ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset or resolves to a small ``tmpfs``..

        Returns
        -------
        SpectrumDataset
        """
        pmz: list[float]
        if precursor_mz is None:
            pmz = [0.0] * len(spectra)
        else:
            pmz = list(precursor_mz)
        if len(pmz) != len(spectra):
            raise ValueError(
                f"precursor_mz has {len(pmz)} entries but spectra has {len(spectra)}"
            )

        def _factory():
            for arr, mz_val in zip(spectra, pmz, strict=False):
                yield Spectrum(
                    mz=np.asarray(arr[0], dtype=np.float64),
                    intensity=np.asarray(arr[1], dtype=np.float32),
                    metadata={"precursor_mz": float(mz_val)},
                )

        return cls._from_generator(
            _factory,
            metadata=None,
            columns=None,
            processor=processor,
            transform=transform,
            save_to=save_to,
            tmp_dir=tmp_dir,
        )

    @classmethod
    def _from_generator(
        cls,
        spectra: Callable[[], Iterable[Spectrum]],
        *,
        metadata: list[str] | None = None,
        columns: dict[str, Value] | None = None,
        precursor_mz_field: str = "precursor_mz",
        processor: Callable[[Spectrum], Spectrum] | None = None,
        transform: Callable[[Spectrum], Spectrum] | None = None,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "SpectrumDataset":
        """Materialize a Spectrum factory (either from list or from mgf) into an Arrow dataset."""
        metadata = list(metadata or [])
        columns = dict(columns or {})
        unknown = set(metadata) - set(METADATA_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown metadata field(s) {sorted(unknown)}; allowed: {METADATA_FIELDS}"
            )
        collision = set(columns) & set(METADATA_FIELDS)
        if collision:
            raise ValueError(
                f"Keys {sorted(collision)} are reserved for `metadata=`; "
                f"remove them from `columns=`."
            )

        features = Features(
            {
                "mz": Sequence(Value("float64")),
                "intensity": Sequence(Value("float32")),
                "precursor_mz": Value("float32"),
                **{k: METADATA_PARSERS[k][1] for k in metadata},
                **columns,
            }
        )
        str_cols = {k for k, v in columns.items() if v.dtype == "string"}

        def gen():
            for s in tqdm(spectra(), desc="Parsing spectra", unit=" spectra"):
                if processor is not None:
                    s = processor(s)
                row = {
                    "mz": s.mz,
                    "intensity": s.intensity,
                    "precursor_mz": _parse_precursor_mz(
                        s.metadata.get(precursor_mz_field)
                    ),
                }
                for k in metadata:
                    row[k] = METADATA_PARSERS[k][0](s.metadata.get(k))
                for k in columns:
                    val = s.metadata.get(k)
                    if k in str_cols and val is not None and not isinstance(val, str):
                        val = str(val)
                    row[k] = val
                yield row

        with hf_tempcache(dir=tmp_dir) as cache_dir:
            with hf_silent():
                ds = Dataset.from_generator(
                    gen,
                    features=features,
                    keep_in_memory=save_to is None,
                    cache_dir=cache_dir,
                )
            ds = hf_persist(ds, save_to)
        return cls._create(ds, transform=transform)

    @classmethod
    def _create(
        cls,
        ds: Dataset,
        *,
        transform: Callable[[Spectrum], Spectrum] | None = None,
        subformulae_name: str | None = None,
        drop_peaks_without_subformula: bool = False,
        adduct_from_subformula: bool = False,
    ) -> "SpectrumDataset":
        """
        Private initialization helper method

        Returns the final SpectrumDataset object after optionally
        pruning subformula sets and checking for the presence of precursor_mz.
        """
        if subformulae_name is not None:
            all_sf_names = {
                k.removesuffix("_subformula_vec")
                for k in ds.column_names
                if k.endswith("_subformula_vec")
            }
            if subformulae_name not in all_sf_names:
                raise ValueError(
                    f"subformulae_name='{subformulae_name}' not found in dataset. "
                    f"Available: {sorted(all_sf_names) or '(none)'}"
                )
            drop = [
                c
                for n in all_sf_names - {subformulae_name}
                for c in (
                    f"{n}_subformula_vec",
                    f"{n}_parent_formula_vec",
                    f"{n}_adduct",
                )
                if c in ds.column_names
            ]
            if drop:
                ds = ds.remove_columns(drop)
        if (
            drop_peaks_without_subformula or adduct_from_subformula
        ) and subformulae_name is None:
            raise ValueError(
                "drop_peaks_without_subformula and adduct_from_subformula require subformulae_name to be set."
            )
        if "precursor_mz" not in ds.column_names:
            warnings.warn(
                "Dataset has no 'precursor_mz' column. SpectrumEncoder requires it. "
                "Build via from_list or from_mgf to include it automatically.",
                stacklevel=2,
            )
        obj = object.__new__(cls)
        obj.ds = ds.with_format("torch")
        obj.transform = transform
        obj.subformulae_name = subformulae_name
        obj.drop_peaks_without_subformula = drop_peaks_without_subformula
        obj.adduct_from_subformula = adduct_from_subformula
        return obj

    @classmethod
    def from_disk(
        cls,
        path: str | PathLike,
        *,
        transform: Callable[[Spectrum], Spectrum] | None = None,
        subformulae_name: str | None = None,
        drop_peaks_without_subformula: bool = False,
        adduct_from_subformula: bool = False,
    ) -> "SpectrumDataset":
        """Load a previously saved ``SpectrumDataset`` from an Arrow directory.

        Parameters
        ----------
        path : str or PathLike
            Directory previously written by :meth:`save_to`,
            :meth:`from_list` / :meth:`from_mgf` with ``save_to=...``.
        transform : Callable[[Spectrum], Spectrum], optional
            Iteration-time transform stored on the returned dataset.
        subformulae_name : str or None, optional
            Name of a subformula set previously added via
            :meth:`add_subformulae`. When set, all other subformula sets'
            columns are dropped from the in-memory dataset (the on-disk
            Arrow files are untouched). Enables the two bool flags below.
        drop_peaks_without_subformula : bool
            When ``True`` (requires ``subformulae_name``), ``__getitem__``
            filters each spectrum to only peaks matched by the selected set.
            ``mz``, ``intensity``, and all ``*_subformula_vec`` columns are
            filtered together; scalar and parent-formula columns are unchanged.
        adduct_from_subformula : bool
            When ``True`` (requires ``subformulae_name``), ``__getitem__``
            reads the raw adduct from ``{subformulae_name}_adduct``, encodes
            it, and injects it as ``row["adduct"]`` for
            :class:`~metabo_depthcharge.encoders.spectra.MetadataEncoder`.

        Returns
        -------
        SpectrumDataset
        """
        return cls._create(
            Dataset.load_from_disk(str(path)),
            transform=transform,
            subformulae_name=subformulae_name,
            drop_peaks_without_subformula=drop_peaks_without_subformula,
            adduct_from_subformula=adduct_from_subformula,
        )

    def save_to(self, path: str | PathLike) -> None:
        """Persist the underlying HF Dataset to disk in a directory of Arrow shards.

        Parameters
        ----------
        path : str or PathLike
            Destination directory. Loadable via :meth:`from_disk`.
        """
        self.ds.save_to_disk(str(path))

    def filter(self, condition: Callable[[dict], bool], **kwargs) -> "SpectrumDataset":
        """Return a new dataset keeping only rows where ``condition`` is truthy.

        Parameters
        ----------
        condition : Callable[[dict], bool]
            Row-wise condition as a callable taking a dict and returning a bool.
            The input dict keys are the dataset's columns
            (``mz``, ``intensity``, plus every column declared via
            ``metadata=`` / ``columns=`` at build time).
        **kwargs
            Passed to :meth:`datasets.Dataset.filter`. Common knobs:
            ``num_proc=8`` for parallelism, ``batched=True`` if the
            condition accepts a column-wise batch dict and returns a list
            of bools, ``load_from_cache_file=False`` to bypass HF's
            content-addressable cache.

        Returns
        -------
        SpectrumDataset
            New dataset preserving ``self.transform`` and the concrete
            subclass type.

        Examples
        --------
        .. code-block:: python

            ds.filter(lambda row: row["adduct"] == 1)       # encoded int; use encode_adduct("[M+H]+")
            ds.filter(lambda row: row["adduct"] in {1, 3})  # multiple adducts
            ds.filter(lambda row: row["precursor_mz"] < 500)
            ds.filter(lambda row: row["fold"] == "train" and len(row["mz"]) >= 5)
        """
        return type(self)._create(
            self.ds.filter(condition, **kwargs),
            transform=self.transform,
            subformulae_name=self.subformulae_name,
            drop_peaks_without_subformula=self.drop_peaks_without_subformula,
            adduct_from_subformula=self.adduct_from_subformula,
        )

    def select(self, indices: Iterable[int], **kwargs) -> "SpectrumDataset":
        """Return a new dataset with only the rows at ``indices``, in that order.

        Thin wrapper over :meth:`datasets.Dataset.select` that preserves
        ``self.transform`` and the subformula configuration and returns the
        concrete subclass type. Use it to materialize a row subset (e.g. a
        train/val/test fold) as a standalone dataset.

        Parameters
        ----------
        indices : Iterable[int]
            Row indices to keep, in the desired output order. Duplicates and
            arbitrary orderings are allowed.
        **kwargs
            Passed to :meth:`datasets.Dataset.select` (e.g.
            ``keep_in_memory=True``).

        Returns
        -------
        SpectrumDataset

        Examples
        --------
        .. code-block:: python

            ds.select([0, 5, 5, 2])
        """
        return type(self)._create(
            self.ds.select(indices, **kwargs),
            transform=self.transform,
            subformulae_name=self.subformulae_name,
            drop_peaks_without_subformula=self.drop_peaks_without_subformula,
            adduct_from_subformula=self.adduct_from_subformula,
        )

    def add_subformulae(
        self,
        name: str,
        *,
        source: str | PathLike,
        source_formula_col: str = "formula",
        source_adduct_col: str = "adduct",
        ppm: float = 10.0,
        num_workers: int = 1,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "SpectrumDataset":
        """Assign subformulae to peaks and add them as named columns.

        Reads molecular formulae and adducts from an external CSV/TSV ``source`` and
        runs per-peak subformula assignment (as in MIST). Three columns are
        added:

        - ``{name}_subformula_vec``: flat ``int16`` sequence of length
          ``N_peaks * ELEMENT_DIM`` (bag-of-atoms per peak).
        - ``{name}_parent_formula_vec``: ``int16`` sequence of length
          ``ELEMENT_DIM`` (parent molecular formula bag-of-atoms).
        - ``{name}_adduct``: raw adduct string used for the assignment.

        Unmatched peaks (outside ppm tolerance, or missing/invalid
        formula/adduct) receive zero vectors.
        The behavior of ``__getitem__`` can be modified to filter out unmatched peaks
        and/or inject the adduct as metadata for the encoder,
        by adjusting e.g. ``ds.drop_peaks_without_subformula=True``.

        Multiple named sets can be added by chaining calls::

            ds = ds.add_subformulae("gt", source="gt.tsv")
            ds = ds.add_subformulae("pred", source="pred.tsv")

        Parameters
        ----------
        name : str
            Identifier for this subformula set (used as column prefix).
        source : str or PathLike
            CSV or TSV file with one row per spectrum **in dataset order**,
            containing at least a formula column and an adduct column.
        source_formula_col : str
            Column name for the molecular formula in ``source``.
        source_adduct_col : str
            Column name for the adduct string in ``source``.
        ppm : float
            Mass tolerance in ppm for subformula matching.
        num_workers : int
            Number of parallel workers for :meth:`datasets.Dataset.map`.
        save_to : str or PathLike, optional
            If given, persist the result to this path (loadable with
            :meth:`from_disk`). Pass the same path as the original dataset
            to update it in place.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults to
            ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset or resolves to a small ``tmpfs``.

        Returns
        -------
        SpectrumDataset
            New dataset with three new columns: ``{name}_subformula_vec``,
            ``{name}_parent_formula_vec``, and ``{name}_adduct``.
        """

        src_df = pd.read_csv(source, sep=None, engine="python")
        if len(src_df) != len(self):
            raise ValueError(
                f"Source has {len(src_df)} rows but dataset has {len(self)}"
            )

        for col in (source_formula_col, source_adduct_col):
            if col not in src_df.columns:
                raise ValueError(
                    f"Column '{col}' not found in source file; "
                    f"available: {list(src_df.columns)}"
                )

        vec_col = f"{name}_subformula_vec"
        parent_col = f"{name}_parent_formula_vec"
        adduct_col = f"{name}_adduct"

        formulas = src_df[source_formula_col].fillna("").astype(str).tolist()
        adducts = src_df[source_adduct_col].fillna("").astype(str).tolist()

        def _fn(row, idx):
            mz = np.asarray(row["mz"], dtype=np.float64)
            adduct = adducts[idx]
            vecs, parent_vec, _ = assign_peak_subformulae(
                mz, formulas[idx], adduct, ppm
            )
            return {vec_col: vecs, parent_col: parent_vec, adduct_col: adduct}

        in_memory = save_to is None
        with hf_tempcache(dir=tmp_dir) as cache_dir:
            mapped_ds = self.ds.map(
                _fn,
                with_indices=True,
                num_proc=num_workers if num_workers > 1 else None,
                keep_in_memory=in_memory,
                cache_file_name=hf_cache_file(cache_dir, in_memory),
                desc=f"Assigning subformulae ({name})",
            )

            mapped_ds = hf_persist(mapped_ds, save_to)

        return type(self)._create(mapped_ds, transform=self.transform)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        row = self.ds[i]
        row = self._parse_subformula_vecs(row)
        if self.drop_peaks_without_subformula and self.subformulae_name is not None:
            row = self._filter_unmatched(row, self.subformulae_name)
        if self.adduct_from_subformula and self.subformulae_name is not None:
            row = self._inject_metadata_adduct(row, self.subformulae_name)
        if self.transform is None:
            return row
        return self._to_row(self.transform(self._to_spectrum(row)), row)

    @staticmethod
    def _parse_subformula_vecs(row: dict) -> dict:
        """Reshape flat *_subformula_vec tensors from (N_peaks * D,) to (N_peaks, D)."""
        if not any(k.endswith("_subformula_vec") for k in row):
            return row
        n_peaks = len(row["mz"])
        for k in list(row):
            if k.endswith("_subformula_vec") and isinstance(row[k], torch.Tensor):
                row[k] = row[k].reshape(n_peaks, ELEMENT_DIM)
        return row

    @staticmethod
    def _filter_unmatched(row: dict, name: str) -> dict:
        """Filter each spectrum to peaks matched by the named subformula set."""
        if len(row["mz"]) == 0:
            return row
        valid = (
            row[f"{name}_subformula_vec"].sum(dim=-1) != 0
        )  # (N_peaks,); vec is (N_peaks, D) after _parse
        filter_key = {"mz", "intensity"} | {
            k for k in row if k.endswith("_subformula_vec")
        }
        return {k: v[valid] if k in filter_key else v for k, v in row.items()}

    @staticmethod
    def _inject_metadata_adduct(row: dict, name: str) -> dict:
        """Override row["adduct"] with the encoded adduct from the named subformula set."""
        row["adduct"] = torch.tensor(
            encode_adduct(str(row[f"{name}_adduct"])), dtype=torch.int64
        )
        return row

    @staticmethod
    def _to_spectrum(row: dict) -> Spectrum:
        return Spectrum(
            mz=np.asarray(row["mz"], dtype=np.float64),
            intensity=np.asarray(row["intensity"], dtype=np.float32),
            metadata={k: v for k, v in row.items() if k not in ("mz", "intensity")},
        )

    @staticmethod
    def _to_row(spectrum: Spectrum, template: dict | None = None) -> dict:
        row = dict(template) if template else {}
        row["mz"] = torch.as_tensor(spectrum.mz, dtype=torch.float32)
        row["intensity"] = torch.as_tensor(spectrum.intensity, dtype=torch.float32)
        return row

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """Pad ragged ``mz``/``intensity`` into batched torch tensors.

        Pass as ``collate_fn`` to :class:`torch.utils.data.DataLoader`.

        Parameters
        ----------
        batch : list[dict]
            Rows produced by :meth:`__getitem__`.

        Returns
        -------
        dict
            - ``mz``, ``intensity``: ``(B, L)`` tensors zero-padded to the
              longest spectrum in the batch.
            - ``mask``: ``(B, L)`` bool tensor, ``True`` at real peaks.
            - ``metadata``: dict of ``(B,)`` tensors for NN-encoded fields
              (``adduct``, ``collision_energy``, ``instrument_type``), present
              only when those columns exist. Ready to pass directly to
              :class:`~metabo_depthcharge.encoders.spectra.MetadataEncoder`.
            - ``subformulae``: dict keyed by subformula set name, each value
              a dict ``{"form_vec": (B, L, ELEMENT_DIM), "parent_form_vec":
              (B, ELEMENT_DIM)}``. Present only when subformula columns exist.
            - All other tensor columns: stacked to ``(B,)`` at the top level.
            - String/object columns: passed through as Python lists.
        """
        mz = pad_sequence(
            [torch.as_tensor(r["mz"], dtype=torch.float32) for r in batch],
            batch_first=True,
        )
        intensity = pad_sequence(
            [torch.as_tensor(r["intensity"], dtype=torch.float32) for r in batch],
            batch_first=True,
        )
        lengths = torch.tensor([len(r["mz"]) for r in batch])
        mask = torch.arange(mz.size(1))[None, :] < lengths[:, None]
        out = {"mz": mz, "intensity": intensity, "mask": mask}

        metadata = {}
        subformulae = {}
        for k in batch[0]:
            if k in ("mz", "intensity") or k.endswith("_parent_formula_vec"):
                continue
            if k.endswith("_subformula_vec"):
                name = k.removesuffix("_subformula_vec")
                parents = torch.stack([r[f"{name}_parent_formula_vec"] for r in batch])
                subformulae[name] = {
                    "form_vec": pad_sequence([r[k] for r in batch], batch_first=True),
                    "parent_form_vec": parents,
                }
            else:
                vals = [r[k] for r in batch]
                stacked = (
                    torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
                )
                if k in METADATA_FIELDS:
                    metadata[k] = stacked
                else:
                    out[k] = stacked
        if metadata:
            out["metadata"] = metadata
        if subformulae:
            # Flatten when only one set is present (i.e. subformulae_name was set at load time)
            out["subformulae"] = (
                next(iter(subformulae.values()))
                if len(subformulae) == 1
                else subformulae
            )

        return out
