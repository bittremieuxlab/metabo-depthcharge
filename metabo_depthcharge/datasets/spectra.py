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

from metabo_depthcharge.datasets._common import hf_silent, hf_tempcache, infer_value
from metabo_depthcharge.spec.adducts import encode_adduct
from metabo_depthcharge.spec.metadata_parsers import METADATA_FIELDS, METADATA_PARSERS
from metabo_depthcharge.spec.preprocessing import Spectrum
from metabo_depthcharge.spec.subformulae import ELEMENT_DIM, assign_peak_subformulae


def read_mgf(path: str | PathLike) -> Iterable[Spectrum]:
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
    :meth:`SpectrumDataset.from_spectra`.
    """
    inferred: dict[str, Value] = {}
    for i, s in enumerate(read_mgf(path)):
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
    """Torch :class:`Dataset` over an HF spectra :class:`Dataset`.

    Construct via the alternative constructors :meth:`from_spectra` (a
    factory yielding :class:`Spectrum`), :meth:`from_mgf` (parse an MGF), or
    :meth:`from_disk` (load a previously saved directory). To wrap an
    existing HF :class:`Dataset`, pass it to ``__init__`` directly.

    Parameters
    ----------
    ds : datasets.Dataset
        The underlying HF Dataset whose rows have ``mz`` and ``intensity``
        columns plus any metadata columns declared at build time. The
        dataset is wrapped with ``with_format("torch")``, so numeric
        columns surface as torch tensors at iteration time.
    transform : Callable[[Spectrum], Spectrum], optional
        Iteration-time transform applied fresh on every ``__getitem__``.
        Use for stochastic augmentations. ``None`` (default) skips the
        Spectrum roundtrip and returns the HF row untouched.
    subformulae_name : str or None, optional
        Name of a subformula set previously added via :meth:`add_subformulae`.
        When set, all other subformula sets' columns (``*_subformula_vec``,
        ``*_parent_formula_vec``, ``*_adduct``) are dropped from the
        in-memory dataset at construction time (the on-disk Arrow files
        are untouched). Required for the two bool flags below.
    drop_peaks_without_subformula : bool
        When ``True`` (requires ``subformulae_name``), each ``__getitem__``
        call filters the spectrum to only peaks matched by the selected
        subformula set. See :meth:`from_disk` for details.
    adduct_from_subformula : bool
        When ``True`` (requires ``subformulae_name``), each ``__getitem__``
        call reads the raw adduct from ``{subformulae_name}_adduct``, encodes
        it, and injects it as ``row["adduct"]`` for
        :class:`~metabo_depthcharge.encoders.spectra.MetadataEncoder`.
        See :meth:`from_disk` for details.

    Attributes
    ----------
    ds : datasets.Dataset
        The torch-formatted HF Dataset; use it as an escape hatch for HF
        operations not delegated by this class (``select_columns``,
        ``shuffle``, ``train_test_split``, ``concatenate_datasets``, ...).
    transform : Callable[[Spectrum], Spectrum] or None
        The active iteration-time transform.
    subformulae_name : str or None
        The active subformula set, or ``None``.
    drop_peaks_without_subformula : bool
        Whether unmatched peaks are dropped at iteration time.
    adduct_from_subformula : bool
        Whether the adduct is sourced from the subformula set at iteration time.

    Notes
    -----
    For non-trivial per-sample logic (pair sampling, hard negatives,
    multi-view), subclass and override :meth:`__getitem__`. The base
    ``__getitem__`` reads ``self.ds[i]`` and applies ``self.transform``,
    so ``super().__getitem__(i)`` is the right delegation primitive.
    """

    def __init__(
        self,
        ds: Dataset,
        transform: Callable[[Spectrum], Spectrum] | None = None,
        subformulae_name: str | None = None,
        drop_peaks_without_subformula: bool = False,
        adduct_from_subformula: bool = False,
    ):
        if subformulae_name is not None:
            all_sf_names = {
                k.removesuffix("_subformula_vec")
                for k in ds.column_names
                if k.endswith("_subformula_vec")
            }
            if subformulae_name not in all_sf_names:
                raise ValueError(
                    f"subformulae_name='{subformulae_name}' not found in dataset. Available: {sorted(all_sf_names) or '(none)'}"
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
                "Dataset has no 'precursor_mz' column. SpectrumEmbedder requires it. "
                "Build via from_mgf or from_spectra to include it automatically.",
                stacklevel=2,
            )
        self.ds = ds.with_format("torch")
        self.transform = transform
        self.subformulae_name = subformulae_name
        self.drop_peaks_without_subformula = drop_peaks_without_subformula
        self.adduct_from_subformula = adduct_from_subformula

    @classmethod
    def from_spectra(
        cls,
        spectra: Callable[[], Iterable[Spectrum]],
        *,
        metadata: list[str] | None = None,
        columns: dict[str, Value] | None = None,
        processor: Callable[[Spectrum], Spectrum] | None = None,
        transform: Callable[[Spectrum], Spectrum] | None = None,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "SpectrumDataset":
        """Materialize an iterable of :class:`Spectrum` into a preprocessed Arrow dataset.

        Parameters
        ----------
        spectra : Callable[[], Iterable[Spectrum]]
            Zero-arg factory returning a fresh iterable of spectra on each
            call. A factory (rather than a live iterable) is required
            because HF Datasets pickles the source to compute its cache
            fingerprint and may re-invoke it; a primed generator is not
            picklable.
        metadata : list[str], optional
            Any of: {``adduct``, ``collision_energy``, ``instrument_type``}.
            Metadata fields to be parsed and processed by the neural network encoders.
            See :class:`~metabo_depthcharge.encoders.spectra.MetadataEncoder`.
            ``None`` (default) means no metadata parsing.
        columns : dict[str, datasets.Value], optional
            Passthrough columns: ``Spectrum.metadata`` key → HF
            :class:`Value` dtype. Missing keys per spectrum yield
            ``None``. Keys in :data:`METADATA_FIELDS` are reserved for
            ``metadata=`` and must not appear here. ``None`` (default)
            means no passthrough columns.
        processor : Callable[[Spectrum], Spectrum], optional
            Build-time preprocessor applied once per spectrum and baked
            into the Arrow output. ``None`` (default) skips preprocessing;
            pass :data:`~metabo_depthcharge.spec.preprocessing.DefaultSpectrumProcessor`
            (or a custom callable) to opt in.
        transform : Callable[[Spectrum], Spectrum], optional
            Iteration-time transform stored on the returned dataset.
        save_to : str or PathLike, optional
            If given, stream the build to disk and persist a portable copy
            at this path (loadable with :meth:`from_disk`). Use for datasets
            that don't fit in memory. If ``None`` (default), the dataset is
            built in memory and nothing is written to disk.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults to
            :func:`tempfile.gettempdir` (honors ``$TMPDIR``), which is
            usually correct: SLURM/PBS-style HPC schedulers point this at
            per-job local scratch, and workstations point it at ``/tmp``.
            Override on systems where ``$TMPDIR`` is unset, points at a
            small ``/tmp``, or resolves to a ``tmpfs`` (which would
            silently consume RAM during a ``save_to=...`` build).

        Returns
        -------
        SpectrumDataset
        """
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
        if "precursor_mz" in columns:
            raise ValueError(
                "'precursor_mz' is always parsed as a float32 column; "
                "remove it from `columns=`."
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
                    "precursor_mz": _parse_precursor_mz(s.metadata.get("precursor_mz")),
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
            with hf_silent():  # gen() has its own tqdm; silence HF's dual bar
                ds = Dataset.from_generator(
                    gen,
                    features=features,
                    keep_in_memory=save_to is None,
                    cache_dir=cache_dir,
                )
            if save_to is not None:
                ds.save_to_disk(str(save_to))
                ds = Dataset.load_from_disk(str(save_to))
        return cls(ds, transform=transform)

    @classmethod
    def from_mgf(
        cls,
        path: str | PathLike,
        *,
        metadata: list[str] | None = None,
        columns: dict[str, Value] | None = None,
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
            See :meth:`from_spectra`. Strictly opt-in: never auto-enabled.
        columns : dict[str, datasets.Value], optional
            See :meth:`from_spectra`. ``None`` (default) auto-sniffs the
            schema from the first 1000 spectra via
            :func:`_sniff_mgf_columns` (non-scalars fall back to
            ``Value("string")``). Keys also listed in ``metadata`` are
            dropped from the sniffed schema.
        processor : Callable[[Spectrum], Spectrum], optional
            See :meth:`from_spectra`.
        transform : Callable[[Spectrum], Spectrum], optional
            See :meth:`from_spectra`.
        save_to : str or PathLike, optional
            See :meth:`from_spectra`.
        tmp_dir : str or PathLike, optional
            See :meth:`from_spectra`.

        Returns
        -------
        SpectrumDataset
        """
        path = str(path)
        if columns is None:
            columns = _sniff_mgf_columns(path)
        for k in metadata or []:
            columns.pop(k, None)
        columns.pop("precursor_mz", None)  # always parsed separately as float32

        return cls.from_spectra(
            lambda: read_mgf(path),
            metadata=metadata,
            columns=columns,
            processor=processor,
            transform=transform,
            save_to=save_to,
            tmp_dir=tmp_dir,
        )

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
        """Load a previously saved Arrow directory.

        Parameters
        ----------
        path : str or PathLike
            Directory previously written by :meth:`save_to`,
            :meth:`from_spectra` / :meth:`from_mgf` with ``save_to=...``,
            or :meth:`datasets.Dataset.save_to_disk`.
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
        return cls(
            Dataset.load_from_disk(str(path)),
            transform=transform,
            subformulae_name=subformulae_name,
            drop_peaks_without_subformula=drop_peaks_without_subformula,
            adduct_from_subformula=adduct_from_subformula,
        )

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

    def filter(self, predicate: Callable[[dict], bool], **kwargs) -> "SpectrumDataset":
        """Return a new dataset keeping only rows where ``predicate`` is truthy.

        Parameters
        ----------
        predicate : Callable[[dict], bool]
            Row-wise predicate. The argument is a dict whose keys are the
            dataset's columns (``mz``, ``intensity``, plus every column
            declared via ``metadata=`` / ``columns=`` at build time).
            ``mz`` and ``intensity`` come through as torch tensors
            (matching ``__getitem__``); scalar columns come through as
            native ``str`` / ``float`` / ``int`` per the schema.
        **kwargs
            Passed to :meth:`datasets.Dataset.filter`. Common knobs:
            ``num_proc=8`` for parallelism, ``batched=True`` if the
            predicate accepts a column-wise batch dict and returns a list
            of bools, ``load_from_cache_file=False`` to bypass HF's
            content-addressable cache.

        Returns
        -------
        SpectrumDataset
            New dataset preserving ``self.transform`` and the concrete
            subclass type.

        Examples
        --------
        >>> ds.filter(lambda r: r["adduct"] == 1)       # encoded int; use encode_adduct("[M+H]+")
        >>> ds.filter(lambda r: r["adduct"] in {1, 3})  # multiple adducts
        >>> ds.filter(lambda r: r["precursor_mz"] < 500)
        >>> ds.filter(lambda r: r["fold"] == "train" and len(r["mz"]) >= 5)
        """
        return type(self)(
            self.ds.filter(predicate, **kwargs),
            transform=self.transform,
            subformulae_name=self.subformulae_name,
            drop_peaks_without_subformula=self.drop_peaks_without_subformula,
            adduct_from_subformula=self.adduct_from_subformula,
        )

    def save_to(self, path: str | PathLike) -> None:
        """Persist the underlying HF Dataset to disk as Arrow shards.

        Use this to snapshot a dataset that was built in memory (or
        derived from one via :meth:`filter` etc.) into a portable
        directory loadable with :meth:`from_disk`. To stream a build
        directly to disk without materializing it in memory, pass
        ``save_to=...`` to :meth:`from_spectra` / :meth:`from_mgf`
        instead.

        Parameters
        ----------
        path : str or PathLike
            Destination directory. Loadable via :meth:`from_disk`.
        """
        self.ds.save_to_disk(str(path))

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
    ) -> "SpectrumDataset":
        """Assign subformulae to peaks and add them as named columns.

        Reads molecular formulae and adducts from an external CSV/TSV and
        runs per-peak subformula assignment (as in MIST). Three columns are
        added:

        - ``{name}_subformula_vec``: flat ``int16`` sequence of length
          ``N_peaks x ELEMENT_DIM`` (bag-of-atoms per peak).
        - ``{name}_parent_formula_vec``: ``int16`` sequence of length
          ``ELEMENT_DIM`` (parent molecular formula bag-of-atoms).
        - ``{name}_adduct``: raw adduct string used for the assignment.

        Unmatched peaks (outside ppm tolerance, or missing/invalid
        formula/adduct) receive zero vectors.

        Multiple named sets can be added by chaining calls::

            ds = ds.add_subformulae("gt", source="gt.tsv")
            ds = ds.add_subformulae("pred", source="pred.tsv")

        Parameters
        ----------
        name : str
            Identifier for this subformula set (used as column prefix).
        source : str or PathLike
            CSV or TSV file with one row per spectrum in **dataset order**,
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
            If given, persist the result to this path (loadable via
            :meth:`from_disk`). Pass the same path as the original dataset
            to update it in place.

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

        mapped_ds = self.ds.map(
            _fn,
            with_indices=True,
            num_proc=num_workers if num_workers > 1 else None,
            desc=f"Assigning subformulae ({name})",
        )

        if save_to is not None:
            mapped_ds.save_to_disk(str(save_to))
            mapped_ds = Dataset.load_from_disk(str(save_to))

        return type(self)(
            mapped_ds,
            transform=self.transform,
            subformulae_name=None,
            drop_peaks_without_subformula=False,
            adduct_from_subformula=False,
        )

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
