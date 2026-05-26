"""Spectrum dataset creation and dataloading utilities"""

from collections.abc import Callable, Iterable
from os import PathLike

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from pyteomics.mgf import read
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from metabo_depthcharge.datasets._common import hf_silent, hf_tempcache, infer_value
from metabo_depthcharge.spec.metadata_parsers import METADATA_FIELDS, METADATA_PARSERS
from metabo_depthcharge.spec.preprocessing import Spectrum


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

    Attributes
    ----------
    ds : datasets.Dataset
        The torch-formatted HF Dataset; use it as an escape hatch for HF
        operations not delegated by this class (``select_columns``,
        ``shuffle``, ``train_test_split``, ``concatenate_datasets``, ...).
    transform : Callable[[Spectrum], Spectrum] or None
        The active iteration-time transform.

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
    ):
        self.ds = ds.with_format("torch")
        self.transform = transform

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
            See :class:`~metabo_depthcharge.encoders.transformers.MetadataEncoder`.
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

        features = Features(
            {
                "mz": Sequence(Value("float64")),
                "intensity": Sequence(Value("float32")),
                **{k: METADATA_PARSERS[k][1] for k in metadata},
                **columns,
            }
        )
        str_cols = {k for k, v in columns.items() if v.dtype == "string"}

        def gen():
            for s in tqdm(spectra(), desc="Parsing spectra", unit=" spectra"):
                if processor is not None:
                    s = processor(s)
                row = {"mz": s.mz, "intensity": s.intensity}
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

        Returns
        -------
        SpectrumDataset
        """
        return cls(Dataset.load_from_disk(str(path)), transform=transform)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        row = self.ds[i]
        if self.transform is None:
            return row
        return self._to_row(self.transform(self._to_spectrum(row)), row)

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
        >>> ds.filter(lambda r: r["adduct"] == "[M+H]+")
        >>> ds.filter(lambda r: r["adduct"] in {"[M+H]+", "[M+Na]+"})
        >>> ds.filter(lambda r: r["precursor_mz"] < 500)
        >>> ds.filter(lambda r: r["fold"] == "train" and len(r["mz"]) >= 5)
        """
        return type(self)(self.ds.filter(predicate, **kwargs), transform=self.transform)

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
            ``mz`` and ``intensity`` as ``(B, L)`` torch tensors padded
            with zeros, ``mask`` as a ``(B, L)`` bool tensor (``True`` at
            real peaks). NN-encoded metadata fields (those declared via
            ``metadata=`` at build time) are stacked into ``(B,)`` tensors
            and nested under ``batch["metadata"]``, ready to pass directly
            to :class:`MetadataEncoder`. Other tensor-valued columns are
            stacked into ``(B,)`` tensors at the top level; string/object
            columns pass through as Python lists.
        """
        mz = pad_sequence(
            [torch.as_tensor(r["mz"], dtype=torch.float64) for r in batch],
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
        for k in batch[0]:
            if k in ("mz", "intensity"):
                continue
            vals = [r[k] for r in batch]
            stacked = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
            if k in METADATA_FIELDS:
                metadata[k] = stacked
            else:
                out[k] = stacked
        if metadata:
            out["metadata"] = metadata
        return out
