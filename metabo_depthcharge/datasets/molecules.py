"""Molecule dataset creation and dataloading utilities"""

import uuid
from collections.abc import Callable, Iterable
from os import PathLike

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from tqdm.auto import tqdm

from metabo_depthcharge.datasets._common import hf_silent, hf_tempcache
from metabo_depthcharge.molecules.fingerprints import (
    SmilesToBiosynfoni,
    SmilesToChemBERTa,
    SmilesToMACCS,
    SmilesToMAP4,
    SmilesToMolFormer,
    SmilesToMorgan,
    SmilesToRdkit,
    SmilesToUniMol,
)
from metabo_depthcharge.molecules.molecule import Molecule


_FP_INFO: dict[str, dict] = {
    "morgan": {
        "kind": "binary",
        "size": 4096,
        "build": lambda **kw: SmilesToMorgan(**{"fp_size": 4096, **kw}),
    },
    "morgan_count": {
        "kind": "count",
        "size": 4096,
        "build": lambda **kw: SmilesToMorgan(**{"fp_size": 4096, "counts": True, **kw}),
    },
    "rdkit": {
        "kind": "binary",
        "size": 4096,
        "build": lambda **kw: SmilesToRdkit(**{"fp_size": 4096, **kw}),
    },
    "rdkit_count": {
        "kind": "count",
        "size": 4096,
        "build": lambda **kw: SmilesToRdkit(**{"fp_size": 4096, "counts": True, **kw}),
    },
    "maccs": {"kind": "binary", "size": 167, "build": lambda **kw: SmilesToMACCS(**kw)},
    "biosynfoni": {
        "kind": "count",
        "size": 39,
        "build": lambda **kw: SmilesToBiosynfoni(**kw),
    },
    "map4": {
        "kind": "binary",
        "size": 4096,
        "build": lambda **kw: SmilesToMAP4(**{"fp_size": 4096, **kw}),
    },
    "unimol": {
        "kind": "dense",
        "size": 512,
        "neural": True,
        "build": lambda **kw: SmilesToUniMol(**kw),
    },
    "chemberta": {
        "kind": "dense",
        "size": 768,
        "neural": True,
        # default model pinned to match registered size; override via
        # build_kwargs={"model_name": ...} (caller must keep size at 768).
        "build": lambda **kw: SmilesToChemBERTa(
            **{"model_name": "seyonec/ChemBERTa-zinc-base-v1", **kw}
        ),
    },
    "molformer": {
        "kind": "dense",
        "size": 768,
        "neural": True,
        "build": lambda **kw: SmilesToMolFormer(**kw),
    },
}


def _add_fingerprint_column(
    ds: Dataset,
    name: str,
    build_kwargs: dict | None = None,
    *,
    batch_size: int,
    num_proc: int,
    keep_in_memory: bool,
) -> Dataset:
    """Compute and append fingerprint column(s) onto ``ds`` for ``name``.

    ``name`` must be a key of :data:`_FP_INFO`; ``build_kwargs`` is
    forwarded to the registered builder (e.g. ``{"device": "cuda"}`` for
    neural FPs, ``{"model_name": "..."}`` for ChemBERTa/MolFormer).
    """
    if name not in _FP_INFO:
        raise ValueError(
            f"Unknown fingerprint column {name!r}; must be one of {sorted(_FP_INFO)}"
        )
    spec = _FP_INFO[name]
    kind = spec["kind"]
    fp_size = spec["size"]
    build_kwargs = build_kwargs or {}
    # HF convention: num_proc=None means in-process; num_proc=1 still spawns
    # a Pool of one worker (which would try to pickle the closure). Built
    # representations often hold unpicklable C++ state (e.g. RDKit
    # FingerprintGenerator64), so we don't capture an instance — we build
    # lazily on first batch via the [None] cell below. This means each
    # worker builds its own copy after unpickling; neural FPs (in-process)
    # also build exactly once and reuse across batches.
    nproc = None if spec.get("neural") or num_proc <= 1 else num_proc
    rep_cache: list = [None]

    if kind == "dense":
        new_cols = {name: Sequence(Value("float32"), length=fp_size)}
    elif kind == "binary":
        new_cols = {name: Sequence(Value("int16"))}
    else:  # count
        new_cols = {
            name: Sequence(Value("int16")),
            f"{name}_values": Sequence(Value("uint16")),
        }

    def map_batched(batch):
        if rep_cache[0] is None:
            rep_cache[0] = _FP_INFO[name]["build"](**build_kwargs)
        fps = np.asarray(rep_cache[0](batch["smiles"]))
        # Pass numpy arrays straight to HF — pyarrow handles 2D arrays for
        # fixed-length Sequence and lists-of-1D-arrays for variable-length.
        # Going through ``.tolist()`` would allocate millions of Python
        # floats/ints per batch, which dominates wall-time on big datasets.
        if kind == "dense":
            return {name: fps.astype(np.float32, copy=False)}
        if kind == "binary":
            return {name: [np.where(r > 0)[0].astype(np.int16) for r in fps]}
        idx = [np.where(r > 0)[0] for r in fps]
        return {
            name: [i.astype(np.int16) for i in idx],
            f"{name}_values": [
                r[i].astype(np.uint16) for r, i in zip(fps, idx, strict=False)
            ],
        }

    # Bypass HF's map-function fingerprinting (some representations hold
    # unpicklable C++ objects, e.g. RDKit fp generators). We don't rely on
    # cross-run cache hits — builds run inside a transient tempdir.
    return ds.map(
        map_batched,
        batched=True,
        batch_size=batch_size,
        num_proc=nproc,
        features=Features({**ds.features, **new_cols}),
        keep_in_memory=keep_in_memory,
        new_fingerprint=uuid.uuid4().hex,
        desc=f"Computing {name}",
    )


class MoleculeDataset(torch.utils.data.Dataset):
    """Torch :class:`Dataset` over an HF molecules :class:`Dataset`.

    Construct via :meth:`from_smiles`, :meth:`from_csv`, or
    :meth:`from_disk`. To wrap an existing HF :class:`Dataset`, pass it to
    ``__init__`` directly.

    Multiple fingerprint columns may share one ``smiles`` column. Column
    names are pinned to a canonical vocabulary (:data:`_FP_INFO`) — Morgan,
    MACCS, ChemBERTa, etc. each have a fixed ``(kind, fp_size)`` so
    :meth:`collate` can densify sparse columns from the column name alone
    (no per-dataset bookkeeping, no sidecar files). Sparse columns store
    set-bit indices as ``Sequence(Value("int16"))``; count columns add a
    parallel ``<name>_values`` ``Sequence(Value("uint16"))``; dense columns
    are ``Sequence(Value("float32"), length=fp_size)``.

    Parameters
    ----------
    ds : datasets.Dataset
        The underlying HF Dataset whose rows have a ``smiles`` column plus
        any metadata and fingerprint columns. Wrapped with
        ``with_format("torch")``.
    transform : Callable[[Molecule], Molecule], optional
        Iteration-time transform applied fresh on every ``__getitem__``.
        Receives a :class:`Molecule` reconstructed from the ``smiles`` and
        metadata columns; fingerprint columns pass through untouched.

    Attributes
    ----------
    ds : datasets.Dataset
        The torch-formatted HF Dataset; use it as an escape hatch for HF
        operations not delegated by this class (``select_columns``,
        ``shuffle``, ``train_test_split``, ``concatenate_datasets``, ...).
    transform : Callable[[Molecule], Molecule] or None
        The active iteration-time transform.
    """

    def __init__(
        self,
        ds: Dataset,
        transform: Callable[[Molecule], Molecule] | None = None,
    ):
        self.ds = ds.with_format("torch")
        self.transform = transform

    @classmethod
    def from_smiles(
        cls,
        molecules: Callable[[], Iterable[Molecule | str]],
        *,
        metadata: dict[str, Value] | None = None,
        representations: dict[str, dict] | None = None,
        batch_size: int = 512,
        num_proc: int = 1,
        transform: Callable[[Molecule], Molecule] | None = None,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "MoleculeDataset":
        """Materialize an iterable of molecules into an Arrow dataset.

        Parameters
        ----------
        molecules : Callable[[], Iterable[Molecule | str]]
            Zero-arg factory returning a fresh iterable on each call.
        metadata : dict[str, datasets.Value], optional
            Mapping from metadata key to HF :class:`Value` dtype. Each
            entry becomes a top-level column; missing keys per molecule
            yield ``None``.
        representations : dict[str, dict], optional
            Canonical fingerprint column names (keys of :data:`_FP_INFO`)
            mapped to ``build_kwargs`` forwarded to the registered builder.
            Use ``{}`` for defaults, e.g. ``{"morgan": {}, "chemberta":
            {"device": "cuda", "model_name": "..."}}``.
        batch_size : int
            Batch size passed to :meth:`datasets.Dataset.map` for each
            fingerprint computation.
        num_proc : int
            Worker count for :meth:`datasets.Dataset.map`. Forced to
            in-process for neural fingerprinters so the model is loaded
            once.
        transform : Callable[[Molecule], Molecule], optional
            Iteration-time transform stored on the returned dataset.
        save_to : str or PathLike, optional
            If given, stream the build to disk and persist to this path
            (loadable with :meth:`from_disk`). If ``None`` (default), the
            dataset is built in memory.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults
            to ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset, points at a small ``/tmp``, or resolves to a ``tmpfs``.
        """
        metadata = metadata or {}
        features = Features({"smiles": Value("string"), **metadata})
        in_memory = save_to is None

        def gen():
            for m in tqdm(molecules(), desc="Parsing molecules", unit=" mols"):
                if isinstance(m, str):
                    m = Molecule(smiles=m)
                yield {"smiles": m.smiles, **{k: m.metadata.get(k) for k in metadata}}

        with hf_tempcache(dir=tmp_dir) as cache_dir:
            with hf_silent():  # gen() has its own tqdm; silence HF's dual bar
                ds = Dataset.from_generator(
                    gen,
                    features=features,
                    keep_in_memory=in_memory,
                    cache_dir=cache_dir,
                )
            for name, kw in (representations or {}).items():
                ds = _add_fingerprint_column(
                    ds,
                    name,
                    kw,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    keep_in_memory=in_memory,
                )
            if save_to is not None:
                ds.save_to_disk(str(save_to))
                ds = Dataset.load_from_disk(str(save_to))

        return cls(ds, transform=transform)

    @classmethod
    def from_csv(
        cls,
        path: str | PathLike,
        *,
        sep: str = ",",
        smiles_column: str = "smiles",
        representations: dict[str, dict] | None = None,
        batch_size: int = 512,
        num_proc: int = 1,
        transform: Callable[[Molecule], Molecule] | None = None,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "MoleculeDataset":
        """Parse a CSV/TSV file and materialize an Arrow dataset.

        Schema is inferred by HF's :meth:`datasets.Dataset.from_csv` (via
        pyarrow). For full control over column dtypes, pre-build a
        :class:`datasets.Dataset` yourself and pass it to ``__init__``.

        Parameters
        ----------
        path : str or PathLike
            Path to a delimited text file with a header row.
        sep : str
            Field delimiter. Default ``","``; pass ``"\\t"`` for TSV.
        smiles_column : str
            Header name of the SMILES column. Renamed to ``"smiles"`` if
            different.
        representations, batch_size, num_proc, transform, save_to, tmp_dir
            See :meth:`from_smiles`.
        """
        in_memory = save_to is None
        with hf_tempcache(dir=tmp_dir) as cache_dir:
            tqdm.write(f"Reading molecules from {path}...")
            with hf_silent():
                ds = Dataset.from_csv(
                    str(path),
                    sep=sep,
                    keep_in_memory=in_memory,
                    cache_dir=cache_dir,
                )
            tqdm.write(f"  loaded {len(ds)} molecules")
            if smiles_column != "smiles":
                ds = ds.rename_column(smiles_column, "smiles")
            for name, kw in (representations or {}).items():
                ds = _add_fingerprint_column(
                    ds,
                    name,
                    kw,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    keep_in_memory=in_memory,
                )
            if save_to is not None:
                ds.save_to_disk(str(save_to))
                ds = Dataset.load_from_disk(str(save_to))
        return cls(ds, transform=transform)

    @classmethod
    def from_disk(
        cls,
        path: str | PathLike,
        *,
        transform: Callable[[Molecule], Molecule] | None = None,
    ) -> "MoleculeDataset":
        """Load a previously saved Arrow directory."""
        return cls(Dataset.load_from_disk(str(path)), transform=transform)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        row = self.ds[i]
        if self.transform is None:
            return row
        md = {k: v for k, v in row.items() if k != "smiles"}
        mol = self.transform(Molecule(smiles=row["smiles"], metadata=md))
        return {**row, "smiles": mol.smiles}

    def filter(self, predicate: Callable[[dict], bool], **kwargs) -> "MoleculeDataset":
        """Return a new dataset keeping rows where ``predicate`` is truthy."""
        return type(self)(
            self.ds.filter(predicate, **kwargs),
            transform=self.transform,
        )

    def add_fingerprint(
        self,
        name: str,
        build_kwargs: dict | None = None,
        *,
        batch_size: int = 512,
        num_proc: int = 1,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "MoleculeDataset":
        """Compute and append a new fingerprint column.

        ``name`` must be a key of :data:`_FP_INFO`. ``build_kwargs`` is
        forwarded to the registered builder (e.g. ``{"device": "cuda"}``).
        Returns a new dataset with the column added.
        """
        in_memory = save_to is None
        with hf_tempcache(dir=tmp_dir):
            ds = _add_fingerprint_column(
                self.ds,
                name,
                build_kwargs,
                batch_size=batch_size,
                num_proc=num_proc,
                keep_in_memory=in_memory,
            )
        if save_to is not None:
            ds.save_to_disk(str(save_to))
            ds = Dataset.load_from_disk(str(save_to))
        return type(self)(ds, transform=self.transform)

    def save_to(self, path: str | PathLike) -> None:
        """Persist the underlying HF Dataset to disk as Arrow shards."""
        self.ds.save_to_disk(str(path))

    @staticmethod
    def densify(
        indices: list,
        fp_size: int,
        values: list | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Densify sparse fingerprint rows into a ``(B, fp_size)`` tensor.

        Parameters
        ----------
        indices : list of 1-D Tensor
            Per-row set-bit indices (binary FPs) or non-zero indices
            (count FPs).
        fp_size : int
            Output width.
        values : list of 1-D Tensor, optional
            Per-row counts. If ``None`` (default), output is binary 0/1.
        dtype : torch.dtype
            Output dtype. Default :data:`torch.float32`.
        """
        out = torch.zeros((len(indices), fp_size), dtype=dtype)
        for b, idx in enumerate(indices):
            idx = idx.long()
            if values is None:
                out[b, idx] = 1
            else:
                out[b, idx] = values[b].to(dtype)
        return out

    def collate(self, batch: list[dict]) -> dict:
        """Stack/densify a batch of rows into batched torch tensors.

        Pass as ``collate_fn`` to :class:`torch.utils.data.DataLoader`.
        Fingerprint columns present in :data:`_FP_INFO` are densified to
        ``(B, fp_size)`` tensors using the canonical ``fp_size``; other
        columns pass through as Python lists (``smiles`` always a list of
        strings).
        """
        out: dict = {"smiles": [r["smiles"] for r in batch]}
        skip = {"smiles"}
        for name, spec in _FP_INFO.items():
            if name not in batch[0]:
                continue
            kind, size = spec["kind"], spec["size"]
            if kind == "dense":
                out[name] = torch.stack(
                    [torch.as_tensor(r[name], dtype=torch.float32) for r in batch]
                )
                skip.add(name)
            elif kind == "binary":
                out[name] = self.densify([r[name] for r in batch], size)
                skip.add(name)
            else:  # count
                out[name] = self.densify(
                    [r[name] for r in batch],
                    size,
                    values=[r[f"{name}_values"] for r in batch],
                )
                skip.add(name)
                skip.add(f"{name}_values")
        for k in batch[0]:
            if k in skip:
                continue
            out[k] = [r[k] for r in batch]
        return out
