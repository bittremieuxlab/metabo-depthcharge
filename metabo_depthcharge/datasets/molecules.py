"""Molecule dataset creation and dataloading utilities."""

import os
import tempfile
import uuid
from collections.abc import Callable, Iterable
from os import PathLike

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from tqdm.auto import tqdm

from metabo_depthcharge.chem.molecule import PROPERTIES, Molecule
from metabo_depthcharge.chem.representations import (
    MoleculeToBiosynfoni,
    MoleculeToChemBERTa,
    MoleculeToMACCS,
    MoleculeToMAP4,
    MoleculeToMolFormer,
    MoleculeToMorgan,
    MoleculeToRdkit,
)
from metabo_depthcharge.datasets._common import hf_silent, hf_tempcache


_FP_INFO: dict[str, dict] = {
    "morgan": {
        "kind": "binary",
        "size": 4096,
        "build": lambda **kw: MoleculeToMorgan(**{"fp_size": 4096, **kw}),
    },
    "morgan_count": {
        "kind": "count",
        "size": 4096,
        "build": lambda **kw: MoleculeToMorgan(
            **{"fp_size": 4096, "counts": True, **kw}
        ),
    },
    "rdkit": {
        "kind": "binary",
        "size": 4096,
        "build": lambda **kw: MoleculeToRdkit(**{"fp_size": 4096, **kw}),
    },
    "rdkit_count": {
        "kind": "count",
        "size": 4096,
        "build": lambda **kw: MoleculeToRdkit(
            **{"fp_size": 4096, "counts": True, **kw}
        ),
    },
    "maccs": {
        "kind": "binary",
        "size": 167,
        "build": lambda **kw: MoleculeToMACCS(**kw),
    },
    "biosynfoni": {
        "kind": "count",
        "size": 39,
        "build": lambda **kw: MoleculeToBiosynfoni(**kw),
    },
    "map4": {
        "kind": "binary",
        "size": 4096,
        "build": lambda **kw: MoleculeToMAP4(**{"fp_size": 4096, **kw}),
    },
    "chemberta": {
        "kind": "dense",
        "size": 768,
        "neural": True,
        # Default model pinned to match the registered size of 768;
        # override via build_kwargs={"model_name": ...}.
        "build": lambda **kw: MoleculeToChemBERTa(
            **{"model_name": "seyonec/ChemBERTa-zinc-base-v1", **kw}
        ),
    },
    "molformer": {
        "kind": "dense",
        "size": 768,
        "neural": True,
        "build": lambda **kw: MoleculeToMolFormer(**kw),
    },
}


class MoleculeDataset(torch.utils.data.Dataset):
    """Molecule dataset backed by an Arrow table, materialized via HF Datasets.

    Build via :meth:`from_smiles`, :meth:`from_csv`, :meth:`from_disk`,
    or wrap an existing HF :class:`Dataset` via ``__init__``.

    Parameters
    ----------
    ds : datasets.Dataset
        A pre-existing HF Dataset.

    Attributes
    ----------
    ds : datasets.Dataset
        The torch-formatted HF Dataset.
    """

    def __init__(self, ds: Dataset):
        self.ds = ds.with_format("torch")

    @classmethod
    def from_csv(
        cls,
        path: str | PathLike,
        *,
        sep: str = ",",
        smiles_column: str = "smiles",
        standardize: bool = False,
        standardize_kwargs: dict | None = None,
        properties: list[str] | None = None,
        recompute_properties: bool = False,
        representations: dict[str, dict] | None = None,
        batch_size: int = 512,
        num_proc: int = 1,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "MoleculeDataset":
        """Parse a CSV/TSV file with molecules in rows, and materialize an Arrow dataset.

        Parameters
        ----------
        path : str or PathLike
            Path to a delimited text file with a header row.
        sep : str
            Field delimiter. Default ``","``; pass ``"\\t"`` for TSV.
        smiles_column : str
            Header name of the SMILES column. Renamed to ``"smiles"`` if
            different.
        standardize : bool, default False
            If True, run :meth:`Molecule.standardize` on each row and
            replace the ``smiles`` column with the result. Rows whose
            standardization returns ``None`` are dropped.
        standardize_kwargs : dict, optional
            Forwarded to :meth:`Molecule.standardize` (e.g.
            ``{"canonicalize_tautomers": True, "remove_stereo": True}``).
        properties : list[str], optional
            Names from :data:`PROPERTIES` to materialize as columns
            (``canonical_smiles``, ``inchi``, ``inchikey``, ``inchikey_2d``,
            ``formula``, ``exact_mass``). Column names match the property
            names exactly; no aliases are recognised.
        recompute_properties : bool, default False
            If False, skip computing a property whose column already
            exists in the input. If True, always (re)compute. Note:
            ``standardize=True`` invalidates pre-existing property
            columns; either drop them first or set this flag.
        representations : dict[str, dict], optional
            Canonical fingerprint column names (keys of :data:`_FP_INFO`)
            mapped to ``build_kwargs`` forwarded to the registered
            builder. Use ``{}`` for defaults, e.g. ``{"morgan": {},
            "chemberta": {"device": "cuda", "model_name": "..."}}``.
        batch_size : int, default 512
            Per-batch row count for :meth:`datasets.Dataset.map` during
            the build pipeline.
        num_proc : int, default 1
            Worker count for :meth:`datasets.Dataset.map`. Neural
            representations are forced in-process so the model is loaded
            once.
        save_to : str or PathLike, optional
            If given, stream the build to disk and persist to this path
            (loadable with :meth:`from_disk`). If ``None`` (default), the
            dataset is built in memory.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults
            to ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset, points at a small ``/tmp``, or resolves to a ``tmpfs``.
        """
        for prop in properties or []:
            if prop not in PROPERTIES:
                raise ValueError(
                    f"Unknown property {prop!r}; choose from {list(PROPERTIES)}"
                )
        in_memory = save_to is None
        nproc = num_proc if num_proc > 1 else None

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

            # Standardize → mutates smiles in place, drops failed rows.
            # Must run before property/fingerprint computation since
            # those derive from the (possibly updated) smiles column.
            if standardize:
                std_kw = standardize_kwargs or {}

                def _std(s):
                    if s is None:
                        return None
                    m = Molecule(s).standardize(**std_kw)
                    return m.canonical_smiles if m is not None else None

                ds = ds.map(
                    lambda b: {"smiles": [_std(s) for s in b["smiles"]]},
                    batched=True,
                    batch_size=batch_size,
                    num_proc=nproc,
                    keep_in_memory=in_memory,
                    desc="Standardizing",
                )
                before = len(ds)
                ds = ds.filter(
                    lambda r: r["smiles"] is not None,
                    num_proc=nproc,
                    keep_in_memory=in_memory,
                )
                if (dropped := before - len(ds)) > 0:
                    tqdm.write(
                        f"Standardization dropped {dropped} row(s) "
                        "that could not be cleaned."
                    )

            needed = [
                p
                for p in properties or []
                if p not in ds.column_names or recompute_properties
            ]
            if needed:

                def _props(batch):
                    mols = [Molecule(s) for s in batch["smiles"]]
                    return {p: [getattr(m, p) for m in mols] for p in needed}

                ds = ds.map(
                    _props,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=nproc,
                    keep_in_memory=in_memory,
                    desc="Computing properties",
                )

            for name, kw in (representations or {}).items():
                ds = cls._add_fingerprint_column(
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
        return cls(ds)

    @classmethod
    def from_smiles(
        cls,
        smiles: Iterable[Molecule | str],
        **kwargs,
    ) -> "MoleculeDataset":
        """Build from an in-memory iterable of SMILES or :class:`Molecule`.

        Convenience wrapper over :meth:`from_csv` for the "no extra
        columns" case: writes the SMILES to a temp file and delegates.
        ``**kwargs`` forward to :meth:`from_csv`; don't pass ``sep`` or
        ``smiles_column``. For datasets with extra columns, use
        :meth:`from_csv` directly or wrap your own HF Dataset.

        Parameters
        ----------
        smiles : iterable of str or Molecule
            Per-row SMILES (mixed types accepted).
        """

        smiles_list = [s.smiles if isinstance(s, Molecule) else s for s in smiles]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("smiles\n")
            f.writelines(s + "\n" for s in smiles_list)
            path = f.name
        try:
            return cls.from_csv(path, **kwargs)
        finally:
            os.unlink(path)

    @classmethod
    def from_disk(cls, path: str | PathLike) -> "MoleculeDataset":
        """Load a previously saved Arrow directory."""
        return cls(Dataset.load_from_disk(str(path)))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        return self.ds[i]

    def col_to_numpy(self, name: str) -> np.ndarray:
        """Return a column as a numpy array."""
        return self.ds.data.column(name).to_numpy(zero_copy_only=False)

    def filter(self, predicate: Callable[[dict], bool], **kwargs) -> "MoleculeDataset":
        """Return a new dataset keeping rows where ``predicate`` is truthy."""
        return type(self)(self.ds.filter(predicate, **kwargs))

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
        """Append a fingerprint column to an already-built dataset.

        Parameters
        ----------
        name : str
            Canonical fingerprint name (key of :data:`_FP_INFO`).
        build_kwargs : dict, optional
            Forwarded to the registered builder (e.g.
            ``{"device": "cuda"}`` for neural FPs).
        batch_size : int, default 512
            Per-batch row count for :meth:`datasets.Dataset.map`.
        num_proc : int, default 1
            Worker count for :meth:`datasets.Dataset.map`. Neural FPs
            are forced in-process.
        save_to : str or PathLike, optional
            If given, persist the resulting dataset to this path.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults
            to ``$TMPDIR``.

        Returns
        -------
        MoleculeDataset
            New dataset with the column added.
        """
        in_memory = save_to is None
        with hf_tempcache(dir=tmp_dir):
            ds = self._add_fingerprint_column(
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
        return type(self)(ds)

    @staticmethod
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
        forwarded to the registered builder (e.g. ``{"device": "cuda"}``
        for neural FPs, ``{"model_name": "..."}`` for ChemBERTa/MolFormer).
        """
        if name not in _FP_INFO:
            raise ValueError(
                f"Unknown fingerprint column {name!r}; "
                f"must be one of {sorted(_FP_INFO)}"
            )
        repr_info = _FP_INFO[name]
        kind = repr_info["kind"]
        fp_size = repr_info["size"]
        build_kwargs = build_kwargs or {}

        nproc = None if repr_info.get("neural") or num_proc <= 1 else num_proc
        rep_cache = [None]

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
            mols = [Molecule(s) for s in batch["smiles"]]
            fps = np.asarray(rep_cache[0](mols))
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

    def save_to(self, path: str | PathLike) -> None:
        """Persist the underlying HF Dataset to disk as Arrow shards."""
        self.ds.save_to_disk(str(path))

    @staticmethod
    def _densify(
        indices: list,
        fp_size: int,
        values: list | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Densify sparse fingerprint rows into a ``(B, fp_size)`` tensor.
        Used by the collater to construct batches of sparse representations.
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
        """
        out = {"smiles": [r["smiles"] for r in batch]}
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
                out[name] = self._densify([r[name] for r in batch], size)
                skip.add(name)
            else:  # count
                out[name] = self._densify(
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
