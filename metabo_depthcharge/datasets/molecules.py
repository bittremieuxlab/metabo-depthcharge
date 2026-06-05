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
from metabo_depthcharge.encoders.molecules import MolMLP, MultiMolMLP


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
    """Molecule dataset backed by an Arrow table, via HF Datasets.

    Build once via :meth:`from_csv` or :meth:`from_list`, persist to disk,
    and reload with :meth:`from_disk`. Direct instantiation is not supported.

    Example
    -------
    .. code-block:: python

        # build and save (CSV must have a "smiles" column)
        ds = MoleculeDataset.from_csv("molecules.csv", save_to="molecules/")

        # load from previously-built
        ds = MoleculeDataset.from_disk("molecules/")

        # add fingerprint columns
        ds = ds.add_representations({"morgan": None, "rdkit": None}, save_to="molecules/")

        # Usage like torch Dataset:
        len(ds)        # number of molecules
        ds[0]          # dict with "smiles" and any property/fingerprint columns
        ds[0]["smiles"]  # SMILES string

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=32,
            collate_fn=MoleculeDataset.collate
        )

    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            f"Use {type(self).__name__}.from_csv(), .from_list(), or .from_disk() "
            "to construct a MoleculeDataset."
        )

    @classmethod
    def from_csv(
        cls,
        path: str | PathLike,
        *,
        sep: str = ",",
        smiles_column: str = "smiles",
        properties: list[str] | None = None,
        recompute_properties: bool = False,
        batch_size: int = 512,
        num_proc: int = 1,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "MoleculeDataset":
        """Parse a CSV/TSV file with molecules in rows and materialize an Arrow dataset.

        Parameters
        ----------
        path : str or PathLike
            Path to a delimited text file with a header row.
        sep : str
            Field delimiter. Default ``","``; pass ``"\\t"`` for TSV.
        smiles_column : str
            Header name of the SMILES column. Renamed to ``"smiles"`` if
            different.
        properties : list[str], optional
            Properties to materialize as columns
            Any of (``canonical_smiles``, ``inchi``, ``inchikey``,
            `inchikey_2d``, ``formula``, ``exact_mass``).
            Other names/RDKit properties are not implemented.
        recompute_properties : bool, default False
            If False, skip computing a property whose column already exists in
            the input. If True, always (re)compute.
            Note: Rows whose SMILES fails parsing for any requested property are dropped
        batch_size : int, default 512
            Per-batch row count for :meth:`datasets.Dataset.map`.
        num_proc : int, default 1
            Worker count for :meth:`datasets.Dataset.map`.
        save_to : str or PathLike, optional
            If given, persist the dataset to this path (loadable with
            :meth:`from_disk`). If ``None`` (default), built in memory.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults to
            ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset or resolves to a small ``tmpfs``.
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

            needed = [
                p
                for p in properties or []
                if p not in ds.column_names or recompute_properties
            ]
            if needed:
                ds = cls._recompute_properties(
                    ds,
                    needed,
                    batch_size=batch_size,
                    nproc=nproc,
                    in_memory=in_memory,
                    cache_dir=cache_dir,
                )

            if save_to is not None:
                ds.save_to_disk(str(save_to))
                ds = Dataset.load_from_disk(str(save_to))
        return cls._create(ds)

    @classmethod
    def from_list(
        cls,
        smiles: Iterable[Molecule | str],
        **kwargs,
    ) -> "MoleculeDataset":
        """Build from an in-memory iterable of SMILES or :class:`Molecule`.

        Convenience wrapper over :meth:`from_csv` for the "no extra
        columns" case.

        Parameters
        ----------
        smiles : iterable of str or Molecule
            Per-row SMILES (mixed types accepted).
        **kwargs
            Forwarded to :meth:`from_csv`. Accepts ``properties``,
            ``recompute_properties``, ``batch_size``, ``num_proc``,
            ``save_to``, and ``tmp_dir``. Do not pass ``sep`` or
            ``smiles_column``.
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
    def _create(cls, ds: Dataset) -> "MoleculeDataset":
        """
        Private initialization helper method.
        """
        obj = object.__new__(cls)
        obj.ds = ds.with_format("torch")
        return obj

    @classmethod
    def from_disk(cls, path: str | PathLike) -> "MoleculeDataset":
        """
        Load a previously saved ``MoleculeDataset`` from an Arrow directory.

        Parameters
        ----------
        path : str or PathLike
            Directory previously written by :meth:`save_to`,
            :meth:`from_list` / :meth:`from_mgf` with ``save_to=...``.

        Returns
        -------
        MoleculeDataset
        """
        return cls._create(Dataset.load_from_disk(str(path)))

    def save_to(self, path: str | PathLike) -> None:
        """Persist the underlying HF Dataset to disk in a directory of Arrow shards.

        Parameters
        ----------
        path : str or PathLike
            Destination directory. Loadable via :meth:`from_disk`.
        """
        self.ds.save_to_disk(str(path))

    @staticmethod
    def _recompute_properties(
        ds: Dataset,
        properties: list[str],
        *,
        batch_size: int,
        nproc: int | None,
        in_memory: bool,
        cache_dir: str | None = None,
    ) -> Dataset:
        """Drop and recompute ``properties`` columns from the ``smiles`` column.

        Rows whose SMILES fails parsing for any requested property are dropped.
        """
        cols_to_drop = [p for p in properties if p in ds.column_names]
        if cols_to_drop:
            ds = ds.remove_columns(cols_to_drop)

        def _props(batch):
            out = {p: [] for p in properties}
            for s in batch["smiles"]:
                try:
                    m = Molecule(s)
                    vals = {p: getattr(m, p) for p in properties}
                except Exception:
                    vals = {p: None for p in properties}
                for p in properties:
                    out[p].append(vals[p])
            return out

        ds = ds.map(
            _props,
            batched=True,
            batch_size=batch_size,
            num_proc=nproc,
            keep_in_memory=in_memory,
            cache_file_name=None
            if (in_memory or cache_dir is None)
            else os.path.join(cache_dir, uuid.uuid4().hex + ".arrow"),
            desc="Computing properties",
        )
        before = len(ds)
        ds = ds.filter(
            lambda r: all(r[p] is not None for p in properties),
            num_proc=nproc,
            keep_in_memory=in_memory,
            cache_file_name=None
            if (in_memory or cache_dir is None)
            else os.path.join(cache_dir, uuid.uuid4().hex + ".arrow"),
        )
        if (dropped := before - len(ds)) > 0:
            tqdm.write(
                f"Property computation dropped {dropped} row(s) "
                "with unparseable SMILES."
            )
        return ds

    def standardize(
        self,
        *,
        standardize_kwargs: dict | None = None,
        batch_size: int = 512,
        num_proc: int = 1,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "MoleculeDataset":
        """Standardize the ``smiles`` column, dropping rows that fail cleanup.

        Replaces each SMILES with its standardized canonical form via
        :meth:`metabo_depthcharge.chem.Molecule.standardize`. Rows whose standardization returns
        ``None`` are dropped.

        Parameters
        ----------
        standardize_kwargs : dict, optional
            Forwarded to :meth:`metabo_depthcharge.chem.Molecule.standardize` (e.g.
            ``{"canonicalize_tautomers": True, "remove_stereo": True}``).
        batch_size : int, default 512
            Per-batch row count for :meth:`datasets.Dataset.map`.
        num_proc : int, default 1
            Worker count for :meth:`datasets.Dataset.map`.
        save_to : str or PathLike, optional
            If given, persist the result to this path. If ``None`` (default),
            built in memory.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults to
            ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset or resolves to a small ``tmpfs``.
        """
        std_kw = standardize_kwargs or {}
        in_memory = save_to is None
        nproc = num_proc if num_proc > 1 else None

        def _std(s):
            if s is None:
                return None
            m = Molecule(s).standardize(**std_kw)
            return m.canonical_smiles if m is not None else None

        with hf_tempcache(dir=tmp_dir) as cache_dir:
            ds = self.ds.map(
                lambda b: {"smiles": [_std(s) for s in b["smiles"]]},
                batched=True,
                batch_size=batch_size,
                num_proc=nproc,
                keep_in_memory=in_memory,
                cache_file_name=None
                if (in_memory or cache_dir is None)
                else os.path.join(cache_dir, uuid.uuid4().hex + ".arrow"),
                desc="Standardizing",
            )
            before = len(ds)
            ds = ds.filter(
                lambda r: r["smiles"] is not None,
                num_proc=nproc,
                keep_in_memory=in_memory,
                cache_file_name=None
                if (in_memory or cache_dir is None)
                else os.path.join(cache_dir, uuid.uuid4().hex + ".arrow"),
            )
            if (dropped := before - len(ds)) > 0:
                tqdm.write(
                    f"Standardization dropped {dropped} row(s) "
                    "that could not be cleaned."
                )

            # Property columns derived from the old SMILES are now stale; recompute.
            stale = [p for p in PROPERTIES if p in ds.column_names]
            if stale:
                ds = self._recompute_properties(
                    ds,
                    stale,
                    batch_size=batch_size,
                    nproc=nproc,
                    in_memory=in_memory,
                    cache_dir=cache_dir,
                )

            if save_to is not None:
                ds.save_to_disk(str(save_to))
                ds = Dataset.load_from_disk(str(save_to))
        return type(self)._create(ds)

    def add_representations(
        self,
        representations: dict[str, dict | None],
        *,
        batch_size: int = 512,
        num_proc: int = 1,
        save_to: str | PathLike | None = None,
        tmp_dir: str | PathLike | None = None,
    ) -> "MoleculeDataset":
        """Append one or more molecular representation columns.

        Parameters
        ----------
        representations : dict[str, dict | None]
            A dict of molecular representation type names as keys, each mapped to a dict of keyword arguments
            for the builder or ``None`` for defaults.
            Valid names: ``"morgan"``, ``"morgan_count"``, ``"rdkit"``,
            ``"rdkit_count"``, ``"maccs"``, ``"biosynfoni"``, ``"map4"``,
            ``"chemberta"``, ``"molformer"``.
            See :mod:`~metabo_depthcharge.chem.representations` for builder
            keyword arguments.
            E.g. ``{"morgan": None, "chemberta": {"device": "cuda"}}``.
        batch_size : int, default 512
            Per-batch row count for :meth:`datasets.Dataset.map`.
        num_proc : int, default 1
            Worker count for :meth:`datasets.Dataset.map`. Neural
            representations are forced in-process so the model is loaded once.
        save_to : str or PathLike, optional
            If given, persist the result to this path. (loadable with
            :meth:`from_disk`). Pass the same path as the original dataset
            to update it in place.
        tmp_dir : str or PathLike, optional
            Parent directory for the build's transient cache. Defaults to
            ``$TMPDIR``; override on HPC systems where ``$TMPDIR`` is
            unset or resolves to a small ``tmpfs``.
        """
        in_memory = save_to is None
        with hf_tempcache(dir=tmp_dir) as cache_dir:
            ds = self.ds
            for name, kw in representations.items():
                ds = self._add_representation_column(
                    ds,
                    name,
                    kw,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    keep_in_memory=in_memory,
                    cache_dir=cache_dir,
                )
            if save_to is not None:
                ds.save_to_disk(str(save_to))
                ds = Dataset.load_from_disk(str(save_to))
        return type(self)._create(ds)

    @staticmethod
    def _add_representation_column(
        ds: Dataset,
        name: str,
        build_kwargs: dict | None = None,
        *,
        batch_size: int,
        num_proc: int,
        keep_in_memory: bool,
        cache_dir: str | None = None,
    ) -> Dataset:
        """Compute and append representation column(s) onto ``ds`` for ``name``.

        Helper function for add_representation().
        See add_representation() for parameter descriptions.
        """
        if name not in _FP_INFO:
            raise ValueError(
                f"Unknown representation name {name!r}; "
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
            try:
                mols = [Molecule(s) for s in batch["smiles"]]
            except Exception as exc:
                bad = next(
                    (s for s in batch["smiles"] if not isinstance(s, str) or s == ""),
                    None,
                )
                raise ValueError(
                    f"Representation computation failed — dataset contains unparseable "
                    f"SMILES (first bad value: {bad!r}). "
                    "Filter rows with None/invalid SMILES before calling add_representations."
                ) from exc
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
            cache_file_name=None
            if (keep_in_memory or cache_dir is None)
            else os.path.join(cache_dir, uuid.uuid4().hex + ".arrow"),
            new_fingerprint=uuid.uuid4().hex,
            desc=f"Computing {name}",
        )

    def col_to_numpy(self, name: str) -> np.ndarray:
        """Return a dataset column as a numpy array.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        np.ndarray
        """
        return self.ds.data.column(name).to_numpy(zero_copy_only=False)

    def filter(self, condition: Callable[[dict], bool], **kwargs) -> "MoleculeDataset":
        """Return a new dataset keeping only rows where ``condition`` is truthy.

        Parameters
        ----------
        condition : Callable[[dict], bool]
            Row-wise condition as a callable taking a dict and returning a bool.
            The input dict keys are the dataset's columns, e.g. properties.
        **kwargs
            Passed to :meth:`datasets.Dataset.filter`. Common knobs:
            ``num_proc=8`` for parallelism, ``keep_in_memory=True`` to avoid
            writing a cache file.

        Returns
        -------
        MoleculeDataset

        Examples
        --------
        .. code-block:: python

            ds.filter(lambda r: r["exact_mass"] < 1000)
        """
        return type(self)._create(self.ds.filter(condition, **kwargs))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        return self.ds[i]

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

    @staticmethod
    def collate(batch: list[dict]) -> dict:
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
                out[name] = MoleculeDataset._densify([r[name] for r in batch], size)
                skip.add(name)
            else:  # count
                out[name] = MoleculeDataset._densify(
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

    def get_molmlp(
        self,
        fp_names: list[str],
        n_layers: int,
        d_model: int = 512,
        *,
        compute_max_counts: bool = True,
    ):
        """Build a :class:`~metabo_depthcharge.encoders.molecules.MolMLP`
        or :class:`~metabo_depthcharge.encoders.molecules.MultiMolMLP`
        wired to the fingerprint columns present in this dataset.

        Parameters
        ----------
        fp_names : list[str]
            One or more fingerprint column names (keys of :data:`_FP_INFO`).
            A single-element list returns :class:`~metabo_depthcharge.encoders.molecules.MolMLP`;
            two or more return :class:`~metabo_depthcharge.encoders.molecules.MultiMolMLP`.
        n_layers : int
            Depth passed to the underlying
            :class:`~metabo_depthcharge.encoders.molecules.MolMLP`.
        d_model : int, default 512
            Output embedding dimension.
        compute_max_counts : bool, default True
            If ``True``, scan count-fingerprint columns to derive per-bit
            ``max_counts`` tensors used for normalisation inside
            :class:`~metabo_depthcharge.encoders.molecules.MolMLP`.

        Returns
        -------
        MolMLP or MultiMolMLP
        """
        for name in fp_names:
            if name not in _FP_INFO:
                raise ValueError(
                    f"Unknown representation name {name!r}; must be one of {sorted(_FP_INFO)}"
                )
            if name not in self.ds.column_names:
                raise ValueError(
                    f"Column {name!r} not found in this dataset; "
                    "call add_representations() first."
                )

        fp_types = [_FP_INFO[n]["kind"] for n in fp_names]
        fp_sizes = [_FP_INFO[n]["size"] for n in fp_names]

        max_counts: dict[str, torch.Tensor] = {}
        if compute_max_counts:
            for name, kind, size in zip(fp_names, fp_types, fp_sizes, strict=False):
                if kind == "count":
                    max_counts[name] = self._max_counts_from_sparse(self.ds, name, size)

        if len(fp_names) == 1:
            return MolMLP(
                fp_size=fp_sizes[0],
                n_layers=n_layers,
                d_model=d_model,
                fp_type=fp_types[0],
                max_counts=max_counts.get(fp_names[0]),
            )

        return MultiMolMLP(
            fp_names=fp_names,
            fp_sizes=fp_sizes,
            n_layers=n_layers,
            d_model=d_model,
            fp_types=fp_types,
            max_counts=max_counts or None,
        )

    @staticmethod
    def _max_counts_from_sparse(ds: Dataset, name: str, fp_size: int) -> torch.Tensor:
        """Compute per-bit max count across all rows of a sparse count column.

        Used in get_mol_embedder() to normalise count fingerprints for MolMLP.
        """
        flat_idx = (
            ds.data.column(name)
            .combine_chunks()
            .values.to_numpy(zero_copy_only=False)
            .astype(np.intp)
        )
        flat_val = (
            ds.data.column(f"{name}_values")
            .combine_chunks()
            .values.to_numpy(zero_copy_only=False)
            .astype(np.float32)
        )
        max_c = np.zeros(fp_size, dtype=np.float32)
        np.maximum.at(max_c, flat_idx, flat_val)
        return torch.from_numpy(max_c)
