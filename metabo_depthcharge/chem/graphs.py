"""Molecular graph featurization for graph-based molecule encoders.

Turns RDKit molecules into the flat tensors a message-passing encoder consumes, in a
format compact enough to keep a very large candidate pool resident in memory.

The compact part rests on one observation: an atom's feature row is categorical.
:func:`atom_features` returns 78 numbers, but all of them are indicator bits except the
leading mass, which itself follows from the element and isotope. Distinct rows are
therefore few -- around a hundred across millions of real atoms -- so the table holds
each distinct row once and gives every atom a small integer pointing at it. That is two
bytes per atom rather than the 312 the row itself would cost.

Vocabulary
----------
feature row
    The 78 floats :func:`atom_features` computes for one atom.
atom key
    A 64-bit hash of a feature row (:func:`atom_key`). Equal keys mean equal rows. Only
    an intermediate: it lets the per-molecule stage summarize a row without carrying
    it, and lets rows computed in separate worker processes be compared.
atom type id
    A small integer in ``[0, n_types)`` indexing the ``types`` table that
    :func:`build_atom_types` builds. This is what is actually stored per atom.
packed table
    A whole dataset's graphs as flat tensors plus offsets -- see :data:`TABLE_KEYS`.
bond vs. edge
    A *bond* is stored once, as the molecule has it. An *edge* is directed and exists
    only at batch time: :func:`expand_bonds` turns each atom into one self-loop and
    each bond into two opposed edges.

Pipeline
--------
1. :func:`featurize` -- one molecule at a time, during preprocessing. Emits atom keys
   and bonds, which a dataset stores as columns.
2. :func:`pack` -- once per dataset. Concatenates every molecule into flat tensors and
   trades the atom keys for dense atom type ids.
3. :func:`gather` (many rows of a packed table at once) or :func:`collate` (rows a
   DataLoader already fetched) -- once per batch.
4. :func:`expand_bonds` -- inside the encoder, once per forward pass.
"""

import hashlib
from collections.abc import Callable, Iterable, Sequence
from os import PathLike

import numpy as np
import torch
from rdkit import Chem

from metabo_depthcharge.chem.molecule import _lenient_mol_from_smiles


ELEMENTS = ("H", "C", "O", "N", "P", "S", "Cl", "F", "Br", "I", "B", "As", "Si", "Se")

#: The code :func:`expand_bonds` gives a self-loop, one past the 16 real bond codes
SELF_LOOP_CODE = 16

#: Size of a bond-code vocabulary: 16 real codes (4 bond types x conjugated x in-ring)
#: plus :data:`SELF_LOOP_CODE`.
N_BOND_CODES = 17

#: Width of one atom feature row: 1 mass + 77 indicator bits. See :func:`atom_features`.
FEAT_DIM = 78

_BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
_HYBRIDIZATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
_CHIRAL_TAGS = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_OTHER,
)

#: Keys of a packed graph table, for validation on load. For a table of ``M`` molecules
#: holding ``A`` atoms and ``B`` bonds in total:
#:
#: * ``smiles`` -- list of ``M`` str, the molecule each row came from.
#: * ``types`` -- ``(n_types, FEAT_DIM)`` float32, every distinct feature row, once.
#: * ``atom_type`` -- ``(A,)`` uint16, the row of ``types`` each atom has.
#: * ``bsrc``, ``bdst`` -- ``(B,)`` int16, a bond's two atoms, numbered within their own
#:   molecule rather than across the table.
#: * ``bcode`` -- ``(B,)`` uint8, each bond's :func:`bond_code`.
#: * ``nptr``, ``bptr`` -- ``(M + 1,)`` int64 offsets: molecule ``i`` owns atoms
#:   ``[nptr[i]:nptr[i + 1]]`` and bonds ``[bptr[i]:bptr[i + 1]]``.
TABLE_KEYS = (
    "smiles",
    "types",
    "atom_type",
    "bsrc",
    "bdst",
    "bcode",
    "nptr",
    "bptr",
)


def atom_key(atom: Chem.Atom) -> np.int64:
    """Hash an atom's feature row down to a single integer.

    Parameters
    ----------
    atom : rdkit.Chem.Atom
        The atom to key.

    Returns
    -------
    np.int64
        A digest of the atom's feature row.
    """
    row = np.asarray(atom_features(atom), dtype=np.float32)
    digest = hashlib.blake2b(row.tobytes(), digest_size=8).digest()
    # Signed, so that the key sorts and compares identically everywhere it travels.
    return np.int64(int.from_bytes(digest, "little", signed=True))


def build_atom_types(
    keys: np.ndarray, exemplars: "Callable[[int], Chem.Atom]"
) -> tuple[np.ndarray, np.ndarray]:
    """Replace atom keys with small ids into a table of the distinct feature rows.

    Two things come out. ``types`` is the vocabulary: every feature row occurring
    anywhere in ``keys``, listed once, so a row that a million atoms share is stored a
    single time. ``ids`` says which entry of that vocabulary each atom is, such that
    ``types[ids[i]]`` is exactly the row :func:`atom_features` would return for atom
    ``i``. Nothing is approximated -- the pair reconstructs the original rows exactly.

    The ids are *dense* in that they run ``0, 1, ... n_types - 1`` with no gaps. Keys
    are hashes scattered over the whole int64 range and can only be looked up, whereas
    an id indexes ``types`` directly and fits in two bytes. This is the step that makes
    a table small, and it needs the entire dataset at once: which row is number 7
    depends on what else is present.

    Rebuilding the vocabulary needs real feature rows and only hashes were kept, so
    ``exemplars`` supplies one representative atom per distinct key to recompute from.
    That is ``n_types`` calls, not ``n_atoms``.

    Parameters
    ----------
    keys : np.ndarray
        ``(n_atoms,)`` int64 of :func:`atom_key` values, concatenated over molecules.
    exemplars : callable
        Given a position in ``keys``, returns that atom. Called once per distinct key,
        at the first position that key appears, to recompute its row with
        :func:`atom_features`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``types``, the ``(n_types, FEAT_DIM)`` float32 vocabulary, and ``ids``, the
        ``(n_atoms,)`` uint16 entry of it that each atom maps to.

    Raises
    ------
    ValueError
        If more than 65535 distinct rows occur, which a uint16 id cannot address, or if
        two different rows hashed to the same key. Real molecules reach neither --
        around a hundred distinct rows occur across millions of atoms -- but both would
        silently corrupt every graph in the table, so they are checked rather than
        assumed.
    """
    uniq, first, ids = np.unique(keys, return_index=True, return_inverse=True)
    if len(uniq) > np.iinfo(np.uint16).max:
        raise ValueError(f"{len(uniq)} atom types exceeds what a uint16 id can hold")
    types = np.stack(
        [np.asarray(atom_features(exemplars(int(i))), dtype=np.float32) for i in first]
    )
    if len(np.unique(types, axis=0)) != len(types):
        raise ValueError("two distinct atom rows hashed to the same key")
    return types, ids.astype(np.uint16)


def expand_bonds(
    nsize: torch.Tensor,
    bsize: torch.Tensor,
    bsrc: torch.Tensor,
    bdst: torch.Tensor,
    bcode: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild the directed edges a message-passing encoder needs from stored bonds.

    A table keeps bonds as the molecule has them, once each. An encoder instead wants
    edges: directed, and including a self-loop per atom so that an atom's own state
    reaches its update. Storing those would cost the table several times what the bonds
    do, so they are rebuilt per batch -- ``n_atoms + 2 * n_bonds`` of them, laid out per
    graph as every self-loop in atom order, then each bond forwards and backwards. The
    result is exactly what storing the expanded form would have given.

    Indices come out batch-local: ``bsrc``/``bdst`` number atoms within their own
    molecule, and the offset of each graph's block is added here, so the returned
    indices address the batch's concatenated atom tensor directly.

    Parameters
    ----------
    nsize, bsize : torch.Tensor
        ``(K,)`` atom and bond counts per graph.
    bsrc, bdst, bcode : torch.Tensor
        Per-bond endpoints (numbered within their own molecule) and bond codes,
        concatenated over the batch.

    Returns
    -------
    tuple of torch.Tensor
        ``esrc``, ``edst`` and ``ecode`` per edge, plus ``erev``, the position of each
        edge's opposite. D-MPNN needs ``erev`` to subtract the reverse message and so
        keep a message from flowing straight back where it came from; a self-loop is
        its own reverse.
    """
    device = nsize.device
    natoms = int(nsize.sum())
    # Where each graph's atoms and edges start in the concatenated tensors.
    noff = torch.cumsum(nsize, 0) - nsize
    esize = nsize + 2 * bsize
    eoff = torch.cumsum(esize, 0) - esize

    # Self-loops: one per atom, in atom order, at the front of each graph's block.
    loop_graph = torch.repeat_interleave(torch.arange(len(nsize), device=device), nsize)
    loop_at = torch.arange(natoms, device=device)
    loop_pos = eoff[loop_graph] + (loop_at - noff[loop_graph])

    # Bonds: two directed edges each, straight after that graph's self-loops.
    bond_graph = torch.repeat_interleave(torch.arange(len(nsize), device=device), bsize)
    within = torch.arange(len(bsrc), device=device) - torch.repeat_interleave(
        torch.cumsum(bsize, 0) - bsize, bsize
    )
    fwd = eoff[bond_graph] + nsize[bond_graph] + 2 * within
    src_g = bsrc.long() + noff[bond_graph]
    dst_g = bdst.long() + noff[bond_graph]

    n_edges = int(esize.sum())
    esrc = torch.empty(n_edges, dtype=torch.long, device=device)
    edst = torch.empty(n_edges, dtype=torch.long, device=device)
    ecode = torch.empty(n_edges, dtype=torch.long, device=device)
    erev = torch.empty(n_edges, dtype=torch.long, device=device)

    esrc[loop_pos] = edst[loop_pos] = loop_at
    ecode[loop_pos] = SELF_LOOP_CODE
    erev[loop_pos] = loop_pos  # a self-loop is its own reverse

    esrc[fwd], edst[fwd] = src_g, dst_g
    esrc[fwd + 1], edst[fwd + 1] = dst_g, src_g
    ecode[fwd] = ecode[fwd + 1] = bcode.long()
    erev[fwd], erev[fwd + 1] = fwd + 1, fwd

    return esrc, edst, ecode, erev


def _one_hot(value, allowed: Iterable, unknown: bool = False) -> list[bool]:
    """dgllife's ``one_hot_encoding``, including its trailing catch-all slot."""
    hot = [value == a for a in allowed]
    return hot + [value not in allowed] if unknown else hot


def bond_code(bond: Chem.Bond) -> int:
    """Encode a bond's type, conjugation and ring membership as one integer.

    Parameters
    ----------
    bond : rdkit.Chem.Bond
        The bond to encode.

    Returns
    -------
    int
        ``bond_type * 4 + conjugated * 2 + in_ring``, a value in ``[0, 16)`` and so
        disjoint from :data:`SELF_LOOP_CODE`, which is what keeps a real bond from
        being mistaken for a self-loop once :func:`expand_bonds` mixes the two. Bond
        types outside the four supported orders fall back to the single-bond slot.
    """
    kind = bond.GetBondType()
    t = _BOND_TYPES.index(kind) if kind in _BOND_TYPES else 0
    return t * 4 + int(bond.GetIsConjugated()) * 2 + int(bond.IsInRing())


def atom_features(atom: Chem.Atom) -> list[float]:
    """Featurize one atom, reproducing dgllife's ``atom_feature='full'`` block.

    The sole definition of what a feature row is, and so of :data:`FEAT_DIM`. Every
    entry but the leading mass is an indicator bit, and the mass follows from the
    element and isotope -- which is what makes rows categorical, and so what lets
    :func:`atom_key` stand in for a whole row and :func:`build_atom_types` store each
    distinct row only once.

    Parameters
    ----------
    atom : rdkit.Chem.Atom
        The atom to featurize.

    Returns
    -------
    list of float
        :data:`FEAT_DIM` values in ``ConcatFeaturizer`` order: the mass scaled by
        0.01, then one-hot blocks for element, bond types present, degree, total
        degree, explicit and implicit valence, hybridization, attached hydrogens,
        formal charge, radical electrons, aromaticity, ring membership and chirality.
    """
    bonds = atom.GetBonds()
    # dgllife's atom_bond_type_one_hot indexes ``bt[:, i]`` and so raises IndexError on
    # an atom with no bonds at all -- a lone counter-ion, e.g. the "Br." of a
    # hydrobromide. Zeros is the only sensible reading of "no bond type is present".
    bond_types = [any(b.GetBondType() == t for b in bonds) for t in _BOND_TYPES]
    return (
        [atom.GetMass() * 0.01]
        + _one_hot(atom.GetSymbol(), ELEMENTS, unknown=True)
        + bond_types
        + _one_hot(atom.GetDegree(), range(11))
        + _one_hot(atom.GetTotalDegree(), range(6))
        + _one_hot(atom.GetExplicitValence(), range(1, 7))
        + _one_hot(atom.GetImplicitValence(), range(7))
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalNumHs(), range(5))
        + _one_hot(atom.GetFormalCharge(), range(-2, 3))
        + _one_hot(atom.GetNumRadicalElectrons(), range(5))
        + _one_hot(atom.GetIsAromatic(), [False, True])
        + _one_hot(atom.IsInRing(), [False, True])
        + _one_hot(atom.GetChiralTag(), _CHIRAL_TAGS)
    )


def featurize(mol: Chem.Mol) -> dict[str, np.ndarray]:
    """Reduce one molecule to the small arrays a packed table is built from.

    Atoms become :func:`atom_key` hashes rather than feature rows, and bonds are kept as
    the molecule has them -- one entry each, not the self-loops and opposed edges an
    encoder consumes, which :func:`expand_bonds` adds per batch. Both choices trade a
    little work at batch time for a table small enough to stay resident.

    This is the per-molecule stage: it runs in preprocessing workers, and its output is
    what a dataset stores as columns.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule to featurize.

    Returns
    -------
    dict
        ``atom_key`` ``(n_atoms,)`` int64, one per atom in RDKit's atom order, plus
        ``bsrc``/``bdst`` ``(n_bonds,)`` uint16 -- each bond's two atoms, as positions
        in that order -- and ``bcode`` ``(n_bonds,)`` uint8.
    """
    bonds = list(mol.GetBonds())
    return {
        "atom_key": np.array([atom_key(a) for a in mol.GetAtoms()], dtype=np.int64),
        "bsrc": np.array([b.GetBeginAtomIdx() for b in bonds], dtype=np.uint16),
        "bdst": np.array([b.GetEndAtomIdx() for b in bonds], dtype=np.uint16),
        "bcode": np.array([bond_code(b) for b in bonds], dtype=np.uint8),
    }


def pack(per_molecule: Sequence[dict], smiles: Sequence[str]) -> dict:
    """Fuse per-molecule graph arrays into one flat table for a whole dataset.

    Ragged arrays do not stack, so every molecule's atoms and bonds are concatenated end
    to end and the boundaries recorded in the ``nptr``/``bptr`` offsets, leaving each
    molecule recoverable as a slice. A batch of graphs is then gathered with two
    index-selects instead of a Python loop, which is what makes scoring a large
    candidate pool tractable.

    This is also where atom keys become dense ids. It is the first point at which the
    whole dataset is in hand, which is what :func:`build_atom_types` needs to know the
    full set of distinct rows; recomputing a row means having its atom back, so the
    SMILES are re-parsed for the one representative atom per distinct key.

    Parameters
    ----------
    per_molecule : sequence of dict
        One :class:`~metabo_depthcharge.chem.MoleculeToGraph` result per molecule.
    smiles : sequence of str
        The molecules' SMILES, in the same order. Row ``i`` of the table is
        ``smiles[i]``, which is what a dataset's row numbers refer to, so the order
        is meaningful and must match.

    Returns
    -------
    dict
        The packed table -- see :data:`TABLE_KEYS`.

    Raises
    ------
    ValueError
        If the two arguments differ in length.
    """
    rows = list(per_molecule)
    smiles = list(smiles)
    if len(rows) != len(smiles):
        raise ValueError(f"{len(rows)} graphs but {len(smiles)} SMILES")

    nptr = np.zeros(len(rows) + 1, dtype=np.int64)
    bptr = np.zeros(len(rows) + 1, dtype=np.int64)
    nptr[1:] = np.cumsum([len(r["atom_key"]) for r in rows])
    bptr[1:] = np.cumsum([len(r["bsrc"]) for r in rows])
    keys = np.concatenate([np.asarray(r["atom_key"], dtype=np.int64) for r in rows])

    def exemplar(pos: int) -> Chem.Atom:
        """The atom at flat position ``pos``, for recomputing its feature row.

        ``nptr`` says which molecule owns that position; re-parsing gives the atom back.
        """
        row = int(np.searchsorted(nptr, pos, side="right") - 1)
        mol = _lenient_mol_from_smiles(smiles[row])
        return mol.GetAtomWithIdx(pos - int(nptr[row]))

    types, ids = build_atom_types(keys, exemplar)

    def cat(key, dtype):
        return torch.from_numpy(
            np.concatenate([np.asarray(r[key], dtype=dtype) for r in rows])
        )

    return {
        "smiles": smiles,
        "types": torch.from_numpy(types),
        "atom_type": torch.from_numpy(ids),
        "bsrc": cat("bsrc", np.int16),
        "bdst": cat("bdst", np.int16),
        "bcode": cat("bcode", np.uint8),
        "nptr": torch.from_numpy(nptr),
        "bptr": torch.from_numpy(bptr),
    }


def _batch(atom_type, bsrc, bdst, bcode, nsize, bsize) -> dict:
    """Assemble the batched-graph dict every graph encoder consumes.

    The flat tensors arrive already concatenated; this turns the per-graph sizes into
    the ``nptr``/``bptr`` offsets that delimit each graph within them.
    """
    nptr = torch.zeros(len(nsize) + 1, dtype=torch.long)
    bptr = torch.zeros(len(bsize) + 1, dtype=torch.long)
    nptr[1:] = torch.as_tensor(nsize).cumsum(0)
    bptr[1:] = torch.as_tensor(bsize).cumsum(0)
    return {
        "atom_type": atom_type,
        "bsrc": bsrc,
        "bdst": bdst,
        "bcode": bcode,
        "nptr": nptr,
        "bptr": bptr,
    }


def gather(table: dict, indices: torch.Tensor) -> dict:
    """Select rows of a packed table into one batched graph.

    The bulk path, for graphs already in a table: two index-selects over its flat
    tensors, with no Python loop over molecules, which is what makes gathering thousands
    of candidates per step affordable. :func:`collate` is the counterpart for rows a
    DataLoader fetched one at a time.

    Parameters
    ----------
    table : dict
        A packed table from :func:`pack`.
    indices : torch.Tensor
        ``(K,)`` row numbers, in the order the batch should hold them. Repeats are
        allowed -- one molecule can be a candidate for several spectra.

    Returns
    -------
    dict
        A batched graph: the selected molecules' ``atom_type``, ``bsrc``, ``bdst`` and
        ``bcode`` concatenated, plus fresh ``nptr``/``bptr`` offsets delimiting them.
    """
    nsel, nsize = _ragged(table["nptr"], indices)
    bsel, bsize = _ragged(table["bptr"], indices)
    return _batch(
        table["atom_type"][nsel].long(),
        table["bsrc"][bsel].long(),
        table["bdst"][bsel].long(),
        table["bcode"][bsel],
        nsize,
        bsize,
    )


def collate(rows: Sequence[dict]) -> dict:
    """Fuse per-row graph slices from a DataLoader batch into one batched graph.

    The counterpart to :func:`gather` for rows fetched one at a time: same output, other
    input. Ragged per-molecule arrays do not stack, so they are concatenated into flat
    tensors delimited by offsets -- the layout :func:`pack` uses, and the one a graph
    encoder consumes.

    Parameters
    ----------
    rows : sequence of dict
        One graph slice per molecule, as
        :meth:`~metabo_depthcharge.datasets.MoleculeDataset.__getitem__` returns.

    Returns
    -------
    dict
        A batched graph: ``atom_type``, ``bsrc``, ``bdst`` and ``bcode`` concatenated,
        plus the ``nptr``/``bptr`` offsets delimiting each molecule.
    """
    rows = list(rows)

    def cat(key, dtype):
        parts = [torch.as_tensor(r[key]).to(dtype) for r in rows]
        return torch.cat(parts) if parts else torch.zeros(0, dtype=dtype)

    return _batch(
        cat("atom_type", torch.long),
        cat("bsrc", torch.long),
        cat("bdst", torch.long),
        cat("bcode", torch.uint8),
        [len(r["atom_type"]) for r in rows],
        [len(r["bsrc"]) for r in rows],
    )


def _ragged(ptr: torch.Tensor, idx: torch.Tensor):
    """Concatenate the ``[ptr[i]:ptr[i + 1]]`` row ranges of every ``i`` in ``idx``.

    Returns the flat positions to index the table with, and the length of each range,
    both without a Python loop over ``idx``.
    """
    sizes = ptr[idx + 1] - ptr[idx]
    total = int(sizes.sum())
    starts = torch.repeat_interleave(ptr[idx], sizes)
    within = torch.arange(total, device=idx.device) - torch.repeat_interleave(
        torch.cumsum(sizes, 0) - sizes, sizes
    )
    return starts + within, sizes


def save(table: dict, path: str | PathLike) -> None:
    """Write a packed table to disk.

    Parameters
    ----------
    table : dict
        A packed table from :func:`pack`.
    path : str or PathLike
        Destination file.
    """
    torch.save(table, path)


def load(path: str | PathLike) -> dict:
    """Read a packed table, checking it has every key an encoder will index.

    Parameters
    ----------
    path : str or PathLike
        A file written by :func:`save`.

    Returns
    -------
    dict
        The packed table -- see :data:`TABLE_KEYS`.

    Raises
    ------
    ValueError
        If the file is missing any of :data:`TABLE_KEYS`, e.g. because it predates
        the compact format.
    """
    table = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in TABLE_KEYS if k not in table]
    if missing:
        raise ValueError(f"{path} is not a graph table: missing {missing}")
    return table
