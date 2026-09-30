"""Molecular graph featurization for graph-based molecule encoders.

Turns RDKit molecules into the flat tensors a message-passing encoder consumes, in a
format compact enough to keep a very large candidate pool resident in memory.

The compact part rests on one observation: an atom's feature row is categorical. Every
entry :func:`atom_features` computes but the leading mass is an indicator bit drawn from
a small, fixed, hardcoded vocabulary (``ELEMENTS``, bond types, degree/valence buckets,
hybridization, ...) -- and mass itself follows from the element. So a whole row is a
handful of small integers, and :func:`atom_code` packs them into a single int64.
:func:`decode_atom_codes` reconstructs the dense row an encoder consumes.

Vocabulary
----------
feature row
    The 78 floats :func:`atom_features` computes for one atom.
atom code
    A single int64 (:func:`atom_code`) a whole feature row packs into: what is
    actually stored per atom and what :func:`decode_atom_codes` unpacks back into
    the dense row an encoder consumes.
packed table
    A whole dataset's graphs as flat tensors plus offsets -- see :data:`TABLE_KEYS`.
bond vs. edge
    A *bond* is stored once, as the molecule has it. An *edge* is directed and exists
    only at batch time: :func:`expand_bonds` turns each atom into one self-loop and
    each bond into two opposed edges.

Pipeline
--------
1. :func:`featurize` -- one molecule at a time, during preprocessing. Emits atom codes
   and bonds, which a dataset stores as columns.
2. :func:`pack` -- once per dataset. Concatenates every molecule into flat tensors.
3. :func:`gather` (many rows of a packed table at once) or :func:`collate` (rows a
   DataLoader already fetched) -- once per batch.
4. :func:`decode_atom_codes` and :func:`expand_bonds` -- inside the encoder, once per
   forward pass.
"""

from collections.abc import Iterable, Sequence
from os import PathLike

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem


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
#: * ``atom_type`` -- ``(A,)`` int64, each atom's self-describing :func:`atom_code`.
#: * ``bsrc``, ``bdst`` -- ``(B,)`` int16, a bond's two atoms, numbered within their own
#:   molecule rather than across the table.
#: * ``bcode`` -- ``(B,)`` uint8, each bond's :func:`bond_code`.
#: * ``nptr``, ``bptr`` -- ``(M + 1,)`` int64 offsets: molecule ``i`` owns atoms
#:   ``[nptr[i]:nptr[i + 1]]`` and bonds ``[bptr[i]:bptr[i + 1]]``.
TABLE_KEYS = (
    "smiles",
    "atom_type",
    "bsrc",
    "bdst",
    "bcode",
    "nptr",
    "bptr",
)

#: (name, getter, allowed values, has an explicit "unknown" catch-all slot) for every
#: pure one-hot field of :func:`atom_features` except the element and bond-types blocks
#: (handled separately in both :func:`atom_features` and :func:`atom_code` -- element
#: because :func:`decode_atom_codes` also recovers mass from it, bond types because they
#: are independent flags rather than "at most one of these"). The single source of truth
#: both :func:`atom_features` and the compact :func:`atom_code`/:func:`decode_atom_codes`
#: pack and unpack from, so the two can never quietly drift apart.
_ONE_HOT_FIELDS = (
    ("degree", lambda a: a.GetDegree(), tuple(range(11)), False),
    ("total_degree", lambda a: a.GetTotalDegree(), tuple(range(6)), False),
    ("explicit_valence", lambda a: a.GetExplicitValence(), tuple(range(1, 7)), False),
    ("implicit_valence", lambda a: a.GetImplicitValence(), tuple(range(7)), False),
    ("hybridization", lambda a: a.GetHybridization(), _HYBRIDIZATIONS, False),
    ("num_hs", lambda a: a.GetTotalNumHs(), tuple(range(5)), False),
    ("formal_charge", lambda a: a.GetFormalCharge(), tuple(range(-2, 3)), False),
    ("radical_electrons", lambda a: a.GetNumRadicalElectrons(), tuple(range(5)), False),
    ("aromatic", lambda a: a.GetIsAromatic(), (False, True), False),
    ("in_ring", lambda a: a.IsInRing(), (False, True), False),
    ("chiral_tag", lambda a: a.GetChiralTag(), _CHIRAL_TAGS, False),
)

#: Standard atomic weight per :data:`ELEMENTS`, plus a trailing 0.0 for the "unknown
#: element" slot -- what :func:`decode_atom_codes` recovers ``atom_features``' leading
#: mass field from, given only the decoded element index. A fixed property of chemistry
#: (via RDKit's own periodic table), not of any dataset. The one thing this loses
#: relative to :func:`atom_features`: an isotope-labeled atom (e.g. deuterium) decodes
#: to its element's ordinary mass rather than that isotope's exact one.
_ATOMIC_WEIGHTS = tuple(
    Chem.GetPeriodicTable().GetAtomicWeight(e) for e in ELEMENTS
) + (0.0,)


def _index(value, allowed: Sequence) -> int:
    """0-based position of ``value`` in ``allowed``, or ``len(allowed)`` if absent.

    Either way the result is exactly the index :func:`_one_hot` would set -- a real
    match, or (for a field without ``unknown=True``) the "no bit set" sentinel one past
    the end -- so this is the single index space :func:`atom_code` packs and
    :func:`decode_atom_codes` unpacks each field through.
    """
    return allowed.index(value) if value in allowed else len(allowed)


def _nbits(n_states: int) -> int:
    """Bits needed to represent ``n_states`` distinct non-negative integers."""
    return max(1, (n_states - 1).bit_length())


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
    :func:`atom_code` pack a whole row into a single small integer.

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
    row = [atom.GetMass() * 0.01]
    row += _one_hot(atom.GetSymbol(), ELEMENTS, unknown=True)
    row += bond_types
    for _, getter, allowed, unknown in _ONE_HOT_FIELDS:
        row += _one_hot(getter(atom), allowed, unknown=unknown)
    return row


def atom_code(atom: Chem.Atom) -> int:
    """Pack one atom's categorical fields into a single self-describing integer.

    Every field is drawn from a small, fixed vocabulary already hardcoded in this
    module (``ELEMENTS``, ``range(11)``, ``_HYBRIDIZATIONS``, ...), so the packed code
    is a pure function of the atom -- nothing here depends on which dataset the atom
    came from, or on any other atom being present. The same kind of atom always packs
    to the same code, anywhere, forever; see :func:`decode_atom_codes` for the inverse
    and the rationale for replacing the old hash-and-deduplicate scheme with this.

    Parameters
    ----------
    atom : rdkit.Chem.Atom
        The atom to code.

    Returns
    -------
    int
        A non-negative integer under ``2**40``, safe to store as int64.
    """
    code = _index(atom.GetSymbol(), ELEMENTS)
    bonds = atom.GetBonds()
    for t in _BOND_TYPES:
        code = (code << 1) | int(any(b.GetBondType() == t for b in bonds))
    for _, getter, allowed, _unknown in _ONE_HOT_FIELDS:
        code = (code << _nbits(len(allowed) + 1)) | _index(getter(atom), allowed)
    return code


def decode_atom_codes(codes: torch.Tensor) -> torch.Tensor:
    """Reconstruct dense feature rows from packed :func:`atom_code` integers.

    A pure, closed-form unpack: no table, no dataset, so it works for any code
    :func:`atom_code` could ever produce -- a molecule pooled today decodes exactly
    like one that will be seen for the first time tomorrow. Run inside the encoder,
    once per batch, in place of the old buffer lookup.

    The one approximation: mass is recovered from the decoded element's *standard*
    atomic weight (:data:`_ATOMIC_WEIGHTS`), not the atom's own possibly
    isotope-specific mass -- an isotope-labeled atom (e.g. deuterium) decodes to its
    element's ordinary mass. Every other field is exact.

    Parameters
    ----------
    codes : torch.Tensor
        ``(A,)`` int64, from :func:`atom_code` (e.g. a table's ``atom_type`` column).

    Returns
    -------
    torch.Tensor
        ``(A, FEAT_DIM)`` float32, matching :func:`atom_features` stacked over the
        same atoms (mass caveat above aside).
    """
    remaining = codes.long()
    blocks = []
    for _, _getter, allowed, _unknown in reversed(_ONE_HOT_FIELDS):
        n = len(allowed)
        bits = _nbits(n + 1)
        idx = remaining & ((1 << bits) - 1)
        remaining = remaining >> bits
        hit = idx < n
        onehot = F.one_hot(idx.clamp(max=n - 1), num_classes=n).float()
        blocks.append(onehot * hit[:, None])
    blocks.reverse()

    bond_bits = []
    for _ in _BOND_TYPES:
        bond_bits.append((remaining & 1).float())
        remaining = remaining >> 1
    bond_bits.reverse()

    elem_idx = (
        remaining  # only the element index is left once every field is peeled off
    )
    elem_onehot = F.one_hot(elem_idx, num_classes=len(ELEMENTS) + 1).float()
    weights = torch.tensor(_ATOMIC_WEIGHTS, dtype=torch.float32, device=codes.device)
    mass = (weights[elem_idx] * 0.01)[:, None]

    return torch.cat(
        [mass, elem_onehot, torch.stack(bond_bits, dim=-1), *blocks], dim=-1
    )


def featurize(mol: Chem.Mol) -> dict[str, np.ndarray]:
    """Reduce one molecule to the small arrays a packed table is built from.

    Atoms become :func:`atom_code` integers rather than feature rows, and bonds are
    kept as the molecule has them -- one entry each, not the self-loops and opposed
    edges an encoder consumes, which :func:`expand_bonds` adds per batch. Both choices
    trade a little work at batch time for a table small enough to stay resident.

    This is the per-molecule stage: it runs in preprocessing workers, and its output is
    what a dataset stores as columns.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule to featurize.

    Returns
    -------
    dict
        ``atom_code`` ``(n_atoms,)`` int64, one per atom in RDKit's atom order, plus
        ``bsrc``/``bdst`` ``(n_bonds,)`` uint16 -- each bond's two atoms, as positions
        in that order -- and ``bcode`` ``(n_bonds,)`` uint8.
    """
    bonds = list(mol.GetBonds())
    return {
        "atom_code": np.array([atom_code(a) for a in mol.GetAtoms()], dtype=np.int64),
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
    candidate pool tractable. Purely a concatenation: each atom's code (from
    :func:`featurize`) already stands on its own, so unlike the old scheme this needs
    nothing about the dataset as a whole and never re-parses a SMILES.

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
    nptr[1:] = np.cumsum([len(r["atom_code"]) for r in rows])
    bptr[1:] = np.cumsum([len(r["bsrc"]) for r in rows])

    def cat(key, dtype):
        return torch.from_numpy(
            np.concatenate([np.asarray(r[key], dtype=dtype) for r in rows])
        )

    return {
        "smiles": smiles,
        "atom_type": cat("atom_code", np.int64),
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
