"""Tests for molecular graph featurization."""

import itertools

import numpy as np
import pytest
import torch

from metabo_depthcharge.chem import Molecule, MoleculeToGraph, graphs


SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "[Na+].[Cl-]"]


def _table(smiles=SMILES):
    return graphs.pack(MoleculeToGraph()([Molecule(s) for s in smiles]), smiles)


def test_featurize_shapes(aspirin_mol):
    out = graphs.featurize(aspirin_mol.mol)
    mol = aspirin_mol.mol
    assert out["atom_key"].shape == (mol.GetNumAtoms(),)
    for key in ("bsrc", "bdst", "bcode"):
        assert out[key].shape == (mol.GetNumBonds(),)


def test_atom_key_is_stable_and_row_determined(aspirin_mol):
    """Same feature row -> same key; the key must not depend on anything else."""
    atoms = list(aspirin_mol.mol.GetAtoms())
    keys = [graphs.atom_key(a) for a in atoms]
    assert keys == [graphs.atom_key(a) for a in atoms]  # deterministic
    rows = [tuple(graphs.atom_features(a)) for a in atoms]
    for i, j in itertools.combinations(range(len(atoms)), 2):
        assert (keys[i] == keys[j]) == (rows[i] == rows[j])


def test_atom_types_reconstruct_feature_rows_exactly():
    """The type table is what the encoder actually consumes -- it must be lossless."""
    mols = [Molecule(s).mol for s in SMILES]
    table = _table()
    types, nptr = table["types"].numpy(), table["nptr"].numpy()
    ids = table["atom_type"].numpy()
    for m, mol in enumerate(mols):
        for a, atom in enumerate(mol.GetAtoms()):
            want = np.asarray(graphs.atom_features(atom), dtype=np.float32)
            assert np.array_equal(types[ids[nptr[m] + a]], want)


def test_atom_types_are_few():
    """The whole point: a handful of distinct rows across many atoms."""
    table = _table()
    assert len(table["types"]) < len(table["atom_type"])


def test_expand_bonds_matches_explicit_construction():
    """Self-loops first, then each bond both ways, with erev pairing the two."""
    table = _table()
    nsize = torch.diff(table["nptr"])
    bsize = torch.diff(table["bptr"])
    esrc, edst, ecode, erev = graphs.expand_bonds(
        nsize, bsize, table["bsrc"].long(), table["bdst"].long(), table["bcode"]
    )
    assert len(esrc) == int((nsize + 2 * bsize).sum())
    # erev is an involution pairing each directed edge with its opposite
    assert torch.equal(erev[erev], torch.arange(len(erev)))
    assert torch.equal(esrc[erev], edst)
    # every atom has exactly one self-loop
    loops = ecode == graphs.SELF_LOOP_CODE
    assert int(loops.sum()) == int(nsize.sum())
    assert torch.equal(esrc[loops], edst[loops])


def test_expand_bonds_keeps_edges_inside_their_own_graph():
    """A batch-local index that leaked across molecules would silently fuse graphs."""
    table = _table()
    nsize = torch.diff(table["nptr"])
    bsize = torch.diff(table["bptr"])
    esrc, edst, _, _ = graphs.expand_bonds(
        nsize, bsize, table["bsrc"].long(), table["bdst"].long(), table["bcode"]
    )
    owner = torch.repeat_interleave(torch.arange(len(nsize)), nsize)
    assert torch.equal(owner[esrc], owner[edst])


def test_bond_codes_never_collide_with_the_self_loop_code():
    """A bond that hashed to SELF_LOOP_CODE would be dropped as a self-loop."""
    for smiles in ["C=C", "C#C", "c1ccccc1", "C1CCCCC1", "CC=CC=CC"]:
        for bond in Molecule(smiles).mol.GetBonds():
            assert 0 <= graphs.bond_code(bond) < graphs.SELF_LOOP_CODE


def test_atom_with_no_bonds_is_featurizable():
    """A lone counter-ion has no bonds; dgllife's own featurizer raises on these."""
    out = graphs.featurize(Molecule("[Na+].[Cl-]").mol)
    assert len(out["atom_key"]) == 2
    assert len(out["bsrc"]) == 0


def test_pack_offsets_match_contents():
    table = _table()
    assert len(table["nptr"]) == len(SMILES) + 1
    assert table["nptr"][-1] == len(table["atom_type"])
    assert table["bptr"][-1] == len(table["bsrc"])
    for i, s in enumerate(SMILES):
        mol = Molecule(s).mol
        assert table["nptr"][i + 1] - table["nptr"][i] == mol.GetNumAtoms()
        assert table["bptr"][i + 1] - table["bptr"][i] == mol.GetNumBonds()


def test_pack_rejects_mismatched_smiles():
    rows = MoleculeToGraph()([Molecule(s) for s in SMILES])
    with pytest.raises(ValueError, match="but"):
        graphs.pack(rows, SMILES[:-1])


def test_single_molecule_returns_its_own_arrays(aspirin_mol):
    out = MoleculeToGraph()(aspirin_mol)
    assert set(out) == set(MoleculeToGraph.KEYS)
    assert out["atom_key"].shape[0] == aspirin_mol.mol.GetNumAtoms()


def test_batched_call_matches_single_calls():
    gen = MoleculeToGraph()
    rows = gen([Molecule(s) for s in SMILES])
    for row, s in zip(rows, SMILES, strict=True):
        one = gen(Molecule(s))
        for key in MoleculeToGraph.KEYS:
            assert np.array_equal(one[key], row[key])


def test_save_load_round_trip(tmp_path):
    table = _table()
    path = tmp_path / "graphs.pt"
    graphs.save(table, path)
    back = graphs.load(path)
    assert back["smiles"] == table["smiles"]
    assert torch.equal(back["atom_type"], table["atom_type"])
    assert torch.equal(back["types"], table["types"])


def test_load_rejects_a_table_missing_keys(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"smiles": ["CCO"]}, path)
    with pytest.raises(ValueError, match="not a graph table"):
        graphs.load(path)
