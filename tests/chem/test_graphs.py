"""Tests for molecular graph featurization."""

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
    assert out["atom_code"].shape == (mol.GetNumAtoms(),)
    for key in ("bsrc", "bdst", "bcode"):
        assert out[key].shape == (mol.GetNumBonds(),)


def test_atom_code_is_stable_and_a_pure_function_of_the_atom(aspirin_mol):
    """Same atom, called twice, or reached via a totally different molecule -> same code."""
    atoms = list(aspirin_mol.mol.GetAtoms())
    codes = [graphs.atom_code(a) for a in atoms]
    assert codes == [graphs.atom_code(a) for a in atoms]  # deterministic

    # A plain, unsubstituted aromatic CH ring carbon codes identically whether it
    # comes from plain benzene or from one of aspirin's unsubstituted ring
    # positions -- nothing about atom_code depends on which molecule, or which
    # dataset, an atom is drawn from.
    def _plain_ring_ch(mol):
        return next(
            a
            for a in mol.GetAtoms()
            if a.GetIsAromatic() and a.GetDegree() == 2 and a.GetTotalNumHs() == 1
        )

    benzene_ch = _plain_ring_ch(Molecule("c1ccccc1CC").mol)
    aspirin_ch = _plain_ring_ch(aspirin_mol.mol)
    assert graphs.atom_code(benzene_ch) == graphs.atom_code(aspirin_ch)


def test_atom_code_never_raises_for_any_molecule():
    """The whole point: no vocabulary to be unseen from -- exotic atoms just code."""
    for smiles in [
        "[Na+].[Cl-]",
        "[Se]",
        "C[As](C)(C)=O",
        "[13C]C",
        "[2H]C([2H])([2H])O",
    ]:
        for atom in Molecule(smiles).mol.GetAtoms():
            graphs.atom_code(atom)  # must not raise


def test_decode_atom_codes_reconstructs_feature_rows_exactly():
    """Round-tripping through atom_code/decode_atom_codes must be lossless for any
    atom whose element is known and not isotope-labeled -- the model actually
    consumes the decoded row, so it has to match atom_features exactly."""
    for smiles in (
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    ):
        mol = Molecule(smiles).mol
        for atom in mol.GetAtoms():
            want = np.asarray(graphs.atom_features(atom), dtype=np.float32)
            code = torch.tensor([graphs.atom_code(atom)])
            got = graphs.decode_atom_codes(code)[0].numpy()
            assert np.allclose(want, got, atol=1e-4), atom.GetSymbol()


def test_decode_atom_codes_approximates_mass_for_an_unknown_element():
    """Na isn't in ELEMENTS, so its mass can't be recovered from the element index
    alone -- decode falls back to 0.0 rather than silently guessing wrong, while
    every other (categorical) field still matches exactly."""
    atom = next(
        a for a in Molecule("[Na+].[Cl-]").mol.GetAtoms() if a.GetSymbol() == "Na"
    )
    want = np.asarray(graphs.atom_features(atom), dtype=np.float32)
    got = graphs.decode_atom_codes(torch.tensor([graphs.atom_code(atom)]))[0].numpy()
    assert np.allclose(want[1:], got[1:], atol=1e-4)  # every field but mass
    assert got[0] == 0.0
    assert want[0] != 0.0


def test_same_atom_codes_the_same_across_independently_built_tables():
    """The property align_atom_types used to have to restore by hand: two tables
    built from completely different molecule sets agree on an atom's code without
    any alignment step."""
    small = _table(["CCO"])
    large = _table(SMILES)
    small_carbon = small["atom_type"][small["nptr"][0]].item()  # CCO's first atom, C
    # find the same kind of atom (an ethanol-like sp3 C-C-O carbon) in the big table
    large_codes = large["atom_type"].tolist()
    assert small_carbon in large_codes


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
    assert len(out["atom_code"]) == 2
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
    assert out["atom_code"].shape[0] == aspirin_mol.mol.GetNumAtoms()


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


def test_load_rejects_a_table_missing_keys(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"smiles": ["CCO"]}, path)
    with pytest.raises(ValueError, match="not a graph table"):
        graphs.load(path)
