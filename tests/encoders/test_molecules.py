"""Tests for :mod:`metabo_depthcharge.encoders.molecules` — the
fingerprint/representation embedders (:class:`MolMLP`, :class:`MultiMolMLP`)."""

import pytest

from metabo_depthcharge.chem import Molecule, MoleculeToGraph, graphs
from metabo_depthcharge.encoders import BondMolEncoder, GraphMolEncoder


torch = pytest.importorskip("torch")

from metabo_depthcharge.encoders import MolMLP, MultiMolMLP  # noqa: E402


# ---------------------------------------------------------------------------
# MolMLP
# ---------------------------------------------------------------------------


def test_mol_embedder_binary():
    emb = MolMLP(rep_size=64, n_blocks=2, d_model=32, rep_type="binary")
    x = torch.randint(0, 2, (4, 64)).float()
    assert emb(x).shape == (4, 32)


def test_mol_embedder_count():
    max_c = torch.rand(64) * 5 + 1
    emb = MolMLP(
        rep_size=64, n_blocks=2, d_model=32, rep_type="count", max_counts=max_c
    )
    x = torch.randint(0, 5, (4, 64)).float()
    assert emb(x).shape == (4, 32)


def test_mol_embedder_dense():
    emb = MolMLP(rep_size=64, n_blocks=2, d_model=32, rep_type="dense")
    x = torch.randn(4, 64)
    assert emb(x).shape == (4, 32)


def test_mol_embedder_zero_blocks_identity():
    # n_blocks=0 with rep_size == d_model is an exact pass-through (no final norm)
    emb = MolMLP(rep_size=32, n_blocks=0, d_model=32, rep_type="binary")
    x = torch.rand(4, 32)
    out = emb(x)
    assert out.shape == (4, 32)
    assert torch.allclose(out, x)


def test_mol_embedder_zero_blocks_projection():
    # n_blocks=0 with rep_size != d_model is a single linear projection
    emb = MolMLP(rep_size=64, n_blocks=0, d_model=32, rep_type="binary")
    x = torch.rand(4, 64)
    assert emb(x).shape == (4, 32)


# ---------------------------------------------------------------------------
# MultiMolMLP
# ---------------------------------------------------------------------------


def test_multi_mol_embedder_output_shape():
    emb = MultiMolMLP(
        rep_names=["fp1", "fp2"],
        rep_sizes=[64, 32],
        n_blocks=2,
        d_model=32,
    )
    fps = {"fp1": torch.rand(4, 64), "fp2": torch.rand(4, 32)}
    assert emb(fps).shape == (4, 32)


def test_multi_mol_embedder_mixed_types():
    max_c = torch.rand(64) * 5 + 1
    emb = MultiMolMLP(
        rep_names=["bin", "cnt", "dns"],
        rep_sizes=[64, 64, 32],
        n_blocks=2,
        d_model=32,
        rep_types=["binary", "count", "dense"],
        max_counts={"cnt": max_c},
    )
    fps = {
        "bin": torch.randint(0, 2, (3, 64)).float(),
        "cnt": torch.randint(0, 5, (3, 64)).float(),
        "dns": torch.randn(3, 32),
    }
    assert emb(fps).shape == (3, 32)


# --- Graph molecule encoders ---------------------------------------------


GRAPH_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
]


@pytest.fixture
def graph_table():
    mols = [Molecule(s) for s in GRAPH_SMILES]
    return graphs.pack(MoleculeToGraph()(mols), GRAPH_SMILES)


@pytest.fixture
def atom_types(graph_table):
    return graph_table["types"]


@pytest.fixture
def idx(graph_table):
    """A batched graph covering every molecule, which is what an encoder takes."""
    return graphs.gather(graph_table, torch.arange(len(GRAPH_SMILES)))


def test_pooled_shape(atom_types, idx):
    enc = GraphMolEncoder(atom_types, [32, 64], 0.0, 64).eval()
    assert enc(idx).shape == (len(GRAPH_SMILES), 64)


def test_per_atom_returns_tokens_and_mask(atom_types, idx):
    enc = GraphMolEncoder(atom_types, [32, 64], 0.0, 64, per_atom=True).eval()
    tokens, pad = enc(idx)
    assert tokens.shape[:2] == pad.shape
    for i, s in enumerate(GRAPH_SMILES):
        assert (~pad[i]).sum() == Molecule(s).mol.GetNumAtoms()


@pytest.mark.parametrize("channels", [[32, 64], [32, 48]])
def test_padding_rows_stay_zero(atom_types, idx, channels):
    """Regression: `out` is a biased Linear when channels[-1] != d_model, and applying
    it AFTER the scatter turns every padding row into `bias` -- which a scorer that
    infers validity from nonzero rows then reads as a real atom."""
    enc = GraphMolEncoder(atom_types, channels, 0.0, 64, per_atom=True).eval()
    tokens, pad = enc(idx)
    assert pad.any(), "test needs ragged sizes to be meaningful"
    assert tokens[pad].abs().sum() == 0


def test_pooled_equals_mean_of_per_atom_tokens(atom_types, idx):
    torch.manual_seed(0)
    pooled = GraphMolEncoder(atom_types, [32, 64], 0.0, 64).eval()
    torch.manual_seed(0)
    per_atom = GraphMolEncoder(atom_types, [32, 64], 0.0, 64, per_atom=True).eval()
    with torch.no_grad():
        a = pooled(idx)
        tokens, pad = per_atom(idx)
    keep = (~pad)[..., None]
    b = (tokens * keep).sum(1) / keep.sum(1)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("norm", ["batch", "layer"])
def test_norm_choices_build_and_run(atom_types, idx, norm):
    enc = GraphMolEncoder(atom_types, [32, 64], 0.0, 64, norm=norm).eval()
    assert enc(idx).shape == (len(GRAPH_SMILES), 64)


def test_unknown_norm_raises(atom_types):
    with pytest.raises(ValueError, match="Unknown norm"):
        GraphMolEncoder(atom_types, [32], 0.0, 32, norm="group")


def test_bond_encoder_shapes(atom_types, idx):
    enc = BondMolEncoder(atom_types, [32, 64], 0.0, 64, per_atom=True).eval()
    tokens, pad = enc(idx)
    for i, s in enumerate(GRAPH_SMILES):
        assert (~pad[i]).sum() == Molecule(s).mol.GetNumAtoms()


def test_bond_tokens_emit_one_token_per_bond(atom_types, idx):
    enc = BondMolEncoder(atom_types, [32, 64], 0.0, 64, bond_tokens=True).eval()
    tokens, pad = enc(idx)
    for i, s in enumerate(GRAPH_SMILES):
        assert (~pad[i]).sum() == Molecule(s).mol.GetNumBonds()
    assert tokens[pad].abs().sum() == 0
