"""Tests for :mod:`metabo_depthcharge.encoders.molecules` — the
fingerprint/representation embedders (:class:`MolMLP`, :class:`MultiMolMLP`)."""

import pytest


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
