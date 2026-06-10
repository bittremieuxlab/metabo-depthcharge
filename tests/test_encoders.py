"""Standalone tests for metabo_depthcharge.encoders and .data.metadata."""

import pytest


torch = pytest.importorskip("torch")

from metabo_depthcharge.encoders import (  # noqa: E402
    AttnAggregator,
    MetadataEncoder,
    MolMLP,
    MultiMolMLP,
    PeakEncoder,
    ResidualNetwork,
    SpectrumEncoder,
)


# ---------------------------------------------------------------------------
# MetadataEncoder
# ---------------------------------------------------------------------------


def test_metadata_encoder_adduct_output_shape():
    enc = MetadataEncoder(d_model=32, metadata_fields=["adduct"])
    out = enc({"adduct": torch.zeros(4, dtype=torch.long)})
    assert out.shape == (4, 32)


def test_metadata_encoder_unknown_adduct_zero():
    """Adduct index 0 is padding_idx → embedding should be zero."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["adduct"])
    out = enc({"adduct": torch.zeros(3, dtype=torch.long)})
    assert torch.allclose(out, torch.zeros_like(out))


def test_metadata_encoder_ce_output_shape():
    enc = MetadataEncoder(d_model=32, metadata_fields=["collision_energy"])
    out = enc({"collision_energy": torch.rand(4)})
    assert out.shape == (4, 32)


def test_metadata_encoder_ce_missing_zero():
    """CE value 0.0 should be masked to zero contribution."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["collision_energy"])
    out = enc({"collision_energy": torch.zeros(2)})
    assert torch.allclose(out, torch.zeros_like(out))


def test_metadata_encoder_combined_fields():
    enc = MetadataEncoder(
        d_model=32,
        metadata_fields=["adduct", "collision_energy", "instrument_type"],
    )
    meta = {
        "adduct": torch.ones(4, dtype=torch.long),
        "collision_energy": torch.rand(4),
        "instrument_type": torch.ones(4, dtype=torch.long),
    }
    out = enc(meta)
    assert out.shape == (4, 32)


def test_metadata_encoder_empty_metadata_returns_zeros():
    """Passing an empty dict when adduct field is registered → zeros."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["adduct"])
    # Provide a dummy tensor so the shape can be inferred
    dummy = torch.zeros(3, dtype=torch.long)
    out = enc({"adduct": dummy})
    # with all-zero adduct indices (padding_idx=0) output should be zero
    assert out.shape == (3, 16)
    assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# PeakEncoder
# ---------------------------------------------------------------------------


def test_peak_encoder_output_shape():
    enc = PeakEncoder(d_model=32, min_mz_wavelength=0.001, max_mz_wavelength=10_000)
    x = torch.rand(4, 10, 2)  # (B, L, 2)
    out = enc(x)
    assert out.shape == (4, 10, 32)


# ---------------------------------------------------------------------------
# AttnAggregator
# ---------------------------------------------------------------------------


def test_attn_aggregator_output_shape():
    agg = AttnAggregator(hidden_dim=32)
    x = torch.rand(4, 10, 32)  # (B, L, D)
    out = agg(x)
    assert out.shape == (4, 32)


def test_attn_aggregator_with_mask():
    """Mask second half of positions; output should still be finite."""
    B, L, D = 3, 8, 16
    agg = AttnAggregator(hidden_dim=D)
    x = torch.rand(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, L // 2 :] = True  # mask out last half
    out = agg(x, mask=mask)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()


def test_attn_aggregator_no_mask():
    agg = AttnAggregator(hidden_dim=24)
    x = torch.rand(2, 6, 24)
    out = agg(x)
    assert out.shape == (2, 24)


# ---------------------------------------------------------------------------
# ResidualNetwork
# ---------------------------------------------------------------------------


def test_residual_network_same_dim():
    proj = ResidualNetwork(d_in=32, d_out=32, n_blocks=0)
    x = torch.rand(4, 32)
    out = proj(x)
    assert out.shape == (4, 32)


def test_residual_network_different_dim():
    proj = ResidualNetwork(d_in=256, d_out=512, n_blocks=0)
    x = torch.rand(4, 256)
    out = proj(x)
    assert out.shape == (4, 512)


def test_residual_network_with_layers():
    proj = ResidualNetwork(d_in=64, d_out=64, n_blocks=2)
    x = torch.rand(4, 64)
    out = proj(x)
    assert out.shape == (4, 64)


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


# ---------------------------------------------------------------------------
# SpectrumEncoder
# ---------------------------------------------------------------------------

depthcharge = pytest.importorskip("depthcharge")

B, L = 2, 8
D_MODEL = 64
N_LAYERS = 2


def _make_batch():
    mz = torch.rand(B, L).abs() + 0.1
    intensity = torch.rand(B, L).abs()
    precursor_mz = torch.rand(B).abs() + 0.1
    return mz, intensity, precursor_mz


def test_spectrum_encoder_forward_shape():
    enc = SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS)
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert out.shape == (B, D_MODEL)


def test_spectrum_encoder_attention_pool():
    enc = SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS, pool="attention")
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert out.shape == (B, D_MODEL)


def test_spectrum_encoder_cls_pool():
    enc = SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS, pool="cls")
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert out.shape == (B, D_MODEL)


def test_spectrum_encoder_no_pool_returns_sequence_and_mask():
    enc = SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS, pool=None)
    enc.eval()
    with torch.no_grad():
        out, mask = enc(*_make_batch())
    assert out.shape == (B, L + 1, D_MODEL)
    assert mask.shape == (B, L + 1)
    assert mask.dtype == torch.bool


def test_spectrum_encoder_invalid_pool():
    with pytest.raises(ValueError, match="Unknown pool mode"):
        SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS, pool="max")


def test_spectrum_encoder_with_metadata():
    meta_enc = MetadataEncoder(d_model=D_MODEL, metadata_fields=["adduct"])
    enc = SpectrumEncoder(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        metadata_encoder=meta_enc,
    )
    enc.eval()
    mz, intensity, precursor_mz = _make_batch()
    meta = {"adduct": torch.ones(B, dtype=torch.long)}
    with torch.no_grad():
        out = enc(mz, intensity, precursor_mz, metadata=meta)
    assert out.shape == (B, D_MODEL)
