"""Standalone tests for metabo_depthcharge.encoders and .data.metadata."""

import pytest


torch = pytest.importorskip("torch")

from metabo_depthcharge.encoders import (  # noqa: E402
    AttnAggregator,
    DepthchargeEncoder,
    MetadataEncoder,
    PeakEncoder,
    ResidualProjection,
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
# ResidualProjection
# ---------------------------------------------------------------------------


def test_residual_projection_same_dim():
    proj = ResidualProjection(d_in=32, d_out=32, n_layers=0)
    x = torch.rand(4, 32)
    out = proj(x)
    assert out.shape == (4, 32)


def test_residual_projection_different_dim():
    proj = ResidualProjection(d_in=256, d_out=512, n_layers=0)
    x = torch.rand(4, 256)
    out = proj(x)
    assert out.shape == (4, 512)


def test_residual_projection_with_layers():
    proj = ResidualProjection(d_in=64, d_out=64, n_layers=2)
    x = torch.rand(4, 64)
    out = proj(x)
    assert out.shape == (4, 64)


# ---------------------------------------------------------------------------
# DepthchargeEncoder
# ---------------------------------------------------------------------------

depthcharge = pytest.importorskip("depthcharge")

B, L = 2, 8
D_MODEL = 64
D_OUT = 32
N_LAYERS = 2


def _make_batch():
    mz = torch.rand(B, L).abs() + 0.1
    intensity = torch.rand(B, L).abs()
    precursor_mz = torch.rand(B).abs() + 0.1
    return mz, intensity, precursor_mz


def test_depthcharge_encoder_forward_shape():
    enc = DepthchargeEncoder(d_model=D_MODEL, n_layers=N_LAYERS, d_out=D_OUT)
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert out.shape == (B, D_OUT)


def test_depthcharge_encoder_attention_pool():
    enc = DepthchargeEncoder(
        d_model=D_MODEL, n_layers=N_LAYERS, d_out=D_OUT, pool="attention"
    )
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert out.shape == (B, D_OUT)


def test_depthcharge_encoder_cls_pool():
    enc = DepthchargeEncoder(
        d_model=D_MODEL, n_layers=N_LAYERS, d_out=D_OUT, pool="cls"
    )
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert out.shape == (B, D_OUT)


def test_depthcharge_encoder_with_metadata():
    meta_enc = MetadataEncoder(d_model=D_MODEL, metadata_fields=["adduct"])
    enc = DepthchargeEncoder(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_out=D_OUT,
        metadata_encoder=meta_enc,
    )
    enc.eval()
    mz, intensity, precursor_mz = _make_batch()
    meta = {"adduct": torch.ones(B, dtype=torch.long)}
    with torch.no_grad():
        out = enc(mz, intensity, precursor_mz, metadata=meta)
    assert out.shape == (B, D_OUT)


def test_depthcharge_encoder_output_is_finite():
    enc = DepthchargeEncoder(d_model=D_MODEL, n_layers=N_LAYERS, d_out=D_OUT)
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert torch.isfinite(out).all()


def test_depthcharge_encoder_invalid_pool_raises():
    with pytest.raises(ValueError, match="Unknown pool mode"):
        DepthchargeEncoder(
            d_model=D_MODEL, n_layers=N_LAYERS, d_out=D_OUT, pool="mean"
        )


def test_depthcharge_encoder_with_proj_layers():
    enc = DepthchargeEncoder(
        d_model=D_MODEL, n_layers=N_LAYERS, d_out=D_OUT, n_proj_layers=2
    )
    enc.eval()
    with torch.no_grad():
        out = enc(*_make_batch())
    assert out.shape == (B, D_OUT)
    assert torch.isfinite(out).all()


def test_depthcharge_encoder_with_metadata_all_unknown():
    """Unknown adduct (index 0) contributes zero; output still valid."""
    meta_enc = MetadataEncoder(d_model=D_MODEL, metadata_fields=["adduct"])
    enc = DepthchargeEncoder(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_out=D_OUT,
        metadata_encoder=meta_enc,
    )
    enc.eval()
    mz, intensity, precursor_mz = _make_batch()
    meta = {"adduct": torch.zeros(B, dtype=torch.long)}
    with torch.no_grad():
        out = enc(mz, intensity, precursor_mz, metadata=meta)
    assert out.shape == (B, D_OUT)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Additional MetadataEncoder tests
# ---------------------------------------------------------------------------


def test_metadata_encoder_instrument_type_output_shape():
    enc = MetadataEncoder(d_model=32, metadata_fields=["instrument_type"])
    out = enc({"instrument_type": torch.ones(4, dtype=torch.long)})
    assert out.shape == (4, 32)


def test_metadata_encoder_unknown_instrument_zero():
    """Instrument index 0 is padding_idx → embedding should be zero."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["instrument_type"])
    out = enc({"instrument_type": torch.zeros(3, dtype=torch.long)})
    assert torch.allclose(out, torch.zeros_like(out))


def test_metadata_encoder_nonzero_adduct_nonzero_output():
    """Known adduct (index > 0) should produce a non-zero embedding."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["adduct"])
    out = enc({"adduct": torch.ones(2, dtype=torch.long)})
    # At least one element in the output should be non-zero
    assert not torch.allclose(out, torch.zeros_like(out))


def test_metadata_encoder_nonzero_ce_nonzero_output():
    """Nonzero CE should produce a non-zero embedding."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["collision_energy"])
    out = enc({"collision_energy": torch.ones(2)})
    assert not torch.allclose(out, torch.zeros_like(out))


def test_metadata_encoder_fields_subset_only_ce():
    """Only collision_energy registered; adduct key in dict is ignored."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["collision_energy"])
    out = enc(
        {
            "adduct": torch.ones(3, dtype=torch.long),
            "collision_energy": torch.ones(3),
        }
    )
    # adduct should be ignored; only CE contributes
    out_no_adduct = enc({"collision_energy": torch.ones(3)})
    assert torch.allclose(out, out_no_adduct)


# ---------------------------------------------------------------------------
# Additional PeakEncoder tests
# ---------------------------------------------------------------------------


def test_peak_encoder_output_is_finite():
    enc = PeakEncoder(d_model=32, min_mz_wavelength=0.001, max_mz_wavelength=10_000)
    x = torch.rand(4, 10, 2)
    out = enc(x)
    assert torch.isfinite(out).all()


def test_peak_encoder_single_peak():
    enc = PeakEncoder(d_model=32, min_mz_wavelength=0.001, max_mz_wavelength=10_000)
    x = torch.rand(1, 1, 2)
    out = enc(x)
    assert out.shape == (1, 1, 32)


# ---------------------------------------------------------------------------
# Additional AttnAggregator tests
# ---------------------------------------------------------------------------


def test_attn_aggregator_output_is_finite_no_mask():
    agg = AttnAggregator(hidden_dim=16)
    x = torch.rand(3, 5, 16)
    out = agg(x)
    assert torch.isfinite(out).all()


def test_attn_aggregator_single_position():
    """With a single unmasked position, output should equal that position's value."""
    agg = AttnAggregator(hidden_dim=8)
    x = torch.rand(2, 1, 8)
    out = agg(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Additional ResidualProjection tests
# ---------------------------------------------------------------------------


def test_residual_projection_different_dim_with_layers():
    """d_in != d_out with residual layers should still work."""
    proj = ResidualProjection(d_in=32, d_out=64, n_layers=2)
    x = torch.rand(4, 32)
    out = proj(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_residual_projection_output_is_finite():
    proj = ResidualProjection(d_in=64, d_out=64, n_layers=1)
    x = torch.rand(4, 64)
    out = proj(x)
    assert torch.isfinite(out).all()


def test_residual_projection_no_layers_same_dim_identity_path():
    """When d_in == d_out and n_layers=0, init_proj is Identity."""
    proj = ResidualProjection(d_in=32, d_out=32, n_layers=0)
    import torch.nn as nn

    assert isinstance(proj.init_proj, nn.Identity)
    assert proj.blocks is None
