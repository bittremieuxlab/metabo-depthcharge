"""Tests for :mod:`metabo_depthcharge.encoders.nn` — generic neural building
blocks (:class:`AttnAggregator`, :class:`ResidualNetwork`)."""

import pytest


torch = pytest.importorskip("torch")

from metabo_depthcharge.encoders import AttnAggregator, ResidualNetwork  # noqa: E402


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
