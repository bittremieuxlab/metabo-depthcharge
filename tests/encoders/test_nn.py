"""Tests for :mod:`metabo_depthcharge.encoders.nn` — generic neural building
blocks (:class:`AttnAggregator`, :class:`ResidualNetwork`, :class:`GrowableEmbedding`)."""

import pytest


torch = pytest.importorskip("torch")

from metabo_depthcharge.encoders import AttnAggregator, ResidualNetwork  # noqa: E402
from metabo_depthcharge.encoders.nn import GrowableEmbedding  # noqa: E402


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
# GrowableEmbedding
# ---------------------------------------------------------------------------


def test_growable_embedding_forward_like_plain_embedding():
    """No mismatch involved: behaves exactly like nn.Embedding."""
    emb = GrowableEmbedding(5, 4, padding_idx=0)
    out = emb(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 4)
    assert torch.equal(out[0], torch.zeros(4))  # padding_idx row


def test_growable_embedding_pads_grown_vocab():
    """Loading a checkpoint saved under a smaller vocab keeps its rows exactly
    and leaves the newly appended rows untouched (not zeroed, not copied)."""
    old_weight = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    checkpoint = {"weight": old_weight.clone()}

    grown = GrowableEmbedding(8, 4, padding_idx=0)
    fresh_new_rows = grown.weight.data[5:8].clone()

    missing, unexpected = grown.load_state_dict(checkpoint, strict=False)

    assert missing == [] and unexpected == []
    assert torch.equal(grown.weight.data[:5], old_weight)
    assert torch.equal(grown.weight.data[5:8], fresh_new_rows)


def test_growable_embedding_does_not_reenforce_padding_idx_zero():
    """Loading does a verbatim row copy -- it does not re-zero padding_idx
    even if the checkpoint's row there happens not to be zero (e.g. a
    hand-edited or foreign checkpoint). This is a real limitation, not a bug:
    a checkpoint's own padding_idx row is zero only because it was never
    trained (autograd masks its gradient), not because loading enforces it."""
    checkpoint = {"weight": torch.ones(3, 4)}  # row 0 deliberately non-zero
    grown = GrowableEmbedding(5, 4, padding_idx=0)
    grown.load_state_dict(checkpoint, strict=False)
    assert torch.equal(grown.weight.data[0], torch.ones(4))


def test_growable_embedding_same_size_is_a_normal_load():
    old_weight = torch.randn(5, 4)
    checkpoint = {"weight": old_weight.clone()}
    same = GrowableEmbedding(5, 4, padding_idx=0)
    same.load_state_dict(checkpoint)  # strict=True default must still pass
    assert torch.equal(same.weight.data, old_weight)


def test_growable_embedding_rejects_dim_change():
    """embedding_dim mismatch is a real incompatibility, not vocab growth --
    must still raise, not silently reshape."""
    checkpoint = {"weight": torch.randn(5, 4)}
    wrong_dim = GrowableEmbedding(8, 6, padding_idx=0)
    with pytest.raises(RuntimeError):
        wrong_dim.load_state_dict(checkpoint, strict=False)


def test_growable_embedding_rejects_shrink():
    """A checkpoint with MORE rows than the current vocab means the vocab
    shrank or got reordered -- never a supported case, must still raise."""
    checkpoint = {"weight": torch.randn(5, 4)}
    smaller = GrowableEmbedding(3, 4, padding_idx=0)
    with pytest.raises(RuntimeError):
        smaller.load_state_dict(checkpoint, strict=False)


def test_growable_embedding_grows_when_nested_in_a_parent_module():
    """Exercises the real usage pattern: GrowableEmbedding as a submodule, so
    _load_from_state_dict runs with a non-empty ``prefix``."""

    class Wrapper(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.adduct_emb = GrowableEmbedding(n, 4, padding_idx=0)

    old_weight = torch.randn(5, 4)
    small = Wrapper(5)
    small.adduct_emb.weight.data.copy_(old_weight)

    big = Wrapper(8)
    missing, unexpected = big.load_state_dict(small.state_dict(), strict=False)

    assert missing == [] and unexpected == []
    assert torch.equal(big.adduct_emb.weight.data[:5], old_weight)
