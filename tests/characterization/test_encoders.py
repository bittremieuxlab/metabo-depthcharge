"""Characterization tests for metabo_depthcharge.encoders vs spectrawl."""

import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _load_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_spectrawl_path = next(
    p for p in sys.path if (Path(p) / "spectrawl" / "nn" / "transformers.py").exists()
)
_metabo_src_path = next(
    p
    for p in sys.path
    if (Path(p) / "metabo_depthcharge" / "encoders" / "transformers.py").exists()
)

orig = _load_from_file(
    "spectrawl_transformers",
    Path(_spectrawl_path) / "spectrawl" / "nn" / "transformers.py",
)
new = _load_from_file(
    "metabo_encoders",
    Path(_metabo_src_path) / "metabo_depthcharge" / "encoders" / "transformers.py",
)

B = 3
D_MODEL = 32
D_OUT = 16


# ---------------------------------------------------------------------------
# MetadataEncoder
# ---------------------------------------------------------------------------


def test_metadata_encoder_output_matches():
    orig_enc = orig.MetadataEncoder(
        d_model=D_MODEL, metadata_fields=["adduct", "collision_energy"]
    )
    new_enc = new.MetadataEncoder(
        d_model=D_MODEL, metadata_fields=["adduct", "collision_energy"]
    )
    new_enc.load_state_dict(orig_enc.state_dict())

    meta = {
        "adduct": torch.ones(B, dtype=torch.long),
        "collision_energy": torch.rand(B),
    }
    with torch.no_grad():
        out_orig = orig_enc(meta)
        out_new = new_enc(meta)

    torch.testing.assert_close(out_orig, out_new)


# ---------------------------------------------------------------------------
# PeakEncoder
# ---------------------------------------------------------------------------


def test_peak_encoder_output_matches():
    orig_enc = orig.PeakEncoder(
        d_model=D_MODEL, min_mz_wavelength=0.001, max_mz_wavelength=10_000
    )
    new_enc = new.PeakEncoder(
        d_model=D_MODEL, min_mz_wavelength=0.001, max_mz_wavelength=10_000
    )
    new_enc.load_state_dict(orig_enc.state_dict())

    x = torch.rand(B, 8, 2)
    with torch.no_grad():
        out_orig = orig_enc(x)
        out_new = new_enc(x)

    torch.testing.assert_close(out_orig, out_new)


# ---------------------------------------------------------------------------
# AttnAggregator
# ---------------------------------------------------------------------------


def test_attn_aggregator_output_matches():
    orig_agg = orig.AttnAggregator(hidden_dim=D_MODEL)
    new_agg = new.AttnAggregator(hidden_dim=D_MODEL)
    new_agg.load_state_dict(orig_agg.state_dict())

    x = torch.rand(B, 8, D_MODEL)
    with torch.no_grad():
        out_orig = orig_agg(x)
        out_new = new_agg(x)

    torch.testing.assert_close(out_orig, out_new)


# ---------------------------------------------------------------------------
# ResidualProjection
# ---------------------------------------------------------------------------


def test_residual_projection_output_matches():
    orig_proj = orig.ResidualProjection(d_in=D_MODEL, d_out=D_OUT, n_layers=1)
    new_proj = new.ResidualProjection(d_in=D_MODEL, d_out=D_OUT, n_layers=1)
    new_proj.load_state_dict(orig_proj.state_dict())
    orig_proj.eval()
    new_proj.eval()

    x = torch.rand(B, D_MODEL)
    with torch.no_grad():
        out_orig = orig_proj(x)
        out_new = new_proj(x)

    torch.testing.assert_close(out_orig, out_new)


# ---------------------------------------------------------------------------
# DepthchargeEncoder
# ---------------------------------------------------------------------------


def test_depthcharge_encoder_output_matches():
    # spectrawl's DepthchargeEncoder may fail to instantiate if the installed
    # depthcharge version doesn't support attention_backend/rotary_embedding —
    # in that case skip rather than hard-fail.
    try:
        orig_enc = orig.DepthchargeEncoder(d_model=D_MODEL, n_layers=2, d_out=D_OUT)
    except TypeError as e:
        pytest.skip(
            f"spectrawl DepthchargeEncoder incompatible with installed depthcharge: {e}"
        )

    new_enc = new.DepthchargeEncoder(d_model=D_MODEL, n_layers=2, d_out=D_OUT)

    # Copy weights from orig to new (best-effort: only matching keys)
    orig_sd = orig_enc.state_dict()
    new_sd = new_enc.state_dict()
    new_sd.update({k: v for k, v in orig_sd.items() if k in new_sd})
    new_enc.load_state_dict(new_sd)

    mz = torch.rand(B, 8).abs() + 0.1
    intensity = torch.rand(B, 8).abs()
    precursor_mz = torch.rand(B).abs() + 0.1

    orig_enc.eval()
    new_enc.eval()
    with torch.no_grad():
        out_orig = orig_enc(mz, intensity, precursor_mz)
        out_new = new_enc(mz, intensity, precursor_mz)

    assert out_new.shape == (B, D_OUT)
    assert torch.isfinite(out_new).all()
    torch.testing.assert_close(out_orig, out_new)
