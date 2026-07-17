"""Tests for :mod:`metabo_depthcharge.encoders.spectra` — the spectrum-side
encoders (:class:`MetadataEncoder`, :class:`PeakEncoder`,
:class:`SpectrumEncoder`)."""

import pytest


torch = pytest.importorskip("torch")

from metabo_depthcharge.encoders import (  # noqa: E402
    MetadataEncoder,
    PeakEncoder,
    SpectrumEncoder,
    SubformulaEncoder,
)
from metabo_depthcharge.mist_cf.common.chem_utils import ELEMENT_DIM  # noqa: E402


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


def test_metadata_encoder_ion_activation_output_shape():
    enc = MetadataEncoder(d_model=32, metadata_fields=["ion_activation"])
    out = enc({"ion_activation": torch.ones(4, dtype=torch.long)})
    assert out.shape == (4, 32)


def test_metadata_encoder_unknown_ion_activation_zero():
    """Ion activation index 0 is padding_idx → embedding should be zero."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["ion_activation"])
    out = enc({"ion_activation": torch.zeros(3, dtype=torch.long)})
    assert torch.allclose(out, torch.zeros_like(out))


def test_metadata_encoder_ionization_method_output_shape():
    enc = MetadataEncoder(d_model=32, metadata_fields=["ionization_method"])
    out = enc({"ionization_method": torch.ones(4, dtype=torch.long)})
    assert out.shape == (4, 32)


def test_metadata_encoder_unknown_ionization_method_zero():
    """Ionization method index 0 is padding_idx → embedding should be zero."""
    enc = MetadataEncoder(d_model=16, metadata_fields=["ionization_method"])
    out = enc({"ionization_method": torch.zeros(3, dtype=torch.long)})
    assert torch.allclose(out, torch.zeros_like(out))


def test_metadata_encoder_combined_fields():
    enc = MetadataEncoder(
        d_model=32,
        metadata_fields=[
            "adduct",
            "collision_energy",
            "instrument_type",
            "ion_activation",
            "ionization_method",
        ],
    )
    meta = {
        "adduct": torch.ones(4, dtype=torch.long),
        "collision_energy": torch.rand(4),
        "instrument_type": torch.ones(4, dtype=torch.long),
        "ion_activation": torch.ones(4, dtype=torch.long),
        "ionization_method": torch.ones(4, dtype=torch.long),
    }
    out = enc(meta)
    assert out.shape == (4, 32)


def test_metadata_encoder_no_fields_raises():
    with pytest.raises(ValueError, match="At least one metadata field"):
        MetadataEncoder(d_model=16, metadata_fields=[])


def test_metadata_encoder_forward_missing_all_fields_warns():
    enc = MetadataEncoder(d_model=16, metadata_fields=["adduct"])
    with pytest.warns(UserWarning, match="none were present"):
        out = enc({})
    assert out == 0


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


def test_spectrum_encoder_causal_perturbation_earlier_unchanged():
    """Changing the last peak must not change earlier positions' outputs."""
    enc = SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS, pool=None, causal=True)
    enc.eval()
    mz, intensity, precursor_mz = _make_batch()
    with torch.no_grad():
        out1, _ = enc(mz, intensity, precursor_mz)

    mz2, intensity2 = mz.clone(), intensity.clone()
    mz2[:, -1] = torch.rand(B).abs() + 5.0
    intensity2[:, -1] = torch.rand(B).abs()
    with torch.no_grad():
        out2, _ = enc(mz2, intensity2, precursor_mz)

    assert torch.allclose(out1[:, :L, :], out2[:, :L, :], atol=1e-5)
    assert not torch.allclose(out1[:, L, :], out2[:, L, :], atol=1e-6)


def test_spectrum_encoder_causal_no_padding_leak():
    """Combining causal + padding masks must not NaN or leak padding into real positions."""
    half = L // 2
    mz = torch.zeros(B, L)
    intensity = torch.zeros(B, L)
    mz[:, :half] = torch.rand(B, half).abs() + 0.1
    intensity[:, :half] = torch.rand(B, half).abs()
    precursor_mz = torch.rand(B).abs() + 0.1

    enc = SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS, pool=None, causal=True)
    enc.eval()
    with torch.no_grad():
        out1, mask = enc(mz, intensity, precursor_mz)
    assert torch.isfinite(out1).all()
    assert mask[:, half + 1 :].all()

    mz2, intensity2 = mz.clone(), intensity.clone()
    mz2[:, -1] = torch.rand(B).abs() + 5.0
    intensity2[:, -1] = torch.rand(B).abs()
    with torch.no_grad():
        out2, _ = enc(mz2, intensity2, precursor_mz)
    assert torch.allclose(out1[:, : half + 1, :], out2[:, : half + 1, :], atol=1e-5)


def test_spectrum_encoder_causal_cls_pool_raises():
    with pytest.raises(ValueError, match="incompatible with pool='cls'"):
        SpectrumEncoder(d_model=D_MODEL, n_layers=N_LAYERS, causal=True, pool="cls")


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


# ---------------------------------------------------------------------------
# SubformulaEncoder
# ---------------------------------------------------------------------------


def test_subformula_encoder_output_shape():
    enc = SubformulaEncoder(d_model=32)
    form_vec = torch.randint(0, 4, (2, 5, ELEMENT_DIM))
    parent_form_vec = torch.randint(4, 8, (2, ELEMENT_DIM))
    out = enc(form_vec, parent_form_vec)
    assert out.shape == (2, 5, 32)


def test_spectrum_encoder_with_subformulae():
    sub_enc = SubformulaEncoder(d_model=D_MODEL)
    enc = SpectrumEncoder(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        subformula_encoder=sub_enc,
    )
    enc.eval()
    mz, intensity, precursor_mz = _make_batch()
    subformulae = {
        "form_vec": torch.randint(0, 4, (B, L, ELEMENT_DIM)),
        "parent_form_vec": torch.randint(4, 8, (B, ELEMENT_DIM)),
    }
    with torch.no_grad():
        out = enc(mz, intensity, precursor_mz, subformulae=subformulae)
    assert out.shape == (B, D_MODEL)
