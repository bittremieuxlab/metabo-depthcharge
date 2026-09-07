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
from metabo_depthcharge.encoders.spectra import FLARE_ELEMENT_NORM  # noqa: E402
from metabo_depthcharge.mist_cf.common.chem_utils import (  # noqa: E402
    ELEMENT_DIM,
    VALID_ELEMENTS,
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


# --- Configurable peak-embedding paths --------------------------------------


def _spec_batch(b=3, length=6, n_elem=18):
    mz = torch.rand(b, length) * 500 + 50
    intensity = torch.rand(b, length)
    for i in range(b):  # ragged padding, so mask handling is actually exercised
        mz[i, length - i :] = 0.0
        intensity[i, length - i :] = 0.0
    return {
        "mz": mz,
        "intensity": intensity,
        "precursor_mz": mz.max(dim=1).values + 1.0,
        "subformulae": {
            "form_vec": torch.randint(0, 5, (b, length, n_elem)),
            "parent_form_vec": torch.randint(0, 5, (b, n_elem)),
        },
    }


def test_float_norm_flare_preset_maps_by_symbol_and_pads_with_ones():
    """Elements outside FLARE's own 14-element vocabulary must default to 1.0."""
    enc = SubformulaEncoder(64, form_embedder="float", float_norm="flare")
    norm = enc.form_encoder.norm_vec
    assert norm.shape == (len(VALID_ELEMENTS),)
    for el, value in FLARE_ELEMENT_NORM.items():
        assert norm[VALID_ELEMENTS.index(el)] == value
    covered = set(FLARE_ELEMENT_NORM)
    for el in VALID_ELEMENTS:
        if el not in covered:
            assert norm[VALID_ELEMENTS.index(el)] == 1.0


def test_flare_equivalent_subformula_encoder_shape():
    """form_embedder='float' + use_complement=False + mlp_dims reproduces FLARE's path.

    float_norm defaults to "flare", so nothing extra needs to be passed.
    """
    enc = SubformulaEncoder(
        64,
        form_embedder="float",
        use_complement=False,
        mlp_dims=(32, 64),
    ).eval()
    form = torch.randint(0, 5, (3, 6, 18))
    with torch.no_grad():
        out = enc(form, None)
    assert out.shape == (3, 6, 64)


def test_float_norm_mistcf_preset_matches_common_norm_vec():
    from metabo_depthcharge.mist_cf.common.chem_utils import NORM_VEC

    enc = SubformulaEncoder(64, form_embedder="float", float_norm="mistcf")
    assert torch.allclose(
        enc.form_encoder.norm_vec, torch.tensor(NORM_VEC, dtype=torch.float32)
    )


def test_float_norm_custom_sequence():
    custom = [2.0] * len(VALID_ELEMENTS)
    enc = SubformulaEncoder(64, form_embedder="float", float_norm=custom)
    assert torch.allclose(enc.form_encoder.norm_vec, torch.tensor(custom))


def test_float_norm_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown float_norm preset"):
        SubformulaEncoder(64, form_embedder="float", float_norm="bogus")


def test_float_norm_ignored_for_non_float_embedder():
    """A non-'float' embedder shouldn't care what float_norm is (even the default)."""
    enc = SubformulaEncoder(64, form_embedder="abs-sines", float_norm="bogus")
    assert not hasattr(enc.form_encoder, "norm_vec")


def test_mlp_dims_rejects_mismatched_width():
    with pytest.raises(ValueError, match="d_model"):
        SubformulaEncoder(64, mlp_dims=(32, 128))


def test_mlp_dims_with_complement_doubles_input_width():
    enc = SubformulaEncoder(64, "abs-sines", use_complement=True, mlp_dims=(256, 64))
    assert enc.layers[0].in_features == enc.form_encoder.full_dim * 2


def test_subformula_encoder_use_complement_false_ignores_parent():
    enc = SubformulaEncoder(64, "abs-sines", use_complement=False).eval()
    form = torch.randint(0, 5, (3, 6, 18))
    with torch.no_grad():
        a = enc(form, torch.randint(0, 5, (3, 18)))
        b = enc(form, torch.randint(0, 9, (3, 18)))
    assert torch.equal(a, b)


def test_subformula_encoder_use_complement_false_halves_proj_width():
    enc = SubformulaEncoder(64, "abs-sines", use_complement=False)
    assert enc.proj.in_features == enc.form_encoder.full_dim


def test_peak_encoder_use_mz_false_ignores_mz_but_keeps_intensity():
    enc = PeakEncoder(
        d_model=32, min_mz_wavelength=0.001, max_mz_wavelength=10_000, use_mz=False
    )
    mz1, mz2 = torch.rand(4, 10) * 500, torch.rand(4, 10) * 500
    intensity = torch.rand(4, 10)
    x1 = torch.stack([mz1, intensity], dim=2)
    x2 = torch.stack([mz2, intensity], dim=2)
    assert torch.allclose(
        enc(x1), enc(x2)
    )  # different m/z, same intensity -> same output

    x3 = torch.stack([mz1, torch.rand(4, 10)], dim=2)
    assert not torch.allclose(
        enc(x1), enc(x3)
    )  # different intensity -> different output


def test_use_mz_false_keeps_peak_encoder_module():
    """use_mz=False must still build peak_encoder (it still encodes intensity)."""
    enc = SpectrumEncoder(
        d_model=64,
        n_layers=1,
        nhead=2,
        pool=None,
        use_mz=False,
        subformula_encoder=SubformulaEncoder(64, mlp_dims=(32, 64)),
    )
    assert enc.peak_encoder is not None
    assert enc.peak_encoder.mz_encoder is not None
    assert any("peak_encoder" in k for k in enc.state_dict())


def test_use_mz_false_drops_precursor_mz_from_global_token_too():
    """use_mz=False must keep precursor m/z out of the model, not just peak m/z."""
    enc = SpectrumEncoder(
        d_model=64,
        n_layers=1,
        nhead=2,
        pool=None,
        use_mz=False,
        subformula_encoder=SubformulaEncoder(64, mlp_dims=(32, 64)),
    ).eval()
    batch = _spec_batch()
    with torch.no_grad():
        out1, _ = enc(**batch)
        out2, _ = enc(**{**batch, "precursor_mz": batch["precursor_mz"] + 100.0})
    assert torch.allclose(out1, out2, atol=1e-6)


def test_global_token_false_drops_the_cls_and_its_token():
    enc = SpectrumEncoder(
        d_model=64,
        n_layers=1,
        nhead=2,
        pool=None,
        use_mz=False,
        use_global_token=False,
        subformula_encoder=SubformulaEncoder(64, mlp_dims=(32, 64)),
    ).eval()
    assert enc.precursor_cls is None
    batch = _spec_batch()
    with torch.no_grad():
        out, pad = enc(**batch)
    assert out.shape[1] == batch["mz"].shape[1]  # no prepended token


def test_norm_first_builds_a_pre_norm_stack():
    enc = SpectrumEncoder(d_model=64, n_layers=1, nhead=2, norm_first=True)
    assert enc.transformer_encoder.layers[0].norm_first


def test_invalid_configurations_raise():
    mlp = SubformulaEncoder(64, mlp_dims=(32, 64))
    with pytest.raises(ValueError, match="no m/z embedding path"):
        SpectrumEncoder(d_model=64, n_layers=1, nhead=2, use_mz=False)
    with pytest.raises(ValueError, match="pool='cls'"):
        SpectrumEncoder(
            d_model=64,
            n_layers=1,
            nhead=2,
            pool="cls",
            use_global_token=False,
            subformula_encoder=mlp,
        )
    with pytest.raises(ValueError, match="use_global_token=True"):
        SpectrumEncoder(
            d_model=64,
            n_layers=1,
            nhead=2,
            use_global_token=False,
            subformula_encoder=mlp,
            metadata_encoder=MetadataEncoder(64, ["adduct"]),
        )
