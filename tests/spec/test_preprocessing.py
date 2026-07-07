"""Tests for :mod:`metabo_depthcharge.spec.preprocessing` — the
:class:`Spectrum` container and the peak-processing transforms."""

import numpy as np
import pytest
import torch

from metabo_depthcharge.spec import (
    CollapseSteppedCE,
    DefaultSpectrumProcessor,
    Normalizer,
    PeakFilter,
    SequentialPreprocessor,
    Spectrum,
    Trimmer,
)


MZ = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 1500.0, 2500.0])
INTENSITY = np.array([5.0, 10.0, 50.0, 100.0, 30.0, 5.0, 2.0, 1.0])
MZ_UNSIGNED = MZ.astype(np.uint16)
INTENSITY_UNSIGNED = INTENSITY.astype(np.uint16)


def make_spectrum(mz, intensity, **kwargs):
    return Spectrum(mz=mz, intensity=intensity, **kwargs)


@pytest.mark.parametrize(
    "mz,intensity", [(MZ, INTENSITY), (MZ_UNSIGNED, INTENSITY_UNSIGNED)]
)
def test_len(mz, intensity):
    assert len(make_spectrum(mz, intensity)) == 8


def test_getitem():
    s = make_spectrum(MZ, INTENSITY)[2:5]
    print(s)
    assert len(s) == 3
    np.testing.assert_array_equal(s.mz, MZ[2:5])
    empty_s = make_spectrum(None, None)
    assert len(empty_s) == 0


def test_plot_runs_without_error():
    s = make_spectrum(MZ, INTENSITY)
    s.plot(as_peaks=True)
    s.plot(as_peaks=False)


def test_metadata_default_empty():
    assert make_spectrum(MZ, INTENSITY).metadata == {}


def test_metadata_stored():
    s = make_spectrum(MZ, INTENSITY, metadata={"adduct": "[M+H]+", "ce": 30.0})
    assert s.metadata["adduct"] == "[M+H]+"
    assert s.metadata["ce"] == 30.0


def test_normalizer_max_is_one():
    out = Normalizer()(make_spectrum(MZ, INTENSITY))
    assert np.isclose(out.intensity.max(), 1.0)


def test_normalizer_preserves_metadata():
    s = make_spectrum(MZ, INTENSITY, metadata={"adduct": "[M+H]+"})
    out = Normalizer()(s)
    assert out.metadata["adduct"] == "[M+H]+"


def test_trimmer_removes_out_of_range():
    out = Trimmer(min=0, max=2000)(make_spectrum(MZ, INTENSITY))
    assert out.mz.max() < 2000
    assert out.mz.min() > 0


def test_peak_filter_max_number():
    out = PeakFilter(max_number=3)(make_spectrum(MZ, INTENSITY))
    assert len(out) == 3
    # Must keep the 3 highest-intensity peaks
    top3_mz = set(MZ[np.argsort(-INTENSITY)[:3]].tolist())
    assert set(out.mz.tolist()) == top3_mz


def test_peak_filter_min_intensity():
    out = PeakFilter(min_intensity=10.0)(make_spectrum(MZ, INTENSITY))
    assert (out.intensity >= 10.0).all()


@pytest.mark.parametrize(
    "metadata, source",
    [
        (
            {
                "COLLISION_ENERGY_1": "10.0",
                "COLLISION_ENERGY_2": 20.0,
                "COLLISION_ENERGY_3": 30.0,
            },
            "absolute",
        ),
        (
            {
                "NORMALIZED_COLLISION_ENERGY_1": 10.0,
                "NORMALIZED_COLLISION_ENERGY_2": 20.0,
                "NORMALIZED_COLLISION_ENERGY_3": 30.0,
            },
            "normalized",
        ),
    ],
)
def test_collapse_stepped_ce(metadata, source):
    s = make_spectrum(MZ, INTENSITY, metadata=metadata)
    out = CollapseSteppedCE(source=source)(s)
    assert np.isclose(out.metadata["collision_energy"], 20)


def test_collapse_stepped_ce_arg():
    with pytest.raises(ValueError):
        CollapseSteppedCE(source="some value")


def test_collapse_stepped_ce_skips_unrelated_and_unparsable_keys():
    metadata = {
        "normalized_collision_energy_1": 10.0,
        "normalized_collision_energy_2": 30.0,
        "normalized_collision_energy_bad": "not-a-number",
        "instrument_type": "orbitrap",
    }
    s = make_spectrum(MZ, INTENSITY, metadata=metadata)
    out = CollapseSteppedCE(source="normalized")(s)
    assert np.isclose(out.metadata["collision_energy"], 20)


def test_sequential_preprocessor():
    pipeline = SequentialPreprocessor(
        Trimmer(min=0, max=2000),
        PeakFilter(max_number=5),
        Normalizer(),
    )
    out = pipeline(make_spectrum(MZ, INTENSITY))
    assert len(out) <= 5
    assert np.isclose(out.intensity.max(), 1.0)


def test_default_spectrum_processor():
    out = DefaultSpectrumProcessor(make_spectrum(MZ, INTENSITY))
    assert len(out) <= 128
    assert np.isclose(out.intensity.max(), 1.0)
    assert out.mz.max() < 2000


def test_torch():
    t = make_spectrum(MZ, INTENSITY).torch()
    assert set(t.keys()) == {"mz", "intensity"}
    assert t["mz"].shape[-1] == 8
    assert t["intensity"].dtype == torch.float32
