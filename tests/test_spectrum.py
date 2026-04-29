import numpy as np
import pytest

from metabo_depthcharge.spectrum import (
    DefaultSpectrumProcessor,
    Normalizer,
    PeakFilter,
    SequentialPreprocessor,
    Spectrum,
    Trimmer,
)


MZ = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 1500.0, 2500.0])
INTENSITY = np.array([5.0, 10.0, 50.0, 100.0, 30.0, 5.0, 2.0, 1.0])


def make_spectrum(**kwargs):
    return Spectrum(mz=MZ.copy(), intensity=INTENSITY.copy(), **kwargs)


def test_len():
    assert len(make_spectrum()) == 8


def test_getitem():
    s = make_spectrum()[2:5]
    assert len(s) == 3
    np.testing.assert_array_equal(s.mz, MZ[2:5])


def test_metadata_default_empty():
    assert make_spectrum().metadata == {}


def test_metadata_stored():
    s = make_spectrum(metadata={"adduct": "[M+H]+", "ce": 30.0})
    assert s.metadata["adduct"] == "[M+H]+"
    assert s.metadata["ce"] == 30.0


def test_normalizer_max_is_one():
    out = Normalizer()(make_spectrum())
    assert np.isclose(out.intensity.max(), 1.0)


def test_normalizer_preserves_metadata():
    s = make_spectrum(metadata={"adduct": "[M+H]+"})
    out = Normalizer()(s)
    assert out.metadata["adduct"] == "[M+H]+"


def test_trimmer_removes_out_of_range():
    out = Trimmer(min=0, max=2000)(make_spectrum())
    assert out.mz.max() < 2000
    assert out.mz.min() > 0


def test_peak_filter_max_number():
    out = PeakFilter(max_number=3)(make_spectrum())
    assert len(out) == 3
    # Must keep the 3 highest-intensity peaks
    top3_mz = set(MZ[np.argsort(-INTENSITY)[:3]].tolist())
    assert set(out.mz.tolist()) == top3_mz


def test_peak_filter_min_intensity():
    out = PeakFilter(min_intensity=10.0)(make_spectrum())
    assert (out.intensity >= 10.0).all()


def test_sequential_preprocessor():
    pipeline = SequentialPreprocessor(
        Trimmer(min=0, max=2000),
        PeakFilter(max_number=5),
        Normalizer(),
    )
    out = pipeline(make_spectrum())
    assert len(out) <= 5
    assert np.isclose(out.intensity.max(), 1.0)


def test_default_spectrum_processor():
    out = DefaultSpectrumProcessor(make_spectrum())
    assert len(out) <= 128
    assert np.isclose(out.intensity.max(), 1.0)
    assert out.mz.max() < 2000


def test_torch():
    pytest.importorskip("torch")
    import torch

    t = make_spectrum().torch()
    assert set(t.keys()) == {"mz", "intensity"}
    assert t["mz"].shape[-1] == 8
    assert t["intensity"].dtype == torch.float32
