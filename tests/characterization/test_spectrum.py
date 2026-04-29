"""Characterization tests for metabo_depthcharge.spectrum vs spectrawl."""

import numpy as np
import pytest
import spectrawl.data.spectra as orig
import metabo_depthcharge.spectrum as new


MZ = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 1500.0, 2500.0])
INTENSITY = np.array([5.0, 10.0, 50.0, 100.0, 30.0, 5.0, 2.0, 1.0])


def orig_s():
    return orig.SpectrumObject(mz=MZ.copy(), intensity=INTENSITY.copy())


def new_s():
    return new.SpectrumObject(mz=MZ.copy(), intensity=INTENSITY.copy())


def assert_spectra_equal(o, n):
    np.testing.assert_array_equal(o.mz, n.mz)
    np.testing.assert_array_equal(o.intensity, n.intensity)


def test_normalizer():
    assert_spectra_equal(orig.Normalizer()(orig_s()), new.Normalizer()(new_s()))


def test_trimmer():
    assert_spectra_equal(
        orig.Trimmer(min=0, max=2000)(orig_s()),
        new.Trimmer(min=0, max=2000)(new_s()),
    )


def test_peak_filter_max_number():
    assert_spectra_equal(
        orig.PeakFilter(max_number=3)(orig_s()),
        new.PeakFilter(max_number=3)(new_s()),
    )


def test_peak_filter_min_intensity():
    assert_spectra_equal(
        orig.PeakFilter(min_intensity=10.0)(orig_s()),
        new.PeakFilter(min_intensity=10.0)(new_s()),
    )


def test_sequential_preprocessor():
    pipeline_orig = orig.SequentialPreprocessor(
        orig.Trimmer(min=0, max=2000),
        orig.PeakFilter(max_number=5),
        orig.Normalizer(),
    )
    pipeline_new = new.SequentialPreprocessor(
        new.Trimmer(min=0, max=2000),
        new.PeakFilter(max_number=5),
        new.Normalizer(),
    )
    assert_spectra_equal(pipeline_orig(orig_s()), pipeline_new(new_s()))


def test_default_spectrum_processor():
    assert_spectra_equal(
        orig.DefaultSpectrumProcessor(orig_s()),
        new.DefaultSpectrumProcessor(new_s()),
    )


def test_getitem():
    assert_spectra_equal(orig_s()[2:5], new_s()[2:5])


def test_len():
    assert len(orig_s()) == len(new_s())


def test_torch():
    pytest.importorskip("torch")
    import torch
    o = orig_s().torch()
    n = new_s().torch()
    assert torch.equal(o["mz"], n["mz"])
    assert torch.equal(o["intensity"], n["intensity"])
