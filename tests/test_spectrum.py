import numpy as np
import pytest
import torch

from metabo_depthcharge.datasets.spectra import SpectrumDataset
from metabo_depthcharge.spec import (
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
    t = make_spectrum().torch()
    assert set(t.keys()) == {"mz", "intensity"}
    assert t["mz"].shape[-1] == 8
    assert t["intensity"].dtype == torch.float32


# --- SpectrumDataset ----------------------------------------------------


def _spectra_factory():
    """Yield three small spectra; reusable factory for from_spectra."""

    def gen():
        yield Spectrum(mz=MZ.copy(), intensity=INTENSITY.copy(), metadata={"id": "a"})
        yield Spectrum(
            mz=np.array([100.0, 200.0]),
            intensity=np.array([1.0, 2.0]),
            metadata={"id": "b"},
        )
        yield Spectrum(
            mz=np.array([300.0]),
            intensity=np.array([5.0]),
            metadata={"id": "c"},
        )

    return gen


def _make_mgf(path):
    """Write a minimal 2-spectrum MGF file."""
    path.write_text(
        "BEGIN IONS\n"
        "TITLE=spec_one\n"
        "PEPMASS=200.0\n"
        "50.0 5.0\n"
        "100.0 10.0\n"
        "END IONS\n"
        "BEGIN IONS\n"
        "TITLE=spec_two\n"
        "PEPMASS=300.0\n"
        "75.0 3.0\n"
        "150.0 8.0\n"
        "250.0 4.0\n"
        "END IONS\n"
    )


@pytest.fixture
def tiny_mgf(tmp_path):
    path = tmp_path / "tiny.mgf"
    _make_mgf(path)
    return path


def test_from_spectra_basic():
    from datasets import Value as HFValue

    ds = SpectrumDataset.from_spectra(
        _spectra_factory(),
        columns={"id": HFValue("string")},
    )
    assert len(ds) == 3
    row = ds[0]
    assert "mz" in row and "intensity" in row
    assert row["id"] == "a"


def test_from_spectra_with_processor():
    from datasets import Value as HFValue

    ds = SpectrumDataset.from_spectra(
        _spectra_factory(),
        columns={"id": HFValue("string")},
        processor=Normalizer(),
    )
    # Normalized → max intensity is 1.0 per row.
    assert np.isclose(float(ds[0]["intensity"].max()), 1.0)


def test_from_mgf_basic(tiny_mgf):
    ds = SpectrumDataset.from_mgf(tiny_mgf)
    assert len(ds) == 2
    row = ds[0]
    assert "mz" in row and "intensity" in row
    np.testing.assert_array_equal(row["mz"].numpy(), [50.0, 100.0])


def test_spectrum_dataset_from_disk_round_trip(tiny_mgf, tmp_path):
    saved = tmp_path / "saved_spec"
    ds = SpectrumDataset.from_mgf(tiny_mgf, save_to=saved)
    reloaded = SpectrumDataset.from_disk(saved)
    assert len(reloaded) == len(ds)
    assert reloaded.ds.column_names == ds.ds.column_names
    np.testing.assert_array_equal(reloaded[0]["mz"].numpy(), ds[0]["mz"].numpy())


def test_spectrum_dataset_filter(tiny_mgf):
    ds = SpectrumDataset.from_mgf(tiny_mgf)
    only_one = ds.filter(lambda r: len(r["mz"]) == 2)
    assert isinstance(only_one, SpectrumDataset)
    assert len(only_one) == 1


def test_metadata_fields_encoded_and_batched():
    def factory():
        yield Spectrum(
            mz=MZ.copy(),
            intensity=INTENSITY.copy(),
            metadata={
                "adduct": "[M+H]+",
                "collision_energy": "20",
                "instrument_type": "Orbitrap",
            },
        )
        yield Spectrum(
            mz=np.array([100.0, 200.0]),
            intensity=np.array([1.0, 2.0]),
            metadata={
                "adduct": "[M-H]-",
                "collision_energy": "10 20 30",
                "instrument_type": "qtof",
            },
        )
        yield Spectrum(
            mz=np.array([300.0]),
            intensity=np.array([5.0]),
            metadata={},  # all missing
        )

    ds = SpectrumDataset.from_spectra(
        lambda: factory(),
        metadata=["adduct", "collision_energy", "instrument_type"],
    )
    # Canonical scalar dtypes.
    assert ds.ds.features["adduct"].dtype == "int64"
    assert ds.ds.features["collision_energy"].dtype == "float32"
    assert ds.ds.features["instrument_type"].dtype == "int64"

    # Values: known adducts/instruments → nonzero; missing row → 0.
    assert int(ds[0]["adduct"]) == 1  # [M+H]+ → index 1
    assert int(ds[1]["adduct"]) == 6  # [M-H]- → index 6
    assert int(ds[2]["adduct"]) == 0
    # CE divided by 100; stepped CE is mean/100 = 20/100 = 0.2.
    assert float(ds[0]["collision_energy"]) == pytest.approx(0.2)
    assert float(ds[1]["collision_energy"]) == pytest.approx(0.2)
    assert float(ds[2]["collision_energy"]) == 0.0
    assert int(ds[0]["instrument_type"]) == 1  # orbitrap → 1
    assert int(ds[1]["instrument_type"]) == 2  # qtof → 2
    assert int(ds[2]["instrument_type"]) == 0

    # Collate nests metadata into a sub-dict of stacked tensors.
    batch = ds.collate([ds[0], ds[1], ds[2]])
    assert "metadata" in batch
    md = batch["metadata"]
    assert set(md) == {"adduct", "collision_energy", "instrument_type"}
    assert md["adduct"].shape == (3,)
    assert md["adduct"].dtype == torch.int64
    assert md["collision_energy"].dtype == torch.float32
    assert md["instrument_type"].dtype == torch.int64
    assert md["adduct"].tolist() == [1, 6, 0]


def test_metadata_feeds_metadata_encoder():
    from metabo_depthcharge.encoders.transformers import MetadataEncoder

    def factory():
        yield Spectrum(
            mz=MZ.copy(),
            intensity=INTENSITY.copy(),
            metadata={"adduct": "[M+H]+", "collision_energy": "20"},
        )
        yield Spectrum(
            mz=np.array([100.0, 200.0]),
            intensity=np.array([1.0, 2.0]),
            metadata={"adduct": "[M-H]-", "collision_energy": "40"},
        )

    ds = SpectrumDataset.from_spectra(
        lambda: factory(),
        metadata=["adduct", "collision_energy"],
    )
    batch = ds.collate([ds[0], ds[1]])
    enc = MetadataEncoder(d_model=16, metadata_fields=["adduct", "collision_energy"])
    out = enc(batch["metadata"])
    assert out.shape == (2, 16)


def test_spectrum_dataset_collate_pads_ragged(tiny_mgf):
    ds = SpectrumDataset.from_mgf(tiny_mgf)
    batch = ds.collate([ds[0], ds[1]])
    # Two rows: first has 2 peaks, second has 3 → padded to L=3.
    assert batch["mz"].shape == (2, 3)
    assert batch["intensity"].shape == (2, 3)
    assert batch["mask"].shape == (2, 3)
    # Mask should mark trailing pad as False on the shorter row.
    assert batch["mask"][0].tolist() == [True, True, False]
    assert batch["mask"][1].tolist() == [True, True, True]
