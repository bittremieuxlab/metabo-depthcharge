"""Tests for :class:`metabo_depthcharge.datasets.spectra.SpectrumDataset`"""

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Value

from metabo_depthcharge.datasets.spectra import SpectrumDataset
from metabo_depthcharge.mist_cf.common.chem_utils import formula_mass, ion_to_mass
from metabo_depthcharge.spec import Normalizer
from metabo_depthcharge.spec.adducts import encode_adduct
from metabo_depthcharge.spec.subformulae import ELEMENT_DIM, formula_to_dense


MZ = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 1500.0, 2500.0])
INTENSITY = np.array([5.0, 10.0, 50.0, 100.0, 30.0, 5.0, 2.0, 1.0])

SPECTRA_LIST = [
    np.array([MZ.copy(), INTENSITY.copy()]),
    np.array([[100.0, 200.0], [1.0, 2.0]]),
    np.array([[300.0], [5.0]]),
]
PRECURSOR_MZ_LIST = [150.0, 250.0, 350.0]


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


@pytest.fixture
def tiny_mgf_with_metadata(tmp_path):
    path = tmp_path / "tiny_meta.mgf"
    path.write_text(
        "BEGIN IONS\n"
        "PEPMASS=200.0\n"
        "ADDUCT=[M+H]+\n"
        "COLLISION_ENERGY=20\n"
        "INSTRUMENT_TYPE=Orbitrap\n"
        "50.0 5.0\n100.0 10.0\n150.0 3.0\n"
        "END IONS\n"
        "BEGIN IONS\n"
        "PEPMASS=300.0\n"
        "ADDUCT=[M-H]-\n"
        "COLLISION_ENERGY=10 20 30\n"
        "INSTRUMENT_TYPE=qtof\n"
        "75.0 3.0\n150.0 8.0\n"
        "END IONS\n"
        "BEGIN IONS\n"
        "PEPMASS=400.0\n"
        "80.0 1.0\n"
        "END IONS\n"
    )
    return path


FORMULA = "C9H8O4"
ADDUCT = "[M+H]+"
MATCHED_MZ = formula_mass(FORMULA) + ion_to_mass[ADDUCT]


@pytest.fixture
def subformula_source(tmp_path):
    """Formula/adduct per spectrum in `ds_with_subformulae`: row 0 matches, row 1 is empty."""
    path = tmp_path / "formulae.tsv"
    pd.DataFrame({"formula": [FORMULA, ""], "adduct": [ADDUCT, ""]}).to_csv(
        path, sep="\t", index=False
    )
    return path


@pytest.fixture
def ds_with_subformulae(subformula_source):
    ds = SpectrumDataset.from_list(
        [
            np.array(
                [[MATCHED_MZ, 9999.0], [10.0, 1.0]]
            ),  # peak 0 matches, peak 1 doesn't
            np.array([[123.4], [2.0]]),  # no formula -> no match
        ]
    )
    return ds.add_subformulae("gt", source=subformula_source)


def test_from_list_basic():
    ds = SpectrumDataset.from_list(SPECTRA_LIST, precursor_mz=PRECURSOR_MZ_LIST)
    assert len(ds) == 3
    row = ds[0]
    assert "mz" in row and "intensity" in row
    np.testing.assert_array_almost_equal(row["mz"].numpy(), MZ)
    assert float(ds[0]["precursor_mz"]) == pytest.approx(150.0)


def test_from_list_with_processor():
    ds = SpectrumDataset.from_list(SPECTRA_LIST, processor=Normalizer())
    # Normalized → max intensity is 1.0 per row.
    assert np.isclose(float(ds[0]["intensity"].max()), 1.0)


def test_from_list_default_precursor_mz_is_zero():
    ds = SpectrumDataset.from_list(SPECTRA_LIST)
    assert float(ds[0]["precursor_mz"]) == 0.0


def test_from_list_precursor_mz_length_mismatch_raises():
    with pytest.raises(ValueError, match="precursor_mz"):
        SpectrumDataset.from_list(SPECTRA_LIST, precursor_mz=[1.0, 2.0])


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


def test_metadata_fields_encoded_and_batched(tiny_mgf_with_metadata):
    ds = SpectrumDataset.from_mgf(
        tiny_mgf_with_metadata,
        metadata=["adduct", "collision_energy", "instrument_type"],
        columns={},
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


def test_ion_activation_and_ionization_method_encoded(tmp_path):
    path = tmp_path / "tiny_ia.mgf"
    path.write_text(
        "BEGIN IONS\n"
        "PEPMASS=200.0\n"
        "ION_ACTIVATION=HCD\n"
        "IONIZATION_METHOD=ESI\n"
        "50.0 5.0\n100.0 10.0\n"
        "END IONS\n"
        "BEGIN IONS\n"
        "PEPMASS=300.0\n"
        "ION_ACTIVATION=CID\n"
        "IONIZATION_METHOD=NSI\n"
        "75.0 3.0\n"
        "END IONS\n"
        "BEGIN IONS\n"
        "PEPMASS=400.0\n"
        "80.0 1.0\n"
        "END IONS\n"
    )
    ds = SpectrumDataset.from_mgf(
        path,
        metadata=["ion_activation", "ionization_method"],
        columns={},
    )
    assert ds.ds.features["ion_activation"].dtype == "int64"
    assert ds.ds.features["ionization_method"].dtype == "int64"

    assert int(ds[0]["ion_activation"]) == 1  # HCD → 1
    assert int(ds[1]["ion_activation"]) == 2  # CID → 2
    assert int(ds[2]["ion_activation"]) == 0  # missing → 0
    assert int(ds[0]["ionization_method"]) == 2  # ESI → 2
    assert int(ds[1]["ionization_method"]) == 1  # NSI → 1
    assert int(ds[2]["ionization_method"]) == 0  # missing → 0


def test_metadata_feeds_metadata_encoder(tiny_mgf_with_metadata):
    from metabo_depthcharge.encoders.spectra import MetadataEncoder

    ds = SpectrumDataset.from_mgf(
        tiny_mgf_with_metadata,
        metadata=["adduct", "collision_energy"],
        columns={},
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


def test_raises(tiny_mgf):
    with pytest.raises(TypeError, match="from_mgf"):
        SpectrumDataset()
    with pytest.raises(ValueError, match="Unknown metadata field"):
        SpectrumDataset.from_mgf(tiny_mgf, metadata=["not_a_real_field"])
    with pytest.raises(ValueError, match="reserved for"):
        SpectrumDataset._from_generator(
            lambda: iter([]),
            metadata=["adduct"],
            columns={"adduct": Value("string")},
        )


def test_from_mgf_string_column_coerces_non_string_value(tmp_path):
    path = tmp_path / "rt.mgf"
    path.write_text("BEGIN IONS\nPEPMASS=200.0\nRTINSECONDS=12.5\n50.0 5.0\nEND IONS\n")
    ds = SpectrumDataset.from_mgf(path, columns={"rtinseconds": Value("string")})
    assert ds.ds.features["rtinseconds"].dtype == "string"
    assert ds[0]["rtinseconds"] == "12.5"


def test_precursor_mz_missing_field_defaults_to_zero(tmp_path):
    path = tmp_path / "nopepmass.mgf"
    path.write_text("BEGIN IONS\nTITLE=no_pepmass\n50.0 5.0\nEND IONS\n")
    ds = SpectrumDataset.from_mgf(path)
    assert float(ds[0]["precursor_mz"]) == 0.0


def test_precursor_mz_unparseable_string_defaults_to_zero(tmp_path):
    path = tmp_path / "badpmz.mgf"
    path.write_text(
        "BEGIN IONS\nPEPMASS=200.0\nPRECURSOR_MZ=not-a-number\n50.0 5.0\nEND IONS\n"
    )
    ds = SpectrumDataset.from_mgf(path, precursor_mz_field="precursor_mz")
    assert float(ds[0]["precursor_mz"]) == 0.0


def test_spectrum_dataset_select(tiny_mgf):
    ds = SpectrumDataset.from_mgf(tiny_mgf)
    sub = ds.select([1, 0, 0])
    assert isinstance(sub, SpectrumDataset)
    assert len(sub) == 3
    np.testing.assert_array_equal(sub[0]["mz"].numpy(), ds[1]["mz"].numpy())
    np.testing.assert_array_equal(sub[1]["mz"].numpy(), ds[0]["mz"].numpy())


def test_transform_applied_lazily_at_getitem_time(tiny_mgf):
    ds = SpectrumDataset.from_mgf(tiny_mgf, transform=Normalizer())
    row = ds[0]
    assert np.isclose(float(row["intensity"].max()), 1.0)
    assert row["precursor_mz"] == pytest.approx(200.0)  # other columns pass through
    # Underlying stored data is untouched; normalization only happens on read.
    assert not np.isclose(float(ds.ds[0]["intensity"].max()), 1.0)


def test_flags_require_subformulae_name():
    raw = SpectrumDataset.from_list(SPECTRA_LIST).ds
    with pytest.raises(ValueError, match="require subformulae_name"):
        SpectrumDataset._create(raw, drop_peaks_without_subformula=True)


def test_create_warns_when_precursor_mz_missing():
    raw = SpectrumDataset.from_list(SPECTRA_LIST).ds.remove_columns(["precursor_mz"])
    with pytest.warns(UserWarning, match="precursor_mz"):
        SpectrumDataset._create(raw)


def test_add_subformulae_row_count_mismatch_raises(tmp_path):
    ds = SpectrumDataset.from_list(SPECTRA_LIST)  # 3 spectra
    bad_source = tmp_path / "bad_len.tsv"
    pd.DataFrame({"formula": [FORMULA], "adduct": [ADDUCT]}).to_csv(
        bad_source, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="Source has"):
        ds.add_subformulae("gt", source=bad_source)


def test_add_subformulae_missing_column_raises(tmp_path):
    ds = SpectrumDataset.from_list(SPECTRA_LIST)  # 3 spectra
    bad_source = tmp_path / "bad_cols.tsv"
    pd.DataFrame({"formula": [FORMULA] * 3}).to_csv(bad_source, sep="\t", index=False)
    with pytest.raises(ValueError, match="not found in source file"):
        ds.add_subformulae("gt", source=bad_source)


def test_add_subformulae_adds_columns_and_reshapes(ds_with_subformulae):
    ds = ds_with_subformulae
    assert {"gt_subformula_vec", "gt_parent_formula_vec", "gt_adduct"} <= set(
        ds.ds.column_names
    )

    row0 = ds[0]
    assert row0["gt_subformula_vec"].shape == (
        2,
        ELEMENT_DIM,
    )  # reshaped flat -> (peaks, D)
    np.testing.assert_array_equal(
        row0["gt_subformula_vec"][0].numpy(), formula_to_dense(FORMULA)
    )
    assert row0["gt_subformula_vec"][1].sum() == 0  # peak 1 (9999.0) didn't match
    assert row0["gt_adduct"] == ADDUCT

    row1 = ds[1]  # empty formula/adduct -> all-zero vectors
    assert int(row1["gt_subformula_vec"].sum()) == 0
    assert int(row1["gt_parent_formula_vec"].sum()) == 0


def test_create_subformulae_name_filters_other_sets(
    ds_with_subformulae, subformula_source, tmp_path
):
    ds = ds_with_subformulae.add_subformulae("pred", source=subformula_source)
    saved = tmp_path / "sf_multi"
    ds.save_to(saved)  # exercise the instance-level save_to() directly

    only_gt = SpectrumDataset.from_disk(saved, subformulae_name="gt")
    assert "gt_subformula_vec" in only_gt.ds.column_names
    assert "pred_subformula_vec" not in only_gt.ds.column_names
    assert "pred_parent_formula_vec" not in only_gt.ds.column_names
    assert "pred_adduct" not in only_gt.ds.column_names


def test_create_unknown_subformulae_name_raises(ds_with_subformulae, tmp_path):
    saved = tmp_path / "sf_single"
    ds_with_subformulae.save_to(saved)
    with pytest.raises(ValueError, match="not found in dataset"):
        SpectrumDataset.from_disk(saved, subformulae_name="nope")


def test_drop_peaks_without_subformula_filters_row(ds_with_subformulae, tmp_path):
    saved = tmp_path / "sf_drop"
    ds_with_subformulae.save_to(saved)
    ds = SpectrumDataset.from_disk(
        saved, subformulae_name="gt", drop_peaks_without_subformula=True
    )
    row0 = ds[0]
    assert len(row0["mz"]) == 1  # only the matched peak survives
    assert row0["gt_subformula_vec"].shape == (1, ELEMENT_DIM)

    row1 = ds[1]
    assert len(row1["mz"]) == 0  # its only peak was unmatched


def test_adduct_from_subformula_injects_encoded_adduct(ds_with_subformulae, tmp_path):
    saved = tmp_path / "sf_adduct"
    ds_with_subformulae.save_to(saved)
    ds = SpectrumDataset.from_disk(
        saved, subformulae_name="gt", adduct_from_subformula=True
    )
    assert int(ds[0]["adduct"]) == encode_adduct(ADDUCT)
    assert int(ds[1]["adduct"]) == 0  # empty adduct -> unknown


def test_collate_flattens_single_subformula_set(ds_with_subformulae):
    batch = ds_with_subformulae.collate(
        [ds_with_subformulae[0], ds_with_subformulae[1]]
    )
    assert set(batch["subformulae"]) == {"form_vec", "parent_form_vec"}
    assert batch["subformulae"]["form_vec"].shape[0] == 2
    assert batch["subformulae"]["parent_form_vec"].shape == (2, ELEMENT_DIM)


def test_collate_nests_multiple_subformula_sets(ds_with_subformulae, subformula_source):
    ds = ds_with_subformulae.add_subformulae("pred", source=subformula_source)
    batch = ds.collate([ds[0], ds[1]])
    assert set(batch["subformulae"]) == {"gt", "pred"}
    assert batch["subformulae"]["gt"]["form_vec"].shape[-1] == ELEMENT_DIM
    assert batch["subformulae"]["pred"]["parent_form_vec"].shape == (2, ELEMENT_DIM)
