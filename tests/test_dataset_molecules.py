"""Tests for :class:`metabo_depthcharge.datasets.molecules.MoleculeDataset`."""

import numpy as np
import pytest
import torch

from metabo_depthcharge.chem import Molecule
from metabo_depthcharge.chem.molecule import PROPERTIES
from metabo_depthcharge.datasets.molecules import _REP_INFO, MoleculeDataset


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
BENZENE = "c1ccccc1"


# --- from_csv basics ----------------------------------------------------


def test_from_csv_minimal_load(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t")
    assert len(ds) == 3
    assert ds[0]["smiles"] == ASPIRIN


def test_from_csv_renames_smiles_column(tmp_path):
    path = tmp_path / "custom.tsv"
    path.write_text(f"SMI\n{ASPIRIN}\n{CAFFEINE}\n")
    ds = MoleculeDataset.from_csv(path, sep="\t", smiles_column="SMI")
    assert "smiles" in ds.ds.column_names
    assert "SMI" not in ds.ds.column_names


def test_from_csv_preserves_extra_columns(tmp_path):
    path = tmp_path / "labels.tsv"
    path.write_text(f"smiles\tlabel\n{ASPIRIN}\t1\n{CAFFEINE}\t0\n")
    ds = MoleculeDataset.from_csv(path, sep="\t")
    assert "label" in ds.ds.column_names
    assert ds[0]["label"].item() == 1


def test_unknown_property_raises(tiny_tsv):
    with pytest.raises(ValueError, match="Unknown property"):
        MoleculeDataset.from_csv(tiny_tsv, sep="\t", properties=["bogus"])


# --- from_list --------------------------------------------------------


def test_from_list_accepts_strings():
    ds = MoleculeDataset.from_list([ASPIRIN, CAFFEINE, BENZENE])
    assert len(ds) == 3
    assert ds[0]["smiles"] == ASPIRIN


def test_from_list_accepts_mixed_str_and_molecule():
    ds = MoleculeDataset.from_list([ASPIRIN, Molecule(CAFFEINE), BENZENE])
    assert len(ds) == 3
    # Molecule objects contribute their .smiles attribute.
    assert ds[1]["smiles"] == CAFFEINE


def test_from_list_forwards_pipeline_kwargs():
    ds = MoleculeDataset.from_list(
        [ASPIRIN, CAFFEINE],
        properties=["formula"],
    )
    assert "formula" in ds.ds.column_names


# --- standardize --------------------------------------------------------


def test_standardize_strips_salt():
    ds = MoleculeDataset.from_list(["CC(=O)O.[Na]", CAFFEINE]).standardize()
    assert "[Na]" not in ds[0]["smiles"]
    assert ds[0]["smiles"] == "CC(=O)[O-]"


def test_standardize_drops_failed_rows():
    # Hypervalent SMILES that needs sanitization fallback to even parse →
    # Cleanup re-fails sanitization → row is dropped.
    bad = "F[P](F)(F)(F)(F)(F)F"  # 7-coordinate P, fails strict sanitization
    ds = MoleculeDataset.from_list([ASPIRIN, bad, CAFFEINE]).standardize()
    smiles_out = [ds[i]["smiles"] for i in range(len(ds))]
    assert bad not in smiles_out
    assert len(ds) < 3


def test_standardize_recomputes_stale_property_columns():
    # Load with a salt SMILES and pre-compute formula from the unsanitized form.
    ds = MoleculeDataset.from_list(["CC(=O)O.[Na]", CAFFEINE], properties=["formula"])
    formula_before = ds[0]["formula"]
    # Standardize removes the salt → SMILES changes → formula must update too.
    ds2 = ds.standardize()
    assert "formula" in ds2.ds.column_names
    assert ds2[0]["formula"] != formula_before


# --- properties columns -------------------------------------------------


def test_properties_adds_columns(tiny_tsv):
    ds = MoleculeDataset.from_csv(
        tiny_tsv,
        sep="\t",
        properties=["formula", "exact_mass", "inchikey_2d"],
    )
    for col in ("formula", "exact_mass", "inchikey_2d"):
        assert col in ds.ds.column_names
    assert ds[0]["formula"] == "C9H8O4"
    assert ds[0]["exact_mass"].item() == pytest.approx(180.0423, abs=1e-3)


def test_properties_accepts_all_known_names(tiny_tsv):
    ds = MoleculeDataset.from_csv(
        tiny_tsv,
        sep="\t",
        properties=list(PROPERTIES),
    )
    for col in PROPERTIES:
        assert col in ds.ds.column_names


def test_recompute_properties_false_preserves_csv_value(tiny_tsv_with_formula):
    ds = MoleculeDataset.from_csv(
        tiny_tsv_with_formula,
        sep="\t",
        properties=["formula"],
        recompute_properties=False,
    )
    # The sentinel values from the CSV survive (not overwritten by RDKit).
    assert ds[0]["formula"] == "FROM_CSV_A"


def test_recompute_properties_true_overwrites_csv_value(tiny_tsv_with_formula):
    ds = MoleculeDataset.from_csv(
        tiny_tsv_with_formula,
        sep="\t",
        properties=["formula"],
        recompute_properties=True,
    )
    assert ds[0]["formula"] == "C9H8O4"


# --- representations (fingerprints) -------------------------------------


def test_representations_adds_morgan(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan": {}}
    )
    assert "morgan" in ds.ds.column_names


def test_representations_adds_count_with_values_column(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan_count": {}}
    )
    # Count fingerprints store sparse indices + a values column.
    assert "morgan_count" in ds.ds.column_names
    assert "morgan_count_values" in ds.ds.column_names


def test_representations_unknown_name_raises(tiny_tsv):
    with pytest.raises(ValueError, match="Unknown representation name"):
        MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
            {"made_up": {}}
        )


def test_representations_multiple_at_once(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan": {}, "maccs": {}}
    )
    assert "morgan" in ds.ds.column_names
    assert "maccs" in ds.ds.column_names


# --- from_disk round-trip -----------------------------------------------


def test_from_disk_round_trip(tiny_tsv, tmp_path):
    saved = tmp_path / "saved"
    ds = MoleculeDataset.from_csv(
        tiny_tsv, sep="\t", properties=["formula"], save_to=saved
    )
    reloaded = MoleculeDataset.from_disk(saved)
    assert len(reloaded) == len(ds)
    assert reloaded.ds.column_names == ds.ds.column_names
    assert reloaded[0]["smiles"] == ds[0]["smiles"]
    assert reloaded[0]["formula"] == ds[0]["formula"]


# --- instance methods ---------------------------------------------------


def test_len_and_getitem(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t")
    assert len(ds) == 3
    row = ds[0]
    assert isinstance(row, dict)
    assert row["smiles"] == ASPIRIN


def test_col_to_numpy_returns_ndarray(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t")
    arr = ds.col_to_numpy("smiles")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert arr[0] == ASPIRIN


def test_filter_returns_moleculedataset(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t")
    aromatic_only = ds.filter(lambda r: r["smiles"].startswith("c"))
    assert isinstance(aromatic_only, MoleculeDataset)
    assert len(aromatic_only) == 1
    assert aromatic_only[0]["smiles"] == BENZENE


def test_add_representations_appends_columns(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t")
    assert "morgan" not in ds.ds.column_names
    ds2 = ds.add_representations({"morgan": None})
    assert "morgan" in ds2.ds.column_names
    assert isinstance(ds2, MoleculeDataset)


# --- collate ------------------------------------------------------------


def test_collate_binary_fingerprint_dense_shape(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan": {}}
    )
    batch = ds.collate([ds[0], ds[1]])
    assert batch["morgan"].shape == (2, 4096)
    assert batch["morgan"].dtype == torch.float32


def test_collate_count_fingerprint_carries_values(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan_count": {}}
    )
    batch = ds.collate([ds[0], ds[1]])
    assert batch["morgan_count"].shape == (2, 4096)
    # Count FP should have at least one position with value > 1 across
    # the two molecules (else "count" is indistinguishable from binary).
    assert (batch["morgan_count"] > 1).any()


def test_collate_passes_through_non_rep_columns(tiny_tsv):
    ds = MoleculeDataset.from_csv(
        tiny_tsv,
        sep="\t",
        properties=["formula"],
    ).add_representations({"morgan": {}})
    batch = ds.collate([ds[0], ds[1]])
    # smiles always a list of strings.
    assert batch["smiles"] == [ASPIRIN, CAFFEINE]
    # Property columns pass through as lists too.
    assert batch["formula"] == ["C9H8O4", "C8H10N4O2"]


# --- _REP_INFO registry smoke -------------------------------------------


@pytest.mark.parametrize("name", list(_REP_INFO))
def test_rep_info_build_returns_extractor_with_matching_size(name):
    spec = _REP_INFO[name]
    if spec.get("neural"):
        pytest.skip("neural extractor: skip to avoid model download")
    extractor = spec["build"]()
    assert extractor.rep_size == spec["size"]


# --- get_molmlp factory -------------------------------------


def test_get_molmlp_single_binary_returns_mol_embedder(tiny_tsv):
    from metabo_depthcharge.encoders.molecules import MolMLP

    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan": {}}
    )
    enc = ds.get_molmlp(["morgan"], n_blocks=2, d_model=32)
    assert isinstance(enc, MolMLP)
    assert enc.rep_type == "binary"


def test_get_molmlp_single_count_has_max_counts(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan_count": {}}
    )
    enc = ds.get_molmlp(["morgan_count"], n_blocks=2, d_model=32)
    assert enc.max_counts is not None
    assert enc.max_counts.shape == (4096,)


def test_get_molmlp_count_no_max_counts_when_disabled(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan_count": {}}
    )
    enc = ds.get_molmlp(
        ["morgan_count"], n_blocks=2, d_model=32, compute_max_counts=False
    )
    assert enc.max_counts is None


def test_get_molmlp_multi_returns_multi_mol_embedder(tiny_tsv):
    from metabo_depthcharge.encoders.molecules import MultiMolMLP

    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan": {}, "maccs": {}}
    )
    enc = ds.get_molmlp(["morgan", "maccs"], n_blocks=2, d_model=32)
    assert isinstance(enc, MultiMolMLP)
    assert enc.rep_names == ["morgan", "maccs"]


def test_get_molmlp_forward_pass(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t").add_representations(
        {"morgan": {}}
    )
    enc = ds.get_molmlp(["morgan"], n_blocks=2, d_model=32)
    batch = ds.collate([ds[i] for i in range(len(ds))])
    out = enc(batch["morgan"])
    assert out.shape == (len(ds), 32)


def test_get_molmlp_raises_on_missing_column(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t")
    with pytest.raises(ValueError, match="not found"):
        ds.get_molmlp(["morgan"], n_blocks=2, d_model=32)


def test_get_molmlp_raises_on_unknown_fp(tiny_tsv):
    ds = MoleculeDataset.from_csv(tiny_tsv, sep="\t")
    with pytest.raises(ValueError, match="Unknown representation name"):
        ds.get_molmlp(["not_a_fp"], n_blocks=2, d_model=32)
