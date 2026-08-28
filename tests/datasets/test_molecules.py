"""Tests for :class:`metabo_depthcharge.datasets.molecules.MoleculeDataset`."""

import numpy as np
import pytest
import torch

from metabo_depthcharge.chem import Molecule, MoleculeToGraph, graphs
from metabo_depthcharge.chem.molecule import PROPERTIES
from metabo_depthcharge.datasets.molecules import (
    _GRAPH_TABLE_FILE,
    _REP_INFO,
    MoleculeDataset,
)


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


def test__raises(tiny_tsv):
    with pytest.raises(TypeError):
        MoleculeDataset()
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


# --- Molecular graphs -------------------------------------------------------

GRAPH_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
]


def _graph_ds():
    return MoleculeDataset.from_list(GRAPH_SMILES).add_graphs()


def test_add_graphs_adds_columns():
    ds = _graph_ds()
    for key in MoleculeToGraph.KEYS:
        assert f"graph_{key}" in ds.ds.column_names


def test_add_graphs_column_lengths_match_the_molecules():
    ds = _graph_ds()
    keys = ds.col_to_numpy("graph_atom_key")
    bsrc = ds.col_to_numpy("graph_bsrc")
    for i, smiles in enumerate(GRAPH_SMILES):
        mol = Molecule(smiles).mol
        assert len(keys[i]) == mol.GetNumAtoms()
        assert len(bsrc[i]) == mol.GetNumBonds()


def test_add_graphs_streams_in_batches_smaller_than_the_dataset():
    """Memory must be bounded by batch_size, not by the dataset -- the whole point."""
    ds = MoleculeDataset.from_list(GRAPH_SMILES).add_graphs(batch_size=1)
    assert len(ds) == len(GRAPH_SMILES)
    assert ds.graph_table["smiles"] == GRAPH_SMILES


def test_add_graphs_round_trips_through_disk(tmp_path):
    path = tmp_path / "ds"
    MoleculeDataset.from_list(GRAPH_SMILES).add_graphs(save_to=path)
    back = MoleculeDataset.from_disk(path)
    assert "graph_atom_key" in back.ds.column_names
    assert back.graph_table["smiles"] == GRAPH_SMILES


def test_graph_table_preserves_row_order():
    """The encoder indexes the table by row, so table row i MUST be dataset row i."""
    ds = _graph_ds()
    assert ds.graph_table["smiles"] == ds.col_to_numpy("smiles").tolist()


def test_graph_table_matches_direct_featurization():
    table = _graph_ds().graph_table
    direct = graphs.pack(
        MoleculeToGraph()([Molecule(s) for s in GRAPH_SMILES]), GRAPH_SMILES
    )
    for key in ("types", "atom_type", "bsrc", "bdst", "bcode", "nptr", "bptr"):
        assert torch.equal(table[key], direct[key]), key


def test_graph_table_requires_add_graphs():
    ds = MoleculeDataset.from_list(GRAPH_SMILES)
    with pytest.raises(ValueError, match="add_graphs"):
        _ = ds.graph_table


def test_row_alignment_survives_encoding():
    """Encoding a row alone and in a batch must give the same atoms."""
    from metabo_depthcharge.encoders import GraphMolEncoder

    ds = _graph_ds()
    enc = GraphMolEncoder(ds.atom_types(), [16, 32], 0.0, 32, per_atom=True).eval()
    with torch.no_grad():
        batched, _ = enc(ds.gather_graphs(torch.arange(len(GRAPH_SMILES))))
        for i in range(len(GRAPH_SMILES)):
            alone, alone_pad = enc(ds.gather_graphs(torch.tensor([i])))
            n = int((~alone_pad[0]).sum())
            assert n == Molecule(GRAPH_SMILES[i]).mol.GetNumAtoms()
            assert torch.allclose(batched[i, :n], alone[0, :n], atol=1e-5)


def test_graph_table_follows_a_filtered_subset():
    """filter() renumbers rows; the table must follow, or every index is off."""
    ds = _graph_ds().filter(lambda row: row["smiles"] != "c1ccccc1")
    table = ds.graph_table
    assert table["smiles"] == ds.col_to_numpy("smiles").tolist()
    assert "c1ccccc1" not in table["smiles"]
    assert len(table["nptr"]) == len(ds) + 1


def test_add_graphs_composes_with_add_representations():
    ds = MoleculeDataset.from_list(GRAPH_SMILES).add_representations({"morgan": None})
    ds = ds.add_graphs()
    assert "morgan" in ds.ds.column_names
    assert ds.graph_table["smiles"] == GRAPH_SMILES


def test_col_to_numpy_respects_a_filter():
    """Regression: `.data` ignores the indices map filter()/select() leave behind."""
    ds = MoleculeDataset.from_list(GRAPH_SMILES)
    kept = ds.filter(lambda row: row["smiles"] != "c1ccccc1")
    assert kept.col_to_numpy("smiles").tolist() == ["CCO", *GRAPH_SMILES[2:]]
    assert len(kept.col_to_numpy("smiles")) == len(kept)


def test_col_to_numpy_respects_a_reordering_select():
    ds = MoleculeDataset.from_list(GRAPH_SMILES)
    picked = ds.select([2, 0])
    assert picked.col_to_numpy("smiles").tolist() == [GRAPH_SMILES[2], GRAPH_SMILES[0]]


def test_row_access_returns_a_ready_graph():
    """A stored graph must be readable from a row -- and in the form an encoder takes,
    not as four raw ragged columns the caller has to reassemble."""
    ds = _graph_ds()
    row = ds[0]
    assert "graph" in row
    assert not any(k.startswith("graph_") for k in row)
    assert sorted(row["graph"]) == ["atom_type", "bcode", "bdst", "bsrc"]
    assert len(row["graph"]["atom_type"]) == Molecule(GRAPH_SMILES[0]).mol.GetNumAtoms()


def test_row_access_is_served_from_the_packed_table():
    """Rows are views into the one materialized table, not re-decoded per access."""
    ds = _graph_ds()
    table = ds.graph_table
    assert ds.graph_table is table  # built once, then reused
    lo, hi = int(table["nptr"][1]), int(table["nptr"][2])
    assert torch.equal(ds[1]["graph"]["atom_type"], table["atom_type"][lo:hi])


def test_collate_fuses_rows_into_one_batch():
    """Ragged per-molecule arrays do not stack, so collate emits the batched form."""
    ds = _graph_ds()
    batch = MoleculeDataset.collate([ds[i] for i in range(len(ds))])
    graph = batch["graph"]
    assert graph["nptr"][-1] == len(graph["atom_type"])
    assert graph["bptr"][-1] == len(graph["bsrc"])
    assert len(graph["nptr"]) == len(ds) + 1


def test_dataloader_path_matches_the_gather_path():
    """Iterating and indexing must give the same embedding for the same molecule."""
    from torch.utils.data import DataLoader

    from metabo_depthcharge.encoders import GraphMolEncoder

    ds = _graph_ds()
    enc = GraphMolEncoder(ds.atom_types(), [16, 32], 0.0, 32, per_atom=True).eval()
    loader = DataLoader(ds, batch_size=len(ds), collate_fn=MoleculeDataset.collate)
    with torch.no_grad():
        from_loader, mask_loader = enc(next(iter(loader))["graph"])
        from_gather, mask_gather = enc(ds.gather_graphs(torch.arange(len(ds))))
    assert torch.allclose(from_loader, from_gather, atol=1e-6)
    assert torch.equal(mask_loader, mask_gather)


def test_encoder_rejects_an_atom_type_outside_its_table():
    """Graphs keyed against a different table would silently embed wrong molecules."""
    from metabo_depthcharge.encoders import GraphMolEncoder

    narrow = MoleculeDataset.from_list(["CCO"]).add_graphs()
    enc = GraphMolEncoder(narrow.atom_types(), [16, 32], 0.0, 32)
    wide = _graph_ds()
    with pytest.raises(ValueError, match="outside this encoder"):
        enc(wide.gather_graphs(torch.arange(len(wide))))


def test_graph_columns_are_stored_and_readable_by_column():
    """Present in the dataset and reachable in bulk, which is how the table is built."""
    ds = _graph_ds()
    for key in MoleculeToGraph.KEYS:
        assert f"graph_{key}" in ds.ds.column_names
        assert len(ds.col_to_numpy(f"graph_{key}")) == len(ds)


def test_add_representations_after_add_graphs_keeps_both():
    """map() must not drop the columns that row formatting hides."""
    ds = _graph_ds().add_representations({"morgan": None})
    assert "graph_atom_key" in ds.ds.column_names
    assert "morgan" in ds.ds.column_names
    assert ds.graph_table["smiles"] == GRAPH_SMILES


def test_graph_table_is_built_at_creation_not_on_first_use():
    """The cost lands once, at a predictable point, rather than inside whichever
    access happens to touch graphs first."""
    ds = _graph_ds()
    assert ds._graph_table is not None


def test_no_graphs_means_no_table_is_built():
    ds = MoleculeDataset.from_list(GRAPH_SMILES)
    assert ds._graph_table is None


def test_row_preserving_ops_reuse_the_table():
    """map() keeps rows and order, so rebuilding would be pure waste."""
    ds = _graph_ds()
    after = ds.add_representations({"morgan": None})
    assert after._graph_table is ds._graph_table


def test_row_changing_ops_rebuild_the_table():
    """filter/select renumber rows, so a carried-over table would mislabel them."""
    ds = _graph_ds()
    kept = ds.filter(lambda row: row["smiles"] != "c1ccccc1")
    assert kept._graph_table is not ds._graph_table
    assert kept.graph_table["smiles"] == [s for s in GRAPH_SMILES if s != "c1ccccc1"]
    picked = ds.select([2, 0])
    assert picked.graph_table["smiles"] == [GRAPH_SMILES[2], GRAPH_SMILES[0]]


def test_from_disk_builds_the_table(tmp_path):
    ds = _graph_ds()
    ds.save_to(tmp_path / "ds")
    back = MoleculeDataset.from_disk(tmp_path / "ds")
    assert back._graph_table is not None
    assert back.graph_table["smiles"] == GRAPH_SMILES


def test_add_graphs_caches_the_table_beside_the_dataset(tmp_path):
    MoleculeDataset.from_list(GRAPH_SMILES).add_graphs(save_to=tmp_path / "ds")
    assert (tmp_path / "ds" / _GRAPH_TABLE_FILE).is_file()


def test_from_disk_reuses_the_cached_table(tmp_path, monkeypatch):
    """The point of caching: loading a pool must not rebuild its table."""
    MoleculeDataset.from_list(GRAPH_SMILES).add_graphs(save_to=tmp_path / "ds")
    monkeypatch.setattr(
        MoleculeDataset,
        "_build_graph_table",
        lambda self: pytest.fail("rebuilt despite a valid cache"),
    )
    back = MoleculeDataset.from_disk(tmp_path / "ds")
    assert back.graph_table["smiles"] == GRAPH_SMILES


def test_stale_cache_is_rejected_not_trusted(tmp_path):
    """A cache describing other molecules must never be believed: it would return
    the wrong molecule for every row it shifted."""
    import shutil

    MoleculeDataset.from_list(GRAPH_SMILES).add_graphs(save_to=tmp_path / "ds")
    subset = MoleculeDataset.from_disk(tmp_path / "ds").filter(
        lambda row: row["smiles"] != "c1ccccc1"
    )
    subset.ds.save_to_disk(str(tmp_path / "sub"))  # dataset only, cache left behind
    shutil.copy(
        tmp_path / "ds" / _GRAPH_TABLE_FILE, tmp_path / "sub" / _GRAPH_TABLE_FILE
    )

    loaded = MoleculeDataset.from_disk(tmp_path / "sub")
    assert loaded.graph_table["smiles"] == [s for s in GRAPH_SMILES if s != "c1ccccc1"]


def test_save_to_refreshes_the_cache(tmp_path):
    ds = _graph_ds().filter(lambda row: row["smiles"] != "c1ccccc1")
    ds.save_to(tmp_path / "sub")
    back = MoleculeDataset.from_disk(tmp_path / "sub")
    assert back.graph_table["smiles"] == ds.graph_table["smiles"]


def test_no_cache_is_written_for_a_dataset_without_graphs(tmp_path):
    MoleculeDataset.from_list(GRAPH_SMILES).save_to(tmp_path / "ds")
    assert not (tmp_path / "ds" / _GRAPH_TABLE_FILE).exists()


def test_graph_table_is_read_only():
    """Nothing assigns it: the table comes from the columns or from the cache beside
    the dataset, both of which are checked against the rows they describe."""
    ds = _graph_ds()
    with pytest.raises(AttributeError):
        ds.graph_table = ds.graph_table
