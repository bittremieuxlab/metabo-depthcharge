"""Tests for :class:`metabo_depthcharge.chem.molecule.Molecule` — the
RDKit-backed molecule wrapper, its derived properties, equality semantics,
``from_dict`` construction and ``standardize``."""

import pytest

from metabo_depthcharge.chem import Molecule
from metabo_depthcharge.chem.molecule import PROPERTIES


# --- Molecule basics -----------------------------------------------------


def test_repr_only_shows_smiles(aspirin):
    assert repr(Molecule(aspirin)) == f"Molecule(smiles={aspirin!r})"


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        _ = Molecule("not-a-smiles-!!!").mol


def test_canonical_smiles_is_stable(aspirin):
    canon = Molecule(aspirin).canonical_smiles
    assert Molecule(canon).canonical_smiles == canon


def test_canonical_smiles_normalizes_input():
    # Two equivalent SMILES for benzene → same canonical form.
    a = Molecule("c1ccccc1").canonical_smiles
    b = Molecule("C1=CC=CC=C1").canonical_smiles
    assert a == b


def test_properties_are_cached_and_do_not_track_smiles(aspirin, caffeine):
    m = Molecule(aspirin)
    formula = m.formula
    # cached_property is sticky: reassigning smiles does NOT invalidate it.
    m.smiles = caffeine
    assert m.formula == formula


# --- Equality and hashing (by inchikey_2d) -------------------------------


def test_equality_ignores_stereochemistry():
    # Enantiomers of alanine share a 2D skeleton → equal and hash-equal.
    a = Molecule("C[C@H](N)C(=O)O")
    b = Molecule("C[C@@H](N)C(=O)O")
    assert a == b
    assert hash(a) == hash(b)


def test_distinct_molecules_are_unequal(aspirin, caffeine):
    assert Molecule(aspirin) != Molecule(caffeine)


def test_set_dedups_by_inchikey_2d(aspirin, caffeine):
    mols = {Molecule(aspirin), Molecule(aspirin), Molecule(caffeine)}
    assert len(mols) == 2


def test_equality_against_non_molecule_is_false(aspirin):
    assert Molecule(aspirin) != aspirin  # str, not Molecule


# --- Derived properties --------------------------------------------------


def test_aspirin_formula(aspirin_mol):
    assert aspirin_mol.formula == "C9H8O4"


def test_aspirin_exact_mass(aspirin_mol):
    assert aspirin_mol.exact_mass == pytest.approx(180.0423, abs=1e-3)


def test_aspirin_inchikey(aspirin_mol):
    assert aspirin_mol.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_aspirin_inchi_starts_with_inchi_prefix(aspirin_mol):
    assert aspirin_mol.inchi.startswith("InChI=")


def test_inchikey_2d_is_first_block(aspirin_mol):
    assert aspirin_mol.inchikey_2d == aspirin_mol.inchikey.split("-")[0]


def test_property_caching_reuses_cached_value(aspirin_mol):
    # First access computes; result is stored in __dict__ (cached_property
    # behaviour). Override the slot and verify subsequent reads don't
    # invoke RDKit again.
    _ = aspirin_mol.formula
    aspirin_mol.__dict__["formula"] = "OVERRIDE"
    assert aspirin_mol.formula == "OVERRIDE"


def test_properties_constant_matches_class_attributes():
    from functools import cached_property

    for name in PROPERTIES:
        attr = getattr(Molecule, name)
        assert isinstance(attr, property | cached_property), (
            f"PROPERTIES contains {name!r} but Molecule has no property by that name"
        )


# --- from_dict -----------------------------------------------------------


def test_from_dict_minimal_only_requires_smiles(aspirin):
    m = Molecule.from_dict({"smiles": aspirin})
    assert m.smiles == aspirin
    assert m.formula == "C9H8O4"  # computed via RDKit since not in dict


def test_from_dict_prepopulates_cache(aspirin):
    m = Molecule.from_dict(
        {
            "smiles": aspirin,
            "formula": "PRECOMPUTED",
            "exact_mass": 999.0,
            "inchikey": "FAKE-KEY",
            "inchikey_2d": "FAKE",
            "canonical_smiles": "fake_canon",
            "inchi": "InChI=fake",
        }
    )
    # All property reads return the precomputed values — RDKit never runs.
    assert m.formula == "PRECOMPUTED"
    assert m.exact_mass == 999.0
    assert m.inchikey == "FAKE-KEY"
    assert m.inchikey_2d == "FAKE"
    assert m.canonical_smiles == "fake_canon"
    assert m.inchi == "InChI=fake"


def test_from_dict_partial_columns_compute_the_rest(aspirin):
    m = Molecule.from_dict({"smiles": aspirin, "formula": "PRECOMPUTED"})
    assert m.formula == "PRECOMPUTED"
    # exact_mass was not provided → computed from RDKit.
    assert m.exact_mass == pytest.approx(180.0423, abs=1e-3)


def test_from_dict_bypasses_parsing_when_all_props_present():
    # A nonsensical smiles paired with all precomputed properties should
    # still answer property queries without invoking RDKit.
    m = Molecule.from_dict(
        {
            "smiles": "🚫 not a real smiles 🚫",
            "formula": "C9H8O4",
            "exact_mass": 180.04,
            "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "inchikey_2d": "BSYNRYMUTXBXSQ",
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "inchi": "InChI=fake",
        }
    )
    assert m.formula == "C9H8O4"
    assert m.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"


# --- Molecule.standardize ------------------------------------------------


def test_standardize_returns_new_molecule(aspirin):
    out = Molecule(aspirin).standardize(canonicalize_tautomers=True)
    assert isinstance(out, Molecule)
    assert out is not None


def test_standardize_strips_salt_via_fragment_parent():
    # CC(=O)O.[Na] should reduce to the larger fragment. Default standardize
    # (uncharge=False) keeps the charge state of that fragment as-is.
    out = Molecule("CC(=O)O.[Na]").standardize()
    assert out is not None
    assert "[Na]" not in out.canonical_smiles
    assert out.canonical_smiles == "CC(=O)[O-]"


def test_standardize_uncharges_when_opted_in():
    # Carboxylate anion neutralized only when uncharge=True (opt-in).
    out = Molecule("CC(=O)[O-]").standardize(uncharge=True)
    assert out is not None
    assert out.canonical_smiles == "CC(=O)O"


def test_standardize_preserves_charge_by_default():
    # Default uncharge=False keeps charges as the curator wrote them.
    out = Molecule("CC(=O)[O-]").standardize()
    assert out is not None
    assert "-" in out.canonical_smiles  # still anionic


def test_standardize_metadata_is_dropped():
    # Standardize returns a fresh Molecule without carrying any user state.
    out = Molecule("CC(=O)O").standardize()
    assert out is not None
    assert not hasattr(out, "metadata")


def test_standardize_leaves_original_untouched():
    # standardize returns a new Molecule; the source instance keeps its
    # stereochemistry even when remove_stereo strips it from the copy.
    m = Molecule("C[C@H](N)C(=O)O")
    out = m.standardize()  # remove_stereo=True by default
    assert "@" not in out.canonical_smiles  # copy lost stereo
    assert "@" in m.canonical_smiles  # original retained it


def test_standardize_failure_returns_none_without_mutating():
    bad = "F[P](F)(F)(F)(F)(F)F"  # parses via fallback, Cleanup re-fails
    m = Molecule(bad)
    assert m.standardize() is None
    assert m.smiles == bad
