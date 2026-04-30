"""Characterization tests for metabo_depthcharge.molecules vs spectrawl."""

# Direct file import to bypass spectrawl's __init__.py (which pulls in torch/lightning)
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_spectrawl_path = next(
    p for p in sys.path if (Path(p) / "spectrawl" / "data" / "molecules.py").exists()
)
_metabo_src_path = next(
    p for p in sys.path if (Path(p) / "metabo_depthcharge" / "molecules.py").exists()
)

orig = _load_from_file(
    "spectrawl_molecules",
    Path(_spectrawl_path) / "spectrawl" / "data" / "molecules.py",
)
new = _load_from_file(
    "metabo_molecules",
    Path(_metabo_src_path) / "metabo_depthcharge" / "molecules.py",
)


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
INVALID_BUT_PARSEABLE = "C1=CC=CC=C1"  # benzene, simple canonical


def test_canonicalize_smiles_aspirin():
    assert new.canonicalize_smiles(ASPIRIN) == orig.canonicalize_smiles(ASPIRIN)


def test_canonicalize_smiles_caffeine():
    assert new.canonicalize_smiles(CAFFEINE) == orig.canonicalize_smiles(CAFFEINE)


def test_safe_mol_from_smiles_returns_mol():
    mol_orig = orig.safe_mol_from_smiles(ASPIRIN)
    mol_new = new.safe_mol_from_smiles(ASPIRIN)
    from rdkit import Chem

    assert Chem.MolToSmiles(mol_orig) == Chem.MolToSmiles(mol_new)


def test_safe_mol_from_smiles_invalid_raises():
    with pytest.raises(ValueError):
        new.safe_mol_from_smiles("not-a-smiles-!!!")


# --- Morgan ---


def test_smiles_to_morgan_single():
    fp_orig = orig.SmilesToMorgan()(ASPIRIN)
    fp_new = new.SmilesToMorgan()(ASPIRIN)
    np.testing.assert_array_equal(fp_orig, fp_new)


def test_smiles_to_morgan_batch():
    smiles = [ASPIRIN, CAFFEINE]
    np.testing.assert_array_equal(
        orig.SmilesToMorgan()(smiles),
        new.SmilesToMorgan()(smiles),
    )


def test_smiles_to_morgan_counts():
    fp_orig = orig.SmilesToMorgan(counts=True)(ASPIRIN)
    fp_new = new.SmilesToMorgan(counts=True)(ASPIRIN)
    np.testing.assert_array_equal(fp_orig, fp_new)


def test_smiles_to_morgan_fp_size():
    enc = new.SmilesToMorgan(fp_size=2048)
    assert enc(ASPIRIN).shape == (2048,)


# --- RDKit ---


def test_smiles_to_rdkit_single():
    np.testing.assert_array_equal(
        orig.SmilesToRdkit()(ASPIRIN),
        new.SmilesToRdkit()(ASPIRIN),
    )


def test_smiles_to_rdkit_batch():
    smiles = [ASPIRIN, CAFFEINE]
    np.testing.assert_array_equal(
        orig.SmilesToRdkit()(smiles),
        new.SmilesToRdkit()(smiles),
    )


# --- MACCS ---


def test_smiles_to_maccs_single():
    np.testing.assert_array_equal(
        orig.SmilesToMACCS()(ASPIRIN),
        new.SmilesToMACCS()(ASPIRIN),
    )


def test_smiles_to_maccs_batch():
    smiles = [ASPIRIN, CAFFEINE]
    np.testing.assert_array_equal(
        orig.SmilesToMACCS()(smiles),
        new.SmilesToMACCS()(smiles),
    )


def test_smiles_to_maccs_fp_size():
    assert new.SmilesToMACCS().fp_size == 167


# --- Optional deps (skip if not installed) ---


@pytest.fixture(scope="module")
def biosynfoni():
    return pytest.importorskip("biosynfoni")


def test_smiles_to_biosynfoni_single(biosynfoni):
    np.testing.assert_array_equal(
        orig.SmilesToBiosynfoni()(ASPIRIN),
        new.SmilesToBiosynfoni()(ASPIRIN),
    )


@pytest.fixture(scope="module")
def map4():
    return pytest.importorskip("map4")


def test_smiles_to_map4_single(map4):
    np.testing.assert_array_equal(
        orig.SmilesToMAP4()(ASPIRIN),
        new.SmilesToMAP4()(ASPIRIN),
    )
