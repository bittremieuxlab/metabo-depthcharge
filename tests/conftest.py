"""Shared fixtures for the test suite."""

from pathlib import Path

import pytest

from metabo_depthcharge.chem import Molecule


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
ASPIRIN_SODIUM = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"  # salt; FragmentParent drops [Na+]
BENZENE = "c1ccccc1"


@pytest.fixture
def aspirin() -> str:
    return ASPIRIN


@pytest.fixture
def caffeine() -> str:
    return CAFFEINE


@pytest.fixture
def aspirin_mol() -> Molecule:
    return Molecule(ASPIRIN)


@pytest.fixture
def tiny_tsv(tmp_path: Path) -> Path:
    """A 3-row TSV with just a ``smiles`` column."""
    path = tmp_path / "tiny.tsv"
    path.write_text(f"smiles\n{ASPIRIN}\n{CAFFEINE}\n{BENZENE}\n")
    return path


@pytest.fixture
def tiny_tsv_with_salt(tmp_path: Path) -> Path:
    """A 3-row TSV including a salt to exercise standardize+FragmentParent."""
    path = tmp_path / "tiny_salt.tsv"
    path.write_text(f"smiles\n{ASPIRIN}\n{CAFFEINE}\n{ASPIRIN_SODIUM}\n")
    return path


@pytest.fixture
def tiny_tsv_with_formula(tmp_path: Path) -> Path:
    """TSV with a sentinel ``formula`` column to test recompute_properties."""
    path = tmp_path / "tiny_with_formula.tsv"
    path.write_text(f"smiles\tformula\n{ASPIRIN}\tFROM_CSV_A\n{CAFFEINE}\tFROM_CSV_B\n")
    return path
