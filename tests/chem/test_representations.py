"""Tests for :mod:`metabo_depthcharge.chem.representations` — the
molecule-to-vector representation classes (fingerprints and neural
embeddings)."""

import numpy as np
import pytest

from metabo_depthcharge.chem import (
    Molecule,
    MoleculeToBiosynfoni,
    MoleculeToChemBERTa,
    MoleculeToMACCS,
    MoleculeToMAP4,
    MoleculeToMolFormer,
    MoleculeToMorgan,
    MoleculeToRdkit,
    MoleculeToSAFEGPT,
)


# --- MoleculeToMorgan ----------------------------------------------------


def test_morgan_single_shape(aspirin_mol):
    assert MoleculeToMorgan()(aspirin_mol).shape == (4096,)


def test_morgan_custom_rep_size(aspirin_mol):
    enc = MoleculeToMorgan(rep_size=2048)
    assert enc(aspirin_mol).shape == (2048,)
    assert enc.rep_size == 2048


def test_morgan_batch_stacks(aspirin_mol, caffeine):
    fps = MoleculeToMorgan()([aspirin_mol, Molecule(caffeine)])
    assert fps.shape == (2, 4096)


def test_morgan_binary_values(aspirin_mol):
    rep = MoleculeToMorgan()(aspirin_mol)
    assert set(np.unique(rep)).issubset({0, 1})


def test_morgan_counts_nonnegative(aspirin_mol):
    rep = MoleculeToMorgan(counts=True)(aspirin_mol)
    assert (rep >= 0).all()


def test_morgan_distinguishes_molecules(aspirin_mol, caffeine):
    enc = MoleculeToMorgan()
    assert not np.array_equal(enc(aspirin_mol), enc(Molecule(caffeine)))


# --- MoleculeToRdkit -----------------------------------------------------


def test_rdkit_single_shape(aspirin_mol):
    assert MoleculeToRdkit()(aspirin_mol).shape == (4096,)


def test_rdkit_batch(aspirin_mol, caffeine):
    fps = MoleculeToRdkit()([aspirin_mol, Molecule(caffeine)])
    assert fps.shape == (2, 4096)


def test_rdkit_binary(aspirin_mol):
    rep = MoleculeToRdkit()(aspirin_mol)
    assert set(np.unique(rep)).issubset({0, 1})


def test_rdkit_count(aspirin_mol):
    rep = MoleculeToRdkit(counts=True, rep_size=128)(aspirin_mol)
    assert np.max(rep) > 1


# --- MoleculeToMACCS -----------------------------------------------------


def test_maccs_rep_size_constant(aspirin_mol):
    enc = MoleculeToMACCS()
    assert enc.rep_size == 167
    assert enc(aspirin_mol).shape == (167,)


def test_maccs_batch(aspirin_mol, caffeine):
    fps = MoleculeToMACCS()([aspirin_mol, Molecule(caffeine)])
    assert fps.shape == (2, 167)


def test_maccs_binary(aspirin_mol):
    rep = MoleculeToMACCS()(aspirin_mol)
    assert set(np.unique(rep)).issubset({0, 1})


# --- MoleculeToBiosynfoni ------------------------------------------------


def test_biosynfoni_rep_size_constant(aspirin_mol):
    enc = MoleculeToBiosynfoni()
    assert enc.rep_size == 39
    assert enc(aspirin_mol).shape == (39,)


def test_biosynfoni_batch(aspirin_mol, caffeine):
    fps = MoleculeToBiosynfoni()([aspirin_mol, Molecule(caffeine)])
    assert fps.shape == (2, 39)


# --- MoleculeToMAP4 ------------------------------------------------------


def test_map4_single_shape(aspirin_mol):
    assert MoleculeToMAP4()(aspirin_mol).shape == (4096,)


def test_map4_custom_rep_size(aspirin_mol):
    enc = MoleculeToMAP4(rep_size=1024)
    assert enc(aspirin_mol).shape == (1024,)
    assert enc.rep_size == 1024


def test_map4_batch(aspirin_mol, caffeine):
    fps = MoleculeToMAP4()([aspirin_mol, Molecule(caffeine)])
    assert fps.shape == (2, 4096)


def test_map4_binary_values(aspirin_mol):
    # Verified empirically in the package: this library folds MinHash
    # signatures into a binary indicator vector.
    rep = MoleculeToMAP4()(aspirin_mol)
    assert set(np.unique(rep)).issubset({0, 1})


# --- Neural representations (import-only smoke) --------------------------


def test_molformer(aspirin_mol):
    enc = MoleculeToMolFormer()
    assert enc(aspirin_mol).shape == (768,)


def test_chemberta(aspirin_mol):
    enc = MoleculeToChemBERTa()
    assert enc([aspirin_mol]).shape == (1, 768)


def test_molformer_raise_arg(aspirin_mol):
    with pytest.raises(ValueError):
        MoleculeToMolFormer(device="brick")


def test_safegpt(aspirin_mol, caffeine):
    enc = MoleculeToSAFEGPT()
    assert enc([aspirin_mol, Molecule(caffeine)]).shape == (2, 768)
