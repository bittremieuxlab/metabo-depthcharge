"""Tests for :mod:`metabo_depthcharge.chem.tokenizers` — SAFE string
conversion used by :class:`MoleculeToSAFEGPT`."""

from metabo_depthcharge.chem.tokenizers import smiles_to_safe_strings


def test_smiles_to_safe_strings_skips_invalid_entries(aspirin):
    safe_strs = smiles_to_safe_strings([aspirin, "not_a_smiles"])
    assert safe_strs[0] is not None
    assert safe_strs[1] is None
