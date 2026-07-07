"""Tests for :mod:`metabo_depthcharge.spec.subformulae`."""

import numpy as np
import pytest

from metabo_depthcharge.mist_cf.common.chem_utils import (
    ELEMENT_DIM,
    formula_mass,
    ion_to_mass,
)
from metabo_depthcharge.spec.subformulae import (
    assign_peak_subformulae,
    formula_to_dense,
)


def test_formula_to_dense():
    dense = formula_to_dense("C9H8O4")
    assert dense.dtype == np.int16
    assert dense.shape == (ELEMENT_DIM,)
    assert dense.sum() == 9 + 8 + 4  # total atom count

    assert formula_to_dense("").sum() == 0  # empty formula -> all-zero vector

    with pytest.raises(KeyError):
        formula_to_dense("Zn2")  # Zn is not a supported element


def test_assign_peak_subformulae():
    formula = "C9H8O4"
    parent = formula_to_dense(formula)
    matched_mz = formula_mass(formula) + ion_to_mass["[M+H]+"]
    mz = np.array([matched_mz, 9999.0])  # one matching peak, one far off

    vectors, parent_vec, valid = assign_peak_subformulae(mz, formula, "[M+H]+")
    np.testing.assert_array_equal(parent_vec, parent)
    assert valid.tolist() == [True, False]
    assert vectors.shape == (2 * ELEMENT_DIM,)
    # [M+H]+ matches the parent formula itself
    np.testing.assert_array_equal(vectors[:ELEMENT_DIM], parent)
    np.testing.assert_array_equal(vectors[ELEMENT_DIM:], np.zeros(ELEMENT_DIM))

    # every fallback path (bad/empty formula, bad/empty adduct, no peaks)
    # returns all-zero vectors and an all-False valid mask
    fallback_cases = [
        (mz, "", "[M+H]+"),
        (mz, "Zn2", "[M+H]+"),
        (mz, formula, ""),
        (mz, formula, "NOT-AN-ADDUCT"),
        (np.array([]), formula, "[M+H]+"),
    ]
    for args in fallback_cases:
        v, _p, val = assign_peak_subformulae(*args)
        assert not val.any()
        assert (v == 0).all()


def test_assign_peak_subformulae_get_all_subsets_failure(monkeypatch):
    def boom(_formula):
        raise RuntimeError("boom")

    monkeypatch.setattr("metabo_depthcharge.spec.subformulae.get_all_subsets", boom)

    formula = "C9H8O4"
    parent = formula_to_dense(formula)
    vectors, parent_vec, valid = assign_peak_subformulae(
        np.array([100.0, 200.0]), formula, "[M+H]+"
    )
    np.testing.assert_array_equal(parent_vec, parent)  # already computed before the try
    assert not valid.any()
    assert (vectors == 0).all()
