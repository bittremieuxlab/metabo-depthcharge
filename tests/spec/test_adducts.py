"""Tests for :mod:`metabo_depthcharge.spec.adducts`."""

import pytest

from metabo_depthcharge.spec.adducts import (
    encode_adduct,
    mz_to_neutral_mass,
    neutral_mass_to_mz,
)


def test_encode_adduct():
    assert encode_adduct("[M+H]+") == 1
    assert encode_adduct("UNKNOWN") == 0
    assert encode_adduct(None) == 0
    assert encode_adduct("") == 0


@pytest.mark.parametrize(
    "adduct", ["[M+H]+", "[M+Na]+", "[M-H]-", "[2M+H]+", "[2M+CH3COOH-H]-"]
)
def test_mz_to_neutral_mass_and_back(adduct):
    mz = 500.0
    neutral_mass = mz_to_neutral_mass(mz, adduct)
    mz_back = neutral_mass_to_mz(neutral_mass, adduct)
    assert abs(mz - mz_back) < 1e-6


def test_conversion_with_unknown_adduct_raise():
    with pytest.raises(KeyError):
        mz_to_neutral_mass(500, "UNKNOWN")
    with pytest.raises(KeyError):
        neutral_mass_to_mz(500, "UNKNOWN")
