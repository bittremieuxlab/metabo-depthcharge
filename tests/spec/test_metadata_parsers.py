"""Tests for :mod:`metabo_depthcharge.spec.metadata_parsers`."""

import numpy as np

from metabo_depthcharge.spec.metadata_parsers import (
    encode_collision_energy,
    encode_instrument,
    encode_ion_activation,
    encode_ionization_method,
)


# --- encode_instrument ----------------------------------------------


def test_encode_instrument_known():
    assert encode_instrument("orbitrap") == 1
    assert encode_instrument("qtof") == 2
    assert encode_instrument("Q-ToF") == 2
    assert encode_instrument("iontrap") == 3


def test_encode_instrument_unknown():
    assert encode_instrument("") == 0
    assert encode_instrument(None) == 0
    assert encode_instrument("UNKNOWN") == 0


def test_encode_instrument_case_insensitive():
    assert encode_instrument("orbitrap") == encode_instrument("ORBITRAP")


# --- encode_collision_energy ----------------------------------------------


def test_encode_collision_energy_known():
    assert encode_collision_energy(55) == 0.55
    assert encode_collision_energy("75") == 0.75
    assert encode_collision_energy("20,40,60") == 0.40
    assert encode_collision_energy("30 50 70") == 0.50
    assert encode_collision_energy("10;60;80") == 0.50


def test_encode_collision_energy_unknown():
    assert encode_collision_energy("") == 0
    assert encode_collision_energy(np.nan) == 0
    assert encode_collision_energy(None) == 0
    assert encode_collision_energy("SOME RANDOM TEXT") == 0


# --- encode_ion_activation ----------------------------------------------


def test_encode_ion_activation_known():
    assert encode_ion_activation("HCD") == 1
    assert encode_ion_activation("CID") == 2


def test_encode_ion_activation_unknown():
    assert encode_ion_activation("") == 0
    assert encode_ion_activation(None) == 0
    assert encode_ion_activation("UNKNOWN") == 0


def test_encode_ion_activation_case_insensitive():
    assert encode_ion_activation("hcd") == encode_ion_activation("HCD")


# --- encode_ionization_method -------------------------------------------


def test_encode_ionization_method_known():
    assert encode_ionization_method("NSI") == 1
    assert encode_ionization_method("ESI") == 2
    assert encode_ionization_method("APCI") == 3


def test_encode_ionization_method_unknown():
    assert encode_ionization_method("") == 0
    assert encode_ionization_method(None) == 0
    assert encode_ionization_method("MALDI") == 0


def test_encode_ionization_method_case_insensitive():
    assert encode_ionization_method("esi") == encode_ionization_method("ESI")
