"""Tests for metabo_depthcharge.data.metadata encoding utilities."""

import numpy as np
import pytest

from metabo_depthcharge.data.metadata import (
    ADDUCT_VOCAB,
    INSTRUMENT_TYPES,
    METADATA_FIELDS,
    N_ADDUCTS,
    N_INSTRUMENTS,
    encode_adduct,
    encode_instrument,
    encode_metadata_arrays,
    parse_collision_energy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_adduct_vocab_length():
    assert len(ADDUCT_VOCAB) == 9


def test_n_adducts_equals_vocab_plus_one():
    assert N_ADDUCTS == len(ADDUCT_VOCAB) + 1


def test_instrument_types_length():
    assert len(INSTRUMENT_TYPES) == 3


def test_n_instruments_equals_types_plus_one():
    assert N_INSTRUMENTS == len(INSTRUMENT_TYPES) + 1


def test_metadata_fields_content():
    assert set(METADATA_FIELDS) == {"adduct", "collision_energy", "instrument_type"}


def test_metadata_fields_length():
    assert len(METADATA_FIELDS) == 3


def test_adduct_vocab_known_members():
    assert "[M+H]+" in ADDUCT_VOCAB
    assert "[M-H]-" in ADDUCT_VOCAB
    assert "[M+Na]+" in ADDUCT_VOCAB


def test_instrument_types_known_members():
    assert "orbitrap" in INSTRUMENT_TYPES
    assert "qtof" in INSTRUMENT_TYPES
    assert "iontrap" in INSTRUMENT_TYPES


# ---------------------------------------------------------------------------
# encode_adduct
# ---------------------------------------------------------------------------


def test_encode_adduct_known_first():
    # Index 0 is unknown; known adducts start at 1
    assert encode_adduct("[M+H]+") == 1


def test_encode_adduct_all_vocab_members_nonzero():
    """Every vocab member must map to a unique nonzero index."""
    indices = [encode_adduct(a) for a in ADDUCT_VOCAB]
    assert all(i > 0 for i in indices)
    assert len(set(indices)) == len(ADDUCT_VOCAB), "indices must be unique"


def test_encode_adduct_indices_are_consecutive():
    """Vocab indices should be 1..N with no gaps."""
    indices = sorted(encode_adduct(a) for a in ADDUCT_VOCAB)
    assert indices == list(range(1, len(ADDUCT_VOCAB) + 1))


def test_encode_adduct_empty_string():
    assert encode_adduct("") == 0


def test_encode_adduct_nan_string():
    assert encode_adduct("nan") == 0


def test_encode_adduct_none_string():
    assert encode_adduct("None") == 0


def test_encode_adduct_unknown_string():
    assert encode_adduct("[M+X]++") == 0


def test_encode_adduct_strips_whitespace():
    assert encode_adduct("  [M+H]+  ") == encode_adduct("[M+H]+")


def test_encode_adduct_case_sensitive():
    """encode_adduct is case-sensitive: wrong case → 0."""
    assert encode_adduct("[m+h]+") == 0
    assert encode_adduct("[M+H]+".upper()) == 0  # "[M+H]+" fully upper → no match


def test_encode_adduct_negative_mode():
    assert encode_adduct("[M-H]-") > 0


def test_encode_adduct_mplus_only():
    assert encode_adduct("[M]+") > 0


# ---------------------------------------------------------------------------
# encode_instrument
# ---------------------------------------------------------------------------


def test_encode_instrument_known_orbitrap():
    assert encode_instrument("orbitrap") == 1


def test_encode_instrument_all_types_nonzero():
    """Every instrument type must map to a unique nonzero index."""
    indices = [encode_instrument(t) for t in INSTRUMENT_TYPES]
    assert all(i > 0 for i in indices)
    assert len(set(indices)) == len(INSTRUMENT_TYPES)


def test_encode_instrument_indices_are_consecutive():
    indices = sorted(encode_instrument(t) for t in INSTRUMENT_TYPES)
    assert indices == list(range(1, len(INSTRUMENT_TYPES) + 1))


def test_encode_instrument_case_insensitive_upper():
    assert encode_instrument("ORBITRAP") == encode_instrument("orbitrap")


def test_encode_instrument_case_insensitive_mixed():
    assert encode_instrument("Qtof") == encode_instrument("qtof")


def test_encode_instrument_case_insensitive_iontrap():
    assert encode_instrument("IONTRAP") == encode_instrument("iontrap")


def test_encode_instrument_strips_whitespace():
    assert encode_instrument("  orbitrap  ") == encode_instrument("orbitrap")


def test_encode_instrument_empty_string():
    assert encode_instrument("") == 0


def test_encode_instrument_nan_string():
    assert encode_instrument("nan") == 0


def test_encode_instrument_none_string():
    assert encode_instrument("None") == 0


def test_encode_instrument_unknown_string():
    assert encode_instrument("spectrophotometer") == 0


# ---------------------------------------------------------------------------
# parse_collision_energy
# ---------------------------------------------------------------------------


def test_parse_ce_none():
    assert parse_collision_energy(None) == 0.0


def test_parse_ce_nan_float():
    assert parse_collision_energy(float("nan")) == 0.0


def test_parse_ce_np_nan():
    assert parse_collision_energy(np.nan) == 0.0


def test_parse_ce_integer_numeric():
    """Integer input is returned as float directly — NOT divided by 100."""
    result = parse_collision_energy(40)
    assert result == pytest.approx(40.0)


def test_parse_ce_float_numeric():
    """Float input returned directly — NOT divided by 100."""
    result = parse_collision_energy(35.5)
    assert result == pytest.approx(35.5)


def test_parse_ce_np_integer():
    result = parse_collision_energy(np.int64(50))
    assert result == pytest.approx(50.0)


def test_parse_ce_np_float():
    result = parse_collision_energy(np.float32(25.0))
    assert result == pytest.approx(25.0, abs=1e-4)


def test_parse_ce_string_simple():
    """String '40' → 40/100 = 0.4."""
    result = parse_collision_energy("40")
    assert result == pytest.approx(0.40)


def test_parse_ce_string_float():
    result = parse_collision_energy("35.5")
    assert result == pytest.approx(0.355)


def test_parse_ce_string_empty():
    assert parse_collision_energy("") == 0.0


def test_parse_ce_string_nan():
    assert parse_collision_energy("nan") == 0.0


def test_parse_ce_string_none():
    assert parse_collision_energy("None") == 0.0


def test_parse_ce_stepped_space_delimiter():
    """'10 20 40' → mean(10,20,40)/100 = 0.2333..."""
    result = parse_collision_energy("10 20 40")
    assert result == pytest.approx(np.mean([10, 20, 40]) / 100, rel=1e-5)


def test_parse_ce_stepped_comma_delimiter():
    result = parse_collision_energy("10, 20, 40")
    assert result == pytest.approx(np.mean([10, 20, 40]) / 100, rel=1e-5)


def test_parse_ce_stepped_semicolon_delimiter():
    result = parse_collision_energy("10; 20; 40")
    assert result == pytest.approx(np.mean([10, 20, 40]) / 100, rel=1e-5)


def test_parse_ce_stepped_two_values():
    result = parse_collision_energy("20 60")
    assert result == pytest.approx(np.mean([20, 60]) / 100, rel=1e-5)


def test_parse_ce_unparseable_string():
    """Completely unparseable string → 0.0."""
    assert parse_collision_energy("not-a-number") == 0.0


def test_parse_ce_zero_string():
    """String '0' → 0.0 (0/100)."""
    assert parse_collision_energy("0") == pytest.approx(0.0)


def test_parse_ce_zero_int():
    """Numeric 0 → 0.0 (no division)."""
    assert parse_collision_energy(0) == pytest.approx(0.0)


def test_parse_ce_returns_float():
    assert isinstance(parse_collision_energy(40), float)
    assert isinstance(parse_collision_energy("40"), float)


# ---------------------------------------------------------------------------
# encode_metadata_arrays
# ---------------------------------------------------------------------------


def test_encode_metadata_arrays_adduct_string():
    raw = np.array(["[M+H]+", "[M-H]-", "unknown"])
    out = encode_metadata_arrays(raw, "adduct")
    assert out.dtype == np.int64
    assert out[0] == encode_adduct("[M+H]+")
    assert out[1] == encode_adduct("[M-H]-")
    assert out[2] == 0  # unknown → 0


def test_encode_metadata_arrays_adduct_bytes():
    raw = np.array([b"[M+H]+", b"[M+Na]+", b"unknown"])
    out = encode_metadata_arrays(raw, "adduct")
    assert out.dtype == np.int64
    assert out[0] == encode_adduct("[M+H]+")
    assert out[1] == encode_adduct("[M+Na]+")
    assert out[2] == 0


def test_encode_metadata_arrays_adduct_shape():
    raw = np.array(["[M+H]+"] * 5)
    out = encode_metadata_arrays(raw, "adduct")
    assert out.shape == (5,)


def test_encode_metadata_arrays_instrument_string():
    raw = np.array(["orbitrap", "qtof", "unknown_instrument"])
    out = encode_metadata_arrays(raw, "instrument_type")
    assert out.dtype == np.int64
    assert out[0] == encode_instrument("orbitrap")
    assert out[1] == encode_instrument("qtof")
    assert out[2] == 0


def test_encode_metadata_arrays_instrument_bytes():
    raw = np.array([b"orbitrap", b"IONTRAP"])
    out = encode_metadata_arrays(raw, "instrument_type")
    assert out.dtype == np.int64
    assert out[0] == encode_instrument("orbitrap")
    assert out[1] == encode_instrument("iontrap")


def test_encode_metadata_arrays_ce_string():
    raw = np.array(["40", "20", "0"])
    out = encode_metadata_arrays(raw, "collision_energy")
    assert out.dtype == np.float32
    assert float(out[0]) == pytest.approx(0.40, abs=1e-5)
    assert float(out[1]) == pytest.approx(0.20, abs=1e-5)
    assert float(out[2]) == pytest.approx(0.0, abs=1e-5)


def test_encode_metadata_arrays_ce_bytes():
    raw = np.array([b"40", b"nan"])
    out = encode_metadata_arrays(raw, "collision_energy")
    assert out.dtype == np.float32
    assert float(out[0]) == pytest.approx(0.40, abs=1e-5)
    assert float(out[1]) == pytest.approx(0.0, abs=1e-5)


def test_encode_metadata_arrays_ce_numeric():
    """Numeric values in the array are passed directly to parse_collision_energy."""
    raw = np.array([40.0, 20.0, np.nan])
    out = encode_metadata_arrays(raw, "collision_energy")
    assert out.dtype == np.float32
    # numeric 40 → 40.0 (no /100 for numeric path)
    assert float(out[0]) == pytest.approx(40.0, abs=1e-3)
    assert float(out[2]) == pytest.approx(0.0, abs=1e-5)


def test_encode_metadata_arrays_empty_input():
    raw = np.array([], dtype=object)
    out = encode_metadata_arrays(raw, "adduct")
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_encode_metadata_arrays_empty_ce():
    raw = np.array([], dtype=object)
    out = encode_metadata_arrays(raw, "collision_energy")
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_encode_metadata_arrays_unknown_field():
    with pytest.raises(ValueError, match="Unknown metadata field"):
        encode_metadata_arrays(np.array(["x"]), "nonexistent_field")


def test_encode_metadata_arrays_unknown_field_message():
    with pytest.raises(ValueError, match="nonexistent_field"):
        encode_metadata_arrays(np.array(["x"]), "nonexistent_field")


# ---------------------------------------------------------------------------
# Public API from metabo_depthcharge.data
# ---------------------------------------------------------------------------


def test_data_module_exports_adduct_vocab():
    from metabo_depthcharge.data import ADDUCT_VOCAB as av

    assert isinstance(av, list)
    assert len(av) > 0


def test_data_module_exports_instrument_types():
    from metabo_depthcharge.data import INSTRUMENT_TYPES as it

    assert isinstance(it, list)
    assert len(it) > 0


def test_data_module_exports_n_adducts():
    from metabo_depthcharge.data import N_ADDUCTS as na

    assert na == N_ADDUCTS


def test_data_module_exports_n_instruments():
    from metabo_depthcharge.data import N_INSTRUMENTS as ni

    assert ni == N_INSTRUMENTS


def test_data_module_exports_metadata_fields():
    from metabo_depthcharge.data import METADATA_FIELDS as mf

    assert mf == METADATA_FIELDS


def test_data_module_exports_encode_adduct():
    from metabo_depthcharge.data import encode_adduct as ea

    assert callable(ea)
    assert ea("[M+H]+") == encode_adduct("[M+H]+")


def test_data_module_exports_encode_instrument():
    from metabo_depthcharge.data import encode_instrument as ei

    assert callable(ei)
    assert ei("orbitrap") == encode_instrument("orbitrap")


def test_data_module_exports_parse_collision_energy():
    from metabo_depthcharge.data import parse_collision_energy as pce

    assert callable(pce)
    assert pce("40") == parse_collision_energy("40")


def test_data_module_exports_encode_metadata_arrays():
    from metabo_depthcharge.data import encode_metadata_arrays as ema

    assert callable(ema)
