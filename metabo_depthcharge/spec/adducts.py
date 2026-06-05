"""Adducts: vocabulary for categorical encoding and mass deltas for m/z math.

Two distinct uses share this module because they share the same string
keys:

  * ``ADDUCT_VOCAB`` + :func:`encode_adduct` — categorical encoding for an
    embedding layer. The vocab is **checkpoint-stable**: appending new
    entries is safe; reordering or removing entries breaks any model
    trained against the previous indices. Index 0 is reserved for unknown.
  * ``ADDUCT_MASS`` + :func:`mz_to_neutral_mass` /
    :func:`neutral_mass_to_mz` — mass-delta arithmetic for converting
    between observed precursor m/z and neutral monoisotopic mass. Can be
    extended freely.

Every entry in ``ADDUCT_VOCAB`` must have a corresponding mass delta in
``ADDUCT_MASS`` (asserted at import).
"""

_PROTON = 1.007276


ADDUCT_VOCAB = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M+NH4]+",
    "[M]+",
    "[M-H]-",
    "[M+Cl]-",
    "[M+HCOOH-H]-",
    "[M+CH3COOH-H]-",
]
_ADDUCT_TO_IDX = {a: i + 1 for i, a in enumerate(ADDUCT_VOCAB)}
N_ADDUCTS = len(ADDUCT_VOCAB) + 1  # +1 for unknown at index 0


# Mass added to neutral M to give the observed precursor m/z. Keys aligned
# with ADDUCT_VOCAB where overlapping; may include additional adducts not
# in the categorical vocabulary.
ADDUCT_MASS: dict[str, float] = {
    "[M+H]+": +_PROTON,
    "[M-H]-": -_PROTON,
    "[M+Na]+": +22.989218,
    "[M+K]+": +38.963158,
    "[M+NH4]+": +18.033823,
    "[M+Cl]-": +34.969402,
    "[M+CH3COOH-H]-": +59.013851,
    "[M+HCOOH-H]-": +44.998201,
    "[M]+": 0.0,
    "[M]-": 0.0,
}
assert set(ADDUCT_VOCAB) <= set(ADDUCT_MASS), (
    f"vocab adducts missing from ADDUCT_MASS: "
    f"{sorted(set(ADDUCT_VOCAB) - set(ADDUCT_MASS))}"
)


def encode_adduct(adduct_str: str) -> int:
    """Vocab index for the adduct string; 0 for unknown/missing."""
    if not adduct_str or adduct_str in ("nan", "None", ""):
        return 0
    return _ADDUCT_TO_IDX.get(adduct_str.strip(), 0)


def mz_to_neutral_mass(precursor_mz: float, adduct: str) -> float:
    """Convert observed precursor m/z to the neutral monoisotopic mass."""
    if adduct not in ADDUCT_MASS:
        raise KeyError(f"unknown adduct {adduct!r}; supported: {sorted(ADDUCT_MASS)}")
    return float(precursor_mz) - ADDUCT_MASS[adduct]


def neutral_mass_to_mz(neutral_mass: float, adduct: str) -> float:
    """Convert a neutral monoisotopic mass to the expected precursor m/z."""
    if adduct not in ADDUCT_MASS:
        raise KeyError(f"unknown adduct {adduct!r}; supported: {sorted(ADDUCT_MASS)}")
    return float(neutral_mass) + ADDUCT_MASS[adduct]
