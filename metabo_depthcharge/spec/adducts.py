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


#: List of adduct strings carrying a categorical embedding index. Checkpoint-stable:
#: append-only, never reorder or remove without retraining. Each adduct's index
#: is its 1-based position below; index 0 is reserved for unknown/missing:
#:
#: * ``0`` — unknown / missing
#: * ``1`` — ``[M+H]+``
#: * ``2`` — ``[M+Na]+``
#: * ``3`` — ``[M+K]+``
#: * ``4`` — ``[M+NH4]+``
#: * ``5`` — ``[M]+``
#: * ``6`` — ``[M-H]-``
#: * ``7`` — ``[M+Cl]-``
#: * ``8`` — ``[M+HCOOH-H]-``
#: * ``9`` — ``[M+CH3COOH-H]-``
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


#: Mass added to neutral M to give the observed precursor m/z. Keys aligned
#: with ``ADDUCT_VOCAB`` where overlapping; may include additional adducts not
#: in the categorical vocabulary.
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
    """Vocab index for the adduct string; 0 for unknown/missing.

    See Also
    --------
    ADDUCT_VOCAB : The adduct-to-index vocabulary this looks up, including the
        full index assignment.
    ~metabo_depthcharge.spec.metadata_parsers.METADATA_PARSERS : Registry that
        wires this in as the row-wise parser for the ``adduct`` metadata field,
        consumed by ``SpectrumDataset`` at build time.
    """
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
