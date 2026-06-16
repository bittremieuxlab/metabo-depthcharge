"""Adducts: vocabulary for categorical encoding and mass deltas for m/z math.

Two distinct uses share this module because they share the same string
keys:

  * ``ADDUCT_VOCAB`` + :func:`encode_adduct` — categorical encoding for an
    embedding layer. The vocab is **checkpoint-stable**: appending new
    entries is safe; reordering or removing entries breaks any model
    trained against the previous indices. Index 0 is reserved for unknown.
  * ``ADDUCT_MASS`` + ``_ADDUCT_NZ`` + :func:`mz_to_neutral_mass` /
    :func:`neutral_mass_to_mz` — mass arithmetic for converting between
    observed precursor m/z and neutral monoisotopic mass. Can be extended
    freely.
"""

_PROTON = 1.007276


#: List of adduct strings carrying a categorical embedding index. Checkpoint-stable:
#: append-only, never reorder or remove without retraining. Each adduct's index
#: is its 1-based position below. Index 0 is reserved for unknown/missing.
#: These are the *canonical* spellings: alternate spellings are recognized
#: as listed and do not get their own index:
#:
#: * ``0`` — unknown / missing
#: * ``1`` — ``[M+H]+``
#: * ``2`` — ``[M+Na]+``
#: * ``3`` — ``[M+K]+``
#: * ``4`` — ``[M+NH4]+``
#: * ``5`` — ``[M]+``
#: * ``6`` — ``[M-H]-``
#: * ``7`` — ``[M+Cl]-``
#: * ``8`` — ``[M+HCOOH-H]-`` (formic acid; alias ``[M+FA-H]-``)
#: * ``9`` — ``[M+CH3COOH-H]-`` (acetic acid; alias ``[M+Hac-H]-``)
#: * ``10`` — ``[2M+Na]+``
#: * ``11`` — ``[2M+H]+``
#: * ``12`` — ``[2M-H]-``
#: * ``13`` — ``[2M+HCOOH-H]-`` (alias ``[2M+FA-H]-``)
#: * ``14`` — ``[2M+CH3COOH-H]-`` (alias ``[2M+Hac-H]-``)
#: * ``15`` — ``[M+Br]-``
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
    "[2M+Na]+",
    "[2M+H]+",
    "[2M-H]-",
    "[2M+HCOOH-H]-",
    "[2M+CH3COOH-H]-",
    "[M+Br]-",
]

#: Alternate spellings
_ADDUCT_ALIASES: dict[str, str] = {
    "[M+FA-H]-": "[M+HCOOH-H]-",
    "[2M+FA-H]-": "[2M+HCOOH-H]-",
    "[M+Hac-H]-": "[M+CH3COOH-H]-",
    "[2M+Hac-H]-": "[2M+CH3COOH-H]-",
}

_ADDUCT_TO_IDX = {a: i + 1 for i, a in enumerate(ADDUCT_VOCAB)}
N_ADDUCTS = len(ADDUCT_VOCAB) + 1  # +1 for unknown at index 0


#: Additive mass shift ``delta`` added to ``n * M`` before dividing by charge
#: ``z``. For multimers the shift is the same as the
#: monomer; only ``n`` (in ``_ADDUCT_NZ``) differs. Cation shifts subtract the
#: electron mass; anion shifts add it. Keyed by canonical adduct string; may
#: include additional adducts not in the categorical vocabulary.
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
    "[2M+H]+": +_PROTON,
    "[2M-H]-": -_PROTON,
    "[2M+Na]+": +22.989218,
    "[2M+HCOOH-H]-": +44.998201,
    "[2M+CH3COOH-H]-": +59.013851,
    "[M+Br]-": +78.918886,  # 79Br (78.918338) + electron
}

#: Per-adduct ``(n_mer, charge)``. Adducts absent here default to ``(1, 1)``.
_ADDUCT_NZ: dict[str, tuple[int, int]] = {
    "[2M+H]+": (2, 1),
    "[2M-H]-": (2, 1),
    "[2M+Na]+": (2, 1),
    "[2M+HCOOH-H]-": (2, 1),
    "[2M+CH3COOH-H]-": (2, 1),
}

assert set(ADDUCT_VOCAB) <= set(ADDUCT_MASS), (
    f"vocab adducts missing from ADDUCT_MASS: "
    f"{sorted(set(ADDUCT_VOCAB) - set(ADDUCT_MASS))}"
)
assert set(_ADDUCT_NZ) <= set(ADDUCT_MASS), (
    f"_ADDUCT_NZ adducts missing from ADDUCT_MASS: "
    f"{sorted(set(_ADDUCT_NZ) - set(ADDUCT_MASS))}"
)
assert set(_ADDUCT_ALIASES.values()) <= set(ADDUCT_VOCAB), (
    f"alias targets not in ADDUCT_VOCAB: "
    f"{sorted(set(_ADDUCT_ALIASES.values()) - set(ADDUCT_VOCAB))}"
)
assert not (set(_ADDUCT_ALIASES) & set(ADDUCT_MASS)), (
    f"alias spellings collide with canonical keys: "
    f"{sorted(set(_ADDUCT_ALIASES) & set(ADDUCT_MASS))}"
)


def _canonical(adduct: str) -> str:
    """Adduct to canonical form"""
    return _ADDUCT_ALIASES.get(adduct, adduct)


def _nmer_charge(adduct: str) -> tuple[int, int]:
    """``(n_mer, charge)`` for a canonical adduct; ``(1, 1)`` when not specified."""
    return _ADDUCT_NZ.get(adduct, (1, 1))


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
    return _ADDUCT_TO_IDX.get(_canonical(adduct_str.strip()), 0)


def mz_to_neutral_mass(precursor_mz: float, adduct: str) -> float:
    """Convert observed precursor m/z to the neutral monoisotopic mass."""
    adduct = _canonical(adduct)
    if adduct not in ADDUCT_MASS:
        raise KeyError(f"unknown adduct {adduct!r}; supported: {sorted(ADDUCT_MASS)}")
    n, z = _nmer_charge(adduct)
    return (float(precursor_mz) * z - ADDUCT_MASS[adduct]) / n


def neutral_mass_to_mz(neutral_mass: float, adduct: str) -> float:
    """Convert a neutral monoisotopic mass to the expected precursor m/z."""
    adduct = _canonical(adduct)
    if adduct not in ADDUCT_MASS:
        raise KeyError(f"unknown adduct {adduct!r}; supported: {sorted(ADDUCT_MASS)}")
    n, z = _nmer_charge(adduct)
    return (n * float(neutral_mass) + ADDUCT_MASS[adduct]) / z
