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
#: * ``16`` — ``[M-H2O+H]+``
#: * ``17`` — ``[M-2H2O+H]+``
#: * ``18`` — ``[M+2H]2+``
#: * ``19`` — ``[2M+Na-2H]-``
#: * ``20`` — ``[M-H2O]+``
#: * ``21`` — ``[M-CH3]-``
#: * ``22`` — ``[M-2H]-``
#: * ``23`` — ``[M]-``
#: * ``24`` — ``[2M+NH4]+``
#: * ``25`` — ``[M-H2O-H]-``
#: * ``26`` — ``[M-H5O3]+``
#: * ``27`` — ``[M+C2H4N]+`` (acetonitrile adduct, alias ``[M+ACN+H]+``)
#: * ``28`` — ``[M+3H]3+``
#: * ``29`` — ``[M+2Na-H]+``
#: * ``30`` — ``[2M+K]+``
#: * ``31`` — ``[M-H2]2-``
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
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M+2H]2+",
    "[2M+Na-2H]-",
    "[M-H2O]+",
    "[M-CH3]-",
    "[M-2H]-",
    "[M]-",
    "[2M+NH4]+",
    "[M-H2O-H]-",
    "[M-H5O3]+",
    "[M+C2H4N]+",
    "[M+3H]3+",
    "[M+2Na-H]+",
    "[2M+K]+",
    "[M-H2]2-",
]

#: Alternate spellings
_ADDUCT_ALIASES: dict[str, str] = {
    "[M+FA-H]-": "[M+HCOOH-H]-",
    "[2M+FA-H]-": "[2M+HCOOH-H]-",
    "[M+Hac-H]-": "[M+CH3COOH-H]-",
    "[2M+Hac-H]-": "[2M+CH3COOH-H]-",
    "[M+CH2O2-H]-": "[M+HCOOH-H]-",
    "[2M+CH2O2-H]-": "[2M+HCOOH-H]-",
    "[M+C2H4O2-H]-": "[M+CH3COOH-H]-",
    "[2M+C2H4O2-H]-": "[2M+CH3COOH-H]-",
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
    "[M]+": -0.000549,  # radical cation: electron loss, no atom change
    "[M]-": +0.000549,  # radical anion: electron capture, no atom change
    "[2M+H]+": +_PROTON,
    "[2M-H]-": -_PROTON,
    "[2M+Na]+": +22.989218,
    "[2M+HCOOH-H]-": +44.998201,
    "[2M+CH3COOH-H]-": +59.013851,
    "[M+Br]-": +78.918886,  # 79Br (78.918338) + electron
    "[M-H2O+H]+": -17.003289,  # -H2O (neutral) + H+
    "[M-2H2O+H]+": -35.013854,  # -2 H2O (neutral) + H+
    "[M+2H]2+": 2 * _PROTON,
    "[2M+Na-2H]-": 20.974666,  # +Na (as in [M+Na]+) - 2 H+, net charge -1
    "[M-H2O]+": -18.011114,  # -H2O (neutral), cation via electron loss
    "[M-CH3]-": -15.022926,  # -CH3 (neutral methyl radical)
    "[M-2H]-": -2.015101,  # -2 H (neutral), single net charge (field convention)
    "[2M+NH4]+": +18.033823,  # same shift as [M+NH4]+; n=2 via _ADDUCT_NZ
    "[M-H2O-H]-": -19.017841,  # -H2O (neutral) - H+
    "[M-H5O3]+": -53.024419,  # -H5O3 (neutral fragment)
    "[M+C2H4N]+": +42.033825,  # acetonitrile adduct, alias [M+ACN+H]+
    "[M+3H]3+": 3 * _PROTON,
    "[M+2Na-H]+": +44.971166,  # +2 Na (neutral) - H (neutral), cation
    "[2M+K]+": +38.963158,  # same shift as [M+K]+; n=2 via _ADDUCT_NZ
    "[M-H2]2-": -2.014553,  # -H2 (neutral), doubly charged anion
}

#: Per-adduct ``(n_mer, charge)``. Adducts absent here default to ``(1, 1)``.
_ADDUCT_NZ: dict[str, tuple[int, int]] = {
    "[2M+H]+": (2, 1),
    "[2M-H]-": (2, 1),
    "[2M+Na]+": (2, 1),
    "[2M+HCOOH-H]-": (2, 1),
    "[2M+CH3COOH-H]-": (2, 1),
    "[2M+Na-2H]-": (2, 1),
    "[2M+NH4]+": (2, 1),
    "[2M+K]+": (2, 1),
    "[M+2H]2+": (1, 2),
    "[M+3H]3+": (1, 3),
    "[M-H2]2-": (1, 2),
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
