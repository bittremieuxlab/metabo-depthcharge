"""Peak subformula assignment as in MIST."""

import numpy as np

from metabo_depthcharge.mist_cf.common.chem_utils import (
    ELEMENT_DIM,
    ION_LST,
    clipped_ppm,
    get_all_subsets,
    ion_remap,
    ion_to_mass,
)
from metabo_depthcharge.mist_cf.common.chem_utils import (
    formula_to_dense as _formula_to_dense,
)


def formula_to_dense(formula: str) -> np.ndarray:
    """Convert a molecular formula string to a bag-of-atoms count vector as in MIST-CF.

    Parses ``formula`` into per-element atom counts: a dense vector of length
    ``ELEMENT_DIM`` (18). This is the representation consumed by
    :class:`~metabo_depthcharge.encoders.SubformulaEncoder`, and the building
    block applied per peak by :func:`assign_peak_subformulae`.

    Vector positions correspond, in order, to the supported elements::

        C, N, P, O, S, Si, I, H, Cl, F, Br, B, Se, Fe, Co, As, K, Na

    Thin wrapper around the vendored MIST-CF implementation
    (``metabo_depthcharge.mist_cf.common.chem_utils.formula_to_dense``).

    Parameters
    ----------
    formula : str
        Molecular formula, e.g. ``"C9H8O4"``. An empty string yields an
        all-zero vector.

    Returns
    -------
    np.ndarray
        Bag-of-atoms counts, shape ``(ELEMENT_DIM,)``, dtype ``int16``.

    Raises
    ------
    KeyError
        If ``formula`` references an element outside the supported set above.

    Examples
    --------
    >>> formula_to_dense("C9H8O4")
    array([9, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int16)
    """
    return _formula_to_dense(formula).astype(np.int16)


def assign_peak_subformulae(
    mz: np.ndarray,
    formula: str,
    adduct: str,
    ppm: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign MIST-CF-style subformulae to spectrum peaks.

    For each peak, finds the closest subformula of ``formula`` within
    ``ppm`` and records its bag-of-atoms vector. Unmatched peaks get a
    zero vector. Returns zero vectors for empty, invalid, or unsupported
    formula/adduct combinations.

    Parameters
    ----------
    mz : np.ndarray
        Peak m/z values, shape ``(N_peaks,)``.
    formula : str
        Precursor molecular formula, e.g. ``"C9H8O4"``.
    adduct : str
        Adduct string, e.g. ``"[M+H]+"``.
    ppm : float
        Mass tolerance in ppm for subformula matching.

    Returns
    -------
    vectors : np.ndarray
        Flattened bag-of-atoms vectors, shape ``(N_peaks * ELEMENT_DIM,)``,
        dtype ``int16``. Zero for unmatched peaks.
    parent_vec : np.ndarray
        Parent formula bag-of-atoms, shape ``(ELEMENT_DIM,)``, dtype ``int16``.
        Zero if formula is empty or invalid.
    valid : np.ndarray
        Boolean mask, shape ``(N_peaks,)``. ``True`` for peaks that were
        matched to a subformula within ``ppm``.

    See Also
    --------
    ~metabo_depthcharge.datasets.SpectrumDataset.add_subformulae : Dataset-level
        method that applies this per spectrum across a whole ``SpectrumDataset``,
        baking the results into ``{name}_subformula_vec`` and
        ``{name}_parent_formula_vec`` columns.
    """
    n_peaks = len(mz)
    vectors = np.zeros((n_peaks, ELEMENT_DIM), dtype=np.int16)
    parent_vec = np.zeros(ELEMENT_DIM, dtype=np.int16)
    valid = np.zeros(n_peaks, dtype=bool)

    if not isinstance(formula, str) or not formula:
        return vectors.flatten(), parent_vec, valid
    try:
        parent_vec = formula_to_dense(formula)
    except Exception:
        return vectors.flatten(), parent_vec, valid

    if not isinstance(adduct, str) or not adduct:
        return vectors.flatten(), parent_vec, valid

    adduct = ion_remap.get(adduct, adduct)
    if adduct not in ION_LST or n_peaks == 0:
        return vectors.flatten(), parent_vec, valid

    try:
        cross_prod, masses = get_all_subsets(formula)
    except Exception:
        return vectors.flatten(), parent_vec, valid

    masses_with_ion = masses + ion_to_mass[adduct]
    abs_diffs = np.abs(mz[:, None] - masses_with_ion[None, :])
    best_inds = abs_diffs.argmin(axis=1)
    min_abs_diff = abs_diffs[np.arange(n_peaks), best_inds]
    rel_diff = clipped_ppm(min_abs_diff, mz)

    valid = rel_diff < ppm
    vectors[valid] = cross_prod[best_inds[valid]].astype(np.int16)

    return vectors.flatten(), parent_vec, valid
