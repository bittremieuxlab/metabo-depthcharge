"""Peak subformula assignment as in MIST."""

import numpy as np

from metabo_depthcharge.mist_cf.common.chem_utils import (
    ELEMENT_DIM,
    ION_LST,
    clipped_ppm,
    formula_to_dense,
    get_all_subsets,
    ion_remap,
    ion_to_mass,
)


def assign_peak_subformulae(
    mz: np.ndarray,
    formula: str,
    adduct: str,
    ppm: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign MIST-style subformulae to spectrum peaks.

    For each peak, finds the closest subformula of ``formula`` within
    ``ppm`` and records its bag-of-atoms vector. Unmatched peaks get a
    zero vector. Returns zero vectors for empty, invalid, or unsupported
    formula/adduct combinations without raising.

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
    """
    n_peaks = len(mz)
    vectors = np.zeros((n_peaks, ELEMENT_DIM), dtype=np.int16)
    parent_vec = np.zeros(ELEMENT_DIM, dtype=np.int16)
    valid = np.zeros(n_peaks, dtype=bool)

    if not isinstance(formula, str) or not formula:
        return vectors.flatten(), parent_vec, valid
    try:
        parent_vec = formula_to_dense(formula).astype(np.int16)
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
