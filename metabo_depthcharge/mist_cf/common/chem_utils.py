# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""chem_utils.py"""

import json
import re
from functools import reduce

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Atom
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
    "K",
    "Na",
]

VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]

CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze(), strict=False))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

NORM_VEC = np.array([81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 2])
NORM_VEC_MASS = np.array(NORM_VEC.tolist() + [1471])

MAX_ELEMENT_NUM = 64

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS)), strict=False))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS, strict=False))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS, strict=False))

# Canonical adduct order. Indices 0-10 are the original alphabet and MUST stay
# put: the model's ion embedding is one-hot indexed by position in ION_LST, so
# reordering would silently invalidate every prior checkpoint. New adducts are
# appended (indices 11+) so existing indices never move.
#
# Multimer ([2M+...]) adducts are supported via ion_to_nmer: the neutral
# *monomer* mass relates to the precursor m/z by an n-mer factor (see
# precursor_mz_to_neutral_mass / neutral_mass_to_precursor_mz). Charge is
# assumed 1 for every adduct here. This alphabet is aligned with
# metabo_depthcharge.spec.adducts.ADDUCT_VOCAB.
_ION_LST_BASE = [
    # positive (indices 0-6)
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",  # == [M+NH4]+
    "[M]+",
    "[M-H4O2+H]+",
    # negative (indices 7-10)
    "[M-H]-",
    "[M+Cl]-",
    "[M+HCOOH-H]-",  # formate adduct: net +CHO2
    "[M+CH3COOH-H]-",  # acetate adduct: net +C2H3O2
]
_ION_LST_EXTRA = [
    # appended (indices 11+) to align with metabo-depthcharge's ADDUCT_VOCAB
    "[M+Br]-",  # bromide (monomer)
    "[2M+H]+",
    "[2M+Na]+",
    "[2M-H]-",
    "[2M+HCOOH-H]-",
    "[2M+CH3COOH-H]-",
]
ION_LST = _ION_LST_BASE + _ION_LST_EXTRA

ion_to_mode = {
    "[M+H]+": "pos",
    "[M+Na]+": "pos",
    "[M+K]+": "pos",
    "[M-H2O+H]+": "pos",
    "[M+H3N+H]+": "pos",
    "[M]+": "pos",
    "[M-H4O2+H]+": "pos",
    "[2M+H]+": "pos",
    "[2M+Na]+": "pos",
    "[M-H]-": "neg",
    "[M+Cl]-": "neg",
    "[M+HCOOH-H]-": "neg",
    "[M+CH3COOH-H]-": "neg",
    "[M+Br]-": "neg",
    "[2M-H]-": "neg",
    "[2M+HCOOH-H]-": "neg",
    "[2M+CH3COOH-H]-": "neg",
}
assert set(ion_to_mode) == set(ION_LST), "ion_to_mode must cover exactly ION_LST"

# Number of monomers M in the adduct (1 for [M+...], 2 for [2M+...]). With charge
# fixed at 1 this gives precursor_mz = nmer * neutral_monomer_mass + ion_to_mass.
ion_to_nmer = dict.fromkeys(ION_LST, 1)
ion_to_nmer.update(
    {
        "[2M+H]+": 2,
        "[2M+Na]+": 2,
        "[2M-H]-": 2,
        "[2M+HCOOH-H]-": 2,
        "[2M+CH3COOH-H]-": 2,
    }
)

# Mode-membership lists, derived so they always track ION_LST / ion_to_mode.
POS_ION_LST = [i for i in ION_LST if ion_to_mode[i] == "pos"]
NEG_ION_LST = [i for i in ION_LST if ion_to_mode[i] == "neg"]
ion_mode_to_ions = {"pos": POS_ION_LST, "neg": NEG_ION_LST}

ion_remap = dict(zip(ION_LST, ION_LST, strict=False))
ion_remap.update(
    {
        # positive-mode aliases
        "[M+NH4]+": "[M+H3N+H]+",
        "M+H": "[M+H]+",
        "M+Na": "[M+Na]+",
        "M+H-H2O": "[M-H2O+H]+",
        "M-H2O+H": "[M-H2O+H]+",
        "M+NH4": "[M+H3N+H]+",
        "M-2H2O+H": "[M-H4O2+H]+",
        "[M-2H2O+H]+": "[M-H4O2+H]+",
        "[M-2(H2O)+H]+": "[M-H4O2+H]+",
        "[M+H-2H2O]+": "[M-H4O2+H]+",
        "[M+H-H2O]+": "[M-H2O+H]+",
        "2M+H": "[2M+H]+",
        "2M+Na": "[2M+Na]+",
        # negative-mode aliases
        "M-H": "[M-H]-",
        "[M-H]": "[M-H]-",
        "M+Cl": "[M+Cl]-",
        "M+Br": "[M+Br]-",
        "[M+FA-H]-": "[M+HCOOH-H]-",
        "M+HCOO-": "[M+HCOOH-H]-",
        "[M+HCOO]-": "[M+HCOOH-H]-",
        "[M+AcOH-H]-": "[M+CH3COOH-H]-",
        "[M+Hac-H]-": "[M+CH3COOH-H]-",
        "[M+OAc]-": "[M+CH3COOH-H]-",
        "[M+CH3COO]-": "[M+CH3COOH-H]-",
        "2M-H": "[2M-H]-",
        "[2M-H]": "[2M-H]-",
        "[2M+FA-H]-": "[2M+HCOOH-H]-",
        "[2M+Hac-H]-": "[2M+CH3COOH-H]-",
        "[2M+AcOH-H]-": "[2M+CH3COOH-H]-",
    }
)

ion_to_idx = dict(zip(ION_LST, np.arange(len(ION_LST)), strict=False))

# Positive ions LOSE an electron (subtract ELECTRON_MASS).
# Negative ions GAIN an electron (add ELECTRON_MASS).
ion_to_mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M+Cl]-": ELEMENT_TO_MASS["Cl"] + ELECTRON_MASS,
    "[M+HCOOH-H]-": ELEMENT_TO_MASS["C"]
    + ELEMENT_TO_MASS["H"]
    + 2 * ELEMENT_TO_MASS["O"]
    + ELECTRON_MASS,
    "[M+CH3COOH-H]-": 2 * ELEMENT_TO_MASS["C"]
    + 3 * ELEMENT_TO_MASS["H"]
    + 2 * ELEMENT_TO_MASS["O"]
    + ELECTRON_MASS,
    "[M+Br]-": ELEMENT_TO_MASS["Br"] + ELECTRON_MASS,
    # Multimers: same additive shift as their monomer counterpart; only the
    # n-mer factor (ion_to_nmer) differs.
    "[2M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[2M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[2M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[2M+HCOOH-H]-": ELEMENT_TO_MASS["C"]
    + ELEMENT_TO_MASS["H"]
    + 2 * ELEMENT_TO_MASS["O"]
    + ELECTRON_MASS,
    "[2M+CH3COOH-H]-": 2 * ELEMENT_TO_MASS["C"]
    + 3 * ELEMENT_TO_MASS["H"]
    + 2 * ELEMENT_TO_MASS["O"]
    + ELECTRON_MASS,
}

ion_to_add_vec = {
    "[M+H]+": element_to_position["H"],
    "[M+Na]+": element_to_position["Na"],
    "[M+K]+": element_to_position["K"],
    "[M-H2O+H]+": -element_to_position["O"] - element_to_position["H"],
    "[M+H3N+H]+": element_to_position["N"] + element_to_position["H"] * 4,
    "[M]+": np.zeros_like(element_to_position["H"]),
    "[M-H4O2+H]+": -element_to_position["O"] * 2 - element_to_position["H"] * 3,
    "[M-H]-": -element_to_position["H"],
    "[M+Cl]-": element_to_position["Cl"],
    "[M+HCOOH-H]-": element_to_position["C"]
    + element_to_position["H"]
    + 2 * element_to_position["O"],
    "[M+CH3COOH-H]-": 2 * element_to_position["C"]
    + 3 * element_to_position["H"]
    + 2 * element_to_position["O"],
    "[M+Br]-": element_to_position["Br"],
    # Multimers: same atom offset as their monomer counterpart. add_ion combines
    # this with nmer * M to build the full charged-species formula.
    "[2M+H]+": element_to_position["H"],
    "[2M+Na]+": element_to_position["Na"],
    "[2M-H]-": -element_to_position["H"],
    "[2M+HCOOH-H]-": element_to_position["C"]
    + element_to_position["H"]
    + 2 * element_to_position["O"],
    "[2M+CH3COOH-H]-": 2 * element_to_position["C"]
    + 3 * element_to_position["H"]
    + 2 * element_to_position["O"],
}

assert set(ion_to_mass) == set(ION_LST) == set(ion_to_add_vec), (
    "ion_to_mass / ion_to_add_vec must cover exactly ION_LST"
)


def precursor_mz_to_neutral_mass(precursor_mz, ion):
    """Neutral *monomer* mass from an observed precursor m/z.

    Inverts ``precursor_mz = nmer * neutral_mass + ion_to_mass[ion]`` (charge is
    assumed 1 for every adduct in ION_LST). Accepts scalars or numpy arrays.
    """
    return (precursor_mz - ion_to_mass[ion]) / ion_to_nmer[ion]


def neutral_mass_to_precursor_mz(neutral_mass, ion):
    """Expected precursor m/z for a neutral *monomer* mass under ``ion``."""
    return ion_to_nmer[ion] * neutral_mass + ion_to_mass[ion]


instrument_to_type = {
    "Thermo Finnigan Velos Orbitrap": "orbitrap",
    "Thermo Finnigan Elite Orbitrap": "orbitrap",
    "Orbitrap Fusion Lumos": "orbitrap",
    "Q-ToF (LCMS)": "qtof",
    "Unknown (LCMS)": "unknown",
    "Ion Trap (LCMS)": "iontrap",
    "ion trap": "iontrap",
    "FTICR (LCMS)": "fticr",
    "Bruker Q-ToF (LCMS)": "qtof",
    "Orbitrap (LCMS)": "orbitrap",
    "ESI-Orbitrap": "orbitrap",
    "ESI-qTOF": "qtof",
    "ESI-qToF": "qtof",
    "ESI-qTof": "qtof",
}

instruments = sorted(set(instrument_to_type.values()))
max_instr_idx = len(instruments) + 1
instrument_to_idx = dict(zip(instruments, np.arange(len(instruments)), strict=False))
instrument_to_tol = {
    "qtof": 10,
    "orbitrap": 5,
    "iontrap": 15,
    "fticr": 5,
    "unknown": 15,
}

rdbe_mult = np.zeros_like(ELEMENT_VECTORS[0])
els = ["C", "N", "P", "H", "Cl", "Br", "I", "F"]
weights = [2, 1, 1, -1, -1, -1, -1, -1]
for k, v in zip(els, weights, strict=False):
    rdbe_mult[element_to_ind[k]] = v


def get_ion_idx(ionization: str) -> int:
    return ion_to_idx[ionization]


def get_instr_idx(instrument: str) -> int:
    inst = instrument_to_type.get(instrument, "unknown")
    return instrument_to_idx.get(inst, len(instrument_to_idx))


def get_instr_tol(instrument: str) -> int:
    inst = instrument_to_type.get(instrument, "unknown")
    return instrument_to_tol[inst]


def cross_sum(x, y):
    return (np.expand_dims(x, 0) + np.expand_dims(y, 1)).reshape(-1, y.shape[-1])


def get_all_subsets_dense(dense_formula, element_vectors):
    non_zero = np.argwhere(dense_formula > 0).flatten()
    vectorized_formula = []
    for nonzero_ind in non_zero:
        temp = element_vectors[nonzero_ind] * np.arange(
            0, dense_formula[nonzero_ind] + 1
        ).reshape(-1, 1)
        vectorized_formula.append(temp)

    zero_vec = np.zeros((1, element_vectors.shape[-1]))
    cross_prod = reduce(cross_sum, vectorized_formula, zero_vec)

    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]
    all_masses = cross_prod.dot(VALID_MONO_MASSES)
    return cross_prod, all_masses


def get_all_subsets(chem_formula):
    dense_formula = formula_to_dense(chem_formula)
    return get_all_subsets_dense(dense_formula, element_vectors=ELEMENT_VECTORS)


def rdbe_filter(cross_prod):
    rdbe_total = 1 + 0.5 * cross_prod.dot(rdbe_mult)
    filter_inds = np.argwhere(rdbe_total >= 0).flatten()
    return filter_inds


# ---------------------------------------------------------------------------
# Meet-in-the-middle (mitm) subformula assignment (fast path for large cands).
# ---------------------------------------------------------------------------

def _mitm_split(nonzero_inds, counts):
    """Greedily partition present elements into two halves balancing the
    product of (count+1) (i.e. each half's cross-product row count)."""
    order = sorted(nonzero_inds, key=lambda i: -(int(counts[i]) + 1))
    a, b = [], []
    prod_a, prod_b = 1, 1
    for i in order:
        if prod_a <= prod_b:
            a.append(i)
            prod_a *= int(counts[i]) + 1
        else:
            b.append(i)
            prod_b *= int(counts[i]) + 1
    return a, b


def _enumerate_half(inds, counts):
    """Enumerate every subset of the given element indices as dense vectors,
    returning (dense (n x CHEM_ELEMENT_NUM), neutral masses, rdbe contributions).
    An empty `inds` yields the single all-zero subset."""
    zero_vec = np.zeros((1, CHEM_ELEMENT_NUM))
    vectorized = [
        ELEMENT_VECTORS[i] * np.arange(0, int(counts[i]) + 1).reshape(-1, 1)
        for i in inds
    ]
    cross = reduce(cross_sum, vectorized, zero_vec)
    masses = cross.dot(VALID_MONO_MASSES)
    rdbe_contrib = cross.dot(rdbe_mult)
    return cross, masses, rdbe_contrib


def assign_subforms_mitm(form, spec, ion_type, mass_diff_thresh=15, window_factor=2.0):
    """Meet-in-the-middle equivalent of `assign_subforms`.

    Produces the identical output_dict for every peak that survives
    `valid_mask` in the original. Equivalence argument:

    * The original keeps a peak iff clipped_ppm(min_mass_diff) < thresh, where
      min_mass_diff is the distance to the globally closest rdbe-valid subset.
      Hence a retained peak's reported subformula lies within `thresh` ppm.
    * We search a per-peak window of `window_factor * thresh` ppm (>= thresh),
      so any subformula that could be retained is inside the window, and the
      closest one inside the window equals the global closest whenever the
      global closest is within the window (i.e. whenever the peak is retained).
    * rdbe filtering is additive across the two halves
      (1 + 0.5*(rdbe_A + rdbe_B) >= 0), reproducing the original's per-subset
      rdbe_filter exactly.
    * The original's post-filter dedup loop is a no-op (its formula_idx_dict is
      never populated), so retained peaks pass through unchanged and in order.
    """
    dense_formula = formula_to_dense(form)
    counts = dense_formula.astype(np.int64)
    nonzero = np.nonzero(counts)[0]

    a_inds, b_inds = _mitm_split(nonzero, counts)
    A_dense, A_mass, A_rdbe = _enumerate_half(a_inds, counts)
    B_dense, B_mass, B_rdbe = _enumerate_half(b_inds, counts)

    order = np.argsort(B_mass, kind="stable")
    B_dense, B_mass, B_rdbe = B_dense[order], B_mass[order], B_rdbe[order]

    spec_masses, spec_intens = spec[:, 0], spec[:, 1]
    ion_mass = ion_to_mass[ion_type]
    n_peaks = len(spec_masses)

    best_diff = np.full(n_peaks, np.inf)
    best_dense = np.zeros((n_peaks, CHEM_ELEMENT_NUM))

    clip_mass = np.where(spec_masses < 200, 200.0, spec_masses)
    window = window_factor * mass_diff_thresh * clip_mass / 1e6 + 1e-6

    for p in range(n_peaks):
        target_neutral = spec_masses[p] - ion_mass
        need = target_neutral - A_mass  # required B-half mass for each A-half subset
        w = window[p]
        lo = np.searchsorted(B_mass, need - w, side="left")
        hi = np.searchsorted(B_mass, need + w, side="right")
        pair_counts = hi - lo
        nz = np.nonzero(pair_counts)[0]
        if nz.size == 0:
            continue
        a_rep = np.repeat(nz, pair_counts[nz])
        b_idx = np.concatenate([np.arange(lo[a], hi[a]) for a in nz])

        rdbe_total = 1 + 0.5 * (A_rdbe[a_rep] + B_rdbe[b_idx])
        keep = rdbe_total >= 0
        if not keep.any():
            continue
        a_rep, b_idx = a_rep[keep], b_idx[keep]

        comb_mass = A_mass[a_rep] + B_mass[b_idx]
        diff = np.abs(comb_mass - target_neutral)
        k = int(np.argmin(diff))
        best_diff[p] = diff[k]
        best_dense[p] = A_dense[a_rep[k]] + B_dense[b_idx[k]]

    rel_mass_diff = clipped_ppm(best_diff, spec_masses)
    valid_mask = rel_mass_diff < mass_diff_thresh

    spec_masses = spec_masses[valid_mask]
    spec_intens = spec_intens[valid_mask]
    dense_valid = best_dense[valid_mask]

    mono_mass = dense_valid.dot(VALID_MONO_MASSES) + ion_mass
    abs_mass_diff = np.abs(spec_masses - mono_mass)
    rel_mass_diff = clipped_ppm(abs_mass_diff, spec_masses)
    formulas = np.array([vec_to_formula(d) for d in dense_valid])
    ion_types = np.array([ion_type] * len(formulas))

    if spec_intens.size == 0:
        output_tbl = None
    else:
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "mono_mass": list(mono_mass),
            "abs_mass_diff": list(abs_mass_diff),
            "mass_diff": list(rel_mass_diff),
            "formula": list(formulas),
            "ions": list(ion_types),
        }
    return {"cand_form": form, "cand_ion": ion_type, "output_tbl": output_tbl}


def assign_subforms_fast(
    form, spec, ion_type, mass_diff_thresh=15, hybrid_threshold=2_000
):
    """Hybrid dispatch: original full-enumeration for small candidates (where
    its cross-product is cheap and MITM setup overhead isn't worth it), MITM for
    large ones. `hybrid_threshold` is on the estimated cross-product row count
    (product of (count_i + 1)); calibrated empirically on real spectraverse
    data -- old wins below ~1-3k rows, MITM wins decisively above that (>99.9%
    of candidates with >3k estimated rows favor MITM)."""
    dense_formula = formula_to_dense(form)
    nz = dense_formula[dense_formula > 0]
    est_rows = float(np.prod(nz + 1)) if nz.size else 1.0
    if est_rows <= hybrid_threshold:
        return assign_subforms(form, spec, ion_type, mass_diff_thresh)
    return assign_subforms_mitm(form, spec, ion_type, mass_diff_thresh)


def assign_subforms(form, spec, ion_type, mass_diff_thresh=15):
    cross_prod, masses = get_all_subsets(form)
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]

    ion_masses = ion_to_mass[ion_type]
    masses_with_ion = masses + ion_masses
    ion_types = np.array([ion_type] * len(masses_with_ion))

    mass_diffs = np.abs(spec_masses[:, None] - masses_with_ion[None, :])

    formula_inds = mass_diffs.argmin(-1)
    min_mass_diff = mass_diffs[np.arange(len(mass_diffs)), formula_inds]
    rel_mass_diff = clipped_ppm(min_mass_diff, spec_masses)

    valid_mask = rel_mass_diff < mass_diff_thresh
    spec_masses = spec_masses[valid_mask]
    spec_intens = spec_intens[valid_mask]
    min_mass_diff = min_mass_diff[valid_mask]
    rel_mass_diff = rel_mass_diff[valid_mask]
    formula_inds = formula_inds[valid_mask]

    formulas = np.array([vec_to_formula(j) for j in cross_prod[formula_inds]])
    formula_masses = masses_with_ion[formula_inds]
    ion_types = ion_types[formula_inds]

    formula_idx_dict = {}
    uniq_mask = []
    for idx, formula in enumerate(formulas):
        uniq_mask.append(formula not in formula_idx_dict)
        gather_ind = formula_idx_dict.get(formula)
        if gather_ind is None:
            continue
        spec_intens[gather_ind] += spec_intens[idx]
        formula_idx_dict[formula] = idx

    spec_masses = spec_masses[uniq_mask]
    spec_intens = spec_intens[uniq_mask]
    min_mass_diff = min_mass_diff[uniq_mask]
    rel_mass_diff = rel_mass_diff[uniq_mask]
    formula_masses = formula_masses[uniq_mask]
    formulas = formulas[uniq_mask]
    ion_types = ion_types[uniq_mask]

    if spec_intens.size == 0:
        output_tbl = None
    else:
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "mono_mass": list(formula_masses),
            "abs_mass_diff": list(min_mass_diff),
            "mass_diff": list(rel_mass_diff),
            "formula": list(formulas),
            "ions": list(ion_types),
        }
    output_dict = {
        "cand_form": form,
        "cand_ion": ion_type,
        "output_tbl": output_tbl,
    }
    return output_dict


def get_output_dict(spec_name, spec, form, mass_diff_type, mass_diff_thresh, ion_type):
    assert mass_diff_type == "ppm"
    output_dict = {"cand_form": form, "cand_ion": ion_type, "output_tbl": None}
    if spec is not None and ion_type in ION_LST:
        output_dict = assign_subforms_fast(
            form, spec, ion_type, mass_diff_thresh=mass_diff_thresh
        )
    return output_dict


def assign_single_spec(spec_name, export_dicts, output_dir):
    # Nested keying: {formula: {ion: {"cand_tbl": ...}}}
    # A single formula can legitimately pair with multiple ions (e.g. [M+H]+ and
    # [M+Na]+, or [M-H]- and [M+H]+ across modes). Flat keying by formula only
    # would silently collapse all but one.
    res_dict = {}
    for export_dict in export_dicts:
        output = get_output_dict(**export_dict)
        form = output["cand_form"]
        ion = output["cand_ion"]
        res_dict.setdefault(form, {})[ion] = {
            "cand_tbl": output["output_tbl"],
        }

    if output_dir is not None:
        from pathlib import Path

        with open(Path(output_dir) / f"{spec_name}.json", "w") as f:
            json.dump(res_dict, f, indent=4)
    return res_dict


def clipped_ppm(mass_diff, parentmass):
    parentmass_copy = parentmass * 1
    parentmass_copy[parentmass < 200] = 200
    ppm = mass_diff / parentmass_copy * 1e6
    return ppm


def clipped_ppm_single(cls_mass_diff, parentmass):
    div_factor = 200 if parentmass < 200 else parentmass
    cls_ppm = cls_mass_diff / div_factor * 1e6
    return cls_ppm


def clipped_ppm_single_norm(cls_mass_diff, parentmass):
    return norm_mass_diff_ppm(clipped_ppm_single(cls_mass_diff, parentmass))


def norm_mass_diff_ppm(mass_diff):
    return mass_diff / 10


def get_cls_mass_diff(parentmass, form, ion):
    # neutral_mass_to_precursor_mz folds in the n-mer factor and the adduct mass
    # shift (which already carries the electron-mass sign: −ELECTRON_MASS for
    # positive ions, +ELECTRON_MASS for negative), giving the theoretical
    # precursor m/z directly. No extra electron_correct (that would bias ~2 ppm).
    true_val = neutral_mass_to_precursor_mz(formula_mass(form), ion)
    return abs(parentmass - true_val)


def formula_to_dense(chem_formula):
    total_onehot = []
    for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)
    return dense_vec


def formula_mass(chem_formula):
    mass = 0
    for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass


def electron_correct(mass):
    return mass - ELECTRON_MASS


def formula_difference(formula_1, formula_2):
    form_1 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_1)
    }
    form_2 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_2)
    }
    for k, _v in form_2.items():
        form_1[k] = form_1[k] - form_2[k]
    out_formula = "".join([f"{k}{v}" for k, v in form_1.items() if v > 0])
    return out_formula


def vec_to_formula(form_vec):
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def standardize_form(i):
    return vec_to_formula(formula_to_dense(i))


def standardize_adduct(adduct, fail_silent=False):
    adduct = adduct.replace(" ", "")
    adduct = ion_remap.get(adduct, adduct)
    if fail_silent:
        return adduct
    elif adduct not in ION_LST:
        raise ValueError(f"Adduct {adduct} not in ION_LST")
    else:
        return adduct


def ion_mode_from_adduct(adduct: str) -> str:
    """Return "pos" or "neg" for a (canonical or aliased) adduct string."""
    canonical = standardize_adduct(adduct, fail_silent=False)
    return ion_to_mode[canonical]


def add_ion(form, ion):
    """Full charged-species formula for ``ion``: nmer * M + adduct atoms."""
    ion_vec = ion_to_add_vec[ion]
    form_vec = formula_to_dense(form)
    return vec_to_formula(ion_to_nmer[ion] * form_vec + ion_vec)


def form_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    return "" if mol is None else CalcMolFormula(mol)


def mass_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    return 0.0 if mol is None else ExactMolWt(mol)


def atoms_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    return 0 if mol is None else mol.GetNumAtoms()


def min_formal_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return int(np.array([a.GetFormalCharge() for a in mol.GetAtoms()]).min())


def max_formal_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return int(np.array([a.GetFormalCharge() for a in mol.GetAtoms()]).max())


def has_valid_els(chem_formula):
    for chem_symbol, _ in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        if chem_symbol and chem_symbol not in VALID_ELEMENTS:
            return False
    return True
