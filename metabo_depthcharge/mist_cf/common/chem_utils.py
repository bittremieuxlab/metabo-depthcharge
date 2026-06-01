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

POS_ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]

NEG_ION_LST = [
    "[M-H]-",
    "[M+Cl]-",
    "[M+HCOOH-H]-",  # formate adduct: net +CHO2
    "[M+CH3COOH-H]-",  # acetate adduct: net +C2H3O2
]

# Positive indices come first so any previously-trained checkpoint keeps its
# positive-ion embedding indices stable. New negative indices append at the end.
ION_LST = POS_ION_LST + NEG_ION_LST

ion_mode_to_ions = {"pos": POS_ION_LST, "neg": NEG_ION_LST}
ion_to_mode = dict.fromkeys(POS_ION_LST, "pos")
ion_to_mode.update(dict.fromkeys(NEG_ION_LST, "neg"))

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
        # negative-mode aliases
        "M-H": "[M-H]-",
        "[M-H]": "[M-H]-",
        "M+Cl": "[M+Cl]-",
        "[M+FA-H]-": "[M+HCOOH-H]-",
        "M+HCOO-": "[M+HCOOH-H]-",
        "[M+HCOO]-": "[M+HCOOH-H]-",
        "[M+AcOH-H]-": "[M+CH3COOH-H]-",
        "[M+OAc]-": "[M+CH3COOH-H]-",
        "[M+CH3COO]-": "[M+CH3COOH-H]-",
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
}

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
        output_dict = assign_subforms(
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
    # `ion_to_mass[ion]` already incorporates the electron mass with the correct
    # sign (−ELECTRON_MASS for positive ions, +ELECTRON_MASS for negative), so
    # the result is directly the theoretical m/z of the charged species. No
    # additional electron_correct is needed; applying it would bias by ~2 ppm.
    true_val = formula_mass(form) + ion_to_mass[ion]
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
    ion_vec = ion_to_add_vec[ion]
    form_vec = formula_to_dense(form)
    return vec_to_formula(form_vec + ion_vec)


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
