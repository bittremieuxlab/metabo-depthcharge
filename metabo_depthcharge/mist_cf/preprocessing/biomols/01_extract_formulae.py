# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""01_extract_formulae.py

Read a SMILES or formula source (plain text, one per line, or a TSV with a
named column) and produce a deduplicated, filtered list of molecular formulas
suitable as positive examples for fast-filter training.

Filters applied:
- mass <= --max-mass (default 1500)
- formula uses only elements in common.VALID_ELEMENTS
- if going through SMILES: single fragment, formal charges in [-2, 2],
  >= 2 atoms, valid SMILES
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ... import common


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="Input file (TSV with header, or plain text one-per-line)",
    )
    p.add_argument(
        "--smiles-col",
        default=None,
        help="Name of SMILES column in TSV (mutually exclusive with --formula-col)",
    )
    p.add_argument(
        "--formula-col",
        default=None,
        help="Name of formula column in TSV (skips SMILES filtering)",
    )
    p.add_argument("--max-mass", type=float, default=1500.0)
    p.add_argument(
        "--out", required=True, help="Output text file: one formula per line"
    )
    p.add_argument("--num-workers", type=int, default=16)
    return p.parse_args()


def _filter_via_smiles(smiles, max_mass, num_workers):
    print(f"Filtering {len(smiles)} SMILES...")
    forms = common.chunked_parallel(smiles, common.form_from_smi, max_cpu=num_workers)
    masses = common.chunked_parallel(smiles, common.mass_from_smi, max_cpu=num_workers)
    n_atoms = common.chunked_parallel(
        smiles, common.atoms_from_smi, max_cpu=num_workers
    )
    fc_min = common.chunked_parallel(
        smiles, common.min_formal_from_smi, max_cpu=num_workers
    )
    fc_max = common.chunked_parallel(
        smiles, common.max_formal_from_smi, max_cpu=num_workers
    )

    forms = np.array(forms)
    masses = np.array(masses)
    n_atoms = np.array(n_atoms)
    fc_min = np.array(fc_min)
    fc_max = np.array(fc_max)

    is_valid = np.array([f != "" for f in forms])
    single_frag = np.array(["." not in s for s in smiles])
    valid_els = np.array([common.has_valid_els(f) if f else False for f in forms])

    mask = (
        is_valid
        & single_frag
        & valid_els
        & (n_atoms >= 2)
        & (masses <= max_mass)
        & (fc_min >= -2)
        & (fc_max <= 2)
    )
    return forms[mask].tolist()


def _filter_via_formula(formulae, max_mass):
    print(f"Filtering {len(formulae)} formulae...")
    out = []
    for f in formulae:
        if not f or not isinstance(f, str):
            continue
        if not common.has_valid_els(f):
            continue
        try:
            m = common.formula_mass(f)
        except Exception:
            continue
        if m > max_mass:
            continue
        out.append(f)
    return out


def main():
    args = get_args()
    if (args.smiles_col is None) == (args.formula_col is None):
        raise SystemExit("Provide exactly one of --smiles-col or --formula-col")

    inp = Path(args.input)
    if args.smiles_col is not None:
        df = pd.read_csv(inp, sep="\t")
        if args.smiles_col not in df.columns:
            raise SystemExit(
                f"Column '{args.smiles_col}' not in {inp}. Have: {list(df.columns)}"
            )
        smiles = df[args.smiles_col].astype(str).tolist()
        forms = _filter_via_smiles(smiles, args.max_mass, args.num_workers)
    else:
        df = pd.read_csv(inp, sep="\t")
        if args.formula_col not in df.columns:
            raise SystemExit(
                f"Column '{args.formula_col}' not in {inp}. Have: {list(df.columns)}"
            )
        formulae = df[args.formula_col].astype(str).tolist()
        forms = _filter_via_formula(formulae, args.max_mass)

    forms = sorted(set(forms))
    print(f"Kept {len(forms)} unique formulas (from {len(df)} input rows)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fp:
        fp.write("\n".join(forms))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
