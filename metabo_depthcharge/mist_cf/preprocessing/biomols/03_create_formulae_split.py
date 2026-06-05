# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""03_create_formulae_split.py

Build train/val/test split over the masses in the decoys TSV. Any mass that
appears in --exclude-labels (an evaluation labels file) is forced into 'test'
so the eval set is never seen during fast-filter training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ... import common, decomp


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--formula-decoy-file", required=True)
    p.add_argument(
        "--exclude-labels",
        default=None,
        help="Optional labels TSV with a 'formula' column. Any mass overlapping this file is pinned to test.",
    )
    p.add_argument("--exclude-formula-col", default="formula")
    p.add_argument("--out", required=True)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fold-name", default="biomol_split")
    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)

    df = pd.read_csv(args.formula_decoy_file, sep="\t").fillna("")
    df_masses = set(df["mass"].values)

    overlap = set()
    if args.exclude_labels is not None:
        excl = pd.read_csv(args.exclude_labels, sep="\t")
        if args.exclude_formula_col not in excl.columns:
            raise SystemExit(
                f"Column '{args.exclude_formula_col}' missing in {args.exclude_labels}"
            )
        excl_forms = set(excl[args.exclude_formula_col].dropna().astype(str).values)
        excl_masses = set(
            decomp.get_rounded_masses([common.formula_mass(f) for f in excl_forms])
        )
        overlap = df_masses & excl_masses
        print(
            f"Overlap with exclude set: {len(overlap)} / {len(df_masses)} masses pinned to test"
        )

    total = len(df_masses)
    test_n = int(total * (1 - args.train_frac - args.val_frac))
    val_n = int(total * args.val_frac)

    # Pin overlap to test, fill remainder of test slots randomly
    test_set = set(overlap)
    pool = df_masses - test_set
    extra_test = max(0, test_n - len(test_set))
    if extra_test > 0:
        chosen = np.random.choice(list(pool), min(extra_test, len(pool)), replace=False)
        test_set.update(chosen)
        pool = pool - set(chosen)

    val_set = set(np.random.choice(list(pool), min(val_n, len(pool)), replace=False))
    train_set = pool - val_set

    print(
        f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)} (total={total})"
    )

    rows = []
    for mass in tqdm(df["mass"].values, desc="assigning folds"):
        if mass in train_set:
            fold = "train"
        elif mass in val_set:
            fold = "val"
        elif mass in test_set:
            fold = "test"
        else:
            fold = "exclude"
        rows.append({"mass": mass, args.fold_name: fold})

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_index(axis=1, ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
