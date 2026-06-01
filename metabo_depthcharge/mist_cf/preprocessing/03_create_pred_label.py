# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""create_pred_label.py

Build the prediction-label TSV consumed by PredDataset from step 02's
pred_candidates file.

Unlike the legacy MIST-CF script, this does NOT unconditionally append the
true (formula, ion) pair to each spectrum's candidate list. The input
pred_candidates file already reflects the honest deployment pipeline: the
true pair appears iff it survives SIRIUS decomp and the sampler. Specs for
which the pipeline dropped the true pair are scored without it — the
realistic failure mode.
"""

import argparse
import time
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["spec", "cand_form", "cand_ion", "parentmass", "instrument"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-candidates",
        required=True,
        help="Path to pred_candidates_*.tsv produced by step 02.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional split TSV. If given and it has a 'spec' column, only "
        "rows for specs in the split are kept. Also used to name the "
        "output file when --out-name is not set.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory. Defaults to <data-dir>/pred_labels.",
    )
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--data-dir", default="data/canopus_train")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = get_args()
    pred_cand_file = Path(args.pred_candidates)

    if args.save_dir is None:
        save_dir = Path(args.data_dir) / "pred_labels"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    if args.out_name is None:
        split_stem = Path(args.split_file).stem if args.split_file else "all"
        out_name = f"pred_{split_stem}_{pred_cand_file.stem}.tsv"
    else:
        out_name = args.out_name
    save_path = save_dir / out_name

    df = pd.read_csv(pred_cand_file, sep="\t")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"pred_candidates file {pred_cand_file} is missing columns: {missing}. "
            f"Re-run 02_create_decoy_label.py to regenerate it."
        )
    df = df[REQUIRED_COLS]

    n_all_specs = df["spec"].nunique()

    # Optional split filtering. The split file is expected to be a TSV with a
    # 'spec' column listing the specs in this split. If no such column is
    # present we fall back to just using the filename for output naming.
    if args.split_file is not None:
        split_path = Path(args.split_file)
        if split_path.exists():
            split_df = pd.read_csv(split_path, sep="\t")
            if "spec" in split_df.columns:
                keep = set(split_df["spec"].astype(str))
                before = len(df)
                df = df[df["spec"].astype(str).isin(keep)]
                print(
                    f"Split filter: kept {df['spec'].nunique()}/{n_all_specs} specs "
                    f"({len(df)}/{before} rows)"
                )
            else:
                print(
                    f"Note: split file {split_path} has no 'spec' column; "
                    "not filtering, using the filename only for naming."
                )
        else:
            print(f"Warning: split file {split_path} not found; not filtering.")

    if args.debug:
        df = df.head(10000)

    print(f"{len(df)} candidate rows across {df['spec'].nunique()} specs")
    print(f"Save to: {save_path}")
    df.to_csv(save_path, sep="\t", index=None)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
