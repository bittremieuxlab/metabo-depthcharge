# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""00_prepare_msgym.py

Convert MassSpecGym.tsv into the MIST-CF data directory format:
  - labels.tsv (columns: spec, formula, ionization, instrument)
  - spec_files/*.ms (SIRIUS .ms format, one per spectrum)
  - Optionally a split file from MassSpecGym fold assignments

Each MassSpecGym row (one collision energy measurement) becomes
a separate .ms file, matching the dominant pattern in the original
MIST-CF training data (NPLIB1/canopus_train).
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Map MassSpecGym instrument names to MIST-CF instrument names
INSTRUMENT_MAP = {
    "Orbitrap": "Orbitrap (LCMS)",
    "QTOF": "Q-ToF (LCMS)",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msgym-tsv", required=True, help="Path to MassSpecGym.tsv")
    parser.add_argument("--out-dir", required=True, help="Output data directory")
    parser.add_argument(
        "--create-split",
        action="store_true",
        default=False,
        help="Create a split file from MassSpecGym fold column",
    )
    return parser.parse_args()


def write_ms_file(
    path, spec_name, parent_mass, precursor_mz, adduct, instrument, mzs_str, intens_str
):
    """Write a single .ms file with one >ms2 block."""
    with open(path, "w") as f:
        f.write(f">compound {spec_name}\n")
        f.write(f">parentmass {parent_mass}\n")
        f.write(f">precursor_mz {precursor_mz}\n")
        f.write(f">ionization {adduct}\n")
        f.write(f"#INSTRUMENT TYPE {instrument}\n")
        f.write("\n")
        f.write(">ms2\n")
        for mz, inten in zip(mzs_str.split(","), intens_str.split(","), strict=False):
            f.write(f"{mz} {inten}\n")
        f.write("\n")


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = out_dir / "spec_files"
    spec_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.msgym_tsv, sep="\t")
    print(f"Loaded {len(df)} rows from {args.msgym_tsv}")

    # Map instrument names
    df["instrument_mapped"] = df["instrument_type"].map(INSTRUMENT_MAP)
    unmapped = df["instrument_mapped"].isna()
    if unmapped.any():
        unknown_instruments = df.loc[unmapped, "instrument_type"].unique()
        print(
            f"Warning: unmapped instrument types {unknown_instruments}, defaulting to 'Unknown (LCMS)'"
        )
        df.loc[unmapped, "instrument_mapped"] = "Unknown (LCMS)"

    labels_rows = []
    split_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing .ms files"):
        spec_name = row["identifier"]
        formula = row["formula"]
        precursor_mz = row["precursor_mz"]
        parent_mass = row["parent_mass"]
        adduct = row["adduct"]
        instrument = row["instrument_mapped"]
        fold = row["fold"]

        write_ms_file(
            spec_dir / f"{spec_name}.ms",
            spec_name,
            parent_mass,
            precursor_mz,
            adduct,
            instrument,
            row["mzs"],
            row["intensities"],
        )

        labels_rows.append(
            {
                "spec": spec_name,
                "formula": formula,
                "ionization": adduct,
                "instrument": instrument,
            }
        )
        split_rows.append(
            {
                "spec": spec_name,
                "Fold_0": fold,
            }
        )

    labels_df = pd.DataFrame(labels_rows)
    labels_df.to_csv(out_dir / "labels.tsv", sep="\t", index=False)
    print(f"Wrote {len(labels_df)} entries to {out_dir / 'labels.tsv'}")

    if args.create_split:
        split_dir = out_dir / "splits"
        split_dir.mkdir(exist_ok=True)
        split_df = pd.DataFrame(split_rows)
        split_df = split_df.sort_values("spec").reset_index(drop=True)
        split_path = split_dir / "split_msgym.tsv"
        split_df.to_csv(split_path, sep="\t", index=False)
        print(f"Wrote split file to {split_path}")
        for fold_name in ["train", "val", "test"]:
            n = (split_df["Fold_0"] == fold_name).sum()
            print(f"  {fold_name}: {n}")


if __name__ == "__main__":
    main()
