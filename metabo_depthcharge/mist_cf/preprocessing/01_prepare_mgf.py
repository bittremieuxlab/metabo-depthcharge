# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""01_prepare_mgf.py

Convert an MGF file into the MIST-CF data directory format, mirroring
``01_prepare_msgym.py`` but reading spectra from MGF instead of MassSpecGym.tsv:

  - labels.tsv (columns: spec, formula, ionization, instrument, dataset)
  - spec_files/*.ms (SIRIUS .ms format, one per spectrum)
  - Optionally a split file (random, or read from a per-spectrum fold key)

Each ``BEGIN IONS`` block becomes one .ms file. The neutral *monomer*
``parentmass`` is derived from the precursor m/z and the adduct via
``common.precursor_mz_to_neutral_mass`` (n-mer aware, so ``[2M+...]`` precursors
map to the correct monomer mass).

A ground-truth molecular formula is **required** for every spectrum (this is a
retraining pipeline, not prediction): if any record is missing the formula key,
the script aborts. Use the ``mist_cf_score`` prediction entrypoint instead if
you only want to score unannotated spectra.

Both positive- and negative-mode adducts are supported (see ``ION_LST`` /
``ion_remap`` in ``common/chem_utils.py``).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import common

# Map common MGF instrument strings to MIST-CF instrument names. Anything not
# listed (and with no --instrument-override) falls back to --default-instrument.
INSTRUMENT_MAP = {
    "Orbitrap": "Orbitrap (LCMS)",
    "QTOF": "Q-ToF (LCMS)",
    "Q-TOF": "Q-ToF (LCMS)",
    "qtof": "Q-ToF (LCMS)",
    "orbitrap": "Orbitrap (LCMS)",
    "iontrap": "Ion Trap (LCMS)",
    "nan": "Unknown (LCMS)",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert an MGF into the MIST-CF data directory format."
    )
    parser.add_argument("--mgf-file", required=True, help="Path to input .mgf")
    parser.add_argument("--out-dir", required=True, help="Output data directory")
    parser.add_argument(
        "--dataset", default="custom", help="Value for the labels.tsv 'dataset' column"
    )
    parser.add_argument(
        "--max-num", type=int, default=None, help="Cap on number of spectra to read"
    )

    # Metadata key mapping (matched case-insensitively against MGF fields).
    parser.add_argument(
        "--id-key",
        default=None,
        help="MGF field for the spectrum id (e.g. FEATURE_ID, TITLE). "
        "Falls back to sequential 'mgf_000000' ids if unset/missing.",
    )
    parser.add_argument(
        "--formula-key",
        default="FORMULA",
        help="MGF field holding the ground-truth molecular formula (required).",
    )
    parser.add_argument(
        "--precursor-mz-key",
        default="PEPMASS",
        help="MGF field for precursor m/z (first token used; PEPMASS may carry intensity too).",
    )
    parser.add_argument(
        "--adduct-key",
        default="ADDUCT",
        help="MGF field for the adduct/ionization (normalized via common.ion_remap).",
    )
    parser.add_argument(
        "--adduct-override",
        default=None,
        help="Force this adduct for every spectrum (use when the MGF has no adduct field).",
    )
    parser.add_argument(
        "--instrument-key",
        default=None,
        help="MGF field for the instrument string (mapped via INSTRUMENT_MAP).",
    )
    parser.add_argument(
        "--instrument-override",
        default=None,
        help="Force this instrument string for every spectrum.",
    )
    parser.add_argument(
        "--default-instrument",
        default="Unknown (LCMS)",
        help="Instrument used when neither key nor override resolves.",
    )

    # Splitting.
    parser.add_argument(
        "--create-split",
        action="store_true",
        default=False,
        help="Write splits/split_mgf.tsv (random unless --fold-key is given).",
    )
    parser.add_argument(
        "--fold-key",
        default=None,
        help="MGF field giving a per-spectrum fold (train/val/test); overrides random split.",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def write_ms_file(
    path, spec_name, parent_mass, precursor_mz, adduct, instrument, mzs, intensities
):
    """Write a single .ms file with one >ms2 block.

    Mirrors the format written by 01_prepare_msgym.write_ms_file and read back by
    common.parse_spectra, but takes peak arrays rather than comma-joined strings.
    """
    with open(path, "w") as f:
        f.write(f">compound {spec_name}\n")
        f.write(f">parentmass {parent_mass}\n")
        f.write(f">precursor_mz {precursor_mz}\n")
        f.write(f">ionization {adduct}\n")
        f.write(f"#INSTRUMENT TYPE {instrument}\n")
        f.write("\n")
        f.write(">ms2\n")
        for mz, inten in zip(mzs, intensities, strict=False):
            f.write(f"{mz} {inten}\n")
        f.write("\n")


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    spec_dir = out_dir / "spec_files"

    parsed = common.parse_spectra_mgf(args.mgf_file, max_num=args.max_num)
    print(f"Parsed {len(parsed)} spectra from {args.mgf_file}")

    # Pass 1: resolve every spectrum and collect problems. Nothing is written
    # yet, so a single bad record can't leave a half-built dataset on disk.
    resolved = []
    seen_ids = {}
    missing_formula = []
    bad_adduct = []
    missing_mz = []

    for i, (meta, spectra) in enumerate(
        tqdm(parsed, total=len(parsed), desc="Resolving spectra")
    ):
        # Case-insensitive view of the MGF fields.
        meta_ci = {str(k).upper(): v for k, v in meta.items()}

        def field(key, _m=meta_ci):
            return _m.get(key.upper()) if key else None

        # --- spectrum id (deduplicated) ---
        spec_name = field(args.id_key) if args.id_key else None
        if not spec_name:
            spec_name = f"mgf_{i:06d}"
        if spec_name in seen_ids:
            seen_ids[spec_name] += 1
            spec_name = f"{spec_name}_{seen_ids[spec_name]}"
        else:
            seen_ids[spec_name] = 0

        # --- ground-truth formula (mandatory) ---
        formula = field(args.formula_key)
        if not formula:
            missing_formula.append(spec_name)
            continue

        # --- adduct / ionization ---
        adduct = args.adduct_override or field(args.adduct_key)
        if adduct:
            adduct = common.ion_remap.get(adduct, adduct)
        if adduct not in common.ION_LST:
            bad_adduct.append((spec_name, adduct))
            continue

        # --- precursor m/z (first token; PEPMASS may include intensity) ---
        mz_raw = field(args.precursor_mz_key)
        if not mz_raw:
            missing_mz.append(spec_name)
            continue
        precursor_mz = float(str(mz_raw).split()[0])

        # Neutral monomer mass from precursor m/z and adduct (n-mer aware).
        parent_mass = common.precursor_mz_to_neutral_mass(precursor_mz, adduct)

        # --- instrument ---
        if args.instrument_override:
            instrument = args.instrument_override
        else:
            raw_instr = field(args.instrument_key)
            instrument = INSTRUMENT_MAP.get(
                raw_instr, raw_instr or args.default_instrument
            )

        resolved.append(
            {
                "spec": spec_name,
                "formula": formula,
                "ionization": adduct,
                "instrument": instrument,
                "precursor_mz": precursor_mz,
                "parent_mass": parent_mass,
                "peaks": spectra[0][1],
                "fold": field(args.fold_key),
            }
        )

    # Fail loudly on any dropped records so a partial dataset can't slip through.
    problems = []
    if missing_formula:
        problems.append(
            f"{len(missing_formula)} spectra missing formula key '{args.formula_key}' "
            f"(e.g. {missing_formula[:5]})"
        )
    if bad_adduct:
        problems.append(
            f"{len(bad_adduct)} spectra with adduct not in ION_LST "
            f"(e.g. {bad_adduct[:5]}); valid: {common.ION_LST}"
        )
    if missing_mz:
        problems.append(
            f"{len(missing_mz)} spectra missing precursor m/z key "
            f"'{args.precursor_mz_key}' (e.g. {missing_mz[:5]})"
        )
    if problems:
        raise SystemExit("Aborting; fix the input MGF:\n- " + "\n- ".join(problems))

    # Pass 2: everything validated, now write .ms files and labels.tsv.
    spec_dir.mkdir(parents=True, exist_ok=True)
    labels_rows = []
    for rec in tqdm(resolved, desc="Writing .ms files"):
        peaks = rec["peaks"]
        write_ms_file(
            spec_dir / f"{rec['spec']}.ms",
            rec["spec"],
            rec["parent_mass"],
            rec["precursor_mz"],
            rec["ionization"],
            rec["instrument"],
            peaks[:, 0],
            peaks[:, 1],
        )
        labels_rows.append(
            {
                "spec": rec["spec"],
                "formula": rec["formula"],
                "ionization": rec["ionization"],
                "instrument": rec["instrument"],
                "dataset": args.dataset,
            }
        )

    labels_df = pd.DataFrame(labels_rows)
    labels_df.to_csv(out_dir / "labels.tsv", sep="\t", index=False)
    print(f"Wrote {len(labels_df)} entries to {out_dir / 'labels.tsv'}")

    if args.create_split:
        split_df = pd.DataFrame([{"spec": r["spec"], "fold": r["fold"]} for r in resolved])
        if args.fold_key:
            if split_df["fold"].isna().any():
                missing = split_df.loc[split_df["fold"].isna(), "spec"].tolist()
                raise SystemExit(
                    f"--fold-key '{args.fold_key}' missing for {len(missing)} spectra "
                    f"(e.g. {missing[:5]})"
                )
        else:
            rng = np.random.default_rng(args.seed)
            n = len(split_df)
            perm = rng.permutation(n)
            n_train = int(n * args.train_frac)
            n_val = int(n * args.val_frac)
            folds = np.empty(n, dtype=object)
            folds[perm[:n_train]] = "train"
            folds[perm[n_train : n_train + n_val]] = "val"
            folds[perm[n_train + n_val :]] = "test"
            split_df["fold"] = folds

        split_df = split_df.rename(columns={"fold": "Fold_0"})
        split_df = split_df.sort_values("spec").reset_index(drop=True)
        split_dir = out_dir / "splits"
        split_dir.mkdir(exist_ok=True)
        split_path = split_dir / "split_mgf.tsv"
        split_df.to_csv(split_path, sep="\t", index=False)
        print(f"Wrote split file to {split_path}")
        for fold_name in ["train", "val", "test"]:
            print(f"  {fold_name}: {(split_df['Fold_0'] == fold_name).sum()}")


if __name__ == "__main__":
    main()
