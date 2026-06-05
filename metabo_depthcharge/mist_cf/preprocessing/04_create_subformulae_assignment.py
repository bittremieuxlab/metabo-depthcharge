# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""create_subformulae_assignment.py
Given spectra and candidates from a labels file, assign subformulae and save to JSON files.
"""

import argparse
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/canopus_train")
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--max-formulae", default=100, type=int)
    parser.add_argument(
        "--decoy-label",
        type=str,
        default="data/canopus_train/decoy_labels/decoy_label_RDBE.tsv",
    )
    parser.add_argument("--ionization", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--assign-test-only", action="store_true", default=False)
    parser.add_argument(
        "--split-file", type=str, default="data/canopus_train/splits/split_1.tsv"
    )
    parser.add_argument("--mass-diff-type", default="ppm", type=str)
    parser.add_argument("--mass-diff-thresh", action="store", default=0.01, type=float)
    parser.add_argument("--inten-thresh", action="store", default=0.0, type=float)
    parser.add_argument("--num-workers", action="store", default=20, type=int)
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for slicing Phase 3 (subformulae assignment)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for slicing Phase 3 (subformulae assignment)",
    )
    return parser.parse_args()


def single_spec_process(spec_name, data_dir, max_formulae, inten_thresh):
    spec = common.process_spec_file_unbinned(spec_name, data_dir=data_dir)
    return (
        spec[0],
        common.max_thresh_spec(
            spec[1], max_peaks=max_formulae, inten_thresh=inten_thresh
        ),
    )


def main():
    args = get_args()

    data_dir = Path(args.data_dir)
    label_path = Path(args.decoy_label)
    assign_test_only = args.assign_test_only

    subform_dir = data_dir / "subformulae"
    subform_dir.mkdir(exist_ok=True)
    labels_df = pd.read_csv(label_path, sep="\t")
    debug = args.debug
    max_formulae = args.max_formulae
    inten_thresh = args.inten_thresh
    ionization = args.ionization
    mass_diff_thresh = args.mass_diff_thresh
    split_file = args.split_file
    num_workers = args.num_workers
    out_name = args.out_name

    if debug:
        inds = np.random.choice(len(labels_df), 50)
        labels_df = labels_df.loc[inds]

    label_name = label_path.stem
    if out_name is None:
        out_name = f"formulae_spec_{label_name}"

    output_dir = subform_dir / out_name
    output_dir.mkdir(exist_ok=True)

    if assign_test_only:
        split_file_path = Path(split_file)
        split_df = pd.read_csv(split_file_path, sep="\t")
        test_spec_name_lst = set(
            split_df[split_df["Fold_0"] == "test"]["spec"].to_list()
        )
        labels_df = labels_df[labels_df["spec"].isin(test_spec_name_lst)]

    if ionization is not None:
        labels_df = labels_df[labels_df["ionization"] == ionization]

    spec_lst = labels_df["spec"].to_list()
    labels_df = labels_df.fillna("").reset_index()

    proc_spec_full = partial(
        single_spec_process,
        data_dir=data_dir,
        max_formulae=max_formulae,
        inten_thresh=inten_thresh,
    )

    if debug or num_workers == 0:
        input_specs = [proc_spec_full(i) for i in tqdm(spec_lst)]
    else:
        from pathos import multiprocessing as mp

        cpus = min(mp.cpu_count(), num_workers)
        pool = mp.Pool(processes=cpus)
        input_specs = list(
            tqdm(
                pool.imap(proc_spec_full, spec_lst),
                total=len(spec_lst),
                desc="Processing spectra",
            )
        )
        pool.close()
        pool.join()

    input_specs = dict(input_specs)

    spec_to_assigns = defaultdict(lambda: [])
    for _, spec_entry in labels_df.iterrows():
        spec_name = spec_entry["spec"]
        spec = input_specs[spec_name]
        true_form = spec_entry["formula"]
        true_ion = spec_entry["ionization"]
        instrument = spec_entry["instrument"]
        mass_diff_thresh = common.get_instr_tol(instrument)
        export_dicts = spec_to_assigns[spec_name]

        cand_forms = spec_entry["decoy_formulae"]
        cand_ions = spec_entry["decoy_ions"]
        if cand_forms == "":
            cand_forms = []
            cand_ions = []
        else:
            cand_forms = cand_forms.split(",")
            cand_ions = cand_ions.split(",")

        cand_forms.append(true_form)
        cand_ions.append(true_ion)
        for cand_ion, cand_form in zip(cand_ions, cand_forms, strict=False):
            export_dicts.append(
                {
                    "spec": spec,
                    "mass_diff_type": "ppm",
                    "spec_name": spec_name,
                    "mass_diff_thresh": mass_diff_thresh,
                    "form": cand_form,
                    "ion_type": cand_ion,
                }
            )
        print(f"There are {len(export_dicts)} spec-cand pairs for this spec file")

        if debug:
            spec_to_assigns[spec_name] = export_dicts[:2]

    parallel_list = [
        {"spec_name": k, "export_dicts": v, "output_dir": output_dir}
        for k, v in spec_to_assigns.items()
    ]

    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(parallel_list)
    parallel_list = parallel_list[start_idx:end_idx]

    def export_wrapper(x):
        return common.assign_single_spec(**x)

    print(
        f"Processing {len(parallel_list)} different spectra (slice [{start_idx}:{end_idx}])"
    )

    if num_workers == 0 or debug:
        [export_wrapper(i) for i in tqdm(parallel_list)]
    else:
        from pathos import multiprocessing as mp

        cpus = min(mp.cpu_count(), num_workers)
        pool = mp.Pool(processes=cpus)
        list(
            tqdm(
                pool.imap_unordered(export_wrapper, parallel_list),
                total=len(parallel_list),
                desc="Assigning subformulae",
            )
        )
        pool.close()
        pool.join()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
