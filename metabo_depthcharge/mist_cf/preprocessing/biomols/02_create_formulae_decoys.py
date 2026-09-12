# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""02_create_formulae_decoys.py

For each unique mass in the input formula list, run SIRIUS decomp to enumerate
candidate formulas at that mass, then write a TSV with columns:
    mass, pos (comma-separated true formulas at that mass), neg (comma-separated decoys).

Decoys are formulas at the same mass that are NOT in the positive list, optionally
sub-sampled to --num-decoys per mass to bound output size.

Restartable: SIRIUS is run one batch at a time (decomp.iter_sirius_batches),
each batch's raw output cached to <out>.cache/batch_XXXX.tsv and its decoy
rows checkpointed to <out>.cache/rows_progress.tsv + done_batches.txt right
after that batch finishes. Re-running the same command skips both the SIRIUS
call and the row-building for every already-completed batch. This also
bounds peak memory to a single batch's decompositions instead of every
batch's at once (see decomp.run_sirius's docstring for why that OOM'd).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ... import common, decomp


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--formulae-list", required=True, help="Text file: one formula per line"
    )
    p.add_argument(
        "--num-decoys", type=int, default=256, help="Cap on decoys retained per mass"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--decomp-filter", type=str, default="RDBE")
    p.add_argument("--ppm", type=int, default=10)
    p.add_argument(
        "--elements",
        type=str,
        default=None,
        help="SIRIUS element alphabet. Defaults to decomp.sirius_decomp.EL_STR_DEFAULT.",
    )
    p.add_argument("--cores", type=int, default=16, help="Parallel SIRIUS workers")
    p.add_argument("--max-batch", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Dir for SIRIUS batch cache + progress checkpoint, for restart. "
        "Defaults to <out>.cache/",
    )
    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)

    formulae = [j.strip() for j in open(args.formulae_list) if j.strip()]
    if args.debug:
        formulae = formulae[:1024]
    print(f"Loaded {len(formulae)} formulas")

    formulae_masses = decomp.get_rounded_masses(
        [common.formula_mass(f) for f in formulae]
    )
    mass_to_pos = defaultdict(list)
    for mass, formula in zip(formulae_masses, formulae, strict=False):
        mass_to_pos[mass].append(formula)

    unique_masses = sorted(set(formulae_masses))
    print(f"Running SIRIUS decomp on {len(unique_masses)} unique masses...")

    sirius_kwargs = {
        "filter_": args.decomp_filter,
        "ppm": args.ppm,
        "loglevel": "NONE",
        "cores": args.cores,
        "max_batch": args.max_batch,
        "mass_sort": False,
    }
    if args.elements is not None:
        sirius_kwargs["el_str"] = args.elements

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(f"{args.out}.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "rows_progress.tsv"
    done_batches_path = cache_dir / "done_batches.txt"

    done_batches = set()
    if done_batches_path.exists():
        done_batches = {int(line) for line in open(done_batches_path) if line.strip()}

    rows = []
    if progress_path.exists() and done_batches:
        rows = pd.read_csv(progress_path, sep="\t", dtype={"pos": str, "neg": str}).fillna("").to_dict("records")
        print(f"Resuming: {len(rows)} rows already computed from {len(done_batches)} completed batch(es)")

    for batch_idx, batch_masses, mass_to_form_lists in decomp.iter_sirius_batches(
        unique_masses, cache_dir=cache_dir / "sirius_batches", **sirius_kwargs
    ):
        if batch_idx in done_batches:
            continue
        for mass in batch_masses:
            pos = mass_to_pos.get(mass, [])
            neg = list(set(mass_to_form_lists.get(mass, [])) - set(pos))
            if args.num_decoys is not None and len(neg) > args.num_decoys:
                neg = list(np.random.choice(neg, args.num_decoys, replace=False))
            rows.append({"mass": mass, "pos": ",".join(pos), "neg": ",".join(neg)})
        done_batches.add(batch_idx)
        pd.DataFrame(rows).to_csv(progress_path, sep="\t", index=False)
        with open(done_batches_path, "a") as f:
            f.write(f"{batch_idx}\n")

    df = pd.DataFrame(rows).sort_values(by="mass").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, sep="\t")
    print(f"Wrote {args.out} with {len(df)} rows")
    print(
        f"  median |pos| = {
            int(
                df['pos']
                .str.split(',')
                .apply(lambda x: len([i for i in x if i]))
                .median()
            )
        }, "
        f"median |neg| = {
            int(
                df['neg']
                .str.split(',')
                .apply(lambda x: len([i for i in x if i]))
                .median()
            )
        }"
    )


if __name__ == "__main__":
    main()
