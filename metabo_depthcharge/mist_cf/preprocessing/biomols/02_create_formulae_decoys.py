# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""02_create_formulae_decoys.py

For each unique mass in the input formula list, run SIRIUS decomp to enumerate
candidate formulas at that mass, then write a TSV with columns:
    mass, pos (comma-separated true formulas at that mass), neg (comma-separated decoys).

Decoys are formulas at the same mass that are NOT in the positive list, optionally
sub-sampled to --num-decoys per mass to bound output size.
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

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

    out_dict = decomp.run_sirius(unique_masses, **sirius_kwargs)

    rows = []
    for mass in tqdm(unique_masses, desc="building rows"):
        pos = mass_to_pos.get(mass, [])
        neg = list(set(out_dict.get(mass, [])) - set(pos))
        if args.num_decoys is not None and len(neg) > args.num_decoys:
            neg = list(np.random.choice(neg, args.num_decoys, replace=False))
        rows.append({"mass": mass, "pos": ",".join(pos), "neg": ",".join(neg)})

    df = pd.DataFrame(rows).sort_values(by="mass").reset_index(drop=True)
    from pathlib import Path

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
