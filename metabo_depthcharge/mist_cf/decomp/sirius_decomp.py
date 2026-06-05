# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""sirius_decomp.py

Wrapper calls around SIRIUS to extract formula decompositions.
Requires SIRIUS_PATH environment variable to be set.
"""

import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import common


SIRIUS_LOC = Path(os.getenv("SIRIUS_PATH", "sirius"))
ROUND_FACTOR = 4
EL_STR_DEFAULT = "C[0-]N[0-]O[0-]H[0-]S[0-3]P[0-1]I[0-1]Cl[0-3]F[0-6]Br[0-1]"


def run_sirius(
    masses,
    adduct=None,
    verbose=False,
    mass_sort=True,
    filter_="NONE",
    ppm=15,
    max_batch=10000,
    el_str=EL_STR_DEFAULT,
    cores=16,
    loglevel="WARNING",
):
    # Serial loop, SIRIUS handles parallelism internally via --cores. Earlier
    # ThreadPoolExecutor variant raced on the shared ~/.sirius-6.x/ refresh
    # token (auth0 rotates it on each refresh, invalidating concurrent users)
    # and produced "Please Login" errors mid-run. One process at a time, with
    # all the cores given to SIRIUS, is both simpler and race-free.
    if loglevel == "NONE":
        loglevel = "WARNING"

    if cores == 0:
        cores = 1

    masses = [np.round(i, ROUND_FACTOR) for i in masses]

    # Dedup; heaviest-first so the longest batches start earliest.
    unique_masses = np.unique(masses)
    unique_masses = np.sort(unique_masses)[::-1].astype(str)

    num_sections = max(1, math.ceil(unique_masses.shape[0] / max_batch))
    mass_splits = np.array_split(unique_masses, num_sections)

    out_dfs = []
    print(
        f"  Running {num_sections} SIRIUS batches serially with --cores {cores} "
        f"({unique_masses.shape[0]} unique masses, heaviest-first)...",
        flush=True,
    )
    progress_step = max(1, num_sections // 20)
    for batch_idx, mass_split in enumerate(mass_splits):
        with tempfile.NamedTemporaryFile() as temp_file:
            file_name = temp_file.name
            # shell=False with list argv: each mass is its own arg, so we hit
            # the ~2 MB total-argv limit, not the 128 KB per-arg limit that
            # MAX_ARG_STRLEN imposes on a single shell command string.
            cmd = [
                str(SIRIUS_LOC),
                "--cores",
                str(cores),
                "--log",
                loglevel,
                "decomp",
                "--mass",
                *mass_split.tolist(),
                "--output",
                file_name,
                "--elements",
                el_str,
                "--ppm",
                str(ppm),
            ]
            if adduct is not None:
                cmd.extend(["--ion", adduct])
            if filter_ is not None:
                cmd.extend(["--filter", filter_])

            if verbose:
                print(f"Running sirius command:\n {' '.join(cmd[:8])} ...")

            result = subprocess.run(cmd, capture_output=True, text=True)
            try:
                out_dfs.append(pd.read_csv(file_name, sep="\t"))
            except (pd.errors.EmptyDataError, FileNotFoundError) as e:
                tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
                print(
                    f"WARNING: SIRIUS batch {batch_idx + 1} produced no readable output ({e}). "
                    f"Exit={result.returncode}. Last stderr:\n    "
                    + "\n    ".join(tail)
                )

        completed = batch_idx + 1
        if completed % progress_step == 0 or completed == num_sections:
            print(f"    {completed}/{num_sections} batches done", flush=True)

    if not out_dfs:
        print("WARNING: All SIRIUS batches failed, returning empty results")
        return {np.round(m, ROUND_FACTOR): [] for m in masses}

    df = pd.concat(out_dfs).reset_index(drop=True)
    mass_to_forms = dict(df[["m/z", "decompositions"]].values)

    mass_to_form_lists = {}
    for i, j in mass_to_forms.items():
        cands = [] if not isinstance(j, str) else j.strip().split(",")
        if mass_sort:
            cands_masses = np.array([common.formula_mass(cand) for cand in cands])
            new_inds = np.argsort(np.abs(cands_masses - i))
            cands = np.array(cands)[new_inds].tolist()
        mass_to_form_lists[i] = cands

    return mass_to_form_lists


def get_rounded_masses(masses):
    return [np.round(i, ROUND_FACTOR) for i in masses]
