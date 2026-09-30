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
from tqdm import tqdm

from .. import common


SIRIUS_LOC = Path(os.getenv("SIRIUS_PATH", "sirius"))
ROUND_FACTOR = 4
EL_STR_DEFAULT = "C[0-]N[0-]O[0-]H[0-]S[0-3]P[0-1]I[0-1]Cl[0-3]F[0-6]Br[0-1]"


def iter_sirius_batches(
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
    desc="SIRIUS decomp",
    cache_dir=None,
):
    """Yield (batch_idx, mass_to_form_lists) one SIRIUS batch at a time.

    Unlike run_sirius, this never holds more than one batch's decompositions
    in memory at once, so peak memory is bounded by max_batch rather than by
    the total number of unique masses. If cache_dir is given, each batch's
    raw SIRIUS output is cached to <cache_dir>/batch_XXXX.tsv (atomically, via
    a .tmp + rename) and reused on a re-run instead of re-invoking SIRIUS -
    this is what makes a restart skip already-completed (and possibly
    multi-hour) batches instead of redoing them.
    """
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

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"  Running {num_sections} SIRIUS batches serially with --cores {cores} "
        f"({unique_masses.shape[0]} unique masses, heaviest-first)..."
    )
    for batch_idx, mass_split in enumerate(tqdm(mass_splits, desc=desc, unit="batch")):
        cache_path = cache_dir / f"batch_{batch_idx:04d}.tsv" if cache_dir is not None else None

        if cache_path is not None and cache_path.exists():
            try:
                df = pd.read_csv(cache_path, sep="\t")
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(columns=["m/z", "decompositions"])
        else:
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
                    df = pd.read_csv(file_name, sep="\t")
                except (pd.errors.EmptyDataError, FileNotFoundError) as e:
                    tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
                    tqdm.write(
                        f"WARNING: SIRIUS batch {batch_idx + 1}/{num_sections} produced no readable "
                        f"output ({e}). Exit={result.returncode}. Last stderr:\n    "
                        + "\n    ".join(tail)
                    )
                    df = pd.DataFrame(columns=["m/z", "decompositions"])

            if cache_path is not None:
                tmp_path = cache_path.with_suffix(".tmp")
                df.to_csv(tmp_path, sep="\t", index=False)
                tmp_path.rename(cache_path)

        mass_to_forms = dict(df[["m/z", "decompositions"]].values) if len(df) else {}
        mass_to_form_lists = {}
        for i, j in mass_to_forms.items():
            cands = [] if not isinstance(j, str) else j.strip().split(",")
            if mass_sort:
                cands_masses = np.array([common.formula_mass(cand) for cand in cands])
                new_inds = np.argsort(np.abs(cands_masses - i))
                cands = np.array(cands)[new_inds].tolist()
            mass_to_form_lists[i] = cands

        yield batch_idx, [float(m) for m in mass_split], mass_to_form_lists


def run_sirius(masses, **kwargs):
    """Back-compat wrapper: merges every batch into one dict, all in memory.

    Prefer iter_sirius_batches for large mass lists - this holds every
    batch's decompositions in memory simultaneously and has no restart
    support, which is what OOM'd on ~200k masses with a permissive element
    alphabet (each batch's comma-separated candidate-formula strings get
    parsed out into full-size Python dicts/lists, and all of them are kept
    alive at once here).
    """
    out = {}
    for _, _, batch_dict in iter_sirius_batches(masses, **kwargs):
        out.update(batch_dict)
    return out


def get_rounded_masses(masses):
    return [np.round(i, ROUND_FACTOR) for i in masses]
