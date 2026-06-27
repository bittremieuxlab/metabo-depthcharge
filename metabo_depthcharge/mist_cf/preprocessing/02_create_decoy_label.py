# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""create_decoy_label.py
Take the original labels file and generate decoy file using SIRIUS decomp.
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .. import common, decomp
from ..fast_form_score import fast_form_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-file", default="data/canopus_train/labels.tsv")
    parser.add_argument("--max-decoy", type=int, default=int(1e10))
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--decomp-filter", type=str, default="RDBE")
    parser.add_argument("--data-dir", type=str, default="data/canopus_train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-strat", action="store", default="uniform")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--resample-precursor-mz", action="store_true", default=False)
    parser.add_argument("--decoy-suffix", action="store", default=None)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=20_000,
        help="Masses per SIRIUS subprocess call. Larger = fewer JVM startups.",
    )
    parser.add_argument(
        "--fast-model", type=str, default=None, help="Path to fast filter checkpoint"
    )
    parser.add_argument(
        "--elements",
        type=str,
        default=None,
        help="SIRIUS element alphabet string (e.g. 'C[0-]N[0-]O[0-]H[0-]S[0-4]'). "
        "Defaults to decomp.sirius_decomp.EL_STR_DEFAULT if omitted.",
    )
    parser.add_argument(
        "--max-pred-candidates",
        type=int,
        default=None,
        help="Cap on candidates per spec for the honest pred-candidates file. "
        "Defaults to --max-decoy. Pred sampling runs on the FULL SIRIUS set "
        "(true form NOT excluded); true survives iff the deployment pipeline "
        "would have kept it.",
    )
    parser.add_argument(
        "--adducts",
        type=str,
        default=None,
        help="Comma-separated adducts to consider as candidates, restricting the "
        "candidate universe (e.g. '[M+H]+,[M+Na]+' for a MassSpecGym model that "
        "only saw those). Each must be in common.ION_LST (aliases are normalized). "
        "Default: every adduct of each spectrum's ion mode.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Run the fast-filter model on CUDA instead of CPU. Only affects the "
        "fast_filter sample strategy; no effect without --fast-model.",
    )
    return parser.parse_args()


def sample_decoys(
    spec,
    decoy_ion_lst,
    decoy_ions,
    parentmass,
    max_decoy,
    sample_strat,
    temperature,
    fast_model=None,
    device=None,
    precomputed_scores=None,
):
    decoy_ion_lst = np.array(decoy_ion_lst)
    adduct_masses = np.array([common.ion_to_mass[i] for i in decoy_ions])
    nmers = np.array([common.ion_to_nmer[i] for i in decoy_ions])
    form_masses = np.array([common.formula_mass(i) for i in decoy_ion_lst])
    # Predicted precursor m/z of each candidate (nmer * M + adduct shift).
    decoy_masses = nmers * form_masses + adduct_masses
    decoy_ppm = (
        common.clipped_ppm(
            np.abs(parentmass - decoy_masses), np.ones_like(decoy_masses) * parentmass
        )
        + 1e-20
    )
    sample_num = min(max_decoy, len(decoy_ion_lst))
    sample_ind_choices = np.arange(len(decoy_ion_lst))

    if sample_strat == "uniform":
        sampled_decoys = np.random.choice(sample_ind_choices, sample_num, replace=False)
    elif sample_strat == "normalized_inverse":
        weights = np.reciprocal(decoy_ppm)
        sampled_decoys = np.random.choice(
            sample_ind_choices,
            sample_num,
            replace=False,
            p=weights / (np.sum(weights)),
        )
    elif sample_strat == "softmax_inverse":
        weights = np.reciprocal(decoy_ppm)
        if np.sum(weights) == 0:
            sampled_decoys = []
        else:
            weights = weights / (np.sum(weights))
            weights = np.exp(weights / temperature)
            sampled_decoys = np.random.choice(
                sample_ind_choices,
                sample_num,
                replace=False,
                p=weights / (np.sum(weights)),
            )
    elif sample_strat == "sorted":
        sorted_idx = np.argsort(decoy_ppm)
        sampled_decoys = sample_ind_choices[sorted_idx][:sample_num]
    elif sample_strat == "fast_filter":
        if precomputed_scores is not None:
            # Reuse NN scores already computed on a superset; argsort is cheap.
            # Higher score = more plausible, keep top-k (matches fast_filter_sampling).
            sorted_idx = np.argsort(precomputed_scores)[::-1]
            sampled_decoys = sorted_idx[:sample_num]
        else:
            sampled_decoys = fast_model.fast_filter_sampling(
                spec, decoy_ion_lst, decoy_ions, max_decoy, device, batch_size=1024
            )
    else:
        raise ValueError(f"Weighting method {sample_strat} is not defined")

    np.random.shuffle(sampled_decoys)
    return sampled_decoys


def calculate_resampling_std(true_mass, error=15):
    return true_mass * error / (5 * 1e6)


def resample_precursor_fn(true_masses, errors):
    within_error_thresh = np.zeros_like(true_masses).astype(bool)
    resampling_std = calculate_resampling_std(true_masses, error=errors)
    resample_masses = true_masses * 1
    while not np.all(within_error_thresh):
        new_masses = np.random.normal(loc=true_masses, scale=resampling_std)
        new_masses = np.round(new_masses, 4)
        resample_masses[~within_error_thresh] = new_masses[~within_error_thresh]
        abs_mass_diff = np.abs(resample_masses - true_masses)
        rel_mass_diff = common.clipped_ppm(abs_mass_diff, true_masses)
        within_error_thresh = rel_mass_diff <= errors
    return new_masses


def _spec_seed(spec, seed, salt=0):
    # Deterministic per-spectrum RNG seed so sampling is reproducible and
    # independent of processing order (required once the per-spec loop is
    # parallelized — workers no longer share one sequential global RNG). salt=0
    # reproduces the original pred-path seed exactly, so the pred-candidate file
    # stays byte-identical.
    h = int.from_bytes(hashlib.md5(str(spec).encode("utf-8")).digest()[:4], "big")
    return (seed ^ h ^ salt) & 0x7FFFFFFF


def filter_spec_entries(
    spec,
    ion_masses,
    *,
    mass_to_formulas,
    formula_to_score,
    spec2form,
    spec2parentmass,
    max_decoy,
    max_pred_candidates,
    sample_strat,
    softmax_temperature,
    seed,
    fast_model=None,
    device=None,
):
    # Pair each formula only with the ion that actually produced its mass
    # for THIS spec — no cross-spec ion leakage.
    dict_entry = set()
    for ion, mass in ion_masses:
        forms = mass_to_formulas.get(mass, [])
        dict_entry.update((ion, f) for f in forms)
    if len(dict_entry) > 0:
        ions_all, cands_all = zip(*dict_entry, strict=False)
        ions_all = np.array(ions_all)
        cands_all = np.array(cands_all)
    else:
        ions_all = np.array([], dtype=object)
        cands_all = np.array([], dtype=object)

    # Reuse the globally-precomputed per-formula scores (the fast filter is a
    # pure function of cand_form, not spec/ion) instead of re-running the NN.
    scores_all = None
    if formula_to_score is not None and len(cands_all) > 0:
        scores_all = np.array(
            [formula_to_score[f] for f in cands_all], dtype=np.float32
        )

    # --- Training decoy path (true form excluded before sampling). ---
    inds = cands_all != spec2form[spec]
    cands = cands_all[inds]
    ions = ions_all[inds]
    was_found = np.sum(~inds) > 0

    if len(cands) < max_decoy:
        pass
    else:
        # Per-spec deterministic RNG -> order-independent under parallelism.
        # Only reached when sampling actually triggers; with the default
        # max_decoy this branch never runs and the training output is unchanged.
        np.random.seed(_spec_seed(spec, seed, salt=0x5EED))
        cand_inds = sample_decoys(
            spec,
            cands,
            ions,
            spec2parentmass[spec],
            max_decoy,
            sample_strat,
            softmax_temperature,
            fast_model=fast_model,
            device=device,
            precomputed_scores=(scores_all[inds] if scores_all is not None else None),
        )
        ions = ions[cand_inds]
        cands = cands[cand_inds]

    # --- Honest pred-candidate path (true form NOT excluded). ---
    # Sample on the full SIRIUS union; true survives only if the real deployment
    # pipeline would have kept it.
    if len(cands_all) <= max_pred_candidates:
        pred_ions = ions_all
        pred_cands = cands_all
    else:
        np.random.seed(_spec_seed(spec, seed, salt=0))
        pred_inds = sample_decoys(
            spec,
            cands_all,
            ions_all,
            spec2parentmass[spec],
            max_pred_candidates,
            sample_strat,
            softmax_temperature,
            fast_model=fast_model,
            device=device,
            precomputed_scores=scores_all,
        )
        pred_ions = ions_all[pred_inds]
        pred_cands = cands_all[pred_inds]

    return {
        "spec": spec,
        "was_found": was_found,
        "out_ions": ions,
        "out_cands": cands,
        "pred_ions": pred_ions,
        "pred_cands": pred_cands,
    }


# Read-only context shared with worker processes via copy-on-write fork
# inheritance (Linux). Populated in main() BEFORE the pool is created so the
# large tables (mass_to_formulas, formula_to_score) are never pickled to workers.
_WORKER_CTX = {}


def _filter_spec_entries_worker(item):
    spec, ion_masses = item
    return filter_spec_entries(spec, ion_masses, **_WORKER_CTX)


def main():
    args = get_args()
    labels_file = Path(args.label_file)
    debug = args.debug
    decomp_filter = args.decomp_filter
    max_decoy = args.max_decoy
    resample_precursor_mz = args.resample_precursor_mz
    data_dir = args.data_dir
    sample_strat = args.sample_strat
    softmax_temperature = args.softmax_temperature
    seed = args.seed
    decoy_suffix = args.decoy_suffix
    num_workers = args.num_workers
    max_pred_candidates = (
        args.max_pred_candidates if args.max_pred_candidates is not None else max_decoy
    )
    elements = args.elements

    # Optional restriction of the candidate adduct universe (request: let a model
    # consider only a user-chosen adduct set). None => all adducts of each mode.
    allowed_ions = None
    if args.adducts is not None:
        allowed_ions = set()
        for raw in args.adducts.split(","):
            raw = raw.strip()
            if not raw:
                continue
            canon = common.ion_remap.get(raw, raw)
            if canon not in common.ION_LST:
                raise SystemExit(
                    f"--adducts: '{raw}' is not a recognized adduct. "
                    f"Valid (canonical): {common.ION_LST}"
                )
            allowed_ions.add(canon)
        if not allowed_ions:
            raise SystemExit("--adducts was empty after parsing")
        print(f"Restricting candidate adducts to: {sorted(allowed_ions)}")

    fast_model_path = args.fast_model
    if fast_model_path is not None:
        fast_model_obj = fast_form_model.FastFFN.load_from_checkpoint(fast_model_path)
    else:
        fast_model_obj = None
    if args.gpu:
        if not torch.cuda.is_available():
            raise SystemExit("--gpu was set but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(seed)

    df = pd.read_csv(labels_file, sep="\t")
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    if debug:
        df = df[:100]

    # Drop labels whose true formula contains an element outside the alphabet we
    # decompose into. Those spectra are unrecoverable (SIRIUS never proposes such
    # a formula, the model has no output dimension for it) and would otherwise
    # crash formula_mass with a KeyError on the unknown element (e.g. Sn).
    el_str = elements if elements is not None else decomp.EL_STR_DEFAULT
    allowed_elements = set(re.findall(r"([A-Z][a-z]*)\[", el_str))

    def formula_in_alphabet(formula):
        return all(
            sym in allowed_elements
            for sym, _ in re.findall(common.CHEM_FORMULA_SIZE, formula)
        )

    in_alphabet = df["formula"].map(formula_in_alphabet)
    n_dropped = int((~in_alphabet).sum())
    if n_dropped:
        dropped_elements = sorted(
            {
                sym
                for formula in df.loc[~in_alphabet, "formula"]
                for sym, _ in re.findall(common.CHEM_FORMULA_SIZE, formula)
                if sym not in allowed_elements
            }
        )
        print(
            f"Dropping {n_dropped}/{len(df)} labels whose true formula has elements "
            f"outside the alphabet {sorted(allowed_elements)}: {dropped_elements}"
        )
        df = df[in_alphabet].reset_index(drop=True)

    specs = df["spec"].to_list()
    true_formulae = df["formula"].to_list()
    true_ionizations = df["ionization"].to_list()
    # Predicted precursor m/z of the true (formula, ion); folds in the n-mer
    # factor so [2M+...] precursors are placed correctly.
    true_masses = [
        common.neutral_mass_to_precursor_mz(common.formula_mass(true_form), true_ion)
        for true_form, true_ion in zip(true_formulae, true_ionizations, strict=False)
    ]

    df["true_mass"] = true_masses
    instruments = df["instrument"].values
    true_masses = np.array(true_masses)

    #  resample ms1 mass and get decoy by different instrument
    if resample_precursor_mz:
        errors = [common.get_instr_tol(i) for i in instruments]
        precursor_mz = resample_precursor_fn(true_masses, errors)
    else:
        # if do not resample ms1 mass, then get the measured massby inspecting the files
        precursor_mz = []
        for spec in specs:
            spec_file = Path(data_dir) / "spec_files" / f"{spec}.ms"
            meta, tuples = common.parse_spectra(spec_file)
            parentmass = float(meta.get("precursor_mz", meta["parentmass"]))
            precursor_mz.append(parentmass)
    precursor_mz = np.array(precursor_mz)

    abs_diffs = np.abs(true_masses - precursor_mz)
    rel_diffs = common.clipped_ppm(abs_diffs, true_masses)

    if np.any(abs_diffs > 3):
        raise ValueError(
            "Some spectra have abs mass diff > 3 Da between theoretical and precursor mass"
        )

    spec2form = dict(zip(specs, true_formulae, strict=False))
    spec2ion = dict(zip(specs, true_ionizations, strict=False))
    spec2parentmass = dict(zip(specs, precursor_mz, strict=False))
    spec2instrument = dict(zip(specs, instruments, strict=False))

    spec_to_form_list = {}
    spec_to_ion_list = {}
    spec_to_found = defaultdict(lambda: False)
    # mass_to_formulas is safe to share across specs/ions: SIRIUS decomp output
    # for a given neutral mass depends only on the mass, not on the (spec, ion)
    # that produced it. But the (ion, formula) pairing MUST be kept per-spec:
    # two different specs can yield the same rounded neutral mass under
    # different ion hypotheses, and we must not cross-contaminate their ions.
    mass_to_formulas = {}
    spec_to_ion_masses = defaultdict(list)  # spec -> [(ion, mass), ...]

    # Per-spec ion mode, derived from the (canonical) true-ion adduct sign.
    # Each spectrum only considers adducts of its own mode — no [M+H]+ decoys
    # for a [M-H]- spectrum, etc.
    spec_modes = [common.ion_mode_from_adduct(ion) for ion in true_ionizations]
    specs_arr = np.array(specs)
    precursor_mz_arr = np.array(precursor_mz)
    modes_arr = np.array(spec_modes)

    # First pass: enumerate every (spec, ion, neutral mass) triple WITHOUT
    # calling SIRIUS. SIRIUS decomp depends only on the neutral mass, so we
    # defer it until we have the global union of masses across all ions.
    for mode, ion_subset in common.ion_mode_to_ions.items():
        mask = modes_arr == mode
        if not mask.any():
            continue
        mode_specs = specs_arr[mask]
        mode_pmz = precursor_mz_arr[mask]
        for ion in ion_subset:
            # Optionally restrict the candidate adduct universe (e.g. a model
            # trained only on [M+H]+/[M+Na]+ should consider only those).
            if allowed_ions is not None and ion not in allowed_ions:
                continue
            # Neutral *monomer* mass for SIRIUS; divides out the n-mer factor so
            # [2M+...] candidates decompose the correct monomer mass.
            decoy_masses = [
                common.precursor_mz_to_neutral_mass(pm, ion) for pm in mode_pmz
            ]
            decoy_masses = decomp.get_rounded_masses(decoy_masses)
            for spec, mass in zip(mode_specs, decoy_masses, strict=False):
                spec_to_ion_masses[spec].append((ion, mass))

    # Single global SIRIUS call over the union of neutral masses across all
    # (spec, ion) pairs. Dedup happens inside run_sirius, so overlapping
    # masses across adducts are computed exactly once.
    all_masses = sorted(
        {m for ion_masses in spec_to_ion_masses.values() for _, m in ion_masses}
    )
    print(
        f"Global SIRIUS decomp over {len(all_masses)} unique neutral masses across all ions..."
    )
    sirius_kwargs = {
        "filter_": decomp_filter,
        "ppm": 10,
        "loglevel": "NONE",
        "cores": num_workers if num_workers > 0 else 1,
        "max_batch": args.max_batch,
        "mass_sort": False,
    }
    if elements is not None:
        sirius_kwargs["el_str"] = elements
    out_dict = decomp.run_sirius(all_masses, **sirius_kwargs)
    mass_to_formulas.update(out_dict)
    del out_dict

    # The fast filter score is a pure function of the candidate FORMULA: the
    # model never sees the spectrum or the adduct (FastFFN.forward consumes only
    # the formula's element-count embedding). The original code re-embedded and
    # re-scored the same formulae once per spectrum, so a formula shared by N
    # specs/ions was scored N times — the dominant cost for large inputs, none
    # of which the GPU helps with. Instead, score every unique formula ONCE here
    # and reuse the scores via a {formula: score} lookup in filter_spec_entries.
    # Output is byte-identical: the per-spec path consumes the exact same
    # per-formula scores (the MLP has no batch-coupled layers, so a formula's
    # score is independent of how candidates are batched), then argsorts them
    # in the unchanged candidate order.
    formula_to_score = None
    if sample_strat == "fast_filter" and fast_model_obj is not None:
        unique_forms = np.array(
            sorted({f for forms in mass_to_formulas.values() for f in forms}),
            dtype=object,
        )
        print(
            f"Fast-filter: scoring {len(unique_forms)} unique candidate formulae "
            f"once (reused across all spectra)..."
        )
        formula_to_score = {}
        # Chunk so the transient embedding list + DataFrame inside
        # fast_filter_score stays bounded for very large formula universes.
        # GPU memory is bounded by the DataLoader batch_size (batches are
        # streamed to device inside fast_filter_score), NOT by the chunk size,
        # so the chunk only caps host RAM. batch_size is kept at the original
        # per-spec value (1024) so the model math is unchanged.
        score_chunk = 1_000_000
        for start in range(0, len(unique_forms), score_chunk):
            chunk = unique_forms[start : start + score_chunk]
            chunk_scores = fast_model_obj.fast_filter_score(
                "global", chunk, chunk, device, batch_size=1024
            )
            formula_to_score.update(zip(chunk, chunk_scores, strict=False))

    # Build the per-spectrum candidate sets. This phase is pure-Python (set
    # building + per-candidate score lookups) and was the remaining single-core
    # bottleneck after the fast filter was vectorized, so fan it out across
    # processes. On Linux, fork lets workers read the large mass_to_formulas /
    # formula_to_score tables via copy-on-write without pickling them.
    ctx_kwargs = dict(
        mass_to_formulas=mass_to_formulas,
        formula_to_score=formula_to_score,
        spec2form=spec2form,
        spec2parentmass=spec2parentmass,
        max_decoy=max_decoy,
        max_pred_candidates=max_pred_candidates,
        sample_strat=sample_strat,
        softmax_temperature=softmax_temperature,
        seed=seed,
    )

    items = list(spec_to_ion_masses.items())
    n_proc = num_workers if num_workers and num_workers > 1 else (os.cpu_count() or 1)
    n_proc = min(n_proc, max(1, len(items)))

    if n_proc > 1:
        # Workers never need the NN: every candidate score was precomputed above
        # and is passed via precomputed_scores, so hand them fast_model=None and
        # avoid forking a live CUDA context into the children.
        _WORKER_CTX.clear()
        _WORKER_CTX.update(ctx_kwargs)
        _WORKER_CTX["fast_model"] = None
        _WORKER_CTX["device"] = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Building candidate sets per spectrum across {n_proc} processes...")
        chunksize = max(1, len(items) // (n_proc * 8))
        with mp.get_context("fork").Pool(n_proc) as pool:
            output_dicts = list(
                tqdm(
                    pool.imap(_filter_spec_entries_worker, items, chunksize=chunksize),
                    total=len(items),
                    desc="Candidate sets",
                )
            )
    else:
        output_dicts = [
            filter_spec_entries(
                spec,
                ion_masses,
                fast_model=fast_model_obj,
                device=device,
                **ctx_kwargs,
            )
            for spec, ion_masses in tqdm(items, desc="Candidate sets")
        ]
    for out_dict in output_dicts:
        spec = out_dict["spec"]
        cands = out_dict["out_cands"]
        ions = out_dict["out_ions"]
        was_found = out_dict["was_found"]

        spec_to_form_list[spec] = [str(i) for i in cands]
        spec_to_ion_list[spec] = [str(i) for i in ions]
        spec_to_found[spec] = was_found

    df["decoy_formulae"] = [
        ",".join(spec_to_form_list[i]) if len(spec_to_form_list.get(i, [])) > 0 else ""
        for i in specs
    ]
    df["decoy_ions"] = [
        ",".join(spec_to_ion_list[i]) if len(spec_to_ion_list.get(i, [])) > 0 else ""
        for i in specs
    ]

    decoy_forms_is_null = [str(i).strip() == "[]" for i in df["decoy_formulae"]]
    df.loc[decoy_forms_is_null, "decoy_formulae"] = ""
    df.loc[decoy_forms_is_null, "decoy_ions"] = ""

    df["decomp_recover"] = [spec_to_found[i] for i in specs]
    df["parentmass"] = precursor_mz
    df["abs_mass_diff"] = abs_diffs
    df["rel_mass_diff"] = rel_diffs

    save_dir = labels_file.parent / "decoy_labels"
    save_dir.mkdir(exist_ok=True)

    if decoy_suffix is None:
        decoy_suffix = f"{decomp_filter}"

    save_path = save_dir / f"decoy_label_{decoy_suffix}.tsv"
    print(f"Save to {save_path}")
    df.to_csv(save_path, sep="\t", index=None)

    # --- Honest pred-candidates file. ---
    # One row per surviving candidate from the full deployment pipeline
    # (SIRIUS decomp + sampler, true form NOT pre-excluded). The true pair
    # appears iff it would have been kept in practice; specs whose true pair
    # was dropped are recorded without it — the realistic failure mode.
    # Build the pred-candidate table by concatenating per-spec arrays instead of
    # appending one Python dict per (spec, candidate). That dict loop was a
    # second single-core hotspot — there can be hundreds of millions of rows.
    pipeline_recover = {}
    spec_cols, cand_cols, ion_cols, pm_cols, instr_cols = [], [], [], [], []
    for out_dict in tqdm(output_dicts, desc="Pred candidates"):
        spec = out_dict["spec"]
        pcands = np.asarray(out_dict["pred_cands"], dtype=object)
        pions = np.asarray(out_dict["pred_ions"], dtype=object)
        n = len(pcands)
        pipeline_recover[spec] = (
            bool(np.any((pcands == spec2form[spec]) & (pions == spec2ion[spec])))
            if n
            else False
        )
        if n == 0:
            continue
        spec_cols.append(np.full(n, spec, dtype=object))
        cand_cols.append(pcands)
        ion_cols.append(pions)
        pm_cols.append(np.full(n, spec2parentmass[spec]))
        instr_cols.append(np.full(n, spec2instrument[spec], dtype=object))

    pred_cols = ["spec", "cand_form", "cand_ion", "parentmass", "instrument"]
    if spec_cols:
        pred_df = pd.DataFrame(
            {
                "spec": np.concatenate(spec_cols),
                "cand_form": np.concatenate(cand_cols),
                "cand_ion": np.concatenate(ion_cols),
                "parentmass": np.concatenate(pm_cols),
                "instrument": np.concatenate(instr_cols),
            }
        )
    else:
        pred_df = pd.DataFrame(columns=pred_cols)
    pred_path = save_dir / f"pred_candidates_{decoy_suffix}.tsv"
    print(f"Save to {pred_path}")
    pred_df.to_csv(pred_path, sep="\t", index=None)

    # Summary: two recovery rates along the pipeline.
    n_total = len(specs)
    n_decomp = sum(1 for s in specs if spec_to_found.get(s, False))
    n_pipeline = sum(1 for s in specs if pipeline_recover.get(s, False))
    print(
        f"Pipeline recovery: SIRIUS decomp recovered true form for "
        f"{n_decomp}/{n_total} ({100 * n_decomp / max(n_total, 1):.1f}%); "
        f"after sampling, {n_pipeline}/{n_total} ({100 * n_pipeline / max(n_total, 1):.1f}%) "
        f"retained the true pair in the pred-candidate set."
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
