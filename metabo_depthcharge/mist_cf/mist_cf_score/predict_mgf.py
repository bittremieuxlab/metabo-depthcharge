# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""predict_mgf.py - Predict molecular formulae for a raw MGF, no ground truth needed.

Pipeline: SIRIUS mass decomposition (per spectrum's precursor m/z + resolved
ion mode) -> subformula assignment -> mist_cf_score model -> ranked
(cand_form, cand_ion) per spectrum.

Ion mode (positive/negative) is required to pick which adducts to decompose
against, and is resolved per spectrum in one of two ways:
  --adduct-key    an adduct field already on the spectrum (e.g. from a prior
                   annotation); mapped to a mode via common.ion_mode_from_adduct.
  --ionmode-key   a separate polarity field (e.g. "Ion_Mode"), matched
                   case-insensitively against "positive"/"negative" or "+"/"-",
                   for MGFs where the adduct itself is unknown (the whole point
                   of this script is normally to predict it).
--adduct-key is tried first when both are given and resolves to a known
adduct; the model still predicts which specific adduct within that mode.
Spectra for which neither resolves are skipped (not aborted).

--benchmark reports the true (formula, adduct) retainment/accuracy at each
pipeline stage -- SIRIUS decomp, the fast-filter cap (if used), and the
mist_cf model at top-1/5/10 -- using ground truth read from
--benchmark-formula-field/--benchmark-adduct-field. The candidate-generation
pipeline itself is unaffected by ground truth (same as a real blind run);
truth is only used to measure where it survives.
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import common, decomp
from ..fast_form_score import fast_form_model
from . import mist_cf_data, mist_cf_model
from .predict import run_inference

# Raw MGF instrument strings -> mist-cf canonical instrument names. Mirrors
# preprocessing/01_prepare_mgf.py's map (kept separate: each MGF-ingestion
# entrypoint owns its own raw-field mapping).
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mgf-file", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--checkpoint-pth", required=True)
    parser.add_argument("--max-num", type=int, default=None)

    parser.add_argument("--id-key", default="TITLE")
    parser.add_argument("--precursor-mz-key", default="PEPMASS")
    parser.add_argument("--adduct-key", default=None, help="MGF field with an adduct; resolves ion mode via common.ion_mode_from_adduct.")
    parser.add_argument("--ionmode-key", default=None, help="MGF field with ion-mode text ('positive'/'negative' or '+'/'-'); fallback when --adduct-key is absent/unresolvable.")
    parser.add_argument("--instrument-key", default=None)
    parser.add_argument("--instrument-override", default=None)
    parser.add_argument("--default-instrument", default="Unknown (LCMS)")

    parser.add_argument("--decomp-filter", default="RDBE")
    parser.add_argument("--decomp-ppm", type=int, default=10)
    parser.add_argument("--elements", default=None, help="SIRIUS element alphabet string; defaults to decomp.EL_STR_DEFAULT.")
    parser.add_argument(
        "--adducts",
        default=None,
        help="Comma-separated adducts to restrict candidate generation to (e.g. '[M+H]+,[M+Na]+' "
        "for a model trained on only those, per 02_create_decoy_label.py's --adducts). Each must "
        "be in common.ION_LST (aliases normalized). Default: every adduct of each spectrum's ion "
        "mode -- only safe if the checkpoint was trained on the full mode vocabulary.",
    )

    parser.add_argument("--fast-model", default=None, help="Optional fast-filter checkpoint to cap candidates per spec before scoring.")
    parser.add_argument("--fast-num", type=int, default=None, help="Candidates to keep per spec after fast filtering.")

    parser.add_argument("--output-num", type=int, default=5, help="Top candidates to keep per spectrum in the final output.")

    parser.add_argument("--benchmark", action="store_true", default=False, help="Report true-formula retainment/accuracy at each pipeline stage. Requires --benchmark-formula-field/--benchmark-adduct-field on the MGF.")
    parser.add_argument("--benchmark-formula-field", default="formula")
    parser.add_argument("--benchmark-adduct-field", default="adduct")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def resolve_ion_mode(meta_ci, adduct_key, ionmode_key):
    """"pos"/"neg" for a spectrum, or None if unresolvable from either field."""
    if adduct_key:
        raw = meta_ci.get(adduct_key.upper())
        if raw:
            canon = common.standardize_adduct(str(raw), fail_silent=True)
            if canon in common.ION_LST:
                return common.ion_to_mode[canon]
    if ionmode_key:
        raw = meta_ci.get(ionmode_key.upper())
        if raw:
            v = str(raw).strip().lower()
            if "neg" in v or v == "-":
                return "neg"
            if "pos" in v or v == "+":
                return "pos"
    return None


def canon_formula(formula):
    """Round-trip a formula string through common's dense-vector encoding so
    it's comparable to SIRIUS/model output regardless of element order in the
    source MGF (e.g. "H12C6O6" vs "C6H12O6"). Returns None if unparseable
    (e.g. an element outside common.VALID_ELEMENTS).
    """
    try:
        return common.vec_to_formula(common.formula_to_dense(str(formula)))
    except KeyError:
        return None


def predict_mgf():
    args = get_args()
    kwargs = args.__dict__
    debug = kwargs["debug"]

    save_dir = Path(kwargs["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    subform_dir = save_dir / "subform_assigns"
    subform_dir.mkdir(exist_ok=True)
    save_name = save_dir / "formatted_output.tsv"

    common.setup_logger(kwargs["save_dir"], log_name="mist_cf_pred_mgf.log", debug=debug)
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    logging.info(f"Args:\n{yaml_args}")
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    num_workers = kwargs["num_workers"]
    gpu = kwargs["gpu"]
    device = torch.device("cuda") if gpu else torch.device("cpu")

    model = mist_cf_model.MistNet.load_from_checkpoint(kwargs["checkpoint_pth"])
    logging.info(f"Loaded model from {kwargs['checkpoint_pth']}")

    # --- Pass 1: resolve every spectrum's id/mass/mode/instrument; skip (not
    # abort) anything unresolvable -- a prediction run over real-world data
    # shouldn't die on one bad row. ---
    parsed = common.parse_spectra_mgf(kwargs["mgf_file"], max_num=kwargs["max_num"])
    logging.info(f"Parsed {len(parsed)} spectra from {kwargs['mgf_file']}")

    benchmark = kwargs["benchmark"]
    resolved = {}
    seen_ids = {}
    skipped = []
    n_bench_missing = 0
    for i, (meta, spectra) in enumerate(tqdm(parsed, desc="Resolving spectra")):
        meta_ci = {str(k).upper(): v for k, v in meta.items()}

        def field(key, _m=meta_ci):
            return _m.get(key.upper()) if key else None

        spec_id = field(kwargs["id_key"])
        if not spec_id:
            spec_id = f"mgf_{i:06d}"
        if spec_id in seen_ids:
            seen_ids[spec_id] += 1
            spec_id = f"{spec_id}_{seen_ids[spec_id]}"
        else:
            seen_ids[spec_id] = 0

        mz_raw = field(kwargs["precursor_mz_key"])
        if not mz_raw:
            skipped.append((spec_id, "missing precursor m/z"))
            continue
        precursor_mz = float(str(mz_raw).split()[0])

        mode = resolve_ion_mode(meta_ci, kwargs["adduct_key"], kwargs["ionmode_key"])
        if mode is None:
            skipped.append((spec_id, "unresolvable ion mode"))
            continue

        if kwargs["instrument_override"]:
            instrument = kwargs["instrument_override"]
        else:
            raw_instr = field(kwargs["instrument_key"])
            instrument = INSTRUMENT_MAP.get(raw_instr, raw_instr or kwargs["default_instrument"])

        merged = common.merge_spec_tuples([spectra[0][1]], parent_mass=precursor_mz)
        peaks = common.max_thresh_spec(merged, max_peaks=model.max_subpeak, inten_thresh=0.003)
        if peaks.size == 0:
            skipped.append((spec_id, "no peaks after thresholding"))
            continue

        resolved[spec_id] = {
            "precursor_mz": precursor_mz,
            "mode": mode,
            "instrument": instrument,
            "peaks": peaks,
        }

        # Ground truth is only used for the --benchmark report below; it never
        # feeds back into candidate generation (same pipeline as a blind run).
        if benchmark:
            true_form_raw = field(kwargs["benchmark_formula_field"])
            true_adduct_raw = field(kwargs["benchmark_adduct_field"])
            true_form = canon_formula(true_form_raw) if true_form_raw else None
            true_ion = (
                common.standardize_adduct(str(true_adduct_raw), fail_silent=True)
                if true_adduct_raw
                else None
            )
            if true_form is None or true_ion is None:
                n_bench_missing += 1
            else:
                resolved[spec_id]["true_form"] = true_form
                resolved[spec_id]["true_ion"] = true_ion

    if skipped:
        logging.warning(f"Skipped {len(skipped)}/{len(parsed)} spectra: {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
    if not resolved:
        raise SystemExit("No spectra resolved; nothing to predict.")
    logging.info(f"Resolved {len(resolved)}/{len(parsed)} spectra")

    bench_specs = []
    if benchmark:
        bench_specs = [s for s, r in resolved.items() if "true_form" in r]
        if n_bench_missing:
            logging.warning(
                f"--benchmark: {n_bench_missing}/{len(resolved)} resolved spectra have no usable "
                f"'{kwargs['benchmark_formula_field']}'/'{kwargs['benchmark_adduct_field']}' field "
                "and are excluded from the benchmark report (still predicted normally)."
            )
        if not bench_specs:
            raise SystemExit("--benchmark given but no spectrum has a usable ground-truth formula/adduct.")
        logging.info(f"Benchmarking against {len(bench_specs)} spectra with ground truth")

    # Optional restriction of the candidate adduct universe -- required when
    # the checkpoint was trained on fewer adducts than a mode's full ION_LST
    # subset (e.g. massspecgym's mist_cf model only saw [M+H]+/[M+Na]+); see
    # 02_create_decoy_label.py's --adducts.
    allowed_ions = None
    if kwargs["adducts"] is not None:
        allowed_ions = set()
        for raw in kwargs["adducts"].split(","):
            raw = raw.strip()
            if not raw:
                continue
            canon = common.ion_remap.get(raw, raw)
            if canon not in common.ION_LST:
                raise SystemExit(f"--adducts: '{raw}' is not a recognized adduct. Valid (canonical): {common.ION_LST}")
            allowed_ions.add(canon)
        if not allowed_ions:
            raise SystemExit("--adducts was empty after parsing")
        logging.info(f"Restricting candidate adducts to: {sorted(allowed_ions)}")

    # --- Mass-only candidate generation: global SIRIUS decomp over the union
    # of neutral masses across all (spec, ion-of-its-mode) pairs. No ground
    # truth formula involved anywhere in this pipeline. ---
    spec_to_ion_masses = defaultdict(list)
    for mode, ion_subset in common.ion_mode_to_ions.items():
        mode_specs = [s for s, r in resolved.items() if r["mode"] == mode]
        if not mode_specs:
            continue
        for ion in ion_subset:
            if allowed_ions is not None and ion not in allowed_ions:
                continue
            masses = [
                common.precursor_mz_to_neutral_mass(resolved[s]["precursor_mz"], ion)
                for s in mode_specs
            ]
            masses = decomp.get_rounded_masses(masses)
            for s, m in zip(mode_specs, masses, strict=False):
                spec_to_ion_masses[s].append((ion, m))

    all_masses = sorted({m for lst in spec_to_ion_masses.values() for _, m in lst})
    logging.info(f"Global SIRIUS decomp over {len(all_masses)} unique neutral masses...")
    sirius_kwargs = dict(
        filter_=kwargs["decomp_filter"],
        ppm=kwargs["decomp_ppm"],
        loglevel="NONE",
        cores=num_workers if num_workers > 0 else 1,
        mass_sort=False,
    )
    if kwargs["elements"] is not None:
        sirius_kwargs["el_str"] = kwargs["elements"]
    mass_to_formulas = decomp.run_sirius(all_masses, **sirius_kwargs)

    fast_model_obj = None
    if kwargs["fast_model"] is not None:
        fast_model_obj = fast_form_model.FastFFN.load_from_checkpoint(kwargs["fast_model"])

    rows = []
    n_no_cands = 0
    stage1_hits, stage2_hits = {}, {}
    for spec, ion_masses in spec_to_ion_masses.items():
        is_bench = benchmark and spec in resolved and "true_form" in resolved[spec]
        true_pair = (resolved[spec]["true_ion"], resolved[spec]["true_form"]) if is_bench else None

        cand = {(ion, f) for ion, m in ion_masses for f in mass_to_formulas.get(m, [])}
        if not cand:
            n_no_cands += 1
            if is_bench:
                stage1_hits[spec] = False
                stage2_hits[spec] = False
            continue
        if is_bench:
            # Compare canonicalized on both sides -- SIRIUS's own element
            # ordering need not match canon_formula's, only self-consistency
            # matters here (true_pair already went through canon_formula too).
            stage1_hits[spec] = any(ion == true_pair[0] and canon_formula(f) == true_pair[1] for ion, f in cand)

        cand_ions, cand_forms = map(np.array, zip(*cand, strict=False))
        if fast_model_obj is not None and kwargs["fast_num"] is not None and len(cand_forms) > kwargs["fast_num"]:
            idx = fast_model_obj.fast_filter_sampling(spec, cand_forms, cand_ions, kwargs["fast_num"], device)
            cand_forms, cand_ions = cand_forms[idx], cand_ions[idx]
        if is_bench:
            stage2_hits[spec] = any(
                ion == true_pair[0] and canon_formula(f) == true_pair[1]
                for ion, f in zip(cand_ions.tolist(), cand_forms.tolist(), strict=False)
            )
        for ion, form in zip(cand_ions, cand_forms, strict=False):
            rows.append(
                {
                    "spec": spec,
                    "cand_form": form,
                    "cand_ion": ion,
                    "parentmass": resolved[spec]["precursor_mz"],
                    "instrument": resolved[spec]["instrument"],
                }
            )
    logging.info(f"SIRIUS decomp recovered candidates for {len(spec_to_ion_masses) - n_no_cands}/{len(spec_to_ion_masses)} spectra")
    if not rows:
        raise SystemExit("No candidates survived SIRIUS decomp for any spectrum.")
    label_df = pd.DataFrame(rows)
    label_df.to_csv(save_dir / "pred_labels.tsv", sep="\t", index=None)

    # --- Subformula assignment, no true (formula, ion) appended anywhere. ---
    spec_to_entries = defaultdict(lambda: {"forms": [], "ions": []})
    for row in label_df.itertuples():
        spec_to_entries[row.spec]["forms"].append(row.cand_form)
        spec_to_entries[row.spec]["ions"].append(row.cand_ion)

    all_entries = []
    for spec, ent in spec_to_entries.items():
        mass_diff_thresh = common.get_instr_tol(resolved[spec]["instrument"])
        export_dicts = [
            {
                "spec": resolved[spec]["peaks"],
                "mass_diff_type": "ppm",
                "spec_name": spec,
                "mass_diff_thresh": mass_diff_thresh,
                "form": form,
                "ion_type": ion,
            }
            for form, ion in zip(ent["forms"], ent["ions"], strict=False)
        ]
        all_entries.append({"spec_name": spec, "export_dicts": export_dicts, "output_dir": subform_dir})

    logging.info("Assigning subformulae")

    def export_wrapper(x):
        return common.assign_single_spec(**x)

    if num_workers in (0, 1) or debug:
        [export_wrapper(e) for e in tqdm(all_entries)]
    else:
        common.chunked_parallel(all_entries, export_wrapper, chunks=100, max_cpu=num_workers)

    # --- Score candidates with the trained mist_cf model. ---
    pred_dataset = mist_cf_data.PredDataset(
        label_df,
        subform_dir=subform_dir,
        num_workers=num_workers,
        max_subpeak=model.max_subpeak,
        ablate_cls_error=not model.cls_mass_diff,
    )
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model.eval()
    model = model.to(device)

    logging.info("Predicting with mist_cf")
    if benchmark:
        # Unbounded ranking so top-10 recall is measurable even when
        # --output-num < 10; the saved file is still truncated to --output-num.
        full_out_df = run_inference(model, pred_loader, device, output_num=None)
        save_df = (
            full_out_df.groupby("spec").head(kwargs["output_num"])
            if kwargs["output_num"] is not None
            else full_out_df
        )
    else:
        save_df = run_inference(model, pred_loader, device, kwargs["output_num"])
    save_df.to_csv(save_name, sep="\t", index=None)
    logging.info(f"Wrote {save_name}")

    if benchmark:
        # Canonicalize the model's echoed cand_form the same way true_form was
        # canonicalized, so the merge doesn't depend on SIRIUS's raw element
        # ordering matching canon_formula's.
        full_out_df = full_out_df.assign(_cand_form_canon=full_out_df["cand_form"].map(canon_formula))
        truth_df = pd.DataFrame(
            [
                {"spec": s, "_cand_form_canon": resolved[s]["true_form"], "cand_ion": resolved[s]["true_ion"]}
                for s in bench_specs
            ]
        )
        hits = full_out_df.merge(truth_df, on=["spec", "_cand_form_canon", "cand_ion"], how="inner")
        best_rank = hits.groupby("spec")["rank"].min()

        n_bench = len(bench_specs)
        stage1_recall = sum(stage1_hits.get(s, False) for s in bench_specs) / n_bench
        stage2_recall = (
            sum(stage2_hits.get(s, False) for s in bench_specs) / n_bench
            if fast_model_obj is not None
            else None
        )
        stage3_recall = {k: float((best_rank <= k).sum()) / n_bench for k in (1, 5, 10)}

        report_lines = [
            f"=== Benchmark (n={n_bench} spectra with usable ground truth) ===",
            f"Stage 1 (SIRIUS decomp)        : {stage1_recall:.1%} true formula retained",
            "Stage 2 (+ fast filter)        : "
            + (f"{stage2_recall:.1%} true formula retained" if stage2_recall is not None else "not used (no --fast-model)"),
            f"Stage 3 (mist_cf model) top-1  : {stage3_recall[1]:.1%} accuracy",
            f"Stage 3 (mist_cf model) top-5  : {stage3_recall[5]:.1%} accuracy",
            f"Stage 3 (mist_cf model) top-10 : {stage3_recall[10]:.1%} accuracy",
        ]
        for line in report_lines:
            logging.info(line)
            print(line)

        metrics = {
            "n_benchmark_spectra": n_bench,
            "sirius_decomp_recall": stage1_recall,
            "fast_filter_recall": stage2_recall,
            "model_recall_at_1": stage3_recall[1],
            "model_recall_at_5": stage3_recall[5],
            "model_recall_at_10": stage3_recall[10],
        }
        metrics_path = save_dir / "benchmark_metrics.json"
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2)
        logging.info(f"Wrote {metrics_path}")


if __name__ == "__main__":
    start_time = time.time()
    predict_mgf()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
