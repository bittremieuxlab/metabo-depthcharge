# Using MIST-CF

The main entry-point to using our vendored `mist-cf` should be `python -m metabo_depthcharge.mist_cf.mist_cf_score.predict_mgf`:


```bash
python -m metabo_depthcharge.mist_cf.mist_cf_score.predict_mgf --help


usage: predict_mgf.py [-h] --mgf-file MGF_FILE --save-dir SAVE_DIR --checkpoint-pth CHECKPOINT_PTH [--max-num MAX_NUM]
                      [--id-key ID_KEY] [--precursor-mz-key PRECURSOR_MZ_KEY] [--adduct-key ADDUCT_KEY]
                      [--ionmode-key IONMODE_KEY] [--instrument-key INSTRUMENT_KEY]
                      [--instrument-override INSTRUMENT_OVERRIDE] [--default-instrument DEFAULT_INSTRUMENT]
                      [--decomp-filter DECOMP_FILTER] [--decomp-ppm DECOMP_PPM] [--elements ELEMENTS] [--adducts ADDUCTS]
                      [--fast-model FAST_MODEL] [--fast-num FAST_NUM] [--output-num OUTPUT_NUM] [--benchmark]
                      [--benchmark-formula-field BENCHMARK_FORMULA_FIELD] [--benchmark-adduct-field BENCHMARK_ADDUCT_FIELD]
                      [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--gpu] [--debug]

predict_mgf.py - Predict molecular formulae for a raw MGF, no ground truth needed.

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
--benchmark-formula-field/--benchmark-adduct-field. The mist_cf model's
top-1/5/10 accuracy is also broken down per ground-truth adduct. The
candidate-generation pipeline itself is unaffected by ground truth (same as
a real blind run); truth is only used to measure where it survives.

options:
  -h, --help            show this help message and exit
  --mgf-file MGF_FILE
  --save-dir SAVE_DIR
  --checkpoint-pth CHECKPOINT_PTH
  --max-num MAX_NUM
  --id-key ID_KEY
  --precursor-mz-key PRECURSOR_MZ_KEY
  --adduct-key ADDUCT_KEY
                        MGF field with an adduct; resolves ion mode via common.ion_mode_from_adduct.
  --ionmode-key IONMODE_KEY
                        MGF field with ion-mode text ('positive'/'negative' or '+'/'-'); fallback when --adduct-key is
                        absent/unresolvable.
  --instrument-key INSTRUMENT_KEY
  --instrument-override INSTRUMENT_OVERRIDE
  --default-instrument DEFAULT_INSTRUMENT
  --decomp-filter DECOMP_FILTER
  --decomp-ppm DECOMP_PPM
  --elements ELEMENTS   SIRIUS element alphabet string; defaults to decomp.EL_STR_DEFAULT.
  --adducts ADDUCTS     Comma-separated adducts to restrict candidate generation to (e.g. '[M+H]+,[M+Na]+' for a model trained
                        on only those, per 02_create_decoy_label.py's --adducts). Each must be in common.ION_LST (aliases
                        normalized). Default: every adduct of each spectrum's ion mode -- only safe if the checkpoint was
                        trained on the full mode vocabulary.
  --fast-model FAST_MODEL
                        Optional fast-filter checkpoint to cap candidates per spec before scoring.
  --fast-num FAST_NUM   Candidates to keep per spec after fast filtering.
  --output-num OUTPUT_NUM
                        Top candidates to keep per spectrum in the final output.
  --benchmark           Report true-formula retainment/accuracy at each pipeline stage. Requires --benchmark-formula-
                        field/--benchmark-adduct-field on the MGF.
  --benchmark-formula-field BENCHMARK_FORMULA_FIELD
  --benchmark-adduct-field BENCHMARK_ADDUCT_FIELD
  --batch-size BATCH_SIZE
  --num-workers NUM_WORKERS
  --gpu
  --debug
```
