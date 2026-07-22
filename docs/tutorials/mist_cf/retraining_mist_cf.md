# Retraining MIST-CF

A growing body of computational metabolomics research uses molecular formulae as either "given" or as conditioning information.
In `metabo-depthcharge`, for example, there is a {class}`~metabo_depthcharge.encoders.SubformulaEncoder` adopting the peak subformulae embedding strategies of [MIST](https://github.com/samgoldman97/mist) and [MIST-CF](https://github.com/samgoldman97/mist-cf).
In practical model usage, access to ground truth molecular formulae may not be available.
As such, any machine learning model that utilizes such information should also ship with a molecular formula predictor.
Ideally, that formula predictor should be trained on the same data splits, to ensure fair evaluation of the performance of the overall pipeline.
This tutorial will outline how to re-train the our vendored `mist-cf` implementation.

## Prerequisites

- A `labels.tsv` with columns `spec, formula, ionization, instrument, dataset`,
  and the corresponding `.ms` spectrum files under `spec_files/`.
- For decoy generation:
  [SIRIUS](https://bio.informatik.uni-jena.de/software/sirius/) — set
  `$SIRIUS_PATH` and log in (tested on SIRIUS v6.3.3).
- The `fast_filter_best.ckpt` model from the
  [MIST-CF repo](https://github.com/samgoldman97/mist-cf).

```{note}
Both positive- and negative-mode adducts are supported, and the alphabet is
aligned with `metabo_depthcharge.spec.adducts.ADDUCT_VOCAB` — including the
multimers `[2M+H]+`, `[2M+Na]+`, `[2M-H]-`, `[2M+HCOOH-H]-`, `[2M+CH3COOH-H]-`
and `[M+Br]-`. Recognized adducts and their aliases live in `ION_LST` /
`ion_remap` in `metabo_depthcharge/mist_cf/common/chem_utils.py`; the
electron-mass sign is handled per mode in `ion_to_mass`, and the monomer count
in `ion_to_nmer` (so `[2M+...]` precursors map to the correct neutral *monomer*
mass via `precursor_mz_to_neutral_mass` / `neutral_mass_to_precursor_mz`). An
adduct outside `ION_LST` raises an error during data preparation rather than
being silently mishandled.

New adducts are appended to `ION_LST` (indices 11+) so the original indices
0–10 never move; because the model's ion embedding is indexed by position,
**adding adducts means retraining** — an old checkpoint cannot be loaded against
the wider vocabulary.
```

```{note}
**Multimers: precursor is exact, fragments are monomer-only.** `ion_to_nmer`
places the `[2M+...]` *precursor* correctly, but `assign_subforms` still
enumerates subformulae of the monomer `M`, so peaks above `M` (genuine
dimer-retaining fragments) are dropped rather than assigned. On
`enveda_180` (≈28% dimer spectra) the dropped signal is small — ~3% of intensity
on average (≈6% for `[2M+H]+`, near-zero for `[2M+Na]+`/`[2M-H]-`); most apparent
above-`M` intensity is just the monomer ion's isotope envelope, which is dropped
for every spectrum anyway.

Caveat for `(adduct, formula)` accuracy: those dropped peaks are exactly the ones
that separate a true `[2M+X]+`/`M` from the same-precursor `[M+X]+`/`~2M`
competitor (the monomer fragments fit both), so treat it as *measure it*, not
*ignore it*. **Recommended:** train *with* dimers (do not `--adducts`-filter them
out), then check an adduct-confusion matrix on dimer spectra; only if
`[2M+X]+ → [M+X]+` confusion is systematic and correlates with high-mass
intensity is it worth extending subformula assignment to the `n·M` space for
dimer candidates.
```

## 1. Prepare the data

Run the preprocessing scripts in order (`01` → `02` → `03` → `04`).
Example using MassSpecGym:

```bash
python -m metabo_depthcharge.mist_cf.preprocessing.01_prepare_mgf \
    --mgf-file ./MassSpecGym.mgf \
    --out-dir ./mistcf/ \
    --dataset msgym \
    --id-key IDENTIFIER \
    --formula-key FORMULA \
    --precursor-mz-key PRECURSOR_MZ \
    --adduct-key ADDUCT \
    --instrument-key INSTRUMENT_TYPE
```

```{important}
This is a **retraining** pipeline, so a ground-truth molecular formula is
**required for every spectrum**.
```

## 2. Generate decoys (negative candidate formulae) with SIRIUS

```bash
python -m metabo_depthcharge.mist_cf.preprocessing.02_create_decoy_label \
    --decomp-filter COMMON \
    --label-file ./mistcf/labels.tsv \
    --data-dir ./mistcf/ \
    --max-decoy 256 \
    --sample-strat fast_filter \
    --fast-model ./fast_filter_best.ckpt \
    --resample-precursor-mz \
    --num-workers 24 \
    --max-batch 10000 \
    --elements "C[0-]N[0-]O[0-]H[0-]S[0-3]P[0-1]I[0-1]Cl[0-3]F[0-6]Br[0-1]" \
    --adducts "[M+H]+,[M+Na]+"
```

Adjust `--adducts` to what is available in your dataset.
Passing this argument makes it so that only these adducts are considered when generating negative candidates.
A model will, hence, also only ever be able to predict the adducts passed here.

Note that the fast filter should be appropriate for your use-case.
See {doc}`fast_filter` for when (and how) to retrain it.

## 3. Build prediction labels and split.

```bash
# Extract fold labels from MassSpecGym splits:
awk -F'\t' 'BEGIN{OFS="\t"} NR==1{print "spec","Fold_0"; next} {print $1,$13}' \
    ./MassSpecGym.tsv \
    > ./mistcf/splits/split_msgym.tsv
# ... Or generate your own split_....tsv

python -m metabo_depthcharge.mist_cf.preprocessing.03_create_pred_label \
    --pred-candidates ./mistcf/decoy_labels/pred_candidates_COMMON.tsv \
    --split-file ./mistcf/splits/split_msgym.tsv \
    --data-dir ./mistcf/ \
```

## 4. Assign subformulae for all negative candidates per spectrum

```bash
python -m metabo_depthcharge.mist_cf.preprocessing.04_create_subformulae_assignment \
    --data-dir "$DATA_DIR" \
    --decoy-label "$DECOY_LABEL" \
    --max-formulae 20 \
    --inten-thresh 0.003 \
    --mass-diff-thresh 15 \
    --mass-diff-type ppm \
    --num-workers 10 \
    --start-idx 0 \
    --end-idx 10_000
```

`--start-idx` and `--end-idx` are optional arguments to run slice the total spectra and compute only the slice.
Useful for parallellism e.g. slurm batch arrays.
When omitted, the entire dataset is run at once.

## 5. Train MIST-CF

To train, plug in your own paths:

```bash
python -m metabo_depthcharge.mist_cf.mist_cf_score.train \
    --gpu \
    --num-workers 8 \
    --batch-size 64 \
    --max-epochs 200 \
    --max-decoy 32 \
    --max-subpeak 20 \
    --save-dir "$DATA_DIR/results/train/" \
    --dataset-name msgym \
    --split-file "$DATA_DIR/splits/split_own.tsv" \
    --decoy-label "$DATA_DIR/decoy_labels/decoy_label_COMMON.tsv" \
    --subform-dir "$DATA_DIR/subformulae/formulae_spec_decoy_label_COMMON" \
    --layers 4 \
    --dropout 0.1 \
    --learning-rate 0.0003 \
    --lr-decay-frac 0.8830 \
    --weight-decay 0 \
    --form-encoder 'abs-sines' \
    --val-check-interval 1000 \
    --patience 50
```

Differences with original MIST-CF:
- Early stopping on val acc instead of val loss (better copes with OOD splits)
- `batch_size=4` to  `batch_size=64`
- `layers=2` to  `layers=4`
- `learning-rate=0.00045` to  `learning-rate=0.0003`

## 6. Predict

Generate formula predictions for any spectrum in any mgf file via (plug in your own paths):

```bash
python -m metabo_depthcharge.mist_cf.mist_cf_score.predict_mgf \
    --gpu \
    --num-workers 16 \
    --mgf-file /your/file.mgf \
    --checkpoint-pth /your/trained/mistcf.ckpt \
    --save-dir /your/save/dir \
    --id-key IDENTIFIER \
    --precursor-mz-key PRECURSOR_MZ \
    --adduct-key ADDUCT \
    --instrument-key INSTRUMENT_TYPE \
    --adducts "[M+H]+,[M+Na]+" \
    --fast-model /your/trained/fastfilter.ckpt \
    --fast-num 256 \
    --output-num 5 \
    --benchmark \
    --benchmark-formula-field FORMULA \
    --benchmark-adduct-field ADDUCT
```

The `--benchmark` flag makes the script report accuracy metrics for mgf files where a ground truth formula and adduct field are known (read via the `--benchmark-formula-field` and `--benchmark-adduct-field` flags).
