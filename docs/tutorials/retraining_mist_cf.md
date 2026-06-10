# Retraining MIST-CF

`metabo_depthcharge.mist_cf` is a vendored copy of
[MIST-CF](https://github.com/samgoldman97/mist-cf) (Goldman et al., 2023,
MIT-licensed). It is included so the formula/subformula utilities behind
{doc}`subformulae` are self-contained, but it also ships the full retraining
pipeline if you want to **train the formula-prediction model on your own data**.

```{note}
This is an advanced workflow. Most users only need the high-level subformula
features described in {doc}`subformulae` and never invoke `mist_cf` directly.
```

## Prerequisites

- A `labels.tsv` with columns `spec, formula, ionization, instrument, dataset`,
  and the corresponding `.ms` spectrum files under `spec_files/`.
- For decoy generation:
  [SIRIUS](https://bio.informatik.uni-jena.de/software/sirius/) — set
  `$SIRIUS_PATH` and log in (tested on SIRIUS v6.3.3).
- The `fast_filter_best.ckpt` model from the
  [MIST-CF repo](https://github.com/samgoldman97/mist-cf).

```{warning}
The vendored code currently supports **positive ion mode only**. Negative mode
requires adding entries to `ION_LST`, `ion_to_mass`, `ion_to_add_vec`, and
`ion_remap` in `metabo_depthcharge/mist_cf/common/chem_utils.py` with the
correct electron-mass sign (`+ELECTRON_MASS` for negative-mode adducts such as
`[M-H]-`).
```

## 1. Prepare the data

Run the preprocessing scripts in order (`01` → `02` → `03` → `04`), pointing
`--data-dir` at your dataset. Example using MassSpecGym:

```bash
python -m metabo_depthcharge.mist_cf.preprocessing.01_prepare_msgym \
    --msgym-tsv data/msmsdbs/MassSpecGym.tsv \
    --out-dir data/mistcf/msgym \
    --create-split
```

Generate negative candidate formulae with SIRIUS:

```bash
python -m metabo_depthcharge.mist_cf.preprocessing.02_create_decoy_label \
    --decomp-filter COMMON \
    --label-file data/mistcf/msgym/labels.tsv \
    --data-dir data/mistcf/msgym \
    --max-decoy 256 \
    --sample-strat fast_filter \
    --fast-model data/mistcf/fast_filter_best.ckpt \
    --resample-precursor-mz \
    --num-workers 8
```

Build the prediction labels:

```bash
python -m metabo_depthcharge.mist_cf.preprocessing.03_create_pred_label \
    --decoy-label data/mistcf/msgym/decoy_label_COMMON.tsv \
    --split-file data/mistcf/msgym/splits/split_msgym.tsv \
    --data-dir data/mistcf/msgym
```

Then run `04_create_subformulae_assignment` to assign subformulae over the
candidate set.

## 2. Train the fast filter (optional)

The fast filter pre-ranks candidate formulae before the main model scores them.

```bash
CUDA_VISIBLE_DEVICES=0 python -m metabo_depthcharge.mist_cf.fast_form_score.train \
    --gpu \
    --seed 1 \
    --num-workers 8 \
    --dataset-file data/biomols/biomols_with_decoys.txt \
    --split-file data/biomols/biomols_with_decoys_split.tsv \
    --batch-size 64 \
    --max-decoy 32 \
    --max-epochs 200 \
    --learning-rate 0.00036 \
    --lr-decay-frac 0.86425 \
    --weight-decay 0 \
    --layers 3 \
    --dropout 0.1 \
    --hidden-size 256 \
    --form-encoder abs-sines \
    --save-dir results/public_fast_filter/split
```

## 3. Train MIST-CF

```bash
CUDA_VISIBLE_DEVICES=0 python -m metabo_depthcharge.mist_cf.mist_cf_score.train \
    --gpu \
    --dataset-name canopus_train \
    --split-file data/canopus_train/splits/split_1.tsv \
    --decoy-label data/canopus_train/decoy_labels/decoy_label_COMMON.tsv \
    --subform-dir data/canopus_train/subformulae/formulae_spec_decoy_label_COMMON \
    --seed 1 \
    --num-workers 8 \
    --batch-size 4 \
    --max-decoy 32 \
    --max-epochs 200 \
    --learning-rate 0.00045 \
    --lr-decay-frac 0.88 \
    --weight-decay 0 \
    --max-subpeak 20 \
    --layers 2 \
    --dropout 0.1 \
    --hidden-size 128 \
    --form-encoder abs-sines \
    --no-cls-mass-diff \
    --save-dir results/public_mist_cf/split_1
```

## 4. Predict

Score an MGF file with a trained checkpoint (the fast filter pre-selects
`--fast-num` candidates per spectrum):

```bash
python -m metabo_depthcharge.mist_cf.mist_cf_score.predict_mgf \
    --id-key FEATURE_ID \
    --num-workers 0 \
    --batch-size 8 \
    --save-dir quickstart/mist_cf_out/ \
    --mgf-file data/demo_specs.mgf \
    --checkpoint-pth quickstart/models/mist_cf_best.ckpt \
    --fast-model quickstart/models/fast_filter_best.ckpt \
    --fast-num 256 \
    --instrument-override "Orbitrap (LCMS)" \
    --decomp-ppm 5 \
    --decomp-filter RDBE
    # --gpu
```

Use `predict` instead of `predict_mgf` to score from a labels file rather than
an MGF.

Every script accepts `--help` for the full argument list.
