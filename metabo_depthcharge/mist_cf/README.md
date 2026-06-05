## MIST-CF retraining modules

Adapted from MIST-CF (Goldman et al., 2023)

GitHub link: https://github.com/samgoldman97/mist-cf

Licensed under MIT License - see LICENSE in this directory

### Retrain on own data:


1. Prepare a labels.tsv (columns: spec, formula, ionization, instrument, dataset) and .ms spectrum files in spec_files/
2. Run the preprocessing scripts in order (01→02→03→04), pointing --data-dir to your data
3. Train with python -m spectrawl.mist_cf.mist_cf_score.train --data-dir <your-data> ...
4. Predict with python -m spectrawl.mist_cf.mist_cf_score.predict --checkpoint-pth <ckpt> ...

Prepare MassSpecGym:
```bash
python -m spectrawl.mist_cf.preprocessing.01_prepare_msgym \
    --msgym-tsv data/msmsdbs/MassSpecGym.tsv \
    --out-dir data/mistcf/msgym \
    --create-split
```

Prepare negative candidate formulae with SIRIUS (note: $SIRIUS_PATH needs to be set and needs to be logged in. Code adapted to and tested on SIRIUS v6.3.3). Also, download the fast_filter_best.ckpt model from MIST-CF repo.
```bash
python -m spectrawl.mist_cf.preprocessing.02_create_decoy_label \
    --decomp-filter COMMON \
    --label-file data/mistcf/msgym/labels.tsv \
    --data-dir data/mistcf/msgym \
    --max-decoy 256 \
    --sample-strat fast_filter \
    --fast-model data/mistcf/fast_filter_best.ckpt \
    --resample-precursor-mz \
    --num-workers 8
```

```bash
python -m spectrawl.mist_cf.preprocessing.03_create_pred_label \
    --decoy-label data/mistcf/msgym/decoy_label_COMMON.tsv \
    --split-file data/mistcf/msgym/splits/split_msgym.tsv \
    --data-dir data/mistcf/msgym \
```

python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_2_with_nist.tsv  --data-dir data/nist_canopus/


### To-do:

- Add support for negative ion mode: add entries to ION_LST, ion_to_mass, ion_to_add_vec, and ion_remap in common/chem_utils.py with the correct electron mass sign (+ELECTRON_MASS for negative mode adducts like [M-H]-).


Train fast filter:
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/mist_cf/fast_form_score/train.py \
--gpu \
--save-dir 'split' \
--seed 1 \
--num-workers 8 \
--dataset-file 'data/biomols/biomols_with_decoys.txt' \
--split-file 'data/biomols/biomols_with_decoys_split.tsv' \
--batch-size 64 \
--max-decoy 32 \
--max-epochs 200 \
--learning-rate 0.00036 \
--lr-decay-frac 0.86425 \
--weight-decay 0 \
--layers 3 \
--dropout 0.1 \
--hidden-size 256 \
--form-encoder 'abs-sines' \
--save-dir results/public_fast_filter/split
```
Train mist-cf:
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/mist_cf/mist_cf_score/train.py \
--gpu \
--save-dir 'split_1' \
--dataset-name 'canopus_train' \
--split-file 'data/canopus_train/splits/split_1.tsv' \
--decoy-label 'data/canopus_train/decoy_labels/decoy_label_COMMON.tsv' \
--subform-dir 'data/canopus_train/subformulae/formulae_spec_decoy_label_COMMON' \
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
--form-encoder 'abs-sines' \
--no-cls-mass-diff \
--save-dir results/public_mist_cf/split_1
```

Run: (use predict or predict_mgf)?
```bash
fast_filter_model="quickstart/models/fast_filter_best.ckpt"
mist_cf_model="quickstart/models/mist_cf_best.ckpt"
out_dir="quickstart/mist_cf_out/"
mgf_file="data/demo_specs.mgf"

mkdir $out_dir

python src/mist_cf/mist_cf_score/predict_mgf.py \
    --id-key FEATURE_ID \
    --num-workers 0 \
    --batch-size 8 \
    --save-dir $out_dir \
    --mgf-file $mgf_file \
    --checkpoint-pth $mist_cf_model \
    --fast-model $fast_filter_model \
    --fast-num 256 \
    --instrument-override "Orbitrap (LCMS)" \
    --decomp-ppm 5 \
    --decomp-filter "RDBE"
    # --gpu
```
