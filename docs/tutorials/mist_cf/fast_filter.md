# Fast filter and when to retrain

## Whether to retrain

MIST-CF's `FastFFN` is a small MLP that scores how chemically plausible a molecular formula is.
It's used to prune the SIRIUS-decomposition candidates for a given mass down to a shortlist.
Broadly speaking, two factors decide whether or not `FastFFN` should be re-trained for a user's purpose:
- The alphabet (element budget) cannot represent the compounds in your dataset.
- The molecular formula distribution `FastFFN` saw as "positives" differs drastically from the one in your dataset.

Both points down below:

**Regarding alphabet/element budget:**
The element budget lives in the SIRIUS decomposition step, not in `FastFFN` itself: decomposition can only ever propose formulae that fit the budget, so any true formula outside it can never be recovered: a hard ceiling that no amount of filter training can lift.
The original MIST-CF ships with [`"C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]"`](https://github.com/samgoldman97/mist-cf/blob/71c820b43825e0b0a58680a3690f29a7352290f4/src/mist_cf/decomp/sirius_decomp.py#L22), meaning it will only ever consider (i.e., be able to recover) formulae with a max of one `F` atom, etc.
We found that this specific RegEx covers 94.6% and 87.9% of the spectra in MassSpecGym and Enveda-180 respectively, imposing a hard upper ceiling on the performance of the overall `mist-cf` pipeline.
In `metabo-depthcharge`, our default is `"C[0-]N[0-]O[0-]H[0-]S[0-3]P[0-1]I[0-1]Cl[0-3]F[0-6]Br[0-1]"`, delivering a 99.3% and 100% coverage of MassSpecGym and Enveda-180 spectra, respectively, without completely blowing up the proportions of the formula search space.

Users can check the coverage for their own data via the code snippet below.
Point this at an MGF (each spectrum needs a `FORMULA=` field) and an alphabet string:

```python
import re
from pyteomics import mgf

mgf_path = "spectra.mgf"
alphabet = "C[0-]N[0-]O[0-]H[0-]S[0-3]P[0-1]I[0-1]Cl[0-3]F[0-6]Br[0-1]"

budget = {el: int(hi) if hi else float("inf")
          for el, _, hi in re.findall(r"([A-Z][a-z]*)\[(\d+)-(\d*)\]", alphabet)}
fits = lambda f: all(el in budget and int(n or 1) <= budget[el]
                     for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", f))
forms = [s["params"]["formula"] for s in mgf.read(mgf_path, use_index=False) if "formula" in s["params"]]
print(f"{sum(map(fits, forms)) / len(forms):.1%} covered ({len(forms)} formulae)")
```

If this coverage is not sufficient for your training data, we recommend re-training `FastFFN`.

**Regarding molecular formula distribution:**
The `FastFFN` is trained by decomposing masses into molecular formulae, employing a user-defined input list of molecular formulae as ones that are "chemically plausible".
In `metabo-depthcharge`, we re-trained the model using [MassSpecGym_retrieval_molecules_1M.tsv](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/blob/main/data/molecules/candidate_pools/MassSpecGym_retrieval_molecules_1M.tsv).
This database contain >1M molecules originating from biological databases. Users should check whether the molecular formulae in their data roughly occupy the same manifold as those in this set.

```{note}
During training, it is possible to exclude certain masses from `FastFFN` training to make sure no data leakage happens.
In our shipped checkpoint, this was performed using MassSpecGym formulae.
Maximum scientific rigor would demand that this exclusion is satisfied for each dataset the model is ever used on.
In practice the exposure is limited: exclusion is done by mass and `FastFFN` is a pure function of the formula, so holding out a dataset's masses also removes its formulae (and their same-mass decoys) from the positives.
What remains is the model's general prior over biomolecule-like formulae: the intended generalization, not leakage.
As such, the leakage issue is probably fine for everyday use.
```

## Retraining fast-filter

These mostly just follow the files in the original mist-cf repository.

### 1. Extract formulae

Read a molecule source and emit a filtered, deduplicated formula list.

```bash
# change to desired molecular formula distribution
INPUT="./MassSpecGym_retrieval_molecules_1M.tsv"

python -m metabo_depthcharge.mist_cf.preprocessing.biomols.01_extract_formulae \
    --input "$INPUT" \
    --formula-col formula \
    --max-mass 1500 \
    --out fast_filter/formulae.txt
```

If your source has SMILES rather than formulae, pass `--smiles-col <name>` instead.

### 2. Create decoys

For each unique mass, decompose with SIRIUS to get plausible-but-wrong formulae.
These are the negatives the filter learns to reject.

```bash
python -m metabo_depthcharge.mist_cf.preprocessing.biomols.02_create_formulae_decoys \
    --formulae-list fast_filter/formulae.txt \
    --num-decoys 256 \
    --decomp-filter COMMON \
    --ppm 10 \
    --elements "C[0-]N[0-]O[0-]H[0-]S[0-3]P[0-1]I[0-1]Cl[0-3]F[0-6]Br[0-1]" \
    --cores 32 \
    --max-batch 20_000 \
    --out fast_filter/formulae_with_decoys.tsv
```

```{warning}
`--elements` must match the alphabet your downstream decomposition uses, see above.
```

```{note}
Step 2 runs SIRIUS mass decomposition and is the most expensive step: hours to a day on many cores.
It needs SIRIUS installed with `$SIRIUS_PATH` set and a valid login. See [here](https://github.com/samgoldman97/mist-cf#sirius--).
```

### 3. Split

Train/val/test split over masses.
`--exclude-labels` is your evaluation set: any mass appearing there is pinned to the test fold, so the filter never trains on it.

```bash
# Change to your own file with formula column
EXCLUDE_MASSES_FROM="./MassSpecGym.tsv"

python -m metabo_depthcharge.mist_cf.preprocessing.biomols.03_create_formulae_split \
    --formula-decoy-file fast_filter/formulae_with_decoys.tsv \
    --exclude-labels "$EXCLUDE_MASSES_FROM" \
    --exclude-formula-col formula \
    --train-frac 0.8 --val-frac 0.1 \
    --out fast_filter/formulae_split.tsv
```

Exclusion is by *mass*, not exact formula: each formula in `--exclude-labels` is converted to its rounded neutral mass, and every decoy-set mass that collides with one is sent to test.

### 4. Train

```bash
python -m metabo_depthcharge.mist_cf.fast_form_score.train \
    --gpu \
    --dataset-file fast_filter/formulae_with_decoys.tsv \
    --split-file fast_filter/formulae_split.tsv \
    --max-decoy 32 \
    --batch-size 64 \
    --max-epochs 200 \
    --learning-rate 0.00036 \
    --lr-decay-frac 0.86425 \
    --weight-decay 0 \
    --layers 3 \
    --dropout 0.1 \
    --hidden-size 256 \
    --patience 5 \
    --form-encoder abs-sines \
    --save-dir fast_filter/results/run_1
```

The checkpoint lands at `<save-dir>/version_0/best.ckpt`.
