# Subformulae

This is the **supported, high-level path** for using MIST-style subformula
features. You attach per-peak subformulae to a
{class}`~metabo_depthcharge.datasets.SpectrumDataset` once (baked into Arrow),
then consume them in a model with
{class}`~metabo_depthcharge.encoders.SubformulaEncoder`.

Under the hood this uses formula utilities vendored from
[MIST-CF](https://github.com/samgoldman97/mist-cf) (`metabo_depthcharge.mist_cf`).
You do **not** need to touch `mist_cf` directly for this workflow — if you instead
want to *retrain* the MIST-CF formula-prediction model on your own data, see
{doc}`retraining_mist_cf`.

## 1. Attach subformulae to a dataset

{meth}`~metabo_depthcharge.datasets.SpectrumDataset.add_subformulae` reads a
molecular formula and an adduct **per spectrum** from an external CSV/TSV
(one row per spectrum, in dataset order), assigns a subformula to each peak
within a ppm tolerance, and writes three new columns.

```python
from metabo_depthcharge.datasets import SpectrumDataset

ds = SpectrumDataset.from_mgf("spectra.mgf", save_to="spectra/")

# formulae.tsv has one row per spectrum (same order as the dataset),
# with at least a `formula` and an `adduct` column.
ds = ds.add_subformulae(
    "gt",                       # name -> column prefix
    source="formulae.tsv",
    source_formula_col="formula",
    source_adduct_col="adduct",
    ppm=10.0,
    num_workers=4,
    save_to="spectra/",         # persist in place
)
```

This adds, for the set named `gt`:

- `gt_subformula_vec` — flat `int16` sequence of length `N_peaks * ELEMENT_DIM`
  (a bag-of-atoms per peak).
- `gt_parent_formula_vec` — `int16` sequence of length `ELEMENT_DIM`
  (the parent molecular formula).
- `gt_adduct` — the raw adduct string used for the assignment.

Peaks that fall outside the ppm tolerance (or whose formula/adduct is missing
or invalid) get zero vectors.

You can attach **multiple named sets** by chaining calls — e.g. a ground-truth
set and a predicted set:

```python
ds = ds.add_subformulae("gt", source="formulae.tsv")
ds = ds.add_subformulae("pred", source="predicted_formulae.tsv")
```

## 2. Select a set at load time

When you reload a dataset that carries one or more subformula sets, pick which
one is active with `subformulae_name`. Two optional iteration-time behaviours
build on it:

```python
ds = SpectrumDataset.from_disk(
    "spectra/",
    subformulae_name="gt",
    drop_peaks_without_subformula=True,  # drop unmatched peaks per __getitem__
    adduct_from_subformula=True,         # inject gt_adduct as encoder metadata
)
```

- `drop_peaks_without_subformula` — `__getitem__` returns only the peaks that
  received a subformula (filtering `mz`, `intensity`, and the `*_subformula_vec`
  columns together).
- `adduct_from_subformula` — reads the raw adduct from `{name}_adduct` and
  encodes it as metadata for the encoder.

Both require `subformulae_name` to be set.

## 3. Batch it

{meth}`~metabo_depthcharge.datasets.SpectrumDataset.collate` detects the
subformula columns and pads them into `batch["subformulae"]`. When exactly one
set is active (i.e. `subformulae_name` was set at load time), the dict is
flattened to:

```python
batch["subformulae"] = {
    "form_vec":        ...,  # (B, L, ELEMENT_DIM) int
    "parent_form_vec": ...,  # (B, ELEMENT_DIM) int
}
```

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=32, collate_fn=SpectrumDataset.collate)
batch = next(iter(loader))
```

## 4. Encode it

Pass a {class}`~metabo_depthcharge.encoders.SubformulaEncoder` to
{class}`~metabo_depthcharge.encoders.SpectrumEncoder`. The subformula embedding
is added to the per-peak embeddings before the transformer.

```python
from metabo_depthcharge.encoders import SpectrumEncoder, SubformulaEncoder

d_model = 512
encoder = SpectrumEncoder(
    d_model=d_model,
    subformula_encoder=SubformulaEncoder(d_model, form_embedder="abs-sines"),
)

emb = encoder(
    mz=batch["mz"],
    intensity=batch["intensity"],
    precursor_mz=batch["precursor_mz"],
    subformulae=batch["subformulae"],
)
```

`form_embedder` selects the element-count featurizer (`"abs-sines"`,
`"fourier"`, `"rbf"`, `"learnt"`, ...); see
{class}`~metabo_depthcharge.encoders.SubformulaEncoder` for the full list.

## Building vectors directly

If you need the bag-of-atoms vectors outside the dataset path, the primitives
live in {mod}`metabo_depthcharge.spec.subformulae`:

```python
from metabo_depthcharge.spec.subformulae import (
    formula_to_dense,        # formula string -> (ELEMENT_DIM,) vector
    assign_peak_subformulae, # (mz, formula, adduct, ppm) -> per-peak vectors
)
```
