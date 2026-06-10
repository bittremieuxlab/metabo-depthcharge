# metabo-depthcharge

Library for depthcharge-based metabolomics models. It is organized into four
submodules:

- **{doc}`spec <api/spec>`** — per-spectrum primitives: preprocessing,
  adduct/metadata encoding, subformula assignment.
- **{doc}`chem <api/chem>`** — per-molecule primitives: representation
  generators (fingerprints, embeddings) and similarity measures.
- **{doc}`datasets <api/datasets>`** — HF Datasets-backed PyTorch datasets for
  spectra and molecules.
- **{doc}`encoders <api/encoders>`** — neural encoders for spectra and
  molecules.

```{toctree}
:maxdepth: 1

getting_started
tutorials/index
api/index
```
