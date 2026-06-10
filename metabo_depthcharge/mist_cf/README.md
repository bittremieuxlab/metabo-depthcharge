# Vendored MIST-CF

Adapted from MIST-CF (Goldman et al., 2023) —
<https://github.com/samgoldman97/mist-cf>.
Licensed under the MIT License; see `LICENSE` in this directory.

This subpackage is vendored so the formula/subformula utilities used by
metabo-depthcharge are self-contained. Most users never import it directly —
attach subformula features via
`metabo_depthcharge.datasets.SpectrumDataset.add_subformulae` and encode them
with `metabo_depthcharge.encoders.SubformulaEncoder`.

## Documentation

- **Using subformula features:** see the *Subformulae* tutorial —
  <https://metabo-depthcharge.readthedocs.io/en/latest/tutorials/subformulae.html>
- **Retraining the MIST-CF formula-prediction model on your own data**
  (preprocessing `01→04`, training the fast filter and MIST-CF, prediction):
  see the *Retraining MIST-CF* tutorial —
  <https://metabo-depthcharge.readthedocs.io/en/latest/tutorials/retraining_mist_cf.html>

The tutorial sources live at `docs/tutorials/` in this repository.
