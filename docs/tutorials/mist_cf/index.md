# MIST-CF Tutorials

`metabo-depthcharge` ships with a vendored [`mist-cf`](https://github.com/samgoldman97/mist-cf) implementation where slight changes have been made to processing scripts and default settings.

Changes of note:

- An expanded SIRIUS formula decomp budget to allow recovering more formulae
- Support for negative and dimer adducts.
- Faster peak subformulae assignment, which makes preprocessing/prediction scripts more efficient to run.
- Dropped the SIRIUS dependency with a python implementation of the mass decomposition algorithm.

Throughout, credits go to the original authors of [`mist-cf`](https://github.com/samgoldman97/mist-cf).
The tutorials below are merely meant to illustrate when and how to retrain `mist-cf` using our vendored version.

```{toctree}
:maxdepth: 1

fast_filter
retraining_mist_cf
using_mist_cf
```
