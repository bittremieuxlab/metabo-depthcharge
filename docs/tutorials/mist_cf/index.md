# MIST-CF Tutorials

`metabo-deptcharge` ships with a vendored [`mist-cf`](https://github.com/samgoldman97/mist-cf) implementation where slight changes have been made to processing scripts and default settings.

Changes of note:

- An expanded SIRIUS formula decomp budget to allow recovering more formulae
- Support for negative and dimer adducts.
- Faster peak subformulae assignment, which makes preprocessing/prediction scripts more efficient to run.

Throughout, credits go to the original authors of [`mist-cf`](https://github.com/samgoldman97/mist-cf).
The tutorials below are merely meant to illustrate when and how to retrain `mist-cf` using our vendored version.

```{warning}
`MIST-CF` relies on [`SIRIUS`](https://v6.docs.sirius-ms.io/) for decomposing neutral masses into candidate formulae.
This step is **necessary** for both preprocessing for training, as well as for inference.
Throughout our work, all testing with `MIST-CF` was performed using `SIRIUS 6.3.3`.

The code-base requires that an environment variable `$SIRIUS_PATH` exists pointing to the sirius binary executable.
This can be set through e.g.: `echo 'export SIRIUS_PATH=${your_sirius_path}' >> ~/.bashrc`.

In addition, as of recent SIRIUS versions, users are required to register and login on the command-line.
Run, e.g. `$SIRIUS_PATH login --email=your@email.com --pwd`.
```

```{toctree}
:maxdepth: 1

fast_filter
retraining_mist_cf
using_mist_cf
```
