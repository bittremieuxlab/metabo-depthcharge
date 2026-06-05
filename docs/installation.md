# Installation

`metabo-depthcharge` requires Python ≥ 3.10. All runtime ML dependencies
(`torch`, `lightning`, `depthcharge-ms`, `transformers`) are installed by
default — there is no `nn` extra to opt into.

The package is not yet on PyPI; install directly from GitHub.

::::{tab-set}

:::{tab-item} pip
```bash
pip install git+https://github.com/bittremieuxlab/metabo-depthcharge.git
```
:::

:::{tab-item} conda + pip
Conda is used only to manage the Python environment; the package itself is
installed with pip because one of its dependencies (`depthcharge-ms`) is not
available on conda-forge.

```bash
conda create -n mdc python=3.11
conda activate mdc
pip install git+https://github.com/bittremieuxlab/metabo-depthcharge.git
```
:::

:::{tab-item} uv (contributors)
For local development with a checked-out clone:

```bash
git clone https://github.com/bittremieuxlab/metabo-depthcharge.git
cd metabo-depthcharge
uv sync --extra dev
```

See [Contributing](contributing.md) for the full dev workflow.
:::

::::

## Optional extras

There is a single `dev` extra that bundles everything a contributor needs:
tests (`pytest`), lint (`ruff`, `pre-commit`), and the docs toolchain
(`sphinx`, `furo`, `myst-parser`, …). Install it from a clone with
`pip install -e ".[dev]"` or `uv sync --extra dev`.
