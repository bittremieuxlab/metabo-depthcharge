# Installation

`metabo-depthcharge` requires Python ≥ 3.10.

The package is not yet on PyPI; install directly from GitHub.

::::{tab-set}

:::{tab-item} pip
```bash
pip install git+https://github.com/bittremieuxlab/metabo-depthcharge.git
```
:::

:::{tab-item} conda + pip
Conda is used only to manage the Python environment.
The package itself is installed with pip.

```bash
conda create -n mdc python=3.11
conda activate mdc
pip install git+https://github.com/bittremieuxlab/metabo-depthcharge.git
```
:::

:::{tab-item} uv

```bash
git clone https://github.com/bittremieuxlab/metabo-depthcharge.git
cd metabo-depthcharge
uv sync
```

:::

::::

## Optional extras

There is a single `dev` extra that bundles everything a contributor needs:
tests (`pytest`), lint (`ruff`, `pre-commit`), and the docs toolchain
(`sphinx`, `myst-parser`, ...). Install it from a clone with
`pip install -e ".[dev]"` or `uv sync --extra dev`.
