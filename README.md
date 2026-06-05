# metabo-depthcharge

Library for depthcharge-based metabolomics models.

## Installation

Not yet on PyPI — install from GitHub.

```bash
# pip
pip install git+https://github.com/bittremieuxlab/metabo-depthcharge.git

# conda (env only; package itself via pip)
conda create -n mdc python=3.11 && conda activate mdc
pip install git+https://github.com/bittremieuxlab/metabo-depthcharge.git
```

See the [documentation](https://metabo-depthcharge.readthedocs.io/) for the
full user guide.

## Development setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

### Run tests

```bash
uv run pytest tests/ --ignore=tests/characterization
```

### Lint and format

```bash
# Check only
uv run ruff check .
uv run ruff format --check .

# Fix in place
uv run ruff check --fix .
uv run ruff format .
```

### Pre-commit hooks

```bash
uv run pre-commit install
```

After that, ruff and pytest run automatically on every commit.

### Characterization tests

Characterization tests compare behaviour against the original upstream implementations.
Run locally only, before modifying a module:

```bash
uv run pytest tests/characterization/ \
    --spectrawl-path /path/to/spectrawl \
    --metabo-src-path /path/to/metabo-depthcharge
```
