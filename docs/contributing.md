# Contributing

We recommend [uv](https://docs.astral.sh/uv/) for local development.

## Setup

```bash
git clone https://github.com/bittremieuxlab/metabo-depthcharge.git
cd metabo-depthcharge
uv sync --extra dev
uv run pre-commit install
```

## Tests

```bash
uv run pytest tests/                                       # full suite
uv run pytest tests/spec/                                   # one subpackage
uv run pytest tests/spec/test_preprocessing.py::test_name -v  # single test
```

A failing test blocks `git commit` via the pre-commit hook.

## Lint and format

```bash
uv run ruff check . && uv run ruff format --check .        # check
uv run ruff check --fix . && uv run ruff format .          # fix
```

## Building the docs

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.
