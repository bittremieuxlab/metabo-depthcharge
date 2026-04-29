# metabo-depthcharge

Shared metabolomics library for depthcharge-based MS/MS retrieval models.

## Modules

| Module | Contents |
|---|---|
| `spectrum.py` | Spectrum object, normalizer, trimmer, peak filter, preprocessor pipeline |
| `molecules.py` | SMILES → fingerprint converters |
| `similarities.py` | Tanimoto, cosine, MCES distance metrics |
| `metadata.py` | Spectrum acquisition metadata encoding (adduct, CE, instrument) |
| `encoders.py` | Spectrum transformer encoders (depthcharge-based) |
| `formulae.py` | Chemical formula / subformula embeddings |
| `losses.py` | Loss functions and composer |
| `data.py` | HDF5-backed dataset and Lightning datamodule |
| `retrieval.py` | Retrieval model (Lightning training wrapper) |

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
    --metabo-src-path /path/to/metabo-depthcharge/src
```
