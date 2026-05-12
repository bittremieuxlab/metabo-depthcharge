# metabo-depthcharge

Shared metabolomics library for depthcharge-based MS/MS retrieval models.

## Modules

Layout follows [depthcharge](https://github.com/wfondrie/depthcharge) — organized by *role*, not by data type. Heavy ML deps are gated behind the optional `nn` extra.

| Path | Contents | Deps |
|---|---|---|
| `spectra/` | Spectrum object, preprocessing primitives, (later) HF dataset loaders | base |
| `molecules/` | SMILES utilities, fingerprint extractors, similarity metrics (Tanimoto, cosine, MCES) | base |
| `tokenizers/` | m/z bucketing, SMILES, subformula tokenizers | base |
| `data/` | HF Datasets-backed loaders for spectra and molecules | base |
| `encoders/` | Neural encoders (spectrum, molecule, metadata, subformula) | `nn` |
| `nn/` | Shared model machinery: primitives, losses, retrievers | `nn` |

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
