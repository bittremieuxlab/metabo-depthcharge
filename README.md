# metabo-depthcharge

Shared metabolomics library for depthcharge-based MS/MS retrieval models.

## Modules

Layout inspired by [depthcharge](https://github.com/wfondrie/depthcharge) — organized by *role*, not by data type. Role folders are named `spec/` and `chem/` (rather than `spectra/`, `molecules/`) so they don't collide path-wise with the `datasets/spectra.py` / `datasets/molecules.py` wrappers. All ML deps (torch, lightning, depthcharge-ms, transformers) are required runtime dependencies; only `dev` is an optional extra.

| Path | Contents |
|---|---|
| `spec/` | Spectrum object, preprocessing primitives, parsers |
| `chem/` | `Molecule` wrapper, SMILES standardization, molecule-to-vector representations (fingerprints + neural embeddings), similarity metrics (Tanimoto, cosine, MCES) |
| `tokenizers/` | m/z bucketing, SMILES, subformula tokenizers |
| `datasets/` | HF Datasets-backed loaders for spectra and molecules |
| `encoders/` | Neural encoders (spectrum, molecule, metadata, subformula) |
| `nn/` | Shared model machinery: primitives, losses, retrievers |

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
