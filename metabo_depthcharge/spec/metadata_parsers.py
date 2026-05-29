"""
Metadata encoding utilities for spectrum acquisition conditions.

Adduct handling (vocab, encoder, mass deltas) lives in
:mod:`metabo_depthcharge.spec.adducts`. This module owns the rest:
- Instrument type (categorical index)
- Collision energy (scalar float)
- The :data:`METADATA_PARSERS` registry consumed by ``SpectrumDataset``.
"""

import numpy as np
from datasets import Value

# Used internally by METADATA_PARSERS and encode_metadata_arrays. Adduct
# names should be imported from spec.adducts directly by external callers.
from metabo_depthcharge.spec.adducts import encode_adduct


# Instrument type vocabulary: index 0 is reserved for unknown.
INSTRUMENT_TYPES = [
    "orbitrap",
    "qtof",
    "iontrap",
]
_INSTRUMENT_TO_IDX = {t: i + 1 for i, t in enumerate(INSTRUMENT_TYPES)}
N_INSTRUMENTS = len(INSTRUMENT_TYPES) + 1  # +1 for unknown at index 0

# All supported metadata field names.
METADATA_FIELDS = ["adduct", "collision_energy", "instrument_type"]


def encode_instrument(instrument_str: str) -> int:
    """Encode an instrument type string to a vocabulary index.

    Returns 0 for unknown/missing instruments.
    Case-insensitive matching.
    """
    if not instrument_str or instrument_str in ("nan", "None", ""):
        return 0
    return _INSTRUMENT_TO_IDX.get(instrument_str.strip().lower(), 0)


def parse_collision_energy(raw) -> float:
    """Parse a collision energy value to a float. Divides by 100 for numerical stability.

    Handles:
    - Numeric values (int/float)
    - String representations of numbers
    - Missing/NaN values → 0.0
    - Stepped CE strings like "10 20 40" or "10, 20, 40" → mean
    """
    if raw is None:
        return 0.0

    if isinstance(raw, int | float | np.integer | np.floating):
        if np.isnan(raw):
            return 0.0
        return float(raw) / 100

    # String handling
    s = str(raw).strip()
    if not s or s in ("nan", "None", ""):
        return 0.0

    # Try direct float parse first (most common case)
    try:
        val = float(s)
        return 0.0 if np.isnan(val) else val / 100
    except ValueError:
        pass

    # Handle stepped CE: split on common delimiters
    for sep in [",", ";", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            vals = []
            for p in parts:
                try:
                    v = float(p)
                    if not np.isnan(v):
                        vals.append(v)
                except ValueError:
                    continue
            if vals:
                return float(np.mean(vals)) / 100

    return 0.0


class CollapseSteppedCE:
    """Spectrum processor: collapse suffixed CE keys into ``collision_energy``.

    Some MGFs have stepped collision energy as ``COLLISION_ENERGY_1``,
    ``COLLISION_ENERGY_2``. Or ``NORMALIZED_COLLISION_ENERGY_N`` variants.

    Parameters
    ----------
    source : {"normalized", "absolute"}, default "normalized"
        Which family to collapse. ``"normalized"`` matches
        ``normalized_collision_energy_N``; ``"absolute"`` matches
        ``collision_energy_N``. Pyteomics lowercases MGF keys.
    """

    def __init__(self, source: str = "normalized"):
        if source not in ("normalized", "absolute"):
            raise ValueError(
                f"source must be 'normalized' or 'absolute', got {source!r}"
            )
        self._prefix = (
            "normalized_collision_energy_"
            if source == "normalized"
            else "collision_energy_"
        )

    def __call__(self, spectrum):
        vals = []
        for k, v in spectrum.metadata.items():
            if not k.lower().startswith(self._prefix):
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isnan(f):
                vals.append(f)
        if vals:
            spectrum.metadata["collision_energy"] = sum(vals) / len(vals)
        return spectrum


# Per-field (row-wise parser, HF dtype) for the NN-encoded metadata fields.
# Single source of truth: consumed by SpectrumDataset (build-time encoding into
# typed Arrow columns) and by MetadataEncoder (the only fields it ever reads).
METADATA_PARSERS = {
    "adduct": (encode_adduct, Value("int64")),
    "collision_energy": (parse_collision_energy, Value("float32")),
    "instrument_type": (encode_instrument, Value("int64")),
}
assert list(METADATA_PARSERS) == METADATA_FIELDS


def encode_metadata_arrays(
    raw_values: np.ndarray,
    field: str,
) -> np.ndarray:
    """Batch-encode a raw metadata column into a numeric array.

    Args:
        raw_values: 1-D array of raw values (bytes or numeric) from HDF5.
        field: One of METADATA_FIELDS.

    Returns:
        Encoded numpy array: int64 for categorical fields, float32 for CE.
    """
    n = len(raw_values)

    if field == "adduct":
        out = np.zeros(n, dtype=np.int64)
        for i, v in enumerate(raw_values):
            s = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            out[i] = encode_adduct(s)
        return out

    elif field == "instrument_type":
        out = np.zeros(n, dtype=np.int64)
        for i, v in enumerate(raw_values):
            s = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            out[i] = encode_instrument(s)
        return out

    elif field == "collision_energy":
        out = np.zeros(n, dtype=np.float32)
        for i, v in enumerate(raw_values):
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            out[i] = parse_collision_energy(v)
        return out

    else:
        raise ValueError(
            f"Unknown metadata field: {field}. Must be one of {METADATA_FIELDS}"
        )
