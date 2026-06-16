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

from metabo_depthcharge.spec.adducts import encode_adduct


#: Instrument type strings carrying a categorical embedding index.
#: Checkpoint-stable: append-only, never reorder or remove without retraining.
#: Each type's index is its 1-based position below; index 0 is reserved for
#: unknown/missing:
#:
#: * ``0`` — unknown / missing
#: * ``1`` — ``orbitrap``
#: * ``2`` — ``qtof`` or `q-tof`
#: * ``3`` — ``iontrap``
INSTRUMENT_TYPES = [
    "orbitrap",
    "qtof",
    "iontrap",
]
_INSTRUMENT_TO_IDX = {t: i + 1 for i, t in enumerate(INSTRUMENT_TYPES)}
N_INSTRUMENTS = len(INSTRUMENT_TYPES) + 1  # +1 for unknown at index 0
#: Alternate spellings for instrument strings that are automatically resolved.
_INSTRUMENT_ALIASES = {"q-tof": "qtof"}


def encode_instrument(instrument_str: str) -> int:
    """Encode an instrument type string to a vocabulary index.

    Returns 0 for unknown/missing instruments. Case-insensitive matching.

    See Also
    --------
    INSTRUMENT_TYPES : The instrument-to-index vocabulary this looks up,
        including the full index assignment.
    METADATA_PARSERS : Registry that wires this in as the row-wise parser for
        the ``instrument_type`` metadata field, consumed by ``SpectrumDataset``
        at build time.
    """
    if not instrument_str or instrument_str in ("nan", "None", ""):
        return 0
    key = instrument_str.strip().lower()
    return _INSTRUMENT_TO_IDX.get(_INSTRUMENT_ALIASES.get(key, key), 0)


def encode_collision_energy(raw) -> float:
    """Encode a collision energy value to a float. Divides by 100 for numerical stability.

    Handles:
    - Numeric values (int/float)
    - String representations of numbers
    - Missing/NaN values → 0.0
    - Stepped CE strings like "10 20 40" or "10, 20, 40" → mean

    See Also
    --------
    ~metabo_depthcharge.spec.preprocessing.CollapseSteppedCE : Upstream
        spectrum processor that merges stepped CE spread across multiple
        metadata keys (``COLLISION_ENERGY_1``, ``COLLISION_ENERGY_2``, ...)
        into a single ``collision_energy`` field before this parser runs.
    METADATA_PARSERS : Registry that wires this in as the row-wise parser for
        the ``collision_energy`` metadata field, consumed by ``SpectrumDataset``
        at build time.
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


#: The canonical, ordered set of NN-encoded metadata field names. Exists as an
#: explicit allow-list so :class:`~metabo_depthcharge.datasets.SpectrumDataset`
#: can validate user-requested metadata columns, and so
#: :class:`~metabo_depthcharge.encoders.MetadataEncoder` knows which fields it
#: may read.
METADATA_FIELDS = ["adduct", "collision_energy", "instrument_type"]

#: Dictionary mapping string to tuple.
#: e.g., {"adduct": (encode_adduct, Value("int64"))}
#: i.e. (row-wise parser, HF dtype)
#: Delineates which metadata fields are supported for neural network encoding:
#: consumed by :class:`~metabo_depthcharge.datasets.SpectrumDataset` (build-time encoding
#: into typed Arrow columns) and by
#: :class:`~metabo_depthcharge.encoders.MetadataEncoder` (the only fields it
#: ever reads).
METADATA_PARSERS = {
    "adduct": (encode_adduct, Value("int64")),
    "collision_energy": (encode_collision_energy, Value("float32")),
    "instrument_type": (encode_instrument, Value("int64")),
}
assert list(METADATA_PARSERS) == METADATA_FIELDS
