"""Internal shared helpers for dataset modules."""

import os
import tempfile
import uuid
from contextlib import contextmanager
from os import PathLike

from datasets import Dataset, Value, disable_progress_bars, enable_progress_bars


def hf_cache_file(cache_dir: str | None, in_memory: bool) -> str | None:
    """Per-call Arrow ``cache_file_name`` for a HF ``map``/``filter``.

    Returns ``None`` for in-memory builds (or when no cache dir is given),
    otherwise a unique ``<cache_dir>/<uuid>.arrow`` path. Centralizes the
    naming convention so every dataset build site stays consistent.
    """
    if in_memory or cache_dir is None:
        return None
    return os.path.join(cache_dir, uuid.uuid4().hex + ".arrow")


def hf_persist(ds: Dataset, save_to: str | PathLike | None) -> Dataset:
    """Persist ``ds`` to ``save_to`` and return the disk-backed reload.

    Passthrough when ``save_to`` is ``None``. The save-then-reload makes the
    returned dataset memory-mapped from the final on-disk shards, matching
    what :meth:`from_disk` would load.
    """
    if save_to is None:
        return ds
    ds.save_to_disk(str(save_to))
    return Dataset.load_from_disk(str(save_to))


@contextmanager
def hf_tempcache(dir: str | PathLike | None = None):
    """Route the HF builder cache to a tempdir that's wiped on exit.

    Keeps HF's content-addressable cache from accumulating under
    ``~/.cache/huggingface``. ``dir`` overrides the parent (defaults to
    :func:`tempfile.gettempdir`, honoring ``$TMPDIR``); set it on HPC
    systems where ``$TMPDIR`` is unset, points at a small ``/tmp``, or
    resolves to a ``tmpfs``.
    """
    with tempfile.TemporaryDirectory(prefix="metabo_hf_", dir=dir) as tmp:
        yield tmp


@contextmanager
def hf_silent():
    """Silence HF's progress bars within this scope.

    Use around a single HF call (e.g. :meth:`datasets.Dataset.from_generator`)
    when a caller-side ``tqdm`` is already showing the same progress.
    """
    disable_progress_bars()
    try:
        yield
    finally:
        enable_progress_bars()


def infer_value(v) -> Value:
    """Infer an HF :class:`Value` dtype from a Python scalar."""
    if isinstance(v, bool):
        return Value("bool")
    if isinstance(v, int):
        return Value("int64")
    if isinstance(v, float):
        return Value("float64")
    return Value("string")
