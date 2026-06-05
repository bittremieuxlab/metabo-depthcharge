"""Internal shared helpers for dataset modules."""

import tempfile
from contextlib import contextmanager
from os import PathLike

from datasets import Value, disable_progress_bars, enable_progress_bars


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
