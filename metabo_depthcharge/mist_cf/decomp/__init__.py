# mass_decomp.py is a pure-Python/NumPy reimplementation of SIRIUS's `decomp`
# command (no JVM/SIRIUS login required), validated to exact candidate-set
# parity against a real SIRIUS 6.3.3 binary (see mass_decomp.py's module
# docstring) and ~3-9x faster in every regime tested. It exposes the same
# names/signatures as sirius_decomp (EL_STR_DEFAULT, get_rounded_masses,
# iter_sirius_batches, run_sirius), so this is the only line that needed to
# change for every existing `decomp.X` call site to stop depending on
# SIRIUS. sirius_decomp itself is untouched and still importable directly
# (`from metabo_depthcharge.mist_cf.decomp import sirius_decomp`) for anyone
# who wants the real SIRIUS explicitly, or as a reference to validate against.
from .mass_decomp import *
