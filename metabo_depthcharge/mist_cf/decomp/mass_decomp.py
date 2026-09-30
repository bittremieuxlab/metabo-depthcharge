"""mass_decomp.py

Pure-Python/NumPy mass decomposition: given a mass window and an element
alphabet with per-element count bounds, enumerate every molecular formula
whose mass falls in the window. This is the same "money changing problem"
that sirius_decomp.py solves by shelling out to the SIRIUS `decomp` CLI
(JVM + SIRIUS login required) -- this module needs neither.

ENGINE (decompose_array / decompose_batch_array): branch-and-bound DFS over
element counts, heaviest element first, pruned at each level by the
min/max mass still achievable from the remaining (lighter) elements.
Iterative (an explicit per-level stack, not real recursion -- the same
trick SIRIUS's own Java uses) rather than a plain recursive function: ~9x
faster than a naive recursive version even single-threaded, and it's what
makes multi-threading possible at all (numba's recursive-function support
segfaults when called concurrently from multiple prange threads). Both DFS
cores are numba-jitted (a count pass sizes an exact output array, a fill
pass writes into it -- no per-candidate Python object construction during
the search). decompose_batch_array() parallelizes ACROSS masses with a
thread pool (embarrassingly parallel, each mass independent) rather than
splitting one mass's own branches -- real batches are wildly heterogeneous
in candidate count (a handful to 500K+ in the same run), so a thread
grabbing the next mass as soon as it's free load-balances automatically;
this only gets real parallelism (not just GIL-bound concurrency) because
the numba core is compiled with nogil=True.

CHEMICAL FILTER (chemical_filter_mask): replicates `sirius decomp --filter`
(ChemicalValidator's RDBE / heteroatom-ratio / C-H-ratio check) as a cheap
vectorized post-filter over already-decomposed candidates. The constants
here for "COMMON" (the only filter level actually used by any caller in
this codebase) were verified empirically against a running SIRIUS 6.3.3
binary -- see VALENCES' comment for why they differ from what
ChemicalValidator.java's source reads as.

SIRIUS-COMPATIBLE STREAMING API (iter_sirius_batches / run_sirius /
get_rounded_masses / EL_STR_DEFAULT): same names, same call signatures,
same yielded/returned shapes as sirius_decomp.py, so this module is a
drop-in replacement at every existing call site (decomp.iter_sirius_batches,
decomp.run_sirius, decomp.EL_STR_DEFAULT, decomp.get_rounded_masses) --
see decomp/__init__.py.

Validated against a real SIRIUS 6.3.3 binary: exact candidate-set parity on
1000+ real spectraverse masses (ground truth formula recovery 1000/1000,
candidate-set parity 1000/1000 including with --filter COMMON at ppm=10,
matching what 02_create_decoy_label.py actually calls); see db_search's
decomp/test_mass_decomp.py for the standalone version of this validation
suite. ~3-9x faster than SIRIUS in every regime tested (small/no-filter
queries: 10-500x; large batches at ppm=10 + COMMON filter, matching real
production calls: ~3.3x).

See: Boecker & Liptak, "The Money Changing Problem revisited", COCOON 2005.
"""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path

import numba
import numpy as np
from tqdm import tqdm

# Monoisotopic masses (Da). Matches metabo_depthcharge.mist_cf.common.chem_utils.ELEMENT_TO_MASS
# for these elements; extend as needed for other alphabets.
MASSES = {
    "C": 12.0,
    "H": 1.007825032,
    "N": 14.003074,
    "O": 15.99491462,
    "S": 31.972071,
    "P": 30.97376163,
    "I": 126.904473,
    "Cl": 34.96885268,
    "F": 18.99840322,
    "Br": 78.9183371,
}

ROUND_FACTOR = 4
EL_STR_DEFAULT = "C[0-]N[0-]O[0-]H[0-]S[0-3]P[0-1]I[0-1]Cl[0-3]F[0-6]Br[0-1]"

# Valences for the RDBE filter. sirius-libs's elements.json (read from
# source) lists Br/I as valence 7, and ChemicalValidator.java's
# getCommonThreshold() reads as rdbe_lowerbound=-0.5 -- but neither matches
# the running SIRIUS 6.3.3 binary's actual --filter COMMON output (verified
# empirically: querying SIRIUS directly and back-solving against its
# accept/reject decisions across thousands of real candidates, 0
# mismatches after correction). The values below are what SIRIUS 6.3.3
# actually enforces: standard textbook halogen valence (1) for Br/I too,
# an RDBE lower bound of 0 (not -0.5), and an inclusive upper RDBE bound.
VALENCES = {"C": 4, "H": 1, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1, "Cl": 1, "Br": 1, "I": 1}

# SIRIUS's --filter levels, as (rdbe_threshold, rdbe_lowerbound,
# heteroatom/C threshold, C/H threshold). COMMON verified empirically as
# above; STRICT/PERMISSIVE/RDBE follow the same (source-read) shape scaled
# from ChemicalValidator.java but are NOT independently verified against a
# running SIRIUS binary the way COMMON is -- treat them as best-effort.
_FILTER_PRESETS = {
    "STRICT": (40.0, 0.0, 3.0, 3.0),
    "COMMON": (50.0, 0.0, 3.0, 6.0),
    "PERMISSIVE": (60.0, -2.0, 4.0, 9.0),
    "RDBE": (np.inf, 0.0, np.inf, np.inf),
}

_EL_RE = re.compile(r"([A-Z][a-z]?)\[(\d*)-(\d*)\]")
_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

# Hill order: C, H, then remaining elements alphabetically.
_HILL_ORDER = {"C": 0, "H": 1}


def parse_alphabet(el_str, max_mass):
    """Parse a SIRIUS-style element string, e.g. "C[0-]H[0-]S[0-3]", into a
    list of (symbol, mass, lo, hi, explicit_bound) tuples sorted
    heaviest-first. An open upper bound ("[0-]") is capped by how many
    atoms could fit under max_mass (explicit_bound=False); a literal bound
    like "S[0-3]" is kept as-is (explicit_bound=True).
    """
    out = []
    for sym, lo, hi in _EL_RE.findall(el_str):
        mass = MASSES[sym]
        lo = int(lo) if lo else 0
        explicit = bool(hi)
        hi_val = int(hi) if explicit else int(max_mass // mass)
        out.append((sym, mass, lo, hi_val, explicit))
    out.sort(key=lambda t: -t[1])
    return out


def _suffix_bounds(masses, los, his):
    # suffix_{min,max}[i] = min/max mass achievable using elements i..n-1.
    # suffix_{min,max}[n] = 0 (nothing left to place).
    suffix_min = np.concatenate([np.cumsum((los * masses)[::-1])[::-1], [0.0]])
    suffix_max = np.concatenate([np.cumsum((his * masses)[::-1])[::-1], [0.0]])
    return suffix_min, suffix_max


# --- DFS core, iterative (explicit per-level stack instead of real
# recursion -- the same trick SIRIUS's own Java uses, and required here
# because numba's recursive-function support isn't safe to call
# concurrently from multiple prange threads). Starts at level i0 (elements
# before i0 are already fixed by the caller); used directly by
# decompose_array (i0=0) and by the parallel top-level split below
# (i0=1, one call per thread). ------------------------------------------

@numba.njit(cache=True, nogil=True)
def _count_iter(i0, n, lo0, hi0, masses, los, his, suffix_min, suffix_max):
    if i0 == n:
        return 1
    lo_rem = np.empty(n + 1)
    hi_rem = np.empty(n + 1)
    c = np.zeros(n, dtype=np.int64)
    c_lo = np.zeros(n, dtype=np.int64)
    c_hi = np.zeros(n, dtype=np.int64)
    lo_rem[i0], hi_rem[i0] = lo0, hi0
    m = masses[i0]
    c_lo[i0] = max(los[i0], int(np.ceil((lo0 - suffix_max[i0 + 1]) / m)))
    c_hi[i0] = min(his[i0], int(np.floor((hi0 - suffix_min[i0 + 1]) / m)))
    c[i0] = c_lo[i0] - 1
    i = i0
    total = 0
    while i >= i0:
        c[i] += 1
        if c[i] > c_hi[i]:
            i -= 1
            continue
        new_lo = lo_rem[i] - c[i] * masses[i]
        new_hi = hi_rem[i] - c[i] * masses[i]
        if i + 1 == n:
            total += 1
            continue
        lo_rem[i + 1], hi_rem[i + 1] = new_lo, new_hi
        mi1 = masses[i + 1]
        c_lo[i + 1] = max(los[i + 1], int(np.ceil((new_lo - suffix_max[i + 2]) / mi1)))
        c_hi[i + 1] = min(his[i + 1], int(np.floor((new_hi - suffix_min[i + 2]) / mi1)))
        c[i + 1] = c_lo[i + 1] - 1
        i += 1
    return total


@numba.njit(cache=True, nogil=True)
def _fill_iter(i0, n, lo0, hi0, masses, los, his, suffix_min, suffix_max, base_counts, out, row_offset):
    if i0 == n:
        out[row_offset] = base_counts
        return
    lo_rem = np.empty(n + 1)
    hi_rem = np.empty(n + 1)
    c = np.zeros(n, dtype=np.int64)
    c_lo = np.zeros(n, dtype=np.int64)
    c_hi = np.zeros(n, dtype=np.int64)
    counts = base_counts.copy()
    lo_rem[i0], hi_rem[i0] = lo0, hi0
    m = masses[i0]
    c_lo[i0] = max(los[i0], int(np.ceil((lo0 - suffix_max[i0 + 1]) / m)))
    c_hi[i0] = min(his[i0], int(np.floor((hi0 - suffix_min[i0 + 1]) / m)))
    c[i0] = c_lo[i0] - 1
    i = i0
    row = row_offset
    while i >= i0:
        c[i] += 1
        if c[i] > c_hi[i]:
            counts[i] = 0
            i -= 1
            continue
        counts[i] = c[i]
        new_lo = lo_rem[i] - c[i] * masses[i]
        new_hi = hi_rem[i] - c[i] * masses[i]
        if i + 1 == n:
            out[row] = counts
            row += 1
            continue
        lo_rem[i + 1], hi_rem[i + 1] = new_lo, new_hi
        mi1 = masses[i + 1]
        c_lo[i + 1] = max(los[i + 1], int(np.ceil((new_lo - suffix_max[i + 2]) / mi1)))
        c_hi[i + 1] = min(his[i + 1], int(np.floor((new_hi - suffix_min[i + 2]) / mi1)))
        c[i + 1] = c_lo[i + 1] - 1
        i += 1


# --- parallel top-level split: branch on element 0 across threads, each
# thread runs the iterative DFS above for elements 1..n-1 --------------

@numba.njit(parallel=True, cache=True)
def _count_top_branches(n, lo_rem, hi_rem, masses, los, his, suffix_min, suffix_max, c_lo, n_branches):
    totals = np.zeros(n_branches, dtype=np.int64)
    m0 = masses[0]
    for idx in numba.prange(n_branches):
        c = c_lo + idx
        totals[idx] = _count_iter(1, n, lo_rem - c * m0, hi_rem - c * m0, masses, los, his, suffix_min, suffix_max)
    return totals


@numba.njit(parallel=True, cache=True)
def _fill_top_branches(n, lo_rem, hi_rem, masses, los, his, suffix_min, suffix_max, c_lo, offsets, out):
    n_branches = offsets.shape[0] - 1
    m0 = masses[0]
    for idx in numba.prange(n_branches):
        c = c_lo + idx
        base_counts = np.zeros(n, dtype=np.int32)
        base_counts[0] = c
        _fill_iter(1, n, lo_rem - c * m0, hi_rem - c * m0, masses, los, his, suffix_min, suffix_max,
                   base_counts, out, offsets[idx])


def _search(masses, los, his, lo_rem, hi_rem, n_threads=1):
    """Run the (optionally multi-threaded) DFS core over one alphabet
    (masses/los/his already sorted heaviest-first). Returns the
    (n_candidates, n_elements) int32 count array.
    """
    n = len(masses)
    suffix_min, suffix_max = _suffix_bounds(masses, los, his)
    if n == 0:
        return np.zeros((1, 0), dtype=np.int32) if lo_rem <= 0.0 <= hi_rem else np.zeros((0, 0), dtype=np.int32)

    m0 = masses[0]
    c_lo = max(los[0], int(np.ceil((lo_rem - suffix_max[1]) / m0)))
    c_hi = min(his[0], int(np.floor((hi_rem - suffix_min[1]) / m0)))
    n_branches = max(0, c_hi - c_lo + 1)
    if n_branches == 0:
        return np.zeros((0, n), dtype=np.int32)

    if n_threads > 1:
        numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
        totals = _count_top_branches(n, lo_rem, hi_rem, masses, los, his, suffix_min, suffix_max, c_lo, n_branches)
        offsets = np.zeros(n_branches + 1, dtype=np.int64)
        np.cumsum(totals, out=offsets[1:])
        out = np.zeros((offsets[-1], n), dtype=np.int32)
        _fill_top_branches(n, lo_rem, hi_rem, masses, los, his, suffix_min, suffix_max, c_lo, offsets, out)
        return out

    n_results = _count_iter(0, n, lo_rem, hi_rem, masses, los, his, suffix_min, suffix_max)
    out = np.zeros((n_results, n), dtype=np.int32)
    base_counts = np.zeros(n, dtype=np.int32)
    _fill_iter(0, n, lo_rem, hi_rem, masses, los, his, suffix_min, suffix_max, base_counts, out, 0)
    return out


def chemical_filter_mask(out, symbols, filter_="COMMON"):
    """Boolean mask replicating SIRIUS's --filter option (ChemicalValidator):
    rdbe = 1 + sum(count_i * (valence_i - 2)) / 2 (standard RDBE, using
    VALENCES above), then reject unless rdbe is within [lowerbound,
    threshold] (both inclusive), the heteroatom/carbon ratio is <= its
    threshold, and C/H is <= its threshold (elements missing from
    `symbols`, e.g. no S in the alphabet, contribute 0 atoms and don't
    affect the ratios). filter_ is one of "NONE", "STRICT", "COMMON",
    "PERMISSIVE", "RDBE". Every constant here (including the inclusive
    upper rdbe bound) was verified empirically against the running SIRIUS
    6.3.3 binary for COMMON -- see VALENCES' comment.
    """
    if filter_ in (None, "NONE"):
        return np.ones(out.shape[0], dtype=bool)
    rdbe_threshold, rdbe_lowerbound, hetero_thr, ch_thr = _FILTER_PRESETS[filter_]

    valence = np.array([VALENCES.get(s, 2) for s in symbols], dtype=np.float64)
    counts = out.astype(np.float64)
    rdbe = 1.0 + (counts * (valence - 2.0)).sum(axis=1) / 2.0
    total_atoms = counts.sum(axis=1)
    c = counts[:, symbols.index("C")] if "C" in symbols else np.zeros(out.shape[0])
    h = counts[:, symbols.index("H")] if "H" in symbols else np.zeros(out.shape[0])
    c_safe = np.where(c == 0, 0.8, c)
    h_safe = np.maximum(h, 1e-12)
    hetero_ratio = (total_atoms - c - h) / c_safe
    ch_ratio = c / h_safe
    return (rdbe >= rdbe_lowerbound) & (rdbe <= rdbe_threshold) & (hetero_ratio <= hetero_thr) & (ch_ratio <= ch_thr)


def decompose_array(mass, tol, el_str=EL_STR_DEFAULT, n_threads=1, filter_="NONE"):
    """All formulas with mass in [mass-tol, mass+tol] for the given element
    alphabet/bounds, as a raw (n_candidates, n_elements) int array plus the
    symbol order (skips per-candidate dict/string construction -- use this
    in perf-sensitive code). Pass n_threads>1 to split the heaviest
    element's branches across threads. filter_ applies SIRIUS's chemical
    plausibility filter (see chemical_filter_mask) after the search --
    "NONE" (default) or "STRICT"/"COMMON"/"PERMISSIVE"/"RDBE" to match
    `sirius decomp --filter`.
    """
    alphabet = parse_alphabet(el_str, mass + tol)
    symbols = [sym for sym, *_ in alphabet]
    masses = np.array([m for _, m, _, _, _ in alphabet])
    los = np.array([lo for _, _, lo, _, _ in alphabet], dtype=np.int64)
    his = np.array([hi for _, _, _, hi, _ in alphabet], dtype=np.int64)
    out = _search(masses, los, his, mass - tol, mass + tol, n_threads=n_threads)
    if filter_ not in (None, "NONE"):
        out = out[chemical_filter_mask(out, symbols, filter_=filter_)]
    return out, symbols


def decompose_batch_array(masses, ppm=15, el_str=EL_STR_DEFAULT, n_threads=1, filter_="NONE"):
    """Decompose many masses in parallel: a thread pool across masses
    (embarrassingly parallel -- each mass is independent), not per-mass
    top-level-branch splitting -- see module docstring. Each individual
    mass runs single-threaded internally to avoid nested-parallelism
    oversubscription. Returns a list of (out_array, symbols) in the same
    order as `masses`.
    """
    def _one(mass):
        return decompose_array(mass, mass * ppm * 1e-6, el_str=el_str, n_threads=1, filter_=filter_)

    if n_threads <= 1:
        return [_one(m) for m in masses]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(_one, masses))


@lru_cache(maxsize=32)
def _enumerate_bounded_group(bounded):
    """All combinations of an explicitly-bounded element group (small -- a
    few hundred to a few thousand for a typical heteroatom budget), with
    their masses. Independent of the query mass, so this is cached and
    computed only once no matter how many masses are decomposed against
    the same alphabet.
    """
    if not bounded:
        return np.zeros((1, 0), dtype=np.int32), np.zeros(1)
    ranges = [range(lo, hi + 1) for _, _, lo, hi, _ in bounded]
    masses = np.array([m for _, m, _, _, _ in bounded])
    combos = np.array(list(product(*ranges)), dtype=np.int32)
    combo_masses = combos @ masses
    order = np.argsort(combo_masses)
    return combos[order], combo_masses[order]


def _concat_ranges(starts, lengths):
    """[range(s0, s0+l0), range(s1, s1+l1), ...] concatenated, vectorized."""
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(starts, lengths) + (np.arange(total) - offsets)


def decompose_array_mitm(mass, tol, el_str=EL_STR_DEFAULT, n_threads=1):
    """Meet-in-the-middle version: same result as decompose_array(),
    computed by splitting the alphabet into an explicitly-bounded
    "heteroatom" group and the remaining "core" group, then merging with a
    vectorized binary search. Slower than decompose_array() for the
    default alphabet (the heteroatom group's mass spread is wide relative
    to typical ppm tolerances, so the shared core table ends up bigger
    than the plain DFS ever visits) -- kept for the record, not used by
    default.
    """
    alphabet = parse_alphabet(el_str, mass + tol)
    core = tuple(a for a in alphabet if not a[4])
    bounded = tuple(a for a in alphabet if a[4])

    if not bounded:
        return decompose_array(mass, tol, el_str=el_str, n_threads=n_threads)

    bounded_counts, bounded_masses = _enumerate_bounded_group(bounded)
    b_min, b_max = bounded_masses[0], bounded_masses[-1]

    core_symbols = [sym for sym, *_ in core]
    core_masses = np.array([m for _, m, _, _, _ in core])
    core_los = np.array([lo for _, _, lo, _, _ in core], dtype=np.int64)
    core_his = np.array([hi for _, _, _, hi, _ in core], dtype=np.int64)

    core_lo_rem = max(0.0, (mass - tol) - b_max)
    core_hi_rem = (mass + tol) - b_min
    core_out = _search(core_masses, core_los, core_his, core_lo_rem, core_hi_rem, n_threads=n_threads)
    core_mass_vals = core_out @ core_masses if core_out.shape[1] else np.zeros(core_out.shape[0])
    core_order = np.argsort(core_mass_vals)
    core_out = core_out[core_order]
    core_mass_sorted = core_mass_vals[core_order]

    lo_idx = np.searchsorted(core_mass_sorted, (mass - tol) - bounded_masses, side="left")
    hi_idx = np.searchsorted(core_mass_sorted, (mass + tol) - bounded_masses, side="right")
    lengths = hi_idx - lo_idx

    core_rows = _concat_ranges(lo_idx, lengths)
    bounded_rows = np.repeat(np.arange(len(bounded_masses)), lengths)

    out = np.concatenate([core_out[core_rows], bounded_counts[bounded_rows]], axis=1)
    symbols = core_symbols + [sym for sym, *_ in bounded]
    return out, symbols


def decompose(mass, tol, el_str=EL_STR_DEFAULT, n_threads=1, filter_="NONE"):
    """All formulas (dict[symbol] -> count) with mass in [mass-tol, mass+tol]
    for the given element alphabet/bounds.
    """
    out, symbols = decompose_array(mass, tol, el_str=el_str, n_threads=n_threads, filter_=filter_)
    return [
        {symbols[j]: int(row[j]) for j in np.nonzero(row)[0]}
        for row in out
    ]


def decompose_ppm(mass, ppm=15, el_str=EL_STR_DEFAULT, n_threads=1, filter_="NONE"):
    """Convenience wrapper: tolerance given as ppm of the target mass."""
    return decompose(mass, mass * ppm * 1e-6, el_str=el_str, n_threads=n_threads, filter_=filter_)


def format_formula(counts):
    """dict[symbol] -> count, e.g. {"C": 15, "H": 24, "O": 5} -> "C15H24O5" (Hill order)."""
    def key(sym):
        return (_HILL_ORDER.get(sym, 2), sym)

    return "".join(
        f"{sym}{counts[sym]}" if counts[sym] > 1 else sym
        for sym in sorted(counts, key=key)
    )


def parse_formula(formula):
    """"C15H24O5" -> {"C": 15, "H": 24, "O": 5}"""
    return {
        sym: (int(num) if num else 1)
        for sym, num in _FORMULA_RE.findall(formula)
        if sym
    }


def formula_mass(formula_or_counts):
    counts = parse_formula(formula_or_counts) if isinstance(formula_or_counts, str) else formula_or_counts
    return sum(MASSES[sym] * n for sym, n in counts.items())


# =========================================================================
# SIRIUS-compatible streaming API: same names/signatures/return shapes as
# sirius_decomp.py, so every existing `decomp.X` call site works unchanged
# once decomp/__init__.py points at this module instead. See module
# docstring.
# =========================================================================

def get_rounded_masses(masses):
    return [np.round(i, ROUND_FACTOR) for i in masses]


def _format_batch(masses_sorted, out_list, symbols_list, mass_sort):
    """out_list[i]/symbols_list[i] (decompose_array's raw output for
    masses_sorted[i]) -> {mass: [formula_str, ...]}. mass_sort reproduces
    sirius_decomp's nearest-first ordering, computed from the count arrays
    directly (no string reparse needed).
    """
    mass_to_form_lists = {}
    for mass, out, symbols in zip(masses_sorted, out_list, symbols_list, strict=False):
        if out.shape[0] == 0:
            mass_to_form_lists[mass] = []
            continue
        if mass_sort:
            elem_masses = np.array([MASSES[s] for s in symbols])
            cand_masses = out @ elem_masses
            order = np.argsort(np.abs(cand_masses - mass))
            out = out[order]
        mass_to_form_lists[mass] = [
            format_formula({symbols[j]: int(row[j]) for j in row.nonzero()[0]})
            for row in out
        ]
    return mass_to_form_lists


def _load_batch_cache(cache_path):
    with np.load(cache_path, allow_pickle=False) as npz:
        masses = npz["masses"]
        offsets = npz["offsets"]
        forms = npz["forms"]
    out = {}
    for i, mass in enumerate(masses):
        out[float(mass)] = forms[offsets[i]:offsets[i + 1]].tolist()
    return out


def _save_batch_cache(cache_path, mass_to_form_lists):
    masses = np.array(list(mass_to_form_lists.keys()), dtype=np.float64)
    lengths = np.array([len(v) for v in mass_to_form_lists.values()], dtype=np.int64)
    offsets = np.zeros(len(masses) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    all_forms = [f for v in mass_to_form_lists.values() for f in v]
    forms = np.array(all_forms, dtype="<U64") if all_forms else np.array([], dtype="<U1")
    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, masses=masses, offsets=offsets, forms=forms)
    tmp_path.rename(cache_path)


def iter_sirius_batches(
    masses,
    adduct=None,
    verbose=False,
    mass_sort=True,
    filter_="NONE",
    ppm=15,
    max_batch=10000,
    el_str=EL_STR_DEFAULT,
    cores=16,
    loglevel="WARNING",
    desc="mass_decomp",
    cache_dir=None,
):
    """Yield (batch_idx, batch_masses, mass_to_form_lists) one batch at a
    time -- same shape as sirius_decomp.iter_sirius_batches, backed by the
    pure-Python engine above instead of shelling out to SIRIUS. Peak memory
    is bounded by max_batch, same guarantee as the original.

    adduct/verbose/loglevel are accepted for signature compatibility but
    unused: no caller in this codebase passes adduct (every call site
    already converts precursor m/z to neutral mass before calling decomp,
    so an ion-mode-specific decomposition step was never needed), and
    there's no subprocess to log.
    """
    del adduct, verbose, loglevel  # signature compatibility only, see docstring

    masses = [np.round(i, ROUND_FACTOR) for i in masses]
    unique_masses = np.unique(masses)
    unique_masses = np.sort(unique_masses)[::-1]  # heaviest-first, matches sirius_decomp

    num_sections = max(1, math.ceil(unique_masses.shape[0] / max_batch))
    mass_splits = np.array_split(unique_masses, num_sections)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"  Running {num_sections} mass_decomp batches serially with up to {cores} "
        f"threads/batch ({unique_masses.shape[0]} unique masses, heaviest-first)..."
    )
    for batch_idx, mass_split in enumerate(tqdm(mass_splits, desc=desc, unit="batch")):
        cache_path = cache_dir / f"batch_{batch_idx:04d}.npz" if cache_dir is not None else None

        if cache_path is not None and cache_path.exists():
            mass_to_form_lists = _load_batch_cache(cache_path)
        else:
            batch_masses_list = mass_split.tolist()
            results = decompose_batch_array(
                batch_masses_list, ppm=ppm, el_str=el_str, n_threads=cores, filter_=filter_
            )
            out_list = [r[0] for r in results]
            symbols_list = [r[1] for r in results]
            mass_to_form_lists = _format_batch(batch_masses_list, out_list, symbols_list, mass_sort)
            del results, out_list, symbols_list  # done with raw arrays for this batch

            if cache_path is not None:
                _save_batch_cache(cache_path, mass_to_form_lists)

        yield batch_idx, mass_split.tolist(), mass_to_form_lists
        del mass_to_form_lists  # this batch's strings are consumed by the caller; don't hold on to them


def run_sirius(masses, **kwargs):
    """Back-compat wrapper: merges every batch into one dict, all in memory.

    Prefer iter_sirius_batches for large mass lists -- this holds every
    batch's decompositions in memory simultaneously, same caveat as
    sirius_decomp.run_sirius.
    """
    out = {}
    for _, _, batch_dict in iter_sirius_batches(masses, **kwargs):
        out.update(batch_dict)
    return out


def run_batch(masses, ppm=15, el_str=EL_STR_DEFAULT, n_threads=1, filter_="NONE"):
    """mass -> sorted list of formula strings, for a batch of masses.
    Simple non-streaming convenience wrapper for small inputs/tests; for
    real workloads use iter_sirius_batches.
    """
    out = {}
    for mass in masses:
        decomps = decompose_ppm(mass, ppm=ppm, el_str=el_str, n_threads=n_threads, filter_=filter_)
        out[mass] = sorted(format_formula(c) for c in decomps)
    return out
