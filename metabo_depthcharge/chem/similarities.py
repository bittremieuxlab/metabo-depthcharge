import multiprocessing
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm


class BinaryTanimoto:
    """Binary (bitwise) Tanimoto similarity between fingerprints.

    Instances are callable. The similarity is computed in :meth:`__call__`.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
        """Compute Binary Tanimoto similarity using numpy broadcasting.

        The last axis of each input is the fingerprint dimension and is
        reduced. All leading axes follow standard numpy broadcasting rules.

        Parameters
        ----------
        fp1, fp2: np.ndarray
            Binary fingerprints. Last axis is the bit dimension. Any leading
            axes must be broadcast-compatible.

        Returns
        -------
        float or np.ndarray
            Similarity score(s) with shape equal to the broadcasted leading
            axes (scalar if both inputs are 1-D). For an all-pairs matrix,
            pass e.g. ``fp1[:, None, :]`` and ``fp2[None, :, :]``.
        """
        fp1_bool = fp1.astype(bool)
        fp2_bool = fp2.astype(bool)
        intersection = np.sum(fp1_bool & fp2_bool, axis=-1)
        union = np.sum(fp1_bool | fp2_bool, axis=-1)
        out = np.zeros_like(intersection, dtype=float)
        result = np.divide(intersection, union, out=out, where=union > 0)
        if result.ndim == 0:
            return float(result)
        return result


class CountTanimoto:
    """Count (weighted) Tanimoto similarity between count fingerprints.

    Instances are callable; the similarity is computed in :meth:`__call__`.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
        """Compute Count Tanimoto similarity using numpy broadcasting.

        The last axis of each input is the fingerprint dimension and is
        reduced. All leading axes follow standard numpy broadcasting rules.

        Parameters
        ----------
        fp1, fp2: np.ndarray
            Count fingerprints. Last axis is the bit dimension. Any leading
            axes must be broadcast-compatible.

        Returns
        -------
        float or np.ndarray
            Similarity score(s) with shape equal to the broadcasted leading
            axes (scalar if both inputs are 1-D). For an all-pairs matrix,
            pass e.g. ``fp1[:, None, :]`` and ``fp2[None, :, :]``.
        """
        min_sum = np.sum(np.minimum(fp1, fp2), axis=-1)
        max_sum = np.sum(np.maximum(fp1, fp2), axis=-1)
        out = np.zeros_like(min_sum, dtype=float)
        result = np.divide(min_sum, max_sum, out=out, where=max_sum > 0)
        if result.ndim == 0:
            return float(result)
        return result


class CosineSimilarity:
    """Cosine similarity between (sets of) vectors.

    Instances are callable; the similarity is computed in :meth:`__call__`.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Compute Cosine similarity using numpy broadcasting.

        The last axis of each input is the embedding dimension and is
        reduced. All leading axes follow standard numpy broadcasting rules.

        Parameters
        ----------
        emb1, emb2: np.ndarray
            (Sets of) vectors. Last axis is the embedding dimension; any leading
            axes must be broadcast-compatible.

        Returns
        -------
        float or np.ndarray
            Similarity score(s) with shape equal to the broadcasted leading
            axes (scalar if both inputs are 1-D). For an all-pairs matrix,
            pass e.g. ``emb1[:, None, :]`` and ``emb2[None, :, :]``.
        """
        dot = np.sum(emb1 * emb2, axis=-1)
        norm1 = np.linalg.norm(emb1, axis=-1)
        norm2 = np.linalg.norm(emb2, axis=-1)
        denom = norm1 * norm2
        out = np.zeros(np.broadcast_shapes(dot.shape, denom.shape), dtype=float)
        result = np.divide(dot, denom, out=out, where=denom > 0)
        if result.ndim == 0:
            return float(result)
        return result


def _mces_worker(args):
    """Top-level function for multiprocessing Pool.map."""
    from myopic_mces import MCES

    smi1, smi2, threshold, always_stronger_bound, solver_options, solver = args
    _, dist, _, _ = MCES(
        smi1,
        smi2,
        threshold=threshold,
        always_stronger_bound=always_stronger_bound,
        solver_options=solver_options,
        solver=solver,
    )
    return dist


def _mces_worker_queue(args, result_queue):
    """Worker that puts result in a queue, for Process-based timeout."""
    try:
        result_queue.put(_mces_worker(args))
    except Exception:
        result_queue.put(np.nan)


def _mces_lb_stats(smiles):
    """Precompute the per-molecule structures the ``filter2`` lower bound needs."""
    from myopic_mces.graph import construct_graph

    G = construct_graph(str(smiles))
    atom = {i: G.nodes[i]["atom"] for i in G.nodes}
    halfdeg = {}
    nbr = {}
    by_type = defaultdict(list)
    for i in G.nodes:
        by_type[atom[i]].append(i)
        weights = defaultdict(list)
        deg = 0.0
        for k in G.neighbors(i):
            w = G[i][k]["weight"]
            deg += w
            weights[atom[k]].append(w)
        halfdeg[i] = deg / 2.0
        nbr[i] = {t: sorted(ws, reverse=True) for t, ws in weights.items()}
    return by_type, halfdeg, nbr


def _mces_pair_cost(nbr1, nbr2):
    """Cost of mapping one atom to another from their neighbour-weight lists.

    Faithful port of ``myopic_mces.filter_MCES.get_cost``.
    """
    diff = 0.0
    for t, w1 in nbr1.items():
        w2 = nbr2.get(t)
        if w2 is None:
            diff += sum(w1) / 2.0
            continue
        n = min(len(w1), len(w2))
        for a in range(n):
            diff += abs(w1[a] - w2[a]) / 2.0
        diff += (sum(w1[n:]) + sum(w2[n:])) / 2.0
    for t, w2 in nbr2.items():
        if t not in nbr1:
            diff += sum(w2) / 2.0
    return diff


def _mces_filter2_lb(stats1, stats2):
    """Strong myopic-MCES lower bound (``filter2``) from precomputed stats.

    Faithful port of ``myopic_mces.filter_MCES.filter2``, but the per-element
    minimum-weight full matching is solved with
    :func:`scipy.optimize.linear_sum_assignment` instead of building a
    ``networkx`` graph on every call.
    """
    by_type1, halfdeg1, nbr1 = stats1
    by_type2, halfdeg2, nbr2 = stats2
    total = 0.0
    for t in by_type1.keys() | by_type2.keys():
        nodes1 = by_type1.get(t, [])
        nodes2 = by_type2.get(t, [])
        n1, n2 = len(nodes1), len(nodes2)
        m = max(n1, n2)
        cost = np.zeros((m, m))
        for a, i in enumerate(nodes1):
            for b, j in enumerate(nodes2):
                cost[a, b] = _mces_pair_cost(nbr1[i], nbr2[j])
        if n1 < n2:  # dummy query atoms -> cost = half degree of target atom
            for b, j in enumerate(nodes2):
                cost[n1:, b] = halfdeg2[j]
        elif n2 < n1:  # dummy target atoms -> cost = half degree of query atom
            for a, i in enumerate(nodes1):
                cost[a, n2:] = halfdeg1[i]
        rows, cols = linear_sum_assignment(cost)
        total += float(cost[rows, cols].sum())
    return total


_LB_MIN_PAIRS_FOR_POOL = 512
_LB_POOL_STATS = None


def _lb_pool_init(stats):
    """Pool initializer: stash the shared per-molecule stats in each worker."""
    global _LB_POOL_STATS
    _LB_POOL_STATS = stats


def _lb_pool_worker(ij):
    """Worker: look up the precomputed stats for an ``(i, j)`` index pair."""
    i, j = ij
    return _mces_filter2_lb(_LB_POOL_STATS[i], _LB_POOL_STATS[j])


class MCESDistance:
    """Compute myopic-MCES (Maximum Common Edge Subgraph) distance between molecules.

    Parameters
    ----------
    threshold: int
        MCES threshold parameter (default: 15).
    always_stronger_bound: bool
        Use stronger bound in MCES (default: True).
    n_jobs: int
        Number of parallel workers. -1 uses all CPUs (default: -1).
    solver_options: dict
        Options for the underlying solver
        Default: None resolves to ``dict(msg=0)``.
    solver: str
        Solver to use (default: "HiGHS").
    timeout: float or None
        Timeout in seconds for each pairwise MCES computation.
        If a computation exceeds this time, NaN is returned.
        None means no timeout.
    progress : bool
        Show a tqdm progress bar over pairs.
    """

    def __init__(
        self,
        threshold: int = 15,
        always_stronger_bound: bool = True,
        n_jobs: int = -1,
        solver_options: dict = None,
        solver: str = "HiGHS",
        timeout: float | None = None,
        progress: bool = False,
    ) -> None:
        if solver_options is None:
            solver_options = {"msg": 0}
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.solver_options = solver_options
        self.solver = solver
        self.timeout = timeout
        self.progress = progress

    def _make_worker_args(self, smi1, smi2):
        return (
            smi1,
            smi2,
            self.threshold,
            self.always_stronger_bound,
            self.solver_options,
            self.solver,
        )

    def _dispatch(self, pairs):
        """Compute MCES for a list of (smi1, smi2) pairs, returning a list of distances."""
        worker_args = [self._make_worker_args(s1, s2) for s1, s2 in pairs]

        if len(worker_args) == 0:
            return []

        bar = tqdm(total=len(worker_args), disable=not self.progress, unit="pair")

        if self.timeout is None:
            if len(worker_args) == 1:
                try:
                    result = [_mces_worker(worker_args[0])]
                except Exception:
                    result = [np.nan]
                bar.update(1)
                bar.close()
                return result
            with Pool(self.n_jobs) as pool:
                results = []
                for r in pool.imap(_mces_worker, worker_args):
                    results.append(r)
                    bar.update(1)
            bar.close()
            return results

        # Timeout path: use Process + kill() for reliable enforcement.
        # Process pairs in chunks of n_jobs for parallelism.
        results = []
        for chunk_start in range(0, len(worker_args), self.n_jobs):
            chunk = worker_args[chunk_start : chunk_start + self.n_jobs]
            queues = []
            procs = []
            for args in chunk:
                q = multiprocessing.Queue(1)
                p = multiprocessing.Process(target=_mces_worker_queue, args=(args, q))
                queues.append(q)
                procs.append(p)

            for p in procs:
                p.start()

            # All processes in the chunk started ~simultaneously;
            # give them all `timeout` seconds from now.
            deadline = time.monotonic() + self.timeout

            for p, q in zip(procs, queues, strict=False):
                remaining = max(0, deadline - time.monotonic())
                p.join(timeout=remaining)
                if p.is_alive():
                    p.kill()
                    p.join()
                    results.append(np.nan)
                else:
                    try:
                        results.append(q.get_nowait())
                    except Exception:
                        results.append(np.nan)
                bar.update(1)

        bar.close()
        return results

    def _dispatch_lb(self, pairs):
        """Lower-bound analog of :meth:`_dispatch`: ``(smi1, smi2)`` pairs -> bounds."""
        if not pairs:
            return []

        uniq: dict[str, int] = {}

        def index(smi: str) -> int:
            k = uniq.get(smi)
            if k is None:
                k = uniq[smi] = len(uniq)
            return k

        index_pairs = [(index(a), index(b)) for a, b in pairs]
        stats = [_mces_lb_stats(smi) for smi in uniq]

        if self.n_jobs == 1 or len(index_pairs) < _LB_MIN_PAIRS_FOR_POOL:
            bar = tqdm(index_pairs, disable=not self.progress, unit="pair")
            return [_mces_filter2_lb(stats[i], stats[j]) for i, j in bar]

        bar = tqdm(total=len(index_pairs), disable=not self.progress, unit="pair")
        chunksize = max(1, len(index_pairs) // (self.n_jobs * 8))
        results = []
        with Pool(self.n_jobs, initializer=_lb_pool_init, initargs=(stats,)) as pool:
            for d in pool.imap(_lb_pool_worker, index_pairs, chunksize=chunksize):
                results.append(d)
                bar.update(1)
        bar.close()
        return results

    def _broadcast(self, smiles1, smiles2, compute):
        """Broadcast two SMILES inputs, run ``compute`` over the flat pair list."""
        s1 = np.asarray(smiles1)
        s2 = np.asarray(smiles2)
        shape = np.broadcast_shapes(s1.shape, s2.shape)
        s1_flat = np.broadcast_to(s1, shape).ravel()
        s2_flat = np.broadcast_to(s2, shape).ravel()
        pairs = [(str(a), str(b)) for a, b in zip(s1_flat, s2_flat, strict=False)]
        result = np.array(compute(pairs), dtype=np.float64).reshape(shape)
        if result.ndim == 0:
            return float(result)
        return result

    def _pairwise(self, smiles, compute):
        """Symmetric ``(n, n)`` matrix: run ``compute`` over the upper triangle.

        Only the ``i > j`` pairs are evaluated and mirrored; the diagonal stays
        zero (self-distances are not computed).
        """
        arr = np.asarray(smiles).ravel()
        n = len(arr)
        index_pairs = [(i, j) for i in range(n) for j in range(i)]
        pairs = [(str(arr[i]), str(arr[j])) for i, j in index_pairs]
        dists = compute(pairs)
        result = np.zeros((n, n), dtype=np.float64)
        for (i, j), d in zip(index_pairs, dists, strict=False):
            result[i, j] = result[j, i] = d
        return result

    def __call__(
        self,
        smiles1: str | list[str] | np.ndarray,
        smiles2: str | list[str] | np.ndarray,
    ) -> float | np.ndarray:
        """Compute MCES distance using numpy broadcasting.

        Inputs are broadcast together following standard numpy rules.
        Each element-pair produces one scalar distance.

        Parameters
        ----------
        smiles1, smiles2: str, List[str], or np.ndarray of strings
            SMILES inputs with broadcast-compatible shapes.

        Returns
        -------
        float or np.ndarray
            Distance(s) with shape equal to the broadcasted shape
            (scalar if both inputs are single str). For an all-pairs
            matrix, pass e.g. ``smiles1[:, None]`` and ``smiles2[None, :]``.
        """
        return self._broadcast(smiles1, smiles2, self._dispatch)

    def pairwise(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        """Symmetric pairwise distance matrix, computing only the upper triangle.

        Parameters
        ----------
        smiles : list[str] or np.ndarray
            SMILES inputs; flattened to 1-D, giving ``n`` molecules.

        Returns
        -------
        np.ndarray
            ``(n, n)`` symmetric matrix of pairwise MCES distances. The
            diagonal is zero (self-distances are not computed); only the
            upper triangle is computed and mirrored to the lower triangle.
        """
        return self._pairwise(smiles, self._dispatch)

    def lower_bound(
        self,
        smiles1: str | list[str] | np.ndarray,
        smiles2: str | list[str] | np.ndarray,
    ) -> float | np.ndarray:
        """Compute Lower Bounds on MCES distances using numpy broadcasting.

        Inputs are broadcast together following standard numpy rules.
        Each element-pair produces one scalar LB on distance (like :meth:`__call__`).

        This method never invokes the ILP solver. Equivalent to calling
        :meth:`__call__` when ``MCES(..., threshold=0)``, but with a more efficient
        backbone implementation . The ``threshold``, ``solver`` and
        ``always_stronger_bound`` attributes do not affect this method.

        Parameters
        ----------
        smiles1, smiles2: str, List[str], or np.ndarray of strings
            SMILES inputs with broadcast-compatible shapes. For an all-pairs
            matrix, pass e.g. ``smiles1[:, None]`` and ``smiles2[None, :]``.

        Returns
        -------
        float or np.ndarray
            Lower bounds(s) with shape equal to the broadcasted shape
            (scalar if both inputs are single str). For an all-pairs
            matrix, pass e.g. ``smiles1[:, None]`` and ``smiles2[None, :]``.
        """
        return self._broadcast(smiles1, smiles2, self._dispatch_lb)

    def lower_bound_pairwise(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        """Symmetric pairwise LB distance matrix, computing only the upper triangle.

        Like :meth:`pairwise` but uses the solver-free strong lower bound of
        :meth:`lower_bound`.

        Parameters
        ----------
        smiles : list[str] or np.ndarray
            SMILES inputs; flattened to 1-D, giving ``n`` molecules.

        Returns
        -------
        np.ndarray
            ``(n, n)`` symmetric matrix of pairwise MCES distances. The
            diagonal is zero (self-distances are not computed); only the
            upper triangle is computed and mirrored to the lower triangle.
        """
        return self._pairwise(smiles, self._dispatch_lb)
