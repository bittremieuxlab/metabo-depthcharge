import multiprocessing
import time
from multiprocessing import Pool, cpu_count

import numpy as np


class BinaryTanimoto:
    def __init__(self) -> None:
        pass

    def __call__(self, fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
        """Compute Binary Tanimoto similarity using numpy broadcasting.

        The last axis of each input is the fingerprint dimension and is
        reduced. All leading axes follow standard numpy broadcasting rules.

        Parameters
        ----------
        fp1, fp2: np.ndarray
            Binary fingerprints. Last axis is the bit dimension; any leading
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
    def __init__(self) -> None:
        pass

    def __call__(self, fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
        """Compute Count Tanimoto similarity using numpy broadcasting.

        The last axis of each input is the fingerprint dimension and is
        reduced. All leading axes follow standard numpy broadcasting rules.

        Parameters
        ----------
        fp1, fp2: np.ndarray
            Count fingerprints. Last axis is the bit dimension; any leading
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
    def __init__(self) -> None:
        pass

    def __call__(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Compute Cosine similarity using numpy broadcasting.

        The last axis of each input is the embedding dimension and is
        reduced. All leading axes follow standard numpy broadcasting rules.

        Parameters
        ----------
        emb1, emb2: np.ndarray
            Embeddings. Last axis is the embedding dimension; any leading
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
        Options for the underlying solver (default: dict(msg=0)).
    solver: str
        Solver to use (default: "HiGHS").
    timeout: float or None
        Timeout in seconds for each pairwise MCES computation.
        If a computation exceeds this time, NaN is returned.
        None means no timeout (default: None).
    """

    def __init__(
        self,
        threshold: int = 15,
        always_stronger_bound: bool = True,
        n_jobs: int = -1,
        solver_options: dict = None,
        solver: str = "HiGHS",
        timeout: float | None = None,
    ) -> None:
        if solver_options is None:
            solver_options = {"msg": 0}
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.solver_options = solver_options
        self.solver = solver
        self.timeout = timeout

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

        if self.timeout is None:
            if len(worker_args) == 1:
                try:
                    return [_mces_worker(worker_args[0])]
                except Exception:
                    return [np.nan]
            with Pool(self.n_jobs) as pool:
                return pool.map(_mces_worker, worker_args)

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

        return results

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
            (scalar if both inputs are scalar/str). For an all-pairs
            matrix, pass e.g. ``smiles1[:, None]`` and ``smiles2[None, :]``.
        """
        s1 = np.asarray(smiles1)
        s2 = np.asarray(smiles2)

        shape = np.broadcast_shapes(s1.shape, s2.shape)
        s1_b = np.broadcast_to(s1, shape)
        s2_b = np.broadcast_to(s2, shape)

        s1_flat = s1_b.ravel()
        s2_flat = s2_b.ravel()

        pairs = [(str(a), str(b)) for a, b in zip(s1_flat, s2_flat, strict=False)]

        if len(pairs) == 0:
            return np.array([], dtype=np.float64).reshape(shape)

        dists = self._dispatch(pairs)
        result = np.array(dists, dtype=np.float64).reshape(shape)

        if result.ndim == 0:
            return float(result)
        return result

    def pairwise(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        """Symmetric pairwise distance matrix, computing only the upper triangle."""
        arr = np.asarray(smiles).ravel()
        n = len(arr)
        pairs = [(str(arr[i]), str(arr[j])) for i in range(n) for j in range(i)]
        dists = self._dispatch(pairs) if pairs else []
        result = np.zeros((n, n), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i):
                result[i, j] = result[j, i] = dists[idx]
                idx += 1
        return result
