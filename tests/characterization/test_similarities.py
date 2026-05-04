"""Characterization tests for metabo_depthcharge.similarities vs spectrawl."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_spectrawl_path = next(
    p for p in sys.path if (Path(p) / "spectrawl" / "data" / "similarities.py").exists()
)
_metabo_src_path = next(
    p for p in sys.path if (Path(p) / "metabo_depthcharge" / "similarities.py").exists()
)

orig = _load_from_file(
    "spectrawl_similarities",
    Path(_spectrawl_path) / "spectrawl" / "data" / "similarities.py",
)
new = _load_from_file(
    "metabo_similarities",
    Path(_metabo_src_path) / "metabo_depthcharge" / "similarities.py",
)


FP_BINARY = np.array([[1, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=float)
FP_COUNT = np.array([[2, 1, 0], [1, 1, 1], [0, 2, 3]], dtype=float)
EMBS = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 1.0, 0.0]])
MCES_KWARGS = {"solver": "PULP_CBC_CMD", "solver_options": {"msg": 0}}

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


# --- BinaryTanimoto ---


def test_binary_tanimoto_pair():
    np.testing.assert_allclose(
        orig.BinaryTanimoto()(FP_BINARY[0], FP_BINARY[1]),
        new.BinaryTanimoto()(FP_BINARY[0], FP_BINARY[1]),
    )


def test_binary_tanimoto_batch():
    np.testing.assert_allclose(
        orig.BinaryTanimoto()(FP_BINARY[:-1], FP_BINARY[1:]),
        new.BinaryTanimoto()(FP_BINARY[:-1], FP_BINARY[1:]),
    )


def test_binary_tanimoto_all_pairs():
    np.testing.assert_allclose(
        orig.BinaryTanimoto()(FP_BINARY[:, None, :], FP_BINARY[None, :, :]),
        new.BinaryTanimoto()(FP_BINARY[:, None, :], FP_BINARY[None, :, :]),
    )


def test_count_tanimoto_pair():
    np.testing.assert_allclose(
        orig.CountTanimoto()(FP_COUNT[0], FP_COUNT[1]),
        new.CountTanimoto()(FP_COUNT[0], FP_COUNT[1]),
    )


def test_count_tanimoto_batch():
    np.testing.assert_allclose(
        orig.CountTanimoto()(FP_COUNT[:-1], FP_COUNT[1:]),
        new.CountTanimoto()(FP_COUNT[:-1], FP_COUNT[1:]),
    )


def test_cosine_pair():
    np.testing.assert_allclose(
        orig.CosineSimilarity()(EMBS[0], EMBS[1]),
        new.CosineSimilarity()(EMBS[0], EMBS[1]),
    )


def test_cosine_batch():
    np.testing.assert_allclose(
        orig.CosineSimilarity()(EMBS[:-1], EMBS[1:]),
        new.CosineSimilarity()(EMBS[:-1], EMBS[1:]),
    )


def test_cosine_all_pairs():
    np.testing.assert_allclose(
        orig.CosineSimilarity()(EMBS[:, None, :], EMBS[None, :, :]),
        new.CosineSimilarity()(EMBS[:, None, :], EMBS[None, :, :]),
    )


def test_mces_identical():
    d_orig = orig.MCESDistance(**MCES_KWARGS)(ASPIRIN, ASPIRIN)
    d_new = new.MCESDistance(**MCES_KWARGS)(ASPIRIN, ASPIRIN)
    assert d_orig == pytest.approx(d_new)


def test_mces_pair():
    d_orig = orig.MCESDistance(**MCES_KWARGS)(ASPIRIN, CAFFEINE)
    d_new = new.MCESDistance(**MCES_KWARGS)(ASPIRIN, CAFFEINE)
    assert d_orig == pytest.approx(d_new)


def test_mces_pairwise():
    smiles = [ASPIRIN, CAFFEINE]
    np.testing.assert_allclose(
        orig.MCESDistance(**MCES_KWARGS).pairwise(smiles),
        new.MCESDistance(**MCES_KWARGS).pairwise(smiles),
    )
