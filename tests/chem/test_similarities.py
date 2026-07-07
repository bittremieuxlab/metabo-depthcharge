"""Tests for :mod:`metabo_depthcharge.chem.similarities` — bitwise/vectorwise
similarity metrics and myopic-MCES distance."""

import numpy as np
import pytest

from metabo_depthcharge.chem.similarities import (
    BinaryTanimoto,
    CosineSimilarity,
    CountTanimoto,
    MCESDistance,
)


SMI_A, SMI_B, SMI_C = "CC", "CO", "CN"
BAD_SMI = "not_a_smiles!!!"


# --- BinaryTanimoto / CountTanimoto / CosineSimilarity -------------------


def test_binary_tanimoto():
    fp1 = np.array([[1, 0, 1, 1], [0, 0, 0, 0]]).reshape(2, 1, 4)
    fp2 = np.array([[1, 1, 1, 0]]).reshape(1, 1, 4)
    assert np.allclose(BinaryTanimoto()(fp1, fp2), np.array([0.5, 0]).reshape(2, 1))
    assert BinaryTanimoto()(np.array([0, 0]), np.array([0, 0])) == 0.0


def test_count_tanimoto():
    fp1, fp2 = (
        np.array([2, 0, 3, 1]).reshape(1, 1, 4),
        np.array([1, 1, 2, 0]).reshape(1, 1, 4),
    )
    assert CountTanimoto()(fp1, fp2)[0] == pytest.approx(3 / 7)
    assert CountTanimoto()(np.array([0, 0]), np.array([0, 0])) == 0.0


def test_cosine_similarity():
    fp1 = np.array([1.0, 0.0]).reshape(1, 1, 2)
    fp2 = np.array([0.0, 1.0]).reshape(1, 1, 2)
    assert CosineSimilarity()(fp1, fp2)[0] == 0.0
    sim = CosineSimilarity()(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert sim == pytest.approx(1.0)
    zero_sim = CosineSimilarity()(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert zero_sim == 0.0


# --- MCESDistance ----------------------------------------------------------


def test_mces_distance(monkeypatch):
    monkeypatch.setattr(
        "metabo_depthcharge.chem.similarities._LB_MIN_PAIRS_FOR_POOL", 2
    )
    d = MCESDistance(n_jobs=2)
    assert d(SMI_A, SMI_B) == 2.0
    assert d.lower_bound(SMI_A, SMI_B) <= 2.0
    assert np.all(d(np.array([SMI_A]), np.array([SMI_B])) == 2.0)
    assert np.isnan(d.lower_bound(SMI_A, BAD_SMI))
    assert d._dispatch([]) == []
    assert d._dispatch_lb([]) == []
    assert np.isnan(d._dispatch([(BAD_SMI, SMI_A)])[0])
    d = MCESDistance(n_jobs=2, progress=True)
    assert np.all(d.lower_bound_pairwise([SMI_A, SMI_B, SMI_C] * 3) <= 5.0)
    assert np.all(d.pairwise([SMI_A, SMI_B, SMI_C]) <= 5.0)


def test_dispatch_timeout_kills_slow_worker():
    assert MCESDistance(timeout=30.0)._dispatch([(SMI_A, SMI_B)]) == [2.0]
    d = MCESDistance(n_jobs=2, timeout=1e-9)
    result = d._dispatch([(SMI_A, SMI_B)])
    assert np.isnan(result[0])
