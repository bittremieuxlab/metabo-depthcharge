import numpy as np
import pytest

from metabo_depthcharge.similarities import (
    BinaryTanimoto,
    CosineSimilarity,
    CountTanimoto,
    MCESDistance,
)


def test_binary_tanimoto_identical():
    fp = np.array([1, 1, 0, 1, 0], dtype=float)
    assert BinaryTanimoto()(fp, fp) == pytest.approx(1.0)


def test_binary_tanimoto_disjoint():
    fp1 = np.array([1, 1, 0, 0], dtype=float)
    fp2 = np.array([0, 0, 1, 1], dtype=float)
    assert BinaryTanimoto()(fp1, fp2) == pytest.approx(0.0)


def test_binary_tanimoto_partial():
    fp1 = np.array([1, 1, 1, 0], dtype=float)
    fp2 = np.array([1, 1, 0, 1], dtype=float)
    # intersection=2, union=4
    assert BinaryTanimoto()(fp1, fp2) == pytest.approx(0.5)


def test_binary_tanimoto_zero_vectors():
    fp = np.array([0, 0, 0], dtype=float)
    assert BinaryTanimoto()(fp, fp) == pytest.approx(0.0)


def test_binary_tanimoto_batch():
    fps1 = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
    fps2 = np.array([[1, 1, 0], [0, 0, 1]], dtype=float)
    result = BinaryTanimoto()(fps1, fps2)
    assert result.shape == (2,)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_binary_tanimoto_all_pairs():
    fps = np.array([[1, 1, 0], [1, 0, 1]], dtype=float)
    result = BinaryTanimoto()(fps[:, None, :], fps[None, :, :])
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[1, 1] == pytest.approx(1.0)
    assert result[0, 1] == result[1, 0]


# --- CountTanimoto ---


def test_count_tanimoto_identical():
    fp = np.array([2, 3, 0, 1], dtype=float)
    assert CountTanimoto()(fp, fp) == pytest.approx(1.0)


def test_count_tanimoto_disjoint():
    fp1 = np.array([2, 0, 0], dtype=float)
    fp2 = np.array([0, 0, 3], dtype=float)
    assert CountTanimoto()(fp1, fp2) == pytest.approx(0.0)


def test_count_tanimoto_partial():
    fp1 = np.array([2, 1, 0], dtype=float)
    fp2 = np.array([1, 1, 0], dtype=float)
    # min_sum=2, max_sum=3
    assert CountTanimoto()(fp1, fp2) == pytest.approx(2 / 3)


def test_count_tanimoto_zero_vectors():
    fp = np.array([0, 0, 0], dtype=float)
    assert CountTanimoto()(fp, fp) == pytest.approx(0.0)


def test_cosine_identical():
    emb = np.array([1.0, 2.0, 3.0])
    assert CosineSimilarity()(emb, emb) == pytest.approx(1.0)


def test_cosine_orthogonal():
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.0, 1.0])
    assert CosineSimilarity()(emb1, emb2) == pytest.approx(0.0)


def test_cosine_opposite():
    emb = np.array([1.0, 0.0])
    assert CosineSimilarity()(emb, -emb) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    emb1 = np.array([0.0, 0.0])
    emb2 = np.array([1.0, 0.0])
    assert CosineSimilarity()(emb1, emb2) == pytest.approx(0.0)


def test_cosine_batch():
    embs1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    embs2 = np.array([[1.0, 0.0], [1.0, 0.0]])
    result = CosineSimilarity()(embs1, embs2)
    assert result.shape == (2,)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_cosine_all_pairs():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = CosineSimilarity()(embs[:, None, :], embs[None, :, :])
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(0.0)


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
MCES_KWARGS = {"solver": "PULP_CBC_CMD", "solver_options": {"msg": 0}}


def test_mces_identical():
    dist = MCESDistance(**MCES_KWARGS)(ASPIRIN, ASPIRIN)
    assert dist == pytest.approx(0.0)


def test_mces_returns_float():
    dist = MCESDistance(**MCES_KWARGS)(ASPIRIN, CAFFEINE)
    assert isinstance(dist, float)
    assert dist >= 0.0


def test_mces_batch():
    dist = MCESDistance(**MCES_KWARGS)([ASPIRIN, ASPIRIN], [ASPIRIN, CAFFEINE])
    assert dist.shape == (2,)
    assert dist[0] == pytest.approx(0.0)


def test_mces_pairwise_shape():
    result = MCESDistance(**MCES_KWARGS).pairwise([ASPIRIN, CAFFEINE])
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, 1] == pytest.approx(0.0)
    assert result[0, 1] == result[1, 0]


def test_mces_symmetric():
    d1 = MCESDistance(**MCES_KWARGS)(ASPIRIN, CAFFEINE)
    d2 = MCESDistance(**MCES_KWARGS)(CAFFEINE, ASPIRIN)
    assert d1 == pytest.approx(d2)
