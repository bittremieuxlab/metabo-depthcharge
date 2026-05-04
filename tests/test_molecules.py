import numpy as np
import pytest

from metabo_depthcharge.molecules import (
    SmilesToBiosynfoni,
    SmilesToMACCS,
    SmilesToMAP4,
    SmilesToMorgan,
    SmilesToRdkit,
    canonicalize_smiles,
    safe_mol_from_smiles,
)


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def test_canonicalize_smiles_is_stable():
    canon = canonicalize_smiles(ASPIRIN)
    assert canonicalize_smiles(canon) == canon


def test_canonicalize_smiles_returns_str():
    assert isinstance(canonicalize_smiles(ASPIRIN), str)


def test_safe_mol_from_smiles_valid():
    from rdkit import Chem

    mol = safe_mol_from_smiles(ASPIRIN)
    assert mol is not None
    assert Chem.MolToSmiles(mol) == canonicalize_smiles(ASPIRIN)


def test_safe_mol_from_smiles_invalid_raises():
    with pytest.raises(ValueError):
        safe_mol_from_smiles("not-a-smiles-!!!")


def test_morgan_single_shape():
    enc = SmilesToMorgan()
    fp = enc(ASPIRIN)
    assert fp.shape == (4096,)


def test_morgan_fp_size():
    enc = SmilesToMorgan(fp_size=2048)
    assert enc(ASPIRIN).shape == (2048,)
    assert enc.fp_size == 2048


def test_morgan_batch_shape():
    enc = SmilesToMorgan()
    fps = enc([ASPIRIN, CAFFEINE])
    assert fps.shape == (2, 4096)


def test_morgan_binary_values():
    fp = SmilesToMorgan()(ASPIRIN)
    assert set(np.unique(fp)).issubset({0, 1})


def test_morgan_counts_nonnegative():
    fp = SmilesToMorgan(counts=True)(ASPIRIN)
    assert (fp >= 0).all()


def test_morgan_different_molecules():
    enc = SmilesToMorgan()
    assert not np.array_equal(enc(ASPIRIN), enc(CAFFEINE))


def test_rdkit_fp_single_shape():
    enc = SmilesToRdkit()
    assert enc(ASPIRIN).shape == (4096,)


def test_rdkit_fp_batch():
    fps = SmilesToRdkit()([ASPIRIN, CAFFEINE])
    assert fps.shape == (2, 4096)


def test_rdkit_fp_binary():
    fp = SmilesToRdkit()(ASPIRIN)
    assert set(np.unique(fp)).issubset({0, 1})


def test_maccs_fp_size():
    enc = SmilesToMACCS()
    assert enc.fp_size == 167
    assert enc(ASPIRIN).shape == (167,)


def test_maccs_batch():
    fps = SmilesToMACCS()([ASPIRIN, CAFFEINE])
    assert fps.shape == (2, 167)


def test_maccs_binary():
    fp = SmilesToMACCS()(ASPIRIN)
    assert set(np.unique(fp)).issubset({0, 1})


def test_biosynfoni_fp_size():
    enc = SmilesToBiosynfoni()
    assert enc.fp_size == 39
    assert enc(ASPIRIN).shape == (39,)


def test_biosynfoni_batch():
    fps = SmilesToBiosynfoni()([ASPIRIN, CAFFEINE])
    assert fps.shape == (2, 39)


def test_map4_single_shape():
    enc = SmilesToMAP4()
    fp = enc(ASPIRIN)
    assert fp.shape == (4096,)


def test_map4_fp_size():
    enc = SmilesToMAP4(fp_size=1024)
    assert enc(ASPIRIN).shape == (1024,)
    assert enc.fp_size == 1024


def test_map4_batch():
    fps = SmilesToMAP4()([ASPIRIN, CAFFEINE])
    assert fps.shape == (2, 4096)


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not installed",
)
def test_molformer_import():
    pytest.importorskip("transformers")
    from metabo_depthcharge.molecules import SmilesToMolFormer  # noqa: F401


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not installed",
)
def test_chemberta_import():
    pytest.importorskip("transformers")
    from metabo_depthcharge.molecules import SmilesToChemBERTa  # noqa: F401
