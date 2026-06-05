"""Numerical molecular representation generator classes."""

from collections.abc import Iterable

import numpy as np
import torch
from biosynfoni import Biosynfoni
from map4 import MAP4
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from transformers import AutoModel, AutoTokenizer

from metabo_depthcharge.chem.molecule import Molecule


def _batched(fn):
    def wrapper(self, mol):
        if isinstance(mol, Molecule):
            return fn(self, mol)
        return np.array([fn(self, m) for m in mol])

    return wrapper


class MoleculeToMorgan:
    """Morgan circular fingerprints.

    Parameters
    ----------
    fp_size : int, default 4096
        Hashed fingerprint length.
    radius : int, default 2
        Circular substructure radius.
    counts : bool, default False
        Return integer counts; otherwise binary 0/1.

    Notes
    -----
    Callable. A single :class:`Molecule` returns ``(fp_size,)``; an
    iterable of molecules returns ``(N, fp_size)``.
    """

    def __init__(self, fp_size: int = 4096, radius: int = 2, counts: bool = False):
        self.fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )
        self.counts = counts
        self.fp_size = fp_size

    @_batched
    def __call__(self, mol: Molecule) -> np.ndarray:
        if self.counts:
            return self.fpgen.GetCountFingerprintAsNumPy(mol.mol)
        return self.fpgen.GetFingerprintAsNumPy(mol.mol)


class MoleculeToRdkit:
    """RDKit topological (path-based) fingerprints.

    Parameters
    ----------
    fp_size : int, default 4096
        Hashed fingerprint length.
    counts : bool, default False
        Return integer counts; otherwise binary 0/1.

    Notes
    -----
    Callable. A single :class:`Molecule` returns ``(fp_size,)``; an
    iterable of molecules returns ``(N, fp_size)``.
    """

    def __init__(self, fp_size: int = 4096, counts: bool = False):
        self.fpgen = Chem.rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
        self.counts = counts
        self.fp_size = fp_size

    @_batched
    def __call__(self, mol: Molecule) -> np.ndarray:
        if self.counts:
            return self.fpgen.GetCountFingerprintAsNumPy(mol.mol)
        return self.fpgen.GetFingerprintAsNumPy(mol.mol)


class MoleculeToMACCS:
    """MACCS structural keys (167 binary bits).

    Notes
    -----
    Callable. A single :class:`Molecule` returns ``(167,)``; an iterable
    of molecules returns ``(N, 167)``.
    """

    fp_size = 167

    @_batched
    def __call__(self, mol: Molecule) -> np.ndarray:
        fp = AllChem.GetMACCSKeysFingerprint(mol.mol)
        out = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, out)
        return out


class MoleculeToBiosynfoni:
    """Biosynfoni biochemically-informed fingerprint (39 count dims).

    Notes
    -----
    Callable. A single :class:`Molecule` returns ``(39,)``; an iterable
    of molecules returns ``(N, 39)``.
    """

    fp_size = 39

    @_batched
    def __call__(self, mol: Molecule) -> np.ndarray:
        return np.array(Biosynfoni(mol.mol).fingerprint)


class MoleculeToMAP4:
    """MAP4 MinHashed atom-pair fingerprint.

    Parameters
    ----------
    fp_size : int, default 4096
        MinHash dimensionality.
    radius : int, default 2
        Atom-pair shingle radius.

    Notes
    -----
    Callable. A single :class:`Molecule` returns ``(fp_size,)``; an
    iterable of molecules returns ``(N, fp_size)``.
    """

    def __init__(self, fp_size: int = 4096, radius: int = 2):
        self.map_calc = MAP4(
            dimensions=fp_size, radius=radius, include_duplicated_shingles=False
        )
        self.fp_size = fp_size

    @_batched
    def __call__(self, mol: Molecule) -> np.ndarray:
        return self.map_calc.calculate(mol.mol)


class _HFEmbedder:
    """Shared base class for Hugging Face SMILES embedders."""

    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        trust_remote_code: bool = False,
        pooling: str = "pooler_or_cls",
    ):
        if device == "cpu":
            self.torch_device = torch.device("cpu")
        elif device.startswith("cuda"):
            self.torch_device = torch.device(device if ":" in device else "cuda")
        else:
            raise ValueError(f"Invalid device: {device}")
        if pooling not in ("pooler_or_cls", "cls"):
            raise ValueError(f"Invalid pooling: {pooling!r}")

        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        kw = {"trust_remote_code": True} if trust_remote_code else {}
        model_kw = {"deterministic_eval": True} if trust_remote_code else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        self.model = AutoModel.from_pretrained(model_name, **kw, **model_kw)
        self.model.to(self.torch_device).eval()
        self.fp_size = self.model.config.hidden_size

    def _embed(self, smiles_list: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            if (
                self.pooling == "pooler_or_cls"
                and getattr(outputs, "pooler_output", None) is not None
            ):
                emb = outputs.pooler_output.cpu().numpy()
            else:
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        del inputs
        del outputs
        if self.torch_device.type == "cuda":
            torch.cuda.empty_cache()
        return emb.astype(np.float32)

    def __call__(self, mol: Molecule | Iterable[Molecule]) -> np.ndarray:
        if isinstance(mol, Molecule):
            return self._embed([mol.canonical_smiles])[0]
        return self._embed([m.canonical_smiles for m in mol])


class MoleculeToMolFormer(_HFEmbedder):
    """IBM MolFormer pretrained embeddings.

    Parameters
    ----------
    model_name : str, default ``"ibm-research/MoLFormer-XL-both-10pct"``
        HuggingFace model id. Requires ``trust_remote_code=True`` for
        IBM's custom linear-attention module.
    device : str, default ``"cpu"``
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.

    Notes
    -----
    Callable. A single :class:`Molecule` returns ``(fp_size,)``; an
    iterable of molecules returns ``(N, fp_size)``. Uses
    ``outputs.pooler_output`` if available (the IBM model exposes one),
    else CLS.
    """

    def __init__(
        self,
        model_name: str = "ibm-research/MoLFormer-XL-both-10pct",
        device: str = "cpu",
    ):
        super().__init__(
            model_name, device, trust_remote_code=True, pooling="pooler_or_cls"
        )


class MoleculeToChemBERTa(_HFEmbedder):
    """ChemBERTa pretrained embeddings.

    Parameters
    ----------
    model_name : str, default ``"Derify/ChemBERTa_augmented_pubchem_13m"``
        HuggingFace model id.
    device : str, default ``"cpu"``
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.

    Notes
    -----
    Callable. A single :class:`Molecule` returns ``(fp_size,)``; an
    iterable of molecules returns ``(N, fp_size)``. Always uses CLS
    from ``last_hidden_state`` (ignores any pooler the model exposes).
    """

    def __init__(
        self,
        model_name: str = "Derify/ChemBERTa_augmented_pubchem_13m",
        device: str = "cpu",
    ):
        super().__init__(model_name, device, trust_remote_code=False, pooling="cls")
