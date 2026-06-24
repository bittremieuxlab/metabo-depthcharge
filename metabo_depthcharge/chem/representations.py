"""Numerical molecular representation generator classes."""

import functools
from collections.abc import Iterable

import numpy as np
import torch
from biosynfoni import Biosynfoni
from map4 import MAP4
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from transformers import AutoModel, AutoTokenizer

from metabo_depthcharge.chem.molecule import Molecule, _lenient_mol_from_smiles


def _batched(fn):
    @functools.wraps(fn)
    def wrapper(self, mol):
        if isinstance(mol, Molecule):
            return fn(self, mol)
        return np.array([fn(self, m) for m in mol])

    return wrapper


class MoleculeToMorgan:
    """Morgan circular fingerprints.

    Parameters
    ----------
    rep_size : int, default 4096
        Hashed fingerprint length.
    radius : int, default 2
        Circular substructure radius.
    counts : bool, default False
        Return integer counts. Otherwise binary 0/1.
    """

    def __init__(self, rep_size: int = 4096, radius: int = 2, counts: bool = False):
        self.fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=rep_size
        )
        self.counts = counts
        self.rep_size = rep_size

    @_batched
    def __call__(self, mol: Molecule | Iterable[Molecule]) -> np.ndarray:
        """Compute the Morgan fingerprint.

        Parameters
        ----------
        mol : Molecule or iterable of Molecule
            A single molecule or an iterable of molecules.

        Returns
        -------
        np.ndarray
            ``(rep_size,)`` for a single molecule, or ``(N, rep_size)`` for an
            iterable of ``N`` molecules.
        """
        if self.counts:
            return self.fpgen.GetCountFingerprintAsNumPy(mol.mol)
        return self.fpgen.GetFingerprintAsNumPy(mol.mol)


class MoleculeToRdkit:
    """RDKit topological (path-based) fingerprints.

    Parameters
    ----------
    rep_size : int, default 4096
        Hashed fingerprint length.
    counts : bool, default False
        Return integer counts. Otherwise binary 0/1.
    """

    def __init__(self, rep_size: int = 4096, counts: bool = False):
        self.fpgen = Chem.rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=rep_size)
        self.counts = counts
        self.rep_size = rep_size

    @_batched
    def __call__(self, mol: Molecule | Iterable[Molecule]) -> np.ndarray:
        """Compute the RDKit topological fingerprint.

        Parameters
        ----------
        mol : Molecule or iterable of Molecule
            A single molecule or an iterable of molecules.

        Returns
        -------
        np.ndarray
            ``(rep_size,)`` for a single molecule, or ``(N, rep_size)`` for an
            iterable of ``N`` molecules.
        """
        if self.counts:
            return self.fpgen.GetCountFingerprintAsNumPy(mol.mol)
        return self.fpgen.GetFingerprintAsNumPy(mol.mol)


class MoleculeToMACCS:
    """MACCS structural keys (167 binary bits).

    Takes no constructor arguments.
    """

    rep_size = 167

    @_batched
    def __call__(self, mol: Molecule | Iterable[Molecule]) -> np.ndarray:
        """Compute the 167-bit MACCS key fingerprint.

        Parameters
        ----------
        mol : Molecule or iterable of Molecule
            A single molecule or an iterable of molecules.

        Returns
        -------
        np.ndarray
            ``(167,)`` for a single molecule, or ``(N, 167)`` for an iterable
            of ``N`` molecules.
        """
        rep = AllChem.GetMACCSKeysFingerprint(mol.mol)
        out = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(rep, out)
        return out


class MoleculeToBiosynfoni:
    """Biosynfoni biochemically-informed fingerprint (39 count dims).

    Takes no constructor arguments.
    """

    rep_size = 39

    @_batched
    def __call__(self, mol: Molecule | Iterable[Molecule]) -> np.ndarray:
        """Compute the Biosynfoni fingerprint.

        Parameters
        ----------
        mol : Molecule or iterable of Molecule
            A single molecule or an iterable of molecules.

        Returns
        -------
        np.ndarray
            ``(39,)`` for a single molecule, or ``(N, 39)`` for an iterable of
            ``N`` molecules.
        """
        return np.array(Biosynfoni(mol.mol).fingerprint)


class MoleculeToMAP4:
    """MAP4 MinHashed atom-pair fingerprint.

    Computed via the `map4 <https://pypi.org/project/map4/>`_ package.

    Parameters
    ----------
    rep_size : int, default 4096
        MinHash dimensionality.
    radius : int, default 2
        Atom-pair shingle radius.
    """

    def __init__(self, rep_size: int = 4096, radius: int = 2):
        self.map_calc = MAP4(
            dimensions=rep_size, radius=radius, include_duplicated_shingles=False
        )
        self.rep_size = rep_size

    @_batched
    def __call__(self, mol: Molecule | Iterable[Molecule]) -> np.ndarray:
        """Compute the MAP4 fingerprint.

        Parameters
        ----------
        mol : Molecule or iterable of Molecule
            A single molecule or an iterable of molecules.

        Returns
        -------
        np.ndarray
            ``(rep_size,)`` for a single molecule, or ``(N, rep_size)`` for an
            iterable of ``N`` molecules.
        """
        return self.map_calc.calculate(
            _lenient_mol_from_smiles(mol.canonical_smiles)
        )


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
        self.rep_size = self.model.config.hidden_size

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
        """Embed molecule SMILES into a fixed-size vector.

        Parameters
        ----------
        mol : Molecule or iterable of Molecule
            A single molecule or an iterable of molecules.

        Returns
        -------
        np.ndarray
            ``(rep_size,)`` ``float32`` for a single molecule, or
            ``(N, rep_size)`` for an iterable of ``N`` molecules.
        """
        if isinstance(mol, Molecule):
            return self._embed([mol.canonical_smiles])[0]
        return self._embed([m.canonical_smiles for m in mol])


class MoleculeToMolFormer(_HFEmbedder):
    """IBM MolFormer pretrained embeddings.

    Uses ``outputs.pooler_output`` if available (the IBM model exposes one),
    else the CLS token.

    Parameters
    ----------
    model_name : str, default ``"ibm-research/MoLFormer-XL-both-10pct"``
        HuggingFace model id of a MolFormer checkpoint. Requires
        ``trust_remote_code=True`` for the custom linear-attention module.
        The default is the IBM XL model trained on 10% of ZINC + PubChem
        (``"ibm/MoLFormer-XL-both-10pct"``);
        Another possible checkpoint is e.g., ``"DeepChem/MoLFormer-c3-1.1B"``.
    device : str, default ``"cpu"``
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
    """

    def __init__(
        self,
        model_name: str = "ibm-research/MoLFormer-XL-both-10pct",
        device: str = "cpu",
    ):
        super().__init__(
            model_name, device, trust_remote_code=True, pooling="pooler_or_cls"
        )

    # Re-expose the inherited call operator so it is documented on this class.
    __call__ = _HFEmbedder.__call__


class MoleculeToChemBERTa(_HFEmbedder):
    """ChemBERTa pretrained embeddings.

    Always uses the CLS token from ``last_hidden_state`` (ignores any pooler
    the model exposes).

    Parameters
    ----------
    model_name : str, default ``"Derify/ChemBERTa_augmented_pubchem_13m"``
        Any HuggingFace Hub id for a RoBERTa/BERT-style SMILES encoder
        works, since only the CLS token of ``last_hidden_state`` is pooled.
        Other public checkpoints include ``"DeepChem/ChemBERTa-77M-MLM"``,
        ``"DeepChem/ChemBERTa-77M-MTR"``, and
        ``"seyonec/ChemBERTa-zinc-base-v1"``.
    device : str, default ``"cpu"``
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
    """

    def __init__(
        self,
        model_name: str = "Derify/ChemBERTa_augmented_pubchem_13m",
        device: str = "cpu",
    ):
        super().__init__(model_name, device, trust_remote_code=False, pooling="cls")

    # Re-expose the inherited call operator so it is documented on this class.
    __call__ = _HFEmbedder.__call__
