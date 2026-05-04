import warnings

import numpy as np
from biosynfoni import Biosynfoni
from map4 import MAP4
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MolStandardize, SanitizeFlags


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string via RDKit."""
    mol = safe_mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol)


def safe_mol_from_smiles(smiles: str):
    # Use original MolFromSmiles if it was saved (to avoid recursion from monkey-patch)
    mol_from_smiles = getattr(Chem, "_original_MolFromSmiles", Chem.MolFromSmiles)

    mol = mol_from_smiles(smiles)
    if mol is not None:
        return mol

    mol = mol_from_smiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES even without sanitization: {smiles}")

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES,
        )
        warnings.warn(
            f"Fallback parsing (skip SANITIZE_PROPERTIES) used for SMILES: {smiles}",
            stacklevel=2,
        )
        return mol
    except Chem.KekulizeException:
        pass
    except Exception as e:
        raise ValueError(f"Invalid SMILES string: {smiles}") from e

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=SanitizeFlags.SANITIZE_ALL
            ^ SanitizeFlags.SANITIZE_PROPERTIES
            ^ SanitizeFlags.SANITIZE_KEKULIZE,
        )
        warnings.warn(
            f"Fallback parsing (skip SANITIZE_PROPERTIES and SANITIZE_KEKULIZE) used for SMILES: {smiles}",
            stacklevel=2,
        )
        return mol
    except Exception:
        pass

    warnings.warn(
        f"Fallback parsing (UNSANITIZED) used for SMILES: {smiles}", stacklevel=2
    )
    return mol


class SmilesToMorgan:
    """Convert SMILES strings to Morgan circular fingerprints.

    Parameters
    ----------
    fp_size: int
        Fingerprint size (default: 4096).
    radius: int
        Circular radius (default: 2).
    counts: bool
        Return count fingerprint instead of binary (default: False).
    """

    def __init__(
        self, fp_size: int = 4096, radius: int = 2, counts: bool = False
    ) -> None:
        self.fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )
        self.counts = counts
        self.fp_size = fp_size

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        # Handle single SMILES (str, numpy.str_, or any non-list/array)
        if not isinstance(smiles, list | np.ndarray):
            mol = safe_mol_from_smiles(smiles)
            if self.counts:
                return self.fpgen.GetCountFingerprintAsNumPy(mol)
            else:
                return self.fpgen.GetFingerprintAsNumPy(mol)
        else:
            return np.array([self(smi) for smi in smiles])


class SmilesToRdkit:
    """Convert SMILES strings to RDKit topological fingerprints.

    Parameters
    ----------
    fp_size: int
        Fingerprint size (default: 4096).
    counts: bool
        Return count fingerprint instead of binary (default: False).
    """

    def __init__(self, fp_size: int = 4096, counts: bool = False) -> None:
        self.fpgen = Chem.rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
        self.counts = counts
        self.fp_size = fp_size

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        # Handle single SMILES (str, numpy.str_, or any non-list/array)
        if not isinstance(smiles, list | np.ndarray):
            mol = safe_mol_from_smiles(smiles)
            if self.counts:
                return self.fpgen.GetCountFingerprintAsNumPy(mol)
            else:
                return self.fpgen.GetFingerprintAsNumPy(mol)
        else:
            return np.array([self(smi) for smi in smiles])


class SmilesToMACCS:
    """Convert SMILES strings to MACCS structural key fingerprints."""

    def __init__(self) -> None:
        self.fp_size = 167

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        # Handle single SMILES (str, numpy.str_, or any non-list/array)
        if not isinstance(smiles, list | np.ndarray):
            mol = safe_mol_from_smiles(smiles)

            fp = AllChem.GetMACCSKeysFingerprint(mol)
            fp_np = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            return fp_np
        else:
            return np.array([self(smi) for smi in smiles])


class SmilesToBiosynfoni:
    """Convert SMILES strings to Biosynfoni biochemically-informed fingerprints."""

    def __init__(self) -> None:
        self.fp_size = 39

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        # Handle single SMILES (str, numpy.str_, or any non-list/array)
        if not isinstance(smiles, list | np.ndarray):
            mol = safe_mol_from_smiles(smiles)
            fp = Biosynfoni(mol).fingerprint
            return np.array(fp)
        else:
            return np.array([self(smi) for smi in smiles])


class SmilesToMAP4:
    """Convert SMILES strings to MAP4 MinHashed atom-pair fingerprints.

    Parameters
    ----------
    fp_size: int
        Fingerprint size (default: 4096).
    radius: int
        Atom-pair radius (default: 2).
    """

    def __init__(self, fp_size: int = 4096, radius: int = 2) -> None:
        self.map_calc = MAP4(
            dimensions=fp_size, radius=radius, include_duplicated_shingles=False
        )
        self.fp_size = fp_size

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        # Handle single SMILES (str, numpy.str_, or any non-list/array)
        if not isinstance(smiles, list | np.ndarray):
            mol = safe_mol_from_smiles(smiles)
            fp = self.map_calc.calculate(mol)
            return fp
        else:
            return np.array([self(smi) for smi in smiles])


class SmilesToUniMol:
    """Convert SMILES strings to Uni-Mol pretrained embeddings.

    Parameters
    ----------
    remove_hs: bool
        Remove hydrogen atoms from representation (default: False).
    device: str
        Device to run model on: 'cpu' or 'cuda' (default: 'cpu').
    """

    def __init__(self, remove_hs: bool = False, device: str = "cpu") -> None:
        # Monkey-patch rdkit.Chem.MolFromSmiles BEFORE importing unimol_tools
        # This makes UniMol use our safe_mol_from_smiles instead
        import rdkit.Chem as Chem

        if not hasattr(Chem, "_original_MolFromSmiles"):
            # Save original and patch with safe version
            Chem._original_MolFromSmiles = Chem.MolFromSmiles
            Chem.MolFromSmiles = lambda smi, *args, **kwargs: safe_mol_from_smiles(smi)

        # Now import UniMol - it will use the patched version
        from unimol_tools import UniMolRepr

        if device == "cpu":
            use_cuda = False
            use_gpu = None
        elif device.startswith("cuda"):
            use_cuda = True
            if ":" in device:
                gpu_idx = device.split(":")[1]
                use_gpu = gpu_idx
            else:
                use_gpu = "all"
        else:
            raise ValueError(
                f"Invalid device: {device}. Expected 'cpu', 'cuda', or 'cuda:X'"
            )

        self.device = device
        self.fp_size = 512
        self.model = UniMolRepr(
            data_type="molecule",
            remove_hs=remove_hs,
            use_cuda=use_cuda,
            use_gpu=use_gpu,
        )

    def _sanitize_smiles(self, smiles: str) -> str:
        """Clean SMILES for UniMol: remove salts, take largest fragment."""
        try:
            mol = safe_mol_from_smiles(smiles)
            remover = MolStandardize.rdMolStandardize.LargestFragmentChooser()
            mol = remover.choose(mol)
            clean_smiles = Chem.MolToSmiles(mol)
            if Chem.MolFromSmiles(clean_smiles) is not None:
                return clean_smiles

            if "." in smiles:
                first_component = smiles.split(".")[0]
                if Chem.MolFromSmiles(first_component) is not None:
                    return first_component
            raise ValueError(f"Cannot create UniMol-compatible SMILES: {smiles}")

        except Exception as e:
            if "." in smiles:
                first_component = smiles.split(".")[0]
                if Chem.MolFromSmiles(first_component) is not None:
                    return first_component

            raise ValueError(f"Cannot create UniMol-compatible SMILES: {smiles}") from e

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        # Handle single SMILES (str, numpy.str_, or any non-list/array)
        is_single = not isinstance(smiles, list | np.ndarray)
        smiles_list = [
            canonicalize_smiles(s) for s in ([smiles] if is_single else smiles)
        ]

        # UniMol will use our monkey-patched safe_mol_from_smiles internally
        unimol_repr = self.model.get_repr(smiles_list)

        if is_single:
            return unimol_repr[0]
        else:
            return np.array(unimol_repr)


class SmilesToMolFormer:
    """Convert SMILES strings to MolFormer pretrained embeddings.

    Parameters
    ----------
    model_name: str
        HuggingFace model name (default: 'ibm-research/MoLFormer-XL-both-10pct').
        Examples: 'ibm-research/MoLFormer-XL-both-10pct', 'DeepChem/MoLFormer-c3-1.1B'
    device: str
        Device to run model on: 'cpu' or 'cuda' (default: 'cpu').
    """

    def __init__(
        self,
        model_name: str = "ibm-research/MoLFormer-XL-both-10pct",
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device

        if device == "cpu":
            self.torch_device = torch.device("cpu")
        elif device.startswith("cuda"):
            if ":" in device:
                self.torch_device = torch.device(device)
            else:
                self.torch_device = torch.device("cuda")
        else:
            raise ValueError(
                f"Invalid device: {device}. Expected 'cpu', 'cuda', or 'cuda:X'"
            )

        # IBM MolFormer models require trust_remote_code for custom linear attention
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, deterministic_eval=True
        )
        self.model.to(self.torch_device)
        self.model.eval()

        self.fp_size = self.model.config.hidden_size

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        import torch

        is_single = not isinstance(smiles, list | np.ndarray)
        smiles_list = [
            canonicalize_smiles(s) for s in ([smiles] if is_single else smiles)
        ]

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
            # Use pooler_output if available (IBM MolFormer), else CLS token
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output.cpu().numpy()
            else:
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        del inputs
        del outputs
        if self.torch_device.type == "cuda":
            torch.cuda.empty_cache()

        if is_single:
            return embeddings[0].astype(np.float32)
        else:
            return embeddings.astype(np.float32)


class SmilesToChemBERTa:
    """Convert SMILES strings to ChemBERTa pretrained embeddings.

    Parameters
    ----------
    model_name: str
        HuggingFace model name (default: 'seyonec/ChemBERTa-zinc-base-v1').
        Examples: 'seyonec/ChemBERTa-zinc-base-v1', 'DeepChem/ChemBERTa-77M-MLM',
                  'Derify/ChemBERTa_augmented_pubchem_13m'
    device: str
        Device to run model on: 'cpu' or 'cuda' (default: 'cpu').
    pooling: str
        Pooling strategy: 'cls' (use CLS token) or 'mean' (mean pooling) (default: 'cls').
    """

    def __init__(
        self,
        model_name: str = "Derify/ChemBERTa_augmented_pubchem_13m",
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device

        if device == "cpu":
            self.torch_device = torch.device("cpu")
        elif device.startswith("cuda"):
            if ":" in device:
                self.torch_device = torch.device(device)
            else:
                self.torch_device = torch.device("cuda")
        else:
            raise ValueError(
                f"Invalid device: {device}. Expected 'cpu', 'cuda', or 'cuda:X'"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.torch_device)
        self.model.eval()

        self.fp_size = self.model.config.hidden_size

    def __call__(self, smiles: str | list[str]) -> np.ndarray:
        is_single = not isinstance(smiles, list | np.ndarray)
        smiles_list = [
            canonicalize_smiles(s) for s in ([smiles] if is_single else smiles)
        ]

        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        del inputs
        del outputs
        if self.torch_device.type == "cuda":
            torch.cuda.empty_cache()

        if is_single:
            return embeddings[0].astype(np.float32)
        else:
            return embeddings.astype(np.float32)
