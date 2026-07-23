"""Molecular tokenization converters (SAFE, SMILES, etc.)."""

import importlib.util
import io
import json
import sys
import types
from contextlib import redirect_stderr

from huggingface_hub import hf_hub_download
from rdkit import Chem
from tokenizers import Regex, Tokenizer
from tokenizers.pre_tokenizers import Split
from transformers import AutoModel, PreTrainedTokenizerFast


SAFE_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
MODEL_NAME = "datamol-io/safe-gpt"


def load_tokenizer():
    """Load SAFE-GPT tokenizer with custom pre-tokenizer."""
    with open(hf_hub_download(MODEL_NAME, "tokenizer.json")) as f:
        data = json.load(f)
    attrs = data.get("tokenizer_attrs", {})
    for k in ("custom_pre_tokenizer", "tokenizer_type", "tokenizer_attrs"):
        data.pop(k, None)
    raw = Tokenizer.from_str(json.dumps(data))
    raw.pre_tokenizer = Split(Regex(SAFE_PATTERN), behavior="isolated")
    special = (
        "pad_token",
        "unk_token",
        "bos_token",
        "eos_token",
        "cls_token",
        "sep_token",
        "mask_token",
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=raw, **{k: attrs[k] for k in special if k in attrs}
    )


def load_model(device: str, dtype):
    """Load SAFE-GPT model."""
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()
    return model


def smiles_to_safe_strings(smiles_list: list[str]) -> list:
    """Convert SMILES list to SAFE strings (or SMILES if unfragmentable)."""
    if "safe.converter" not in sys.modules:
        spec = importlib.util.find_spec("safe")
        stub = types.ModuleType("safe")
        stub.__path__ = list(spec.submodule_search_locations)
        sys.modules["safe"] = stub

    # Suppress SAFE converter warnings by redirecting stderr for the entire operation
    with redirect_stderr(io.StringIO()):
        from safe.converter import SAFEConverter

        converter = SAFEConverter()

        safe_strs = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                safe_strs.append(
                    converter.encoder(Chem.MolToSmiles(mol), allow_empty=True)
                    if mol
                    else None
                )
            except Exception:
                safe_strs.append(None)

    return safe_strs


def smiles_to_tokens(smiles_list: list[str], tokenizer) -> dict:
    """Convert SMILES to tokenized SAFE strings.

    Returns dict with 'input_ids', 'attention_mask', and 'keep_idx' (indices of valid molecules).
    """
    safe_strs = smiles_to_safe_strings(smiles_list)
    keep_idx = [i for i, s in enumerate(safe_strs) if s]

    if not keep_idx:
        return {"input_ids": None, "attention_mask": None, "keep_idx": []}

    inputs = tokenizer(
        [safe_strs[i] for i in keep_idx],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
        add_special_tokens=False,
    )

    inputs["keep_idx"] = keep_idx
    return inputs
