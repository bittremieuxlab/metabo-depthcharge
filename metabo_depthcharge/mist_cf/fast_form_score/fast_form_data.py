# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""fast_form_data.py"""

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from .. import common


def _extract_ar(form_str):
    return [i for i in form_str.split(",") if len(i) > 0]


class FormDataset(Dataset):
    """Training dataset: per-mass groups of (1 positive formula, K decoy formulas)."""

    def __init__(self, df, num_workers=0, decoys_per_pos=16, val_test=False, **kwargs):
        self.df = df
        self.num_workers = num_workers
        self.decoys_per_pos = decoys_per_pos
        self.val_test = val_test

        self.pos_str = self.df["pos"].values
        self.neg_str = self.df["neg"].values
        self.masses = self.df["mass"].values

        self.pos_ar = [_extract_ar(s) for s in tqdm(self.pos_str, desc="parse pos")]
        self.neg_ar = [_extract_ar(s) for s in tqdm(self.neg_str, desc="parse neg")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pos = self.pos_ar[idx]
        neg = self.neg_ar[idx]

        pos = np.random.choice(pos)
        sample_num = min(len(neg), self.decoys_per_pos)
        if self.val_test:
            neg = neg[:sample_num]
        else:
            neg = (
                np.random.choice(neg, sample_num, replace=False)
                if sample_num > 0
                else []
            )

        ars = [common.formula_to_dense(pos)] + [common.formula_to_dense(i) for i in neg]
        labels = [1] + [0] * len(neg)
        return {"x": ars, "y": labels}

    @classmethod
    def get_collate_fn(cls):
        return FormDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        x_vals = torch.FloatTensor([i for j in input_list for i in j["x"]])
        y_vals = torch.LongTensor([i for j in input_list for i in j["y"]])
        return {"x": x_vals, "y": y_vals}


class PredDataset(Dataset):
    """Dataset for fast filter prediction/inference."""

    def __init__(self, df, num_workers=0, **kwargs):
        self.df = df
        self.num_workers = num_workers

        self.cand_forms = self.df["cand_form"].values
        self.cand_ions = self.df["cand_ion"].values

        if self.num_workers == 0:
            self.embedded_forms = [
                common.formula_to_dense(i) for i in tqdm(self.cand_forms)
            ]
        else:
            self.embedded_forms = common.chunked_parallel(
                self.cand_forms,
                common.formula_to_dense,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
        self.names = self.df["spec"].values

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        spec_name = self.names[idx]
        ar = self.embedded_forms[idx]
        form = self.cand_forms[idx]
        ion = self.cand_ions[idx]

        return {
            "spec": spec_name,
            "form": form,
            "x": ar,
            "ion": ion,
        }

    @classmethod
    def get_collate_fn(cls):
        return PredDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        str_forms = np.array([i["form"] for i in input_list])
        x = torch.FloatTensor(np.array([i["x"] for i in input_list]))
        names = np.array([i["spec"] for i in input_list])
        ions = np.array([i["ion"] for i in input_list])
        return {"names": names, "str_forms": str_forms, "x": x, "ions": ions}
