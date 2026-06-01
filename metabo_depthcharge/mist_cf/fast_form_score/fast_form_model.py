# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""fast_form_model.py"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .. import nn_utils
from . import fast_form_data


class FastFFN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        formula_size: int = 17,
        learning_rate: float = 7e-4,
        lr_decay_frac: float = 1.0,
        weight_decay: float = 0.0,
        form_encoder: str = "abs-sines",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_frac = lr_decay_frac
        self.weight_decay = weight_decay
        self.form_embedder = nn_utils.get_embedder(form_encoder)
        self.input_dim = self.form_embedder.full_dim

        self.mlp = nn_utils.MLPBlocks(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=self.layers,
        )
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, formulae):
        inputs = self.form_embedder(formulae)
        output = self.mlp(inputs)
        output = self.output_layer(output)
        output = self.output_activation(output)
        return output.squeeze()

    def _common_step(self, batch, name="train"):
        x, y = batch["x"], batch["y"].float()
        model_outs = self.forward(x.float())
        bce_loss = F.binary_cross_entropy(model_outs, y)
        self.log(f"{name}_loss", bce_loss)
        return {"loss": bce_loss}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, name="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer, lr_decay_rate=self.lr_decay_frac
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }

    def fast_filter_score(
        self,
        spec,
        decoy_ion_lst,
        decoy_ions,
        device: torch.device,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        """Run the NN forward pass over every (cand_form, cand_ion) and return
        raw plausibility scores in input order.

        Kept separate from selection so callers that sample multiple times from
        the same candidate set (e.g. one set for training decoys, another for
        honest pred candidates) can score once and argsort twice.
        """
        if len(decoy_ion_lst) == 0:
            return np.array([], dtype=np.float32)
        num = len(decoy_ion_lst)
        data = {
            "spec": [spec] * num,
            "cand_form": decoy_ion_lst,
            "cand_ion": decoy_ions,
        }
        label_df = pd.DataFrame.from_dict(data)

        pred_dataset = fast_form_data.PredDataset(label_df, num_workers=num_workers)
        collate_fn = pred_dataset.get_collate_fn()
        pred_loader = DataLoader(
            pred_dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=batch_size,
        )

        self.eval()
        self = self.to(device)

        out_scores = []
        with torch.no_grad():
            for batch in pred_loader:
                x = batch["x"].to(device)
                model_outs = self.forward(x.float())
                scores = model_outs.squeeze().cpu().numpy()
                out_scores.extend(np.atleast_1d(scores).reshape(-1))

        return np.asarray(out_scores)

    def fast_filter_sampling(
        self,
        spec,
        decoy_ion_lst,
        decoy_ions,
        max_decoy,
        device: torch.device,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        """Score candidate formulae and return indices of top-k most plausible ones."""
        if len(decoy_ion_lst) == 0:
            return []
        out_scores = self.fast_filter_score(
            spec,
            decoy_ion_lst,
            decoy_ions,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        # Higher score = more plausible formula, keep top-k
        sorted_idx = np.argsort(out_scores)[::-1]
        sorted_idx = sorted_idx[:max_decoy]
        return sorted_idx
