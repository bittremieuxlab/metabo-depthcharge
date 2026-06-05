# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""train.py

Train fast formula filter: a 3-layer MLP that scores a single molecular formula
on chemical/biological plausibility (no spectrum input). Trained per-mass
contrastively against decoys produced by SIRIUS decomp at the same mass.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from .. import common
from . import fast_form_data, fast_form_model


def add_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)

    parser.add_argument("--learning-rate", default=0.00036, type=float)
    parser.add_argument("--lr-decay-frac", default=0.86425, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)

    parser.add_argument("--max-decoy", default=32, type=int)
    parser.add_argument("--max-epochs", default=200, type=int)

    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_fast_form_score/")

    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--split-file", required=True)

    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--form-encoder", type=str, default="abs-sines")
    parser.add_argument("--patience", default=5, type=int)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = vars(args)

    save_dir = kwargs["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    common.setup_logger(save_dir, log_name="ffn_train.log", debug=kwargs["debug"])
    pl.seed_everything(kwargs["seed"])

    with open(Path(save_dir) / "args.yaml", "w") as fp:
        yaml.dump(kwargs, fp, indent=2, default_flow_style=False)
    logging.info(f"Args:\n{yaml.dump(kwargs, indent=2, default_flow_style=False)}")

    df = pd.read_csv(kwargs["dataset_file"], sep="\t").fillna("")
    if kwargs["debug"]:
        df = df[:1000]

    masses = df["mass"].values
    train_inds, val_inds, test_inds = common.get_splits(
        masses,
        kwargs["split_file"],
        key="mass",
    )
    train_df, val_df, test_df = (
        df.iloc[train_inds],
        df.iloc[val_inds],
        df.iloc[test_inds],
    )
    logging.info(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_dataset = fast_form_data.FormDataset(
        train_df,
        num_workers=kwargs["num_workers"],
        decoys_per_pos=kwargs["max_decoy"],
    )
    val_dataset = fast_form_data.FormDataset(
        val_df,
        num_workers=kwargs["num_workers"],
        decoys_per_pos=kwargs["max_decoy"],
        val_test=True,
    )
    test_dataset = fast_form_data.FormDataset(
        test_df,
        num_workers=kwargs["num_workers"],
        decoys_per_pos=kwargs["max_decoy"],
        val_test=True,
    )

    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model = fast_form_model.FastFFN(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        dropout=kwargs["dropout"],
        learning_rate=kwargs["learning_rate"],
        lr_decay_frac=kwargs["lr_decay_frac"],
        weight_decay=kwargs["weight_decay"],
        form_encoder=kwargs["form_encoder"],
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()
    tb_path = tb_logger.log_dir

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_path,
        filename="best",
        save_weights_only=True,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=kwargs["patience"])

    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        devices=1,
        callbacks=[earlystop_callback, checkpoint_callback],
        gradient_clip_val=5,
        gradient_clip_algorithm="value",
        max_epochs=kwargs["max_epochs"],
    )

    trainer.fit(model, train_loader, val_loader)
    best_path = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score.item()
    logging.info(f"Best ckpt: {best_path}  (val_loss={best_score:.4f})")

    model = fast_form_model.FastFFN.load_from_checkpoint(best_path)
    model.eval()
    test_out = trainer.test(model, dataloaders=test_loader)

    with open(Path(save_dir) / "test_results.yaml", "w") as fp:
        yaml.dump(
            {
                "args": kwargs,
                "test_metrics": test_out[0],
                "best_ckpt": best_path,
                "best_val_loss": best_score,
            },
            fp,
            indent=2,
            default_flow_style=False,
        )


if __name__ == "__main__":
    import time

    t0 = time.time()
    train_model()
    logging.info(f"Done in {time.time() - t0:.1f} s")
