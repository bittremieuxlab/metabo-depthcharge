# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""splitter.py"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def get_splits(names, split_file, key="spec"):
    if not Path(split_file).exists():
        logging.info(f"Unable to find {split_file}")
        raise ValueError()

    split_df = pd.read_csv(split_file, sep="\t")

    folds = set(split_df.columns)
    folds.remove(key)
    num_folds = len(folds)
    folds = sorted(folds)
    if len(folds) == 0:
        raise ValueError
    elif len(folds) > 1:
        logging.info(f"Found {num_folds} folds; choosing one")

    fold = folds[0]
    fold_entries = split_df[fold]
    names_to_index = dict(zip(names, np.arange(len(names)), strict=False))
    train_entries = fold_entries == "train"
    test_entries = fold_entries == "test"
    val_entries = fold_entries == "val"
    train_inds = [
        names_to_index.get(i)
        for i in split_df[key][train_entries]
        if i in names_to_index
    ]
    test_inds = np.array(
        [
            names_to_index.get(i)
            for i in split_df[key][test_entries]
            if i in names_to_index
        ]
    )
    val_inds = np.array(
        [
            names_to_index.get(i)
            for i in split_df[key][val_entries]
            if i in names_to_index
        ]
    )

    def convert(x):
        return np.array(list(x))

    return convert(train_inds), convert(val_inds), convert(test_inds)
