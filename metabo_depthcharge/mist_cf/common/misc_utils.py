# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""misc_utils.py"""

import copy
import logging
import sys
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


def get_data_dir(dataset_name):
    return Path("data/") / dataset_name


class ConsoleLogger(pl.loggers.Logger):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "console"

    @property
    def version(self):
        return "0"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics = copy.deepcopy(metrics)
        epoch_num = metrics.pop("epoch", "??")
        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass


class _LevelAwareFormatter(logging.Formatter):
    """Tags WARNING+ so they stand out; INFO (the common case) stays plain --
    a bare level tag on every line is just noise at INFO."""

    def format(self, record):
        fmt = (
            "%(asctime)s: %(message)s"
            if record.levelno == logging.INFO
            else "%(asctime)s %(levelname)s: %(message)s"
        )
        self._style._fmt = fmt
        return super().format(record)


def setup_logger(save_dir, log_name="output.log", debug=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_file = save_dir / log_name

    level = logging.DEBUG if debug is not False else logging.INFO
    formatter = _LevelAwareFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[stream_handler, file_handler])

    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))


def batches(it: Iterable, chunk_size: int) -> Iterator[list]:
    """Consume an iterable in batches of size chunk_size"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])
