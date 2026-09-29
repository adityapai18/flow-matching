"""Shared paths, logging, and JSON helpers."""
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_DIR = RESULTS / "synthetic"
RUNS_DIR = RESULTS / "runs"
LOG_PATH = RESULTS / "run.log"

D_LEVELS = [10, 30, 60, 90, 120, 150, 175]
MODELS = ["M1_riemannian", "M2_euclid6d", "M3_quat", "M4_euler"]
EVAL_SEEDS = [0, 1, 2, 3, 4]
NFES = [1, 2, 4, 8, 16, 32, 64]
N_WAYPOINTS = 32


def get_logger(name="se3"):
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter(f"%(asctime)s [pid{os.getpid()}] %(levelname)s %(message)s")
        fh = logging.FileHandler(LOG_PATH, mode="a")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path, obj):
    """Atomic JSON write (tmp + rename) so a crash never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)
