"""Shared loaders: features from the one-pass cache + the repo's own grouped folds."""
import os, sys
import numpy as np
import pandas as pd

REPO = "/Users/jerlshin/FieldOfInterest/ResearchWork/HSI_RGB_seeds/Code"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
sys.path.insert(0, os.path.join(REPO, "src"))

from spectralquadnet.data.loaders import grouped_split  # noqa: E402

LABELS = np.load(f"{REPO}/dataset/labels.npy").astype(np.int64)
GROUPS = np.load(f"{REPO}/dataset/groups.npy").astype(np.int64)
WL = pd.read_csv(f"{REPO}/dataset/wavelengths.csv").iloc[:, -1].values.astype(np.float64)
C = 256


def fold(f):
    """train / calib / heldout rows, built by the training pipeline's own split builder
    with the shipped parameters (grouped, eval_frac=0.3, calib_frac=0.15)."""
    b = grouped_split(LABELS, GROUPS, eval_frac=0.3, calib_frac=0.15, fold=f, single_group_policy="error")
    held = np.sort(np.concatenate([b.val, b.test]))
    return np.asarray(b.train), np.asarray(b.calib), held


def load(name, mmap=False):
    return np.load(os.path.join(CACHE, f"{name}.npy"), mmap_mode="r" if mmap else None)
