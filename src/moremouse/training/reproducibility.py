"""Reproducibility helpers."""

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy RNGs.

    Parameters
    ----------
    seed:
        Non-negative random seed.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)

