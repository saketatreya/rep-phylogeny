"""Test-sentence bootstrap utilities used by all three tiers.

The unit of resampling is a sentence index in ``[0, N_TEST)``. Every per-
sentence statistic (residuals, sentence-level classifier predictions,
sentence-level feature vectors) is precomputed once, then the bootstrap
loop just gathers and averages — so 1000 resamples cost very little.
"""
from __future__ import annotations
import numpy as np


def bootstrap_indices(n_test: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """Return an (n_boot, n_test) matrix of sampled indices."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_test, size=(n_boot, n_test))


def ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    lo = float(np.quantile(values, alpha / 2))
    hi = float(np.quantile(values, 1 - alpha / 2))
    return lo, hi
