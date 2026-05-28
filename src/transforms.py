"""Fit and apply linear transforms between language representations.

Supports two methods:
- ``procrustes``: orthogonal Procrustes (closed-form SVD).
- ``ridge``: closed-form ridge regression W = (X^T X + αI)^{-1} X^T Y.

Both are wrapped in a single transform dict so downstream composition logic
doesn't care which method produced the maps.
"""
from __future__ import annotations
from itertools import combinations
from typing import Iterable
import numpy as np
from scipy.linalg import svd, solve

from .config import N_TRAIN

ALL_PAIRS = list(combinations(range(5), 2))


# ---------- core fitters ----------

def fit_procrustes_np(X_A: np.ndarray, X_B: np.ndarray) -> np.ndarray:
    """Closed-form orthogonal Procrustes in float64. Inputs already centered."""
    M = X_A.astype(np.float64, copy=False).T @ X_B.astype(np.float64, copy=False)
    U, _, Vt = svd(M, full_matrices=False)
    return U @ Vt


def fit_ridge_np(X_A: np.ndarray, X_B: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge in float64. Inputs already centered."""
    A = X_A.astype(np.float64, copy=False)
    B = X_B.astype(np.float64, copy=False)
    d = A.shape[1]
    G = A.T @ A + alpha * np.eye(d)
    rhs = A.T @ B
    return solve(G, rhs, assume_a="pos")


# ---------- transform dict ----------

def _make_transform(
    X_A: np.ndarray,
    X_B: np.ndarray,
    method: str,
    ridge_alpha: float,
) -> dict:
    mean_A = X_A.mean(axis=0)
    mean_B = X_B.mean(axis=0)
    Ac = X_A - mean_A
    Bc = X_B - mean_B
    if method == "procrustes":
        W = fit_procrustes_np(Ac, Bc)
        return {
            "W": W,
            "mean_A": mean_A,
            "mean_B": mean_B,
            "is_orthogonal": True,
        }
    if method == "ridge":
        W = fit_ridge_np(Ac, Bc, alpha=ridge_alpha)
        W_inv = fit_ridge_np(Bc, Ac, alpha=ridge_alpha)  # explicit reverse
        return {
            "W": W,
            "W_inv": W_inv,
            "mean_A": mean_A,
            "mean_B": mean_B,
            "is_orthogonal": False,
        }
    raise ValueError(f"unknown method {method!r}")


def fit_all_transforms(
    reps: dict[int, np.ndarray] | list[np.ndarray],
    method: str = "procrustes",
    ridge_alpha: float = 1.0,
    n_train: int = N_TRAIN,
) -> dict[tuple[int, int], dict]:
    """Fit transforms for all 10 unordered pairs on the training split."""
    if isinstance(reps, list):
        reps = {i: r for i, r in enumerate(reps)}
    out = {}
    for A, B in ALL_PAIRS:
        XA = reps[A][:n_train].astype(np.float64)
        XB = reps[B][:n_train].astype(np.float64)
        out[(A, B)] = _make_transform(XA, XB, method=method, ridge_alpha=ridge_alpha)
    return out


def apply_transform(
    x: np.ndarray,
    src: int,
    dst: int,
    transforms: dict[tuple[int, int], dict],
) -> np.ndarray:
    """Map representations from src to dst. Handles direction automatically."""
    if src == dst:
        return x.copy() if hasattr(x, "copy") else x
    A, B = min(src, dst), max(src, dst)
    t = transforms[(A, B)]
    W = t["W"]
    mean_A = t["mean_A"]
    mean_B = t["mean_B"]
    if src == A:
        return (x - mean_A) @ W + mean_B
    # reverse direction
    if t.get("is_orthogonal", False):
        return (x - mean_B) @ W.T + mean_A
    return (x - mean_B) @ t["W_inv"] + mean_A


def compute_composition_errors(
    X_test: dict[int, np.ndarray] | list[np.ndarray],
    transforms: dict[tuple[int, int], dict],
    n_langs: int = 5,
    per_sentence: bool = False,
) -> dict[tuple[frozenset, int], float | np.ndarray]:
    """For every (triple, intermediate), compute the mean L2 composition
    error on the held-out test split.

    Both directed orderings ``(s0 -> I -> s1)`` and ``(s1 -> I -> s0)`` are
    computed and averaged. If ``per_sentence`` is True, returns per-sentence
    arrays (averaged across the two directions) instead of scalar means.
    """
    if isinstance(X_test, list):
        X_test = {i: r for i, r in enumerate(X_test)}
    out = {}
    for triple in combinations(range(n_langs), 3):
        triple_set = frozenset(triple)
        for intermediate in triple:
            others = [x for x in triple if x != intermediate]
            s0, s1 = others[0], others[1]
            per_fwd = _per_sentence_ce(X_test[s0], s0, intermediate, s1, transforms)
            per_rev = _per_sentence_ce(X_test[s1], s1, intermediate, s0, transforms)
            if per_sentence:
                out[(triple_set, intermediate)] = 0.5 * (per_fwd + per_rev)
            else:
                out[(triple_set, intermediate)] = 0.5 * (float(per_fwd.mean()) + float(per_rev.mean()))
    return out


def _per_sentence_ce(x_src, src, inter, dst, transforms) -> np.ndarray:
    indirect = apply_transform(apply_transform(x_src, src, inter, transforms), inter, dst, transforms)
    direct = apply_transform(x_src, src, dst, transforms)
    return np.linalg.norm(indirect - direct, axis=1)


def pairwise_reconstruction_error(
    X_test: dict[int, np.ndarray] | list[np.ndarray],
    transforms: dict[tuple[int, int], dict],
) -> dict[tuple[int, int], float]:
    if isinstance(X_test, list):
        X_test = {i: r for i, r in enumerate(X_test)}
    out = {}
    for (A, B) in transforms:
        pred = apply_transform(X_test[A], A, B, transforms)
        err = float(np.mean(np.linalg.norm(pred - X_test[B], axis=1)))
        out[(A, B)] = err
    return out


def get_effective_W(
    src: int,
    dst: int,
    transforms: dict[tuple[int, int], dict],
) -> np.ndarray:
    """Return the matrix W such that (x - mean_src) @ W ≈ (y - mean_dst).

    Used by closure diagnostics. Handles direction the same way
    ``apply_transform`` does.
    """
    A, B = min(src, dst), max(src, dst)
    t = transforms[(A, B)]
    if src == A:
        return t["W"]
    if t.get("is_orthogonal", False):
        return t["W"].T
    return t["W_inv"]


# ---------- ridge α selection ----------

def select_ridge_alpha(
    reps: dict[int, np.ndarray] | list[np.ndarray],
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    n_train: int = N_TRAIN,
    n_splits: int = 5,
    seed: int = 42,
    log_fn=None,
) -> float:
    """5-fold CV on training data, averaged across all 10 pairs."""
    from sklearn.model_selection import KFold

    if isinstance(reps, list):
        reps = {i: r for i, r in enumerate(reps)}

    if log_fn is None:
        log_fn = lambda _msg: None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_alpha = None
    best_err = float("inf")
    for alpha in alphas:
        fold_errs = []
        for tr, va in kf.split(np.arange(n_train)):
            for A, B in ALL_PAIRS:
                XA = reps[A][:n_train].astype(np.float64)
                XB = reps[B][:n_train].astype(np.float64)
                mA = XA[tr].mean(axis=0)
                mB = XB[tr].mean(axis=0)
                W = fit_ridge_np(XA[tr] - mA, XB[tr] - mB, alpha=alpha)
                pred = (XA[va] - mA) @ W + mB
                err = float(np.mean(np.linalg.norm(pred - XB[va], axis=1)))
                fold_errs.append(err)
        mean_err = float(np.mean(fold_errs))
        log_fn(f"  alpha={alpha}: mean_cv_err={mean_err:.6f}")
        if mean_err < best_err:
            best_err = mean_err
            best_alpha = float(alpha)
    log_fn(f"  selected alpha={best_alpha}")
    return best_alpha
