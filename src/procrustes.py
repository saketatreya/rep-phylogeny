"""Fit orthogonal Procrustes maps between pairs of language representations."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.linalg import svd


def fit_procrustes(X_A: np.ndarray, X_B: np.ndarray) -> np.ndarray:
    """Solve orthogonal Procrustes: find R minimizing ||X_A R - X_B||_F.

    Returns R of shape (d, d). Both inputs should already be centered.
    Computation runs in float64 for numerical stability.
    """
    A64 = X_A.astype(np.float64, copy=False)
    B64 = X_B.astype(np.float64, copy=False)
    M = A64.T @ B64  # (d, d)
    U, _, Vt = svd(M, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def fit_all_pairs(
    X_train: list[np.ndarray],
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Fit Procrustes maps for all C(5,2)=10 unordered pairs (A, B) with A<B.

    X_train[i] is the (n_train, d) train-split representation matrix for
    language index i.

    Returns dict[(A,B)] -> {'R': (d,d), 'mean_A': (d,), 'mean_B': (d,)}.
    Reverse direction is given by R.T (do not refit).
    """
    n_langs = len(X_train)
    out = {}
    for A, B in combinations(range(n_langs), 2):
        XA, XB = X_train[A], X_train[B]
        mean_A = XA.mean(axis=0)
        mean_B = XB.mean(axis=0)
        R = fit_procrustes(XA - mean_A, XB - mean_B)
        out[(A, B)] = {
            "R": R,
            "mean_A": mean_A.astype(np.float32),
            "mean_B": mean_B.astype(np.float32),
        }
    return out


def apply_transform(
    x: np.ndarray,
    src: int,
    dst: int,
    proc: dict[tuple[int, int], dict[str, np.ndarray]],
) -> np.ndarray:
    """Map representations from src language to dst language.

    Uses the stored (A,B) map if (src,dst) is the stored direction; otherwise
    uses the transpose. x may be a single (d,) vector or (n, d) batch.
    """
    if (src, dst) in proc:
        d = proc[(src, dst)]
        return (x - d["mean_A"]) @ d["R"] + d["mean_B"]
    if (dst, src) in proc:
        d = proc[(dst, src)]
        # Reverse: R.T inverts an orthogonal R
        return (x - d["mean_B"]) @ d["R"].T + d["mean_A"]
    raise KeyError(f"No Procrustes map for ({src}, {dst})")


def pairwise_reconstruction_error(
    X_test: list[np.ndarray],
    proc: dict[tuple[int, int], dict[str, np.ndarray]],
) -> dict[tuple[int, int], float]:
    """Mean L2 error per test sentence for each unordered pair (A<B).

    Defined as mean over test sentences of ||apply(A->B, x_A) - x_B||_2.
    """
    out = {}
    for (A, B) in proc:
        x_pred = apply_transform(X_test[A], A, B, proc)
        err = float(np.mean(np.linalg.norm(x_pred - X_test[B], axis=1)))
        out[(A, B)] = err
    return out
