"""Two geometric distance variants between languages.

(a) **Procrustes residual** — per-pair fit, residual on the held-out test
    split. Measures "how alignable are these two clouds (shape)". The fit
    is per-pair so this is a strictly local metric.
(b) **Hub centroid** — after a single shared hub alignment, the distance
    between the per-language test-split centroids. Measures "in a common
    frame, how far apart are the languages' offsets (location)".

Both run in float64. Procrustes uses the robust gesdd→gesvd SVD fallback
inherited from prior runs (deep XLM-R / Gemma layers can break gesdd).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np

from .config import N_TRAIN
from .hub import hub_align, _safe_svd


# ---------- Procrustes residual ----------

def fit_procrustes_np(X_A: np.ndarray, X_B: np.ndarray) -> np.ndarray:
    """Closed-form orthogonal Procrustes in float64. Inputs already centered."""
    M = X_A.astype(np.float64, copy=False).T @ X_B.astype(np.float64, copy=False)
    U, _, Vt = _safe_svd(M)
    return U @ Vt


def procrustes_residual_distance(
    X_A: np.ndarray, X_B: np.ndarray, n_train: int = N_TRAIN,
    return_per_sentence: bool = False,
):
    """Mean L2 residual of mapping A→B on the test split.

    Lower = more alignable = closer. This is the pairwise base case of the
    composition method (one fitted transformation, residual measures how
    well two languages live in compatible geometries).

    Note: when comparing (X_A, X_B) for "is English close to French or to
    German", X_A appears on both sides of the comparison — so any
    English-specific representation quality issues largely cancel, which
    defuses the per-language-quality confound from prior runs.
    """
    A_tr = X_A[:n_train].astype(np.float64)
    B_tr = X_B[:n_train].astype(np.float64)
    A_te = X_A[n_train:].astype(np.float64)
    B_te = X_B[n_train:].astype(np.float64)
    mA, mB = A_tr.mean(axis=0), B_tr.mean(axis=0)
    R = fit_procrustes_np(A_tr - mA, B_tr - mB)
    pred = (A_te - mA) @ R
    resid = np.linalg.norm(pred - (B_te - mB), axis=1)
    if return_per_sentence:
        return resid
    return float(resid.mean())


def procrustes_distance_matrix(
    reps_by_lang: dict[str, np.ndarray], n_train: int = N_TRAIN,
) -> tuple[list[str], np.ndarray]:
    """Symmetric language×language matrix; mean of A→B and B→A residuals."""
    langs = list(reps_by_lang.keys())
    n = len(langs)
    D = np.zeros((n, n), dtype=np.float64)
    for i, j in combinations(range(n), 2):
        a = procrustes_residual_distance(reps_by_lang[langs[i]],
                                         reps_by_lang[langs[j]], n_train)
        b = procrustes_residual_distance(reps_by_lang[langs[j]],
                                         reps_by_lang[langs[i]], n_train)
        D[i, j] = D[j, i] = 0.5 * (a + b)
    return langs, D


# ---------- Hub centroid ----------

def hub_centroid_distance_matrix(
    reps_by_lang: dict[str, np.ndarray], n_train: int = N_TRAIN, n_iters: int = 5,
) -> tuple[list[str], np.ndarray, dict]:
    """One hub alignment over all languages, then pairwise distance between
    test-split centroids in the common frame.

    Returns ``(langs, dist_matrix, hub_artifacts)`` where ``hub_artifacts``
    holds the aligned reps / grand mean / rotations so callers can re-use
    them for axis fitting and form residuals.
    """
    aligned, grand_mean, V = hub_align(reps_by_lang, n_train=n_train, n_iters=n_iters)
    langs = list(reps_by_lang.keys())
    form = {L: aligned[L][n_train:].mean(axis=0) - grand_mean for L in langs}
    n = len(langs)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            D[i, j] = float(np.linalg.norm(form[langs[i]] - form[langs[j]]))
    return langs, D, {"aligned": aligned, "grand_mean": grand_mean, "V": V,
                      "centroids": form}
