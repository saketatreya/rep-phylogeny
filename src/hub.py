"""Hub alignment — generalized **orthogonal** Procrustes across languages.

Why orthogonal and not affine: an affine fit can rescale away the very
low-variance direction that may carry the genealogical signal — that's the
mechanism by which ridge erased it in prior runs. Orthogonal rotations
preserve all variance ratios; only a single global scalar per language
removes gross magnitude differences (e.g. norm growth across layers).

After alignment, two things matter to downstream code:

- **Centroids in the common frame** — the test-split language centroid,
  relative to the grand mean, is the genealogical signal for the
  hub-centroid distance (Tier 0, Tier 2a).
- **Per-sentence form residuals** — aligned_X[L, i] minus the per-sentence
  meaning vector (the mean across languages of aligned_X[L, i]). These are
  what the held-out family discriminant (Tier 1) and the universal axis
  (Tier 2b) operate on; subtracting the per-sentence meaning isolates
  language-specific structure from shared sentence content.

Critical: **do not center per language** before/after alignment. The
language centroid relative to the grand mean *is* the signal we're trying
to recover; centering would zero it out.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import svd

from .config import N_TRAIN


def _safe_svd(M: np.ndarray):
    try:
        return svd(M, full_matrices=False, lapack_driver="gesdd")
    except np.linalg.LinAlgError:
        # gesdd divide-and-conquer can fail to converge on ill-conditioned
        # matrices (deep-layer outliers, post-permutation). gesvd is slower
        # but uses QR and almost always converges.
        return svd(M, full_matrices=False, lapack_driver="gesvd")


def hub_align(
    reps_by_lang: dict[str, np.ndarray],
    n_train: int = N_TRAIN,
    n_iters: int = 5,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Generalized orthogonal Procrustes with one global scale per language.

    Returns ``(aligned, grand_mean, rotations)``:
      - ``aligned[L]``: (N, d) float64, X[L] / scale[L] @ V[L]. NOT centered.
      - ``grand_mean``: (d,) global mean of aligned data.
      - ``rotations[L]``: (d, d) the V[L] rotation used.
    """
    langs = list(reps_by_lang.keys())
    d = next(iter(reps_by_lang.values())).shape[1]

    # One global scale per language: RMS of train cloud after de-meaning.
    Xs: dict[str, np.ndarray] = {}
    for L in langs:
        X = reps_by_lang[L].astype(np.float64, copy=True)
        center = X[:n_train].mean(axis=0)
        rms = float(np.sqrt(((X[:n_train] - center) ** 2).sum(axis=1).mean()))
        if rms < 1e-12:
            rms = 1.0
        Xs[L] = X / rms

    # Initial reference = per-sentence mean of train clouds across languages.
    ref = np.mean(np.stack([Xs[L][:n_train] for L in langs], axis=0), axis=0)
    V: dict[str, np.ndarray] = {L: np.eye(d) for L in langs}

    for _ in range(n_iters):
        ref_centered = ref - ref.mean(axis=0)
        for L in langs:
            Xc = Xs[L][:n_train] - Xs[L][:n_train].mean(axis=0)
            M = Xc.T @ ref_centered
            U, _, Vt = _safe_svd(M)
            V[L] = U @ Vt
        aligned_tr = np.stack([Xs[L][:n_train] @ V[L] for L in langs], axis=0)
        ref = aligned_tr.mean(axis=0)

    # Final aligned data (apply V to UNCENTERED scale-normed reps).
    aligned = {L: Xs[L] @ V[L] for L in langs}
    grand_mean = np.mean(np.stack(list(aligned.values()), axis=0), axis=0).mean(axis=0)
    return aligned, grand_mean, V


def form_residuals(
    aligned: dict[str, np.ndarray],
    n_train: int | None = N_TRAIN,
    split: str = "test",
) -> dict[str, np.ndarray]:
    """Per-sentence aligned vector minus the cross-language per-sentence mean.

    The mean acts as the (genealogically inert) sentence-content vector for a
    given parallel index; subtracting it isolates language-specific form.

    ``split`` is one of ``"train"``, ``"test"``, ``"all"``.
    """
    langs = list(aligned.keys())
    stack = np.stack([aligned[L] for L in langs], axis=0)  # (n_lang, N, d)
    meaning = stack.mean(axis=0)                            # (N, d)
    resid = {L: aligned[L] - meaning for L in langs}
    if split == "all" or n_train is None:
        return resid
    if split == "train":
        return {L: r[:n_train] for L, r in resid.items()}
    if split == "test":
        return {L: r[n_train:] for L, r in resid.items()}
    raise ValueError(f"unknown split {split!r}")
