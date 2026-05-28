"""Diagnostic blocks A-H from diagnostics.md.

Each ``block_*`` function takes the per-configuration inputs it needs and a
``log`` callable that captures one line of output. The driver in
``run_diagnostics.py`` is responsible for routing those lines to per-config
files and to the master summary tables.

All numerical work is float64 on CPU. Inputs are .npy files loaded by
``representations.load_representations``.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import svd

from .config import LANG_NAMES_SHORT, N_TRAIN, GROUND_TRUTH_SPLITS
from .transforms import (
    ALL_PAIRS,
    apply_transform,
    compute_composition_errors,
    fit_all_transforms,
    get_effective_W,
    pairwise_reconstruction_error,
    select_ridge_alpha,
)
from .topologies import (
    classify_triple,
    enumerate_splits,
    enumerate_topologies,
    get_rank_of_ground_truth,
    is_ground_truth_split,
    is_ground_truth,
    rank_topologies,
    score_all_splits,
    topology_id,
)

ALL_TRIPLES = list(combinations(range(5), 3))
LANGS = LANG_NAMES_SHORT


# ---------- shared helpers ----------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(an * bn, axis=1)))


def _cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear CKA between two (n, d) matrices."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = float(np.linalg.norm(b.T @ a, "fro") ** 2)
    den = float(np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro"))
    return num / (den + 1e-12)


def split_train_test(X: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return [x[:N_TRAIN] for x in X], [x[N_TRAIN:] for x in X]


# ---------- Block A: representation statistics ----------

def block_a_representation_stats(
    reps: list[np.ndarray],
    log: Callable[[str], None],
) -> dict:
    """A1 per-lang norms + effective dim; A2 lang-ID subspace; A3 distance tables.

    Returns dict with at least ``langid_basis`` (top-4 right-singular vectors
    of the centered per-language means) used by Block G.
    """
    log("--- BLOCK A: representation statistics ---")

    # A1: per-language norms + effective dim
    log("A1: per-language norms and effective dim (train split)")
    for L in range(5):
        X = reps[L][:N_TRAIN].astype(np.float64)
        Xc = X - X.mean(axis=0)
        norms = np.linalg.norm(Xc, axis=1)
        _, S, _ = svd(Xc, full_matrices=False)
        cumvar = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-30)
        log(
            f"  {LANGS[L]}: mean_norm={norms.mean():.4f}  std_norm={norms.std():.4f}  "
            f"eff_dim_90={int(np.searchsorted(cumvar,0.90)+1)}  "
            f"eff_dim_95={int(np.searchsorted(cumvar,0.95)+1)}  "
            f"eff_dim_99={int(np.searchsorted(cumvar,0.99)+1)}  "
            f"rank={int(np.sum(S > 1e-6))}"
        )

    # A2: language-ID subspace
    log("A2: language-ID subspace")
    lang_means = np.stack([reps[L][:N_TRAIN].mean(axis=0) for L in range(5)]).astype(np.float64)
    grand_mean = lang_means.mean(axis=0)
    lang_means_c = lang_means - grand_mean
    _, S_lang, Vt_lang = svd(lang_means_c, full_matrices=False)
    log(f"  lang-ID singular values: {np.array2string(S_lang, precision=4)}")

    all_train_c = np.vstack([
        reps[L][:N_TRAIN].astype(np.float64) - grand_mean
        for L in range(5)
    ])
    total_var = float(np.sum(all_train_c ** 2))
    between_var = float(N_TRAIN * np.sum(lang_means_c ** 2))
    log(f"  between-lang var fraction: {between_var/total_var:.6f}")

    # A3: pairwise distances on test split (cosine, L2 mean, CKA)
    log("A3: pairwise distances (test split)")
    cos_table = np.zeros((5, 5))
    l2_table = np.zeros((5, 5))
    cka_table = np.zeros((5, 5))
    test_reps = [r[N_TRAIN:].astype(np.float64) for r in reps]
    for A in range(5):
        cos_table[A, A] = 1.0
        cka_table[A, A] = 1.0
        for B in range(A + 1, 5):
            c = _cos(test_reps[A], test_reps[B])
            l = float(np.mean(np.linalg.norm(test_reps[A] - test_reps[B], axis=1)))
            k = _cka(test_reps[A], test_reps[B])
            cos_table[A, B] = cos_table[B, A] = c
            l2_table[A, B] = l2_table[B, A] = l
            cka_table[A, B] = cka_table[B, A] = k
            log(
                f"  {LANGS[A]}-{LANGS[B]}: cos={c:+.4f}  l2_mean={l:.4f}  cka={k:.4f}"
            )

    return {
        "langid_basis": Vt_lang[:4],          # (4, d), float64
        "cos_table": cos_table,
        "l2_table": l2_table,
        "cka_table": cka_table,
    }


# ---------- Block B: transform statistics ----------

def block_b_transform_stats(
    reps: list[np.ndarray],
    transforms: dict,
    log: Callable[[str], None],
) -> dict:
    """B1 per-pair properties + train/test gap; B2 triangle closure."""
    log("--- BLOCK B: transform statistics ---")
    d = transforms[(0, 1)]["W"].shape[0]
    I = np.eye(d)
    train_test_ratios = {}
    log("B1: per-pair transform properties")
    recon_train_table = {}
    recon_test_table = {}
    for A, B in ALL_PAIRS:
        W = transforms[(A, B)]["W"]
        dist_I = float(np.linalg.norm(W - I, "fro"))
        tr = float(np.trace(W))
        try:
            S_W = np.linalg.svd(W, compute_uv=False)
            sv_min = float(S_W.min())
            sv_max = float(S_W.max())
            cond = sv_max / (sv_min + 1e-30)
        except np.linalg.LinAlgError:
            sv_min = sv_max = cond = float("nan")
        log(
            f"  W({LANGS[A]},{LANGS[B]}): ||W-I||_F={dist_I:.4f}  "
            f"trace={tr:.4f}/{d}  sv=[{sv_min:.4f},{sv_max:.4f}]  cond={cond:.4f}"
        )

        train_err = float(np.mean(np.linalg.norm(
            apply_transform(reps[A][:N_TRAIN].astype(np.float64), A, B, transforms)
            - reps[B][:N_TRAIN].astype(np.float64),
            axis=1,
        )))
        test_err = float(np.mean(np.linalg.norm(
            apply_transform(reps[A][N_TRAIN:].astype(np.float64), A, B, transforms)
            - reps[B][N_TRAIN:].astype(np.float64),
            axis=1,
        )))
        recon_train_table[(A, B)] = train_err
        recon_test_table[(A, B)] = test_err
        ratio = train_err / (test_err + 1e-30)
        train_test_ratios[(A, B)] = ratio
        log(
            f"  W({LANGS[A]},{LANGS[B]}): recon_train={train_err:.4f}  "
            f"recon_test={test_err:.4f}  ratio={ratio:.4f}"
        )

    # B2: triangle closure
    log("B2: triangle closure (matrix + data)")
    closure_table = {}
    test_reps = [r[N_TRAIN:].astype(np.float64) for r in reps]
    for triple in ALL_TRIPLES:
        for I_ in triple:
            others = sorted([x for x in triple if x != I_])
            S_, D_ = others
            W_SI = get_effective_W(S_, I_, transforms)
            W_ID = get_effective_W(I_, D_, transforms)
            W_SD = get_effective_W(S_, D_, transforms)
            W_composed = W_SI @ W_ID
            closure_fro = float(np.linalg.norm(W_composed - W_SD, "fro"))
            closure_rel = closure_fro / (float(np.linalg.norm(W_SD, "fro")) + 1e-30)
            indirect = apply_transform(
                apply_transform(test_reps[S_], S_, I_, transforms), I_, D_, transforms
            )
            direct = apply_transform(test_reps[S_], S_, D_, transforms)
            per_sent = np.linalg.norm(indirect - direct, axis=1)
            closure_table[(S_, I_, D_)] = closure_rel
            log(
                f"  ({LANGS[S_]},{LANGS[I_]},{LANGS[D_]}): "
                f"closure_fro={closure_fro:.6f}  closure_rel={closure_rel:.6f}  "
                f"ce_mean={per_sent.mean():.6f}  ce_std={per_sent.std():.6f}"
            )
    return {
        "train_test_ratios": train_test_ratios,
        "closure_rel": closure_table,
        "recon_train": recon_train_table,
        "recon_test": recon_test_table,
    }


# ---------- Block C: composition error analysis ----------

def block_c_composition(
    reps: list[np.ndarray],
    transforms: dict,
    log: Callable[[str], None],
    save_per_sentence_dir: Path | None = None,
    config_tag: str = "",
) -> dict:
    """C1 distribution stats; C2 within-triple ranking; C3 per-lang bias."""
    log("--- BLOCK C: composition errors ---")
    test_reps = [r[N_TRAIN:].astype(np.float64) for r in reps]
    ce_mean = compute_composition_errors(test_reps, transforms)
    ce_per_sent = compute_composition_errors(test_reps, transforms, per_sentence=True)

    log("C1: per (triple, intermediate) mean/std/median")
    for triple in ALL_TRIPLES:
        for I_ in triple:
            arr = ce_per_sent[(frozenset(triple), I_)]
            tname = ",".join(LANGS[i] for i in triple)
            log(
                f"  {{{tname}}} int={LANGS[I_]}: "
                f"mean={arr.mean():.6f}  std={arr.std():.6f}  "
                f"med={np.median(arr):.6f}  q25={np.percentile(arr,25):.6f}  "
                f"q75={np.percentile(arr,75):.6f}"
            )
            if save_per_sentence_dir is not None:
                p = save_per_sentence_dir / f"ce_{config_tag}_{'-'.join(map(str,triple))}_int{I_}.npy"
                np.save(p, arr)

    log("C2: within-triple intermediate ranking")
    for triple in ALL_TRIPLES:
        errs = {i: ce_mean[(frozenset(triple), i)] for i in triple}
        ranked = sorted(errs.items(), key=lambda x: x[1])
        rng = ranked[2][1] - ranked[0][1]
        rel = rng / (ranked[0][1] + 1e-30)
        tname = ",".join(LANGS[i] for i in triple)
        log(
            f"  {{{tname}}}: best={LANGS[ranked[0][0]]}({ranked[0][1]:.6f})  "
            f"mid={LANGS[ranked[1][0]]}({ranked[1][1]:.6f})  "
            f"worst={LANGS[ranked[2][0]]}({ranked[2][1]:.6f})  "
            f"range={rng:.6f}  rel_range={rel:.4f}"
        )

    log("C3: per-language intermediate bias")
    per_lang_bias = {}
    for I_ in range(5):
        triples_with_I = [t for t in ALL_TRIPLES if I_ in t]
        errs = [ce_mean[(frozenset(t), I_)] for t in triples_with_I]
        per_lang_bias[I_] = float(np.mean(errs))
        log(
            f"  {LANGS[I_]} as intermediate: mean={np.mean(errs):.6f}  "
            f"std={np.std(errs):.6f}  n={len(errs)}"
        )

    return {
        "ce_mean": ce_mean,
        "per_lang_bias": per_lang_bias,
    }


def block_c4_rank_consistency(
    ce_means_by_layer: dict[str, dict],
    log: Callable[[str], None],
) -> None:
    """C4 — run AFTER all layers in a (model, pool, method) sweep are done.

    For each triple, log the worst-intermediate consistency across layers.
    """
    log("--- BLOCK C4: rank consistency across layers ---")
    from collections import Counter
    for triple in ALL_TRIPLES:
        worsts = []
        for layer, ce in ce_means_by_layer.items():
            errs = {i: ce[(frozenset(triple), i)] for i in triple}
            worsts.append(max(errs, key=errs.get))
        counts = Counter(worsts)
        most_common, count = counts.most_common(1)[0]
        consistency = count / len(worsts)
        tname = ",".join(LANGS[i] for i in triple)
        log(
            f"  {{{tname}}}: worst across {len(worsts)} layers: "
            f"{ {LANGS[k]: v for k, v in counts.items()} }  "
            f"consistency={consistency:.2f}"
        )


# ---------- Block D: permutation test ----------

def block_d_permutation(
    reps: list[np.ndarray],
    aligned_ce: dict,
    aligned_gt_rank: int,
    method: str,
    ridge_alpha: float,
    n_perms: int,
    log: Callable[[str], None],
) -> dict:
    """Permutation test: misalign training sentences within each pair, refit
    transforms, recompute composition errors on aligned test data, score
    topologies, and report z-scores.
    """
    log(f"--- BLOCK D: permutation test (n={n_perms}, method={method}) ---")

    test_reps = [r[N_TRAIN:].astype(np.float64) for r in reps]
    perm_ces = []
    perm_gt_ranks = []
    for p_idx in range(n_perms):
        rng = np.random.default_rng(seed=p_idx)
        perm_T = {}
        for A, B in ALL_PAIRS:
            XA = reps[A][:N_TRAIN].astype(np.float64)
            XB = reps[B][:N_TRAIN].astype(np.float64)
            perm = rng.permutation(N_TRAIN)
            XB_p = XB[perm]
            mA = XA.mean(axis=0)
            mB = XB_p.mean(axis=0)
            Ac = XA - mA
            Bc = XB_p - mB
            if method == "procrustes":
                from .transforms import fit_procrustes_np
                W = fit_procrustes_np(Ac, Bc)
                perm_T[(A, B)] = {"W": W, "mean_A": mA, "mean_B": mB, "is_orthogonal": True}
            else:
                from .transforms import fit_ridge_np
                W = fit_ridge_np(Ac, Bc, alpha=ridge_alpha)
                W_inv = fit_ridge_np(Bc, Ac, alpha=ridge_alpha)
                perm_T[(A, B)] = {
                    "W": W, "W_inv": W_inv, "mean_A": mA, "mean_B": mB,
                    "is_orthogonal": False,
                }
        ce_p = compute_composition_errors(test_reps, perm_T)
        gt_rank_p = get_rank_of_ground_truth(ce_p, use_resolved=False)
        perm_ces.append(ce_p)
        perm_gt_ranks.append(gt_rank_p)
        log(f"  perm {p_idx}: GT_rank(legacy)={gt_rank_p}/15")

    # Z-scores per (triple, intermediate)
    z_scores = {}
    log("D summary: z-scores (aligned vs permutation distribution)")
    for key in sorted(aligned_ce.keys(), key=lambda k: (sorted(k[0]), k[1])):
        perm_vals = np.array([perm_ces[p][key] for p in range(n_perms)])
        z = float((aligned_ce[key] - perm_vals.mean()) / (perm_vals.std() + 1e-10))
        z_scores[key] = z
        triple = tuple(sorted(key[0]))
        tname = ",".join(LANGS[i] for i in triple)
        log(
            f"  {{{tname}}} int={LANGS[key[1]]}: aligned={aligned_ce[key]:.6f}  "
            f"perm_mean={perm_vals.mean():.6f}  perm_std={perm_vals.std():.6f}  z={z:.4f}"
        )
    mean_z = float(np.mean(list(z_scores.values())))
    log(
        f"  GT_RANK: aligned={aligned_gt_rank}  "
        f"perm_mean={np.mean(perm_gt_ranks):.1f}  "
        f"perm_std={np.std(perm_gt_ranks):.1f}  "
        f"perm_median={int(np.median(perm_gt_ranks))}"
    )
    log(f"  mean_z over all 30 (triple,intermediate) entries: {mean_z:.4f}")
    return {"z_scores": z_scores, "mean_z": mean_z, "perm_gt_ranks": perm_gt_ranks}


# ---------- Block E: scoring diagnostics ----------

def block_e_scoring(
    ce_mean: dict,
    log: Callable[[str], None],
) -> dict:
    """E1 classify triples; E2 split-level scores; E3 dual topology rankings."""
    log("--- BLOCK E: scoring diagnostics ---")

    log("E1: triple classification under GROUND TRUTH topology")
    n_resolved = n_resolved_ok = n_star = 0
    for triple in ALL_TRIPLES:
        status, pred = classify_triple(triple, GROUND_TRUTH_SPLITS)
        errs = {i: ce_mean[(frozenset(triple), i)] for i in triple}
        emp = max(errs, key=errs.get)
        tname = ",".join(LANGS[i] for i in triple)
        if status == "resolved":
            ok = (pred == emp)
            n_resolved += 1
            n_resolved_ok += int(ok)
            log(
                f"  {{{tname}}}: RESOLVED  pred={LANGS[pred]}  "
                f"emp={LANGS[emp]}  {'OK' if ok else 'MISS'}"
            )
        else:
            n_star += 1
            log(f"  {{{tname}}}: STAR  emp_worst={LANGS[emp]}")
    log(f"  resolved: {n_resolved_ok}/{n_resolved}  star: {n_star}")

    log("E2: split-level scores (all 10 splits)")
    split_scores = score_all_splits(ce_mean)
    for (small, big), correct in split_scores.items():
        details = []
        for outsider in sorted(big):
            triple = tuple(sorted(small | {outsider}))
            errs = {i: ce_mean[(frozenset(triple), i)] for i in triple}
            emp = max(errs, key=errs.get)
            details.append(f"{LANGS[outsider]}:{'OK' if emp == outsider else 'MISS'}")
        gt_tag = "  ** GT SPLIT **" if is_ground_truth_split((small, big)) else ""
        small_str = ",".join(LANGS[i] for i in sorted(small))
        big_str = ",".join(LANGS[i] for i in sorted(big))
        log(f"  {{{small_str}}}|{{{big_str}}}: {correct}/3  [{', '.join(details)}]{gt_tag}")

    log("E3: dual topology rankings")
    legacy_rows = rank_topologies(ce_mean, use_resolved=False)
    resolved_rows = rank_topologies(ce_mean, use_resolved=True)
    legacy_gt = get_rank_of_ground_truth(ce_mean, use_resolved=False)
    resolved_gt = get_rank_of_ground_truth(ce_mean, use_resolved=True)
    log(f"  LEGACY (sum-of-ingroup-errors)  GT rank = {legacy_gt}/15")
    for i, (topo, s, v, gt) in enumerate(legacy_rows, start=1):
        marker = "  <-- GT" if gt else ""
        log(f"    [{i:>2}] score={s:.6f}  votes={v}  {topology_id(topo)}{marker}")
    log(f"  RESOLVED-ONLY (correct outgroup count)  GT rank = {resolved_gt}/15")
    for i, (topo, s, v, gt) in enumerate(resolved_rows, start=1):
        correct = int(-s)
        marker = "  <-- GT" if gt else ""
        log(f"    [{i:>2}] correct={correct}/6  votes={v}  {topology_id(topo)}{marker}")

    return {
        "split_scores": split_scores,
        "gt_rank_legacy": legacy_gt,
        "gt_rank_resolved": resolved_gt,
        "split_gt_0_1": split_scores[GROUND_TRUTH_SPLITS[0]],
        "split_gt_3_4": split_scores[GROUND_TRUTH_SPLITS[1]],
    }


# ---------- Block F: PCA sweep ----------

def block_f_pca_sweep(
    reps: list[np.ndarray],
    method: str,
    ridge_alpha: float,
    target_dims: list[int],
    log: Callable[[str], None],
) -> dict:
    """For each d', project all reps onto the top-d' principal axes of the
    pooled per-language-centered training data, refit, rescore.
    """
    log(f"--- BLOCK F: PCA sweep ({target_dims}) ---")
    pooled = np.vstack([
        reps[L][:N_TRAIN].astype(np.float64) - reps[L][:N_TRAIN].astype(np.float64).mean(axis=0)
        for L in range(5)
    ])
    _, S_pca, Vt_pca = svd(pooled, full_matrices=False)
    log(f"  pooled-centered top singular values: {np.array2string(S_pca[:8], precision=2)}")

    results = {}
    for d_prime in target_dims:
        if d_prime > Vt_pca.shape[0]:
            log(f"  PCA d'={d_prime}: skipping (d'>{Vt_pca.shape[0]} available)")
            continue
        P = Vt_pca[:d_prime].T  # (d, d')
        proj = []
        for L in range(5):
            X = reps[L].astype(np.float64)
            mean_train = reps[L][:N_TRAIN].astype(np.float64).mean(axis=0)
            proj.append((X - mean_train) @ P)
        T = fit_all_transforms(proj, method=method, ridge_alpha=ridge_alpha)
        test_proj = [p[N_TRAIN:] for p in proj]
        ce_p = compute_composition_errors(test_proj, T)
        legacy_gt = get_rank_of_ground_truth(ce_p, use_resolved=False)
        resolved_gt = get_rank_of_ground_truth(ce_p, use_resolved=True)
        split_p = score_all_splits(ce_p)
        log(
            f"  PCA d'={d_prime}: gt_rank_legacy={legacy_gt}/15  "
            f"gt_rank_resolved={resolved_gt}/15  "
            f"split{{Spa,Por}}={split_p[GROUND_TRUTH_SPLITS[0]]}/3  "
            f"split{{Ita,Ron}}={split_p[GROUND_TRUTH_SPLITS[1]]}/3"
        )
        # Per-triple within-ranking
        for triple in ALL_TRIPLES:
            errs = {i: ce_p[(frozenset(triple), i)] for i in triple}
            ranked = sorted(errs.items(), key=lambda x: x[1])
            rng = ranked[2][1] - ranked[0][1]
            rel = rng / (ranked[0][1] + 1e-30)
            tname = ",".join(LANGS[i] for i in triple)
            log(
                f"    PCA d'={d_prime}: {{{tname}}}: "
                f"best={LANGS[ranked[0][0]]}  worst={LANGS[ranked[2][0]]}  "
                f"range={rng:.6f}  rel_range={rel:.4f}"
            )
        results[d_prime] = {
            "gt_rank_legacy": legacy_gt,
            "gt_rank_resolved": resolved_gt,
            "split_gt_0_1": split_p[GROUND_TRUTH_SPLITS[0]],
            "split_gt_3_4": split_p[GROUND_TRUTH_SPLITS[1]],
        }
    return results


# ---------- Block G: language-ID projection ----------

def block_g_langid(
    reps: list[np.ndarray],
    langid_basis: np.ndarray,
    method: str,
    ridge_alpha: float,
    log: Callable[[str], None],
) -> dict:
    """Project out the top k lang-ID directions before fitting transforms."""
    log("--- BLOCK G: language-ID projection sweep ---")
    d = reps[0].shape[1]
    results = {}
    for k in range(5):
        if k == 0:
            proj = [r.astype(np.float64).copy() for r in reps]
        else:
            V = langid_basis[:k].astype(np.float64)
            P = np.eye(d) - V.T @ V
            proj = [r.astype(np.float64) @ P for r in reps]
        T = fit_all_transforms(proj, method=method, ridge_alpha=ridge_alpha)
        test_proj = [p[N_TRAIN:] for p in proj]
        ce_g = compute_composition_errors(test_proj, T)
        legacy_gt = get_rank_of_ground_truth(ce_g, use_resolved=False)
        resolved_gt = get_rank_of_ground_truth(ce_g, use_resolved=True)
        split_g = score_all_splits(ce_g)
        log(
            f"  LangID k={k}: gt_rank_legacy={legacy_gt}/15  "
            f"gt_rank_resolved={resolved_gt}/15  "
            f"split{{Spa,Por}}={split_g[GROUND_TRUTH_SPLITS[0]]}/3  "
            f"split{{Ita,Ron}}={split_g[GROUND_TRUTH_SPLITS[1]]}/3"
        )
        for A, B in ALL_PAIRS:
            err = float(np.mean(np.linalg.norm(
                apply_transform(test_proj[A], A, B, T) - test_proj[B], axis=1,
            )))
            log(f"    LangID k={k}: recon({LANGS[A]},{LANGS[B]})={err:.4f}")
        results[k] = {
            "gt_rank_legacy": legacy_gt,
            "gt_rank_resolved": resolved_gt,
            "split_gt_0_1": split_g[GROUND_TRUTH_SPLITS[0]],
            "split_gt_3_4": split_g[GROUND_TRUTH_SPLITS[1]],
        }
    return results


# ---------- Block H: distance-tree recovery ----------

def _interpret_linkage(Z: np.ndarray, n_leaves: int) -> set[frozenset]:
    """Extract the set of non-trivial splits induced by a SciPy linkage Z."""
    cluster_members: dict[int, frozenset] = {i: frozenset({i}) for i in range(n_leaves)}
    splits = set()
    next_id = n_leaves
    all_leaves = frozenset(range(n_leaves))
    for row in Z:
        c1, c2 = int(row[0]), int(row[1])
        merged = cluster_members[c1] | cluster_members[c2]
        cluster_members[next_id] = merged
        if 1 < len(merged) < n_leaves:
            # Represent the split by the smaller side (canonical form).
            other = all_leaves - merged
            small = merged if len(merged) <= len(other) else other
            big = all_leaves - small
            splits.add((small, big))
        next_id += 1
    return splits


def block_h_distance_tree(
    reps: list[np.ndarray],
    transforms_for_recon: dict | None,
    cos_table: np.ndarray,
    log: Callable[[str], None],
) -> dict:
    """Try multiple distance metrics × multiple linkages. For each tree,
    report the induced non-trivial splits and whether they match the
    ground-truth pair.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    log("--- BLOCK H: pairwise distance tree recovery ---")
    test_reps = [r[N_TRAIN:].astype(np.float64) for r in reps]

    metrics = {}
    if transforms_for_recon is not None:
        recon = pairwise_reconstruction_error(test_reps, transforms_for_recon)
        D_recon = np.zeros((5, 5))
        for (A, B), v in recon.items():
            D_recon[A, B] = D_recon[B, A] = v
        metrics["recon_error"] = D_recon
    D_cos = 1.0 - cos_table
    np.fill_diagonal(D_cos, 0.0)
    metrics["1-cosine"] = D_cos
    D_means = np.zeros((5, 5))
    for A, B in ALL_PAIRS:
        m = float(np.linalg.norm(
            reps[A][:N_TRAIN].astype(np.float64).mean(axis=0)
            - reps[B][:N_TRAIN].astype(np.float64).mean(axis=0)
        ))
        D_means[A, B] = D_means[B, A] = m
    metrics["l2_of_means"] = D_means

    target = {GROUND_TRUTH_SPLITS[0], GROUND_TRUTH_SPLITS[1]}
    matches = {}
    for metric_name, D in metrics.items():
        for link_method in ["average", "single", "complete", "ward"]:
            condensed = squareform(D, checks=False)
            Z = linkage(condensed, method=link_method)
            splits = _interpret_linkage(Z, 5)
            n_gt = len(splits & target)
            log(f"  TREE [{metric_name}, {link_method}]: GT splits matched = {n_gt}/2")
            for i, row in enumerate(Z):
                c1, c2, dist, size = row
                c1n = LANGS[int(c1)] if c1 < 5 else f"c{int(c1)-5}"
                c2n = LANGS[int(c2)] if c2 < 5 else f"c{int(c2)-5}"
                log(f"    merge {i}: ({c1n}, {c2n})  dist={dist:.6f}  size={int(size)}")
            matches[(metric_name, link_method)] = n_gt
    return matches
