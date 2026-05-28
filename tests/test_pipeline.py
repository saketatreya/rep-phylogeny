"""Smoke tests with synthetic data: validates Procrustes, composition errors,
topology enumeration, and end-to-end scoring without needing a GPU/model."""
from __future__ import annotations
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import N_TRAIN, N_TEST
from src.procrustes import (
    fit_procrustes,
    fit_all_pairs,
    apply_transform,
    pairwise_reconstruction_error,
)
from src.composition import compute_composition_errors
from src.topologies import (
    GROUND_TRUTH_SPLITS,
    enumerate_splits,
    enumerate_topologies,
    get_outgroup,
    is_ground_truth,
    rank_topologies,
    score_topology,
    topology_id,
)


# ---- topology enumeration ---------------------------------------------------

def test_enumerate_splits():
    splits = enumerate_splits(5)
    assert len(splits) == 10
    # every split: |small|=2, |big|=3, disjoint, union=full
    full = frozenset(range(5))
    for small, big in splits:
        assert len(small) == 2
        assert len(big) == 3
        assert small.isdisjoint(big)
        assert small | big == full


def test_enumerate_topologies():
    topos = enumerate_topologies()
    assert len(topos) == 15
    # every topology is two distinct compatible splits
    for s1, s2 in topos:
        assert s1 != s2


def test_ground_truth_is_in_enumeration():
    topos = enumerate_topologies()
    matches = [t for t in topos if is_ground_truth(t)]
    assert len(matches) == 1, f"expected exactly 1 ground-truth topology, got {len(matches)}"


def test_get_outgroup_returns_member_of_triple():
    topos = enumerate_topologies()
    for topo in topos:
        for triple in combinations(range(5), 3):
            og = get_outgroup(topo, triple)
            assert og in triple, f"outgroup {og} not in triple {triple} for topo {topo}"


# ---- Procrustes -------------------------------------------------------------

def test_procrustes_recovers_known_rotation():
    """If Y = X @ R for an orthogonal R, fit_procrustes should recover R."""
    rng = np.random.default_rng(0)
    d = 32
    n = 200
    # generate a random orthogonal R via QR
    M = rng.standard_normal((d, d))
    R_true, _ = np.linalg.qr(M)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X = X - X.mean(axis=0)
    Y = X @ R_true
    R_hat = fit_procrustes(X, Y)
    err = np.linalg.norm(R_hat - R_true)
    assert err < 1e-3, f"||R_hat - R_true||={err}"


def test_apply_transform_direction_consistency():
    """apply(A->B) on x_A should equal x_B (within fit error), and the
    reverse path should be exactly the transpose."""
    rng = np.random.default_rng(1)
    d = 16
    n = 100
    X_A = rng.standard_normal((n, d)).astype(np.float32)
    mean_A = X_A.mean(axis=0)
    M = rng.standard_normal((d, d))
    R, _ = np.linalg.qr(M)
    X_B = (X_A - mean_A) @ R + 7.0  # constant offset = mean_B

    X_train = [X_A, X_B, X_A, X_B, X_A]
    proc = fit_all_pairs(X_train)
    # Map A->B using fitted Procrustes
    Y = apply_transform(X_A, 0, 1, proc)
    err = np.mean(np.linalg.norm(Y - X_B, axis=1))
    assert err < 1e-2, f"forward map error {err}"
    # reverse must invert
    X_back = apply_transform(Y, 1, 0, proc)
    err_back = np.mean(np.linalg.norm(X_back - X_A, axis=1))
    assert err_back < 1e-2, f"reverse map error {err_back}"


# ---- end-to-end with planted phylogenetic signal ---------------------------

def make_phylo_reps(rng: np.random.Generator, d: int = 64) -> list[np.ndarray]:
    """Build 5 representation matrices with a phylogenetic structure that
    matches the Romance tree. Lower hidden dim, deterministic, just enough
    to exercise the pipeline.

    Structure: sample one "shared" latent matrix Z (n, d). Each language is
    Z @ R_i + lang_specific_noise where R_i are orthogonal rotations close
    to identity for closely related languages and farther apart for distant
    ones. Sister languages share more of the random rotation.
    """
    n = N_TRAIN + N_TEST
    Z = rng.standard_normal((n, d)).astype(np.float32)
    # Centered base
    Z = Z - Z.mean(axis=0)

    # Build small rotations from skew-symmetric matrices
    def small_rot(scale: float, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        A = r.standard_normal((d, d)).astype(np.float32) * scale
        A = (A - A.T) * 0.5  # skew-symmetric
        # exp(A) via Pade / scipy
        from scipy.linalg import expm
        return expm(A).astype(np.float32)

    # Tree distances roughly matching the Romance phylogeny:
    # Spa <-> Por: very close
    # Spa <-> Fre: medium
    # Spa <-> Ita: medium-far
    # Spa <-> Ron: far
    # Ita <-> Ron: medium-close (they share a recent common ancestor in unrooted)
    # Per-leaf rotation scale (larger = farther from the central "true" Z).
    # We rotate each language's reps relative to Z with branch-length-scaled
    # small rotations stacked along the tree.
    # Branch lengths (toy):
    rot_spa = small_rot(0.05, 1)                                # 0
    rot_por = small_rot(0.05, 2)                                # 1
    rot_spapor_int = small_rot(0.10, 10)
    rot_fre = small_rot(0.10, 3) @ rot_spapor_int               # 2
    rot_spaporfre_int = small_rot(0.10, 11)
    rot_ita = small_rot(0.10, 4) @ rot_spaporfre_int            # 3
    rot_itaron_int = small_rot(0.10, 12)
    rot_ron = small_rot(0.10, 5) @ rot_spaporfre_int @ rot_itaron_int  # 4
    rot_ita = rot_ita @ rot_itaron_int

    rots = [rot_spa, rot_por, rot_fre, rot_ita, rot_ron]
    # Shifts (means)
    means = [rng.standard_normal(d).astype(np.float32) * 0.3 for _ in range(5)]
    noise_scale = 0.02
    X = []
    for i in range(5):
        Xi = Z @ rots[i] + means[i]
        Xi = Xi + rng.standard_normal((n, d)).astype(np.float32) * noise_scale
        X.append(Xi)
    return X


def test_full_pipeline_runs():
    rng = np.random.default_rng(42)
    X_all = make_phylo_reps(rng)
    X_train = [x[:N_TRAIN] for x in X_all]
    X_test = [x[N_TRAIN:] for x in X_all]

    proc = fit_all_pairs(X_train)
    assert len(proc) == 10

    pair_errs = pairwise_reconstruction_error(X_test, proc)
    # Sanity: Spa-Por (sisters) should have the lowest reconstruction error,
    # and at least be lower than Spa-Ron (distant).
    e_spa_por = pair_errs[(0, 1)]
    e_spa_ron = pair_errs[(0, 4)]
    print(f"Spa-Por err={e_spa_por:.4f}, Spa-Ron err={e_spa_ron:.4f}")
    assert e_spa_por < e_spa_ron, "expected sister languages to reconstruct better"

    ce = compute_composition_errors(X_test, proc)
    assert len(ce) == 30  # 10 triples x 3 intermediates

    rows = rank_topologies(ce)
    assert len(rows) == 15
    # Print rank of ground truth for diagnostic
    for i, (topo, score, votes, gt) in enumerate(rows, start=1):
        if gt:
            print(f"Synthetic-data ground-truth rank: {i}/15 (score={score:.4f}, votes={votes}/10)")
            break


# ---- run all ---------------------------------------------------------------

def main() -> int:
    tests = [
        ("enumerate_splits", test_enumerate_splits),
        ("enumerate_topologies", test_enumerate_topologies),
        ("ground_truth_is_in_enumeration", test_ground_truth_is_in_enumeration),
        ("get_outgroup_returns_member_of_triple", test_get_outgroup_returns_member_of_triple),
        ("procrustes_recovers_known_rotation", test_procrustes_recovers_known_rotation),
        ("apply_transform_direction_consistency", test_apply_transform_direction_consistency),
        ("full_pipeline_runs", test_full_pipeline_runs),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} FAILED")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
