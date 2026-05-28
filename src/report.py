"""Pretty-printed reports for a single (model, layer) configuration."""
from __future__ import annotations
from itertools import combinations
import numpy as np

from .config import LANG_NAMES
from .topologies import (
    Topology,
    enumerate_topologies,
    get_outgroup,
    is_ground_truth,
    rank_topologies,
    topology_id,
    vote_count_score,
)


def _short(name: str) -> str:
    return name[:3]


def cosine_similarity_table(X: list[np.ndarray]) -> str:
    """Mean cosine similarity per language pair (sanity check)."""
    lines = ["Pairwise mean cosine similarity (higher = closer):"]
    n = len(X)
    # normalize once
    Xn = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in X]
    for A, B in combinations(range(n), 2):
        sim = float(np.mean(np.sum(Xn[A] * Xn[B], axis=1)))
        lines.append(f"  {_short(LANG_NAMES[A])}-{_short(LANG_NAMES[B])}: {sim:+.4f}")
    return "\n".join(lines)


def reconstruction_error_table(errs: dict[tuple[int, int], float]) -> str:
    """Sanity check: closer language pairs should reconstruct each other better."""
    lines = ["Pairwise Procrustes reconstruction error (test set, lower = closer):"]
    for (A, B), e in errs.items():
        lines.append(f"  {_short(LANG_NAMES[A])}-{_short(LANG_NAMES[B])}: {e:.4f}")
    return "\n".join(lines)


def topology_ranking_table(ce: dict, top_n: int | None = None) -> str:
    rows = rank_topologies(ce)
    if top_n is None:
        top_n = len(rows)
    lines = ["TOPOLOGY RANKING (by composition score, lower = better):"]
    lines.append("")
    lines.append(f"{'Rank':<5}| {'Score':<10}| {'Votes':<6}| {'Splits':<42}| Ground truth?")
    lines.append("-" * 90)
    gt_rank = None
    for i, (topo, score, votes, gt) in enumerate(rows, start=1):
        marker = "  <-- GROUND TRUTH" if gt else ""
        if gt:
            gt_rank = i
        tid = topology_id(topo)
        lines.append(
            f"{i:<5}| {score:<10.6f}| {votes:<6}| {tid:<42}| {'YES' if gt else ''}{marker}"
        )
        if top_n is not None and i >= top_n and not gt:
            # peek ahead to find ground truth even if outside top_n
            continue
    lines.append("")
    lines.append(f"Ground-truth topology rank: {gt_rank} / {len(rows)}")
    return "\n".join(lines)


def outgroup_prediction_table(
    ce: dict,
    topology: Topology,
) -> str:
    """For each triple, show predicted vs empirical outgroup."""
    lines = ["OUTGROUP PREDICTION (ground-truth topology, split-based):"]
    correct = 0
    total = 0
    for triple in combinations(range(5), 3):
        triple_set = frozenset(triple)
        predicted = get_outgroup(topology, triple)
        empirical = max(triple, key=lambda i: ce[(triple_set, i)])
        ok = predicted == empirical
        correct += int(ok)
        total += 1
        tname = ",".join(_short(LANG_NAMES[i]) for i in triple)
        lines.append(
            f"  {{{tname}}}  pred={_short(LANG_NAMES[predicted])}  "
            f"emp={_short(LANG_NAMES[empirical])}  "
            f"{'OK' if ok else '--'}"
        )
    lines.append(f"  Correct: {correct}/{total}")
    return "\n".join(lines)


def full_report(
    title: str,
    X_train: list[np.ndarray],
    X_test: list[np.ndarray],
    ce: dict,
    pair_errs: dict[tuple[int, int], float],
) -> str:
    """Compose the full report for one (model, layer) configuration."""
    from .topologies import GROUND_TRUTH_SPLITS

    out = []
    out.append("=" * 72)
    out.append(f"=== RESULTS: {title}")
    out.append("=" * 72)
    out.append("")
    out.append(topology_ranking_table(ce))
    out.append("")
    out.append(outgroup_prediction_table(ce, GROUND_TRUTH_SPLITS))
    out.append("")
    out.append(cosine_similarity_table(X_test))
    out.append("")
    out.append(reconstruction_error_table(pair_errs))
    out.append("")
    return "\n".join(out)
