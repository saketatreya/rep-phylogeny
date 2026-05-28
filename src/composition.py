"""Compute composition errors on the test split.

For each unordered triple {A, B, C} and each intermediate I in {A, B, C}:
    error(A, I, C) = mean_x ||T_{I->C}(T_{A->I}(x_A)) - T_{A->C}(x_A)||_2
The two directed orderings of the (src, dst) pair are averaged into a single
value per (triple, intermediate). Output is keyed by (frozenset(triple), I).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np

from .procrustes import apply_transform


def compute_composition_errors(
    X_test: list[np.ndarray],
    proc: dict,
    n_langs: int = 5,
) -> dict[tuple[frozenset, int], float]:
    """Return dict mapping (frozenset(triple), intermediate) -> mean L2 error."""
    out: dict[tuple[frozenset, int], float] = {}

    for triple in combinations(range(n_langs), 3):
        triple_set = frozenset(triple)
        for intermediate in triple:
            others = [x for x in triple if x != intermediate]
            src, dst = others[0], others[1]

            err_fwd = _composition_error(X_test, src, intermediate, dst, proc)
            err_rev = _composition_error(X_test, dst, intermediate, src, proc)

            out[(triple_set, intermediate)] = 0.5 * (err_fwd + err_rev)

    return out


def _composition_error(
    X_test: list[np.ndarray],
    src: int,
    intermediate: int,
    dst: int,
    proc: dict,
) -> float:
    x_src = X_test[src]
    indirect = apply_transform(
        apply_transform(x_src, src, intermediate, proc),
        intermediate, dst, proc,
    )
    direct = apply_transform(x_src, src, dst, proc)
    diff = indirect - direct
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def comp_error_table_str(ce: dict, lang_names: list[str]) -> str:
    """Pretty 10x3 table: rows=triples, columns=intermediate."""
    lines = []
    triples = list(combinations(range(len(lang_names)), 3))
    short = [n[:3] for n in lang_names]
    # Header
    header = f"{'Triple':<20} | " + "  ".join(f"int={s:<3}" for s in short) + "    (intermediate)"
    lines.append(header)
    lines.append("-" * len(header))
    for triple in triples:
        tname = ",".join(short[i] for i in triple)
        tset = frozenset(triple)
        cells = []
        for i in range(len(lang_names)):
            if i in triple:
                cells.append(f"{ce[(tset, i)]:8.4f}")
            else:
                cells.append("    -   ")
        lines.append(f"{tname:<20} | " + "  ".join(cells))
    return "\n".join(lines)
