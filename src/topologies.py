"""Enumerate the 15 unrooted binary tree topologies on 5 labeled taxa and
score each one against the composition-error table.

A split = a partition of {0..4} into a size-2 group and a size-3 group.
A topology = a pair of compatible splits.

Ground-truth Romance topology (consensus linguistic phylogeny, unrooted):
  splits = ({0,1}, {2,3,4}) and ({3,4}, {0,1,2})
i.e. (Spanish,Portuguese) cherry and (Italian,Romanian) cherry, with French
on the central edge. NB: in the *unrooted* tree, Italian and Romanian form
a cherry — this is unavoidable when unrooting ((((Spa,Por),Fre),Ita),Ron).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np


Split = tuple[frozenset, frozenset]   # (small, big), |small|=2, |big|=3
Topology = tuple[Split, Split]


def enumerate_splits(n: int = 5) -> list[Split]:
    """All non-trivial splits of {0..n-1} into groups of size 2 and (n-2)."""
    taxa = set(range(n))
    splits = []
    for pair in combinations(range(n), 2):
        small = frozenset(pair)
        big = frozenset(taxa - small)
        splits.append((small, big))
    return splits


def are_compatible(s1: Split, s2: Split) -> bool:
    """Two splits are compatible iff one of the four intersections is empty."""
    a1, b1 = s1
    a2, b2 = s2
    return (not a1 & a2) or (not a1 & b2) or (not b1 & a2) or (not b1 & b2)


def enumerate_topologies() -> list[Topology]:
    """All 15 unrooted binary trees on 5 labeled taxa as pairs of splits."""
    splits = enumerate_splits()
    topologies = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            if splits[i] == splits[j]:
                continue
            if are_compatible(splits[i], splits[j]):
                topologies.append((splits[i], splits[j]))
    assert len(topologies) == 15, f"expected 15 topologies, got {len(topologies)}"
    return topologies


def get_outgroup(topology: Topology, triple: tuple[int, int, int]) -> int:
    """Return the split-based outgroup. The first split that 1-vs-2-separates
    the triple decides. NB: for 5-leaf unrooted trees, a triple may be
    1-vs-2-separated by *both* splits (giving conflicting answers) or by
    *neither* (a star triple). This function is kept for backward compat
    with the original scoring; use ``classify_triple`` for the diagnostic.
    """
    triple_set = set(triple)
    for split in topology:
        small, big = split
        in_small = triple_set & small
        in_big = triple_set & big
        if len(in_small) == 1 and len(in_big) == 2:
            return next(iter(in_small))
        if len(in_big) == 1 and len(in_small) == 2:
            return next(iter(in_big))
    raise ValueError(
        f"Topology {topology} does not 1-vs-2-separate triple {triple}; "
        f"this is impossible for a valid 5-leaf binary topology."
    )


def classify_triple(triple, splits) -> tuple[str, int | None]:
    """Resolved vs star classification under a given (split-pair) topology.

    Returns ``('resolved', outgroup_idx)`` if both splits agree on the
    outgroup (or only one split splits the triple non-trivially), and
    ``('star', None)`` if the two splits disagree (the triple straddles
    the central edge) or neither does.

    For n=5, exactly the 6 triples whose outgroup is *not* an internal
    edge-incident taxon end up resolved under the Romance ground truth.
    The other 4 induce star topologies — the scoring loop must skip them.
    """
    triple_set = set(triple)
    preds = []
    for small, big in splits:
        in_small = triple_set & small
        in_big = triple_set & big
        if len(in_small) == 1 and len(in_big) == 2:
            preds.append(next(iter(in_small)))
        elif len(in_big) == 1 and len(in_small) == 2:
            preds.append(next(iter(in_big)))
        # else: this split does not separate the triple non-trivially
    if len(preds) == 0:
        return "star", None
    if len(preds) == 1:
        return "resolved", preds[0]
    if preds[0] == preds[1]:
        return "resolved", preds[0]
    return "star", None


def score_topology(topology: Topology, ce: dict[tuple[frozenset, int], float]) -> float:
    """LEGACY sum-of-ingroup-errors score. Kept for back-compat reporting.

    Use ``score_topology_resolved`` for the resolved-triples-only scorer
    that avoids the n=5 degeneracy (every star triple silently corrupts
    this sum).
    """
    total = 0.0
    for triple in combinations(range(5), 3):
        outgroup = get_outgroup(topology, triple)
        for intermediate in triple:
            if intermediate != outgroup:
                total += ce[(frozenset(triple), intermediate)]
    return total


def score_topology_resolved(
    topology: Topology,
    ce: dict[tuple[frozenset, int], float],
) -> tuple[int, int]:
    """Resolved-triples-only outgroup-prediction score.

    Returns ``(correct, n_resolved)``: number of resolved triples whose
    empirical worst intermediate matches the topology's predicted outgroup,
    over the number of resolved triples (typically 6/10 for n=5).
    """
    correct = 0
    n_resolved = 0
    for triple in combinations(range(5), 3):
        status, pred = classify_triple(triple, topology)
        if status != "resolved":
            continue
        n_resolved += 1
        emp = max(triple, key=lambda i: ce[(frozenset(triple), i)])
        if emp == pred:
            correct += 1
    return correct, n_resolved


def vote_count_score(topology: Topology, ce: dict[tuple[frozenset, int], float]) -> int:
    """Diagnostic (legacy): per-triple votes using split-based outgroup."""
    correct = 0
    for triple in combinations(range(5), 3):
        empirical_outgroup = max(triple, key=lambda i: ce[(frozenset(triple), i)])
        predicted_outgroup = get_outgroup(topology, triple)
        if empirical_outgroup == predicted_outgroup:
            correct += 1
    return correct


def score_all_splits(
    ce: dict[tuple[frozenset, int], float],
) -> dict[Split, int]:
    """For each of the 10 possible splits, count how many of its 3
    cherry-triples have the *correct* empirical outgroup.

    A split (small, big) is tested on each triple of the form
    ``small ∪ {outsider}`` for outsider in big. The split predicts the
    outsider as the outgroup.

    Returns dict mapping split -> correct (0..3).
    """
    out = {}
    for small, big in enumerate_splits(5):
        correct = 0
        for outsider in sorted(big):
            triple = tuple(sorted(small | {outsider}))
            errs = {i: ce[(frozenset(triple), i)] for i in triple}
            emp = max(errs, key=errs.get)
            if emp == outsider:
                correct += 1
        out[(small, big)] = correct
    return out


def topology_id(topology: Topology) -> str:
    """Canonical string representation: 'a,b|c,d,e ; f,g|h,i,j' (small|big)."""
    parts = []
    for small, big in topology:
        s = ",".join(str(x) for x in sorted(small))
        b = ",".join(str(x) for x in sorted(big))
        parts.append(f"{{{s}}}|{{{b}}}")
    return " ; ".join(sorted(parts))


# The consensus Romance topology, identified by its two splits.
GROUND_TRUTH_SPLITS = (
    (frozenset({0, 1}), frozenset({2, 3, 4})),  # (Spa,Por) | (Fre,Ita,Ron)
    (frozenset({3, 4}), frozenset({0, 1, 2})),  # (Ita,Ron) | (Spa,Por,Fre)
)


def is_ground_truth(topology: Topology) -> bool:
    """True iff this topology's splits match the consensus Romance tree."""
    target = {GROUND_TRUTH_SPLITS[0], GROUND_TRUTH_SPLITS[1]}
    return set(topology) == target


def is_ground_truth_split(split: Split) -> bool:
    return split in GROUND_TRUTH_SPLITS or (split[1], split[0]) in GROUND_TRUTH_SPLITS


def rank_topologies(
    ce: dict[tuple[frozenset, int], float],
    use_resolved: bool = False,
) -> list[tuple[Topology, float, int, bool]]:
    """Return list of (topology, score, vote_count, is_ground_truth)
    sorted ascending by score.

    If ``use_resolved`` is True, score = -(correct_resolved_predictions),
    so lower is better. Ties remain ties (15 topologies, 6 resolved each).
    """
    topologies = enumerate_topologies()
    rows = []
    for topo in topologies:
        if use_resolved:
            correct, _ = score_topology_resolved(topo, ce)
            s = -float(correct)
        else:
            s = score_topology(topo, ce)
        v = vote_count_score(topo, ce)
        rows.append((topo, s, v, is_ground_truth(topo)))
    rows.sort(key=lambda r: r[1])
    return rows


def get_rank_of_ground_truth(
    ce: dict[tuple[frozenset, int], float],
    use_resolved: bool = False,
) -> int:
    """1-based rank (ties resolved by sort order) of the GT topology."""
    rows = rank_topologies(ce, use_resolved=use_resolved)
    for i, (_, _, _, gt) in enumerate(rows, start=1):
        if gt:
            return i
    return -1
