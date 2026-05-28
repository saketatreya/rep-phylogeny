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
    """Return the taxon in `triple` predicted to be the outgroup under
    `topology`, defined as the one separated from the other two by a split
    (1-vs-2 partition). The first split that 1-vs-2-separates the triple
    decides. For 5-leaf binary trees this always returns a value.

    NB: this is the *split-based* outgroup. For triples that span the
    central edge it may differ from the rooted-tree outgroup (see header
    comment). The algorithm is deterministic and is what scores topologies.
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


def score_topology(topology: Topology, ce: dict[tuple[frozenset, int], float]) -> float:
    """Sum of composition errors through the 2 ingroup intermediates of each
    triple. The topology predicts the outgroup is a bad intermediate, so a
    topology that correctly identifies the outgroups has the lowest sum.
    """
    total = 0.0
    for triple in combinations(range(5), 3):
        outgroup = get_outgroup(topology, triple)
        for intermediate in triple:
            if intermediate != outgroup:
                total += ce[(frozenset(triple), intermediate)]
    return total


def vote_count_score(topology: Topology, ce: dict[tuple[frozenset, int], float]) -> int:
    """Diagnostic: for each triple find the empirical outgroup
    (intermediate with the highest composition error), count agreements."""
    correct = 0
    for triple in combinations(range(5), 3):
        empirical_outgroup = max(triple, key=lambda i: ce[(frozenset(triple), i)])
        predicted_outgroup = get_outgroup(topology, triple)
        if empirical_outgroup == predicted_outgroup:
            correct += 1
    return correct


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


def rank_topologies(
    ce: dict[tuple[frozenset, int], float],
) -> list[tuple[Topology, float, int, bool]]:
    """Return list of (topology, score, vote_count, is_ground_truth)
    sorted ascending by score."""
    topologies = enumerate_topologies()
    rows = []
    for topo in topologies:
        s = score_topology(topo, ce)
        v = vote_count_score(topo, ce)
        rows.append((topo, s, v, is_ground_truth(topo)))
    rows.sort(key=lambda r: r[1])
    return rows
