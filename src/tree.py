"""Neighbor-joining trees + clade support + Mantel test.

Self-contained NJ implementation (Saitou-Nei 1987) so we don't pull in
skbio just for this. Sufficient at our scale (~25 taxa).

Three things this module provides:

- ``neighbor_joining(labels, D)`` → rooted binary tree object.
- ``newick(tree)`` → Newick string for downstream viewers.
- ``in_family_clade(tree, target, family_members)`` → bool, used for the
  Tier 2a bootstrap clade support stat (per conflict language, fraction
  of resamples in which it lands within its true family).
- ``mantel(D1, D2, n_perms)`` → (r, p) for Tier 2c (vs ASJP).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ---------- tree object ----------

@dataclass
class Node:
    name: str | None = None              # set on leaves; None on internal nodes
    children: list["Node"] = field(default_factory=list)
    branch_length: float = 0.0           # length of the edge to parent

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> list[str]:
        if self.is_leaf:
            return [self.name]  # type: ignore[list-item]
        out: list[str] = []
        for c in self.children:
            out.extend(c.leaves())
        return out


# ---------- neighbor joining ----------

def neighbor_joining(labels: list[str], D: np.ndarray) -> Node:
    """Saitou-Nei NJ. Returns the root of a rooted binary tree.

    The "root" is just the last join — NJ produces an unrooted tree; we
    expose it rooted at the final triplet's merge for traversal convenience.
    Branch lengths are NJ's standard estimates.
    """
    n = len(labels)
    assert D.shape == (n, n), f"D must be {n}x{n}"

    active: list[Node] = [Node(name=L) for L in labels]
    Dm = D.astype(np.float64, copy=True)
    # zero diagonal explicitly (numeric noise)
    np.fill_diagonal(Dm, 0.0)

    while len(active) > 2:
        m = len(active)
        row_sum = Dm.sum(axis=1)
        # Q(i,j) = (m-2)*D(i,j) - row_sum[i] - row_sum[j]
        Q = (m - 2) * Dm - row_sum[:, None] - row_sum[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i

        # Branch lengths from new node u to i and j.
        delta = (row_sum[i] - row_sum[j]) / (m - 2)
        d_ui = 0.5 * Dm[i, j] + 0.5 * delta
        d_uj = Dm[i, j] - d_ui

        # New distances from u to every other k.
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[i] = keep_mask[j] = False
        d_uk = 0.5 * (Dm[i, keep_mask] + Dm[j, keep_mask] - Dm[i, j])

        # Build new node.
        ci, cj = active[i], active[j]
        ci.branch_length = max(d_ui, 0.0)
        cj.branch_length = max(d_uj, 0.0)
        u = Node(children=[ci, cj])

        # Reduce matrix and active list.
        new_active = [active[k] for k in range(m) if k != i and k != j] + [u]
        D_sub = Dm[np.ix_(keep_mask, keep_mask)]
        m_new = m - 1
        Dm = np.zeros((m_new, m_new), dtype=np.float64)
        Dm[:m_new - 1, :m_new - 1] = D_sub
        Dm[:m_new - 1, m_new - 1] = d_uk
        Dm[m_new - 1, :m_new - 1] = d_uk
        active = new_active

    # Final two nodes: join with a single edge between them.
    a, b = active
    a.branch_length = max(Dm[0, 1], 0.0)
    b.branch_length = 0.0
    root = Node(children=[a, b])
    return root


# ---------- Newick ----------

def newick(node: Node) -> str:
    def emit(n: Node) -> str:
        if n.is_leaf:
            return f"{n.name}:{n.branch_length:.6f}"
        inner = ",".join(emit(c) for c in n.children)
        return f"({inner}):{n.branch_length:.6f}"

    if node.is_leaf:
        return f"{node.name};"
    inner = ",".join(emit(c) for c in node.children)
    return f"({inner});"


# ---------- clade support ----------

def _bipartitions(node: Node, all_leaves: frozenset[str]) -> list[frozenset[str]]:
    """Every non-trivial bipartition implied by the tree.

    Internal edge → frozenset(leaves below the child). The complement gives
    the leaves on the other side. We drop the trivial bipartitions (a single
    leaf vs everything else) since those add no clade info.
    """
    parts: list[frozenset[str]] = []
    def rec(n: Node):
        if n.is_leaf:
            return
        for c in n.children:
            leaves_below = frozenset(c.leaves())
            if 1 < len(leaves_below) < len(all_leaves):
                parts.append(leaves_below)
            rec(c)
    rec(node)
    return parts


def in_family_clade(
    tree: Node, target: str, family_members: list[str],
) -> bool:
    """Is ``target`` inside the smallest clade containing all ``family_members``?

    "Clade" in the unrooted sense: take every bipartition of the leaves; pick
    the side that contains all family members (must exist; if both sides
    contain a member then no single clade holds them and we return False);
    pick the smallest such side; check if target is in it.
    """
    all_leaves = frozenset(tree.leaves())
    fam = frozenset(family_members)
    parts = _bipartitions(tree, all_leaves)

    candidate_sides: list[frozenset[str]] = []
    for half in parts:
        other = all_leaves - half
        if fam.issubset(half):
            candidate_sides.append(half)
        elif fam.issubset(other):
            candidate_sides.append(other)
    if not candidate_sides:
        # Family is split — no single clade contains all members. Target
        # can't be "inside its family clade" because there isn't one.
        return False
    smallest = min(candidate_sides, key=len)
    return target in smallest


# ---------- Mantel test ----------

def mantel(D1: np.ndarray, D2: np.ndarray, n_perms: int = 1000,
           seed: int = 0) -> tuple[float, float]:
    """Pearson Mantel test on upper-triangle entries.

    Returns ``(r, p)``. ``p`` is two-sided: fraction of permutations whose
    |r'| ≥ |r_observed|.
    """
    assert D1.shape == D2.shape and D1.shape[0] == D1.shape[1]
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    v1 = D1[iu]
    v2 = D2[iu]
    r_obs = float(np.corrcoef(v1, v2)[0, 1])

    rng = np.random.default_rng(seed)
    rs = np.empty(n_perms, dtype=np.float64)
    for p in range(n_perms):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        rs[p] = np.corrcoef(v1, D2p[iu])[0, 1]
    p = float(np.mean(np.abs(rs) >= abs(r_obs)))
    return r_obs, p


def submatrix(D: np.ndarray, labels: list[str], keep: list[str],
              ) -> tuple[list[str], np.ndarray]:
    """Restrict distance matrix to a subset of labels."""
    idx = [labels.index(L) for L in keep]
    return keep, D[np.ix_(idx, idx)]
