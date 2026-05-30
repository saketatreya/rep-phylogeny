"""Tier 2 — Universal axis. The prize.

Runs four sub-experiments on the full ~25 language set:

  2a. NJ trees + bootstrap clade support per conflict language.
  2b. Single Germanic-vs-Romance axis; transfer-test on Maltese, Urdu, Romanian.
  2c. Mantel correlation against a reference genealogical distance.
  2d. Transliteration control: romanized surface vs geometry for cross-script
      conflict pairs.

Designed so each sub-tier can fail independently and the others still publish.
"""
from __future__ import annotations
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.bootstrap import bootstrap_indices, ci
from src.classify import fit_family_discriminant, stack_residuals
from src.config import (
    BY_NAME, CLEAN_ANCHORS, CONFLICT_CASES, LANGS, LAYER_SWEEP,
    N_TEST, N_TRAIN, POOL_SWEEP, anchors_in_family, conflict_truth,
)
from src.data import load_sentences
from src.geometry import (
    fit_procrustes_np, hub_centroid_distance_matrix,
    procrustes_distance_matrix, procrustes_residual_distance,
)
from src.hub import form_residuals, hub_align
from src.representations import load_reps
from src.surface import language_surface_distance, romanize_for_surface
from src.tree import (
    in_family_clade, mantel, neighbor_joining, newick, submatrix,
)


# ---------- 2a. NJ trees + bootstrap clade support ----------

def _per_sentence_distance_block(
    reps_by_lang: dict[str, np.ndarray], variant: str, n_train: int,
):
    """Precompute per-sentence test-split residual contributions so the
    bootstrap can resample without refitting transforms each time.

    Returns a callable ``avg(idx) -> (n_lang, n_lang) distance matrix`` for
    the given variant. Refits happen on TRAIN; only the test-side residuals
    are bootstrapped.
    """
    langs = list(reps_by_lang.keys())
    n = len(langs)

    if variant == "procrustes_resid":
        # For each unordered pair, store per-sentence residual on test split.
        per_pair = {}
        for i, j in combinations(range(n), 2):
            ri_te = procrustes_residual_distance(
                reps_by_lang[langs[i]], reps_by_lang[langs[j]],
                n_train=n_train, return_per_sentence=True)
            rj_te = procrustes_residual_distance(
                reps_by_lang[langs[j]], reps_by_lang[langs[i]],
                n_train=n_train, return_per_sentence=True)
            per_pair[(i, j)] = 0.5 * (ri_te + rj_te)  # (n_test,)

        def avg(idx: np.ndarray) -> np.ndarray:
            D = np.zeros((n, n), dtype=np.float64)
            for (i, j), v in per_pair.items():
                D[i, j] = D[j, i] = float(v[idx].mean())
            return D
        return langs, avg

    if variant == "hub_centroid":
        # Fit hub align on full data (TRAIN only inside hub_align); apply
        # to test; bootstrap centroids by averaging aligned-test rows.
        aligned, grand_mean, _ = hub_align(reps_by_lang, n_train=n_train, n_iters=5)
        aligned_test = {L: aligned[L][n_train:] - grand_mean for L in langs}
        # Stack to (n_lang, n_test, d).
        stack = np.stack([aligned_test[L] for L in langs], axis=0)

        def avg(idx: np.ndarray) -> np.ndarray:
            cents = stack[:, idx, :].mean(axis=1)  # (n_lang, d)
            diff = cents[:, None, :] - cents[None, :, :]
            return np.linalg.norm(diff, axis=2)
        return langs, avg

    raise ValueError(f"unknown variant {variant!r}")


def tier2a_clade_support(
    reps_by_lang: dict[str, np.ndarray],
    raw_text_by_lang: dict[str, list[str]],
    n_boot: int, seed: int, out_dir: Path, tag: str,
) -> dict:
    """Build NJ trees from surface, procrustes_resid, hub_centroid; bootstrap
    each over test sentences; report per-conflict-language clade support.
    """
    langs = list(reps_by_lang.keys())

    # Reference NJ trees (no bootstrap).
    surf_langs, S = language_surface_distance(
        {L: raw_text_by_lang[L] for L in langs})
    tree_surf = neighbor_joining(surf_langs, S)
    _, proc_avg = _per_sentence_distance_block(reps_by_lang, "procrustes_resid", N_TRAIN)
    P = proc_avg(np.arange(N_TEST))
    tree_proc = neighbor_joining(langs, P)
    _, hub_avg = _per_sentence_distance_block(reps_by_lang, "hub_centroid", N_TRAIN)
    H = hub_avg(np.arange(N_TEST))
    tree_hub = neighbor_joining(langs, H)

    trees_dir = out_dir / "trees"; trees_dir.mkdir(parents=True, exist_ok=True)
    (trees_dir / f"{tag}_surface.nwk").write_text(newick(tree_surf))
    (trees_dir / f"{tag}_procrustes.nwk").write_text(newick(tree_proc))
    (trees_dir / f"{tag}_hub.nwk").write_text(newick(tree_hub))

    # Bootstrap clade support.
    boot_idx = bootstrap_indices(N_TEST, n_boot, seed=seed)
    families = {L: BY_NAME[L].family for L in langs}

    results: dict[str, dict] = {}
    for cname in CONFLICT_CASES:
        if cname not in langs:
            continue
        truth_family = conflict_truth(cname)
        members = [L for L in langs if families[L] == truth_family and L != cname]
        if not members:
            results[cname] = {"truth_family": truth_family, "note": "no anchors in tree"}
            continue

        in_count = {"surface": 0, "procrustes_resid": 0, "hub_centroid": 0}

        # Surface tree doesn't change under sentence resampling (uses full
        # devtest text). So its "bootstrap" is just the point estimate
        # repeated. Record it as a single binary outcome.
        ref_in = {
            "surface": in_family_clade(tree_surf, cname, members),
            "procrustes_resid_point": in_family_clade(tree_proc, cname, members),
            "hub_centroid_point": in_family_clade(tree_hub, cname, members),
        }

        for b in range(n_boot):
            idx = boot_idx[b]
            P_b = proc_avg(idx)
            t_p = neighbor_joining(langs, P_b)
            if in_family_clade(t_p, cname, members):
                in_count["procrustes_resid"] += 1
            H_b = hub_avg(idx)
            t_h = neighbor_joining(langs, H_b)
            if in_family_clade(t_h, cname, members):
                in_count["hub_centroid"] += 1

        results[cname] = {
            "truth_family": truth_family,
            "surface_in_clade": bool(ref_in["surface"]),
            "procrustes_in_clade_point": bool(ref_in["procrustes_resid_point"]),
            "hub_in_clade_point": bool(ref_in["hub_centroid_point"]),
            "p_in_clade_procrustes": in_count["procrustes_resid"] / n_boot,
            "p_in_clade_hub": in_count["hub_centroid"] / n_boot,
        }

    # Write CSV.
    rows = []
    for cname, r in results.items():
        rows.append({
            "conflict_lang": cname,
            "truth_family": r["truth_family"],
            "surface_in_clade": r.get("surface_in_clade", ""),
            "procrustes_in_clade_point": r.get("procrustes_in_clade_point", ""),
            "hub_in_clade_point": r.get("hub_in_clade_point", ""),
            "p_in_clade_procrustes": f"{r.get('p_in_clade_procrustes', float('nan')):.4f}",
            "p_in_clade_hub": f"{r.get('p_in_clade_hub', float('nan')):.4f}",
        })
    csv_path = out_dir / f"tier2a_clade_support_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return {"tag": tag, "results": results, "csv": str(csv_path)}


# ---------- 2b. Universal axis ----------

def tier2b_axis_transfer(
    reps_by_lang: dict[str, np.ndarray],
    out_dir: Path, tag: str,
) -> dict:
    """Fit LDA on Germanic-vs-Romance form residuals, transfer-test each
    conflict case by comparing its axis projection to its descent vs
    contact family-anchor projections.

    A conflict case "passes" iff |proj(C) - mean_proj(descent anchors)| <
    |proj(C) - mean_proj(contact anchors)|.
    """
    aligned, _, _ = hub_align(reps_by_lang, n_train=N_TRAIN, n_iters=5)
    train_resid = form_residuals(aligned, n_train=N_TRAIN, split="train")
    test_resid = form_residuals(aligned, n_train=N_TRAIN, split="test")

    germ = anchors_in_family("Germanic")
    rom = anchors_in_family("Romance")
    label_map = {L: "Germanic" for L in germ} | {L: "Romance" for L in rom}
    label_map = {L: v for L, v in label_map.items() if L in train_resid}

    disc = fit_family_discriminant({L: train_resid[L] for L in label_map},
                                   label_map, method="lda")
    axis = disc.axis()
    if axis is None:
        raise RuntimeError("LDA produced no single axis (unexpected for 2-class)")

    # Mean projection per language (test centroid · axis).
    proj = {L: float(test_resid[L].mean(axis=0) @ axis) for L in reps_by_lang}

    # Contact mapping: for each conflict case, surface-similar family (the
    # one whose vocabulary is borrowed in).
    contact = {
        "eng": "Romance",     # ~56% Romance vocabulary
        "mlt": "Romance",     # Italian/Sicilian vocabulary
        "ron": "Slavic",      # Slavic admixture
        "urd": "Iranian",     # Perso-Arabic vocabulary
    }

    rows = []
    for cname in CONFLICT_CASES:
        if cname not in reps_by_lang:
            continue
        descent = conflict_truth(cname)
        contact_fam = contact.get(cname, "?")
        descent_anchors = [L for L in anchors_in_family(descent)
                           if L in reps_by_lang and L != cname]
        contact_anchors = [L for L in anchors_in_family(contact_fam)
                           if L in reps_by_lang and L != cname]
        if not descent_anchors or not contact_anchors:
            continue
        c_proj = proj[cname]
        d_proj = float(np.mean([proj[L] for L in descent_anchors]))
        x_proj = float(np.mean([proj[L] for L in contact_anchors]))
        d_dist = abs(c_proj - d_proj)
        x_dist = abs(c_proj - x_proj)
        pass_ = d_dist < x_dist
        rows.append({
            "conflict_lang": cname,
            "descent_family": descent,
            "contact_family": contact_fam,
            "proj_conflict": f"{c_proj:.4f}",
            "proj_descent_mean": f"{d_proj:.4f}",
            "proj_contact_mean": f"{x_proj:.4f}",
            "abs_diff_descent": f"{d_dist:.4f}",
            "abs_diff_contact": f"{x_dist:.4f}",
            "axis_orients_correctly": "Y" if pass_ else "N",
        })

    csv_path = out_dir / f"tier2b_axis_transfer_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return {"tag": tag, "rows": rows, "csv": str(csv_path), "axis": axis,
            "projections": proj}


# ---------- 2c. Mantel test ----------

def glottolog_reference_distance(langs: list[str]) -> np.ndarray:
    """Coarse reference distance from family membership in config.LANGS.

    Categorical: 0 (same family) / 1 (different family). For more
    fine-grained comparison, swap this for ASJP lexical distances —
    download from https://asjp.clld.org/ and replace this function.
    """
    n = len(langs)
    fam = {L: BY_NAME[L].family for L in langs}
    D = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            if fam[langs[i]] == fam[langs[j]]:
                D[i, j] = D[j, i] = 0.0
    return D


def tier2c_mantel(
    surf_labels: list[str], S: np.ndarray,
    geom_langs: list[str], Geom: np.ndarray,
    out_dir: Path, tag: str, conflict_only: bool = False,
) -> dict:
    # Reorder surface to match geometry order.
    perm = [surf_labels.index(L) for L in geom_langs]
    S_re = S[np.ix_(perm, perm)]
    ref = glottolog_reference_distance(geom_langs)

    r_geom, p_geom = mantel(Geom, ref, n_perms=1000)
    r_surf, p_surf = mantel(S_re, ref, n_perms=1000)

    row = {
        "scope": "all",
        "r_surface_vs_ref": f"{r_surf:.4f}",
        "p_surface_vs_ref": f"{p_surf:.4f}",
        "r_geometry_vs_ref": f"{r_geom:.4f}",
        "p_geometry_vs_ref": f"{p_geom:.4f}",
    }
    rows = [row]

    # Submatrix on conflict cases + their relevant anchors (the test where
    # geometry should beat surface most cleanly).
    conflict_sub = list(set(CONFLICT_CASES + CLEAN_ANCHORS) & set(geom_langs))
    if len(conflict_sub) >= 4:
        idx = [geom_langs.index(L) for L in conflict_sub]
        S_sub = S_re[np.ix_(idx, idx)]
        G_sub = Geom[np.ix_(idx, idx)]
        ref_sub = glottolog_reference_distance(conflict_sub)
        r_gs, p_gs = mantel(G_sub, ref_sub, n_perms=1000)
        r_ss, p_ss = mantel(S_sub, ref_sub, n_perms=1000)
        rows.append({
            "scope": "with_conflicts",
            "r_surface_vs_ref": f"{r_ss:.4f}",
            "p_surface_vs_ref": f"{p_ss:.4f}",
            "r_geometry_vs_ref": f"{r_gs:.4f}",
            "p_geometry_vs_ref": f"{p_gs:.4f}",
        })

    csv_path = out_dir / f"tier2c_mantel_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return {"tag": tag, "rows": rows, "csv": str(csv_path)}


# ---------- 2d. Transliteration control ----------

def tier2d_translit(
    raw_text_by_lang: dict[str, list[str]],
    geom_langs: list[str], Geom: np.ndarray,
    out_dir: Path, tag: str,
) -> dict:
    """Romanize all texts, recompute surface, compare geometry to romanized
    surface for cross-script conflict pairs.
    """
    print("  romanizing all sentences (this may take a moment) ...")
    romanized = {L: [romanize_for_surface(s) for s in raw_text_by_lang[L]]
                 for L in raw_text_by_lang}
    surf_langs, S_rom = language_surface_distance(romanized)

    # The pairs we care about — cross-script genealogy-vs-contact.
    pairs = [("urd", "hin"),   # both Indo-Aryan, different script
             ("urd", "pes"),   # Urdu surface neighbour
             ("hin", "pes"),   # Indo-Aryan vs Iranian baseline
             ("mlt", "arb"),   # Maltese vs Semitic anchor
             ("mlt", "ita")]   # Maltese vs surface neighbour

    rows = []
    for A, B in pairs:
        if A not in surf_langs or B not in surf_langs:
            continue
        ia, ib = surf_langs.index(A), surf_langs.index(B)
        ga, gb = geom_langs.index(A), geom_langs.index(B)
        rows.append({
            "pair": f"{A}-{B}",
            "romanized_surface_distance": f"{S_rom[ia, ib]:.4f}",
            "geometry_distance": f"{Geom[ga, gb]:.4f}",
        })

    csv_path = out_dir / f"tier2d_translit_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return {"tag": tag, "rows": rows, "csv": str(csv_path)}


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default="results/reps")
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--layers", nargs="+", default=list(LAYER_SWEEP))
    ap.add_argument("--pools", nargs="+", default=list(POOL_SWEEP))
    ap.add_argument("--n-boot", type=int, default=200,
                    help="bootstrap iterations for clade support "
                         "(200 is enough at ~25 taxa; expensive otherwise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-translit", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = Path(args.reps_dir)

    # Use the full set of available languages from disk.
    avail = []
    for L in [Ll.name for Ll in LANGS]:
        if (reps_dir / args.model / args.pools[0] / args.layers[0]
                / f"{L}.npy").exists():
            avail.append(L)
    print(f"Available languages on disk: {avail}")

    raw_text = load_sentences(avail)

    summary_blocks: list[str] = ["=== TIER 2 ==="]
    for pool in args.pools:
        for layer in args.layers:
            tag = f"{pool}_{layer}"
            print(f"\n===== TIER 2 — pool={pool}  layer={layer} =====")
            reps = load_reps(reps_dir, args.model, pool, layer, avail)

            r2a = tier2a_clade_support(reps, raw_text, args.n_boot,
                                       args.seed, out_dir, tag)
            r2b = tier2b_axis_transfer(reps, out_dir, tag)

            # Build full geometry distance matrix once for 2c/2d.
            geom_langs, Geom, _ = hub_centroid_distance_matrix(reps)
            surf_langs, S = language_surface_distance(raw_text)
            r2c = tier2c_mantel(surf_langs, S, geom_langs, Geom, out_dir, tag)

            if not args.skip_translit:
                r2d = tier2d_translit(raw_text, geom_langs, Geom, out_dir, tag)
            else:
                r2d = None

            block = [f"--- {tag} ---"]
            block.append("clade support (P_in_clade, hub):  " + "  ".join(
                f"{c}={r2a['results'][c].get('p_in_clade_hub', float('nan')):.2f}"
                for c in r2a["results"]))
            block.append("axis transfer:  " + "  ".join(
                f"{r['conflict_lang']}={r['axis_orients_correctly']}"
                for r in r2b["rows"]))
            mantel_all = next((r for r in r2c["rows"] if r["scope"] == "all"), None)
            if mantel_all:
                block.append(
                    f"Mantel vs glottolog: surf r={mantel_all['r_surface_vs_ref']}  "
                    f"geom r={mantel_all['r_geometry_vs_ref']}"
                )
            summary_blocks.append("\n".join(block))

    summary_blocks.append("")
    with open(out_dir / "SUMMARY.txt", "a") as f:
        f.write("\n".join(summary_blocks) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
