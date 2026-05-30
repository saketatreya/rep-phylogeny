"""Tier 0 — Atomic flip on {English, German, French}.

For each (layer, pool):
  surface(E,F), surface(E,G)             [constant across geometry sweep]
  procrustes_resid(E,F), procrustes_resid(E,G)
  hub_centroid(E,F),     hub_centroid(E,G)

Precondition (the experiment is dead if this fails):
  surface(E,F) < surface(E,G)            — surface really does pull E toward F
Eureka condition (under any geometry variant):
  geometry(E,F) > geometry(E,G)          — geometry pulls E toward G instead

Writes results/tier0_atomic.csv and SUMMARY rows for the decision table.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from src.config import LAYER_SWEEP, POOL_SWEEP
from src.data import load_sentences
from src.geometry import (
    hub_centroid_distance_matrix,
    procrustes_residual_distance,
)
from src.representations import load_reps
from src.surface import language_surface_distance


TIER0_LANGS = ["eng", "deu", "fra"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default="results/reps")
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--layers", nargs="+", default=list(LAYER_SWEEP))
    ap.add_argument("--pools", nargs="+", default=list(POOL_SWEEP))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = Path(args.reps_dir)

    # ---------- surface precondition ----------
    sents = load_sentences(TIER0_LANGS)
    langs, S = language_surface_distance(sents)
    si = {L: i for i, L in enumerate(langs)}
    s_ef = S[si["eng"], si["fra"]]
    s_eg = S[si["eng"], si["deu"]]
    precondition = s_ef < s_eg
    print(f"\nPRECONDITION  surface(E,F)={s_ef:.4f}  surface(E,G)={s_eg:.4f}  "
          f"=> surface pulls Eng toward {'French' if s_ef < s_eg else 'German'}  "
          f"[{'OK' if precondition else 'FAIL'}]")

    # ---------- geometry sweep ----------
    rows = []
    for pool in args.pools:
        for layer in args.layers:
            reps = load_reps(reps_dir, args.model, pool, layer, TIER0_LANGS)

            pr_ef = procrustes_residual_distance(reps["eng"], reps["fra"])
            pr_fe = procrustes_residual_distance(reps["fra"], reps["eng"])
            pr_eg = procrustes_residual_distance(reps["eng"], reps["deu"])
            pr_ge = procrustes_residual_distance(reps["deu"], reps["eng"])
            pr_ef_sym = 0.5 * (pr_ef + pr_fe)
            pr_eg_sym = 0.5 * (pr_eg + pr_ge)

            hub_langs, Hub, _ = hub_centroid_distance_matrix(reps)
            hi = {L: i for i, L in enumerate(hub_langs)}
            hc_ef = Hub[hi["eng"], hi["fra"]]
            hc_eg = Hub[hi["eng"], hi["deu"]]

            flip_pr = pr_ef_sym > pr_eg_sym
            flip_hc = hc_ef > hc_eg

            rows.append({
                "pool": pool, "layer": layer,
                "surface_E_F": f"{s_ef:.4f}", "surface_E_G": f"{s_eg:.4f}",
                "procrustes_resid_E_F": f"{pr_ef_sym:.4f}",
                "procrustes_resid_E_G": f"{pr_eg_sym:.4f}",
                "hub_centroid_E_F": f"{hc_ef:.4f}",
                "hub_centroid_E_G": f"{hc_eg:.4f}",
                "flip_procrustes": "Y" if flip_pr else "N",
                "flip_hub_centroid": "Y" if flip_hc else "N",
                "flip_any": "Y" if (flip_pr or flip_hc) else "N",
            })
            print(f"  {pool:>9} / {layer}:  "
                  f"proc E-F={pr_ef_sym:.3f} E-G={pr_eg_sym:.3f} "
                  f"[{'FLIP' if flip_pr else 'no'}]   "
                  f"hub E-F={hc_ef:.3f} E-G={hc_eg:.3f} "
                  f"[{'FLIP' if flip_hc else 'no'}]")

    csv_path = out_dir / "tier0_atomic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # Summary block — appended to SUMMARY.txt.
    any_flip = any(r["flip_any"] == "Y" for r in rows)
    best = next((r for r in rows if r["flip_any"] == "Y"), None)
    summary_lines = [
        "=== TIER 0 ===",
        f"PRECONDITION (surface pulls Eng→Fra): "
        f"{'PASS' if precondition else 'FAIL'}  "
        f"(surf E-F={s_ef:.4f}, E-G={s_eg:.4f})",
        f"FLIP (geometry pulls Eng→Deu, any variant): "
        f"{'PASS' if any_flip else 'FAIL'}",
    ]
    if best is not None:
        summary_lines.append(
            f"  best config: pool={best['pool']} layer={best['layer']}  "
            f"procrustes_flip={best['flip_procrustes']}  "
            f"hub_flip={best['flip_hub_centroid']}"
        )
    summary_lines.append("")
    summary_path = out_dir / "SUMMARY.txt"
    with open(summary_path, "a") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"appended to {summary_path}")
    return 0 if any_flip else 1


if __name__ == "__main__":
    raise SystemExit(main())
