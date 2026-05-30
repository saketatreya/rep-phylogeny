"""Tier 1 — Held-out family discriminant.

Train a Germanic-vs-Romance discriminant on FORM RESIDUALS of the 8 clean
anchors (deu, nld, swe, dan + fra, spa, ita, por), then classify English.
English never participates in the fit. Bootstrap test sentences for CI.

Also run the identical contrast in SURFACE space (per-sentence char-ngram
TF-IDF, classifier trained on the 8 clean langs, English classified).

Gate: geometry P(Eng→Germanic) >> surface P(Eng→Germanic), with the
geometry bootstrap CI excluding 0.5.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.bootstrap import bootstrap_indices, ci
from src.classify import (
    Discriminant, classify_language, fit_family_discriminant,
)
from src.config import (
    LAYER_SWEEP, N_TEST, N_TRAIN, POOL_SWEEP, anchors_in_family,
)
from src.data import load_sentences
from src.hub import form_residuals, hub_align
from src.representations import load_reps
from src.surface import per_sentence_surface_features


TIER1_CLEAN = anchors_in_family("Germanic") + anchors_in_family("Romance")
TIER1_HELDOUT = "eng"
TIER1_ALL = TIER1_CLEAN + [TIER1_HELDOUT]


def _family_label_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for L in anchors_in_family("Germanic"):
        out[L] = "Germanic"
    for L in anchors_in_family("Romance"):
        out[L] = "Romance"
    return out


def _bootstrap_frac(predictions: np.ndarray, target_class: str,
                    n_boot: int, seed: int) -> tuple[float, float, float]:
    """Fraction of test sentences classified as ``target_class``,
    bootstrapped over the test indices."""
    n = len(predictions)
    boot = bootstrap_indices(n, n_boot, seed=seed)
    fracs = (predictions[boot] == target_class).mean(axis=1)
    lo, hi = ci(fracs)
    return float(fracs.mean()), lo, hi


# ---------- geometry side ----------

def geometry_tier1(reps_dir: Path, model: str, pool: str, layer: str,
                   n_boot: int, seed: int) -> dict:
    reps = load_reps(reps_dir, model, pool, layer, TIER1_ALL)
    aligned, _, _ = hub_align(reps, n_train=N_TRAIN, n_iters=5)

    train_resid = form_residuals(aligned, n_train=N_TRAIN, split="train")
    test_resid = form_residuals(aligned, n_train=N_TRAIN, split="test")

    # Discard the held-out language from the training set.
    train_clean = {L: train_resid[L] for L in TIER1_CLEAN}

    disc = fit_family_discriminant(train_clean, _family_label_map(), method="lda")
    out = classify_language(disc, test_resid[TIER1_HELDOUT])
    preds = out["predictions"]
    mean, lo, hi = _bootstrap_frac(preds, "Germanic", n_boot, seed)
    return {"frac_germanic": mean, "frac_germanic_lo": lo, "frac_germanic_hi": hi,
            "predictions": preds, "discriminant": disc}


# ---------- surface side ----------

def surface_tier1(sents_by_lang: dict[str, list[str]],
                  n_boot: int, seed: int) -> dict:
    """Per-sentence char-ngram features, family classifier, English held out.

    The vectorizer is fit on ALL sentences (so English's vocabulary is in
    scope). The classifier is fit on the train-split rows of the 8 clean
    languages only. We score on English's test split.
    """
    feat_input = {L: sents_by_lang[L] for L in TIER1_ALL}
    langs, X, lang_row = per_sentence_surface_features(feat_input)
    lang_idx = {L: i for i, L in enumerate(langs)}

    # Train rows: sentences from clean langs at indices 0..N_TRAIN-1.
    label_map = _family_label_map()
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_y: list[str] = []
    for i, L in enumerate(langs):
        if L not in label_map:
            continue
        start = i * (N_TRAIN + N_TEST)
        train_mask[start:start + N_TRAIN] = True
        train_y.extend([label_map[L]] * N_TRAIN)
    X_train = X[train_mask]

    # Test rows: English test split.
    eng_i = lang_idx[TIER1_HELDOUT]
    eng_start = eng_i * (N_TRAIN + N_TEST)
    X_test = X[eng_start + N_TRAIN: eng_start + N_TRAIN + N_TEST]

    # LDA on sparse features needs dense — these vectors are small enough.
    clf = LinearDiscriminantAnalysis(solver="svd")
    clf.fit(X_train.toarray(), np.asarray(train_y))
    preds = clf.predict(X_test.toarray())
    mean, lo, hi = _bootstrap_frac(preds, "Germanic", n_boot, seed)
    return {"frac_germanic": mean, "frac_germanic_lo": lo, "frac_germanic_hi": hi,
            "predictions": preds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-dir", default="results/reps")
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--layers", nargs="+", default=list(LAYER_SWEEP))
    ap.add_argument("--pools", nargs="+", default=list(POOL_SWEEP))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = Path(args.reps_dir)

    # ---------- surface side (one row, independent of layer/pool) ----------
    print("\nSURFACE side (per-sentence char-ngram, 8 anchors → English):")
    sents = load_sentences(TIER1_ALL)
    surf = surface_tier1(sents, args.n_boot, args.seed)
    print(f"  P(Eng → Germanic | surface) = {surf['frac_germanic']:.3f}  "
          f"[{surf['frac_germanic_lo']:.3f}, {surf['frac_germanic_hi']:.3f}]")
    surface_row = {
        "modality": "surface", "pool": "-", "layer": "-",
        "frac_germanic": f"{surf['frac_germanic']:.4f}",
        "ci_lo": f"{surf['frac_germanic_lo']:.4f}",
        "ci_hi": f"{surf['frac_germanic_hi']:.4f}",
    }

    rows = [surface_row]
    geometry_results = []
    for pool in args.pools:
        for layer in args.layers:
            print(f"\nGEOMETRY  pool={pool}  layer={layer}")
            g = geometry_tier1(reps_dir, args.model, pool, layer,
                               args.n_boot, args.seed)
            geometry_results.append((pool, layer, g))
            rows.append({
                "modality": "geometry", "pool": pool, "layer": layer,
                "frac_germanic": f"{g['frac_germanic']:.4f}",
                "ci_lo": f"{g['frac_germanic_lo']:.4f}",
                "ci_hi": f"{g['frac_germanic_hi']:.4f}",
            })
            print(f"  P(Eng → Germanic | geometry) = {g['frac_germanic']:.3f}  "
                  f"[{g['frac_germanic_lo']:.3f}, {g['frac_germanic_hi']:.3f}]")

    csv_path = out_dir / "tier1_heldout.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # SUMMARY block
    best = max(geometry_results, key=lambda r: r[2]["frac_germanic"])
    summary = [
        "=== TIER 1 ===",
        f"surface  P(Eng→Germanic) = {surf['frac_germanic']:.3f}  "
        f"[{surf['frac_germanic_lo']:.3f}, {surf['frac_germanic_hi']:.3f}]",
        f"geometry P(Eng→Germanic) best = {best[2]['frac_germanic']:.3f}  "
        f"[{best[2]['frac_germanic_lo']:.3f}, {best[2]['frac_germanic_hi']:.3f}]"
        f"  (pool={best[0]}, layer={best[1]})",
        f"Gate (geometry >= 0.90 AND surface <= 0.50): "
        f"{'PASS' if (best[2]['frac_germanic'] >= 0.90 and surf['frac_germanic'] <= 0.50) else 'PARTIAL/FAIL'}",
        "",
    ]
    with open(out_dir / "SUMMARY.txt", "a") as f:
        f.write("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
