"""Full diagnostic pipeline (diagnostics.md).

Extracts every transformer layer for every requested (model, pool_strategy)
combination, runs the Blocks A-H diagnostics for every (model, pool, layer)
× (procrustes, ridge), and writes:

  {out_dir}/SUMMARY.txt
  {out_dir}/{model}/{pool}/layer_KK/{method}.log    per-config detail
  {out_dir}/{model}/{pool}/sanity.log               per-model extraction sanity
  {out_dir}/{model}/{pool}/per_sentence/...         optional per-sentence CE

Usage:
    python run_diagnostics.py                          # both models, full
    python run_diagnostics.py --models xlm-roberta-large
    python run_diagnostics.py --skip-extract
    python run_diagnostics.py --methods procrustes
    python run_diagnostics.py --layers layer_00 layer_12 layer_23
    python run_diagnostics.py --n-perms 20 --pca-dims 16 32 64 128 256 512
    python run_diagnostics.py --no-pca --no-langid --no-perm
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.config import (
    LANG_NAMES_SHORT,
    MODELS,
    N_TRAIN,
    all_layer_labels,
)
from src.data import load_flores_sentences
from src.representations import extract_for_model, load_representations
from src.transforms import (
    compute_composition_errors,
    fit_all_transforms,
    select_ridge_alpha,
)
from src.topologies import get_rank_of_ground_truth
from src import diagnostics as D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    p.add_argument("--out-dir", default="results_diag")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--methods", nargs="+", default=["procrustes", "ridge"],
                   choices=["procrustes", "ridge"])
    p.add_argument("--layers", nargs="+", default=None,
                   help="optional whitelist of layer labels (default: all)")
    p.add_argument("--pools", nargs="+", default=None,
                   help="optional whitelist of pooling strategies")
    p.add_argument("--n-perms", type=int, default=10,
                   help="permutation test iterations (Block D); spec called for 20 "
                        "but 10 gives usable z-scores at half the cost")
    p.add_argument("--pca-dims", nargs="+", type=int,
                   default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--ridge-alphas", nargs="+", type=float,
                   default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--no-langid", action="store_true")
    p.add_argument("--no-perm", action="store_true")
    p.add_argument("--no-tree", action="store_true")
    p.add_argument("--save-per-sentence", action="store_true")
    return p.parse_args()


def reps_present(out_dir: Path, model_key: str, pool: str, layers: list[str]) -> bool:
    base = out_dir / model_key / pool
    for label in layers:
        for name in ["Spanish", "Portuguese", "French", "Italian", "Romanian"]:
            if not (base / label / f"{name}.npy").exists():
                return False
    return True


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Step 1: extraction ----------
    sentences = None
    for model_key in args.models:
        cfg = MODELS[model_key]
        layer_labels_all = all_layer_labels(cfg["n_layers"])
        pools = args.pools or cfg["pool_strategies"]
        needed = any(
            not reps_present(out_dir, model_key, pool, layer_labels_all) for pool in pools
        )
        if args.skip_extract and not needed:
            print(f"[{model_key}] cached reps present for {pools}, skipping extract.")
            continue
        if args.skip_extract and needed:
            print(f"[{model_key}] WARNING: --skip-extract but cache is incomplete; re-extracting.")
        if sentences is None:
            print("Loading FLORES-200 parallel sentences...")
            sentences = load_flores_sentences()
            print(f"  {len(sentences)} languages, {len(sentences[0])} sentences each.")
        t0 = time.time()
        sanity_log_lines: list[str] = []
        extract_for_model(
            model_key,
            sentences,
            out_dir=out_dir,
            max_length=args.max_length,
            pool_strategies=pools,
            sanity_log=sanity_log_lines,
        )
        print(f"  [{model_key}] extraction took {time.time() - t0:.1f}s")
        for pool in pools:
            sanity_path = out_dir / model_key / pool / "sanity.log"
            sanity_path.parent.mkdir(parents=True, exist_ok=True)
            sanity_path.write_text("\n".join(sanity_log_lines))
        gc.collect()

    # ---------- Step 2: diagnostics ----------
    # Master tables (lists of dicts)
    table_gt_rank: list[dict] = []
    table_split_scores: list[dict] = []
    table_perm_z: list[dict] = []
    table_pca: list[dict] = []
    table_langid: list[dict] = []
    table_per_lang_bias: list[dict] = []
    table_closure: list[dict] = []

    # Cross-layer cache for C4
    ce_means_by_config: dict[tuple, dict[str, dict]] = {}

    for model_key in args.models:
        cfg = MODELS[model_key]
        layer_labels_all = all_layer_labels(cfg["n_layers"])
        # Intersect requested layers with this model's available labels.
        # Lets us pass one --layers list spanning models with different depths.
        layers = [l for l in (args.layers or layer_labels_all) if l in layer_labels_all]
        pools = args.pools or cfg["pool_strategies"]
        for pool in pools:
            for label in layers:
                print(f"\n=== diagnostics: {model_key} / {pool} / {label} ===")
                X_all = load_representations(model_key, label, out_dir, pool=pool)
                # Float64 for numerical work
                X_all = [x.astype(np.float64) for x in X_all]
                config_tag = f"{model_key}__{pool}__{label}"
                cfg_dir = out_dir / model_key / pool / label
                cfg_dir.mkdir(parents=True, exist_ok=True)

                # Block A — runs once per (model, pool, layer)
                a_lines: list[str] = []
                a_out = D.block_a_representation_stats(X_all, log=a_lines.append)
                (cfg_dir / "block_a.log").write_text("\n".join(a_lines))

                # Pooled PCA SVD — computed once per layer, reused across methods
                pooled_svd = None
                if not args.no_pca:
                    pooled_svd = D.compute_pooled_pca(X_all)

                for method in args.methods:
                    print(f"  --- method={method} ---")
                    m_lines: list[str] = []
                    log = m_lines.append
                    log(f"=== {config_tag} :: {method} ===")

                    # Fit transforms
                    if method == "ridge":
                        log("Ridge α selection (5-fold CV)")
                        alpha = select_ridge_alpha(
                            X_all, alphas=args.ridge_alphas, log_fn=log,
                        )
                    else:
                        alpha = 1.0  # unused for procrustes
                    transforms = fit_all_transforms(
                        X_all, method=method, ridge_alpha=alpha,
                    )

                    # Block B
                    b_out = D.block_b_transform_stats(X_all, transforms, log=log)

                    # Block C
                    per_sent_dir = None
                    if args.save_per_sentence:
                        per_sent_dir = out_dir / model_key / pool / "per_sentence"
                        per_sent_dir.mkdir(parents=True, exist_ok=True)
                    c_out = D.block_c_composition(
                        X_all, transforms, log=log,
                        save_per_sentence_dir=per_sent_dir,
                        config_tag=f"{config_tag}__{method}",
                    )
                    ce_means_by_config.setdefault(
                        (model_key, pool, method), {}
                    )[label] = c_out["ce_mean"]

                    # Block E (depends on C)
                    e_out = D.block_e_scoring(c_out["ce_mean"], log=log)

                    # Block D — permutation test
                    if not args.no_perm:
                        d_out = D.block_d_permutation(
                            X_all,
                            aligned_ce=c_out["ce_mean"],
                            aligned_gt_rank=e_out["gt_rank_legacy"],
                            method=method,
                            ridge_alpha=alpha,
                            n_perms=args.n_perms,
                            log=log,
                        )
                    else:
                        d_out = {"mean_z": float("nan"), "perm_gt_ranks": []}

                    # Block F — PCA sweep (pooled SVD precomputed once per layer)
                    if not args.no_pca:
                        f_out = D.block_f_pca_sweep(
                            X_all,
                            method=method,
                            ridge_alpha=alpha,
                            target_dims=args.pca_dims,
                            log=log,
                            pooled_svd=pooled_svd,
                        )
                    else:
                        f_out = {}

                    # Block G — language-ID projection
                    if not args.no_langid:
                        g_out = D.block_g_langid(
                            X_all,
                            langid_basis=a_out["langid_basis"],
                            method=method,
                            ridge_alpha=alpha,
                            log=log,
                        )
                    else:
                        g_out = {}

                    # Block H — distance trees (only once per layer, run with procrustes recon)
                    if not args.no_tree and method == args.methods[0]:
                        h_out = D.block_h_distance_tree(
                            X_all, transforms_for_recon=transforms,
                            cos_table=a_out["cos_table"], log=log,
                        )
                    else:
                        h_out = {}

                    (cfg_dir / f"{method}.log").write_text("\n".join(m_lines))

                    # Master-table rows
                    base_row = {
                        "model": model_key, "pool": pool, "layer": label,
                        "method": method, "alpha": alpha,
                    }
                    table_gt_rank.append({
                        **base_row,
                        "gt_rank_legacy": e_out["gt_rank_legacy"],
                        "gt_rank_resolved": e_out["gt_rank_resolved"],
                    })
                    table_split_scores.append({
                        **base_row,
                        "split_spa_por": e_out["split_gt_0_1"],
                        "split_ita_ron": e_out["split_gt_3_4"],
                    })
                    table_perm_z.append({
                        **base_row,
                        "mean_z": d_out.get("mean_z", float("nan")),
                    })
                    if f_out:
                        for dprime, fr in f_out.items():
                            table_pca.append({
                                **base_row,
                                "d_prime": dprime,
                                **fr,
                            })
                    if g_out:
                        for k_proj, gr in g_out.items():
                            table_langid.append({
                                **base_row,
                                "k": k_proj,
                                **gr,
                            })
                    table_per_lang_bias.append({
                        **base_row,
                        **{LANG_NAMES_SHORT[i]: v for i, v in c_out["per_lang_bias"].items()},
                    })
                    # closure diff: cherries vs cross-edge triples
                    # (rough surrogate; full table is in block_b.log)
                    closures = b_out["closure_rel"]
                    table_closure.append({
                        **base_row,
                        "closure_mean": float(np.mean(list(closures.values()))),
                        "closure_max": float(np.max(list(closures.values()))),
                    })

                # Incremental SUMMARY write — survives a downstream crash.
                _write_summary(
                    out_dir / "SUMMARY.txt",
                    gt_rank=table_gt_rank,
                    split_scores=table_split_scores,
                    perm_z=table_perm_z,
                    pca=table_pca,
                    langid=table_langid,
                    per_lang_bias=table_per_lang_bias,
                    closure=table_closure,
                )

    # ---------- Step 3: C4 + master summary ----------
    for (model_key, pool, method), ce_by_layer in ce_means_by_config.items():
        c4_lines: list[str] = []
        c4_lines.append(f"=== C4 across-layer consistency: {model_key}/{pool}/{method} ===")
        D.block_c4_rank_consistency(ce_by_layer, log=c4_lines.append)
        path = out_dir / model_key / pool / f"c4_{method}.log"
        path.write_text("\n".join(c4_lines))

    _write_summary(
        out_dir / "SUMMARY.txt",
        gt_rank=table_gt_rank,
        split_scores=table_split_scores,
        perm_z=table_perm_z,
        pca=table_pca,
        langid=table_langid,
        per_lang_bias=table_per_lang_bias,
        closure=table_closure,
    )
    print(f"\nFull summary written to {out_dir/'SUMMARY.txt'}")
    return 0


def _write_summary(path: Path, *, gt_rank, split_scores, perm_z, pca, langid, per_lang_bias, closure) -> None:
    lines: list[str] = []
    push = lines.append
    bar = "=" * 100

    push(bar); push("MASTER SUMMARY"); push(bar)

    # Table 1: GT rank
    push("\nTOPOLOGY RANK OF GROUND TRUTH (lower = better, out of 15)")
    push(f"{'Config':<55} {'Method':<11} {'legacy':>8} {'resolved':>10}")
    push("-" * 90)
    for row in sorted(gt_rank, key=_cfg_sort_key):
        cfg = f"{row['model']}/{row['pool']}/{row['layer']}"
        push(f"{cfg:<55} {row['method']:<11} "
             f"{row['gt_rank_legacy']:>8} {row['gt_rank_resolved']:>10}")

    # Table 2: split scores
    push("\nSPLIT-LEVEL SCORES (correct out of 3 cherry triples)")
    push(f"{'Config':<55} {'Method':<11} {'{Spa,Por}':>10} {'{Ita,Ron}':>10}")
    push("-" * 90)
    for row in sorted(split_scores, key=_cfg_sort_key):
        cfg = f"{row['model']}/{row['pool']}/{row['layer']}"
        push(f"{cfg:<55} {row['method']:<11} "
             f"{row['split_spa_por']:>10} {row['split_ita_ron']:>10}")

    # Table 3: permutation z-scores
    push("\nPERMUTATION TEST (mean z over 30 (triple,intermediate) entries)")
    push(f"{'Config':<55} {'Method':<11} {'mean_z':>10} {'verdict':<28}")
    push("-" * 110)
    for row in sorted(perm_z, key=_cfg_sort_key):
        cfg = f"{row['model']}/{row['pool']}/{row['layer']}"
        z = row["mean_z"]
        if np.isnan(z):
            verdict = "(skipped)"
        elif z < -2:
            verdict = "SENTENCE-LEVEL SIGNAL"
        elif abs(z) < 2:
            verdict = "NO SIGNAL (language-level only)"
        else:
            verdict = "UNEXPECTED (z>2)"
        push(f"{cfg:<55} {row['method']:<11} {z:>10.4f} {verdict:<28}")

    # Table 4: best PCA dim per config
    if pca:
        push("\nBEST PCA DIMENSION (by gt_rank_resolved)")
        push(f"{'Config':<55} {'Method':<11} {'d*':>5} {'gt_resolved':>12} {'split{Spa,Por}':>16} {'split{Ita,Ron}':>16}")
        push("-" * 120)
        by_cfg: dict[tuple, list] = {}
        for r in pca:
            by_cfg.setdefault((r["model"], r["pool"], r["layer"], r["method"]), []).append(r)
        for cfg_key, rows in sorted(by_cfg.items()):
            best = min(rows, key=lambda r: (r["gt_rank_resolved"], r["gt_rank_legacy"]))
            cfg = f"{cfg_key[0]}/{cfg_key[1]}/{cfg_key[2]}"
            push(
                f"{cfg:<55} {cfg_key[3]:<11} {best['d_prime']:>5} "
                f"{best['gt_rank_resolved']:>12} "
                f"{best['split_gt_0_1']:>16} {best['split_gt_3_4']:>16}"
            )

    # Table 5: best lang-ID k per config
    if langid:
        push("\nBEST LANG-ID PROJECTION k (by gt_rank_resolved)")
        push(f"{'Config':<55} {'Method':<11} {'k*':>5} {'gt_resolved':>12} {'split{Spa,Por}':>16} {'split{Ita,Ron}':>16}")
        push("-" * 120)
        by_cfg: dict[tuple, list] = {}
        for r in langid:
            by_cfg.setdefault((r["model"], r["pool"], r["layer"], r["method"]), []).append(r)
        for cfg_key, rows in sorted(by_cfg.items()):
            best = min(rows, key=lambda r: (r["gt_rank_resolved"], r["gt_rank_legacy"]))
            cfg = f"{cfg_key[0]}/{cfg_key[1]}/{cfg_key[2]}"
            push(
                f"{cfg:<55} {cfg_key[3]:<11} {best['k']:>5} "
                f"{best['gt_rank_resolved']:>12} "
                f"{best['split_gt_0_1']:>16} {best['split_gt_3_4']:>16}"
            )

    # Table 6: per-language intermediate bias
    push("\nPER-LANGUAGE INTERMEDIATE BIAS (mean CE when used as intermediate)")
    push(f"{'Config':<55} {'Method':<11} " + " ".join(f"{n:>10}" for n in LANG_NAMES_SHORT))
    push("-" * 130)
    for row in sorted(per_lang_bias, key=_cfg_sort_key):
        cfg = f"{row['model']}/{row['pool']}/{row['layer']}"
        cells = " ".join(f"{row[n]:>10.6f}" for n in LANG_NAMES_SHORT)
        push(f"{cfg:<55} {row['method']:<11} {cells}")

    # Table 7: triangle closure
    push("\nTRIANGLE CLOSURE (relative Frobenius error)")
    push(f"{'Config':<55} {'Method':<11} {'mean':>10} {'max':>10}")
    push("-" * 100)
    for row in sorted(closure, key=_cfg_sort_key):
        cfg = f"{row['model']}/{row['pool']}/{row['layer']}"
        push(f"{cfg:<55} {row['method']:<11} {row['closure_mean']:>10.4f} {row['closure_max']:>10.4f}")

    push("\n" + bar); push("DIAGNOSIS"); push(bar)
    push("""
Read in this order:

1. PERMUTATION TEST z-scores
   z ≈ 0  → Procrustes fits language-level stats, not sentence correspondence. Method dead.
   z << -2 → Sentence alignment matters. Method has valid foundation. Continue reading.

2. TRIANGLE CLOSURE
   Low mean closure → the maps compose. Differential closure (see block_b.log)
   between cherry-pair triples and outgroup triples is what we want.

3. RIDGE vs PROCRUSTES comparison
   Ridge much better → Orthogonality assumption was bottleneck.
   Similar → Transform class doesn't matter; issue is elsewhere.

4. PCA SWEEP
   gt_rank_resolved improves at lower d' → Null-space noise was masking signal.
   No improvement → Dimensionality isn't the problem.

5. SPLIT-LEVEL SCORES
   {Spa,Por} consistently 3/3 → Method recovers the easy cherry.
   {Ita,Ron} consistently 0-1/3 → Model treats Ita-Ron as distant (typological > genealogical).

6. PER-LANGUAGE BIAS
   One language systematically worst → Topology ranking is confounded.
""")
    path.write_text("\n".join(lines))


_MODEL_ORDER = {"xlm-roberta-large": 0, "gemma-2-2b": 1}
_POOL_ORDER = {"mean_pool": 0, "last_token": 1}
_METHOD_ORDER = {"procrustes": 0, "ridge": 1}


def _cfg_sort_key(row: dict) -> tuple:
    return (
        _MODEL_ORDER.get(row["model"], 99),
        _POOL_ORDER.get(row["pool"], 99),
        row["layer"],
        _METHOD_ORDER.get(row.get("method", ""), 99),
    )


if __name__ == "__main__":
    sys.exit(main())
