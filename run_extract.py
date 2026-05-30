"""Extract per-layer XLM-R-large hidden states for the full language set.

This is the GPU-heavy stage that runs once on Kaggle. Saves to
``--out-dir/{model}/{pool}/{layer}/{lang}.npy``. Idempotent: rerunning skips
languages whose files already exist for every (pool, layer) combination.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from src.config import (
    LANG_NAMES, MODELS, POOL_SWEEP, all_layer_labels,
)
from src.data import load_sentences
from src.representations import extract_for_model


def _all_files_exist(out_dir: Path, model: str, pools: list[str],
                     labels: list[str], lang: str) -> bool:
    for pool in pools:
        for lab in labels:
            if not (out_dir / model / pool / lab / f"{lang}.npy").exists():
                return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/reps")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="restrict to these lang names (default: all 25 in config)")
    ap.add_argument("--pools", nargs="+", default=list(POOL_SWEEP),
                    choices=["mean_pool", "high_freq"])
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_names = args.langs or LANG_NAMES

    print(f"Loading FLORES devtest for {len(lang_names)} languages ...")
    sents = load_sentences(lang_names, cache_dir=args.cache_dir)
    for L, s in sents.items():
        print(f"  {L}: {len(s)} sentences")

    cfg = MODELS[args.model]
    labels = all_layer_labels(cfg["n_layers"])

    # Skip languages already fully extracted (idempotent rerun).
    pending = {L: s for L, s in sents.items()
               if not _all_files_exist(out_dir, args.model, args.pools, labels, L)}
    skipped = [L for L in sents if L not in pending]
    if skipped:
        print(f"\nSkipping (already extracted): {skipped}")
    if not pending:
        print("All languages already extracted. Done.")
        return 0

    sanity: list[str] = []
    extract_for_model(
        args.model, pending, out_dir,
        pool_strategies=tuple(args.pools),
        batch_size=args.batch_size,
        sanity_log=sanity,
    )

    (out_dir / args.model / "sanity.log").write_text("\n".join(sanity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
