"""End-to-end driver for the phylogenetic viability experiment.

Usage:
    python run.py                          # both models, all 4 layers
    python run.py --models xlm-roberta-large
    python run.py --skip-extract           # reuse cached representations
    python run.py --out-dir /kaggle/working/results

Outputs a single text report per (model, layer) configuration to stdout
and to {out_dir}/report.txt.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.config import MODELS, layer_indices, N_TRAIN, LANG_NAMES
from src.data import load_flores_sentences
from src.representations import extract_for_model, load_representations
from src.procrustes import fit_all_pairs, pairwise_reconstruction_error
from src.composition import compute_composition_errors, comp_error_table_str
from src.report import full_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    p.add_argument("--out-dir", default="results",
                   help="directory for cached reps and reports")
    p.add_argument("--skip-extract", action="store_true",
                   help="reuse .npy files in out-dir if present")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--show-comp-errors", action="store_true",
                   help="also print the 10x3 composition-error table")
    return p.parse_args()


def representations_present(out_dir: Path, model_key: str) -> bool:
    cfg = MODELS[model_key]
    layers = layer_indices(cfg["n_layers"])
    for label in layers:
        for name in LANG_NAMES:
            if not (out_dir / model_key / label / f"{name}.npy").exists():
                return False
    return True


def run_one_config(
    model_key: str,
    layer_label: str,
    out_dir: Path,
    show_comp_errors: bool,
) -> str:
    X_all = load_representations(model_key, layer_label, out_dir)
    X_train = [x[:N_TRAIN] for x in X_all]
    X_test = [x[N_TRAIN:] for x in X_all]

    proc = fit_all_pairs(X_train)
    pair_errs = pairwise_reconstruction_error(X_test, proc)
    ce = compute_composition_errors(X_test, proc)

    title = f"{model_key}, layer={layer_label}"
    report = full_report(title, X_train, X_test, ce, pair_errs)

    if show_comp_errors:
        report += "\nCOMPOSITION ERROR TABLE:\n" + comp_error_table_str(ce, LANG_NAMES) + "\n"

    return report


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load sentences once and extract per model
    sentences = None
    for model_key in args.models:
        if args.skip_extract and representations_present(out_dir, model_key):
            print(f"[{model_key}] cached representations found, skipping extract.")
            continue
        if sentences is None:
            print("Loading FLORES-200 parallel sentences...")
            sentences = load_flores_sentences()
            print(f"  {len(sentences)} languages, {len(sentences[0])} sentences each.")
        t0 = time.time()
        extract_for_model(
            model_key,
            sentences,
            out_dir=out_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print(f"  [{model_key}] extraction took {time.time() - t0:.1f}s")

    # Step 2-6: analysis per (model, layer)
    report_lines: list[str] = []
    for model_key in args.models:
        cfg = MODELS[model_key]
        layers = layer_indices(cfg["n_layers"])
        for label in layers:
            section = run_one_config(model_key, label, out_dir, args.show_comp_errors)
            print(section)
            report_lines.append(section)

    # Write combined report
    full = "\n".join(report_lines)
    report_path = out_dir / "report.txt"
    report_path.write_text(full)
    print(f"\nFull report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
