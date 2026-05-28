"""Extract pooled hidden states at every transformer layer for each language.

Two pooling strategies are supported:
- ``mean_pool``: mean over content tokens (excluding BOS/CLS and EOS/SEP/pad).
- ``last_token``: hidden state at the last non-special content position.

For ``xlm-roberta-large`` (bidirectional encoder), only ``mean_pool`` is run.
For ``google/gemma-2-2b`` (decoder LM), both strategies are extracted.

Storage layout::

    {out_dir}/{model_key}/{pool_strategy}/layer_KK/{lang_name}.npy
        shape (n_sentences, d_model), dtype float32

Note: hidden_states[k+1] is the output of transformer block k, where k is
0-indexed. hidden_states[0] is the embedding output. So `layer_00` =
hidden_states[1], ..., `layer_NN` = hidden_states[n_layers].
"""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from .config import LANG_NAMES, MODELS, all_layer_labels


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@torch.no_grad()
def extract_for_model(
    model_key: str,
    sentences_per_lang: list[list[str]],
    out_dir: Path,
    batch_size: int | None = None,
    max_length: int = 256,
    device: str | None = None,
    pool_strategies: list[str] | None = None,
    sanity_log: list[str] | None = None,
) -> None:
    """Run a model on all 5 languages × all sentences, save per-layer pooled
    vectors for every transformer layer and every requested pooling strategy.
    """
    from transformers import AutoModel, AutoTokenizer

    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    is_decoder = cfg["is_decoder"]
    n_layers_expected = cfg["n_layers"]
    if batch_size is None:
        batch_size = cfg.get("batch_size", 16)
    if pool_strategies is None:
        pool_strategies = list(cfg["pool_strategies"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_dtype = cfg.get("dtype", "float16" if device == "cuda" else "float32")
    if device == "cpu":
        cfg_dtype = "float32"  # fp16 on CPU is pointless and slow
    dtype = _DTYPE_MAP[cfg_dtype]

    print(f"\n[{model_key}] loading from {hf_id} (dtype={cfg_dtype}, pool={pool_strategies})")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=dtype, output_hidden_states=True)
    if is_decoder:
        # Gemma-2 needs eager attention to expose hidden states reliably on
        # Kaggle's transformers version.
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModel.from_pretrained(hf_id, **model_kwargs)
    model.to(device)
    model.eval()

    n_layers_actual = model.config.num_hidden_layers
    if n_layers_actual != n_layers_expected:
        print(f"  [warn] expected {n_layers_expected} layers, got {n_layers_actual} — using actual.")
    labels = all_layer_labels(n_layers_actual)
    print(f"  layers: {len(labels)} ({labels[0]} .. {labels[-1]})")

    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    out_dir = Path(out_dir)
    model_dir = out_dir / model_key
    for pool in pool_strategies:
        for label in labels:
            (model_dir / pool / label).mkdir(parents=True, exist_ok=True)

    d = model.config.hidden_size

    for lang_idx, sents in enumerate(sentences_per_lang):
        lang_name = LANG_NAMES[lang_idx]
        n = len(sents)
        # Per-pool, per-layer output buffers
        pooled: dict[str, dict[str, np.ndarray]] = {
            pool: {label: np.zeros((n, d), dtype=np.float32) for label in labels}
            for pool in pool_strategies
        }

        pbar = tqdm(range(0, n, batch_size), desc=f"  {lang_name:<10}")
        for start in pbar:
            end = min(start + batch_size, n)
            batch = sents[start:end]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            out = model(**enc)
            hidden_states = out.hidden_states  # tuple length n_layers+1

            attn = enc["attention_mask"].clone()  # (B, T)
            ids = enc["input_ids"]
            attn[:, 0] = 0  # drop BOS / CLS
            if eos_id is not None:
                eos_mask = (ids == eos_id)
                attn = attn * (~eos_mask).to(attn.dtype)
            content_mask = attn.unsqueeze(-1).float()  # (B, T, 1)
            content_count = content_mask.sum(dim=1).clamp_min(1.0)  # (B, 1)

            # Precompute last-content-token index per row (for last_token pool).
            if "last_token" in pool_strategies:
                row_sums = attn.sum(dim=1)  # (B,)
                # For each row, find the last position where attn==1.
                # If all zero (shouldn't happen), fall back to position 1.
                last_idx = torch.zeros(attn.shape[0], dtype=torch.long, device=device)
                for i in range(attn.shape[0]):
                    nz = torch.nonzero(attn[i], as_tuple=False).flatten()
                    last_idx[i] = nz[-1].item() if nz.numel() > 0 else 1

            for k, label in enumerate(labels):
                hs = hidden_states[k + 1].float()  # (B, T, d) in fp32 for pooling
                if "mean_pool" in pool_strategies:
                    pooled_batch = (hs * content_mask).sum(dim=1) / content_count  # (B, d)
                    pooled["mean_pool"][label][start:end] = pooled_batch.cpu().numpy()
                if "last_token" in pool_strategies:
                    rows = torch.arange(hs.shape[0], device=device)
                    last_batch = hs[rows, last_idx]  # (B, d)
                    pooled["last_token"][label][start:end] = last_batch.cpu().numpy()

            del out, hidden_states

        for pool in pool_strategies:
            for label, arr in pooled[pool].items():
                path = model_dir / pool / label / f"{lang_name}.npy"
                np.save(path, arr)
        print(f"  saved {lang_name}: {len(labels)} layers × {len(pool_strategies)} pool(s)")

    # Inline sanity checks (norms + Spa-Por/Fre/Ron cosine per layer).
    _emit_sanity_checks(model_key, model_dir, pool_strategies, labels, sanity_log)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _emit_sanity_checks(
    model_key: str,
    model_dir: Path,
    pool_strategies: list[str],
    labels: list[str],
    sanity_log: list[str] | None,
) -> None:
    def emit(msg: str) -> None:
        print(msg)
        if sanity_log is not None:
            sanity_log.append(msg)

    emit("")
    emit(f"=== sanity checks: {model_key} ===")
    for pool in pool_strategies:
        for label in labels:
            reps = []
            for name in LANG_NAMES:
                reps.append(np.load(model_dir / pool / label / f"{name}.npy"))

            # Norm stats per language
            for L, name in enumerate(LANG_NAMES):
                ns = np.linalg.norm(reps[L], axis=1)
                emit(
                    f"  {model_key}:{pool}:{label}:{name[:3]}: "
                    f"mean_norm={ns.mean():.2f}  std={ns.std():.2f}  "
                    f"min={ns.min():.2f}  max={ns.max():.2f}"
                )
            # Cosine of Spa with Por / Fre / Ron
            def cos_mean(a, b):
                an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
                bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
                return float(np.mean(np.sum(an * bn, axis=1)))
            cos_sp = cos_mean(reps[0], reps[1])
            cos_sf = cos_mean(reps[0], reps[2])
            cos_sr = cos_mean(reps[0], reps[4])
            emit(
                f"  {model_key}:{pool}:{label}: "
                f"cos(Spa,Por)={cos_sp:.4f}  cos(Spa,Fre)={cos_sf:.4f}  "
                f"cos(Spa,Ron)={cos_sr:.4f}"
            )
            if cos_sp < 0.5 or cos_sf < 0.5:
                emit(f"  *** ANOMALY: very low cosine at {label}. Possible numerical issue.")


def load_representations(
    model_key: str,
    layer_label: str,
    out_dir: Path,
    pool: str = "mean_pool",
) -> list[np.ndarray]:
    """Return list of 5 arrays in language-index order, shape (n_total, d)."""
    out_dir = Path(out_dir)
    arrs = []
    base = out_dir / model_key
    # Back-compat: legacy layout had no pooling subdir.
    legacy = base / layer_label
    pool_dir = base / pool / layer_label
    src_dir = pool_dir if pool_dir.exists() else legacy
    for name in LANG_NAMES:
        arrs.append(np.load(src_dir / f"{name}.npy"))
    return arrs
