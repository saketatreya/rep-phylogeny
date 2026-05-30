"""Extract per-layer pooled hidden states for an arbitrary language set.

Two pool strategies (both computed in one pass to amortize the forward):

- ``mean_pool``: mean over all non-special content tokens.
- ``high_freq``: mean restricted to the per-language top-K most frequent
  subword ids (HIGH_FREQ_K in config). Targets the borrowing-resistant
  grammatical core (function words / morphology), per Rabinovich et al. 2017.

Storage layout::

    {out_dir}/{model_key}/{pool}/layer_KK/{lang_name}.npy
        shape (1012, d_model), dtype float32

`layer_KK` = output of transformer block K (0-indexed). `hidden_states[0]` is
the embedding output, so `layer_KK` = `hidden_states[K+1]`.
"""
from __future__ import annotations
import gc
from collections import Counter
from pathlib import Path

import numpy as np

from .config import HIGH_FREQ_K, MODELS, all_layer_labels


# torch is only needed for extraction; load_reps below uses only numpy. Lazy
# imports so the analysis scripts can use load_reps without a torch install.


def _compute_high_freq_ids(
    tokenizer, sentences: list[str], k: int, special_ids: set[int],
) -> set[int]:
    """Top-K most frequent non-special subword ids across all sentences."""
    counts: Counter[int] = Counter()
    for s in sentences:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        counts.update(ids)
    for sid in special_ids:
        counts.pop(sid, None)
    return {tid for tid, _ in counts.most_common(k)}


def extract_for_model(
    model_key: str,
    sentences_per_lang: dict[str, list[str]],
    out_dir: Path,
    pool_strategies: tuple[str, ...] = ("mean_pool", "high_freq"),
    batch_size: int | None = None,
    max_length: int = 256,
    device: str | None = None,
    sanity_log: list[str] | None = None,
) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    _DTYPE = {"float16": torch.float16, "float32": torch.float32,
              "bfloat16": torch.bfloat16}

    cfg = MODELS[model_key]
    if batch_size is None:
        batch_size = cfg.get("batch_size", 16)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "float16" if device == "cuda" else "float32")
    if device == "cpu":
        cfg_dtype = "float32"
    dtype = _DTYPE[cfg_dtype]

    print(f"\n[{model_key}] loading from {cfg['hf_id']} "
          f"(dtype={cfg_dtype}, pool={list(pool_strategies)})")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        cfg["hf_id"], torch_dtype=dtype, output_hidden_states=True,
    )
    model.to(device).eval()

    n_layers = model.config.num_hidden_layers
    labels = all_layer_labels(n_layers)
    d = model.config.hidden_size
    print(f"  layers: {len(labels)} ({labels[0]}..{labels[-1]})  d_model={d}")

    special_ids = set(tokenizer.all_special_ids)

    out_dir = Path(out_dir)
    for pool in pool_strategies:
        for lab in labels:
            (out_dir / model_key / pool / lab).mkdir(parents=True, exist_ok=True)

    for lang_name, sents in sentences_per_lang.items():
        n = len(sents)

        # Per-language top-K ids for the high_freq pool. Cheap (one tokenizer pass).
        hf_ids: set[int] = set()
        if "high_freq" in pool_strategies:
            hf_ids = _compute_high_freq_ids(tokenizer, sents, HIGH_FREQ_K, special_ids)
            print(f"  [{lang_name}] high_freq vocab = {len(hf_ids)} token ids")

        pooled = {
            p: {lab: np.zeros((n, d), dtype=np.float32) for lab in labels}
            for p in pool_strategies
        }

        pbar = tqdm(range(0, n, batch_size), desc=f"  {lang_name:<5}")
        for start in pbar:
            end = min(start + batch_size, n)
            batch = sents[start:end]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model(**enc)
            hidden_states = out.hidden_states  # tuple of length n_layers+1

            ids = enc["input_ids"]                  # (B, T)
            attn = enc["attention_mask"].clone()    # (B, T)
            # Drop ALL special tokens (BOS, EOS, pad, mask, etc.) from content mask.
            for sid in special_ids:
                attn = attn * (ids != sid).to(attn.dtype)

            content_mask = attn.unsqueeze(-1).float()      # (B, T, 1)
            content_count = content_mask.sum(dim=1).clamp_min(1.0)

            if "high_freq" in pool_strategies and hf_ids:
                hf_id_tensor = torch.tensor(list(hf_ids), device=device)
                # (B, T): True at positions whose token id is in the per-lang top-K
                hf_pos = torch.isin(ids, hf_id_tensor) & (attn > 0)
                hf_mask = hf_pos.unsqueeze(-1).float()
                hf_count = hf_mask.sum(dim=1).clamp_min(1.0)
                # Sentences with zero matches fall back to plain mean (rare with K=80
                # — function words land in nearly every sentence).
                hf_fallback = (hf_pos.sum(dim=1) == 0)  # (B,)

            for k, lab in enumerate(labels):
                hs = hidden_states[k + 1].float()  # (B, T, d) in fp32 for stable pool

                if "mean_pool" in pool_strategies:
                    mp = (hs * content_mask).sum(dim=1) / content_count
                    pooled["mean_pool"][lab][start:end] = mp.cpu().numpy()

                if "high_freq" in pool_strategies and hf_ids:
                    hp = (hs * hf_mask).sum(dim=1) / hf_count
                    if hf_fallback.any():
                        mp_fb = (hs * content_mask).sum(dim=1) / content_count
                        hp = torch.where(hf_fallback.unsqueeze(-1), mp_fb, hp)
                    pooled["high_freq"][lab][start:end] = hp.cpu().numpy()

            del out, hidden_states

        for p in pool_strategies:
            for lab, arr in pooled[p].items():
                np.save(out_dir / model_key / p / lab / f"{lang_name}.npy", arr)
        print(f"  saved {lang_name}: {len(labels)} layers × {len(pool_strategies)} pool(s)")

    _emit_sanity(model_key, out_dir, list(pool_strategies), labels,
                 list(sentences_per_lang.keys()), sanity_log)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _emit_sanity(model_key, out_dir, pools, labels, lang_names, sanity_log) -> None:
    def emit(msg):
        print(msg)
        if sanity_log is not None:
            sanity_log.append(msg)

    emit("")
    emit(f"=== sanity checks: {model_key} ===")
    for pool in pools:
        for lab in labels:
            norms = []
            for name in lang_names:
                X = np.load(out_dir / model_key / pool / lab / f"{name}.npy")
                norms.append(np.linalg.norm(X, axis=1).mean())
            emit(f"  {pool}:{lab}: mean per-lang norms "
                 f"min={min(norms):.2f}  max={max(norms):.2f}  ratio={max(norms)/max(min(norms),1e-9):.2f}")


def load_reps(
    out_dir: Path, model_key: str, pool: str, layer_label: str,
    lang_names: list[str],
) -> dict[str, np.ndarray]:
    base = Path(out_dir) / model_key / pool / layer_label
    return {name: np.load(base / f"{name}.npy") for name in lang_names}
