"""Extract mean-pooled hidden states at 4 layer depths for each language."""
from __future__ import annotations
import os
import gc
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from .config import LANG_NAMES, MODELS, layer_indices


@torch.no_grad()
def extract_for_model(
    model_key: str,
    sentences_per_lang: list[list[str]],
    out_dir: Path,
    batch_size: int = 16,
    max_length: int = 256,
    device: str | None = None,
) -> None:
    """Run a model on all 5 languages × all sentences, save per-layer pooled
    vectors as .npy files.

    Layout:
        {out_dir}/{model_key}/{layer_label}/{lang_name}.npy
        shape (n_sentences, d_model), dtype float32
    """
    from transformers import AutoModel, AutoTokenizer

    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    is_decoder = cfg["is_decoder"]
    n_layers_expected = cfg["n_layers"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[{model_key}] loading from {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    # Some decoder tokenizers (e.g. Gemma) have no pad token defined.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs = dict(torch_dtype=dtype, output_hidden_states=True)
    if is_decoder:
        # Gemma-2 needs eager attention to expose all hidden states reliably
        # on Kaggle's transformers version; sdpa hidden_states are fine but
        # eager is the safest default.
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModel.from_pretrained(hf_id, **model_kwargs)
    model.to(device)
    model.eval()

    # Verify layer count
    n_layers_actual = model.config.num_hidden_layers
    if n_layers_actual != n_layers_expected:
        print(f"  [warn] expected {n_layers_expected} layers, got {n_layers_actual} — using actual.")
    layers = layer_indices(n_layers_actual)
    # +1 because hidden_states[0] is the embedding output (before any layer);
    # hidden_states[i+1] is the output of layer i.
    print(f"  layers to probe: {layers}")

    # Special-token handling per model family
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    out_dir = Path(out_dir)
    model_dir = out_dir / model_key
    for label in layers:
        (model_dir / label).mkdir(parents=True, exist_ok=True)

    for lang_idx, sents in enumerate(sentences_per_lang):
        lang_name = LANG_NAMES[lang_idx]
        n = len(sents)
        # Allocate output: one (n, d) array per layer
        pooled: dict[str, np.ndarray] = {
            label: np.zeros((n, model.config.hidden_size), dtype=np.float32)
            for label in layers
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
            hidden_states = out.hidden_states  # tuple of (B, T, d), length n_layers+1

            # Build a content mask: attention_mask minus BOS/CLS at start and
            # EOS/SEP at the last non-pad position.
            attn = enc["attention_mask"].clone()  # (B, T), 1 for real tokens
            ids = enc["input_ids"]
            # Always drop position 0 (BOS / CLS).
            attn[:, 0] = 0
            # Drop the last real token if it is EOS/SEP.
            if eos_id is not None:
                # last real position = sum(attn before clearing position 0) - 1
                # but easier: for each row, find indices where input == eos_id, set attn=0
                eos_mask = (ids == eos_id)
                attn = attn * (~eos_mask).to(attn.dtype)

            content_mask = attn.unsqueeze(-1).float()  # (B, T, 1)
            content_count = content_mask.sum(dim=1).clamp_min(1.0)  # (B, 1)

            for label, layer_idx in layers.items():
                hs = hidden_states[layer_idx + 1].float()  # (B, T, d) in fp32 for pooling
                pooled_batch = (hs * content_mask).sum(dim=1) / content_count  # (B, d)
                pooled[label][start:end] = pooled_batch.cpu().numpy()

            del out, hidden_states

        for label, arr in pooled.items():
            path = model_dir / label / f"{lang_name}.npy"
            np.save(path, arr)
        print(f"  saved {lang_name}: {pooled[next(iter(layers))].shape}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_representations(model_key: str, layer_label: str, out_dir: Path) -> list[np.ndarray]:
    """Return list of 5 arrays in language-index order, shape (n_total, d)."""
    out_dir = Path(out_dir)
    arrs = []
    for name in LANG_NAMES:
        p = out_dir / model_key / layer_label / f"{name}.npy"
        arrs.append(np.load(p))
    return arrs
