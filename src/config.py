"""Shared configuration constants."""
from __future__ import annotations

LANGUAGES = [
    ("Spanish",    "spa_Latn"),
    ("Portuguese", "por_Latn"),
    ("French",     "fra_Latn"),
    ("Italian",    "ita_Latn"),
    ("Romanian",   "ron_Latn"),
]

LANG_NAMES = [name for name, _ in LANGUAGES]
LANG_NAMES_SHORT = ["Spa", "Por", "Fre", "Ita", "Ron"]
LANG_CODES = [code for _, code in LANGUAGES]

N_TRAIN = 812
N_TEST = 200
N_TOTAL = N_TRAIN + N_TEST  # 1012

# Mean-pool is the default; last_token is only meaningful for decoder LMs.
POOLING_STRATEGIES = ["mean_pool", "last_token"]

MODELS = {
    "xlm-roberta-large": {
        "hf_id": "xlm-roberta-large",
        "n_layers": 24,
        "d_model": 1024,
        "is_decoder": False,
        # XLM-R is stable in fp16 on T4.
        "dtype": "float16",
        # Bidirectional encoder — last_token is not meaningful.
        "pool_strategies": ["mean_pool"],
        "batch_size": 16,
    },
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b",
        "n_layers": 26,
        "d_model": 2304,
        "is_decoder": True,
        # Gemma-2 is designed for bf16; T4 lacks bf16 and fp16 produced
        # anomalies last run (Spa-Fre cosine 0.318 at L_final, recon errs
        # 24->73 across depth). Force fp32 even though it's ~2x slower.
        "dtype": "float32",
        # Decoder LM: both pooling strategies make sense.
        "pool_strategies": ["mean_pool", "last_token"],
        "batch_size": 8,
    },
}


def layer_indices(n_layers: int) -> dict[str, int]:
    """The four layer probes used by the legacy `run.py` driver.

    Kept for backward-compat with the original viability run; the new
    diagnostic pipeline uses `all_layer_labels` instead and probes every
    transformer layer.
    """
    return {
        "L_quarter":       n_layers // 4,
        "L_half":          n_layers // 2,
        "L_three_quarter": 3 * n_layers // 4,
        "L_final":         n_layers - 1,
    }


def all_layer_labels(n_layers: int) -> list[str]:
    """`['layer_00', ..., 'layer_NN']` for every transformer layer.

    `layer_k` is the output of transformer block k (0-indexed), i.e.
    `hidden_states[k+1]` because index 0 is the embedding output.
    """
    return [f"layer_{k:02d}" for k in range(n_layers)]


# Ground-truth Romance topology, as a pair of splits.
# Spa,Por cherry; Ita,Ron cherry; French on the central edge.
GROUND_TRUTH_SPLITS = (
    (frozenset({0, 1}), frozenset({2, 3, 4})),  # {Spa,Por} | {Fre,Ita,Ron}
    (frozenset({3, 4}), frozenset({0, 1, 2})),  # {Ita,Ron} | {Spa,Por,Fre}
)
