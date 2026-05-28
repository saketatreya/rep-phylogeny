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
LANG_CODES = [code for _, code in LANGUAGES]

N_TRAIN = 812
N_TEST = 200
N_TOTAL = N_TRAIN + N_TEST  # 1012

MODELS = {
    "xlm-roberta-large": {
        "hf_id": "xlm-roberta-large",
        "n_layers": 24,
        "d_model": 1024,
        "is_decoder": False,
    },
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b",
        "n_layers": 26,
        "d_model": 2304,
        "is_decoder": True,
    },
}


def layer_indices(n_layers: int) -> dict[str, int]:
    """The four layer probes: quarter, half, three-quarter, final."""
    return {
        "L_quarter":       n_layers // 4,
        "L_half":          n_layers // 2,
        "L_three_quarter": 3 * n_layers // 4,
        "L_final":         n_layers - 1,
    }
