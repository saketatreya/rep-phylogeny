"""Language set, model config, and sweep grid for the genealogy-vs-surface
experiment.

Single source of truth — every driver imports from here so we never get a
mismatch between extraction time and analysis time.

Genealogy labels come from Glottolog top-level families. ``is_conflict`` flags
the languages where surface similarity and genealogy disagree — these are the
test cases that the experiment is designed to rule on. Clean anchors are the
languages used to *build* the discriminating axes (Tier 1, Tier 2b); conflict
cases are never shown to those fits.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Lang:
    name: str          # short slug, used in filenames (e.g. "eng")
    flores: str        # FLORES-200 code, e.g. "eng_Latn"
    family: str        # genealogy bucket (used as classifier label)
    script: str        # writing system, parsed from FLORES suffix
    is_conflict: bool  # True iff surface and genealogy disagree for this lang


# ---------- the language set ----------
#
# Clean anchors: descent and surface agree. Used to *fit* every discriminant /
# axis. Conflict cases: held out, then tested.
#
# Romanian is genealogically Romance but is listed as a conflict case in
# new_spec.md because of heavy Slavic admixture pulling its surface form
# sideways. So it carries the family label "Romance" but is_conflict=True
# (never used in anchor sets).

LANGS: list[Lang] = [
    # Germanic anchors
    Lang("deu", "deu_Latn", "Germanic", "Latn", False),
    Lang("nld", "nld_Latn", "Germanic", "Latn", False),
    Lang("swe", "swe_Latn", "Germanic", "Latn", False),
    Lang("dan", "dan_Latn", "Germanic", "Latn", False),
    # Romance anchors
    Lang("fra", "fra_Latn", "Romance",  "Latn", False),
    Lang("spa", "spa_Latn", "Romance",  "Latn", False),
    Lang("ita", "ita_Latn", "Romance",  "Latn", False),
    Lang("por", "por_Latn", "Romance",  "Latn", False),
    # Slavic anchors
    Lang("rus", "rus_Cyrl", "Slavic",   "Cyrl", False),
    Lang("pol", "pol_Latn", "Slavic",   "Latn", False),
    Lang("ces", "ces_Latn", "Slavic",   "Latn", False),
    Lang("bul", "bul_Cyrl", "Slavic",   "Cyrl", False),
    # Semitic anchors
    Lang("arb", "arb_Arab", "Semitic",  "Arab", False),
    Lang("heb", "heb_Hebr", "Semitic",  "Hebr", False),
    # Indo-Aryan anchors
    Lang("hin", "hin_Deva", "IndoAryan", "Deva", False),
    Lang("ben", "ben_Beng", "IndoAryan", "Beng", False),
    Lang("npi", "npi_Deva", "IndoAryan", "Deva", False),
    # Iranian anchor (single, since the spec only needs one for Urdu's contrast)
    Lang("pes", "pes_Arab", "Iranian",  "Arab", False),
    # Uralic control (non-IE — must sit apart from everything IE)
    Lang("fin", "fin_Latn", "Uralic",   "Latn", False),
    Lang("hun", "hun_Latn", "Uralic",   "Latn", False),
    Lang("est", "est_Latn", "Uralic",   "Latn", False),
    # Conflict cases
    Lang("eng", "eng_Latn", "Germanic", "Latn", True),   # surface → Romance, genealogy → Germanic
    Lang("mlt", "mlt_Latn", "Semitic",  "Latn", True),   # surface → Romance, genealogy → Semitic
    Lang("ron", "ron_Latn", "Romance",  "Latn", True),   # surface → mixed/Slavic, genealogy → Romance
    Lang("urd", "urd_Arab", "IndoAryan", "Arab", True),  # surface → Iranian, genealogy → IndoAryan
]


# Convenience indexes.
BY_NAME: dict[str, Lang] = {L.name: L for L in LANGS}
LANG_NAMES: list[str] = [L.name for L in LANGS]
CLEAN_ANCHORS: list[str] = [L.name for L in LANGS if not L.is_conflict]
CONFLICT_CASES: list[str] = [L.name for L in LANGS if L.is_conflict]


def anchors_in_family(family: str) -> list[str]:
    """All clean anchors with the given family label."""
    return [L.name for L in LANGS if L.family == family and not L.is_conflict]


def conflict_truth(name: str) -> str:
    """Genealogical family that a conflict case *should* end up in."""
    L = BY_NAME[name]
    assert L.is_conflict, f"{name} is not a conflict case"
    return L.family


# ---------- splits ----------

N_TOTAL = 1012          # FLORES-200 devtest size
N_TRAIN = 812           # used for fitting Procrustes, ridge, hub alignment, discriminants
N_TEST = N_TOTAL - N_TRAIN  # 200 sentences for evaluation + bootstrap


# ---------- model ----------
#
# Gemma-2 is excluded per new_spec.md (Romanian-intermediate bias contaminates
# exactly the conflict cases). XLM-R-large is the clean instrument.

MODELS = {
    "xlm-roberta-large": {
        "hf_id": "xlm-roberta-large",
        "n_layers": 24,
        "d_model": 1024,
        "is_decoder": False,
        "dtype": "float16",    # stable on T4
        "batch_size": 16,
    },
}


# ---------- sweep grid ----------

LAYER_SWEEP = ["layer_04", "layer_08", "layer_12", "layer_18"]
POOL_SWEEP = ["mean_pool", "high_freq"]
GEOM_VARIANTS = ["procrustes_resid", "hub_centroid"]

# Top-K subword ids per language for the high-frequency-token pool.
# Spec range: 50-100. Pick 80 as the middle.
HIGH_FREQ_K = 80


def layer_label(k: int) -> str:
    return f"layer_{k:02d}"


def all_layer_labels(n_layers: int) -> list[str]:
    return [layer_label(k) for k in range(n_layers)]
