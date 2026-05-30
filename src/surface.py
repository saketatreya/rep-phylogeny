"""Character n-gram surface distance — the baseline that *should* be fooled
by English / Maltese / Romanian / Urdu.

We give it the strongest fair shot: TF-IDF over char_wb (3,5) on the raw
devtest text, cosine similarity between per-language profiles, and the option
to romanize cross-script pairs (transliteration control, Tier 2d).
"""
from __future__ import annotations
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def language_surface_distance(
    raw_text_by_lang: dict[str, list[str]],
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> tuple[list[str], np.ndarray]:
    """One TF-IDF profile per language, distance = 1 - cosine.

    Returns (lang_names_in_order, dist_matrix). The matrix is symmetric with
    zeros on the diagonal. Cross-script pairs will saturate to ~1.0 (no shared
    char n-grams) — that's expected and handled by the transliteration
    control in :func:`romanize_for_surface`.
    """
    langs = list(raw_text_by_lang.keys())
    docs = [" ".join(raw_text_by_lang[L]) for L in langs]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return langs, dist


def per_sentence_surface_features(
    raw_text_by_lang: dict[str, list[str]],
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
):
    """Per-sentence TF-IDF features over the union vocabulary.

    Used by the held-out family discriminant (Tier 1) so the surface
    classifier sees per-sentence vectors, just like the geometry one.
    Returns (lang_names, sparse_matrix, row_lang_idx) where row_lang_idx[i]
    is the language index of row i.
    """
    langs = list(raw_text_by_lang.keys())
    rows: list[str] = []
    row_lang: list[int] = []
    for i, L in enumerate(langs):
        for s in raw_text_by_lang[L]:
            rows.append(s)
            row_lang.append(i)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(rows)
    return langs, X, np.asarray(row_lang, dtype=np.int64)


def romanize_for_surface(text: str) -> str:
    """Best-effort romanization for the surface baseline only.

    Uses :mod:`uroman` if installed (preferred — broad coverage); falls back
    to Unicode-normalize → ASCII fold which works for Latin-script langs but
    is a no-op for Cyrillic/Arabic/Devanagari and so will be visibly worse.

    This is the control referenced in Tier 2d: recompute surface distance on
    romanized text so Hindi/Urdu and Persian/Arabic are not artificially far
    apart purely by script.
    """
    try:
        import uroman  # type: ignore
        ur = uroman.Uroman()
        return ur.romanize_string(text)
    except ImportError:
        import unicodedata
        nfkd = unicodedata.normalize("NFKD", text)
        return nfkd.encode("ascii", "ignore").decode("ascii")
