"""Load FLORES-200 parallel sentences for the 5 Romance languages."""
from __future__ import annotations
from .config import LANG_CODES, N_TOTAL


def load_flores_sentences() -> list[list[str]]:
    """Return parallel sentences as list-of-lists indexed [lang][sentence].

    Returns a list of 5 lists, each of length 1012, in the order
    [Spanish, Portuguese, French, Italian, Romanian].
    """
    from datasets import load_dataset

    # Try a sequence of dataset locations, since FLORES has shifted on HF
    # over time. Each loader returns parallel sentence lists in the order
    # of LANG_CODES.
    loaders = [
        _try_facebook_flores,
        _try_openlanguagedata_flores_plus,
        _try_muennighoff_flores200,
    ]
    last_err = None
    for fn in loaders:
        try:
            sents = fn()
            assert len(sents) == 5
            for s in sents:
                assert len(s) == N_TOTAL, f"expected {N_TOTAL}, got {len(s)}"
            return sents
        except Exception as e:
            last_err = e
            print(f"  [data] {fn.__name__} failed: {type(e).__name__}: {e}")
            continue
    raise RuntimeError(f"Could not load FLORES from any source. Last error: {last_err}")


def _try_facebook_flores() -> list[list[str]]:
    """facebook/flores with the all_languages config (devtest split)."""
    from datasets import load_dataset
    ds = load_dataset("facebook/flores", "all", split="devtest")
    out = []
    for code in LANG_CODES:
        col = f"sentence_{code}"
        out.append(list(ds[col]))
    return out


def _try_openlanguagedata_flores_plus() -> list[list[str]]:
    """openlanguagedata/flores_plus — per-language configs."""
    from datasets import load_dataset
    out = []
    for code in LANG_CODES:
        ds = load_dataset("openlanguagedata/flores_plus", code, split="devtest")
        # the text column varies — try common names
        for col in ("text", "sentence"):
            if col in ds.column_names:
                out.append(list(ds[col]))
                break
        else:
            raise KeyError(f"no text column in {ds.column_names}")
    return out


def _try_muennighoff_flores200() -> list[list[str]]:
    """Muennighoff/flores200 — per-language configs, devtest split."""
    from datasets import load_dataset
    out = []
    for code in LANG_CODES:
        ds = load_dataset("Muennighoff/flores200", code, split="devtest")
        for col in ("sentence", "text"):
            if col in ds.column_names:
                out.append(list(ds[col]))
                break
        else:
            raise KeyError(f"no text column in {ds.column_names}")
    return out
