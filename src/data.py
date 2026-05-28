"""Load FLORES-200 parallel sentences for the 5 Romance languages.

Tries, in order:
  1. Direct HTTP download of the canonical Meta FLORES-200 tarball (most
     robust — bypasses HuggingFace entirely, no script/version issues).
  2. HuggingFace datasets loaders (facebook/flores, openlanguagedata,
     Muennighoff) — these work only with datasets<4.0 because the older
     ones are script-based.
"""
from __future__ import annotations
import io
import os
import tarfile
import urllib.request
from pathlib import Path

from .config import LANG_CODES, N_TOTAL

FLORES_TAR_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def load_flores_sentences(cache_dir: str | Path = "data_cache") -> list[list[str]]:
    """Return parallel sentences as a list-of-lists indexed [lang][sentence].

    Returns 5 lists of 1012 sentences each in the order
    [Spanish, Portuguese, French, Italian, Romanian].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def direct():
        return _try_direct_download(cache_dir)
    direct.__name__ = "_try_direct_download"
    loaders = [
        direct,
        _try_facebook_flores,
        _try_openlanguagedata_flores_plus,
        _try_muennighoff_flores200,
    ]
    last_err = None
    for fn in loaders:
        name = fn.__name__
        try:
            sents = fn()
            assert len(sents) == 5, f"got {len(sents)} langs"
            for i, s in enumerate(sents):
                assert len(s) == N_TOTAL, f"lang {i}: expected {N_TOTAL}, got {len(s)}"
            print(f"  [data] loaded via {name}")
            return sents
        except Exception as e:
            last_err = e
            print(f"  [data] {name} failed: {type(e).__name__}: {e}")
            continue
    raise RuntimeError(f"Could not load FLORES from any source. Last error: {last_err}")


# ---------- preferred: direct download ----------

def _try_direct_download(cache_dir: Path) -> list[list[str]]:
    """Download Meta's FLORES-200 tarball directly and extract the 5 files."""
    cached = {code: cache_dir / f"{code}.devtest" for code in LANG_CODES}
    if not all(p.exists() for p in cached.values()):
        tar_path = cache_dir / "flores200_dataset.tar.gz"
        if not tar_path.exists():
            print(f"  [data] downloading FLORES-200 ({FLORES_TAR_URL}) ...")
            req = urllib.request.Request(
                FLORES_TAR_URL,
                headers={"User-Agent": "rep-phylogeny/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as r, open(tar_path, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"  [data] downloaded {tar_path.stat().st_size / 1e6:.1f} MB")

        print(f"  [data] extracting 5 devtest files ...")
        with tarfile.open(tar_path, mode="r:gz") as tar:
            for code in LANG_CODES:
                target = f"devtest/{code}.devtest"
                member = next(
                    (m for m in tar.getmembers() if m.name.endswith(target)),
                    None,
                )
                if member is None:
                    raise FileNotFoundError(f"missing {target} in tarball")
                f = tar.extractfile(member)
                cached[code].write_bytes(f.read())

    out = []
    for code in LANG_CODES:
        text = cached[code].read_text(encoding="utf-8")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        out.append(lines)
    return out


# ---------- HuggingFace fallbacks (require datasets<4.0 for the script ones) ----------

def _try_facebook_flores() -> list[list[str]]:
    from datasets import load_dataset
    ds = load_dataset("facebook/flores", "all", split="devtest")
    return [list(ds[f"sentence_{code}"]) for code in LANG_CODES]


def _try_openlanguagedata_flores_plus() -> list[list[str]]:
    """Non-script HF dataset. Gated — user must request access on HF."""
    from datasets import load_dataset
    out = []
    for code in LANG_CODES:
        ds = load_dataset("openlanguagedata/flores_plus", code, split="devtest")
        for col in ("text", "sentence"):
            if col in ds.column_names:
                out.append(list(ds[col]))
                break
        else:
            raise KeyError(f"no text column in {ds.column_names}")
    return out


def _try_muennighoff_flores200() -> list[list[str]]:
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
