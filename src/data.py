"""Load FLORES-200 devtest sentences for an arbitrary set of languages.

Downloads Meta's canonical NLLB tarball once into ``data_cache/`` and extracts
the ``devtest/{flores_code}.devtest`` files on demand. No HuggingFace
dependency — the prior pipeline learned the hard way that the HF loaders for
FLORES are fragile across ``datasets`` versions.
"""
from __future__ import annotations
import tarfile
import urllib.request
from pathlib import Path

from .config import LANGS, BY_NAME, N_TOTAL


FLORES_TAR_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def _ensure_tarball(cache_dir: Path) -> Path:
    tar_path = cache_dir / "flores200_dataset.tar.gz"
    if tar_path.exists():
        return tar_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [data] downloading FLORES-200 ({FLORES_TAR_URL}) ...")
    req = urllib.request.Request(FLORES_TAR_URL,
                                 headers={"User-Agent": "geneal-rep/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tar_path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"  [data] downloaded {tar_path.stat().st_size / 1e6:.1f} MB")
    return tar_path


def _extract_one(tar_path: Path, flores_code: str, dest: Path) -> None:
    target_suffix = f"devtest/{flores_code}.devtest"
    with tarfile.open(tar_path, mode="r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith(target_suffix)),
            None,
        )
        if member is None:
            raise FileNotFoundError(
                f"FLORES-200 tarball has no entry ending in {target_suffix!r} "
                f"(language code likely wrong)"
            )
        f = tar.extractfile(member)
        if f is None:
            raise RuntimeError(f"tar entry {member.name} is not a regular file")
        dest.write_bytes(f.read())


def load_sentences(
    lang_names: list[str] | None = None,
    cache_dir: str | Path = "data_cache",
) -> dict[str, list[str]]:
    """Return ``{lang_name: [sentence_0, ..., sentence_1011]}``.

    Sentence indices are parallel across languages (sentence ``i`` in language
    A is a translation of sentence ``i`` in language B), which is the property
    every downstream alignment / hub / classifier relies on.
    """
    if lang_names is None:
        lang_names = [L.name for L in LANGS]

    cache_dir = Path(cache_dir)
    needed = []
    for name in lang_names:
        L = BY_NAME[name]
        dest = cache_dir / f"{L.flores}.devtest"
        if not dest.exists():
            needed.append((L, dest))

    if needed:
        tar_path = _ensure_tarball(cache_dir)
        print(f"  [data] extracting {len(needed)} devtest file(s) ...")
        for L, dest in needed:
            _extract_one(tar_path, L.flores, dest)

    out: dict[str, list[str]] = {}
    for name in lang_names:
        L = BY_NAME[name]
        text = (cache_dir / f"{L.flores}.devtest").read_text(encoding="utf-8")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) != N_TOTAL:
            raise AssertionError(
                f"{L.flores}.devtest has {len(lines)} non-empty lines, "
                f"expected {N_TOTAL}"
            )
        out[name] = lines
    return out
