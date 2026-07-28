"""Result cache keyed on image content, not filename.

Two levels:

* exact  — SHA-256 of the file bytes. A re-sent WhatsApp file is byte-identical
           surprisingly often, and this makes it free.
* near   — 64-bit perceptual hash within a small Hamming radius. Catches the
           same page photographed twice, or the same image re-encoded at a
           different quality by a messaging app.

Near-duplicate reuse is the riskier of the two, so it is bounded: only results
that passed validation are ever served from the near cache, and the reused
record is stamped with the document it was borrowed from.
"""
from __future__ import annotations

import json
import pathlib

from .imaging import hamming

NEAR_DUP_MAX_DISTANCE = 3


class ResultCache:
    def __init__(self, path: str | pathlib.Path, enable_near_dup: bool = True):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_near_dup = enable_near_dup
        self._exact: dict[str, dict] = {}
        self._phash: list[tuple[str, str]] = []  # (phash, content_hash)
        self.hits_exact = 0
        self.hits_near = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._exact = blob.get("exact", {})
        self._phash = [tuple(x) for x in blob.get("phash", [])]

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"exact": self._exact, "phash": self._phash}, ensure_ascii=False),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self._exact.clear()
        self._phash.clear()
        self.hits_exact = self.hits_near = 0
        if self.path.exists():
            self.path.unlink()

    def get(self, content_hash: str, phash: str | None) -> tuple[dict | None, str]:
        hit = self._exact.get(content_hash)
        if hit is not None:
            self.hits_exact += 1
            return hit, "exact"
        if self.enable_near_dup and phash:
            for known_phash, known_content in self._phash:
                if hamming(known_phash, phash) <= NEAR_DUP_MAX_DISTANCE:
                    near = self._exact.get(known_content)
                    if near is not None:
                        self.hits_near += 1
                        return near, "near"
        return None, ""

    def put(self, content_hash: str, phash: str | None, record: dict) -> None:
        self._exact[content_hash] = record
        if phash and not any(p == phash for p, _ in self._phash):
            self._phash.append((phash, content_hash))
