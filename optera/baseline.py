"""The naive 1x baseline.

One call per image, to the biggest multimodal model, at full resolution, with a
single prompt covering every document class. No routing, no caching, no
batching, no prefilter — deliberately. This is the number everything else is
measured against, so it has to be an honest, competent implementation of the
obvious approach rather than a strawman: same schemas, same field descriptions,
same extraction rules as the optimized path.
"""
from __future__ import annotations

import json
import pathlib

from .config import BASELINE_MODEL, IMAGE_DIR, OUT_DIR, PRICING
from .imaging import prepare
from .ledger import Ledger, Usage
from .prompts import BASELINE_INSTRUCTION, BASELINE_SYSTEM
from .schemas import BASELINE_SCHEMA, envelope


def run_baseline(provider, ledger: Ledger, doc_paths: list[pathlib.Path], *, batched: bool = False) -> dict[str, dict]:
    out_dir = pathlib.Path(OUT_DIR) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    # Full resolution for the model's vision tier — the naive choice, and the
    # single biggest reason the baseline is expensive.
    edge = PRICING[BASELINE_MODEL].max_image_edge_px

    for path in doc_paths:
        doc_id = path.stem
        img = prepare(path, edge)

        if not img.ok:
            # Even the naive path cannot send bytes that are not an image. We
            # still record the outcome so per-doc counts line up across runs,
            # at zero cost — the baseline gets this one for free too.
            record = envelope(doc_id, "unreadable", {"reason": img.reason}, model=None, prefiltered=True)
            ledger.record("baseline", doc_id, "prefilter", "none", Usage(),
                          live=provider.live, note=img.reason or "")
            results[doc_id] = record
            (out_dir / f"{doc_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        completion = provider.complete(
            model=BASELINE_MODEL,
            system=BASELINE_SYSTEM,
            images=[img],
            instruction=BASELINE_INSTRUCTION,
            schema=BASELINE_SCHEMA,
            max_tokens=4096,
            cache_system=False,  # naive: no caching
        )
        ledger.record(
            "baseline", doc_id, "extract", BASELINE_MODEL, completion.usage,
            batched=batched, live=completion.live,
            image_px=f"{img.width}x{img.height}", note="single-call baseline",
        )

        data = completion.data or {}
        cls = data.get("doc_class", "unreadable")
        payload = data.get(cls) if isinstance(data.get(cls), dict) else data
        record = envelope(doc_id, cls, payload or {}, model=BASELINE_MODEL, confidence=data.get("confidence"))
        results[doc_id] = record
        (out_dir / f"{doc_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return results
