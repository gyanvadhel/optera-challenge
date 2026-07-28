"""Stage 1: classify a batch of thumbnails with the cheapest model.

The router answers one four-way question, so it gets the smallest image the
question survives (448px long edge, ~130 image tokens) and several documents
share a single call so the system prompt is paid once per batch instead of once
per document.
"""
from __future__ import annotations

from .config import ROUTER_CONFIDENCE_FLOOR, ROUTER_MODEL, ESCALATION_MODEL
from .imaging import Prepared
from .ledger import Ledger
from .prompts import ROUTER_INSTRUCTION, ROUTER_SYSTEM
from .schemas import ROUTER_SCHEMA


def route_batch(
    provider,
    ledger: Ledger,
    images: list[Prepared],
    *,
    run: str = "optimized",
    batched: bool = False,
) -> dict[str, dict]:
    """Classify a batch. Returns {doc_id: {doc_class, confidence, subject?}}."""
    if not images:
        return {}

    completion = provider.complete(
        model=ROUTER_MODEL,
        system=ROUTER_SYSTEM,
        images=images,
        instruction=ROUTER_INSTRUCTION,
        schema=ROUTER_SCHEMA,
        max_tokens=64 * len(images) + 128,
        cache_system=True,
    )
    ledger.record(
        run, [i.doc_id for i in images], "route", ROUTER_MODEL, completion.usage,
        batched=batched, live=completion.live,
        image_px=f"{images[0].width}x{images[0].height}",
        note=f"batch of {len(images)}",
    )

    out: dict[str, dict] = {}
    for row in completion.data.get("results", []):
        if isinstance(row, dict) and row.get("id"):
            out[str(row["id"])] = row

    # Any image the router failed to answer for is treated as low confidence
    # rather than silently dropped.
    for img in images:
        out.setdefault(img.doc_id, {"doc_class": "unreadable", "confidence": 0.0,
                                    "subject": None, "_missing_from_router": True})
    return out


def needs_second_opinion(row: dict) -> bool:
    """A cheap classification we should not act on without a closer look."""
    conf = row.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return True
    return conf < ROUTER_CONFIDENCE_FLOOR or row.get("doc_class") == "unreadable"


def reroute_one(provider, ledger: Ledger, image: Prepared, *, run: str = "optimized", batched: bool = False) -> dict:
    """Re-classify a single uncertain image on the stronger model.

    Only reached for the small minority the cheap router is unsure about, which
    is the whole point: pay Opus prices for the hard 10%, not the easy 90%.
    """
    completion = provider.complete(
        model=ESCALATION_MODEL,
        system=ROUTER_SYSTEM,
        images=[image],
        instruction=ROUTER_INSTRUCTION,
        schema=ROUTER_SCHEMA,
        max_tokens=192,
        cache_system=True,
    )
    ledger.record(
        run, image.doc_id, "route_escalate", ESCALATION_MODEL, completion.usage,
        batched=batched, live=completion.live,
        image_px=f"{image.width}x{image.height}", note="router confidence below floor",
    )
    rows = completion.data.get("results") or []
    if rows and isinstance(rows[0], dict):
        return rows[0]
    return {"doc_class": "unreadable", "confidence": 0.0}
