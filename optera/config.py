"""Pricing tables, model tiers, and tunable thresholds.

Prices are USD per 1M tokens, Anthropic first-party list prices as of 2026-07-28.
Everything cost-related in this repo derives from this one table, so a price
change is a one-line edit and every number in the report moves with it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_write_5m: float  # 1.25x input
    cache_read: float  # 0.10x input
    # Longest image edge the model accepts before server-side downscale, and the
    # resulting hard ceiling on image tokens for a single image.
    max_image_edge_px: int
    max_image_tokens: int


PRICING: dict[str, Price] = {
    # High-resolution vision tier (2576px long edge, up to 4784 image tokens).
    "claude-opus-5": Price(5.00, 25.00, 6.25, 0.50, 2576, 4784),
    # Sonnet 5 list price. An introductory $2/$10 runs through 2026-08-31; we
    # deliberately bill at list so the reported savings are not flattered by a
    # promotion that expires.
    "claude-sonnet-5": Price(3.00, 15.00, 3.75, 0.30, 2576, 4784),
    # Standard vision tier (1568px long edge, ~1600 image tokens).
    "claude-haiku-4-5": Price(1.00, 5.00, 1.25, 0.10, 1568, 1600),

    # --- Google Gemini (AI Studio) ---------------------------------------
    # Google's PAID-tier rates, verified against ai.google.dev/gemini-api/docs/
    # pricing on 2026-07-28. A free-tier AI Studio key costs nothing to run,
    # which would make the cost comparison read $0.00 vs $0.00 and say nothing.
    # Pricing at paid rates keeps the report answering the question that matters
    # — what this costs at scale — while the run itself stays free.
    #
    # Pro rates are the <=200k-context tier; every request here is far under it.
    # Gemini has no cache-write premium (implicit caching is automatic), and
    # cached input reads at ~25% of the input rate.
    # Gemini bills images in 768px tiles at 258 tokens each, so the two
    # "edge/ceiling" columns are unused for these models (see imaging.py).
    "gemini-2.5-pro": Price(1.25, 10.00, 1.25, 0.3125, 3072, 6192),
    "gemini-2.5-flash": Price(0.30, 2.50, 0.30, 0.075, 3072, 6192),
    "gemini-2.5-flash-lite": Price(0.10, 0.40, 0.10, 0.025, 3072, 6192),
    # Newer generation, also available on the same key. Not the default only
    # because the 2.5 ladder gives a cleaner same-family three-tier comparison;
    # switch with OPTERA_* env vars if you want the accuracy.
    "gemini-3.1-pro-preview": Price(2.00, 12.00, 2.00, 0.50, 3072, 6192),
    "gemini-3.6-flash": Price(1.50, 7.50, 1.50, 0.375, 3072, 6192),
    "gemini-3.5-flash": Price(1.50, 9.00, 1.50, 0.375, 3072, 6192),
    "gemini-3.5-flash-lite": Price(0.30, 2.50, 0.30, 0.075, 3072, 6192),
    "gemini-3.1-flash-lite": Price(0.25, 1.50, 0.25, 0.0625, 3072, 6192),
}

# Providers whose image billing is tile-based rather than area-based.
GEMINI_MODELS = {m for m in PRICING if m.startswith("gemini-")}

# Message Batches API: 50% off all token usage, <24h turnaround.
BATCH_DISCOUNT = 0.50

# Minimum cacheable prefix, in tokens, per model. Below this the API silently
# does not cache — no error, just cache_creation_input_tokens == 0. This is the
# single most important gotcha in the whole cost model; see DESIGN.md.
CACHE_MIN_TOKENS: dict[str, int] = {
    "claude-opus-5": 512,
    "claude-sonnet-5": 1024,
    "claude-haiku-4-5": 4096,
    **{m: 1024 for m in PRICING if m.startswith("gemini-")},
}

# Model tier sets, selected by --provider.
MODEL_SETS: dict[str, dict[str, str]] = {
    "anthropic": {
        "baseline": "claude-opus-5",
        "router": "claude-haiku-4-5",
        "cheap": "claude-haiku-4-5",
        "escalation": "claude-opus-5",
    },
    # Three-tier ladder, constrained by what an AI Studio FREE key can reach.
    # Measured empirically, because the docs do not make this obvious:
    #   * Pro (2.5-pro, 3.x-pro, pro-latest) -> 429, zero free quota.
    #   * 2.5-flash / 2.5-flash-lite -> 404 "no longer available to new users".
    #   * 3.6-flash and flash-latest -> 20 requests PER DAY. Unusable for a
    #     47-image benchmark; a single run exhausts them.
    # What is left, and what the ladder is built from:
    #   router     3.1-flash-lite  $0.25 / $1.50
    #   cheap      3.5-flash-lite  $0.30 / $2.50
    #   baseline   3.5-flash       $1.50 / $9.00   (most capable with real quota)
    # a 6x input / 6x output spread end to end. On a PAID key the top of the
    # ladder becomes Pro and the spread widens, so the savings measured here are
    # a conservative floor rather than a best case. See DESIGN.md 2b.
    "gemini": {
        "baseline": os.getenv("OPTERA_GEMINI_BASELINE", "gemini-3.5-flash"),
        "router": os.getenv("OPTERA_GEMINI_ROUTER", "gemini-3.1-flash-lite"),
        "cheap": os.getenv("OPTERA_GEMINI_CHEAP", "gemini-3.5-flash-lite"),
        "escalation": os.getenv("OPTERA_GEMINI_ESCALATION", "gemini-3.5-flash"),
    },
}


# Gemini bills vision in 768px tiles, which makes resolution a step function
# instead of a smooth curve. Two consequences worth exploiting:
#   * A 1568px page costs 6 tiles; a 1536px page costs 4. That is 33% off for a
#     2% resolution drop, purely from landing inside a tile boundary.
#   * The router thumbnail is one tile at anything up to 768px, so 448px and
#     768px cost exactly the same — take the larger one for free accuracy.
GEMINI_ROUTER_EDGE_PX = 768
GEMINI_EXTRACT_EDGE_PX = {
    "work_report": 1536,
    "vendor_invoice": 1536,
    "meter_reading": 768,
}
GEMINI_ESCALATION_EDGE_PX = 2304


def use_model_set(provider: str) -> None:
    """Point the tier globals at one provider's models. Called once at startup."""
    tiers = MODEL_SETS.get(provider)
    if not tiers:
        return
    global BASELINE_MODEL, ROUTER_MODEL, CHEAP_MODEL, ESCALATION_MODEL
    global ROUTER_EDGE_PX, EXTRACT_EDGE_PX, ESCALATION_EDGE_PX
    BASELINE_MODEL = tiers["baseline"]
    ROUTER_MODEL = tiers["router"]
    CHEAP_MODEL = tiers["cheap"]
    ESCALATION_MODEL = tiers["escalation"]

    if provider == "gemini":
        ROUTER_EDGE_PX = GEMINI_ROUTER_EDGE_PX
        EXTRACT_EDGE_PX = dict(GEMINI_EXTRACT_EDGE_PX)
        ESCALATION_EDGE_PX = GEMINI_ESCALATION_EDGE_PX
        import optera.extract as _ex_edges
        import optera.pipeline as _pl
        _ex_edges.EXTRACT_EDGE_PX = EXTRACT_EDGE_PX
        _ex_edges.ESCALATION_EDGE_PX = ESCALATION_EDGE_PX
        _pl.ROUTER_EDGE_PX = ROUTER_EDGE_PX

    # The stages captured these names at import time; rebind them too.
    import optera.baseline as _bl
    import optera.extract as _ex
    import optera.router as _rt
    _bl.BASELINE_MODEL = BASELINE_MODEL
    _rt.ROUTER_MODEL, _rt.ESCALATION_MODEL = ROUTER_MODEL, ESCALATION_MODEL
    _ex.CHEAP_MODEL, _ex.ESCALATION_MODEL = CHEAP_MODEL, ESCALATION_MODEL


# --------------------------------------------------------------------------
# Model tiers
# --------------------------------------------------------------------------

BASELINE_MODEL = os.getenv("OPTERA_BASELINE_MODEL", "claude-opus-5")
ROUTER_MODEL = os.getenv("OPTERA_ROUTER_MODEL", "claude-haiku-4-5")
CHEAP_MODEL = os.getenv("OPTERA_CHEAP_MODEL", "claude-haiku-4-5")
ESCALATION_MODEL = os.getenv("OPTERA_ESCALATION_MODEL", "claude-opus-5")


# --------------------------------------------------------------------------
# Image budgets (the dominant cost lever — see DESIGN.md §2)
# --------------------------------------------------------------------------

# Router only needs to answer "what kind of thing is this?", which survives
# aggressive downscaling. 448px long edge ~= 130 image tokens.
ROUTER_EDGE_PX = 448

# Per-class extraction resolution. Handwriting needs more pixels than a printed
# invoice; a 7-segment odometer needs almost none.
EXTRACT_EDGE_PX: dict[str, int] = {
    "work_report": 1568,  # multilingual handwriting — the hardest class
    "vendor_invoice": 1400,  # printed text, tolerates a little less
    "meter_reading": 900,  # a handful of large digits
}
ESCALATION_EDGE_PX = 2000  # retry bigger when the cheap tier fails validation

# How many thumbnails to pack into a single router call. Amortises the router
# system prompt across N documents instead of paying it N times.
ROUTER_BATCH_SIZE = int(os.getenv("OPTERA_ROUTER_BATCH", "6"))


# --------------------------------------------------------------------------
# Routing / escalation thresholds
# --------------------------------------------------------------------------

# Router self-reported confidence below this goes straight to the escalation
# model for classification rather than being trusted.
ROUTER_CONFIDENCE_FLOOR = 0.70

# Extraction confidence below this triggers escalation even if the deterministic
# validators pass.
EXTRACT_CONFIDENCE_FLOOR = 0.60

# Invoice arithmetic tolerance, in rupees. Handwritten bills round; a 1-rupee
# discrepancy is not an extraction error.
INVOICE_TOTAL_TOLERANCE = 1.50

MAX_ESCALATIONS_PER_DOC = 1


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

IMAGE_DIR = os.getenv("OPTERA_IMAGE_DIR", "data/images")
OUT_DIR = os.getenv("OPTERA_OUT_DIR", "out")
CACHE_DIR = os.getenv("OPTERA_CACHE_DIR", ".optera_cache")
GROUND_TRUTH_DIR = "data/ground_truth"
