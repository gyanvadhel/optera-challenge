"""Stage 2: routed extraction with validator-gated escalation.

The cheap model gets first attempt at every real document. Its output is then
checked by free deterministic validators (arithmetic, formats, completeness).
Only when that check fails — or the model itself reports low confidence — is the
expensive model paid, and it is told what went wrong so the retry is targeted.

This is the mechanism that lets the pipeline be much cheaper without being
cheaper-but-wrong: the failure mode of the cheap tier is *caught*, not shipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CHEAP_MODEL, ESCALATION_EDGE_PX, ESCALATION_MODEL, EXTRACT_EDGE_PX
from .imaging import Prepared, prepare
from .ledger import Ledger
from .prompts import ESCALATION_SUFFIX, EXTRACT_INSTRUCTION, EXTRACT_SYSTEM
from .schemas import SCHEMA_BY_CLASS
from .validate import validate


@dataclass
class ExtractResult:
    doc_class: str
    data: dict
    model: str
    escalated: bool
    validation: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)


def extract_document(
    provider,
    ledger: Ledger,
    doc_path: str,
    doc_class: str,
    *,
    run: str = "optimized",
    batched: bool = False,
    allow_escalation: bool = True,
) -> ExtractResult:
    schema = SCHEMA_BY_CLASS[doc_class]
    system = EXTRACT_SYSTEM[doc_class]

    # not_a_document needs a fraction of the tokens a real extraction does: a
    # small image and a short answer are enough to confirm "this is a battery".
    edge = EXTRACT_EDGE_PX.get(doc_class, 700)
    max_out = 4096 if doc_class in ("work_report", "vendor_invoice") else 700

    img = prepare(doc_path, edge)
    completion = provider.complete(
        model=CHEAP_MODEL, system=system, images=[img],
        instruction=EXTRACT_INSTRUCTION, schema=schema,
        max_tokens=max_out, cache_system=True,
    )
    ledger.record(
        run, img.doc_id, "extract", CHEAP_MODEL, completion.usage,
        batched=batched, live=completion.live,
        image_px=f"{img.width}x{img.height}", note=doc_class,
    )

    data = completion.data
    reasons = validate(doc_class, data)
    tiers = [CHEAP_MODEL]

    if reasons and allow_escalation:
        big = prepare(doc_path, ESCALATION_EDGE_PX)
        completion2 = provider.complete(
            model=ESCALATION_MODEL, system=system,
            images=[big],
            instruction=EXTRACT_INSTRUCTION + ESCALATION_SUFFIX.format(reasons="; ".join(reasons[:4])),
            schema=schema, max_tokens=max_out, cache_system=True,
        )
        ledger.record(
            run, big.doc_id, "escalate", ESCALATION_MODEL, completion2.usage,
            batched=batched, live=completion2.live,
            image_px=f"{big.width}x{big.height}",
            note=f"{doc_class}: {reasons[0][:90]}",
        )
        reasons2 = validate(doc_class, completion2.data)
        # Keep the escalated answer unless it is strictly worse than the cheap
        # one — a retry that fails more checks than the original is not progress.
        if len(reasons2) <= len(reasons):
            return ExtractResult(doc_class, completion2.data, ESCALATION_MODEL, True, reasons2,
                                 tiers + [ESCALATION_MODEL])
        return ExtractResult(doc_class, data, CHEAP_MODEL, True, reasons, tiers + [ESCALATION_MODEL])

    return ExtractResult(doc_class, data, CHEAP_MODEL, False, reasons, tiers)
