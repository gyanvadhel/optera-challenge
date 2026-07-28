"""Deterministic offline provider — no API key, no network.

Purpose: let `python run.py` work end to end on a clean checkout so the routing,
validation, escalation, caching and accounting machinery is all exercised and
testable. It is NOT a substitute for a real run:

  * TOKEN COUNTS ARE REAL-ISH.  Image tokens come from the same (w*h)/750 tile
    math the API bills on, so the cost comparison is a genuine model of the
    request shapes the pipeline builds. Text tokens are estimated at ~3.6
    chars/token. Every ledger row it writes is stamped live=False and the report
    labels the whole run SIMULATED.
  * ACCURACY IS NOT MEASURED.  Answers are derived from the committed ground
    truth with seeded, deliberate corruption injected at `error_rate` so that
    the validators and the escalation ladder actually fire. Accuracy from an
    offline run tells you the harness works; it tells you nothing about a model.

Run with a real key to turn both into measurements.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random

from ..config import GROUND_TRUTH_DIR
from ..imaging import Prepared, estimate_text_tokens, image_tokens
from ..ledger import Usage
from .base import Completion


class OfflineProvider:
    name = "offline"
    live = False

    def __init__(self, error_rate: float = 0.18, seed: int = 7):
        self.error_rate = error_rate
        self.seed = seed
        self._routing = _load(f"{GROUND_TRUTH_DIR}/routing.json")
        self._fields = _load(f"{GROUND_TRUTH_DIR}/fields.json")

    # -- helpers ---------------------------------------------------------

    def _rng(self, *parts: str) -> random.Random:
        key = "|".join(parts) + f"|{self.seed}"
        return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))

    def _class_of(self, doc_id: str) -> str:
        return (self._routing.get(doc_id) or {}).get("doc_class", "work_report")

    # -- provider API ----------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: str,
        images: list[Prepared],
        instruction: str,
        schema: dict,
        max_tokens: int = 4096,
        cache_system: bool = False,
    ) -> Completion:
        in_tok = estimate_text_tokens(system) + estimate_text_tokens(instruction)
        for img in images:
            in_tok += image_tokens(img.width, img.height, model)
            if len(images) > 1:
                in_tok += 6  # per-image "[label]" text block

        is_router = "results" in schema.get("properties", {})
        if is_router:
            data = self._route(images, model)
        elif "work_report" in schema.get("properties", {}):
            data = self._baseline(images[0], model)
        else:
            data = self._extract(images[0], model)

        text = json.dumps(data, ensure_ascii=False)
        usage = Usage(input_tokens=in_tok, output_tokens=estimate_text_tokens(text))
        return Completion(data=data, usage=usage, live=False, model=model, raw_text=text)

    # -- synthesis -------------------------------------------------------

    def _route(self, images: list[Prepared], model: str) -> dict:
        results = []
        for img in images:
            rng = self._rng("route", img.doc_id, model)
            cls = self._class_of(img.doc_id)
            conf = round(rng.uniform(0.82, 0.99), 2)
            # Cheap routers do get things wrong; simulate that so the confidence
            # floor and the escalation path are genuinely exercised.
            if model.startswith("claude-haiku") and rng.random() < self.error_rate * 0.5:
                conf = round(rng.uniform(0.30, 0.68), 2)
            row = {"id": img.doc_id, "doc_class": cls, "confidence": conf}
            if cls == "not_a_document":
                row["subject"] = (self._routing.get(img.doc_id) or {}).get("subject", "other")
            results.append(row)
        return {"results": results}

    def _payload(self, doc_id: str, cls: str, model: str) -> dict:
        rng = self._rng("extract", doc_id, model)
        gt = {k: v for k, v in (self._fields.get(doc_id) or {}).items() if not k.startswith("_")}
        # Ground-truth-only projections are not part of the wire schema.
        entry_codes = gt.pop("entry_vehicle_codes", None)
        gt.pop("doc_class", None)
        payload = json.loads(json.dumps(_defaults(cls) | gt))
        payload["doc_class"] = cls
        _synthesise_body(payload, cls, rng, entry_codes)

        degrade = model.startswith("claude-haiku")
        conf = round(rng.uniform(0.78, 0.97), 2)
        if degrade and rng.random() < self.error_rate:
            conf = round(rng.uniform(0.25, 0.58), 2)
            # Inject a defect the deterministic validators are designed to catch,
            # so escalation is triggered by evidence rather than by a coin flip.
            if cls == "vendor_invoice" and payload.get("total") is not None:
                payload["total"] = round(float(payload["total"]) * 1.11 + 3, 2)
            elif cls == "work_report" and payload.get("entries"):
                payload["entries"] = payload["entries"][:-1] or payload["entries"]
            elif cls == "meter_reading":
                payload["odometer_km"] = None
        payload["confidence"] = conf
        return payload

    def _extract(self, img: Prepared, model: str) -> dict:
        return self._payload(img.doc_id, self._class_of(img.doc_id), model)

    def _baseline(self, img: Prepared, model: str) -> dict:
        cls = self._class_of(img.doc_id)
        payload = self._payload(img.doc_id, cls, model)
        return {"doc_class": cls, cls: payload, "confidence": payload["confidence"]}


_DEPOT_CODES = ("TCM", "TAM", "MAM", "MAC", "TOM")
_JOBS = (
    "All wheel brake set", "engine oil topup 5 ltr", "battery was faulty",
    "coolant hose change", "spark plug change", "head light bulb change",
    "axle packing change rear left side", "radiator change, coolant topup",
    "steering oil topup 500 ml", "clutch oiling, gear oiling",
)
_PARTS = ("HOSE CLIP STEEL HD", "COTTON WEST CLOTH", "SIDE MIRROR STAND TC",
          "BRAKE FUEL 250ML TVS", "DAMAR PATTI", "LST101L FUEL STAINER")


def _synthesise_body(payload: dict, cls: str, rng, entry_codes) -> None:
    """Fill in the bulk fields the committed ground truth deliberately omits.

    The ground truth records header fields only (transcribing 25 multilingual
    handwritten pages in full was out of scope). Without a body the validators
    would fire on every single document and the simulated run would escalate
    everything — measuring the simulator's gaps rather than the pipeline. This
    generates a plausible, internally consistent body so the escalation rate
    reflects the injected defect rate instead.
    """
    if cls == "work_report" and not payload.get("entries"):
        codes = entry_codes or [
            f"{rng.choice(_DEPOT_CODES)}{rng.randint(1, 65)}" for _ in range(rng.randint(6, 14))
        ]
        payload["entries"] = [
            {"sr": f"{i + 1:02d}", "vehicle_code": c, "work_done": rng.choice(_JOBS),
             "mechanic": None, "material": None, "struck_through": False}
            for i, c in enumerate(codes)
        ]
    elif cls == "vendor_invoice" and not payload.get("line_items"):
        total = payload.get("total")
        subtotal = payload.get("subtotal")
        taxes = sum(payload.get(k) or 0 for k in ("cgst", "sgst", "igst"))
        if subtotal is None and total is not None:
            subtotal = round(total - taxes - (payload.get("round_off") or 0), 2)
            payload["subtotal"] = subtotal
        if subtotal:
            # Split the subtotal across 1-3 lines so the arithmetic validator
            # reconciles exactly.
            n = rng.randint(1, 3)
            cuts = sorted(round(rng.uniform(0.15, 0.85), 4) for _ in range(n - 1))
            bounds = [0.0, *cuts, 1.0]
            amounts = [round(subtotal * (bounds[i + 1] - bounds[i]), 2) for i in range(n)]
            amounts[-1] = round(subtotal - sum(amounts[:-1]), 2)
            payload["line_items"] = [
                {"sr": str(i + 1), "description": rng.choice(_PARTS), "hsn_sac": None,
                 "qty": 1.0, "unit": "Pcs", "rate": a, "amount": a}
                for i, a in enumerate(amounts)
            ]
        else:
            payload["line_items"] = [
                {"sr": "1", "description": rng.choice(_PARTS), "hsn_sac": None,
                 "qty": 1.0, "unit": "Pcs", "rate": None, "amount": None}
            ]
    elif cls == "meter_reading":
        if payload.get("reading_kind") == "odometer" and payload.get("odometer_km") is None:
            payload["odometer_km"] = round(rng.uniform(20_000, 400_000), 1)


def _defaults(cls: str) -> dict:
    if cls == "work_report":
        return {"depot": None, "report_date": None, "page_no": None, "entries": [], "unreadable_regions": []}
    if cls == "vendor_invoice":
        return {
            "vendor_name": None, "vendor_gstin": None, "buyer_name": None, "buyer_gstin": None,
            "invoice_no": None, "invoice_date": None, "vehicle_no": None, "document_kind": None,
            "line_items": [], "subtotal": None, "cgst": None, "sgst": None, "igst": None,
            "round_off": None, "total": None, "currency": "INR", "amount_in_words": None,
        }
    if cls == "meter_reading":
        return {
            "reading_kind": "odometer", "odometer_km": None, "trip_km": None, "litres": None,
            "amount_inr": None, "rate_per_litre": None, "urea_concentration_pct": None,
            "vehicle_hint": None, "captured_at": None,
        }
    return {"subject": "other", "visible_text": [], "reason": "Photograph of an object, not a record."}


def _load(path: str) -> dict:
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
