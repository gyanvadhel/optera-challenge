"""Deterministic validators — the accuracy floor under the cost optimisation.

This is what makes "cheaper" safe. The cheap tier is allowed to be wrong, as
long as being wrong is *detectable* without another model call. Every check
here is free arithmetic or a regex; a document that fails one is escalated to
the expensive model rather than shipped.

Each validator returns a list of human-readable failure reasons. Empty means
the extraction is internally consistent and structurally complete.
"""
from __future__ import annotations

import datetime as _dt
import re

from .config import EXTRACT_CONFIDENCE_FLOOR, INVOICE_TOTAL_TOLERANCE

GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d][Zz][A-Z\d]$")
VEHICLE_RE = re.compile(r"^[A-Z]{2}\s?-?\d{1,2}\s?-?[A-Z]{1,3}\s?-?\d{1,4}$")
# Most depots write a lettered fleet code (TCM35, MAM43, MAC-31). Some — the
# REMCO/MEMCO sheets in the starter set — write a bare bus number in the
# "BUS NO" column instead. A first live run escalated every one of those pages
# to the expensive tier because this pattern rejected bare digits: the model was
# right and the validator was wrong. Escalating correct work is pure cost.
BUS_CODE_RE = re.compile(r"^([A-Z]{2,4}[- ]?\d{1,3}|\d{1,4})$")


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _valid_date(s) -> bool:
    if not isinstance(s, str):
        return False
    try:
        d = _dt.date.fromisoformat(s)
    except ValueError:
        return False
    # These are 2026 operational documents; anything far outside that is a
    # misread year (a 2016/2026 confusion is a classic OCR failure).
    return _dt.date(2015, 1, 1) <= d <= _dt.date.today() + _dt.timedelta(days=365)


def validate(doc_class: str, data: dict) -> list[str]:
    if not isinstance(data, dict):
        return ["payload is not an object"]
    if data.get("_parse_error"):
        return ["model output was not valid JSON"]

    fn = {
        "work_report": _work_report,
        "vendor_invoice": _vendor_invoice,
        "meter_reading": _meter_reading,
        "not_a_document": _not_a_document,
    }.get(doc_class)
    if fn is None:
        return [f"unknown doc_class {doc_class!r}"]

    reasons = fn(data)
    conf = _num(data.get("confidence"))
    if conf is None:
        reasons.append("no confidence reported")
    elif conf < EXTRACT_CONFIDENCE_FLOOR:
        reasons.append(f"self-reported confidence {conf:.2f} below floor {EXTRACT_CONFIDENCE_FLOOR}")
    return reasons


# --------------------------------------------------------------------------


def _work_report(d: dict) -> list[str]:
    out: list[str] = []
    entries = d.get("entries")
    if not isinstance(entries, list) or not entries:
        out.append("no entries extracted from a work report")
        return out

    empty = sum(1 for e in entries if not (e.get("vehicle_code") or e.get("work_done")))
    if empty:
        out.append(f"{empty} entry rows are entirely empty")

    codes = [e.get("vehicle_code") for e in entries if e.get("vehicle_code")]
    if codes:
        bad = [c for c in codes if not BUS_CODE_RE.match(str(c).upper().replace(" ", ""))]
        # A couple of odd codes is normal handwriting; a majority is a misread page.
        if len(bad) > max(2, len(codes) * 0.4):
            out.append(f"{len(bad)}/{len(codes)} vehicle codes do not look like bus codes (e.g. {bad[:3]})")
    else:
        out.append("no vehicle codes found on a work report")

    if d.get("report_date") is not None and not _valid_date(d["report_date"]):
        out.append(f"report_date {d['report_date']!r} is not a plausible ISO date")
    return out


def _vendor_invoice(d: dict) -> list[str]:
    out: list[str] = []
    items = d.get("line_items")
    if not isinstance(items, list) or not items:
        out.append("no line items extracted from an invoice")
    items = items if isinstance(items, list) else []

    total = _num(d.get("total"))
    # A delivery challan legitimately has blank rate/amount columns — demanding
    # a total there would escalate a perfectly good extraction forever.
    # optera_doc_47 in the starter set is exactly this case.
    total_required = d.get("document_kind") != "delivery_challan"
    if total is None:
        if total_required:
            out.append("no grand total extracted")
    elif total <= 0:
        out.append(f"grand total {total} is not positive")

    # The arithmetic cross-check: line items + tax + round-off should reconcile
    # with the printed grand total. This catches transposed digits and misread
    # decimals that a confidence score alone will never surface.
    line_sum = sum(x for x in (_num(i.get("amount")) for i in items or []) if x is not None)
    subtotal = _num(d.get("subtotal"))
    taxes = sum(x for x in (_num(d.get(k)) for k in ("cgst", "sgst", "igst")) if x is not None)
    round_off = _num(d.get("round_off")) or 0.0

    if subtotal is not None and line_sum > 0 and abs(line_sum - subtotal) > max(INVOICE_TOTAL_TOLERANCE, subtotal * 0.02):
        out.append(f"line items sum to {line_sum:.2f} but subtotal reads {subtotal:.2f}")

    base = subtotal if subtotal is not None else (line_sum or None)
    if total is not None and base is not None:
        expected = base + taxes + round_off
        if abs(expected - total) > max(INVOICE_TOTAL_TOLERANCE, total * 0.02):
            out.append(f"subtotal+tax+roundoff = {expected:.2f} but total reads {total:.2f}")

    for field in ("vendor_gstin", "buyer_gstin"):
        v = d.get(field)
        if v and not GSTIN_RE.match(str(v).replace(" ", "").upper()):
            out.append(f"{field} {v!r} fails GSTIN checksum format")

    if d.get("invoice_date") is not None and not _valid_date(d["invoice_date"]):
        out.append(f"invoice_date {d['invoice_date']!r} is not a plausible ISO date")

    if not d.get("vendor_name"):
        out.append("no vendor name extracted")
    return out


def _meter_reading(d: dict) -> list[str]:
    out: list[str] = []
    kind = d.get("reading_kind")
    odo = _num(d.get("odometer_km"))
    trip = _num(d.get("trip_km"))
    litres = _num(d.get("litres"))
    amount = _num(d.get("amount_inr"))
    rate = _num(d.get("rate_per_litre"))

    if kind == "odometer":
        if odo is None:
            out.append("odometer reading has no odometer_km value")
        elif not (0 < odo < 3_000_000):
            out.append(f"odometer_km {odo} outside plausible range")
        if odo is not None and trip is not None and trip > odo:
            out.append(f"trip_km {trip} exceeds odometer_km {odo} — values likely swapped")
    elif kind in ("def_dispenser", "fuel_dispenser"):
        if litres is None and amount is None:
            out.append("dispenser reading has neither litres nor amount")
        # amount = litres x rate is a free, exact cross-check on all three digits.
        if None not in (litres, amount, rate) and rate > 0:
            expected = litres * rate
            if abs(expected - amount) > max(2.0, amount * 0.02):
                out.append(f"litres x rate = {expected:.2f} but amount reads {amount:.2f}")
        pct = _num(d.get("urea_concentration_pct"))
        if pct is not None and not (0 <= pct <= 100):
            out.append(f"urea concentration {pct} out of range")
    elif kind is None:
        out.append("no reading_kind")
    return out


def _not_a_document(d: dict) -> list[str]:
    out: list[str] = []
    if not d.get("subject"):
        out.append("no subject named for a rejected photo")
    if not d.get("reason"):
        out.append("no reason given for rejection")
    # Guard against the exact failure the brief calls out: a rejection that
    # still smuggles extracted rows through.
    for leak in ("line_items", "entries", "total", "odometer_km", "invoice_no"):
        if d.get(leak):
            out.append(f"rejected photo carries extracted field {leak!r} — hallucinated structure")
    return out
