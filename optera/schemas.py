"""Canonical output schemas.

Four document classes plus two non-extraction outcomes. Every image in the
stream resolves to exactly one of these, and `not_a_document` / `unreadable`
are first-class results rather than errors — refusing is a correct answer.

The JSON Schemas here are used three ways:
  1. as the tool/structured-output schema sent to the model,
  2. as the contract the deterministic validators check, and
  3. as the shape the evaluator compares against ground truth.
"""
from __future__ import annotations

DOC_CLASSES = ("work_report", "vendor_invoice", "meter_reading", "not_a_document")
TERMINAL_CLASSES = ("not_a_document", "unreadable")

# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The label printed under the image, e.g. A1"},
                    "doc_class": {"type": "string", "enum": list(DOC_CLASSES) + ["unreadable"]},
                    "confidence": {"type": "number", "description": "0.0-1.0"},
                    "subject": {
                        "type": "string",
                        "description": "For not_a_document only: what the photo shows (battery, tyre, number_plate, fuel_cap, other).",
                    },
                },
                "required": ["id", "doc_class", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Extraction schemas
# --------------------------------------------------------------------------

_CONF = {"type": "number", "description": "Your confidence in this extraction, 0.0-1.0."}

WORK_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_class": {"type": "string", "enum": ["work_report"]},
        "depot": {"type": ["string", "null"], "description": "Site/depot printed or written in the header, e.g. VADAJ, PALDI, SARANGPUR, REMCO."},
        "report_date": {"type": ["string", "null"], "description": "ISO 8601 YYYY-MM-DD. Convert DD/MM/YY as written; do not guess a missing date."},
        "page_no": {"type": ["string", "null"]},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sr": {"type": ["string", "null"], "description": "Serial number as written."},
                    "vehicle_code": {"type": ["string", "null"], "description": "Bus/vehicle code, normalised uppercase no spaces, e.g. TCM35, MAM43, MAC-31."},
                    "work_done": {"type": ["string", "null"], "description": "Verbatim transcription. Keep the original script for Hindi/Gujarati; do not translate."},
                    "mechanic": {"type": ["string", "null"]},
                    "material": {"type": ["string", "null"]},
                    "struck_through": {"type": "boolean", "description": "True if the row is crossed out."},
                },
                "required": ["vehicle_code", "work_done"],
                "additionalProperties": False,
            },
        },
        "unreadable_regions": {"type": "array", "items": {"type": "string"}, "description": "Short notes on parts you could not read (finger over text, ink bleed, cut off)."},
        "confidence": _CONF,
    },
    "required": ["doc_class", "entries", "confidence"],
    "additionalProperties": False,
}

VENDOR_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_class": {"type": "string", "enum": ["vendor_invoice"]},
        "vendor_name": {"type": ["string", "null"]},
        "vendor_gstin": {"type": ["string", "null"]},
        "buyer_name": {"type": ["string", "null"]},
        "buyer_gstin": {"type": ["string", "null"]},
        "invoice_no": {"type": ["string", "null"]},
        "invoice_date": {"type": ["string", "null"], "description": "ISO 8601 YYYY-MM-DD."},
        "vehicle_no": {"type": ["string", "null"], "description": "Registration as printed, e.g. GJ05AU4001."},
        "document_kind": {"type": ["string", "null"], "enum": ["tax_invoice", "bill_of_supply", "cash_memo", "delivery_challan", "credit_memo", None]},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sr": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "hsn_sac": {"type": ["string", "null"]},
                    "qty": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "rate": {"type": ["number", "null"]},
                    "amount": {"type": ["number", "null"]},
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        },
        "subtotal": {"type": ["number", "null"]},
        "cgst": {"type": ["number", "null"]},
        "sgst": {"type": ["number", "null"]},
        "igst": {"type": ["number", "null"]},
        "round_off": {"type": ["number", "null"]},
        "total": {"type": ["number", "null"], "description": "Grand total payable."},
        "currency": {"type": ["string", "null"], "description": "ISO code, almost always INR."},
        "amount_in_words": {"type": ["string", "null"]},
        "confidence": _CONF,
    },
    "required": ["doc_class", "line_items", "total", "confidence"],
    "additionalProperties": False,
}

METER_READING_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_class": {"type": "string", "enum": ["meter_reading"]},
        "reading_kind": {"type": "string", "enum": ["odometer", "def_dispenser", "fuel_dispenser", "other"]},
        "odometer_km": {"type": ["number", "null"], "description": "Main ODO value in km. Do not report the trip meter here."},
        "trip_km": {"type": ["number", "null"]},
        "litres": {"type": ["number", "null"]},
        "amount_inr": {"type": ["number", "null"]},
        "rate_per_litre": {"type": ["number", "null"]},
        "urea_concentration_pct": {"type": ["number", "null"]},
        "vehicle_hint": {"type": ["string", "null"], "description": "Any vehicle identifier visible in the frame."},
        "captured_at": {"type": ["string", "null"], "description": "Timestamp burnt into the photo, ISO 8601, if present."},
        "confidence": _CONF,
    },
    "required": ["doc_class", "reading_kind", "confidence"],
    "additionalProperties": False,
}

NOT_A_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_class": {"type": "string", "enum": ["not_a_document"]},
        "subject": {"type": "string", "description": "battery, tyre, number_plate, fuel_cap, engine_part, other."},
        "visible_text": {"type": "array", "items": {"type": "string"}, "description": "Any text legible on the object. Text alone does not make it a document."},
        "reason": {"type": "string", "description": "One sentence on why no record can be extracted."},
        "confidence": _CONF,
    },
    "required": ["doc_class", "subject", "reason", "confidence"],
    "additionalProperties": False,
}

SCHEMA_BY_CLASS = {
    "work_report": WORK_REPORT_SCHEMA,
    "vendor_invoice": VENDOR_INVOICE_SCHEMA,
    "meter_reading": METER_READING_SCHEMA,
    "not_a_document": NOT_A_DOCUMENT_SCHEMA,
}

# The baseline uses one union schema for everything — that is the naive design
# on purpose: one big prompt, one big model, one call per image.
BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_class": {"type": "string", "enum": list(DOC_CLASSES) + ["unreadable"]},
        "work_report": {"type": ["object", "null"], "properties": WORK_REPORT_SCHEMA["properties"], "additionalProperties": False},
        "vendor_invoice": {"type": ["object", "null"], "properties": VENDOR_INVOICE_SCHEMA["properties"], "additionalProperties": False},
        "meter_reading": {"type": ["object", "null"], "properties": METER_READING_SCHEMA["properties"], "additionalProperties": False},
        "not_a_document": {"type": ["object", "null"], "properties": NOT_A_DOCUMENT_SCHEMA["properties"], "additionalProperties": False},
        "confidence": _CONF,
    },
    "required": ["doc_class", "confidence"],
    "additionalProperties": False,
}


def envelope(doc_id: str, doc_class: str, payload: dict, **meta) -> dict:
    """Uniform on-disk record. Baseline and optimized emit the same envelope so
    the evaluator can diff them field-by-field without special-casing."""
    return {
        "doc_id": doc_id,
        "doc_class": doc_class,
        "status": "rejected" if doc_class in TERMINAL_CLASSES else "extracted",
        "data": payload,
        "meta": meta,
    }
