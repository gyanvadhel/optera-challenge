"""Accuracy measurement.

Three separate things are measured, because they have very different evidence
behind them and conflating them would be dishonest:

  A. Routing accuracy — against hand labels for all 47 images. Complete.
  B. Field accuracy   — against hand-transcribed fields for a 30-document
                        subset. Partial by design; only keys present in the
                        ground truth are scored.
  C. Baseline agreement — field-by-field agreement between the baseline and the
                        optimized run across ALL documents. This is what
                        supports "cheaper without dropping accuracy" on the
                        documents nobody hand-transcribed: if the two runs
                        agree, the cheap path did not lose anything the
                        expensive path found.

A regression in C with no ground truth is still a finding — it says the two
paths diverge and a human should look.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

from .config import GROUND_TRUTH_DIR

_PUNCT = re.compile(r"[^a-z0-9]+")


def _norm(v):
    """Comparison normalisation. Deliberately forgiving about formatting and
    strict about content: 'Shiv Shakti Motors' == 'SHIV SHAKTI MOTORS', and
    2000 == 2000.0, but 2000 != 2000.06."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    s = str(v).strip()
    if not s:
        return None
    try:
        return round(float(s.replace(",", "")), 2)
    except ValueError:
        pass
    return _PUNCT.sub("", s.lower()) or None


def _eq(a, b) -> bool:
    na, nb = _norm(a), _norm(b)
    if isinstance(na, list) and isinstance(nb, list):
        return na == nb
    if isinstance(na, float) and isinstance(nb, float):
        return abs(na - nb) < 0.011
    return na == nb


@dataclass
class FieldScore:
    correct: int = 0
    wrong: int = 0
    missing: int = 0  # ground truth has a value, extraction has null
    details: list = field(default_factory=list)

    @property
    def scored(self) -> int:
        return self.correct + self.wrong + self.missing

    @property
    def accuracy(self) -> float:
        return self.correct / self.scored if self.scored else 0.0


def load_ground_truth() -> tuple[dict, dict]:
    def load(name):
        p = pathlib.Path(GROUND_TRUTH_DIR) / name
        blob = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return {k: v for k, v in blob.items() if not k.startswith("_")}

    return load("routing.json"), load("fields.json")


# --------------------------------------------------------------------------
# A. Routing
# --------------------------------------------------------------------------


def score_routing(results: dict[str, dict], routing_gt: dict) -> dict:
    correct = 0
    confusion: dict[str, int] = {}
    errors = []
    hallucinated_structure = 0

    for doc_id, gt in routing_gt.items():
        got = results.get(doc_id)
        if got is None:
            errors.append({"doc_id": doc_id, "expected": gt["doc_class"], "got": "<not processed>"})
            confusion[f"{gt['doc_class']}->missing"] = confusion.get(f"{gt['doc_class']}->missing", 0) + 1
            continue
        exp, act = gt["doc_class"], got.get("doc_class")
        if exp == act:
            correct += 1
        else:
            errors.append({"doc_id": doc_id, "expected": exp, "got": act})
        confusion[f"{exp}->{act}"] = confusion.get(f"{exp}->{act}", 0) + 1

        # The specific failure the brief warns about: structure invented from a
        # photo of an object. Counted separately because it is worse than a
        # plain misroute.
        if exp in ("not_a_document", "unreadable") and act not in ("not_a_document", "unreadable"):
            data = got.get("data") or {}
            if data.get("line_items") or data.get("entries") or data.get("total") is not None:
                hallucinated_structure += 1

    total = len(routing_gt)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "errors": errors,
        "confusion": confusion,
        "hallucinated_structure_on_non_documents": hallucinated_structure,
    }


# --------------------------------------------------------------------------
# B. Field accuracy vs ground truth
# --------------------------------------------------------------------------

# Derived keys that need a projection out of the extracted payload before
# they can be compared to the ground truth.
_DERIVED = {
    "entry_vehicle_codes": lambda d: [
        str(e.get("vehicle_code", "")).upper().replace(" ", "")
        for e in (d.get("entries") or [])
        if e.get("vehicle_code")
    ],
}


def score_fields(results: dict[str, dict], fields_gt: dict) -> dict:
    per_field: dict[str, FieldScore] = {}
    per_doc: dict[str, dict] = {}

    for doc_id, gt in fields_gt.items():
        got = results.get(doc_id)
        data = (got or {}).get("data") or {}
        doc_correct = doc_wrong = doc_missing = 0

        for key, expected in gt.items():
            if key.startswith("_") or key == "doc_class":
                continue
            score = per_field.setdefault(key, FieldScore())
            actual = _DERIVED[key](data) if key in _DERIVED else data.get(key)

            if _eq(expected, actual):
                score.correct += 1
                doc_correct += 1
            elif actual in (None, "", []):
                score.missing += 1
                doc_missing += 1
                score.details.append({"doc_id": doc_id, "expected": expected, "got": None})
            else:
                score.wrong += 1
                doc_wrong += 1
                score.details.append({"doc_id": doc_id, "expected": expected, "got": actual})

        per_doc[doc_id] = {"correct": doc_correct, "wrong": doc_wrong, "missing": doc_missing}

    correct = sum(s.correct for s in per_field.values())
    wrong = sum(s.wrong for s in per_field.values())
    missing = sum(s.missing for s in per_field.values())
    scored = correct + wrong + missing

    return {
        "documents_with_ground_truth": len(fields_gt),
        "fields_scored": scored,
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "accuracy": correct / scored if scored else 0.0,
        # Precision-style view: of the fields we DID emit a value for, how many
        # were right? This is the number that matters for hallucination risk.
        "precision_when_answered": correct / (correct + wrong) if (correct + wrong) else 0.0,
        "per_field": {
            k: {"correct": v.correct, "wrong": v.wrong, "missing": v.missing, "accuracy": round(v.accuracy, 4)}
            for k, v in sorted(per_field.items())
        },
        "per_doc": per_doc,
        "mismatches": [d for s in per_field.values() for d in s.details],
    }


# --------------------------------------------------------------------------
# C. Baseline vs optimized agreement (all documents, no ground truth needed)
# --------------------------------------------------------------------------

_AGREEMENT_KEYS = {
    "work_report": ["depot", "report_date"],
    "vendor_invoice": ["vendor_name", "invoice_no", "invoice_date", "vehicle_no", "total"],
    "meter_reading": ["reading_kind", "odometer_km", "trip_km", "litres", "amount_inr", "rate_per_litre"],
    "not_a_document": ["subject"],
}


def score_agreement(baseline: dict[str, dict], optimized: dict[str, dict]) -> dict:
    class_agree = class_total = 0
    field_agree = field_total = 0
    divergences = []

    for doc_id, base in baseline.items():
        opt = optimized.get(doc_id)
        if opt is None:
            continue
        class_total += 1
        if base.get("doc_class") == opt.get("doc_class"):
            class_agree += 1
        else:
            divergences.append({"doc_id": doc_id, "field": "doc_class",
                                "baseline": base.get("doc_class"), "optimized": opt.get("doc_class")})
            continue

        for key in _AGREEMENT_KEYS.get(base.get("doc_class"), []):
            bv = (base.get("data") or {}).get(key)
            ov = (opt.get("data") or {}).get(key)
            if bv is None and ov is None:
                continue  # neither run found it; nothing to compare
            field_total += 1
            if _eq(bv, ov):
                field_agree += 1
            else:
                divergences.append({"doc_id": doc_id, "field": key, "baseline": bv, "optimized": ov})

    return {
        "documents_compared": class_total,
        "class_agreement": class_agree / class_total if class_total else 0.0,
        "fields_compared": field_total,
        "field_agreement": field_agree / field_total if field_total else 0.0,
        "divergences": divergences,
    }


def evaluate(baseline: dict | None, optimized: dict | None) -> dict:
    routing_gt, fields_gt = load_ground_truth()
    out: dict = {}
    if optimized:
        out["optimized"] = {
            "routing": score_routing(optimized, routing_gt),
            "fields": score_fields(optimized, fields_gt),
        }
    if baseline:
        out["baseline"] = {
            "routing": score_routing(baseline, routing_gt),
            "fields": score_fields(baseline, fields_gt),
        }
    if baseline and optimized:
        out["agreement"] = score_agreement(baseline, optimized)
    return out
