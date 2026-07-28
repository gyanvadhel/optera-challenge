"""Tests for the parts that must not silently break.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from optera.cache import ResultCache  # noqa: E402
from optera.imaging import dhash, hamming, image_tokens, prepare, sniff  # noqa: E402
from optera.ledger import Ledger, Usage, cost_usd  # noqa: E402
from optera.validate import validate  # noqa: E402

IMAGES = pathlib.Path("data/images")


class TestPrefilter(unittest.TestCase):
    def test_html_masquerading_as_jpeg_is_rejected_for_free(self):
        """optera_doc_33 is a Cloudflare error page saved with a .jpg extension.
        It must never reach a model."""
        path = IMAGES / "optera_doc_33.jpg"
        if not path.exists():
            self.skipTest("starter images not present")
        out = prepare(path, 448)
        self.assertFalse(out.ok)
        self.assertIn("not an image", out.reason)

    def test_truncated_jpeg_is_repaired_not_rejected(self):
        """optera_doc_47 is a real invoice whose file is 4 bytes short.
        Rejecting it would be a false negative on a genuine document."""
        path = IMAGES / "optera_doc_47.jpg"
        if not path.exists():
            self.skipTest("starter images not present")
        out = prepare(path, 448)
        self.assertTrue(out.ok, out.reason)
        self.assertTrue(out.was_repaired)
        self.assertGreater(out.width, 0)

    def test_sniff_rejects_non_images(self):
        self.assertIsNone(sniff(b"<!doctype html><html>"))
        self.assertEqual(sniff(b"\xff\xd8\xff\xe0rest"), "image/jpeg")


class TestImageTokens(unittest.TestCase):
    def test_area_scaling(self):
        """Tokens scale with area, so halving the edge should quarter the cost."""
        big = image_tokens(1000, 1000, "claude-opus-5")
        small = image_tokens(500, 500, "claude-opus-5")
        self.assertAlmostEqual(big / small, 4.0, delta=0.05)

    def test_tier_ceilings_are_enforced(self):
        self.assertLessEqual(image_tokens(6000, 8000, "claude-haiku-4-5"), 1600)
        self.assertLessEqual(image_tokens(6000, 8000, "claude-opus-5"), 4784)

    def test_router_thumbnail_is_an_order_of_magnitude_cheaper(self):
        router = image_tokens(448, 597, "claude-haiku-4-5")
        full = image_tokens(3000, 4000, "claude-haiku-4-5")
        self.assertLess(router * 4, full)


class TestGeminiAdapter(unittest.TestCase):
    """Gemini's response_schema is an OpenAPI subset, not JSON Schema. If this
    translation regresses, every structured call 400s."""

    def _walk(self, node, path="root"):
        bad = []
        if isinstance(node, dict):
            if "additionalProperties" in node:
                bad.append(f"{path}: additionalProperties is rejected by Gemini")
            if isinstance(node.get("type"), list):
                bad.append(f"{path}: type unions are rejected by Gemini")
            if node.get("type") and not str(node["type"]).isupper():
                bad.append(f"{path}: type must be upper-case")
            for k, v in node.items():
                bad += self._walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                bad += self._walk(v, f"{path}[{i}]")
        return bad

    def test_all_schemas_translate_cleanly(self):
        from optera.providers.gemini import to_gemini_schema
        from optera.schemas import BASELINE_SCHEMA, ROUTER_SCHEMA, SCHEMA_BY_CLASS

        for name, schema in [("router", ROUTER_SCHEMA), ("baseline", BASELINE_SCHEMA),
                             *SCHEMA_BY_CLASS.items()]:
            with self.subTest(schema=name):
                self.assertEqual(self._walk(to_gemini_schema(schema)), [])

    def test_nullable_union_becomes_flag(self):
        from optera.providers.gemini import to_gemini_schema

        out = to_gemini_schema({"type": ["string", "null"], "description": "d"})
        self.assertEqual(out, {"type": "STRING", "nullable": True, "description": "d"})

    def test_null_in_enum_becomes_nullable(self):
        from optera.providers.gemini import to_gemini_schema

        out = to_gemini_schema({"type": ["string", "null"], "enum": ["a", "b", None]})
        self.assertTrue(out["nullable"])
        self.assertEqual(out["enum"], ["a", "b"])

    def test_tile_billing_is_a_step_function(self):
        """1536px costs 4 tiles where 1568px costs 6 — the whole reason the
        Gemini image budgets are tile-aligned."""
        self.assertEqual(image_tokens(1176, 1568, "gemini-2.5-flash"), 1548)
        self.assertEqual(image_tokens(1152, 1536, "gemini-2.5-flash"), 1032)
        # And a router thumbnail is one tile anywhere up to 768px.
        self.assertEqual(image_tokens(336, 448, "gemini-2.5-flash-lite"),
                         image_tokens(576, 768, "gemini-2.5-flash-lite"))

    def test_model_set_switch_rebinds_every_stage(self):
        import optera.baseline as bl
        import optera.config as cfg
        import optera.extract as ex
        import optera.router as rt

        saved = (cfg.BASELINE_MODEL, cfg.ROUTER_MODEL, cfg.CHEAP_MODEL, cfg.ESCALATION_MODEL)
        try:
            cfg.use_model_set("gemini")
            self.assertTrue(bl.BASELINE_MODEL.startswith("gemini"))
            self.assertTrue(rt.ROUTER_MODEL.startswith("gemini"))
            self.assertTrue(ex.CHEAP_MODEL.startswith("gemini"))
            self.assertEqual(ex.EXTRACT_EDGE_PX["work_report"], 1536)
        finally:
            cfg.use_model_set("anthropic")
            self.assertEqual(
                (cfg.BASELINE_MODEL, cfg.ROUTER_MODEL, cfg.CHEAP_MODEL, cfg.ESCALATION_MODEL),
                saved)


class TestDedup(unittest.TestCase):
    def test_known_duplicate_pair_is_detected(self):
        """08 and 13 are the same page photographed twice."""
        a, b = IMAGES / "optera_doc_08.jpg", IMAGES / "optera_doc_13.jpg"
        if not (a.exists() and b.exists()):
            self.skipTest("starter images not present")
        self.assertLessEqual(hamming(prepare(a, 448).phash, prepare(b, 448).phash), 3)

    def test_distinct_documents_are_not_confused(self):
        a, b = IMAGES / "optera_doc_01.jpg", IMAGES / "optera_doc_30.jpg"
        if not (a.exists() and b.exists()):
            self.skipTest("starter images not present")
        self.assertGreater(hamming(prepare(a, 448).phash, prepare(b, 448).phash), 8)

    def test_cache_roundtrip(self):
        cache = ResultCache(pathlib.Path(".optera_cache/test_isolated/test.json"))
        cache.clear()
        cache.put("abc", "0000000000000000", {"doc_id": "d1"})
        hit, how = cache.get("abc", None)
        self.assertEqual((hit["doc_id"], how), ("d1", "exact"))
        hit, how = cache.get("zzz", "0000000000000001")  # 1 bit away
        self.assertEqual((hit["doc_id"], how), ("d1", "near"))
        self.assertIsNone(cache.get("zzz", "ffffffffffffffff")[0])
        cache.clear()


class TestValidators(unittest.TestCase):
    def test_invoice_arithmetic_mismatch_is_caught(self):
        bad = {
            "doc_class": "vendor_invoice", "vendor_name": "X", "confidence": 0.95,
            "line_items": [{"description": "part", "amount": 100.0}],
            "subtotal": 100.0, "cgst": 9.0, "sgst": 9.0, "total": 999.0,
        }
        self.assertTrue(any("total reads" in r for r in validate("vendor_invoice", bad)))

    def test_consistent_invoice_passes(self):
        good = {
            "doc_class": "vendor_invoice", "vendor_name": "M K MOTORS", "confidence": 0.93,
            "line_items": [{"description": "HOSE KIT", "amount": 544.92}],
            "subtotal": 544.92, "cgst": 49.04, "sgst": 49.04, "total": 643.00,
        }
        self.assertEqual(validate("vendor_invoice", good), [])

    def test_delivery_challan_without_total_is_allowed(self):
        """A challan has blank amount columns; requiring a total would escalate
        a correct extraction forever (optera_doc_47)."""
        challan = {
            "doc_class": "vendor_invoice", "vendor_name": "ANUPAM ENTERPRISE",
            "document_kind": "delivery_challan", "confidence": 0.9,
            "line_items": [{"description": "EXIDE XP-1000"}], "total": None,
        }
        self.assertEqual(validate("vendor_invoice", challan), [])

    def test_dispenser_cross_check(self):
        """43.24 L x Rs 74.00 = Rs 3199.76 exactly (optera_doc_41)."""
        good = {"doc_class": "meter_reading", "reading_kind": "def_dispenser", "confidence": 0.95,
                "litres": 43.24, "rate_per_litre": 74.00, "amount_inr": 3199.76}
        self.assertEqual(validate("meter_reading", good), [])
        bad = dict(good, amount_inr=5000.0)
        self.assertTrue(any("litres x rate" in r for r in validate("meter_reading", bad)))

    def test_swapped_odometer_and_trip_is_caught(self):
        swapped = {"doc_class": "meter_reading", "reading_kind": "odometer",
                   "confidence": 0.9, "odometer_km": 38.6, "trip_km": 37057.9}
        self.assertTrue(any("swapped" in r for r in validate("meter_reading", swapped)))

    def test_hallucinated_structure_on_object_photo_is_caught(self):
        """The exact failure the brief warns about: an invoice row invented from
        a battery photo."""
        leaked = {"doc_class": "not_a_document", "subject": "battery",
                  "reason": "photo of a battery", "confidence": 0.9,
                  "line_items": [{"description": "EXIDE", "amount": 5000}]}
        self.assertTrue(any("hallucinated structure" in r for r in validate("not_a_document", leaked)))

    def test_low_confidence_triggers_escalation(self):
        low = {"doc_class": "meter_reading", "reading_kind": "odometer",
               "odometer_km": 100000, "confidence": 0.2}
        self.assertTrue(any("below floor" in r for r in validate("meter_reading", low)))


class TestCosting(unittest.TestCase):
    def test_batch_discount_halves_cost(self):
        u = Usage(input_tokens=10_000, output_tokens=1_000)
        self.assertAlmostEqual(cost_usd(u, "claude-opus-5", True),
                               cost_usd(u, "claude-opus-5", False) / 2, places=9)

    def test_cheap_tier_is_five_times_cheaper_than_opus(self):
        u = Usage(input_tokens=10_000, output_tokens=1_000)
        self.assertAlmostEqual(
            cost_usd(u, "claude-opus-5") / cost_usd(u, "claude-haiku-4-5"), 5.0, places=6)

    def test_ledger_totals_match_row_sum(self):
        led = Ledger(pathlib.Path(".optera_cache/test_ledger.jsonl"))
        led.reset()
        led.record("optimized", "d1", "extract", "claude-haiku-4-5", Usage(1000, 100))
        led.record("optimized", "d2", "extract", "claude-haiku-4-5", Usage(2000, 200))
        self.assertAlmostEqual(led.total_cost("optimized"), sum(e.cost_usd for e in led.entries))
        self.assertEqual(led.total_usage("optimized").input_tokens, 3000)
        led.reset()


class TestEndToEnd(unittest.TestCase):
    def test_offline_pipeline_routes_every_document(self):
        from optera.evaluate import load_ground_truth, score_routing
        from optera.pipeline import run_pipeline
        from optera.providers.offline import OfflineProvider

        if not IMAGES.exists():
            self.skipTest("starter images not present")
        paths = sorted(IMAGES.glob("optera_doc_*.jpg"))
        led = Ledger(pathlib.Path(".optera_cache/test_e2e.jsonl"))
        led.reset()

        # Use an isolated cache directory. Pointing the tests at the real one
        # would let `python -m unittest` wipe the cache out from under a live
        # run happening in another terminal.
        import optera.pipeline as pl

        saved_cache_dir, saved_out = pl.CACHE_DIR, pl.OUT_DIR
        pl.CACHE_DIR = ".optera_cache/test_isolated"
        pl.OUT_DIR = "out/test_isolated"
        try:
            results, stats = run_pipeline(OfflineProvider(), led, paths, batched=True)
        finally:
            pl.CACHE_DIR, pl.OUT_DIR = saved_cache_dir, saved_out

        self.assertEqual(len(results), len(paths), "every input must produce exactly one record")
        routing_gt, _ = load_ground_truth()
        score = score_routing(results, routing_gt)
        self.assertEqual(score["hallucinated_structure_on_non_documents"], 0)
        # The corrupt file must have cost nothing.
        self.assertEqual(results["optera_doc_33"]["doc_class"], "unreadable")
        self.assertGreaterEqual(stats.prefiltered, 1)
        led.reset()


if __name__ == "__main__":
    unittest.main()
