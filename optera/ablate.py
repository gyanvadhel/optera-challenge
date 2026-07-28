"""Lever-by-lever ablation.

"5x cheaper" is not a useful claim on its own — the useful question is which
decision bought which part of it, and what each one costs in accuracy. This
re-runs the corpus under progressively more optimisation and reports cost per
document and routing/field accuracy for each rung.

Runs the whole corpus once per configuration, so use it offline (free) or
against a small --limit when live.
"""
from __future__ import annotations

import pathlib

from . import config as cfg
from .baseline import run_baseline
from .evaluate import evaluate
from .ledger import Ledger
from .pipeline import run_pipeline

RUNGS = [
    ("A. Baseline: opus-5, full res, 1 call/doc", {}),
    ("B. + per-class image downscaling (still opus-5 everywhere)",
     {"CHEAP_MODEL": "claude-opus-5", "ROUTER_MODEL": "claude-opus-5", "batched": False, "escalate": False}),
    ("C. + cheap model for routing and first-pass extraction",
     {"batched": False, "escalate": False}),
    ("D. + validator-gated escalation to opus-5 (the accuracy floor)",
     {"batched": False, "escalate": True}),
    ("E. + Message Batches API (-50%)",
     {"batched": True, "escalate": True}),
]


def run_ablation(provider, doc_paths: list[pathlib.Path], out_dir: str) -> list[dict]:
    from .cache import ResultCache

    rows: list[dict] = []
    n = len(doc_paths)

    for label, knobs in RUNGS:
        ledger = Ledger(pathlib.Path(out_dir) / "ledger_ablation.jsonl")
        ledger.reset()
        ResultCache(pathlib.Path(cfg.CACHE_DIR) / "ablation.json").clear()

        saved = (cfg.CHEAP_MODEL, cfg.ROUTER_MODEL)
        try:
            if not knobs:
                results = run_baseline(provider, ledger, doc_paths)
                run_key = "baseline"
            else:
                cfg.CHEAP_MODEL = knobs.get("CHEAP_MODEL", saved[0])
                cfg.ROUTER_MODEL = knobs.get("ROUTER_MODEL", saved[1])
                # Rebind the module-level names the stages captured at import.
                import optera.extract as _ex
                import optera.router as _rt
                _ex.CHEAP_MODEL = cfg.CHEAP_MODEL
                _rt.ROUTER_MODEL = cfg.ROUTER_MODEL

                results, _ = run_pipeline(
                    provider, ledger, doc_paths,
                    batched=knobs.get("batched", False),
                    use_cache=False,  # isolate the model levers from caching
                    allow_escalation=knobs.get("escalate", True),
                )
                run_key = "optimized"
        finally:
            cfg.CHEAP_MODEL, cfg.ROUTER_MODEL = saved
            import optera.extract as _ex
            import optera.router as _rt
            _ex.CHEAP_MODEL, _rt.ROUTER_MODEL = saved

        cost = ledger.total_cost(run_key)
        ev = evaluate(results if run_key == "baseline" else None,
                      results if run_key == "optimized" else None)
        scored = ev.get(run_key, {})
        rows.append({
            "rung": label,
            "cost": cost,
            "per_doc": cost / n if n else 0.0,
            "routing_accuracy": scored.get("routing", {}).get("accuracy", 0.0),
            "field_accuracy": scored.get("fields", {}).get("accuracy", 0.0),
            "hallucinated": scored.get("routing", {}).get("hallucinated_structure_on_non_documents", 0),
        })
    return rows


def render(rows: list[dict]) -> str:
    if not rows:
        return ""
    base = rows[0]["per_doc"] or 1.0
    out = ["\n## Ablation — where the savings come from\n",
           "Each rung adds one lever to the one above it. Same 47 images, same "
           "ground truth, caching disabled throughout so the model levers are "
           "isolated. Rung E therefore reads slightly higher than the headline "
           "figure above, which also benefits from in-run deduplication.\n",
           "| Configuration | Cost/doc | vs baseline | Routing acc | Field acc | Invented rows |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        factor = base / r["per_doc"] if r["per_doc"] else float("inf")
        out.append(
            f"| {r['rung']} | ${r['per_doc']:.6f} | {factor:.1f}x | "
            f"{r['routing_accuracy']:.1%} | {r['field_accuracy']:.1%} | {r['hallucinated']} |"
        )
    return "\n".join(out) + "\n"
