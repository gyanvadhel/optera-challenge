#!/usr/bin/env python3
"""Document extraction pipeline — one command.

    python run.py                 # baseline + optimized + eval + report
    python run.py --optimized     # skip the baseline (it is the expensive one)
    python run.py --offline       # force the simulator even if a key is set
    python run.py --limit 12      # first N images, for a quick smoke test

With no ANTHROPIC_API_KEY set it runs the deterministic offline simulator so a
clean checkout works. With a key set it makes real calls and every number
becomes a measurement.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from optera import baseline as baseline_mod
from optera import evaluate as eval_mod
from optera import report as report_mod
from optera.cache import ResultCache
from optera import config as cfg
from optera.config import CACHE_DIR, IMAGE_DIR, OUT_DIR
from optera.ledger import Ledger
from optera.pipeline import run_pipeline
from optera.providers import detect_provider, get_provider, has_credentials


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and choke on the arrows and rupee
    signs in the report. The file on disk is always UTF-8 regardless."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def discover(image_dir: str, limit: int | None) -> list[pathlib.Path]:
    root = pathlib.Path(image_dir)
    if not root.exists():
        sys.exit(f"error: image directory {root} does not exist")
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        sys.exit(f"error: no images found in {root}")
    return paths[:limit] if limit else paths


def main() -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description="Document extraction pipeline")
    ap.add_argument("--images", default=IMAGE_DIR, help="directory of input images")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N images")
    ap.add_argument("--offline", action="store_true", help="force the deterministic simulator")
    ap.add_argument("--provider", default="auto", choices=["auto", "anthropic", "gemini", "offline"],
                    help="which model vendor to use. 'auto' picks whichever key is set "
                         "(GEMINI_API_KEY wins), else the offline simulator.")
    ap.add_argument("--optimized", action="store_true", help="run only the optimized pipeline")
    ap.add_argument("--baseline", action="store_true", help="run only the naive baseline")
    ap.add_argument("--no-batch", action="store_true", help="do not price the optimized run through the Batches API")
    ap.add_argument("--no-cache", action="store_true", help="disable the dedup result cache entirely")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do NOT clear the result cache first. Off by default: the headline "
                         "number must be a cold-start measurement, or re-running the same corpus "
                         "silently reports a warm cache as if it were the real cost.")
    ap.add_argument("--replay", action="store_true",
                    help="after the cold run, run the optimized pipeline a second time against the "
                         "warm cache and report the re-send cost separately")
    ap.add_argument("--no-escalation", action="store_true", help="ablation: never escalate to the big model")
    ap.add_argument("--ablate", action="store_true",
                    help="run the lever-by-lever ablation table (re-runs the corpus once per rung)")
    ap.add_argument("--error-rate", type=float, default=0.18, help="offline simulator defect injection rate")
    args = ap.parse_args()

    run_baseline = not args.optimized
    run_optimized = not args.baseline

    paths = discover(args.images, args.limit)

    chosen = "offline" if args.offline else (
        detect_provider() if args.provider == "auto" else args.provider)
    cfg.use_model_set(chosen if chosen != "offline" else detect_provider())
    provider = get_provider(force_offline=args.offline, error_rate=args.error_rate,
                            provider=args.provider)
    live = getattr(provider, "live", False)

    print(f"Document extraction pipeline")
    print(f"  images   : {len(paths)} from {args.images}")
    print(f"  provider : {provider.name} ({'LIVE' if live else 'offline simulator, $0'})")
    print(f"  models   : baseline={cfg.BASELINE_MODEL}  router={cfg.ROUTER_MODEL}")
    print(f"             cheap={cfg.CHEAP_MODEL}  escalation={cfg.ESCALATION_MODEL}")
    if provider.name == "gemini":
        print(f"  note     : AI Studio free tier is rate-limited; a full run is paced")
        print(f"             and takes a few minutes. Costs are priced at PAID-tier")
        print(f"             rates so the comparison is meaningful (the run is free).")
    if not live and has_credentials() and not args.offline:
        print("  note     : credentials found but offline forced")
    print()

    ledger = Ledger(pathlib.Path(OUT_DIR) / "ledger.jsonl")
    ledger.reset()

    # A benchmark that reuses a cache warmed by a previous run is not a
    # benchmark. Start cold unless explicitly told otherwise.
    if not args.keep_cache:
        ResultCache(pathlib.Path(CACHE_DIR) / "results.json").clear()

    baseline_results: dict = {}
    optimized_results: dict = {}
    stats = None

    if run_baseline:
        t0 = time.time()
        print(f"[1/3] baseline — one big-model call per image ...")
        baseline_results = baseline_mod.run_baseline(provider, ledger, paths)
        print(f"      {len(baseline_results)} docs, {ledger.total_cost('baseline'):.4f} USD, {time.time() - t0:.1f}s\n")

    if run_optimized:
        t0 = time.time()
        print(f"[2/3] optimized — prefilter -> route -> extract -> escalate ...")
        optimized_results, stats = run_pipeline(
            provider, ledger, paths,
            batched=not args.no_batch,
            use_cache=not args.no_cache,
            allow_escalation=not args.no_escalation,
        )
        print(f"      {len(optimized_results)} docs, {ledger.total_cost('optimized'):.4f} USD, {time.time() - t0:.1f}s\n")

    replay = None
    if args.replay and run_optimized:
        print("[2b/3] replay — same corpus a second time, warm cache ...")
        replay_ledger = Ledger(pathlib.Path(OUT_DIR) / "ledger_replay.jsonl")
        replay_ledger.reset()
        _, replay_stats = run_pipeline(
            provider, replay_ledger, paths,
            batched=not args.no_batch, use_cache=True,
            allow_escalation=not args.no_escalation,
        )
        replay = {
            "cost": replay_ledger.total_cost("optimized"),
            "per_doc": replay_ledger.total_cost("optimized") / max(1, len(paths)),
            "cache_exact": replay_stats.cache_exact,
            "cache_near": replay_stats.cache_near,
            "docs": len(paths),
        }
        print(f"      {replay['cache_exact']} exact + {replay['cache_near']} near cache hits, "
              f"{replay['cost']:.4f} USD\n")

    ablation_rows = None
    if args.ablate:
        from optera.ablate import run_ablation
        print("[2c/3] ablation — one run per lever ...")
        ablation_rows = run_ablation(provider, paths, OUT_DIR)
        print()

    print("[3/3] evaluating ...")
    evaluation = eval_mod.evaluate(baseline_results or None, optimized_results or None)
    text = report_mod.build_report(
        ledger, evaluation, stats, n_docs=len(paths), batched=not args.no_batch, replay=replay
    )
    if ablation_rows:
        from optera.ablate import render
        text += render(ablation_rows)
    path = report_mod.write_report(text, evaluation)

    print()
    print(text)
    print(f"Written: {path}")
    print(f"         {OUT_DIR}/ledger.jsonl      (every call, every token)")
    print(f"         {OUT_DIR}/evaluation.json   (per-field scoring detail)")
    print(f"         {OUT_DIR}/optimized/*.json  (the structured output)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
