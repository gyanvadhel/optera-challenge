"""Render the numbers. Everything here is derived from the ledger and the
evaluator — no figure is typed in by hand.
"""
from __future__ import annotations

import json
import pathlib

from .config import BATCH_DISCOUNT, OUT_DIR
from .ledger import Ledger


def _fmt_usd(x: float) -> str:
    return f"${x:,.6f}" if x < 0.01 else f"${x:,.4f}"


def build_report(ledger: Ledger, evaluation: dict, stats, *, n_docs: int, batched: bool,
                 replay: dict | None = None) -> str:
    base_cost = ledger.total_cost("baseline")
    opt_cost = ledger.total_cost("optimized")
    base_live = ledger.is_live("baseline")
    opt_live = ledger.is_live("optimized")

    base_docs = len(ledger.docs_seen("baseline")) or n_docs
    opt_docs = len(ledger.docs_seen("optimized")) or n_docs

    base_per = base_cost / base_docs if base_docs else 0.0
    opt_per = opt_cost / opt_docs if opt_docs else 0.0
    factor = base_per / opt_per if opt_per > 0 else float("inf")

    L: list[str] = []
    add = L.append

    mode = "MEASURED (live API)" if (base_live and opt_live) else "SIMULATED (no API key)"
    add(f"# Document extraction pipeline — cost & accuracy report\n")
    add(f"**Mode: {mode}**\n")
    if mode.startswith("SIMULATED"):
        add(
            "> Token counts come from the real image-tile math and a text-token "
            "estimate, so the *cost* comparison reflects the request shapes the "
            "pipeline actually builds. **Accuracy in this mode is not a "
            "measurement** — the offline provider synthesises answers from the "
            "committed ground truth with seeded corruption injected so the "
            "validators and escalation path are exercised. Export "
            "`ANTHROPIC_API_KEY` and re-run to turn both into measurements.\n"
        )

    # ---- headline ----
    add("## Cost per document\n")
    add("| | Baseline (1x) | Optimized | Change |")
    add("|---|---:|---:|---:|")
    add(f"| Model(s) | opus-5, full res, 1 call/doc | haiku-4.5 routed, opus-5 on escalation | |")
    add(f"| Documents | {base_docs} | {opt_docs} | |")
    add(f"| Total cost | {_fmt_usd(base_cost)} | {_fmt_usd(opt_cost)} | **-{(1 - opt_cost / base_cost) * 100:.1f}%** |" if base_cost else "| Total cost | n/a | n/a | |")
    add(f"| **Cost per doc** | **{_fmt_usd(base_per)}** | **{_fmt_usd(opt_per)}** | **{factor:.1f}x cheaper** |")
    add(f"| Cost per 1,000 docs | ${base_per * 1000:,.2f} | ${opt_per * 1000:,.2f} | |")
    if batched:
        add(f"\n_Optimized run priced through the Message Batches API ({BATCH_DISCOUNT:.0%} discount). "
            f"Baseline is priced at standard rates — batching the baseline too would make it "
            f"{_fmt_usd(base_per * (1 - BATCH_DISCOUNT))}/doc, still {base_per * (1 - BATCH_DISCOUNT) / opt_per:.1f}x the optimized cost._")

    add("\n_This is a **cold-start** run: the result cache is cleared before the "
        "optimized pipeline begins, so no document benefits from a previous run._")

    if replay:
        add("\n### Re-sent corpus (warm cache)\n")
        add("Operations inboxes redeliver the same photo constantly. Running the "
            "identical corpus a second time:\n")
        add(f"- Cache hits: **{replay['cache_exact']} exact + {replay['cache_near']} near-duplicate** "
            f"of {replay['docs']} documents")
        add(f"- Cost of the second pass: **{_fmt_usd(replay['cost'])}** "
            f"({_fmt_usd(replay['per_doc'])}/doc)")
        if opt_per > 0 and replay["per_doc"] > 0:
            add(f"- Marginal cost of a re-send vs a first send: **{replay['per_doc'] / opt_per:.1%}**")
        elif replay["cost"] == 0:
            add("- Marginal cost of a re-send: **zero**")

    # ---- token totals ----
    bu, ou = ledger.total_usage("baseline"), ledger.total_usage("optimized")
    add("\n## Tokens\n")
    add("| | Baseline | Optimized |")
    add("|---|---:|---:|")
    add(f"| Input tokens | {bu.input_tokens:,} | {ou.input_tokens:,} |")
    add(f"| Output tokens | {bu.output_tokens:,} | {ou.output_tokens:,} |")
    add(f"| Cache writes | {bu.cache_creation_input_tokens:,} | {ou.cache_creation_input_tokens:,} |")
    add(f"| Cache reads | {bu.cache_read_input_tokens:,} | {ou.cache_read_input_tokens:,} |")

    # ---- where the money goes ----
    add("\n## Optimized run — cost by stage\n")
    add("| Stage | Calls | Input tok | Output tok | Cost | % of run |")
    add("|---|---:|---:|---:|---:|---:|")
    for stage, row in sorted(ledger.by_stage("optimized").items(), key=lambda kv: -kv[1]["cost_usd"]):
        pct = row["cost_usd"] / opt_cost * 100 if opt_cost else 0.0
        add(f"| {stage} | {row['calls']} | {row['input']:,} | {row['output']:,} | {_fmt_usd(row['cost_usd'])} | {pct:.1f}% |")

    # ---- pipeline behaviour ----
    if stats is not None:
        add("\n## Routing behaviour\n")
        add(f"- Documents in: **{stats.total}**")
        add(f"- Rejected free by the local prefilter (not an image): **{stats.prefiltered}**")
        add(f"- Served from cache — exact dup **{stats.cache_exact}**, near dup **{stats.cache_near}** (zero model cost)")
        add(f"- Re-routed on the big model for low router confidence: **{stats.rerouted}**")
        add(f"- Extracted as real documents: **{stats.extracted}**")
        add(f"- Refused (not a document / unreadable): **{stats.rejected}**")
        add(f"- Escalated to {'the escalation tier'}: **{stats.escalated}**")
        add(f"- Still failing validation after escalation: **{stats.still_invalid}**")
        if stats.by_class:
            add("\nBy class: " + ", ".join(f"`{k}` {v}" for k, v in sorted(stats.by_class.items())))

    # ---- accuracy ----
    add("\n## Accuracy\n")
    for run in ("baseline", "optimized"):
        ev = evaluation.get(run)
        if not ev:
            continue
        r, f = ev["routing"], ev["fields"]
        add(f"### {run}\n")
        add(f"- **Routing**: {r['correct']}/{r['total']} = **{r['accuracy']:.1%}** "
            f"(hand-labelled, all images)")
        add(f"- **Fields**: {f['correct']}/{f['fields_scored']} = **{f['accuracy']:.1%}** "
            f"across {f['documents_with_ground_truth']} hand-transcribed documents; "
            f"precision when a value was emitted **{f['precision_when_answered']:.1%}**")
        add(f"- Invented structure on a non-document: **{r['hallucinated_structure_on_non_documents']}**")
        if r["errors"]:
            add(f"- Routing errors: " + ", ".join(f"`{e['doc_id']}` {e['expected']}→{e['got']}" for e in r["errors"][:8]))
        weak = [k for k, v in f["per_field"].items() if v["accuracy"] < 0.8 and (v["correct"] + v["wrong"] + v["missing"]) >= 3]
        if weak:
            add(f"- Weakest fields (<80%): " + ", ".join(f"`{k}` {f['per_field'][k]['accuracy']:.0%}" for k in weak))
        add("")

    ag = evaluation.get("agreement")
    if ag:
        add("### Baseline vs optimized agreement\n")
        add("The check that supports \"cheaper without dropping accuracy\" on the "
            "documents nobody hand-transcribed.\n")
        add(f"- Class agreement: **{ag['class_agreement']:.1%}** over {ag['documents_compared']} documents")
        add(f"- Field agreement: **{ag['field_agreement']:.1%}** over {ag['fields_compared']} compared fields")
        if ag["divergences"]:
            add(f"- {len(ag['divergences'])} divergence(s); first few:")
            for d in ag["divergences"][:6]:
                add(f"  - `{d['doc_id']}` `{d['field']}`: baseline `{d['baseline']}` vs optimized `{d['optimized']}`")

    add(f"\n---\n_Every figure above is computed from `{OUT_DIR}/ledger.jsonl` "
        f"({len(ledger.entries)} rows) and `{OUT_DIR}/evaluation.json`. Nothing is hand-entered._")
    return "\n".join(L) + "\n"


def write_report(text: str, evaluation: dict) -> pathlib.Path:
    out = pathlib.Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    path = out / "report.md"
    path.write_text(text, encoding="utf-8")
    return path
