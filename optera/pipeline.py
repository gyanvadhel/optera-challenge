"""The optimized pipeline.

Five stages, cheapest first, each one able to end the document's journey:

  0. prefilter   free   magic-byte check, truncation repair, dedup by hash
  1. route       ~$0.0002/doc   batched thumbnails on the cheapest model
  1b. re-route   rare   single image on the big model when the router is unsure
  2. reject      free   not_a_document / unreadable stop here
  3. extract     cheap  per-class prompt, per-class resolution, cheap model
  4. escalate    rare   big model, bigger image, told what failed validation

The ordering is the whole design: work is done at the lowest tier that can
answer, and every stage has a free, deterministic way to detect that it was the
wrong tier.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .cache import NEAR_DUP_MAX_DISTANCE, ResultCache
from .config import CACHE_DIR, OUT_DIR, ROUTER_BATCH_SIZE, ROUTER_EDGE_PX
from .extract import extract_document
from .imaging import hamming, prepare
from .ledger import Ledger, Usage
from .router import needs_second_opinion, reroute_one, route_batch
from .schemas import envelope


@dataclass
class PipelineStats:
    total: int = 0
    prefiltered: int = 0
    cache_exact: int = 0
    cache_near: int = 0
    rerouted: int = 0
    rejected: int = 0
    extracted: int = 0
    escalated: int = 0
    still_invalid: int = 0
    by_class: dict = field(default_factory=dict)


def run_pipeline(
    provider,
    ledger: Ledger,
    doc_paths: list[pathlib.Path],
    *,
    batched: bool = False,
    use_cache: bool = True,
    allow_escalation: bool = True,
) -> tuple[dict[str, dict], PipelineStats]:
    out_dir = pathlib.Path(OUT_DIR) / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(pathlib.Path(CACHE_DIR) / "results.json")
    if not use_cache:
        cache.clear()

    stats = PipelineStats(total=len(doc_paths))
    results: dict[str, dict] = {}

    def emit(record: dict) -> None:
        results[record["doc_id"]] = record
        (out_dir / f"{record['doc_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stats.by_class[record["doc_class"]] = stats.by_class.get(record["doc_class"], 0) + 1

    # ---- stage 0: free local prefilter + dedup ------------------------
    # Deduplication works two ways and both matter:
    #   * against the persistent cache, for documents seen on a previous run;
    #   * within this run, so a page sent twice in the same batch is only
    #     extracted once. The starter set contains two such pairs.
    pending: list[tuple[pathlib.Path, object]] = []
    seen_in_run: list[tuple[str, str, str]] = []  # (phash, content_hash, doc_id)
    followers: dict[str, list[str]] = {}  # representative doc_id -> duplicates

    for path in doc_paths:
        thumb = prepare(path, ROUTER_EDGE_PX)

        if not thumb.ok:
            stats.prefiltered += 1
            ledger.record("optimized", thumb.doc_id, "prefilter", "none", Usage(),
                          live=provider.live, note=thumb.reason or "")
            emit(envelope(thumb.doc_id, "unreadable", {"reason": thumb.reason},
                          stage="prefilter", cost_usd=0.0))
            continue

        if use_cache:
            hit, how = cache.get(thumb.content_hash, thumb.phash)
            if hit is not None:
                stats.cache_exact += how == "exact"
                stats.cache_near += how == "near"
                ledger.record("optimized", thumb.doc_id, "cache_hit", "none", Usage(),
                              live=provider.live, note=f"{how} duplicate of {hit.get('doc_id')}")
                record = dict(hit)
                record["doc_id"] = thumb.doc_id
                record["meta"] = dict(record.get("meta") or {}) | {
                    "cache": how, "reused_from": hit.get("doc_id"), "cost_usd": 0.0
                }
                emit(record)
                continue

            twin = _twin_in_run(thumb, seen_in_run)
            if twin is not None:
                stats.cache_near += 1
                ledger.record("optimized", thumb.doc_id, "cache_hit", "none", Usage(),
                              live=provider.live, note=f"in-run duplicate of {twin}")
                followers.setdefault(twin, []).append(thumb.doc_id)
                continue
            seen_in_run.append((thumb.phash, thumb.content_hash, thumb.doc_id))

        pending.append((path, thumb))

    # ---- stage 1: batched classification -------------------------------
    classified: dict[str, dict] = {}
    for i in range(0, len(pending), ROUTER_BATCH_SIZE):
        chunk = pending[i : i + ROUTER_BATCH_SIZE]
        classified |= route_batch(provider, ledger, [t for _, t in chunk], batched=batched)

    # ---- stage 1b: second opinion on low-confidence classifications ----
    for path, thumb in pending:
        row = classified.get(thumb.doc_id, {})
        if needs_second_opinion(row):
            stats.rerouted += 1
            classified[thumb.doc_id] = reroute_one(provider, ledger, thumb, batched=batched)

    # ---- stages 2-4: reject or extract ---------------------------------
    for path, thumb in pending:
        row = classified.get(thumb.doc_id, {})
        cls = row.get("doc_class", "unreadable")

        if cls == "unreadable":
            stats.rejected += 1
            record = envelope(thumb.doc_id, "unreadable",
                              {"reason": "could not be classified with confidence"},
                              stage="route", router_confidence=row.get("confidence"))
            cache.put(thumb.content_hash, thumb.phash, record)
            emit(record)
            continue

        result = extract_document(
            provider, ledger, str(path), cls,
            batched=batched, allow_escalation=allow_escalation,
        )
        if result.escalated:
            stats.escalated += 1
        if result.validation:
            stats.still_invalid += 1
        if cls == "not_a_document":
            stats.rejected += 1
        else:
            stats.extracted += 1

        record = envelope(
            thumb.doc_id, cls, result.data,
            model=result.model, escalated=result.escalated,
            validation_failures=result.validation,
            router_confidence=row.get("confidence"),
            tiers=result.tiers,
        )
        # Only cache clean results — never propagate a known-bad extraction to
        # a near-duplicate.
        if not result.validation:
            cache.put(thumb.content_hash, thumb.phash, record)
        emit(record)
        _emit_followers(record, followers, emit)

    # A representative that never reached extraction (rejected at routing) still
    # owes its duplicates an answer.
    for rep, dupes in followers.items():
        if rep in results and dupes:
            _emit_followers(results[rep], followers, emit)

    if use_cache:
        cache.save()
    return results, stats


def _twin_in_run(thumb, seen: list[tuple[str, str, str]]) -> str | None:
    """Has an image indistinguishable from this one already been queued in this
    same run? Returns the representative doc_id, or None."""
    for phash, content_hash, doc_id in seen:
        if content_hash == thumb.content_hash:
            return doc_id
        if thumb.phash and hamming(phash, thumb.phash) <= NEAR_DUP_MAX_DISTANCE:
            return doc_id
    return None


def _emit_followers(record: dict, followers: dict[str, list[str]], emit) -> None:
    for dupe_id in followers.pop(record["doc_id"], []):
        clone = json.loads(json.dumps(record))
        clone["doc_id"] = dupe_id
        clone["meta"] = dict(clone.get("meta") or {}) | {
            "cache": "in_run_duplicate", "reused_from": record["doc_id"], "cost_usd": 0.0
        }
        emit(clone)
