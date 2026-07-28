"""Token / cost ledger.

Every model call — baseline or optimized, live or simulated — appends one JSONL
row here. Nothing in the report is computed any other way, so the numbers in
README.md are exactly the sum of this file and can be audited line by line.
"""
from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import asdict, dataclass, field

from .config import BATCH_DISCOUNT, PRICING


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


def cost_usd(usage: Usage, model: str, batched: bool = False) -> float:
    p = PRICING[model]
    total = (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + usage.cache_creation_input_tokens * p.cache_write_5m
        + usage.cache_read_input_tokens * p.cache_read
    ) / 1_000_000
    return total * (1 - BATCH_DISCOUNT) if batched else total


@dataclass
class Entry:
    run: str  # "baseline" | "optimized"
    doc_ids: list[str]  # >1 when several docs share one batched call
    stage: str  # "prefilter" | "route" | "extract" | "escalate" | "cache_hit"
    model: str
    usage: dict
    cost_usd: float
    batched: bool
    live: bool  # False => tokens are computed, not returned by the API
    image_px: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)


class Ledger:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.entries: list[Entry] = []

    def reset(self) -> None:
        with self._lock:
            self.entries.clear()
            if self.path.exists():
                self.path.unlink()

    def record(
        self,
        run: str,
        doc_ids: list[str] | str,
        stage: str,
        model: str,
        usage: Usage,
        *,
        batched: bool = False,
        live: bool = False,
        image_px: str = "",
        note: str = "",
        **extra,
    ) -> Entry:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        entry = Entry(
            run=run,
            doc_ids=doc_ids,
            stage=stage,
            model=model,
            usage=asdict(usage),
            cost_usd=cost_usd(usage, model, batched) if model in PRICING else 0.0,
            batched=batched,
            live=live,
            image_px=image_px,
            note=note,
            extra=extra,
        )
        with self._lock:
            self.entries.append(entry)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    # -- rollups ---------------------------------------------------------

    def total_cost(self, run: str | None = None) -> float:
        return sum(e.cost_usd for e in self.entries if run is None or e.run == run)

    def total_usage(self, run: str | None = None) -> Usage:
        acc = Usage()
        for e in self.entries:
            if run is None or e.run == run:
                acc = acc + Usage(**e.usage)
        return acc

    def by_stage(self, run: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in self.entries:
            if e.run != run:
                continue
            row = out.setdefault(e.stage, {"calls": 0, "cost_usd": 0.0, "input": 0, "output": 0})
            row["calls"] += 1
            row["cost_usd"] += e.cost_usd
            row["input"] += e.usage["input_tokens"] + e.usage["cache_read_input_tokens"] + e.usage["cache_creation_input_tokens"]
            row["output"] += e.usage["output_tokens"]
        return out

    def docs_seen(self, run: str) -> set[str]:
        seen: set[str] = set()
        for e in self.entries:
            if e.run == run:
                seen.update(e.doc_ids)
        return seen

    def is_live(self, run: str) -> bool:
        rows = [e for e in self.entries if e.run == run]
        return bool(rows) and all(e.live for e in rows if e.model in PRICING)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Ledger":
        led = cls(path)
        if led.path.exists():
            for line in led.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    led.entries.append(Entry(**json.loads(line)))
        return led
