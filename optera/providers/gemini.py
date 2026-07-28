"""Google Gemini provider (AI Studio).

Chosen because AI Studio issues a free API key with a usable vision quota, so
the pipeline can be run live at zero spend.

Two things here are not just plumbing:

1. **Gemini's `response_schema` is not JSON Schema.** It is an OpenAPI-3.0
   subset: no `additionalProperties`, and nullability is a `nullable: true`
   flag rather than a `["string", "null"]` type union. `to_gemini_schema()`
   translates our canonical schemas across. Without it every structured call
   400s.

2. **The free tier is rate-limited per minute.** A full run is ~60 calls, which
   will trip the limit unless calls are paced. There is a token-bucket pacer and
   429 backoff built in, so a free-tier run completes rather than half-failing.

Cost note: on the free tier the run genuinely costs $0, which would make the
cost comparison meaningless. So the ledger prices every call at Google's
*paid-tier* published rates. The report therefore answers "what would this cost
at scale", which is the question that matters, while the run itself is free.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time

from ..imaging import Prepared
from ..ledger import Usage
from .base import Completion

# Free-tier requests per minute. Deliberately conservative — being throttled
# mid-run is far more annoying than being 30 seconds slower.
DEFAULT_RPM = int(os.getenv("OPTERA_GEMINI_RPM", "10"))
MAX_RETRIES = 5


# --------------------------------------------------------------------------
# Schema translation
# --------------------------------------------------------------------------

_ALLOWED = {"type", "format", "description", "nullable", "enum", "items",
            "properties", "required", "propertyOrdering"}


def to_gemini_schema(schema: dict) -> dict:
    """Translate a standard JSON Schema into Gemini's OpenAPI-subset dialect."""
    if not isinstance(schema, dict):
        return schema

    out: dict = {}
    t = schema.get("type")

    # ["string", "null"] -> type string + nullable
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        out["type"] = (non_null[0] if non_null else "string").upper()
        if len(non_null) < len(t):
            out["nullable"] = True
    elif isinstance(t, str):
        out["type"] = t.upper()

    if "description" in schema:
        out["description"] = schema["description"]

    if "enum" in schema:
        values = [v for v in schema["enum"] if v is not None]
        if len(values) < len(schema["enum"]):
            out["nullable"] = True
        # Gemini only supports string enums.
        out["enum"] = [str(v) for v in values]
        out["type"] = "STRING"

    if "properties" in schema:
        out["properties"] = {k: to_gemini_schema(v) for k, v in schema["properties"].items()}
        # Stabilises field order in the response, which keeps output tokens and
        # diffs predictable across runs.
        out["propertyOrdering"] = list(schema["properties"].keys())
        out.setdefault("type", "OBJECT")

    if "items" in schema:
        out["items"] = to_gemini_schema(schema["items"])
        out.setdefault("type", "ARRAY")

    if "required" in schema:
        out["required"] = list(schema["required"])

    # additionalProperties is silently unsupported and 400s if sent.
    return {k: v for k, v in out.items() if k in _ALLOWED}


# --------------------------------------------------------------------------
# Rate pacing
# --------------------------------------------------------------------------


class _Pacer:
    """Minimum-interval pacer, so a free-tier key does not 429 on every call."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


# --------------------------------------------------------------------------


class GeminiProvider:
    name = "gemini"
    live = True

    def __init__(self, api_key: str | None = None, rpm: int = DEFAULT_RPM):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The google-genai SDK is not installed. Run:\n"
                "    pip install google-genai\n"
                "or run with --offline to use the deterministic simulator."
            ) from exc
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key. Get a free one at https://aistudio.google.com/apikey "
                "then:  set GEMINI_API_KEY=..."
            )
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.pacer = _Pacer(rpm)

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
        import base64

        from google.genai import types

        parts: list = []
        for img in images:
            if len(images) > 1:
                parts.append(types.Part.from_text(text=f"[{img.doc_id}]"))
            parts.append(
                types.Part.from_bytes(
                    data=base64.standard_b64decode(img.b64),
                    mime_type=img.media_type,
                )
            )
        parts.append(types.Part.from_text(text=instruction))

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=to_gemini_schema(schema),
            max_output_tokens=max_tokens,
            temperature=0.0,
        )

        resp = self._call_with_retry(model, parts, config)

        text = (getattr(resp, "text", None) or "{}").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"_parse_error": True, "_raw": text[:2000]}

        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            cache_read_input_tokens=getattr(um, "cached_content_token_count", 0) or 0,
        )
        # Gemini reports cached tokens inside the prompt count; the ledger prices
        # them separately, so avoid billing them twice.
        usage.input_tokens = max(0, usage.input_tokens - usage.cache_read_input_tokens)

        return Completion(data=data, usage=usage, live=True, model=model, raw_text=text)

    def _call_with_retry(self, model: str, parts: list, config):
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self.pacer.wait()
            try:
                return self.client.models.generate_content(
                    model=model, contents=parts, config=config
                )
            except Exception as exc:  # SDK raises a variety of transport errors
                last = exc
                msg = str(exc).lower()

                # A PER-DAY quota is not transient. Retrying it just burns the
                # remaining allowance and then fails anyway — which is exactly
                # what happened on the first live run, where gemini-3.6-flash
                # turned out to allow 20 requests *per day* on the free tier.
                # Fail fast with an actionable message instead.
                if _is_daily_quota(msg):
                    raise RuntimeError(
                        f"Daily free-tier quota exhausted for {model}. This is a per-day "
                        f"cap, not a per-minute one — waiting will not help today.\n"
                        f"  Pick a model with a larger daily allowance, e.g.:\n"
                        f"    set OPTERA_GEMINI_BASELINE=gemini-3.5-flash\n"
                        f"    set OPTERA_GEMINI_ESCALATION=gemini-3.5-flash\n"
                        f"  or run with --offline / --limit N."
                    ) from exc

                retryable = any(s in msg for s in ("429", "resource_exhausted", "quota",
                                                   "503", "unavailable", "500", "internal",
                                                   "deadline", "timeout"))
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise
                backoff = min(60.0, (2 ** attempt) * 2.0) + random.uniform(0, 1.5)
                print(f"      rate limited, retrying in {backoff:.0f}s "
                      f"({attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
        raise last  # pragma: no cover


def _is_daily_quota(msg: str) -> bool:
    """Distinguish a per-day cap from a per-minute one. Google signals the
    former with a PerDay quota id; the wording of the human message varies, so
    match the machine-readable field first."""
    return ("perday" in msg.replace("_", "").replace("-", "")
            or "requests per day" in msg
            or "generate_content_free_tier_requests" in msg)
