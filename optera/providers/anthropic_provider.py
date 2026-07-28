"""Anthropic Claude provider.

Uses structured outputs (`output_config.format`) so the model is constrained to
the schema instead of being asked politely for JSON, and prompt caching on the
system block when the prompt is long enough to clear the model's minimum
cacheable prefix (see config.CACHE_MIN_TOKENS — this matters a lot on Haiku).
"""
from __future__ import annotations

import json

from ..config import CACHE_MIN_TOKENS
from ..imaging import Prepared, estimate_text_tokens
from ..ledger import Usage
from .base import Completion


class AnthropicProvider:
    name = "anthropic"
    live = True

    def __init__(self, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The anthropic SDK is not installed. Run `pip install -r requirements.txt`, "
                "or run with --offline to use the deterministic simulator."
            ) from exc
        self._sdk = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

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
        content: list[dict] = []
        for img in images:
            if len(images) > 1:
                # Batched router call: label each image so the model can key its
                # answers back to a document.
                content.append({"type": "text", "text": f"[{img.doc_id}]"})
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": img.media_type, "data": img.b64},
                }
            )
        content.append({"type": "text", "text": instruction})

        # Only mark the system block cacheable when it can actually clear the
        # model's minimum prefix; below it the API silently does not cache and
        # we would just be adding a no-op field.
        system_block: list[dict] | str
        if cache_system and estimate_text_tokens(system) >= CACHE_MIN_TOKENS.get(model, 1024):
            system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            system_block = system

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_block,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        # Haiku 4.5 predates the effort parameter; the newer tiers accept it and
        # `low` is right for mechanical extraction.
        if model.startswith(("claude-opus-5", "claude-sonnet-5")):
            kwargs["output_config"]["effort"] = "low"

        resp = self.client.messages.create(**kwargs)

        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"_parse_error": True, "_raw": text[:2000]}

        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )
        return Completion(data=data, usage=usage, live=True, model=model, raw_text=text)
