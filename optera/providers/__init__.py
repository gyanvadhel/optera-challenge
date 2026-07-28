"""Provider registry.

`get_provider()` picks a live provider when credentials are present and falls
back to the deterministic offline simulator otherwise, so a clean checkout with
no API key still runs the whole pipeline.
"""
from __future__ import annotations

import os

from .base import Completion, Provider  # noqa: F401

ANTHROPIC_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
GEMINI_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _any(names: tuple[str, ...]) -> bool:
    return any(os.getenv(n) for n in names)


def has_credentials(provider: str = "auto") -> bool:
    if provider == "anthropic":
        return _any(ANTHROPIC_ENV)
    if provider == "gemini":
        return _any(GEMINI_ENV)
    return _any(ANTHROPIC_ENV) or _any(GEMINI_ENV)


def detect_provider() -> str:
    """Which live provider a bare `python run.py` should use."""
    if _any(GEMINI_ENV):
        return "gemini"
    if _any(ANTHROPIC_ENV):
        return "anthropic"
    return "offline"


def get_provider(force_offline: bool = False, error_rate: float = 0.18,
                 provider: str = "auto"):
    if provider == "auto":
        provider = "offline" if force_offline else detect_provider()
    elif force_offline:
        provider = "offline"

    if provider == "offline":
        from .offline import OfflineProvider

        return OfflineProvider(error_rate=error_rate)
    if provider == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    from .anthropic_provider import AnthropicProvider

    return AnthropicProvider()
