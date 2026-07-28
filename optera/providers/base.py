"""Provider interface.

One method, `complete`, which takes a system prompt, a list of images, a user
instruction and a JSON schema, and returns parsed JSON plus a Usage record.
Adding a provider means implementing this and registering it in __init__.py —
nothing else in the pipeline knows which model vendor is behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..imaging import Prepared
from ..ledger import Usage


@dataclass
class Completion:
    data: dict
    usage: Usage
    live: bool
    model: str
    raw_text: str = ""


class Provider(Protocol):
    name: str
    live: bool

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
        ...
