"""Hosted model APIs the console offers by name (S13, D171): where each one answers and which of
the two API formats it speaks, so adding one is choosing it rather than typing a URL.

A preset is a starting point, nothing more. What is saved is an ordinary hosted host — its
`base_url` and `engine` as the operator left them — and a provider missing here is added as
"another OpenAI-format API" or "another Anthropic-format API" with its own URL. Nothing here is
a price: a provider's prices are on its own page, change without notice, and are typed by the
operator, who answers for them (D164).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from .config import ANTHROPIC_ENGINE

OPENAI_ENGINE = "openai-api"


@dataclasses.dataclass(frozen=True)
class HostedPreset:
    id: str
    name: str
    #: The provider's whole API base, ending where its `/models` and `/chat/completions` (or
    #: `/messages`) follow. Empty for "another API": the operator gives it.
    base_url: str
    engine: str
    #: Shown beside the key field, so the operator knows which key is meant.
    key_hint: str = ""
    #: A short note on what the provider serves through this format.
    note: str = ""

    @property
    def custom(self) -> bool:
        return not self.base_url

    def as_dict(self) -> dict:
        return {**dataclasses.asdict(self), "custom": self.custom}


PRESETS: tuple[HostedPreset, ...] = (
    HostedPreset("anthropic", "Anthropic", "https://api.anthropic.com/v1", ANTHROPIC_ENGINE,
                 key_hint="an API key from the Claude Console",
                 note="Claude, through Anthropic's own API; prompts cached by default (D170)"),
    HostedPreset("openai", "OpenAI", "https://api.openai.com/v1", OPENAI_ENGINE,
                 key_hint="an API key from the OpenAI platform"),
    HostedPreset("groq", "Groq", "https://api.groq.com/openai/v1", OPENAI_ENGINE),
    HostedPreset("together", "Together AI", "https://api.together.xyz/v1", OPENAI_ENGINE),
    HostedPreset("ollama-cloud", "Ollama's cloud", "https://ollama.com/v1", OPENAI_ENGINE,
                 note="the models Ollama serves on its own machines (D168)"),
    HostedPreset("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", OPENAI_ENGINE),
    HostedPreset("deepseek", "DeepSeek", "https://api.deepseek.com/v1", OPENAI_ENGINE),
    HostedPreset("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1", OPENAI_ENGINE),
    HostedPreset("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", OPENAI_ENGINE),
    HostedPreset("openai-compatible", "Another OpenAI-format API", "", OPENAI_ENGINE,
                 note="any provider serving /models and /chat/completions under one base URL"),
    HostedPreset("anthropic-compatible", "Another Anthropic-format API", "", ANTHROPIC_ENGINE,
                 note="any provider serving /models and /messages under one base URL"),
)


def preset(preset_id: str) -> Optional[HostedPreset]:
    return next((p for p in PRESETS if p.id == preset_id), None)


def preset_for(base_url: str, engine: str) -> Optional[HostedPreset]:
    """The preset a saved host came from, read back from its URL and engine — never stored."""
    normal = (base_url or "").rstrip("/")
    return next((p for p in PRESETS if p.base_url and p.base_url == normal and p.engine == engine), None)
