"""A build's thinking default (D187): what a chat request carries when the app did not say.

One model can think on one host and answer at once on another — Gemma 4 on a machine's Ollama
thinks before it answers, its build on Ollama's cloud does not — and a reply's token limit spent
on thinking comes back empty. A catalog build's `thinking: on | off` makes the pool's answer the
same wherever it is served: the pool adds the engine's own switch to a chat request that names
none, and never changes one the app wrote.

Added the way the model is rewritten — one member put in after the opening brace, every other
byte as the app sent it (the passthrough the app contract promises).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .openai_api import members

#: Every field by which a request already says how much to think, per path. One of them present,
#: the request is left alone: the app's word wins, whatever the build's default.
_SAID = {
    "/api/chat": frozenset({"think"}),
    "/api/generate": frozenset({"think"}),
    "/v1/chat/completions": frozenset({"reasoning_effort", "reasoning", "think", "chat_template_kwargs"}),
    "/v1/messages": frozenset({"thinking"}),
}


def switch(engine: str, path: str, on: bool) -> Optional[tuple[str, Any]]:
    """The member that turns thinking on or off for this engine on this path, or None where the
    pool knows of none (then the engine's own default stands)."""
    if path in ("/api/chat", "/api/generate"):
        return "think", on  # Ollama's own API
    if path == "/v1/chat/completions":
        if engine == "vllm":
            # vLLM reads a model's thinking switch from its chat template's arguments.
            return "chat_template_kwargs", {"enable_thinking": on}
        # Ollama's OpenAI-shaped path, and OpenAI-shaped providers: "none" asks for no thinking.
        return "reasoning_effort", ("medium" if on else "none")
    if path == "/v1/messages":
        if not on:
            return "thinking", {"type": "disabled"}
        if engine == "anthropic-api":
            return "thinking", {"type": "adaptive"}
    return None


def says_thinking(path: str, body: bytes) -> bool:
    """Whether a request names any thinking field of its own, on the path it arrived on."""
    if path not in _SAID:
        return False
    found = members(body)
    return found is not None and any(key in _SAID[path] for key, *_ in found)


def with_thinking(engine: str, path: str, body: bytes, setting: Optional[str]) -> bytes:
    """`body` with the build's thinking default added, where the build has one, the path takes
    one and the request says nothing of its own. A request translated to another format is asked
    on the path it arrived on too (the router does that): a field the translation dropped was
    still the app's word."""
    if setting not in ("on", "off") or path not in _SAID:
        return body
    found = members(body)
    if found is None or any(key in _SAID[path] for key, *_ in found):
        return body
    member = switch(engine, path, setting == "on")
    if member is None:
        return body
    text = body.decode("utf-8")
    brace = text.index("{") + 1
    added = json.dumps(member[0]) + ":" + json.dumps(member[1], separators=(",", ":"))
    return (text[:brace] + added + ("," if found else "") + text[brace:]).encode("utf-8")
