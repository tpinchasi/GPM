"""Test connection — what makes adding a host safe.

docs/spec/console-and-control-api.md §2.1. Given a host definition **as filled in, without
saving**: reach the endpoint, authenticate, list models, check the pool's whole model set is
resident together, inspect capabilities, run the concurrency check, and report the worker
count that would apply. Each step passes or fails with its actual error.

The concurrency check is the part that cannot be guessed: an engine left at a lower parallelism
than the pool's worker count queues requests *inside itself*, and every latency figure the pool
records then lies.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any, Mapping, Optional, Sequence

import httpx

from .catalog import variants_for_host
from .config import PoolSettings, TransportConfig
from .transports import build_client


@dataclasses.dataclass
class Step:
    name: str
    ok: bool
    detail: str = ""
    #: What would put a failure right, as a name a client can act on — never instructions to
    #: parse. `engine_settings`: the engine must be restarted with more parallelism, which the
    #: pool can do itself on a host that runs an agent (docs/spec/host-agent.md, D41).
    fix: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "fix": self.fix}


def _one_token(engine: Any, tag: str) -> Optional[tuple[str, dict[str, Any]]]:
    """The shortest generation, on a path this engine serves — Ollama's own, or OpenAI's
    completions, which needs no chat template. None for an engine that generates no text."""
    paths = engine.inference_paths()
    if "/api/generate" in paths:
        return "/api/generate", {"model": tag, "prompt": "hi", "stream": False, "options": {"num_predict": 1}}
    if "/v1/completions" in paths:
        return "/v1/completions", {"model": tag, "prompt": "hi", "max_tokens": 1, "stream": False}
    return None


def _chat_builds(required: set[str], model_set: Sequence[str], catalog: Mapping[str, Any]) -> list[str]:
    """The builds in `required` that generate text: the concurrency check measures a chat build,
    never a speech model or a ranker, which answer a different shape of request (D142)."""
    chat = set()
    for name in model_set:
        entry = catalog.get(name)
        if entry is None:
            chat.add(name)  # a model with no catalog entry is a chat model under its own name
            continue
        chat.update(v.tag for v in entry.variants if "chat" in v.kinds())
    return sorted(required & chat)


async def _concurrency(client: httpx.AsyncClient, engine: Any, tag: str, workers: int) -> Step:
    """n short generations at once, against one. Near n× wall time means the engine is
    serialising, whatever it was asked for."""
    if workers < 2:
        return Step("concurrency", True, "one worker: nothing to check")
    request = _one_token(engine, tag)
    if request is None:
        return Step("concurrency", True, f"{engine.name} generates no text: nothing to measure")
    path, body = request

    async def once() -> bool:
        try:
            response = await client.post(path, json=body, timeout=120)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    started = time.monotonic()
    if not await once():
        return Step("concurrency", False, "a single short generation did not succeed")
    single = time.monotonic() - started

    started = time.monotonic()
    results = await asyncio.gather(*(once() for _ in range(workers)))
    together = time.monotonic() - started

    if not all(results):
        return Step("concurrency", False, f"{results.count(False)} of {workers} parallel requests failed")
    if single <= 0:
        return Step("concurrency", True, "too fast to measure")

    ratio = together / single
    if ratio >= workers * 0.75:
        return Step(
            "concurrency",
            False,
            f"{workers} at once took {ratio:.1f}x one ({together:.2f}s against {single:.2f}s): the "
            f"engine is serialising. Set its parallelism to at least {workers} — for Ollama, "
            f"OLLAMA_NUM_PARALLEL={workers}",
            fix="engine_settings",
        )
    return Step("concurrency", True, f"{workers} at once took {ratio:.1f}x one: really parallel")


async def test_connection(
    host: Mapping[str, Any],
    *,
    engine: Any,
    settings: PoolSettings,
    model_set: Sequence[str],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Runs the checks in order and stops at the first that makes the rest meaningless."""
    steps: list[Step] = []
    workers = int(host.get("workers") or 1)
    capabilities = list(host.get("capabilities") or [])

    transport_body = dict(host.get("transport") or {})
    if not transport_body:
        base_url = host.get("base_url")
        if not base_url:
            raise ValueError("give either a transport block or a base_url")
        transport_body = {
            "type": "https" if str(base_url).startswith("https") else "http",
            "base_url": base_url,
            "allow_insecure": bool(host.get("allow_insecure", True)),
        }
    if transport_body.get("type") == "tunnel":
        raise ValueError("a tunnel host cannot be tested before it is saved: the pool must open the forward first")

    try:
        transport = TransportConfig.model_validate(transport_body)
    except Exception as exc:  # pydantic validation, or a rule such as plain http off loopback
        return {"ok": False, "steps": [Step("definition", False, str(exc)).as_dict()], "workers": 0}
    steps.append(Step("definition", True, f"{transport.type} → {transport.base_url}"))

    try:
        client = build_client(transport, settings, transport.base_url)
    except Exception as exc:
        steps.append(Step("credentials", False, str(exc)))
        return {"ok": False, "steps": [s.as_dict() for s in steps], "workers": 0}
    steps.append(Step("credentials", True, "set ✓" if transport.auth_headers() else "none needed"))

    try:
        health = await engine.health(client)
        if getattr(engine, "residency_from_agent", False):
            # Its agent fetches the model and starts the server, and before the host is saved
            # the agent has been asked for nothing (D148): a server not answering yet is the
            # expected state, and what it holds is reported once the host is saved.
            steps.append(Step("reach", True, "the server answered" if health.ok
                              else "not answering yet: the host's agent starts it once the host is saved"))
            steps.append(Step("model set", True, "fetched and started by the host's agent"))
            return {"ok": True, "steps": [s.as_dict() for s in steps], "workers": workers}
        steps.append(Step("reach", health.ok, health.detail or "the engine answered"))
        if not health.ok:
            return {"ok": False, "steps": [s.as_dict() for s in steps], "workers": 0}

        variants = variants_for_host(list(model_set), catalog, frozenset(capabilities), engine.name)
        required = {group[0].tag for group in variants.values() if group}

        resident = await engine.models_resident(client)
        available = await engine.models_available(client)
        steps.append(Step("models", True, f"resident now: {', '.join(sorted(resident)) or 'none'}"))

        residency = host.get("residency", "pinned")
        if residency == "on_demand":
            missing = required - available
            ok_detail = "the whole set is on disk; models load on first use"
            fail_detail = f"missing {sorted(missing)}: not on disk. Pull them; the pool never downloads for a request"
        else:
            missing = required - resident
            ok_detail = "the whole set is resident together"
            fail_detail = f"missing {sorted(missing)}: this host would not join the pool until they are loaded"
        steps.append(Step("model set", not missing, ok_detail if not missing else fail_detail))

        steps.append(Step("capabilities", True, ", ".join(capabilities) if capabilities else "none declared; the no-requirements variant is served"))

        chat = _chat_builds(required, model_set, catalog)
        if not missing and chat:
            steps.append(await _concurrency(client, engine, chat[0], workers))
    except httpx.HTTPError as exc:
        steps.append(Step("reach", False, str(exc)))
    finally:
        await client.aclose()

    return {
        "ok": all(step.ok for step in steps),
        "steps": [step.as_dict() for step in steps],
        "workers": workers,
    }
