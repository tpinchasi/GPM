"""The supervisor's side of the host agent (docs/spec/host-agent.md).

The pool dials the agent; only the supervisor does, and only on its control pass. An agent is
auxiliary: a host whose agent is silent keeps serving, verified by the probe as it always was,
and **nothing is inferred from the silence** — no capability, no fact, no state.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Iterable, Optional

import httpx

from ..config import AgentConfig

PROTOCOL = "1"
#: Short on purpose: an agent that does not answer must not stretch the control pass.
_TIMEOUT = httpx.Timeout(connect=2.0, read=4.0, write=4.0, pool=2.0)
#: The capabilities a machine's hardware decides. Anything else in a host's list (a precision
#: a card supports, say) is the operator's statement, and the agent has no opinion on it.
PLATFORM_CAPABILITIES = frozenset({"apple-silicon", "cuda", "rocm"})


@dataclasses.dataclass(frozen=True)
class AgentView:
    """What the last attempt to ask the agent produced."""

    reachable: bool
    detail: Optional[str] = None
    facts: Optional[dict[str, Any]] = None
    #: Where the machine stands against the model set it was asked to hold, when it manages one.
    models: Optional[dict[str, Any]] = None
    asked_at: float = dataclasses.field(default_factory=time.time)

    @property
    def derived_capabilities(self) -> Optional[frozenset[str]]:
        """None when the agent is silent or could not tell — which is not "none"."""
        if not self.reachable or self.facts is None:
            return None
        derived = self.facts.get("capabilities")
        return None if derived is None else frozenset(derived) & PLATFORM_CAPABILITIES


async def ask(agent: AgentConfig, *, transport: Optional[httpx.AsyncBaseTransport] = None) -> AgentView:
    if agent.url is None:
        return AgentView(False, "the forward to this host's agent is not open yet")
    key = agent.key()
    if key is None:
        return AgentView(False, f"agent key missing: environment variable {agent.bearer_env} is not set")
    try:
        async with httpx.AsyncClient(
            base_url=agent.url, timeout=_TIMEOUT, verify=agent.verify, transport=transport,
            headers={"Authorization": f"Bearer {key}"},
        ) as client:
            response = await client.get(f"/agent/v{PROTOCOL}/facts")
    except httpx.HTTPError as exc:
        return AgentView(False, f"agent unreachable: {str(exc) or type(exc).__name__}")
    if response.status_code in (401, 403):
        return AgentView(False, "the agent refused the key in " + agent.bearer_env)
    if response.status_code != 200:
        return AgentView(False, f"the agent answered {response.status_code}")
    try:
        facts = response.json()
    except ValueError:
        return AgentView(False, "the agent's answer was not JSON")
    # The host is not trusted: an answer of the wrong shape is no answer, never an exception in
    # the supervisor's pass.
    if not isinstance(facts, dict) or not isinstance(facts.get("agent"), dict) or (
            "engine" in facts and not isinstance(facts["engine"], dict)):
        return AgentView(False, "the agent's answer was not in the shape of its facts")
    spoken = str(facts["agent"].get("protocol"))
    if spoken != PROTOCOL:
        return AgentView(False, f"the agent speaks protocol {spoken}; this pool speaks {PROTOCOL}")
    return AgentView(True, facts=facts)


def _client(agent: AgentConfig, key: str, transport: Optional[httpx.AsyncBaseTransport]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=agent.url, timeout=_TIMEOUT, verify=agent.verify, transport=transport,
        headers={"Authorization": f"Bearer {key}"},
    )


async def hold(
    agent: AgentConfig, tags: Iterable[str], residency: str,
    *, transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Optional[dict[str, Any]]:
    """Send the whole desired state; read back where the machine stands. The agent works in
    the background and answers at once, so this never waits on a download. None on any failure:
    the facts call already said whether the agent answers, and the next pass says it all again."""
    key = agent.key()
    if key is None:
        return None
    try:
        async with _client(agent, key, transport) as client:
            response = await client.put(f"/agent/v{PROTOCOL}/models", json={"tags": sorted(tags), "residency": residency})
        return response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def beat(
    agent, *, transport: Optional[httpx.AsyncBaseTransport] = None
) -> Optional[str]:
    """Postpone the host's dead-man timer through its agent (D63). None on success.

    One request on a connection the pool already holds, in place of an SSH session per host
    per pass. It can only ever postpone a shutdown the pool could equally cause by going
    silent, so there is nothing here to bound.
    """
    key = agent.key()
    if key is None:
        return "no key for this agent"
    try:
        async with _client(agent, key, transport) as client:
            response = await client.post(f"/agent/v{PROTOCOL}/heartbeat")
    except httpx.HTTPError as exc:
        return f"agent unreachable: {str(exc) or type(exc).__name__}"
    if response.status_code != 200:
        return f"the agent answered {response.status_code} to a heartbeat"
    return None


async def delete_model(
    agent: AgentConfig, tag: str, *, transport: Optional[httpx.AsyncBaseTransport] = None
) -> tuple[int, dict[str, Any]]:
    """An operator's explicit act, passed on. The agent has the last word and its words are
    returned as they are."""
    key = agent.key()
    if key is None:
        return 503, {"error": "agent_key_missing", "detail": f"environment variable {agent.bearer_env} is not set"}
    try:
        async with _client(agent, key, transport) as client:
            response = await client.request("DELETE", f"/agent/v{PROTOCOL}/models", json={"tag": tag})
        return response.status_code, response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return 502, {"error": "agent_unreachable", "detail": str(exc) or type(exc).__name__}


async def restart_engine(
    agent: AgentConfig, settings: Optional[dict[str, int]],
    *, transport: Optional[httpx.AsyncBaseTransport] = None,
) -> tuple[int, dict[str, Any]]:
    """An operator's explicit act (D41). `settings` is numbers or nothing: the pool never sends
    the engine a variable name, a value as text, or a command. Waits as long as a restart may
    take — which is why this is never called from the control pass."""
    key = agent.key()
    if key is None:
        return 503, {"error": "agent_key_missing", "detail": f"environment variable {agent.bearer_env} is not set"}
    try:
        async with httpx.AsyncClient(
            base_url=agent.url, verify=agent.verify, transport=transport,
            timeout=httpx.Timeout(connect=2.0, read=400.0, write=10.0, pool=2.0),
            headers={"Authorization": f"Bearer {key}"},
        ) as client:
            response = await client.post(f"/agent/v{PROTOCOL}/engine", json={"settings": settings})
        return response.status_code, response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return 502, {"error": "agent_unreachable", "detail": str(exc) or type(exc).__name__}


def wanted_engine_settings(workers: int, models_held: int, cards_per_copy: int = 1) -> dict[str, int]:
    """What this host's engine must run with for the pool's numbers to be true: as many
    requests at once as the host has workers, the whole model set held together, and each
    model split across as many cards as it was bought for (D114).

    `cards_per_copy` is sent only when it is more than one: an agent from before it existed
    refuses a name it does not know, and one card per copy is what every agent already does."""
    wanted = {"workers": workers, "models_held": max(1, models_held)}
    if cards_per_copy > 1:
        wanted["cards_per_copy"] = cards_per_copy
    return wanted


def model_events(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    """What changed on the machine between two answers, as (kind, summary, numbers). Pure, so
    the decision log's account of a download is tested without one."""
    if not after:
        return []
    was = {m["tag"]: m for m in (before or {}).get("models", [])}
    events = []
    for model in after.get("models", []):
        tag, old = model["tag"], was.get(model["tag"], {})
        if model.get("pulling") and not old.get("pulling"):
            total = model["pulling"].get("total_bytes")
            events.append(("agent_pull_started", f"pulling {tag}", {"total_gb": round(total / 1e9, 2) if total else None}))
        if model.get("on_disk") and old and not old.get("on_disk"):
            events.append(("agent_model_on_disk", f"{tag} is on disk", {"size_gb": round((model.get('size_bytes') or 0) / 1e9, 2)}))
        if model.get("error") and model.get("error") != old.get("error"):
            events.append(("agent_model_failed", f"{tag}: {model['error']}", {}))
    return events


def effective_capabilities(typed: Iterable[str], view: Optional[AgentView]) -> frozenset[str]:
    """What variants are resolved against. The operator's list always stands; where it names no
    platform at all, the platform the agent found fills the gap. Never an override: a list that
    names a platform is used as written, and a disagreement is reported (`capability_conflict`)
    rather than settled silently in either direction."""
    typed = frozenset(typed)
    derived = view.derived_capabilities if view else None
    if derived and not typed & PLATFORM_CAPABILITIES:
        return typed | derived
    return typed


def capability_conflict(typed: Iterable[str], view: Optional[AgentView]) -> Optional[str]:
    derived = view.derived_capabilities if view else None
    stated = frozenset(typed) & PLATFORM_CAPABILITIES
    if derived is None or not stated or stated == derived:
        return None
    return (
        f"configuration says {sorted(stated)} but the agent on the machine found "
        f"{sorted(derived) or 'no accelerator'}; the configured list is in use until it is corrected"
    )
