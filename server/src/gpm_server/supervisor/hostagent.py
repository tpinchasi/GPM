"""Putting the host agent on a machine the pool rents (D63, D72).

The pool created the host, so the pool configures it: it copies the packed agent over the SSH
connection it already needs for the dead-man timer, has the agent mint a key for that host
alone, starts it on the host's loopback, and reaches it through a second forward.

Three rules keep this off the list of ways to lose money:

1. **The dead-man timer stays independent.** It is armed by the start-up script, before any of
   this, and knows nothing about the agent.
2. **No agent is never fatal.** A host with no interpreter, or one where any step here fails,
   is prepared the way it was before and joins the pool; the reason is recorded.
3. **Nothing here blocks the model download**, which is the long pole in preparing a host.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shlex
from pathlib import Path
from typing import Optional

log = logging.getLogger("gpm.hostagent")

#: Where the agent and its settings live on a rented host — beside the timer's own state.
REMOTE_DIR = "/var/run/gpm"
ARCHIVE = f"{REMOTE_DIR}/gpm-agent.pyz"
SETTINGS = f"{REMOTE_DIR}/agent.json"
HEARTBEAT = f"{REMOTE_DIR}/heartbeat"
ENGINE_RESTART = f"{REMOTE_DIR}/restart-engine.sh"
ENGINE_ENV = f"{REMOTE_DIR}/engine.env"
#: The agent listens here, on the host's loopback only. Never exposed; the pool forwards to it.
AGENT_PORT = 8195

_KEY = re.compile(r"\b(gpmg_[0-9a-f]{64})\b")


@dataclasses.dataclass
class RentedAgent:
    """How the pool reaches the agent on a host it rents.

    Deliberately *not* an `AgentConfig`: that is the operator's file schema, where a key may
    only ever be named as an environment variable. This key was minted minutes ago on a machine
    the pool created, and lives in the supervisor's memory for as long as the host does.
    """

    url: str
    secret: str
    verify: bool = True
    #: Named for the message an agent client writes when a key is refused.
    bearer_env: str = "the key the pool installed on this host"
    manage_models: bool = True

    def key(self) -> Optional[str]:
        return self.secret


class AgentInstallFailed(Exception):
    """The host gets no agent. It still joins the pool."""


async def install(
    *,
    run,
    push,
    archive: Path,
    engine_port: int,
    engine: str = "ollama",
    agent_port: int = AGENT_PORT,
) -> str:
    """Put the agent on one host and return the key it minted.

    `run(command) -> (code, output)` and `push(data, path) -> (code, output)` are the pool's
    existing ways of reaching a rented host: one SSH exec, and one with a file on its input.
    """
    code, found = await run("command -v python3 || true")
    if code != 0 or not found.strip():
        raise AgentInstallFailed(
            "this host has no python3, so it runs without an agent "
            "(an image that carries one — the provider's own build, say — would get it one)"
        )

    code, output = await push(archive.read_bytes(), ARCHIVE)
    if code != 0:
        raise AgentInstallFailed(f"the agent could not be copied: {output.strip()[:200]}")

    code, output = await run(
        f"mkdir -p {REMOTE_DIR} && chmod 700 {REMOTE_DIR} && rm -f {SETTINGS} && "
        f"python3 {ARCHIVE} -c {SETTINGS} init "
        f"--host 127.0.0.1 --port {agent_port} "
        # Which engine this machine runs (D93): the agent holds models and reads settings in
        # that engine's terms, and one told the wrong name would fetch the wrong weights.
        f"--engine {shlex.quote(engine)} "
        f"--engine-url http://127.0.0.1:{engine_port} "
        f"--heartbeat-file {HEARTBEAT} "
        # On a host the pool created, what restarts the engine is the pool's own start-up
        # material — fixed at installation, never sent over the protocol (D63 amending D41).
        f"--restart-command {ENGINE_RESTART} "
        f"--engine-env-file {ENGINE_ENV}"
    )
    if code != 0:
        raise AgentInstallFailed(f"the agent could not be set up: {output.strip()[:200]}")
    minted = _KEY.search(output)
    if minted is None:
        raise AgentInstallFailed("the agent did not mint a key this pool could read")

    # Started detached, so it outlives this SSH session; its log stays on the host for an
    # operator who goes looking, and carries no request content.
    code, output = await run(
        f"(setsid nohup python3 {ARCHIVE} -c {SETTINGS} serve "
        f">{REMOTE_DIR}/agent.log 2>&1 &) ; sleep 1; "
        f"kill -0 $(pgrep -f 'gpm-agent.pyz.*serve' | head -1) 2>/dev/null && echo started"
    )
    if code != 0 or "started" not in output:
        raise AgentInstallFailed(f"the agent did not start: {output.strip()[:200]}")
    return minted.group(1)


def push_command(path: str) -> str:
    """The far side of a file copy: one shell command reading the file from its input."""
    return f"mkdir -p {shlex.quote(REMOTE_DIR)} && cat > {shlex.quote(path)} && chmod 600 {shlex.quote(path)}"
