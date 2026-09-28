"""Restarting the engine, and the settings it starts with (docs/spec/host-agent.md §4, D41).

Two things make this safe to expose. **What runs is the owner's**: the only command this
module can start is `restart_command` from the agent's own settings file, as an argument list,
never through a shell; nothing in a request can name, extend or replace it. **What the pool
sends is numbers**: a closed set of bounded integers, turned into the engine's environment
variables *here*, and written to a file whose path the owner chose. No text from the pool is
ever written to disk or put in a process's environment.

And it only happens when an operator asks. A restart drops whatever the engine is doing, so
it is never something the pool's control loop does on its own.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .engines import OllamaFacts
from .models import Refused
from .settings import Settings

_BOUNDS = {
    "workers": (1, 64), "models_held": (1, 64), "context": (256, 1_048_576),
    # How many cards each copy of a model is split across (D114). A power of two: the engines
    # that split a model this way divide its attention heads between the cards.
    "cards_per_copy": (1, 8),
}
_HEADER = "# Written by gpm-agent from the pool's settings for this host. Edits are overwritten.\n"
_OUTPUT_TAIL = 2000


def validated(body: Any) -> Optional[dict[str, int]]:
    """The settings asked for, or None for a plain restart. Anything but the known names with
    integers inside their bounds is refused — this is the whole of what the pool can say."""
    if not isinstance(body, dict) or set(body) - {"settings"}:
        raise Refused(400, "bad_request", "the body may carry `settings` and nothing else")
    wanted = body.get("settings")
    if wanted is None:
        return None
    if not isinstance(wanted, dict) or set(wanted) - set(_BOUNDS):
        raise Refused(400, "bad_settings", f"settings may name only {sorted(_BOUNDS)}")
    for name, value in wanted.items():
        low, high = _BOUNDS[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise Refused(400, "bad_settings", f"{name} must be a whole number from {low} to {high}")
    if wanted.get("cards_per_copy", 1) not in (1, 2, 4, 8):
        raise Refused(400, "bad_settings", "cards_per_copy must be 1, 2, 4 or 8")
    if "workers" not in wanted or "models_held" not in wanted:
        raise Refused(400, "bad_settings", "settings must state workers and models_held")
    return dict(wanted)


def read_applied(path: Optional[str]) -> Optional[dict[str, str]]:
    """The environment last written for the engine, or None if there is none to read."""
    if not path:
        return None
    try:
        lines = Path(path).expanduser().read_text().splitlines()
    except OSError:
        return None
    pairs = (line.partition("=") for line in lines if line.strip() and not line.startswith("#"))
    return {name.strip(): value.strip() for name, _, value in pairs}


def _write(path: str, environment: dict[str, str]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    draft = target.with_name(target.name + ".new")
    # World-readable on purpose: it holds no secret, and the engine's service user reads it.
    draft.write_text(_HEADER + "".join(f"{name}={value}\n" for name, value in sorted(environment.items())))
    os.chmod(draft, 0o644)
    os.replace(draft, target)


class EngineControl:
    def __init__(self, settings: Settings, engine: OllamaFacts, client: httpx.AsyncClient):
        self.settings, self.engine, self.client = settings, engine, client
        self._lock = asyncio.Lock()

    @property
    def can_restart(self) -> bool:
        return bool(self.settings.restart_command)

    @property
    def can_apply_settings(self) -> bool:
        return self.can_restart and bool(self.settings.engine_env_file)

    async def act(self, body: Any) -> dict[str, Any]:
        wanted = validated(body)
        if not self.can_restart:
            raise Refused(409, "owner_has_not_enabled_restart",
                          "this machine's owner has set no restart_command in the agent's settings; the pool cannot supply one")
        if wanted is not None and not self.settings.engine_env_file:
            raise Refused(409, "owner_has_not_enabled_settings",
                          "this machine's owner has set no engine_env_file in the agent's settings; the pool cannot choose where to write")
        if wanted is not None and wanted.get("cards_per_copy", 1) > 1 and not getattr(self.engine, "splits_across_cards", False):
            raise Refused(409, "engine_cannot_split",
                          f"{self.engine.name} does not split a model across cards on request")
        if self._lock.locked():
            raise Refused(409, "restart_in_progress", "the engine is already being restarted")
        async with self._lock:
            changed = False
            if wanted is not None:
                environment = self.engine.launch_environment(**wanted)
                changed = read_applied(self.settings.engine_env_file) != environment
                if changed:
                    _write(self.settings.engine_env_file, environment)
            code, output = await self._restart()
            answers = await self._wait_for_engine()
        return {
            "settings_written": changed,
            "applied": read_applied(self.settings.engine_env_file),
            "restart_exit_code": code,
            "restart_output": output[-_OUTPUT_TAIL:],
            "engine_answers": answers,
        }

    async def _restart(self) -> tuple[Optional[int], str]:
        # The owner's argument list, exactly as written in their file. No shell.
        try:
            process = await asyncio.create_subprocess_exec(
                *self.settings.restart_command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return None, f"could not start the restart command: {exc}"
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=self.settings.restart_timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            return None, f"the restart command did not finish in {self.settings.restart_timeout_s}s and was stopped"
        return process.returncode, output.decode(errors="replace")

    async def _wait_for_engine(self) -> bool:
        deadline = time.monotonic() + self.settings.restart_timeout_s
        while time.monotonic() < deadline:
            if (await self.engine.describe(self.client)).get("answers"):
                return True
            await asyncio.sleep(0.5)
        return False
