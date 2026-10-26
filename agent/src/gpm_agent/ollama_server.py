"""Ollama's server, run by this agent on a machine whose owner lets it install engines (D150).

Until S9, Ollama was the machine owner's service: the agent asked it for models and, to apply the
pool's numbers, ran the owner's restart command. On a machine with `install_engines: true` the
agent installs Ollama itself (`installs`) and runs it, as it runs whisper.cpp's server (D148):

- **A fixed command**: `<version directory>/ollama serve`, nothing else — the program from the
  version the agent installed, and one constant argument.
- **On loopback, at the engine URL's port**, with the models directory the agent's settings name
  (`~/.ollama/models` unless set, so nothing already downloaded is downloaded again), and the
  pool's numbers — workers, models held, context — as the environment Ollama reads them from,
  built from integers by the engine's adapter. The hub's and GitHub's credentials are withheld.
- **Recognised as its own** by a record of what it started and the process's command line, so
  that it is still known as the agent's own after the agent restarts, and a process number that
  has since gone to another program is never signalled.
- **Never over something it did not start.** If anything else answers on the port — the Ollama
  app, or an `ollama serve` the owner started — the agent reports it and leaves it alone.

The record, the pool's numbers and the log live in the engine's own directory under the engines
directory, beside its versions.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from . import platform as host_platform

#: What was started, the pool's numbers, and the server's log, in the engine's directory.
RECORD_FILE = "server.json"
ENVIRONMENT_FILE = "environment.json"
LOG_FILE = "ollama.log"

#: The pool's numbers as Ollama reads them — the only names the agent ever writes for it, and the
#: only ones read back from its own file (`engines.OllamaFacts.launch_environment` builds them).
POOL_NAMES = ("OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_CONTEXT_LENGTH")

#: Credentials the agent may hold — for the model hub, and for GitHub's releases. The server has
#: no use for either, so it is started without them.
WITHHELD = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN")

#: The longest one look at the server may take. Short: the facts look on every pass.
ANSWER_TIMEOUT_S = 3.0

#: Between looks while the server starts.
POLL_S = 0.25

#: How far back the log is read for the line that says why a server stopped.
_LOG_TAIL_BYTES = 16384


def serve_argv(program: Path) -> list[str]:
    """The whole command the server is started with: the installed program, and `serve`."""
    return [str(program), "serve"]


def _last_line(log: Optional[str], offset: int) -> Optional[str]:
    if not log:
        return None
    try:
        with open(log, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(offset, handle.tell() - _LOG_TAIL_BYTES))
            lines = handle.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:300]
    return None


class OllamaServer:
    def __init__(
        self,
        state_dir: Path,
        *,
        engine_url: str,
        models_path: str,
        client: httpx.AsyncClient,
        os_name: Optional[str] = None,
    ) -> None:
        self.state_dir = state_dir
        #: The settings refuse an engine URL that is not loopback with a port, for such a host.
        self.port = int(urlsplit(engine_url).port or 11434)
        self.models_path = models_path
        self.client = client
        self.os_name = os_name
        self.foreign_reason = (
            f"something this agent did not start already answers on port {self.port}: quit the Ollama "
            "you started yourself, and remove it from login items, to let the agent run it"
        )
        self._process: Optional[host_platform.Process] = None
        self._record: Optional[dict[str, Any]] = None
        self._record_read = False
        #: Whether a process recorded before this agent started is its server, checked once each.
        self._identity: dict[int, bool] = {}

    # --- the pool's numbers ---

    def pool_environment(self) -> Optional[dict[str, str]]:
        """The pool's numbers the server is started with, as last stored — None if never."""
        try:
            stored = json.loads((self.state_dir / ENVIRONMENT_FILE).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(stored, dict):
            return None
        return {
            name: value for name, value in stored.items()
            if name in POOL_NAMES and isinstance(value, str) and value.isdigit()
        }

    def set_pool_environment(self, environment: dict[str, str]) -> bool:
        """Store the pool's numbers for the next start; True when they changed."""
        if self.pool_environment() == environment:
            return False
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = self.state_dir / ENVIRONMENT_FILE
        draft = target.with_name(target.name + ".new")
        draft.write_text(json.dumps(environment, indent=2, sort_keys=True))
        os.replace(draft, target)
        return True

    def environment(self) -> dict[str, str]:
        """The server's whole environment: the agent's own, less its credentials, with the pool's
        numbers and — last, so nothing overrides them — the address and the models directory."""
        environment = {name: value for name, value in os.environ.items() if name not in WITHHELD}
        environment.update(self.pool_environment() or {})
        environment["OLLAMA_HOST"] = f"127.0.0.1:{self.port}"
        environment["OLLAMA_MODELS"] = str(Path(self.models_path).expanduser())
        return environment

    # --- what was started ---

    def _current(self) -> Optional[dict[str, Any]]:
        if not self._record_read:
            self._record_read = True
            try:
                record = json.loads((self.state_dir / RECORD_FILE).read_text())
            except (OSError, ValueError):
                record = None
            if isinstance(record, dict) and isinstance(record.get("pid"), int) and isinstance(record.get("program"), str):
                self._record = record
        return self._record

    def _remember(self, record: dict[str, Any]) -> None:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = self.state_dir / RECORD_FILE
        draft = target.with_name(target.name + ".new")
        draft.write_text(json.dumps(record, indent=2))
        os.replace(draft, target)
        self._record, self._record_read = record, True

    def _forget(self) -> None:
        (self.state_dir / RECORD_FILE).unlink(missing_ok=True)
        self._record, self._record_read, self._process = None, True, None

    def _own_process(self, record: dict[str, Any]) -> Optional[host_platform.Process]:
        if self._process is not None and self._process.pid == record.get("pid"):
            return self._process
        return None

    def _is_ours(self, pid: int, record: dict[str, Any]) -> bool:
        """Is process `pid` the server the record describes? Started with the recorded program
        and `serve`; a program that has since taken the number is not."""
        said = host_platform.command_line(pid, os_name=self.os_name) or b""
        return str(record["program"]).encode() in said and b"serve" in said

    def _alive(self, record: dict[str, Any]) -> bool:
        own = self._own_process(record)
        if own is not None:
            return own.poll() is None
        pid = record["pid"]
        if not host_platform.alive(pid, os_name=self.os_name):
            self._identity.pop(pid, None)  # the number may go to another program next
            return False
        if pid not in self._identity:
            self._identity[pid] = self._is_ours(pid, record)
        return self._identity[pid]

    def ours(self) -> bool:
        """Is the server this agent started running now — started by this agent process, or by
        the one before it, and still the same program?"""
        record = self._current()
        return record is not None and self._alive(record)

    def running_version(self) -> Optional[str]:
        record = self._current()
        return record.get("version") if record is not None and self._alive(record) else None

    def why_stopped(self) -> Optional[str]:
        """The server's last words, when the one this agent started has stopped."""
        record = self._current()
        if record is None or self._alive(record):
            return None
        own = self._own_process(record)
        code = own.returncode if own is not None else None
        line = _last_line(record.get("log"), int(record.get("log_offset") or 0))
        return (
            f"ollama serve exited{f' with code {code}' if code is not None else ''}: "
            f"{line or 'nothing in its log'}"
        )

    # --- what answers on the port ---

    async def answers(self) -> Optional[str]:
        """The version the server on the port says it is, or None if nothing answers so."""
        try:
            response = await self.client.get("/api/version", timeout=ANSWER_TIMEOUT_S)
            if response.status_code != 200:
                return None
            version = response.json().get("version")
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        return version if isinstance(version, str) else None

    async def listening(self) -> bool:
        """Does anything at all answer on the port — Ollama or not?"""
        try:
            await self.client.get("/api/version", timeout=ANSWER_TIMEOUT_S)
        except httpx.ConnectError:
            return False
        except httpx.HTTPError:
            return True  # something took the connection, and did not answer as a server should
        return True

    async def foreign(self) -> bool:
        """Something this agent did not start answers on the port."""
        return not self.ours() and await self.listening()

    # --- starting and stopping ---

    async def start(self, program: Path, version: str, timeout_s: float) -> Optional[str]:
        """Start `program serve` and wait for it to answer as `version`. None once it does;
        otherwise why not — and then whatever was started is left for the caller to stop."""
        if self.ours():
            return "this agent's server is already running"
        if await self.listening():
            return self.foreign_reason
        if not program.is_file():
            return f"{program} is not there"
        logs = self.state_dir / "logs"
        logs.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_path = logs / LOG_FILE
        argv = serve_argv(program)
        try:
            process, offset = host_platform.start(
                argv, env=self.environment(), log=log_path, cwd=self.state_dir,
                note=f"starting ollama {version}", os_name=self.os_name,
            )
        except OSError as exc:
            return f"could not start {program}: {exc}"
        self._process = process
        record = {
            "pid": process.pid, "version": version, "program": str(program), "argv": argv,
            "log": str(log_path), "log_offset": offset, "started_at": time.time(),
        }
        self._remember(record)
        deadline = time.monotonic() + timeout_s
        while self._alive(record):
            said = await self.answers()
            # An answer counts only from a server still running after it.
            if said is not None and self._alive(record):
                if said == version:
                    return None
                return f"it answered as version {said}, not {version}"
            if time.monotonic() >= deadline:
                return f"it did not answer within {timeout_s:.0f}s of starting"
            await asyncio.sleep(POLL_S)
        return self.why_stopped() or "it exited as it started"

    async def stop(self) -> None:
        """Stop the server this agent started, if it runs, and forget it. A process known only
        from the record is signalled only once its command line says it is that server."""
        record = self._current()
        if record is None:
            return
        own = self._own_process(record)
        if own is not None:
            await host_platform.stop(own.pid, process=own, os_name=self.os_name)
        else:
            pid = record["pid"]
            if host_platform.alive(pid, os_name=self.os_name) and self._is_ours(pid, record):
                await host_platform.stop(pid, os_name=self.os_name)
        self._forget()
