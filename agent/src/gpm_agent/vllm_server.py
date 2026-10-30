"""vLLM, run by this agent on a machine whose owner lets it install engines (D159).

On a machine the pool rents, or one whose owner runs vLLM, the engine is started by the machine's
own restart script running `gpm-agent vllm-start` (D97, D160). Where the owner set
`install_engines: true` the agent installs vLLM itself (D157) and runs it, as it runs Ollama and
whisper.cpp (D148, D150):

- **Its own launcher, as a program.** `python -m gpm_agent.cli vllm-start` with fixed arguments —
  the models directory and the engine URL's port from the agent's settings, `--proxy` when there
  is more than one model, `--vllm` naming the installed version's own program, which the launcher
  takes only from inside the agent's engines directory (D41), and `--models-from` naming a file in
  the engine's own directory that lists the models to start. Never imported here: the launcher
  starts processes, and the agent that answers the pool does not (D63). The pool's numbers reach
  it as the environment it reads them from (`GPM_VLLM_*`), stored in the engine's own directory as
  Ollama's are, on top of only what `platform.program_environment` keeps of the agent's own — no
  credential; and no `HF_TOKEN` either, since the models are already on disk.
- **The models it holds, once every one is on disk** — never one on disk that the pool no longer
  asks for, which would take its share of the card. vLLM serves what it was started with, and a
  start takes minutes, so it is not started for each model as it lands: only when every model the
  pool told this machine to hold is complete — or, until then, with what it ran before and still
  holds, while every one of those still is, so that a switch or a crash while another model is
  fetched does not leave the machine serving nothing. Until either holds, a start starts nothing
  and says why (`waiting`), which is not a failure: the version is still the one that runs.
- **Again whenever that set changes** (`wants_restart`) — a model held that has landed, one no
  longer held, or one deleted — when every engine of a start that served has died while its router
  lives on, and when the version is switched (the installer's own stop and start).
- **A start in which a model does not serve is that model's failure**, never the version's: the
  launcher refusing the set, starting none of it, every model exiting, or the wait running out.
- **Recognised as its own** by a record of what it started — version, program, models, port and
  the launcher's argument list — and by the launcher's own record of the processes it started in
  this boot of the machine, each still alive and still, by its command line, this version's
  `vllm serve` of a model in this models directory, the router for this models directory on this
  port, or the helper that starts its models in turn. A process number from a file is only a
  number; one that has since gone to another program is not this agent's.
- **Stopped by `gpm-agent vllm-stop`**, a program too, and only when what runs is its own.
  Anything else answering on the port is reported and left alone: the agent never stops a vLLM it
  did not start.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Collection, Optional, Sequence
from urllib.parse import urlsplit

import httpx

from . import platform as host_platform
from . import vllm_state
from .engines import complete_tags

log = logging.getLogger("gpm_agent.vllm")

#: What was started, the pool's numbers, and the launcher's own output, in the engine's directory.
RECORD_FILE = "server.json"
ENVIRONMENT_FILE = "environment.json"
LOG_FILE = "launcher.log"
#: The models it is started with, which the launcher reads (`--models-from`): the agent's own file,
#: in the engine's directory its settings place — never a path the pool names.
MODELS_FILE = "models.json"

#: The pool's numbers as the launcher reads them (`vllm_launch.NUMBER_FLAGS`, `CARDS_PER_COPY`) —
#: written out here, since the agent never imports the launcher (D63); a test holds them equal.
POOL_NAMES = (
    "GPM_VLLM_MAX_NUM_SEQS", "GPM_VLLM_MAX_NUM_BATCHED_TOKENS", "GPM_VLLM_MAX_MODEL_LEN",
    "GPM_VLLM_CARDS_PER_COPY",
)

#: How the agent runs its own launcher: its own module, as a program of its own.
AGENT_MODULE = ("-m", "gpm_agent.cli")

#: How long vLLM is given to serve every model it was started with before a start is called a
#: failure. Long: a large model loads for minutes, the models on one card start one after another
#: (D155), and a switch that gives up too soon rolls back a version that was only slow.
START_TIMEOUT_S = 1800.0

#: The launcher itself: it stops what ran before — each process given 30 seconds — and starts.
LAUNCH_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 180.0

#: The longest one look at the engine may take. Short: the facts look on every pass.
ANSWER_TIMEOUT_S = 3.0

#: Between looks while it starts.
POLL_S = 1.0

#: How long the processes found running are taken to be this agent's on the strength of one look
#: at their command lines — well under one of the pool's passes (as `ollama_server`).
IDENTITY_TRUSTED_S = 2.0


def switch_bound_s(start_timeout_s: float = START_TIMEOUT_S) -> float:
    """The longest one switch to a version of vLLM — or one start or restart of it — takes this
    agent from its beginning until it has an outcome (`engine.install.switch_bound_s`, which the
    pool waits for before it judges the switch, a rollback or an operator's restart).

    Sized for a switch that fails, whose whole path is one switch: what runs is stopped
    (`vllm-stop`, `STOP_TIMEOUT_S`), the port looked at, and the version started — a look, its
    launcher (`LAUNCH_TIMEOUT_S`) and the wait for it to serve (`start_timeout_s`, overrun by at
    most one look and one pause); then, as it did not start, it is stopped again and the version
    before started and waited for the same way. Twice each, so: a start that began and was waited
    for has an outcome of its own today (`start`), and only one that could not begin fails, but
    the pool's wait does not rest on that."""
    one_try = STOP_TIMEOUT_S + ANSWER_TIMEOUT_S + ANSWER_TIMEOUT_S + LAUNCH_TIMEOUT_S + start_timeout_s
    return 2 * (one_try + ANSWER_TIMEOUT_S + POLL_S)


#: Runs a fixed argument list and waits for it: `platform.run_command`, or a test's stand-in.
Runner = Callable[..., Awaitable[host_platform.Ran]]


def _last_line(output: str) -> str:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else "it said nothing"


def interpreter_guard() -> list[str]:
    """The guard the launcher's interpreter runs under, as this agent's own does: isolated where
    the agent is (`-I`, as it is started with the machine, D154), and otherwise never importing
    from the directory it happens to run in (`-P`) — `-m` would put that first on its path."""
    return ["-I"] if sys.flags.isolated else ["-P"]


class VllmServer:
    """vLLM as the installer drives it (`installs.Server`)."""

    def __init__(
        self,
        state_dir: Path,
        *,
        engine_url: str,
        models_path: str,
        client: httpx.AsyncClient,
        engines_dir: Path,
        held: Optional[Callable[[], Optional[Sequence[str]]]] = None,
        run: Optional[Runner] = None,
        python: str = sys.executable,
        os_name: Optional[str] = None,
        poll_s: float = POLL_S,
    ) -> None:
        self.state_dir = state_dir
        #: The settings refuse an engine URL that is not loopback with a port, for such a host.
        self.port = int(urlsplit(engine_url).port or 8000)
        self.models_dir = Path(models_path).expanduser()
        self.client = client
        self.engines_dir = Path(engines_dir)
        #: The models the pool told this machine to hold — None until it has said, after the agent
        #: starts (`app.create_app` reads them from the model work).
        self.held: Callable[[], Optional[Sequence[str]]] = held or (lambda: None)
        self.run: Runner = run or host_platform.run_command
        self.python = python
        self.os_name = os_name
        self.poll_s = poll_s
        self.foreign_reason = (
            f"something this agent did not start already answers on port {self.port}: stop the vLLM "
            "you run yourself, and take it out of the machine's start-up, to let the agent run it"
        )
        #: Why the last start started nothing, while it waits for the models it holds.
        self.waiting: Optional[str] = None
        self._record: Optional[dict[str, Any]] = None
        self._record_read = False
        #: When the processes found running were last found to be this agent's.
        self._confirmed: Optional[float] = None
        #: Set while this agent is starting it: whatever answers on the port meanwhile is its own.
        self._starting = False

    # --- the pool's numbers ---

    def pool_environment(self) -> Optional[dict[str, str]]:
        """The pool's numbers vLLM is started with, as last stored — None if never."""
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
        """The launcher's whole environment, which it passes on to every vLLM process: what
        `platform.program_environment` keeps of the agent's own, and the pool's numbers."""
        return host_platform.program_environment(dict(self.pool_environment() or {}), os_name=self.os_name)

    # --- what it would be started with ---

    def launcher_argv(self, program: Path, models: Sequence[str]) -> list[str]:
        """The launcher's whole command: fixed but for the agent's own settings — its models
        directory, its engine port, its engines directory, and the file in its engine's directory
        naming the models to start — and the installed version's program. Nothing the pool says
        is in it (D41): the models it holds reach the launcher only in that file, as repository ids
        the launcher checks as the fetch does, choosing among the directories already on disk."""
        argv = [
            self.python, *interpreter_guard(), *AGENT_MODULE, "vllm-start",
            "--models-dir", str(self.models_dir), "--port", str(self.port),
        ]
        if len(models) > 1:
            argv.append("--proxy")
        return [
            *argv, "--vllm", str(program), "--engines-dir", str(self.engines_dir),
            "--models-from", str(self.state_dir / MODELS_FILE),
        ]

    def stop_argv(self) -> list[str]:
        return [self.python, *interpreter_guard(), *AGENT_MODULE, "vllm-stop", "--models-dir", str(self.models_dir)]

    def to_start(self) -> tuple[Optional[list[str]], Optional[str]]:
        """The models vLLM would be started with now — every model the pool told this machine to
        hold, once each is complete on disk — or None, and why not yet. Never a model on disk that
        it no longer holds: it would take its share of the card from those it does. Until they are
        all there — or until the pool has said which — what it ran before and still holds, while
        every one of those still is, so that a switch or a crash while another model is fetched
        does not leave the machine serving nothing."""
        on_disk = complete_tags(self.models_dir)
        held = self.held()
        record = self._current()
        before = set(record.get("models") or []) if record is not None else set()
        missing: list[str] = []
        if held is not None:
            if not held:
                return None, "the pool holds no model on this machine"
            missing = sorted(set(held) - on_disk)
            if not missing:
                return sorted(set(held)), None
            before &= set(held)
        if before and before <= on_disk:
            # What ran before, still all here: started again while the rest is fetched.
            return sorted(before), None
        if held is None:
            return None, "waiting for the pool to say which models this machine holds"
        return None, f"waiting for {', '.join(missing)} to be on disk before starting vLLM"

    def wants_restart(self) -> bool:
        """Should this agent's vLLM be started again — for the models it holds, once every one is
        on disk and they are not what it was started with (a model held that has landed, one no
        longer held, or one deleted), or because a start that served has since lost every engine
        while its router lives on (`_crashed`)? Only once every model held is on disk: a start
        takes minutes, and each model landing in turn would cost one."""
        record = self._current()
        if record is None or record.get("stopped") or not self.ours():
            return False
        if self._crashed(record):
            return True
        held = set(self.held() or ())
        if not held:
            return False
        on_disk = complete_tags(self.models_dir)
        return held <= on_disk and set(record.get("models") or []) != held

    def _crashed(self, record: dict[str, Any]) -> bool:
        """Has the start on `record`, which served, lost every engine process since — a driver
        reset, a process killed for memory — with nothing left to start one in turn? With several
        models, or one on several cards, the router outlives them, and the vLLM would otherwise
        pass for running for ever. A start that never served is never started again for it: its
        models' failure is said, and stands until they change (`start`)."""
        if not record.get("served"):
            return False
        launched = vllm_state.read_record(self.models_dir) or {}
        engines = [e.get("pid") for e in launched.get("engines") or [] if isinstance(e, dict)]
        pids = [pid for pid in engines if isinstance(pid, int)]
        if not pids:
            return False
        helper = (launched.get("in_turn") or {}).get("pid")
        if isinstance(helper, int) and host_platform.alive(helper, os_name=self.os_name) and (
                self._is_ours(helper, "helper", record) is not False):
            return False
        return not any(
            host_platform.alive(pid, os_name=self.os_name) and self._is_ours(pid, "engine", record) is not False
            for pid in pids
        )

    # --- what was started ---

    def _current(self) -> Optional[dict[str, Any]]:
        if not self._record_read:
            self._record_read = True
            try:
                record = json.loads((self.state_dir / RECORD_FILE).read_text())
            except (OSError, ValueError):
                record = None
            if isinstance(record, dict) and isinstance(record.get("program"), str) and isinstance(record.get("models"), list):
                self._record = record
        return self._record

    def _write(self, name: str, value: Any) -> None:
        """A file of the engine's own directory, written whole or not at all."""
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = self.state_dir / name
        draft = target.with_name(target.name + ".new")
        draft.write_text(json.dumps(value, indent=2))
        os.replace(draft, target)

    def _remember(self, record: dict[str, Any]) -> None:
        self._write(RECORD_FILE, record)
        self._record, self._record_read = record, True

    def _launched(self) -> list[tuple[int, str]]:
        """Every process the launcher recorded starting, and what it is: an engine, the router, or
        the helper that starts a card's models in turn. None of what it recorded before the machine
        last restarted: those numbers name nothing it started (`vllm_state.another_boot`)."""
        found: list[tuple[int, str]] = []
        launched = vllm_state.read_record(self.models_dir) or {}
        if vllm_state.another_boot(launched, vllm_state.boot_id()):
            return found
        for engine in launched.get("engines") or []:
            if isinstance(engine, dict) and isinstance(engine.get("pid"), int):
                found.append((engine["pid"], "engine"))
        for helper, kind in ((launched.get("proxy"), "router"), (launched.get("in_turn"), "helper")):
            if isinstance(helper, dict) and isinstance(helper.get("pid"), int):
                found.append((helper["pid"], kind))
        return found

    def _is_ours(self, pid: int, kind: str, record: dict[str, Any]) -> Optional[bool]:
        """Is process `pid` what this agent's launcher started — this version's `vllm serve` of a
        model in this models directory, its router for this models directory on this port, or its
        helper for this models directory (`vllm_state.is_engine`, `is_router`, `is_helper`)? None
        where its command line cannot be read now, which says nothing either way."""
        said = host_platform.command_line(pid, os_name=self.os_name)
        if said is None:
            return None
        if kind == "engine":
            return vllm_state.is_engine(said, self.models_dir, program=str(record["program"]))
        if kind == "router":
            return vllm_state.is_router(said, self.models_dir, port=self.port)
        return vllm_state.is_helper(said, self.models_dir)

    def ours(self, *, fresh: bool = False) -> bool:
        """Is the vLLM this agent started running now — any process of it, started by this agent
        process or the one before it, and still what it was? `fresh` looks again now. A look that
        could not be made is no answer: the last one stands, and is not kept any longer for it."""
        if self._starting:
            return True
        record = self._current()
        if record is None or record.get("stopped") or record.get("port") != self.port:
            self._confirmed = None
            return False
        if not fresh and self._confirmed is not None and time.monotonic() - self._confirmed < IDENTITY_TRUSTED_S:
            return True
        unknown = False
        for pid, kind in self._launched():
            if not host_platform.alive(pid, os_name=self.os_name):
                continue
            said = self._is_ours(pid, kind, record)
            if said:
                self._confirmed = time.monotonic()
                return True
            unknown = unknown or said is None
        if unknown and self._confirmed is not None:
            return True
        self._confirmed = None
        return False

    def running_version(self) -> Optional[str]:
        record = self._current()
        return record.get("version") if record is not None and self.ours() else None

    def served_when_started(self) -> list[str]:
        """The models the last start served — recorded once they did — until it is stopped."""
        record = self._current()
        if record is None or record.get("stopped"):
            return []
        return [model for model in record.get("served") or [] if isinstance(model, str)]

    # --- what answers on the port ---

    async def listening(self) -> bool:
        """Does anything at all answer on the port — vLLM or not?"""
        try:
            await self.client.get("/health", timeout=ANSWER_TIMEOUT_S)
        except httpx.ConnectError:
            return False
        except httpx.HTTPError:
            return True  # something took the connection, and did not answer as a server should
        return True

    async def foreign(self) -> bool:
        """Something this agent did not start answers on the port."""
        return not self.ours() and await self.listening()

    async def _served(self) -> Optional[set[str]]:
        """The models the engine — or the router in front of several — serves now; None if it
        does not answer."""
        try:
            response = await self.client.get("/v1/models", timeout=ANSWER_TIMEOUT_S)
            if response.status_code != 200:
                return None
            data = response.json().get("data") or []
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        return {entry["id"] for entry in data if isinstance(entry, dict) and isinstance(entry.get("id"), str)}

    # --- starting and stopping ---

    async def start(self, program: Path, version: str, timeout_s: float) -> Optional[str]:
        """Start vLLM with `program` for the models it holds, once every one is on disk, and wait
        for it to serve them. None once it does, or while it waits for the models (said in
        `waiting`) — and None too where the start began and no model serves (`_did_not_serve`).
        Otherwise why not — the version's own fault, or a start that could not begin: its program
        missing, its launcher refusing it, a record that could not be written, something else on
        the port, or this agent's vLLM already running — and then whatever was started is left
        for the caller to stop."""
        if self.ours(fresh=True):
            return "this agent's vLLM is already running"
        if await self.listening():
            return self.foreign_reason
        if not program.is_file():
            return f"{program} is not there"
        models, waiting = self.to_start()
        record = self._current()
        if (
            models is not None and record is not None and record.get("failed") and not record.get("served")
            and not record.get("stopped") and record.get("version") == version and record.get("models") == models
        ):
            models, waiting = None, self._not_again(record)
        if models is None:
            if waiting != self.waiting:
                log.info("vLLM %s is not started yet: %s", version, waiting)
            self.waiting = waiting
            return None
        self.waiting = None
        argv = self.launcher_argv(program, models)
        logs = self.state_dir / "logs"
        self._starting = True
        try:
            began = time.time()
            try:
                logs.mkdir(mode=0o700, parents=True, exist_ok=True)
                # The models to start, where the launcher reads them (`--models-from`).
                self._write(MODELS_FILE, models)
                # Recorded before the launcher runs, so an agent that restarts meanwhile still
                # knows what it was starting, and which version.
                self._remember({
                    "version": version, "program": str(program), "models": models, "argv": argv,
                    "port": self.port, "models_dir": str(self.models_dir), "started_at": began,
                    "log": str(logs / LOG_FILE),
                })
            except OSError as exc:
                return f"could not record what it starts in {self.state_dir}: {exc}"
            self._confirmed = None
            ran = await self.run(argv, env=self.environment(), timeout_s=LAUNCH_TIMEOUT_S, log=logs / LOG_FILE,
                                 os_name=self.os_name)
            launched = vllm_state.read_record(self.models_dir) or {}
            refused = launched.get("refused")
            if ran.code == 1 and refused and float(launched.get("at") or 0) >= began - 1:
                # The launcher refused the set as it stands — it does not fit the card, or cannot be
                # split as the pool asked: the models', as the pool configured them, not the version's.
                return self._did_not_serve(version, models, {model: str(refused) for model in models})
            if ran.code != 0:
                return f"its launcher did not start it — {_last_line(ran.output)}"
            return await self._serving_within(version, models, timeout_s)
        finally:
            self._starting = False

    def _not_again(self, record: dict[str, Any]) -> str:
        """Why the models a start served none of are not started again by that version yet."""
        failed = record.get("failed") or {}
        reasons = sorted(set(failed.values()))
        said = reasons[0] if len(reasons) == 1 else "; ".join(f"{model}: {why}" for model, why in sorted(failed.items()))
        return (
            f"vLLM {record.get('version')} did not serve {', '.join(sorted(failed))} when last started ({said}); "
            "started again when the models it holds change, or when it is restarted"
        )

    def _did_not_serve(self, version: str, models: Sequence[str], failed: dict[str, str],
                       served: Collection[str] = ()) -> Optional[str]:
        """A start that began and in which a model does not serve: that model's failure, said for
        it, never the version's — None, unless it cannot even be recorded.

        The version answered its own `vllm --version` when it was installed, so what keeps it from
        serving these models is theirs: the launcher refusing the set (it does not fit the card),
        starting none of it, each model exiting, or the wait running out — where its processes are
        left running, since they may yet serve. A model too large for the card and a version that
        cannot run on it look the same from here, and only the first is common: called the
        version's failure, it would cost a machine on its first version its vLLM until the next
        release (D153), for a model. So each model's failure is said for it — in `waiting`, and in
        the facts (`models_failed`) — and, where nothing served, the same models are not started
        again by this version until they change or it is stopped or restarted: each start holds
        the card for minutes. An upgrade that serves nothing is rolled back by the pool's check
        after the switch (supervisor §3.1)."""
        record = self._current() or {}
        update: dict[str, Any] = {"failed": {model: str(why) for model, why in sorted(failed.items())}}
        if served:
            update["served"] = sorted(served)
        try:
            self._remember({**record, **update})
        except OSError as exc:
            return f"could not record in {self.state_dir} that {', '.join(sorted(failed))} did not serve: {exc}"
        if served:
            log.warning("vLLM %s serves %s; not %s", version, sorted(served), failed)
            return None
        self.waiting = self._not_again(self._current() or {})
        log.warning("%s", self.waiting)
        return None

    async def _serving_within(self, version: str, models: Sequence[str], timeout_s: float) -> Optional[str]:
        """Wait for every model the launcher started to be served — or each one that is not to
        have failed, or the wait to run out (`_did_not_serve`). A start that served is recorded so
        (`served`), for a crash of every engine after it to be started again (`_crashed`)."""
        deadline = time.monotonic() + timeout_s
        while True:
            launched = vllm_state.read_record(self.models_dir) or {}
            if launched.get("refused"):
                return self._did_not_serve(version, models, {model: str(launched["refused"]) for model in models})
            started = {e["model"] for e in launched.get("engines") or [] if isinstance(e, dict) and e.get("model")}
            if not started:
                return self._did_not_serve(version, models, {
                    model: "the launcher started none of the models: none was complete on disk when it looked"
                    for model in models
                })
            served = await self._served() or set()
            if started <= served:
                try:
                    self._remember({**(self._current() or {}), "served": sorted(started)})
                except OSError as exc:
                    log.warning("could not record in %s that vLLM serves: %s", self.state_dir, exc)
                return None
            failed = vllm_state.failed_engines(self.models_dir, sorted(served))
            if started <= served | set(failed):
                return self._did_not_serve(version, models, failed, served=started & served)
            if not any(host_platform.alive(pid, os_name=self.os_name) for pid, _ in self._launched()):
                return self._did_not_serve(version, models, {
                    **{model: "everything the launcher started has exited" for model in started - served},
                    **failed,
                }, served=started & served)
            if time.monotonic() >= deadline:
                late = f"it did not serve within {timeout_s:.0f}s of starting"
                return self._did_not_serve(version, models, {
                    **{model: late for model in started - served}, **failed,
                }, served=started & served)
            await asyncio.sleep(self.poll_s)

    async def stop(self) -> None:
        """Stop the vLLM this agent started, if it runs — every process of it, by `gpm-agent
        vllm-stop` — and remember that it stopped, with the models it ran. Whatever runs that is
        not its own is left alone. Where that cannot be told now, or the stop fails, it raises
        (an OSError, as an error from the system is), to be asked again."""
        record = self._current()
        if record is None or record.get("stopped"):
            return
        if not self.ours(fresh=True):
            if any(host_platform.alive(pid, os_name=self.os_name) and self._is_ours(pid, kind, record) is None
                   for pid, kind in self._launched()):
                raise OSError("whether the processes it started are still vLLM could not be read now")
        else:
            logs = self.state_dir / "logs"
            logs.mkdir(mode=0o700, parents=True, exist_ok=True)
            ran = await self.run(self.stop_argv(), env=self.environment(), timeout_s=STOP_TIMEOUT_S,
                                 log=logs / LOG_FILE, os_name=self.os_name)
            if ran.code != 0:
                raise OSError(f"gpm-agent vllm-stop did not stop it: {_last_line(ran.output)}")
        self._remember({**record, "stopped": True, "stopped_at": time.time()})
        self._confirmed = None
