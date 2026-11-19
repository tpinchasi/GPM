"""The `onnx` engine, as the host's agent runs it (S15, D179).

The agent puts each ONNX build on disk — a `.onnx` file at the root of a hub repository
(`owner/repo/file.onnx`), or a file the pool sent it (`upload/<sha256>`, D182) — and runs the pool's
own server (`onnx_server.py`) on loopback with every build it is told to hold, prepared as the pool
declared (`specs` in `PUT /models`). One server holds them all; it is started again whenever that
set changes, and a build that will not load is that build's failure, not the server's.

What runs the server: the version this agent installed (`onnx_install`, where its owner set
`install_engines`), or a Python the owner named (`engine_binary`) with onnxruntime, numpy and pillow
in it. Never a program from the pool.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Collection, Optional
from urllib.parse import urlsplit

import httpx

from . import modelhub
from . import platform as host_platform
from .onnx_install import program_in, server_argv, server_in
from .settings import loopback_engine_url_problem
from .vllm_state import state_dir_for
from .whisper_cpp import (
    _is_our_server,
    _last_error_line,
    _own_directory,
    _refused,
    _refusing,
    _server_environment,
)

DEFAULT_MODELS_PATH = "~/gpm-onnx-models"
DEFAULT_ENGINE_URL = "http://127.0.0.1:8190"
PROGRAM = "the ONNX server"
RECORD_FILE = "onnx-server.json"
PLAN_FILE = "onnx-plan.json"
LOG_FILE = "onnx-server.log"
START_TIMEOUT_S = 180.0
POLL_S = 0.25
ANSWER_TIMEOUT_S = 3.0
STOP_GRACE_S = 10.0
IDENTITY_TRUSTED_S = 2.0
#: An uploaded build's one file, in its own directory (D182).
UPLOADED_FILE = "model.onnx"

_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UPLOAD = re.compile(r"^upload/([0-9a-f]{64})$")


@dataclass(frozen=True)
class Build:
    """One ONNX build: a file in a hub repository, or one the pool sent."""

    tag: str
    repo: Optional[str]
    file: str

    @property
    def uploaded(self) -> bool:
        return self.repo is None

    @property
    def directory(self) -> str:
        return self.tag.replace("/", "__")


def build_of(tag: Any) -> Build:
    if isinstance(tag, str):
        uploaded = _UPLOAD.match(tag)
        if uploaded:
            return Build(tag=tag, repo=None, file=UPLOADED_FILE)
        parts = tag.split("/")
        if (len(parts) == 3 and ".." not in tag and parts[2].endswith(".onnx") and len(parts[2]) > len(".onnx")
                and all(_PART.match(p) and "__" not in p and not p.endswith("_") for p in parts)):
            return Build(tag=tag, repo=f"{parts[0]}/{parts[1]}", file=parts[2])
    raise _refused(
        f"{tag!r} is not an ONNX build: expected `owner/repo/file.onnx` — a hub repository and one `.onnx` file "
        "at its root — or `upload/<sha256>`, a file the pool sent"
    )


def build_in(directory_name: str) -> Optional[Build]:
    parts = directory_name.split("__")
    if len(parts) not in (2, 3):
        return None
    try:
        return build_of("/".join(parts))
    except Exception:  # noqa: BLE001 — not a build's directory: never reported, never touched
        return None


#: What the pool may declare of a build, as the agent checks it: the shape of `specs` (D179).
_SPEC_KEYS = {"engine", "spec", "max_batch", "needs", "port"}


def checked_spec(tag: str, said: Any) -> dict[str, Any]:
    """One build's declarations, checked as data — never a path, a URL or a command. The server
    reads them again as it prepares each input; here they are bounded and of the right shape."""
    if not isinstance(said, dict) or not set(said) <= _SPEC_KEYS or said.get("engine") != "onnx":
        raise ValueError(f"{tag}: not an ONNX build's declarations")
    spec = said.get("spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict) or not isinstance(spec.get("outputs"), dict):
        raise ValueError(f"{tag}: `spec` names its inputs and outputs")
    if not (1 <= len(spec["inputs"]) <= 64 and 1 <= len(spec["outputs"]) <= 64):
        raise ValueError(f"{tag}: from 1 to 64 inputs and outputs")
    if len(json.dumps(spec)) > 4 * 1024 * 1024:
        raise ValueError(f"{tag}: declarations of more than 4 MB")
    for name, decl in [*spec["inputs"].items(), *spec["outputs"].items()]:
        if not isinstance(name, str) or not isinstance(decl, dict):
            raise ValueError(f"{tag}: each input and output is a name and its declaration")
        kind = decl.get("as")
        if kind not in ("tensor", "image", "features", "scores", "vector", "value"):
            raise ValueError(f"{tag}: {name!r} is declared as {kind!r}")
        if not isinstance(decl.get("input", decl.get("output")), str):
            raise ValueError(f"{tag}: {name!r} names no tensor of the file")
    batch = said.get("max_batch", 1)
    if not isinstance(batch, int) or not 1 <= batch <= 1024:
        raise ValueError(f"{tag}: max_batch is a whole number from 1 to 1024")
    return {"spec": spec, "max_batch": batch}


class OnnxFacts:
    name = "onnx"
    #: Builds come from a hub or the pool, and the server cannot run until one is here.
    fetches_without_engine = True
    loads_by_restart = False
    splits_across_cards = False

    def __init__(self, settings: Any = None, *, hub_transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.settings = settings
        self.hub_transport = hub_transport
        #: The pool's declarations for each build it named, as last checked (`PUT /models`).
        self.specs: dict[str, dict[str, Any]] = {}
        #: The builds the pool wants served now.
        self._wanted: set[str] = set()
        self._process: Optional[host_platform.Process] = None
        self._record: Optional[dict[str, Any]] = None
        self._record_read = False
        self._confirmed: Optional[tuple[int, float]] = None
        self._lock = asyncio.Lock()
        self.installer: Any = None

    def use_installer(self, installer: Any) -> None:
        self.installer = installer

    def set_specs(self, specs: dict[str, dict[str, Any]]) -> None:
        self.specs = dict(specs)

    @staticmethod
    def checked_spec(tag: str, said: Any) -> dict[str, Any]:
        try:
            build_of(tag)
        except Exception as exc:  # noqa: BLE001 — the engine's refusal, said as the body's fault
            raise ValueError(str(exc)) from exc
        return checked_spec(tag, said)

    # --- where things are ---

    @property
    def _models_dir(self) -> Path:
        return Path(getattr(self.settings, "models_path", None) or DEFAULT_MODELS_PATH).expanduser()

    @property
    def _state_dir(self) -> Path:
        return state_dir_for(self._models_dir)

    def _prepare_state(self) -> Path:
        state = self._state_dir
        with _refusing(f"could not prepare this agent's state directory {state}", state):
            host_platform.private_dir(state, parents=True, exist_ok=True)
            _own_directory(state, follow=True)
        return state

    def _port(self) -> int:
        url = getattr(self.settings, "engine_url", "") or DEFAULT_ENGINE_URL
        problem = loopback_engine_url_problem(url)
        if problem:
            raise _refused(f"engine_url {url!r} {problem}; the ONNX server is started on 127.0.0.1")
        return int(urlsplit(url).port or 0)

    def _program(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """The interpreter, the server file it runs, and the version: the installer's, or the owner's Python."""
        if self.installer is not None:
            running = self.installer.record.running
            if running is None:
                return None, None, None
            version_dir = self.installer.program(running).parent.parent.parent
            python = program_in(version_dir)
            return (str(python) if python.is_file() else None), str(server_in(version_dir)), running
        named = getattr(self.settings, "engine_binary", None)
        if named:
            path = Path(named).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path), str(Path(__file__).with_name("onnx_server.py")), None
        return None, None, None

    def _no_program(self) -> str:
        if self.installer is not None:
            return ("ONNX Runtime is not installed here yet: this agent installs it when the pool names a version, "
                    "and its facts say how that stands (engine.install)")
        named = getattr(self.settings, "engine_binary", None)
        if named:
            return f"engine_binary names {named}, which is not a program here: a Python with onnxruntime, numpy and pillow"
        return ("nothing here runs ONNX builds: set install_engines in this agent's settings to let it install ONNX "
                "Runtime, or engine_binary to a Python with onnxruntime, numpy and pillow")

    # --- what was started ---

    def _current(self) -> Optional[dict[str, Any]]:
        if not self._record_read:
            try:
                record = json.loads((self._state_dir / RECORD_FILE).read_text())
            except (FileNotFoundError, ValueError):
                record = None
            except OSError:
                return None
            self._record_read = True
            if isinstance(record, dict) and isinstance(record.get("pid"), int):
                self._record = record
        return self._record

    def _remember(self, record: dict[str, Any]) -> None:
        target = self._state_dir / RECORD_FILE
        draft = target.with_name(target.name + ".new")
        draft.write_text(json.dumps(record, indent=2))
        draft.replace(target)
        self._record, self._record_read = record, True

    def _forget(self) -> None:
        self._record, self._record_read, self._process, self._confirmed = None, True, None, None
        with _refusing(f"could not remove {PROGRAM}'s record from {self._state_dir}"):
            (self._state_dir / RECORD_FILE).unlink(missing_ok=True)

    def _own_process(self, record: dict[str, Any]) -> Optional[host_platform.Process]:
        if self._process is not None and self._process.pid == record.get("pid"):
            return self._process
        return None

    def _alive(self, record: dict[str, Any], *, fresh: bool = False) -> bool:
        if record.get("stopped"):
            return False
        own = self._own_process(record)
        if own is not None:
            return own.poll() is None
        pid = record["pid"]
        if not host_platform.alive(pid):
            self._confirmed = None
            return False
        confirmed = self._confirmed if self._confirmed is not None and self._confirmed[0] == pid else None
        if not fresh and confirmed is not None and time.monotonic() - confirmed[1] < IDENTITY_TRUSTED_S:
            return True
        said = _is_our_server(pid, record)
        if said is None:
            return confirmed is not None
        self._confirmed = (pid, time.monotonic()) if said else None
        return said

    def _why_stopped(self, record: dict[str, Any]) -> str:
        own = self._own_process(record)
        code = own.returncode if own is not None else None
        line = _last_error_line(record.get("log"), int(record.get("log_offset") or 0))
        return f"{PROGRAM} exited{f' with code {code}' if code is not None else ''}" + (f": {line}" if line else "")

    # --- what the agent reports ---

    async def describe(self, client: httpx.AsyncClient) -> dict[str, Any]:
        python, _, version = self._program()
        record = self._current()
        alive = record is not None and self._alive(record)
        loaded, failed = (await self._served(client)) if alive else ([], {})
        if record is not None and not alive and not record.get("stopped"):
            failed = {tag: self._why_stopped(record) for tag in record.get("tags", [])}
        return {
            "name": self.name, "answers": alive and await self._answers(client), "version": version,
            "binary": python, "models_on_disk": self._on_disk(), "models_loaded": loaded, "models_failed": failed,
        }

    def _on_disk(self) -> list[dict[str, Any]]:
        root = self._models_dir
        found: list[dict[str, Any]] = []
        if not root.is_dir():
            return found
        for directory in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            build = build_in(directory.name)
            if build is None or not modelhub.is_complete(directory):
                continue
            size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file() and not f.name.startswith("."))
            found.append({"tag": build.tag, "size_bytes": size})
        return found

    async def _answers(self, client: httpx.AsyncClient) -> bool:
        try:
            return (await client.get("/health", timeout=ANSWER_TIMEOUT_S)).status_code == 200
        except httpx.HTTPError:
            return False

    async def _served(self, client: httpx.AsyncClient) -> tuple[list[str], dict[str, str]]:
        """What the server says it loaded, and why the rest did not load."""
        try:
            response = await client.get("/v1/models", timeout=ANSWER_TIMEOUT_S)
            rows = response.json().get("data") if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError, AttributeError):
            return [], {}
        loaded, failed = [], {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            if row.get("loaded") is True:
                loaded.append(row["id"])
            else:
                failed[row["id"]] = str(row.get("error") or "did not load")[:500]
        return loaded, failed

    # --- the things the agent may do to a model ---

    async def pull(self, client: httpx.AsyncClient, tag: str) -> AsyncIterator[tuple[int, int]]:
        build = build_of(tag)
        if build.uploaded:
            raise _refused(f"{tag} is a file the pool sends this machine; it has not arrived yet")
        async with httpx.AsyncClient(transport=self.hub_transport, timeout=httpx.Timeout(60.0, read=300.0),
                                     follow_redirects=True) as hub:
            try:
                async for done, total in modelhub.fetch(build.repo, self._models_dir, client=hub, only=[build.file],
                                                        into=build.directory):
                    yield done, total
            except modelhub.HubRefused as no:
                raise _refused(str(no)) from no

    async def hold(self, client: httpx.AsyncClient, tag: str, *, pinned: bool) -> None:
        build = build_of(tag)
        async with self._lock:
            if pinned:
                self._wanted.add(build.tag)
            else:
                self._wanted.discard(build.tag)
            with _refusing(f"{PROGRAM} could not be started with {build.tag}"):
                await self._serve(client)
            if pinned:
                loaded, failed = await self._served(client)
                if build.tag in failed:
                    raise _refused(failed[build.tag])
                if build.tag not in loaded:
                    raise _refused(f"{build.tag} is not loaded by {PROGRAM}")

    async def keep_only(self, client: httpx.AsyncClient, tags: Collection[str]) -> None:
        async with self._lock:
            self._wanted &= set(tags)
            with _refusing(f"{PROGRAM} could not be restarted"):
                await self._serve(client)

    def _plan(self) -> dict[str, dict[str, Any]]:
        """Every wanted build that is whole on disk and declared, with its file."""
        plan = {}
        for tag in sorted(self._wanted):
            build = build_of(tag)
            directory = self._models_dir / build.directory
            model_file = directory / build.file
            if tag in self.specs and modelhub.is_complete(directory) and model_file.is_file():
                plan[tag] = {"path": str(model_file), **self.specs[tag]}
        return plan

    async def _serve(self, client: httpx.AsyncClient) -> None:
        missing = sorted(t for t in self._wanted if t not in self.specs)
        if missing:
            raise _refused(f"the pool has not said what {missing[0]} takes in and gives back")
        plan = self._plan()
        digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        record = self._current()
        if record is not None and record.get("plan") == digest and self._alive(record):
            if await self._answered_within(client, record, max(15.0, float(record.get("started_at") or 0) + START_TIMEOUT_S - time.time())):
                return
        if self.installer is not None and self.installer.starting:
            return
        await self._stop()
        if not plan:
            return
        python, server, version = self._program()
        if python is None or server is None:
            raise _refused(self._no_program())
        if await self._answers(client):
            raise _refused(f"something else already answers on port {self._port()}, where this agent starts {PROGRAM}")
        record = self._start(python, server, plan, digest, version)
        if await self._answered_within(client, record, START_TIMEOUT_S):
            return
        if not self._alive(record):
            raise _refused(self._why_stopped(record))
        raise _refused(f"{PROGRAM} has not answered in {START_TIMEOUT_S:.0f}s; it is started afresh on the next attempt")

    async def _answered_within(self, client: httpx.AsyncClient, record: dict[str, Any], seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while self._alive(record):
            if await self._answers(client) and self._alive(record):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(POLL_S)
        return False

    def _start(self, python: str, server: str, plan: dict[str, Any], digest: str, version: Optional[str]) -> dict[str, Any]:
        state = self._prepare_state()
        plan_path = state / PLAN_FILE
        with _refusing(f"could not write {PROGRAM}'s plan in {state}"):
            draft = plan_path.with_name(PLAN_FILE + ".new")
            draft.write_text(json.dumps(plan))
            draft.replace(plan_path)
            logs = state / "logs"
            host_platform.private_dir(logs, exist_ok=True)
        argv = server_argv(python, server, self._port(), plan_path)
        try:
            process, offset = host_platform.start(argv, env=_server_environment(), log=logs / LOG_FILE, cwd=state,
                                                  note=f"starting {PROGRAM} with {len(plan)} build(s)")
        except OSError as exc:
            raise _refused(f"could not start {python}: {exc}") from exc
        self._process = process
        record = {"pid": process.pid, "tags": sorted(plan), "plan": digest, "argv": argv, "program": python,
                  "version": version, "created": host_platform.creation_time(process.pid), "log": str(logs / LOG_FILE),
                  "log_offset": offset, "started_at": time.time()}
        try:
            self._remember(record)
        except OSError as exc:
            host_platform.end_now(process)
            self._process = None
            raise _refused(f"could not record the {PROGRAM} it started in {state}: {exc}") from exc
        return record

    async def _stop(self, *, forget: bool = True) -> None:
        record = self._current()
        if record is None:
            return
        pid = record["pid"]
        own = self._own_process(record)
        with _refusing(f"could not stop {PROGRAM} (process {pid})"):
            if own is not None:
                await host_platform.stop(pid, process=own, grace_s=STOP_GRACE_S, poll_s=POLL_S / 2)
            elif not record.get("stopped") and host_platform.alive(pid):
                said = _is_our_server(pid, record)
                if said is None:
                    raise _refused(f"whether process {pid} is the {PROGRAM} it started could not be read now; it is not stopped")
                if said:
                    await host_platform.stop(pid, grace_s=STOP_GRACE_S, poll_s=POLL_S / 2)
        if forget:
            self._forget()
        else:
            self._process, self._confirmed = None, None
            with _refusing(f"could not record in {self._state_dir} that {PROGRAM} was stopped"):
                self._remember({**record, "stopped": True})

    async def delete(self, client: httpx.AsyncClient, tag: str) -> None:
        build = build_of(tag)
        async with self._lock:
            if build.tag in self._wanted:
                self._wanted.discard(build.tag)
                with _refusing(f"{PROGRAM} could not be restarted without {build.tag}"):
                    await self._serve(client)
            directory = self._models_dir / build.directory
            if directory.is_dir():
                try:
                    shutil.rmtree(directory)
                except OSError as exc:
                    raise _refused(f"could not remove {tag} from disk: {exc}") from exc

    def launch_environment(self, workers: int, models_held: int, context: Optional[int] = None,
                           cards_per_copy: int = 1) -> dict[str, str]:
        return {}

    def settings_from_environment(self, environment: Optional[dict[str, str]]) -> Optional[dict[str, Optional[int]]]:
        return None


class InstalledServer:
    """The ONNX server as the installer drives it (`installs.Server`): a switch stops it and starts
    the new version with the same builds; with none held, a version is checked by importing its runtime."""

    def __init__(self, engine: OnnxFacts, client: httpx.AsyncClient, *, run: Any = None) -> None:
        self.engine = engine
        self.client = client
        self.run = run or host_platform.run_command
        self.foreign_reason = (f"something this agent did not start already answers on port {engine._port()}, where it "
                               f"starts {PROGRAM}: stop it, or give this agent another engine_url")
        self._checked: set[str] = set()

    def ours(self, *, fresh: bool = False) -> bool:
        record = self.engine._current()
        return record is not None and self.engine._alive(record, fresh=fresh)

    def running_version(self) -> Optional[str]:
        record = self.engine._current()
        return record.get("version") if record is not None and self.engine._alive(record) else None

    async def foreign(self) -> bool:
        return not self.ours() and await self.engine._answers(self.client)

    def pool_environment(self) -> Optional[dict[str, str]]:
        return None

    def set_pool_environment(self, environment: dict[str, str]) -> bool:
        return False

    async def start(self, program: Path, version: str, timeout_s: float) -> Optional[str]:
        from .engines import EngineRefused

        engine = self.engine
        async with engine._lock:
            plan = engine._plan()
            if not plan:
                if str(program) in self._checked:
                    return None
                if not program.is_file():
                    return f"{program} is not there"
                checked = await self.run([str(program), "-I", "-c", "import numpy, PIL, onnxruntime"],
                                         env=_server_environment(), timeout_s=120.0)
                if checked.code != 0:
                    return f"ONNX Runtime {version} does not import here: {checked.output.strip()[-300:]}"
                self._checked.add(str(program))
                return None
            try:
                await engine._stop()
                if await engine._answers(self.client):
                    return self.foreign_reason
                version_dir = program.parent.parent.parent
                digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
                record = engine._start(str(program), str(server_in(version_dir)), plan, digest, version)
            except EngineRefused as no:
                return str(no)
            if await engine._answered_within(self.client, record, timeout_s):
                return None
            return engine._why_stopped(record) if not engine._alive(record) else \
                f"it did not answer within {timeout_s:.0f}s of starting"

    async def stop(self) -> None:
        async with self.engine._lock:
            await self.engine._stop(forget=False)
