"""whisper.cpp's `whisper-server`, run by this agent (D148).

The third engine differs from the other two in the one way that matters most here: **nobody else
starts it.** Ollama is the machine owner's service (unless the owner lets the agent install and
run it, D150), and vLLM is started by the machine's restart script; whisper.cpp's server is a
plain program that serves the one file it was started with,
cannot say which file that is, and has no authentication of its own. So this agent fetches the
build, starts the server on loopback with it, and reports what it is serving — the pool asks for
a model set and never learns there is a process to start.

What keeps that inside the agent's bounds (D40, D41):

- **The pool names a build, never a path or a program.** A build is one file in a model-hub
  repository, `owner/repo/file.bin`. It must be listed at the repository's root, and lands in a
  directory of its own under the models directory, named after it, so nothing the pool says
  climbs out of that directory.
- **The program is the owner's, or the one this agent installed.** `whisper-server` on this
  agent's PATH, or the absolute path in the agent's own `engine_binary` setting — or, where the
  machine's owner lets the agent install engines (D150), the version its installer runs, built
  here or taken from whisper.cpp's own Windows archives (`whisper_build`), and never either of
  the others. It is started with an argument list written here — the file, the port from the
  agent's own settings, and constants — never through a shell.
- **On loopback, always.** The server has no authentication, so it is started on 127.0.0.1 and
  the agent's settings refuse an engine address that is anything else; the pool reaches it on
  this machine or through a tunnel.
- **Its state is kept beside the models directory, never in it** (D139): the record of what was
  started, its log, and a directory of its own where the server writes each recording it is sent
  — readable by this user alone, and emptied before every start and after every stop, since the
  server leaves a recording it could not convert behind.

Holding a build is starting the server with it, and the agent's ordinary loop drives the rest: a
server that has stopped is not serving, so the next pass holds it again — the restart when it
stops — and a build that will not start is left alone for a minute before the next attempt.

Where the agent installs whisper.cpp, its installer drives the same server through
`InstalledServer`: a version switch stops it and starts the new program with the file it was
serving (D153) — with the same arguments, and the same uploads directory, emptied around it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Collection, Iterator, Optional
from urllib.parse import urlsplit

import httpx

from . import modelhub
from . import platform as host_platform
from .settings import loopback_engine_url_problem
from .vllm_state import state_dir_for

#: The program whisper.cpp installs for serving, looked for on this agent's PATH when the owner
#: has not named one.
PROGRAM = "whisper-server"

#: Where this engine's builds live when the agent's settings do not say.
DEFAULT_MODELS_PATH = "~/gpm-whisper-models"

#: OpenAI's transcription path. The server answers its own `/inference` unless told otherwise,
#: and that is a path the pool never forwards.
INFERENCE_PATH = "/v1/audio/transcriptions"

#: How long a server just started is given to answer `GET /`. Loading even the largest build onto
#: the accelerator takes seconds; two minutes means something is wrong. Generous and finite, for
#: the reason the other engines' loads are: one stuck start must not stall a host silently.
START_TIMEOUT_S = 120.0

#: Between looks while a server starts or stops.
POLL_S = 0.25

#: The longest one `GET /` may take before the server is called not answering. Short, because
#: the facts ask it on every pass.
ANSWER_TIMEOUT_S = 3.0

#: A server that is running this build and has gone quiet is given this long to answer again
#: before it is started afresh — a machine busy transcribing can be slow to answer anything, and
#: restarting it would drop the recording it is working on.
QUIET_GRACE_S = 15.0

#: How long a server is given to stop when asked, before it is made to.
STOP_GRACE_S = 10.0

#: What was started, kept beside the models directory: process, build, log.
RECORD_FILE = "whisper-server.json"
LOG_FILE = "whisper-server.log"

#: The server's working directory and `--tmp-dir`, under the state directory: where it writes
#: each recording it is sent, to hand it to `ffmpeg`. Its own, so that emptying it touches nothing
#: else; readable by this user alone.
UPLOADS_DIR = "tmp"

#: How long a process found running is taken to be the server on the strength of one look at its
#: command line — well under one of the pool's passes, so a number that has gone to another
#: process is not mistaken for the server for long.
IDENTITY_TRUSTED_S = 2.0

#: Credentials the agent may hold — for the hub, and for GitHub's releases where it installs
#: engines. The server has no use for any of them, so it is started, and checked, without them.
_WITHHELD = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN")

#: How long `whisper-server --help` is given. It prints its usage and exits, loading nothing.
HELP_TIMEOUT_S = 30.0

#: One part of a build's name: a letter or digit, then letters, digits, dots, dashes and
#: underscores. The tag is flattened into one directory name with `__`, so a part may neither hold
#: `__` nor end in `_` (it would run into the separator), or the name could not be read back as
#: the tag it came from. The same rule as the pool's own check of a build's name.
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: whisper.cpp's file format.
_SUFFIX = ".bin"

#: How far back a log is read for the line that says why a server stopped.
_LOG_TAIL_BYTES = 16384

#: What whisper-server prints about each request it serves — among it the name the client gave
#: its recording. Never a reason, and never reported.
_PER_REQUEST = ("Received request:", "Successfully loaded", "Running whisper.cpp inference on", "processing '")

#: What a line of the log says when it is a reason.
_ERROR_WORDS = ("error", "failed", "not found")


def _refused(why: str) -> Exception:
    """The engine registry's refusal. Imported when needed: the registry imports this module."""
    from .engines import EngineRefused

    return EngineRefused(why)


@dataclass(frozen=True)
class Build:
    """One whisper.cpp build: a file in a hub repository."""

    tag: str
    repo: str
    file: str

    @property
    def directory(self) -> str:
        """Its own directory under the models directory, named after the whole tag."""
        return self.tag.replace("/", "__")


def build_of(tag: Any) -> Build:
    """The build a tag names, or a refusal saying what a build looks like.

    Exactly three parts: the repository as the hub names it, and a file at its root. Every part
    must read back from the one directory name the tag is flattened into (`_PART`).
    """
    parts = tag.split("/") if isinstance(tag, str) else []
    if (
        len(parts) == 3
        and ".." not in tag
        and all(_PART.match(part) and "__" not in part and not part.endswith("_") for part in parts)
        and parts[2].endswith(_SUFFIX)
        and len(parts[2]) > len(_SUFFIX)
    ):
        return Build(tag=tag, repo=f"{parts[0]}/{parts[1]}", file=parts[2])
    raise _refused(
        f"{tag!r} is not a whisper.cpp build: expected `owner/repo/file.bin` — a model-hub "
        "repository and one `.bin` file at its root, each part starting with a letter or digit, "
        "in letters, digits, dots, dashes and underscores, never ending in `_` or holding `__`"
    )


def build_in(directory_name: str) -> Optional[Build]:
    """The build a directory under the models directory holds, if it is one this agent made."""
    parts = directory_name.split("__")
    if len(parts) != 3:
        return None
    try:
        return build_of("/".join(parts))
    except Exception:  # noqa: BLE001 - not a build's directory: not reported, never touched
        return None


def server_argv(binary: str, model_file: Path, port: int, uploads: Path) -> list[str]:
    """The whole command the server is started with — a fixed argument list, never a shell line.

    Every part is the agent's: the owner's program or the one it installed, a file under the
    agent's models directory, the port from the agent's own settings, the agent's own directory
    for what it is sent. whisper.cpp 1.9.2 serves on 8080 and transcribes as English unless told
    otherwise, reads only WAV unless `--convert` lets it hand other audio to `ffmpeg` — writing
    each recording to `--tmp-dir` first, its working directory unless told — and ignores the
    model a request names.
    """
    return [
        binary, "-m", str(model_file),
        "--host", "127.0.0.1", "--port", str(port),
        "--inference-path", INFERENCE_PATH,
        "--convert", "--tmp-dir", str(uploads),
        "-l", "auto",
    ]


def _server_environment() -> dict[str, str]:
    """The agent's own environment, less the credentials the server has no use for."""
    return {name: value for name, value in os.environ.items() if name not in _WITHHELD}


def help_argv(program: str) -> list[str]:
    """The command that checks a program runs on this machine: its own `--help`, which prints
    its usage and exits 0 having loaded nothing. Fixed; only the program is the agent's."""
    return [program, "--help"]


async def program_runs(
    program: Path, run: Optional[Callable[..., Awaitable[host_platform.Ran]]] = None
) -> Optional[str]:
    """None when `program --help` exits 0 — the program starts here, its libraries and its
    accelerator's runtime found; otherwise what it said instead."""
    run = run or host_platform.run_command
    said = await run(help_argv(str(program)), env=_server_environment(), timeout_s=HELP_TIMEOUT_S)
    if said.code == 0:
        return None
    lines = [line.strip() for line in said.output.splitlines() if line.strip()]
    how = f"exited with code {said.code}" if said.code is not None else "did not run"
    return f"{program.name} --help {how}: {lines[-1][:300] if lines else 'it said nothing'}"


_VERSIONED_LIBRARY = re.compile(r"^libwhisper\.(?:so\.)?(\d+\.\d+\.\d+)(?:\.dylib)?$")
_VERSIONED_DIRECTORY = re.compile(r"^(?:.*[-_])?v?(\d+\.\d+\.\d+)$")


def installed_version(binary: str) -> Optional[str]:
    """The version whisper.cpp's installation names, or None where it names none.

    The server cannot say — it has no version flag and prints none — but an installation usually
    does: the shared library beside it carries the version in its name (`libwhisper.1.9.2.dylib`,
    `libwhisper.so.1.9.2`), as does a package manager's directory for it. Read from names on
    disk; nothing is run.
    """
    real = Path(binary).resolve()
    library = real.parent.parent / "lib"
    try:
        names = sorted(entry.name for entry in library.iterdir()) if library.is_dir() else []
    except OSError:
        names = []
    for name in names:
        found = _VERSIONED_LIBRARY.match(name)
        if found:
            return found.group(1)
    for part in reversed(real.parts[:-1]):
        found = _VERSIONED_DIRECTORY.match(part)
        if found:
            return found.group(1)
    return None


def _is_our_server(pid: int, record: dict[str, Any]) -> bool:
    """Is this process the server the record describes? A number from a file is only a number:
    the process may have ended and its number gone to another. Ours runs the argument list the
    record says it was started with (`platform.started_with`); an unrelated process does not."""
    recorded = record.get("argv")
    if not isinstance(recorded, list) or not recorded or not all(isinstance(part, str) for part in recorded):
        return False
    return host_platform.started_with(pid, recorded)


def _last_error_line(log: Optional[str], offset: int) -> Optional[str]:
    """The last line naming an error that a server wrote to its log since it was started — what
    the operator would read. whisper.cpp's last words are its reason: `error: failed to
    initialize whisper context`, or that `ffmpeg` is not found.

    Only a line that reads as a reason, and never one the server writes about a request: those
    carry the name a client gave its recording, and the reason travels to the pool. Indented
    lines are `ffmpeg` describing a recording — its streams and its tags — and are skipped too.
    """
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
        if not line.strip() or line[:1].isspace() or any(marker in line for marker in _PER_REQUEST):
            continue
        if any(word in line.lower() for word in _ERROR_WORDS):
            return line.strip()[:300]
    return None


@contextlib.contextmanager
def _refusing(doing: str) -> Iterator[None]:
    """An error from the system while `doing` is this engine's refusal, saying so — so it reaches
    the pool as the build's error, and the pause before the next attempt applies."""
    try:
        yield
    except OSError as exc:
        raise _refused(f"{doing}: {exc}") from exc


def _own_directory(path: Path, *, follow: bool) -> None:
    """Refuse a directory that is not one, or not this user's; make it this user's alone. Read
    without following a link, unless `follow` allows one."""
    found = os.stat(path) if follow else os.lstat(path)
    if not stat.S_ISDIR(found.st_mode):
        raise _refused(f"{path} is not a directory; this agent keeps {PROGRAM}'s state there")
    if not host_platform.owned_by_this_user(found):
        raise _refused(
            f"{path} belongs to another user; this agent keeps {PROGRAM}'s state, and the "
            "recordings it is sent, only in a directory of its own"
        )
    host_platform.make_private(path, found)


def _empty(directory: Path) -> None:
    """Remove the regular files directly in the uploads directory — recordings the server was
    sent. Nothing else is touched, and no link is followed."""
    try:
        found = os.lstat(directory)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(found.st_mode) or not host_platform.owned_by_this_user(found):
        return  # not the directory this agent made: nothing of its own to remove
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(entry.path)


class WhisperCppFacts:
    name = "whisper-cpp"
    #: Builds come from a hub, and the server cannot run until one is here.
    fetches_without_engine = True
    #: From the pool's side nothing is restarted: the agent starts the server itself when it
    #: holds a build, so the ordinary load step drives it.
    loads_by_restart = False
    #: One process, one card.
    splits_across_cards = False
    #: One build at a time: the server serves the one file it was started with, so a desired set
    #: naming two would restart it with each in turn on every pass. Refused in the models verb.
    serves_one_model = True

    def __init__(
        self, settings: Any = None, *, hub_transport: Optional[httpx.AsyncBaseTransport] = None
    ) -> None:
        self.settings = settings
        #: How the hub is reached. Only ever replaced by a test; the hub itself is fixed by the
        #: machine's own environment (`modelhub`).
        self.hub_transport = hub_transport
        #: The server this agent process started, while it lives. After the agent's own restart
        #: there is none, and the record on disk says what was started before.
        self._process: Optional[host_platform.Process] = None
        self._record: Optional[dict[str, Any]] = None
        self._record_read = False
        #: A process recorded before this agent started, found to be the server, and when —
        #: looked at again once that is more than `IDENTITY_TRUSTED_S` ago.
        self._confirmed: Optional[tuple[int, float]] = None
        self._versions: dict[str, Optional[str]] = {}
        #: One start, stop or removal at a time — the installer's too (`InstalledServer`).
        self._lock = asyncio.Lock()
        #: The installer, where the machine's owner lets this agent install whisper.cpp (D150).
        #: The program is then the version it runs, and never one from PATH or the settings.
        self.installer: Any = None

    def use_installer(self, installer: Any) -> None:
        """Take the program from `installer`'s running version from now on."""
        self.installer = installer

    # --- where things are ---

    @property
    def _models_dir(self) -> Path:
        return Path(getattr(self.settings, "models_path", None) or DEFAULT_MODELS_PATH).expanduser()

    @property
    def _state_dir(self) -> Path:
        return state_dir_for(self._models_dir)

    @property
    def _uploads_dir(self) -> Path:
        return self._state_dir / UPLOADS_DIR

    def _prepare_state(self) -> Path:
        """The state directory and the server's uploads directory, made or checked: each a
        directory of this user's, readable by it alone (0700), and the uploads emptied. The state
        directory may be a link the owner made; the uploads directory is never one."""
        state, uploads = self._state_dir, self._uploads_dir
        with _refusing(f"could not prepare this agent's state directory {state}"):
            state.mkdir(mode=0o700, parents=True, exist_ok=True)
            _own_directory(state, follow=True)
            uploads.mkdir(mode=0o700, exist_ok=True)
            _own_directory(uploads, follow=False)
            _empty(uploads)
        return uploads

    def _port(self) -> int:
        url = getattr(self.settings, "engine_url", "")
        problem = loopback_engine_url_problem(url)
        if problem:
            # The settings refuse this when loaded; said again here for one built otherwise.
            raise _refused(f"engine_url {url!r} {problem}; whisper-server is started on 127.0.0.1")
        return int(urlsplit(url).port or 0)

    def _binary(self) -> Optional[str]:
        """The program: the owner's `engine_binary` if set — and only that — else the one on PATH."""
        named = getattr(self.settings, "engine_binary", None)
        if named:
            path = Path(named).expanduser()
            return str(path) if path.is_file() and os.access(path, os.X_OK) else None
        return shutil.which(PROGRAM)

    def _program(self) -> tuple[Optional[str], Optional[str]]:
        """The program and its version. Where this agent installs whisper.cpp, the version its
        installer runs — none until one is installed; otherwise the owner's, as `_binary` finds it."""
        if self.installer is not None:
            running = self.installer.record.running
            if running is None:
                return None, None
            program = self.installer.program(running)
            return (str(program) if program.is_file() else None), running
        binary = self._binary()
        return binary, self._version(binary)

    def _no_binary(self) -> str:
        if self.installer is not None:
            running = self.installer.record.running
            if running:
                return (
                    f"whisper.cpp {running}, installed by this agent, has no {PROGRAM} at "
                    f"{self.installer.program(running)}: something other than this agent changed its "
                    "directory; switching to another version replaces it"
                )
            return (
                "whisper.cpp is not installed here yet: this agent installs it when the pool names a "
                "version, and its facts say how that stands (engine.install)"
            )
        named = getattr(self.settings, "engine_binary", None)
        if named:
            return (
                f"engine_binary in this agent's settings names {named}, which is not a program "
                "here; install whisper.cpp there, or correct engine_binary"
            )
        return (
            f"{PROGRAM} is not on this agent's PATH: install whisper.cpp (it provides "
            f"{PROGRAM}), or set engine_binary in the agent's settings to where it is"
        )

    def _version(self, binary: Optional[str]) -> Optional[str]:
        if binary is None:
            return None
        if binary not in self._versions:
            self._versions[binary] = installed_version(binary)
        return self._versions[binary]

    # --- what was started ---

    def _current(self) -> Optional[dict[str, Any]]:
        """What this agent last started here — from memory, or after the agent's own restart from
        its record, so a server that outlived the agent is still known as its own."""
        if not self._record_read:
            self._record_read = True
            try:
                record = json.loads((self._state_dir / RECORD_FILE).read_text())
            except (OSError, ValueError):
                record = None
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

    def _alive(self, record: dict[str, Any]) -> bool:
        """Is the recorded server still running? Asked of the process itself when this agent
        started it; otherwise of the system, with its identity looked at again whenever the last
        look is more than a moment old. A server the installer stopped to start another version
        is not, whatever has its number since."""
        if record.get("stopped"):
            return False
        own = self._own_process(record)
        if own is not None:
            return own.poll() is None
        pid = record["pid"]
        if not host_platform.alive(pid):
            return False
        confirmed = self._confirmed
        if confirmed is not None and confirmed[0] == pid and time.monotonic() - confirmed[1] < IDENTITY_TRUSTED_S:
            return True
        if _is_our_server(pid, record):
            self._confirmed = (pid, time.monotonic())
            return True
        return False

    def _why_stopped(self, record: dict[str, Any]) -> str:
        own = self._own_process(record)
        code = own.returncode if own is not None else None
        line = _last_error_line(record.get("log"), int(record.get("log_offset") or 0))
        reason = f"its whisper-server exited{f' with code {code}' if code is not None else ''}"
        if line:
            reason += f": {line}"
        if shutil.which("ffmpeg") is None:
            # whisper.cpp 1.9.2 started with --convert ends at once when it cannot run ffmpeg.
            reason += "; ffmpeg is not on this agent's PATH, and the server is started with --convert, which needs it"
        return reason

    # --- what the agent reports ---

    async def describe(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """The program, the builds on disk, and what the server is serving.

        A build is loaded only while the server this agent started runs **and** answers: the
        server cannot say which file it holds, and something else answering on the port is not
        it. A server that has stopped is reported as that build's failure, with its last words —
        unless the installer stopped it, to start another version with the same file.
        """
        binary, version = self._program()
        record = self._current()
        alive = record is not None and self._alive(record)
        answers = await self._answers(client)
        return {
            "name": self.name,
            "answers": answers,
            "version": version,
            "binary": binary,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "models_on_disk": self._on_disk(),
            "models_loaded": [record["tag"]] if record is not None and alive and answers else [],
            "models_failed": (
                {record["tag"]: self._why_stopped(record)}
                if record is not None and not alive and not record.get("stopped") else {}
            ),
        }

    def _on_disk(self) -> list[dict[str, Any]]:
        root = self._models_dir
        found: list[dict[str, Any]] = []
        if not root.is_dir():
            return found
        for directory in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            build = build_in(directory.name)
            if build is None or not modelhub.is_complete(directory):
                continue  # not a build, or one still arriving
            size = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file() and not f.name.startswith(".")
            )
            found.append({"tag": build.tag, "size_bytes": size})
        return found

    async def _answers(self, client: httpx.AsyncClient) -> bool:
        """`GET /` answered 200: the server is up and its model loaded. It has no `/health`."""
        try:
            response = await client.get("/", timeout=ANSWER_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _listening(self, client: httpx.AsyncClient) -> bool:
        """Does anything at all answer on the server's port?"""
        try:
            await client.get("/", timeout=ANSWER_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return True

    # --- the things the agent may do to a model (docs/spec/host-agent.md §4) ---

    async def pull(self, client: httpx.AsyncClient, tag: str) -> AsyncIterator[tuple[int, int]]:
        """Fetch the build's one file from the hub into its own directory, yielding (bytes done,
        bytes total). The engine's client is unused: this engine has no pull of its own."""
        build = build_of(tag)
        async with httpx.AsyncClient(
            transport=self.hub_transport, timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True
        ) as hub:
            try:
                async for done, total in modelhub.fetch(
                    build.repo, self._models_dir, client=hub, only=[build.file], into=build.directory
                ):
                    yield done, total
            except modelhub.HubRefused as no:
                raise _refused(str(no)) from no

    async def hold(self, client: httpx.AsyncClient, tag: str, *, pinned: bool) -> None:
        """Pinned: the server is running this build and answering — started, or started afresh,
        if not. Not pinned: a server running this build is stopped; nothing else is touched."""
        build = build_of(tag)
        async with self._lock:
            with _refusing(f"{PROGRAM} could not be {'started' if pinned else 'stopped'} with {build.tag}"):
                if pinned:
                    await self._serve(client, build)
                    return
                record = self._current()
                if record is not None and record.get("tag") == build.tag:
                    await self._stop()

    async def keep_only(self, client: httpx.AsyncClient, tags: Collection[str]) -> None:
        """Stop the server this agent started unless the build it runs is one of `tags` — the
        builds the pool wants held — whatever this agent's pins say. A pin is recorded only once
        a start is answered, so a server still starting, or left running silent, when its build
        left the set would otherwise be left running with nothing to stop it. A build the
        installer stopped the server of, to start another version with it, is forgotten too once
        it leaves the set: the new version then has no file to start with, and is checked alone."""
        async with self._lock:
            record = self._current()
            if record is not None and record.get("tag") not in tags:
                with _refusing(f"{PROGRAM} could not be stopped"):
                    await self._stop()

    async def _serve(self, client: httpx.AsyncClient, build: Build) -> None:
        directory = self._models_dir / build.directory
        model_file = directory / build.file
        if not modelhub.is_complete(directory) or not model_file.is_file():
            raise _refused(f"{build.tag} is not on disk yet; it is fetched before it is served")
        record = self._current()
        if record is not None and record.get("file") == str(model_file) and self._alive(record):
            # Already running this build: given the rest of its start window to answer, or, once
            # that has passed, a short grace — and started afresh only if it stays silent.
            window = max(QUIET_GRACE_S, float(record.get("started_at") or 0) + START_TIMEOUT_S - time.time())
            if await self._answered_within(client, record, window):
                return
        if self.installer is not None and self.installer.starting:
            # The installer is stopping or starting this server — a switch, a restart — and starts
            # it with the file it holds. Started here as well, the two would take turns at the
            # port; the next pass holds this build if that one is not it.
            return
        binary, version = self._program()
        if binary is None:
            raise _refused(self._no_binary())
        port = self._port()
        await self._stop()
        if await self._listening(client):
            raise _refused(
                f"something else already answers on port {port}, where this agent starts "
                f"{PROGRAM}; stop it, or give this agent another engine_url"
            )
        record = self._start(binary, model_file, port, build, version)
        if await self._answered_within(client, record, START_TIMEOUT_S):
            return
        if not self._alive(record):
            raise _refused(self._why_stopped(record))
        raise _refused(
            f"{PROGRAM} has not answered in {START_TIMEOUT_S:.0f}s of starting with {build.tag}; "
            "it is left running, and started afresh if it is still silent on the next attempt"
        )

    async def _answered_within(self, client: httpx.AsyncClient, record: dict[str, Any], seconds: float) -> bool:
        """True once the recorded server answers; False if it stops, or the time runs out."""
        deadline = time.monotonic() + seconds
        while self._alive(record):
            # An answer counts only from a server still running after it: something else on
            # the port would answer for a server that is on its way out.
            if await self._answers(client) and self._alive(record):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(POLL_S)
        return False

    def _start(
        self, binary: str, model_file: Path, port: int, build: Build, version: Optional[str] = None
    ) -> dict[str, Any]:
        uploads = self._prepare_state()
        logs = self._state_dir / "logs"
        log_path = logs / LOG_FILE
        argv = server_argv(binary, model_file, port, uploads)
        environment = _server_environment()
        with _refusing(f"could not open {PROGRAM}'s log {log_path}"):
            logs.mkdir(mode=0o700, exist_ok=True)
        try:
            # Its own session, so it outlives an agent restart and keeps serving — and leads a
            # process group of its own, which is what is stopped. It works in the uploads
            # directory, so whatever it writes there is emptied with them.
            process, offset = host_platform.start(
                argv, env=environment, log=log_path, cwd=uploads,
                note=f"starting {PROGRAM} with {build.tag}",
            )
        except OSError as exc:
            raise _refused(f"could not start {binary}: {exc}") from exc
        self._process = process
        record = {
            "pid": process.pid, "tag": build.tag, "file": str(model_file), "argv": argv,
            "program": binary, "version": version,
            "log": str(log_path), "log_offset": offset, "started_at": time.time(),
        }
        try:
            self._remember(record)
        except OSError as exc:
            # A server nothing records is one nothing would stop once this agent restarts.
            host_platform.end_now(process)
            self._process = None
            raise _refused(f"could not record the {PROGRAM} it started in {self._state_dir}: {exc}") from exc
        return record

    async def _stop(self, *, forget: bool = True) -> None:
        """Stop the server this agent started, if it runs — with whatever it started — and forget
        it, or, for the installer's switch, remember the build it served, marked stopped, so the
        next version is started with the same file; then empty its uploads. A process found only
        in the record is signalled only once its own command line, read now, says it is that
        server."""
        record = self._current()
        if record is None:
            return
        pid = record["pid"]
        own = self._own_process(record)
        with _refusing(f"could not stop {PROGRAM} (process {pid})"):
            if own is not None:
                await host_platform.stop(pid, process=own, grace_s=STOP_GRACE_S, poll_s=POLL_S / 2)
            elif not record.get("stopped") and host_platform.alive(pid) and _is_our_server(pid, record):
                await host_platform.stop(pid, grace_s=STOP_GRACE_S, poll_s=POLL_S / 2)
        if forget:
            self._forget()
        else:
            self._process, self._confirmed = None, None
            with _refusing(f"could not record in {self._state_dir} that {PROGRAM} was stopped"):
                self._remember({**record, "stopped": True})
        with _refusing(f"could not empty {PROGRAM}'s uploads in {self._uploads_dir}"):
            _empty(self._uploads_dir)

    def _held(self) -> Optional[tuple[Build, Path]]:
        """The build this agent's server serves, or served until the installer stopped it, and its
        file — while that file is still whole on disk."""
        record = self._current()
        if record is None:
            return None
        try:
            build = build_of(record.get("tag"))
        except Exception:  # noqa: BLE001 - a record that names no build holds nothing
            return None
        directory = self._models_dir / build.directory
        model_file = directory / build.file
        if str(model_file) != record.get("file") or not modelhub.is_complete(directory) or not model_file.is_file():
            return None
        return build, model_file

    async def delete(self, client: httpx.AsyncClient, tag: str) -> None:
        """Remove a build from disk — stopping the server first if it is serving it."""
        build = build_of(tag)
        async with self._lock:
            record = self._current()
            if record is not None and record.get("tag") == build.tag:
                with _refusing(f"{PROGRAM} could not be stopped before {build.tag} is removed"):
                    await self._stop()
            directory = self._models_dir / build.directory
            if not directory.is_dir():
                return
            try:
                shutil.rmtree(directory)
            except OSError as exc:
                raise _refused(f"could not remove {tag} from disk: {exc}") from exc

    # --- the pool's numbers ---

    def launch_environment(
        self, workers: int, models_held: int, context: Optional[int] = None, cards_per_copy: int = 1
    ) -> dict[str, str]:
        """Nothing: the server serves one recording at a time and one build, and is started by
        this agent with its own arguments — none of the pool's numbers has a name here."""
        return {}

    def settings_from_environment(self, environment: Optional[dict[str, str]]) -> Optional[dict[str, Optional[int]]]:
        return None


class InstalledServer:
    """whisper-server as the installer drives it (`installs.Server`), where this agent installs
    whisper.cpp (D150, D153). The same server the engine's model work holds builds with — one
    record, one lock — started from the version the installer names.

    What a switch needs that holding a build does not: stopping the server while keeping the
    build it served, so that the next version is started with **the same file** the pool chose
    (`PUT /models` still chooses it); and, before any file is held, a check that a version runs
    here at all. With no file there is nothing to start, so that check is the program's own
    `--help` — which loads nothing, but does load the program, its libraries and its
    accelerator's runtime: a build for the wrong card or a missing library fails it. The version
    is then recorded as running, and the server is started with it once the pool holds a build.
    """

    def __init__(
        self,
        engine: WhisperCppFacts,
        client: httpx.AsyncClient,
        *,
        run: Optional[Callable[..., Awaitable[host_platform.Ran]]] = None,
    ) -> None:
        self.engine = engine
        self.client = client
        #: How `--help` is run — only ever replaced by a test.
        self.run = run or host_platform.run_command
        port = urlsplit(getattr(engine.settings, "engine_url", "")).port
        self.foreign_reason = (
            f"something this agent did not start already answers on port {port}, where it starts "
            f"{PROGRAM}: stop the whisper-server you run yourself, or give this agent another engine_url"
        )
        #: Programs that answered `--help` here, so one with no file to serve is checked once.
        self._runs: set[str] = set()

    def ours(self) -> bool:
        record = self.engine._current()
        return record is not None and self.engine._alive(record)

    def running_version(self) -> Optional[str]:
        record = self.engine._current()
        return record.get("version") if record is not None and self.engine._alive(record) else None

    async def foreign(self) -> bool:
        return not self.ours() and await self.engine._listening(self.client)

    def pool_environment(self) -> Optional[dict[str, str]]:
        """None: none of the pool's numbers has a name for this server (`launch_environment`)."""
        return None

    def set_pool_environment(self, environment: dict[str, str]) -> bool:
        return False

    async def start(self, program: Path, version: str, timeout_s: float) -> Optional[str]:
        """Start `program` with the file this agent's server holds, and wait for it to answer;
        with no file held, check that `program` runs. None once it does; otherwise why not — and
        then whatever was started is left for the caller to stop."""
        from .engines import EngineRefused

        engine = self.engine
        async with engine._lock:
            held = engine._held()
            if held is None:
                return await self._check(program)
            build, model_file = held
            record = engine._current()
            try:
                if record is not None and engine._alive(record):
                    # Started meanwhile, from the version before, by a pass holding this build:
                    # this start is the installer's, and replaces it.
                    await engine._stop(forget=False)
                if await engine._listening(self.client):
                    return self.foreign_reason
                if not program.is_file():
                    return f"{program} is not there"
                # The same command a pass holding the build starts — its uploads directory
                # emptied first, as `_stop` empties it after — with this version's program.
                record = engine._start(str(program), model_file, engine._port(), build, version)
            except EngineRefused as no:
                return str(no)
            if await engine._answered_within(self.client, record, timeout_s):
                return None
            if not engine._alive(record):
                return engine._why_stopped(record)
            return f"it did not answer within {timeout_s:.0f}s of starting with {build.tag}"

    async def _check(self, program: Path) -> Optional[str]:
        if str(program) in self._runs:
            return None
        if not program.is_file():
            return f"{program} is not there"
        problem = await program_runs(program, self.run)
        if problem is None:
            self._runs.add(str(program))
        return problem

    async def stop(self) -> None:
        """Stop the server, keeping the build it served for the start that follows, and empty its
        uploads. An error from the system is raised as the engine's refusal (`_refusing`)."""
        async with self.engine._lock:
            await self.engine._stop(forget=False)
