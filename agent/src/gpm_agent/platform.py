"""What differs between the operating systems the agent runs on (D154).

The agent runs on macOS, Linux and Windows, as its machine owner's user. Most of it is the same
everywhere; what is not lives here, in one place, so the rest of the agent never asks which
system it is on:

- **Where installed engines live** — the per-user place each system keeps an application's data.
- **Starting a program the agent runs** (whisper.cpp's server, an installed Ollama) so that it
  outlives the agent: its own session on POSIX, its own process group with no console window on
  Windows. Always a fixed argument list built by the caller, never a shell.
- **Stopping one it started**, with whatever it started: its process group asked first, then
  made to, on POSIX; its process tree ended on Windows, where a program has no signal to be
  asked with.
- **Recognising its own process** after the agent restarts, by the process's command line — a
  process number from a file is only a number, and may since belong to something else. Ours runs
  exactly the argument list it was started with (`started_with`).
- **Keeping a directory private**: this user's alone (0700) on POSIX; on Windows, where Python
  reports no owner and no such mode, by where it lives, under the user's profile.
- **Checking a program's signature** where the system has one: `codesign` on macOS,
  Authenticode on Windows, each with the publisher pinned by the caller (D152).
- **Running a command and waiting for it** (`run_command`) — a build, or a check of a program the
  agent installed — at low priority if asked, and ended, with everything it started, when the
  wait is given up.

Every command run here is a literal argument list written in this module — `ps`, `taskkill`,
`codesign`, and PowerShell with a fixed script — and none takes text from the pool; `start` and
`run_command` run their caller's own fixed argument list. The system is read from `sys.platform`
and can be named instead, so each branch is tested on any machine.
"""

from __future__ import annotations

import asyncio
import os
import platform as _stdlib_platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Optional, Sequence

MACOS, LINUX, WINDOWS = "macos", "linux", "windows"

#: A process this agent started, as `subprocess` hands it back.
Process = subprocess.Popen

#: Windows' process-creation flags, by value: the names exist in `subprocess` only on Windows.
#: Its own process group, so a console signal meant for the agent does not reach it; and no
#: console window — a hidden one, which the engine's own helper processes then share, where a
#: detached process (`DETACHED_PROCESS`) would have each of those open a window of its own.
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
WINDOWS_START_FLAGS = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
#: Windows' "below normal" priority, a process-creation flag there.
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

#: How far a command run at low priority is lowered, on POSIX's scale of 0 to 19: below
#: anything the machine's owner runs at the default, so a build on their laptop yields to them.
LOW_PRIORITY = 10

#: How much of a command's output `run_command` hands back: its last words, not its whole log.
_OUTPUT_TAIL_BYTES = 16384

#: How long a program is given to stop when asked, before it is made to.
STOP_GRACE_S = 10.0

#: Between looks while a program stops.
POLL_S = 0.125

#: The longest one of this module's own commands may take. `codesign` reads the whole file it
#: checks, and an engine's libraries run to hundreds of megabytes.
COMMAND_TIMEOUT_S = 120.0

#: The longest reading a process's command line may take: it is asked while the agent answers.
BRIEF_TIMEOUT_S = 5.0

#: How many words a system may put in front of a program's own argument list when it runs it: the
#: interpreter a script's `#!` line names, and the one argument that line may add.
SCRIPT_WORDS = 2


def system(sys_platform: Optional[str] = None) -> str:
    """`macos`, `linux` or `windows` — or, for any other system, its own name, which nothing in
    the agent's tables knows and so is refused where it matters."""
    name = sys.platform if sys_platform is None else sys_platform
    if name == "darwin":
        return MACOS
    if name.startswith("linux"):
        return LINUX
    if name in ("win32", "cygwin"):
        return WINDOWS
    return name


#: The processor names each system reports, in the names engines publish their builds under.
_ARCHITECTURES = {
    "x86_64": "amd64", "amd64": "amd64", "x64": "amd64",
    "arm64": "arm64", "aarch64": "arm64", "armv8": "arm64",
}


def architecture(machine: Optional[str] = None) -> Optional[str]:
    """`amd64` or `arm64`, or None for a processor no engine publishes a build for."""
    name = (_stdlib_platform.machine() if machine is None else machine).lower()
    return _ARCHITECTURES.get(name)


def engines_dir_for(os_name: str, environ: Mapping[str, str], home: PurePath) -> PurePath:
    """Where installed engines live, per user, on `os_name` — computed in that system's own path
    form, so every system's answer can be checked from any one of them.

    macOS: `~/Library/Application Support/gpm/engines`. Windows: `%LOCALAPPDATA%\\gpm\\engines`.
    Linux and the rest: `$XDG_DATA_HOME/gpm/engines`, or `~/.local/share/gpm/engines` where that is
    unset or, against the specification, not absolute.
    """
    if os_name == WINDOWS:
        local = environ.get("LOCALAPPDATA")
        base = PureWindowsPath(local) if local else PureWindowsPath(str(home)) / "AppData" / "Local"
        return base / "gpm" / "engines"
    home = PurePosixPath(str(home))
    if os_name == MACOS:
        return home / "Library" / "Application Support" / "gpm" / "engines"
    data = environ.get("XDG_DATA_HOME") or ""
    base = PurePosixPath(data) if data.startswith("/") else home / ".local" / "share"
    return base / "gpm" / "engines"


def engines_dir() -> Path:
    """Where installed engines live for the user running this agent, on this system."""
    return Path(engines_dir_for(system(), os.environ, Path.home()))


# --- running a program ---


@dataclass(frozen=True)
class Ran:
    """What one of this module's commands said: its exit code and its output, both streams."""

    code: Optional[int]
    output: str


Runner = Callable[..., Ran]


def _run(argv: list[str], env: Optional[Mapping[str, str]] = None, timeout_s: float = COMMAND_TIMEOUT_S) -> Ran:
    """Run one of this module's literal commands, without a shell. A command that cannot start or
    does not finish is an answer too — `code` None — never an exception."""
    try:
        done = subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
            timeout=timeout_s, check=False, env=None if env is None else dict(env),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Ran(None, str(exc) or type(exc).__name__)
    return Ran(done.returncode, (done.stdout or "") + (done.stderr or ""))


def _run_briefly(argv: list[str], env: Optional[Mapping[str, str]] = None) -> Ran:
    return _run(argv, env, timeout_s=BRIEF_TIMEOUT_S)


def start(
    argv: list[str],
    *,
    env: Mapping[str, str],
    log: Path,
    cwd: Path,
    note: str,
    os_name: Optional[str] = None,
    popen: Optional[Callable[..., Any]] = None,
) -> tuple[Process, int]:
    """Start `argv` — the caller's fixed argument list — so that it outlives this agent, its
    output appended to `log` after a line saying what was started. Returns the process and where
    in the log its own output begins. Raises OSError when the program cannot be started.

    On POSIX in its own session, so the agent's restart or its terminal closing does not take it
    down. On Windows in its own process group, with no console window (`WINDOWS_START_FLAGS`).
    """
    os_name = os_name or system()
    launch = popen or subprocess.Popen
    with open(log, "ab") as out:
        out.write(f"--- gpm-agent: {note} at {time.ctime()}\n".encode())
        out.flush()
        offset = out.tell()
        if os_name == WINDOWS:
            placement: dict[str, Any] = {"creationflags": WINDOWS_START_FLAGS}
        else:
            placement = {"start_new_session": True}
        process = launch(
            argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            cwd=cwd, env=dict(env), **placement,
        )
    return process, offset


def _windows_alive(pid: int) -> bool:
    """Asked of Windows itself: on Windows, `os.kill` with any signal ends the process."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    query_limited_information, still_active, access_denied = 0x1000, 259, 5
    handle = kernel32.OpenProcess(query_limited_information, False, pid)
    if not handle:
        # There, and not this user's to open — which the command-line check then says.
        return ctypes.get_last_error() == access_denied  # type: ignore[attr-defined]
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def alive(pid: int, *, os_name: Optional[str] = None) -> bool:
    """Is process `pid` running? Says nothing about *which* program it is: `command_line`."""
    if (os_name or system()) == WINDOWS:
        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # there, and not this user's — which the command-line check then says
    except OSError:
        return False
    return True


def command_line(pid: int, *, os_name: Optional[str] = None, run: Optional[Runner] = None) -> Optional[bytes]:
    """A process's command line, or None where it cannot be read: from /proc where there is one,
    else `ps`; on Windows, the process's record from PowerShell. Asked only of a process this
    agent process did not start itself — one it finds in a record after a restart — before
    signalling it, and to recognise it, a positive answer kept a moment (`started_with`)."""
    run = run or _run_briefly
    pid = int(pid)
    if (os_name or system()) == WINDOWS:
        said = run([
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            f'(Get-CimInstance Win32_Process -Filter "ProcessId={pid}").CommandLine',
        ])
        return said.output.strip().encode() if said.code == 0 and said.output.strip() else None
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        pass
    said = run(["ps", "-ww", "-o", "command=", "-p", str(pid)])
    return said.output.encode() if said.code == 0 else None


def arguments(pid: int, *, os_name: Optional[str] = None) -> Optional[list[str]]:
    """A process's arguments one by one, where the system lists them so (/proc); else None, and
    only its command line, joined, can be read (`command_line`)."""
    if (os_name or system()) == WINDOWS:
        return None
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return None
    listed = raw.decode(errors="surrogateescape").split("\0")
    return listed[:-1] if listed and listed[-1] == "" else listed


def started_with(
    pid: int, argv: Sequence[str], *, os_name: Optional[str] = None, run: Optional[Runner] = None
) -> bool:
    """Is process `pid` running exactly the argument list `argv`?

    Exactly — or as the system rewrites it to run a script: up to `SCRIPT_WORDS` words in front
    of the list, the interpreter a `#!` line names and the one argument it may add. Never a part
    of it found somewhere in another process's command line. Where the arguments cannot be read
    one by one, the command line is compared whole: joined by spaces, as `ps` prints it, or
    quoted as Windows reads a command line back.
    """
    os_name = os_name or system()
    expected = list(argv)
    if not expected:
        return False
    listed = arguments(pid, os_name=os_name)
    if listed is not None:
        extra = len(listed) - len(expected)
        return 0 <= extra <= SCRIPT_WORDS and listed[extra:] == expected
    said = command_line(pid, os_name=os_name, run=run)
    if said is None:
        return False
    line = said.decode(errors="replace").strip()
    # `list2cmdline` only quotes the list as Windows' own process creation did; it runs nothing.
    joined = subprocess.list2cmdline(expected) if os_name == WINDOWS else " ".join(expected)
    if line == joined:
        return True
    return line.endswith(f" {joined}") and 1 <= len(line[: -len(joined) - 1].split()) <= SCRIPT_WORDS


async def _until(done: Callable[[], bool], seconds: float, poll_s: float) -> bool:
    deadline = time.monotonic() + seconds
    while not done():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(poll_s)
    return True


async def stop(
    pid: int,
    *,
    process: Optional[Process] = None,
    os_name: Optional[str] = None,
    grace_s: float = STOP_GRACE_S,
    poll_s: float = POLL_S,
    run: Optional[Runner] = None,
) -> None:
    """Stop a program this agent started: `process` when this agent process started it, else the
    process number from its record — which the caller has already checked is that program.

    POSIX: asked (SIGTERM), and after `grace_s` made to (SIGKILL) — its whole process group, and
    waited for until every process in it has ended, since it was started as the leader of a
    session of its own: an engine's helper processes, or the real server a wrapper script runs,
    are in that group, and a wrapper that exits at the signal would leave the server behind.
    A process that does not lead its own group is signalled alone. An error from the system other
    than the process having gone — not this user's to signal — is raised, for the caller to say.
    Windows: its process tree ended (`taskkill /T /F`), as Windows has no signal to ask a program
    to stop that a program without a console receives.
    """
    os_name = os_name or system()

    def leader_gone() -> bool:
        if process is not None:
            return process.poll() is not None
        return not alive(pid, os_name=os_name)

    if leader_gone():
        return
    if os_name == WINDOWS:
        (run or _run)(["taskkill", "/PID", str(int(pid)), "/T", "/F"])
        if not await _until(leader_gone, grace_s, poll_s):
            # taskkill missing or refused: the process itself, at least.
            try:
                if process is not None:
                    process.terminate()
                else:
                    os.kill(pid, signal.SIGTERM)  # on Windows: TerminateProcess
            except OSError:
                pass
            await _until(leader_gone, grace_s, poll_s)
        return
    group = _leads_group(pid)

    def gone() -> bool:
        return leader_gone() and (not group or _group_gone(pid))

    _signal(pid, signal.SIGTERM, group=group, process=process)
    if await _until(gone, grace_s, poll_s):
        return
    _signal(pid, signal.SIGKILL, group=group, process=process)
    await _until(gone, grace_s, poll_s)


def _leads_group(pid: int) -> bool:
    """Does this process lead a process group numbered as it is? A program `start` started does:
    it leads a session of its own — and whatever it starts is in that group too. POSIX only."""
    try:
        return os.getpgid(pid) == pid
    except (ProcessLookupError, PermissionError):
        return False


def _signal(pid: int, signum: int, *, group: bool, process: Optional[Process] = None) -> None:
    """Signal a program's whole process group, or, for one that does not lead its own (asked while
    it ran), that process alone. A process already gone is not an error. POSIX only.

    The group's number cannot have gone to another process while any process is in that group, so
    it is signalled even once its leader has ended — a wrapper's server left behind."""
    try:
        if group:
            os.killpg(pid, signum)
        elif process is not None:
            process.send_signal(signum)
        else:
            os.kill(pid, signum)
    except ProcessLookupError:
        pass


def _group_gone(pid: int) -> bool:
    """Has every process in the group this program led ended? POSIX only."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def end_now(process: Process, *, os_name: Optional[str] = None) -> None:
    """End a program this agent has just started, and everything it started, at once — for a
    start that cannot be kept, such as one whose record could not be written, which nothing
    would stop once the agent restarts."""
    _end_now(process, os_name or system())


# --- a directory of this user's alone ---


def owned_by_this_user(found: os.stat_result, *, os_name: Optional[str] = None) -> bool:
    """Does a file, as `os.stat` or `os.lstat` found it, belong to the user running this agent?
    On Windows, where Python reports no owner, yes: a directory there is kept to its user by
    where it lives, under the user's profile."""
    if (os_name or system()) == WINDOWS:
        return True
    return found.st_uid == os.getuid()


def make_private(path: Path, found: os.stat_result, *, os_name: Optional[str] = None) -> None:
    """Make a directory, as `os.stat` found it, readable by this user alone (0700). Nothing on
    Windows, where such a mode means nothing (`owned_by_this_user`)."""
    if (os_name or system()) != WINDOWS and stat.S_IMODE(found.st_mode) != 0o700:
        os.chmod(path, 0o700)


# --- running a command and waiting for it ---


def _end_now(process: Process, os_name: str) -> None:
    """End a command `run_command` started, and everything it started: its session on POSIX —
    a build's compilers are its children — its process tree on Windows."""
    if process.poll() is not None:
        return
    if os_name == WINDOWS:
        _run(["taskkill", "/PID", str(int(process.pid)), "/T", "/F"])
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except OSError:
        pass
    process.wait()


def _tail(handle: Any, offset: int) -> str:
    """What a command wrote after `offset`, its last `_OUTPUT_TAIL_BYTES` at most."""
    handle.flush()
    handle.seek(0, os.SEEK_END)
    handle.seek(max(offset, handle.tell() - _OUTPUT_TAIL_BYTES))
    return handle.read().decode(errors="replace")


async def run_command(
    argv: list[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout_s: float = COMMAND_TIMEOUT_S,
    low_priority: bool = False,
    log: Optional[Path] = None,
    os_name: Optional[str] = None,
    poll_s: float = POLL_S,
    popen: Optional[Callable[..., Any]] = None,
) -> Ran:
    """Run `argv` — the caller's fixed argument list, never a shell line — and wait for it,
    without holding up the event loop. Answers what `_run` answers: the exit code (None when it
    could not start or did not finish) and its output — the tail of what it wrote, which goes to
    `log` when one is named (appended, after a line saying what ran) and is otherwise kept only
    until it is read here.

    `low_priority` lowers it below the machine owner's own programs: started through POSIX's
    `nice`, which every process it starts inherits, or as below-normal on Windows. It runs in its
    own session (or process group), so that when the wait is given up — the time is up, or the
    agent is stopping and cancels it — the command and everything it started end with it, and a
    half-finished build is not left running.
    """
    os_name = os_name or system()
    launch = popen or subprocess.Popen
    placement: dict[str, Any]
    if os_name == WINDOWS:
        placement = {"creationflags": WINDOWS_START_FLAGS | (BELOW_NORMAL_PRIORITY_CLASS if low_priority else 0)}
    else:
        placement = {"start_new_session": True}
        nice = shutil.which("nice") if low_priority else None
        if nice:
            # `nice` runs the command in its own place, so its process number is the command's.
            argv = [nice, "-n", str(LOW_PRIORITY), *argv]
    if log is not None:
        log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        sink: Any = open(log, "a+b")  # noqa: SIM115 - closed below, after the command ends
        sink.write(f"--- gpm-agent: {' '.join(argv)} at {time.ctime()}\n".encode())
    else:
        sink = tempfile.TemporaryFile()  # noqa: SIM115 - closed below, after the command ends
    with sink:
        sink.flush()
        offset = sink.tell()
        try:
            process = launch(
                argv, stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT,
                env=None if env is None else dict(env), **placement,
            )
        except OSError as exc:
            return Ran(None, str(exc) or type(exc).__name__)
        if low_priority and os_name != WINDOWS and not nice:
            # No `nice` program here: lowered from outside, as early as can be.
            try:
                os.setpriority(os.PRIO_PROCESS, process.pid, LOW_PRIORITY)
            except OSError:
                pass
        deadline = time.monotonic() + timeout_s
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    _end_now(process, os_name)
                    return Ran(None, f"{_tail(sink, offset)}\nit did not finish within {timeout_s:.0f}s and was stopped")
                await asyncio.sleep(poll_s)
        except BaseException:
            _end_now(process, os_name)  # given up on — by the agent stopping, most likely
            raise
        return Ran(process.returncode, _tail(sink, offset))


# --- signatures (D152) ---

#: Authenticode, read by PowerShell. The file's path reaches the script through its environment,
#: never through its text, so no file name can become part of a command.
_AUTHENTICODE_SCRIPT = (
    "$s = Get-AuthenticodeSignature -LiteralPath $env:GPM_SIGNED_FILE; "
    "$s.Status.ToString(); "
    "if ($s.SignerCertificate) { $s.SignerCertificate.Subject } else { '' }"
)

_TEAM = re.compile(r"^TeamIdentifier=(\S+)\s*$", re.MULTILINE)
_COMMON_NAME = re.compile(r'(?:^|,\s*)CN=(?:"([^"]*)"|([^,]*))')


def _last_words(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1][:200] if lines else "no output"


def signature_problem(
    path: Path,
    publisher: str,
    *,
    os_name: Optional[str] = None,
    run: Optional[Runner] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """None when `path` carries a valid signature by `publisher`; otherwise what is wrong.

    macOS: `codesign --verify --strict` must pass, and `codesign -dv` must name `publisher` as
    the signing team (`TeamIdentifier=`). Windows: `Get-AuthenticodeSignature` must say `Valid`,
    with `publisher` the signing certificate's common name. On any other system there is no
    signature to check, and asking is a mistake the caller made.
    """
    os_name = os_name or system()
    run = run or _run
    if os_name == MACOS:
        verified = run(["codesign", "--verify", "--strict", str(path)])
        if verified.code != 0:
            return f"{path.name} fails `codesign --verify --strict`: {_last_words(verified.output)}"
        described = run(["codesign", "-dv", str(path)])
        found = _TEAM.search(described.output) if described.code == 0 else None
        team = found.group(1) if found else None
        if team != publisher:
            return f"{path.name} is signed by team {team or '(none)'}, not {publisher}"
        return None
    if os_name == WINDOWS:
        environment = dict(os.environ if environ is None else environ)
        environment["GPM_SIGNED_FILE"] = str(path)
        said = run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _AUTHENTICODE_SCRIPT],
            env=environment,
        )
        lines = [line.strip() for line in said.output.splitlines()]
        status = lines[0] if said.code == 0 and lines else None
        if status != "Valid":
            return f"{path.name}'s Authenticode signature is {status or 'unreadable'}"
        subject = lines[1] if len(lines) > 1 else ""
        names = [quoted or plain.strip() for quoted, plain in _COMMON_NAME.findall(subject)]
        if publisher not in names:
            return f"{path.name} is signed by {subject or '(nobody named)'}, not {publisher}"
        return None
    return f"{os_name} has no program signatures this agent checks"
