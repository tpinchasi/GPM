"""The agent starting with the machine: `gpm-agent init --start-at-login` and `gpm-agent startup`.

docs/spec/host-agent.md §2.3, D154. The machine's owner runs this, once, at a terminal; nothing on
the agent's protocol reaches it, and the pool cannot ask for it. Afterwards the agent starts as
its owner's user, with no administrator rights, whenever the owner's session does — and again
when it fails:

- **macOS**: a launchd agent, `~/Library/LaunchAgents/<name>.plist`, loaded into the owner's
  login session with `launchctl bootstrap gui/<uid>`.
- **Linux**: a systemd user unit, `~/.config/systemd/user/<name>.service`, enabled and started.
  With *lingering* on — which takes administrator rights to switch on, so this only prints the
  command unless the owner said `--admin` at a terminal — the owner's user manager runs from boot
  without anyone logging in.
- **Windows**: a scheduled task at the owner's logon, registered with `schtasks` from a task
  definition written beside the settings.

macOS and Windows start the agent when the owner logs in, not at boot: a machine nobody logs in
to — a box in a cupboard after a power cut — needs its system's automatic login for the agent to
come back by itself. A system service would not, but it needs administrator rights to install
and runs outside the owner's session and profile, where the models are (D154).

**One command everywhere** (`serve_command`): this agent's own interpreter, isolated, serving the
one settings file. A definition already there for *another* settings file is never replaced, so
two agents on one machine — one for each engine — live side by side, each under a name taken from
its settings file. Everything that decides what is written and run is a pure function of a
`Machine`, which a test supplies; `run_command` is the only place a process starts, always from an
argument list written here, never through a shell.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ntpath
import os
import platform
import plistlib
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from xml.parsers.expat import ExpatError
from xml.sax.saxutils import escape, unescape

#: The name every definition starts with: `gpm-agent` for the usual settings file, `agent.json`,
#: and `gpm-agent-<its name>` for any other — `gpm-agent-whisper` for `whisper.json`.
BASE_NAME = "gpm-agent"

#: How the agent's own module is run. `cli.py` runs `main()` when run as a program.
MODULE = "gpm_agent.cli"

#: How long a query of the system's service manager may take.
QUERY_TIMEOUT_S = 30.0


@dataclasses.dataclass(frozen=True)
class Ran:
    """How a command ended, and what it printed — empty when it printed to the owner's terminal."""

    returncode: int
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(argv: Sequence[str], *, on_terminal: bool = False) -> Ran:
    """Run one argument list, never a shell. Every command reaching here is written out in this
    module or in `prerequisites`; the only values in them are paths and names this machine gave.

    `on_terminal` hands the command the owner's terminal — for `sudo`'s password prompt and a
    package manager's progress — and captures nothing; otherwise its output is captured, with no
    input, and it is given `QUERY_TIMEOUT_S`."""
    options: dict[str, Any] = {} if on_terminal else {
        "capture_output": True, "text": True, "errors": "replace",
        "stdin": subprocess.DEVNULL, "timeout": QUERY_TIMEOUT_S,
    }
    try:
        done = subprocess.run(list(argv), check=False, **options)
    except FileNotFoundError:
        return Ran(127, f"{argv[0]} was not found")
    except subprocess.TimeoutExpired:
        return Ran(124, f"{argv[0]} did not answer within {QUERY_TIMEOUT_S:.0f} s")
    except OSError as exc:
        return Ran(126, f"{argv[0]} could not be run: {exc}")
    return Ran(done.returncode, ((done.stdout or "") + (done.stderr or "")).strip() if not on_terminal else "")


def shown(argv: Sequence[str], system: str) -> str:
    """An argument list as the owner would type it at this system's prompt."""
    return subprocess.list2cmdline(list(argv)) if system == "Windows" else shlex.join(argv)


class Files:
    """The file system, as this module uses it. A test supplies a dictionary instead, so each
    system's definitions are written and read back in that system's own paths."""

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def read(self, path: str) -> Optional[bytes]:
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def write(self, path: str, content: bytes) -> None:
        """Replace `path` whole: written beside it, then moved over it. Readable by all and
        writable by the owner only, as launchd and systemd require of a definition."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        draft = target.with_name(target.name + ".new")
        descriptor = os.open(draft, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(draft, target)

    def create_private(self, path: str) -> None:
        """Create `path` owner-only if it is not there — a log, before anything appends to it."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))

    def delete(self, path: str) -> None:
        Path(path).unlink(missing_ok=True)


@dataclasses.dataclass
class Machine:
    """What this module needs to know about the machine, and the way it acts on it — every part of
    which a test replaces."""

    #: As `platform.system()` names it: Darwin, Linux or Windows.
    system: str
    home: str
    #: The owner's login name; on Windows, `DOMAIN\\name` as Task Scheduler names a user.
    user: str
    #: The owner's user id, on macOS and Linux.
    uid: Optional[int]
    #: The interpreter running this agent, and the environment it belongs to.
    executable: str
    prefix: str
    #: The PATH of the owner's terminal, from which the agent's is taken.
    path: str
    environ: Mapping[str, str]
    #: Whether a person is at a terminal to answer `sudo`.
    interactive: bool
    run: Callable[..., Ran] = run_command
    which: Callable[[str], Optional[str]] = shutil.which
    files: Files = dataclasses.field(default_factory=Files)
    sleep: Callable[[float], None] = time.sleep

    @classmethod
    def this(cls) -> "Machine":
        system = platform.system()
        environ = dict(os.environ)
        if system == "Windows":
            name = environ.get("USERNAME", "")
            domain = environ.get("USERDOMAIN", "")
            user, uid = (f"{domain}\\{name}" if domain else name), None
        else:
            import pwd  # POSIX only, so imported here

            uid = os.getuid()
            user = pwd.getpwuid(uid).pw_name
        return cls(
            system=system, home=str(Path.home()), user=user, uid=uid,
            executable=sys.executable or "", prefix=sys.prefix, path=environ.get("PATH", ""),
            environ=environ, interactive=bool(sys.stdin and sys.stdin.isatty()),
        )

    @property
    def paths(self) -> Any:
        """The path rules of the machine described, whatever machine this runs on."""
        return ntpath if self.system == "Windows" else posixpath


# --- what is written, and what it runs: pure ---


def name_for(settings_path: str, system: str = "") -> str:
    """The definition's name, from the settings file's: `gpm-agent` for `agent.json`, else
    `gpm-agent-<name>` in lower-case letters, digits and dashes — valid as a launchd label, a
    systemd unit and a task name alike."""
    paths = ntpath if system == "Windows" else posixpath
    stem = paths.splitext(paths.basename(settings_path))[0]
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    if slug == "agent":
        return BASE_NAME
    if not slug:  # a name with none of those characters: the file's path decides instead
        slug = hashlib.sha256(settings_path.encode()).hexdigest()[:8]
    return f"{BASE_NAME}-{slug}"


def windowless(executable: str, exists: Callable[[str], bool]) -> str:
    """`pythonw.exe` beside a Windows interpreter, where there is one: the same interpreter
    without a console window — a window that would stop the agent when closed."""
    folder, program = ntpath.split(executable)
    if program.lower() == "python.exe":
        candidate = ntpath.join(folder, "pythonw.exe")
        if exists(candidate):
            return candidate
    return executable


def log_path(machine: Machine, settings_path: str) -> str:
    """Where the agent's output goes when nothing else keeps it — beside its settings. On Linux
    the journal keeps it (`journalctl --user -u <name>`)."""
    paths = machine.paths
    return paths.join(paths.dirname(settings_path), name_for(settings_path, machine.system) + ".log")


def serve_command(machine: Machine, settings_path: str) -> list[str]:
    """What starts this agent: the interpreter running now, isolated, serving one settings file.

    The interpreter rather than the `gpm-agent` program: it is an absolute path on every system,
    and it belongs to the very environment this package is installed in, so an upgrade by the
    owner's package manager (`uv tool upgrade gpm-agent`) is what the next start runs. Isolated
    (`-I`): no `PYTHON*` variables, no user site, and nothing from the working directory, so what
    runs is what that environment holds. On Windows it is `pythonw.exe`, which has no window to
    close and no standard streams, so the agent is told where to write its output.
    """
    executable = machine.executable
    if machine.system == "Windows":
        executable = windowless(executable, machine.files.exists)
    command = [executable, "-I", "-m", MODULE, "--config", settings_path, "serve"]
    if machine.system == "Windows":
        command += ["--log-file", log_path(machine, settings_path)]
    return command


def agent_path(machine: Machine) -> str:
    """The PATH the agent is started with: the owner's, as their terminal has it, without relative
    entries — the agent finds `ffmpeg`, `whisper-server` and `nvidia-smi` where the owner's own
    `gpm-agent serve` would, which launchd's and systemd's bare default PATH would not. Only PATH:
    no other variable is copied, so no credential of the owner's is ever written into a file."""
    paths = machine.paths
    separator = ";" if machine.system == "Windows" else ":"
    kept: list[str] = []
    for entry in machine.path.split(separator):
        if entry and paths.isabs(entry) and entry not in kept:
            kept.append(entry)
    return separator.join(kept)


def launchd_plist(name: str, command: Sequence[str], *, working_directory: str, path: str, log: str) -> bytes:
    """A launchd agent: started at login (`RunAtLoad`), started again when it exits on failure
    (`KeepAlive` unless it ended cleanly, as when stopped), its output appended to `log`.

    `ProcessType` is `Interactive` — no resource limits, as for an app — because the agent starts
    the engines that serve the pool, which inherit its limits; left unset, launchd throttles a
    job's processor and disk use."""
    return plistlib.dumps({
        "Label": name,
        "ProgramArguments": list(command),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Interactive",
        "WorkingDirectory": working_directory,
        "EnvironmentVariables": {"PATH": path},
        "StandardOutPath": log,
        "StandardErrorPath": log,
    })


#: A word systemd takes as it is; anything else is quoted.
_PLAIN_UNIT_WORD = re.compile(r"[A-Za-z0-9_@+=:,./-]+")


def _unit_text(text: str) -> str:
    """Text in a unit, with systemd's `%` specifiers made literal."""
    return text.replace("%", "%%")


def _unit_quoted(text: str) -> str:
    """One quoted value in a unit: backslashes and quotes escaped, `%` made literal."""
    return '"' + _unit_text(text.replace("\\", "\\\\").replace('"', '\\"')) + '"'


def _unit_word(word: str) -> str:
    """One word of `ExecStart=`, as it is when plain, else quoted with `$` made literal too, since
    systemd expands variables there."""
    if _PLAIN_UNIT_WORD.fullmatch(word):
        return word
    return _unit_quoted(word.replace("$", "$$"))


#: How a unit says which settings file it serves, for `owner_of` to read back.
_UNIT_SETTINGS = "# Settings: "


def systemd_unit(command: Sequence[str], *, settings_path: str, working_directory: str, path: str) -> str:
    """A systemd user unit: started with the owner's user manager (`WantedBy=default.target`) and
    started again when it fails (`Restart=on-failure`); its output goes to the journal.

    `KillMode=process`: stopping or restarting the unit stops the agent and not the engine servers
    it started, which go on serving and which the agent finds again from its record when it is
    back (D148) — as they do under launchd, where they run in a session of their own. systemd's
    default would end every process the unit ever started, interrupting inference whenever the
    agent restarts."""
    return "\n".join([
        "# Written by `gpm-agent startup add`; `gpm-agent startup remove` takes it away.",
        f"{_UNIT_SETTINGS}{settings_path}",
        "[Unit]",
        f"Description=GPM host agent, {_unit_text(settings_path)}",
        "",
        "[Service]",
        "Type=simple",
        "ExecStart=" + " ".join(_unit_word(word) for word in command),
        f"WorkingDirectory={_unit_text(working_directory)}",
        f"Environment={_unit_quoted('PATH=' + path)}",
        "Restart=on-failure",
        "RestartSec=5",
        "KillMode=process",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])


#: How a task definition says which settings file it serves, for `owner_of` to read back.
_TASK_DESCRIPTION = "GPM host agent, "
_TASK_SETTINGS = re.compile(f"<Description>{_TASK_DESCRIPTION}(.*?)</Description>")


def task_xml(command: Sequence[str], *, user: str, settings_path: str, working_directory: str) -> bytes:
    """A Windows task definition: at the owner's logon, as the owner, with the rights of an
    ordinary user (`LeastPrivilege`) and only while they are logged on (`InteractiveToken`), so
    no password is ever stored.

    A definition rather than `schtasks /Create /SC ONLOGON /TR …`, because the short form keeps
    Task Scheduler's defaults, each wrong for an agent: it would stop the agent after three days,
    never start it on battery and stop it when the power cable comes out, run it — and the engines
    it starts — below normal priority, and fit the command line in 261 characters. Here the program
    and its arguments are separate fields, so nothing is split by a shell: the arguments are
    quoted by the rules the interpreter reads them back with, then escaped for XML.

    Encoded as UTF-16, the form Task Scheduler itself writes and reads."""
    arguments = subprocess.list2cmdline(list(command[1:]))
    body = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{_TASK_DESCRIPTION}{escape(settings_path)}</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{escape(user)}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(user)}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(command[0])}</Command>
      <Arguments>{escape(arguments)}</Arguments>
      <WorkingDirectory>{escape(working_directory)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""
    return body.replace("\n", "\r\n").encode("utf-16")


@dataclasses.dataclass(frozen=True)
class Entry:
    """What makes one agent start with the machine: a definition file and what it runs."""

    kind: str
    name: str
    file: str
    content: bytes
    command: list[str]
    #: The file the agent's output is appended to, where the system keeps none.
    log: Optional[str] = None


def schtasks(machine: Machine) -> str:
    """Windows' `schtasks`, by its full path under the system directory: a program by that name in
    the working directory, which Windows searches first, is never the one run."""
    root = machine.environ.get("SystemRoot") or machine.environ.get("SYSTEMROOT") or "C:\\Windows"
    return ntpath.join(root, "System32", "schtasks.exe")


def unit_directory(machine: Machine) -> str:
    configured = machine.environ.get("XDG_CONFIG_HOME", "")
    base = configured if posixpath.isabs(configured) else posixpath.join(machine.home, ".config")
    return posixpath.join(base, "systemd", "user")


def entry_for(machine: Machine, settings_path: str) -> Entry:
    """The definition for this machine's system, serving `settings_path` (absolute)."""
    name = name_for(settings_path, machine.system)
    command = serve_command(machine, settings_path)
    working_directory = machine.paths.dirname(settings_path)
    if machine.system == "Darwin":
        log = log_path(machine, settings_path)
        return Entry(
            kind="launchd agent", name=name,
            file=posixpath.join(machine.home, "Library", "LaunchAgents", f"{name}.plist"),
            content=launchd_plist(name, command, working_directory=working_directory,
                                  path=agent_path(machine), log=log),
            command=command, log=log,
        )
    if machine.system == "Linux":
        return Entry(
            kind="systemd user unit", name=f"{name}.service",
            file=posixpath.join(unit_directory(machine), f"{name}.service"),
            content=systemd_unit(command, settings_path=settings_path, working_directory=working_directory,
                                 path=agent_path(machine)).encode(),
            command=command,
        )
    if machine.system == "Windows":
        return Entry(
            kind="scheduled task", name=name,
            file=ntpath.join(working_directory, f"{name}.task.xml"),
            content=task_xml(command, user=machine.user, settings_path=settings_path,
                             working_directory=working_directory),
            command=command, log=log_path(machine, settings_path),
        )
    raise ValueError(f"no way to start with the machine on {machine.system}")


# --- the commands, per system: pure ---


def launchd_commands(machine: Machine, entry: Entry) -> dict[str, list[str]]:
    target = f"gui/{machine.uid}"
    return {
        "query": ["launchctl", "print", f"{target}/{entry.name}"],
        "stop": ["launchctl", "bootout", f"{target}/{entry.name}"],
        "start": ["launchctl", "bootstrap", target, entry.file],
    }


def systemd_commands(entry: Entry) -> dict[str, list[str]]:
    return {
        "reload": ["systemctl", "--user", "daemon-reload"],
        "enable": ["systemctl", "--user", "enable", entry.name],
        # A restart starts a stopped unit, and makes a running one take up changed settings or a
        # changed definition — a key rotated by `init --rotate`, say.
        "restart": ["systemctl", "--user", "restart", entry.name],
        "disable": ["systemctl", "--user", "disable", "--now", entry.name],
        "enabled": ["systemctl", "--user", "is-enabled", entry.name],
        "active": ["systemctl", "--user", "is-active", entry.name],
    }


def linger_commands(machine: Machine) -> dict[str, list[str]]:
    return {
        "query": ["loginctl", "show-user", machine.user, "--property=Linger"],
        "enable": ["sudo", "loginctl", "enable-linger", machine.user],
    }


def task_commands(machine: Machine, entry: Entry) -> dict[str, list[str]]:
    program = schtasks(machine)
    return {
        "create": [program, "/Create", "/TN", entry.name, "/XML", entry.file, "/F"],
        "start": [program, "/Run", "/TN", entry.name],
        "stop": [program, "/End", "/TN", entry.name],
        "delete": [program, "/Delete", "/TN", entry.name, "/F"],
        "query": [program, "/Query", "/TN", entry.name, "/FO", "LIST", "/V"],
    }


# --- before anything is written ---


def temporary_environment(machine: Machine) -> Optional[str]:
    """The cache directory this agent's environment lies in, or None.

    `uvx gpm-agent` runs the agent from an environment in uv's cache, which uv deletes or rebuilds
    whenever its cache is cleaned: a start at login pointing into it would one day point at
    nothing — or, rebuilt, at a version nobody chose. A cache directory marks itself with
    `CACHEDIR.TAG`, as uv's does."""
    paths = machine.paths
    folder = machine.prefix
    while folder:
        if machine.files.exists(paths.join(folder, "CACHEDIR.TAG")):
            return folder
        parent = paths.dirname(folder)
        if parent == folder:
            return None
        folder = parent
    return None


def _installed_version() -> str:
    """This agent's version as installed, or "" from a source tree that was never installed."""
    try:
        return metadata.version("gpm-agent")
    except metadata.PackageNotFoundError:
        return ""


def owner_of(machine: Machine, entry: Entry) -> tuple[bool, Optional[str]]:
    """Whether a definition by this name is already there, and which settings file it serves —
    None when it is there and says nothing this module wrote, so is someone else's."""
    existing = machine.files.read(entry.file)
    if machine.system == "Windows":
        # The task lives in Task Scheduler; the definition kept beside the settings is this
        # module's mark on it. A task by this name with no such definition is someone else's.
        if existing is None:
            return machine.run(task_commands(machine, entry)["query"]).ok, None
        found = _TASK_SETTINGS.search(existing.decode("utf-16", errors="replace"))
        return True, (unescape(found.group(1)) if found else None)
    if existing is None:
        return False, None
    if machine.system == "Darwin":
        try:
            arguments = plistlib.loads(existing).get("ProgramArguments") or []
            return True, str(arguments[arguments.index("--config") + 1])
        except (plistlib.InvalidFileException, ExpatError, AttributeError, ValueError, IndexError):
            return True, None  # not a plist this module wrote
    for line in existing.decode(errors="replace").splitlines():
        if line.startswith(_UNIT_SETTINGS):
            return True, line[len(_UNIT_SETTINGS):]
    return True, None


def problem_with_adding(machine: Machine, settings_path: str) -> Optional[str]:
    """Why this agent cannot be made to start with this machine — said before anything is written
    — or None."""
    if machine.system not in ("Darwin", "Linux", "Windows"):
        return (
            f"Starting with the machine is done for macOS, Linux and Windows, not {machine.system}. "
            "Have this system's own service manager run, as you:\n\n  "
            + shown(serve_command(machine, settings_path), machine.system)
        )
    if not machine.executable or not machine.paths.isabs(machine.executable):
        return "This interpreter does not know its own path, so there is nothing to start at login."
    if any(ord(c) < 32 for c in machine.executable + settings_path + machine.home):
        return "A path here holds a control character; a start at login cannot name it safely."
    if machine.system == "Windows" and "%" in machine.executable + settings_path:
        return (
            "A path here holds `%`, which Task Scheduler reads as the start of a variable. Keep the "
            "agent and its settings under paths without it."
        )
    if machine.system == "Linux" and machine.which("systemctl") is None:
        return (
            "This Linux has no systemd, so there is no user unit to write. Have its own service "
            "manager run, as you:\n\n  " + shown(serve_command(machine, settings_path), machine.system)
        )
    cache = temporary_environment(machine)
    if cache is not None:
        version = _installed_version()
        pinned = f"gpm-agent=={version}" if version else "gpm-agent"
        return (
            f"This agent is running from a temporary environment in {cache} — as `uvx` runs it — "
            "which is deleted or rebuilt whenever that cache is cleaned, so a start at login cannot "
            f"point into it. Install it to stay, then run this again from it:\n\n  uv tool install {pinned}\n"
        )
    entry = entry_for(machine, settings_path)
    there, said = owner_of(machine, entry)
    if there and said != settings_path:
        whose = f"for {said}" if said else "not written by this agent"
        return (
            f"A {entry.kind} named {entry.name} is already there, {whose}. Nothing was changed. "
            "Remove it first, or give these settings a file name of their own — the name follows it."
        )
    return None


# --- acting: writing the definition and running the commands ---


def _say_failed(command: Sequence[str], ran: Ran, system: str) -> None:
    print(f"  `{shown(command, system)}` failed ({ran.returncode})" + (f": {ran.output}" if ran.output else ""))


def linger(machine: Machine) -> Optional[bool]:
    """Whether the owner's user manager runs from boot, or None when `loginctl` would not say."""
    if not machine.user or machine.user.startswith("-"):  # it would be read as an option
        return None
    ran = machine.run(linger_commands(machine)["query"])
    if not ran.ok:
        return None
    for line in ran.output.splitlines():
        if line.startswith("Linger="):
            return line.partition("=")[2].strip() == "yes"
    return None


def _ensure_linger(machine: Machine, admin: bool) -> bool:
    """Lingering on, or the exact command for the owner. Only with `--admin`, at a terminal, is it
    run here, through `sudo` — which asks for the password itself; none is ever held here."""
    if not machine.user or machine.user.startswith("-"):
        print("  This user's name cannot be handed to loginctl safely; lingering was not checked.")
        return False
    lingering = linger(machine)
    if lingering:
        print("  Lingering is on: it starts at boot, before anyone logs in.")
        return True
    state = "off" if lingering is False else "not known (loginctl would not say)"
    enable = linger_commands(machine)["enable"]
    if admin and machine.interactive:
        print(f"  Lingering is {state}. Running, with administrator rights: {shown(enable, machine.system)}")
        ran = machine.run(enable, on_terminal=True)
        if ran.ok and linger(machine):
            print("  Lingering is on: it starts at boot, before anyone logs in.")
            return True
        _say_failed(enable, ran, machine.system)
    elif admin:
        print("  --admin needs a terminal for sudo to ask its password at; lingering was not changed.")
    print(
        f"  Lingering is {state}: the agent runs only while you are logged in. To start it at boot,\n"
        f"  run with administrator rights:\n\n    {shown(enable, machine.system)}\n"
    )
    return False


def add(machine: Machine, settings_path: str, *, admin: bool = False) -> int:
    """Make the agent serving `settings_path` start with the machine, and (re)start it now with the
    settings as they are."""
    problem = problem_with_adding(machine, settings_path)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    entry = entry_for(machine, settings_path)
    machine.files.write(entry.file, entry.content)
    if entry.log:
        machine.files.create_private(entry.log)
    print(f"Wrote the {entry.kind} {entry.name}: {entry.file}")
    print(f"It runs: {shown(entry.command, machine.system)}")
    started = _start(machine, entry, admin)
    if started:
        print("The agent is running now, and starts with the machine; there is no need to run `gpm-agent serve`.")
    if entry.log:
        print(f"Its output: {entry.log}")
    elif machine.system == "Linux":
        print(f"Its output: journalctl --user -u {entry.name}")
    return 0 if started else 1


def _start(machine: Machine, entry: Entry, admin: bool) -> bool:
    if machine.system == "Darwin":
        commands = launchd_commands(machine, entry)
        if machine.run(commands["query"]).ok:  # loaded already: out first, to load what was written
            machine.run(commands["stop"])
        # launchd can still be letting go of the old one; a few tries, a second apart.
        for attempt in range(3):
            ran = machine.run(commands["start"])
            if ran.ok:
                return True
            if attempt < 2:
                machine.sleep(1.0)
        _say_failed(commands["start"], ran, machine.system)
        print(
            "  launchd loads it at your next login. Without a login session at this Mac's screen —\n"
            "  over SSH, say — `gui/` has nothing to load it into until someone logs in."
        )
        return False
    if machine.system == "Linux":
        commands = systemd_commands(entry)
        for step in ("reload", "enable", "restart"):
            ran = machine.run(commands[step])
            if not ran.ok:
                _say_failed(commands[step], ran, machine.system)
                return False
        _ensure_linger(machine, admin)
        return True
    commands = task_commands(machine, entry)
    ran = machine.run(commands["create"])
    if not ran.ok:
        _say_failed(commands["create"], ran, machine.system)
        return False
    print("  It starts when you log on, as you, and runs while you are logged on: a machine nobody")
    print("  logs on to needs Windows' automatic sign-in for the agent to start by itself.")
    machine.run(commands["stop"])  # one running from before is ended, to start with these settings
    ran = machine.run(commands["start"])
    if not ran.ok:
        _say_failed(commands["start"], ran, machine.system)
        return False
    return True


def remove(machine: Machine, settings_path: str) -> int:
    """Stop the agent serving `settings_path` and take its definition away. Lingering is left as
    it is: other units of the owner's may need it."""
    if machine.system not in ("Darwin", "Linux", "Windows"):
        print(f"Nothing to remove: starting with the machine is not done on {machine.system}.")
        return 0
    entry = entry_for(machine, settings_path)
    there, said = owner_of(machine, entry)
    if not there:
        print(f"There is no {entry.kind} {entry.name}; nothing to remove.")
        return 0
    if said != settings_path:
        whose = f"serves {said}" if said else "was not written by this agent"
        print(f"The {entry.kind} {entry.name} {whose}; it was left alone.", file=sys.stderr)
        return 1
    failed = False
    if machine.system == "Darwin":
        machine.run(launchd_commands(machine, entry)["stop"])  # fails only when it was not loaded
    elif machine.system == "Linux":
        commands = systemd_commands(entry)
        machine.run(commands["disable"])
    else:
        commands = task_commands(machine, entry)
        machine.run(commands["stop"])
        ran = machine.run(commands["delete"])
        if not ran.ok and machine.run(commands["query"]).ok:
            _say_failed(commands["delete"], ran, machine.system)
            failed = True
    if not failed:
        machine.files.delete(entry.file)
        if machine.system == "Linux":
            machine.run(systemd_commands(entry)["reload"])
        print(f"Removed the {entry.kind} {entry.name}; the agent no longer starts with the machine.")
    return 1 if failed else 0


def status(machine: Machine, settings_path: str) -> int:
    """Say whether the agent serving `settings_path` starts with the machine, and whether it runs.
    0 when it is set up, 1 when not."""
    if machine.system not in ("Darwin", "Linux", "Windows"):
        print(f"Starting with the machine is not done on {machine.system}.")
        return 1
    entry = entry_for(machine, settings_path)
    there, said = owner_of(machine, entry)
    if not there:
        print(f"Not set up: no {entry.kind} {entry.name}. `gpm-agent startup add` sets it up.")
        return 1
    if said != settings_path:
        print(f"A {entry.kind} {entry.name} is there, but for {said or 'something else'}, not {settings_path}.")
        return 1
    print(f"{entry.kind} {entry.name}: {entry.file}")
    if machine.system == "Darwin":
        ran = machine.run(launchd_commands(machine, entry)["query"])
        if not ran.ok:
            print("  not loaded (it loads at your next login)")
        else:
            state = re.search(r"^\s*state = (.+)$", ran.output, re.MULTILINE)
            pid = re.search(r"^\s*pid = (\d+)$", ran.output, re.MULTILINE)
            print(f"  loaded; {state.group(1).strip() if state else 'state unknown'}"
                  + (f", process {pid.group(1)}" if pid else ""))
    elif machine.system == "Linux":
        commands = systemd_commands(entry)
        enabled = machine.run(commands["enabled"]).output or "unknown"
        active = machine.run(commands["active"]).output or "unknown"
        lingering = linger(machine)
        print(f"  {enabled}; {active}")
        print("  lingering: " + {True: "on (starts at boot)", False: "off (starts at login)", None: "unknown"}[lingering])
    else:
        ran = machine.run(task_commands(machine, entry)["query"])
        if not ran.ok:
            print("  its definition is here but the task is not registered")
            return 1
        for line in ran.output.splitlines():
            if line.strip():
                print("  " + line.strip())
    if entry.log:
        print(f"  output: {entry.log}")
    return 0


def send_output_to(path: str) -> None:
    """Append this process's output to `path`, owner-only — for an agent started at logon on
    Windows by `pythonw.exe`, which has no standard streams to write to."""
    descriptor = os.open(os.path.expanduser(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    stream = os.fdopen(descriptor, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr = stream


# --- the command line ---


def settings_path_of(args: Any) -> str:
    return str(Path(args.config).expanduser().absolute())


def _verb(args: Any) -> int:
    from .settings import load

    path = settings_path_of(args)
    machine = Machine.this()
    if args.action == "add":
        load(path)  # an agent that would not start is not set to start at login
        return add(machine, path, admin=args.admin)
    if args.action == "remove":
        return remove(machine, path)
    return status(machine, path)


def add_verbs(verbs: Any) -> None:
    """`gpm-agent startup add|remove|status`."""
    startup = verbs.add_parser("startup", help="start this agent with the machine, or stop doing so")
    startup.add_argument("action", choices=("add", "remove", "status"))
    startup.add_argument(
        "--admin", action="store_true",
        help="on Linux, switch lingering on through sudo so the agent starts at boot (asks for the password)",
    )
    startup.set_defaults(run=_verb)
