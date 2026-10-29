"""What only the machine's owner can install: `gpm-agent init --prerequisites`.

docs/spec/host-agent.md §2.3, D154. A few things an engine needs come from the system's package
manager, and some of those need administrator rights — which the agent never asks for itself. So
they are installed once, here, at a terminal, by the person who owns the machine:

- **ffmpeg**, for whisper.cpp's server, which hands it every recording that is not WAV
  (`--convert`). Homebrew on macOS; apt or dnf on Linux, through `sudo`; winget on Windows, for
  this user only.
- **A compiler**, where this agent builds whisper.cpp from source (D152): on macOS Xcode's
  command-line tools — whose installer is a dialog, so it is only named; on Linux a C++ compiler,
  `make` and `cmake`, through `sudo`. Windows takes whisper.cpp's own builds and needs none.
- **For vLLM**, where this agent installs it (D157, Linux only): Python's `venv` for the agent's
  own interpreter, which makes each version's environment (`python3-venv` on Debian and Ubuntu);
  a locked set for that Python and the machine's processor; and an NVIDIA driver at the newest
  set's floor. The last two are never installed here — a set comes with a release of the agent,
  and a driver is the owner's to choose and install, with a restart — only said, with what is
  needed.

What is already there is found first and left alone. `sudo` runs only when the owner said
`--admin` at a terminal, where it asks for the password itself; otherwise the exact command is
printed for the owner to run. Every command is an argument list written here, run through
`startup.run_command` — never a shell, never a script fetched from anywhere.
"""

from __future__ import annotations

import dataclasses
import platform
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from . import vllm_install
from .startup import Machine, shown
from .vllm_install import LockedSet

#: The engine that needs ffmpeg and, when this agent installs it, a compiler.
NEEDS_FFMPEG = frozenset({"whisper-cpp"})
BUILT_FROM_SOURCE = frozenset({"whisper-cpp"})
#: The engine installed into a Python environment of its own, from a locked set (D157, D158).
IN_AN_ENVIRONMENT = frozenset({"vllm"})
#: What only a release of the agent, or the machine's owner, can change: said, never installed.
NEVER_INSTALLED = frozenset({"linux", "vllm set", "nvidia driver"})

#: winget's name for the ffmpeg build it carries — Gyan Doshi's, the one ffmpeg's own download
#: page points Windows users to — installed for this user, without administrator rights.
WINGET_FFMPEG = [
    "install", "--id", "Gyan.FFmpeg", "-e", "--scope", "user",
    "--accept-package-agreements", "--accept-source-agreements",
]

#: Each Linux package manager's names for what is needed. The build (`whisper_build`) runs `c++`,
#: `make` and `cmake` — cmake from PyPI where the machine has none, in a private environment made
#: with Python's `venv`, which Debian and Ubuntu ship apart from Python itself (`python3-venv`).
#: vLLM's environments are made the same way. Fedora's Python carries `venv` and the pip wheel
#: it installs, so dnf has nothing to name for it: a Python without them there is not the
#: system's, and is said as such.
LINUX_PACKAGES = {
    "apt-get": {"ffmpeg": ["ffmpeg"], "compiler": ["g++", "make", "cmake", "python3-venv"], "venv": ["python3-venv"]},
    "dnf": {"ffmpeg": ["ffmpeg"], "compiler": ["gcc-c++", "make", "cmake"]},
}

#: `nvidia-smi`'s answer with the driver's version, one line a card.
NVIDIA_DRIVER = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]

#: The programs the build runs, each looked for by the name it runs it by.
BUILD_PROGRAMS = ("c++", "make", "cmake")

#: What each need is called, for the owner.
TITLES = {
    "ffmpeg": "ffmpeg",
    "command-line tools": "Xcode's command-line tools",
    "compiler": "a C++ compiler (c++), make and cmake",
    "venv": "Python's venv module, for the agent's own Python",
    "vllm set": "a vLLM set for the agent's Python and this processor",
    "nvidia driver": "an NVIDIA driver new enough for vLLM's CUDA",
    "linux": "Linux, for vLLM",
}


@dataclasses.dataclass(frozen=True)
class ForVllm:
    """What a vLLM set must be for: the agent's own Python minor version and the machine's
    processor — and the sets to choose from, None for the ones the agent carries."""

    python: str
    processor: str
    sets: Optional[Sequence[LockedSet]] = None

    @classmethod
    def this(cls) -> "ForVllm":
        return cls(vllm_install.this_python(), platform.machine())

    def newest(self) -> Optional[LockedSet]:
        found = vllm_install.sets_for(self.python, self.processor, self.sets)
        return found[0] if found else None


@dataclasses.dataclass
class Report:
    """What was there, what was installed now, and what is still needed with how to get it."""

    present: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    installed: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    still_needed: list[tuple[str, str]] = dataclasses.field(default_factory=list)


def needs(engine: str, install_engines: bool, system: str) -> list[str]:
    """What this machine needs for `engine`: ffmpeg for whisper.cpp; and a compiler where this
    agent builds whisper.cpp itself — on macOS and Linux, when the owner allowed installs."""
    wanted = []
    if engine in NEEDS_FFMPEG:
        wanted.append("ffmpeg")
    if engine in BUILT_FROM_SOURCE and install_engines:
        if system == "Darwin":
            wanted.append("command-line tools")
        elif system == "Linux":
            wanted.append("compiler")
    if engine in IN_AN_ENVIRONMENT and install_engines:
        wanted += ["venv", "vllm set", "nvidia driver"] if system == "Linux" else ["linux"]
    return wanted


def found(machine: Machine, need: str, vllm: Optional[ForVllm] = None) -> Optional[str]:
    """Where `need` is on this machine, or None."""
    if need == "ffmpeg":
        return machine.which("ffmpeg")
    if need == "command-line tools":
        ran = machine.run(["xcode-select", "-p"])
        return (ran.output.strip() or "installed") if ran.ok else None
    if need == "compiler":
        # As the build looks: `c++` itself — a `g++` without it is refused there.
        programs = [machine.which(name) for name in BUILD_PROGRAMS]
        return ", ".join(str(program) for program in programs) if all(programs) else None
    if need == "venv":
        return _venv_works(machine)
    if need == "linux":
        return machine.system if machine.system == "Linux" else None
    vllm = vllm or ForVllm.this()
    if need == "vllm set":
        newest = vllm.newest()
        return f"vLLM {newest.vllm}'s, for Python {vllm.python} on {vllm.processor}" if newest else None
    if need == "nvidia driver":
        have, newest = nvidia_driver(machine), vllm.newest()
        key = vllm_install.driver_key(have)
        floor = vllm_install.driver_key(newest.driver_floor) if newest else None
        if key is None or (floor is not None and key < floor):
            return None
        return f"{have}" + (f", at or above vLLM {newest.vllm}'s {newest.driver_floor}" if newest else "")
    raise ValueError(need)


def _venv_works(machine: Machine) -> Optional[str]:
    """Whether the agent's own interpreter makes an environment as the installer will — pip
    included, which Debian and Ubuntu leave out without `python3-venv` — tried in a directory of
    its own, removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="gpm-venv-check-") as scratch:
        ran = machine.run(vllm_install.venv_argv(machine.executable, Path(scratch) / "venv"))
    return f"{machine.executable} makes one" if ran.ok else None


def nvidia_driver(machine: Machine) -> Optional[str]:
    """The NVIDIA driver's version as `nvidia-smi` says it — None where it does not answer."""
    ran = machine.run(NVIDIA_DRIVER)
    if not ran.ok:
        return None
    versions = [line.strip() for line in ran.output.splitlines() if vllm_install.driver_key(line.strip())]
    return versions[0] if versions else None


def never_installed(machine: Machine, need: str, vllm: ForVllm) -> str:
    """What to do about something this command never installs, in words for the owner."""
    if need == "linux":
        return "vLLM publishes its builds for Linux only: on this machine the agent serves on Ollama"
    if need == "vllm set":
        if vllm.processor not in vllm_install.PLATFORMS:
            return f"vLLM publishes no build for the {vllm.processor} processor"
        carried = list(vllm_install.load_sets() if vllm.sets is None else vllm.sets)
        pythons = sorted(
            {s.python for s in carried if s.platform == vllm.processor}, key=lambda p: tuple(map(int, p.split(".")))
        )
        if not pythons:
            return f"this agent carries no vLLM set for {vllm.processor}: a release of the agent brings them"
        return (
            f"this agent runs on Python {vllm.python}, and carries vLLM's sets for Python {', '.join(pythons)} "
            f"on {vllm.processor}: install it on one of those (`uv tool install --python {pythons[-1]} gpm-agent`)"
        )
    newest = vllm.newest()
    wanted = f"{newest.driver_floor} or later, which vLLM {newest.vllm}'s CUDA needs" if newest else "a recent one"
    have = nvidia_driver(machine)
    where = "no NVIDIA driver answers here (nvidia-smi does not run)" if have is None else f"the NVIDIA driver here is {have}"
    return (
        f"{where}: install NVIDIA's driver {wanted}, from the distribution's packages or NVIDIA's own, "
        "and restart the machine — the agent never installs a driver"
    )


def linux_commands(manager: str, missing: list[str]) -> list[list[str]]:
    """The commands that install `missing` with `manager`, through `sudo`. apt's package lists are
    refreshed first: on a machine fresh from its image they are often empty."""
    packages = [package for need in missing for package in LINUX_PACKAGES[manager][need]]
    if manager == "apt-get":
        return [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", *packages]]
    return [["sudo", "dnf", "install", "-y", *packages]]


def ensure(
    machine: Machine, *, engine: str, install_engines: bool, admin: bool, vllm: Optional[ForVllm] = None,
) -> Report:
    """Find what `engine` needs here, install what can be, and say what is left. `vllm` is what a
    vLLM set must be for: the agent's own Python and this processor, unless a test says."""
    report = Report()
    missing = []
    wanted = needs(engine, install_engines, machine.system)
    if vllm is None and any(need in NEVER_INSTALLED for need in wanted):
        vllm = ForVllm.this()
    for need in wanted:
        where = found(machine, need, vllm)
        if where:
            report.present.append((need, where))
        elif need in NEVER_INSTALLED and vllm is not None:
            report.still_needed.append((need, never_installed(machine, need, vllm)))
        else:
            missing.append(need)
    if not missing:
        return report
    if machine.system == "Darwin":
        _darwin(machine, missing, report)
    elif machine.system == "Linux":
        _linux(machine, missing, report, admin)
    elif machine.system == "Windows":
        _windows(machine, missing, report)
    else:
        for need in missing:
            report.still_needed.append((need, f"install {TITLES[need]} with this system's package manager"))
    return report


def _darwin(machine: Machine, missing: list[str], report: Report) -> None:
    if "ffmpeg" in missing:
        brew = machine.which("brew")
        if brew is None:
            report.still_needed.append(("ffmpeg", "brew install ffmpeg   (Homebrew: https://brew.sh)"))
        else:
            # Homebrew installs as the owner and refuses to run as root, so no sudo here.
            command = [brew, "install", "ffmpeg"]
            ran = machine.run(command, on_terminal=True)
            if ran.ok:
                report.installed.append(("ffmpeg", shown(command, machine.system)))
            else:
                report.still_needed.append(("ffmpeg", f"brew install ffmpeg   (it failed here with {ran.returncode})"))
    if "command-line tools" in missing:
        # Its installer is a dialog at the Mac's screen, to be clicked through: named, not run.
        report.still_needed.append(("command-line tools", "xcode-select --install   (opens Apple's installer)"))


def _linux(machine: Machine, missing: list[str], report: Report, admin: bool) -> None:
    manager = next((name for name in LINUX_PACKAGES if machine.which(name)), None)
    if manager is None:
        for need in missing:
            report.still_needed.append((need, f"install {TITLES[need]} with this distribution's package manager"))
        return
    for need in [need for need in missing if need not in LINUX_PACKAGES[manager]]:
        report.still_needed.append((
            need, f"install {TITLES[need]} ({machine.executable}) with this distribution's package manager",
        ))
    missing = [need for need in missing if need in LINUX_PACKAGES[manager]]
    if not missing:
        return
    commands = linux_commands(manager, missing)
    how = "\n      ".join(shown(command, machine.system) for command in commands)
    if manager == "dnf" and "ffmpeg" in missing:
        how += "\n      (on Fedora and its relatives, ffmpeg comes from RPM Fusion, which must be enabled first)"
    if not (admin and machine.interactive):
        why = "needs a terminal for sudo" if admin else "needs administrator rights; --admin runs it through sudo"
        for need in missing:
            report.still_needed.append((need, f"{how}\n      ({why})"))
        return
    for command in commands:
        ran = machine.run(command, on_terminal=True)
        if not ran.ok:
            for need in missing:
                report.still_needed.append((need, f"{how}\n      (`{shown(command, machine.system)}` failed with {ran.returncode})"))
            return
    for need in missing:
        report.installed.append((need, shown(commands[-1], machine.system)))


def _windows(machine: Machine, missing: list[str], report: Report) -> None:
    if "ffmpeg" not in missing:
        return
    winget = machine.which("winget")
    typed = shown(["winget", *WINGET_FFMPEG], machine.system)
    if winget is None:
        report.still_needed.append((
            "ffmpeg",
            f"{typed}\n      (winget comes with Windows' App Installer, from the Microsoft Store)",
        ))
        return
    command = [winget, *WINGET_FFMPEG]
    ran = machine.run(command, on_terminal=True)
    if ran.ok:
        report.installed.append(("ffmpeg", typed))
    else:
        report.still_needed.append(("ffmpeg", f"{typed}\n      (it failed here with {ran.returncode})"))


def say(report: Report, *, engine: str, system: str) -> None:
    """The report, for the owner."""
    if not (report.present or report.installed or report.still_needed):
        print(f"Nothing to install for {engine} on this machine.")
        return
    print(f"What {engine} needs on this machine:")
    for need, where in report.present:
        print(f"  {TITLES[need]}: already here ({where})")
    for need, command in report.installed:
        print(f"  {TITLES[need]}: installed now ({command})")
        if system == "Windows":
            print("    — on the PATH of terminals opened from now on, and of the agent started at logon")
    if report.still_needed:
        print("Still needed:")
        shown_already = set()
        for need, how in report.still_needed:
            print(f"  {TITLES[need]}:")
            if how not in shown_already:  # one command often covers several needs
                print(f"      {how}")
                shown_already.add(how)
            else:
                print("      (the command above)")


def run(
    machine: Machine, *, engine: str, install_engines: bool, admin: bool, vllm: Optional[ForVllm] = None,
) -> int:
    """Install what is missing and say how it went: 0 when nothing is still needed."""
    report = ensure(machine, engine=engine, install_engines=install_engines, admin=admin, vllm=vllm)
    say(report, engine=engine, system=machine.system)
    return 1 if report.still_needed else 0


#: An engine the agent chooses itself as it starts (D161).
AUTO = "auto"


def auto_choice(models_path: str, probes: Any = None) -> tuple[str, list[str]]:
    """The engine an agent of `auto` would choose on this machine as it starts, and every reason
    against vLLM — made exactly as the agent makes it, from the machine's own facts."""
    from .app import choose_engine  # the agent's own choice; not loaded for any other engine
    from .facts import Probes, gather

    choice = choose_engine(gather(probes or Probes(), models_path))
    return choice["engine"], list(choice["reasons"])


def run_for(machine: Machine, settings: Any, *, admin: bool, probes: Any = None, vllm: Optional[ForVllm] = None) -> int:
    """`run` for the agent's settings: under `auto`, for the engine it would choose here, said
    first with why not vLLM — a driver to upgrade, a Python the agent carries no set for."""
    engine = settings.engine
    if engine == AUTO:
        engine, reasons = auto_choice(settings.models_path, probes)
        print(f"engine: auto — on this machine the agent runs {engine}"
              + (f"; not vLLM: {'; '.join(reasons)}" if reasons else ""))
    return run(machine, engine=engine, install_engines=bool(getattr(settings, "install_engines", False)),
               admin=admin, vllm=vllm)


# --- the command line ---


def _verb(args: Any) -> int:
    from .settings import load
    from .startup import settings_path_of

    settings = load(settings_path_of(args))
    return run_for(Machine.this(), settings, admin=args.admin)


def add_verbs(verbs: Any) -> None:
    """`gpm-agent prerequisites`: what `init --prerequisites` does, for an agent already set up."""
    prerequisites = verbs.add_parser(
        "prerequisites", help="install what this machine's engine needs from the system's package manager",
    )
    prerequisites.add_argument(
        "--admin", action="store_true",
        help="on Linux, install through sudo (asks for the password); otherwise the command is printed",
    )
    prerequisites.set_defaults(run=_verb)
