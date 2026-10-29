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

What is already there is found first and left alone. `sudo` runs only when the owner said
`--admin` at a terminal, where it asks for the password itself; otherwise the exact command is
printed for the owner to run. Every command is an argument list written here, run through
`startup.run_command` — never a shell, never a script fetched from anywhere.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from .startup import Machine, shown

#: The engine that needs ffmpeg and, when this agent installs it, a compiler.
NEEDS_FFMPEG = frozenset({"whisper-cpp"})
BUILT_FROM_SOURCE = frozenset({"whisper-cpp"})

#: winget's name for the ffmpeg build it carries — Gyan Doshi's, the one ffmpeg's own download
#: page points Windows users to — installed for this user, without administrator rights.
WINGET_FFMPEG = [
    "install", "--id", "Gyan.FFmpeg", "-e", "--scope", "user",
    "--accept-package-agreements", "--accept-source-agreements",
]

#: Each Linux package manager's names for what is needed. The build (`whisper_build`) runs `c++`,
#: `make` and `cmake` — cmake from PyPI where the machine has none, in a private environment made
#: with Python's `venv`, which Debian and Ubuntu ship apart from Python itself (`python3-venv`).
LINUX_PACKAGES = {
    "apt-get": {"ffmpeg": ["ffmpeg"], "compiler": ["g++", "make", "cmake", "python3-venv"]},
    "dnf": {"ffmpeg": ["ffmpeg"], "compiler": ["gcc-c++", "make", "cmake"]},
}

#: The programs the build runs, each looked for by the name it runs it by.
BUILD_PROGRAMS = ("c++", "make", "cmake")

#: What each need is called, for the owner.
TITLES = {
    "ffmpeg": "ffmpeg",
    "command-line tools": "Xcode's command-line tools",
    "compiler": "a C++ compiler (c++), make and cmake",
}


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
    return wanted


def found(machine: Machine, need: str) -> Optional[str]:
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
    raise ValueError(need)


def linux_commands(manager: str, missing: list[str]) -> list[list[str]]:
    """The commands that install `missing` with `manager`, through `sudo`. apt's package lists are
    refreshed first: on a machine fresh from its image they are often empty."""
    packages = [package for need in missing for package in LINUX_PACKAGES[manager][need]]
    if manager == "apt-get":
        return [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", *packages]]
    return [["sudo", "dnf", "install", "-y", *packages]]


def ensure(machine: Machine, *, engine: str, install_engines: bool, admin: bool) -> Report:
    """Find what `engine` needs here, install what can be, and say what is left."""
    report = Report()
    missing = []
    for need in needs(engine, install_engines, machine.system):
        where = found(machine, need)
        if where:
            report.present.append((need, where))
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


def run(machine: Machine, *, engine: str, install_engines: bool, admin: bool) -> int:
    """Install what is missing and say how it went: 0 when nothing is still needed."""
    report = ensure(machine, engine=engine, install_engines=install_engines, admin=admin)
    say(report, engine=engine, system=machine.system)
    return 1 if report.still_needed else 0


# --- the command line ---


def _verb(args: Any) -> int:
    from .settings import load
    from .startup import settings_path_of

    settings = load(settings_path_of(args))
    return run(
        Machine.this(), engine=settings.engine,
        install_engines=bool(getattr(settings, "install_engines", False)), admin=args.admin,
    )


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
