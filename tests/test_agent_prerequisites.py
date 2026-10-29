"""What only the machine's owner can install, installed once at `init` (D154).

docs/spec/host-agent.md §2.3. Each system is described by what is on its PATH and what its
commands answer; nothing here runs brew, apt-get, dnf, winget, xcode-select or sudo. What matters:
what is already there is left alone, `sudo` runs only on the owner's word at a terminal, every
command is the exact argument list, and what is left is said with the command that does it.
"""

from __future__ import annotations

import pytest
from gpm_agent import cli, prerequisites, startup
from gpm_agent.settings import Settings, fingerprint, save
from gpm_agent.startup import Machine, Ran


class Runner:
    """Records each command and answers from a table, or with success."""

    def __init__(self, answers=None):
        self.answers = {tuple(argv): answer for argv, answer in (answers or {}).items()}
        self.calls = []

    def __call__(self, argv, *, on_terminal=False):
        self.calls.append((list(argv), on_terminal))
        return self.answers.get(tuple(argv), Ran(0))

    @property
    def commands(self):
        return [argv for argv, _ in self.calls]


def a_machine(system, on_path=(), runner=None, interactive=True):
    """A machine with these programs on its PATH, under /usr/bin (C:\\Tools on Windows)."""
    folder = "C:\\Tools\\" if system == "Windows" else "/usr/bin/"
    return Machine(
        system=system, home="/home/owner", user="owner", uid=1000, executable="/usr/bin/python3",
        prefix="/usr", path="/usr/bin", environ={}, interactive=interactive, run=runner or Runner(),
        which=lambda name: folder + name if name in on_path else None, files=startup.Files(),
        sleep=lambda seconds: None,
    )


def test_only_whisper_cpp_needs_anything_and_a_compiler_only_where_the_agent_builds_it():
    assert prerequisites.needs("ollama", True, "Linux") == []
    assert prerequisites.needs("vllm", True, "Linux") == []
    assert prerequisites.needs("whisper-cpp", False, "Linux") == ["ffmpeg"]
    assert prerequisites.needs("whisper-cpp", True, "Darwin") == ["ffmpeg", "command-line tools"]
    assert prerequisites.needs("whisper-cpp", True, "Linux") == ["ffmpeg", "compiler"]
    # Windows takes whisper.cpp's own builds: no compiler.
    assert prerequisites.needs("whisper-cpp", True, "Windows") == ["ffmpeg"]


def test_what_is_already_there_is_left_alone():
    runner = Runner({("xcode-select", "-p"): Ran(0, "/Library/Developer/CommandLineTools")})
    machine = a_machine("Darwin", on_path=("ffmpeg", "brew"), runner=runner)
    report = prerequisites.ensure(machine, engine="whisper-cpp", install_engines=True, admin=True)
    assert report.present == [("ffmpeg", "/usr/bin/ffmpeg"), ("command-line tools", "/Library/Developer/CommandLineTools")]
    assert report.installed == [] and report.still_needed == []
    assert runner.commands == [["xcode-select", "-p"]]  # a look, and nothing installed

    runner = Runner()
    machine = a_machine("Linux", on_path=("ffmpeg", "c++", "make", "cmake", "apt-get", "sudo"), runner=runner)
    report = prerequisites.ensure(machine, engine="whisper-cpp", install_engines=True, admin=True)
    assert [need for need, _ in report.present] == ["ffmpeg", "compiler"] and runner.commands == []


def test_macos_installs_ffmpeg_with_homebrew_as_the_owner_and_only_names_xcodes_installer(capsys):
    runner = Runner({("xcode-select", "-p"): Ran(2, "xcode-select: error: unable to get active developer directory")})
    machine = a_machine("Darwin", on_path=("brew",), runner=runner)
    assert prerequisites.run(machine, engine="whisper-cpp", install_engines=True, admin=False) == 1
    assert runner.calls == [
        (["xcode-select", "-p"], False),
        (["/usr/bin/brew", "install", "ffmpeg"], True),  # no sudo: Homebrew refuses root
    ]
    out = capsys.readouterr().out
    assert "ffmpeg: installed now" in out and "xcode-select --install" in out


def test_macos_without_homebrew_is_told_how():
    runner = Runner()
    report = prerequisites.ensure(a_machine("Darwin", runner=runner), engine="whisper-cpp", install_engines=False, admin=True)
    assert report.still_needed == [("ffmpeg", "brew install ffmpeg   (Homebrew: https://brew.sh)")]
    assert runner.commands == []


def test_linux_installs_through_sudo_only_with_admin_at_a_terminal():
    runner = Runner()
    machine = a_machine("Linux", on_path=("apt-get",), runner=runner)
    report = prerequisites.ensure(machine, engine="whisper-cpp", install_engines=True, admin=True)
    assert runner.calls == [
        (["sudo", "apt-get", "update"], True),
        (["sudo", "apt-get", "install", "-y", "ffmpeg", "g++", "make", "cmake", "python3-venv"], True),
    ]
    assert [need for need, _ in report.installed] == ["ffmpeg", "compiler"] and report.still_needed == []


def test_the_compiler_is_looked_for_by_the_names_the_build_runs():
    """whisper.cpp's build runs `c++`, `make` and `cmake`: a `g++` without `c++` is refused there,
    so it is not "already here"; and cmake comes with the compiler, from the same packages."""
    assert prerequisites.found(a_machine("Linux", on_path=("g++", "make", "cmake")), "compiler") is None
    assert prerequisites.found(a_machine("Linux", on_path=("c++", "make")), "compiler") is None
    assert prerequisites.found(a_machine("Linux", on_path=("c++", "make", "cmake")), "compiler") == (
        "/usr/bin/c++, /usr/bin/make, /usr/bin/cmake"
    )


@pytest.mark.parametrize("admin, interactive", [(False, True), (True, False), (False, False)])
def test_linux_without_the_owners_word_at_a_terminal_prints_the_exact_command(admin, interactive):
    runner = Runner()
    machine = a_machine("Linux", on_path=("apt-get",), runner=runner, interactive=interactive)
    report = prerequisites.ensure(machine, engine="whisper-cpp", install_engines=False, admin=admin)
    assert runner.commands == []
    ((need, how),) = report.still_needed
    assert need == "ffmpeg" and how.startswith("sudo apt-get update\n      sudo apt-get install -y ffmpeg\n")


def test_dnf_gets_its_own_package_names_and_a_failure_is_still_needed(capsys):
    install = ("sudo", "dnf", "install", "-y", "ffmpeg", "gcc-c++", "make", "cmake")
    runner = Runner({install: Ran(1)})
    machine = a_machine("Linux", on_path=("dnf",), runner=runner)
    assert prerequisites.run(machine, engine="whisper-cpp", install_engines=True, admin=True) == 1
    assert runner.commands == [list(install)]
    out = capsys.readouterr().out
    assert "failed with 1" in out and "RPM Fusion" in out and "(the command above)" in out


def test_a_linux_without_apt_or_dnf_is_told_what_to_install():
    report = prerequisites.ensure(a_machine("Linux"), engine="whisper-cpp", install_engines=True, admin=True)
    assert [need for need, _ in report.still_needed] == ["ffmpeg", "compiler"]


def test_windows_installs_ffmpeg_with_winget_for_this_user_only():
    runner = Runner()
    machine = a_machine("Windows", on_path=("winget",), runner=runner)
    report = prerequisites.ensure(machine, engine="whisper-cpp", install_engines=True, admin=False)
    assert runner.calls == [([
        "C:\\Tools\\winget", "install", "--id", "Gyan.FFmpeg", "-e", "--scope", "user",
        "--accept-package-agreements", "--accept-source-agreements",
    ], True)]
    assert [need for need, _ in report.installed] == ["ffmpeg"]


def test_windows_without_winget_is_told_the_command():
    runner = Runner()
    report = prerequisites.ensure(a_machine("Windows", runner=runner), engine="whisper-cpp", install_engines=False, admin=True)
    ((need, how),) = report.still_needed
    assert how.startswith("winget install --id Gyan.FFmpeg -e --scope user ") and runner.commands == []


def test_nothing_needed_says_so(capsys):
    assert prerequisites.run(a_machine("Linux"), engine="ollama", install_engines=True, admin=True) == 0
    assert "Nothing to install for ollama" in capsys.readouterr().out


def test_init_and_the_verb_install_what_the_settings_engine_needs(monkeypatch, tmp_path, capsys):
    runner = Runner()
    machine = a_machine("Linux", on_path=("apt-get",), runner=runner)
    monkeypatch.setattr(startup.Machine, "this", classmethod(lambda cls: machine))
    path = tmp_path / "whisper.json"
    assert cli.main(["-c", str(path), "init", "--engine", "whisper-cpp", "--prerequisites", "--admin"]) == 0
    assert runner.commands == [["sudo", "apt-get", "update"], ["sudo", "apt-get", "install", "-y", "ffmpeg"]]
    assert "Agent key" in capsys.readouterr().out  # the key is still shown, first

    runner.calls.clear()
    assert cli.main(["-c", str(path), "prerequisites"]) == 1  # no --admin: printed, not run
    assert runner.commands == [] and "sudo apt-get install -y ffmpeg" in capsys.readouterr().out

    ollama = save(Settings(key_hash=fingerprint("gpmg_notreal")), tmp_path / "agent.json")
    assert cli.main(["-c", str(ollama), "prerequisites", "--admin"]) == 0


def test_init_fails_while_a_prerequisite_is_still_needed(monkeypatch, tmp_path, capsys):
    """The settings are written and the key shown, but `init` ends non-zero, so a script running
    it learns that something is still to do."""
    runner = Runner()
    machine = a_machine("Linux", on_path=("apt-get",), runner=runner)
    monkeypatch.setattr(startup.Machine, "this", classmethod(lambda cls: machine))
    path = tmp_path / "whisper.json"
    assert cli.main(["-c", str(path), "init", "--engine", "whisper-cpp", "--prerequisites"]) == 1
    out = capsys.readouterr().out
    assert path.exists() and "Agent key" in out and "Still needed" in out and runner.commands == []
