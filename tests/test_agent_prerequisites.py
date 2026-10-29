"""What only the machine's owner can install, installed once at `init` (D154).

docs/spec/host-agent.md §2.3. Each system is described by what is on its PATH and what its
commands answer; nothing here runs brew, apt-get, dnf, winget, xcode-select or sudo. What matters:
what is already there is left alone, `sudo` runs only on the owner's word at a terminal, every
command is the exact argument list, and what is left is said with the command that does it.
"""

from __future__ import annotations

import sys

import pytest
from gpm_agent import cli, prerequisites, startup
from gpm_agent.settings import Settings, fingerprint, save
from gpm_agent.startup import Machine, Ran
from gpm_agent.vllm_install import LockedSet


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
    # vLLM its owner installed needs nothing of this command; one the agent installs, below.
    assert prerequisites.needs("vllm", False, "Linux") == []
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


# --- vLLM, where the agent installs it (D157, D158) ---

SETS = (
    LockedSet(vllm="0.31.0", python="3.12", platform="x86_64", driver_floor="525.60.13", requirements="", size_bytes=1),
    LockedSet(vllm="0.32.0", python="3.12", platform="x86_64", driver_floor="580.65.06", requirements="", size_bytes=1),
    LockedSet(vllm="0.32.0", python="3.13", platform="x86_64", driver_floor="580.65.06", requirements="", size_bytes=1),
)
HERE = prerequisites.ForVllm(python="3.12", processor="x86_64", sets=SETS)


class VllmMachine(Runner):
    """A Linux machine as vLLM's checks see it: whether the agent's Python makes an environment,
    and what `nvidia-smi` says of the driver (None: it does not run)."""

    def __init__(self, venv_works=True, driver="580.95.05"):
        super().__init__()
        self.venv_works, self.driver = venv_works, driver

    def __call__(self, argv, *, on_terminal=False):
        self.calls.append((list(argv), on_terminal))
        if argv[1:4] == ["-I", "-m", "venv"]:
            return Ran(0) if self.venv_works else Ran(1, "The virtual environment was not successfully created because ensurepip is not available.")
        if argv[0] == "nvidia-smi":
            return Ran(0, f"{self.driver}\n{self.driver}\n") if self.driver else Ran(127, "nvidia-smi was not found")
        return Ran(0)


def test_vllm_the_agent_installs_needs_venv_a_set_and_a_driver_on_linux_and_linux_elsewhere():
    assert prerequisites.needs("vllm", True, "Linux") == ["venv", "vllm set", "nvidia driver"]
    assert prerequisites.needs("vllm", True, "Darwin") == ["linux"]
    assert prerequisites.needs("vllm", True, "Windows") == ["linux"]


def test_a_machine_ready_for_vllm_is_told_so_and_nothing_runs_with_sudo():
    runner = VllmMachine()
    report = prerequisites.ensure(a_machine("Linux", ("apt-get",), runner), engine="vllm", install_engines=True, admin=True, vllm=HERE)
    assert [need for need, _ in report.present] == ["venv", "vllm set", "nvidia driver"]
    assert report.installed == [] and report.still_needed == []
    assert dict(report.present)["nvidia driver"] == "580.95.05, at or above vLLM 0.32.0's 580.65.06"
    # The environment is tried as the installer makes it: the agent's own Python, isolated, in a
    # directory of its own.
    venv = next(argv for argv in runner.commands if "venv" in argv)
    assert venv[:4] == ["/usr/bin/python3", "-I", "-m", "venv"] and venv[4].endswith("/venv")
    assert not any(argv[0] == "sudo" for argv in runner.commands)


def test_pythons_venv_is_installed_through_apt_on_the_owners_word_and_printed_otherwise(capsys):
    runner = VllmMachine(venv_works=False)
    machine = a_machine("Linux", ("apt-get",), runner)
    report = prerequisites.ensure(machine, engine="vllm", install_engines=True, admin=True, vllm=HERE)
    assert report.installed == [("venv", "sudo apt-get install -y python3-venv")]
    assert ["sudo", "apt-get", "install", "-y", "python3-venv"] in runner.commands

    runner = VllmMachine(venv_works=False)
    assert prerequisites.run(a_machine("Linux", ("apt-get",), runner), engine="vllm", install_engines=True, admin=False, vllm=HERE) == 1
    assert "sudo apt-get install -y python3-venv" in capsys.readouterr().out
    assert not any(argv[0] == "sudo" for argv in runner.commands)


def test_dnf_names_no_package_for_venv_and_says_whose_python_lacks_it():
    runner = VllmMachine(venv_works=False)
    report = prerequisites.ensure(a_machine("Linux", ("dnf",), runner), engine="vllm", install_engines=True, admin=True, vllm=HERE)
    assert report.still_needed == [(
        "venv", "install Python's venv module, for the agent's own Python (/usr/bin/python3) with this distribution's package manager",
    )]
    assert not any(argv[0] == "sudo" for argv in runner.commands)


@pytest.mark.parametrize(("driver", "said"), [
    ("575.51.03", "the NVIDIA driver here is 575.51.03: install NVIDIA's driver 580.65.06 or later, which vLLM 0.32.0's CUDA needs"),
    (None, "no NVIDIA driver answers here (nvidia-smi does not run): install NVIDIA's driver 580.65.06 or later"),
])
def test_a_driver_missing_or_older_than_the_newest_sets_floor_is_said_and_never_installed(driver, said, capsys):
    """The newest set's floor, not an older one's that the driver would meet; and with --admin at
    a terminal still nothing is installed for it: a driver is the owner's to choose, with a restart."""
    runner = VllmMachine(driver=driver)
    machine = a_machine("Linux", ("apt-get",), runner)
    assert prerequisites.run(machine, engine="vllm", install_engines=True, admin=True, vllm=HERE) == 1
    out = capsys.readouterr().out
    assert said in out and "the agent never installs a driver" in out
    assert "an NVIDIA driver new enough for vLLM's CUDA" in out
    assert not any(argv[0] == "sudo" for argv in runner.commands)


def test_a_python_the_agent_carries_no_set_for_is_told_which_it_does():
    runner = VllmMachine()
    other = prerequisites.ForVllm(python="3.15", processor="x86_64", sets=SETS)
    report = prerequisites.ensure(a_machine("Linux", ("apt-get",), runner), engine="vllm", install_engines=True, admin=True, vllm=other)
    (need, how), = report.still_needed
    assert need == "vllm set"
    assert how == (
        "this agent runs on Python 3.15, and carries vLLM's sets for Python 3.12, 3.13 on x86_64: install it on one "
        "of those (`uv tool install --python 3.13 gpm-agent`)"
    )
    odd = prerequisites.ForVllm(python="3.12", processor="ppc64le", sets=SETS)
    report = prerequisites.ensure(a_machine("Linux", ("apt-get",), runner), engine="vllm", install_engines=True, admin=True, vllm=odd)
    assert ("vllm set", "vLLM publishes no build for the ppc64le processor") in report.still_needed


def test_vllm_on_a_mac_is_told_it_runs_on_linux_only(capsys):
    machine = a_machine("Darwin", ("brew",), Runner())
    assert prerequisites.run(machine, engine="vllm", install_engines=True, admin=True, vllm=HERE) == 1
    out = capsys.readouterr().out
    assert "Linux, for vLLM" in out and "vLLM publishes its builds for Linux only" in out
    assert machine.run.commands == []


def test_the_sets_the_agent_carries_are_what_prerequisites_judge_by_unless_told():
    """`ForVllm.this()`: the agent's own Python and this processor, and the sets in the agent."""
    here = prerequisites.ForVllm.this()
    assert here.sets is None and here.python == f"{sys.version_info.major}.{sys.version_info.minor}"
