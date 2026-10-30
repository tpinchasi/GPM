"""The agent installs vLLM on a Linux machine with an NVIDIA card, from a locked set (D157, D158).

docs/spec/host-agent.md §2.3. Nothing here reaches PyPI, needs a GPU or runs on Linux: the sets are
built here, the machine's cards and C library are what a test says they are, and wheels are small
zips made here — a `vllm` whose console script prints its version — installed by real pip from a
directory (`GPM_PYPI_INDEX`) into a real environment, where a test needs pip's own answer: that a
set installs and its `vllm` runs from where the version lives, and that a file not matching its
set's digest refuses the version. The rest stands in for the commands. The release's tool that
makes the sets (`tools/vllm-lock`) is run against a stand-in for uv and for PyPI's JSON API.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import importlib.machinery
import importlib.util
import json
import os
import socketserver
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Optional

import pytest
from gpm_agent import installs, vllm_install
from gpm_agent import platform as host_platform
from gpm_agent.installs import Installer, InstallRefused, Target
from gpm_agent.platform import LINUX, MACOS, WINDOWS, Ran
from gpm_agent.vllm_install import LockedSet, VllmSource

REPO = Path(__file__).resolve().parent.parent
PYTHON = vllm_install.this_python()
LINUX_X86 = Target(LINUX, "amd64")

AMPERE = {"kind": "cuda", "name": "NVIDIA RTX A6000", "compute_capability": "8.6", "driver": "580.95.05"}
TURING = {"kind": "cuda", "name": "Tesla T4", "compute_capability": "7.5", "driver": "580.95.05"}
PASCAL = {"kind": "cuda", "name": "Tesla P100", "compute_capability": "6.0", "driver": "580.95.05"}


def locked(version="0.32.0", python=PYTHON, platform="x86_64", *, floor="580.65.06", requirements="", size=4 * 10**9,
           glibc: Optional[str] = "2.31") -> LockedSet:
    return LockedSet(vllm=version, python=python, platform=platform, driver_floor=floor,
                     requirements=requirements, size_bytes=size, glibc=glibc)


def source(*sets, target=LINUX_X86, python=PYTHON, cards=(AMPERE,), glibc="2.35", run=None, environ=None) -> VllmSource:
    return VllmSource(
        target, python=python, sets=list(sets), cards=lambda: None if cards is None else [dict(c) for c in cards],
        glibc=lambda: glibc, run=run or never_run, environ=environ if environ is not None else {},
    )


async def never_run(argv, **_):
    raise AssertionError(f"nothing is run before the machine is found able to run vLLM: {argv}")


def nothing(*_):
    return None


# --- the versions: the sets the agent carries, for its Python and this processor ---


SETS = (
    locked("0.31.0", "3.12"), locked("0.32.0", "3.12"), locked("0.31.0", "3.13"),
    locked("0.30.0", "3.12", "aarch64", glibc="2.34"),
)


async def test_latest_is_the_newest_set_for_this_python_and_processor():
    assert (await source(*SETS, python="3.12").latest(None)).version == "0.32.0"
    assert (await source(*SETS, python="3.13").latest(None)).version == "0.31.0"
    assert (await source(*SETS, python="3.12", target=Target(LINUX, "arm64")).latest(None)).version == "0.30.0"
    with pytest.raises(InstallRefused) as refused:
        await source(*SETS, python="3.15").latest(None)
    assert str(refused.value) == (
        "this agent carries no vLLM set for Python 3.15 on x86_64: its sets are for Python 3.12, 3.13 on "
        "aarch64, x86_64; run the agent with one of those"
    )


async def test_a_version_is_only_one_the_agent_carries_a_set_for_here():
    release = await source(*SETS, python="3.12").release(None, "0.31.0")
    assert (release.version, release.tag, release.assets) == ("0.31.0", "v0.31.0", ())
    with pytest.raises(InstallRefused, match=r"^vLLM 0\.32\.0 has no locked set for Python 3\.13 on x86_64, only for Python 3\.12 on x86_64$"):
        await source(*SETS, python="3.13").release(None, "0.32.0")
    with pytest.raises(InstallRefused, match=r"vLLM 0\.99\.0 is not a version this agent carries a locked set for \(it carries 0\.32\.0, 0\.31\.0, 0\.30\.0\)"):
        await source(*SETS, python="3.12").release(None, "0.99.0")
    with pytest.raises(InstallRefused, match="is not a version"):
        await source(*SETS).release(None, "0.32.0; rm -rf /")


@pytest.mark.parametrize(("target", "says"), [
    (Target(MACOS, "arm64"), "vLLM publishes its builds for Linux only, and this machine runs macOS"),
    (Target(WINDOWS, "amd64"), "vLLM publishes its builds for Linux only, and this machine runs Windows"),
    (Target(LINUX, None), "vLLM publishes no build for the unknown processor"),
])
async def test_a_machine_vllm_publishes_nothing_for_is_refused_at_every_step(target, says, tmp_path):
    vllm = source(*SETS, target=target)
    for step in (vllm.latest(None), vllm.release(None, "0.32.0"),
                 vllm.stage(None, installs.Release("0.32.0", "v0.32.0", ()), tmp_path, nothing, nothing)):
        with pytest.raises(InstallRefused, match=says):
            await step


async def test_a_free_threaded_python_is_one_no_set_is_for(monkeypatch):
    """A free-threaded build loads only extensions built for its own ABI (`cp313t`); every set's
    wheels are built for the regular one. So it is named apart — "3.13t" — and no set matches it,
    where taking it for "3.13" would install a set that cannot be imported."""
    regular = f"{sys.version_info.major}.{sys.version_info.minor}"
    sets = [locked(python=regular)]
    monkeypatch.setattr(vllm_install.sysconfig, "get_config_var", lambda name: 1 if name == "Py_GIL_DISABLED" else None)
    assert vllm_install.this_python() == f"{regular}t"
    assert vllm_install.sets_for(vllm_install.this_python(), "x86_64", sets) == []
    assert vllm_install.newest_driver_floor(vllm_install.this_python(), "x86_64", sets) is None
    with pytest.raises(InstallRefused) as refused:
        await VllmSource(LINUX_X86, sets=sets, cards=lambda: [AMPERE], glibc=lambda: "2.35", run=never_run, environ={}).latest(None)
    assert str(refused.value).startswith(
        f"this agent carries no vLLM set for Python {regular}t (free-threaded, which vLLM's wheels are not built for) on x86_64"
    )
    monkeypatch.setattr(vllm_install.sysconfig, "get_config_var", lambda name: 0 if name == "Py_GIL_DISABLED" else None)
    assert vllm_install.this_python() == regular


def test_the_installer_takes_vllm_from_its_table():
    found = installs.source_for("vllm", LINUX_X86)
    assert isinstance(found, VllmSource) and found.engine == "vllm" and found.signature == "not checked"
    assert found.processor == "x86_64" and found.python == PYTHON
    assert found.program(Path("/e/vllm/8000/0.32.0")) == vllm_install.program_in(Path("/e/vllm/8000/0.32.0"))
    assert installs.RELEASES["vllm"] == installs.VLLM


# --- refused before anything is fetched ---


@pytest.mark.parametrize(("cards", "glibc", "says"), [
    (None, "2.35", "nvidia-smi does not answer here: no NVIDIA card, or no driver for one"),
    ((), "2.35", "no NVIDIA card"),
    ((PASCAL,), "2.35", "Tesla P100 has compute capability 6.0; this needs 7.5 or higher"),
    (({**AMPERE, "driver": "575.51.03"},), "2.35", "NVIDIA RTX A6000's driver 575.51.03 is older than 580.65.06, which vLLM's CUDA needs"),
    (({"kind": "cuda", "name": "NVIDIA RTX A6000", "driver": "580.95.05"},), "2.35", "its compute capability could not be read"),
    ((AMPERE,), "2.28", "its wheels need glibc 2.31 or later, and this machine has 2.28"),
    ((AMPERE,), None, "its wheels need glibc 2.31 or later, and this machine's C library is not glibc"),
])
async def test_a_machine_that_cannot_run_the_set_is_refused_before_anything_is_fetched(cards, glibc, says, tmp_path):
    asked = []
    vllm = source(locked(), cards=cards, glibc=glibc)
    with pytest.raises(InstallRefused) as refused:
        await vllm.stage(None, installs.Release("0.32.0", "v0.32.0", ()), tmp_path / "0.32.0.partial",
                         nothing, lambda needed, what: asked.append(needed))
    assert says in str(refused.value) and str(refused.value).startswith("vLLM 0.32.0 is not installed here: ")
    assert "gpm-agent prerequisites" in str(refused.value)
    assert asked == []  # not even the room asked for


async def test_a_turing_card_runs_vllm_where_the_host_names_it(tmp_path):
    """vLLM's own floor is 7.5; only `engine: auto` asks for 8.0 (D161)."""
    assert await source(locked(), cards=(TURING,)).reasons_against(locked()) == []
    assert vllm_install.machine_can_run([TURING], os_name="Linux", arch="x86_64", minimum=vllm_install.AUTO_MIN_CAPABILITY)


async def test_the_room_asked_for_is_four_times_the_sets_wheels_before_anything_runs(tmp_path):
    def no_room(needed, what):
        raise InstallRefused(f"no room for {needed}: {what}")

    with pytest.raises(InstallRefused) as refused:
        await source(locked(size=4_015_673_214)).stage(
            None, installs.Release("0.32.0", "v0.32.0", ()), tmp_path / "0.32.0.partial", nothing, no_room,
        )
    assert str(refused.value) == (
        "no room for 16062692856: its locked set's wheels, 4.0 GB, held while they install, and the environment "
        "they make: up to 4 times that"
    )


@pytest.mark.parametrize(("wheels", "unpacked", "source"), [
    (4.02, 8.51, 0.40),  # vLLM 0.32.0 on x86_64, from each wheel's own listing of its files
    (4.07, 8.18, 0.40),  # on aarch64
])
def test_the_room_holds_every_wheel_beside_the_environment_and_its_pyc(wheels, unpacked, source):
    """pip holds every wheel of the set while it installs them
    (`test_pip_holds_every_wheel_of_the_set_before_it_installs_any`), so at its peak the volume
    holds the wheels, what they unpack to, and the `.pyc` compiled beside each Python file — about
    as large as the source. What is asked for must cover that, with room to spare as sets grow."""
    peak = wheels + unpacked + source
    assert vllm_install.UNPACKED_MARGIN * wheels >= peak * 1.2


async def test_an_install_stopped_half_way_leaves_nothing_counted_against_the_room(tmp_path):
    """The wheels a stopped install left in pip's work directory are gone before the room is
    asked for: a set's wheels are gigabytes, and would be counted twice."""
    engine_dir = tmp_path / "vllm" / "8000"
    (engine_dir / "pip-work" / "pip-unpack-1").mkdir(parents=True)
    (engine_dir / "pip-work" / "pip-unpack-1" / "torch.whl").write_bytes(b"x" * 1000)
    seen = []

    def room(needed, what):
        seen.append((engine_dir / "pip-work").exists())
        raise InstallRefused("no room")

    with pytest.raises(InstallRefused, match="no room"):
        await source(locked()).stage(
            None, installs.Release("0.32.0", "v0.32.0", ()), engine_dir / "0.32.0.partial", nothing, room,
        )
    assert seen == [False]


async def test_an_index_the_environment_names_must_be_an_address_or_a_directory(tmp_path):
    assert vllm_install.index_arguments({}) == ["--index-url", "https://pypi.org/simple"]
    assert vllm_install.index_arguments({"GPM_PYPI_INDEX": "https://mirror.example/simple"}) == [
        "--index-url", "https://mirror.example/simple",
    ]
    assert vllm_install.index_arguments({"GPM_PYPI_INDEX": str(tmp_path)}) == ["--no-index", "--find-links", str(tmp_path)]
    for named in ("wheels", str(tmp_path / "missing"), "ftp://example/simple"):
        with pytest.raises(InstallRefused, match="neither an http"):
            vllm_install.index_arguments({"GPM_PYPI_INDEX": named})
    with pytest.raises(InstallRefused, match="neither an http"):
        await source(locked(), environ={"GPM_PYPI_INDEX": "wheels"}).stage(
            None, installs.Release("0.32.0", "v0.32.0", ()), tmp_path / "0.32.0.partial", nothing, nothing,
        )


@pytest.mark.parametrize("named", [
    "https://owner:notreal@mirror.example/simple",
    "http://token-notreal@mirror.example:8080/simple",
    "ftp://owner:notreal@mirror.example/simple",
])
async def test_an_index_with_a_credential_in_its_address_is_refused_and_the_credential_never_said(named, tmp_path):
    """Every command the agent runs is written to the install log, pip's arguments with it: a
    password in the index's address would be kept there in plain text (CLAUDE.md: no secret in a
    log). Refused before anything runs — and the refusal, which reaches the pool's console, does
    not repeat it."""
    with pytest.raises(InstallRefused) as refused:
        vllm_install.index_arguments({"GPM_PYPI_INDEX": named})
    said = str(refused.value)
    assert said.startswith("GPM_PYPI_INDEX names an index with a user name or password in its address (")
    assert "notreal" not in said and "mirror.example" in said and "install log" in said
    engine_dir = tmp_path / "vllm" / "8000"
    with pytest.raises(InstallRefused, match="user name or password"):
        await source(locked(), environ={"GPM_PYPI_INDEX": named}).stage(
            None, installs.Release("0.32.0", "v0.32.0", ()), engine_dir / "0.32.0.partial", nothing, nothing,
        )
    assert not (engine_dir / "logs").exists()


# --- staging, with the commands stood in for ---


class Commands:
    """pip, venv and the check, as the source runs them: each recorded, and each doing what a
    test says — by default making an environment whose `vllm` answers as `says`."""

    def __init__(self, *, venv=0, pip=(0, ""), check=None, says="0.32.0", first_line=None):
        self.venv, self.pip, self.check, self.says = venv, pip, check, says
        #: The program's first line pip writes — None: the environment's own interpreter.
        self.first_line = first_line
        self.calls: list[tuple[list[str], dict]] = []

    async def __call__(self, argv, **options):
        self.calls.append((list(argv), options))
        if argv[1:4] == ["-I", "-m", "venv"]:
            if self.venv == 0:
                (Path(argv[4]) / "bin").mkdir(parents=True)
                (Path(argv[4]) / "bin" / "python").write_text("")
            return Ran(self.venv, "" if self.venv == 0 else "Error: ensurepip is not available")
        if argv[1:5] == ["-I", "-m", "pip", "install"]:
            code, said = self.pip
            if code == 0:
                venv = Path(argv[0]).parent.parent
                (venv / "bin" / "vllm").write_text(f"{self.first_line or f'#!{venv}/bin/python'}\nimport sys\n")
            return Ran(code, said)
        if argv[-1] == "--version":
            return self.check or Ran(0, f"INFO 10-30 a line of logging\n{self.says}\n")
        raise AssertionError(argv)

    def argv(self, which: str) -> list[str]:
        return next(argv for argv, _ in self.calls if which in argv)

    def options(self, which: str) -> dict:
        return next(options for argv, options in self.calls if which in argv)


async def stage(commands, tmp_path, *sets, environ=None):
    engine_dir = tmp_path / "vllm" / "8000"
    into = engine_dir / "0.32.0.partial"
    into.mkdir(parents=True)
    steps = []
    vllm = source(*(sets or (locked(requirements="vllm==0.32.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"),)),
                  run=commands, environ=environ)
    await vllm.stage(None, installs.Release("0.32.0", "v0.32.0", ()), into,
                     lambda step, done, total: steps.append(step), nothing)
    return engine_dir, into, steps


async def test_a_version_is_installed_with_exactly_its_set_wheels_only_checked_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_notreal")
    monkeypatch.setenv("PIP_INDEX_URL", "https://elsewhere.example/simple")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    commands = Commands()
    engine_dir, into, steps = await stage(commands, tmp_path, environ={"GPM_PYPI_INDEX": str(tmp_path)})

    venv = into / "venv"
    assert commands.argv("venv") == [sys.executable, "-I", "-m", "venv", str(venv)]
    assert commands.argv("pip") == [
        str(venv / "bin" / "python"), "-I", "-m", "pip", "install", "--isolated",
        "--require-hashes", "--only-binary=:all:", "--no-deps", "--no-input",
        "--disable-pip-version-check", "--no-cache-dir", "--progress-bar=off",
        "--no-index", "--find-links", str(tmp_path), "-r", str(into / "requirements.txt"),
    ]
    assert (into / "requirements.txt").read_text().startswith("vllm==0.32.0 \\\n    --hash=sha256:aaaa")
    pip = commands.options("pip")
    # At low priority, on the engines' volume, with the agent's short list of names: no token,
    # none of pip's own settings — the cards' are the owner's and pass.
    assert pip["low_priority"] is True and pip["env"]["TMPDIR"] == str(engine_dir / "pip-work")
    assert "HF_TOKEN" not in pip["env"] and "PIP_INDEX_URL" not in pip["env"] and pip["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    # No configuration file at all: the machine's and the environment's are read even --isolated.
    assert pip["env"]["PIP_CONFIG_FILE"] == os.devnull
    assert pip["log"] == engine_dir / "logs" / "install.log" and pip["timeout_s"] == vllm_install.INSTALL_TIMEOUT_S
    assert commands.argv("--version") == [str(venv / "bin" / "python"), "-I", str(venv / "bin" / "vllm"), "--version"]
    assert commands.options("--version")["timeout_s"] == vllm_install.CHECK_TIMEOUT_S
    assert steps == ["downloading", "verifying"]
    assert not (engine_dir / "pip-work").exists()
    # The program names the place the version will live, not the one it was made in.
    assert (venv / "bin" / "vllm").read_text().splitlines()[0] == f"#!{engine_dir / '0.32.0' / 'venv' / 'bin' / 'python'}"


@pytest.mark.parametrize(("commands", "says"), [
    (Commands(venv=1), "could not be made with this agent's Python (exit code 1: Error: ensurepip is not available): "
                       "it needs Python's venv module (python3-venv on Debian and Ubuntu)"),
    (Commands(pip=(1, "ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE.\n    torch==2.13.0")),
     "a file fetched for vLLM 0.32.0 does not match the digest its locked set records; nothing of it is kept"),
    (Commands(pip=(1, "ERROR: No matching distribution found for torch==2.13.0")),
     "installing vLLM 0.32.0's locked set failed (exit code 1: ERROR: No matching distribution found for torch==2.13.0)"),
    (Commands(check=Ran(1, "Traceback (most recent call last):\nImportError: libcudart.so.13: cannot open shared object file")),
     "vLLM 0.32.0, installed, does not run: exit code 1: ImportError: libcudart.so.13"),
    (Commands(check=Ran(None, "it did not finish within 300s and was stopped")),
     "vLLM 0.32.0, installed, does not run: it did not run to the end"),
    (Commands(says="0.31.0"), "vLLM 0.32.0, installed, says it is another version (exit code 0: 0.31.0)"),
    (Commands(says="0.32.01"), "says it is another version"),
    # The check runs the program as a script, by its environment's interpreter, and would pass
    # whatever its first line names — the line the launcher runs it by.
    (Commands(first_line="#!/usr/bin/python3"), "vLLM 0.32.0, installed, has a `vllm` that names /usr/bin/python3 to run it"),
    (Commands(first_line="import sys"), "has a `vllm` that names no interpreter to run it"),
])
async def test_a_version_that_does_not_install_or_run_is_refused(commands, says, tmp_path):
    with pytest.raises(InstallRefused) as refused:
        await stage(commands, tmp_path)
    assert says in str(refused.value)
    assert not (tmp_path / "vllm" / "8000" / "pip-work").exists()


def test_the_environment_is_moved_to_where_the_version_will_live(tmp_path):
    made, lives = tmp_path / "0.32.0.partial" / "venv", tmp_path / "0.32.0" / "venv"
    bin_ = made / "bin"
    bin_.mkdir(parents=True)
    body = f"import re\nimport sys\nprint('{made}/bin/python is a string here')\n"
    (bin_ / "vllm").write_bytes(f"#!{made}/bin/python\n{body}".encode())
    # pip's form for a path too long for `#!`: /bin/sh, running the interpreter it names.
    (bin_ / "pip").write_bytes(f"#!/bin/sh\n'''exec' '{made}/bin/python' \"$0\" \"$@\"\n' '''\nimport pip\n".encode())
    (bin_ / "activate").write_text(f"VIRTUAL_ENV='{made}'\nexport VIRTUAL_ENV\n")
    (bin_ / "ninja").write_bytes(b"\x7fELF" + str(made).encode())  # a program: never rewritten
    (bin_ / "python").symlink_to(sys.executable)
    (made / "pyvenv.cfg").write_text(f"home = /usr/bin\ncommand = /usr/bin/python3 -m venv {made}\n")

    vllm_install.move_environment(made, lives)

    # Only where a script names its interpreter — its first lines — never in what it says.
    assert (bin_ / "vllm").read_text() == f"#!{lives}/bin/python\n{body}"
    assert (bin_ / "pip").read_text().splitlines()[1] == f"'''exec' '{lives}/bin/python' \"$0\" \"$@\""
    assert (bin_ / "activate").read_text() == f"VIRTUAL_ENV='{lives}'\nexport VIRTUAL_ENV\n"
    assert (bin_ / "ninja").read_bytes() == b"\x7fELF" + str(made).encode()
    assert os.readlink(bin_ / "python") == sys.executable
    assert (made / "pyvenv.cfg").read_text().endswith(f"-m venv {lives}\n")


def test_an_environment_named_through_dotdot_is_moved_as_python_names_it(tmp_path):
    """Python names its own interpreter absolute with no `..`, and pip writes that name: an
    environment made under a path holding `..` is found by its normalised name, or not at all."""
    (tmp_path / "data").mkdir()
    made = tmp_path / "data" / ".." / "0.32.0.partial" / "venv"
    lives = tmp_path / "data" / ".." / "0.32.0" / "venv"
    (made / "bin").mkdir(parents=True)
    written = os.path.normpath(made)
    (made / "bin" / "vllm").write_text(f"#!{written}/bin/python\nimport sys\n")

    vllm_install.move_environment(made, lives)

    assert vllm_install.interpreter_named(made / "bin" / "vllm") == os.fsencode(os.path.normpath(lives / "bin" / "python"))


@pytest.mark.parametrize(("head", "names"), [
    ("#!/e/0.32.0/venv/bin/python\n", b"/e/0.32.0/venv/bin/python"),
    # pip's form for an interpreter path too long for `#!`, or holding a space (quoted).
    ("#!/bin/sh\n'''exec' /e/0.32.0/venv/bin/python \"$0\" \"$@\"\n' '''\n", b"/e/0.32.0/venv/bin/python"),
    ("#!/bin/sh\n'''exec' \"/e e/venv/bin/python\" \"$0\" \"$@\"\n' '''\n", b"/e e/venv/bin/python"),
    ("#!/bin/sh\necho elsewhere\n", None),
    ("import sys\n", None),
])
def test_the_interpreter_a_program_names_is_read_from_its_first_lines(tmp_path, head, names):
    (tmp_path / "vllm").write_text(f"{head}import sys\n")
    assert vllm_install.interpreter_named(tmp_path / "vllm") == names


# --- with real pip, from a directory of wheels made here ---


def wheel(directory: Path, name: str, version: str, files: dict[str, str], entry_points: str = "") -> Path:
    """A pure-Python wheel, as a release would publish one."""
    dist = f"{name}-{version}.dist-info"
    contents = {path: text.encode() for path, text in files.items()}
    contents[f"{dist}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    contents[f"{dist}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: gpm-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    if entry_points:
        contents[f"{dist}/entry_points.txt"] = entry_points.encode()
    record = []
    for path, data in contents.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(data)}")
    record.append(f"{dist}/RECORD,,")
    contents[f"{dist}/RECORD"] = ("\n".join(record) + "\n").encode()
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in contents.items():
            archive.writestr(member, data)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_release(directory: Path) -> tuple[Path, Path]:
    directory.mkdir()
    vllm = wheel(directory, "vllm", "0.32.0", {
        "vllm/__init__.py": "from fakedep import NAME\n",
        "vllm/cli.py": "def main():\n    print('0.32.0')\n",
    }, "[console_scripts]\nvllm = vllm.cli:main\n")
    dep = wheel(directory, "fakedep", "1.0", {"fakedep/__init__.py": "NAME = 'fakedep'\n"})
    return vllm, dep


class Started:
    """The engine's server, as the installer drives it: each start recorded, each answering."""

    foreign_reason = "not used here"

    def __init__(self):
        self.started: list[tuple[Path, str]] = []

    def ours(self, *, fresh=False):
        return bool(self.started)

    def running_version(self):
        return self.started[-1][1] if self.started else None

    async def foreign(self):
        return False

    async def start(self, program, version, timeout_s):
        self.started.append((program, version))
        return None

    async def stop(self):
        return None


async def test_a_set_installs_with_real_pip_and_its_vllm_runs_from_where_the_version_lives(tmp_path):
    vllm, dep = fake_release(tmp_path / "wheels")
    requirements = (
        f"fakedep==1.0 \\\n    --hash=sha256:{sha256(dep)}\n"
        f"vllm==0.32.0 \\\n    --hash=sha256:{sha256(vllm)}\n"
    )
    vllm_source = VllmSource(
        LINUX_X86, sets=[locked(requirements=requirements, size=vllm.stat().st_size + dep.stat().st_size)],
        cards=lambda: [AMPERE], glibc=lambda: "2.35", environ={"GPM_PYPI_INDEX": str(tmp_path / "wheels")},
    )
    engine_dir, server = tmp_path / "engines" / "vllm" / "8000", Started()
    installer = Installer(vllm_source, engine_dir, server, free_disk_bytes=lambda: 10**12)  # not this disk's
    await installer.want("latest", False)
    await installer.finished()

    state = installer.state()
    assert state["error"] is None and state["running"] == "0.32.0" and state["signature"] == "not checked"
    program = engine_dir / "0.32.0" / "venv" / "bin" / "vllm"
    assert server.started == [(program, "0.32.0")]
    assert sorted(p.name for p in engine_dir.iterdir()) == ["0.32.0", "logs", "state.json"]
    assert (engine_dir / "0.32.0" / "requirements.txt").read_text() == requirements
    assert program.read_text().splitlines()[0] == f"#!{engine_dir / '0.32.0' / 'venv' / 'bin' / 'python'}"
    ran = subprocess.run([str(program), "--version"], capture_output=True, text=True, timeout=60, check=False)
    assert (ran.returncode, ran.stdout) == (0, "0.32.0\n")
    assert "Successfully installed" in (engine_dir / "logs" / "install.log").read_text()


def a_set_of(directory: Path) -> tuple[str, int]:
    """The fake release made in `directory`, as a set's requirements and size."""
    vllm, dep = fake_release(directory)
    requirements = (
        f"fakedep==1.0 \\\n    --hash=sha256:{sha256(dep)}\n"
        f"vllm==0.32.0 \\\n    --hash=sha256:{sha256(vllm)}\n"
    )
    return requirements, vllm.stat().st_size + dep.stat().st_size


async def test_an_engines_directory_named_through_dotdot_gets_a_vllm_that_runs_where_it_lives(tmp_path):
    """`engines_path` is absolute, and may hold `..`. pip writes each program's first line with the
    interpreter's name as Python makes it — with no `..` — so the version is made, moved and
    checked by that name: its `vllm` runs as the launcher runs it, by its first line."""
    requirements, size = a_set_of(tmp_path / "wheels")
    vllm_source = VllmSource(
        LINUX_X86, sets=[locked(requirements=requirements, size=size)], cards=lambda: [AMPERE], glibc=lambda: "2.35",
        environ={"GPM_PYPI_INDEX": str(tmp_path / "wheels")},
    )
    (tmp_path / "data").mkdir()
    engine_dir, server = tmp_path / "data" / ".." / "engines" / "vllm" / "8000", Started()
    installer = Installer(vllm_source, engine_dir, server, free_disk_bytes=lambda: 10**12)  # not this disk's
    await installer.want("latest", False)
    await installer.finished()

    assert installer.state()["error"] is None and installer.state()["running"] == "0.32.0"
    program = engine_dir / "0.32.0" / "venv" / "bin" / "vllm"
    lives = os.path.normpath(engine_dir / "0.32.0" / "venv" / "bin" / "python")
    assert program.read_text().splitlines()[0] == f"#!{lives}"
    ran = subprocess.run([str(program), "--version"], capture_output=True, text=True, timeout=60, check=False)
    assert (ran.returncode, ran.stdout) == (0, "0.32.0\n")


async def test_pip_reads_no_configuration_file_not_even_the_environments_own(tmp_path):
    """`--isolated` leaves out pip's environment variables and the user's file, but pip still reads
    the machine's (`/etc/pip.conf`, `/etc/xdg/pip/pip.conf`) and the environment's own `pip.conf`;
    `PIP_CONFIG_FILE` naming the null device leaves out every one. One the environment carries,
    asking for a `--user` install — which pip refuses inside an environment — stands for them."""
    requirements, size = a_set_of(tmp_path / "wheels")

    async def with_a_pip_conf(argv, **options):
        ran = await host_platform.run_command(argv, **options)
        if argv[1:4] == ["-I", "-m", "venv"] and ran.code == 0:
            (Path(argv[4]) / "pip.conf").write_text("[install]\nuser = true\n")
        return ran

    vllm_source = VllmSource(
        LINUX_X86, sets=[locked(requirements=requirements, size=size)], cards=lambda: [AMPERE], glibc=lambda: "2.35",
        environ={"GPM_PYPI_INDEX": str(tmp_path / "wheels")}, run=with_a_pip_conf,
    )
    engine_dir = tmp_path / "engines" / "vllm" / "8000"
    installer = Installer(vllm_source, engine_dir, Started(), free_disk_bytes=lambda: 10**12)  # not this disk's
    await installer.want("latest", False)
    await installer.finished()
    assert installer.state()["error"] is None and installer.state()["running"] == "0.32.0"
    assert (engine_dir / "0.32.0" / "venv" / "pip.conf").is_file()  # there, and not read


async def test_pip_holds_every_wheel_of_the_set_before_it_installs_any(tmp_path):
    """Why the room asked for counts the wheels beside the environment they make
    (`UNPACKED_MARGIN`): under `--require-hashes` pip fetches the whole set into its work directory
    while it resolves, before it installs anything. Served by an index here, as PyPI serves it."""
    files, simple = tmp_path / "index" / "files", tmp_path / "index" / "simple"
    files.parent.mkdir()
    made = [*fake_release(files), *(wheel(files, name, "1.0", {f"{name}/__init__.py": "X = 1\n"}) for name in ("pkga", "pkgb"))]
    requirements = ""
    for path in made:
        name = path.name.split("-")[0]
        (simple / name).mkdir(parents=True)
        (simple / name / "index.html").write_text(f'<a href="../../files/{path.name}#sha256={sha256(path)}">{path.name}</a>\n')
        requirements += f"{name}=={path.name.split('-')[1]} \\\n    --hash=sha256:{sha256(path)}\n"
    engine_dir = tmp_path / "engines" / "vllm" / "8000"
    work, venv = engine_dir / "pip-work", engine_dir / "0.32.0.partial" / "venv"
    seen: list[tuple[str, int, bool]] = []

    class Index(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path / "index"), **kwargs)

        def do_GET(self):
            if self.path.endswith(".whl"):
                held = len(list(work.rglob("*.whl"))) if work.is_dir() else 0
                installed = [p.name for p in venv.rglob("*.dist-info") if not p.name.startswith("pip-")]
                seen.append((Path(self.path).name, held, bool(installed)))
            super().do_GET()

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Index)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        vllm_source = VllmSource(
            LINUX_X86, sets=[locked(requirements=requirements)], cards=lambda: [AMPERE], glibc=lambda: "2.35",
            environ={"GPM_PYPI_INDEX": f"http://127.0.0.1:{server.server_address[1]}/simple"},
        )
        installer = Installer(vllm_source, engine_dir, Started(), free_disk_bytes=lambda: 10**12)  # not this disk's
        await installer.want("0.32.0", False)
        await installer.finished()
    finally:
        server.shutdown()
        server.server_close()
    assert installer.state()["error"] is None and installer.state()["running"] == "0.32.0"
    # Each wheel fetched while every one fetched before it is still held, and none installed yet.
    assert sorted(name for name, _, _ in seen) == sorted(path.name for path in made)
    assert [(held, installed) for _, held, installed in seen] == [(count, False) for count in range(len(made))]


async def test_a_wheel_that_does_not_match_its_sets_digest_refuses_the_version_and_nothing_stays(tmp_path):
    vllm, dep = fake_release(tmp_path / "wheels")
    requirements = (
        f"fakedep==1.0 \\\n    --hash=sha256:{'0' * 64}\n"
        f"vllm==0.32.0 \\\n    --hash=sha256:{sha256(vllm)}\n"
    )
    vllm_source = VllmSource(
        LINUX_X86, sets=[locked(requirements=requirements)], cards=lambda: [AMPERE], glibc=lambda: "2.35",
        environ={"GPM_PYPI_INDEX": str(tmp_path / "wheels")},
    )
    engine_dir, server = tmp_path / "engines" / "vllm" / "8000", Started()
    installer = Installer(vllm_source, engine_dir, server, free_disk_bytes=lambda: 10**12)  # not this disk's
    await installer.want("0.32.0", False)
    await installer.finished()

    state = installer.state()
    assert state["running"] is None and state["staged"] is None and server.started == []
    assert state["error"].startswith("a file fetched for vLLM 0.32.0 does not match the digest its locked set records")
    assert sorted(p.name for p in engine_dir.iterdir()) == ["logs", "state.json"]


# --- the sets the agent carries ---


def test_every_set_the_agent_carries_is_whole_pinned_and_hashed_once():
    """What GPM's release generated (`tools/vllm-lock`): each wheel pinned, with the one digest pip
    may take — so every machine installs exactly the same files — and small enough to carry."""
    paths = sorted(vllm_install.SETS_DIR.glob("*.json"))
    assert paths, "the agent carries no vLLM set"
    assert sum(p.stat().st_size for p in paths) < 2_000_000
    carried = vllm_install.load_sets()
    assert len({(s.vllm, s.python, s.platform) for s in carried}) == len(carried) == len(paths)
    for path, found in zip(paths, carried, strict=True):
        assert path.name == f"vllm-{found.vllm}-py{found.python}-{found.platform}.json"
        assert installs.VERSION.fullmatch(found.vllm) and found.platform in vllm_install.PLATFORMS
        assert vllm_install.driver_key(found.driver_floor) and vllm_install.driver_key(found.glibc)
        lines = found.requirements.replace("\\\n", "").splitlines()
        names = [line.split("==")[0] for line in lines]
        assert len(names) == len(set(names)) and f"vllm=={found.vllm}" in [line.split()[0] for line in lines]
        for line in lines:
            pin, *hashes = line.split()
            assert "==" in pin and len(hashes) == 1 and hashes[0].startswith("--hash=sha256:") and len(hashes[0]) == 14 + 64
        assert found.size_bytes > 10**9
    # Each version the agent carries, for every Python vLLM supports and both processors.
    for version in {s.vllm for s in carried}:
        assert {(s.python, s.platform) for s in carried if s.vllm == version} == {
            (python, platform) for python in ("3.11", "3.12", "3.13", "3.14") for platform in vllm_install.PLATFORMS
        }


def test_vllm_0_32_0s_sets_need_cuda_13s_driver():
    """torch 2.13.0 on PyPI is built for CUDA 13 (`cuda-toolkit==13.0.3`), whose drivers start at
    580 — not CUDA 12.9's 525 (decisions.md, Unverified assumptions)."""
    assert vllm_install.newest_driver_floor("3.12", "x86_64") == "580.65.06"
    assert all(s.driver_floor == "580.65.06" for s in vllm_install.load_sets() if s.vllm == "0.32.0")


# --- the release's tool ---


def load_tool():
    """It is a script, not a package: run by GPM's release, never imported by the agent."""
    path = REPO / "tools" / "vllm-lock"
    spec = importlib.util.spec_from_loader("vllm_lock", importlib.machinery.SourceFileLoader("vllm_lock", str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()

#: What PyPI lists for each pinned release: every file, for every platform, as its JSON API does.
PYPI = {
    ("vllm", "0.32.0"): [
        ("vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl", 326_518_482),
        ("vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl", 321_437_289),
        ("vllm-0.32.0.tar.gz", 44_549_956),
    ],
    ("torch", "2.13.0"): [
        ("torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl", 526_605_292),
        ("torch-2.13.0-cp312-cp312-manylinux_2_28_aarch64.whl", 427_210_998),
        ("torch-2.13.0-cp313-cp313-manylinux_2_28_x86_64.whl", 526_602_329),
        ("torch-2.13.0-cp312-cp312-win_amd64.whl", 122_057_313),
    ],
    ("llguidance", "1.7.6"): [
        ("llguidance-1.7.6-cp314-cp314t-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", 3_000_000),
        ("llguidance-1.7.6-cp39-abi3-manylinux_2_31_x86_64.whl", 2_900_000),
        ("llguidance-1.7.6-cp39-abi3-manylinux_2_31_aarch64.whl", 2_800_000),
    ],
    ("numpy", "2.3.5"): [
        ("numpy-2.3.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", 16_000_000),
        ("numpy-2.3.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", 15_000_000),
        ("numpy-2.3.5-cp312-cp312-musllinux_1_2_x86_64.whl", 14_000_000),
    ],
    ("regex", "2026.9.29"): [("regex-2026.9.29-py3-none-any.whl", 400_000)],
    ("nvidia-cuda-runtime", "13.0.96"): [("nvidia_cuda_runtime-13.0.96-py3-none-manylinux_2_27_x86_64.whl", 2_200_000)],
}


def digest(filename: str) -> str:
    return hashlib.sha256(filename.encode()).hexdigest()


def pypi_files(name, version):
    return [
        {"filename": filename, "size": size, "sha256": digest(filename),
         "packagetype": "sdist" if filename.endswith(".tar.gz") else "bdist_wheel"}
        for filename, size in PYPI[(name, version)]
    ]


def uv_says(pins=None, missing=()):
    """What uv writes: every pin with the digest of every file PyPI lists for it."""
    text = ""
    for (name, version), files in sorted(pins or PYPI.items()):
        hashes = "".join(f" \\\n    --hash=sha256:{digest(f)}" for f, _ in files if f not in missing)
        text += f"{name}=={version}{hashes}\n"
    return text


def uv(levels_that_resolve=("2.31",), said=None):
    asked = []

    def compile(requirement, python, platform, glibc, index, exclude_newer):
        asked.append((requirement, python, platform, glibc, index, exclude_newer))
        if glibc not in levels_that_resolve:
            raise TOOL.LockFailed(f"× No solution found\nBecause llguidance has no wheels for manylinux_{glibc}")
        return said if said is not None else uv_says()

    return compile, asked


def test_a_set_is_resolved_at_the_oldest_glibc_every_wheel_is_published_for():
    compile, asked = uv(levels_that_resolve=("2.31", "2.34"))
    made = TOOL.lock("0.32.0", "3.12", "x86_64", compile=compile, files=pypi_files)
    assert [glibc for *_, glibc, _, _ in asked] == ["2.28", "2.31"]
    assert asked[0][:3] == ("vllm[audio]==0.32.0", "3.12", "x86_64") and asked[0][4] == "https://pypi.org/simple"
    assert list(made) == ["vllm", "python", "platform", "driver_floor", "glibc", "cuda_runtime", "wheels", "size_bytes", "requirements"]
    assert (made["vllm"], made["python"], made["platform"], made["glibc"]) == ("0.32.0", "3.12", "x86_64", "2.31")
    assert (made["driver_floor"], made["cuda_runtime"], made["wheels"]) == ("580.65.06", "13.0.96", 6)
    assert LockedSet.from_json(made).glibc == "2.31"


def test_each_pin_keeps_only_the_one_wheel_pip_would_install_there():
    made = TOOL.lock("0.32.0", "3.12", "x86_64", compile=uv()[0], files=pypi_files)
    chosen = {
        "llguidance": "llguidance-1.7.6-cp39-abi3-manylinux_2_31_x86_64.whl",  # not the free-threaded build
        "numpy": "numpy-2.3.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",  # the newer manylinux
        "nvidia-cuda-runtime": "nvidia_cuda_runtime-13.0.96-py3-none-manylinux_2_27_x86_64.whl",
        "regex": "regex-2026.9.29-py3-none-any.whl",
        "torch": "torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "vllm": "vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl",
    }
    assert made["requirements"] == "".join(
        f"{name}=={version} \\\n    --hash=sha256:{digest(chosen[name])}\n" for (name, version) in sorted(PYPI)
    )
    sizes = {filename: size for files in PYPI.values() for filename, size in files}
    assert made["size_bytes"] == sum(sizes[filename] for filename in chosen.values())


def test_a_wheel_whose_digest_uv_did_not_list_is_refused():
    said = uv_says(missing=("torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl",))
    with pytest.raises(TOOL.LockFailed, match="digest on PyPI is not one uv listed for torch 2.13.0"):
        TOOL.lock("0.32.0", "3.12", "x86_64", compile=uv(said=said)[0], files=pypi_files)


def test_a_pin_with_no_wheel_for_the_platform_is_refused():
    with pytest.raises(TOOL.LockFailed, match="numpy 2.3.5 has no wheel for Python 3.11 on x86_64 with glibc 2.31"):
        TOOL.lock("0.32.0", "3.11", "x86_64", compile=uv()[0], files=pypi_files)


@pytest.mark.parametrize(("runtime", "floor"), [
    (("nvidia-cuda-runtime-cu12", "12.8.90"), "525.60.13"),  # named with its major, up to CUDA 12
    (("nvidia-cuda-runtime", "13.0.96"), "580.65.06"),       # by its version, from 13
    (("cuda-toolkit", "13.0.3"), "580.65.06"),               # no runtime pinned on its own
])
def test_the_driver_floor_is_the_cuda_major_of_the_sets_runtime(runtime, floor):
    major, version = TOOL.cuda_runtime({"vllm": ("0.32.0", set()), runtime[0]: (runtime[1], set())})
    assert (TOOL.DRIVER_FLOORS[major], version) == (floor, runtime[1])


def test_a_set_with_no_cuda_runtime_or_one_the_table_does_not_name_is_refused():
    with pytest.raises(TOOL.LockFailed, match="no CUDA runtime"):
        TOOL.cuda_runtime({"vllm": ("0.32.0", set()), "nvidia-cudnn-cu13": ("9.20.0.48", set())})
    with pytest.raises(TOOL.LockFailed, match="more than one major"):
        TOOL.cuda_runtime({"nvidia-cuda-runtime-cu12": ("12.8.90", set()), "nvidia-cuda-runtime": ("13.0.96", set())})
    said = uv_says().replace("nvidia-cuda-runtime==13.0.96", "nvidia-cuda-runtime==11.8.89")

    def files(name, version):
        return pypi_files(name, "13.0.96" if name == "nvidia-cuda-runtime" else version)

    with pytest.raises(TOOL.LockFailed, match=r"CUDA 11\), whose least driver this tool's table does not name"):
        TOOL.lock("0.32.0", "3.12", "x86_64", compile=uv(said=said)[0], files=files)


def test_the_order_of_tags_is_pips_own():
    """The tags `packaging` — pip's own — gives CPython on these platforms, in its order."""
    tags = pytest.importorskip("packaging.tags")
    platforms = TOOL.platform_tags("x86_64", "2.31")
    assert platforms[:2] == ["manylinux_2_31_x86_64", "manylinux_2_30_x86_64"]
    assert platforms[platforms.index("manylinux_2_17_x86_64") + 1] == "manylinux2014_x86_64"
    assert platforms[-3:] == ["manylinux_2_5_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    assert TOOL.platform_tags("aarch64", "2.34")[-3:] == ["manylinux_2_17_aarch64", "manylinux2014_aarch64", "linux_aarch64"]
    pips = [*tags.cpython_tags((3, 12), ["cp312"], platforms), *tags.compatible_tags((3, 12), "cp312", platforms)]
    assert TOOL.supported_tags("3.12", "x86_64", "2.31") == [(t.interpreter, t.abi, t.platform) for t in pips]


def test_the_tool_writes_every_set_or_none(tmp_path, capsys):
    """Pure-Python and stable-ABI wheels only, so one resolution serves every Python asked."""
    portable = [("vllm", "0.32.0"), ("regex", "2026.9.29"), ("nvidia-cuda-runtime", "13.0.96")]
    said = uv_says(pins=[(pin, PYPI[pin]) for pin in portable])
    asked = []

    def compile(requirement, python, platform, glibc, index, exclude_newer):
        asked.append(python)
        if python == "3.11":
            raise TOOL.LockFailed("× No solution found when resolving dependencies")
        return said

    out = tmp_path / "sets"
    assert TOOL.main(["0.32.0", "--python", "3.12", "3.13", "--platform", "x86_64", "--out", str(out)],
                     compile=compile, files=pypi_files) == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == ["vllm-0.32.0-py3.12-x86_64.json", "vllm-0.32.0-py3.13-x86_64.json"]
    assert [LockedSet.from_json(json.loads((out / name).read_text())).python for name in written] == ["3.12", "3.13"]
    assert "2 sets" in capsys.readouterr().out

    # One that does not resolve: nothing is written — a release never carries part of a version.
    elsewhere = tmp_path / "none"
    assert TOOL.main(["0.32.0", "--python", "3.12", "3.11", "--platform", "x86_64", "--out", str(elsewhere)],
                     compile=compile, files=pypi_files) == 1
    assert not elsewhere.exists()
    assert "vllm[audio]==0.32.0 does not resolve for Python 3.11 on x86_64" in capsys.readouterr().err


def test_every_set_of_a_run_is_resolved_from_pypi_as_it_stood_when_the_run_began(tmp_path, capsys):
    """A version's sets are resolved one after another, for minutes: a dependency released
    meanwhile would make one Python's or processor's set differ from another's. Every resolution
    is given the one moment the run began (`--exclude-newer`)."""
    portable = [("vllm", "0.32.0"), ("regex", "2026.9.29"), ("nvidia-cuda-runtime", "13.0.96")]
    compile, asked = uv(said=uv_says(pins=[(pin, PYPI[pin]) for pin in portable]))
    moments = iter(["2026-10-30T09:00:00Z", "2026-10-30T09:05:00Z"])
    assert TOOL.main(["0.32.0", "--python", "3.12", "3.13", "--platform", "x86_64", "--out", str(tmp_path / "sets")],
                     compile=compile, files=pypi_files, clock=lambda: next(moments)) == 0
    assert [python for _, python, *_ in asked] == ["3.12", "3.12", "3.13", "3.13"]  # 2.28, then 2.31, each
    assert {exclude_newer for *_, exclude_newer in asked} == {"2026-10-30T09:00:00Z"}
    assert "as it stood at 2026-10-30T09:00:00Z" in capsys.readouterr().out
    assert TOOL.now_rfc3339().endswith("Z") and len(TOOL.now_rfc3339()) == len("2026-10-30T09:00:00Z")


def test_uv_is_asked_for_nothing_uploaded_after_that_moment(monkeypatch):
    ran = []

    def run(argv, **options):
        ran.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="vllm==0.32.0\n", stderr="")

    monkeypatch.setattr(TOOL.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(TOOL.subprocess, "run", run)
    TOOL.uv_compile("vllm[audio]==0.32.0", "3.12", "x86_64", "2.31", TOOL.INDEX, "2026-10-30T09:00:00Z")
    TOOL.uv_compile("vllm[audio]==0.32.0", "3.12", "x86_64", "2.31", TOOL.INDEX)
    assert ran[0][ran[0].index("--exclude-newer") + 1] == "2026-10-30T09:00:00Z"
    assert "--exclude-newer" not in ran[1]


def test_a_version_the_agent_could_not_name_a_directory_by_is_refused():
    for version in ("0.32.0rc1", "0.7.3.post1", "../0.32.0"):
        with pytest.raises(TOOL.LockFailed, match="not a version of digits and dots"):
            TOOL.lock(version, "3.12", "x86_64", compile=uv()[0], files=pypi_files)


def test_the_real_tool_runs_with_the_standard_library_alone():
    """GPM's release runs it with any Python 3.9 or later, and `uv` on PATH: no package of GPM's."""
    ran = subprocess.run([sys.executable, "-I", str(REPO / "tools" / "vllm-lock"), "--help"],
                         capture_output=True, text=True, timeout=60, check=False)
    assert ran.returncode == 0 and "vllm-lock" in ran.stdout
    text = (REPO / "tools" / "vllm-lock").read_text()
    assert "import gpm_" not in text and "from gpm_" not in text
