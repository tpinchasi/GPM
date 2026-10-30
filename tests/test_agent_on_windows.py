"""The agent on Windows: what it reports about the machine, and that it loads and serves there
(D154).

docs/spec/host-agent.md §3. Run on any system: Windows is described by its probes — what
`GlobalMemoryStatusEx` fills in, what `nvidia-smi` prints — and the import check runs the agent in
a fresh interpreter with the names Windows lacks taken away.
"""

from __future__ import annotations

import collections
import ctypes
import ntpath
import subprocess
import sys
import textwrap

import pytest
from gpm_agent import facts as agent_facts
from gpm_agent import settings as agent_settings
from gpm_agent.settings import Settings, SettingsError, load, save

Usage = collections.namedtuple("Usage", "total used free")
GIB = 1024**3


def windows(outputs=None, memory=(64 * GIB, 40 * GIB), free_gb=300):
    """A Windows machine, by what its commands would print and what its memory call fills in."""
    outputs = outputs or {}
    asked = []

    def run(command):
        asked.append(list(command))
        return outputs.get(" ".join(command), outputs.get(command[0]))

    probes = agent_facts.Probes(
        system=lambda: "Windows",
        machine=lambda: "AMD64",
        run=run,
        read=lambda path: None,
        disk_usage=lambda path: Usage(1000 * 10**9, (1000 - free_gb) * 10**9, free_gb * 10**9),
        windows_memory=lambda: memory,
    )
    return probes, asked


def test_a_windows_machine_with_nvidia_cards_is_described_as_linux_would_describe_them():
    probes, asked = windows({
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits": "NVIDIA GeForce RTX 4090, 24564\r\n",
        "nvidia-smi --query-gpu=compute_cap,driver_version --format=csv,noheader": "8.9, 581.57\r\n",
    })
    found = agent_facts.gather(probes, "C:\\Users\\owner\\.ollama\\models")
    assert found["os"] == "Windows" and found["arch"] == "AMD64"
    assert found["memory"] == {"total_bytes": 64 * GIB, "available_bytes": 40 * GIB}
    assert found["accelerators"] == [{
        "kind": "cuda", "name": "NVIDIA GeForce RTX 4090", "memory_bytes": 24564 * 1024 * 1024, "unified": False,
        "compute_capability": "8.9", "driver": "581.57",
    }]
    assert found["capabilities"] == ["cuda"]
    assert found["disk"]["free_bytes"] == 300 * 10**9
    # The same queries Linux asks, and nothing else: no rocm-smi, no sysctl.
    assert asked == [
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader"],
    ]


def test_without_nvidias_answer_windows_says_it_could_not_tell_rather_than_none():
    """Only NVIDIA's cards are asked about on Windows; an AMD card may still be there."""
    probes, _ = windows()
    assert agent_facts.gather(probes, "C:\\models")["capabilities"] is None


def test_memory_that_cannot_be_read_is_reported_as_unknown():
    probes, _ = windows(memory=None)
    assert agent_facts.memory(probes) == {"total_bytes": None, "available_bytes": None}


def test_the_memory_call_is_given_the_structure_windows_expects_and_read_back():
    assert ctypes.sizeof(agent_facts.MemoryStatus) == 64  # MEMORYSTATUSEX

    class Kernel32:
        def __init__(self, answer):
            self.answer, self.length = answer, None

        def GlobalMemoryStatusEx(self, pointer):  # noqa: N802 — Windows' own name
            status = pointer.contents
            self.length = status.dwLength
            status.ullTotalPhys, status.ullAvailPhys = 32 * GIB, 12 * GIB
            return self.answer

    kernel32 = Kernel32(answer=1)
    assert agent_facts._windows_memory(kernel32) == (32 * GIB, 12 * GIB)
    assert kernel32.length == 64  # Windows refuses the call unless the length is filled in
    assert agent_facts._windows_memory(Kernel32(answer=0)) is None


@pytest.mark.skipif(sys.platform == "win32", reason="checks the answer away from Windows")
def test_away_from_windows_there_is_no_memory_call_to_make():
    assert agent_facts._windows_memory() is None


def test_linux_and_macos_are_asked_what_they_were_asked_before():
    asked = []

    def run(command):
        asked.append(command[0])
        return None

    for system in ("Linux", "Darwin"):
        agent_facts.accelerators(agent_facts.Probes(system=lambda s=system: s, machine=lambda: "x86_64", run=run))
    assert asked == ["nvidia-smi", "rocm-smi", "nvidia-smi", "rocm-smi"]


def test_facts_run_their_commands_without_a_console_window_on_windows():
    """The agent started at logon runs under `pythonw.exe`: each console program it starts would
    otherwise open a window on the owner's screen, `nvidia-smi` on every facts call."""
    calls = []

    def run(argv, **options):
        calls.append(options)
        return subprocess.CompletedProcess(argv, 0, stdout="NVIDIA GeForce RTX 4090, 24564\n")

    nvidia = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    found = agent_facts._run(nvidia, system=lambda: "Windows", which=lambda name: "C:\\nvidia-smi.exe", run=run)
    assert found == "NVIDIA GeForce RTX 4090, 24564\n"
    assert calls[-1]["creationflags"] == 0x08000000  # CREATE_NO_WINDOW
    agent_facts._run(nvidia, system=lambda: "Linux", which=lambda name: "/usr/bin/nvidia-smi", run=run)
    assert "creationflags" not in calls[-1]  # refused off Windows unless 0


def test_settings_load_on_windows_where_file_modes_mean_nothing_but_the_place_does(tmp_path, monkeypatch):
    profile = tmp_path / "Users" / "owner"
    path = save(Settings(key_hash="x"), profile / ".config" / "gpm-agent" / "agent.json")
    path.chmod(0o644)  # as Python reports every writable file on Windows
    with pytest.raises(SettingsError, match="readable"):
        load(path)
    monkeypatch.setattr(agent_settings, "WINDOWS", True)
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert load(path).key_hash == "x"


def test_on_windows_settings_outside_the_profile_are_refused(tmp_path, monkeypatch):
    """A folder made at a drive's root inherits "Authenticated Users: Modify": any user of the
    machine could then name the program this agent runs as its owner. The standard library
    reads no access control list, so on Windows the place decides."""
    profile, shared = tmp_path / "Users" / "owner", tmp_path / "Shared"
    elsewhere = save(Settings(key_hash="x"), shared / "agent.json")
    monkeypatch.setattr(agent_settings, "WINDOWS", True)
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    with pytest.raises(SettingsError, match="outside your profile"):
        load(elsewhere)
    with pytest.raises(SettingsError, match="outside your profile"):
        save(Settings(key_hash="x"), shared / "other.json")  # `init` writes nothing there
    assert not (shared / "other.json").exists()
    # A link out of the profile is where it leads.
    (profile / "link").parent.mkdir(parents=True)
    (profile / "link").symlink_to(shared, target_is_directory=True)
    with pytest.raises(SettingsError, match="outside your profile"):
        load(profile / "link" / "agent.json")
    # Under the profile, or the local application data moved out of it: loaded.
    assert load(save(Settings(key_hash="x"), profile / "agent.json")).key_hash == "x"
    assert load(save(Settings(key_hash="x"), tmp_path / "Local" / "gpm" / "agent.json")).key_hash == "x"
    # What the settings name for the agent to run, likewise.
    with pytest.raises(SettingsError, match="engine_binary .* outside your profile"):
        Settings(key_hash="x", engine="whisper-cpp", engine_url="http://127.0.0.1:8178",
                 engine_binary=str(shared / "whisper-server.exe")).check()
    with pytest.raises(SettingsError, match="engines_path .* outside your profile"):
        Settings(key_hash="x", engines_path=str(shared / "engines")).check()
    Settings(key_hash="x", engines_path=str(profile / "engines"),
             engine_binary=str(profile / "bin" / "whisper-server.exe")).check()
    monkeypatch.delenv("USERPROFILE")
    monkeypatch.delenv("LOCALAPPDATA")
    with pytest.raises(SettingsError, match="USERPROFILE does not say"):
        load(profile / "agent.json")


def test_the_profile_is_read_with_windows_path_rules():
    environ = {"USERPROFILE": "C:\\Users\\Owner", "LOCALAPPDATA": "D:\\Local\\owner"}
    assert agent_settings.outside_profile("c:\\users\\owner\\.config\\gpm-agent\\agent.json", environ, ntpath) is None
    assert agent_settings.outside_profile("D:\\Local\\owner\\gpm\\engines", environ, ntpath) is None
    assert "outside your profile (C:\\Users\\Owner)" in agent_settings.outside_profile("C:\\GPM\\agent.json", environ, ntpath)
    assert agent_settings.outside_profile("C:\\Users\\Owner2\\agent.json", environ, ntpath)  # not a prefix of a name
    assert agent_settings.outside_profile("E:\\agent.json", environ, ntpath)  # another drive
    assert agent_settings.outside_profile("C:\\Users\\Owner\\x.json", {"USERPROFILE": "Users\\Owner"}, ntpath)


def test_the_agent_imports_and_builds_its_app_without_the_names_only_posix_has(tmp_path):
    """Windows has no `O_NOFOLLOW`, `O_DIRECTORY`, `getuid`, `SIGKILL`, `pwd` or `fcntl`. The
    agent is imported in a fresh interpreter with all of them taken away — after the standard
    library and the agent's dependencies, which import as they do on Windows — and must still
    load, build its app and parse its command line."""
    script = textwrap.dedent("""
        import asyncio, ctypes, importlib.metadata, os, platform, plistlib, shlex, signal
        import subprocess, sys, xml.parsers.expat, xml.sax.saxutils
        import httpx, starlette.applications, uvicorn
        for name in ("O_NOFOLLOW", "O_DIRECTORY", "getuid", "geteuid", "setsid", "killpg",
                     "getpgid", "fork", "uname", "getpgrp", "setpgrp"):
            if hasattr(os, name):
                delattr(os, name)
        for name in ("SIGKILL", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGCHLD"):
            if hasattr(signal, name):
                delattr(signal, name)
        for module in ("pwd", "grp", "fcntl", "termios", "resource"):
            sys.modules[module] = None
        import gpm_agent.app, gpm_agent.cli, gpm_agent.startup, gpm_agent.prerequisites
        from gpm_agent.settings import Settings
        gpm_agent.app.create_app(Settings(key_hash="x"))
        print("loaded")
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "loaded"
