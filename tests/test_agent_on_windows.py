"""The agent on Windows: what it reports about the machine, and that it loads and serves there
(D154).

docs/spec/host-agent.md §3. Run on any system: Windows is described by its probes — what
`GlobalMemoryStatusEx` fills in, what `nvidia-smi` prints — and the import check runs the agent in
a fresh interpreter with the names Windows lacks taken away.
"""

from __future__ import annotations

import collections
import ctypes
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
        return outputs.get(command[0])

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
    probes, asked = windows({"nvidia-smi": "NVIDIA GeForce RTX 4090, 24564\r\n"})
    found = agent_facts.gather(probes, "C:\\Users\\owner\\.ollama\\models")
    assert found["os"] == "Windows" and found["arch"] == "AMD64"
    assert found["memory"] == {"total_bytes": 64 * GIB, "available_bytes": 40 * GIB}
    assert found["accelerators"] == [
        {"kind": "cuda", "name": "NVIDIA GeForce RTX 4090", "memory_bytes": 24564 * 1024 * 1024, "unified": False},
    ]
    assert found["capabilities"] == ["cuda"]
    assert found["disk"]["free_bytes"] == 300 * 10**9
    # The same query Linux asks, and nothing else: no rocm-smi, no sysctl.
    assert asked == [["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]]


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


def test_settings_load_on_windows_where_file_modes_mean_nothing(tmp_path, monkeypatch):
    path = save(Settings(key_hash="x"), tmp_path / "agent.json")
    path.chmod(0o644)  # as Python reports every writable file on Windows
    with pytest.raises(SettingsError, match="readable"):
        load(path)
    monkeypatch.setattr(agent_settings, "CHECKS_FILE_MODE", False)
    assert load(path).key_hash == "x"


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
