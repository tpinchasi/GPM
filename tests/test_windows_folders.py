"""Folders the agent and the pool make for themselves, and runs with administrator rights, on
Windows (D163). Written to pass on every system; CI runs this file on Windows too.

Since Python 3.12.4, `mkdir(mode=0o700)` on Windows gives a folder an owner-only access list. An
agent run elevated has the Administrators group for its owner, so an engine version it staged was
unreadable to the same user afterwards, and the next ordinary start could not run it. On Windows
these folders now inherit the user's profile's own permissions instead.
"""

import logging
import os
import re
from pathlib import Path

import pytest
from gpm_agent import platform as host_platform
from gpm_server import processes

ROOT = Path(__file__).resolve().parent.parent


def test_a_private_folder_is_owner_only_by_its_mode_on_macos_and_linux(tmp_path):
    if os.name == "nt":
        pytest.skip("Windows has no mode bits; its case is below")
    host_platform.private_dir(tmp_path / "a" / "b", parents=True, os_name=host_platform.LINUX)
    assert (tmp_path / "a" / "b").stat().st_mode & 0o777 == 0o700


def test_on_windows_a_private_folder_is_made_without_an_owner_only_mode(tmp_path, monkeypatch):
    asked = []
    real = Path.mkdir

    def mkdir(self, *args, **kwargs):
        asked.append((args, kwargs))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    host_platform.private_dir(tmp_path / "engines" / "ollama", parents=True, exist_ok=True, os_name=host_platform.WINDOWS)
    modes = [kwargs.get("mode", args[0] if args else 0o777) for args, kwargs in asked]
    assert modes and 0o700 not in modes, "never owner-only: the profile's permissions are inherited"
    assert (tmp_path / "engines" / "ollama").is_dir()


def test_a_private_folder_made_here_is_readable_and_writable_by_this_user(tmp_path):
    folder = tmp_path / "engines" / "whisper-cpp" / "8178"
    host_platform.private_dir(folder, parents=True)
    (folder / "server.json").write_text("{}")
    assert os.access(folder, os.R_OK | os.W_OK | os.X_OK) and (folder / "server.json").read_text() == "{}"


def test_nothing_in_the_agent_or_the_pool_makes_a_folder_owner_only_by_its_mode_alone():
    """`mode=0o700` is said only beside its Windows case: `platform.private_dir`, and the pool's
    credential directory."""
    found = []
    for path in [*(ROOT / "agent" / "src").rglob("*.py"), *(ROOT / "server" / "src").rglob("*.py")]:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"mkdir\([^)]*0o700", line) and "private_dir" not in path.name:
                found.append(f"{path.relative_to(ROOT)}:{number}")
    # `private_dir` itself, and the pool's credential directory, which says its Windows case beside it.
    allowed = {"agent/src/gpm_agent/platform.py", "server/src/gpm_server/credentials.py"}
    assert {f.rsplit(":", 1)[0].replace("\\", "/") for f in found} <= allowed, found


def test_an_elevated_run_is_known_on_windows_and_never_assumed_elsewhere():
    assert host_platform.elevated(os_name=host_platform.LINUX) is False
    assert isinstance(host_platform.elevated(), bool)
    assert isinstance(processes.elevated(), bool)


def test_the_pool_says_when_it_runs_elevated(monkeypatch, caplog):
    monkeypatch.setattr(processes, "elevated", lambda: True)
    with caplog.at_level(logging.WARNING, logger="gpm"):
        processes.warn_if_elevated("pool's supervisor")
    assert "pool's supervisor runs with administrator rights" in caplog.text
    caplog.clear()
    monkeypatch.setattr(processes, "elevated", lambda: False)
    processes.warn_if_elevated("pool's supervisor")
    assert caplog.text == ""
