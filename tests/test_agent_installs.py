"""The agent installs, runs and upgrades Ollama on a machine whose owner allows it (D150–D153).

docs/spec/host-agent.md §2.3 and §7. Against a fake GitHub — releases with per-file digests, and
small archives built here — and a fake `ollama` program: a small Python program that answers
`GET /api/version` as the version it was packed as. Nothing here reaches the network, downloads
a real engine, or needs a GPU; the platform's commands (`codesign`, PowerShell) are stood in for —
but for one check that runs this Mac's own `codesign`, where there is one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import httpx
import pytest
from gpm_agent import cli, installs
from gpm_agent import platform as host_platform
from gpm_agent.app import create_app
from gpm_agent.engines import OllamaFacts
from gpm_agent.installs import OLLAMA, ArchiveSource, Installer, InstallRefused, Target, unpack
from gpm_agent.models import ModelWork, Refused
from gpm_agent.ollama_server import POOL_NAMES, OllamaServer
from gpm_agent.settings import Settings, SettingsError, fingerprint, load

AGENT_KEY = "gpmg_" + "a" * 64
V1, V2, V3 = "0.40.10", "0.40.11", "0.41.0"

#: The fake program. Packed into each release's archive behind a small shell wrapper, which names
#: the version it was packed as in its environment and runs the fake as `python fake.py <program>
#: <arguments>`, the way the system runs a script — so the process's command line ends in exactly
#: the argument list the agent started it with, as the real one's is. Told how to behave by a file
#: beside the test, per version.
FAKE_OLLAMA = '''
import http.server, json, os, sys, time

version, control_path, starts_path = (os.environ[name] for name in ("FAKE_VERSION", "FAKE_CONTROL", "FAKE_STARTS"))
program, args = sys.argv[1], sys.argv[2:]
try:
    control = json.load(open(control_path)).get(version, {})
except OSError:
    control = {}
watched = ["OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS",
           "OLLAMA_CONTEXT_LENGTH", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN",
           "CONTAINER_API_KEY", "SOME_SERVICE_TOKEN", "OLLAMA_ORIGINS", "CUDA_VISIBLE_DEVICES", "PATH", "HOME"]
with open(starts_path, "a") as starts:
    starts.write(json.dumps({"version": version, "program": program, "args": args, "pid": os.getpid(),
                             "cwd": os.getcwd(), "env": {k: os.environ.get(k) for k in watched}}) + "\\n")
print("ollama", version, "starting", flush=True)
if control.get("exit"):
    print(control["exit"], flush=True)
    sys.exit(3)
if control.get("silent"):
    time.sleep(3600)
answer_as = control.get("answer_as", version)
host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        bodies = {"/api/version": {"version": answer_as}, "/api/tags": {"models": []}, "/api/ps": {"models": []}}
        body = bodies.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(json.dumps(body or {}).encode())

    def log_message(self, *_):
        pass


http.server.ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
'''

#: The table the running tests install from: Ollama's own macOS row, with the signature left to
#: its own tests — the fake program is a script, which nothing signs.
UNSIGNED = dataclasses.replace(OLLAMA, signers={})
MAC = Target(host_platform.MACOS, "arm64")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- archives, built here ---


def tgz(entries: list[tarfile.TarInfo | tuple[str, bytes, int]]) -> bytes:
    """A gzipped tar of (name, content, mode) files, or of TarInfo entries as given."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for entry in entries:
            if isinstance(entry, tarfile.TarInfo):
                archive.addfile(entry)
                continue
            name, content, mode = entry
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), mode
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def link(name: str, to: str, kind: bytes = tarfile.SYMTYPE) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, to
    return info


def zipped(entries: list[tuple[str, bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content, mode in entries:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            archive.writestr(info, content)
    return buffer.getvalue()


# --- a fake GitHub ---


class FakeGitHub:
    """Releases of `ollama/ollama` as GitHub's API describes them, and their files."""

    def __init__(self) -> None:
        self.releases: dict[str, dict[str, bytes]] = {}
        self.latest: Optional[str] = None
        self.requests: list[httpx.Request] = []
        #: (version, file) → the digest to publish instead of the true one; None publishes none.
        self.digests: dict[tuple[str, str], Optional[str]] = {}
        #: file → how many bytes to send before the connection "closes", once.
        self.cut: dict[str, int] = {}
        self.with_checksums = True
        self.checksum_text: dict[str, str] = {}
        self._ids: dict[tuple[str, str], int] = {}

    def publish(self, version: str, files: dict[str, bytes], *, latest: bool = True) -> None:
        self.releases[version] = dict(files)
        if latest:
            self.latest = version

    def _files(self, version: str) -> dict[str, bytes]:
        files = dict(self.releases[version])
        if self.with_checksums:
            files["sha256sum.txt"] = self.checksum_text.get(version, "".join(
                f"{sha(content)}  ./{name}\n" for name, content in sorted(files.items())
            )).encode()
        files["install.sh"] = b"#!/bin/sh\necho never run\n"
        return files

    def _id(self, version: str, name: str) -> int:
        return self._ids.setdefault((version, name), 1000 + len(self._ids))

    def _release(self, version: str) -> dict:
        assets = []
        for name, content in self._files(version).items():
            digest = self.digests.get((version, name), f"sha256:{sha(content)}")
            entry = {"name": name, "id": self._id(version, name), "size": len(content),
                     "browser_download_url": f"https://github.com/ollama/ollama/releases/download/v{version}/{name}"}
            if digest is not None:
                entry["digest"] = digest
            assets.append(entry)
        return {"tag_name": f"v{version}", "draft": False, "prerelease": False, "assets": assets}

    def asked(self, path_part: str) -> int:
        return sum(1 for request in self.requests if path_part in request.url.path)

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            path = request.url.path
            if path == "/repos/ollama/ollama/releases/latest":
                return httpx.Response(200, json=self._release(self.latest)) if self.latest else httpx.Response(404)
            if path.startswith("/repos/ollama/ollama/releases/tags/v"):
                version = path.rsplit("/v", 1)[1]
                return httpx.Response(200, json=self._release(version)) if version in self.releases else httpx.Response(404)
            if path.startswith("/repos/ollama/ollama/releases/assets/"):
                number = int(path.rsplit("/", 1)[1])
                (version, name), = [key for key, value in self._ids.items() if value == number]
                content = self._files(version)[name]
                assert request.headers["accept"] == "application/octet-stream"
                ranged = request.headers.get("range")
                if ranged:
                    start = int(ranged.removeprefix("bytes=").rstrip("-"))
                    return httpx.Response(206, content=content[start:],
                                          headers={"content-range": f"bytes {start}-{len(content) - 1}/{len(content)}"})
                if name in self.cut:
                    return httpx.Response(200, content=content[: self.cut.pop(name)])
                return httpx.Response(200, content=content)
            return httpx.Response(404)

        return httpx.MockTransport(handler)


# --- a machine ---


class Rig:
    """A machine whose agent installs Ollama: an engines directory, a port, a fake GitHub, and
    the installer and server as the agent builds them."""

    def __init__(self, root: Path, *, table=UNSIGNED, target=MAC, run=None, switch_timeout_s=20.0,
                 min_free_disk_gb=0.0, free_bytes=None):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.engines = root / "engines"
        self.control = root / "fake-control.json"
        self.starts_file = root / "fake-starts.jsonl"
        (root / "fake_ollama.py").write_text(FAKE_OLLAMA)
        self.github = FakeGitHub()
        self.table, self.target, self.run = table, target, run
        self.switch_timeout_s = switch_timeout_s
        self.min_free_disk_gb, self.free_bytes = min_free_disk_gb, free_bytes
        self.now = 1_800_000_000.0
        self.clients: list[httpx.AsyncClient] = []
        self.servers: list[OllamaServer] = []
        self.installer = self.new_installer()

    def settings(self, **more) -> Settings:
        return Settings(
            key_hash=fingerprint(AGENT_KEY), engine="ollama", engine_url=self.url,
            models_path=str(self.root / "models"), install_engines=True, engines_path=str(self.engines), **more,
        )

    def new_installer(self) -> Installer:
        """The installer as a freshly started agent builds it."""
        client = httpx.AsyncClient(base_url=self.url, timeout=10)
        self.clients.append(client)
        server = OllamaServer(
            self.engines / "ollama", engine_url=self.url, models_path=str(self.root / "models"), client=client,
        )
        self.servers.append(server)
        source = ArchiveSource(self.table, self.target, run=self.run)
        return Installer(
            source, self.engines / "ollama", server, http_transport=self.github.transport(),
            clock=lambda: self.now, switch_timeout_s=self.switch_timeout_s,
            min_free_disk_gb=self.min_free_disk_gb,
            free_disk_bytes=(lambda: self.free_bytes) if self.free_bytes is not None else None,
        )

    def program(self, version: str) -> bytes:
        return (
            f'#!/bin/sh\nFAKE_VERSION="{version}" FAKE_CONTROL="{self.control}" FAKE_STARTS="{self.starts_file}"\n'
            "export FAKE_VERSION FAKE_CONTROL FAKE_STARTS\n"
            f'exec "{sys.executable}" "{self.root / "fake_ollama.py"}" "$0" "$@"\n'
        ).encode()

    def release(self, version: str, *, latest: bool = True, extra: Optional[list] = None) -> None:
        """Publish `version` with a macOS archive holding the fake program and a library."""
        archive = tgz([
            ("ollama", self.program(version), 0o755),
            ("lib/libggml-base.dylib", b"not really a library " + version.encode(), 0o644),
            *(extra or []),
        ])
        self.github.publish(version, {"ollama-darwin.tgz": archive, "ollama-linux-amd64.tar.zst": b"other"}, latest=latest)

    def behave(self, version: str, **control) -> None:
        current = json.loads(self.control.read_text()) if self.control.exists() else {}
        current[version] = control
        self.control.write_text(json.dumps(current))

    def starts(self) -> list[dict]:
        if not self.starts_file.exists():
            return []
        return [json.loads(line) for line in self.starts_file.read_text().splitlines()]

    def versions_on_disk(self) -> set[str]:
        root = self.engines / "ollama"
        return {p.name for p in root.iterdir() if p.is_dir() and installs.VERSION.match(p.name)} if root.is_dir() else set()

    async def want(self, version: str, switch: bool = False, installer: Optional[Installer] = None) -> dict:
        installer = installer or self.installer
        answer = await installer.want(version, switch)
        await installer.finished()
        return answer

    async def answering(self) -> Optional[str]:
        async with httpx.AsyncClient(base_url=self.url, timeout=3) as client:
            try:
                return (await client.get("/api/version")).json()["version"]
            except httpx.HTTPError:
                return None

    async def close(self) -> None:
        for record in self.starts():
            # Only a process still the fake: a number from the log may have gone to another.
            said = host_platform.command_line(record["pid"]) or b""
            if host_platform.alive(record["pid"]) and str(self.root / "fake_ollama.py").encode() in said:
                try:
                    os.kill(record["pid"], signal.SIGKILL)
                except OSError:
                    pass
        for server in self.servers:
            if server._process is not None:
                server._process.wait(timeout=10)
        for client in self.clients:
            await client.aclose()


@pytest.fixture
async def rig(tmp_path):
    made = Rig(tmp_path)
    yield made
    await made.close()


async def wait_until(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.05)


# --- the platform layer (D154) ---


@pytest.mark.parametrize("os_name, environ, home, expected", [
    ("macos", {}, "/Users/me", PurePosixPath("/Users/me/Library/Application Support/gpm/engines")),
    ("linux", {}, "/home/me", PurePosixPath("/home/me/.local/share/gpm/engines")),
    ("linux", {"XDG_DATA_HOME": "/data/me"}, "/home/me", PurePosixPath("/data/me/gpm/engines")),
    ("linux", {"XDG_DATA_HOME": "relative/data"}, "/home/me", PurePosixPath("/home/me/.local/share/gpm/engines")),
    ("windows", {"LOCALAPPDATA": r"C:\Users\me\AppData\Local"}, r"C:\Users\me",
     PureWindowsPath(r"C:\Users\me\AppData\Local\gpm\engines")),
    ("windows", {}, r"C:\Users\me", PureWindowsPath(r"C:\Users\me\AppData\Local\gpm\engines")),
])
def test_each_system_keeps_engines_in_its_own_per_user_place(os_name, environ, home, expected):
    assert host_platform.engines_dir_for(os_name, environ, PureWindowsPath(home) if os_name == "windows" else PurePosixPath(home)) == expected


def test_systems_and_processors_are_named_as_engines_publish_them():
    assert [host_platform.system(name) for name in ("darwin", "linux", "win32", "freebsd14")] == [
        "macos", "linux", "windows", "freebsd14",
    ]
    assert [host_platform.architecture(name) for name in ("x86_64", "AMD64", "arm64", "aarch64", "riscv64")] == [
        "amd64", "amd64", "arm64", "arm64", None,
    ]


def test_a_program_is_started_in_its_own_session_on_posix_and_its_own_group_without_a_window_on_windows(tmp_path):
    seen = []

    def popen(argv, **options):
        seen.append((argv, options))
        return type("Started", (), {"pid": 4242})()

    for os_name in ("windows", "linux", "macos"):
        process, offset = host_platform.start(
            ["/x/ollama", "serve"], env={"A": "1"}, log=tmp_path / f"{os_name}.log", cwd=tmp_path,
            note="starting ollama", os_name=os_name, popen=popen,
        )
        assert process.pid == 4242 and offset == (tmp_path / f"{os_name}.log").stat().st_size
    (windows_argv, windows), (_, linux), (_, mac) = seen
    assert windows_argv == ["/x/ollama", "serve"]
    assert windows["creationflags"] == host_platform.CREATE_NEW_PROCESS_GROUP | host_platform.CREATE_NO_WINDOW
    assert "start_new_session" not in windows
    for posix in (linux, mac):
        assert posix["start_new_session"] is True and "creationflags" not in posix
    assert all(options["env"] == {"A": "1"} and options["cwd"] == tmp_path for _, options in seen)


async def test_on_windows_a_process_is_read_and_stopped_by_fixed_commands():
    ran = []

    def run(argv, env=None):
        ran.append(argv)
        return host_platform.Ran(0, "C:\\gpm\\engines\\ollama\\0.40.11\\ollama.exe serve\r\n")

    said = host_platform.command_line(1234, os_name="windows", run=run)
    assert said == b"C:\\gpm\\engines\\ollama\\0.40.11\\ollama.exe serve"

    class Stopped:
        pid = 1234
        calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 0

    await host_platform.stop(1234, process=Stopped(), os_name="windows", run=run, poll_s=0.01)
    assert ran == [
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         '(Get-CimInstance Win32_Process -Filter "ProcessId=1234").CommandLine'],
        ["taskkill", "/PID", "1234", "/T", "/F"],
    ]


SERVER_ARGV = ["/e/whisper-server", "-m", "/m/o__r__f.bin", "--port", "8178", "-l", "auto"]


@pytest.mark.parametrize("listed, ours", [
    (SERVER_ARGV, True),
    (["/bin/sh", *SERVER_ARGV], True),                                  # a script, run by its `#!`
    (["/usr/bin/python3", "-I", *SERVER_ARGV], True),                   # … with its one argument
    (["/usr/bin/env", "python3", "-I", *SERVER_ARGV], False),           # more than a `#!` adds
    ([*SERVER_ARGV, "--more"], False),
    (["/usr/bin/python3", "-c", "pass", *SERVER_ARGV[1:]], False),      # its words, not its program
])
def test_a_process_is_its_own_by_exactly_its_argument_list_where_they_are_listed(monkeypatch, listed, ours):
    monkeypatch.setattr(host_platform, "arguments", lambda pid, os_name=None: listed)
    assert host_platform.started_with(4242, SERVER_ARGV, os_name="linux") is ours


@pytest.mark.parametrize("line, ours", [
    (" ".join(SERVER_ARGV), True),
    ("/usr/bin/python3 /t/fake.py " + " ".join(SERVER_ARGV), True),
    ("/usr/bin/env python3 -I " + " ".join(SERVER_ARGV), False),
    ("sleep 60 " + " ".join(SERVER_ARGV[1:]), False),
    (" ".join(SERVER_ARGV) + " --more", False),
])
def test_where_only_a_joined_command_line_is_read_it_is_compared_whole(monkeypatch, line, ours):
    monkeypatch.setattr(host_platform, "arguments", lambda pid, os_name=None: None)
    said = host_platform.started_with(
        4242, SERVER_ARGV, os_name="macos", run=lambda argv, env=None: host_platform.Ran(0, line + "\n"),
    )
    assert said is ours


def test_on_windows_a_command_line_is_compared_as_windows_quotes_it():
    argv = [r"C:\Users\me\AppData\Local\gpm\engines\whisper-cpp\1.9.5\Release\whisper-server.exe",
            "-m", r"C:\Users\me\gpm whisper models\o__r__f.bin", "--port", "8178"]
    quoted = (r'C:\Users\me\AppData\Local\gpm\engines\whisper-cpp\1.9.5\Release\whisper-server.exe '
              r'-m "C:\Users\me\gpm whisper models\o__r__f.bin" --port 8178')

    def says(line):
        return lambda argv, env=None: host_platform.Ran(0, line + "\r\n")

    assert host_platform.started_with(7, argv, os_name="windows", run=says(quoted))
    assert not host_platform.started_with(7, argv, os_name="windows", run=says(quoted + " --more"))
    assert not host_platform.started_with(7, argv, os_name="windows", run=says(quoted.replace('"', "")))
    assert not host_platform.started_with(7, argv, os_name="windows", run=lambda argv, env=None: host_platform.Ran(1, ""))


def test_why_the_server_stopped_is_an_error_line_never_a_request(tmp_path):
    """The reason travels to the pool: Ollama's lines about requests, and what it prints when
    asked to debug — which can carry a client's prompt — are never it."""
    from gpm_agent.ollama_server import _last_error_line

    log = tmp_path / "ollama.log"
    log.write_text(
        "--- gpm-agent: starting ollama 0.40.11\n"
        'time=2026-10-29T12:00:00 level=INFO source=routes.go:1 msg="Listening on 127.0.0.1:11434"\n'
        "Error: could not load the runner\n"
        '[GIN] 2026/10/29 - 12:00:01 | 500 | 1.2s | 127.0.0.1 | POST "/api/chat"\n'
        'time=2026-10-29T12:00:01 level=DEBUG msg="chat request" prompt="the error in my report"\n'
        "    error: an indented line, describing something else\n"
    )
    assert _last_error_line(str(log), 0) == "Error: could not load the runner"
    log.write_text('[GIN] 2026/10/29 - 12:00:01 | 200 | 1.2s | 127.0.0.1 | POST "/api/chat"\n')
    assert _last_error_line(str(log), 0) is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes where the test forbids")
async def test_a_state_directory_it_may_not_write_is_a_reason_not_an_exception(tmp_path):
    """An error from the system is said as the start's reason, for the installer to record —
    not an exception that ends its work unsaid."""
    locked = tmp_path / "locked"
    locked.mkdir()
    program = tmp_path / "ollama"
    program.write_text("#!/bin/sh\nexit 1\n")
    program.chmod(0o755)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{free_port()}", timeout=3) as client:
        server = OllamaServer(locked / "ollama", engine_url=str(client.base_url), models_path="~/.ollama/models",
                              client=client)
        locked.chmod(0o500)
        try:
            said = await server.start(program, V1, 5)
        finally:
            locked.chmod(0o755)
    assert said is not None and "ermission denied" in said and server._process is None


def test_on_windows_liveness_is_asked_of_windows_and_never_with_a_signal(tmp_path, monkeypatch):
    """On Windows `os.kill(pid, 0)` does not look at a process: it ends it. A recorded number
    that has gone to another program must never be ended by looking — not by this agent's own
    servers' records, not by whisper.cpp's, not by the vLLM launcher's."""
    import ctypes

    from gpm_agent import vllm_state
    from gpm_agent.whisper_cpp import WhisperCppFacts

    signals = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(host_platform, "system", lambda sys_platform=None: "windows")
    opened = []

    class Kernel32:
        def OpenProcess(self, access, inherit, pid):
            opened.append((access, pid))
            return 0 if pid == 999 else 77

        def GetExitCodeProcess(self, handle, code):
            code._obj.value = 259  # STILL_ACTIVE
            return 1

        def CloseHandle(self, handle):
            return 1

    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: Kernel32(), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 87, raising=False)  # no such process
    monkeypatch.setattr(host_platform, "_run_briefly", lambda argv, env=None: host_platform.Ran(0, "someone else's program\n"))

    assert host_platform.alive(1234) is True and host_platform.alive(999) is False
    assert (0x1000, 1234) in opened

    vllm_models = tmp_path / "vllm-models"
    vllm_models.mkdir()
    launched = vllm_state.state_dir_for(vllm_models)
    launched.mkdir()
    (launched / vllm_state.STARTED_FILE).write_text(json.dumps({"engines": [{"model": "m", "pid": 999, "log": ""}]}))
    assert "m" in vllm_state.failed_engines(vllm_models, [])

    whisper = WhisperCppFacts(Settings(
        key_hash="x", engine="whisper-cpp", engine_url="http://127.0.0.1:8178", models_path=str(tmp_path / "w"),
    ))
    whisper._state_dir.mkdir(parents=True)
    (whisper._state_dir / "whisper-server.json").write_text(json.dumps({
        "pid": 1234, "tag": "o/r/f.bin", "file": "/w/f.bin", "log": "", "log_offset": 0, "started_at": 0,
    }))
    assert whisper._alive(whisper._current()) is False, "running, but not the server, by its command line"

    engine_dir = tmp_path / "engines" / "ollama"
    engine_dir.mkdir(parents=True)
    (engine_dir / "server.json").write_text(json.dumps({"pid": 1234, "program": "/x/ollama", "version": V1}))
    server = OllamaServer(engine_dir, engine_url="http://127.0.0.1:11434", models_path="~/.ollama/models",
                          client=httpx.AsyncClient())
    assert server.ours() is False
    assert signals == [], "a process was signalled on Windows"


#: Ollama's team as the requirement names it: a Developer ID chain anchored at Apple's root.
OLLAMA_REQUIREMENT = (
    "anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists"
    " and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
    ' and certificate leaf[subject.OU] = "3MU9H2V9Y9"'
)


def test_on_macos_a_signature_is_one_requirement_anchored_at_apple_with_the_team_pinned(tmp_path):
    ran = []

    def run(argv, env=None):
        ran.append(argv)
        return host_platform.Ran(0, "")

    program = tmp_path / "ollama"
    assert host_platform.signature_problem(program, "3MU9H2V9Y9", os_name="macos", run=run) is None
    assert ran == [["codesign", "--verify", "--strict", f"-R={OLLAMA_REQUIREMENT}", str(program)]]
    failed = host_platform.signature_problem(
        program, "3MU9H2V9Y9", os_name="macos",
        run=lambda argv, env=None: host_platform.Ran(3, "test-requirement: code failed to satisfy specified code requirement(s)"),
    )
    assert "not signed for distribution by team 3MU9H2V9Y9" in failed and "failed to satisfy" in failed
    # The requirement is built from a team's own form and nothing else; the table's is one.
    ran.clear()
    for odd in ('3MU9H2V9Y9" or anchor apple', "3mu9h2v9y9", "3MU9H2V9Y", "", "3MU9H2V9Y9\n"):
        assert "is not an Apple Developer ID team" in host_platform.signature_problem(program, odd, os_name="macos", run=run)
    assert ran == []
    assert host_platform.TEAM_ID.fullmatch(OLLAMA.signers["macos"])


def test_a_file_name_cannot_write_the_team_into_what_is_checked(tmp_path):
    """`codesign` as it answers for a program signed ad hoc: valid on its own, failing any
    Developer ID requirement — and describing itself with its file's name first, so a name can
    print a line of its own. The answer is the requirement's exit code alone."""
    named = tmp_path / "ollama\nTeamIdentifier=3MU9H2V9Y9"

    def codesign(argv, env=None):
        if any(part.startswith("-R=") for part in argv):
            return host_platform.Ran(3, f"{argv[-1]}: test-requirement: code failed to satisfy specified code requirement(s)")
        if "-dv" in argv:
            return host_platform.Ran(0, f"Executable={argv[-1]}\nIdentifier=adhoc\nSignature=adhoc\nTeamIdentifier=not set\n")
        return host_platform.Ran(0, "")

    assert host_platform.signature_problem(named, "3MU9H2V9Y9", os_name="macos", run=codesign) is not None


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("codesign") is None, reason="codesign is macOS's")
def test_on_this_mac_codesign_refuses_an_ad_hoc_signature_and_apples_own(tmp_path):
    """The real `codesign`: a program signed ad hoc — named to print the team, as an archive could
    name it — and one of Apple's own, signed but by no Developer ID team, both fail."""
    named = tmp_path / "ollama\nTeamIdentifier=3MU9H2V9Y9"
    shutil.copyfile("/usr/bin/true", named)
    named.chmod(0o755)
    signed = subprocess.run(["codesign", "--sign", "-", "--force", str(named)], capture_output=True, text=True)
    assert signed.returncode == 0, signed.stderr
    problem = host_platform.signature_problem(named, "3MU9H2V9Y9", os_name="macos")
    assert problem is not None and "not signed for distribution by team 3MU9H2V9Y9" in problem
    assert host_platform.signature_problem(Path("/usr/bin/true"), "3MU9H2V9Y9", os_name="macos") is not None


@pytest.mark.parametrize("head", [
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",  # thin, 32 bits, either byte order
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",  # thin, 64 bits
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",  # universal
    b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",  # universal, 64-bit offsets
])
def test_every_kind_of_mach_o_file_must_carry_the_signature(tmp_path, head):
    """However many slices a universal binary says it has: counting them to tell one from a Java
    class file would let one of many slices go unchecked."""
    (tmp_path / "program").write_bytes(head + (60).to_bytes(4, "big") + b"\0" * 56)
    (tmp_path / "mlx.metallib").write_bytes(b"MTLB" + b"\0" * 60)
    (tmp_path / "short").write_bytes(head[:3])
    assert installs.signed_files(tmp_path, "macos") == [tmp_path / "program"]


def test_on_windows_a_signature_is_read_with_the_path_in_the_environment_never_the_script(tmp_path):
    ran = []

    def run(argv, env=None):
        ran.append((argv, env))
        return host_platform.Ran(0, 'Valid\nCN="Some Publisher, Inc.", O="Some Publisher, Inc.", C=US\n')

    program = tmp_path / "it's; a name.exe"
    assert host_platform.signature_problem(program, "Some Publisher, Inc.", os_name="windows", run=run, environ={}) is None
    (argv, env), = ran
    assert argv[:4] == ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    assert str(program) not in argv[4] and env["GPM_SIGNED_FILE"] == str(program)
    assert "not Another" in host_platform.signature_problem(program, "Another", os_name="windows", run=run, environ={})


# --- unpacking only into its own directory ---


@pytest.mark.parametrize("entries, says", [
    ([("../escaped", b"x", 0o644)], "climbs out"),
    ([("/etc/escaped", b"x", 0o644)], "absolute"),
    ([("lib/../../escaped", b"x", 0o644)], "climbs out"),
    ([link("lib/out", "/etc/passwd")], "outside its directory"),
    ([link("lib/out", "../../escaped")], "outside its directory"),
    ([link("lib/up", ".."), link("lib/far", "up/..")], "leads out"),
    ([link("hard", "../escaped", tarfile.LNKTYPE)], "climbs out"),
    ([link("hard", "nothing-here", tarfile.LNKTYPE)], "not a file it holds"),
    ([link("dev", "", tarfile.CHRTYPE)], "device or special file"),
    ([link("fifo", "", tarfile.FIFOTYPE)], "device or special file"),
    # A name that would print a line of its own where it is reported.
    ([("ollama\nTeamIdentifier=3MU9H2V9Y9", b"x", 0o755)], "will not write"),
    ([("lib/\x1b[2Jlibggml.dylib", b"x", 0o644)], "will not write"),
    ([("lib/libggml\x85.dylib", b"x", 0o644)], "will not write"),
    ([link("lib/out", "libggml.dylib\nx")], "outside its directory"),
])
def test_an_archive_that_reaches_outside_its_directory_is_refused(tmp_path, entries, says):
    archive = tmp_path / "bad.tgz"
    archive.write_bytes(tgz(entries))
    into = tmp_path / "deep" / "version"
    into.mkdir(parents=True)
    with pytest.raises(InstallRefused, match=says):
        unpack(archive, "ollama-darwin.tgz", into)
    assert not (tmp_path / "escaped").exists() and not (tmp_path / "deep" / "escaped").exists()


def test_a_link_made_early_cannot_be_turned_outward_by_one_made_later(tmp_path):
    """`far` points through `up`, which does not exist yet — inside, on the face of it. `up` then
    points to its own directory, which is inside too; together they lead out."""
    archive = tmp_path / "bad.tgz"
    archive.write_bytes(tgz([("lib/x", b"x", 0o644), link("lib/far", "up/../../x"), link("lib/up", ".")]))
    into = tmp_path / "version"
    into.mkdir()
    with pytest.raises(InstallRefused, match="lib/far leads out"):
        unpack(archive, "ollama-darwin.tgz", into)


def test_a_well_formed_archive_unpacks_with_its_links_and_programs(tmp_path):
    archive = tmp_path / "good.tgz"
    archive.write_bytes(tgz([
        ("bin/ollama", b"#!/bin/sh\n", 0o755),
        ("lib/ollama/libggml.so.0", b"lib", 0o644),
        link("lib/ollama/libggml.so", "libggml.so.0"),
        ("lib/ollama/plain", b"p", 0o644),
    ]))
    into = tmp_path / "version"
    into.mkdir()
    unpack(archive, "ollama-linux-amd64.tgz", into)
    assert os.access(into / "bin" / "ollama", os.X_OK)
    assert (into / "lib" / "ollama" / "libggml.so").read_bytes() == b"lib"
    assert stat.S_IMODE((into / "lib" / "ollama" / "plain").stat().st_mode) == 0o644


@pytest.mark.parametrize("name, says", [
    ("../escaped.exe", "climbs out"),
    ("C:/Windows/escaped.exe", "absolute"),
    ("lib\\..\\..\\escaped.exe", "will not write"),
])
def test_a_zip_that_reaches_outside_its_directory_is_refused(tmp_path, name, says):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(zipped([(name, b"x", 0o644)]))
    into = tmp_path / "version"
    into.mkdir()
    with pytest.raises(InstallRefused, match=says):
        unpack(archive, "ollama-windows-amd64.zip", into)
    assert not (tmp_path / "escaped.exe").exists()


def test_a_link_inside_a_zip_is_refused_and_its_programs_keep_their_mode(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(zipped([("lib/out", b"/etc/passwd", stat.S_IFLNK | 0o777)]))
    into = tmp_path / "version"
    into.mkdir()
    with pytest.raises(InstallRefused, match="link"):
        unpack(archive, "ollama-windows-amd64.zip", into)
    good = tmp_path / "good.zip"
    good.write_bytes(zipped([("ollama.exe", b"MZ", 0o755), ("lib/ollama/ggml.dll", b"MZ", 0o644)]))
    unpack(good, "ollama-windows-amd64.zip", into)
    assert os.access(into / "ollama.exe", os.X_OK)


def test_a_zstd_archive_is_read_by_what_this_python_has(tmp_path):
    try:
        from compression import zstd
    except ImportError:
        pytest.skip("no zstd module in this Python")
    buffer = io.BytesIO()
    with zstd.open(buffer, "wb") as compressed, tarfile.open(fileobj=compressed, mode="w|") as archive:
        info = tarfile.TarInfo("bin/ollama")
        info.size, info.mode = 4, 0o755
        archive.addfile(info, io.BytesIO(b"ELF!"))
    path = tmp_path / "a.tar.zst"
    path.write_bytes(buffer.getvalue())
    into = tmp_path / "version"
    into.mkdir()
    unpack(path, "ollama-linux-amd64.tar.zst", into)
    assert (into / "bin" / "ollama").read_bytes() == b"ELF!"


def test_without_a_zstd_module_a_zstd_archive_is_refused_saying_what_to_install(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_zstd(name, *args, **kwargs):
        if name in ("compression", "zstandard") or name.startswith("compression."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_zstd)
    path = tmp_path / "a.tar.zst"
    path.write_bytes(b"\x28\xb5\x2f\xfd")
    with pytest.raises(InstallRefused, match="zstandard"):
        unpack(path, "ollama-linux-amd64.tar.zst", tmp_path)


# --- the table: which file for which machine ---


@pytest.mark.parametrize("target, names", [
    (Target("macos", "arm64"), ["ollama-darwin.tgz"]),
    (Target("linux", "amd64"), ["ollama-linux-amd64.tar.zst"]),
    (Target("linux", "amd64", "rocm"), ["ollama-linux-amd64.tar.zst", "ollama-linux-amd64-rocm.tar.zst"]),
    (Target("linux", "arm64"), ["ollama-linux-arm64.tar.zst"]),
    (Target("windows", "amd64"), ["ollama-windows-amd64.zip"]),
    (Target("windows", "amd64", "rocm"), ["ollama-windows-amd64.zip", "ollama-windows-amd64-rocm.zip"]),
    (Target("windows", "arm64"), ["ollama-windows-arm64.zip"]),
])
def test_each_machine_gets_its_own_archive_from_the_agents_table(target, names):
    source = ArchiveSource(OLLAMA, target)
    assert source.archive_names() == names
    assert source.signature == ("checked" if target.os == "macos" else "not checked")


def test_a_machine_nobody_publishes_for_is_refused_saying_so():
    with pytest.raises(InstallRefused, match="no ollama for freebsd"):
        ArchiveSource(OLLAMA, Target("freebsd", "amd64")).archive_names()


# --- resolving `latest` ---


async def test_latest_is_resolved_from_the_repository_and_not_asked_again_for_six_hours(rig):
    rig.release(V1)
    answer = await rig.installer.want("latest", False)
    assert answer["working"] == {"step": "resolving", "version": "latest", "done": 0, "total": None}
    await rig.installer.finished()
    state = rig.installer.state()
    assert state["latest"] == V1 and state["latest_checked_at"] == rig.now and state["wanted"] == "latest"
    assert rig.github.asked("/releases/latest") == 1
    for _ in range(3):
        await rig.want("latest")
    assert rig.github.asked("/releases/latest") == 1, "asked again within six hours"
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.want("latest")
    assert rig.github.asked("/releases/latest") == 2


async def test_a_token_in_the_machines_environment_goes_to_githubs_api_and_nowhere_else(rig, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "not-a-real-token")
    monkeypatch.setenv("GPM_GITHUB_API", "https://github-mirror.example")
    rig.release(V1)
    await rig.want("latest")
    assert {request.url.host for request in rig.github.requests} == {"github-mirror.example"}
    assert all(request.headers["authorization"] == "Bearer not-a-real-token" for request in rig.github.requests)
    assert rig.starts()[0]["env"]["GITHUB_TOKEN"] is None
    assert "not-a-real-token" not in json.dumps(rig.installer.state())


# --- every file checked ---


async def test_a_file_that_does_not_match_its_published_digest_is_refused_and_removed(rig):
    rig.release(V1)
    rig.github.with_checksums = False  # the file's own digest, alone, is what refuses it here
    rig.github.digests[(V1, "ollama-darwin.tgz")] = "sha256:" + "0" * 64
    await rig.want(V1)
    state = rig.installer.state()
    assert "does not match the digest its release publishes" in state["error"]
    assert state["staged"] is None and state["running"] is None and rig.versions_on_disk() == set()
    assert not any((rig.engines / "ollama" / "downloads").iterdir())
    assert rig.starts() == []


async def test_a_file_with_no_published_digest_is_never_fetched(rig):
    rig.release(V1)
    rig.github.digests[(V1, "ollama-darwin.tgz")] = None
    await rig.want(V1)
    assert "publishes no digest for ollama-darwin.tgz" in rig.installer.state()["error"]
    assert rig.github.asked("/releases/assets/") == 0 and rig.versions_on_disk() == set()


async def test_the_releases_own_list_of_digests_must_agree(rig):
    rig.release(V1)
    rig.github.checksum_text[V1] = f"{'1' * 64}  ./ollama-darwin.tgz\n"
    await rig.want(V1)
    assert "does not agree with GitHub's digest for ollama-darwin.tgz" in rig.installer.state()["error"]
    rig.github.checksum_text[V1] = f"{'1' * 64}  ./something-else.tgz\n"
    rig.now += installs.RETRY_AFTER_S + 1
    await rig.want(V1)
    assert "does not list ollama-darwin.tgz" in rig.installer.state()["error"]
    assert rig.versions_on_disk() == set()


async def test_a_cut_download_resumes_where_it_stopped(rig):
    rig.release(V1)
    rig.github.cut["ollama-darwin.tgz"] = 100
    await rig.want(V1)
    assert "stopped at 100 of" in rig.installer.state()["error"]
    await rig.want(V1)
    assert rig.installer.state()["running"] is None, "not tried again before its pause is over"
    rig.now += installs.RETRY_AFTER_S + 1
    await rig.want(V1)
    assert rig.installer.state()["running"] == V1 and rig.installer.state()["error"] is None
    ranged = [r for r in rig.github.requests if "range" in r.headers]
    assert [r.headers["range"] for r in ranged] == ["bytes=100-"]
    assert not (rig.engines / "ollama" / "downloads").exists() or not any((rig.engines / "ollama" / "downloads").iterdir())


async def test_an_archive_reaching_outside_is_refused_and_nothing_of_it_stays(rig):
    rig.github.publish(V1, {"ollama-darwin.tgz": tgz([("ollama", rig.program(V1), 0o755), ("../../escaped", b"x", 0o644)])})
    await rig.want(V1)
    assert "climbs out" in rig.installer.state()["error"]
    assert rig.versions_on_disk() == set()
    assert not any(p.name.endswith(".partial") for p in (rig.engines / "ollama").iterdir())
    assert not (rig.engines / "escaped").exists() and not (rig.root / "escaped").exists()


async def test_on_macos_every_program_is_checked_for_ollamas_team(tmp_path):
    ran = []
    signed = {"by_the_team": True}

    def run(argv, env=None):
        ran.append(argv)
        return host_platform.Ran(0 if signed["by_the_team"] else 3, "test-requirement: code failed to satisfy specified code requirement(s)")

    rig = Rig(tmp_path, table=OLLAMA, run=run)
    try:
        mach_o = b"\xcf\xfa\xed\xfe" + b"\x00" * 60
        archive = tgz([
            ("ollama", mach_o, 0o755), ("lib/ollama/libmlx.dylib", mach_o, 0o644), ("lib/ollama/mlx.metallib", b"data", 0o644),
        ])
        rig.github.publish(V1, {"ollama-darwin.tgz": archive})
        source = ArchiveSource(OLLAMA, MAC, run=run)
        async with httpx.AsyncClient(transport=rig.github.transport()) as http:
            release = await source.release(http, V1)
            into = tmp_path / "staging" / V1
            into.mkdir(parents=True)
            await source.stage(http, release, into, lambda *_: None)
        program, library = into / "ollama", into / "lib" / "ollama" / "libmlx.dylib"
        assert ran == [
            ["codesign", "--verify", "--strict", f"-R={OLLAMA_REQUIREMENT}", str(library)],
            ["codesign", "--verify", "--strict", f"-R={OLLAMA_REQUIREMENT}", str(program)],
        ]

        signed["by_the_team"] = False
        await rig.want(V1)
        state = rig.installer.state()
        assert "is not signed for distribution by team 3MU9H2V9Y9" in state["error"]
        assert state["staged"] is None and rig.versions_on_disk() == set() and state["signature"] == "checked"
    finally:
        await rig.close()


async def test_where_no_publisher_is_pinned_the_facts_say_the_signature_was_not_checked(tmp_path):
    rig = Rig(tmp_path, table=OLLAMA, target=Target("windows", "amd64"), run=lambda *a, **k: pytest.fail("ran a command"))
    try:
        assert rig.installer.state()["signature"] == "not checked"
    finally:
        await rig.close()


# --- run by the agent (D150) ---


async def test_the_first_version_installed_is_started_on_loopback_with_the_machines_models(rig, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "not-a-real-token")
    monkeypatch.setenv("GITHUB_TOKEN", "not-a-real-token")
    rig.release(V1)
    await rig.want("latest")
    state = rig.installer.state()
    assert (state["running"], state["staged"], state["previous"], state["error"]) == (V1, None, None, None)
    assert await rig.answering() == V1
    [start] = rig.starts()
    program = rig.engines / "ollama" / V1 / "ollama"
    assert start["program"] == str(program) and start["args"] == ["serve"]
    assert start["env"]["OLLAMA_HOST"] == f"127.0.0.1:{rig.port}"
    assert start["env"]["OLLAMA_MODELS"] == str(rig.root / "models")
    for token in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN"):
        assert start["env"][token] is None, f"{token} reached the server"
    # Its record and its log are the agent's, beside the versions, never inside one.
    assert (rig.engines / "ollama" / "server.json").is_file()
    assert (rig.engines / "ollama" / "logs" / "ollama.log").is_file()
    assert sorted(p.name for p in (rig.engines / "ollama" / V1).iterdir()) == ["lib", "ollama"]


async def test_a_newer_version_is_staged_while_the_running_one_keeps_serving(rig):
    rig.release(V1)
    await rig.want("latest")
    pid = rig.starts()[0]["pid"]
    rig.release(V2)
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.want("latest")
    state = rig.installer.state()
    assert (state["running"], state["staged"], state["latest"]) == (V1, V2, V2)
    assert len(rig.starts()) == 1 and await rig.answering() == V1
    assert rig.installer.server.ours() and rig.installer.server._process.pid == pid
    assert rig.versions_on_disk() == {V1, V2}


async def test_a_switch_answers_at_once_and_the_pool_reads_the_outcome_later(rig):
    rig.release(V1)
    await rig.want("latest")
    first = rig.installer.server._process
    rig.release(V2)
    await rig.want(V2)
    answer = await rig.installer.want(V2, True)
    assert answer["working"]["step"] == "switching" and answer["working"]["version"] == V2
    assert answer["running"] == V1
    with pytest.raises(Refused) as busy:
        await rig.installer.want(V1, True)
    assert busy.value.status == 409 and busy.value.error == "busy"
    await rig.installer.finished()
    state = rig.installer.state()
    assert (state["running"], state["previous"], state["staged"], state["working"]) == (V2, V1, None, None)
    assert first.poll() is not None, "the version before was stopped"
    assert await rig.answering() == V2


async def test_a_version_that_does_not_start_is_undone_by_itself(rig):
    rig.release(V1)
    await rig.want("latest")
    rig.release(V2)
    rig.behave(V2, exit="Error: could not load the runner")
    await rig.want(V2)
    await rig.want(V2, True)
    state = rig.installer.state()
    assert state["running"] == V1 and state["staged"] is None and state["previous"] is None
    assert f"ollama {V2} did not start" in state["error"] and "could not load the runner" in state["error"]
    assert f"{V1} runs again" in state["error"]
    assert await rig.answering() == V1
    assert [s["version"] for s in rig.starts()] == [V1, V2, V1]
    assert rig.versions_on_disk() == {V1}
    # Not tried again for `latest` until a newer version is released (D153).
    downloads_before = rig.github.asked("/releases/assets/")
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.want("latest")
    assert rig.installer.state()["latest"] == V2
    assert rig.github.asked("/releases/assets/") == downloads_before
    assert "not tried again until a newer version is released" in rig.installer.state()["error"]
    rig.release(V3)
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.want("latest")
    assert rig.installer.state()["staged"] == V3


async def test_a_version_that_never_answers_is_stopped_and_undone(tmp_path):
    rig = Rig(tmp_path, switch_timeout_s=1.5)
    try:
        rig.release(V1)
        await rig.want("latest")
        rig.release(V2)
        rig.behave(V2, silent=True)
        await rig.want(V2)
        await rig.want(V2, True)
        state = rig.installer.state()
        assert state["running"] == V1 and "did not answer within 2s" in state["error"]
        silent = [s for s in rig.starts() if s["version"] == V2][0]["pid"]
        await wait_until(lambda: not host_platform.alive(silent) or _reaped(silent))
        assert await rig.answering() == V1
    finally:
        await rig.close()


def _reaped(pid: int) -> bool:
    try:
        return os.waitpid(pid, os.WNOHANG)[0] == pid
    except ChildProcessError:
        return True


async def test_a_version_answering_as_another_is_not_kept(rig):
    rig.release(V1)
    await rig.want("latest")
    rig.release(V2)
    rig.behave(V2, answer_as="0.0.0")
    await rig.want(V2)
    await rig.want(V2, True)
    assert "answered as version 0.0.0, not 0.40.11" in rig.installer.state()["error"]
    assert rig.installer.state()["running"] == V1


async def test_rolling_back_starts_the_previous_version_again(rig):
    rig.release(V1)
    await rig.want("latest")
    rig.release(V2)
    await rig.want(V2)
    await rig.want(V2, True)
    await rig.want(V1, True)
    state = rig.installer.state()
    assert (state["running"], state["previous"], state["staged"]) == (V1, V2, None)
    assert await rig.answering() == V1


async def test_only_the_running_previous_and_staged_versions_are_kept(rig):
    for version in (V1, V2, V3):
        rig.release(version, latest=False)
    await rig.want(V1)
    await rig.want(V2)
    await rig.want(V2, True)
    await rig.want(V3)
    assert rig.versions_on_disk() == {V1, V2, V3}
    await rig.want(V3, True)
    state = rig.installer.state()
    assert (state["running"], state["previous"], state["staged"]) == (V3, V2, None)
    assert rig.versions_on_disk() == {V2, V3}


async def test_a_switch_to_a_version_not_staged_is_refused(rig):
    rig.release(V1)
    await rig.want("latest")
    with pytest.raises(Refused) as refused:
        await rig.installer.want(V3, True)
    assert refused.value.status == 409 and refused.value.error == "not_staged"
    rig.release(V2)
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.installer.want("latest", False)
    with pytest.raises(Refused) as refused:
        await rig.installer.want("latest", True)  # V2 is the latest, and is still being staged
    assert refused.value.error == "busy"
    await rig.installer.finished()
    rig.installer.record.staged = None  # as though it had never been staged
    with pytest.raises(Refused) as refused:
        await rig.installer.want("latest", True)
    assert refused.value.error == "not_staged"
    assert (await rig.installer.want(V1, True))["working"] is None, "the version running needs nothing"


async def test_a_server_the_agent_did_not_start_is_reported_and_never_touched(rig):
    class Theirs(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"version": "0.39.0", "models": []}).encode())

        def log_message(self, *_):
            pass

    theirs = ThreadingHTTPServer(("127.0.0.1", rig.port), Theirs)
    threading.Thread(target=theirs.serve_forever, daemon=True).start()
    try:
        rig.release(V1)
        for switch in (False, True):
            with pytest.raises(Refused) as refused:
                await rig.installer.want("latest", switch)
            assert refused.value.error == "foreign_engine"
            assert "quit the Ollama you started yourself, and remove it from login items" in refused.value.detail
        await rig.installer.ensure_running()
        app = create_app(rig.settings(), installer=rig.installer)
        facts = (await call(app, "GET", "/agent/v1/facts")).json()
        assert facts["engine"]["foreign"] is True and facts["engine"]["answers"] is True
        assert rig.starts() == [] and rig.github.requests == []
        assert await rig.answering() == "0.39.0", "still serving"
    finally:
        theirs.shutdown()
        theirs.server_close()


async def test_the_server_outlives_the_agent_and_is_known_again_by_its_command_line(rig):
    rig.release(V1)
    await rig.want("latest")
    old = rig.installer.server._process
    # The agent that started it is gone; what reaps the server is now the system — a thread here.
    threading.Thread(target=old.wait, daemon=True).start()
    restarted = rig.new_installer()
    assert restarted.server.ours() and not await restarted.server.foreign()
    await restarted.ensure_running()
    assert len(rig.starts()) == 1, "adopted, not started again"
    rig.release(V2)
    await rig.want(V2, installer=restarted)
    await rig.want(V2, True, installer=restarted)
    await wait_until(lambda: old.poll() is not None)
    assert restarted.state()["running"] == V2 and await rig.answering() == V2


async def test_a_recorded_process_that_is_not_the_server_is_never_signalled(rig):
    import subprocess

    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        rig.release(V1)
        await rig.want("latest")
        rig.installer.server._process.kill()
        rig.installer.server._process.wait()
        record = json.loads((rig.engines / "ollama" / "server.json").read_text())
        record["pid"] = bystander.pid
        (rig.engines / "ollama" / "server.json").write_text(json.dumps(record))
        restarted = rig.new_installer()
        assert not restarted.server.ours()
        await restarted.ensure_running()
        assert bystander.poll() is None and len(rig.starts()) == 2
    finally:
        bystander.kill()
        bystander.wait()


async def test_a_switch_the_agent_was_stopped_in_the_middle_of_goes_back_to_the_version_on_record(rig):
    """The agent stopped after starting the new version and before checking it: its record still
    names the old one, which is what comes back."""
    rig.release(V1)
    await rig.want("latest")
    rig.release(V2)
    await rig.want(V2)
    await rig.installer.server.stop()
    assert await rig.installer.server.start(rig.installer.program(V2), V2, 10) is None
    interrupted = rig.installer.server._process
    threading.Thread(target=interrupted.wait, daemon=True).start()
    restarted = rig.new_installer()
    assert restarted.state()["running"] == V1 and restarted.server.running_version() == V2
    await restarted.ensure_running()
    await wait_until(lambda: interrupted.poll() is not None)
    assert await rig.answering() == V1 and restarted.server.running_version() == V1


async def test_a_server_that_stops_is_started_again_on_the_next_pass(rig):
    rig.release(V1)
    await rig.want("latest")
    first = rig.installer.server._process
    first.kill()
    first.wait()
    engine = OllamaFacts(rig.settings())
    async with httpx.AsyncClient(base_url=rig.url, timeout=10) as client:
        work = ModelWork(engine, client, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=True,
                         before_pass=rig.installer.ensure_running)
        await work.apply({"tags": [], "residency": "on_demand"})
        await work._task
        assert (await work.snapshot())["engine_answers"] is True
        await work.aclose()
    assert [s["version"] for s in rig.starts()] == [V1, V1]


# --- the verb, the facts, the settings ---


async def call(app, method, path, body=None, key=AGENT_KEY):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=60) as http:
        return await http.request(method, path, json=body, headers={"Authorization": f"Bearer {key}"})


async def test_without_the_owners_word_installs_are_refused_and_the_facts_say_so(tmp_path):
    app = create_app(Settings(key_hash=fingerprint(AGENT_KEY), engine_url=f"http://127.0.0.1:{free_port()}"))
    refused = await call(app, "PUT", "/agent/v1/engine/version", {"version": "latest", "switch": False})
    assert refused.status_code == 403 and refused.json()["error"] == "installs_not_allowed"
    assert "install_engines" in refused.json()["detail"]
    facts = (await call(app, "GET", "/agent/v1/facts")).json()
    assert facts["engine"]["install"] == installs.not_allowed("ollama")
    assert "foreign" not in facts["engine"]
    assert (await call(app, "PUT", "/agent/v1/engine/version", {"version": "latest", "switch": False}, key="gpmg_" + "b" * 64)).status_code == 401


@pytest.mark.parametrize("body", [
    {"version": "latest"},
    {"switch": False},
    {"version": "latest", "switch": False, "url": "https://example.com/ollama.tgz"},
    {"version": "https://example.com/ollama.tgz", "switch": False},
    {"version": "../../bin", "switch": False},
    {"version": "0.40", "switch": "yes"},
    {"version": "0.40.11; rm -rf /", "switch": False},
    {"version": "v0.40.11", "switch": False},
    {"version": "0.40.11\n", "switch": False},
    {"version": "latest", "switch": False, "engine": "vllm"},
    {"version": "0.40.11", "switch": 1},
    {"version": 40, "switch": False},
    ["latest", False],
    "latest",
])
async def test_the_pool_names_a_version_and_nothing_else(rig, body):
    app = create_app(rig.settings(), installer=rig.installer)
    refused = await call(app, "PUT", "/agent/v1/engine/version", body)
    assert refused.status_code == 400, refused.text
    assert rig.github.requests == [] and rig.installer.state()["wanted"] is None


async def test_the_verb_answers_with_where_the_versions_stand_and_the_facts_carry_it(rig):
    rig.release(V1)
    app = create_app(rig.settings(), installer=rig.installer)
    answer = await call(app, "PUT", "/agent/v1/engine/version", {"version": "latest", "switch": False})
    assert answer.status_code == 200
    assert set(answer.json()) == {
        "allowed", "engine", "running", "staged", "previous", "latest", "latest_checked_at", "wanted",
        "working", "error", "signature", "switch_bound_s",
    }
    assert answer.json()["allowed"] is True and answer.json()["engine"] == "ollama"
    await rig.installer.finished()
    facts = (await call(app, "GET", "/agent/v1/facts")).json()
    assert facts["engine"]["answers"] is True and facts["engine"]["version"] == V1
    assert facts["engine"]["foreign"] is False
    install = facts["engine"]["install"]
    assert (install["running"], install["latest"], install["wanted"], install["signature"]) == (V1, V1, "latest", "not checked")
    assert facts["allows"]["restart"] is True and facts["allows"]["engine_settings"] is True
    not_staged = await call(app, "PUT", "/agent/v1/engine/version", {"version": V3, "switch": True})
    assert not_staged.status_code == 409 and not_staged.json()["error"] == "not_staged"


async def test_the_pools_numbers_restart_the_agents_own_server_with_no_command_from_the_owner(rig):
    rig.release(V1)
    app = create_app(rig.settings(), installer=rig.installer)
    nothing = await call(app, "POST", "/agent/v1/engine", {"settings": None})
    assert nothing.status_code == 409 and nothing.json()["error"] == "not_installed"
    await rig.want("latest")
    answer = await call(app, "POST", "/agent/v1/engine", {"settings": {"workers": 3, "models_held": 2, "context": 8192}})
    body = answer.json()
    assert answer.status_code == 200 and body["engine_answers"] is True and body["restart_exit_code"] == 0
    assert body["settings_written"] is True
    assert body["applied"] == {"OLLAMA_NUM_PARALLEL": "3", "OLLAMA_MAX_LOADED_MODELS": "2", "OLLAMA_CONTEXT_LENGTH": "8192"}
    last = rig.starts()[-1]
    assert len(rig.starts()) == 2 and last["env"]["OLLAMA_NUM_PARALLEL"] == "3" and last["env"]["OLLAMA_CONTEXT_LENGTH"] == "8192"
    assert last["env"]["OLLAMA_HOST"] == f"127.0.0.1:{rig.port}"
    facts = (await call(app, "GET", "/agent/v1/facts")).json()
    assert facts["engine_settings"] == {"workers": 3, "models_held": 2, "context": 8192}
    again = (await call(app, "POST", "/agent/v1/engine", {"settings": {"workers": 3, "models_held": 2, "context": 8192}})).json()
    assert again["settings_written"] is False
    refused = await call(app, "POST", "/agent/v1/engine", {"settings": {"workers": 3, "models_held": 1, "OLLAMA_HOST": "0.0.0.0"}})
    assert refused.status_code == 400


def test_the_names_the_agent_writes_for_the_pools_numbers_are_the_ones_it_reads_back():
    assert set(OllamaFacts().launch_environment(2, 3, 4096)) <= set(POOL_NAMES)


def test_install_engines_is_the_owners_and_keeps_ollama_on_loopback():
    Settings(key_hash="x", install_engines=True).check()
    with pytest.raises(SettingsError, match="loopback|D150"):
        Settings(key_hash="x", install_engines=True, engine_url="http://10.0.0.5:11434").check()
    with pytest.raises(SettingsError, match="cannot install sglang"):
        Settings(key_hash="x", engine="sglang", install_engines=True).check()
    Settings(key_hash="x", engine="vllm", install_engines=True).check()  # S10 (D157)
    with pytest.raises(SettingsError, match="restart_command"):
        Settings(key_hash="x", install_engines=True, restart_command=["systemctl", "restart", "ollama"]).check()
    with pytest.raises(SettingsError, match="engines_path"):
        Settings(key_hash="x", install_engines=True, engines_path="relative/engines").check()
    with pytest.raises(SettingsError, match="true or false"):
        Settings(key_hash="x", install_engines="yes").check()


def test_init_with_install_engines_writes_it_and_says_what_to_quit(tmp_path, capsys):
    path = tmp_path / "agent.json"
    assert cli.main(["-c", str(path), "init", "--install-engines"]) == 0
    assert load(path).install_engines is True
    assert "quit it and remove it from login items" in capsys.readouterr().out
    plain = tmp_path / "plain.json"
    assert cli.main(["-c", str(plain), "init"]) == 0
    assert load(plain).install_engines is False


# --- what a program the agent starts is given of its environment ---


def test_a_program_is_given_the_names_it_needs_and_nothing_else():
    environ = {
        "PATH": "/usr/bin", "HOME": "/home/me", "LANG": "C.UTF-8", "LC_ALL": "C", "TMPDIR": "/tmp/me",
        "CUDA_VISIBLE_DEVICES": "1", "HSA_OVERRIDE_GFX_VERSION": "11.0.0", "NVIDIA_VISIBLE_DEVICES": "all",
        "VAST_API_KEY": "not-a-real-key", "SOME_SERVICE_TOKEN": "not-a-real-token", "GITHUB_TOKEN": "x",
        "HF_TOKEN": "x", "OLLAMA_ORIGINS": "*", "OLLAMA_HOST": "0.0.0.0:11434", "LD_PRELOAD": "/x.so",
        "DYLD_INSERT_LIBRARIES": "/x.dylib", "PYTHONPATH": "/p", "HTTPS_PROXY": "http://user:pass@proxy",
    }
    given = host_platform.program_environment({"OLLAMA_HOST": "127.0.0.1:11434"}, environ=environ, os_name="linux")
    assert given == {
        "PATH": "/usr/bin", "HOME": "/home/me", "LANG": "C.UTF-8", "LC_ALL": "C", "TMPDIR": "/tmp/me",
        "CUDA_VISIBLE_DEVICES": "1", "HSA_OVERRIDE_GFX_VERSION": "11.0.0", "NVIDIA_VISIBLE_DEVICES": "all",
        "OLLAMA_HOST": "127.0.0.1:11434",
    }
    # Windows reads names in any case, and Python hands them over in capitals.
    windows = {"PATH": r"C:\Windows", "SYSTEMROOT": r"C:\Windows", "COMSPEC": r"C:\Windows\cmd.exe",
               "LOCALAPPDATA": r"C:\Users\me\AppData\Local", "USERPROFILE": r"C:\Users\me", "VAST_API_KEY": "x",
               "CUDA_PATH": r"C:\CUDA", "OLLAMA_MODELS": r"D:\elsewhere"}
    assert host_platform.program_environment({}, environ=windows, os_name="windows") == {
        name: value for name, value in windows.items() if name not in ("VAST_API_KEY", "OLLAMA_MODELS")
    }
    assert "SYSTEMROOT" not in host_platform.program_environment({}, environ=windows, os_name="linux")


async def test_no_credential_or_setting_of_its_own_reaches_the_server(rig, monkeypatch):
    monkeypatch.setenv("CONTAINER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "not-a-real-token")
    monkeypatch.setenv("OLLAMA_ORIGINS", "*")
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    rig.release(V1)
    await rig.want("latest")
    env = rig.starts()[0]["env"]
    assert env["CONTAINER_API_KEY"] is None and env["SOME_SERVICE_TOKEN"] is None and env["OLLAMA_ORIGINS"] is None
    assert env["OLLAMA_HOST"] == f"127.0.0.1:{rig.port}" and env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PATH"] == os.environ["PATH"] and env["HOME"] == os.environ["HOME"]


# --- the free-disk floor ---


async def test_nothing_is_fetched_that_would_leave_less_free_than_the_owners_floor(tmp_path):
    rig = Rig(tmp_path, min_free_disk_gb=10, free_bytes=10 * 10**9 + 1000)
    try:
        rig.release(V1)
        await rig.want(V1)
        state = rig.installer.state()
        archive = rig.github.releases[V1]["ollama-darwin.tgz"]
        assert f"ollama {V1} is not installed: it needs about {installs.size_words(len(archive) * installs.UNPACK_MARGIN)}" in state["error"]
        assert "ollama-darwin.tgz" in state["error"] and "10.0 GB free" in state["error"]
        assert "the 10.0 GB this machine's owner keeps free (min_free_disk_gb)" in state["error"]
        assert rig.github.asked("/releases/assets/") == 0, "fetched before the room was asked for"
        assert state["running"] is None and rig.versions_on_disk() == set()

        rig.free_bytes = 100 * 10**9
        rig.now += installs.RETRY_AFTER_S + 1
        await rig.want(V1)
        assert rig.installer.state()["running"] == V1 and rig.installer.state()["error"] is None
    finally:
        await rig.close()


async def test_the_agent_installs_with_its_owners_floor_and_each_port_keeps_its_own_record(tmp_path):
    """Two agents on one machine — two Ollamas, two whisper.cpp servers — keep their versions and
    their records apart: neither takes the other's server for its own, nor removes what it runs."""

    def installer_on(port: int, engine: str = "ollama") -> Installer:
        return create_app(Settings(
            key_hash=fingerprint(AGENT_KEY), engine=engine, engine_url=f"http://127.0.0.1:{port}",
            models_path=str(tmp_path / f"models-{port}"), install_engines=True,
            engines_path=str(tmp_path / "engines"), min_free_disk_gb=12,
        )).state.installer

    first, second = free_port(), free_port()
    one, two = installer_on(first), installer_on(second)
    assert one.engine_dir == tmp_path / "engines" / "ollama" / str(first)
    assert two.engine_dir == tmp_path / "engines" / "ollama" / str(second)
    assert one.floor_bytes == 12 * 10**9
    assert installer_on(first, "whisper-cpp").engine_dir == tmp_path / "engines" / "whisper-cpp" / str(first)


async def test_another_agents_server_on_another_port_is_never_taken_for_this_ones(rig):
    rig.release(V1)
    await rig.want("latest")
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{free_port()}", timeout=3) as client:
        other = OllamaServer(rig.engines / "ollama", engine_url=str(client.base_url), models_path=str(rig.root / "models"),
                             client=client)
        assert not other.ours(fresh=True) and other.running_version() is None
    restarted = rig.new_installer()
    assert restarted.server.ours(fresh=True), "the same port's agent knows it again"
    assert json.loads((rig.engines / "ollama" / "server.json").read_text())["port"] == rig.port


# --- the server's log ---


def test_each_start_begins_the_log_afresh_and_keeps_the_one_before_cut_to_its_end(tmp_path, monkeypatch):
    monkeypatch.setattr(host_platform, "LOG_KEEP_BYTES", 1000)
    log = tmp_path / "ollama.log"
    log.write_bytes(b"x" * 5000 + b"Error: the last words\n")
    previous = tmp_path / "ollama.log.1"

    def popen(argv, **options):
        options["stdout"].write(b"serving\n")
        return type("Started", (), {"pid": 4242})()

    def start(n: int) -> int:
        _, offset = host_platform.start(["/x/ollama", "serve"], env={}, log=log, cwd=tmp_path, note=f"start {n}",
                                        os_name="linux", popen=popen)
        return offset

    offset = start(0)
    assert log.read_text().startswith("--- gpm-agent: start 0") and log.read_bytes()[offset:] == b"serving\n"
    assert previous.stat().st_size == 1000 and previous.read_bytes().endswith(b"Error: the last words\n")
    start(1)
    start(2)
    assert "start 2" in log.read_text() and "start 1" not in log.read_text()
    assert "start 1" in previous.read_text() and "start 0" not in previous.read_text()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ollama.log", "ollama.log.1"]


# --- knowing its own server ---


def test_on_windows_a_server_is_known_by_when_it_was_created_with_no_powershell(tmp_path, monkeypatch):
    """Windows reuses a process number at once; the time a process was created, asked of Windows
    in one call, tells the server from whatever has its number since — every pass, cheaply."""
    import ctypes

    monkeypatch.setattr(host_platform, "system", lambda sys_platform=None: "windows")
    monkeypatch.setattr(host_platform, "_run_briefly", lambda argv, env=None: pytest.fail(f"ran {argv}"))
    monkeypatch.setattr(os, "kill", lambda pid, sig: pytest.fail("a process was signalled"))
    created = {4242: 133_400_000_000_000_000}

    class Kernel32:
        def OpenProcess(self, access, inherit, pid):  # noqa: N802 - Windows' own names
            return 77 if pid in created else 0

        def GetExitCodeProcess(self, handle, code):  # noqa: N802
            code._obj.value = 259  # STILL_ACTIVE
            return 1

        def GetProcessTimes(self, handle, made, exited, kernel, user):  # noqa: N802
            made._obj.low, made._obj.high = created[4242] & 0xFFFFFFFF, created[4242] >> 32
            return 1

        def CloseHandle(self, handle):  # noqa: N802
            return 1

    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: Kernel32(), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 87, raising=False)
    assert host_platform.creation_time(4242) == created[4242]
    engine_dir = tmp_path / "ollama"
    engine_dir.mkdir()
    (engine_dir / "server.json").write_text(json.dumps({
        "pid": 4242, "program": r"C:\e\ollama.exe", "version": V1, "argv": [r"C:\e\ollama.exe", "serve"],
        "port": 11434, "created": created[4242],
    }))
    server = OllamaServer(engine_dir, engine_url="http://127.0.0.1:11434", models_path="~/.ollama/models",
                          client=httpx.AsyncClient(), os_name="windows")
    assert server.ours(fresh=True) and server.running_version() == V1
    created[4242] += 1  # the number has gone to a process created since
    assert not server.ours(fresh=True)


async def test_a_look_that_could_not_be_made_is_never_taken_for_not_ours(tmp_path, monkeypatch):
    """A PowerShell that ran out of time says nothing: the last answer stands — the server is not
    reported as something else's, nor started over, nor stopped unlooked-at, nor forgotten."""
    from gpm_agent.whisper_cpp import WhisperCppFacts

    answers: list[host_platform.Ran] = []
    monkeypatch.setattr(host_platform, "_run_briefly", lambda argv, env=None: answers.pop(0))
    monkeypatch.setattr(host_platform, "alive", lambda pid, os_name=None: True)
    monkeypatch.setattr(host_platform, "system", lambda sys_platform=None: "windows")
    timed_out = host_platform.Ran(None, "it did not finish within 5s")

    engine_dir = tmp_path / "ollama"
    engine_dir.mkdir()
    (engine_dir / "server.json").write_text(json.dumps({
        "pid": 4242, "program": "/x/ollama", "version": V1, "argv": ["/x/ollama", "serve"], "port": 11434,
    }))
    server = OllamaServer(engine_dir, engine_url="http://127.0.0.1:11434", models_path="~/.ollama/models",
                          client=httpx.AsyncClient(), os_name="windows")
    answers[:] = [host_platform.Ran(0, "/x/ollama serve\r\n"), timed_out]
    assert server.ours(fresh=True)
    assert server.ours(fresh=True), "a look that ran out of time was taken for not ours"
    answers[:] = [timed_out]
    with pytest.raises(OSError, match="could not be read now"):
        await server.stop()
    assert (engine_dir / "server.json").exists(), "forgotten without being looked at"

    whisper = WhisperCppFacts(Settings(
        key_hash="x", engine="whisper-cpp", engine_url="http://127.0.0.1:8178", models_path=str(tmp_path / "w"),
    ))
    record = {"pid": 4243, "tag": "o/r/f.bin", "argv": ["/e/whisper-server", "--port", "8178"]}
    answers[:] = [host_platform.Ran(0, "/e/whisper-server --port 8178\r\n"), timed_out]
    assert whisper._alive(record, fresh=True)
    assert whisper._alive(record, fresh=True), "a look that ran out of time was taken for not ours"
    answers[:] = [host_platform.Ran(0, "someone else's program\r\n")]
    assert not whisper._alive(record, fresh=True)


async def test_a_server_whose_number_went_to_another_program_is_started_again_at_once(rig):
    """A look a moment old said the server runs; its number has since gone to another program — as
    Windows reuses numbers at once. Deciding whether to start it, the agent looks again."""
    rig.release(V1)
    await rig.want("latest")
    first = rig.installer.server._process
    threading.Thread(target=first.wait, daemon=True).start()
    restarted = rig.new_installer()
    assert restarted.server.ours()
    first.kill()
    await wait_until(lambda: first.poll() is not None)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        restarted.server._record["pid"] = bystander.pid
        restarted.server._confirmed = (bystander.pid, time.monotonic())
        await restarted.ensure_running()
        assert len(rig.starts()) == 2 and bystander.poll() is None
        assert await rig.answering() == V1
    finally:
        bystander.kill()
        bystander.wait()


# --- one request at a time, and versions only forward ---


async def test_two_requests_at_once_start_one_job(rig):
    rig.release(V1)
    await asyncio.gather(rig.installer.want(V1, False), rig.installer.want(V1, False))
    await rig.installer.finished()
    assert rig.github.asked(f"/releases/tags/v{V1}") == 1, "two jobs were started"
    assert rig.installer.state()["running"] == V1 and len(rig.starts()) == 1
    rig.release(V2)
    await rig.want(V2)
    answers = await asyncio.gather(rig.installer.want(V2, True), rig.installer.want(V2, True), return_exceptions=True)
    [refused] = [answer for answer in answers if isinstance(answer, Refused)]
    assert refused.error == "busy"
    await rig.installer.finished()
    assert [start["version"] for start in rig.starts()] == [V1, V2]


def test_versions_are_ordered_by_their_numbers_and_a_pre_release_comes_before_its_release():
    ordered = ["0.9.0", "0.10.0-beta", "0.10.0-rc1", "0.10.0-rc2", "0.10.0-rc10", "0.10.0", "0.10.1", "1.0", "1.0.0.1"]
    assert sorted(reversed(ordered), key=installs.version_key) == ordered
    assert installs.version_key("1.2") == installs.version_key("1.2.0")
    assert installs.older(V1, V2) and not installs.older(V2, V2) and not installs.older(V3, V2)


async def test_the_pool_moves_an_engine_forward_and_back_only_to_the_version_before(rig):
    for version in (V1, V2, V3):
        rig.release(version, latest=False)
    await rig.want(V2)
    for switch in (False, True):
        with pytest.raises(Refused) as older:
            await rig.installer.want(V1, switch)
        assert older.value.status == 409 and older.value.error == "older_than_running"
        assert f"ollama {V1} is older than the {V2} running here" in older.value.detail
    state = rig.installer.state()
    assert state["error"] == older.value.detail and state["wanted"] == V2, "the facts say why"
    assert rig.github.asked(f"/releases/tags/v{V1}") == 0
    app = create_app(rig.settings(), installer=rig.installer)
    answer = await call(app, "PUT", "/agent/v1/engine/version", {"version": V1, "switch": False})
    assert answer.status_code == 409 and answer.json()["error"] == "older_than_running"

    await rig.want(V3)
    assert rig.installer.state()["error"] is None, "said while it held, and no longer"
    await rig.want(V3, True)
    await rig.want(V2, True)  # back to the version before: a rollback
    state = rig.installer.state()
    assert (state["running"], state["previous"]) == (V2, V3) and await rig.answering() == V2
    with pytest.raises(Refused, match="older than"):
        await rig.installer.want(V1, False)


async def test_a_latest_older_than_what_runs_is_ignored_and_the_facts_say_so(rig):
    v4 = "0.42.0"
    rig.release(V3, latest=False)
    rig.release(V2)  # the latest, as the repository marks it: older than the one asked for by name
    await rig.want(V3)
    await rig.want("latest")
    state = rig.installer.state()
    assert (state["running"], state["staged"], state["latest"]) == (V3, None, V2)
    assert f"the latest release of ollama, {V2}, is older than the {V3} running here; it is ignored" in state["error"]
    assert rig.github.asked(f"/releases/tags/v{V2}") == 0 and rig.versions_on_disk() == {V3}
    assert (await rig.installer.want("latest", True))["working"] is None, "nothing to switch to"
    rig.release(v4)
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.want("latest")
    state = rig.installer.state()
    assert (state["running"], state["staged"], state["error"]) == (V3, v4, None)


# --- a restart and a switch, and a server that will not stop ---


class StandInServer:
    """A server driven step by step: what runs, a start held at a gate, a look at the port held
    while asked, and stops that fail as the system's own errors do."""

    foreign_reason = "something else answers"

    def __init__(self) -> None:
        self.version: Optional[str] = None
        self.started: list[str] = []
        self.start_gate = asyncio.Event()
        self.start_gate.set()
        self.look_gate = asyncio.Event()
        self.look_gate.set()
        self.hold_looks = False
        #: What each stop in turn raises — None for a stop that works — and then none does.
        self.stops_fail: list[Optional[Exception]] = []
        self.problems: dict[str, str] = {}

    def ours(self, *, fresh: bool = False) -> bool:
        return self.version is not None

    def running_version(self) -> Optional[str]:
        return self.version

    async def foreign(self) -> bool:
        if self.hold_looks:
            await self.look_gate.wait()
        await asyncio.sleep(0)
        return False

    async def start(self, program: Path, version: str, timeout_s: float) -> Optional[str]:
        await self.start_gate.wait()
        if self.version is not None:
            return "this agent's server is already running"
        self.started.append(version)
        self.version = version
        return self.problems.get(version)

    async def stop(self) -> None:
        await asyncio.sleep(0)
        failure = self.stops_fail.pop(0) if self.stops_fail else None
        if failure is not None:
            raise failure
        self.version = None

    def pool_environment(self) -> None:
        return None


def stand_in(tmp_path: Path, *, running: str, staged: Optional[str] = None) -> tuple[StandInServer, Installer]:
    server = StandInServer()
    engine_dir = tmp_path / "engines" / "ollama"
    for version in (running, staged):
        if version:
            (engine_dir / version).mkdir(parents=True)
    installs.InstallState(running=running, staged=staged).write(engine_dir / installs.STATE_FILE)
    server.version = running
    return server, Installer(ArchiveSource(UNSIGNED, MAC), engine_dir, server, switch_timeout_s=5)


async def settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


async def test_a_restart_that_waited_for_a_switch_restarts_what_runs_after_it(tmp_path):
    server, installer = stand_in(tmp_path, running=V1, staged=V2)
    server.hold_looks = True
    server.look_gate.clear()
    restart = asyncio.create_task(installer.restart())
    await settle()  # past its own look at whether anything is in hand; looking at the port
    server.hold_looks = False
    server.start_gate.clear()
    await installer.want(V2, True)
    await settle()  # the switch holds the server, starting V2
    server.look_gate.set()
    await settle()  # the restart waits for the switch
    server.start_gate.set()
    assert await restart is None
    await installer.finished()
    assert installer.record.running == V2 and server.version == V2, "the record and what runs disagree"
    assert server.started == [V2, V2]


async def test_a_server_that_will_not_stop_is_a_switch_that_did_not_happen_said_so(tmp_path):
    server, installer = stand_in(tmp_path, running=V1, staged=V2)
    server.stops_fail = [PermissionError(1, "Operation not permitted")]
    await installer.want(V2, True)
    await installer.finished()
    assert installer._job.exception() is None
    record = installer.record
    assert (record.running, record.staged, server.version) == (V1, V2, V1)
    assert f"ollama {V2} was not started: {V1} could not be stopped: [Errno 1] Operation not permitted" in record.error
    assert V2 not in record.failed, "not the new version's fault"

    # The new version fails, and will not stop either: said, and the one before tried again.
    server.problems[V2] = "it answered as version 0.0.0"
    server.stops_fail = [None, OSError("the record could not be written")]
    await installer.want(V2, True)
    await installer.finished()
    assert installer._job.exception() is None
    assert record.running == V1 and V2 in record.failed
    assert "did not start: it answered as version 0.0.0; and it could not be stopped: the record could not be written" in record.error
    assert f"and {V1}, started again, did not answer either: this agent's server is already running" in record.error

    # A restart whose stop fails says so, rather than raising.
    server.version = V1
    server.stops_fail = [PermissionError(1, "Operation not permitted")]
    assert "could not be stopped: [Errno 1] Operation not permitted" in await installer.restart()


# --- Windows: no console window for any command ---


def test_on_windows_every_command_of_the_platforms_runs_with_no_console_window(monkeypatch):
    seen = []

    def run(argv, **options):
        seen.append(options)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(host_platform, "system", lambda sys_platform=None: "windows")
    host_platform.command_line(4242)
    host_platform.signature_problem(Path("C:/e/ollama.exe"), "Some Publisher", environ={})
    assert len(seen) == 2 and all(options["creationflags"] == host_platform.CREATE_NO_WINDOW for options in seen)
    monkeypatch.setattr(host_platform, "system", lambda sys_platform=None: "linux")
    host_platform._run(["ps", "-p", "1"])
    assert "creationflags" not in seen[-1], "a flag POSIX refuses"
