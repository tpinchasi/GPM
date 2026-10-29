"""The agent installs, runs and upgrades Ollama on a machine whose owner allows it (D150–D153).

docs/spec/host-agent.md §2.3 and §7. Against a fake GitHub — releases with per-file digests, and
small archives built here — and a fake `ollama` program: a small Python program that answers
`GET /api/version` as the version it was packed as. Nothing here reaches the network, downloads
a real engine, or needs a GPU; the platform's commands (`codesign`, PowerShell) are stood in for.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import os
import signal
import socket
import stat
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
           "OLLAMA_CONTEXT_LENGTH", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN"]
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

    def __init__(self, root: Path, *, table=UNSIGNED, target=MAC, run=None, switch_timeout_s=20.0):
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


def test_on_macos_a_signature_is_checked_with_codesign_and_the_team_pinned(tmp_path):
    ran = []

    def run(argv, env=None):
        ran.append(argv)
        if argv[1] == "--verify":
            return host_platform.Ran(0, "")
        return host_platform.Ran(0, "Executable=/x/ollama\nAuthority=Developer ID Application: Someone (3MU9H2V9Y9)\nTeamIdentifier=3MU9H2V9Y9\n")

    program = tmp_path / "ollama"
    assert host_platform.signature_problem(program, "3MU9H2V9Y9", os_name="macos", run=run) is None
    assert ran == [["codesign", "--verify", "--strict", str(program)], ["codesign", "-dv", str(program)]]
    assert "not 0000000000" in host_platform.signature_problem(program, "0000000000", os_name="macos", run=run)
    unsigned = host_platform.signature_problem(
        program, "3MU9H2V9Y9", os_name="macos", run=lambda argv, env=None: host_platform.Ran(1, "code object is not signed at all"),
    )
    assert "fails `codesign --verify --strict`: code object is not signed at all" in unsigned


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
    team = {"value": "3MU9H2V9Y9"}

    def run(argv, env=None):
        ran.append(argv)
        return host_platform.Ran(0, "" if argv[1] == "--verify" else f"TeamIdentifier={team['value']}\n")

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
            ["codesign", "--verify", "--strict", str(library)], ["codesign", "-dv", str(library)],
            ["codesign", "--verify", "--strict", str(program)], ["codesign", "-dv", str(program)],
        ]

        team["value"] = "0000000000"
        await rig.want(V1)
        state = rig.installer.state()
        assert "signed by team 0000000000, not 3MU9H2V9Y9" in state["error"]
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
        "working", "error", "signature",
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
    with pytest.raises(SettingsError, match="cannot install vllm"):
        Settings(key_hash="x", engine="vllm", install_engines=True).check()
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
