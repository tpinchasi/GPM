"""The agent installs whisper.cpp, builds it here, and switches its versions (D150–D153).

docs/spec/host-agent.md §2.1 and §2.3. Against a fake GitHub — version and build releases, tags
light and annotated, source archives made of a commit, Windows archives with their digests — and
the machine's commands stood in for through the runner the source takes: the compiler checks,
`cmake` (whose build drops a small fake `whisper-server` into the build tree), `nvidia-smi`,
`venv` and pip. The fake server is a small Python program answering `GET /` as the real one does.
Nothing here reaches the network, compiles anything, or needs a GPU.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import signal
import socket
import sys
import tarfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx
import pytest
from gpm_agent import cli, installs, whisper_build
from gpm_agent import platform as host_platform
from gpm_agent.app import create_app
from gpm_agent.engines import EngineRefused
from gpm_agent.installs import WHISPER_CPP, Installer, InstallRefused, Target
from gpm_agent.settings import Settings, SettingsError, fingerprint, load
from gpm_agent.vllm_state import state_dir_for
from gpm_agent.whisper_build import (
    BUILD_TIMEOUT_S,
    CMAKE_VERSION,
    CMAKE_WHEELS,
    WhisperCppSource,
    requirements_text,
)
from gpm_agent.whisper_cpp import InstalledServer, WhisperCppFacts, build_of

AGENT_KEY = "gpmg_" + "c" * 64
REPO = "ggml-org/whisper.cpp"
V1, V2 = "1.9.4", "1.9.5"
C1, C2, C_OTHER = "1" * 40, "2" * 40, "3" * 40
MAC = Target(host_platform.MACOS, "arm64")
LINUX = Target(host_platform.LINUX, "amd64")
WIN = Target(host_platform.WINDOWS, "amd64")
WIN_ARM = Target(host_platform.WINDOWS, "arm64")

HUB_REPO = "someone/whisper-builds"
TAG = f"{HUB_REPO}/ggml-small.bin"
HUB_FILES = {"ggml-small.bin": b"s" * 3000}

#: The fake server. Packed by the fake build behind a small shell wrapper, which names the version
#: it was built as in its environment and runs the fake as `python fake.py <program> <arguments>`,
#: the way the system runs a script — so its command line ends in exactly the argument list the
#: agent started it with, as the real one's is. Told how to behave by a file beside the test, per
#: version.
FAKE_SERVER = '''
import http.server, json, os, sys

version, control_path, starts_path = (os.environ[name] for name in ("FAKE_VERSION", "FAKE_CONTROL", "FAKE_STARTS"))
program, args = sys.argv[1], sys.argv[2:]
try:
    control = json.load(open(control_path)).get(version, {})
except OSError:
    control = {}
if args == ["--help"]:
    print("usage: whisper-server [options]", flush=True)
    sys.exit(control.get("help_code", 0))
with open(starts_path, "a") as starts:
    starts.write(json.dumps({"version": version, "program": program, "args": args, "pid": os.getpid(),
                             "token": os.environ.get("GITHUB_TOKEN"),
                             "env": {k: os.environ.get(k) for k in ("VAST_API_KEY", "SOME_SERVICE_TOKEN", "PATH")}}) + "\\n")
print("whisper_init_from_file_with_params_no_state: loading model", flush=True)
if control.get("exit"):
    print(control["exit"], flush=True)
    sys.exit(3)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<h1>Whisper.cpp Server</h1>")

    def log_message(self, *_):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), Handler).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_tarball(commit: str, version: str, *, names: Optional[str] = None) -> bytes:
    """GitHub's archive of a commit: one top directory, and the commit written as the archive's
    global header — `names` writes another there."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": names or commit}) as archive:
        top = f"ggml-org-whisper.cpp-{commit[:7]}"
        directory = tarfile.TarInfo(top)
        directory.type, directory.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(directory)
        for name, content in (("CMakeLists.txt", b"project(whisper.cpp)\n"), ("VERSION", version.encode())):
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size, info.mode = len(content), 0o644
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def windows_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


WINDOWS_BUILD = {
    "Release/whisper-server.exe": b"MZ whisper-server",
    "Release/whisper.dll": b"MZ whisper",
    "Release/ggml.dll": b"MZ ggml",
}


# --- a fake GitHub ---


class FakeGitHub:
    """whisper.cpp's releases, tags and archives, as GitHub's API describes and serves them."""

    def __init__(self) -> None:
        #: Newest first, as GitHub lists them.
        self.releases: list[dict] = []
        self.refs: dict[str, dict] = {}
        self.tag_objects: dict[str, dict] = {}
        self.tarballs: dict[str, bytes] = {}
        self.files: dict[int, bytes] = {}
        self.latest: Optional[str] = None
        self.requests: list[httpx.Request] = []
        self._next_id = 5000

    def _assets(self, files: dict[str, bytes], digests: Optional[dict[str, Optional[str]]] = None) -> list[dict]:
        assets = []
        for name, content in files.items():
            self._next_id += 1
            self.files[self._next_id] = content
            entry = {"name": name, "id": self._next_id, "size": len(content)}
            digest = (digests or {}).get(name, f"sha256:{sha(content)}")
            if digest is not None:
                entry["digest"] = digest
            assets.append(entry)
        return assets

    def version(self, version: str, commit: str, *, annotated: bool = True, files: Optional[dict] = None,
                prerelease: bool = False, latest: bool = True, archive: Optional[bytes] = None) -> None:
        tag = f"v{version}"
        self.releases.insert(0, {"tag_name": tag, "draft": False, "prerelease": prerelease,
                                 "target_commitish": commit, "assets": self._assets(files or {})})
        if annotated:
            tag_sha = sha(tag.encode())[:40]
            self.refs[tag] = {"ref": f"refs/tags/{tag}", "object": {"type": "tag", "sha": tag_sha}}
            self.tag_objects[tag_sha] = {"tag": tag, "object": {"type": "commit", "sha": commit}}
        else:
            self.refs[tag] = {"ref": f"refs/tags/{tag}", "object": {"type": "commit", "sha": commit}}
        self.tarballs[commit] = archive if archive is not None else source_tarball(commit, version)
        if latest:
            self.latest = tag

    def build_tag(self, tag: str, commit: str, files: dict[str, bytes], *, says: Optional[str] = None,
                  digests: Optional[dict] = None, latest: bool = False, at: int = 0) -> None:
        """A build release, listed at `at` among the newest; `says` is the commit its release names,
        which need not be the one its tag names."""
        self.releases.insert(at, {"tag_name": tag, "draft": False, "prerelease": True,
                                  "target_commitish": says or commit, "assets": self._assets(files, digests)})
        self.refs[tag] = {"ref": f"refs/tags/{tag}", "object": {"type": "commit", "sha": commit}}
        if latest:
            self.latest = tag

    def asked(self, part: str) -> list[httpx.Request]:
        return [request for request in self.requests if part in request.url.path]

    def transport(self) -> httpx.MockTransport:
        def release(tag: str) -> Optional[dict]:
            return next((entry for entry in self.releases if entry["tag_name"] == tag), None)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            path = request.url.path.removeprefix("/api")  # a mirror's own prefix, where one is named
            if request.url.host == "codeload.github.com":
                commit = path.rsplit("/", 1)[1]
                return httpx.Response(200, content=self.tarballs[commit]) if commit in self.tarballs else httpx.Response(404)
            base = f"/repos/{REPO}"
            if path == f"{base}/releases/latest":
                found = release(self.latest) if self.latest else None
                return httpx.Response(200, json=found) if found else httpx.Response(404)
            if path.startswith(f"{base}/releases/tags/"):
                found = release(path.rsplit("/", 1)[1])
                return httpx.Response(200, json=found) if found else httpx.Response(404)
            if path == f"{base}/releases":
                per_page, page = int(request.url.params["per_page"]), int(request.url.params["page"])
                return httpx.Response(200, json=self.releases[(page - 1) * per_page: page * per_page])
            if path.startswith(f"{base}/git/ref/tags/"):
                found = self.refs.get(path.rsplit("/", 1)[1])
                return httpx.Response(200, json=found) if found else httpx.Response(404)
            if path.startswith(f"{base}/git/tags/"):
                found = self.tag_objects.get(path.rsplit("/", 1)[1])
                return httpx.Response(200, json=found) if found else httpx.Response(404)
            if path.startswith(f"{base}/tarball/"):
                # As GitHub does: to codeload, another host, which a token is not sent on to.
                commit = path.rsplit("/", 1)[1]
                return httpx.Response(302, headers={"location": f"https://codeload.github.com/{REPO}/legacy.tar.gz/{commit}"})
            if path.startswith(f"{base}/releases/assets/"):
                assert request.headers["accept"] == "application/octet-stream"
                return httpx.Response(200, content=self.files[int(path.rsplit("/", 1)[1])])
            return httpx.Response(404)

        return httpx.MockTransport(handler)


# --- the machine's commands, stood in for ---


class Commands:
    """What the source runs, answered by program and recorded with how it was asked: `xcode-select`,
    `nvidia-smi`, `venv`, pip, `cmake` — whose build writes a fake `whisper-server` built as the
    version the source says it is — and the program's `--help`."""

    def __init__(self, root: Path, *, xcode: bool = True, driver: Optional[str] = None) -> None:
        self.root = root
        self.xcode, self.driver = xcode, driver
        self.calls: list[tuple[list[str], dict, Optional[dict]]] = []
        self.configure_says: Optional[host_platform.Ran] = None
        self.builds_program = True
        self.help_says: Optional[host_platform.Ran] = None
        self._source: Optional[Path] = None

    def argvs(self) -> list[list[str]]:
        return [argv for argv, _, _ in self.calls]

    async def __call__(self, argv, env=None, **options) -> host_platform.Ran:
        argv = list(argv)
        self.calls.append((argv, options, env))
        ran = host_platform.Ran
        if argv[:2] == ["xcode-select", "-p"]:
            return ran(0, "/Library/Developer/CommandLineTools\n") if self.xcode else ran(
                2, "xcode-select: error: unable to get active developer directory")
        if argv[0] == "nvidia-smi":
            return ran(0, self.driver) if self.driver else ran(None, "[Errno 2] No such file or directory: 'nvidia-smi'")
        if argv[1:4] == ["-I", "-m", "venv"]:
            venv = Path(argv[4])
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("")
            return ran(0, "")
        if argv[1:5] == ["-I", "-m", "pip", "install"]:
            cmake = Path(argv[0]).parent / "cmake"
            cmake.write_text("")
            cmake.chmod(0o755)
            return ran(0, f"Successfully installed cmake-{CMAKE_VERSION}")
        if argv[1:] == ["--version"]:
            return ran(0, f"cmake version {CMAKE_VERSION}")
        if "-S" in argv:
            self._source = Path(argv[argv.index("-S") + 1])
            assert (self._source / "CMakeLists.txt").is_file(), "configured from the unpacked source"
            return self.configure_says or ran(0, "-- Build files have been written")
        if "--build" in argv:
            build = Path(argv[argv.index("--build") + 1])
            if self.builds_program:
                version = (self._source / "VERSION").read_text()
                program = build / "bin" / "whisper-server"
                program.parent.mkdir(parents=True)
                program.write_text(
                    f'#!/bin/sh\nFAKE_VERSION="{version}" FAKE_CONTROL="{self.root / "fake-control.json"}" '
                    f'FAKE_STARTS="{self.root / "fake-starts.jsonl"}"\n'
                    "export FAKE_VERSION FAKE_CONTROL FAKE_STARTS\n"
                    f'exec "{sys.executable}" "{self.root / "fake_whisper_server.py"}" "$0" "$@"\n'
                )
                (build / "bin" / "whisper-cli").write_text("not kept")
            return ran(0, "[100%] Built target whisper-server")
        if argv[1:] == ["--help"]:
            return self.help_says or ran(0, "usage: whisper-server [options]")
        raise AssertionError(f"an unexpected command: {argv}")


def tools(table: dict[str, Optional[str]]):
    return lambda name: table.get(name)


ON_PATH = {"cmake": "/opt/tools/bin/cmake", "c++": "/usr/bin/c++", "make": "/usr/bin/make"}


class Rig:
    """A machine whose agent installs whisper.cpp: engines and models directories, a port, a fake
    GitHub, a fake hub, the machine's commands stood in for, and the agent's own pieces."""

    def __init__(self, root: Path, *, target: Target = MAC, which: Optional[dict] = None,
                 min_free_disk_gb: float = 0.0, free_bytes: Optional[int] = None, **commands) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        (root / "fake_whisper_server.py").write_text(FAKE_SERVER)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.engines = root / "engines"
        self.github = FakeGitHub()
        self.commands = Commands(root, **commands)
        self.which = dict(ON_PATH if which is None else which)
        self.target = target
        self.min_free_disk_gb, self.free_bytes = min_free_disk_gb, free_bytes
        self.now = 1_800_000_000.0
        self.settings = Settings(
            key_hash=fingerprint(AGENT_KEY), engine="whisper-cpp", engine_url=self.url,
            models_path=str(root / "models"), install_engines=True, engines_path=str(self.engines),
        )
        self.settings.check()
        self.clients: list[httpx.AsyncClient] = []
        self.engine_facts: list[WhisperCppFacts] = []
        self.engine, self.installer = self.new_agent()

    def source(self) -> WhisperCppSource:
        return WhisperCppSource(
            WHISPER_CPP, self.target, tools_dir=self.engines / installs.TOOLS, run=self.commands,
            which=tools(self.which), cpu_count=lambda: 12,
        )

    def new_agent(self) -> tuple[WhisperCppFacts, Installer]:
        """The engine and its installer, as a freshly started agent builds them."""
        engine = WhisperCppFacts(self.settings, hub_transport=self.hub())
        client = httpx.AsyncClient(base_url=self.url, timeout=10)
        self.clients.append(client)
        self.engine_facts.append(engine)
        installer = Installer(
            self.source(), self.engines / "whisper-cpp", InstalledServer(engine, client),
            http_transport=self.github.transport(), clock=lambda: self.now, switch_timeout_s=20.0,
            min_free_disk_gb=self.min_free_disk_gb,
            free_disk_bytes=(lambda: self.free_bytes) if self.free_bytes is not None else None,
        )
        engine.use_installer(installer)
        return engine, installer

    def hub(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == f"/api/models/{HUB_REPO}/tree/main":
                return httpx.Response(200, json=[{"type": "file", "path": n, "size": len(b)} for n, b in HUB_FILES.items()])
            name = request.url.path.split("/resolve/main/", 1)[-1]
            return httpx.Response(200, content=HUB_FILES[name]) if name in HUB_FILES else httpx.Response(404)

        return httpx.MockTransport(handler)

    def publish(self, version: str, commit: str, **more) -> None:
        self.github.version(version, commit, **more)

    def behave(self, version: str, **control) -> None:
        path = self.root / "fake-control.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current[version] = control
        path.write_text(json.dumps(current))

    def starts(self) -> list[dict]:
        path = self.root / "fake-starts.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def program(self, version: str) -> Path:
        return self.engines / "whisper-cpp" / version / "bin" / "whisper-server"

    def model_file(self) -> Path:
        build = build_of(TAG)
        return self.root / "models" / build.directory / build.file

    async def want(self, version: str, switch: bool = False) -> dict:
        answer = await self.installer.want(version, switch)
        await self.installer.finished()
        return answer

    async def hold(self, tag: str = TAG) -> None:
        async with httpx.AsyncClient(base_url=self.url, timeout=10) as client:
            async for _ in self.engine.pull(client, tag):
                pass
            await self.engine.hold(client, tag, pinned=True)

    async def describe(self) -> dict:
        async with httpx.AsyncClient(base_url=self.url, timeout=10) as client:
            return await self.engine.describe(client)

    async def close(self) -> None:
        for record in self.starts():
            said = host_platform.command_line(record["pid"]) or b""
            if host_platform.alive(record["pid"]) and b"fake_whisper_server.py" in said:
                try:
                    os.kill(record["pid"], signal.SIGKILL)
                except OSError:
                    pass
        for engine in self.engine_facts:
            if engine._process is not None:
                engine._process.wait(timeout=10)
        for client in self.clients:
            await client.aclose()


@pytest.fixture
async def rig(tmp_path):
    made = Rig(tmp_path)
    yield made
    await made.close()


def staged(rig: Rig, version: str) -> Path:
    return rig.engines / "whisper-cpp" / version


async def stage_directly(rig: Rig, version: str) -> tuple[Path, list[tuple[str, int, Optional[int]]]]:
    """Stage `version` with the source alone, into a fresh directory, recording its progress."""
    steps: list[tuple[str, int, Optional[int]]] = []
    source = rig.source()
    into = rig.engines / "whisper-cpp" / f"{version}.partial"
    into.mkdir(parents=True)
    async with httpx.AsyncClient(transport=rig.github.transport(), follow_redirects=True) as http:
        release = await source.release(http, version)
        await source.stage(http, release, into, lambda step, done, total: steps.append((step, done, total)))
    return into, steps


# --- versions and their commits ---


async def test_latest_is_the_newest_version_even_where_a_build_tag_is_marked_latest(rig):
    rig.publish(V1, C1)
    rig.github.build_tag("b5130", C1, {"whisper-bin-x64.zip": b"x"})
    rig.publish(V2, C2, prerelease=True, latest=False)  # a version marked pre-release is not taken
    rig.github.build_tag("b5454", C2, {"whisper-bin-x64.zip": b"x"}, latest=True)
    async with httpx.AsyncClient(transport=rig.github.transport()) as http:
        assert (await rig.source().latest(http)).version == V1
        rig.github.latest = f"v{V2}"
        assert (await rig.source().latest(http)).version == V2


async def test_a_tag_is_read_to_its_commit_through_an_annotated_tag(rig):
    rig.publish(V1, C1, annotated=True)
    rig.github.build_tag("b5130", C1, {})
    rig.publish(V2, C2, annotated=False)
    rig.github.refs["v9.9.9"] = {"object": {"type": "tree", "sha": "4" * 40}}
    rig.github.refs["v9.9.8"] = {"object": {"type": "commit", "sha": "not a commit"}}
    source = rig.source()
    async with httpx.AsyncClient(transport=rig.github.transport()) as http:
        assert await source.commit_of(http, f"v{V1}") == C1
        assert rig.github.asked("/git/tags/"), "the annotated tag was followed"
        assert await source.commit_of(http, "b5130") == C1
        assert await source.commit_of(http, f"v{V2}") == C2
        with pytest.raises(InstallRefused, match="names a tree"):
            await source.commit_of(http, "v9.9.9")
        with pytest.raises(InstallRefused, match="names no commit"):
            await source.commit_of(http, "v9.9.8")
        before = len(rig.github.requests)
        for odd in ("../v1.9.5", "v1.9.5/../../x", "master", "b12x"):
            with pytest.raises(InstallRefused, match="not a tag"):
                await source.commit_of(http, odd)
        assert len(rig.github.requests) == before, "an odd tag never reaches GitHub"


# --- macOS and Linux: built here ---


async def test_on_macos_the_server_alone_is_built_statically_with_metal_from_the_tags_commit(rig, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "not-a-real-token")
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    rig.publish(V2, C2)
    into, steps = await stage_directly(rig, V2)

    # The source is the commit the tag names, fetched by that commit through the API.
    [tarball] = rig.github.asked(f"/repos/{REPO}/tarball/")
    assert tarball.url.path == f"/repos/{REPO}/tarball/{C2}" and tarball.url.host == "api.github.com"
    assert tarball.headers["authorization"] == "Bearer not-a-real-token"
    [codeload] = [r for r in rig.github.requests if r.url.host == "codeload.github.com"]
    assert "authorization" not in codeload.headers, "the token went on to another host"

    work = rig.engines / "whisper-cpp" / "build"
    source = work / "src" / f"ggml-org-whisper.cpp-{C2[:7]}"
    build = work / "out"
    assert rig.commands.argvs() == [
        ["xcode-select", "-p"],
        ["/opt/tools/bin/cmake", "-S", str(source), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DWHISPER_BUILD_TESTS=OFF",
         "-DWHISPER_BUILD_EXAMPLES=ON", "-DWHISPER_SDL2=OFF", "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"],
        ["/opt/tools/bin/cmake", "--build", str(build), "--config", "Release", "--target", "whisper-server", "-j", "6"],
        [str(into / "bin" / "whisper-server"), "--help"],
    ]
    log = rig.engines / "whisper-cpp" / "logs" / "build.log"
    for _, options, env in rig.commands.calls[1:3]:
        assert options == {"timeout_s": BUILD_TIMEOUT_S, "low_priority": True, "log": log}
        assert "GITHUB_TOKEN" not in env and "HF_TOKEN" not in env
        assert env["GIT_CEILING_DIRECTORIES"] == str(work)
    assert "GITHUB_TOKEN" not in rig.commands.calls[3][2], "the program is checked without the agent's credentials"

    # Only the program is kept; the source and its build are gone.
    assert sorted(str(p.relative_to(into)) for p in into.rglob("*")) == ["bin", "bin/whisper-server"]
    assert os.access(into / "bin" / "whisper-server", os.X_OK)
    assert not work.exists()
    assert C2 in log.read_text()
    assert [step for step, _, _ in steps if step != "downloading"] == ["unpacking", "building"]
    assert steps[0][0] == "downloading" and steps[-3][0] == "downloading" and steps[-3][1] > 0


@pytest.mark.parametrize("on_path, at_cuda_home, flags", [
    ("/usr/bin/nvcc", False, ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_COMPILER=/usr/bin/nvcc"]),
    (None, True, ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_COMPILER={nvcc}"]),
    (None, False, []),
])
async def test_on_linux_cuda_is_built_in_where_its_compiler_is_and_the_cpu_otherwise(
    tmp_path, monkeypatch, on_path, at_cuda_home, flags,
):
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    if at_cuda_home:
        nvcc.parent.mkdir(parents=True)
        nvcc.write_text("")
        nvcc.chmod(0o755)
    monkeypatch.setattr(whisper_build, "CUDA_NVCC", str(nvcc))
    rig = Rig(tmp_path / "machine", target=LINUX, which={**ON_PATH, "nvcc": on_path})
    try:
        rig.publish(V2, C2)
        await stage_directly(rig, V2)
        configure = next(argv for argv in rig.commands.argvs() if "-S" in argv)
        assert configure[-len(flags) or len(configure):] == [flag.format(nvcc=nvcc) for flag in flags]
        assert "-DGGML_METAL=ON" not in configure
        assert ["xcode-select", "-p"] not in rig.commands.argvs(), "Linux has no Xcode to ask about"
        assert "-j" in next(argv for argv in rig.commands.argvs() if "--build" in argv)
    finally:
        await rig.close()


@pytest.mark.parametrize("target, which, xcode, missing", [
    (MAC, ON_PATH, False, "Xcode's command-line tools"),
    (LINUX, {**ON_PATH, "c++": None}, True, "a C++ compiler (c++) and make"),
    (LINUX, {**ON_PATH, "make": None}, True, "a C++ compiler (c++) and make"),
])
async def test_without_a_compiler_a_version_is_refused_naming_prerequisites_before_anything_is_fetched(
    tmp_path, target, which, xcode, missing,
):
    rig = Rig(tmp_path, target=target, which=which, xcode=xcode)
    try:
        rig.publish(V2, C2)
        await rig.want("latest")
        state = rig.installer.state()
        assert state["running"] is None and state["staged"] is None
        assert f"has no {missing}" in state["error"]
        assert "gpm-agent prerequisites" in state["error"]
        assert not rig.github.asked("/tarball/") and not rig.github.asked("/git/ref/")
        assert not any("cmake" in argv[0] for argv in rig.commands.argvs())
    finally:
        await rig.close()


async def test_cmake_from_pypi_is_pinned_to_one_version_and_its_published_digests(rig):
    rig.which["cmake"] = None
    rig.publish(V2, C2)
    await stage_directly(rig, V2)
    venv = rig.engines / ".tools" / f"cmake-{CMAKE_VERSION}"
    argvs = rig.commands.argvs()
    assert argvs[1] == [sys.executable, "-I", "-m", "venv", str(venv)]
    assert argvs[2] == [
        str(venv / "bin" / "python"), "-I", "-m", "pip", "install",
        "--require-hashes", "--only-binary=:all:", "--no-deps", "--no-input", "--disable-pip-version-check",
        "-r", str(venv / "gpm-requirements.txt"),
    ]
    assert argvs[3] == [str(venv / "bin" / "cmake"), "--version"]
    assert argvs[4][0] == str(venv / "bin" / "cmake") and "-S" in argvs[4]
    written = (venv / "gpm-requirements.txt").read_text()
    assert written == requirements_text() and written.startswith(f"cmake=={CMAKE_VERSION} \\\n")
    assert sorted(line.split("sha256:")[1].rstrip(" \\") for line in written.splitlines()[1:]) == sorted(CMAKE_WHEELS.values())
    assert len(CMAKE_WHEELS) == 3 and all(len(digest) == 64 for digest in CMAKE_WHEELS.values())
    assert all(f"-{CMAKE_VERSION}-" in name for name in CMAKE_WHEELS)

    # Kept for the next build, and not fetched again.
    rig.commands.calls.clear()
    rig.publish(V1, C1, latest=False)
    await stage_directly(rig, V1)
    assert not any("venv" in argv or "pip" in argv for argv in rig.commands.argvs())


async def test_without_cmake_on_a_machine_pypi_has_no_wheel_for_it_says_to_install_cmake(tmp_path):
    rig = Rig(tmp_path, target=Target(host_platform.LINUX, None), which={**ON_PATH, "cmake": None})
    try:
        rig.publish(V2, C2)
        await rig.want("latest")
        assert "install cmake" in rig.installer.state()["error"]
        assert not rig.github.asked("/tarball/")
    finally:
        await rig.close()


async def test_a_source_archive_that_names_another_commit_is_not_built(rig):
    rig.publish(V2, C2, archive=source_tarball(C2, V2, names=C_OTHER))
    await rig.want("latest")
    assert f"says it is of commit {C_OTHER}, not {C2}" in rig.installer.state()["error"]
    assert not any("-S" in argv for argv in rig.commands.argvs())
    assert not (rig.engines / "whisper-cpp" / "build").exists()
    assert rig.installer.state()["running"] is None and not staged(rig, V2).exists()


async def test_a_build_that_fails_is_refused_with_its_last_words_and_leaves_nothing(rig):
    rig.publish(V2, C2)
    rig.commands.configure_says = host_platform.Ran(1, "-- The C compiler identification is AppleClang\nCMake Error: no Metal here")
    await rig.want("latest")
    error = rig.installer.state()["error"]
    assert "building whisper.cpp v1.9.5 failed (exit code 1: CMake Error: no Metal here)" in error
    assert str(rig.engines / "whisper-cpp" / "logs" / "build.log") in error
    assert not (rig.engines / "whisper-cpp" / "build").exists()
    assert not any(p.name.startswith(V2) for p in (rig.engines / "whisper-cpp").iterdir())

    rig.commands.configure_says = None
    rig.commands.builds_program = False
    rig.now += installs.RETRY_AFTER_S + 1
    await rig.want("latest")
    assert "made no whisper-server" in rig.installer.state()["error"]


async def test_a_built_program_that_does_not_run_is_not_kept(rig):
    rig.publish(V2, C2)
    rig.commands.help_says = host_platform.Ran(None, "dyld: Library not loaded: libggml.dylib")
    await rig.want("latest")
    assert "built here, does not run: whisper-server --help did not run: dyld: Library not loaded" in rig.installer.state()["error"]
    assert not staged(rig, V2).exists()


async def test_no_credential_reaches_the_build_pip_the_checks_or_the_server(rig, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "not-a-real-key")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "not-a-real-token")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    rig.which["cmake"] = None  # cmake from PyPI: pip runs too
    rig.publish(V1, C1)
    await rig.want("latest")
    await rig.hold(TAG)
    ran = [argv[0] if "-m" not in argv[:3] else " ".join(argv[1:4]) for argv in rig.commands.argvs()]
    assert any("venv" in name for name in ran) and any("pip" in name for name in ran)
    for argv, _, env in rig.commands.calls:
        assert env is not None, f"{argv} was given the agent's whole environment"
        assert "VAST_API_KEY" not in env and "SOME_SERVICE_TOKEN" not in env, argv
        assert env["PATH"] == os.environ["PATH"] and env["CUDA_VISIBLE_DEVICES"] == "0", argv
    [start] = rig.starts()
    assert start["env"]["VAST_API_KEY"] is None and start["env"]["SOME_SERVICE_TOKEN"] is None
    assert start["env"]["PATH"] == os.environ["PATH"]


async def test_a_build_is_not_begun_that_would_leave_less_free_than_the_owners_floor(tmp_path):
    rig = Rig(tmp_path, min_free_disk_gb=10, free_bytes=11 * 10**9)
    try:
        rig.publish(V2, C2)
        await rig.want("latest")
        error = rig.installer.state()["error"]
        assert "whisper-cpp 1.9.5 is not installed: it needs about 2.0 GB on the engines' volume" in error
        assert "a build from its source, estimated at 2.0 GB" in error and "11.0 GB free" in error
        assert "min_free_disk_gb" in error
        assert not rig.github.asked("/tarball/") and not any("-S" in argv for argv in rig.commands.argvs())
        rig.free_bytes = 13 * 10**9
        rig.now += installs.RETRY_AFTER_S + 1
        await rig.want("latest")
        assert rig.installer.state()["running"] == V2
    finally:
        await rig.close()


async def test_on_windows_an_archive_is_not_fetched_that_would_leave_less_free_than_the_floor(tmp_path):
    rig = Rig(tmp_path, target=WIN, min_free_disk_gb=10, free_bytes=10 * 10**9 + 10)
    try:
        rig.publish(V2, C2)
        rig.github.build_tag("b5454", C2, {"whisper-blas-bin-x64.zip": windows_zip(WINDOWS_BUILD)}, at=1)
        await rig.want("latest")
        error = rig.installer.state()["error"]
        assert "whisper-blas-bin-x64.zip" in error and "and up to 3 times that unpacked" in error
        assert "min_free_disk_gb" in error and not rig.github.asked("/releases/assets/")
    finally:
        await rig.close()


async def test_the_source_comes_through_the_api_the_machines_owner_names(rig, monkeypatch):
    monkeypatch.setenv("GPM_GITHUB_API", "https://mirror.example/api")
    rig.publish(V2, C2)
    await stage_directly(rig, V2)
    [tarball] = rig.github.asked("/tarball/")
    assert (tarball.url.host, tarball.url.path) == ("mirror.example", f"/api/repos/{REPO}/tarball/{C2}")


# --- Windows: whisper.cpp's own archives ---


@pytest.mark.parametrize("target, driver, archive", [
    (WIN, "566.36\n", "whisper-bin-win-cuda-12.4.0-x64.zip"),
    (WIN, "550.00\n566.36\n", "whisper-bin-win-cuda-12.4.0-x64.zip"),
    (WIN, "537.13\n", "whisper-bin-win-cuda-11.8.0-x64.zip"),
    (WIN, "452.39\n", "whisper-bin-win-cuda-11.8.0-x64.zip"),
    (WIN, "452.06\n", "whisper-blas-bin-x64.zip"),
    (WIN, None, "whisper-blas-bin-x64.zip"),
    (WIN_ARM, "566.36\n", "whisper-bin-win-cpu-arm64.zip"),
])
async def test_on_windows_the_archive_is_chosen_by_the_card_and_its_driver(tmp_path, target, driver, archive):
    rig = Rig(tmp_path, target=target, driver=driver)
    try:
        rig.publish(V2, C2)
        rig.github.build_tag("b5454", C2, {name: windows_zip(WINDOWS_BUILD) for names in WHISPER_CPP.windows.values()
                                           for name in names[:1]}, at=1)
        into, _ = await stage_directly(rig, V2)
        [download] = rig.github.asked("/releases/assets/")
        chosen = next(entry for entry in rig.github.releases if entry["tag_name"] == "b5454")["assets"]
        assert next(asset["name"] for asset in chosen if str(asset["id"]) == download.url.path.rsplit("/", 1)[1]) == archive
        asked_driver = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"] in rig.commands.argvs()
        assert asked_driver is (target.arch == "amd64")
        assert rig.source().program(into) == into / "Release" / "whisper-server.exe"
        assert rig.commands.argvs()[-1] == [str(into / "Release" / "whisper-server.exe"), "--help"]
        assert not any("cmake" in argv[0] or argv[0] == "xcode-select" for argv in rig.commands.argvs())
    finally:
        await rig.close()


async def test_on_windows_the_build_tag_of_the_same_commit_is_taken_never_its_neighbour(tmp_path):
    rig = Rig(tmp_path, target=WIN)
    try:
        rig.publish(V2, C2)
        blas = {"whisper-blas-bin-x64.zip": windows_zip(WINDOWS_BUILD)}
        # Listed right beside the version, and saying it was made from its commit — but its tag
        # names another. The one of the same commit is further down, saying nothing useful.
        rig.github.build_tag("b5455", C_OTHER, blas, says=C2, at=0)
        rig.github.build_tag("b5454", C2, {"whisper-blas-bin-x64.zip": windows_zip({**WINDOWS_BUILD, "Release/b5454": b""})},
                             says="master", at=2)
        rig.github.build_tag("b5450", C_OTHER, blas, at=3)
        into, _ = await stage_directly(rig, V2)
        assert (into / "Release" / "b5454").exists()
        assert {r.url.path.rsplit("/", 1)[1] for r in rig.github.asked("/git/ref/tags/")} == {f"v{V2}", "b5455", "b5454"}
    finally:
        await rig.close()


async def test_on_windows_a_version_with_no_build_tag_of_its_commit_is_refused(tmp_path):
    rig = Rig(tmp_path, target=WIN)
    try:
        rig.publish(V2, C2)
        rig.github.build_tag("b5454", C_OTHER, {"whisper-blas-bin-x64.zip": windows_zip(WINDOWS_BUILD)}, at=1)
        await rig.want("latest")
        error = rig.installer.state()["error"]
        assert f"no build tag released with it names its commit {C2[:12]}" in error
        assert not rig.github.asked("/releases/assets/")
    finally:
        await rig.close()


async def test_on_windows_an_older_version_carrying_its_own_archives_needs_no_build_tag(tmp_path):
    rig = Rig(tmp_path, target=WIN, driver="566.36")
    try:
        # 1.9.2 and before: the version's own release carries them, under their older names.
        rig.publish(V1, C1, files={"whisper-cublas-12.4.0-bin-x64.zip": windows_zip(WINDOWS_BUILD),
                                   "whisper-blas-bin-x64.zip": windows_zip(WINDOWS_BUILD)})
        into, _ = await stage_directly(rig, V1)
        assert (into / "Release" / "whisper-server.exe").is_file()
        assert not rig.github.asked("/git/ref/") and not any("page" in str(r.url) for r in rig.github.requests)
    finally:
        await rig.close()


@pytest.mark.parametrize("files, digests, says", [
    ({"Release/whisper.dll": b"MZ"}, {}, "holds no whisper-server.exe"),
    ({"a/whisper-server.exe": b"MZ", "b/whisper-server.exe": b"MZ"}, {}, "holds 2 whisper-server.exe"),
    (WINDOWS_BUILD, {"whisper-blas-bin-x64.zip": None}, "publishes no digest"),
    (WINDOWS_BUILD, {"whisper-blas-bin-x64.zip": "sha256:" + "0" * 64}, "does not match the digest"),
])
async def test_on_windows_an_archive_is_checked_and_must_hold_one_server(tmp_path, files, digests, says):
    rig = Rig(tmp_path, target=WIN)
    try:
        rig.publish(V2, C2)
        rig.github.build_tag("b5454", C2, {"whisper-blas-bin-x64.zip": windows_zip(files)}, digests=digests, at=1)
        await rig.want("latest")
        assert says in rig.installer.state()["error"]
        assert rig.installer.state()["running"] is None and not staged(rig, V2).exists()
        assert rig.installer.state()["signature"] == "not checked"
    finally:
        await rig.close()


# --- run by the agent: a version switch keeps the file ---


async def test_until_a_file_is_held_a_version_is_checked_by_running_it_and_nothing_is_started(rig):
    rig.publish(V2, C2)
    await rig.want("latest")
    state = rig.installer.state()
    assert (state["running"], state["error"], state["engine"]) == (V2, None, "whisper-cpp")
    assert rig.starts() == [], "nothing to serve, so nothing started"
    described = await rig.describe()
    assert (described["binary"], described["version"]) == (str(rig.program(V2)), V2)
    assert described["models_failed"] == {} and described["answers"] is False


async def test_a_version_whose_program_does_not_run_is_not_switched_to(rig):
    rig.publish(V2, C2)
    rig.behave(V2, help_code=1)
    await rig.want("latest")
    state = rig.installer.state()
    assert state["running"] is None and V2 in rig.installer.record.failed
    assert "whisper-server --help exited with code 1" in state["error"]


async def test_a_switch_restarts_the_server_with_the_new_program_and_the_same_model_file(rig, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "not-a-real-token")
    rig.publish(V1, C1)
    await rig.want("latest")
    await rig.hold(TAG)
    [first] = rig.starts()
    assert first["program"] == str(rig.program(V1)) and first["version"] == V1
    assert first["args"][:2] == ["-m", str(rig.model_file())] and first["token"] is None
    # The same command as the owner's program is started with: its uploads in the agent's own
    # directory, readable by this user alone.
    uploads = state_dir_for(rig.root / "models") / "tmp"
    assert first["args"][-4:] == ["--tmp-dir", str(uploads), "-l", "auto"]
    assert uploads.stat().st_mode & 0o777 == 0o700

    rig.publish(V2, C2)
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.want("latest")
    assert rig.installer.state()["staged"] == V2 and len(rig.starts()) == 1, "staged while V1 serves"
    (uploads / "whisper-server-left.wav").write_bytes(b"a recording ffmpeg could not read")
    await rig.want(V2, switch=True)

    state = rig.installer.state()
    assert (state["running"], state["previous"], state["staged"], state["error"]) == (V2, V1, None, None)
    first, second = rig.starts()
    assert second["program"] == str(rig.program(V2)) and second["args"] == first["args"]
    assert list(uploads.iterdir()) == [], "emptied around the restart"
    assert not host_platform.alive(first["pid"]) or b"fake_whisper_server" not in (host_platform.command_line(first["pid"]) or b"")
    described = await rig.describe()
    assert described["models_loaded"] == [TAG] and described["version"] == V2 and described["models_failed"] == {}
    assert rig.engine._current()["version"] == V2

    # Holding the build again on the next pass leaves the switched-to server alone.
    await rig.hold(TAG)
    assert len(rig.starts()) == 2


async def test_a_version_that_will_not_serve_the_file_is_undone_with_the_same_file(rig):
    rig.publish(V1, C1)
    await rig.want("latest")
    await rig.hold(TAG)
    rig.publish(V2, C2)
    rig.behave(V2, exit="error: failed to initialize whisper context")
    rig.now += installs.LATEST_EVERY_S + 1
    await rig.want("latest")
    await rig.want(V2, switch=True)
    state = rig.installer.state()
    assert state["running"] == V1 and "failed to initialize whisper context" in state["error"] and "1.9.4 runs again" in state["error"]
    assert [start["version"] for start in rig.starts()] == [V1, V2, V1]
    assert len({tuple(start["args"]) for start in rig.starts()}) == 1, "every start served the same file"
    assert (await rig.describe())["models_loaded"] == [TAG]


async def test_while_the_installer_starts_the_server_holding_a_build_leaves_it_to_the_installer(rig):
    rig.publish(V1, C1)
    await rig.want("latest")
    async with rig.installer._lock:
        await rig.hold(TAG)
    assert rig.starts() == []
    await rig.hold(TAG)
    assert len(rig.starts()) == 1


async def test_a_server_that_stops_is_started_again_with_its_file_by_the_installer(rig):
    rig.publish(V1, C1)
    await rig.want("latest")
    await rig.hold(TAG)
    pid = rig.engine._current()["pid"]
    os.kill(pid, signal.SIGKILL)
    rig.engine._process.wait(timeout=10)
    assert TAG in (await rig.describe())["models_failed"]
    await rig.installer.ensure_running()
    assert [start["args"] for start in rig.starts()][-1] == rig.starts()[0]["args"]
    assert (await rig.describe())["models_loaded"] == [TAG]


async def test_the_server_outlives_the_agent_and_is_known_again_with_its_version(rig):
    rig.publish(V1, C1)
    await rig.want("latest")
    await rig.hold(TAG)
    pid = rig.engine._current()["pid"]
    threading.Thread(target=rig.engine._process.wait, daemon=True).start()
    rig.engine, rig.installer = rig.new_agent()
    assert rig.installer.server.ours() and rig.installer.server.running_version() == V1
    await rig.installer.ensure_running()
    await rig.hold(TAG)
    assert len(rig.starts()) == 1 and rig.engine._current()["pid"] == pid


async def test_before_any_version_is_installed_holding_a_build_says_so(rig):
    async with httpx.AsyncClient(base_url=rig.url) as client:
        async for _ in rig.engine.pull(client, TAG):
            pass
        with pytest.raises(EngineRefused, match="not installed here yet"):
            await rig.engine.hold(client, TAG, pinned=True)
    assert (await rig.describe())["binary"] is None and rig.starts() == []


async def test_something_else_on_the_port_refuses_installs_and_is_never_touched(rig):
    class Other(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    other = ThreadingHTTPServer(("127.0.0.1", rig.port), Other)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    try:
        rig.publish(V1, C1)
        with pytest.raises(installs.Refused) as refused:
            await rig.installer.want("latest", False)
        assert refused.value.error == "foreign_engine" and "whisper-server you run yourself" in refused.value.detail
    finally:
        other.shutdown()
        other.server_close()


# --- the agent: settings, wiring, facts ---


AGENT_HEADERS = {"Authorization": f"Bearer {AGENT_KEY}"}


async def test_the_facts_say_where_whisper_cpps_versions_stand(rig):
    rig.publish(V1, C1)
    app = create_app(rig.settings, installer=rig.installer)
    assert app.state.work.engine is rig.engine, "the model work and the installer drive one server"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=60) as http:
        answer = await http.put("/agent/v1/engine/version", json={"version": "latest", "switch": False}, headers=AGENT_HEADERS)
        assert answer.status_code == 200 and answer.json()["engine"] == "whisper-cpp"
        await rig.installer.finished()
        facts = (await http.get("/agent/v1/facts", headers=AGENT_HEADERS)).json()
    install = facts["engine"]["install"]
    assert (install["engine"], install["running"], install["signature"]) == ("whisper-cpp", V1, "not checked")
    assert facts["engine"]["foreign"] is False and facts["engine"]["binary"] == str(rig.program(V1))


async def test_an_agent_set_to_install_whisper_cpp_builds_its_installer_around_its_own_engine(tmp_path):
    settings = Settings(
        key_hash=fingerprint(AGENT_KEY), engine="whisper-cpp", engine_url=f"http://127.0.0.1:{free_port()}",
        models_path=str(tmp_path / "models"), install_engines=True, engines_path=str(tmp_path / "engines"),
    )
    app = create_app(settings)
    installer = app.state.installer
    assert isinstance(installer.server, InstalledServer) and installer.server.engine is app.state.work.engine
    assert app.state.work.engine.installer is installer
    assert isinstance(installer.source, WhisperCppSource)
    assert installer.source.tools_dir == tmp_path / "engines" / ".tools"
    assert installer.engine_dir == tmp_path / "engines" / "whisper-cpp" / str(urlsplit(settings.engine_url).port)

    plain = create_app(Settings(key_hash="x", engine="whisper-cpp", engine_url=f"http://127.0.0.1:{free_port()}"))
    assert plain.state.installer is None and plain.state.work.engine.installer is None


def test_install_engines_takes_whisper_cpp_and_refuses_the_owners_program_beside_it():
    Settings(key_hash="x", engine="whisper-cpp", engine_url="http://127.0.0.1:8178", install_engines=True).check()
    with pytest.raises(SettingsError, match="engine_binary"):
        Settings(key_hash="x", engine="whisper-cpp", engine_url="http://127.0.0.1:8178", install_engines=True,
                 engine_binary="/opt/whisper/bin/whisper-server").check()
    with pytest.raises(SettingsError, match="D148|D150"):
        Settings(key_hash="x", engine="whisper-cpp", engine_url="http://10.0.0.5:8178", install_engines=True).check()


def test_init_with_install_engines_for_whisper_cpp_says_nothing_of_path(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    path = tmp_path / "agent.json"
    assert cli.main(["-c", str(path), "init", "--engine", "whisper-cpp", "--install-engines"]) == 0
    assert load(path).install_engines is True and load(path).engine_binary is None
    out = capsys.readouterr().out
    assert "is not on this PATH" not in out and "This agent installs whisper-cpp" in out
    refused = tmp_path / "refused.json"
    assert cli.main(["-c", str(refused), "init", "--engine", "whisper-cpp", "--install-engines",
                     "--engine-binary", "/opt/whisper/bin/whisper-server"]) == 2
    assert not refused.exists()


# --- the platform's runner (D152, D154) ---


async def test_a_command_run_at_low_priority_runs_below_the_agent():
    said = await host_platform.run_command(
        [sys.executable, "-I", "-c", "import os; print(os.nice(0))"], low_priority=True,
    )
    assert said.code == 0 and int(said.output.strip()) == min(19, os.nice(0) + host_platform.LOW_PRIORITY)
    plain = await host_platform.run_command([sys.executable, "-I", "-c", "import os; print(os.nice(0))"])
    assert int(plain.output.strip()) == os.nice(0)


async def test_a_commands_output_goes_to_its_log_and_its_last_words_come_back(tmp_path):
    log = tmp_path / "logs" / "build.log"
    said = await host_platform.run_command(
        [sys.executable, "-I", "-c", "import sys; print('configuring', flush=True); print('CMake Error: x', file=sys.stderr); sys.exit(1)"],
        log=log,
    )
    assert said.code == 1 and said.output.splitlines() == ["configuring", "CMake Error: x"]
    assert log.read_text().splitlines()[1:] == ["configuring", "CMake Error: x"]
    assert (await host_platform.run_command([str(tmp_path / "no-such-program")])).code is None


async def test_a_command_given_up_on_ends_with_everything_it_started(tmp_path):
    child_file = tmp_path / "child"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(child_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    task = asyncio.create_task(host_platform.run_command([sys.executable, "-I", "-c", script], poll_s=0.05))
    deadline = time.monotonic() + 10
    while not (child_file.exists() and child_file.read_text()):
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    child = int(child_file.read_text())
    deadline = time.monotonic() + 10
    while host_platform.alive(child):
        assert time.monotonic() < deadline, "the build's own children outlived it"
        await asyncio.sleep(0.05)

    timed_out = await host_platform.run_command([sys.executable, "-I", "-c", "import time; time.sleep(30)"], timeout_s=0.3, poll_s=0.05)
    assert timed_out.code is None and "did not finish within 0s" in timed_out.output


def test_on_windows_a_low_priority_command_is_started_below_normal_in_its_own_group():
    seen = []

    class Done:
        pid, returncode = 4242, 0

        def poll(self):
            return 0

    def popen(argv, **options):
        seen.append((argv, options))
        return Done()

    said = asyncio.run(host_platform.run_command(["whisper-server.exe", "--help"], low_priority=True,
                                                 os_name="windows", popen=popen))
    [(argv, options)] = seen
    assert said.code == 0 and argv == ["whisper-server.exe", "--help"], "no `nice` on Windows"
    assert options["creationflags"] == host_platform.WINDOWS_START_FLAGS | host_platform.BELOW_NORMAL_PRIORITY_CLASS
    assert "start_new_session" not in options
