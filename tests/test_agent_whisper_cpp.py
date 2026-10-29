"""The agent runs whisper.cpp's server itself (D148): fetches one build, starts the program on
loopback with it, restarts it when the build changes or it stops, and reports what it serves.

docs/spec/host-agent.md §2.1. Against a fake hub and a fake `whisper-server` — a small Python
program that answers `GET /` the way the real one does once its model is loaded. Nothing here
reaches the network, needs a GPU, or runs whisper.cpp.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from gpm_agent import cli, modelhub, whisper_cpp
from gpm_agent import platform as host_platform
from gpm_agent.engines import EngineRefused, engine_facts
from gpm_agent.models import ModelWork, Refused
from gpm_agent.settings import Settings, SettingsError, load
from gpm_agent.vllm_state import state_dir_for
from gpm_agent.whisper_cpp import (
    INFERENCE_PATH,
    WhisperCppFacts,
    build_in,
    build_of,
    installed_version,
    server_argv,
)

REPO = "someone/whisper-builds"
TAG = f"{REPO}/ggml-small.bin"
OTHER = f"{REPO}/ggml-small-q5_0.bin"
#: The repository as the hub lists it: two sizes of one model side by side, and other things.
FILES = {
    "ggml-small.bin": b"s" * 3000,
    "ggml-small-q5_0.bin": b"q" * 1000,
    "model.safetensors": b"w" * 5000,
    "README.md": b"# builds",
    "nested/ggml-deep.bin": b"d" * 10,
}

#: The fake program. Started by the agent exactly as the real one is; told how to behave by a
#: file beside it, and records each start — its arguments, where it ran, what it was given. Run
#: as `python fake.py <program> <arguments>`, the way the system runs a script, so its command
#: line ends in exactly the argument list the agent started it with.
FAKE_SERVER = '''
import http.server, json, os, signal, sys, time

root = os.path.dirname(os.path.dirname(sys.argv[1]))
control_path, starts_path = os.path.join(root, "fake-control.json"), os.path.join(root, "fake-starts.jsonl")
args = sys.argv[2:]
try:
    control = json.load(open(control_path))
except OSError:
    control = {}
with open(starts_path, "a") as starts:
    starts.write(json.dumps({"args": args, "cwd": os.getcwd(), "pid": os.getpid(),
                             "hub_token": os.environ.get("HF_TOKEN")}) + "\\n")
print("whisper_init_from_file_with_params_no_state: loading model from", args[args.index("-m") + 1], flush=True)
if control.get("exit"):
    print(control["exit"], flush=True)
    sys.exit(control.get("code", 3))


def die(*_):
    print(control.get("last_words", "error: the fake server lost its device"), flush=True)
    os._exit(1)


signal.signal(signal.SIGUSR1, die)
started = time.monotonic()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ready = time.monotonic() - started >= control.get("delay", 0)
        self.send_response(200 if ready else 503)
        self.end_headers()
        self.wfile.write(b"<h1>Whisper.cpp Server</h1>" if ready else b"loading")

    def do_POST(self):
        """As whisper.cpp 1.9.2 does with a recording ffmpeg cannot read: it is written to
        --tmp-dir, the request and the recording's own tags are printed, and the file is left
        behind with a 500."""
        name = self.headers.get("x-recording-name", "recording.mp3")
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        print(f"Received request: {name}", flush=True)
        kept = os.path.join(args[args.index("--tmp-dir") + 1], f"whisper-server-{time.monotonic_ns()}.wav")
        with open(kept, "wb") as recording:
            recording.write(body)
        print(f"Running whisper.cpp inference on {name}", flush=True)
        print(f"operator(): processing '{name}' (16000 samples, 1.0 sec), 4 threads, lang = auto ...", flush=True)
        print(f"  Metadata:\\n    title           : {name}", flush=True)
        self.send_response(500)
        self.end_headers()
        self.wfile.write(b'{"error":"FFmpeg conversion failed."}')

    def log_message(self, *_):
        pass


port = int(args[args.index("--port") + 1])
http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Machine:
    """A machine with a models directory, a fake `whisper-server` and an agent's engine."""

    def __init__(self, tmp_path: Path, *, binary: str | None = "fake", **settings):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.root = tmp_path
        self.port = free_port()
        self.control = tmp_path / "fake-control.json"
        self.starts = tmp_path / "fake-starts.jsonl"
        program = tmp_path / "bin" / "whisper-server"
        program.parent.mkdir(exist_ok=True)
        (tmp_path / "fake_whisper_server.py").write_text(FAKE_SERVER)
        # A shell wrapper, so the fake runs under this interpreter whatever its path's length.
        self.wrap(program)
        self.program = str(program)
        self.settings = Settings(
            key_hash="x", engine="whisper-cpp", engine_url=f"http://127.0.0.1:{self.port}",
            models_path=str(tmp_path / "models"),
            engine_binary=self.program if binary == "fake" else binary,
            **settings,
        )
        self.settings.check()
        self.hub_requests: list[str] = []
        self.engines: list[WhisperCppFacts] = []
        self.engine = self.new_engine()

    def wrap(self, program: Path, *, replaced: bool = True) -> None:
        """The program, as a shell script running the fake: replaced by it (`exec`), or as a
        wrapper that stays, with the fake its child — as an owner's own wrapper script might."""
        program.write_text(
            f'#!/bin/sh\n{"exec " if replaced else ""}"{sys.executable}" '
            f'"{self.root / "fake_whisper_server.py"}" "$0" "$@"\n'
        )
        program.chmod(0o755)

    def new_engine(self) -> WhisperCppFacts:
        """The engine as a freshly started agent builds it."""
        engine = WhisperCppFacts(self.settings, hub_transport=self.hub())
        self.engines.append(engine)
        return engine

    def hub(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.hub_requests.append(request.url.path)
            if request.url.path == f"/api/models/{REPO}/tree/main":
                return httpx.Response(200, json=[
                    {"type": "file", "path": name, "size": len(body)} for name, body in FILES.items()
                ])
            name = request.url.path.split("/resolve/main/", 1)[-1]
            if name not in FILES:
                return httpx.Response(404)
            return httpx.Response(200, content=FILES[name])

        return httpx.MockTransport(handler)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.settings.engine_url, timeout=10)

    def behave(self, **control) -> None:
        self.control.write_text(json.dumps(control))

    def started(self) -> list[dict]:
        if not self.starts.exists():
            return []
        return [json.loads(line) for line in self.starts.read_text().splitlines()]

    def file_of(self, tag: str) -> Path:
        build = build_of(tag)
        return self.root / "models" / build.directory / build.file

    async def fetch(self, *tags: str) -> None:
        async with self.client() as client:
            for tag in tags:
                async for _ in self.engine.pull(client, tag):
                    pass

    def stop_everything(self) -> None:
        for record in self.started():
            try:
                os.kill(record["pid"], signal.SIGKILL)
            except OSError:
                pass
        for engine in self.engines:
            if engine._process is not None:
                engine._process.wait(timeout=10)


@pytest.fixture
def machine(tmp_path):
    made = Machine(tmp_path)
    yield made
    made.stop_everything()


async def wait_until(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.05)


# --- what a build may be called ---


@pytest.mark.parametrize("tag", [
    "../../etc/passwd/x.bin",
    "someone/../ggml-small.bin",
    "someone/whisper-builds",                     # two parts: a repository, no file
    "someone/whisper-builds/nested/ggml-deep.bin",  # four: a file below the root
    "someone/whisper-builds/ggml-small.gguf",     # not whisper.cpp's format
    "someone/whisper-builds/.hidden.bin",
    "someone/whisper-builds/ggml small.bin",
    "someone/whisper-builds/ggml-small.bin:latest",
    "/someone/whisper-builds/ggml-small.bin",
    "some__one/whisper-builds/ggml-small.bin",    # could not be read back from its directory
    "someone_/whisper-builds/ggml-small.bin",     # … nor could a part that runs into the separator
    "someone/whisper-builds_/ggml-small.bin",
    "someone/_whisper-builds/ggml-small.bin",
    "someone/whisper-builds/_ggml-small.bin",
    "someone/whisper-builds/.bin",
    "",
])
def test_a_tag_that_is_not_one_file_in_one_repository_is_refused(tag):
    """The tag becomes a directory and a file on this machine. The pool says which build, never
    where it goes — anything not shaped `owner/repo/file.bin` is refused, saying what is (D40)."""
    with pytest.raises(EngineRefused, match="owner/repo/file.bin"):
        build_of(tag)


def test_a_build_lands_in_a_directory_of_its_own_named_after_it():
    build = build_of(TAG)
    assert (build.repo, build.file, build.directory) == (
        REPO, "ggml-small.bin", "someone__whisper-builds__ggml-small.bin",
    )


@pytest.mark.parametrize("tag", [
    TAG, OTHER, "a/b/c.bin", "some.one-1/whisper_builds.v2/ggml_small-q5_0.bin", "x/y/z_.bin", "x./y-/z.bin",
])
def test_every_build_accepted_reads_back_from_its_directory(tag):
    """What is on disk is reported by reading the directory's name back as a tag. A build that
    could be fetched and never read back would be fetched again on every pass."""
    assert build_in(build_of(tag).directory).tag == tag


async def test_a_refused_tag_never_reaches_the_hub(machine):
    async with machine.client() as client:
        with pytest.raises(EngineRefused):
            async for _ in machine.engine.pull(client, "someone/whisper-builds/../../x.bin"):
                pass
    assert machine.hub_requests == []


# --- fetching one file ---


async def test_one_file_is_fetched_into_its_own_directory_and_marked_complete(machine):
    seen = []
    async with machine.client() as client:
        async for done, total in machine.engine.pull(client, TAG):
            seen.append((done, total))
    directory = machine.root / "models" / "someone__whisper-builds__ggml-small.bin"
    # The one file asked for, and nothing else the repository holds — not the other size beside it.
    assert sorted(p.name for p in directory.iterdir()) == sorted([".gpm-complete", "ggml-small.bin"])
    assert (directory / "ggml-small.bin").read_bytes() == FILES["ggml-small.bin"]
    assert modelhub.is_complete(directory)
    assert seen and all(total == 3000 for _, total in seen) and seen[-1][0] == 3000
    async with machine.client() as client:
        described = await machine.engine.describe(client)
    assert described["models_on_disk"] == [{"tag": TAG, "size_bytes": 3000}]


async def test_a_file_the_repository_does_not_list_at_its_root_is_refused(machine):
    async with machine.client() as client:
        with pytest.raises(EngineRefused, match=f"the hub's listing of {REPO} has no ggml-large.bin"):
            async for _ in machine.engine.pull(client, f"{REPO}/ggml-large.bin"):
                pass
        # A file of that name below the root is not the file at the root.
        with pytest.raises(EngineRefused, match="has no ggml-deep.bin"):
            async for _ in machine.engine.pull(client, f"{REPO}/ggml-deep.bin"):
                pass
    assert not (machine.root / "models").exists() or not any((machine.root / "models").iterdir())


async def test_the_hubs_fetch_takes_a_chosen_file_into_a_chosen_directory_and_nowhere_else(tmp_path):
    machine = Machine(tmp_path)
    async with httpx.AsyncClient(transport=machine.hub()) as hub:
        for escape in ("../outside", "a/b", ".hidden", ""):
            with pytest.raises(modelhub.HubRefused, match="not a directory name"):
                async for _ in modelhub.fetch(REPO, tmp_path / "models", client=hub,
                                              only=["ggml-small.bin"], into=escape):
                    pass
        async for _ in modelhub.fetch(REPO, tmp_path / "models", client=hub,
                                      only=["ggml-small-q5_0.bin"], into="chosen"):
            pass
    assert (tmp_path / "models" / "chosen" / "ggml-small-q5_0.bin").read_bytes() == FILES["ggml-small-q5_0.bin"]
    assert not (tmp_path / "outside").exists()


# --- running the server ---


async def test_holding_a_build_starts_the_server_with_exactly_this_command(machine):
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        described = await machine.engine.describe(client)

    [start] = machine.started()
    model_file = machine.file_of(TAG)
    state = state_dir_for(machine.root / "models")
    uploads = state / "tmp"
    assert start["args"] == [
        "-m", str(model_file), "--host", "127.0.0.1", "--port", str(machine.port),
        "--inference-path", "/v1/audio/transcriptions", "--convert", "--tmp-dir", str(uploads),
        "-l", "auto",
    ]
    assert machine.engine._current()["argv"] == [machine.program, *start["args"]]
    assert server_argv(machine.program, model_file, machine.port, uploads) == [machine.program, *start["args"]]
    # Its state — the record, its log, the directory it works in — beside the models, never in them.
    assert Path(start["cwd"]).resolve() == uploads.resolve()
    assert (state / "whisper-server.json").is_file() and (state / "logs" / "whisper-server.log").is_file()
    assert not any(p.name.startswith(("whisper-server", "logs")) for p in (machine.root / "models").rglob("*"))
    assert described["answers"] is True and described["models_loaded"] == [TAG]


async def test_the_hubs_credential_is_not_handed_to_the_server(machine, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
    assert machine.started()[0]["hub_token"] is None


async def test_holding_again_leaves_a_running_server_alone(machine):
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        pid = machine.engine._current()["pid"]
        await machine.engine.hold(client, TAG, pinned=True)
        await machine.engine.hold(client, TAG, pinned=True)
    assert len(machine.started()) == 1
    assert machine.engine._current()["pid"] == pid


async def test_a_changed_build_starts_the_server_again_with_the_new_file(machine):
    await machine.fetch(TAG, OTHER)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        first = machine.engine._process
        await machine.engine.hold(client, OTHER, pinned=True)
        described = await machine.engine.describe(client)

    starts = machine.started()
    assert [s["args"][1] for s in starts] == [str(machine.file_of(TAG)), str(machine.file_of(OTHER))]
    assert first.poll() is not None, "the first server was stopped before the second started"
    assert described["models_loaded"] == [OTHER] and described["models_failed"] == {}


async def test_a_server_that_takes_its_time_to_load_is_waited_for(machine):
    machine.behave(delay=1.0)  # 503 until its model is loaded, as the real one answers
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        assert (await client.get("/")).status_code == 200


async def test_a_server_that_exits_as_it_starts_is_refused_with_its_own_last_words(machine):
    machine.behave(exit="error: failed to initialize whisper context", code=3)
    await machine.fetch(TAG)
    async with machine.client() as client:
        with pytest.raises(EngineRefused, match="exited with code 3: error: failed to initialize whisper context"):
            await machine.engine.hold(client, TAG, pinned=True)
        described = await machine.engine.describe(client)
    assert described["models_loaded"] == []
    assert "failed to initialize whisper context" in described["models_failed"][TAG]


async def test_without_ffmpeg_the_server_is_still_started_and_the_facts_say_so(machine, monkeypatch):
    """whisper.cpp 1.9.2 started with --convert ends at once when it cannot run ffmpeg. The
    agent does not decide that for it; it says ffmpeg is missing, beside the server's own words."""
    monkeypatch.setenv("PATH", str(machine.root / "empty"))
    machine.behave(exit="ffmpeg is not found. Please ensure that ffmpeg is installed", code=0)
    await machine.fetch(TAG)
    async with machine.client() as client:
        with pytest.raises(EngineRefused, match="ffmpeg is not found.*ffmpeg is not on this agent's PATH"):
            await machine.engine.hold(client, TAG, pinned=True)
        described = await machine.engine.describe(client)
    assert len(machine.started()) == 1
    assert described["ffmpeg"] is False


async def test_a_build_not_on_disk_is_not_started(machine):
    async with machine.client() as client:
        with pytest.raises(EngineRefused, match="not on disk"):
            await machine.engine.hold(client, TAG, pinned=True)
    assert machine.started() == []


async def test_a_missing_program_says_how_to_install_it_or_where_to_point(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    on_path = Machine(tmp_path / "on-path", binary=None)
    named = Machine(tmp_path / "named", binary=str(tmp_path / "nowhere" / "whisper-server"))
    try:
        for machine, says in ((on_path, "install whisper.cpp"), (named, "engine_binary")):
            await machine.fetch(TAG)
            async with machine.client() as client:
                with pytest.raises(EngineRefused, match=says):
                    await machine.engine.hold(client, TAG, pinned=True)
                assert (await machine.engine.describe(client))["binary"] is None
        assert on_path.started() == [] and named.started() == []
    finally:
        on_path.stop_everything()
        named.stop_everything()


async def test_something_else_on_the_port_is_named_rather_than_mistaken_for_the_server(machine):
    class Other(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    other = ThreadingHTTPServer(("127.0.0.1", machine.port), Other)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    try:
        await machine.fetch(TAG)
        async with machine.client() as client:
            with pytest.raises(EngineRefused, match="something else already answers"):
                await machine.engine.hold(client, TAG, pinned=True)
            described = await machine.engine.describe(client)
    finally:
        other.shutdown()
        other.server_close()
    assert machine.started() == []
    assert described["answers"] is True and described["models_loaded"] == []


async def test_releasing_stops_the_server_only_if_it_is_serving_that_build(machine):
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        process = machine.engine._process
        await machine.engine.hold(client, OTHER, pinned=False)
        assert process.poll() is None, "another build's release touches nothing"
        await machine.engine.hold(client, TAG, pinned=False)
        described = await machine.engine.describe(client)
    assert process.poll() is not None
    # Stopped on purpose: not a failure.
    assert described["models_loaded"] == [] and described["models_failed"] == {}


async def test_deleting_a_build_stops_the_server_serving_it_first(machine):
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        process = machine.engine._process
        await machine.engine.delete(client, TAG)
        assert process.poll() is not None
        assert not machine.file_of(TAG).parent.exists()
        assert (await machine.engine.describe(client))["models_on_disk"] == []
        await machine.engine.delete(client, TAG)  # a build already gone is not an error
        with pytest.raises(EngineRefused):
            await machine.engine.delete(client, "../../etc")


async def test_a_server_started_before_the_agent_restarted_is_still_its_own(machine):
    """It runs in its own session, so it outlives the agent; the agent that comes back knows it
    by its record, checks by its command line that it is that server, and stops it before
    starting another."""
    await machine.fetch(TAG, OTHER)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        old_pid = machine.engine._current()["pid"]

        # The agent that started it is gone; what reaps the server when it ends is no longer the
        # agent but the system — here, a thread standing in for it.
        threading.Thread(target=machine.engine._process.wait, daemon=True).start()
        restarted = machine.new_engine()
        assert (await restarted.describe(client))["models_loaded"] == [TAG]
        await restarted.hold(client, TAG, pinned=True)
        assert len(machine.started()) == 1, "the running server is adopted, not started again"

        await restarted.hold(client, OTHER, pinned=True)
    await wait_until(lambda: machine.engine._process.poll() is not None)
    assert [s["args"][1] for s in machine.started()][-1] == str(machine.file_of(OTHER))
    assert restarted._current()["pid"] != old_pid


async def test_a_recorded_process_that_is_not_the_server_is_never_signalled(machine):
    """A process number from a file is only a number: the server may have ended, and its number
    gone to something else."""
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        state = state_dir_for(machine.root / "models")
        state.mkdir(parents=True)
        (state / "whisper-server.json").write_text(json.dumps({
            "pid": bystander.pid, "tag": TAG, "file": str(machine.file_of(TAG)),
            "log": str(state / "nothing.log"), "log_offset": 0, "started_at": time.time(),
        }))
        await machine.fetch(TAG)
        engine = machine.new_engine()
        async with machine.client() as client:
            described = await engine.describe(client)
            assert described["models_loaded"] == [] and TAG in described["models_failed"]
            await engine.hold(client, TAG, pinned=True)
        assert bystander.poll() is None, "the bystander is still running"
        assert len(machine.started()) == 1
    finally:
        bystander.kill()
        bystander.wait()


async def test_a_process_whose_command_line_only_mentions_the_server_is_not_it(machine):
    """Ours runs the argument list recorded for it, exactly — not a command line that happens to
    hold the build's file and the transcription path among its words."""
    await machine.fetch(TAG)
    state = state_dir_for(machine.root / "models")
    argv = server_argv(machine.program, machine.file_of(TAG), machine.port, state / "tmp")
    lookalike = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(60)", str(machine.file_of(TAG)), INFERENCE_PATH,
    ])
    try:
        state.mkdir(parents=True)
        (state / "whisper-server.json").write_text(json.dumps({
            "pid": lookalike.pid, "tag": TAG, "file": str(machine.file_of(TAG)), "argv": argv,
            "log": str(state / "nothing.log"), "log_offset": 0, "started_at": time.time() - 1000,
        }))
        engine = machine.new_engine()
        async with machine.client() as client:
            described = await engine.describe(client)
            assert described["models_loaded"] == [] and TAG in described["models_failed"]
            await engine.hold(client, TAG, pinned=True)
            assert (await engine.describe(client))["models_loaded"] == [TAG]
        assert lookalike.poll() is None, "never signalled"
        assert len(machine.started()) == 1
    finally:
        lookalike.kill()
        lookalike.wait()


async def test_a_server_found_running_is_looked_at_again_on_the_next_pass(machine, monkeypatch):
    """A positive answer is not kept: the number may go to another process after the server
    ends, and that process must not be taken for it for longer than a moment."""
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        threading.Thread(target=machine.engine._process.wait, daemon=True).start()
        restarted = machine.new_engine()
        assert (await restarted.describe(client))["models_loaded"] == [TAG]

        monkeypatch.setattr(whisper_cpp, "IDENTITY_TRUSTED_S", 0.0)
        monkeypatch.setattr(whisper_cpp, "_is_our_server", lambda pid, record: False)
        described = await restarted.describe(client)
    assert described["models_loaded"] == [] and TAG in described["models_failed"]


# --- what it is sent, and what it leaves ---


async def test_recordings_land_in_a_directory_of_the_agents_alone_and_are_cleared(machine):
    """whisper.cpp writes each recording it is sent to its temp directory, and leaves one ffmpeg
    could not read behind. That directory is the agent's own, readable by it alone, emptied
    before every start and after every stop — of regular files only, following no link."""
    state = state_dir_for(machine.root / "models")
    uploads = state / "tmp"
    # As an earlier agent left it: open to everyone, with a recording left behind.
    uploads.mkdir(parents=True)
    state.chmod(0o755)
    uploads.chmod(0o755)
    (uploads / "whisper-server-old.wav").write_bytes(b"an earlier recording")
    outside = machine.root / "not-a-recording.txt"
    outside.write_text("kept")
    (uploads / "a-link").symlink_to(outside)
    (uploads / "a-directory").mkdir()
    (uploads / "a-directory" / "inside.txt").write_text("kept")

    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        assert machine.started()[0]["args"][-4:] == ["--tmp-dir", str(uploads), "-l", "auto"]
        assert (state.stat().st_mode & 0o777, uploads.stat().st_mode & 0o777) == (0o700, 0o700)
        assert not (uploads / "whisper-server-old.wav").exists(), "emptied before the start"

        refused = await client.post(INFERENCE_PATH, content=b"not audio", headers={"x-recording-name": "meeting.mp3"})
        assert refused.status_code == 500
        assert [p.read_bytes() for p in uploads.glob("whisper-server-*.wav")] == [b"not audio"]

        await machine.engine.hold(client, TAG, pinned=False)
    assert list(uploads.glob("whisper-server-*")) == [], "emptied after the stop"
    assert outside.read_text() == "kept" and (uploads / "a-link").is_symlink()
    assert (uploads / "a-directory" / "inside.txt").read_text() == "kept"


async def test_a_wrapper_script_and_the_server_it_runs_are_stopped_together(machine):
    """An owner's `engine_binary` may be a script that runs the server as its child. The server
    leads a process group of its own, and the whole group is stopped — not the wrapper alone,
    which would leave the real server holding the port."""
    machine.wrap(Path(machine.program), replaced=False)
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        wrapper = machine.engine._process
        [server] = [start["pid"] for start in machine.started()]
        assert server != wrapper.pid, "the server is the wrapper's child"
        await machine.engine.hold(client, TAG, pinned=False)
        assert wrapper.poll() is not None
        await wait_until(lambda: not host_platform.alive(server))
        assert (await machine.engine.describe(client))["answers"] is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes where the test forbids")
async def test_a_state_directory_it_may_not_write_is_a_refusal_saying_so(tmp_path):
    """A models directory at the top of a volume puts the state beside it, where this user may
    not write. That is the build's error, said as one — not an exception lost in the background
    with the start tried again on every pass."""
    machine = Machine(tmp_path / "volume")
    try:
        await machine.fetch(TAG)
        (tmp_path / "volume").chmod(0o500)
        async with machine.client() as client:
            with pytest.raises(EngineRefused, match="could not prepare this agent's state directory.*ermission denied"):
                await machine.engine.hold(client, TAG, pinned=True)
            work = ModelWork(
                machine.engine, client, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=True,
            )
            await work.apply({"tags": [TAG], "residency": "pinned"})
            await work._task
            held = (await work.snapshot())["models"][0]
            assert not held["loaded"] and "ermission denied" in held["error"]
            assert not work._may_retry(TAG), "left alone for a minute"
            await work.aclose()
        assert machine.started() == []
    finally:
        (tmp_path / "volume").chmod(0o755)
        machine.stop_everything()


async def test_the_reason_a_server_stopped_never_carries_what_a_client_sent(machine):
    """whisper-server prints the name a client gave its recording on every request, and ffmpeg
    the recording's own tags. The reason travels to the pool: only a line naming an error is
    reported, never one about a request — and without one, the exit alone."""
    machine.behave(last_words="")
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        await client.post(INFERENCE_PATH, content=b"audio", headers={"x-recording-name": "error-report-0417.wav"})
        process = machine.engine._process
        os.kill(process.pid, signal.SIGUSR1)
        await wait_until(lambda: process.poll() is not None)
        reason = (await machine.engine.describe(client))["models_failed"][TAG]
    assert "0417" not in reason
    assert reason.split(";")[0] == "its whisper-server exited with code 1"


async def test_an_error_after_requests_is_still_the_reason(machine):
    machine.behave(last_words="error: the device was lost")
    await machine.fetch(TAG)
    async with machine.client() as client:
        await machine.engine.hold(client, TAG, pinned=True)
        await client.post(INFERENCE_PATH, content=b"audio", headers={"x-recording-name": "error-report-0417.wav"})
        process = machine.engine._process
        os.kill(process.pid, signal.SIGUSR1)
        await wait_until(lambda: process.poll() is not None)
        reason = (await machine.engine.describe(client))["models_failed"][TAG]
    assert reason.startswith("its whisper-server exited with code 1: error: the device was lost")
    assert "0417" not in reason


# --- what the agent reports ---


async def test_the_facts_say_what_the_program_is_and_what_it_serves(machine):
    async with machine.client() as client:
        before = await machine.engine.describe(client)
        await machine.fetch(TAG)
        await machine.engine.hold(client, TAG, pinned=True)
        after = await machine.engine.describe(client)
    assert set(before) == {
        "name", "answers", "version", "binary", "ffmpeg", "models_on_disk", "models_loaded", "models_failed",
    }
    assert before["name"] == "whisper-cpp" and before["binary"] == machine.program
    assert before["answers"] is False and before["models_loaded"] == [] and before["models_failed"] == {}
    assert before["version"] is None  # the fake's installation names none
    assert isinstance(before["ffmpeg"], bool)
    assert after["answers"] is True and after["models_loaded"] == [TAG]
    assert after["models_on_disk"] == [{"tag": TAG, "size_bytes": 3000}]


@pytest.mark.parametrize("layout, version", [
    (("Cellar", "whisper.cpp", "1.9.2"), "1.9.2"),        # a package manager's own directory
    (("usr", "local"), None),
])
def test_the_version_is_what_the_installation_names(tmp_path, layout, version):
    prefix = tmp_path.joinpath(*layout)
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "whisper-server").write_text("")
    assert installed_version(str(prefix / "bin" / "whisper-server")) == version
    (prefix / "lib").mkdir()
    (prefix / "lib" / "libwhisper.so.1.8.4").write_text("")  # the library beside it, on Linux
    assert installed_version(str(prefix / "bin" / "whisper-server")) == "1.8.4"


def test_the_pools_numbers_have_no_name_here(machine):
    assert machine.engine.launch_environment(workers=1, models_held=1, context=4096) == {}
    assert machine.engine.settings_from_environment({"ANYTHING": "1"}) is None


def test_the_registry_builds_this_engine(machine):
    assert isinstance(engine_facts("whisper-cpp", machine.settings), WhisperCppFacts)


# --- the agent's own loop drives it ---


async def test_a_server_that_stops_is_reported_with_its_last_words_and_started_again(machine):
    """Level-triggered: nothing watches the process. A server that has stopped is not serving,
    so the facts say why, and the next pass holds the build again."""
    machine.behave(last_words="error: lost the device")
    async with machine.client() as client:
        work = ModelWork(
            machine.engine, client, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=True,
        )
        await work.apply({"tags": [TAG], "residency": "pinned"})
        await work._task
        held = (await work.snapshot())["models"][0]
        assert held["on_disk"] and held["loaded"] and held["pinned_by_agent"] and held["error"] is None
        assert held["awaiting_restart"] is False

        process = machine.engine._process
        os.kill(process.pid, signal.SIGUSR1)
        await wait_until(lambda: process.poll() is not None)
        stopped = (await work.snapshot())["models"][0]
        assert not stopped["loaded"] and "error: lost the device" in stopped["error"]

        await work.apply({"tags": [TAG], "residency": "pinned"})
        await work._task
        again = (await work.snapshot())["models"][0]
        assert again["loaded"] and again["error"] is None
        assert len(machine.started()) == 2
        await work.aclose()


async def test_going_on_demand_stops_the_server_the_agent_started(machine):
    async with machine.client() as client:
        work = ModelWork(
            machine.engine, client, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=True,
        )
        await work.apply({"tags": [TAG], "residency": "pinned"})
        await work._task
        process = machine.engine._process
        await work.apply({"tags": [TAG], "residency": "on_demand"})
        await work._task
        assert process.poll() is not None
        assert not (await work.snapshot())["models"][0]["loaded"]
        await work.aclose()


async def test_a_build_that_leaves_the_set_while_its_server_starts_is_stopped(machine):
    """The pass that starts it is still waiting for it to answer when the pool's next pass says
    the set is empty. The pin it records on answering is released in the same pass — not dropped
    as stale by what the pass saw before the start, with the server left running."""
    machine.behave(delay=1.0)
    await machine.fetch(TAG)
    async with machine.client() as client:
        work = ModelWork(
            machine.engine, client, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=True,
        )
        await work.apply({"tags": [TAG], "residency": "pinned"})
        await wait_until(lambda: machine.started() != [])
        starting = work._task
        process = machine.engine._process
        await work.apply({"tags": [], "residency": "pinned"})
        await starting
        assert process.poll() is not None, "stopped"
        assert work.pinned_by_agent == set() and machine.engine._current() is None
        assert (await machine.engine.describe(client))["answers"] is False
        await work.aclose()


async def test_a_server_left_starting_is_stopped_once_its_build_leaves_the_set(machine, monkeypatch):
    """A start that has not answered in time is left running, pinned by no one. Once its build
    is no longer wanted, the agent stops the server it started — whatever its pins say."""
    monkeypatch.setattr(whisper_cpp, "START_TIMEOUT_S", 0.5)
    machine.behave(delay=60)
    await machine.fetch(TAG)
    async with machine.client() as client:
        work = ModelWork(
            machine.engine, client, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=True,
        )
        await work.apply({"tags": [TAG], "residency": "pinned"})
        await work._task
        process = machine.engine._process
        assert process.poll() is None and "has not answered" in (await work.snapshot())["models"][0]["error"]
        assert work.pinned_by_agent == set()

        await work.apply({"tags": [], "residency": "pinned"})
        await work._task
        assert process.poll() is not None, "stopped"
        assert machine.engine._current() is None
        await work.aclose()


async def test_more_than_one_build_in_the_set_is_refused(machine):
    """The server serves one file. Holding each of two in turn would restart it on every pass,
    so a set naming two is refused, saying why, and nothing is fetched or started."""
    async with machine.client() as client:
        work = ModelWork(
            machine.engine, client, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=True,
        )
        with pytest.raises(Refused) as refused:
            await work.apply({"tags": [TAG, OTHER], "residency": "pinned"})
        assert (refused.value.status, refused.value.error) == (400, "one_model_only")
        assert "one model at a time" in refused.value.detail and TAG in refused.value.detail
        assert work.desired.tags == () and work._task is None
        # One build — named twice or not — is the set it always was.
        await work.apply({"tags": [TAG, TAG], "residency": "pinned"})
        await work._task
        assert (await work.snapshot())["models"][0]["loaded"]
        await work.aclose()
    assert machine.hub_requests and len(machine.started()) == 1


# --- the agent's settings ---


@pytest.mark.parametrize("url", [
    "http://10.0.0.5:8178",
    "http://0.0.0.0:8178",
    "http://[::1]:8178",
    "https://127.0.0.1:8178",
    "http://127.0.0.1",
    "http://127.0.0.1:8178/v1",
    "not a url",
])
def test_the_server_is_only_ever_on_loopback(url):
    """It has no authentication: the agent starts it on 127.0.0.1 and refuses to be told it is
    anywhere else, so the pool reaches it on the machine or through a tunnel (D148)."""
    with pytest.raises(SettingsError, match="D148"):
        Settings(key_hash="x", engine="whisper-cpp", engine_url=url).check()


def test_a_loopback_address_with_a_port_is_accepted_and_other_engines_are_unaffected():
    Settings(key_hash="x", engine="whisper-cpp", engine_url="http://127.0.0.1:8178").check()
    Settings(key_hash="x", engine="whisper-cpp", engine_url="http://localhost:8178/").check()
    Settings(key_hash="x", engine="ollama", engine_url="http://10.0.0.5:11434").check()


def test_the_program_is_named_by_an_absolute_path_or_not_at_all():
    with pytest.raises(SettingsError, match="absolute path"):
        Settings(key_hash="x", engine="whisper-cpp", engine_url="http://127.0.0.1:8178",
                 engine_binary="bin/whisper-server").check()


def test_init_for_whisper_cpp_writes_its_own_defaults(tmp_path, capsys):
    path = tmp_path / "agent.json"
    assert cli.main(["-c", str(path), "init", "--engine", "whisper-cpp"]) == 0
    written = load(path)
    assert written.engine == "whisper-cpp"
    assert written.engine_url == "http://127.0.0.1:8178"
    assert written.models_path == "~/gpm-whisper-models"
    assert written.engine_binary is None

    named = tmp_path / "named.json"
    assert cli.main(["-c", str(named), "init", "--engine", "whisper-cpp",
                     "--engine-binary", "/opt/whisper/bin/whisper-server"]) == 0
    assert load(named).engine_binary == "/opt/whisper/bin/whisper-server"

    refused = tmp_path / "refused.json"
    assert cli.main(["-c", str(refused), "init", "--engine", "whisper-cpp",
                     "--engine-url", "http://192.168.1.4:8178"]) == 2
    assert not refused.exists()


def test_init_for_the_other_engines_is_unchanged(tmp_path):
    path = tmp_path / "agent.json"
    assert cli.main(["-c", str(path), "init"]) == 0
    assert load(path).engine_url == "http://127.0.0.1:11434"
    assert load(path).models_path == "~/.ollama/models"


def test_the_transcription_path_is_the_one_the_pool_forwards():
    from gpm_server.engines import openai_api

    assert INFERENCE_PATH in openai_api.TRANSCRIBE_PATHS
