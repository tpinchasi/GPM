"""The agent runs vLLM on a machine its owner gives it, and chooses its engine under `auto`
(S10: D159, D161).

docs/spec/host-agent.md §2.1 and §2.3. Against a fake `vllm`: a small Python program, put where an
installed version's program goes (`<engines>/vllm/<port>/<version>/venv/bin/vllm`), that answers
`/health`, `/version` and `/v1/models` for the model it was started with — started by the agent's
own launcher, run for real as `python -m gpm_agent.cli vllm-start`, with the router and the helper
that starts a card's models in turn when there are several. Nothing here needs a GPU, a network or
vLLM itself; the model hub is pointed at a closed port, so nothing could be fetched by mistake.
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
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import gpm_agent
import httpx
import pytest
from gpm_agent import app as agent_app
from gpm_agent import cli, installs, modelhub, vllm_install, vllm_launch, vllm_server
from gpm_agent import platform as host_platform
from gpm_agent.app import CHOICE_FILE, choose_engine, create_app
from gpm_agent.engines import VllmFacts
from gpm_agent.installs import Installer, Release
from gpm_agent.models import Refused
from gpm_agent.ollama_server import OllamaServer
from gpm_agent.settings import Settings, SettingsError, fingerprint, load
from gpm_agent.vllm_server import VllmServer

AGENT_KEY = "gpmg_" + "c" * 64
AGENT_SRC = str(Path(gpm_agent.__file__).parents[1])
V1, V2 = "0.31.0", "0.32.0"
CHAT, SPEECH = "acme/chat-model", "acme/speech-model"

#: The fake `vllm`. Told how to behave per model by a file beside the test; writes down every start
#: — its own command line and what of its environment matters — and serves the one model it was
#: started with, as `vllm serve` does.
FAKE_VLLM = '''
import http.server, json, os, sys, time

STARTS, CONTROL, VERSION = {starts!r}, {control!r}, {version!r}
WATCHED = ["GPM_VLLM_MAX_NUM_SEQS", "GPM_VLLM_MAX_NUM_BATCHED_TOKENS", "GPM_VLLM_MAX_MODEL_LEN",
           "HF_TOKEN", "SOME_SERVICE_TOKEN", "GITHUB_TOKEN", "CUDA_VISIBLE_DEVICES", "HOME", "PATH"]
args = sys.argv[1:]
assert args[0] == "serve", args
name = args[args.index("--served-model-name") + 1]
port = int(args[args.index("--port") + 1])
try:
    control = json.load(open(CONTROL)).get(name, {{}})
except OSError:
    control = {{}}
with open(STARTS, "a") as starts:
    starts.write(json.dumps({{"version": VERSION, "argv": sys.argv, "pid": os.getpid(), "model": name,
                             "env": {{k: os.environ.get(k) for k in WATCHED}}}}) + "\\n")
print("INFO starting vLLM", VERSION, "for", name, flush=True)
if control.get("exit"):
    print(control["exit"], flush=True)
    sys.exit(1)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        bodies = {{"/health": {{}}, "/version": {{"version": VERSION}},
                  "/v1/models": {{"object": "list", "data": [{{"id": name, "object": "model"}}]}}}}
        body = bodies.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(json.dumps(body or {{}}).encode())

    def log_message(self, *_):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''

#: A fake `ollama`, for an agent of `auto` that ran Ollama before: answers its version on the
#: address the agent gives it, and nothing else.
FAKE_OLLAMA = '''
import http.server, json, os, sys
host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"version": "0.40.11"}).encode())

    def log_message(self, *_):
        pass


http.server.ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
'''


def free_ports(count: int = 3) -> int:
    """A port with the `count - 1` above it free too: the router's, and its engines' (D96)."""
    for _ in range(100):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        if base + count > 65535:
            continue
        try:
            for port in range(base, base + count):
                with socket.socket() as taken:
                    taken.bind(("127.0.0.1", port))
        except OSError:
            continue
        return base
    raise RuntimeError("no free ports")


async def wait_until(condition, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.05)


def gone(pid: int) -> bool:
    return not host_platform.alive(pid)


class FakeSource:
    """vLLM's versions as the agent's installer sees them, without PyPI: each version is the fake
    program, put where a version's own `vllm` goes (`vllm_install.program_in`)."""

    engine = "vllm"
    signature = "not checked"

    def __init__(self, machine: "Machine", latest: str = V1) -> None:
        self.machine, self.newest = machine, latest

    async def latest(self, http: httpx.AsyncClient) -> Release:
        return Release(self.newest, f"v{self.newest}", ())

    async def release(self, http: httpx.AsyncClient, version: str) -> Release:
        return Release(version, f"v{version}", ())

    async def stage(self, http, release, into, progress, room=None) -> None:
        self.machine.write_program(vllm_install.program_in(into), release.version)

    def program(self, version_dir: Path) -> Path:
        return vllm_install.program_in(version_dir)


class Machine:
    """The owner's machine: an engines directory, a models directory, an engine port with the two
    above it free, and the fake vLLM — what the agent starts, as it starts it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.engines = root / "engines"
        self.models = root / "models"
        self.models.mkdir(parents=True)
        self.port = free_ports()
        self.url = f"http://127.0.0.1:{self.port}"
        self.engine_dir = self.engines / "vllm" / str(self.port)
        self.starts_file = root / "fake-vllm-starts.jsonl"
        self.control = root / "fake-vllm-control.json"
        #: What the pool told this machine to hold — None until it has said.
        self.held: Optional[list[str]] = None
        #: Every command the agent ran, with its environment.
        self.ran: list[tuple[list[str], dict[str, str]]] = []
        self.clients: list[httpx.AsyncClient] = []
        self.started: list[subprocess.Popen] = []

    def write_program(self, path: Path, version: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!{sys.executable}\n" + FAKE_VLLM.format(
            starts=str(self.starts_file), control=str(self.control), version=version,
        ))
        path.chmod(0o755)
        return path

    def install(self, version: str) -> Path:
        """`version` as the installer leaves it: its own program in its own directory."""
        return self.write_program(vllm_install.program_in(self.engine_dir / version), version)

    def downloaded(self, repo: str, size: int = 1000) -> Path:
        into = modelhub.directory_for(self.models, repo)
        into.mkdir(parents=True, exist_ok=True)
        config = b'{"model_type": "qwen3"}'
        (into / "config.json").write_bytes(config)
        (into / "model.safetensors").write_bytes(b"w" * size)
        (into / modelhub.COMPLETE_MARKER).write_text(f"{size + len(config)}\n")
        assert modelhub.is_complete(into)
        return into

    def behave(self, model: str, **control) -> None:
        current = json.loads(self.control.read_text()) if self.control.exists() else {}
        current[model] = control
        self.control.write_text(json.dumps(current))

    async def run(self, argv, *, env=None, **options) -> host_platform.Ran:
        """The agent's own runner, with this checkout's code on the path of the program it runs —
        the environment the agent gives it carries no PYTHONPATH."""
        self.ran.append((list(argv), dict(env or {})))
        return await host_platform.run_command(argv, env={**(env or {}), "PYTHONPATH": AGENT_SRC}, **options)

    def client(self) -> httpx.AsyncClient:
        made = httpx.AsyncClient(base_url=self.url, timeout=10)
        self.clients.append(made)
        return made

    def server(self) -> VllmServer:
        """The server as a freshly started agent builds it."""
        return VllmServer(
            self.engine_dir, engine_url=self.url, models_path=str(self.models), client=self.client(),
            engines_dir=self.engines, held=lambda: self.held, run=self.run, poll_s=0.05,
        )

    def installer(self, server: Optional[VllmServer] = None, latest: str = V1) -> Installer:
        return Installer(
            FakeSource(self, latest), self.engine_dir, server or self.server(),
            http_transport=httpx.MockTransport(lambda request: httpx.Response(404)), switch_timeout_s=30,
        )

    def settings(self, **more) -> Settings:
        return Settings(
            key_hash=fingerprint(AGENT_KEY), engine=more.pop("engine", "vllm"), engine_url=self.url,
            models_path=str(self.models), install_engines=True, engines_path=str(self.engines), **more,
        )

    def starts(self) -> list[dict]:
        if not self.starts_file.exists():
            return []
        return [json.loads(line) for line in self.starts_file.read_text().splitlines()]

    def launches(self) -> list[list[str]]:
        return [argv for argv, _ in self.ran if "vllm-start" in argv]

    async def serving(self) -> Optional[set[str]]:
        async with httpx.AsyncClient(base_url=self.url, timeout=3) as http:
            try:
                answer = await http.get("/v1/models")
            except httpx.HTTPError:
                return None
        return {m["id"] for m in answer.json()["data"]} if answer.status_code == 200 else None

    async def close(self) -> None:
        vllm_launch.stop(self.models, grace_s=5)
        for start in self.starts():
            said = host_platform.command_line(start["pid"]) or b""
            if host_platform.alive(start["pid"]) and str(self.engines).encode() in said:
                os.kill(start["pid"], signal.SIGKILL)
        for process in self.started:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        for client in self.clients:
            await client.aclose()


@pytest.fixture
async def machine(tmp_path, monkeypatch):
    # Anything that tried to fetch a model would find a closed port, never the real hub.
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    made = Machine(tmp_path)
    yield made
    await made.close()


async def call(app, method, path, body=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=60) as http:
        return await http.request(method, path, json=body, headers={"Authorization": f"Bearer {AGENT_KEY}"})


# --- the agent's own settings ---


def test_vllm_is_an_engine_the_agent_installs_and_auto_is_only_where_it_does():
    Settings(key_hash="x", engine="vllm", install_engines=True).check()
    Settings(key_hash="x", engine="auto", install_engines=True).check()
    with pytest.raises(SettingsError, match="install_engines: true") as refused:
        Settings(key_hash="x", engine="auto").check()
    assert "D161" in str(refused.value)
    with pytest.raises(SettingsError, match="loopback"):
        Settings(key_hash="x", engine="auto", install_engines=True, engine_url="http://10.0.0.5:8000").check()


def test_init_writes_an_engine_of_auto_or_vllm_with_installs_and_refuses_auto_without(tmp_path, capsys):
    for engine in ("auto", "vllm"):
        path = tmp_path / f"{engine}.json"
        assert cli.main(["-c", str(path), "init", "--engine", engine, "--install-engines"]) == 0
        written = load(path)
        assert (written.engine, written.install_engines) == (engine, True)
    out = capsys.readouterr().out
    assert "chooses its engine each time it starts" in out and "This agent installs vllm" in out
    refused = tmp_path / "refused.json"
    assert cli.main(["-c", str(refused), "init", "--engine", "auto"]) == 2
    assert "install_engines" in capsys.readouterr().err
    assert not refused.exists(), "nothing written for settings that do not load"


# --- the choice of `auto` (D161) ---

SET = vllm_install.LockedSet(
    vllm=V2, python="3.12", platform="x86_64", driver_floor="575.51.03", requirements="", size_bytes=1,
)


def card(name: str, capability: str, driver: str = "580.65.06", kind: str = "cuda") -> dict:
    return {"kind": kind, "name": name, "memory_bytes": 24 * 2**30, "unified": False,
            "compute_capability": capability, "driver": driver}


def facts_of(os_name="Linux", arch="x86_64", accelerators=None) -> dict:
    return {"os": os_name, "arch": arch, "accelerators": [card("NVIDIA RTX A6000", "8.6")] if accelerators is None else accelerators}


@pytest.mark.parametrize("machine_facts, python, engine, reasons", [
    (facts_of(), "3.12", "vllm", []),
    (facts_of(accelerators=[card("NVIDIA H100 80GB HBM3", "9.0"), card("NVIDIA L40S", "8.9")]), "3.12", "vllm", []),
    (facts_of(accelerators=[card("Tesla T4", "7.5")]), "3.12", "ollama",
     ["Tesla T4 has compute capability 7.5; this needs 8.0 or higher"]),
    (facts_of(accelerators=[{"kind": "rocm", "name": "AMD Radeon PRO W7900", "memory_bytes": None, "unified": False}]),
     "3.12", "ollama", ["no NVIDIA card (found: rocm)"]),
    (facts_of("Darwin", "arm64", [{"kind": "apple-silicon", "name": "Apple M4 Max", "memory_bytes": 1, "unified": True}]),
     "3.12", "ollama", ["vLLM runs on Linux only, and this machine runs Darwin",
                        "vLLM publishes no build for the arm64 processor", "no NVIDIA card (found: apple-silicon)"]),
    (facts_of("Windows", "AMD64", [card("NVIDIA GeForce RTX 4090", "8.9")]), "3.12", "ollama",
     ["vLLM runs on Linux only, and this machine runs Windows", "vLLM publishes no build for the AMD64 processor"]),
    (facts_of(), "3.13", "ollama", ["the agent carries no vLLM set for Python 3.13 on x86_64"]),
    (facts_of(accelerators=[card("NVIDIA RTX A6000", "8.6", driver="550.54.15")]), "3.12", "ollama",
     ["NVIDIA RTX A6000's driver 550.54.15 is older than 575.51.03, which vLLM's CUDA needs"]),
    (facts_of(accelerators=[]), "3.12", "ollama", ["no NVIDIA card"]),
])
def test_auto_chooses_vllm_only_where_the_machine_runs_its_builds_and_says_why_not(machine_facts, python, engine, reasons):
    assert choose_engine(machine_facts, python=python, sets=[SET]) == {"engine": engine, "reasons": reasons}


async def test_an_agent_of_auto_is_an_agent_of_the_engine_it_chose_and_says_so(machine, monkeypatch):
    monkeypatch.setattr(agent_app, "this_python", lambda: "3.12")  # the set's own Python
    real = agent_app.gather
    monkeypatch.setattr(agent_app, "gather", lambda probes, path: {**real(probes, path), **facts_of(
        accelerators=[card("Tesla T4", "7.5")])})
    app = create_app(machine.settings(engine="auto"), vllm_sets=[SET])
    facts = (await call(app, "GET", "/agent/v1/facts")).json()
    assert facts["engine"]["choice"] == {
        "engine": "ollama", "reasons": ["Tesla T4 has compute capability 7.5; this needs 8.0 or higher"],
    }
    assert facts["engine"]["name"] == "ollama" and facts["engine"]["install"]["engine"] == "ollama"
    assert app.state.installer.engine_dir == machine.engines / "ollama" / str(machine.port)

    monkeypatch.setattr(agent_app, "gather", lambda probes, path: {**real(probes, path), **facts_of()})
    chosen = create_app(machine.settings(engine="auto"), vllm_sets=[SET], installer=machine.installer())
    facts = (await call(chosen, "GET", "/agent/v1/facts")).json()
    assert facts["engine"]["choice"] == {"engine": "vllm", "reasons": []}
    assert set(facts["engine"]["choice"]) == {"engine", "reasons"}, "the shape the pool takes"
    assert facts["engine"]["name"] == "vllm" and facts["engine"]["install"]["engine"] == "vllm"


async def test_an_agent_that_names_its_engine_reports_no_choice(machine):
    facts = (await call(create_app(machine.settings(), installer=machine.installer()), "GET", "/agent/v1/facts")).json()
    assert "choice" not in facts["engine"]
    install = facts["engine"]["install"]
    assert install["allowed"] is True and install["engine"] == "vllm"


async def test_without_the_owners_word_a_vllm_host_says_installs_are_not_allowed(machine):
    """A rented vLLM host, or one whose owner runs vLLM: the same `engine.install` the other
    engines report, saying no."""
    settings = Settings(key_hash=fingerprint(AGENT_KEY), engine="vllm", engine_url=machine.url, models_path=str(machine.models))
    facts = (await call(create_app(settings), "GET", "/agent/v1/facts")).json()
    assert facts["engine"]["install"] == installs.not_allowed("vllm")
    assert "choice" not in facts["engine"] and "foreign" not in facts["engine"]


# --- waiting for the models it holds (D159) ---


async def test_nothing_starts_until_every_model_it_holds_is_on_disk(machine):
    program = machine.install(V1)
    server = machine.server()
    machine.downloaded(CHAT)
    assert await server.start(program, V1, 10) is None, "waiting is not a failure"
    assert "the pool to say" in server.waiting and machine.ran == []

    machine.held = [CHAT, SPEECH]
    assert await server.start(program, V1, 10) is None
    assert server.waiting == f"waiting for {SPEECH} to be on disk before starting vLLM"
    assert machine.ran == [] and not server.ours() and not await server.foreign()

    machine.downloaded(SPEECH)
    assert await server.start(program, V1, 30) is None
    assert server.waiting is None
    assert await machine.serving() == {CHAT, SPEECH}
    assert server.ours() and server.running_version() == V1


async def test_a_download_in_progress_is_not_a_model_on_disk(machine):
    program = machine.install(V1)
    server = machine.server()
    machine.held = [CHAT]
    partial = modelhub.directory_for(machine.models, CHAT)
    partial.mkdir(parents=True)
    (partial / "model.safetensors").write_bytes(b"w" * 10)
    assert await server.start(program, V1, 10) is None
    assert machine.ran == [] and CHAT in server.waiting


async def test_it_starts_the_installed_program_through_its_own_launcher_with_the_pools_numbers(machine, monkeypatch):
    for secret in ("HF_TOKEN", "SOME_SERVICE_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.setenv(secret, "not-a-real-secret")
    program = machine.install(V1)
    server = machine.server()
    numbers = {"GPM_VLLM_MAX_NUM_SEQS": "6", "GPM_VLLM_MAX_NUM_BATCHED_TOKENS": "8192", "GPM_VLLM_MAX_MODEL_LEN": "4096"}
    assert server.set_pool_environment(numbers) is True
    assert server.set_pool_environment(numbers) is False, "unchanged"
    machine.downloaded(CHAT)
    machine.held = [CHAT]

    assert await server.start(program, V1, 30) is None
    (launcher, env), = machine.ran
    assert launcher == [
        sys.executable, "-P", "-m", "gpm_agent.cli", "vllm-start", "--models-dir", str(machine.models),
        "--port", str(machine.port), "--vllm", str(program), "--engines-dir", str(machine.engines),
    ], "one model: no router"
    assert {k: env[k] for k in numbers} == numbers
    assert not {"HF_TOKEN", "SOME_SERVICE_TOKEN", "GITHUB_TOKEN", "PYTHONPATH"} & set(env)

    (started,) = machine.starts()
    argv = started["argv"]
    assert argv[0] == str(program) and argv[1:3] == ["serve", str(modelhub.directory_for(machine.models, CHAT))]
    assert argv[argv.index("--port") + 1] == str(machine.port) and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--max-num-seqs") + 1] == "6" and argv[argv.index("--max-model-len") + 1] == "4096"
    assert started["env"]["HF_TOKEN"] is None and started["env"]["SOME_SERVICE_TOKEN"] is None
    assert started["env"]["GPM_VLLM_MAX_NUM_SEQS"] == "6"

    record = json.loads((machine.engine_dir / vllm_server.RECORD_FILE).read_text())
    assert (record["version"], record["program"], record["models"], record["argv"], record["port"]) == (
        V1, str(program), [CHAT], launcher, machine.port,
    )
    assert server.ours() and server.running_version() == V1 and not await server.foreign()
    assert await server.start(program, V1, 30) == "this agent's vLLM is already running"

    # An agent started afresh knows it again, by its record and the processes' command lines.
    again = machine.server()
    assert again.ours(fresh=True) and again.running_version() == V1 and not await again.foreign()


async def test_the_pools_numbers_through_the_restart_verb_reach_the_next_start(machine):
    machine.downloaded(CHAT)
    installer = machine.installer()
    await installer.want("latest", False)
    await installer.finished()
    assert installer.record.running == V1 and machine.ran == [], "installed; waiting for the pool's set"
    app = create_app(machine.settings(), installer=installer)

    answer = (await call(app, "POST", "/agent/v1/engine", {"settings": {"workers": 4, "models_held": 1, "context": 8192}})).json()
    assert answer["settings_written"] is True and answer["engine_answers"] is False
    assert answer["restart_output"].startswith(f"vllm {V1} is not started yet: waiting for the pool")
    facts = (await call(app, "GET", "/agent/v1/facts")).json()
    assert facts["engine_settings"] == {"workers": 4, "models_held": 1, "context": 8192}
    assert facts["allows"]["restart"] is True and facts["allows"]["engine_settings"] is True

    held = await call(app, "PUT", "/agent/v1/models", {"tags": [CHAT], "residency": "pinned"})
    assert held.status_code == 200
    await app.state.look()
    await wait_until(lambda: machine.starts())
    argv = machine.starts()[0]["argv"]
    assert argv[argv.index("--max-num-seqs") + 1] == "4" and argv[argv.index("--max-model-len") + 1] == "8192"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "8192"


# --- starting again when the set changes, and on a switch ---


async def test_it_starts_again_when_a_model_held_lands_and_not_for_one_nobody_asked_for(machine):
    machine.downloaded(CHAT)
    installer = machine.installer()
    await installer.want("latest", False)
    await installer.finished()
    app = create_app(machine.settings(), installer=installer)
    await call(app, "PUT", "/agent/v1/models", {"tags": [CHAT], "residency": "pinned"})
    await app.state.look()
    assert await machine.serving() == {CHAT}
    first = machine.starts()[0]["pid"]

    machine.downloaded(SPEECH)  # on disk, and held by no one: not worth a restart
    await app.state.look()
    assert len(machine.launches()) == 1 and not gone(first)

    await call(app, "PUT", "/agent/v1/models", {"tags": [CHAT, SPEECH], "residency": "pinned"})
    await app.state.look()
    await wait_until(lambda: gone(first))
    assert "--proxy" in machine.launches()[-1], "several models: the router in front of them"
    await wait_until(lambda: len(machine.starts()) == 3)
    for _ in range(200):
        if await machine.serving() == {CHAT, SPEECH}:
            break
        await asyncio.sleep(0.05)
    assert await machine.serving() == {CHAT, SPEECH}
    record = json.loads((machine.engine_dir / vllm_server.RECORD_FILE).read_text())
    assert record["models"] == [CHAT, SPEECH]

    # A model it runs, deleted: started again without it.
    await call(app, "PUT", "/agent/v1/models", {"tags": [CHAT], "residency": "pinned"})
    assert (await call(app, "DELETE", "/agent/v1/models", {"tag": SPEECH})).status_code == 200
    await app.state.look()
    assert "--proxy" not in machine.launches()[-1]
    assert await machine.serving() == {CHAT}


async def test_a_switch_stops_the_old_version_with_the_stop_verb_and_starts_the_new(machine):
    machine.downloaded(CHAT)
    machine.held = [CHAT]
    installer = machine.installer(latest=V1)
    await installer.want("latest", False)
    await installer.finished()
    assert await machine.serving() == {CHAT}
    old = machine.starts()[0]["pid"]

    installer.source.newest = V2
    installer.record.latest_checked_at = None
    await installer.want("latest", False)
    await installer.finished()
    assert installer.record.staged == V2 and not gone(old), "staged while it serves"
    await installer.want("latest", True)
    await installer.finished()

    assert installer.record.running == V2 and installer.record.previous == V1
    assert gone(old)
    assert [s["version"] for s in machine.starts()] == [V1, V2]
    stops = [argv for argv, _ in machine.ran if "vllm-stop" in argv]
    assert stops == [[sys.executable, "-P", "-m", "gpm_agent.cli", "vllm-stop", "--models-dir", str(machine.models)]]
    assert machine.launches()[-1][machine.launches()[-1].index("--vllm") + 1] == str(installer.program(V2))
    assert installer.server.running_version() == V2 and await machine.serving() == {CHAT}


async def test_a_switch_while_another_model_is_fetched_starts_again_what_ran_before(machine):
    """A switch stops what runs; the model still fetching must not leave the machine serving
    nothing until it lands."""
    machine.downloaded(CHAT)
    machine.held = [CHAT]
    installer = machine.installer()
    await installer.want("latest", False)
    await installer.finished()
    machine.held = [CHAT, SPEECH]  # SPEECH is still on its way
    installer.source.newest = V2
    installer.record.latest_checked_at = None
    await installer.want("latest", False)
    await installer.finished()
    await installer.want("latest", True)
    await installer.finished()
    assert installer.record.running == V2 and await machine.serving() == {CHAT}


# --- stopping, and what it did not start ---


async def test_stop_runs_the_stop_verb_and_forgets_nothing_it_needs_to_start_again(machine):
    program = machine.install(V1)
    server = machine.server()
    machine.downloaded(CHAT)
    machine.held = [CHAT]
    assert await server.start(program, V1, 30) is None
    pid = machine.starts()[0]["pid"]

    await server.stop()
    await wait_until(lambda: gone(pid))
    assert not server.ours(fresh=True) and server.running_version() is None
    assert vllm_launch.read_record(machine.models) is None, "stopped, not failed: no record of a failure"
    assert VllmFacts(machine.settings())._failed([]) == {}
    record = json.loads((machine.engine_dir / vllm_server.RECORD_FILE).read_text())
    assert record["stopped"] is True and record["models"] == [CHAT]
    await server.stop()  # nothing runs: nothing to do
    assert len([argv for argv, _ in machine.ran if "vllm-stop" in argv]) == 1

    # Started again by an agent that has not heard from the pool yet — after a reboot, say: what
    # it ran before, all still on disk.
    fresh = machine.server()
    machine.held = None
    assert await fresh.start(program, V1, 30) is None and await machine.serving() == {CHAT}


async def test_a_vllm_the_agent_did_not_start_is_reported_and_left_alone(machine):
    class Theirs(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"object": "list", "data": [{"id": "their/model"}]}')

        def log_message(self, *_):
            pass

    theirs = ThreadingHTTPServer(("127.0.0.1", machine.port), Theirs)
    threading.Thread(target=theirs.serve_forever, daemon=True).start()
    try:
        program = machine.install(V1)
        machine.downloaded(CHAT)
        machine.held = [CHAT]
        server = machine.server()
        assert await server.foreign() is True
        assert await server.start(program, V1, 10) == server.foreign_reason
        await server.stop()
        assert machine.ran == [], "nothing run, nothing stopped"
        installer = machine.installer(server)
        with pytest.raises(Refused) as refused:
            await installer.want("latest", False)
        assert refused.value.error == "foreign_engine"
        app = create_app(machine.settings(), installer=installer)
        facts = (await call(app, "GET", "/agent/v1/facts")).json()
        assert facts["engine"]["foreign"] is True and facts["engine"]["models_loaded"] == ["their/model"]
    finally:
        theirs.shutdown()
        theirs.server_close()


async def test_a_vllm_its_owner_started_with_the_launcher_is_not_the_agents(machine):
    """The owner's own `gpm-agent vllm-start`, on the same port and models: the launcher's record
    names its processes, and none is the agent's program — so not the agent's to stop."""
    machine.downloaded(CHAT)
    owners = machine.write_program(machine.root / "owners" / "bin" / "vllm", "0.30.0")
    done = subprocess.run(
        [sys.executable, "-m", "gpm_agent.cli", "vllm-start", "--models-dir", str(machine.models), "--port", str(machine.port)],
        env={**os.environ, "PYTHONPATH": AGENT_SRC, "PATH": f"{owners.parent}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True, timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    await wait_until(lambda: machine.starts())
    pid = machine.starts()[0]["pid"]
    for _ in range(100):
        if await machine.serving():
            break
        await asyncio.sleep(0.05)

    # Even with a record of its own from before, naming another program.
    program = machine.install(V1)
    machine.engine_dir.mkdir(parents=True, exist_ok=True)
    (machine.engine_dir / vllm_server.RECORD_FILE).write_text(json.dumps({
        "version": V1, "program": str(program), "models": [CHAT], "argv": [], "port": machine.port,
    }))
    server = machine.server()
    assert not server.ours(fresh=True) and await server.foreign()
    await server.stop()
    assert not gone(pid) and machine.ran == []


# --- a model that will not start ---


async def test_a_model_that_dies_beside_one_that_serves_is_that_models_failure(machine):
    program = machine.install(V1)
    server = machine.server()
    machine.downloaded(CHAT)
    machine.downloaded(SPEECH)
    machine.held = [CHAT, SPEECH]
    machine.behave(SPEECH, exit="ValueError: 2.14 GiB KV cache is needed, larger than the available 1.83 GiB")
    assert await server.start(program, V1, 30) is None, "the version serves; one model does not"
    assert await machine.serving() == {CHAT}
    facts = await VllmFacts(machine.settings()).describe(machine.client())
    assert facts["models_failed"][SPEECH].endswith("ValueError: 2.14 GiB KV cache is needed, larger than the available 1.83 GiB")


async def test_a_start_where_nothing_serves_says_why(machine):
    program = machine.install(V1)
    server = machine.server()
    machine.downloaded(CHAT)
    machine.held = [CHAT]
    machine.behave(CHAT, exit="RuntimeError: CUDA error: no kernel image is available for execution on the device")
    problem = await server.start(program, V1, 30)
    assert problem == (
        f"{CHAT}: its vLLM process exited before serving it: "
        "RuntimeError: CUDA error: no kernel image is available for execution on the device"
    )


# --- `auto` choosing again: the engine it ran before is handed over ---


def started_ollama(machine: Machine, version: str = "0.40.11") -> tuple[OllamaServer, subprocess.Popen]:
    """Ollama as this agent started it while it chose Ollama: its program in its version
    directory, its record in its engine directory."""
    program = machine.engines / "ollama" / str(machine.port) / version / "ollama"
    program.parent.mkdir(parents=True)
    program.write_text(f"#!{sys.executable}\n{FAKE_OLLAMA}")
    program.chmod(0o755)
    server = OllamaServer(
        machine.engines / "ollama" / str(machine.port), engine_url=machine.url, models_path=str(machine.models),
        client=machine.client(),
    )
    return server, program


def chose_before(machine: Machine, engine: str) -> Path:
    """What an agent of `auto` recorded choosing the last time it started."""
    choice = machine.engines / "auto" / str(machine.port) / CHOICE_FILE
    choice.parent.mkdir(parents=True)
    choice.write_text(json.dumps({"engine": engine}))
    return choice


async def test_an_auto_agent_that_now_chooses_vllm_stops_the_ollama_it_started_first(machine, monkeypatch):
    ollama, program = started_ollama(machine)
    assert await ollama.start(program, "0.40.11", 10) is None
    process = ollama._process
    machine.started.append(process)
    # The agent that started it is gone; what reaps it now is the system — a thread here.
    threading.Thread(target=process.wait, daemon=True).start()
    choice = chose_before(machine, "ollama")

    # The machine now has a card vLLM runs on: the agent starts again, and chooses vLLM.
    real = agent_app.gather
    monkeypatch.setattr(agent_app, "gather", lambda probes, path: {**real(probes, path), **facts_of()})
    monkeypatch.setattr(agent_app, "this_python", lambda: "3.12")
    machine.downloaded(CHAT)
    installer = machine.installer()
    app = create_app(machine.settings(engine="auto"), vllm_sets=[SET], installer=installer)
    assert (await call(app, "GET", "/agent/v1/facts")).json()["engine"]["choice"]["engine"] == "vllm"
    await app.state.look()
    await wait_until(lambda: process.poll() is not None)
    assert json.loads(choice.read_text()) == {"engine": "vllm"}
    assert not (ollama.state_dir / "server.json").exists(), "Ollama's record forgotten with it"

    # And then the engine it chose: installed, and started once the pool has said what it holds.
    await installer.want("latest", False)
    await installer.finished()
    await call(app, "PUT", "/agent/v1/models", {"tags": [CHAT], "residency": "pinned"})
    await app.state.look()
    assert await machine.serving() == {CHAT}


async def test_an_auto_agent_that_now_chooses_ollama_stops_the_vllm_it_started(machine, monkeypatch):
    program = machine.install(V1)
    machine.downloaded(CHAT)
    machine.held = [CHAT]
    assert await machine.server().start(program, V1, 30) is None
    pid = machine.starts()[0]["pid"]
    choice = chose_before(machine, "vllm")

    real = agent_app.gather
    monkeypatch.setattr(agent_app, "gather", lambda probes, path: {**real(probes, path), **facts_of(
        accelerators=[card("Tesla T4", "7.5")])})
    monkeypatch.setattr(agent_app.VllmServer, "__init__", _with_runner(agent_app.VllmServer.__init__, machine))
    app = create_app(machine.settings(engine="auto"), vllm_sets=[SET])
    assert app.state.installer.server.__class__ is OllamaServer
    await app.state.look()
    await wait_until(lambda: gone(pid))
    assert json.loads(choice.read_text()) == {"engine": "ollama"}
    assert [argv[argv.index("vllm-stop"):] for argv, _ in machine.ran if "vllm-stop" in argv] == [
        ["vllm-stop", "--models-dir", str(machine.models)],
    ]
    await app.state.look()
    assert len([argv for argv, _ in machine.ran if "vllm-stop" in argv]) == 1, "handed over once"


def _with_runner(init, machine: Machine):
    """VllmServer's constructor, giving every one made after this the test's runner."""
    def made(self, *args, **kwargs):
        kwargs.setdefault("run", machine.run)
        init(self, *args, **kwargs)
    return made


async def test_the_hand_over_never_stops_what_the_agent_did_not_start(machine, monkeypatch):
    """Ollama on the port that this agent did not start — its owner's own — and a choice that
    moved to vLLM: nothing is stopped, and the port is reported as held by something else."""
    owners = subprocess.Popen(
        [sys.executable, "-c", FAKE_OLLAMA], env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{machine.port}"},
    )
    machine.started.append(owners)
    for _ in range(100):
        try:
            httpx.get(f"{machine.url}/api/version", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    choice = chose_before(machine, "ollama")
    real = agent_app.gather
    monkeypatch.setattr(agent_app, "gather", lambda probes, path: {**real(probes, path), **facts_of()})
    monkeypatch.setattr(agent_app, "this_python", lambda: "3.12")
    installer = machine.installer()
    app = create_app(machine.settings(engine="auto"), vllm_sets=[SET], installer=installer)
    await app.state.look()
    assert owners.poll() is None, "not the agent's to stop"
    assert json.loads(choice.read_text()) == {"engine": "vllm"}
    with pytest.raises(Refused) as refused:
        await installer.want("latest", False)
    assert refused.value.error == "foreign_engine"


# --- the launcher's `--vllm` (D159, D41) ---


def test_vllm_names_only_a_program_inside_the_agents_engines_directory(tmp_path):
    engines = tmp_path / "engines"
    inside = engines / "vllm" / "8000" / V1 / "venv" / "bin" / "vllm"
    inside.parent.mkdir(parents=True)
    inside.write_text("#!/bin/sh\n")
    outside = tmp_path / "elsewhere" / "vllm"
    outside.parent.mkdir()
    outside.write_text("#!/bin/sh\n")
    linked = engines / "vllm" / "8000" / V2 / "venv" / "bin" / "vllm"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside)

    assert vllm_launch.vllm_refusal(str(inside), engines) is None
    assert "absolute" in vllm_launch.vllm_refusal("vllm", engines)
    assert "absolute" in vllm_launch.vllm_refusal("engines/vllm/8000/0.31.0/venv/bin/vllm", engines)
    assert "only a program named 'vllm'" in vllm_launch.vllm_refusal(str(inside.with_name("python")), engines)
    assert "not inside" in vllm_launch.vllm_refusal(str(outside), engines)
    assert "not inside" in vllm_launch.vllm_refusal(str(linked), engines), "a link that leads out"
    assert "not inside" in vllm_launch.vllm_refusal(str(engines / ".." / "elsewhere" / "vllm"), engines)
    assert "not a file" in vllm_launch.vllm_refusal(str(inside.parent.parent / "missing" / "vllm"), engines)
    # The helper that starts a card's models in turn starts such a program as `vllm serve`.
    assert vllm_launch._is_vllm_serve([str(inside), "serve", "/models/acme__chat-model"])


def test_vllm_start_refuses_a_program_outside_the_engines_directory_and_starts_nothing(tmp_path, capsys, monkeypatch):
    engines = tmp_path / "engines"
    models = tmp_path / "models"
    started = []
    monkeypatch.setattr(vllm_launch, "launch", lambda *args, **kwargs: started.append(kwargs) or vllm_launch.Started())
    code = cli.main(["vllm-start", "--models-dir", str(models), "--port", "8000",
                     "--vllm", "/usr/local/bin/vllm", "--engines-dir", str(engines)])
    assert code == 2 and started == []
    assert "not inside this agent's engines directory" in capsys.readouterr().out

    inside = engines / "vllm" / "8000" / V1 / "venv" / "bin" / "vllm"
    inside.parent.mkdir(parents=True)
    inside.write_text("#!/bin/sh\n")
    assert cli.main(["vllm-start", "--models-dir", str(models), "--port", "8000",
                     "--vllm", str(inside), "--engines-dir", str(engines)]) == 0
    assert started[-1]["vllm"] == str(inside)
    assert cli.main(["vllm-start", "--models-dir", str(models), "--port", "8000"]) == 0
    assert started[-1]["vllm"] == "vllm", "the PATH's, as before"


def test_without_engines_dir_the_systems_own_place_is_the_only_one(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(vllm_launch.host_platform, "engines_dir", lambda: tmp_path / "system-engines")
    monkeypatch.setattr(vllm_launch, "launch", lambda *args, **kwargs: vllm_launch.Started())
    elsewhere = tmp_path / "engines" / "vllm" / "8000" / V1 / "venv" / "bin" / "vllm"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("#!/bin/sh\n")
    assert cli.main(["vllm-start", "--models-dir", str(tmp_path / "m"), "--port", "8000", "--vllm", str(elsewhere)]) == 2


def test_the_helpers_run_the_agent_as_the_launcher_was_run(monkeypatch):
    module_run = types.SimpleNamespace(__spec__=types.SimpleNamespace(name="gpm_agent.cli"))
    monkeypatch.setitem(sys.modules, "__main__", module_run)
    monkeypatch.setattr(sys, "flags", types.SimpleNamespace(isolated=0, safe_path=True))
    assert vllm_launch.agent_command() == ["-P", "-m", "gpm_agent.cli"]
    monkeypatch.setattr(sys, "flags", types.SimpleNamespace(isolated=1, safe_path=True))
    assert vllm_launch.agent_command() == ["-I", "-m", "gpm_agent.cli"]
    monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace(__spec__=None))
    monkeypatch.setattr(sys, "argv", ["/var/run/gpm/gpm-agent.pyz", "vllm-start"])
    assert vllm_launch.agent_command() == ["/var/run/gpm/gpm-agent.pyz"]


def test_a_listed_process_that_is_no_longer_the_launchers_is_never_signalled(tmp_path):
    """After a reboot the numbers in the launcher's list belong to the owner's own programs."""
    state = vllm_launch.state_dir_for(tmp_path)
    state.mkdir()
    (state / vllm_launch.PIDS_FILE).write_text(json.dumps([4001, 4002, 4003]))
    lines = {
        4001: b"/e/vllm/8000/0.31.0/venv/bin/vllm\0serve\0/m/a",
        4002: b"/usr/bin/python3\0-m\0gpm_agent.cli\0proxy\0--upstreams\0/m.gpm-state/u.json",
        4003: b"/usr/bin/vim\0notes.txt",
    }
    killed = []
    vllm_launch.stop_previous(tmp_path, kill=lambda pid, sig: killed.append(pid), alive=lambda pid: pid not in killed,
                              cmdline=lines.get, grace_s=0)
    assert killed == [4001, 4002]


def test_vllm_stop_with_nothing_started_is_not_an_error(tmp_path, capsys):
    assert cli.main(["vllm-stop", "--models-dir", str(tmp_path / "models")]) == 0
    assert "stopped 0 processes" in capsys.readouterr().out


# --- what the agent and the launcher share ---


def test_the_pools_numbers_are_named_as_the_launcher_reads_them():
    assert set(vllm_server.POOL_NAMES) == {name for name, _ in vllm_launch.NUMBER_FLAGS} | {vllm_launch.CARDS_PER_COPY}
    assert set(VllmFacts().launch_environment(8, 1, 4096, cards_per_copy=2)) <= set(vllm_server.POOL_NAMES)


def test_the_agent_that_answers_the_pool_never_imports_the_launcher():
    """D63: the launcher starts processes; the agent runs it as a program of its own."""
    done = subprocess.run(
        [sys.executable, "-c", "import sys, gpm_agent.app, gpm_agent.vllm_server; "
                               "print('gpm_agent.vllm_launch' in sys.modules)"],
        env={**os.environ, "PYTHONPATH": AGENT_SRC}, capture_output=True, text=True, timeout=60,
    )
    assert done.stdout.strip() == "False", done.stderr


async def test_the_audio_check_asks_the_installed_versions_own_program(machine):
    engine = VllmFacts(machine.settings())
    installer = machine.installer()
    engine.use_installer(installer)
    assert engine._command() is None, "nothing installed: nothing to ask"
    await installer.want("latest", False)
    await installer.finished()
    assert engine._command() == str(installer.program(V1))
