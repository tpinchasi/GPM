"""One Ollama server per card (S17, D190–D192).

A machine with several NVIDIA cards is given work for all of them (D107), and Ollama keeps one
loaded instance of a model — so with mostly one model the extra work queued on one card. Now the
machine runs an Ollama on each card, pinned with `CUDA_VISIBLE_DEVICES`, behind the agent's router;
a model is pulled once through the first and held on each; the machine holds a model only once
every server does. Against fakes: a fake `ollama` and a fake `nvidia-smi` as programs, the real
router and launcher as processes, and fake servers behind mock transports. No GPU, no Ollama.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from gpm_agent import ollama_cards
from gpm_agent import platform as host_platform
from gpm_agent.engines import OllamaFacts
from gpm_agent.ollama_cards import Layout, OllamaCards, plan
from gpm_agent.proxy import OLLAMA_FORWARDED_PATHS, create_app
from gpm_agent.settings import Settings, fingerprint

GIB = 1024**3
AGENT_KEY = "gpmg_" + "a" * 64
VERSION = "0.40.11"

#: The fake `ollama serve`: answers as `VERSION` on `OLLAMA_HOST`, holds what it is asked to load,
#: and writes the environment it was started with — what the per-card start is judged by — to a
#: file beside it.
FAKE_OLLAMA = r'''
import http.server, json, os, sys
host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
watched = ["OLLAMA_HOST", "CUDA_VISIBLE_DEVICES", "OLLAMA_NUM_PARALLEL", "OLLAMA_NOPRUNE", "OLLAMA_MODELS",
           "OLLAMA_KEEP_ALIVE", "OLLAMA_NO_CLOUD", "HF_TOKEN"]
with open(os.environ.get("FAKE_STARTS") or os.path.join(os.path.dirname(sys.argv[0]), "starts.jsonl"), "a") as f:
    f.write(json.dumps({"pid": os.getpid(), "args": sys.argv[1:], "env": {k: os.environ.get(k) for k in watched}}) + "\n")
try:
    # Cards whose server hangs as it starts, and never answers: "all", or card numbers.
    silent = open(os.path.join(os.path.dirname(sys.argv[0]), "silent")).read().split()
except OSError:
    silent = []
if "all" in silent or os.environ.get("CUDA_VISIBLE_DEVICES") in silent:
    import time
    time.sleep(3600)
loaded = []


class Handler(http.server.BaseHTTPRequestHandler):
    def answer(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            return self.answer({"version": "0.40.11"})
        if self.path == "/api/tags":
            return self.answer({"models": [{"name": "small:1", "size": 10}]})
        if self.path == "/api/ps":
            return self.answer({"models": [{"name": n, "size": 10, "size_vram": 10} for n in loaded]})
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        if self.path == "/api/generate":
            if body.get("model") not in loaded:
                loaded.append(body.get("model"))
            return self.answer({"response": "on " + str(os.environ.get("CUDA_VISIBLE_DEVICES")), "done": True})
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_):
        pass


http.server.ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
'''

#: The fake `nvidia-smi`: two 24 GB cards, as the driver lists them.
FAKE_SMI = r'''
import sys
field = [a for a in sys.argv if a.startswith("--query-gpu=")][0].split("=", 1)[1]
print("\n".join({"index": ["0", "1"], "memory.total": ["24564", "24564"]}.get(field, [])))
'''


def free_ports(count: int) -> int:
    """A port with the `count - 1` above it free too: the router's, and a server per card's —
    below the systems' ranges for outgoing connections (macOS from 49152, Linux from 32768), as an
    engine's port is: there any loopback connection of the suite's could hold one for a moment."""
    import random

    for _ in range(200):
        base = random.randint(20000, 32000 - count)
        try:
            for port in range(base, base + count):
                with socket.socket() as taken:
                    taken.bind(("127.0.0.1", port))
        except OSError:
            continue
        return base
    raise RuntimeError("no free ports")


def program(directory: Path, name: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)
    return path


def starts(directory: Path) -> list[dict]:
    path = directory / "starts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def manifest(models: Path, tag: str, size: int) -> None:
    """A model in Ollama's store, as its manifest names it."""
    path = ollama_cards.manifest_path(models, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "size": size}],
                                "config": {"size": 500}}))


def own_command() -> list[str]:
    return [sys.executable, "-P", "-m", "gpm_agent.cli"]


async def version_on(port: int) -> int:
    async with httpx.AsyncClient(timeout=3) as client:
        try:
            return (await client.get(f"http://127.0.0.1:{port}/api/version")).status_code
        except httpx.HTTPError:
            return 0


# --- the plan (D191) ---


def test_several_cards_holding_models_that_fit_one_get_a_server_each():
    layout = plan(["0", "1"], {"small:1": 4 * GIB}, 24 * GIB)
    assert layout.cards == ("0", "1") and "2 cards" in layout.why


def test_models_too_large_for_one_card_get_one_server_across_them():
    layout = plan(["0", "1"], {"big:1": 30 * GIB}, 24 * GIB)
    assert layout.cards == ()
    assert "big:1" in layout.why and "one server spans every card" in layout.why


def test_a_reserve_per_model_counts_as_the_vllm_launcher_counts_it():
    """Two 9 GiB models are 19.8 GiB of weights and 6 GiB of reserve: past nine tenths of 24."""
    assert plan(["0", "1"], {"a:1": 9 * GIB}, 24 * GIB).cards == ("0", "1")
    assert plan(["0", "1"], {"a:1": 9 * GIB, "b:1": 9 * GIB}, 24 * GIB).cards == ()


@pytest.mark.parametrize("cards", [[], ["0"]])
def test_one_card_or_none_is_one_server(cards):
    assert plan(cards, {"small:1": GIB}, 24 * GIB).cards == ()


def test_cards_whose_memory_cannot_be_read_are_not_split():
    assert plan(["0", "1"], {}, None).cards == ()


def test_models_not_on_disk_yet_are_counted_once_they_are():
    assert plan(["0", "1"], {"coming:1": None}, 24 * GIB).cards == ("0", "1")


def test_the_cards_are_those_the_driver_lists_or_the_environment_limits_it_to():
    smi = {"index": "0\n1\n2\n", "memory.total": "24564\n81559\n"}

    def run(command):
        return smi[command[1].split("=", 1)[1]]

    assert ollama_cards.nvidia_cards({}, run) == ["0", "1", "2"]
    assert ollama_cards.nvidia_cards({"CUDA_VISIBLE_DEVICES": "1,2"}, run) == ["1", "2"]
    assert ollama_cards.smallest_card_bytes(run) == 24564 * 1024 * 1024


def test_a_machine_without_an_nvidia_card_is_unchanged():
    """An Apple-silicon Mac, an AMD card, the processor: `nvidia-smi` is not there to answer."""
    assert ollama_cards.nvidia_cards({"CUDA_VISIBLE_DEVICES": "0,1"}, lambda command: None) == []


# --- what is on disk ---


def test_a_models_size_is_read_from_its_manifest(tmp_path):
    manifest(tmp_path, "gemma3:4b", 3_000)
    manifest(tmp_path, "someone/model:q4", 7_000)
    manifest(tmp_path, "hub.example.com/team/model:v1", 9_000)
    assert ollama_cards.size_on_disk(tmp_path, "gemma3:4b") == 3_500
    assert ollama_cards.size_on_disk(tmp_path, "someone/model:q4") == 7_500
    assert ollama_cards.size_on_disk(tmp_path, "missing:1") is None
    assert ollama_cards.tags_on_disk(tmp_path) == ["gemma3:4b", "hub.example.com/team/model:v1", "someone/model:q4"]


@pytest.mark.parametrize("tag", ["../x:1", "a/../../b:1", "a/b/c/d:1", "x:../y"])
def test_a_tag_never_names_a_path_out_of_the_store(tmp_path, tag):
    assert ollama_cards.manifest_path(tmp_path, tag) is None


# --- the router in front of them ---


def answered(body: dict) -> httpx.Response:
    """An answer as an engine streams one: the router relays every answer as it comes."""
    async def once():
        yield json.dumps(body).encode()

    return httpx.Response(200, content=once(), headers={"content-type": "application/json"})


class Servers:
    """Two Ollama servers, one per card, each holding what it is told."""

    def __init__(self, loaded=(("small:1",), ("small:1",)), down=()):
        self.loaded = [list(models) for models in loaded]
        self.down = set(down)
        self.seen: list[tuple[int, str, bytes]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            card = request.url.port - 9001
            self.seen.append((card, request.url.path, request.content))
            if card in self.down:
                raise httpx.ConnectError("refused", request=request)
            path = request.url.path
            if path == "/api/version":
                return httpx.Response(200, json={"version": VERSION})
            if path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "small:1", "size": 10}, {"name": "big:1"}]})
            if path == "/api/ps":
                return httpx.Response(200, json={"models": [
                    {"name": name, "size": 10, "size_vram": 10 if card == 0 else 4} for name in self.loaded[card]]})
            if path == "/api/show":
                return httpx.Response(200, json={"capabilities": ["completion"], "from_card": card})
            return answered({"served_by_card": card})

        return httpx.MockTransport(handler)


def router_for(tmp_path, servers: Servers) -> httpx.AsyncClient:
    listed = tmp_path / "copies.json"
    listed.write_text(json.dumps({"cards": ["0", "1"], "why": "two cards", "servers": [
        {"card": "0", "url": "http://127.0.0.1:9001"}, {"card": "1", "url": "http://127.0.0.1:9002"}]}))
    app = create_app(str(listed), engine="ollama", client=httpx.AsyncClient(transport=servers.transport()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router")


def test_the_paths_forwarded_to_ollama_are_a_closed_list():
    """Inference only — Ollama's own and the OpenAI- and Anthropic-shaped ones it serves. Its
    pull, delete, create, copy and push change the machine, and are never forwarded (D190)."""
    assert OLLAMA_FORWARDED_PATHS == {
        "/api/chat", "/api/generate", "/api/embed", "/api/embeddings", "/v1/chat/completions",
        "/v1/completions", "/v1/embeddings", "/v1/responses", "/v1/messages",
    }


async def test_a_request_goes_to_the_server_with_the_fewest_in_flight(tmp_path):
    """The point of the story: with one model, the second request runs on the second card while
    the first still runs on the first, rather than queueing behind it."""
    import asyncio

    seen: list[int] = []
    hold = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        card = request.url.port - 9001
        seen.append(card)
        if card == 0:
            await hold.wait()
        return answered({"served_by_card": card})

    listed = tmp_path / "copies.json"
    listed.write_text(json.dumps({"servers": [{"card": "0", "url": "http://127.0.0.1:9001"},
                                              {"card": "1", "url": "http://127.0.0.1:9002"}]}))
    app = create_app(str(listed), engine="ollama", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as router:
        first = asyncio.create_task(router.post("/api/chat", json={"model": "small:1", "stream": False}))
        while not seen:
            await asyncio.sleep(0)
        second = await router.post("/api/chat", json={"model": "small:1", "stream": False})
        hold.set()
        first = await first
    assert first.json()["served_by_card"] == 0 and second.json()["served_by_card"] == 1
    # And idle, the first again: none in flight.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as router:
        third = await router.post("/v1/chat/completions", json={"model": "small:1"})
    assert third.json()["served_by_card"] == 0


async def test_a_server_refusing_the_connection_is_passed_over(tmp_path):
    servers = Servers(down={0})
    async with router_for(tmp_path, servers) as router:
        answer = await router.post("/v1/chat/completions", json={"model": "small:1"})
    assert answer.status_code == 200 and answer.json()["served_by_card"] == 1


@pytest.mark.parametrize("path", ["/api/pull", "/api/delete", "/api/create", "/api/copy", "/api/push",
                                  "/api/blobs/sha256:00", "/v1/audio/transcriptions"])
async def test_nothing_that_changes_the_machine_is_forwarded(tmp_path, path):
    servers = Servers()
    async with router_for(tmp_path, servers) as router:
        answer = await router.post(path, json={"model": "small:1"})
    assert answer.status_code == 404 and answer.json()["error"]["type"] == "path_not_served"
    assert not [seen for seen in servers.seen if seen[1] == path]


async def test_a_model_is_loaded_only_once_every_server_holds_it(tmp_path):
    """D192: the pool reads `/api/ps` as what the machine holds; one card of two is not it."""
    servers = Servers(loaded=(("small:1", "other:1"), ("small:1",)))
    async with router_for(tmp_path, servers) as router:
        listed = (await router.get("/api/ps")).json()["models"]
    assert [entry["name"] for entry in listed] == ["small:1"]
    # The entry with the least on its card, so a copy put on the processor is seen.
    assert listed[0]["size_vram"] == 4


async def test_the_machine_answers_only_when_every_server_does(tmp_path):
    servers = Servers(down={1})
    async with router_for(tmp_path, servers) as router:
        version = await router.get("/api/version")
        loaded = await router.get("/api/ps")
        health = await router.get("/health")
    assert version.status_code == loaded.status_code == health.status_code == 503
    assert version.json() == {"error": "the Ollama server for card 1 did not answer"}


async def test_the_version_models_on_disk_and_details_come_through(tmp_path):
    servers = Servers()
    async with router_for(tmp_path, servers) as router:
        version = (await router.get("/api/version")).json()
        tags = (await router.get("/api/tags")).json()
        shown = await router.post("/api/show", json={"model": "small:1"})
        running = await router.get("/")
    assert version == {"version": VERSION}
    assert sorted(entry["name"] for entry in tags["models"]) == ["big:1", "small:1"]
    assert shown.status_code == 200 and shown.json()["from_card"] == 0
    assert running.text == "Ollama is running"


async def test_the_pools_headers_never_reach_a_server(tmp_path):
    seen: list[httpx.Headers] = []
    servers = Servers()
    inner = servers.transport()

    def handler(request):
        seen.append(request.headers)
        return inner.handle_request(request)

    listed = tmp_path / "copies.json"
    listed.write_text(json.dumps({"servers": [{"card": "0", "url": "http://127.0.0.1:9001"}]}))
    app = create_app(str(listed), engine="ollama", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as router:
        await router.post("/api/generate", json={"model": "small:1"}, headers={"X-GPM-Model": "small:1"})
    assert seen and all("x-gpm-model" not in headers for headers in seen)


def test_a_server_listed_off_this_machine_is_never_taken(tmp_path):
    listed = tmp_path / "copies.json"
    listed.write_text(json.dumps({"servers": [{"card": "0", "url": "http://10.0.0.5:9001"},
                                              {"card": "1", "url": "http://127.0.0.1:9002"}]}))
    from gpm_agent.proxy import OllamaServers

    assert OllamaServers(listed).servers() == [("1", "http://127.0.0.1:9002")]


# --- the agent's model work through them (D190, D192) ---


@pytest.fixture
def router_up(monkeypatch):
    """The router in front of the servers is running, as its record says (the processes are fakes
    behind a mock transport here)."""
    monkeypatch.setattr(ollama_cards, "router_running", lambda *args, **kwargs: True)


def facts_with_servers(tmp_path, servers: Servers) -> tuple[OllamaFacts, httpx.AsyncClient]:
    models = tmp_path / "models"
    state = ollama_cards.state_dir(models, 9000)
    state.mkdir(parents=True)
    (state / "copies.json").write_text(json.dumps({"cards": ["0", "1"], "why": "two cards", "servers": [
        {"card": "0", "url": "http://127.0.0.1:9001"}, {"card": "1", "url": "http://127.0.0.1:9002"}]}))
    settings = Settings(key_hash=fingerprint(AGENT_KEY), engine_url="http://127.0.0.1:9000", models_path=str(models))
    inner = servers.transport()

    def handler(request):
        if request.url.port == 9000:
            # The router: what it says for the machine, as above.
            if request.url.path == "/api/ps":
                common = set(servers.loaded[0]) & set(servers.loaded[1])
                return httpx.Response(200, json={"models": [{"name": n} for n in sorted(common)]})
            return httpx.Response(200, json={"version": VERSION, "models": []})
        if request.url.path == "/api/pull":
            servers.seen.append((request.url.port - 9001, "/api/pull", request.content))
            return httpx.Response(200, content=b'{"status":"success"}\n')
        if request.url.path in ("/api/generate", "/api/embed"):
            card = request.url.port - 9001
            servers.seen.append((card, request.url.path, request.content))
            body = json.loads(request.content)
            if body["keep_alive"] == -1 and body["model"] not in servers.loaded[card]:
                servers.loaded[card].append(body["model"])
            if body["keep_alive"] != -1 and body["model"] in servers.loaded[card]:
                servers.loaded[card].remove(body["model"])
            return httpx.Response(200, json={})
        return inner.handle_request(request)

    client = httpx.AsyncClient(base_url="http://127.0.0.1:9000", transport=httpx.MockTransport(handler))
    return OllamaFacts(settings), client


async def test_a_model_is_pulled_once_through_the_first_server(tmp_path, router_up):
    servers = Servers(loaded=((), ()))
    facts, client = facts_with_servers(tmp_path, servers)
    async with client:
        async for _ in facts.pull(client, "small:1"):
            pass
    assert [(card, path) for card, path, _ in servers.seen if path == "/api/pull"] == [(0, "/api/pull")]


async def test_a_pinned_model_is_held_on_every_server(tmp_path, router_up):
    servers = Servers(loaded=((), ()))
    facts, client = facts_with_servers(tmp_path, servers)
    async with client:
        await facts.hold(client, "small:1", pinned=True)
        described = await facts.describe(client)
    assert servers.loaded == [["small:1"], ["small:1"]]
    assert described["models_loaded"] == ["small:1"]
    assert described["layout"] == {"cards": ["0", "1"], "servers": 2, "why": "two cards", "running": True}
    assert [copy["models_loaded"] for copy in described["copies"]] == [["small:1"], ["small:1"]]


async def test_a_release_goes_only_to_the_servers_that_hold_it(tmp_path, router_up):
    """Releasing a model a server does not hold would load it there (D192)."""
    servers = Servers(loaded=(("small:1",), ()))
    facts, client = facts_with_servers(tmp_path, servers)
    async with client:
        described = await facts.describe(client)
        await facts.hold(client, "small:1", pinned=False)
    assert described["models_loaded"] == [] and described["models_loaded_anywhere"] == ["small:1"]
    loads = [(card, path) for card, path, _ in servers.seen if path in ("/api/generate", "/api/embed")]
    assert loads == [(0, "/api/generate")]
    assert servers.loaded == [[], []]


async def test_a_pin_held_by_one_server_is_still_released(tmp_path, router_up):
    """A model left by the set, pinned by this agent, held on one card of two: released there —
    not forgotten as an engine restart's doing, which would keep it on that card for good."""
    from gpm_agent.models import ModelWork

    servers = Servers(loaded=(("old:1",), ()))
    facts, client = facts_with_servers(tmp_path, servers)
    async with client:
        work = ModelWork(facts, client, free_disk_bytes=lambda: 10**12, min_free_disk_gb=0, allow_delete=False)
        work.pinned_by_agent = {"old:1"}
        await work.apply({"tags": [], "residency": "pinned"})
        await work._task
    assert servers.loaded == [[], []]
    assert work.pinned_by_agent == set()


async def test_one_server_is_worked_with_as_before(tmp_path):
    """No servers recorded: the engine URL is the one server, and nothing else is asked."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"version": VERSION, "models": []})

    settings = Settings(key_hash=fingerprint(AGENT_KEY), engine_url="http://127.0.0.1:9000",
                        models_path=str(tmp_path / "models"))
    async with httpx.AsyncClient(base_url="http://127.0.0.1:9000", transport=httpx.MockTransport(handler)) as client:
        described = await OllamaFacts(settings).describe(client)
        await OllamaFacts(settings).hold(client, "small:1", pinned=True)
    assert "layout" not in described and "copies" not in described
    assert all(url.startswith("http://127.0.0.1:9000/") for url in seen)
    assert not ollama_cards.state_dir(tmp_path / "models", 9000).exists()


# --- the agent's own runner, on a machine its owner gives it ---


class Machine:
    """A machine with two cards (as told), a fake `ollama` on its PATH, and the ports to run on."""

    def __init__(self, root: Path):
        self.root = root
        self.port = free_ports(3)
        self.bin = root / "bin"
        self.ollama = program(self.bin, "ollama", FAKE_OLLAMA)
        self.made: list[OllamaCards] = []
        self.clients: list[httpx.AsyncClient] = []

    def cards(self, cards=("0", "1"), card_gib=24, held=()) -> OllamaCards:
        """The agent's runner; `held` what the pool told it to hold — None for not told yet."""
        client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=10)
        self.clients.append(client)
        runner = OllamaCards(
            self.root / "engines" / "ollama" / str(self.port), engine_url=f"http://127.0.0.1:{self.port}",
            models_path=str(self.root / "models"), client=client,
            cards=lambda: list(cards), card_bytes=lambda: card_gib * GIB, command=own_command(),
        )
        runner.held = lambda: held
        self.made.append(runner)
        return runner

    async def close(self) -> None:
        for runner in self.made:
            try:
                await runner.stop()
            except OSError:
                pass
        for record in starts(self.bin):
            # Only a process still the fake: a number from the log may have gone to another.
            said = host_platform.command_line(record["pid"]) or b""
            if host_platform.alive(record["pid"]) and str(self.ollama).encode() in said:
                try:
                    os.kill(record["pid"], 9)
                except OSError:
                    pass
        for client in self.clients:
            await client.aclose()


@pytest.fixture
async def machine(tmp_path):
    made = Machine(tmp_path)
    yield made
    await made.close()


async def test_a_server_runs_on_each_card_behind_the_router(machine):
    runner = machine.cards()
    runner.set_pool_environment({"OLLAMA_NUM_PARALLEL": "8", "OLLAMA_MAX_LOADED_MODELS": "1"})
    try:
        assert await runner.start(machine.ollama, VERSION, 30) is None
        started = starts(machine.bin)
        assert sorted(start["env"]["CUDA_VISIBLE_DEVICES"] for start in started) == ["0", "1"]
        by_card = {start["env"]["CUDA_VISIBLE_DEVICES"]: start["env"] for start in started}
        # The host's eight workers shared between its servers; one store; only the first prunes.
        assert by_card["0"]["OLLAMA_NUM_PARALLEL"] == by_card["1"]["OLLAMA_NUM_PARALLEL"] == "4"
        assert by_card["0"]["OLLAMA_MODELS"] == by_card["1"]["OLLAMA_MODELS"] == str(machine.root / "models")
        assert by_card["0"]["OLLAMA_NOPRUNE"] is None and by_card["1"]["OLLAMA_NOPRUNE"] == "1"
        assert {by_card[c]["OLLAMA_HOST"] for c in "01"} == {f"127.0.0.1:{machine.port + 1}", f"127.0.0.1:{machine.port + 2}"}
        layout = ollama_cards.read_layout(machine.root / "models", machine.port)
        assert [server["card"] for server in layout["servers"]] == ["0", "1"]
        # The router answers for the machine on the engine's port.
        assert await version_on(machine.port) == 200
        assert runner.ours() and runner.running_version() == VERSION and not await runner.foreign()
        assert not runner.wants_restart()
    finally:
        await runner.stop()
    assert await version_on(machine.port) == 0
    assert await version_on(machine.port + 1) == 0 and await version_on(machine.port + 2) == 0
    assert ollama_cards.read_layout(machine.root / "models", machine.port) is None
    assert not runner.ours()


async def test_a_card_whose_server_died_has_the_whole_layout_started_again(machine):
    runner = machine.cards()
    try:
        assert await runner.start(machine.ollama, VERSION, 30) is None
        copy = runner._copies()[1]
        os.kill(copy._current()["pid"], 9)
        deadline = time.monotonic() + 10
        while runner.ours(fresh=True) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not runner.ours(fresh=True)
        assert not await runner.foreign(), "the router is its own: a restart, not a stranger on the port"
        assert await runner.start(machine.ollama, VERSION, 30) is None
        assert runner.ours(fresh=True) and len(starts(machine.bin)) == 4
    finally:
        await runner.stop()


async def test_models_too_large_for_one_card_run_on_one_server_and_say_why(machine):
    manifest(machine.root / "models", "big:1", 30 * GIB)
    runner = machine.cards(held=["big:1"])
    try:
        assert await runner.start(machine.ollama, VERSION, 30) is None
        (only,) = starts(machine.bin)
        assert only["env"]["CUDA_VISIBLE_DEVICES"] is None and only["env"]["OLLAMA_HOST"] == f"127.0.0.1:{machine.port}"
        layout = ollama_cards.read_layout(machine.root / "models", machine.port)
        assert layout["servers"] == [] and "one server spans every card" in layout["why"]
        # The set changes to one that fits: the layout is to change with it (D191).
        runner.held = lambda: None
        assert not runner.wants_restart(), "not on a guess, before the pool has said"
        runner.held = lambda: ["small:1"]
        assert runner.wants_restart()
    finally:
        await runner.stop()


async def test_one_card_runs_the_one_server_as_before(machine):
    runner = machine.cards(cards=("0",))
    try:
        assert await runner.start(machine.ollama, VERSION, 30) is None
        (only,) = starts(machine.bin)
        assert only["env"]["CUDA_VISIBLE_DEVICES"] is None
        assert (machine.root / "engines" / "ollama" / str(machine.port) / "server.json").exists()
        assert not ollama_cards.state_dir(machine.root / "models", machine.port).exists()
        assert not runner.wants_restart()
    finally:
        await runner.stop()


async def test_servers_per_card_that_cannot_start_leave_one_server_serving(machine):
    """A card's port taken by something else: one server, saying why — and not tried again every
    pass while nothing has changed."""
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", machine.port + 2))
        taken.listen()
        runner = machine.cards(held=[])
        try:
            assert await runner.start(machine.ollama, VERSION, 30) is None
            assert len(starts(machine.bin)) == 1
            layout = ollama_cards.read_layout(machine.root / "models", machine.port)
            assert layout["servers"] == [] and f"port {machine.port + 2}" in layout["why"]
            assert not runner.wants_restart()
        finally:
            await runner.stop()


# --- the launcher, on a machine the pool rents ---


def launcher(machine, verb: str, *more: str) -> subprocess.CompletedProcess:
    program(machine.bin, "nvidia-smi", FAKE_SMI)
    env = {**os.environ, "PATH": f"{machine.bin}{os.pathsep}{os.environ['PATH']}",
           "OLLAMA_NUM_PARALLEL": "6", "HF_TOKEN": "hf_not_a_real_token"}
    return subprocess.run(
        [*own_command(), verb, "--port", str(machine.port), "--models-dir", str(machine.root / "models"), *more],
        env=env, capture_output=True, text=True, timeout=120,
    )


async def test_the_launcher_starts_a_server_per_card_and_stops_them(machine):
    try:
        done = launcher(machine, "ollama-start", "--pinned")
        assert done.returncode == 0, done.stderr
        started = starts(machine.bin)
        assert sorted(start["env"]["CUDA_VISIBLE_DEVICES"] for start in started) == ["0", "1"], done.stdout
        for start in started:
            env = start["env"]
            assert env["OLLAMA_NUM_PARALLEL"] == "3" and env["OLLAMA_KEEP_ALIVE"] == "-1"
            assert env["OLLAMA_NO_CLOUD"] == "1" and env["OLLAMA_MODELS"] == str(machine.root / "models")
            assert env["HF_TOKEN"] is None, "the hub's token never reaches an engine (D184)"
        assert await version_on(machine.port) == 200
        async with httpx.AsyncClient(timeout=5) as client:
            answer = await client.post(f"http://127.0.0.1:{machine.port}/api/generate",
                                       json={"model": "small:1", "stream": False})
        assert answer.status_code == 200 and answer.json()["response"] in ("on 0", "on 1")
        # Started again — a resize — what ran is stopped first.
        again = launcher(machine, "ollama-start", "--pinned")
        assert again.returncode == 0, again.stderr
        assert len(starts(machine.bin)) == 4 and await version_on(machine.port) == 200
    finally:
        stopped = launcher(machine, "ollama-stop")
    assert stopped.returncode == 0, stopped.stderr
    assert await version_on(machine.port) == 0 and await version_on(machine.port + 1) == 0


async def test_the_launcher_never_starts_over_an_ollama_it_did_not_start(machine):
    stranger = subprocess.Popen([sys.executable, str(machine.ollama), "serve"],
                                env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{machine.port}",
                                     "FAKE_STARTS": str(machine.root / "stranger.jsonl")})
    try:
        deadline = time.monotonic() + 10
        while await version_on(machine.port) != 200:
            assert time.monotonic() < deadline
            time.sleep(0.1)
        done = launcher(machine, "ollama-start", "--pinned")
        assert done.returncode == 1 and "did not start" in done.stderr
        assert starts(machine.bin) == []
        layout = ollama_cards.read_layout(machine.root / "models", machine.port)
        assert layout["servers"] == [] and "did not start" in layout["why"]
    finally:
        stranger.kill()
        stranger.wait(timeout=10)


def test_the_launcher_is_reached_only_from_the_command_line():
    """Its verbs are the restart script's, never the protocol's: the agent's verbs are unchanged."""
    from gpm_agent import cli

    assert {"ollama-start", "ollama-stop"} <= set(cli.LAUNCHER_VERBS)
    assert Layout((), "x").cards == ()


# --- the pool: its start on a rented host, and asking for it once (D190) ---


def test_the_pools_start_runs_the_launcher_on_several_cards_and_the_one_server_otherwise():
    from gpm_server.engines import get_engine

    start = get_engine("ollama").default_start_command(port=11434, models_dir="/m", agent_archive="/a.pyz", proxy=False)
    head, single = start.split(" else ")
    assert head.startswith("if [ -f /a.pyz ] && ") and "nvidia-smi -L" in head and "-gt 1" in head
    assert "python3 /a.pyz ollama-start --port 11434 --pinned; then :;" in head
    # The one server where the launcher does not run — or could start nothing.
    assert 'OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}" nohup ollama serve' in single and single.endswith("& fi")
    # And forgets any servers per card recorded before: a machine started again keeps its disk only.
    assert single.startswith('rm -f "$HOME/.ollama/models.gpm-state/ollama-11434/copies.json"; ')
    done = subprocess.run(["sh", "-n", "-c", start], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.fixture
def ollama_fleet(tmp_path):
    from gpm_server.config import PoolConfig
    from gpm_server.db import Database
    from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
    from gpm_server.providers import FakeProvider
    from gpm_server.supervisor.renting import Fleet

    databases = []

    def build(sizes=None, **rented):
        sizes = {"size_gb": 4.0} if sizes is None else sizes
        config = PoolConfig.model_validate({
            "pool": {"name": "t", "model_set": ["chat"], "models_per_host": "all"},
            "auth": {"app_keys": ["k"]},
            "engine": "ollama",
            "catalog": {"chat": {"variants": [{"tag": "chat:1", "engine": "ollama", **sizes}]}},
            "hosts": [],
            "rented": {
                "provider": "fake", "workers": 6, "capabilities": ["cuda"], "engine": "ollama",
                "image": "ollama/ollama:0.40.15",
                "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 2.0}, "bidding": {"premium": 0.02},
                **rented,
            },
        })
        database = Database(tmp_path / f"gpm{len(databases)}.sqlite3")
        databases.append(database)
        return Fleet(config, config.rented, FakeProvider(offers=[]), LeaseStore(database), EventLog(database),
                     SpendLedger(database))

    yield build
    for database in databases:
        database.close()


def rented_host(cards: int = 2, facts=None, state: str = "preparing"):
    from gpm_server.providers import default_offer
    from gpm_server.supervisor import hostagent
    from gpm_server.supervisor.renting import RentedHost

    host = RentedHost(host_id="rented-1", instance=None, offer=default_offer(gpus=cards), bid_hourly=1.0,
                      lease_id="lease-1", state=state, workers=12, launch_workers=12)
    host.agent = hostagent.RentedAgent(url="http://agent", secret=AGENT_KEY)
    host.agent_facts = facts if facts is not None else {
        "accelerators": [{"kind": "cuda"} for _ in range(cards)], "engine": {"name": "ollama", "answers": True}}
    return host


def agent_answering(fleet, answer=None) -> list:
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=answer or {"engine_answers": True, "restart_exit_code": 0})

    fleet._agent_transport = httpx.MockTransport(handler)
    return asked


async def test_a_rented_ollama_host_with_two_cards_is_started_again_per_card_once(ollama_fleet):
    fleet = ollama_fleet()
    asked = agent_answering(fleet)
    host = rented_host()
    await fleet.start_per_card(host)
    await fleet.start_per_card(host)
    # The host's own numbers, as any restart: the workers it was launched for, its set, its context.
    assert asked == [("/agent/v1/engine", {"settings": {"workers": 12, "models_held": 1, "context": 8192}})]
    kinds = [event["kind"] for event in fleet.events.recent(limit=20)]
    assert "engine_per_card" in kinds and "engine_per_card_failed" not in kinds


@pytest.mark.parametrize("why", ["one card", "launcher ran", "serving", "no agent", "operator start"])
async def test_a_host_that_should_not_be_restarted_is_not(ollama_fleet, why):
    fleet = ollama_fleet(**({"engine_start": "nohup ollama serve &"} if why == "operator start" else {}))
    asked = agent_answering(fleet)
    host = rented_host(cards=1 if why == "one card" else 2, state="ready" if why == "serving" else "preparing")
    if why == "launcher ran":
        # Adopted after a supervisor restart: its agent says what the launcher chose.
        host.agent_facts["engine"]["layout"] = {"cards": ["0", "1"], "servers": 2, "why": "two cards"}
    if why == "no agent":
        host.agent = None
    await fleet.start_per_card(host)
    assert asked == []


async def test_the_cards_the_agent_counts_win_over_the_offer(ollama_fleet):
    fleet = ollama_fleet()
    asked = agent_answering(fleet)
    host = rented_host(cards=2, facts={"accelerators": [{"kind": "cuda"}], "engine": {"name": "ollama"}})
    await fleet.start_per_card(host)
    assert asked == []


async def test_a_restart_that_does_not_come_back_is_said_and_not_asked_again(ollama_fleet):
    fleet = ollama_fleet()
    asked = agent_answering(fleet, {"engine_answers": False, "restart_exit_code": 1,
                                    "restart_output": "gpm-agent ollama-start: ollama is not on the PATH"})
    host = rented_host()
    await fleet.start_per_card(host)
    await fleet.start_per_card(host)
    assert len(asked) == 1
    failed = [event for event in fleet.events.recent(limit=20) if event["kind"] == "engine_per_card_failed"]
    assert failed and "ollama is not on the PATH" in failed[0]["summary"]


def test_the_agent_that_runs_ollama_runs_it_one_server_per_card_where_it_can(tmp_path):
    """On a machine its owner gives it, the agent's own runner is the per-card one, told the models
    the pool asks it to hold — which decide whether they fit one card (D191)."""
    from gpm_agent.app import create_app as agent_app
    from gpm_agent.models import Desired

    settings = Settings(key_hash=fingerprint(AGENT_KEY), engine="ollama", engine_url="http://127.0.0.1:11999",
                        models_path=str(tmp_path / "models"), install_engines=True,
                        engines_path=str(tmp_path / "engines"))
    app = agent_app(settings)
    server = app.state.installer.server
    assert isinstance(server, OllamaCards)
    assert server.held() is None, "until the pool has said"
    app.state.work.desired = Desired(("small:1",), "pinned")
    assert tuple(server.held()) == ("small:1",)


# --- the review's findings ---


def test_only_the_agent_itself_matches_what_a_reinstall_stops():
    """Finding 1: the install's `pkill -f` stopped every program of the archive whose command line
    held "serve" — the router's `--upstreams …/servers.json` among them — on every reinstall."""
    import re

    from gpm_server.supervisor import hostagent

    pattern = hostagent.AGENT_PROCESS
    archive, models = hostagent.ARCHIVE, "/root/.ollama/models.gpm-state/ollama-11434"
    assert re.search(pattern, f"python3 {archive} -c {hostagent.SETTINGS} serve")
    assert re.search(pattern, f"/usr/bin/python3 {archive} -c {hostagent.SETTINGS} serve")
    for other in (
        f"/usr/bin/python3 {archive} proxy --engine ollama --upstreams {models}/copies.json --port 11434",
        f"/usr/bin/python3 {archive} proxy --engine ollama --upstreams {models}/servers.json --port 11434",
        f"python3 {archive} ollama-start --port 11434 --pinned",
        f"python3 {archive} vllm-start --models-dir /opt/gpm/models --port 8000 --proxy",
        f"python3 {archive} -c {hostagent.SETTINGS} serve-ish",
    ):
        assert not re.search(pattern, other), other
    assert "serve" not in ollama_cards.COPIES_FILE


async def test_a_record_whose_router_is_not_running_is_never_dialled(tmp_path):
    """Finding 3: the record outlives its processes (a machine started again keeps its disk), and
    a pull sent to a port nothing listens on can never land."""
    servers = Servers(loaded=((), ()))
    facts, client = facts_with_servers(tmp_path, servers)
    async with client:
        assert facts.copies() == []
        described = await facts.describe(client)
        async for _ in facts.pull(client, "small:1"):
            pass
    assert described["layout"]["running"] is False and "copies" not in described
    assert not [seen for seen in servers.seen if seen[1] == "/api/pull"], "pulled through the engine URL"


async def test_a_router_is_running_only_as_its_record_says(machine):
    runner = machine.cards()
    try:
        assert await runner.start(machine.ollama, VERSION, 30) is None
        assert ollama_cards.router_running(machine.root / "models", machine.port)
        record = runner._router()._current()
        os.kill(record["pid"], 9)
        deadline = time.monotonic() + 10
        while ollama_cards.router_running(machine.root / "models", machine.port):
            assert time.monotonic() < deadline
            time.sleep(0.1)
        layout = ollama_cards.read_layout(machine.root / "models", machine.port)
        assert layout["servers"] and layout["running"] is False
        assert ollama_cards.copy_urls(machine.root / "models", machine.port) == []
    finally:
        await runner.stop()


async def test_before_the_pool_has_said_models_that_do_not_fit_are_not_split(machine):
    """Finding 2: before the pool's first pass the plan saw no models and split the cards — a set
    too large for one was then pinned partly on the processor, and restarted."""
    manifest(machine.root / "models", "big:1", 30 * GIB)
    runner = machine.cards(held=None)
    try:
        assert await runner.start(machine.ollama, VERSION, 30) is None
        assert len(starts(machine.bin)) == 1
        assert "one server spans every card" in ollama_cards.read_layout(machine.root / "models", machine.port)["why"]
    finally:
        await runner.stop()


async def test_with_nothing_known_one_server_runs_until_the_pool_says(machine):
    runner = machine.cards(held=None)
    try:
        assert await runner.start(machine.ollama, VERSION, 30) is None
        assert len(starts(machine.bin)) == 1
        assert "until the pool says" in runner.why
        assert not runner.wants_restart()
        runner.held = lambda: ["small:1"]
        assert runner.wants_restart(), "told a set that fits: a server on each card"
    finally:
        await runner.stop()


async def test_the_one_server_has_a_wait_of_its_own_after_cards_that_hung(machine):
    """Finding 4: a card's server that hung to the end of the shared wait left the one server a
    second to start in."""
    (machine.bin / "silent").write_text("1")
    runner = machine.cards()
    try:
        assert await runner.start(machine.ollama, VERSION, 4) is None
        layout = ollama_cards.read_layout(machine.root / "models", machine.port)
        assert layout["servers"] == [] and "card 1" in layout["why"]
        assert await version_on(machine.port) == 200
    finally:
        await runner.stop()


async def test_a_launcher_that_could_not_start_stops_what_it_started(machine):
    """Finding 4: a launcher that gave up left its servers starting, racing the restart script's
    one server for the port."""
    (machine.bin / "silent").write_text("all")
    done = launcher(machine, "ollama-start", "--pinned", "--timeout", "2")
    assert done.returncode == 1
    started = starts(machine.bin)
    assert len(started) == 3, "two cards' servers, then the one server"
    deadline = time.monotonic() + 10
    while any(host_platform.alive(record["pid"]) for record in started):
        assert time.monotonic() < deadline, "the launcher left a server running"
        time.sleep(0.1)


def test_the_launcher_gives_up_inside_the_restart_commands_time():
    """Both tries, after the restart script's 30 s for the engine before, inside the 120 s the agent
    gives the pool's restart command."""
    assert 30 + 2 * ollama_cards.START_TIMEOUT_S < Settings(key_hash="x").restart_timeout_s


@pytest.mark.parametrize("sizes, why", [({}, "is not known"), ({"size_gb": 60.0}, "need")])
async def test_a_set_that_cannot_be_shown_to_fit_one_card_stays_one_server(ollama_fleet, sizes, why):
    """Finding 5: on a rented machine nothing is on disk when the engine is started again, so the
    pool decides — every size known, and the set within the smallest card (D191)."""
    fleet = ollama_fleet(sizes=sizes)
    asked = agent_answering(fleet)
    host = rented_host()
    await fleet.start_per_card(host)
    assert asked == []
    said = [event for event in fleet.events.recent(limit=20) if event["kind"] == "engine_one_server"]
    assert said and why in said[0]["summary"]


async def test_the_agents_card_memory_wins_over_the_offers(ollama_fleet):
    fleet = ollama_fleet(sizes={"size_gb": 20.0})
    asked = agent_answering(fleet)
    host = rented_host(facts={"accelerators": [{"kind": "cuda", "memory_bytes": 16 * GIB}] * 2,
                              "engine": {"name": "ollama"}})
    await fleet.start_per_card(host)
    assert asked == []


async def test_a_router_found_not_running_has_the_engine_started_again_a_few_times(ollama_fleet):
    """Finding 1's other half: nothing started a dead router again."""
    fleet = ollama_fleet()
    asked = agent_answering(fleet)
    host = rented_host(state="ready")
    host.per_card_asked = True
    host.agent_facts["engine"]["layout"] = {"cards": ["0", "1"], "servers": 2, "why": "two cards", "running": False}
    for _ in range(5):
        fleet.recover_per_card(host)
        if host.restart_task is not None:
            await host.restart_task
        if host.per_card_recovered_at is not None:
            host.per_card_recovered_at -= fleet.RESTART_RETRY_S + 1
    assert len(asked) == fleet.PER_CARD_RECOVERIES == 3
    host.per_card_recoveries = 0
    host.agent_facts["engine"]["layout"]["running"] = True
    fleet.recover_per_card(host)
    assert len(asked) == 3, "a router running is left alone"
