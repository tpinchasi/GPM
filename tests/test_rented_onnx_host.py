"""A rented machine for a trained model (S15, D179): rented from a plain Python image, its agent
installed with `--install-engines`, told to install ONNX Runtime, told what the build takes in and
gives back, and only then ready — its readiness read from what its agent says the server loaded.

The real supervisor, fake provider and router; the machine's agent is a stand-in that starts the
server when it is told to hold its build."""

import json
import time

import httpx
from fakes.harness import EngineSpec, pool_harness
from gpm_server.supervisor import hostagent
from test_trained_models import CHAT, CLASSIFIER, ONNX_BUILD, ONNX_FILE, SPEC

AGENT_KEY = "gpmg_" + "c" * 64


class MachineAgent:
    """What the pool hears from the rented machine's agent."""

    def __init__(self, server):
        self.server = server
        self.asked: list[tuple[str, dict]] = []
        self.loaded: list[str] = []
        self.transport = httpx.MockTransport(self.answer)

    def answer(self, request):
        body = json.loads(request.content) if request.content else None
        self.asked.append((request.url.path, body))
        if request.url.path.endswith("/facts"):
            return httpx.Response(200, json={"agent": {"version": "t", "protocol": "1"}, "os": "Linux", "arch": "x86_64",
                                             "capabilities": [], "engine": {"name": "onnx", "answers": bool(self.loaded),
                                                                            "models_loaded": self.loaded}})
        if request.url.path.endswith("/engine/version"):
            return httpx.Response(200, json={"allowed": True, "running": "1.32.0"})
        if request.url.path.endswith("/models"):
            # The agent installs the runtime and starts the server with what it was told it takes.
            assert body["specs"][ONNX_FILE]["spec"] == SPEC
            self.server.start(body["tags"])
            self.loaded = list(body["tags"])
            return httpx.Response(200, json={"engine_answers": True, "models": [
                {"tag": t, "on_disk": True, "loaded": True, "size_bytes": 1000} for t in body["tags"]], "surplus": []})
        return httpx.Response(404)


def test_a_rented_machine_is_bought_for_an_onnx_model_and_serves_it(monkeypatch):
    with pool_harness(
        [EngineSpec(id="mac", resident={"chatter:7b"})],
        host_overrides={"mac": {"models": [CHAT]}},
        rentable=[EngineSpec(id="market-1", resident=set(), engine="onnx")],
        model_set=[CLASSIFIER, CHAT],
        catalog={CLASSIFIER: {"variants": [ONNX_BUILD]}, CHAT: {"variants": [{"tag": "chatter:7b"}]}},
        rented={"provider": "fake", "engine": "onnx", "workers": 2, "models": [CLASSIFIER],
                "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02},
                "scale": {"scale_up_after_s": 0}, "teardown": {"idle_minutes": 10}},
        pool_settings={"models_per_host": "declared"},
    ) as pool:
        fleet = pool.supervisor.fleet
        assert pool.supervisor.config.rented_image() == "python:3.12.16-slim-bookworm"
        server = pool.rentable["market-1"].fake
        agent = MachineAgent(server)
        fleet._agent_transport = agent.transport
        fleet.agent_hold_retry_s = 0.01
        installs = []

        async def installed(host):
            if host.agent is None:
                installs.append(bool(getattr(fleet.engine_of(host), "installed_by_agent", False)))
                host.agent = hostagent.RentedAgent(url="http://agent-market-1", secret=AGENT_KEY)

        monkeypatch.setattr(fleet, "install_agent", installed)
        fleet.open_lease(workers=2, max_hours=2, max_spend=2.00, allow_rent=True)

        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            pool.reprobe()
            live = [h for h in fleet.hosts.values() if not h.released]
            if live and all(h.state == "ready" for h in live):
                break
            time.sleep(0.1)
        else:
            raise AssertionError(f"not ready: {[(h.host_id, h.state) for h in fleet.hosts.values()]}; "
                                 f"{[e['summary'] for e in pool.supervisor.events.recent(15)]}")

        (host,) = [h for h in fleet.hosts.values() if not h.released]
        assert installs == [True], "the agent was installed before the engine answered, to install it"
        paths = [path for path, _ in agent.asked]
        assert any(p.endswith("/engine/version") for p in paths), "told to install ONNX Runtime"
        version = next(body for path, body in agent.asked if path.endswith("/engine/version"))
        assert version["version"] == "latest"
        with pool.client() as client:
            answer = client.post("/v1/predict", json={"model": CLASSIFIER, "inputs": {"image": "aGk="}})
        assert answer.status_code == 200, answer.text
        assert answer.headers["X-GPM-Host"] == host.host_id
        assert server.received[-1][1]["model"] == ONNX_FILE


def test_the_agent_is_installed_to_install_its_engine():
    """What a rented ONNX machine's agent is set up with: it installs the runtime itself, and the
    pool's start-up material for an engine it does not run has no place there."""
    import asyncio
    from pathlib import Path

    ran: list[str] = []

    async def run(command):
        ran.append(command)
        if command.startswith("command -v"):
            return 0, "/usr/local/bin/python3"
        if " init " in command:
            return 0, "agent key: gpmg_" + "d" * 64
        return 0, "started"

    async def push(data, path):
        return 0, ""

    archive = Path(__file__)
    asyncio.run(hostagent.install(run=run, push=push, archive=archive, engine_port=8190, engine="onnx", installs_engine=True))
    init = next(c for c in ran if " init " in c)
    assert "--install-engines" in init and "--engine onnx" in init and "--engine-url http://127.0.0.1:8190" in init
    assert "--restart-command" not in init and "--engine-env-file" not in init


TOKEN = "hf_not_a_real_read_token"


def install_with(hub_token):
    import asyncio
    from pathlib import Path

    ran: list[str] = []
    pushed: list[tuple[str, bytes]] = []

    async def run(command):
        ran.append(command)
        if command.startswith("command -v"):
            return 0, "/usr/local/bin/python3"
        if " init " in command:
            return 0, "agent key: gpmg_" + "d" * 64
        return 0, "started"

    async def push(data, path):
        pushed.append((path, data))
        return 0, ""

    asyncio.run(hostagent.install(run=run, push=push, archive=Path(__file__), engine_port=8190, engine="onnx",
                                  installs_engine=True, hub_token=hub_token))
    return ran, pushed


def test_the_hub_token_goes_to_the_machine_as_an_owner_only_file_never_on_a_command_line():
    """D184: copied on the connection's input, read into the agent's environment as it starts."""
    ran, pushed = install_with(TOKEN)
    (data,) = [data for path, data in pushed if path == hostagent.HUB_ENV]
    assert data.decode() == f"HF_TOKEN={TOKEN}\nexport HF_TOKEN\n"
    assert "chmod 600" in hostagent.push_command(hostagent.HUB_ENV)
    assert all(TOKEN not in command for command in ran), "never on a command line the machine could list"
    serve = next(c for c in ran if " serve " in c)
    assert serve.index(f". {hostagent.HUB_ENV};") < serve.index("setsid nohup python3")

    ran, pushed = install_with(None)
    assert [path for path, _ in pushed] == [hostagent.ARCHIVE]
    assert all("hub.env" not in command for command in ran)


def test_a_machine_is_given_the_token_only_for_a_hub_file_its_agent_fetches(monkeypatch):
    with pool_harness(
        [EngineSpec(id="mac", resident={"chatter:7b"})],
        host_overrides={"mac": {"models": [CHAT]}},
        rentable=[EngineSpec(id="market-1", resident=set(), engine="onnx")],
        model_set=[CLASSIFIER, CHAT],
        catalog={CLASSIFIER: {"variants": [ONNX_BUILD]}, CHAT: {"variants": [{"tag": "chatter:7b"}]}},
        rented={"provider": "fake", "engine": "onnx", "workers": 2, "models": [CLASSIFIER],
                "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02},
                "scale": {"scale_up_after_s": 0}, "teardown": {"idle_minutes": 10}},
        pool_settings={"models_per_host": "declared"},
    ) as pool:
        fleet = pool.supervisor.fleet
        assert fleet.hub_read_token == pool.supervisor.hub_read_token
        assert pool.supervisor.hub_read_token() is None, "none saved: none placed"
        server = pool.rentable["market-1"].fake
        agent = MachineAgent(server)
        fleet._agent_transport = agent.transport
        fleet.agent_hold_retry_s = 0.01
        fleet.hub_read_token = lambda: TOKEN
        given: list = []

        async def install(**kwargs):
            given.append(kwargs.get("hub_token"))
            return AGENT_KEY

        async def forward(*_):
            return "http://agent-market-1"

        monkeypatch.setattr(hostagent, "install", install)
        monkeypatch.setattr(fleet, "_open_agent_tunnel", forward)
        fleet.open_lease(workers=2, max_hours=2, max_spend=2.00, allow_rent=True)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline and not given:
            pool.reprobe()
            time.sleep(0.1)
        assert given == [TOKEN], "an ONNX file on the hub: the machine's agent fetches it with the token"
        summaries = [e["summary"] for e in pool.supervisor.events.recent(50)]
        assert any("hub's read-only token" in s for s in summaries)
        assert all(TOKEN not in s for s in summaries)

        (host,) = [h for h in fleet.hosts.values() if not h.released]
        monkeypatch.setattr(fleet, "tags_for", lambda _host: {"upload/" + "a" * 64})
        assert fleet.hub_token_for(host) is None, "a file the pool sends needs no token"
        fleet.hub_read_token = None
        monkeypatch.setattr(fleet, "tags_for", lambda _host: {ONNX_FILE})
        assert fleet.hub_token_for(host) is None


def test_only_the_read_only_token_saved_for_this_hub_is_ever_placed(tmp_path, monkeypatch):
    from gpm_server import hubbuilds
    from gpm_server.credentials import CredentialStore

    with pool_harness([EngineSpec(id="mac", resident={"chatter:7b"})], model_set=[CHAT],
                      catalog={CHAT: {"variants": [{"tag": "chatter:7b"}]}}) as pool:
        store = CredentialStore(pool.supervisor.config.credentials_dir() / "hub")
        store.put("write", "hf_not_a_real_write_token", {"hub": hubbuilds.hub_url()})
        assert pool.supervisor.hub_read_token() is None, "the write token is never the one placed"
        store.put("read", TOKEN, {"hub": "https://another-hub.example"})
        assert pool.supervisor.hub_read_token() is None, "a token saved for another hub is not this hub's"
        store.put("read", TOKEN, {"hub": hubbuilds.hub_url()})
        assert pool.supervisor.hub_read_token() == TOKEN
