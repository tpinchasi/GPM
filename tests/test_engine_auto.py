"""A host whose agent chooses its engine (D161), and an engine its owner installed started once its
models are on disk (D160) — S10, docs/spec/supervisor.md §3.2.

Against a fake agent that says what the real one says — its choice and why, the models it holds,
whether it may restart the engine — and a fake vLLM that serves what it was started with.
"""

import json

import httpx
import pytest
from fakes.harness import EngineSpec, pool_harness
from gpm_server.config import PoolConfig

MODEL = "chat"
VLLM_BUILD, OLLAMA_BUILD = "org/chat-bf16", "chat:7b"
CATALOG = {MODEL: {"variants": [
    {"tag": VLLM_BUILD, "engine": "vllm"},
    {"tag": OLLAMA_BUILD, "engine": "ollama"},
]}}
AGENT = {"url": "http://127.0.0.1:8095", "bearer_env": "TEST_AGENT_KEY"}


class Agent:
    """The host's agent as the pool hears it."""

    def __init__(self, engine, *, chosen=None, reasons=(), installs=True, restart=False, on_disk=()):
        self.engine, self.chosen, self.reasons = engine, chosen, list(reasons)
        self.installs, self.restart = installs, restart
        self.on_disk = set(on_disk)
        self.held: list[list[str]] = []
        self.restarts: list[dict] = []
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request):
        path = request.url.path
        if path.endswith("/facts"):
            engine = {"name": self.chosen or "auto", "answers": bool(self.engine.served)}
            if self.chosen:
                engine["choice"] = {"engine": self.chosen, "reasons": self.reasons}
            engine["install"] = ({"allowed": True, "engine": self.chosen, "running": "0.32.0", "staged": None,
                                  "previous": None, "working": None, "error": None}
                                 if self.installs else {"allowed": False, "engine": self.chosen})
            return httpx.Response(200, json={
                "agent": {"version": "test", "protocol": "1"}, "os": "Linux", "arch": "x86_64",
                "capabilities": None, "engine": engine,
                "allows": {"delete": False, "restart": self.restart, "engine_settings": self.restart},
            })
        if path.endswith("/models"):
            tags = json.loads(request.content)["tags"]
            self.held.append(tags)
            return httpx.Response(200, json={"engine_answers": bool(self.engine.served), "surplus": [], "models": [
                {"tag": t, "on_disk": t in self.on_disk, "loaded": t in self.engine.served,
                 "awaiting_restart": t in self.on_disk and t not in self.engine.served}
                for t in tags
            ]})
        if path.endswith("/engine"):
            self.restarts.append(json.loads(request.content))
            self.engine.start(self.on_disk)
            return httpx.Response(200, json={"restart_exit_code": 0, "engine_answers": True})
        if path.endswith("/engine/version"):
            return httpx.Response(200, json={"allowed": True, "engine": self.chosen, "running": "0.32.0"})
        return httpx.Response(404, json={"error": "not_found"})


def auto_pool(monkeypatch, **override):
    monkeypatch.setenv("TEST_AGENT_KEY", "test-agent-key")
    return pool_harness(
        [EngineSpec(id="box", engine="vllm", resident=set())],
        model_set=[MODEL], catalog=CATALOG,
        host_overrides={"box": {"engine": "auto", "agent": AGENT, **override}},
    )


def kinds(pool):
    return [e["kind"] for e in reversed(pool.supervisor.events.recent(50))]


# --- configuration ---


def config(hosts, **pool):
    return PoolConfig.model_validate({
        "pool": {"name": "p", "model_set": [MODEL]}, "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]},
        "catalog": CATALOG, "hosts": hosts, **pool,
    })


def host(**extra):
    return {"id": "box", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:8000"},
            "engine": "auto", "agent": AGENT, **extra}


def test_an_auto_host_runs_either_engine_and_the_router_knows_both():
    loaded = config([host()])
    assert loaded.engines_of(loaded.hosts[0]) == ("vllm", "ollama")
    assert {"vllm", "ollama"} <= set(loaded.engines_in_use())


def test_auto_is_chosen_by_an_agent_so_a_host_without_one_is_refused():
    with pytest.raises(ValueError, match="no `agent`"):
        config([{**host(), "agent": None}])


def test_each_model_an_auto_host_holds_needs_a_build_for_both_engines():
    only_vllm = {MODEL: {"variants": [{"tag": VLLM_BUILD, "engine": "vllm"}]}}
    with pytest.raises(ValueError, match="needs a build for both"):
        PoolConfig.model_validate({
            "pool": {"name": "p", "model_set": [MODEL]}, "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]},
            "catalog": only_vllm, "hosts": [host()],
        })


def test_an_auto_host_cannot_pin_a_version_of_an_engine_not_yet_chosen():
    with pytest.raises(ValueError, match="takes `engine_version: latest` only"):
        config([host(engine_version="0.32.0")])
    assert config([host(engine_version="latest")]).hosts[0].engine_version == "latest"


def test_auto_is_a_hosts_not_the_pools_nor_the_rented_machines():
    with pytest.raises(ValueError, match="pool's `engine` cannot be `auto`"):
        config([host(engine="ollama", agent=None)], engine="auto")


# --- the pool takes the agent's choice ---


def test_an_auto_host_waits_for_its_agents_choice_and_holds_nothing_meanwhile(monkeypatch):
    with auto_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen=None)
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        box = pool.supervisor.hosts["box"]
        assert box.state.value == "preparing" and "choosing its engine" in box.last_error
        assert agent.held == [], "an agent told to hold nothing may remove what it has"


def test_an_auto_host_whose_agent_does_not_install_engines_says_it_never_will_choose(monkeypatch):
    with auto_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen=None, installs=False)
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        box = pool.supervisor.hosts["box"]
        assert box.state.value == "preparing" and "does not install engines" in box.last_error
        assert agent.held == []


def test_the_engine_its_agent_chose_is_published_and_its_builds_held(monkeypatch):
    with auto_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", on_disk={VLLM_BUILD})
        pool.engines["box"].fake.start({VLLM_BUILD})
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        box = pool.supervisor.hosts["box"]
        assert box.chosen_engine == "vllm" and agent.held[-1] == [VLLM_BUILD]
        assert box.state.value == "ready"
        row = next(r for r in pool.supervisor.table.all() if r.host_id == "box")
        assert row.engine == "vllm"
        assert "engine_chosen" in kinds(pool)


def test_a_machine_that_cannot_run_vllm_is_held_to_ollamas_builds_and_says_why(monkeypatch):
    with auto_pool(monkeypatch) as pool:
        reasons = ["GeForce RTX 2080 has compute capability 7.5; this needs 8.0 or higher"]
        agent = Agent(pool.engines["box"].fake, chosen="ollama", reasons=reasons)
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        assert agent.held[-1] == [OLLAMA_BUILD]
        chosen = next(e for e in pool.supervisor.events.recent(20) if e["kind"] == "engine_chosen")
        assert "runs ollama" in chosen["summary"] and "compute capability 7.5" in chosen["summary"]


# --- an engine its owner installed, started once its models are on disk (D160) ---


def owners_vllm_pool(monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "test-agent-key")
    return pool_harness(
        [EngineSpec(id="box", engine="vllm", resident=set())],
        model_set=[MODEL], catalog=CATALOG,
        host_overrides={"box": {"agent": {**AGENT, "manage_models": True}}},
    )


def test_an_owner_installed_vllm_is_started_once_its_models_are_on_disk(monkeypatch):
    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True, on_disk={VLLM_BUILD})
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        task = pool.supervisor.hosts["box"].restart_task

        async def finished():
            await task

        pool.loop.run(finished())
        assert len(agent.restarts) == 1 and agent.restarts[0]["settings"]["models_held"] == 1
        assert "engine_restart" in kinds(pool)
        pool.reprobe()
        pool.reprobe()
        assert len(agent.restarts) == 1, "once per set on disk"
        assert pool.supervisor.hosts["box"].state.value == "ready"


def test_nothing_is_started_while_a_model_is_still_missing_or_without_the_owners_command(monkeypatch):
    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True, on_disk=set())
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        assert agent.restarts == []
        agent.on_disk, agent.restart = {VLLM_BUILD}, False
        pool.reprobe()
        assert agent.restarts == [], "no restart command: nothing the pool could ask for"


def test_an_engine_the_agent_runs_itself_is_never_restarted_by_the_pool(monkeypatch):
    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=True, restart=True, on_disk={VLLM_BUILD})
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        pool.reprobe()
        assert agent.restarts == [], "its agent starts it (D159)"
