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
    """The host's agent as the pool hears it. A model is loaded once the fake engine serves it —
    which, with a load delay, is a while after a start — and waits for a start until then, as the
    real agent says of an engine its owner installed."""

    def __init__(self, engine, *, chosen=None, reasons=(), installs=True, restart=False, on_disk=(),
                 settings=None, name=None, answer=None):
        self.engine, self.chosen, self.reasons = engine, chosen, list(reasons)
        self.installs, self.restart = installs, restart
        #: Whether its owner set an `engine_env_file`, so the pool's numbers have somewhere to go.
        self.settings = restart if settings is None else settings
        #: The engine it names where it names its own (no choice), in place of `auto`.
        self.name = name
        #: How it answers a restart, in place of starting the engine and saying it answers.
        self.answer = answer
        self.on_disk = set(on_disk)
        #: Says its models wait for a start whatever the engine serves (a guard's test).
        self.claims_waiting = False
        self.reachable = True
        self.held: list[list[str]] = []
        self.restarts: list[dict] = []
        self.versions: list[dict] = []
        self.transport = httpx.MockTransport(self._answer)

    def loaded(self, tag):
        return tag in self.engine.served and getattr(self.engine, "running", True)

    def _answer(self, request):
        path = request.url.path
        if not self.reachable:
            raise httpx.ConnectError("the agent is not answering", request=request)
        if path.endswith("/facts"):
            engine = {"name": self.name or self.chosen or "auto", "answers": bool(self.engine.served)}
            if self.chosen:
                engine["choice"] = {"engine": self.chosen, "reasons": self.reasons}
            engine["install"] = ({"allowed": True, "engine": self.chosen, "running": "0.32.0", "staged": None,
                                  "previous": None, "working": None, "error": None}
                                 if self.installs else {"allowed": False, "engine": self.chosen})
            return httpx.Response(200, json={
                "agent": {"version": "test", "protocol": "1"}, "os": "Linux", "arch": "x86_64",
                "capabilities": None, "engine": engine,
                "allows": {"delete": False, "restart": self.restart, "engine_settings": self.settings},
            })
        if path.endswith("/models"):
            tags = json.loads(request.content)["tags"]
            self.held.append(tags)
            return httpx.Response(200, json={"engine_answers": bool(self.engine.served), "surplus": [], "models": [
                {"tag": t, "on_disk": t in self.on_disk, "loaded": self.loaded(t),
                 "awaiting_restart": t in self.on_disk and (self.claims_waiting or not self.loaded(t))}
                for t in tags
            ]})
        if path.endswith("/engine"):
            sent = json.loads(request.content)
            self.restarts.append(sent)
            if self.answer is not None:
                return self.answer(sent)
            if sent.get("settings") is not None and not self.settings:
                # As engine_control does where the owner set no engine_env_file.
                return httpx.Response(409, json={"error": "owner_has_not_enabled_settings",
                                                 "detail": "this machine's owner has set no engine_env_file"})
            self.engine.start(self.on_disk)
            return httpx.Response(200, json={"restart_exit_code": 0, "restart_output": "",
                                             "engine_answers": all(self.loaded(t) for t in self.on_disk)})
        if path.endswith("/engine/version"):
            self.versions.append(json.loads(request.content))
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


def finish(pool, host_id="box"):
    """Let a restart asked in the background finish — re-raising anything it raised."""
    task = pool.supervisor.hosts[host_id].restart_task
    if task is None:
        return

    async def finished():
        await task

    pool.loop.run(finished())


def events_of(pool, kind):
    return [e for e in pool.supervisor.events.recent(50) if e["kind"] == kind]


def test_it_is_asked_once_per_set_while_the_engine_loads(monkeypatch):
    """A vLLM loads for minutes after its start, and its models wait for a start meanwhile: the
    pool does not start it again, however many passes go by and however long past the retry
    interval — asking again would cut off the load."""
    with owners_vllm_pool(monkeypatch) as pool:
        engine = pool.engines["box"].fake
        engine.load_delay_s = 3600
        agent = Agent(engine, chosen="vllm", installs=False, restart=True, on_disk={VLLM_BUILD})
        pool.supervisor._agent_transport = agent.transport
        box = pool.supervisor.hosts["box"]
        pool.reprobe()
        finish(pool)
        for _ in range(3):
            box.restart_asked_at -= 10_000  # long past the retry interval
            pool.reprobe()
            finish(pool)
        assert len(agent.restarts) == 1, "once per set on disk"
        assert box.state.value != "ready"


def test_a_ready_host_is_never_restarted_whatever_its_agent_says(monkeypatch):
    """An engine that serves its set serves requests: a restart would cut them off (D41)."""
    with owners_vllm_pool(monkeypatch) as pool:
        engine = pool.engines["box"].fake
        engine.start({VLLM_BUILD})
        agent = Agent(engine, chosen="vllm", installs=False, restart=True, on_disk={VLLM_BUILD})
        agent.claims_waiting = True
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        pool.reprobe()
        assert pool.supervisor.hosts["box"].state.value == "ready"
        assert agent.restarts == []


def test_nothing_is_restarted_while_requests_are_in_flight_and_it_is_once_they_end(monkeypatch):
    from gpm_server.db import CounterRow

    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True, on_disk={VLLM_BUILD})
        pool.supervisor._agent_transport = agent.transport
        busy = {"box": CounterRow("box", busy=1, total=1, requests_served=0, failures=0, last_request_at=None)}
        monkeypatch.setattr(pool.supervisor.counters, "all", lambda: busy)
        pool.reprobe()
        pool.reprobe()
        assert agent.restarts == [], "what was already sent to it ends first"
        busy["box"] = CounterRow("box", busy=0, total=1, requests_served=1, failures=0, last_request_at=None)
        pool.reprobe()
        finish(pool)
        assert len(agent.restarts) == 1


def test_where_the_owner_set_no_engine_env_file_it_is_started_without_settings(monkeypatch):
    """`restart_command` alone: the agent refuses numbers it has nowhere to write, so asking with
    them would be refused for ever."""
    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True, settings=False,
                      on_disk={VLLM_BUILD})
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        finish(pool)
        assert agent.restarts == [{"settings": None}]
        pool.reprobe()
        assert pool.supervisor.hosts["box"].state.value == "ready"
        assert not events_of(pool, "engine_restart_failed")


def test_a_restart_whose_command_failed_is_said_in_the_agents_words_and_asked_again(monkeypatch):
    def failing(sent):
        return httpx.Response(200, json={"restart_exit_code": 1, "restart_output": "vllm.service: \x1b[31mfailed\x1b[0m",
                                         "engine_answers": False, "settings_written": False})

    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True,
                      on_disk={VLLM_BUILD}, answer=failing)
        pool.supervisor._agent_transport = agent.transport
        box = pool.supervisor.hosts["box"]
        pool.reprobe()
        finish(pool)
        said = events_of(pool, "engine_restart_failed")
        assert said and "exited 1" in said[0]["summary"] and "vllm.service:" in said[0]["summary"]
        assert "\x1b" not in said[0]["summary"], "printable only"
        pool.reprobe()
        assert len(agent.restarts) == 1, "not before the retry interval"
        box.restart_asked_at -= pool.supervisor.RESTART_RETRY_S + 1
        pool.reprobe()
        finish(pool)
        assert len(agent.restarts) == 2


def test_an_answer_that_is_not_an_object_is_a_failure_not_an_exception(monkeypatch):
    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True,
                      on_disk={VLLM_BUILD}, answer=lambda sent: httpx.Response(200, json=["not", "an", "object"]))
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        finish(pool)  # raised AttributeError inside the task before
        said = events_of(pool, "engine_restart_failed")
        assert said and "not an object" in said[0]["summary"]
        assert pool.supervisor.hosts["box"].restarted_for is None


@pytest.mark.parametrize("how", ["another restart is under way there", "the pool stopped waiting"])
def test_a_restart_not_known_to_have_failed_is_not_asked_again_until_a_long_pause(monkeypatch, how):
    """Another restart holding the agent (409 restart_in_progress), or the pool's own wait running
    out: the engine may be starting. Asked again only after a long pause, if its models still wait."""

    def busy(sent):
        return httpx.Response(409, json={"error": "restart_in_progress", "detail": "the engine is already being restarted"})

    def too_slow(sent):
        raise httpx.ReadTimeout("no answer")

    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True, on_disk={VLLM_BUILD},
                      answer=busy if how.startswith("another") else too_slow)
        pool.supervisor._agent_transport = agent.transport
        box = pool.supervisor.hosts["box"]
        pool.reprobe()
        finish(pool)
        assert box.restarted_for == frozenset({VLLM_BUILD}), "kept: not read as a failed restart"
        assert events_of(pool, "engine_restart_unconfirmed") and not events_of(pool, "engine_restart_failed")
        box.restart_asked_at -= pool.supervisor.RESTART_RETRY_S + 1
        pool.reprobe()
        assert len(agent.restarts) == 1, "not after the short retry interval"
        box.restart_asked_at -= pool.supervisor.RESTART_UNCONFIRMED_S
        box.restart_again_at -= pool.supervisor.RESTART_UNCONFIRMED_S + 1
        pool.reprobe()
        finish(pool)
        assert len(agent.restarts) == 2, "its models still wait, long after"


# --- the agent's choice, as it changes, and as the configuration changes (D161) ---


def test_a_choice_that_changes_flips_the_builds_held_and_published(monkeypatch):
    with auto_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm")
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        assert agent.held[-1] == [VLLM_BUILD]
        agent.chosen = "ollama"  # the agent started again on a machine with another card
        pool.reprobe()
        assert agent.held[-1] == [OLLAMA_BUILD]
        row = next(r for r in pool.supervisor.table.all() if r.host_id == "box")
        assert row.engine == "ollama" and [v[0] for v in row.variants[MODEL]] == [OLLAMA_BUILD]
        assert "(was vllm)" in events_of(pool, "engine_chosen")[0]["summary"]


def test_an_agent_gone_silent_keeps_its_last_choice_and_the_console_says_so(monkeypatch):
    from gpm_server.supervisor.control import _engine_choice

    with auto_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", reasons=[])
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        agent.reachable = False
        pool.reprobe()
        box = pool.supervisor.hosts["box"]
        assert box.chosen_engine == "vllm"
        row = next(r for r in pool.supervisor.table.all() if r.host_id == "box")
        assert row.engine == "vllm" and MODEL in row.variants
        assert _engine_choice(box) == {"engine": "vllm", "reasons": [], "said_now": False}


def with_engine(pool, engine):
    """The running configuration with the host's engine changed, applied as a reload applies it."""
    config = pool.supervisor.config
    hosts = [h.model_copy(update={"engine": engine}) if h.id == "box" else h for h in config.hosts]

    async def apply():
        pool.supervisor.apply_config(config.model_copy(update={"hosts": hosts}))

    pool.loop.run(apply())


def test_a_configuration_moved_to_and_from_auto_keeps_what_the_machine_runs(monkeypatch):
    with owners_vllm_pool(monkeypatch) as pool:
        engine = pool.engines["box"].fake
        engine.start({VLLM_BUILD})
        agent = Agent(engine, chosen="vllm", on_disk={VLLM_BUILD})  # an agent of `auto` on the machine
        pool.supervisor._agent_transport = agent.transport
        box = pool.supervisor.hosts["box"]
        pool.reprobe()
        assert box.state.value == "ready"
        with_engine(pool, "auto")
        assert pool.supervisor.engine_name_of(box) == "vllm" and MODEL in box.variants, "no wait for a choice it has"
        pool.reprobe()
        assert box.state.value == "ready" and agent.held[-1] == [VLLM_BUILD]
        with_engine(pool, "vllm")
        pool.reprobe()
        assert box.state.value == "ready"


def test_a_host_named_auto_whose_agent_runs_an_engine_of_its_own_says_so_and_is_not_upgraded(monkeypatch):
    """No choice in its facts — its agent was set up with `--engine ollama`: the host waited for
    ever, while the upgrade pass, reading an engine it never probes, switched versions and rolled
    back good ones."""
    with auto_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen=None, name="ollama")
        install = {"allowed": True, "engine": "ollama", "running": "0.40.12", "staged": "0.40.13",
                   "previous": None, "working": None, "error": None}
        answer = agent._answer

        def with_install(request):
            response = answer(request)
            if request.url.path.endswith("/facts"):
                body = response.json()
                body["engine"]["install"] = install
                return httpx.Response(200, json=body)
            return response

        pool.supervisor._agent_transport = httpx.MockTransport(with_install)
        pool.supervisor._started_at -= 10_000  # quiet long enough for an upgrade
        for _ in range(3):
            pool.reprobe()
        box = pool.supervisor.hosts["box"]
        assert box.state.value == "preparing"
        assert "its agent runs ollama" in box.last_error
        assert "gpm-agent init --engine auto --install-engines" in box.last_error
        assert agent.versions == [], "nothing said of an engine the pool does not take this host to run"
        assert "box" not in pool.supervisor.upgrades.active and agent.held == []


def test_a_host_named_for_one_engine_whose_agent_chose_another_holds_nothing(monkeypatch):
    with owners_vllm_pool(monkeypatch) as pool:  # the pool names vllm
        agent = Agent(pool.engines["box"].fake, chosen="ollama", reasons=["no NVIDIA card"])
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        pool.reprobe()
        box = pool.supervisor.hosts["box"]
        assert box.state.value == "preparing"
        assert "its agent chose ollama" in box.last_error and "names vllm" in box.last_error
        assert agent.held == [] and agent.versions == [], "not the wrong engine's builds"
        assert len(events_of(pool, "engine_mismatch")) == 1
        with_engine(pool, "auto")  # the configuration made to agree
        pool.reprobe()
        assert agent.held[-1] == [OLLAMA_BUILD]


def test_a_restart_in_flight_ends_with_the_host_and_with_the_supervisor(monkeypatch):
    import asyncio

    async def never(request):
        if request.url.path.endswith("/engine"):
            await asyncio.sleep(3600)
        return agent._answer(request)

    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=True, on_disk={VLLM_BUILD})
        pool.supervisor._agent_transport = httpx.MockTransport(never)
        pool.reprobe()
        first = pool.supervisor.hosts["box"].restart_task
        assert first is not None and not first.done()

        async def retire():
            pool.supervisor._retire_host("box", remove_row=False)
            pool.supervisor.hosts["box"] = pool.supervisor._build(pool.supervisor.config.hosts[0])

        pool.loop.run(retire())
        assert first.cancelling() or first.cancelled()
        pool.reprobe()
        second = pool.supervisor.hosts["box"].restart_task
        assert second is not None and second is not first
    assert second.cancelling() or second.cancelled(), "closed with the supervisor"


# --- the operator's restart (D41), in the background ---


def test_an_operators_restart_answers_at_once_and_waits_as_long_as_the_host_says(monkeypatch):
    import asyncio

    from gpm_server.supervisor import agents
    from test_agent import control

    release = asyncio.Event()
    waited = []

    async def slow(agent_config, settings, *, read_s, transport=None):
        waited.append(read_s)
        await release.wait()
        return 200, {"restart_exit_code": 0, "restart_output": "", "engine_answers": True}

    with auto_pool(monkeypatch) as pool:
        # An agent that runs its vLLM itself, and so may restart it.
        agent = Agent(pool.engines["box"].fake, chosen="vllm", restart=True, on_disk={VLLM_BUILD})
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        pool.supervisor.hosts["box"].install = {"allowed": True, "engine": "vllm", "switch_bound_s": 2160}
        monkeypatch.setattr(agents, "restart_engine", slow)
        server = control(pool)
        try:
            with httpx.Client(base_url=server.base_url, headers={"Authorization": "Bearer gpmx_console"}, timeout=10) as http:
                url = "/pool/hosts/box/engine/restart"
                asked = http.post(url, json={"confirm": "box"})
                assert asked.status_code == 202, asked.text
                again = http.post(url, json={"confirm": "box"})
                assert again.status_code == 409 and again.json()["error"] == "restart_in_progress"
                pool.loop.loop.call_soon_threadsafe(release.set)
                finish(pool)
        finally:
            server.stop()
        assert waited == [2160 + 60], "the host's own bound, not the agent client's 400 s"
        assert events_of(pool, "agent_engine_restarted")


def test_an_operators_restart_the_agent_has_said_it_would_refuse_is_refused_at_once(monkeypatch):
    """In the background, a refusal would only be an event: what the agent last said it allows is
    answered on the spot."""
    from test_agent import control

    with owners_vllm_pool(monkeypatch) as pool:
        agent = Agent(pool.engines["box"].fake, chosen="vllm", installs=False, restart=False)
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        server = control(pool)
        try:
            with httpx.Client(base_url=server.base_url, headers={"Authorization": "Bearer gpmx_console"}, timeout=10) as http:
                url = "/pool/hosts/box/engine/restart"
                refused = http.post(url, json={"confirm": "box"})
                assert refused.status_code == 409 and refused.json()["error"] == "owner_has_not_enabled_restart"
                agent.restart, agent.settings = True, False
                pool.reprobe()
                refused = http.post(url, json={"confirm": "box", "apply_settings": True})
                assert refused.status_code == 409 and refused.json()["error"] == "owner_has_not_enabled_settings"
                agent.reachable = False
                pool.reprobe()
                assert http.post(url, json={"confirm": "box"}).status_code == 502
        finally:
            server.stop()
        assert agent.restarts == []


def test_a_restart_is_waited_on_by_the_engines_own_bound_where_the_agent_runs_it():
    from gpm_server.supervisor.agents import RESTART_READ_S, restart_read_s

    assert restart_read_s(None) == RESTART_READ_S
    assert restart_read_s({"allowed": False, "switch_bound_s": 2160}) == RESTART_READ_S, "the owner's command"
    assert restart_read_s({"allowed": True, "engine": "ollama", "switch_bound_s": 120}) == RESTART_READ_S
    assert restart_read_s({"allowed": True, "engine": "vllm", "switch_bound_s": 2160}) == 2220
    assert restart_read_s({"allowed": True, "engine": "vllm", "switch_bound_s": 10**9}) == 4 * 3600


# --- what the console is told of a choice (D161) ---


class _View:
    def __init__(self, facts, reachable=True):
        self.facts, self.reachable = facts, reachable


class _Host:
    def __init__(self, choice=None, reachable=True, chosen_engine=None):
        engine = {"name": "x"} if choice is None else {"name": "x", "choice": choice}
        self.agent = _View({"engine": engine} if reachable else None, reachable)
        self.chosen_engine = chosen_engine


@pytest.mark.parametrize(("choice", "shown"), [
    ({"engine": "tensorrt", "reasons": []}, None),
    ({"engine": None}, None),
    ("vllm", None),
    ({"engine": "vllm"}, {"engine": "vllm", "reasons": [], "said_now": True}),
    ({"engine": "ollama", "reasons": "a card"}, {"engine": "ollama", "reasons": [], "said_now": True}),
    ({"engine": "ollama", "reasons": ["old card", 7, None, {"x": 1}]},
     {"engine": "ollama", "reasons": ["old card"], "said_now": True}),
])
def test_a_choice_is_shown_only_as_one_auto_can_make(choice, shown):
    from gpm_server.supervisor.control import _engine_choice

    assert _engine_choice(_Host(choice)) == shown


def test_a_choices_reasons_are_bounded_in_number_and_length():
    from gpm_server.supervisor.control import _engine_choice

    shown = _engine_choice(_Host({"engine": "ollama", "reasons": ["x" * 5000] * 50}))
    assert len(shown["reasons"]) == 6 and all(len(r) <= 301 for r in shown["reasons"])


# --- configuration, and the plan ---


TWO = {
    MODEL: {"variants": [{"tag": VLLM_BUILD, "engine": "vllm"}, {"tag": OLLAMA_BUILD, "engine": "ollama"}]},
    "emb": {"variants": [{"tag": "org/emb", "engine": "vllm", "serves": ["embed"]},
                         {"tag": "emb:1", "engine": "ollama", "serves": ["embed"]}]},
}


def two_models(engine):
    return PoolConfig.model_validate({
        "pool": {"name": "p", "model_set": [MODEL, "emb"]}, "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]},
        "catalog": TWO, "hosts": [host(engine=engine)],
    })


def test_an_auto_host_holds_several_models_as_its_agent_runs_either_engine():
    """An agent of `auto` installs and runs its engine itself, and starts vLLM with its router in
    front of several models (`--proxy`): not held to one, as a host that names vLLM still is —
    whether that one's agent runs it is the machine's setting, which the file cannot see."""
    loaded = two_models("auto")
    assert loaded.models_held_by(loaded.hosts[0]) == [MODEL, "emb"]
    with pytest.raises(ValueError, match="serves one model per process"):
        two_models("vllm")


def test_a_declared_model_an_auto_host_cannot_hold_is_named_with_the_build_it_lacks():
    only_vllm = {MODEL: {"variants": [{"tag": VLLM_BUILD, "engine": "vllm"}]}}
    with pytest.raises(ValueError) as refused:
        PoolConfig.model_validate({
            "pool": {"name": "p", "model_set": [MODEL], "models_per_host": "declared"},
            "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]},
            "catalog": only_vllm, "hosts": [host(models=[MODEL])],
        })
    said = str(refused.value)
    assert "'chat' has no build for ollama" in said and "meet no variant" not in said


def test_the_plan_says_what_an_auto_hosts_agent_will_fetch_or_that_it_follows_the_choice():
    from gpm_server.configplan import MachineNow, plan_changes

    pool = config([host(agent={**AGENT, "manage_models": True})])
    undecided = plan_changes(pool, pool, machines={"box": MachineNow(frozenset(), frozenset(), None)})
    said = [c.detail for c in undecided if c.kind == "agent_download"]
    assert said and "follows that choice" in said[0]
    chose = plan_changes(pool, pool, machines={"box": MachineNow(frozenset(), frozenset(), None, chosen_engine="vllm")})
    said = [c.detail for c in chose if c.kind == "agent_download"]
    assert said and "its agent chose vllm" in said[0] and VLLM_BUILD in said[0]
