"""Engines the host's agent installed are upgraded at a quiet moment, drained first, checked
after, and rolled back by themselves (S9, D153; docs/spec/supervisor.md §3.1).

The decisions are tested as functions of what the agent said, what the host holds and the clock;
then once through the pool, against a fake agent that stages, switches and goes back as the real
one does, and a fake engine whose models can be lost by a switch — the way an engine update once
split two models the first adopter held together.
"""

import json
import time

import httpx
import pytest
from fakes.harness import EngineSpec, pool_harness
from gpm_server.supervisor.upgrades import SWITCH_TIMEOUT_S, EngineUpgrades

MODEL = "chat:7b"
WINDOWS = {"idle_needed_s": 600.0, "drain_s": 300.0, "check_s": 600.0}


def install(running="0.40.11", staged="0.40.12", previous=None, **more):
    return {"allowed": True, "engine": "ollama", "running": running, "staged": staged,
            "previous": previous, "working": None, "error": None, **more}


def step(upgrades, info, now, *, busy=0, idle_s=None, held=frozenset({MODEL}), answers=True, another=False):
    return upgrades.next_step("mac", info, busy=busy, idle_s=idle_s, held=held, engine_answers=answers,
                              another_upgrading=another, now=now, **WINDOWS)


# --- the decisions ---


def test_nothing_happens_until_the_host_has_been_quiet_long_enough():
    upgrades = EngineUpgrades()
    assert step(upgrades, install(), 0, idle_s=30).action == "wait"
    assert step(upgrades, install(), 0, busy=1, idle_s=900).action == "wait"
    started = step(upgrades, install(), 0, idle_s=900)
    assert started.action == "start" and started.event == "engine_upgrade_draining"
    assert "0.40.11 → 0.40.12" in started.summary


def test_a_drained_host_switches_then_is_checked_then_serves():
    upgrades = EngineUpgrades()
    step(upgrades, install(), 0)
    assert step(upgrades, install(), 1, busy=2).action == "wait"
    switch = step(upgrades, install(), 2)
    assert (switch.action, switch.version) == ("switch", "0.40.12")
    assert step(upgrades, install(working={"step": "switching", "version": "0.40.12"}), 3).action == "wait"
    switched = step(upgrades, install(running="0.40.12", staged=None, previous="0.40.11"), 4,
                    held=frozenset())
    assert switched.event == "engine_switched"
    assert step(upgrades, install(running="0.40.12", staged=None, previous="0.40.11"), 5, held=frozenset()).action == "wait"
    done = step(upgrades, install(running="0.40.12", staged=None, previous="0.40.11"), 60)
    assert done.action == "done" and done.event == "engine_upgraded" and "mac" not in upgrades.active


def test_requests_that_will_not_finish_put_the_upgrade_off_and_the_host_back():
    upgrades = EngineUpgrades()
    step(upgrades, install(), 0)
    put_off = step(upgrades, install(), 301, busy=1)
    assert put_off.action == "abandon" and put_off.event == "engine_upgrade_postponed"
    assert "mac" not in upgrades.active and not upgrades.refused


def test_a_version_that_loses_the_model_set_is_rolled_back_and_never_taken_again_until_a_newer_one():
    upgrades = EngineUpgrades()
    step(upgrades, install(), 0)
    step(upgrades, install(), 1)
    after = install(running="0.40.12", staged=None, previous="0.40.11")
    step(upgrades, after, 2, held=frozenset())
    back = step(upgrades, after, 2 + 601, held=frozenset())
    assert (back.action, back.version, back.event) == ("rollback", "0.40.11", "engine_rolled_back")
    assert upgrades.refused["mac"] == {"0.40.12"}
    # The agent stages it again, as `latest` still names it: not taken.
    assert step(upgrades, install(), 1000, idle_s=900).action == "wait"
    # A newer release is a new version.
    assert step(upgrades, install(staged="0.40.13"), 1001, idle_s=900).action == "start"


def test_a_version_that_does_not_start_is_given_back_by_the_agent_and_refused():
    upgrades = EngineUpgrades()
    step(upgrades, install(), 0)
    step(upgrades, install(), 1)
    gave_back = step(upgrades, install(error="the engine did not answer in 120 s"), 2)
    assert gave_back.action == "done" and gave_back.event == "engine_rolled_back"
    assert upgrades.refused["mac"] == {"0.40.12"}


def test_a_switch_the_agent_never_finishes_is_given_up_with_a_rollback():
    upgrades = EngineUpgrades()
    step(upgrades, install(), 0)
    step(upgrades, install(), 1)
    stuck = install(working={"step": "switching", "version": "0.40.12"})
    assert step(upgrades, stuck, 1 + SWITCH_TIMEOUT_S - 1).action == "wait"
    gone = step(upgrades, install(running=None), 2 + SWITCH_TIMEOUT_S)
    assert (gone.action, gone.version) == ("rollback", "0.40.11")


def test_an_operator_upgrades_now_without_the_quiet_wait_and_rolls_back_on_purpose():
    upgrades = EngineUpgrades()
    upgrades.ask("mac", "upgrade")
    assert step(upgrades, install(), 0, idle_s=5, busy=1).action == "start"
    upgrades.active.clear()

    upgrades.ask("mac", "rollback")
    on_new = install(running="0.40.12", staged=None, previous="0.40.11")
    started = step(upgrades, on_new, 0, busy=0, idle_s=5)
    assert started.action == "start" and "rolling ollama back to 0.40.11" in started.summary
    assert step(upgrades, on_new, 1).version == "0.40.11"
    back = install(running="0.40.11", staged="0.40.12", previous="0.40.12")
    step(upgrades, back, 2)
    done = step(upgrades, back, 3)
    assert done.action == "done" and upgrades.refused["mac"] == {"0.40.12"}, "rolled away from on purpose"


def test_one_host_at_a_time_and_nothing_where_the_owner_did_not_allow_installs():
    upgrades = EngineUpgrades()
    assert step(upgrades, install(), 0, idle_s=900, another=True).action == "wait"
    assert step(upgrades, install(allowed=False), 0, idle_s=900).action == "wait"
    assert step(upgrades, None, 0).action == "wait"


def test_a_machine_with_nothing_running_yet_is_left_to_its_agent_which_starts_the_first_version_itself():
    upgrades = EngineUpgrades()
    assert step(upgrades, install(running=None, staged="0.40.12"), 0, busy=0, idle_s=1).action == "wait"
    assert "mac" not in upgrades.active


# --- through the pool ---


class InstallingAgent:
    """The host's agent as the pool hears it, staging and switching as the real one does."""

    def __init__(self, engine, *, loses_models=False):
        self.engine = engine
        self.loses_models = loses_models
        self.install = install()
        self.asked: list[dict] = []
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request):
        path = request.url.path
        if path.endswith("/facts"):
            return httpx.Response(200, json={
                "agent": {"version": "test", "protocol": "1"}, "os": "Darwin", "arch": "arm64",
                "capabilities": None, "engine": {"name": "ollama", "answers": True, "install": self.install},
            })
        if path.endswith("/models"):
            return httpx.Response(200, json={"engine_answers": True, "models": [], "surplus": []})
        body = json.loads(request.content)
        self.asked.append(body)
        if body["switch"]:
            info, wanted = self.install, body["version"]
            if wanted == info["staged"]:
                self.install = install(running=wanted, staged=None, previous=info["running"])
                if self.loses_models:
                    self.engine.resident.clear()
            elif wanted == info["previous"]:
                self.install = install(running=wanted, staged=info["running"], previous=info["running"])
                self.engine.resident.add(MODEL)
        return httpx.Response(200, json=self.install)


def upgrading_pool(monkeypatch, **settings):
    monkeypatch.setenv("TEST_AGENT_KEY", "test-agent-key")
    return pool_harness(
        [EngineSpec(id="mac", resident={MODEL})],
        model_set=[MODEL],
        pool_settings={"engine_upgrade_idle_s": 0, **settings},
        host_overrides={"mac": {"agent": {"url": "http://127.0.0.1:8095", "bearer_env": "TEST_AGENT_KEY"}}},
    )


def control(pool):
    from fakes.harness import APP_KEY, ServerHandle
    from gpm_server.config import AuthConfig
    from gpm_server.supervisor.control import create_control_app

    config = pool.supervisor.config.model_copy(
        update={"auth": AuthConfig(app_keys=[APP_KEY], admin_keys=["gpmx_console"])}
    )
    return ServerHandle(create_control_app(pool.supervisor, config), pool.loop)


def kinds(pool):
    return [e["kind"] for e in reversed(pool.supervisor.events.recent(50))]


def test_an_upgrade_runs_through_the_pool_out_of_routing_and_back(monkeypatch):
    with upgrading_pool(monkeypatch) as pool:
        agent = InstallingAgent(pool.engines["mac"].fake)
        pool.supervisor._agent_transport = agent.transport
        host = pool.supervisor.hosts["mac"]

        pool.reprobe()                       # staged and quiet: drained at once
        assert host.state.value == "preparing" and "0.40.11 → 0.40.12" in host.last_error
        assert agent.asked[0] == {"version": "latest", "switch": False}
        pool.reprobe()                       # nothing in flight: switched
        assert {"version": "0.40.12", "switch": True} in agent.asked
        pool.reprobe()                       # running the new one: checked
        pool.reprobe()
        pool.reprobe()
        assert host.state.value == "ready", host.last_error
        assert agent.install["running"] == "0.40.12"
        assert kinds(pool).count("engine_upgraded") == 1
        assert [k for k in kinds(pool) if k.startswith("engine_")][:3] == [
            "engine_upgrade_draining", "engine_upgrade_switching", "engine_switched"]


def test_an_upgrade_that_loses_the_model_set_rolls_back_through_the_pool(monkeypatch):
    with upgrading_pool(monkeypatch, engine_upgrade_check_s=0.01) as pool:
        agent = InstallingAgent(pool.engines["mac"].fake, loses_models=True)
        pool.supervisor._agent_transport = agent.transport
        for _ in range(6):
            pool.reprobe()
            time.sleep(0.02)  # past the check's window, which the passes here outrun
        assert agent.install["running"] == "0.40.11"
        assert "engine_rolled_back" in kinds(pool)
        assert pool.supervisor.upgrades.refused["mac"] == {"0.40.12"}
        assert pool.supervisor.hosts["mac"].state.value == "ready"


def test_the_operator_asks_and_the_status_says_where_versions_stand(monkeypatch):
    with upgrading_pool(monkeypatch, engine_upgrade_idle_s=3600) as pool:
        agent = InstallingAgent(pool.engines["mac"].fake)
        pool.supervisor._agent_transport = agent.transport
        pool.reprobe()
        assert pool.supervisor.hosts["mac"].state.value == "ready", "not quiet for an hour yet"
        server = control(pool)
        try:
            with httpx.Client(base_url=server.base_url, headers={"Authorization": "Bearer gpmx_console"}) as http:
                refused = http.post("/pool/hosts/mac/engine/upgrade", json={"confirm": "wrong"})
                assert refused.status_code == 400
                asked = http.post("/pool/hosts/mac/engine/upgrade", json={"confirm": "mac"})
                assert asked.status_code == 200, asked.text
                assert http.post("/pool/hosts/mac/engine/rollback", json={"confirm": "mac"}).status_code == 409
                pool.reprobe()
                entry = next(h for h in http.get("/pool/status").json()["hosts"] if h["host_id"] == "mac")
        finally:
            server.stop()
        assert entry["agent"]["engine_install"]["staged"] == "0.40.12"
        assert entry["agent"]["upgrade"]["to"] == "0.40.12" and entry["agent"]["upgrade"]["by_operator"]


@pytest.mark.parametrize("version", ["latest", "0.40.11", "1.9.5", "0.41.0-rc1"])
def test_a_host_names_its_engine_version(version):
    from gpm_server.config import HostConfig

    host = HostConfig.model_validate({"id": "h", "kind": "local", "engine_version": version,
                                      "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}})
    assert host.engine_version == version


@pytest.mark.parametrize("version", ["newest", "v0.40.11", "0.40.11; rm -rf", "../0.40", "https://x/y"])
def test_anything_but_a_version_is_refused(version):
    from gpm_server.config import HostConfig

    with pytest.raises(ValueError, match="engine_version"):
        HostConfig.model_validate({"id": "h", "kind": "local", "engine_version": version,
                                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}})
