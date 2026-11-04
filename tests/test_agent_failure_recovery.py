"""A host whose agent fails to fetch or hold a model says why for as long as it fails, and is
ready again on its own once the agent succeeds — with one event when the failure begins, one if
what it says changes, an occasional "still failing", and one when it recovers (D163).

SpaceGarage reads each host's `state` and `last_error` from `GET /pool/status` with the app key,
and shows them to the shop; that answer is checked here as it reads it.
"""

import json
import os

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, pool_harness
from gpm_server.supervisor import agents

MODEL = "chat-model:1"
AGENT = {"url": "http://127.0.0.1:8095", "bearer_env": "TEST_AGENT_KEY"}


def answer(error=None, attempts=0, retry_in_s=None, on_disk=False, loaded=False):
    return {"engine_answers": True, "models": [{
        "tag": MODEL, "on_disk": on_disk, "loaded": loaded, "pulling": None, "error": error,
        "attempts": attempts, "retry_in_s": retry_in_s, "awaiting_restart": False,
    }]}


# --- the decision log ---


def kinds_over(answers):
    seen, before = [], None
    for after in answers:
        seen += [(kind, summary) for kind, summary, _ in agents.model_events(before, after)]
        before = after
    return seen


def test_one_event_when_a_failure_begins_or_changes_and_one_when_it_recovers():
    reset, full = "connection reset by peer", "no space left on device"
    said = [(k, s) for k, s in kinds_over([
        answer(reset, 1, 30), answer(reset, 2, 60), answer(reset, 3, 120), answer(reset, 4, 240),
        answer(reset, 5, 480), answer(full, 6, 600), answer(full, 7, 600),
        answer(on_disk=True, loaded=True),
    ]) if k != "agent_model_on_disk"]
    assert [kind for kind, _ in said] == [
        "agent_model_failed", "agent_model_still_failing", "agent_model_failed", "agent_model_recovered"]
    assert said[1][1] == f"{MODEL}: still failing after 5 attempts (next in 8 minutes): {reset}"
    assert said[3][1] == f"{MODEL} works again, after 7 failed attempts"


def test_a_failure_that_lasts_is_said_again_now_and_then_never_on_every_attempt():
    answers = [answer("refused", n, 600) for n in range(1, 101)]
    still = [summary for kind, summary in kinds_over(answers) if kind == "agent_model_still_failing"]
    assert [s.split(" after ")[1].split(" ")[0] for s in still] == ["5", "10", "20", "30", "50", "75", "100"]


def test_what_a_host_says_while_it_fails_is_the_agents_reason_and_its_attempts():
    trouble = agents.model_trouble(answer("Access is denied", 3, 115), [MODEL])
    assert trouble == f"{MODEL}: Access is denied (tried 3 times; next in 2 minutes)"
    assert agents.model_trouble(answer("Access is denied", 1, 30), [MODEL]) == (
        f"{MODEL}: Access is denied (next in 30 seconds)")
    assert agents.model_trouble(answer(on_disk=True), [MODEL]) is None
    assert agents.model_trouble(answer("x", 2, 60), ["another:1"]) is None


# --- the pool, end to end ---


class Agent:
    """The host's agent as the pool hears it: a download that fails until it does not."""

    def __init__(self, engine):
        self.engine = engine
        self.models = answer("pull model manifest: connection reset by peer", 1, 30)
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request):
        if request.url.path.endswith("/facts"):
            return httpx.Response(200, json={
                "agent": {"version": "test", "protocol": "1"}, "os": "Linux", "arch": "x86_64",
                "capabilities": None, "engine": {"name": "ollama", "answers": True},
                "allows": {"delete": False, "restart": False, "engine_settings": False},
            })
        if request.url.path.endswith("/models"):
            assert json.loads(request.content)["tags"] == [MODEL]
            return httpx.Response(200, json=self.models)
        return httpx.Response(404, json={"error": "not_found"})


def status(pool):
    answered = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {APP_KEY}"}, timeout=10)
    assert answered.status_code == 200, answered.text
    return {h["host_id"]: h for h in answered.json()["hosts"]}


@pytest.mark.skipif(os.name == "nt", reason="the pool harness runs its fake engines as POSIX processes")
def test_a_host_whose_download_fails_says_why_until_it_works_and_then_serves_on_its_own(monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "test-agent-key")
    with pool_harness(
        [EngineSpec(id="shop-pc", resident=set())], model_set=[MODEL],
        host_overrides={"shop-pc": {"agent": AGENT}},
    ) as pool:
        agent = Agent(pool.engines["shop-pc"].fake)
        pool.supervisor._agent_transport = agent.transport
        for attempts, wait in ((1, 30), (2, 60), (3, 120)):
            agent.models = answer("pull model manifest: connection reset by peer", attempts, wait)
            pool.reprobe()
            host = status(pool)["shop-pc"]
            assert host["state"] == "preparing"
            assert host["last_error"].startswith(f"{MODEL}: pull model manifest: connection reset by peer")
        assert "tried 3 times" in host["last_error"]
        refused = httpx.post(f"{pool.url}/v1/chat/completions", headers={"Authorization": f"Bearer {APP_KEY}"},
                             json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}, timeout=10)
        assert refused.status_code == 503 and refused.json()["reason"] == "preparing"

        # The network is back: the agent's next attempt fetches and loads it. Nothing restarted.
        pool.engines["shop-pc"].fake.resident.add(MODEL)
        agent.models = answer(on_disk=True, loaded=True)
        pool.reprobe()
        host = status(pool)["shop-pc"]
        assert host["state"] == "ready" and host["last_error"] is None
        failures = ("agent_model_failed", "agent_model_still_failing", "agent_model_recovered")
        kinds = [e["kind"] for e in pool.supervisor.events.recent(100) if e["kind"] in failures]
        assert sorted(kinds) == ["agent_model_failed", "agent_model_recovered"]
