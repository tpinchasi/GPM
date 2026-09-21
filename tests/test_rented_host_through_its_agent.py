"""A rented host, prepared through its real agent, all the way to serving.

Every piece here had its own tests, and the path as a whole had none. It first ran on a live,
billing host, where it failed three separate ways in one afternoon:

1. the control loop raised on every pass as soon as a rented host had an agent that answered;
2. the agent could not pin an embedding model, and the pool destroyed a healthy host for it;
3. the console drew an empty bar for a download that was running.

So this drives the whole of it with nothing faked between the pool and the engine but the
network: the real supervisor, the real `gpm_agent` application, a fake engine that starts with
an empty disk — and a model set with an embedding model in it, because real ones have one.
Only the SSH install is replaced, by handing the host the agent it would have installed.
"""

import time

import httpx
from fakes.harness import EngineSpec, pool_harness
from gpm_agent.app import create_app
from gpm_agent.settings import Settings, fingerprint
from gpm_server.supervisor import hostagent

CHAT = "a-chat-model"
EMBED = "an-embed-model"
AGENT_KEY = "gpmg_" + "a" * 64

RENTED = {
    "provider": "fake",
    "workers": 2,
    "model_set_gb": 5.0,
    "bidding": {"bid_ceiling": 0.60, "premium": 0.02},
    "scale": {"scale_up_after_s": 0},
    "teardown": {"idle_minutes": 10},
}


def event_kinds(pool):
    return [event["kind"] for event in pool.supervisor.events.recent(limit=200)]


def test_a_rented_host_is_prepared_through_its_agent_and_serves(monkeypatch):
    with pool_harness(
        [EngineSpec(id="local-1", resident={CHAT, EMBED}, kind="local", workers=1)],
        rentable=[EngineSpec(id="market-1", resident=set(), available=set(), workers=2)],
        model_set=[CHAT, EMBED],
        rented=RENTED,
    ) as pool:
        fleet = pool.supervisor.fleet
        fleet.agent_hold_retry_s = 0.01
        engine = pool.rentable["market-1"].fake
        # Slow enough that the download is *seen* in flight: an instant pull shows no progress
        # at all, and a test that never sees any cannot say what shape it came in.
        engine.pull_delay_s = 0.15

        # The agent the pool would have installed over SSH, talking to that host's engine.
        agent = create_app(
            Settings(key_hash=fingerprint(AGENT_KEY)),
            engine_transport=httpx.ASGITransport(app=engine.app),
        )
        fleet._agent_transport = httpx.ASGITransport(app=agent)

        async def installed(host):
            if host.agent is None:
                host.agent = hostagent.RentedAgent(url="http://agent", secret=AGENT_KEY)

        monkeypatch.setattr(fleet, "install_agent", installed)

        fleet.open_lease(workers=3, max_hours=2, max_spend=2.00, allow_rent=True)
        passes_that_raised = []
        real_pass = pool.supervisor.pass_once

        async def watched():
            try:
                await real_pass()
            except Exception as exc:  # noqa: BLE001 - the point is to see it, not to hide it
                passes_that_raised.append(repr(exc))
                raise

        monkeypatch.setattr(pool.supervisor, "pass_once", watched)

        shapes_seen = []
        host = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            pool.reprobe()
            live = [h for h in fleet.hosts.values() if not h.released]
            if live:
                host = live[0]
                for tag, entry in (host.progress or {}).items():
                    shapes_seen.append((tag, set(entry)))
                if host.state == "ready":
                    break
            time.sleep(0.1)

        assert not passes_that_raised, f"a control-loop pass raised: {passes_that_raised[:2]}"
        assert "prepare_failed" not in event_kinds(pool), [
            e["summary"] for e in pool.supervisor.events.recent(50) if e["kind"] == "prepare_failed"
        ]
        assert host is not None and host.state == "ready" and not host.released
        assert {CHAT, EMBED} <= engine.resident, "the embedding model is held too"

        # It was the agent that did it, not the pool falling back to the engine's own API.
        assert host.agent_models is not None, "the host was prepared without its agent"

        # Progress was shown while it downloaded, and in the one shape the console reads.
        assert shapes_seen, "no download was ever seen in flight; this test proves nothing about it"
        for tag, fields in shapes_seen:
            assert tag in (CHAT, EMBED), f"progress keyed by {tag!r}, not by a model tag"
            assert {"completed", "total"} <= fields, f"{tag}: the console cannot draw {fields}"

        # And it serves.
        with pool.client() as client:
            rows = {h["host_id"]: h["state"] for h in client.get("/pool/status").json()["hosts"]}
        assert rows[host.host_id] == "ready"
