"""A workload's serving plan carried through the pool (S19, D198–D203), stage 1.

Two levels, as in test_workload_spending. The router's: the supervisor stopped and the tables set
by hand — `laptop` is the shared workload's own computer, `office` a machine the operator owns
elsewhere, `rented` the workload `research`'s, `provider` a hosted API (a fake). The supervisor's
heartbeat and the replacement gaps it marks are written here as it would write them. Then the
supervisor's: it marks a replacement while a serving workload is short of hosts, and a program
creates a workload with a plan within its grant, through the real router, supervisor and SDK.
Nothing here spends money.
"""

import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import APP_KEY, BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_client import WorkloadProvisioner, WorkloadRefused
from gpm_server.providers import default_offer
from gpm_server.serving_plan import compile_policy, parse_serve
from gpm_server.supervisor.control import create_control_app
from gpm_server.workload_store import Workload, WorkloadStore

MODEL = "m1"
BIG = "big-model"
PRICE = {"input": 3.0, "output": 15.0}
ADMIN = "gpmx_admin_test"


@pytest.fixture
def provider():
    loop = BackgroundLoop()
    fake = FakeHostedApi(models=("provider-m1", "provider-big"))
    server = ServerHandle(fake.app, loop)
    yield fake, server.base_url
    server.stop()
    loop.stop()


def hosted_host(url):
    return {"id": "provider", "kind": "hosted", "workers": 4,
            "transport": {"type": "http", "base_url": f"{url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}


# --- the router ---


@pytest.fixture
def pool(monkeypatch, provider):
    _, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    catalog = {MODEL: {"variants": [{"tag": MODEL, "engine": "ollama"},
                                    {"tag": "provider-m1", "engine": "openai-api", "price": PRICE}]}}
    with pool_harness(
        [EngineSpec(id="laptop", resident={MODEL}, workers=4),
         EngineSpec(id="office", kind="fixed-remote", resident={MODEL}, workers=4),
         EngineSpec(id="rented", resident={MODEL}, workers=1, chunk_delay_s=2.0)],
        model_set=[MODEL], catalog=catalog, extra_hosts=[hosted_host(url)],
    ) as harness:
        harness.reprobe()
        harness.stop_supervisor()
        set_host(harness, "rented", workload="research")
        yield harness


def set_host(pool, host_id, *, workload=None, state=None):
    pool.database.execute(
        "UPDATE hosts SET workload = ?, state = COALESCE(?, state), updated_at = ? WHERE host_id = ?",
        (workload, state, time.time() + 1, host_id),
    )
    pool.loop.run(pool.state.registry.refresh())


def heartbeat(pool, at=None):
    """The supervisor's heartbeat, as it writes it every pass."""
    pool.database.execute(
        "INSERT INTO supervisor_lock (pool, owner, pid, heartbeat) VALUES (?, 'test', 1, ?) "
        "ON CONFLICT(pool) DO UPDATE SET heartbeat = excluded.heartbeat",
        (pool.config.pool.name, time.time() if at is None else at))
    pool.loop.run(pool.state.registry.refresh())


def add_workload(pool, serve, *, state="serving", name="research", hosts_at_start=1, may_borrow=True):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name=name, model=MODEL, builds={MODEL: MODEL}, latency_s=20, parallel=1, kind="roi",
        lease_id=f"lease-{name}", state=state, workers_per_host=1, hosts_at_start=hosts_at_start,
        plan={"may_borrow": may_borrow}, created_at=now, updated_at=now, policy=compile_policy(parse_serve(serve)),
    ))
    _, key = store.mint_key(name)
    pool.loop.run(pool.state.registry.refresh())
    return store, key


def ask(pool, key, *, stream=False):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30, headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": stream})


COVER = {"budget": {"per_day": 5.0}}


def test_a_starting_workload_is_covered_by_the_machines_its_plan_names_and_no_other(pool, provider):
    fake, _ = provider
    heartbeat(pool)
    _, key = add_workload(pool, {"until_ready": ["fixed"]}, state="preparing")
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key)
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "office", "fixed, never the laptop"
    assert fake.seen == []


def test_the_stop_gap_goes_to_a_hosted_api_at_once_and_stops_once_the_workload_serves(pool, provider):
    fake, _ = provider
    heartbeat(pool)
    _, key = add_workload(pool, {"until_ready": [{"hosted": COVER}]}, state="preparing")
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key)
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "provider"
    assert float(answer.headers["x-gpm-wait-s"]) < 0.3, "a stop-gap's source has no wait"
    set_host(pool, "rented", workload="research", state="ready")
    answer = ask(pool, key)
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "rented"
    assert len(fake.seen) == 1


def test_a_stop_gap_on_a_stale_supervisor_uses_only_what_spends_nothing(pool, provider):
    fake, _ = provider
    heartbeat(pool, at=time.time() - 10 * 365 * 86400)  # older than three of any pass
    _, key = add_workload(pool, {"until_ready": ["local", {"hosted": COVER}]}, state="preparing")
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key)
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "laptop"
    # Only the provider could take it now, and a supervisor that stopped opens no paid stop-gap.
    _, key = add_workload(pool, {"until_ready": [{"hosted": COVER}]}, state="preparing", name="other")
    answer = ask(pool, key)
    assert answer.status_code == 503 and answer.json()["reason"] == "workload_preparing"
    assert fake.seen == []


def test_a_replacement_the_supervisor_marks_borrows_while_the_workloads_own_host_is_busy(pool):
    heartbeat(pool)
    # Planned two hosts; one is left while the other is replaced.
    store, key = add_workload(pool, {"until_ready": ["local"]}, hosts_at_start=2)
    store.open_gap("research", MODEL, time.time(), time.time() + 600, ["rented"])
    pool.loop.run(pool.state.registry.refresh())
    slow = threading.Thread(target=lambda: ask(pool, key, stream=True))
    slow.start()  # its one machine's one worker, busy for two seconds
    time.sleep(0.3)
    answer = ask(pool, key)
    slow.join()
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "laptop"
    row = next(r for r in pool.wait_for_log(2) if r["host_id"] == "laptop")
    assert row["borrowed"] == 1 and row["workload"] == "research"


def test_a_marked_replacement_ends_the_moment_another_host_is_ready(pool):
    heartbeat(pool)
    store, key = add_workload(pool, {"until_ready": ["local"]}, hosts_at_start=1)
    # Opened with no host ready; `rented` is ready now, and the supervisor has not looked again yet.
    store.open_gap("research", MODEL, time.time(), time.time() + 600)
    pool.loop.run(pool.state.registry.refresh())
    slow = threading.Thread(target=lambda: ask(pool, key, stream=True))
    slow.start()
    time.sleep(0.3)
    answer = ask(pool, key)
    slow.join()
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "rented", "its one planned host is ready"


def test_a_request_already_waiting_takes_the_stop_gap_once_a_replacement_is_marked(pool):
    heartbeat(pool)
    store, key = add_workload(pool, {"until_ready": ["local"]}, hosts_at_start=2)
    slow = threading.Thread(target=lambda: ask(pool, key, stream=True))
    slow.start()  # its one machine's one worker, busy for two seconds
    time.sleep(0.3)
    answers = []
    waiting = threading.Thread(target=lambda: answers.append(ask(pool, key)))
    waiting.start()
    time.sleep(0.3)  # queued behind it, with no gap yet
    store.open_gap("research", MODEL, time.time(), time.time() + 600, ["rented"])
    pool.loop.run(pool.state.registry.refresh())
    waiting.join()
    slow.join()
    assert answers[0].status_code == 200 and answers[0].headers["x-gpm-host"] == "laptop"


def test_a_workload_that_may_not_borrow_never_borrows_through_its_plan(pool):
    heartbeat(pool)
    _, key = add_workload(pool, {"until_ready": ["local"]}, state="preparing", may_borrow=False)
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key)
    assert answer.status_code == 503 and answer.json()["reason"] == "workload_preparing"


def test_a_workloads_status_says_it_borrows_only_where_its_plan_does(pool):
    heartbeat(pool)
    _, key = add_workload(pool, {"when_full": [{"hosted": COVER}]}, state="preparing")
    set_host(pool, "rented", workload="research", state="preparing")
    status = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {key}"}).json()
    assert status["borrowing"] is False, "no stop-gap: nothing to borrow"
    _, key = add_workload(pool, {"until_ready": ["local"]}, state="preparing", name="other")
    status = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {key}"}).json()
    assert status["borrowing"] is True and status["borrowing_models"] == [MODEL]


def test_without_a_marked_replacement_a_serving_workload_waits_for_its_own_host(pool):
    heartbeat(pool)
    _, key = add_workload(pool, {"until_ready": ["local"]})
    slow = threading.Thread(target=lambda: ask(pool, key, stream=True))
    slow.start()
    time.sleep(0.3)
    started = time.monotonic()
    answer = ask(pool, key)
    slow.join()
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "rented"
    assert time.monotonic() - started > 1.0, "it waited for its own machine: no gap, no stop-gap"


# --- the supervisor, and a program's plan ---


SPEC = {"latency_s": 30, "parallel": 2, "hours": 1, "max_spend": 2.0}


@pytest.fixture
def programs(monkeypatch, provider):
    _, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    catalog = {BIG: {"variants": [{"tag": BIG, "size_gb": 4},
                                  {"tag": "provider-big", "engine": "openai-api", "price": PRICE}]}}
    with pool_harness(
        [EngineSpec(id="laptop", resident={BIG}, workers=4)], model_set=[BIG], catalog=catalog,
        extra_hosts=[hosted_host(url)],
        rentable=[EngineSpec(id="market-1", resident={BIG}, workers=4), EngineSpec(id="market-2", resident={BIG}, workers=4)],
        rented={"provider": "fake", "workers": 2, "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 1.0},
                "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0}},
        extra_config={
            "auth": {"app_keys": [APP_KEY], "admin_keys": [ADMIN]},
            "limits": {"max_rented_hosts": 4},
            "workloads": {"min_reliability": 0.0},
            "provisioning": {"enabled": True, "max_spend_per_day": 10.0, "request_poll_s": 0.05,
                             "max_hosted_per_day": 8.0},
        },
    ) as h:
        h.supervisor.fleet.provider.offers = [default_offer("o-1", "m-1", min_bid_hourly=0.2),
                                              default_offer("o-2", "m-2", min_bid_hourly=0.2)]
        control = ServerHandle(create_control_app(h.supervisor, h.config), h.loop)
        h.control_url = control.base_url
        try:
            yield h
        finally:
            control.stop()


def grant(pool, name="evals", **overrides):
    body = {"name": name, "models": [BIG], "max_open": 3, "max_spend": 5.0, "max_spend_per_day": 10.0,
            "max_hours": 4, **overrides}
    answer = httpx.post(f"{pool.control_url}/pool/provisioners", json=body, headers={"Authorization": f"Bearer {ADMIN}"})
    assert answer.status_code == 201, answer.text
    return answer.json()["key"]


def provisioner(pool, key):
    pool.loop.run(pool.state.registry.refresh())
    return WorkloadProvisioner(pool.url, key, poll_s=0.05)


def test_a_program_makes_a_workload_with_a_plan_and_replaces_it_while_it_runs(programs):
    key = grant(programs, hosted_per_day=5.0)
    serve = {"until_ready": ["shared", {"hosted": {"budget": {"per_day": 2.0}}}]}
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC, serve=serve) as w:
        policy = programs.supervisor.workloads.get(w.name).policy
        assert policy["serve"]["until_ready"][0] == "shared" and policy["hosted"]["budget"] == {"per_day": 2.0}
        programs.loop.run(programs.state.registry.refresh())
        assert programs.state.plan_for(w.name).until_ready_hosted().after_s == 0
        with pytest.raises(WorkloadRefused, match="has a serving plan; change it with serve"):
            w.set_bursts({"hosted": {"after_s": 2, "budget": {"per_day": 1.0}}})
        w.set_serve({"when_full": [{"hosted": {"after_s": 3, "budget": {"per_day": 1.0}}}]})
        programs.loop.run(programs.state.registry.refresh())
        plan = programs.state.plan_for(w.name)
        assert plan.until_ready == () and plan.when_full_hosted().after_s == 3, "the whole plan is replaced"
    events = programs.supervisor.events.recent(50, kind="workload_policy")
    assert events and "serving plan: steady on machines rented for it; no stop-gap" in events[0]["summary"]


def test_a_program_names_only_the_sources_its_grant_allows(programs):
    key = grant(programs, may_borrow=False)
    with provisioner(programs, key) as pool_side:
        with pytest.raises(WorkloadRefused, match=r"allows the sources \['rented'\], not \['local'\]"):
            pool_side.workload(BIG, **SPEC, serve={"until_ready": ["local"]})
        with pytest.raises(ValueError, match="bursts or serve, not both"):
            pool_side.workload(BIG, **SPEC, serve={}, bursts={})
        with pytest.raises(WorkloadRefused, match="rules arrive in a later version"):
            pool_side.workload(BIG, **SPEC, serve={"rules": []})
    key = grant(programs, name="office-apps", sources=["rented", "local"])
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC, serve={"until_ready": ["local"]}) as w:
        assert programs.supervisor.workloads.get(w.name).policy["serve"]["until_ready"] == ["local"]
        programs.loop.run(programs.state.registry.refresh())


def test_a_grant_naming_its_sources_holds_the_shorthand_to_them_too(programs):
    key = grant(programs, hosted_per_day=5.0, sources=["rented"])
    with provisioner(programs, key) as pool_side:
        with pytest.raises(WorkloadRefused, match=r"allows the sources \['rented'\], not \['hosted'\]"):
            pool_side.workload(BIG, **SPEC, bursts={"hosted": {"budget": {"per_day": 1.0}}})
        with pool_side.workload(BIG, **SPEC) as w:
            assert programs.supervisor.workloads.get(w.name).plan["may_borrow"] is False, "no borrowing source"
            programs.loop.run(programs.state.registry.refresh())


def test_a_grant_that_may_not_borrow_cannot_allow_borrowed_machines(programs):
    answer = httpx.post(f"{programs.control_url}/pool/provisioners", headers={"Authorization": f"Bearer {ADMIN}"},
                        json={"name": "odd", "models": [BIG], "may_borrow": False, "sources": ["rented", "local"]})
    assert answer.status_code == 400 and "cannot allow them" in answer.text


def test_the_routers_policy_call_refuses_bursts_and_serve_together(programs):
    key = grant(programs)
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC) as w:
        programs.loop.run(programs.state.registry.refresh())
        answer = httpx.post(f"{programs.url}/pool/provisioning/workloads/{w.name}/policy",
                            headers={"Authorization": f"Bearer {key}"},
                            json={"serve": {}, "bursts": {}})
        assert answer.status_code == 400 and "not both" in answer.text
        programs.loop.run(programs.state.registry.refresh())


def test_an_operators_grant_names_only_sources_there_are(programs):
    answer = httpx.post(f"{programs.control_url}/pool/provisioners", headers={"Authorization": f"Bearer {ADMIN}"},
                        json={"name": "odd", "models": [BIG], "sources": ["rented", "moon"]})
    assert answer.status_code == 400 and "sources are some of local, fixed, shared, rented, hosted" in answer.text


def test_the_supervisor_marks_a_replacement_once_per_shortfall_and_closes_it(programs):
    key = grant(programs)
    serve = {"until_ready": ["shared"], "max_min": 30}
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC, serve=serve) as w:
        manager = programs.supervisor.workloads
        store = manager.store
        # Serving, and short of its one planned host: what an eviction leaves while the fleet replaces it.
        store.set_state(w.name, "serving")
        programs.loop.run(manager.advance())
        gaps = store.gaps()
        opened, ends, ready = gaps[(w.name, BIG)]
        assert ends == pytest.approx(opened + 30 * 60) and ready == frozenset()
        programs.loop.run(manager.advance())
        assert store.gaps()[(w.name, BIG)] == (opened, ends, ready), "a pass later, the same gap: never extended"
        covering = programs.supervisor.events.recent(20, kind="workload_covering")
        assert len(covering) == 1 and "covers for at most 30 min" in covering[0]["summary"]
        # Its host ready again: the gap is closed.
        fleet = programs.supervisor.fleet
        hosts_of = fleet.hosts_of
        fleet.hosts_of = lambda name: [SimpleNamespace(host_id="r-1", models=(BIG,), burst=False, state="ready")]
        try:
            programs.loop.run(manager.advance())
        finally:
            fleet.hosts_of = hosts_of
        assert (w.name, BIG) not in store.gaps()
        programs.loop.run(programs.state.registry.refresh())
