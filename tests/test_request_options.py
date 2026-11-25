"""A request's options and the route it took (S19 stage 2, D201; workloads.md §14.4).

`X-GPM-Priority` and `X-GPM-Keep-On` narrow where a request may go: the router takes the hosts
they exclude out when it chooses eligible hosts, on every workload and on the shared one. A plan
gives each a default; `keep_on: any` lifts it, never past the plan. Every answer says which role
and source served it (`X-GPM-Route`), and the request log and the stats keep all three.

As in test_serving_plan_routing: the supervisor stopped and the tables set by hand — `laptop` is
the shared workload's own computer, `office` a machine the operator owns elsewhere, `rented` the
workload `research`'s, `provider` a hosted API (a fake). Nothing here spends money.
"""

import threading
import time

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import APP_KEY, BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_client import PoolClient, PoolOptionNotHonoured, PoolRequestError, PoolUnavailable
from gpm_server.serving_plan import (
    PlanRefused,
    compile_policy,
    parse_serve,
    request_options,
    why_kept_off,
)
from gpm_server.stats import Stats
from gpm_server.workload_store import Workload, WorkloadStore

MODEL = "m1"
PRICE = {"input": 3.0, "output": 15.0}
BURST = {"budget": {"per_day": 5.0}, "after_s": 0}


@pytest.fixture
def provider():
    loop = BackgroundLoop()
    fake = FakeHostedApi(models=("provider-m1",))
    server = ServerHandle(fake.app, loop)
    yield fake, server.base_url
    server.stop()
    loop.stop()


@pytest.fixture
def pool(monkeypatch, provider):
    _, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    catalog = {MODEL: {"variants": [{"tag": MODEL, "engine": "ollama"},
                                    {"tag": "provider-m1", "engine": "openai-api", "price": PRICE}]}}
    hosted = {"id": "provider", "kind": "hosted", "workers": 4,
              "transport": {"type": "http", "base_url": f"{url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}
    with pool_harness(
        [EngineSpec(id="laptop", resident={MODEL}, workers=4),
         EngineSpec(id="office", kind="fixed-remote", resident={MODEL}, workers=4),
         EngineSpec(id="rented", resident={MODEL}, workers=1, chunk_delay_s=2.0)],
        model_set=[MODEL], catalog=catalog, extra_hosts=[hosted],
    ) as harness:
        harness.reprobe()
        harness.stop_supervisor()
        # A machine rented for the workload, as the supervisor publishes one.
        set_host(harness, "rented", workload="research", kind="rented-on-demand")
        heartbeat(harness)
        yield harness


def set_host(pool, host_id, *, workload=None, state=None, burst=None, kind=None):
    pool.database.execute(
        "UPDATE hosts SET workload = ?, state = COALESCE(?, state), burst = COALESCE(?, burst), "
        "kind = COALESCE(?, kind), updated_at = ? WHERE host_id = ?",
        (workload, state, None if burst is None else int(burst), kind, time.time() + 1, host_id),
    )
    pool.loop.run(pool.state.registry.refresh())


def heartbeat(pool):
    pool.database.execute(
        "INSERT INTO supervisor_lock (pool, owner, pid, heartbeat) VALUES (?, 'test', 1, ?) "
        "ON CONFLICT(pool) DO UPDATE SET heartbeat = excluded.heartbeat", (pool.config.pool.name, time.time()))
    pool.loop.run(pool.state.registry.refresh())


def add_workload(pool, serve, *, state="serving", name="research"):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name=name, model=MODEL, builds={MODEL: MODEL}, latency_s=20, parallel=1, kind="roi",
        lease_id=f"lease-{name}", state=state, workers_per_host=1, hosts_at_start=1,
        plan={"may_borrow": True}, created_at=now, updated_at=now, policy=compile_policy(parse_serve(serve)),
    ))
    _, key = store.mint_key(name)
    pool.loop.run(pool.state.registry.refresh())
    return store, key


def ask(pool, key, *, stream=False, **headers):
    sent = {"Authorization": f"Bearer {key}",
            **{f"X-GPM-{name.replace('_', '-').title()}": value for name, value in headers.items()}}
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30, headers=sent,
                      json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": stream})


def busy(pool, key):
    """The workload's one machine's one worker, busy for two seconds."""
    slow = threading.Thread(target=lambda: ask(pool, key, stream=True))
    slow.start()
    time.sleep(0.3)
    return slow


# --- the options, read ---


def test_the_options_take_the_plans_defaults_and_any_lifts_them():
    plan = parse_serve({"defaults": {"keep_on": "no_hosted"}})
    assert request_options(plan, None, None) == ("interactive", "no_hosted")
    assert request_options(plan, "batch", "any") == ("batch", "any")
    assert request_options(None, None, None) == ("interactive", "any")
    with pytest.raises(ValueError, match="X-GPM-Keep-On is no_hosted, owned or any"):
        request_options(plan, None, "nowhere")
    with pytest.raises(ValueError, match="X-GPM-Priority is interactive or batch"):
        request_options(plan, "urgent", None)
    with pytest.raises(PlanRefused, match="defaults.keep_on is no_hosted, owned or any"):
        parse_serve({"defaults": {"keep_on": "cheap"}})
    with pytest.raises(PlanRefused, match="substitute"):
        parse_serve({"defaults": {"substitute": "never"}})
    assert parse_serve({"defaults": {"priority": "batch"}}).as_dict()["defaults"] == {"priority": "batch"}
    with pytest.raises(PlanRefused, match="keep_on owned keeps every request off the workload's own rented machines"):
        parse_serve({"defaults": {"keep_on": "owned"}})


def test_the_refusal_names_the_option_and_whether_lifting_it_would_help():
    plan = parse_serve({"defaults": {"keep_on": "no_hosted"}})
    chosen = request_options(plan, None, None)
    assert why_kept_off(plan, None, None, chosen) == "X-GPM-Keep-On: no_hosted (the plan's default; send any to lift it)"
    chosen = request_options(plan, "batch", None)
    assert why_kept_off(plan, "batch", None, chosen) == "X-GPM-Keep-On: no_hosted (the plan's default); X-GPM-Priority: batch"
    assert why_kept_off(None, None, "owned", request_options(None, None, "owned")) == "X-GPM-Keep-On: owned"


# --- the router ---


def test_no_hosted_keeps_a_request_off_a_hosted_api_its_plan_would_use(pool, provider):
    fake, _ = provider
    _, key = add_workload(pool, {"when_full": [{"hosted": BURST}]})
    slow = busy(pool, key)
    answer = ask(pool, key)
    assert answer.headers["x-gpm-host"] == "provider" and answer.headers["x-gpm-route"] == "when_full/hosted"
    started = time.monotonic()
    answer = ask(pool, key, keep_on="no_hosted")
    slow.join()
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "rented"
    assert answer.headers["x-gpm-route"] == "steady/rented"
    assert time.monotonic() - started > 1.0, "it waited for the machine"
    assert len(fake.seen) == 1


def test_batch_never_reaches_a_hosted_api(pool, provider):
    fake, _ = provider
    _, key = add_workload(pool, {"until_ready": [{"hosted": BURST}]}, state="preparing")
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key, priority="batch")
    # Its own machine is coming up: waiting can help, so it is told to wait — never sent on.
    assert answer.status_code == 503 and answer.json()["reason"] == "workload_preparing"
    assert ask(pool, key).headers["x-gpm-route"] == "until_ready/hosted"
    assert len(fake.seen) == 1


def test_owned_uses_only_the_operators_machines(pool, provider):
    fake, _ = provider
    _, key = add_workload(pool, {"until_ready": [{"hosted": BURST}, "fixed"]}, state="preparing")
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key, keep_on="owned")
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "office"
    assert answer.headers["x-gpm-route"] == "until_ready/fixed"
    assert fake.seen == []


def test_owned_on_a_workload_that_runs_on_rented_machines_has_nowhere_to_go(pool):
    _, key = add_workload(pool, {"when_full": [{"hosted": BURST}]})
    answer = ask(pool, key, keep_on="owned")
    assert answer.status_code == 503 and answer.json()["reason"] == "no_allowed_source"
    assert "X-GPM-Keep-On: owned" in answer.json()["detail"]


def test_a_plans_default_holds_until_a_request_says_any(pool, provider):
    fake, _ = provider
    _, key = add_workload(pool, {"when_full": [{"hosted": BURST}], "defaults": {"keep_on": "no_hosted"}})
    slow = busy(pool, key)
    answer = ask(pool, key)
    assert answer.headers["x-gpm-host"] == "rented", "kept off the hosted API by the plan's default"
    slow.join()
    slow = busy(pool, key)
    answer = ask(pool, key, keep_on="any")
    slow.join()
    assert answer.status_code == 200 and answer.headers["x-gpm-route"] == "when_full/hosted"
    assert len(fake.seen) == 1


def test_an_option_not_on_its_list_is_refused(pool):
    answer = ask(pool, APP_KEY, keep_on="cheap")
    assert answer.status_code == 400 and answer.json()["reason"] == "unknown_option"
    answer = ask(pool, APP_KEY, priority="urgent")
    assert answer.status_code == 400 and answer.json()["reason"] == "unknown_option"


def test_the_shared_workload_honours_the_options_too(pool):
    set_host(pool, "laptop", state="unreachable")
    answer = ask(pool, APP_KEY, keep_on="owned")
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "office"
    assert answer.headers["x-gpm-route"] == "steady/shared"


def test_a_burst_host_answers_as_when_full(pool):
    _, key = add_workload(pool, {"when_full": [{"name": "more", "rented": {"max_hosts": 1,
                                                                           "budget": {"per_day": 1.0}}}]})
    set_host(pool, "rented", workload="research", burst=True)
    answer = ask(pool, key)
    assert answer.headers["x-gpm-route"] == "when_full/more"


def test_the_log_and_the_stats_keep_the_options_and_the_route(pool):
    _, key = add_workload(pool, {"until_ready": ["local"]})
    assert ask(pool, key, priority="batch", keep_on="no_hosted").status_code == 200
    assert ask(pool, APP_KEY).status_code == 200
    rows = pool.wait_for_log(2)
    mine = next(r for r in rows if r["workload"] == "research")
    assert (mine["priority"], mine["keep_on"], mine["route"]) == ("batch", "no_hosted", "steady/rented")
    shared = next(r for r in rows if r["workload"] is None)
    assert (shared["priority"], shared["keep_on"], shared["route"]) == ("interactive", "any", "steady/shared")
    stats = Stats(pool.database, lambda: pool.config)
    stats.roll_up()
    now = time.time()
    by_priority = stats.query(now - 3600, now + 3600, by="priority")
    assert {s["label"]: s["totals"]["requests"] for s in by_priority["series"]} == {"batch": 1, "interactive": 1}
    by_route = stats.query(now - 3600, now + 3600, by="route")
    assert {s["label"] for s in by_route["series"]} == {"steady/rented", "steady/shared"}
    assert any("from gpm-server 0.47.0 on" in note for note in by_route["notes"])
    by_keep_on = stats.query(now - 3600, now + 3600, by="keep_on")
    assert {s["label"] for s in by_keep_on["series"]} == {"no_hosted", "any"}
    with pytest.raises(ValueError, match="a host or provider filter does not apply"):
        stats.query(now - 3600, now + 3600, by="route", filters={"host": "laptop"})


def test_the_pools_status_lists_the_options_it_honours(pool):
    status = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {APP_KEY}"}).json()
    assert status["options"] == {"priority": ["interactive", "batch"], "keep_on": ["no_hosted", "owned", "any"],
                                 "substitute": ["never"]}


# --- the SDK ---


def test_the_sdk_sends_the_options_and_reads_the_route(pool):
    _, key = add_workload(pool, {"until_ready": ["local"]})
    with PoolClient(pool.url, key, keep_on="no_hosted") as client:
        reply = client.chat(MODEL, [{"role": "user", "content": "hi"}])
        assert reply.route == "steady/rented"
        reply = client.chat(MODEL, [{"role": "user", "content": "hi"}], priority="batch")
        assert reply.route == "steady/rented"
    rows = pool.wait_for_log(2)
    assert sorted((r["priority"], r["keep_on"]) for r in rows) == [("batch", "no_hosted"), ("interactive", "no_hosted")]
    with pytest.raises(ValueError, match="keep_on is no_hosted, owned or any"):
        PoolClient(pool.url, key, keep_on="cheap")


def test_the_sdk_fails_fast_where_no_source_is_allowed(pool):
    _, key = add_workload(pool, {})
    with PoolClient(pool.url, key) as client:
        started = time.monotonic()
        with pytest.raises(PoolUnavailable) as caught:
            client.chat(MODEL, [{"role": "user", "content": "hi"}], keep_on="owned")
        assert caught.value.reason == "no_allowed_source" and time.monotonic() - started < 5


def test_the_sdk_refuses_an_option_a_pool_does_not_list():
    def answer(request):
        if request.url.path == "/pool/status":
            return httpx.Response(200, json={"pool": "old", "hosts": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    with PoolClient("http://pool.test", "gpm_app_test", transport=httpx.MockTransport(answer)) as client:
        assert client.chat(MODEL, [{"role": "user", "content": "hi"}]).content == "hi"
        with pytest.raises(PoolOptionNotHonoured, match="this pool does not honour X-GPM-Keep-On") as caught:
            client.chat(MODEL, [{"role": "user", "content": "hi"}], keep_on="no_hosted")
        assert isinstance(caught.value, PoolRequestError) and caught.value.reason == "option_not_honoured"


def test_the_sdk_asks_again_once_an_older_pool_is_upgraded():
    upgraded = []

    def answer(request):
        if request.url.path == "/pool/status":
            options = {"options": {"keep_on": ["no_hosted", "owned", "any"]}} if upgraded else {}
            return httpx.Response(200, json={"pool": "p", "hosts": [], **options})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]},
                              headers={"X-GPM-Route": "steady/rented"})

    with PoolClient("http://pool.test", "gpm_app_test", transport=httpx.MockTransport(answer)) as client:
        with pytest.raises(PoolOptionNotHonoured):
            client.chat(MODEL, [{"role": "user", "content": "hi"}], keep_on="no_hosted")
        upgraded.append(True)
        assert client.chat(MODEL, [{"role": "user", "content": "hi"}], keep_on="no_hosted").route == "steady/rented"


def test_owned_waits_where_the_plans_stop_gap_could_still_reach_an_owned_machine(pool):
    # Serving on its rented machine, with `local` to cover a replacement: no source now, but one
    # may open — a retryable answer, never the final no_allowed_source.
    _, key = add_workload(pool, {"until_ready": ["local"]})
    answer = ask(pool, key, keep_on="owned")
    assert answer.status_code == 503 and answer.json()["reason"] != "no_allowed_source"
