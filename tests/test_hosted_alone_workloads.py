"""Workloads served by hosted APIs alone (D165, workloads.md §13): a model asked for with
`parallel` 0 — and a model no machine can run, which is always asked for so — gets no machine,
rented or borrowed, and goes to the pool's hosted APIs on the workload's hosted budget from its
first request. A workload whose every model is so rents nothing: no market search, no dollar cap,
a lease that cannot rent, serving from the start.

Against a fake hosted API and the fake provider; nothing here spends money."""

import time

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import APP_KEY, BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_client import WorkloadProvisioner, WorkloadRefused
from gpm_server.providers import default_offer
from gpm_server.supervisor.control import create_control_app
from gpm_server.workload_store import Group, ModelTarget, Workload, WorkloadStore

ADMIN = "gpmx_alone_admin"
BIG, CLOSED, SMALL = "big", "closed", "small"
PRICE = {"input": 3.0, "output": 15.0}
HOSTED = {"hosted": {"after_s": 0.3, "budget": {"per_day": 1.0}}}
HEADERS = {"Authorization": f"Bearer {ADMIN}"}


@pytest.fixture
def provider():
    loop = BackgroundLoop()
    fake = FakeHostedApi(models=("provider-big", "provider-closed"))
    server = ServerHandle(fake.app, loop)
    yield fake, server.base_url
    server.stop()
    loop.stop()


@pytest.fixture
def pool(monkeypatch, provider):
    _, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    catalog = {
        BIG: {"variants": [{"tag": BIG, "size_gb": 4},
                           {"tag": "provider-big", "engine": "openai-api", "price": PRICE}]},
        SMALL: {"variants": [{"tag": SMALL, "size_gb": 2}]},
        # No build a machine runs: a closed model, reached only as a hosted API.
        CLOSED: {"variants": [{"tag": "provider-closed", "engine": "openai-api", "price": PRICE}]},
    }
    hosted = {"id": "provider", "kind": "hosted", "workers": 4,
              "transport": {"type": "http", "base_url": f"{url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}
    with pool_harness(
        [EngineSpec(id="laptop", resident={BIG, SMALL}, workers=4)], model_set=[BIG, SMALL, CLOSED], catalog=catalog,
        extra_hosts=[hosted],
        rentable=[EngineSpec(id="market-1", resident={BIG, SMALL}, workers=4)],
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
        h.supervisor.fleet.provider.offers = [default_offer("o-1", "m-1", min_bid_hourly=0.2)]
        h.reprobe()
        control = ServerHandle(create_control_app(h.supervisor, h.config), h.loop)
        h.control_url = control.base_url
        try:
            yield h
        finally:
            control.stop()


def create(pool, **body):
    return httpx.post(f"{pool.control_url}/pool/workloads", headers=HEADERS,
                      json={"name": "night", "hours": 2, **body}, timeout=60)


def ask(pool, key, model):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30, headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


def test_a_workload_of_a_closed_model_rents_nothing_and_serves_from_the_start(pool):
    created = create(pool, model=CLOSED, parallel=0, bursts=HOSTED)
    assert created.status_code == 201, created.text
    answer = created.json()
    plan, view = answer["plan"], answer["workload"]
    assert plan["hosts_at_start"] == 0 and plan["offers_seen"] == 0 and plan["hosted_alone"] == [CLOSED]
    assert view["state"] == "serving" and view["groups"] == [], "nothing to prepare"
    lease = pool.supervisor.leases.get(view["lease_id"])
    assert lease.is_open and not lease.allow_rent, "its lease holds its hours, and can rent nothing"
    pool.loop.run(pool.state.registry.refresh())
    reply = ask(pool, answer["connection"]["api_key"], CLOSED)
    assert reply.status_code == 200, reply.text
    assert reply.headers["x-gpm-served-model"] == "provider-closed" and float(reply.headers["x-gpm-cost-usd"]) > 0
    pool.reprobe()
    assert pool.supervisor.fleet.hosts_of("night") == [], "and no machine was rented for it"
    assert "no machine" in pool.supervisor.events.recent(5, kind="workload_created")[0]["summary"]


@pytest.mark.parametrize("body, words", [
    ({"model": CLOSED, "parallel": 0}, "so it needs a hosted burst"),
    ({"model": CLOSED, "parallel": 0, "bursts": HOSTED, "max_spend": 5}, "so it has no max_spend"),
    ({"model": CLOSED, "latency_s": 10, "parallel": 2, "bursts": HOSTED}, "has no build a machine runs"),
    ({"model": SMALL, "parallel": 0, "bursts": HOSTED}, "no hosted API in this pool serves 'small'"),
    ({"model": CLOSED, "parallel": 0,
      "bursts": {**HOSTED, "rented": {"max_hosts": 1, "budget": {"per_day": 1}}}}, "none of its models runs on one"),
    ({"model": BIG, "parallel": 2, "bursts": HOSTED}, "latency_s is required where parallel is above 0"),
])
def test_what_a_workload_served_by_hosted_apis_alone_may_not_be_is_said(pool, body, words):
    planned = httpx.post(f"{pool.control_url}/pool/workloads/plan", headers=HEADERS,
                         json={"name": "night", "hours": 2, **body}, timeout=60)
    assert planned.status_code == 400 and words in planned.json()["detail"], planned.text


def test_its_hosted_burst_cannot_be_taken_away_while_a_model_has_nothing_else(pool):
    assert create(pool, model=CLOSED, parallel=0, bursts=HOSTED).status_code == 201
    changed = httpx.post(f"{pool.control_url}/pool/workloads/night/policy", headers=HEADERS, json={"bursts": {}})
    assert changed.status_code == 400 and "needs a hosted burst" in changed.json()["detail"]


def test_a_mixed_workload_rents_for_its_machine_model_only(pool):
    created = create(pool, models=[{"model": SMALL, "latency_s": 30, "parallel": 2},
                                   {"model": CLOSED, "parallel": 0}], max_spend=3.0, bursts=HOSTED)
    assert created.status_code == 201, created.text
    plan = created.json()["plan"]
    assert plan["hosted_alone"] == [CLOSED] and plan["hosts_at_start"] == 1
    assert [g["models"] for g in plan["groups"]] == [[SMALL]], "a group for the model that runs on machines"
    workload = pool.supervisor.workloads.get("night")
    assert workload.state == "preparing" and workload.hosted_alone(CLOSED) and not workload.hosted_alone(SMALL)
    view = pool.supervisor.workloads.view(workload)
    assert view["answers"]["by_model"][CLOSED]["hosted_alone"] and CLOSED not in view["borrowing_models"]


def test_a_model_served_by_hosted_apis_alone_never_borrows_a_machine(pool):
    """Its workload prepares and may borrow, and the shared laptop serves the model: still, the
    request goes to the hosted API — the model was asked for with no machine."""
    pool.stop_supervisor()
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name="mixed", model=SMALL, builds={SMALL: SMALL}, latency_s=30, parallel=2, kind="roi", lease_id="lease-mixed",
        state="preparing", workers_per_host=2, hosts_at_start=1, plan={"may_borrow": True}, created_at=now,
        updated_at=now, targets=(ModelTarget(SMALL, 30, 2), ModelTarget(BIG, 0.0, 0)),
        groups=(Group(models=(SMALL,), builds={SMALL: SMALL}, hosts_at_start=1, workers_per_host=2),),
        policy=HOSTED,
    ))
    _, key = store.mint_key("mixed")
    pool.loop.run(pool.state.registry.refresh())
    reply = ask(pool, key, BIG)
    assert reply.status_code == 200, reply.text
    assert reply.headers["x-gpm-served-model"] == "provider-big", "not the laptop, which serves big"
    borrowed = ask(pool, key, SMALL)
    assert borrowed.status_code == 200 and borrowed.headers["x-gpm-served-model"] == SMALL, "its machine model borrows"
    status = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {key}"}).json()
    assert status["borrowing_models"] == [SMALL]


def test_its_hours_end_it_though_no_fleet_pass_looks_at_its_lease(pool):
    created = create(pool, model=CLOSED, parallel=0, bursts=HOSTED)
    lease_id = created.json()["workload"]["lease_id"]
    pool.database.execute("UPDATE leases SET opened_at = opened_at - 3 * 3600 WHERE lease_id = ?", (lease_id,))
    workloads = pool.supervisor.workloads
    pool.loop.run(workloads.advance())
    assert workloads.get("night").state in ("ending", "ended")
    pool.loop.run(workloads.advance())
    assert workloads.get("night").state == "ended"
    assert pool.supervisor.leases.get(lease_id).closed_reason == "time limit reached"


def test_a_program_makes_a_workload_of_a_closed_model_with_no_max_spend(pool):
    grant = httpx.post(f"{pool.control_url}/pool/provisioners", headers=HEADERS, json={
        "name": "evals", "models": [CLOSED, SMALL], "max_open": 2, "max_spend": 5.0, "max_spend_per_day": 10.0,
        "max_hours": 4, "hosted_per_day": 2.0})
    assert grant.status_code == 201, grant.text
    pool.loop.run(pool.state.registry.refresh())
    with WorkloadProvisioner(pool.url, grant.json()["key"], poll_s=0.05) as pool_side:
        with pool_side.workload(CLOSED, parallel=0, hours=1, bursts=HOSTED) as w:
            assert w.answer["state"] == "serving"
            pool.loop.run(pool.state.registry.refresh())
            reply = w.client.chat(CLOSED, [{"role": "user", "content": "hi"}])
            assert reply.cost_usd and reply.cost_usd > 0
        with pytest.raises(WorkloadRefused, match="max_spend is required"):
            pool_side.workload(SMALL, latency_s=30, parallel=2, hours=1)
