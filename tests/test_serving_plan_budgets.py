"""A serving plan's sources, each with its own budget, under one hosted total (S19 stage 3).

Several hosted sources, each paying from its own budget and named in the request log; the
workload's one hosted total over them all; a hosted source's `providers`; a rented source's
`kind`; and covering a scale-up where the plan lists it (workloads.md §14.1, §14.2).

The router's tests run with the supervisor stopped and the tables set by hand, as in
test_serving_plan_routing: `laptop` is the shared workload's computer, `rented` the workload's
own machine, `provider-a` and `provider-b` two hosted APIs (one fake behind both). Nothing here
spends money.
"""

import time
from types import SimpleNamespace

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_client import WorkloadRefused
from gpm_server.config import Budget
from gpm_server.router.budgets import HostedBudgets
from gpm_server.serving_plan import PlanRefused, compile_policy, parse_serve, route
from gpm_server.workload_store import Workload, WorkloadStore
from test_serving_plan_routing import BIG, SPEC, grant, programs, provider, provisioner  # noqa: F401 - fixtures

MODEL = "m1"
PRICE = {"input": 3.0, "output": 15.0}


def hosted(per_day, **more):
    return {"hosted": {"budget": {"per_day": per_day}, **more}}


# --- the plan ---


def test_unnamed_sources_are_numbered_in_their_role_and_budgeted_by_role():
    plan = parse_serve({"until_ready": [hosted(1.0), hosted(2.0)], "when_full": [hosted(3.0)]})
    assert [s.name for s in plan.until_ready] == ["hosted", "hosted-2"]
    assert [s.budget_id for s in plan.hosted_sources()] == ["until_ready/hosted", "until_ready/hosted-2",
                                                             "when_full/hosted"]
    assert parse_serve(plan.as_dict()).as_dict() == plan.as_dict(), "read back as written"
    named = parse_serve({"until_ready": [{"name": "cheap", **hosted(1.0)}], "when_full": ["cheap"]})
    assert named.hosted_sources()[0].budget_id == "cheap" and len(named.hosted_sources()) == 1


def test_one_hosted_total_sums_every_source_by_the_period():
    plan = parse_serve({"until_ready": [hosted(1.0)],
                        "when_full": [{"hosted": {"budget": {"per_month": 30.0}}}]})
    # A source without a day's limit counts its month toward a day; one without a month's, its
    # day times the days in the month.
    assert plan.hosted_total(30) == {"per_day": 31.0, "per_month": 60.0}
    policy = compile_policy(plan)
    assert policy["hosted"]["budget"] == {"per_day": 31.0, "per_month": 61.0}, "a reader of hosted alone: 31 days"


def test_a_hosted_sources_providers_and_a_rented_sources_kind_are_read():
    plan = parse_serve({"until_ready": [hosted(1.0, providers=["provider-b", "provider-a"])],
                        "when_full": [{"rented": {"max_hosts": 2, "budget": {"per_day": 3.0}, "kind": "on_demand"}}]})
    assert plan.until_ready[0].providers == ("provider-b", "provider-a")
    assert plan.rented().kind == "on_demand"
    with pytest.raises(PlanRefused, match="providers is a list of hosted host ids, each once"):
        parse_serve({"until_ready": [hosted(1.0, providers=["a", "a"])]})
    with pytest.raises(PlanRefused, match="rented.kind is one of roi, on_demand, interruptible"):
        parse_serve({"when_full": [{"rented": {"max_hosts": 1, "budget": {"per_day": 1.0}, "kind": "cheap"}}]})


def test_a_scale_up_is_covered_only_where_the_plan_lists_it():
    now = time.time()
    facts = dict(preparing=False, model_ready=True, created_at=now - 7200, now=now, gap_ends_at=now + 60,
                 fresh=True, borrow_share=0.5, gap_reason="scale_up")
    assert route(parse_serve({"until_ready": ["local"]}), **facts).covering is False, "not by default"
    listed = parse_serve({"until_ready": ["local"], "during": ["start", "replacement", "scale_up"]})
    assert route(listed, **facts).covering is True


def test_the_sources_in_force_name_the_hosted_apis_a_request_may_use():
    now = time.time()
    plan = parse_serve({"until_ready": [hosted(1.0, providers=["provider-a"])],
                        "when_full": [hosted(2.0, providers=["provider-b"])]})
    starting = route(plan, preparing=True, model_ready=False, created_at=now, now=now, gap_ends_at=None,
                     fresh=True, borrow_share=0.5)
    assert starting.hosted_roles == ("until_ready", "when_full")
    assert starting.hosted_ids == {"provider-a", "provider-b"} and starting.hosted_order == ("provider-a", "provider-b")
    serving = route(plan, preparing=False, model_ready=True, created_at=now, now=now, gap_ends_at=None,
                    fresh=True, borrow_share=0.5)
    assert serving.hosted_roles == ("when_full",) and serving.hosted_ids == {"provider-b"}


# --- the budgets ---


def test_each_budget_and_the_total_hold_their_limits():
    budgets = HostedBudgets()
    total = {"per_day": 1.5, "per_month": 10.0}
    a, b = Budget(per_day=1.0), Budget(per_day=1.0)
    assert budgets.hold("w", a, 0.8, budget_id="a", total=total) is not None
    assert budgets.hold("w", a, 0.3, budget_id="a", total=total) is None, "a's own day is full"
    assert budgets.hold("w", b, 0.8, budget_id="b", total=total) is None, "the total is full: 0.8 + 0.8 > 1.5"
    assert budgets.hold("w", b, 0.6, budget_id="b", total=total) is not None
    # A workload without a plan has one budget: everything it spent counts against it.
    budgets.load([{"workload": "x", "ts": time.time(), "cost_usd": 0.9, "budget": "renamed"}])
    assert budgets.hold("x", Budget(per_day=1.0), 0.2) is None


# --- the router ---


@pytest.fixture
def two_providers(monkeypatch):
    loop = BackgroundLoop()
    fake = FakeHostedApi(models=("provider-m1",))
    server = ServerHandle(fake.app, loop)
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    catalog = {MODEL: {"variants": [{"tag": MODEL, "engine": "ollama"},
                                    {"tag": "provider-m1", "engine": "openai-api", "price": PRICE}]}}
    hosts = [{"id": name, "kind": "hosted", "workers": 4,
              "transport": {"type": "http", "base_url": f"{server.base_url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}
             for name in ("provider-a", "provider-b")]
    with pool_harness([EngineSpec(id="laptop", resident={MODEL}, workers=4),
                       EngineSpec(id="rented", resident={MODEL}, workers=1)],
                      model_set=[MODEL], catalog=catalog, extra_hosts=hosts) as harness:
        harness.reprobe()
        harness.stop_supervisor()
        harness.database.execute("UPDATE hosts SET workload = 'research', kind = 'rented-on-demand', "
                                 "state = 'preparing', updated_at = ? WHERE host_id = 'rented'", (time.time() + 1,))
        harness.database.execute("INSERT INTO supervisor_lock (pool, owner, pid, heartbeat) VALUES (?, 't', 1, ?) "
                                 "ON CONFLICT(pool) DO UPDATE SET heartbeat = excluded.heartbeat",
                                 (harness.config.pool.name, time.time()))
        harness.loop.run(harness.state.registry.refresh())
        yield harness, fake
    server.stop()
    loop.stop()


def add_workload(pool, serve, state="preparing"):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name="research", model=MODEL, builds={MODEL: MODEL}, latency_s=20, parallel=1, kind="roi",
        lease_id="lease-research", state=state, workers_per_host=1, hosts_at_start=1,
        plan={"may_borrow": False}, created_at=now, updated_at=now, policy=compile_policy(parse_serve(serve)),
    ))
    _, key = store.mint_key("research")
    pool.loop.run(pool.state.registry.refresh())
    return key


def ask(pool, key):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30, headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})


def test_a_hosted_answer_is_paid_from_its_sources_budget_through_its_providers(two_providers):
    pool, _ = two_providers
    key = add_workload(pool, {"until_ready": [{"name": "cover", **hosted(1.0, providers=["provider-b", "provider-a"])}],
                              "when_full": [hosted(2.0, providers=["provider-a"])]})
    answer = ask(pool, key)
    assert answer.status_code == 200, answer.text
    assert answer.headers["x-gpm-host"] == "provider-b", "the first of its providers"
    assert answer.headers["x-gpm-route"] == "until_ready/cover"
    row = pool.wait_for_log(1)[0]
    assert row["budget"] == "cover" and row["cost_usd"] > 0
    status = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {key}"}).json()
    assert set(status["budgets"]["sources"]) == {"cover", "when_full/hosted"}
    assert status["budgets"]["sources"]["cover"]["spent_today_usd"] > 0
    assert status["budgets"]["hosted"]["per_day_usd"] == 3.0, "the one hosted total"


def test_a_source_whose_budget_is_spent_gives_way_to_the_next(two_providers):
    pool, _ = two_providers
    key = add_workload(pool, {"until_ready": [{"name": "tiny", **hosted(0.000001)}, {"name": "more", **hosted(1.0)}]})
    answer = ask(pool, key)
    assert answer.status_code == 200 and answer.headers["x-gpm-route"] == "until_ready/more"
    assert pool.wait_for_log(1)[0]["budget"] == "more"


# --- the supervisor ---


def test_a_provider_the_pool_does_not_have_is_refused(programs):  # noqa: F811
    key = grant(programs, hosted_per_day=5.0)
    with provisioner(programs, key) as pool_side, pytest.raises(WorkloadRefused, match="'nowhere' is not a hosted API"):
        pool_side.workload(BIG, **SPEC, serve={"until_ready": [hosted(1.0, providers=["nowhere"])]})


def test_a_scale_up_is_marked_where_the_plan_lists_it_and_a_burst_rents_its_own_kind(programs):  # noqa: F811
    key = grant(programs)
    serve = {"until_ready": ["shared"], "during": ["start", "replacement", "scale_up"], "max_min": 20,
             "when_full": [{"rented": {"max_hosts": 1, "budget": {"per_day": 1.0}, "kind": "on_demand"}}]}
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC, serve=serve) as w:
        manager, fleet = programs.supervisor.workloads, programs.supervisor.fleet
        manager.store.set_state(w.name, "serving")
        hosts_of = fleet.hosts_of
        fleet.hosts_of = lambda name: [
            SimpleNamespace(host_id="r-1", models=(BIG,), burst=False, state="ready"),
            SimpleNamespace(host_id="r-2", models=(BIG,), burst=True, state="preparing"),
        ]
        try:
            programs.loop.run(manager.advance())
            gap = manager.store.gaps()[(w.name, BIG)]
            assert gap[3] == "scale_up" and gap[2] == frozenset({"r-1"})
            assert gap[1] == pytest.approx(gap[0] + 20 * 60)
            fleet.workloads = {w.name: manager.get(w.name)}  # as the fleet's pass reads them
            assert fleet.burst_kind_of(w.name) == "on_demand", "its burst rents the kind its source names"
        finally:
            fleet.hosts_of = hosts_of
        programs.loop.run(programs.state.registry.refresh())


def test_every_sources_wait_is_a_moment_the_request_looks_again():
    now = time.time()
    plan = parse_serve({"until_ready": [hosted(1.0)], "when_full": [hosted(2.0, after_s=4)]})
    starting = route(plan, preparing=True, model_ready=False, created_at=now, now=now, gap_ends_at=None,
                     fresh=True, borrow_share=0.5)
    assert starting.hosted_after_s == 0 and starting.hosted_waits == (0.0, 4.0)


def test_a_budget_named_like_anything_is_counted_once():
    budgets = HostedBudgets()
    budgets.load([{"workload": "w", "ts": time.time(), "cost_usd": 0.5, "budget": "*"}])
    assert budgets.left("w", {"per_day": 1.0})["spent_today_usd"] == 0.5
    assert budgets.left("w", {"per_day": 1.0}, budget_id="*")["spent_today_usd"] == 0.5


def test_a_programs_burst_rents_only_kinds_its_grant_allows(programs):  # noqa: F811
    key = grant(programs, kinds=["roi"])
    serve = {"when_full": [{"rented": {"max_hosts": 1, "budget": {"per_day": 1.0}, "kind": "on_demand"}}]}
    with provisioner(programs, key) as pool_side, pytest.raises(WorkloadRefused, match="not on_demand machines for its burst"):
        pool_side.workload(BIG, **SPEC, serve=serve)
