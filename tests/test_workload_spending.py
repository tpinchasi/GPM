"""Each workload's own spending policy (S11, D165): a hosted burst on its own budget, set when it
is made or while it runs — by its operator, or by the program that made it within the grant of
its provisioning key and the pool's cap on every program together.

Two levels. The router's: the supervisor stopped and the host table set by hand, as in
test_workload_routing — `laptop` is the shared workload's, `rented` is the workload `research`'s,
`provider` is a hosted API (a fake). The supervisor's: programs asking through the real router,
supervisor and SDK, as in test_provisioning. Nothing here spends money.
"""

import threading
import time

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import APP_KEY, BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_client import WorkloadProvisioner, WorkloadRefused
from gpm_server.config import Budget
from gpm_server.db import RequestLog, RequestRecord
from gpm_server.providers import default_offer
from gpm_server.supervisor.control import create_control_app
from gpm_server.supervisor.workloads import WorkloadRefused as Refused
from gpm_server.supervisor.workloads import parse_policy
from gpm_server.workload_store import ModelTarget, Workload, WorkloadStore

MODEL = "m1"
PRICE = {"input": 3.0, "output": 15.0}
#: The fake provider's usage — 1000 in (200 cached), 500 out — at that price.
COST = (1000 * 3.0 + 500 * 15.0) / 1e6
HOSTED = {"after_s": 0.3, "budget": {"per_day": 1.0}}


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


def set_host(pool, host_id, *, workload=None, state=None):
    pool.database.execute(
        "UPDATE hosts SET workload = ?, state = COALESCE(?, state), updated_at = ? WHERE host_id = ?",
        (workload, state, time.time() + 1, host_id),
    )
    pool.loop.run(pool.state.registry.refresh())


def add_workload(pool, name="research", state="serving", policy=None, may_borrow=True):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name=name, model=MODEL, builds={MODEL: MODEL}, latency_s=20, parallel=4, kind="roi",
        lease_id=f"lease-{name}", state=state, workers_per_host=1, hosts_at_start=1,
        plan={"may_borrow": may_borrow}, created_at=now, updated_at=now, policy=policy or {},
    ))
    _, key = store.mint_key(name)
    pool.loop.run(pool.state.registry.refresh())
    return store, key


def ask(pool, key, *, stream=False):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30, headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": stream})


@pytest.fixture
def pool(monkeypatch, provider):
    _, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    catalog = {MODEL: {"variants": [{"tag": MODEL, "engine": "ollama"},
                                    {"tag": "provider-m1", "engine": "openai-api", "price": PRICE}]}}
    with pool_harness(
        [EngineSpec(id="laptop", resident={MODEL}, workers=4),
         EngineSpec(id="rented", resident={MODEL}, workers=1, chunk_delay_s=2.0)],
        model_set=[MODEL], catalog=catalog, extra_hosts=[hosted_host(url)],
        pool_settings={"bursts": {"hosted": {"after_s": 0.3, "budget": {"per_day": 5.0}}}},
    ) as harness:
        harness.reprobe()
        harness.stop_supervisor()
        set_host(harness, "rented", workload="research")
        yield harness


def test_a_workloads_burst_spills_from_its_own_busy_machine_on_its_own_budget(pool, provider):
    fake, _ = provider
    _, key = add_workload(pool, policy={"hosted": HOSTED})
    slow = threading.Thread(target=lambda: ask(pool, key, stream=True))
    slow.start()  # its one machine's one worker, busy for two seconds
    time.sleep(0.2)
    started = time.monotonic()
    answer = ask(pool, key)
    slow.join()
    assert answer.status_code == 200, answer.text
    assert answer.headers["x-gpm-host"] == "provider" and time.monotonic() - started >= 0.3
    assert float(answer.headers["x-gpm-cost-usd"]) == pytest.approx(COST)
    assert fake.seen[-1]["headers"]["authorization"] == f"Bearer {KEY}"
    row = next(r for r in pool.wait_for_log(2) if r["host_id"] == "provider")
    assert row["workload"] == "research" and row["borrowed"] == 0, "a hosted API is never borrowing"
    budgets = pool.state.budgets
    assert budgets.left("research", Budget(per_day=1.0))["spent_today_usd"] == pytest.approx(COST)
    assert budgets.left(None, Budget(per_day=5.0))["spent_today_usd"] == 0, "not the shared workload's money"


def test_a_workload_without_a_hosted_burst_never_reaches_one_though_the_shared_workload_does(pool, provider):
    fake, _ = provider
    _, key = add_workload(pool)
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key)
    assert answer.status_code == 503 and answer.json()["reason"] == "workload_preparing"
    assert fake.seen == []


def test_a_starting_workload_that_may_not_borrow_is_served_by_its_hosted_burst_meanwhile(pool, provider):
    _, key = add_workload(pool, state="preparing", policy={"hosted": HOSTED}, may_borrow=False)
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key)
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "provider"
    assert float(answer.headers["x-gpm-wait-s"]) < 0.3, "no machine of its own to wait for"


def test_a_starting_workload_that_may_borrow_takes_a_shared_machine_before_a_provider(pool, provider):
    fake, _ = provider
    _, key = add_workload(pool, state="preparing", policy={"hosted": HOSTED})
    set_host(pool, "rented", workload="research", state="preparing")
    answer = ask(pool, key)
    assert answer.status_code == 200 and answer.headers["x-gpm-host"] == "laptop"
    assert fake.seen == []


def test_a_workload_spent_out_is_told_so_and_the_shared_budget_is_untouched(pool):
    _, key = add_workload(pool, state="preparing", may_borrow=False,
                          policy={"hosted": {"after_s": 0, "budget": {"per_day": 0.02}}})
    set_host(pool, "rented", workload="research", state="preparing")
    # Up to 4 096 tokens out at $15 a million: more than two cents could hold.
    answer = ask(pool, key)
    assert answer.status_code == 503 and answer.json()["reason"] == "hosted_budget_spent"
    with pool.client() as client:
        assert client.post("/v1/chat/completions", json={"model": MODEL, "messages": []}).headers["x-gpm-host"] == "laptop"


def test_a_workloads_status_says_its_own_budget(pool):
    _, key = add_workload(pool, policy={"hosted": HOSTED})
    status = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {key}"}).json()
    assert status["budgets"]["hosted"]["per_day_usd"] == 1.0
    shared = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {APP_KEY}"}).json()
    assert shared["budgets"]["hosted"]["per_day_usd"] == 5.0


def test_a_providers_workers_are_not_lent_shared_workers(pool):
    # The laptop's four ready workers at a quarter: one. The provider's four are no one's to lend.
    assert pool.state.dispatcher.borrow_limit(0.25) == 1


# --- the policy, as asked for ---


def test_a_policy_is_read_and_checked_in_words():
    assert parse_policy({}) == {}
    assert parse_policy({"bursts": {"hosted": {"budget": {"per_month": 30}}}})["hosted"]["after_s"] == 5.0
    for body, words in (
        ({"bursts": {"hosted": {"budget": {"per_request": 0.1}}}}, "per_request alone"),
        ({"bursts": {"hosted": {"after_s": -1, "budget": {"per_day": 1}}}}, "bursts.hosted.after_s"),
        ({"bursts": {"rented": {"after_min": 10, "budget": {"per_day": 5}}}}, "bursts.rented.max_hosts: Field required"),
        ({"bursts": {"rented": {"max_hosts": 2, "budget": {"per_day": 5, "per_request": 1}}}},
         "a rented machine is paid by the hour"),
        ({"bursts": {"rented": {"max_hosts": 0, "budget": {"per_day": 5}}}}, "bursts.rented.max_hosts"),
        ({"bursts": {"spot": {}}}, r"not \['spot'\]"),
        ({"bursts": {"hosted": HOSTED}, "steady": {"hosts": 2}}, "steady capacity is its parallel at its latency_s"),
    ):
        with pytest.raises(Refused, match=words):
            parse_policy(body)


def test_a_budget_bounds_a_day_and_a_month():
    assert Budget(per_day=10).most_in_a_day == 10 and Budget(per_day=10).most_in_a_month == 310
    assert Budget(per_day=10, per_month=6).most_in_a_day == 6
    assert Budget(per_month=50).most_in_a_day == 50 and Budget(per_month=50).most_in_a_month == 50


# --- programs, within their grant ---

ADMIN = "gpmx_spending_admin"
BIG = "big"
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


def seen(pool):
    """The router reads the workload table in the background; a test reads it now."""
    pool.loop.run(pool.state.registry.refresh())


def bursts(per_day, **more):
    return {"hosted": {"after_s": 2, "budget": {"per_day": per_day, **more}}}


def test_a_program_makes_a_workload_with_a_hosted_burst_and_changes_it_while_it_runs(programs):
    key = grant(programs, hosted_per_day=5.0)
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC, bursts=bursts(3.0)) as w:
        assert programs.supervisor.workloads.get(w.name).policy["hosted"]["budget"] == {"per_day": 3.0}
        programs.loop.run(programs.state.registry.refresh())
        assert w.hosted_budget()["per_day_usd"] == 3.0
        assert programs.state.hosted_burst_for(w.name).after_s == 2
        w.set_bursts(bursts(4.0))
        programs.loop.run(programs.state.registry.refresh())
        assert w.hosted_budget()["per_day_usd"] == 4.0, "the router reads the change"
        with pytest.raises(WorkloadRefused, match=r"pass its \$5.00"):
            w.set_bursts(bursts(6.0))
        w.set_bursts({})
        programs.loop.run(programs.state.registry.refresh())
        assert w.hosted_budget() is None and programs.state.hosted_burst_for(w.name) is None
    events = programs.supervisor.events.recent(50, kind="workload_policy")
    assert events and "provisioning key evals" in events[0]["summary"]


def test_a_grant_that_allows_no_hosted_dollars_makes_no_hosted_burst(programs):
    key = grant(programs)
    with provisioner(programs, key) as pool_side, pytest.raises(WorkloadRefused, match="allows no hosted"):
        pool_side.workload(BIG, **SPEC, bursts=bursts(1.0))


def test_a_keys_workloads_together_stay_within_its_hosted_day_and_month(programs):
    key = grant(programs, hosted_per_day=5.0, hosted_per_month=100.0)
    with provisioner(programs, key) as pool_side:
        with pool_side.workload(BIG, **SPEC, bursts=bursts(3.0)):  # 31 days of $3: $93 a month
            with pytest.raises(WorkloadRefused, match=r"committed \$3.00 a day"):
                pool_side.workload(BIG, **SPEC, bursts=bursts(2.5))
            with pytest.raises(WorkloadRefused, match=r"committed \$93.00 a month"):
                pool_side.workload(BIG, **SPEC, bursts=bursts(2.0))  # 31 days of $2: $62
            with pool_side.workload(BIG, **SPEC, bursts=bursts(2.0, per_month=5.0)):
                seen(programs)
            seen(programs)


def test_the_pool_bounds_every_programs_hosted_day_together(programs):
    first, second = grant(programs, "evals", hosted_per_day=5.0), grant(programs, "batch", hosted_per_day=5.0)
    with provisioner(programs, first) as a, a.workload(BIG, **SPEC, bursts=bursts(5.0)):
        seen(programs)
        with provisioner(programs, second) as b, pytest.raises(WorkloadRefused, match="max_hosted_per_day"):
            b.workload(BIG, **SPEC, bursts=bursts(4.0))


def test_an_ended_workload_counts_what_it_spent_so_ending_it_buys_no_fresh_day(programs):
    key = grant(programs, hosted_per_day=5.0)
    with provisioner(programs, key) as pool_side:
        with pool_side.workload(BIG, **SPEC, bursts=bursts(5.0)) as w:
            name = w.name
            seen(programs)
        RequestLog(programs.database)._insert(RequestRecord(request_id="r-1", outcome="ok", workload=name, cost_usd=1.5))
        with pytest.raises(WorkloadRefused, match=r"committed \$1.50 a day"):
            pool_side.workload(BIG, **SPEC, bursts=bursts(4.0))
        with pool_side.workload(BIG, **SPEC, bursts=bursts(3.5)):
            seen(programs)


def test_the_operator_sets_a_policy_and_one_nothing_could_spend_is_refused(programs):
    workloads = programs.supervisor.workloads
    with pytest.raises(Refused, match="no hosted API in this pool serves 'nothing'"):
        workloads.check_policy([ModelTarget("nothing", 30, 2)], {"hosted": HOSTED})
    created = httpx.post(f"{programs.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"},
                         json={"name": "night", "model": BIG, **SPEC, "bursts": bursts(9.0)}, timeout=60)
    assert created.status_code == 201, created.text
    assert created.json()["workload"]["policy"]["hosted"]["budget"]["per_day"] == 9.0, "no grant: the operator's own"
    changed = httpx.post(f"{programs.control_url}/pool/workloads/night/policy", headers={"Authorization": f"Bearer {ADMIN}"},
                         json={"bursts": {}})
    assert changed.status_code == 200 and changed.json()["workload"]["policy"] == {}
    assert changed.json()["workload"]["hosted_spend"] is None


def test_a_plan_with_a_rented_burst_derives_a_budget_that_pays_for_it(programs):
    """A rented burst is paid from the workload's lease (D165): the budget a plan derives carries
    the least of its hosts at their price the whole time and its budget over the hours."""
    headers = {"Authorization": f"Bearer {ADMIN}"}
    body = {"name": "night", "model": BIG, "latency_s": 30, "parallel": 2, "hours": 1}
    plain = httpx.post(f"{programs.control_url}/pool/workloads/plan", headers=headers, json=body, timeout=60).json()["plan"]
    burst = {"rented": {"after_min": 5, "max_hosts": 1, "budget": {"per_day": 0.05}}}
    planned = httpx.post(f"{programs.control_url}/pool/workloads/plan", headers=headers,
                         json={**body, "bursts": burst}, timeout=60).json()["plan"]
    assert planned["rented_burst"]["max_hosts"] == 1 and planned["rented_burst"]["in_the_budget"] > 0
    assert planned["derived_budget"] == pytest.approx(plain["derived_budget"] + planned["rented_burst"]["in_the_budget"])
    assert planned["rented_burst"]["in_the_budget"] <= 0.10, "never more than its budget over the days it touches"
    refused = httpx.post(f"{programs.control_url}/pool/workloads/plan", headers=headers,
                         json={**body, "bursts": {"rented": {"max_hosts": 1, "budget": {}}}}, timeout=60)
    assert refused.status_code == 400 and "per_day or per_month" in refused.text


def test_the_command_line_changes_one_burst_and_keeps_the_other(programs, capsys, monkeypatch):
    from gpm_server.cli import main

    monkeypatch.setenv("GPM_ADMIN_KEY", ADMIN)
    monkeypatch.setenv("GPM_CONTROL_URL", programs.control_url)
    created = httpx.post(f"{programs.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"},
                         json={"name": "night", "model": BIG, **SPEC, "bursts": bursts(9.0)}, timeout=60)
    assert created.status_code == 201, created.text
    workloads = programs.supervisor.workloads
    assert main(["workload", "policy", "night", "--rent-max-hosts", "2", "--rent-per-day", "1.5"]) == 0
    policy = workloads.get("night").policy
    assert policy["rented"] == {"after_min": 10.0, "max_hosts": 2, "budget": {"per_day": 1.5}}
    assert policy["hosted"]["budget"]["per_day"] == 9.0, "the burst its flags did not name is kept"
    assert main(["workload", "policy", "night", "--no-hosted"]) == 0
    assert set(workloads.get("night").policy) == {"rented"}
    capsys.readouterr()
    assert main(["workload", "policy", "night", "--rent-per-day", "1"]) == 1, "max_hosts is required"
    said = capsys.readouterr()
    assert "max_hosts" in said.out + said.err
    events = programs.supervisor.events.recent(10, kind="workload_policy")
    assert any("up to 2 more host(s) once a burst has lasted 10 min" in e["summary"] for e in events)


# --- what the operator sees: spending, and a provider out of money ---


def test_the_operator_reads_spending_per_workload_provider_and_burst(programs, capsys, monkeypatch):
    from gpm_server.cli import main
    from gpm_server.router.budgets import period_starts

    policy = {**bursts(9.0), "rented": {"max_hosts": 1, "budget": {"per_day": 2.0}}}
    created = httpx.post(f"{programs.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"},
                         json={"name": "night", "model": BIG, **SPEC, "bursts": policy}, timeout=60)
    assert created.status_code == 201, created.text
    log = RequestLog(programs.database)
    day, _ = period_starts()
    log._insert(RequestRecord(request_id="r-1", outcome="ok", workload="night", host_id="provider", cost_usd=0.5))
    log._insert(RequestRecord(request_id="r-2", outcome="ok", workload="night", host_id="provider", cost_usd=0.25))
    log._insert(RequestRecord(request_id="r-3", outcome="ok", workload=None, host_id="provider", cost_usd=0.1))
    if day > period_starts()[1]:  # not the month's first day: an answer from earlier this month
        log._insert(RequestRecord(request_id="r-4", outcome="ok", workload="night", host_id="provider",
                                  cost_usd=1.0, ts=day - 60))
    spending = httpx.get(f"{programs.control_url}/pool/spending", headers={"Authorization": f"Bearer {ADMIN}"}).json()
    assert spending["hosted"]["today"] == pytest.approx(0.85)
    rows = {(r["workload"], r["provider"]): r for r in spending["hosted"]["by"]}
    assert rows[("night", "provider")]["today"] == pytest.approx(0.75) and rows[("night", "provider")]["answers_today"] == 2
    assert rows[(None, "provider")]["today"] == pytest.approx(0.1)
    (night,) = [b for b in spending["hosted"]["budgets"] if b["workload"] == "night"]
    assert night["per_day"] == 9.0 and night["today"] == pytest.approx(0.75)
    assert spending["rented_bursts"] == [{"workload": "night", "per_day": 2.0, "per_month": None, "max_hosts": 1,
                                          "today": 0.0, "this_month": 0.0, "hosts": 0}]
    assert spending["providers"] == [{"host_id": "provider", "state": spending["providers"][0]["state"],
                                      "out_of_funds": None, "out_of_funds_at": None}]
    monkeypatch.setenv("GPM_ADMIN_KEY", ADMIN)
    monkeypatch.setenv("GPM_CONTROL_URL", programs.control_url)
    capsys.readouterr()
    assert main(["spending"]) == 0 and '"rented_bursts"' in capsys.readouterr().out


def test_the_operator_says_a_providers_money_is_back(programs):
    headers = {"Authorization": f"Bearer {ADMIN}"}
    # As the router marks it on its provider's answer, and publishes it with its counters.
    host = programs.state.registry.by_id["provider"]
    host.out_of_funds, host.out_of_funds_at = '{"error": {"code": "insufficient_quota"}}', time.time() - 30
    programs.loop.run(programs.state.registry.publish_counters())
    status = httpx.get(f"{programs.control_url}/pool/status", headers=headers).json()
    (shown,) = [h for h in status["hosts"] if h["host_id"] == "provider"]
    assert "insufficient_quota" in shown["out_of_funds"], "the console's Hosts screen reads it here"
    answer = httpx.post(f"{programs.control_url}/pool/hosts/provider/funds-restored", headers=headers)
    assert answer.status_code == 200 and answer.json()["was_out_of_funds"], answer.text
    programs.loop.run(programs.state.registry.refresh())
    assert host.out_of_funds is None, "the router reads it with the host table, and routes to it again"
    said = programs.supervisor.events.recent(5, kind="hosted_funds_restored")[0]["summary"]
    assert "its money is back" in said and "insufficient_quota" in said
    refused = httpx.post(f"{programs.control_url}/pool/hosts/laptop/funds-restored", headers=headers)
    assert refused.status_code == 404 and "no hosted API" in refused.text
