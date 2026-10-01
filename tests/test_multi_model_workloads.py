"""A workload of several models (D118, stories/S7): each model with its own target and its own
answers at once, placed on hosts that hold every model or on a group of hosts per model —
whichever is expected to cost less — behind one key, one lease and one budget.

The real supervisor, router and control API over one database; the fake provider's market; fake
engines for the machines it hands out. Nothing here spends money.
"""

import time

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, ServerHandle, pool_harness
from gpm_server.providers import default_offer
from gpm_server.supervisor.control import create_control_app

ADMIN = "gpmx_multi_admin"
SHARED, CHAT, EMBED, THIRD = "m1", "chat", "embed", "third"
CATALOG = {
    CHAT: {"variants": [{"tag": CHAT, "size_gb": 4}], "workloads_only": True},
    EMBED: {"variants": [{"tag": EMBED, "size_gb": 1}], "workloads_only": True},
    THIRD: {"variants": [{"tag": THIRD, "size_gb": 1}], "workloads_only": True},
}


@pytest.fixture
def pool():
    with pool_harness(
        [EngineSpec(id="laptop", resident={SHARED}, workers=4)],
        model_set=[SHARED], catalog=CATALOG,
        rentable=[EngineSpec(id=f"market-{i}", resident={CHAT, EMBED, THIRD}, workers=4) for i in range(4)],
        rented={
            "provider": "fake", "workers": 2,
            "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 1.0},
            "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0},
            "teardown": {"idle_minutes": 30},
        },
        extra_config={
            "auth": {"app_keys": [APP_KEY], "admin_keys": [ADMIN]},
            "limits": {"max_rented_hosts": 4},
            "workloads": {"min_reliability": 0.0, "borrow_share": 0.5},
        },
    ) as harness:
        harness.supervisor.fleet.provider.offers = [
            default_offer(f"o-{i}", f"m-{i}", min_bid_hourly=0.20) for i in range(4)
        ]
        control = ServerHandle(create_control_app(harness.supervisor, harness.config), harness.loop)
        harness.control_url = control.base_url
        try:
            yield harness
        finally:
            control.stop()


def admin(pool):
    return httpx.Client(base_url=pool.control_url, headers={"Authorization": f"Bearer {ADMIN}"}, timeout=30)


def until(pool, check, what, within=20.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        pool.reprobe()
        if check():
            return
        time.sleep(0.05)
    raise AssertionError(f"not {what} within {within}s")


def request(placement="auto", *models):
    models = models or ((CHAT, 30, 2), (EMBED, 5, 2))
    return {"name": "research", "hours": 2, "max_spend": 5.0, "placement": placement,
            "models": [{"model": m, "latency_s": latency, "parallel": parallel} for m, latency, parallel in models]}


def create(pool, body):
    with admin(pool) as control:
        made = control.post("/pool/workloads", json=body)
    assert made.status_code == 201, made.text
    pool.loop.run(pool.state.registry.refresh())
    return made.json()


def ask(pool, key, model):
    with pool.client(key=key) as app:
        return app.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


def test_two_models_that_fit_one_card_are_placed_together_and_both_served(pool):
    with admin(pool) as control:
        plan = control.post("/pool/workloads/plan", json=request()).json()["plan"]
    assert plan["refused"] is None, plan
    assert plan["placement"] == "together", plan["reasons"]
    assert set(plan["placements"]) == {"together", "apart"}
    assert any(r.startswith("placement: together") for r in plan["reasons"])
    (group,) = plan["groups"]
    assert set(group["models"]) == {CHAT, EMBED}
    # Two workers a card for each alone: one of each at once per host, on two hosts.
    assert group["caps"] == {CHAT: 1, EMBED: 1} and group["workers_per_host"] == 2 and group["hosts_at_start"] == 2

    key = create(pool, request())["connection"]["api_key"]
    (lease,) = pool.supervisor.leases.open_leases()
    assert lease.workers == 4, "the lease holds every model's answers at once"
    until(pool, lambda: len([h for h in pool.supervisor.fleet.hosts_of("research") if h.state == "ready"]) == 2,
          "both hosts ready")
    for host in pool.supervisor.fleet.hosts_of("research"):
        assert set(host.models) == {CHAT, EMBED} and host.workers == 2
    until(pool, lambda: pool.supervisor.workloads.get("research").state == "serving", "serving")
    pool.loop.run(pool.state.registry.refresh())
    for model in (CHAT, EMBED):
        answer = ask(pool, key, model)
        assert answer.status_code == 200 and answer.headers["X-GPM-Host"].startswith("rented-"), model
    refused = ask(pool, key, THIRD)
    assert refused.status_code == 404 and "'chat', 'embed'" in refused.json()["detail"]
    rows = [r for r in pool.wait_for_log(3) if r["outcome"] == "ok"]
    assert all(r["model_concurrency"] == 1 for r in rows)
    view = pool.supervisor.workloads.view(pool.supervisor.workloads.get("research"))
    assert view["answers"]["by_model"][CHAT]["cap_per_host"] == 1


def test_apart_gives_each_model_hosts_of_its_own_and_serves_once_each_has_one(pool):
    key = create(pool, request("apart"))["connection"]["api_key"]
    workload = pool.supervisor.workloads.get("research")
    assert workload.placement == "apart" and {g.models for g in workload.groups} == {(CHAT,), (EMBED,)}
    until(pool, lambda: len([h for h in pool.supervisor.fleet.hosts_of("research") if h.state == "ready"]) == 2,
          "a host for each model")
    held = sorted(tuple(h.models) for h in pool.supervisor.fleet.hosts_of("research"))
    assert held == [(CHAT,), (EMBED,)], "each host holds its own group's model and nothing else"
    until(pool, lambda: pool.supervisor.workloads.get("research").state == "serving", "serving")
    pool.loop.run(pool.state.registry.refresh())
    served_on = {}
    for model in (CHAT, EMBED):
        answer = ask(pool, key, model)
        assert answer.status_code == 200, answer.text
        served_on[model] = answer.headers["X-GPM-Host"]
    by_host = {h.host_id: h.models for h in pool.supervisor.fleet.hosts_of("research")}
    assert by_host[served_on[CHAT]] == (CHAT,) and by_host[served_on[EMBED]] == (EMBED,)


def test_a_mix_no_card_holds_at_once_goes_apart_and_says_why(pool):
    """Three models, each served two at a time by a card alone: one of each is already 150% of it."""
    body = request("auto", (CHAT, 30, 1), (EMBED, 5, 1), (THIRD, 5, 1))
    with admin(pool) as control:
        plan = control.post("/pool/workloads/plan", json=body).json()["plan"]
        forced = control.post("/pool/workloads/plan", json=request("together", (CHAT, 30, 1), (EMBED, 5, 1),
                                                                    (THIRD, 5, 1))).json()["plan"]
    assert plan["placement"] == "apart" and plan["placements"]["together"]["refused"]
    assert len(plan["groups"]) == 3 and plan["hosts_at_start"] == 3
    assert forced["refused"] and "none holds chat, embed, third together" in forced["refused"]


def test_two_groups_short_of_their_start_never_hold_each_other_back(pool):
    """Found in review: each group yielded to the other's shortfall, and with one slot left
    neither rented. Groups yield in their order, as workloads do: the first takes the slot."""
    pool.supervisor.config.limits.max_rented_hosts = 1
    body = request("apart", (CHAT, 30, 2), (EMBED, 5, 2))
    workloads = pool.supervisor.workloads
    plan = pool.loop.run(workloads.plan(__import__("gpm_server.supervisor.workloads", fromlist=["x"]).WorkloadRequest.of(
        "research", [__import__("gpm_server.workload_store", fromlist=["x"]).ModelTarget(m["model"], m["latency_s"], m["parallel"])
                     for m in body["models"]], 2, max_spend=5.0, placement="apart")))
    assert plan["refused"], "two hosts to start, and room for one: refused at the plan"
    # Room made after creation shrinks: the first group still rents into what is left.
    pool.supervisor.config.limits.max_rented_hosts = 4
    create(pool, body)
    pool.supervisor.config.limits.max_rented_hosts = 1
    fleet = pool.supervisor.fleet
    fleet.workloads = {w.name: w for w in workloads.store.active()}  # as a pass would read them
    first, second = [g.key for g in workloads.get("research").groups]
    assert fleet.reserved_hosts(besides="research", group=first) == 0, "the first group yields to no sibling"
    assert fleet.reserved_hosts(besides="research", group=second) == 1, "the second yields to the first's shortfall"
    pool.reprobe()
    assert len(fleet.hosts_of("research", first)) == 1 and not fleet.hosts_of("research", second)


@pytest.mark.parametrize("body, words", [
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 1}] * 2}, "each model is named once"),
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 1}], "model": CHAT}, "not both"),
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 0}]}, "parallel must be a number above zero"),
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 7.9}]}, "must be a whole number"),
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 10**9}]}, "at most 100000"),
    ({"models": [{"model": CHAT, "latency_s": -1, "parallel": 1}]}, "above zero"),
    ({"models": [], }, "models is a list"),
    ({"placement": "sideways"}, "placement is one of"),
])
def test_what_a_workload_of_several_models_may_not_be_is_said(pool, body, words):
    with admin(pool) as control:
        answer = control.post("/pool/workloads/plan", json={**request(), **body})
    assert answer.status_code == 400 and words in answer.json()["detail"], answer.text


def test_more_models_than_the_pool_allows_is_refused(pool):
    pool.supervisor.config.workloads.max_models = 2
    with admin(pool) as control:
        answer = control.post("/pool/workloads/plan", json=request("auto", (CHAT, 30, 1), (EMBED, 5, 1), (THIRD, 5, 1)))
    assert answer.status_code == 400 and "at most 2 models" in answer.json()["detail"]


def test_the_command_line_plans_several_models(pool, capsys, monkeypatch):
    import test_workloads_end_to_end
    from test_workloads_end_to_end import cli

    monkeypatch.setattr(test_workloads_end_to_end, "ADMIN", ADMIN)
    code, out, err = cli(pool, capsys, monkeypatch, "workload", "plan", "research",
                         "--model", CHAT, "--latency", "30", "--parallel", "2",
                         "--model", EMBED, "--latency", "5", "--parallel", "2", "--hours", "2", "--max-spend", "5")
    assert code == 0, err
    assert "chat (chat): 2 at once, 30s p95" in out and "together: 2 host(s)" in out and "chat 1, embed 1 each" in out
    assert pool.supervisor.workloads.store.all() == []
    code, _, err = cli(pool, capsys, monkeypatch, "workload", "plan", "research",
                       "--model", CHAT, "--latency", "30", "--model", EMBED, "--hours", "2")
    assert code == 2 and "repeat the three" in err


def test_an_infinite_number_is_refused_in_words_not_a_500(pool):
    """Python's JSON reads 1e400 as infinity; `int()` of it raised past the handler."""
    with admin(pool) as control:
        answer = control.post("/pool/workloads/plan", headers={"content-type": "application/json"},
                              content='{"name": "research", "hours": 2, "model": "chat", "latency_s": 5, "parallel": 1e400}')
    assert answer.status_code == 400 and "above zero" in answer.json()["detail"]


# --- found in review ---


def targets(*models):
    from gpm_server.workload_store import ModelTarget

    return [ModelTarget(m, latency, parallel) for m, latency, parallel in models]


def plan_of(pool, *models, placement="auto", max_spend=5.0):
    from gpm_server.supervisor.workloads import WorkloadRequest

    return pool.loop.run(pool.supervisor.workloads.plan(
        WorkloadRequest.of("research", targets(*models), 2, max_spend=max_spend, placement=placement)))


def test_auto_keeps_the_placement_the_caps_allow_when_the_cheaper_is_refused(pool, monkeypatch):
    """Found in review: auto chose the cheaper placement and only then asked the caps, so a
    workload was refused that the other placement would have served."""
    workloads = pool.supervisor.workloads
    real = workloads._price

    def priced(req, sketch, placement, offers, builds, cards):
        option = real(req, sketch, placement, offers, builds, cards)
        if len(placement) == 1:   # together: one host, dearer per hour
            option.update(hosts=1, hourly=0.60, expected=2.40)
        else:                     # apart: cheaper, on more hosts than the pool has room for
            option.update(hosts=2, hourly=0.50, expected=2.00)
        return option

    monkeypatch.setattr(workloads, "_price", priced)
    pool.supervisor.config.limits.max_rented_hosts = 1
    plan = plan_of(pool, (CHAT, 30, 1), (EMBED, 5, 1))
    assert plan["refused"] is None and plan["placement"] == "together", plan
    assert "room for 1" in plan["placements"]["apart"]["refused"]
    assert any("the only placement the pool's caps allow" in r for r in plan["reasons"])
    pool.supervisor.config.limits.max_rented_hosts = 4
    assert plan_of(pool, (CHAT, 30, 1), (EMBED, 5, 1))["placement"] == "apart", "with room, the cheaper"


def test_a_forced_placement_says_it_was_asked_for(pool):
    plan = plan_of(pool, (CHAT, 30, 2), (EMBED, 5, 2), placement="apart")
    assert "placement: apart, as asked" in plan["reasons"]
    assert set(plan["placements"]) == {"apart"}


def test_a_lost_split_host_is_replaced_on_the_card_it_was_planned_on(pool):
    """Found in review: a split host only runs a model up to its share, so its answers could
    never show the card holds more — and once measured, every offer of that card was refused."""
    import time as clock

    create(pool, request())
    fleet = pool.supervisor.fleet
    until(pool, lambda: len([h for h in fleet.hosts_of("research") if h.state == "ready"]) == 2, "both hosts ready")
    host = fleet.hosts_of("research")[0]
    # Twenty answers of chat alone at its share (1 at once), well within target: the curve now
    # stops at 1, which is all a split host could ever show.
    for i in range(25):
        pool.database.execute(
            "INSERT INTO request_log (ts, request_id, outcome, host_id, model_served, latency_ms, concurrency, "
            "model_concurrency, workload) VALUES (?, ?, 'ok', ?, ?, 500, 1, 1, 'research')",
            (clock.time(), f"m{i}", host.host_id, CHAT))
    fleet._sizing_cache.clear()
    fleet._hardware_cache = None  # read again: this host was rented after it was last read
    spec = pool.supervisor.workloads.get("research")
    assert fleet.model_at_latency(host.offer.hardware, CHAT, 30, ceiling=2).curve, "the answers are read"
    spec = pool.supervisor.workloads.get("research")
    assert all(fleet.split_fits_on(o, spec, spec.groups[0]) for o in fleet.provider.offers)
    pool.loop.run(fleet.destroy(host, "evicted, in this test"))
    until(pool, lambda: len([h for h in fleet.hosts_of("research") if h.state == "ready"]) == 2, "replaced")


def test_a_split_group_is_rented_on_the_cheaper_card_not_the_bigger(pool):
    """Found in review: at rent time a group's offers were ranked by what a card could hold, so a
    bigger card won — and then ran the same fixed split as the smaller one, at a higher price."""
    from gpm_server.config import CapacityProfile
    from gpm_server.providers import default_offer

    fleet = pool.supervisor.fleet
    fleet.provider.offers = [default_offer(f"small-{i}", f"m-small-{i}", min_bid_hourly=0.20) for i in range(3)]
    create(pool, request())
    until(pool, lambda: len([h for h in fleet.hosts_of("research") if h.state == "ready"]) == 2, "both hosts ready")
    # Now a bigger card appears: four of each alone, at a lower price per worker it could hold,
    # and a higher price per host.
    pool.supervisor.config.capacity_profiles.append(
        CapacityProfile.model_validate({"match": {"hardware": "FakeGPU 96GB"}, "max_workers": 8}))
    fleet.provider.offers.append(default_offer("big", "m-big", min_bid_hourly=0.36, hardware="FakeGPU 96GB",
                                               gpu_memory_gb=96.0))
    spec = pool.supervisor.workloads.get("research")
    assert fleet.capacity_on(fleet.provider.offers[-1], spec, (CHAT, EMBED), spec.builds) > spec.groups[0].workers_per_host
    lost = fleet.hosts_of("research")[0]
    pool.loop.run(fleet.destroy(lost, "evicted, in this test"))
    until(pool, lambda: len([h for h in fleet.hosts_of("research") if h.state == "ready"]) == 2, "replaced")
    assert {h.offer.machine_id for h in fleet.hosts_of("research")} <= {"m-small-0", "m-small-1", "m-small-2"}


def test_a_split_hosts_worker_count_is_not_changed_by_hand(pool):
    create(pool, request())
    fleet = pool.supervisor.fleet
    until(pool, lambda: len([h for h in fleet.hosts_of("research") if h.state == "ready"]) == 2, "both hosts ready")
    host = fleet.hosts_of("research")[0]
    done, why = pool.loop.run(fleet.resize(host, 1))
    assert not done and "split" in why and host.workers == 2


def test_each_model_is_judged_against_its_own_target(pool):
    """Found in review: one mixed p95 against the tightest target read a slow chat as a miss for
    both, and a busy model crowded the other out of the window."""
    import time as clock

    create(pool, request())
    rows = [(CHAT, 14_000)] * 30 + [(EMBED, 400)] * 2100
    for i, (model, ms) in enumerate(rows):
        pool.database.execute(
            "INSERT INTO request_log (ts, request_id, outcome, model_requested, latency_ms, workload) "
            "VALUES (?, ?, 'ok', ?, ?, 'research')", (clock.time(), f"r{i}", model, ms))
    view = pool.supervisor.workloads.view(pool.supervisor.workloads.get("research"))
    answers = view["answers"]
    assert answers["by_model"][CHAT]["count"] == 30 and answers["by_model"][CHAT]["meets_target"]
    assert answers["by_model"][EMBED]["meets_target"] and answers["meets_target"] is True


def test_no_host_takes_more_than_its_share_of_a_model_under_load(pool):
    """The split, end to end: many chat answers at once on hosts whose share of chat is one each
    are served one per host at a time, and the rest wait — never two chats on one host."""
    import concurrent.futures

    for engine in pool.rentable.values():
        engine.fake.chunk_delay_s = 0.3
    key = create(pool, request())["connection"]["api_key"]
    fleet = pool.supervisor.fleet
    until(pool, lambda: pool.supervisor.workloads.get("research").state == "serving", "serving")
    pool.loop.run(pool.state.registry.refresh())

    def one(i):
        with pool.client(key=key, timeout=30) as app:
            return app.post("/v1/chat/completions", json={"model": CHAT if i % 3 else EMBED,
                                                          "messages": [{"role": "user", "content": "hi"}]})

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as threads:
        answers = list(threads.map(one, range(12)))
    assert all(a.status_code == 200 for a in answers), [a.text for a in answers if a.status_code != 200]
    served = [r for r in pool.wait_for_log(12) if r["outcome"] == "ok"]
    assert len(served) == 12
    assert max(r["model_concurrency"] for r in served) == 1, "a host's share of each model is one"
    assert max(r["concurrency"] for r in served) <= 2, "and its workers are the shares' sum"
    assert {r["host_id"] for r in served} <= {h.host_id for h in fleet.hosts_of("research")}
