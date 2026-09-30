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
    assert forced["refused"] and "no machine" in forced["refused"]


def test_one_group_filling_its_floor_does_not_hide_anothers(pool):
    """Reservations are per group: the chat group at its start leaves the embed group's own."""
    create(pool, request("apart"))
    fleet = pool.supervisor.fleet
    pool.reprobe()
    workload = pool.supervisor.workloads.get("research")
    groups = {g.key: g for g in workload.groups}
    for key in groups:
        others = fleet.reserved_hosts(besides="research", group=key)
        have = {k: len(fleet.hosts_of("research", k)) for k in groups}
        expected = sum(max(0, g.hosts_at_start - have[k]) for k, g in groups.items() if k != key)
        assert others == expected


@pytest.mark.parametrize("body, words", [
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 1}] * 2}, "each model is named once"),
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 1}], "model": CHAT}, "not both"),
    ({"models": [{"model": CHAT, "latency_s": 5, "parallel": 0}]}, "at least 1"),
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
