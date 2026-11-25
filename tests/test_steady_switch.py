"""A rule's steady switch, handed over make-before-break and paid for before it is allowed
(S19 stage 5b, D200; workloads.md §14.3).

A rule may switch where a workload runs: to the operator's own machines (with workers reserved
for it), or to rented machines of another kind. The supervisor keeps the old steady serving until
the new one is ready, then gives the old hosts up; the plan's budget is priced at the dearest
steady the rules can reach, with both sets of hosts while the new are prepared. Nothing here
spends money.
"""

import time
from types import SimpleNamespace

import httpx
import pytest
from gpm_client import WorkloadRefused
from gpm_server.serving_plan import PlanRefused, end_of_day, parse_serve, steady_wanted
from test_serving_plan_routing import ADMIN, BIG, SPEC, grant, programs, provider, provisioner  # noqa: F401

EVICTED = {"name": "evicted", "when": {"evictions": {"count": 1, "within_h": 1}},
           "then": {"steady": {"rented": {"kind": "on_demand"}}}}


# --- the plan ---


def test_a_switch_is_read_and_refused_in_words():
    plan = parse_serve({"reserve": {"local": 2}, "rules": [
        EVICTED, {"name": "nights", "when": {"between": "22:00-06:00"}, "then": {"steady": "local"},
                  "until": "window_end"}]})
    assert [r.switch for r in plan.rules] == ["rented:on_demand", "local"]
    assert parse_serve(plan.as_dict()).as_dict() == plan.as_dict()
    for serve, says in [
        ({"rules": [{"name": "x", "when": {"between": "22:00-06:00"}, "then": {"steady": "local"}}]}, "add reserve"),
        ({"rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"steady": {"rented": {"kind": "roi"}}}}]},
         "never per request"),
        ({"rules": [{**EVICTED, "until": "window_end"}]}, "needs a window"),
        ({"rules": [{"name": "x", "when": {"between": "22:00-06:00"}, "then": {"use_only": ["rented"]},
                     "until": "end_of_day"}]}, "this rule has none"),
        ({"steady": "local", "reserve": {"local": 1},
          "rules": [{"name": "x", "when": {"between": "22:00-06:00"}, "then": {"steady": {"rented": {"kind": "roi"}}}}]},
         "rent a base the workload was created without"),
        ({"rules": [{"name": "x", "when": {"between": "22:00-06:00"}, "then": {"steady": "rented"}}]}, "rented already"),
    ]:
        with pytest.raises(PlanRefused, match=says):
            parse_serve(serve)


def test_a_switch_is_held_by_its_until():
    plan = parse_serve({"reserve": {"local": 2}, "tz": "America/New_York", "rules": [
        {"name": "busy day", "when": {"evictions": {"count": 2, "within_h": 1}}, "then": {"steady": "local"},
         "until": "end_of_day"}]})
    now = time.time()
    began = steady_wanted(plan, frozenset({"busy day"}), None, now)
    assert began["to"] == "local" and began["until"] == end_of_day("America/New_York", now)
    assert steady_wanted(plan, frozenset({"busy day"}), began, now + 60)["since"] == began["since"], "the same switch"
    assert steady_wanted(plan, frozenset(), began, now + 60) == began, "held to the end of the day"
    assert steady_wanted(plan, frozenset(), began, began["until"] + 1) is None
    window = parse_serve({"reserve": {"local": 2}, "rules": [
        {"name": "w", "when": {"between": "00:00-23:59", "evictions": {"count": 1, "within_h": 1}},
         "then": {"steady": "local"}, "until": "window_end"}]})
    held = steady_wanted(window, frozenset({"w"}), None, now)
    assert steady_wanted(window, frozenset(), held, now) == held, "evictions forgotten, still inside its window"


# --- the supervisor ---


def create(pool, **body):
    return httpx.post(f"{pool.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"},
                      json={"hours": 1, "latency_s": 30, "parallel": 2, "max_spend": 5.0, **body}, timeout=60)


SPOT = {**EVICTED, "then": {"steady": {"rented": {"kind": "interruptible"}}}}


def test_a_switch_to_another_kind_is_priced_and_rents_that_kind_once_in_force(programs):  # noqa: F811
    made = create(programs, name="night", model=BIG, serve={"rules": [SPOT]})
    assert made.status_code == 201, made.text
    plan = made.json()["plan"]
    assert plan["switch_budget"]["switches"] == ["rented:interruptible"] and plan["switch_budget"]["needed"] > 0
    # Both ways, each time its rule can begin: an hour's lease, one eviction window — twice at most.
    assert plan["switch_budget"]["times"] == 2 and plan["switch_budget"]["overlaps"] > 0
    manager, fleet = programs.supervisor.workloads, programs.supervisor.fleet
    programs.loop.run(manager.advance())
    assert fleet.kind_in_force("night") == "roi"
    programs.supervisor.events.record("eviction", "outbid", lease_id=manager.get("night").lease_id)
    programs.loop.run(manager.advance())
    switch = manager.store.steady_switches()["night"]
    assert switch["want"]["to"] == "rented:interruptible" and switch["serving"] is None
    assert fleet.kind_in_force("night") == "interruptible"
    events = programs.supervisor.events.recent(10, kind="workload_steady")
    assert events and "switching to rented:interruptible by rule 'evicted'" in events[0]["summary"]
    refused = create(programs, name="same", model=BIG, kind="interruptible", serve={"rules": [SPOT]})
    assert refused.status_code == 400 and "which it rents already" in refused.text


def test_a_switch_to_the_operators_machines_serves_there_and_gives_the_rented_hosts_up(programs):  # noqa: F811
    serve = {"reserve": {"local": 2}, "rules": [{"name": "evicted", "when": {"evictions": {"count": 1, "within_h": 1}},
                                                 "then": {"steady": "local"}}]}
    made = create(programs, name="night", model=BIG, serve=serve)
    assert made.status_code == 201, made.text
    manager, fleet = programs.supervisor.workloads, programs.supervisor.fleet
    programs.supervisor.events.record("eviction", "outbid", lease_id=manager.get("night").lease_id)
    programs.loop.run(manager.advance())
    switch = manager.store.steady_switches()["night"]
    assert switch["serving"] == "local", "the laptop holds the model and is ready: served there at once"
    drained = []

    async def drain(host, reason, then="destroy"):
        drained.append((host.host_id, reason))

    fleet.drain, hosts_of = drain, fleet.hosts_of
    fleet.hosts_of = lambda name, group=None: [SimpleNamespace(host_id="r-1", state="ready", models=(BIG,), burst=False)]
    try:
        programs.loop.run(fleet.hand_over_steady())
    finally:
        fleet.hosts_of = hosts_of
    assert drained == [("r-1", "workload night: steady switched to the operator's local machines")]
    programs.loop.run(manager.advance())
    assert manager.get("night").state == "serving", "served on the operator's machines: serving, not preparing"
    assert not [k for k in manager.store.gaps() if k[0] == "night"], "its rented hosts are meant to go: no gap"
    programs.loop.run(programs.state.registry.refresh())
    key = made.json()["connection"]["api_key"]
    answer = httpx.post(f"{programs.url}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, timeout=30,
                        json={"model": BIG, "messages": [{"role": "user", "content": "hi"}]})
    assert answer.status_code == 200 and answer.headers["x-gpm-route"] == "steady/local"


def test_a_program_switches_only_to_kinds_its_grant_allows(programs):  # noqa: F811
    key = grant(programs, kinds=["roi"])
    with provisioner(programs, key) as pool_side, pytest.raises(WorkloadRefused, match="switches steady to on_demand"):
        pool_side.workload(BIG, **SPEC, serve={"rules": [EVICTED]})


def test_a_switch_is_not_added_to_a_running_workload_and_one_removed_hands_back(programs):  # noqa: F811
    plain = create(programs, name="plain", model=BIG, serve={"until_ready": ["shared"]})
    assert plain.status_code == 201, plain.text
    added = httpx.post(f"{programs.control_url}/pool/workloads/plain/policy", headers={"Authorization": f"Bearer {ADMIN}"},
                       json={"serve": {"until_ready": ["shared"], "rules": [SPOT]}})
    assert added.status_code == 400 and "checked and priced when a workload is created" in added.text
    serve = {"reserve": {"local": 2}, "rules": [{"name": "evicted", "when": {"evictions": {"count": 1, "within_h": 1}},
                                                 "then": {"steady": "local"}}]}
    made = create(programs, name="night", model=BIG, serve=serve)
    assert made.status_code == 201, made.text
    manager = programs.supervisor.workloads
    programs.supervisor.events.record("eviction", "outbid", lease_id=manager.get("night").lease_id)
    programs.loop.run(manager.advance())
    assert manager.store.steady_switches()["night"]["serving"] == "local"
    removed = httpx.post(f"{programs.control_url}/pool/workloads/night/policy",
                         headers={"Authorization": f"Bearer {ADMIN}"}, json={"serve": {"reserve": {"local": 2}}})
    assert removed.status_code == 200, removed.text
    programs.loop.run(manager.advance())
    switch = manager.store.steady_switches()["night"]
    assert switch["want"] is None and switch["serving"] == "local", "kept on local until its rented base is ready"


def test_a_switch_between_kinds_hands_over_a_host_at_a_time(programs):  # noqa: F811
    fleet = programs.supervisor.fleet
    group = SimpleNamespace(holds=lambda models: True, hosts_at_start=2)
    workloads, hosts_of, drain = fleet.workloads, fleet.hosts_of, fleet.drain
    hosts = [SimpleNamespace(host_id=h, workload="w", state=state, models=(BIG,), burst=False, interruptible=spot,
                             bid_hourly=price)
             for h, state, spot, price in [("old-1", "ready", True, 0.4), ("old-2", "ready", True, 0.5),
                                           ("old-3", "preparing", True, 0.4), ("new-1", "ready", False, 0.9),
                                           ("new-2", "preparing", False, 0.9)]]
    drained = []

    async def gone(host, reason, then="destroy"):
        drained.append(host.host_id)
        host.state = "draining"

    # Its rule was removed: the kind in force is machines= again (on_demand), with no switch held.
    fleet.workloads = {**workloads, "w": SimpleNamespace(kind="on_demand", groups=[group], policy={})}
    fleet.hosts_of, fleet.drain, fleet.steady_targets = (lambda name, g=None: hosts if name == "w" else []), gone, {}
    try:
        programs.loop.run(fleet.hand_over_steady())
        assert drained == ["old-3", "old-2"], "the one preparing at once; one ready, the dearer, for the one ready new"
        hosts[4].state = "ready"
        programs.loop.run(fleet.hand_over_steady())
        assert drained == ["old-3", "old-2", "old-1"], "a whole new base ready: the last old one goes"
    finally:
        fleet.workloads, fleet.hosts_of, fleet.drain = workloads, hosts_of, drain
