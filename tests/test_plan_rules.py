"""A serving plan's rules (S19 stage 5, D200; workloads.md §14.3).

`{name, when, then}`, at most 16, a closed list. The supervisor judges the conditions on the
pass — a time window in the plan's time zone, a budget's spend, evictions — and writes the rules
in force; the router judges a request's options and applies what the rules in force say: sources
kept out (`use_only`, `never_use`) and shorter waits (`after_s`). They only narrow, and add up.
Nothing here spends money.
"""

import datetime
import threading
import time

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_server.serving_plan import (
    PlanRefused,
    compile_policy,
    in_window,
    narrowed,
    narrowing,
    parse_serve,
)
from gpm_server.workload_store import Workload, WorkloadStore
from test_serving_plan_routing import BIG, SPEC, grant, programs, provider, provisioner  # noqa: F401 - fixtures

MODEL = "m1"
PRICE = {"input": 3.0, "output": 15.0}
BURST = {"name": "burst", "hosted": {"after_s": 5, "budget": {"per_day": 5.0, "per_month": 50.0}}}
NIGHT = {"name": "night", "when": {"between": "22:00-06:00"}, "then": {"never_use": ["burst"]}}


def at(text, tz="America/New_York"):
    from zoneinfo import ZoneInfo

    return datetime.datetime.fromisoformat(text).replace(tzinfo=ZoneInfo(tz)).timestamp()


# --- the plan ---


def test_rules_are_read_and_refused_in_words():
    plan = parse_serve({"tz": "America/New_York", "when_full": [BURST], "rules": [
        NIGHT,
        {"name": "fast for people", "when": {"priority": "interactive"}, "then": {"after_s": {"burst": 0}}},
        {"name": "burst nearly spent", "when": {"spent_over": {"budget": "burst", "period": "month", "share": 0.8}},
         "then": {"never_use": ["hosted"]}},
        {"name": "evicted", "when": {"evictions": {"count": 3, "within_h": 24}}, "then": {"use_only": ["rented"]}},
    ]})
    assert [r.name for r in plan.rules] == ["night", "fast for people", "burst nearly spent", "evicted"]
    assert parse_serve(plan.as_dict()).as_dict() == plan.as_dict()
    refused = [
        ({"rules": [{"name": f"r{i}", "when": {"priority": "batch"}, "then": {"never_use": ["burst"]}}
                    for i in range(17)], "when_full": [BURST]}, "at most 16 rules"),
        ({"rules": [{"name": "x", "when": {"weather": "rain"}, "then": {"never_use": ["burst"]}}], "when_full": [BURST]},
         "the conditions are"),
        ({"rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"steady": "local"}}]}, "add reserve"),
        ({"rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"never_use": ["rented"]}}]}, "removes steady"),
        ({"rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"use_only": ["burst"]}}], "when_full": [BURST]},
         "leaves out steady"),
        ({"rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"after_s": {"burst": 9}}}], "when_full": [BURST]},
         "a shorter wait only"),
        ({"rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"never_use": ["moon"]}}]}, "no source of this plan"),
        ({"rules": [{"name": "x", "when": {"spent_over": {"budget": "burst", "period": "week", "share": 1}},
                     "then": {"never_use": ["burst"]}}], "when_full": [BURST]}, "period is day or month"),
        ({"rules": [{"name": "x", "when": {"days": ["mon"]}, "then": {"never_use": ["burst"]}}], "when_full": [BURST]},
         "days goes with between"),
        ({"tz": "Mars/Olympus"}, "not an IANA time zone"),
    ]
    for serve, says in refused:
        with pytest.raises(PlanRefused, match=says):
            parse_serve(serve)


def test_rules_in_force_narrow_and_add_up():
    plan = parse_serve({"until_ready": ["local", {"hosted": {"budget": {"per_day": 1.0}}}], "when_full": [BURST],
                        "rules": [NIGHT,
                                  {"name": "batch", "when": {"priority": "batch"}, "then": {"never_use": ["local"]}},
                                  {"name": "fast", "when": {"keep_on": "any"}, "then": {"after_s": {"burst": 1}}}]})
    by_day = narrowing(plan, frozenset(), "interactive", "any")
    assert by_day.by == ("fast",) and by_day.excluded == frozenset()
    assert narrowed(plan, by_day).when_full[0].after_s == 1
    by_night = narrowing(plan, frozenset({"night"}), "batch", "no_hosted")
    assert by_night.by == ("night", "batch") and by_night.excluded == {"burst", "until_ready/local"}
    seen = narrowed(plan, by_night)
    assert [s.name for s in seen.until_ready] == ["hosted"] and seen.when_full == ()


def test_a_time_window_is_read_in_the_plans_zone_across_midnight_and_dst():
    when = {"between": "22:00-06:00", "days": ["fri"]}
    assert in_window(when, "America/New_York", at("2026-11-27T23:30"))  # a Friday night
    assert in_window(when, "America/New_York", at("2026-11-28T05:59")), "past midnight: the window opened on Friday"
    assert not in_window(when, "America/New_York", at("2026-11-28T23:30")), "Saturday"
    # 02:30 does not exist on 2026-03-08 in New York: a window opening then takes effect at 03:00.
    spring = {"between": "02:30-04:00"}
    assert in_window(spring, "America/New_York", at("2026-03-08T03:00"))
    # 01:00-02:00 repeats on 2026-11-01: a window ending at 01:30 ended at its first occurrence.
    from zoneinfo import ZoneInfo

    second = datetime.datetime(2026, 11, 1, 1, 10, tzinfo=ZoneInfo("America/New_York"), fold=1).timestamp()
    assert not in_window({"between": "00:00-01:30"}, "America/New_York", second)
    assert in_window({"between": "01:20-03:00"}, "America/New_York", second)


# --- the router ---


@pytest.fixture
def pool(monkeypatch):
    loop = BackgroundLoop()
    fake = FakeHostedApi(models=("provider-m1",))
    server = ServerHandle(fake.app, loop)
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    catalog = {MODEL: {"variants": [{"tag": MODEL, "engine": "ollama"},
                                    {"tag": "provider-m1", "engine": "openai-api", "price": PRICE}]}}
    hosted = {"id": "provider", "kind": "hosted", "workers": 4,
              "transport": {"type": "http", "base_url": f"{server.base_url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}
    with pool_harness([EngineSpec(id="rented", resident={MODEL}, workers=1, chunk_delay_s=1.5)],
                      model_set=[MODEL], catalog=catalog, extra_hosts=[hosted]) as harness:
        harness.reprobe()
        harness.stop_supervisor()
        harness.database.execute("UPDATE hosts SET workload = 'research', kind = 'rented-on-demand', updated_at = ? "
                                 "WHERE host_id = 'rented'", (time.time() + 1,))
        harness.loop.run(harness.state.registry.refresh())
        yield harness, fake
    server.stop()
    loop.stop()


def add_workload(pool, serve):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name="research", model=MODEL, builds={MODEL: MODEL}, latency_s=20, parallel=1, kind="roi",
        lease_id="lease-research", state="serving", workers_per_host=1, hosts_at_start=1,
        plan={"may_borrow": False}, created_at=now, updated_at=now, policy=compile_policy(parse_serve(serve)),
    ))
    _, key = store.mint_key("research")
    pool.loop.run(pool.state.registry.refresh())
    return store, key


def ask(pool, key, stream=False, **headers):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30,
                      headers={"Authorization": f"Bearer {key}", **headers},
                      json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": stream})


def test_a_rule_on_a_requests_options_shortens_a_wait(pool):
    harness, fake = pool
    _, key = add_workload(harness, {"when_full": [BURST], "rules": [
        {"name": "people first", "when": {"priority": "interactive"}, "then": {"after_s": {"burst": 0}}}]})
    slow = threading.Thread(target=lambda: ask(harness, key, stream=True))
    slow.start()
    time.sleep(0.3)
    answer = ask(harness, key)
    slow.join()
    assert answer.headers["x-gpm-host"] == "provider" and float(answer.headers["x-gpm-wait-s"]) < 1.0
    assert answer.headers["x-gpm-route"] == "when_full/burst"


def test_a_rule_the_supervisor_put_in_force_keeps_a_source_out_until_it_leaves_it(pool):
    harness, fake = pool
    store, key = add_workload(harness, {"when_full": [{"name": "burst", "hosted": {"after_s": 0,
                                                                                  "budget": {"per_day": 5.0}}}],
                                        "rules": [{**NIGHT}]})
    store.set_armed_rules("research", ["night"])
    harness.loop.run(harness.state.registry.refresh())
    slow = threading.Thread(target=lambda: ask(harness, key, stream=True))
    slow.start()
    time.sleep(0.3)
    started = time.monotonic()
    answer = ask(harness, key)
    slow.join()
    assert answer.headers["x-gpm-host"] == "rented" and time.monotonic() - started > 1.0, "kept off the burst"
    store.set_armed_rules("research", [])
    harness.loop.run(harness.state.registry.refresh())
    slow = threading.Thread(target=lambda: ask(harness, key, stream=True))
    slow.start()
    time.sleep(0.3)
    assert ask(harness, key).headers["x-gpm-host"] == "provider", "the window ended: the burst is back"
    slow.join()


# --- the supervisor ---


def test_the_supervisor_puts_a_rule_in_force_on_evictions_and_holds_a_burst_back(programs):  # noqa: F811
    key = grant(programs)
    serve = {"when_full": [{"name": "extra", "rented": {"max_hosts": 1, "budget": {"per_day": 1.0}}}],
             "rules": [{"name": "evicted", "when": {"evictions": {"count": 2, "within_h": 1}},
                        "then": {"never_use": ["extra"]}}]}
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC, serve=serve) as w:
        manager = programs.supervisor.workloads
        lease = manager.get(w.name).lease_id
        programs.loop.run(manager.advance())
        assert manager.store.armed_rules().get(w.name, frozenset()) == frozenset()
        for _ in range(2):
            programs.supervisor.events.record("eviction", "outbid", lease_id=lease)
        programs.loop.run(manager.advance())
        assert manager.store.armed_rules()[w.name] == {"evicted"}
        assert w.name in programs.supervisor.fleet.bursts_held_back
        events = programs.supervisor.events.recent(10, kind="workload_rules")
        assert events and "rule 'evicted' in force" in events[0]["summary"]
        programs.loop.run(programs.state.registry.refresh())


def test_a_source_is_named_by_its_budget_id_and_an_unnamed_wait_shortened_in_its_role():
    plan = parse_serve({"until_ready": [{"hosted": {"budget": {"per_day": 1.0}}}],
                        "when_full": [{"hosted": {"budget": {"per_day": 1.0}}},
                                      {"rented": {"max_hosts": 1, "budget": {"per_day": 1.0}}}],
                        "rules": [{"name": "night", "when": {"between": "22:00-06:00"},
                                   "then": {"never_use": ["when_full/rented"], "after_s": {"hosted": 2}}}]})
    seen = narrowed(plan, narrowing(plan, frozenset({"night"}), "interactive", "any"))
    assert seen.rented() is None and seen.when_full[0].after_s == 2 and seen.until_ready[0].after_s == 0
    with pytest.raises(PlanRefused, match="cannot keep the rented burst out"):
        parse_serve({"when_full": [{"name": "extra", "rented": {"max_hosts": 1, "budget": {"per_day": 1.0}}}],
                     "rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"never_use": ["extra"]}}]})


def test_a_rule_in_force_holds_for_a_model_served_by_hosted_apis_alone(pool):
    harness, fake = pool
    harness.database.execute("UPDATE hosts SET workload = NULL, updated_at = ? WHERE host_id = 'rented'",
                             (time.time() + 1,))
    store = WorkloadStore(harness.database)
    now = time.time()
    store.create(Workload(
        name="research", model=MODEL, builds={}, latency_s=20, parallel=0, kind="roi", lease_id="lease-research",
        state="serving", workers_per_host=0, hosts_at_start=0, plan={"may_borrow": False}, created_at=now,
        updated_at=now, policy=compile_policy(parse_serve({"when_full": [BURST], "rules": [NIGHT]}))))
    _, key = store.mint_key("research")
    store.set_armed_rules("research", ["night"])
    harness.loop.run(harness.state.registry.refresh())
    answer = ask(harness, key)
    assert answer.status_code == 503 and answer.json()["reason"] == "no_allowed_source"
    assert "the plan's rules in force: night" in answer.json()["detail"]
    assert fake.seen == []
