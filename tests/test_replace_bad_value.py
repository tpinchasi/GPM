"""A host that is far worse value than the rest is given up, not merely shrunk (D75,
hosts-routing-capacity.md §2.4): its cost per unit of work — hourly all-in over tokens a second —
against the other hosts serving the same model, for long enough, on enough answers; and only
within the bounds — one at a time, never the last ready host, never without a lease that covers
the replacement's download. Against the fake provider; nothing here spends money."""

import time
from types import SimpleNamespace

import pytest
from gpm_server.providers import FakeProvider
from gpm_server.strategies import ValueReading, judge_value
from test_workload_fleet import make_fleet, offers, open_workload, run

CFG = SimpleNamespace(replace_above_factor=3.0, replace_after_s=600.0, replace_min_requests=20)
NOW = 10_000.0


def reading(host, *, hourly=0.50, tps=100.0, model="m", requests=50, bad_since=None):
    return ValueReading(host, model, hourly, tps, requests, bad_since)


def verdicts(*readings):
    return {v.host_id: v for v in judge_value(list(readings), CFG, NOW)}


# --- the judgement ---


def test_a_host_costing_far_more_per_token_than_the_others_is_worse():
    found = verdicts(reading("a"), reading("b"), reading("slow", tps=20.0))
    assert found["slow"].worse and found["slow"].ratio == pytest.approx(5.0)
    assert not found["a"].worse and not found["b"].worse
    assert "per million tokens" in found["slow"].reasons[0]


def test_it_is_given_up_only_once_it_has_stayed_worse_for_long_enough():
    assert not verdicts(reading("a"), reading("slow", tps=20.0, bad_since=NOW - 60))["slow"].replace
    assert verdicts(reading("a"), reading("slow", tps=20.0, bad_since=NOW - 601))["slow"].replace


def test_work_is_counted_not_slots():
    # Six workers serving slowly cost more per token than two serving fast: that is the machine
    # this is for, and counting workers would have rated it the better one (D75).
    found = verdicts(reading("two-fast", tps=120.0), reading("six-slow", tps=30.0, hourly=0.80))
    assert found["six-slow"].worse


def test_hosts_are_compared_only_with_others_serving_the_same_model():
    # A larger model is slower everywhere; that does not make its host bad value.
    found = verdicts(reading("small", tps=200.0, model="small"), reading("big", tps=20.0, model="big"))
    assert not found["big"].worse and found["big"].ratio is None


def test_a_dear_market_has_no_bad_host_only_dear_hosts():
    found = verdicts(reading("a", hourly=4.0), reading("b", hourly=4.2))
    assert not any(v.worse for v in found.values())


def test_too_few_answers_or_no_traffic_is_not_judged():
    found = verdicts(reading("a"), reading("thin", tps=5.0, requests=3), reading("idle", tps=None))
    assert not found["thin"].worse and not found["idle"].worse


# --- the bounds, and the act ---


@pytest.fixture
def db(tmp_path):
    from gpm_server.db import Database

    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


async def two_ready(db, max_spend=20.0):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, lease = open_workload(fleet, hosts_at_start=2, max_spend=max_spend)
    await run(fleet, [workload])
    first, second = fleet.hosts_of("research")
    first.state = second.state = "ready"
    return fleet, workload, first, second


async def test_a_bad_value_host_is_drained_its_machine_avoided_and_said(db):
    fleet, _, first, second = await two_ready(db)
    assert fleet.may_replace(second) is None
    await fleet.give_up_for_value(second, "$9.00 per million tokens, 5.0x the others'", 5.0)
    assert second.state == "draining"
    assert second.offer.machine_id in fleet.avoided_now(), "not bought straight back"
    (event,) = [e for e in fleet.events.recent(20) if e["kind"] == "replaced_for_value"]
    assert event["numbers"]["ratio"] == 5.0 and "rents its replacement" in event["summary"]


async def test_one_at_a_time(db):
    fleet, _, first, second = await two_ready(db)
    await fleet.give_up_for_value(second, "worse", 5.0)
    assert "one at a time" in (fleet.may_replace(first) or "")


async def test_never_the_last_ready_host(db):
    fleet, _, first, second = await two_ready(db)
    first.state = "preparing"
    assert "still being prepared" in fleet.may_replace(second)
    first.state = "draining"
    assert "only ready host" in fleet.may_replace(second)


async def test_never_without_a_lease_that_covers_the_replacements_download(db):
    fleet, _, first, second = await two_ready(db)
    fleet.lease_spend = lambda lease: (lease.max_spend - 0.01, 0.0)  # all but a cent spent
    second.download_cost = 0.50
    assert "less than" in fleet.may_replace(second)


def log_answers(db, host, *, answers, tokens, seconds, every, now):
    """`answers` answers of `tokens` tokens each, each taking `seconds`, one every `every` s."""
    for i in range(answers):
        db.execute("INSERT INTO request_log (ts, request_id, host_id, model_requested, model_served, outcome, "
                   "tokens_out, latency_ms) VALUES (?, ?, ?, 'big', 'big', 'ok', ?, ?)",
                   (now - every * i, f"{host.host_id}-{i}", host.host_id, tokens, seconds * 1000))


def bare_supervisor(fleet, db):
    from gpm_server.supervisor.service import Supervisor

    supervisor = Supervisor.__new__(Supervisor)
    supervisor.fleet, supervisor.db = fleet, db
    supervisor.config = fleet.config.model_copy(deep=True)
    supervisor.config.rented.workers_auto.enabled = True
    return supervisor


async def test_the_supervisors_pass_gives_up_a_bad_value_host_on_its_measurements(db):
    # The pass, end to end over the request log: one host serving a fifth of the other's tokens
    # a second, for longer than replace_after_s, is given up; the other stays.
    from gpm_server.supervisor.service import Supervisor

    fleet, workload, first, second = await two_ready(db)
    now = time.time()
    log_answers(db, first, answers=30, tokens=400, seconds=1.0, every=2.0, now=now)
    log_answers(db, second, answers=30, tokens=80, seconds=1.0, every=2.0, now=now)
    second.bad_value_since = now - 3600
    await Supervisor._replace_bad_value(bare_supervisor(fleet, db))
    assert second.state == "draining" and first.state == "ready"


async def test_a_host_the_router_sent_less_work_is_not_bad_value(db):
    # Same speed, a third of the traffic — idle much of the window, or ready only late in it.
    # Counted over the whole window it would read 3x worse; counted while serving, it is equal.
    from gpm_server.supervisor.service import Supervisor

    fleet, workload, first, second = await two_ready(db)
    now = time.time()
    log_answers(db, first, answers=90, tokens=400, seconds=1.0, every=1.0, now=now)
    log_answers(db, second, answers=30, tokens=400, seconds=1.0, every=1.0, now=now)
    second.bad_value_since = now - 3600
    await Supervisor._replace_bad_value(bare_supervisor(fleet, db))
    assert second.state == "ready" and second.bad_value_since is None


def test_busy_time_counts_overlapping_answers_once():
    from gpm_server.strategies import busy_seconds

    assert busy_seconds([(0, 10), (5, 12), (20, 25), (21, 22)]) == 17
    assert busy_seconds([]) == 0
