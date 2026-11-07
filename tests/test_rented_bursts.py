"""A workload's rented burst (D165, workloads.md §13): hosts beyond its base, rented only once
the whole base serves and its load has stayed above it for the burst's `after_min`, at most
`max_hosts`, within the burst's own day and month budget — paid from the workload's lease. Burst
hosts never count towards the base, go first when the load falls, and are drained when the
budget is reached or the burst is taken away.

Against the fake provider; nothing here spends money."""

import datetime
import time

import pytest
from gpm_server.config import Budget, PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.strategies import Load
from gpm_server.supervisor.renting import Fleet
from gpm_server.workload_store import Workload, WorkloadStore

BIG = "big"
NAME = "orders"


def make_fleet(database, max_rented_hosts=8):
    config = PoolConfig.model_validate({
        "pool": {"name": "test", "model_set": [BIG], "models_per_host": "declared"},
        "auth": {"app_keys": ["k"]},
        "catalog": {BIG: {"variants": [{"tag": "big:q4", "size_gb": 20}]}},
        "hosts": [],
        "limits": {"max_rented_hosts": max_rented_hosts},
        "workloads": {"min_reliability": 0.0},
        "rented": {
            "provider": "fake", "engine": "ollama", "workers": 4,
            "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 2.0},
            "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0, "scale_down_after_s": 0},
            "teardown": {"idle_minutes": 1},
            "dynamic": {"window_s": 0, "ramp_backoff_s": 0, "target_utilisation": 1.0},
        },
    })
    offers = [default_offer(f"o-{i}", f"m-{i}") for i in range(8)]
    return Fleet(config, config.rented, FakeProvider(offers=offers), LeaseStore(database), EventLog(database),
                 SpendLedger(database))


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


def rented(after_min=0, max_hosts=2, **budget):
    return {"rented": {"after_min": after_min, "max_hosts": max_hosts, "budget": budget or {"per_day": 50.0}}}


def open_workload(fleet, policy=None, hosts_at_start=1, workers_per_host=4, max_spend=100.0, hours=8.0):
    lease = fleet.leases.open(workers=hosts_at_start * workers_per_host, max_hours=hours, max_spend=max_spend,
                              allow_rent=True, workload=NAME)
    now = time.time()
    workload = Workload(
        name=NAME, model=BIG, builds={BIG: "big:q4"}, latency_s=30.0, parallel=hosts_at_start * workers_per_host,
        kind="interruptible", lease_id=lease.lease_id, state="serving", workers_per_host=workers_per_host,
        hosts_at_start=hosts_at_start, plan={}, created_at=now, updated_at=now, ends_at=now + hours * 3600,
        policy=policy or {},
    )
    WorkloadStore(fleet.leases.db).create(workload)
    return workload, lease


def with_policy(workload, policy):
    import dataclasses

    return dataclasses.replace(workload, policy=policy)


async def run(fleet, workload, load=None, passes=1, idle=None, busy=None):
    for _ in range(passes):
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds=idle or {}, busy=busy or {},
                              workloads=[workload], loads={NAME: load} if load is not None else {})


def ready_all(fleet):
    for host in fleet.hosts_of(NAME):
        if host.state != "draining":
            host.state, host.ready_at = "ready", time.time()


def base_of(fleet):
    return [h for h in fleet.hosts_of(NAME) if not h.burst]


def burst_of(fleet):
    return [h for h in fleet.hosts_of(NAME) if h.burst]


#: Four busy of four, and twenty waiting: far above a one-host base.
HEAVY = Load(busy_workers=4, ready_workers=4, waiting=20)


async def test_without_a_rented_burst_a_workload_never_rents_beyond_its_base(db):
    fleet = make_fleet(db)
    workload, _ = open_workload(fleet)
    await run(fleet, workload)
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=4)
    assert len(fleet.hosts_of(NAME)) == 1, "steady alone: the base, then the queue"


async def test_a_lasting_burst_rents_beyond_the_base_up_to_its_hosts(db):
    fleet = make_fleet(db)
    workload, lease = open_workload(fleet, rented(max_hosts=2))
    await run(fleet, workload)
    (base,) = fleet.hosts_of(NAME)
    assert not base.burst, "the floor is the base"
    ready_all(fleet)
    for _ in range(6):
        await run(fleet, workload, HEAVY)
        ready_all(fleet)
    assert len(burst_of(fleet)) == 2, "never more than max_hosts beyond the base"
    assert len(base_of(fleet)) == 1
    rented_events = fleet.events.recent(20, kind="rented")
    assert sum(1 for e in rented_events if e["numbers"].get("burst")) == 2
    assert all(e["numbers"]["workload"] == NAME for e in rented_events)
    assert fleet.published_ref(burst_of(fleet)[0])["burst"] is True, "what it is survives a restart"


async def test_a_burst_waits_its_minutes_measured_from_a_serving_base(db):
    fleet = make_fleet(db)
    workload, _ = open_workload(fleet, rented(after_min=10))
    await run(fleet, workload)
    # The base is still preparing: however long the load, more load is the base's to meet.
    await run(fleet, workload, HEAVY, passes=3)
    assert len(fleet.hosts_of(NAME)) == 1
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=3)
    assert burst_of(fleet) == [], "a burst shorter than after_min rents nothing"
    unit = fleet.unit(NAME)
    assert unit.burst_since is not None
    unit.burst_since -= 601
    await run(fleet, workload, HEAVY)
    assert len(burst_of(fleet)) == 1, "after ten minutes of it, the first burst host"


async def test_a_base_host_lost_beside_burst_hosts_is_rented_again_at_once(db):
    fleet = make_fleet(db)
    workload, _ = open_workload(fleet, rented(max_hosts=1))
    await run(fleet, workload)
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=2)
    assert len(burst_of(fleet)) == 1
    (base,) = base_of(fleet)
    await fleet.destroy(base, "evicted, in a test")
    await run(fleet, workload, HEAVY)
    assert len(base_of(fleet)) == 1, "a burst host never stands in for the base"


async def test_the_burst_budget_reached_drains_its_hosts_and_keeps_the_base(db):
    fleet = make_fleet(db)
    workload, lease = open_workload(fleet, rented(max_hosts=1, per_day=0.3))
    await run(fleet, workload)
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=2)
    (extra,) = burst_of(fleet)
    # Four hours of it at $0.12/h: past $0.30 a day.
    extra.created_at -= 4 * 3600
    await run(fleet, workload, HEAVY)
    assert extra.state == "draining" or extra.released
    assert [h.state for h in base_of(fleet)] == ["ready"]
    ended = fleet.events.recent(10, kind="workload_burst_ended")
    assert ended and "of its $0.30 a day" in ended[0]["summary"]
    assert fleet.burst_spent(lease.lease_id)["today"] >= 0.45
    # And no burst host is rented again today: the budget has no hour of one left.
    await run(fleet, workload, HEAVY, passes=2)
    assert not [h for h in burst_of(fleet) if h.state != "draining" and not h.released]


async def test_a_burst_budget_without_an_hour_of_a_host_rents_none_and_leaves_the_base_alone(db):
    fleet = make_fleet(db)
    workload, _ = open_workload(fleet, rented(max_hosts=2, per_day=0.1))
    await run(fleet, workload)
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=3)
    assert burst_of(fleet) == []
    bound = fleet.events.recent(10, kind="workload_burst_bound")
    assert len(bound) == 1, "said once, and not searched for again until something changes"
    assert "rented burst's budget has left" in bound[0]["summary"]
    assert fleet.unit(NAME).budget_bound is None, "the base's own renting is not held back"


async def test_taking_the_rented_burst_away_drains_its_hosts(db):
    fleet = make_fleet(db)
    workload, _ = open_workload(fleet, rented(max_hosts=1))
    await run(fleet, workload)
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=2)
    (extra,) = burst_of(fleet)
    await run(fleet, with_policy(workload, {}), HEAVY)
    assert extra.state == "draining" or extra.released
    assert "no longer has a rented burst" in fleet.events.recent(5, kind="workload_burst_ended")[0]["summary"]


async def test_when_the_load_falls_burst_hosts_go_first_and_the_base_stays(db):
    fleet = make_fleet(db)
    workload, _ = open_workload(fleet, rented(max_hosts=1))
    await run(fleet, workload)
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=2)
    ready_all(fleet)
    (extra,) = burst_of(fleet)
    # The burst still runs: eight busy of eight. Nothing is surplus.
    await run(fleet, workload, Load(busy_workers=8, ready_workers=8, waiting=0), passes=2)
    assert not extra.released and extra.state == "ready"
    # It is over: one busy. The burst host goes; the base is kept.
    await run(fleet, workload, Load(busy_workers=1, ready_workers=8, waiting=0), passes=2)
    assert extra.released or extra.state in ("draining", "parked")
    assert [h.state for h in base_of(fleet)] == ["ready"]


async def test_a_burst_host_must_fit_the_lease_to_its_end_only_within_the_bursts_budget(db):
    # $1.50 of lease over 8 hours: the base at $0.12/h is $0.96 to the end. A burst host to the
    # end would take it to $1.92 — but its budget lets it cost $0.40 at most (two days touched).
    fleet = make_fleet(db)
    workload, _ = open_workload(fleet, rented(max_hosts=1, per_day=0.2), max_spend=1.5)
    await run(fleet, workload)
    ready_all(fleet)
    await run(fleet, workload, HEAVY, passes=2)
    assert len(burst_of(fleet)) == 1


def test_a_budget_over_hours_counts_every_day_and_month_they_touch():
    late = datetime.datetime(2026, 11, 30, 22, 0, tzinfo=datetime.timezone.utc).timestamp()
    assert Budget(per_day=10).most_over(late, 6) == 20, "22:00 to 04:00 is two days"
    assert Budget(per_day=10, per_month=15).most_over(late, 6) == 20, "and two months: $30 of month, $20 of days"
    assert Budget(per_month=15).most_over(late, 6) == 30
    noon = datetime.datetime(2026, 11, 10, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
    assert Budget(per_day=10).most_over(noon, 3) == 10
