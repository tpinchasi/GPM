"""The fleet, per workload (D115, workloads.md §5–§8): each workload rents its own hosts, starts on
the hosts its plan said, keeps them for its lease, and lets them go when the lease is gone; the
shared workload is not covered by a workload's hosts, and the pool's caps bound them all.

Against the fake provider; nothing here spends money."""

import json
import time

import pytest
from gpm_server.config import PoolConfig
from gpm_server.db import Database, RequestLog, RequestRecord
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.strategies import Load
from gpm_server.supervisor.renting import Fleet
from gpm_server.workload_store import Workload, WorkloadStore

MODEL, BIG = "m1", "big"


def make_fleet(database, provider, max_rented_hosts=6, **rented_overrides):
    rented = {
        "provider": "fake", "workers": 4,
        "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 2.0},
        "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0},
        "teardown": {"idle_minutes": 1},
    }
    rented.update(rented_overrides)
    config = PoolConfig.model_validate({
        "pool": {"name": "test", "model_set": [MODEL, BIG], "models_per_host": "declared"},
        "auth": {"app_keys": ["k"]},
        "catalog": {BIG: {"variants": [{"tag": "big:q4", "size_gb": 20}]}},
        "hosts": [{"id": "laptop", "kind": "local", "models": [MODEL],
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "limits": {"max_rented_hosts": max_rented_hosts},
        "workloads": {"min_reliability": 0.0},
        "rented": rented,
    })
    return Fleet(config, config.rented, provider, LeaseStore(database), EventLog(database), SpendLedger(database))


def offers(n=6, **overrides):
    return [default_offer(f"o-{i}", f"m-{i}", **overrides) for i in range(n)]


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


def open_workload(fleet, name="research", hosts_at_start=2, workers_per_host=4, kind="interruptible",
                  latency_s=30.0, parallel=8, hours=4.0, max_spend=20.0):
    lease = fleet.leases.open(workers=parallel, max_hours=hours, max_spend=max_spend, allow_rent=True,
                              workload=name)
    now = time.time()
    workload = Workload(
        name=name, model=BIG, builds={BIG: "big:q4"}, latency_s=latency_s, parallel=parallel, kind=kind,
        lease_id=lease.lease_id, state="preparing", workers_per_host=workers_per_host,
        hosts_at_start=hosts_at_start, plan={}, created_at=now, updated_at=now, ends_at=now + hours * 3600,
    )
    WorkloadStore(fleet.leases.db).create(workload)
    return workload, lease


async def run(fleet, workloads, passes=1, idle=None, loads=None, ready_higher=0):
    for _ in range(passes):
        await fleet.pass_once(ready_workers_higher_tiers=ready_higher, idle_seconds=idle or {},
                              workloads=workloads, loads=loads or {})


def mine(fleet, name):
    return fleet.hosts_of(name)


async def test_a_workload_starts_on_the_hosts_its_plan_said(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, lease = open_workload(fleet, hosts_at_start=2)
    await run(fleet, [workload])
    hosts = mine(fleet, "research")
    assert len(hosts) == 2
    assert all(h.lease_id == lease.lease_id and h.models == (BIG,) and h.builds == {BIG: "big:q4"} for h in hosts)
    assert all(h.instance.label.startswith("gpm/test/research/") for h in hosts), "the name is in the label"
    assert fleet.hosts_of(None) == [], "nothing was rented for the shared workload"
    # Once they are rented, the next pass rents nothing more: the floor is met.
    await run(fleet, [workload])
    assert len(mine(fleet, "research")) == 2


async def test_its_hosts_are_kept_for_its_lease_even_idle(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, _ = open_workload(fleet, hosts_at_start=2)
    await run(fleet, [workload])
    for host in mine(fleet, "research"):
        host.state, host.ready_at = "ready", time.time() - 3600
    await run(fleet, [workload], passes=3, idle={h.host_id: 3600 for h in mine(fleet, "research")})
    assert len(mine(fleet, "research")) == 2 and all(not h.released for h in mine(fleet, "research"))


async def test_a_workloads_hosts_never_cover_the_shared_demand(db):
    """The shared lease wants workers; the workload's ready hosts are not its."""
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    for host in mine(fleet, "research"):
        host.state = "ready"
    shared = fleet.leases.open(workers=4, max_hours=2, max_spend=5.0, allow_rent=True)
    await run(fleet, [workload])
    assert [h.lease_id for h in fleet.hosts_of(None)] == [shared.lease_id]


async def test_when_its_lease_closes_its_hosts_go(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, lease = open_workload(fleet, hosts_at_start=2)
    await run(fleet, [workload])
    hosts = list(mine(fleet, "research"))
    fleet.leases.close(lease.lease_id, "workload ended")
    await run(fleet, [])
    assert all(h.released for h in hosts), "not serving anything, so nothing to drain: destroyed at once"
    assert all(h.instance.instance_id not in fleet.provider.instances for h in hosts)


async def test_a_ready_host_with_work_in_flight_drains_when_its_lease_closes(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, lease = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    host.state = "ready"
    fleet.leases.close(lease.lease_id, "workload ended")
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, busy={host.host_id: 2}, workloads=[])
    assert host.state == "draining" and not host.released
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, busy={host.host_id: 0}, workloads=[])
    assert host.released


async def test_spend_is_recorded_for_a_host_whose_lease_has_closed(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, lease = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    host.state = "ready"
    host.created_at -= 3600
    fleet.leases.close(lease.lease_id, "workload ended")
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, busy={host.host_id: 1}, workloads=[])
    spent, _ = fleet.lease_spend(fleet.leases.get(lease.lease_id))
    assert spent > 0, "the draining hour is on the workload's lease"


async def test_the_pools_host_limit_bounds_every_workload_together(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()), max_rented_hosts=3)
    first, _ = open_workload(fleet, name="first", hosts_at_start=2)
    second, _ = open_workload(fleet, name="second", hosts_at_start=2)
    await run(fleet, [first, second])
    assert len(mine(fleet, "first")) == 2 and len(mine(fleet, "second")) == 1
    assert any(e["kind"] == "rent_refused" for e in fleet.events.recent())


async def test_the_room_left_is_taken_in_the_order_the_leases_were_opened(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()), max_rented_hosts=2)
    later, _ = open_workload(fleet, name="later", hosts_at_start=2)
    time.sleep(0.01)
    earlier_lease_opened_second, _ = open_workload(fleet, name="also", hosts_at_start=2)
    await run(fleet, [earlier_lease_opened_second, later])
    assert len(mine(fleet, "later")) == 2 and mine(fleet, "also") == []


async def test_a_machine_one_workload_rents_is_not_bid_on_for_another(db):
    fleet = make_fleet(db, FakeProvider(offers=offers(n=2)))
    first, _ = open_workload(fleet, name="first", hosts_at_start=1)
    second, _ = open_workload(fleet, name="second", hosts_at_start=1)
    await run(fleet, [first, second])
    machines = [h.offer.machine_id for h in fleet.hosts.values()]
    assert sorted(machines) == ["m-0", "m-1"]


async def test_what_a_workload_host_is_survives_a_restart(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    assert fleet.published_ref(host)["workload"] == "research"


async def test_scale_up_above_the_floor_comes_from_load(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()),
                       dynamic={"window_s": 0, "ramp_backoff_s": 0, "target_utilisation": 1.0})
    workload, _ = open_workload(fleet, hosts_at_start=1, parallel=12)
    await run(fleet, [workload])
    for host in mine(fleet, "research"):
        host.state = "ready"
    busy = Load(busy_workers=4, ready_workers=4, waiting=8)
    await run(fleet, [workload], loads={"research": busy})
    await run(fleet, [workload], loads={"research": busy})
    assert len(mine(fleet, "research")) >= 2, "load above the floor adds hosts"
    assert len(mine(fleet, "research")) <= 3, "never past the workload's own parallelism"


# --- the rental-kind rule (D115, workloads.md §5) ---


def market_both_kinds(download_mbps=100.0):
    """A bid and an on-demand machine; at 100 Mbps a 20 GB replacement is 30 minutes from ready,
    which is what makes an eviction expensive."""
    bid = default_offer("bid-1", "m-bid", min_bid_hourly=0.9, interruptible=True, on_demand_hourly=1.5,
                        download_mbps=download_mbps)
    od = default_offer("od-1", "m-od", min_bid_hourly=1.5, all_in_hourly=1.5, interruptible=False,
                       on_demand_hourly=1.5, download_mbps=download_mbps)
    return [bid, od]


async def test_a_lone_first_host_goes_on_demand_when_evictions_are_likely(db):
    fleet = make_fleet(db, FakeProvider(offers=market_both_kinds()), mode="cheaper")
    fleet.config.workloads.eviction_prior_per_hour = 1.0
    workload, _ = open_workload(fleet, hosts_at_start=1, kind="roi")
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    assert not host.interruptible and host.offer.machine_id == "m-od"
    (event,) = [e for e in fleet.events.recent() if e["kind"] == "rental_kind"]
    assert event["summary"].startswith("rental kind: on demand on m-od")


async def test_a_bid_wins_when_evictions_are_rare(db):
    fleet = make_fleet(db, FakeProvider(offers=market_both_kinds()), mode="cheaper")
    fleet.config.workloads.eviction_prior_per_hour = 0.0
    workload, _ = open_workload(fleet, hosts_at_start=1, kind="roi")
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    assert host.interruptible


async def test_a_fixed_kind_is_what_it_says(db):
    fleet = make_fleet(db, FakeProvider(offers=market_both_kinds()), mode="cheaper")
    workload, _ = open_workload(fleet, hosts_at_start=1, kind="on_demand")
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    assert not host.interruptible
    assert not any(e["kind"] == "rental_kind" for e in fleet.events.recent())


# --- workers per host at the latency target ---


def served(db, host_id, concurrency, latency_s, n=25, tag="big:q4"):
    log = RequestLog(db)
    for i in range(n):
        log._insert(RequestRecord(request_id=f"{host_id}-{concurrency}-{i}", outcome="ok", host_id=host_id,
                                  model_served=tag, latency_ms=latency_s * 1000, concurrency=concurrency))


async def test_a_host_is_held_to_what_its_class_served_within_the_target(db):
    fleet = make_fleet(db, FakeProvider(offers=offers(hardware="FakeGPU 48GB")))
    # A host of the same class served this build before: fine at 2 at once, too slow at 4.
    fleet.events.record("rented", "bid on m-old (FakeGPU 48GB)", host_id="rented-old",
                        numbers={"machine": "m-old", "hardware": "FakeGPU 48GB"})
    served(db, "rented-old", 2, 10.0)
    served(db, "rented-old", 4, 40.0)
    workload, _ = open_workload(fleet, hosts_at_start=1, latency_s=20.0)
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    assert host.workers == 2 and host.launch_workers == 2
    rented = next(e for e in fleet.events.recent() if e["kind"] == "rented" and e["host_id"] == host.host_id)
    assert "held to 2 for the 20s target" in rented["summary"]


async def test_nothing_measured_is_said_and_the_cards_number_is_used(db):
    fleet = make_fleet(db, FakeProvider(offers=offers()))
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = mine(fleet, "research")
    assert host.workers == 4
    rented = next(e for e in fleet.events.recent() if e["kind"] == "rented" and e["host_id"] == host.host_id)
    assert "unmeasured" in rented["summary"]


def test_the_eviction_rate_is_per_rented_hour():
    from gpm_server import history

    start = 1_000_000.0
    events = [
        {"kind": "rented", "host_id": "h1", "ts": start, "numbers": {"machine": "m", "hardware": "X"}},
        {"kind": "eviction", "host_id": "h1", "ts": start + 1800, "numbers": {}},
        {"kind": "released", "host_id": "h1", "ts": start + 7200, "numbers": {}},
    ]
    (record,) = history.build(events, []).values()
    assert record.hours_rented == pytest.approx(2.0) and record.evictions_per_hour == pytest.approx(0.5)
    assert json.dumps(record.as_dict())  # the view stays serialisable


async def test_the_shared_workload_does_not_take_room_a_starting_workload_reserved(db):
    """Created with room for its start; the shared lease acquires first each pass, and must not
    take that room while the workload's hosts are still being rented (workloads.md §8)."""
    fleet = make_fleet(db, FakeProvider(offers=offers()), max_rented_hosts=2)
    workload, _ = open_workload(fleet, hosts_at_start=2)
    fleet.workloads = {"research": workload}
    shared = fleet.leases.open(workers=8, max_hours=4, max_spend=5.0, allow_rent=True)
    assert fleet.reserved_hosts(besides=None) == 2
    assert "reserved for workloads still starting" in fleet._refuse_for_caps(None, shared)
    assert fleet._refuse_for_caps(None, fleet.leases.get(workload.lease_id)) is None, "its own room"


def test_a_workloads_name_never_looks_like_a_host_id():
    from gpm_server.workloads import valid_name

    assert not valid_name("rented-a1b2c3") and valid_name("rentals")
