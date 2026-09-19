"""Renting, against the fake provider. Nothing here needs a cloud account.

docs/spec/supervisor.md §3–§6, §9. The rules being checked are the ones that cost money if they
are wrong: nothing rents without a lease, every cap is re-checked after a strategy returns, a
lease stops before its dollar cap, and a release counts only once the provider agrees.
"""

import time

import pytest
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseRefused, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor.renting import Fleet

MODEL = "m1"


def make_config(**rented_overrides):
    rented = {
        "provider": "fake",
        "workers": 2,
        "model_set_gb": 10.0,
        "bidding": {"bid_ceiling": 0.60, "premium": 0.02},
        "offer_policy": {"min_gpu_memory_gb": 24},
        "scale": {"scale_up_after_s": 0},
    }
    rented.update(rented_overrides)
    return PoolConfig.model_validate(
        {
            "pool": {"name": "test", "model_set": [MODEL]},
            "auth": {"app_keys": ["k"]},
            "hosts": [
                {
                    "id": "local-1",
                    "kind": "local",
                    "workers": 1,
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"},
                }
            ],
            "rented": rented,
        }
    )


@pytest.fixture
def fleet(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    config = make_config()
    provider = FakeProvider()
    made = Fleet(
        config,
        config.rented,
        provider,
        LeaseStore(database),
        EventLog(database),
        SpendLedger(database),
    )
    try:
        yield made
    finally:
        database.close()


def open_lease(fleet, **overrides):
    args = dict(workers=6, max_hours=4, max_spend=5.00, allow_rent=True)
    args.update(overrides)
    return fleet.leases.open(**args)


def kinds(fleet):
    return [event["kind"] for event in fleet.events.recent()]


# --- a lease is the only thing that can spend ---


async def test_nothing_is_rented_with_no_lease_open(fleet):
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert fleet.hosts == {}
    assert "create" not in fleet.provider.calls


async def test_a_lease_that_can_rent_must_carry_a_dollar_cap(fleet):
    with pytest.raises(LeaseRefused, match="dollar cap"):
        fleet.leases.open(workers=4, max_hours=2, max_spend=None, allow_rent=True)


async def test_a_lease_that_may_not_rent_needs_no_cap(fleet):
    lease = fleet.leases.open(workers=4, max_hours=2, max_spend=None, allow_rent=False)
    assert lease.is_open
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert fleet.hosts == {}


async def test_a_lease_may_only_be_tightened(fleet):
    lease = open_lease(fleet)
    fleet.leases.tighten(lease.lease_id, max_spend=2.00)
    assert fleet.leases.get(lease.lease_id).max_spend == 2.00
    with pytest.raises(LeaseRefused, match="tightened"):
        fleet.leases.tighten(lease.lease_id, max_spend=50.00)


async def test_a_lease_may_not_loosen_the_pools_bid_ceiling(fleet):
    with pytest.raises(LeaseRefused, match="tighten"):
        fleet.leases.open(
            workers=4, max_hours=2, max_spend=1.0, allow_rent=True,
            bid_ceiling=5.0, pool_bid_ceiling=0.60,
        )


# --- renting ---


async def test_persistent_overflow_rents_one_host_and_says_why(fleet):
    lease = open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=1, idle_seconds={})

    assert len(fleet.hosts) == 1
    host = next(iter(fleet.hosts.values()))
    assert host.bid_hourly == pytest.approx(0.12)  # floor 0.10 + premium 0.02
    assert host.lease_id == lease.lease_id

    rented = [e for e in fleet.events.recent() if e["kind"] == "rented"][0]
    assert rented["numbers"]["floor"] == 0.10
    assert rented["numbers"]["bid"] == pytest.approx(0.12)
    assert rented["numbers"]["reasons"]


async def test_only_one_host_is_rented_at_a_time(fleet):
    open_lease(fleet, workers=20)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert len(fleet.hosts) == 1


async def test_the_pools_host_limit_is_re_checked_after_the_strategy(fleet):
    fleet.config.limits.max_rented_hosts = 1
    open_lease(fleet, workers=20)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    # Pretend the one host finished starting, so "one at a time" no longer blocks.
    next(iter(fleet.hosts.values())).state = "ready"
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert len(fleet.hosts) == 1
    refusals = [e for e in fleet.events.recent() if e["kind"] == "rent_refused"]
    assert refusals and "limit of 1" in refusals[0]["summary"]


async def test_the_hourly_burn_cap_refuses_a_bid_that_would_cross_it(fleet):
    fleet.config.limits.max_hourly_burn = 0.05  # below the cheapest bid available
    open_lease(fleet, workers=20)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert fleet.hosts == {}
    assert any("burn" in e["summary"] for e in fleet.events.recent())


async def test_a_lost_bid_leaves_nothing_behind_and_tries_the_next_offer(fleet):
    fleet.provider.offers = [
        default_offer(offer_id="o-1", machine_id="m-1", min_bid_hourly=0.10),
        default_offer(offer_id="o-2", machine_id="m-2", min_bid_hourly=0.20),
    ]
    fleet.provider.lose_bid_on = {"o-1"}
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert len(fleet.hosts) == 1
    assert next(iter(fleet.hosts.values())).offer.machine_id == "m-2"
    assert "bid_failed" in kinds(fleet)
    assert len(fleet.provider.instances) == 1  # the losing bid left nothing


async def test_an_empty_market_stays_paused_rather_than_relaxing_a_filter(fleet):
    fleet.provider.offers = [default_offer(gpu_memory_gb=8)]  # below the policy's minimum
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert fleet.hosts == {}
    no_offer = [e for e in fleet.events.recent() if e["kind"] == "no_offer"][0]
    assert no_offer["numbers"]["rejected"]  # every rejection, with its reason


async def test_a_bid_over_the_ceiling_is_clamped_by_the_supervisor_not_trusted(fleet):
    """A faulty or hostile strategy cannot spend past the limits."""
    lease = open_lease(fleet)
    assert fleet._cap_bid(99.0, lease) == 0.60
    assert any(e["kind"] == "bid_clamped" for e in fleet.events.recent())


# --- money ---


async def test_a_lease_stops_before_its_cap_when_the_provider_reports_more(fleet):
    lease = open_lease(fleet, max_spend=1.00)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))

    # Half an hour in, the pool's own estimate is $0.06 — comfortably inside the cap...
    host.created_at = time.time() - 1800
    fleet.provider.instances[host.instance.instance_id].created_at = host.created_at
    # ...but the provider says it has cost twenty times that.
    fleet.provider.set_reported_charges(host.instance.instance_id, multiplier=20)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert fleet.leases.get(lease.lease_id).state == "closed"
    assert fleet.hosts == {}  # everything the lease held was released
    capped = [e for e in fleet.events.recent() if e["kind"] == "lease_capped"][0]
    # Enforced at the cap less the safety margin, so it stops *before* the limit.
    assert capped["numbers"]["enforced_at"] == pytest.approx(0.90)
    assert capped["numbers"]["spent"] > capped["numbers"]["estimate"]


async def test_a_provider_that_cannot_report_charges_gets_a_wider_margin(fleet):
    fleet.provider.capabilities = type(fleet.provider.capabilities)(
        interruptible=True, reports_charges=False
    )
    assert fleet.margin() == 0.25


async def test_the_narrow_margin_is_earned_only_once_a_charge_has_actually_been_reported(fleet):
    """Live finding: a provider may declare charge reporting and never return a figure."""
    assert fleet.provider.capabilities.reports_charges
    assert fleet.margin() == 0.25  # declared, but nothing seen yet

    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})  # records a report
    assert fleet.charges_ever_reported
    assert fleet.margin() == 0.10


async def test_a_declaring_provider_that_never_reports_keeps_the_wide_margin(fleet):
    async def nothing(instance):
        return None

    fleet.provider.reported_charges = nothing
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert fleet.margin() == 0.25


async def test_an_expired_lease_releases_what_it_held(fleet):
    lease = open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert len(fleet.hosts) == 1

    fleet.leases.db.execute(
        "UPDATE leases SET opened_at = ? WHERE lease_id = ?",
        (time.time() - 5 * 3600, lease.lease_id),
    )
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert fleet.hosts == {}
    assert fleet.leases.get(lease.lease_id).state == "closed"
    assert "lease_expired" in kinds(fleet)


# --- release what should not exist ---


async def test_a_stray_instance_carrying_the_pools_label_is_swept(fleet):
    fleet.provider.strand(label="gpm/test/rented-nobody-knows")
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert fleet.provider.instances == {}
    assert "orphan_swept" in kinds(fleet)


async def test_rented_instances_carry_the_frameworks_own_prefix(fleet):
    """`gpm/<pool>/<host>` — so a sweep can tell this pool's instances from another tool's on
    the same account, which the live run showed is a real situation."""
    assert fleet.label_prefix == "gpm/test/"
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    label = next(iter(fleet.provider.instances.values())).label
    assert label.startswith("gpm/test/rented-")


async def test_an_instance_under_another_pools_label_is_left_alone(fleet):
    fleet.provider.strand(label="someone-elses-tool/host")
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert len(fleet.provider.instances) == 1


async def test_a_destroy_that_fails_is_retried_and_only_counts_once_verified(fleet):
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))

    fleet.provider.destroy_failures = 1
    await fleet.destroy(host, "test")
    assert host.host_id in fleet.hosts  # not released: the provider still lists it
    assert "destroy_failed" in kinds(fleet)

    await fleet.destroy(host, "test")
    assert fleet.hosts == {}
    assert "released" in kinds(fleet)


async def test_an_idle_rented_host_releases_itself(fleet):
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host_id = next(iter(fleet.hosts))
    fleet.hosts[host_id].state = "ready"  # only a host that is serving can be idle
    fleet.hosts[host_id].ready_at = time.time() - 12 * 60

    # Nothing has been routed to it for longer than the idle limit.
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={host_id: 11 * 60})

    assert fleet.hosts == {} or fleet.hosts[host_id].state == "parked"
    assert {"parked", "released"} & set(kinds(fleet))


# --- recover what is broken ---


async def test_an_eviction_inside_a_lease_is_recovered_unattended(fleet):
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))

    fleet.provider.evict(host.instance.instance_id)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert "eviction" in kinds(fleet)
    # Either back on the same machine, or replaced — but never silently left stopped.
    assert not fleet.hosts or next(iter(fleet.hosts.values())).state != "stopped"


async def test_an_eviction_outside_a_lease_rents_nothing_back(fleet):
    lease = open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))

    fleet.leases.close(lease.lease_id)
    fleet.provider.evict(host.instance.instance_id)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert fleet.hosts == {}
    assert fleet.provider.instances == {}
    assert "create" not in fleet.provider.calls[-3:]


# --- capacity must not flap (found on the first live run) ---


async def test_a_host_still_booting_is_not_reaped_as_surplus(fleet):
    """Live run, 2026-09-19: 2 workers wanted, one 2-worker host in flight — overflow hit
    exactly zero and the host was destroyed the pass after it was rented, five times over."""
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))
    assert host.state == "preparing"

    for _ in range(5):
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert host.host_id in fleet.hosts and not host.released
    assert len(fleet.provider.instances) == 1  # rented once, not five times


async def test_surplus_capacity_is_released_only_after_the_scale_down_window(fleet):
    fleet.rented.scale.scale_down_after_s = 600
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))
    host.state = "ready"

    # Demand is now fully covered by higher tiers: the rented host is surplus — but only once
    # that has stayed true for the window, so capacity does not flap.
    await fleet.pass_once(ready_workers_higher_tiers=2, idle_seconds={})
    assert host.host_id in fleet.hosts

    fleet.overflow_gone_since -= 601
    await fleet.pass_once(ready_workers_higher_tiers=2, idle_seconds={})
    # Surplus inside an open lease is parked (disk kept), not destroyed; either way it stops
    # serving and the pool does not rent it straight back.
    assert host.host_id not in fleet.hosts or fleet.hosts[host.host_id].state == "parked"
    assert any("overflow" in e["summary"] for e in fleet.events.recent() if e["kind"] in ("released", "parked"))
    assert len(fleet.provider.instances) == 1


async def test_a_host_that_never_becomes_ready_is_given_up_on(fleet):
    fleet.rented.teardown.max_preparing_minutes = 30
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))

    host.created_at -= 31 * 60
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert host.host_id not in fleet.hosts
    assert any("not ready after" in e["summary"] for e in fleet.events.recent())
