"""A pool that adds hosts from what the traffic asks for (D66).

The lease stays the only spending authority and becomes the ceiling; what it stops being is the
*demand*. Every host in a round is still chosen by the configured offer rules and re-checked
against every cap on its own — a round is a number of attempts, never a bulk purchase.
"""

import pytest
from fakes.harness import BackgroundLoop  # noqa: F401 - imported for parity with the suite
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.strategies import Load
from gpm_server.supervisor.renting import Fleet

MODEL = "m1"


def make_fleet(database, provider, **rented):
    base = {
        "provider": "fake",
        "workers": 2,
        "model_set_gb": 10.0,
        "allocation": "dynamic",
        "dynamic": {"window_s": 0, "ramp_backoff_s": 0},
        "bidding": {"bid_ceiling": 0.60, "premium": 0.02},
        "scale": {"scale_up_after_s": 0},
    }
    base.update(rented)
    config = PoolConfig.model_validate(
        {
            "pool": {"name": "test", "model_set": [MODEL]},
            "auth": {"app_keys": ["k"]},
            "hosts": [{"id": "local-1", "kind": "local",
                       "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            "limits": {"max_rented_hosts": 20, "max_hourly_burn": 30},
            "rented": base,
        }
    )
    return Fleet(config, config.rented, provider, LeaseStore(database), EventLog(database),
                 SpendLedger(database))


def market(machines=12):
    """A market with room in it: the pool does not bid on a machine it already rents (D59),
    so a ramp needs somewhere to go."""
    return FakeProvider(
        offers=[default_offer(offer_id=f"o-{n}", machine_id=f"m-{n}") for n in range(machines)]
    )


@pytest.fixture
def fleet(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    made = make_fleet(database, market())
    made.run_on_host = lambda *a: (0, "")
    try:
        yield made
    finally:
        database.close()


def busy(fleet, waiting=8):
    """Every ready worker busy, with requests waiting behind them."""
    ready = sum(h.workers for h in fleet.hosts.values() if not h.released and h.state == "ready")
    return Load(busy_workers=ready, ready_workers=ready, waiting=waiting)


def kinds(fleet):
    return [e["kind"] for e in fleet.events.recent(50)]


async def pump(fleet, load, higher_tiers=2, passes=2):
    """These fleets hold for no time at all (`window_s: 0`), so a pass is a round. A pool with
    the shipped window waits two minutes before its first round."""
    for _ in range(passes):
        await fleet.pass_once(ready_workers_higher_tiers=higher_tiers, idle_seconds={}, load=load)


async def test_nothing_is_rented_without_a_lease_however_loud_the_load(fleet):
    await pump(fleet, busy(fleet, waiting=99), higher_tiers=0)
    assert not fleet.hosts and not fleet.provider.instances


async def test_a_quiet_pool_rents_nothing_though_the_lease_would_allow_it(fleet):
    fleet.open_lease(workers=12, max_hours=2, max_spend=5.0, allow_rent=True)

    quiet = Load(busy_workers=0, ready_workers=4, waiting=0)
    await pump(fleet, quiet, higher_tiers=4)

    assert not fleet.hosts, "the lease is the ceiling, not the demand (D66)"


async def test_load_brings_the_first_round_which_is_one_host(fleet):
    fleet.open_lease(workers=12, max_hours=2, max_spend=5.0, allow_rent=True)

    await pump(fleet, busy(fleet))

    assert len(fleet.hosts) == 1
    assert "ramp_round" in kinds(fleet)


async def test_the_next_round_asks_for_more_once_the_last_has_landed(fleet):
    fleet.open_lease(workers=40, max_hours=2, max_spend=50.0, allow_rent=True)
    await pump(fleet, busy(fleet))
    assert len(fleet.hosts) == 1

    for host in fleet.hosts.values():  # the first round lands
        host.state = "ready"
    await pump(fleet, busy(fleet), passes=1)

    assert len(fleet.hosts) == 3, "one, then two"


async def test_a_round_does_not_grow_while_its_hosts_are_still_coming_up(fleet):
    fleet.open_lease(workers=40, max_hours=2, max_spend=50.0, allow_rent=True)
    await pump(fleet, busy(fleet))

    # Still preparing: a host takes minutes, and doubling now buys what it was about to supply.
    await pump(fleet, busy(fleet), passes=3)

    assert len(fleet.hosts) == 1


async def test_the_ramp_starts_from_one_again_when_the_load_clears(fleet):
    fleet.open_lease(workers=40, max_hours=2, max_spend=50.0, allow_rent=True)
    await pump(fleet, busy(fleet))
    for host in fleet.hosts.values():
        host.state = "ready"

    quiet = Load(busy_workers=0, ready_workers=99, waiting=0)
    await pump(fleet, quiet, higher_tiers=99, passes=1)
    assert "ramp_reset" in kinds(fleet)

    await pump(fleet, busy(fleet))
    assert len(fleet.hosts) == 2, "the next ramp starts from one host again, not from two"


async def test_the_lease_remains_the_ceiling(fleet):
    fleet.open_lease(workers=2, max_hours=2, max_spend=50.0, allow_rent=True)

    # Far more load than the lease permits: it is still only asked for what it authorised.
    await pump(fleet, Load(busy_workers=2, ready_workers=2, waiting=500))

    assert not fleet.hosts, "two workers wanted, two already there"


async def test_every_host_in_a_round_is_re_checked_against_the_caps(fleet, tmp_path):
    """A round is a number of attempts, never a bulk purchase."""
    database = Database(tmp_path / "capped.sqlite3")
    try:
        capped = make_fleet(database, market())
        capped.run_on_host = lambda *a: (0, "")
        capped.config.limits.max_rented_hosts = 2
        capped.open_lease(workers=40, max_hours=2, max_spend=50.0, allow_rent=True)

        await pump(capped, busy(capped))
        for host in capped.hosts.values():
            host.state = "ready"
        await pump(capped, busy(capped), passes=1)
        for host in capped.hosts.values():
            host.state = "ready"
        await pump(capped, busy(capped), passes=1)

        assert len(capped.hosts) == 2, "the pool's host limit stopped the round mid-way"
        assert "rent_refused" in kinds(capped)
    finally:
        database.close()


async def test_a_burst_shorter_than_the_window_buys_nothing(tmp_path):
    """A model download takes minutes; a spike that is gone by then is pure cost."""
    database = Database(tmp_path / "burst.sqlite3")
    try:
        slow = make_fleet(database, market(), dynamic={"window_s": 600, "ramp_backoff_s": 0})
        slow.run_on_host = lambda *a: (0, "")
        slow.open_lease(workers=20, max_hours=2, max_spend=20.0, allow_rent=True)

        await pump(slow, busy(slow), passes=4)

        assert not slow.hosts, "the load has not held long enough to be worth a download"
    finally:
        database.close()
