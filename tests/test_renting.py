"""Renting, against the fake provider. Nothing here needs a cloud account.

docs/spec/supervisor.md §3–§6, §9. The rules being checked are the ones that cost money if they
are wrong: nothing rents without a lease, every cap is re-checked after a strategy returns, a
lease stops before its dollar cap, and a release counts only once the provider agrees.
"""

import time
from pathlib import Path

import pytest
from gpm_server.config import OfferPolicy, PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseRefused, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.strategies import reject_reasons
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


async def test_a_lease_is_tightened_freely_but_raised_only_on_purpose(fleet):
    """D49 replaced tighten-only: raising is allowed, but never by accident — the caller has
    to say it meant to, and the console and CLI only do that once the operator retypes it."""
    lease = open_lease(fleet)
    fleet.leases.tighten(lease.lease_id, max_spend=2.00)
    assert fleet.leases.get(lease.lease_id).max_spend == 2.00

    with pytest.raises(LeaseRefused, match="must be confirmed"):
        fleet.leases.tighten(lease.lease_id, max_spend=50.00)

    fleet.leases.tighten(lease.lease_id, max_spend=50.00, loosen=True)
    assert fleet.leases.get(lease.lease_id).max_spend == 50.00


async def test_a_closed_lease_is_not_reopened_by_amending_it(fleet):
    lease = open_lease(fleet)
    fleet.leases.close(lease.lease_id, "done")
    with pytest.raises(LeaseRefused, match="closed"):
        fleet.leases.tighten(lease.lease_id, max_hours=10, loosen=True)


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
    # Everything the lease held is on its way out; idle, so the next pass ends it.
    assert all(h.state == "draining" for h in fleet.hosts.values())
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert fleet.hosts == {}
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

    # It drains first — a lease ending is not a reason to drop requests in flight (D53).
    (host,) = fleet.hosts.values()
    assert host.state == "draining" and "draining" in kinds(fleet)
    assert fleet.leases.get(lease.lease_id).state == "closed"
    assert "lease_expired" in kinds(fleet)

    # Nothing is running on it, so the next pass ends it.
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, busy={host.host_id: 0})
    assert fleet.hosts == {}


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

    host.engine_seen_at = time.time()  # it started — so this is "never got ready", not "stuck"
    host.created_at -= 31 * 60
    host.preparing_since -= 31 * 60  # it has been *preparing* that long, not merely existing
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert host.host_id not in fleet.hosts
    assert any("not ready after" in e["summary"] for e in fleet.events.recent())


async def test_a_long_running_host_is_not_given_up_on_the_moment_it_is_re_verified(fleet):
    """What a supervisor restart did live (D50): adoption marks a host `preparing` so its
    readiness is re-verified, and the deadline was measured from when it was *created* — so a
    host that had been serving for an hour was destroyed a second after it was taken back."""
    fleet.rented.teardown.max_preparing_minutes = 30
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))

    host.created_at -= 3 * 3600  # three hours old, and serving all that time
    host.state = "ready"
    host.mark_preparing()  # exactly what adoption, or a blink of its engine, does
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert host.host_id in fleet.hosts, "a host that has only just started preparing was destroyed"


# --- renting on demand: a host nobody can outbid (D52) ---


def on_demand_offer(**overrides):
    base = dict(offer_id="od-1", machine_id="m-9", min_bid_hourly=0.50, all_in_hourly=0.50,
                on_demand_hourly=0.50, interruptible=False)
    base.update(overrides)
    return default_offer(**base)


async def test_an_on_demand_host_is_rented_at_its_price_with_no_bid(fleet):
    fleet.rented.mode = "on_demand"
    fleet.provider.offers = [on_demand_offer()]
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (host,) = fleet.hosts.values()
    assert host.interruptible is False
    assert host.bid_hourly == 0.50  # the listed price, not floor plus a premium
    # The provider was asked to create it without a price: that is what makes it on-demand.
    (created,) = fleet.provider.instances.values()
    assert created.bid_hourly == 0.50 and "create" in fleet.provider.calls
    rented = next(e for e in fleet.events.recent() if e["kind"] == "rented")
    assert "not outbiddable" in rented["summary"]


async def test_a_fixed_price_above_the_ceiling_is_refused_not_bid_down(fleet):
    """You cannot offer a marketplace less than its asking price and be served."""
    from gpm_server.strategies import price_bid

    bid = price_bid(on_demand_offer(all_in_hourly=2.0), fleet.rented.bidding)
    assert bid.hourly == 0.0
    assert "cannot be lowered" in " ".join(bid.reasons)

    fleet.rented.mode = "on_demand"
    fleet.provider.offers = [on_demand_offer(all_in_hourly=2.0)]  # ceiling is 0.60
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert fleet.hosts == {}


async def test_an_on_demand_host_that_stops_is_not_treated_as_outbid(fleet):
    fleet.rented.mode = "on_demand"
    fleet.provider.offers = [on_demand_offer()]
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()

    fleet.provider.evict(host.instance.instance_id)  # the provider stops it, whatever the reason
    await fleet.handle_evictions()

    assert host.released
    kinds_seen = kinds(fleet)
    assert "host_stopped" in kinds_seen and "eviction" not in kinds_seen


async def test_which_listings_are_searched_follows_the_mode(fleet):
    asked = []

    async def remember(query):
        asked.append((query.interruptible, query.on_demand))
        return []

    fleet.provider.search_offers = remember
    for mode in ("interruptible", "on_demand", "cheaper"):
        fleet.rented.mode = mode
        await fleet._offers()
    assert asked == [(True, False), (False, True), (True, True)]


async def test_with_both_kinds_in_hand_ranking_chooses(fleet):
    """`cheaper` is not "always bid": a cheap fixed price can beat a risky one."""
    fleet.rented.mode = "cheaper"
    fleet.provider.offers = [
        default_offer(offer_id="bid-1", machine_id="m-1", min_bid_hourly=0.40),   # 0.42 with premium
        on_demand_offer(offer_id="od-1", machine_id="m-2", all_in_hourly=0.20),
    ]
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (host,) = fleet.hosts.values()
    assert host.offer.machine_id == "m-2" and host.interruptible is False


async def test_the_pool_does_not_bid_against_its_own_host(fleet):
    """The machine it already rents is still listed — to be outbid. Outbidding yourself buys
    nothing and costs a host."""
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()

    preview = await fleet.market_preview()
    assert preview["passed"] == 0
    assert "already rented" in fleet.avoided_now()[host.offer.machine_id]
    assert await fleet.prepare(max_spend=1.0, max_hours=1.0) is None
    assert len(fleet.provider.instances) == 1

    await fleet.destroy(host, "done")
    assert (await fleet.market_preview())["passed"] == 1  # and it is back once we have let go


# --- choosing the host and the kind of rental yourself (D55) ---


def both_kinds():
    return [
        default_offer(offer_id="bid-1", machine_id="m-1", min_bid_hourly=0.10),
        default_offer(offer_id="bid-2", machine_id="m-2", min_bid_hourly=0.30),
        on_demand_offer(offer_id="od-1", machine_id="m-3", all_in_hourly=0.50),
    ]


async def test_the_market_shows_both_kinds_whatever_the_mode_and_says_which_is_which(fleet):
    fleet.provider.offers = both_kinds()
    assert fleet.rented.mode == "interruptible"

    configured = await fleet.market_preview()
    assert {row["kind"] for row in configured["best"]} == {"interruptible"}

    preview = await fleet.market_preview(kinds="both")
    by_id = {row["offer_id"]: row for row in preview["best"]}
    assert set(by_id) == {"bid-1", "bid-2", "od-1"}
    assert by_id["od-1"]["kind"] == "on_demand" and by_id["od-1"]["would_bid"] == 0.50
    assert by_id["bid-1"]["kind"] == "interruptible"
    assert fleet.provider.instances == {}  # looking rents nothing


async def test_the_chosen_offer_is_the_one_rented_not_the_best_ranked(fleet):
    fleet.provider.offers = both_kinds()
    host = await fleet.prepare(max_spend=1.0, max_hours=1.0, offer_id="bid-2")

    assert host.offer.offer_id == "bid-2" and host.interruptible is True
    assert len(fleet.provider.instances) == 1


async def test_an_on_demand_host_can_be_chosen_while_the_pool_is_set_to_bid(fleet):
    """Mixed renting: the mode is the default for what the pool rents by itself, not a limit
    on what an operator may choose by hand."""
    fleet.provider.offers = both_kinds()
    host = await fleet.prepare(max_spend=1.0, max_hours=1.0, offer_id="od-1")
    assert host.interruptible is False and host.bid_hourly == 0.50

    second = await fleet.prepare(max_spend=1.0, max_hours=1.0, offer_id="bid-1")
    assert second.interruptible is True
    assert sorted(h.interruptible for h in fleet.hosts.values()) == [False, True]


async def test_asking_for_a_kind_rents_the_best_of_that_kind(fleet):
    fleet.provider.offers = both_kinds()
    host = await fleet.prepare(max_spend=1.0, max_hours=1.0, kind="on_demand")
    assert host.offer.offer_id == "od-1"

    with pytest.raises(LeaseRefused, match="kind"):
        await fleet.prepare(max_spend=1.0, max_hours=1.0, kind="spot")


async def test_a_chosen_offer_that_has_gone_is_not_replaced_by_another(fleet):
    """The operator confirmed one machine at one price. Renting a different one "instead" would
    be spending on something nobody agreed to."""
    fleet.provider.offers = both_kinds()
    assert await fleet.prepare(max_spend=1.0, max_hours=1.0, offer_id="no-such-offer") is None

    assert fleet.provider.instances == {} and fleet.hosts == {}
    assert "no-such-offer" in fleet.last_refusal
    assert "chosen_offer_unavailable" in kinds(fleet)
    assert fleet.leases.open_leases() == []  # and the lease it opened does not linger


async def test_a_chosen_offer_that_loses_its_bid_is_not_replaced_either(fleet):
    fleet.provider.offers = both_kinds()
    fleet.provider.lose_bid_on.add("bid-1")
    assert await fleet.prepare(max_spend=1.0, max_hours=1.0, offer_id="bid-1") is None
    assert fleet.provider.instances == {} and fleet.hosts == {}
    assert fleet.leases.open_leases() == []


async def test_choosing_an_offer_does_not_get_it_past_the_pools_ceilings(fleet):
    """Picking by hand picks among what policy allows; it is not a way round policy."""
    fleet.provider.offers = [*both_kinds(),
                             on_demand_offer(offer_id="od-dear", machine_id="m-4", all_in_hourly=2.0)]
    # ceiling is 0.60
    assert await fleet.prepare(max_spend=5.0, max_hours=1.0, offer_id="od-dear") is None
    assert fleet.provider.instances == {}


# --- a host that is going still finishes what it was given (D53) ---


async def test_a_host_with_work_in_flight_is_drained_not_dropped(fleet):
    """A lease ending is not a reason to drop the requests already running on its host."""
    lease = open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()
    host.state = "ready"

    fleet.leases.close(lease.lease_id, "time limit reached")  # as the real path does first
    await fleet.release_lease(lease, "time limit reached")
    assert host.state == "draining" and not host.released

    # Two requests are still running on it: it is left alone, and stays out of routing.
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, busy={host.host_id: 2})
    assert host.host_id in fleet.hosts and host.state == "draining"

    # The last one answers, and only then is it ended.
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, busy={host.host_id: 0})
    assert fleet.hosts == {}
    released = next(e for e in fleet.events.recent() if e["kind"] == "released")
    assert "its work had finished" in released["summary"]


async def test_draining_does_not_wait_for_ever_because_it_is_still_billing(fleet):
    fleet.rented.teardown.drain_timeout_s = 0.0
    lease = open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()
    host.state = "ready"

    fleet.leases.close(lease.lease_id, "dollar cap reached")
    await fleet.release_lease(lease, "dollar cap reached")
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, busy={host.host_id: 1})

    assert fleet.hosts == {}
    released = next(e for e in fleet.events.recent() if e["kind"] == "released")
    assert "still not finished" in released["summary"] and "still billing" in released["summary"]


async def test_a_host_that_is_serving_nothing_yet_is_not_kept_alive_to_drain(fleet):
    """A host that never became ready has no work to protect; draining it would only bill."""
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()
    host.state = "scheduling"

    await fleet.drain(host, "nothing to keep")
    assert host.released and fleet.hosts == {}


def test_the_router_stops_choosing_a_host_that_is_draining():
    """Publishing `draining` is what stops new work reaching it — so the state must be one the
    router understands, and must not be eligible."""
    from gpm_server.models import HostState
    from gpm_server.router.dispatch import Dispatcher

    assert HostState("draining") is HostState.DRAINING
    assert HostState.DRAINING is not HostState.READY
    source = __import__("inspect").getsource(Dispatcher.eligible)
    assert "HostState.READY" in source


# --- watching a rented host while it comes up (D54) ---


async def test_a_host_stuck_before_it_ever_starts_is_given_up_early(fleet):
    """Seen live: a host sat at "scheduling" while billing, until the owner released it by
    hand. The 30-minute preparing window is for downloads; a host that has not even started
    is judged against a much shorter one."""
    fleet.rented.teardown.max_starting_minutes = 10
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))
    machine = host.offer.machine_id
    assert host.engine_seen_at is None

    host.preparing_since -= 11 * 60
    fleet.leases.close(host.lease_id, "test: stop it renting again")
    await fleet.tear_down([], {}, 0)

    assert host.released
    seen = kinds(fleet)
    assert "host_stuck_starting" in seen and "machine_avoided" in seen
    assert machine in fleet.avoided_now()


async def test_a_host_whose_engine_has_answered_is_not_called_stuck(fleet):
    fleet.rented.teardown.max_starting_minutes = 10
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))
    host.engine_seen_at = time.time()  # started; now downloading, which takes as long as it takes
    host.preparing_since -= 11 * 60

    await fleet.tear_down(fleet.leases.open_leases(), {}, 0)
    assert not host.released


async def test_a_machine_that_just_failed_is_not_bid_on_again(fleet):
    """Seen live: the best-ranked offer was the machine that had just failed, four times."""
    fleet.provider.offers = [
        default_offer(offer_id="o-1", machine_id="bad", min_bid_hourly=0.10),   # ranks first
        default_offer(offer_id="o-2", machine_id="good", min_bid_hourly=0.30),
    ]
    fleet.avoid("bad", "could not download m1")
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (host,) = fleet.hosts.values()
    assert host.offer.machine_id == "good"
    preview = await fleet.market_preview(hours=1)
    assert preview["avoided"]["bad"] == "could not download m1"
    assert "already rented" in preview["avoided"]["good"]  # and never against its own host
    assert any("avoid" in reason for reason in preview["rejected_by_reason"])


async def test_an_avoided_machine_comes_back_when_its_time_is_up(fleet):
    fleet.avoid("m-1", "never started")
    assert "m-1" in fleet.avoided_now()
    fleet.avoided["m-1"] = (time.time() - 1, "never started")
    assert fleet.avoided_now() == {}


async def test_avoiding_can_be_switched_off(fleet):
    fleet.rented.teardown.avoid_failed_machine_minutes = 0
    fleet.avoid("m-1", "never started")
    assert fleet.avoided_now() == {}


# --- what the machine itself said, when a host never answered (D78) ---


async def _a_host_stuck_starting(fleet):
    fleet.rented.teardown.max_starting_minutes = 10
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    host = next(iter(fleet.hosts.values()))
    host.preparing_since -= 11 * 60
    fleet.leases.close(host.lease_id, "test: stop it renting again")
    return host


async def test_a_host_given_up_for_never_starting_says_what_its_boot_output_said(fleet):
    """Found live: three hosts were read as refusing the pool's key when their own boot output
    said the provider's proxy had never published them. One call tells the two apart."""
    host = await _a_host_stuck_starting(fleet)
    fleet.provider.boot_output[host.instance.instance_id] = "\n".join(
        ["Warning: Permanently added 'proxy' to the list of known hosts."]
        + ["Error: remote port forwarding failed for listen port 24390"] * 40
    )

    await fleet.tear_down([], {}, 0)

    assert host.released
    stuck = [e for e in fleet.events.recent() if e["kind"] == "host_stuck_starting"][0]
    assert "remote port forwarding failed for listen port 24390" in stuck["summary"]
    assert stuck["summary"].count("remote port forwarding failed") == 1, "said once, not forty times"


async def test_a_provider_with_no_boot_output_changes_nothing(fleet):
    import dataclasses

    fleet.provider.capabilities = dataclasses.replace(
        fleet.provider.capabilities, reports_instance_logs=False
    )
    host = await _a_host_stuck_starting(fleet)

    await fleet.tear_down([], {}, 0)

    assert host.released and "host_stuck_starting" in kinds(fleet)


async def test_boot_output_that_cannot_be_read_never_delays_giving_a_host_up(fleet, monkeypatch):
    host = await _a_host_stuck_starting(fleet)

    async def broken(instance, tail=60):
        raise RuntimeError("the provider's log endpoint is down")

    monkeypatch.setattr(fleet.provider, "instance_logs", broken)
    await fleet.tear_down([], {}, 0)

    assert host.released


# --- a machine whose driver the engine cannot use (D81) ---


def _offer_with_driver(version, **rest):
    return default_offer(driver_version=version, **rest)


def test_a_driver_too_old_for_the_engine_image_is_refused_before_it_is_rented():
    """Found live, in the engine's own log on a billing host:

        WARN "NVIDIA driver too old" device="NVIDIA A100-SXM4-80GB"
             compute=8.0 driver=535 required_driver="550 or newer"
        INFO "inference compute" id=cpu library=cpu

    It loaded a 26B model at 100% CPU, never finished the model set, and was given up half an
    hour later — an A100-80GB rented at $1.06/h that never touched the accelerator.
    """
    policy = OfferPolicy(min_driver_version="550")

    refused = reject_reasons(_offer_with_driver("535.183.01"), policy)
    assert any(r.startswith("driver:") for r in refused), refused
    assert "535.183.01" in refused[0] and "550" in refused[0]

    assert not reject_reasons(_offer_with_driver("595.84"), policy)
    assert not reject_reasons(_offer_with_driver("550"), policy)


def test_a_machine_that_does_not_say_its_driver_is_not_assumed_to_pass():
    """The filter exists because the cost of being wrong is a whole rental."""
    assert reject_reasons(_offer_with_driver(None), OfferPolicy(min_driver_version="550"))
    assert not reject_reasons(_offer_with_driver(None), OfferPolicy())  # nothing asked, nothing refused


def test_driver_versions_compare_by_number_and_not_as_text():
    """"595.84" is above "550"; as text it is below it."""
    from gpm_server.strategies import driver_below

    assert driver_below("535", "550") is True
    assert driver_below("595.84", "550") is False
    assert driver_below("550", "550") is False
    assert driver_below("9.1", "9") is False
    assert driver_below(None, "550") is None
    assert driver_below("not a version", "550") is None


def _all_offers_refused(fleet):
    fleet.rented.offer_policy = OfferPolicy(min_gpu_memory_gb=10_000)  # nothing can pass


async def test_a_market_that_refuses_everything_is_recorded_once_not_once_a_pass(fleet):
    """Live, a pool whose filters rejected every offer wrote the same line every fifteen
    seconds. A decision log that repeats itself is one an operator stops reading — and the
    reading of it is the whole point of recording refusals."""
    _all_offers_refused(fleet)
    open_lease(fleet, workers=2)

    for _ in range(4):
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    said = [e for e in fleet.events.recent(50) if e["kind"] == "no_offer"]
    assert len(said) == 1, f"{len(said)} identical refusals written"
    assert said[0]["numbers"]["seen"] >= 1


async def test_a_market_that_changes_its_mind_is_recorded_again(fleet):
    """Said once per market, not once ever: a different set of reasons is news."""
    _all_offers_refused(fleet)
    open_lease(fleet, workers=2)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    fleet.rented.offer_policy = OfferPolicy(min_disk_gb=10_000)  # refused, for another reason
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    said = [e for e in fleet.events.recent(50) if e["kind"] == "no_offer"]
    assert len(said) == 2, "a market refusing for a new reason was not recorded"


def test_a_provider_back_off_note_does_not_nest():
    """Live: 'rate limited — not asking again for 60s — not asking again for 44s'."""
    source = (Path(__file__).resolve().parent.parent / "server/src/gpm_server/supervisor/renting.py").read_text()
    waiting = source[source.index("if now < self._offer_retry_at:"):]
    waiting = waiting[: waiting.index("return []")]
    assert "self._offer_refusal" in waiting
    assert "self.last_offer_error or" not in waiting, "the note is built from itself again"
