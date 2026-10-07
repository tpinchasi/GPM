"""A pool renting through several provider connections at once (D129, D131, D132,
docs/spec/providers.md, step 2): one search across all of them, every operation on a host at its
own provider, spot prices that the provider sets and moves, and the choice between a bid and a
spot price made on the same terms. Three fake providers; nothing here spends money."""

import time

import pytest
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.providers.base import ProviderCapabilities, ProviderRateLimited
from gpm_server.supervisor.renting import Fleet
from gpm_server.workload_store import Workload, WorkloadStore

BIG = "big"


def bid_offer(offer_id, machine, floor, **over):
    # On demand well above, so the bid is never held down by the on-demand crossover.
    return default_offer(offer_id, machine, min_bid_hourly=floor, on_demand_hourly=3.0, **over)


def spot_offer(offer_id, machine, price, **over):
    return default_offer(offer_id, machine, min_bid_hourly=price, bidding=False, on_demand_hourly=3.0, **over)


def fixed_offer(offer_id, machine, price, **over):
    return default_offer(offer_id, machine, min_bid_hourly=price, all_in_hourly=price, interruptible=False, **over)


SPOT_CAPS = ProviderCapabilities(interruptible=True, self_terminate=True, reports_charges=True,
                                 interruption_notice=True)
FIXED_CAPS = ProviderCapabilities(interruptible=False, self_terminate=False, reports_charges=True)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


def make_fleet(db, providers, enabled=None, **rented):
    enabled = enabled if enabled is not None else list(providers)
    config = PoolConfig.model_validate({
        "pool": {"name": "test", "model_set": ["m1", BIG], "models_per_host": "declared"},
        "auth": {"app_keys": ["k"]},
        "catalog": {BIG: {"variants": [{"tag": "big:q4", "size_gb": 20}]}},
        "hosts": [{"id": "laptop", "kind": "local", "models": ["m1"],
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "limits": {"max_rented_hosts": 6},
        "workloads": {"min_reliability": 0.0, "eviction_prior_per_hour": 0.0},
        # Each its own provider: a pool has one connection per provider (D133).
        "rented": {"providers": {name: {"type": f"fake-{name}", "enabled": name in enabled} for name in providers},
                   "workers": 4, "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 2.0},
                   "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0}, **rented},
    })
    return Fleet(config, config.rented, providers, LeaseStore(db), EventLog(db), SpendLedger(db))


def open_workload(fleet, kind="roi", hours=2.0, max_spend=20.0, hosts_at_start=1):
    lease = fleet.leases.open(workers=4, max_hours=hours, max_spend=max_spend, allow_rent=True, workload="w")
    now = time.time()
    workload = Workload(name="w", model=BIG, builds={BIG: "big:q4"}, latency_s=30, parallel=4, kind=kind,
                        lease_id=lease.lease_id, state="preparing", workers_per_host=4,
                        hosts_at_start=hosts_at_start, plan={}, created_at=now, updated_at=now,
                        ends_at=now + hours * 3600)
    WorkloadStore(fleet.leases.db).create(workload)
    return workload, lease


async def run(fleet, workload, passes=1):
    for _ in range(passes):
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={},
                              workloads=[WorkloadStore(fleet.leases.db).get("w") or workload], loads={})


def market(**extra):
    return {
        "bids": FakeProvider(offers=[bid_offer("b-1", "m-1", 0.90)]),
        "spot": FakeProvider(offers=[spot_offer("s-1", "m-1", 0.60)], capabilities=SPOT_CAPS),
        "fixed": FakeProvider(offers=[fixed_offer("f-1", "m-1", 1.50)], capabilities=FIXED_CAPS),
        **extra,
    }


# --- one search across every enabled connection ---


async def test_a_search_asks_every_enabled_connection_and_names_each_offer(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["bids", "spot"], mode="cheaper")
    offers = await fleet._offers()
    assert sorted((o.connection, o.offer_id) for o in offers) == [("bids", "b-1"), ("spot", "s-1")]
    assert "search_offers" not in providers["fixed"].calls, "a disabled connection is not searched"


async def test_one_connection_failing_leaves_the_others_offers_and_says_why(db):
    providers = market()
    providers["bids"].unavailable = True
    fleet = make_fleet(db, providers, mode="cheaper")
    offers = await fleet._offers()
    assert {o.connection for o in offers} == {"spot", "fixed"}
    assert fleet.searching["bids"].error and fleet.last_offer_error is None
    for p in providers.values():
        p.unavailable = True
    assert await fleet._offers() == []
    assert "bids:" in fleet.last_offer_error and "spot:" in fleet.last_offer_error, "nobody answered: each says why"


async def test_a_rate_limit_backs_off_that_connection_only(db):
    providers = market()

    async def refused(query):
        raise ProviderRateLimited("too many requests")

    providers["bids"].search_offers = refused
    fleet = make_fleet(db, providers, mode="cheaper")
    await fleet._offers()
    assert fleet.searching["bids"].retry_at > time.monotonic() and fleet.searching["spot"].retry_at == 0
    assert {o.connection for o in await fleet._offers()} == {"spot", "fixed"}


# --- a host lives at its own provider ---


async def test_a_host_is_rented_watched_and_ended_at_its_own_provider(db):
    providers = market()
    fleet = make_fleet(db, providers, mode="cheaper")
    workload, lease = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    assert host.connection_name == "spot", "the cheapest interruptible offer was the spot price"
    assert list(providers["spot"].instances) == [host.instance.instance_id]
    assert not providers["bids"].instances
    await fleet.destroy(host, "test")
    assert not providers["spot"].instances


async def test_the_same_instance_id_at_two_providers_is_two_instances(db):
    providers = market()
    fleet = make_fleet(db, providers, mode="cheaper")
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    stray = providers["bids"].strand(fleet.label_prefix + "stray")
    assert stray == host.instance.instance_id, "both providers number their instances from one"
    await fleet.sweep_orphans()
    assert stray not in providers["bids"].instances, "the stray at the other provider is swept"
    assert host.instance.instance_id in providers["spot"].instances, "the pool's own host is not"


async def test_a_disabled_connections_hosts_are_still_swept(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["bids"])
    providers["fixed"].strand(fleet.label_prefix + "stray")
    await fleet.sweep_orphans()
    assert not providers["fixed"].instances


# --- bid and spot, on the same terms (D131, D132) ---


@pytest.mark.parametrize("spot_price, bid_floor, wins", [(0.60, 0.90, "spot"), (1.20, 0.50, "bids")])
async def test_a_bid_and_a_spot_price_compete_on_what_they_cost(db, spot_price, bid_floor, wins):
    providers = market(
        bids=FakeProvider(offers=[bid_offer("b-1", "m-1", bid_floor)]),
        spot=FakeProvider(offers=[spot_offer("s-1", "m-1", spot_price)], capabilities=SPOT_CAPS),
    )
    fleet = make_fleet(db, providers, enabled=["bids", "spot"], mode="cheaper")
    workload, _ = open_workload(fleet, kind="roi")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    assert host.connection_name == wins


async def test_a_spot_rental_pays_the_listed_price_and_tells_the_provider_its_most(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["spot"])
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    assert host.bid_hourly == pytest.approx(0.60), "no premium on a price the pool does not set"
    held = providers["spot"].instances[host.instance.instance_id]
    assert held.max_price == pytest.approx(2.0 - held.offer.storage_hourly), "the ceiling, less storage"


async def test_a_spot_price_that_moves_is_billed_from_then(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["spot"])
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    host.created_at -= 3600  # an hour at the old price
    providers["spot"].set_spot_price(host.instance.instance_id, 0.80)
    await fleet.handle_evictions()
    assert host.bid_hourly == pytest.approx(0.80) and host.state != "draining"
    assert host.accrued == pytest.approx(0.60 + host.offer.storage_hourly, rel=0.02), "the hour before, at the old price"
    assert any(e["kind"] == "spot_price_moved" for e in fleet.events.recent(20))


async def test_a_spot_price_past_the_ceiling_releases_the_host(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["spot"])
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    providers["spot"].set_spot_price(host.instance.instance_id, 2.50)
    await fleet.handle_evictions()
    assert host.released or host.state == "draining"
    (moved,) = [e for e in fleet.events.recent(20) if e["kind"] == "spot_price_moved"]
    assert "releasing it" in moved["summary"]


async def test_an_interruption_warning_drains_the_host_at_once(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["spot"])
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    host.state = "ready"
    providers["spot"].warn_interruption(host.instance.instance_id)
    await fleet.handle_evictions()
    assert host.state == "draining"
    assert any(e["kind"] == "interruption_warned" for e in fleet.events.recent(20))


async def test_a_spot_host_taken_back_is_replaced_never_bid_for(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["spot"])
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    providers["spot"].evict(host.instance.instance_id)
    await fleet.handle_evictions()
    assert "set_bid" not in providers["spot"].calls, "a spot price is not won back with a bid"
    assert host.released


# --- leases, and connections without a dead-man timer ---


async def test_a_long_lease_is_not_rented_on_a_connection_without_a_deadman_timer(db):
    providers = market(fixed=FakeProvider(offers=[fixed_offer("f-1", "m-1", 0.20)], capabilities=FIXED_CAPS))
    fleet = make_fleet(db, providers, mode="cheaper")
    assert fleet.max_lease_hours() is None, "another connection arms a timer, so leases may be long"
    longest = fleet.rented.teardown.max_hours_without_deadman
    workload, _ = open_workload(fleet, kind="roi", hours=longest + 1)
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    assert host.connection_name != "fixed", "the cheapest offer, but nothing there could stop it billing"


async def test_only_connections_without_a_timer_cap_every_lease(db):
    providers = {"fixed": FakeProvider(offers=[fixed_offer("f-1", "m-1", 0.20)], capabilities=FIXED_CAPS)}
    fleet = make_fleet(db, providers)
    assert fleet.max_lease_hours() == fleet.rented.teardown.max_hours_without_deadman


async def test_spend_is_recorded_per_connection(db):
    providers = market()
    fleet = make_fleet(db, providers, mode="cheaper")
    workload, lease = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    await fleet.record_spend(lease)
    assert {r["connection"] for r in db.query("SELECT connection FROM spend WHERE host_id LIKE 'rented-%'")} == {"spot"}


# --- what the reviews found (each held here) ---


async def test_the_shared_pool_also_takes_what_costs_least_per_worker_hour(db):
    """D131 for every rental: a spot price under a dearer bid wins for the shared pool too."""
    providers = market(
        bids=FakeProvider(offers=[bid_offer("b-1", "m-1", 0.90)]),
        spot=FakeProvider(offers=[spot_offer("s-1", "m-2", 0.60)], capabilities=SPOT_CAPS),
    )
    fleet = make_fleet(db, providers, enabled=["bids", "spot"], mode="interruptible")
    lease = fleet.leases.open(workers=4, max_hours=2, max_spend=20, allow_rent=True)
    host = await fleet.rent_one(lease, ["test"])
    assert host is not None and host.connection_name == "spot"
    (note,) = [e for e in fleet.events.recent(20) if e["kind"] == "rental_kind"]
    assert note["summary"].startswith("rental kind: spot on m-2 (spot)")


async def test_an_operator_chosen_offer_is_that_providers_offer(db):
    providers = market(
        bids=FakeProvider(offers=[bid_offer("same", "m-1", 0.40)]),
        spot=FakeProvider(offers=[spot_offer("same", "m-9", 1.60)], capabilities=SPOT_CAPS),
    )
    fleet = make_fleet(db, providers, enabled=["bids", "spot"], mode="interruptible")
    lease = fleet.leases.open(workers=4, max_hours=2, max_spend=20, allow_rent=True)
    assert await fleet.rent_one(lease, ["chosen"], offer_id="same") is None
    assert "more than one provider" in fleet.last_refusal, "an id two providers list is not guessed at"
    host = await fleet.rent_one(lease, ["chosen"], offer_id="same", connection="spot")
    assert host.connection_name == "spot" and host.offer.machine_id == "m-9"


async def test_a_spot_rise_within_the_ceiling_still_answers_to_the_burn_cap(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["spot"])
    fleet.config.limits.max_hourly_burn = 1.0
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    host.state = "ready"
    providers["spot"].set_spot_price(host.instance.instance_id, 1.50)  # under the $2 ceiling, over the burn
    await fleet.handle_evictions()
    assert host.state == "draining" or host.released
    assert any(e["kind"] == "spot_price_refused" for e in fleet.events.recent(20))


async def test_a_host_without_a_deadman_timer_is_ended_when_its_hours_are_up(db):
    """However its lease came to be longer — restarted, extended, amended — nothing on the
    machine could stop it billing."""
    providers = {"fixed": FakeProvider(offers=[fixed_offer("f-1", "m-1", 0.20)], capabilities=FIXED_CAPS)}
    fleet = make_fleet(db, providers, mode="on_demand")
    limit = fleet.rented.teardown.max_hours_without_deadman
    workload, _ = open_workload(fleet, kind="on_demand", hours=limit)
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    host.state = "ready"
    host.created_at -= (limit + 0.1) * 3600
    await fleet.handle_evictions()
    assert host.state == "draining" or host.released
    assert any(e["kind"] == "timerless_limit" for e in fleet.events.recent(20))


async def test_an_event_about_a_host_names_that_hosts_connection(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["bids", "spot"], mode="interruptible")
    workload, _ = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    assert host.connection_name == "spot" and fleet.connection == "bids"
    providers["spot"].set_spot_price(host.instance.instance_id, 0.70)
    await fleet.handle_evictions()
    (moved,) = [e for e in fleet.events.recent(20) if e["kind"] == "spot_price_moved"]
    assert moved["numbers"]["connection"] == "spot", "not the first enabled connection's"


async def test_a_plugin_that_raises_takes_only_its_own_offers_with_it(db):
    providers = market()

    async def broken(query):
        raise RuntimeError("a bug in the plug-in")

    providers["bids"].search_offers = broken
    fleet = make_fleet(db, providers, mode="cheaper")
    assert {o.connection for o in await fleet._offers()} == {"spot", "fixed"}
    assert "plug-in failed" in fleet.searching["bids"].error


async def test_a_parked_host_is_not_restarted_through_a_disabled_connection(db):
    providers = market()
    fleet = make_fleet(db, providers, enabled=["spot"])
    workload, lease = open_workload(fleet, kind="interruptible")
    await run(fleet, workload)
    (host,) = fleet.hosts_of("w")
    host.state = "parked"
    fleet.rented.providers["spot"].enabled = False
    fleet.rented.providers["bids"].enabled = True
    assert await fleet.restart_parked(lease, "w") is None


# --- the second review (each held here) ---


async def test_a_renamed_connections_volumes_are_still_the_pools(db):
    from gpm_server.providers.base import VolumeInfo
    from gpm_server.workload_store import Volume

    caps = ProviderCapabilities(interruptible=True, self_terminate=True, reports_charges=True, volumes=True)
    shared = FakeProvider(capabilities=caps)
    before = make_fleet(db, {"old": shared})
    before.claim_records()
    workload, _ = open_workload(before)
    shared.volumes["v-1"] = VolumeInfo(volume_id="v-1", machine_id="m-1", label=before.label_prefix + "w/models", size_gb=20)
    WorkloadStore(db).add_volume(Volume(volume_id="v-1", workload="w", machine_id="m-1", size_gb=20, hourly=0.01,
                                        lease_id=None, created_at=time.time(), connection="old"))
    # The same provider, renamed (each fake's type is fake-<name>: keep the type, change the name).
    config = PoolConfig.model_validate({**before.config.model_dump(), "rented": {
        **before.config.rented.model_dump(), "providers": {"new": {"type": "fake-old"}}}})
    after = Fleet(config, config.rented, {"new": shared}, before.leases, EventLog(db), SpendLedger(db))
    after.claim_records()
    after.workloads = {"w": workload}
    await after.sweep_volumes()
    assert "v-1" in shared.volumes, "a live workload's volume, under the connection's old name, is kept"


def test_a_name_once_one_providers_is_not_reused_for_another(db):
    from gpm_server.supervisor.renting import ConnectionNameReused

    make_fleet(db, {"main": FakeProvider()}).claim_records()  # type fake-main
    config = PoolConfig.model_validate({**make_fleet(db, {"main": FakeProvider()}).config.model_dump()})
    config.rented.providers["main"].type = "something-else"
    fleet = Fleet(config, config.rented, {"main": FakeProvider()}, LeaseStore(db), EventLog(db), SpendLedger(db))
    with pytest.raises(ConnectionNameReused, match="give the new provider another name"):
        fleet.claim_records()


def test_volumes_and_held_back_rows_stop_a_removal_but_are_not_counted_as_hosts():
    from gpm_server.configplan import RentedNow, plan_changes

    fleet_config = lambda providers, hosts: PoolConfig.model_validate({  # noqa: E731
        "pool": {"name": "t", "model_set": ["m1"]}, "auth": {"app_keys": ["k"]},
        "hosts": [{"id": "l", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "limits": {"max_rented_hosts": hosts},
        "rented": {"providers": providers, "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 2.0}}})
    both = {"a": {"type": "a"}, "b": {"type": "b"}}
    held = [RentedNow("rented-1", 0.5, provider="a"), RentedNow("volume:v-1", 0.01, provider="b")]
    lowered = plan_changes(fleet_config(both, 4), fleet_config(both, 1), held)
    assert not any("drained" in c.detail for c in lowered if c.kind == "max_rented_hosts_lowered"), "one host, not two"
    (dropped,) = [c for c in plan_changes(fleet_config(both, 4), fleet_config({"a": {"type": "a"}}, 4), held)
                  if c.kind == "providers"]
    assert dropped.refused and "volume:v-1" in dropped.refused


async def test_the_cost_of_a_machine_counts_its_download_and_its_record(db):
    cheap_fetch = fixed_offer("near", "m-near", 0.50, download_per_gb=0.0)
    dear_fetch = fixed_offer("far", "m-far", 0.48, download_per_gb=0.20)  # 20 GB: $4 to fetch, once
    fleet = make_fleet(db, {"fixed": FakeProvider(offers=[cheap_fetch, dear_fetch], capabilities=FIXED_CAPS)},
                       mode="on_demand")
    lease = fleet.leases.open(workers=4, max_hours=1, max_spend=20, allow_rent=True)
    host = await fleet.rent_one(lease, ["test"])
    assert host.offer.offer_id == "near", "a cent an hour cheaper does not pay a $4 download for an hour"


async def test_the_markets_best_offer_is_the_one_the_pool_would_rent(db):
    providers = market(
        bids=FakeProvider(offers=[bid_offer("b-1", "m-1", 0.90)]),
        spot=FakeProvider(offers=[spot_offer("s-1", "m-2", 0.60)], capabilities=SPOT_CAPS),
    )
    fleet = make_fleet(db, providers, enabled=["bids", "spot"], mode="interruptible")
    preview = await fleet.market_preview(search=True)
    first = preview["best"][0]
    assert (first["connection"], first["priced"]) == ("spot", "spot") and first["per_worker_hour"] is not None
