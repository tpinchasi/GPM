"""One number for each thing the search and the rental both mean (D108).

Found live: the search let through a machine offering 101 GB because it asked for at least 50,
the rental then asked that machine for 150 — a second number, in another part of the file — and
the provider refused three bids in a row. The price had the same split (a search maximum of
$3.10 beside a bid ceiling of $1.40), and so did the models' size (typed as 29 GB, stale at
36.7). Each is now one number: what is searched for is what is rented and what is paid.

Against the fake provider; nothing here spends money.
"""

import pytest
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.directory import OLLAMA, DirectoryStore
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor.renting import Fleet

MODEL = "m1"


def config(**rented_overrides):
    rented = {
        "provider": "fake",
        "workers": 2,
        "bidding": {"premium": 0.02},
        "offer_policy": {"min_disk_gb": 50, "max_all_in_hourly": 0.60},
        "scale": {"scale_up_after_s": 0},
    }
    rented.update(rented_overrides)
    return PoolConfig.model_validate({
        "pool": {"name": "test", "model_set": [MODEL]},
        "auth": {"app_keys": ["k"]},
        "hosts": [{"id": "local-1", "kind": "local", "workers": 1,
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "rented": rented,
    })


@pytest.fixture
def make_fleet(tmp_path):
    databases = []

    def build(cfg, offers=None):
        database = Database(tmp_path / f"gpm{len(databases)}.sqlite3")
        databases.append(database)
        provider = FakeProvider(offers=offers) if offers is not None else FakeProvider()
        return Fleet(cfg, cfg.rented, provider, LeaseStore(database), EventLog(database), SpendLedger(database))

    yield build
    for database in databases:
        database.close()


def open_lease(fleet):
    return fleet.leases.open(workers=2, max_hours=4, max_spend=5.0, allow_rent=True,
                             pool_max_all_in_hourly=fleet.rented.max_all_in_hourly)


def created(fleet):
    return list(fleet.provider.instances.values())


# --- disk ---


async def test_a_host_is_rented_with_the_disk_the_search_asks_for(make_fleet):
    fleet = make_fleet(config())
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (instance,) = created(fleet)
    assert instance.spec.disk_gb == 50


async def test_under_a_search_profile_the_host_gets_the_profiles_disk(make_fleet):
    fleet = make_fleet(config(
        search_profiles={"big": {"min_disk_gb": 80, "max_all_in_hourly": 0.60}},
        search_profile="big",
    ))
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (instance,) = created(fleet)
    assert instance.spec.disk_gb == 80


async def test_a_machine_with_less_disk_than_the_pool_rents_with_is_never_bid_on(make_fleet):
    """The live failure: 101 GB offered, 150 GB asked for, three refusals."""
    fleet = make_fleet(
        config(offer_policy={"min_disk_gb": 150, "max_all_in_hourly": 2.0}),
        offers=[default_offer("small", "m-small", disk_gb=101.25)],
    )
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert created(fleet) == []
    preview = await fleet.market_preview(hours=1)
    assert preview["passed"] == 0 and "disk" in preview["rejected_by_reason"]


# --- price ---


async def test_an_offer_is_priced_for_the_disk_rented_before_it_is_compared(make_fleet):
    """Storage at $0.003/GB-hour adds $0.15 to a 50 GB host: an offer whose floor alone fits
    the maximum is over it once its storage is counted, and is not considered."""
    cheap_storage = default_offer("fits", "m-1", min_bid_hourly=0.50, all_in_hourly=0.50, on_demand_hourly=None,
                                  storage_hourly=0.0, storage_per_gb_hourly=0.0001)
    dear_storage = default_offer("over", "m-2", min_bid_hourly=0.50, all_in_hourly=0.50, on_demand_hourly=None,
                                 storage_hourly=0.0, storage_per_gb_hourly=0.003)
    fleet = make_fleet(config(), offers=[cheap_storage, dear_storage])
    preview = await fleet.market_preview(hours=1)
    by_id = {offer["offer_id"]: offer for offer in preview["best"]}
    assert set(by_id) == {"fits"}
    assert by_id["fits"]["all_in"] == pytest.approx(0.505)
    assert by_id["fits"]["storage_hourly"] == pytest.approx(0.005)
    assert "all-in ceiling" in preview["rejected_by_reason"]


async def test_the_bid_leaves_room_for_the_storage_so_the_host_costs_no_more_than_the_maximum(make_fleet):
    offer = default_offer("o-1", "m-1", min_bid_hourly=0.55, all_in_hourly=0.55, on_demand_hourly=None,
                          storage_hourly=0.0, storage_per_gb_hourly=0.0004)
    fleet = make_fleet(config(), offers=[offer])
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (instance,) = created(fleet)
    # floor 0.55 + premium 0.02 = 0.57; storage for 50 GB is 0.02; the all-in 0.60 leaves 0.58.
    assert instance.bid_hourly == pytest.approx(0.57)
    assert instance.bid_hourly + 0.0004 * 50 <= 0.60


async def test_the_spending_bound_is_hosts_times_the_one_maximum(make_fleet):
    fleet = make_fleet(config())
    assert fleet.worst_case_hourly() == pytest.approx(fleet.config.limits.max_rented_hosts * 0.60)


# --- the models' size ---


def test_the_models_size_is_read_from_their_builds(make_fleet):
    fleet = make_fleet(config())
    assert fleet.model_set_gb([MODEL]) == 0.0
    assert fleet.model_sizes_unknown == [MODEL], "not measured is said, not counted as nothing silently"

    DirectoryStore(fleet.leases.db).put(OLLAMA, "m", {"tags": [{"name": MODEL, "size_gb": 36.7}]})
    assert fleet.model_set_gb([MODEL]) == pytest.approx(36.7)
    assert fleet.model_sizes_unknown == []


async def test_a_disk_too_small_for_the_models_rents_nothing_and_says_why(make_fleet):
    fleet = make_fleet(config(offer_policy={"min_disk_gb": 30, "max_all_in_hourly": 0.60}))
    DirectoryStore(fleet.leases.db).put(OLLAMA, "m", {"tags": [{"name": MODEL, "size_gb": 36.7}]})
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert created(fleet) == []
    preview = await fleet.market_preview(hours=1)
    assert preview["model_set_gb"] == pytest.approx(36.7)
    assert preview["passed"] == 0 and "disk" in preview["rejected_by_reason"]


# --- the old second numbers are refused by name ---


@pytest.mark.parametrize("old, where", [
    ({"disk_gb": 150}, "min_disk_gb"),
    ({"model_set_gb": 29}, "worked out from their builds"),
    ({"bidding": {"bid_ceiling": 1.4}}, "max_all_in_hourly"),
])
def test_a_second_number_for_the_same_thing_is_refused_saying_where_it_went(old, where):
    with pytest.raises(ValueError, match="D108") as refused:
        config(**old)
    assert where in str(refused.value)


@pytest.mark.parametrize("policy", [
    {"max_all_in_hourly": 0.60},
    {"min_disk_gb": 50},
])
def test_every_search_names_its_disk_and_its_ceiling(policy):
    with pytest.raises(ValueError, match="must be set"):
        config(offer_policy=policy)
    with pytest.raises(ValueError, match="search_profiles.cheap"):
        config(search_profiles={"cheap": policy})
