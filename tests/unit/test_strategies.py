"""Strategies are pure functions that return their reasons — so they are asserted directly."""

import pytest
from gpm_server.config import BiddingConfig, OfferPolicy, ScaleConfig, TeardownConfig
from gpm_server.providers import default_offer
from gpm_server.strategies import (
    Demand,
    HostView,
    LeaseView,
    decide_eviction,
    decide_rent,
    decide_teardown,
    price_bid,
    rank_offers,
    reject_reasons,
)

BIDDING = BiddingConfig(bid_ceiling=0.60, premium=0.02)
SCALE = ScaleConfig()


def lease(**overrides):
    base = dict(lease_id="l-1", allow_rent=True, hours_left=4.0, dollars_left=5.0)
    base.update(overrides)
    return LeaseView(**base)


# --- when to rent ---


def test_nothing_is_rented_without_a_lease():
    decision = decide_rent(Demand(wanted_workers=8, ready_workers_higher_tiers=0, rented_workers=0), None, SCALE, 2)
    assert not decision.rent
    assert "no lease" in decision.reasons[0]


def test_nothing_is_rented_by_a_lease_that_may_not_spend():
    demand = Demand(wanted_workers=8, ready_workers_higher_tiers=0, rented_workers=0, overflow_age_s=999)
    decision = decide_rent(demand, lease(allow_rent=False), SCALE, 2)
    assert not decision.rent
    assert "does not allow renting" in decision.reasons[0]


def test_a_short_burst_is_not_worth_a_model_download():
    demand = Demand(wanted_workers=8, ready_workers_higher_tiers=2, rented_workers=0, overflow_age_s=10)
    decision = decide_rent(demand, lease(), SCALE, 2)
    assert not decision.rent
    assert "short burst" in decision.reasons[0]


def test_persistent_overflow_rents():
    demand = Demand(wanted_workers=8, ready_workers_higher_tiers=2, rented_workers=0, overflow_age_s=200)
    decision = decide_rent(demand, lease(), SCALE, 2)
    assert decision.rent
    assert decision.count == 1  # one at a time: a bad market yields one failed bid, not five
    assert "overflow 6 workers" in decision.reasons[0]


def test_a_lease_that_ends_before_the_host_is_useful_rents_nothing():
    demand = Demand(wanted_workers=8, ready_workers_higher_tiers=0, rented_workers=0, overflow_age_s=200)
    decision = decide_rent(demand, lease(hours_left=0.2), SCALE, 2)
    assert not decision.rent
    assert "below the 1.0h" in decision.reasons[0]


def test_only_one_host_is_brought_up_at_a_time():
    demand = Demand(
        wanted_workers=8, ready_workers_higher_tiers=0, rented_workers=0, overflow_age_s=200, hosts_pending=1
    )
    decision = decide_rent(demand, lease(), SCALE, 2)
    assert not decision.rent
    assert "already on its way" in decision.reasons[0]


def test_no_overflow_rents_nothing():
    demand = Demand(wanted_workers=4, ready_workers_higher_tiers=4, rented_workers=0, overflow_age_s=999)
    assert not decide_rent(demand, lease(), SCALE, 2).rent


# --- which offers are acceptable ---


def test_every_hard_filter_says_why_it_rejected():
    policy = OfferPolicy(
        min_gpu_memory_gb=64, min_disk_gb=200, max_all_in_hourly=0.10,
        max_download_per_gb=0.005, min_download_mbps=1000, min_reliability=0.999,
    )
    reasons = reject_reasons(default_offer(), policy)
    joined = " ".join(reasons)
    assert "gpu memory" in joined
    assert "disk" in joined
    assert "all-in ceiling" in joined
    assert "download price" in joined
    assert "download speed" in joined
    assert "reliability" in joined
    # Every reason names its filter first, so rejections can be counted by filter.
    assert all(":" in reason for reason in reasons)


def test_an_unverified_machine_is_rejected_by_default():
    assert reject_reasons(default_offer(verified=False), OfferPolicy())
    assert not reject_reasons(default_offer(verified=False), OfferPolicy(verified_only=False))


def test_a_machine_on_the_avoid_list_is_rejected():
    policy = OfferPolicy(avoid_machines=["m-1"])
    assert "avoid list" in " ".join(reject_reasons(default_offer(machine_id="m-1"), policy))


def test_excluded_hardware_is_rejected_by_name():
    """A mining card passes every numeric filter and will not run the engine."""
    policy = OfferPolicy(exclude_hardware=["CMP", "P106"])
    reasons = reject_reasons(default_offer(hardware="1x CMP 170HX"), policy)
    assert reasons and reasons[0].startswith("excluded hardware:")
    assert not reject_reasons(default_offer(hardware="1x RTX 4090"), policy)


def test_ranking_keeps_the_rejections_and_their_reasons():
    offers = [
        default_offer(offer_id="cheap", machine_id="m-1", min_bid_hourly=0.10),
        default_offer(offer_id="dear", machine_id="m-2", min_bid_hourly=0.40),
        default_offer(offer_id="tiny", machine_id="m-3", gpu_memory_gb=8),
    ]
    accepted, rejected = rank_offers(offers, OfferPolicy(min_gpu_memory_gb=24), BIDDING, hours=4, model_set_gb=10)
    assert [offer.offer_id for offer, _ in accepted] == ["cheap", "dear"]
    assert "tiny" in rejected


# --- how much to bid ---


def test_the_bid_is_the_floor_plus_an_absolute_premium():
    bid = price_bid(default_offer(min_bid_hourly=0.10), BIDDING)
    assert bid.hourly == pytest.approx(0.12)
    assert "floor $0.100 + premium $0.020" in bid.reasons[0]


def test_the_bid_ceiling_clamps():
    # floor 0.50 + premium 0.02 = 0.52, clamped to a 0.51 ceiling that still clears the floor
    cfg = BiddingConfig(bid_ceiling=0.51, premium=0.02)
    bid = price_bid(default_offer(min_bid_hourly=0.50, on_demand_hourly=None), cfg)
    assert bid.hourly == 0.51
    assert any("bid ceiling" in reason for reason in bid.reasons)


def test_a_ceiling_below_the_floor_means_no_bid_rather_than_a_bid_that_cannot_win():
    bid = price_bid(default_offer(min_bid_hourly=5.0, on_demand_hourly=None), BIDDING)
    assert bid.hourly == 0.0
    assert "below the floor" in bid.reasons[-1]


def test_a_lease_may_tighten_the_ceiling_but_the_pool_still_caps():
    bid = price_bid(default_offer(min_bid_hourly=0.20, on_demand_hourly=None), BIDDING, lease_ceiling=0.21)
    assert bid.hourly == 0.21


def test_the_on_demand_crossover_clamps():
    """Past it an interruptible host carries the eviction risk without the discount."""
    # floor 0.30 + 0.02 = 0.32; crossover 0.8 x 0.39 = 0.312 pulls it down, still above floor
    bid = price_bid(default_offer(min_bid_hourly=0.30, on_demand_hourly=0.39), BIDDING)
    assert bid.hourly == pytest.approx(0.312)
    assert any("on-demand" in reason for reason in bid.reasons)


def test_a_crossover_below_the_floor_refuses_the_offer():
    """Seen on the live market: on-demand barely above the floor, so 0.8x lands under it.
    An interruptible price that close to on-demand is not worth the eviction risk."""
    bid = price_bid(default_offer(min_bid_hourly=0.16, on_demand_hourly=0.16), BIDDING)
    assert bid.hourly == 0.0
    assert "below the floor" in bid.reasons[-1]


def test_a_refused_bid_is_a_rejection_with_its_own_filter_name():
    offers = [default_offer(offer_id="tight", min_bid_hourly=0.16, on_demand_hourly=0.16)]
    accepted, rejected = rank_offers(offers, OfferPolicy(), BIDDING, hours=1, model_set_gb=1)
    assert accepted == []
    assert rejected["tight"][0].startswith("bid:")


# --- eviction ---


def test_re_bidding_in_place_wins_when_the_download_would_cost_more():
    host = HostView("h1", 20, 0, 2, 0, bid_hourly=0.12, machine_id="m-1")
    same = default_offer(machine_id="m-1", min_bid_hourly=0.13)
    alternative = default_offer(offer_id="o-2", machine_id="m-2", min_bid_hourly=0.12, download_per_gb=0.05)
    decision = decide_eviction(host, same, alternative, lease(hours_left=1.0), BIDDING, model_set_gb=20)
    assert decision.action == "rebid"
    assert decision.bid == pytest.approx(0.15)


def test_replacing_wins_when_holding_the_machine_costs_more_than_moving():
    host = HostView("h1", 20, 0, 2, 0, bid_hourly=0.12, machine_id="m-1")
    same = default_offer(machine_id="m-1", min_bid_hourly=0.50, on_demand_hourly=None)
    alternative = default_offer(offer_id="o-2", machine_id="m-2", min_bid_hourly=0.10, download_per_gb=0.0001)
    decision = decide_eviction(host, same, alternative, lease(hours_left=8.0), BIDDING, model_set_gb=1)
    assert decision.action == "replace"


def test_an_eviction_outside_a_lease_just_stops_billing():
    host = HostView("h1", 20, 0, 2, 0, bid_hourly=0.12, machine_id="m-1")
    decision = decide_eviction(host, default_offer(), default_offer(), None, BIDDING, model_set_gb=1)
    assert decision.action == "destroy"


# --- tear-down ---


def test_an_idle_host_is_released():
    host = HostView("h1", 20, 0, 2, idle_seconds=11 * 60, bid_hourly=0.12)
    actions = decide_teardown([host], Demand(4, 0, 2), lease(), TeardownConfig(), lease_open=True)
    assert actions[0].action == "park"
    assert "idle 11.0 min" in actions[0].reasons[0]


def test_an_idle_host_is_destroyed_where_parking_is_off():
    host = HostView("h1", 20, 0, 2, idle_seconds=11 * 60, bid_hourly=0.12)
    actions = decide_teardown(
        [host], Demand(4, 0, 2), lease(), TeardownConfig(park_when_idle=False), lease_open=True
    )
    assert actions[0].action == "destroy"


def test_with_no_lease_open_every_rented_host_goes():
    host = HostView("h1", 20, 0, 2, idle_seconds=0, bid_hourly=0.12)
    actions = decide_teardown([host], Demand(0, 0, 2), None, TeardownConfig(), lease_open=False)
    assert actions[0].action == "destroy"
    assert "no lease" in actions[0].reasons[0]


def test_a_busy_host_inside_a_lease_is_left_alone():
    host = HostView("h1", 20, busy_workers=2, total_workers=2, idle_seconds=0, bid_hourly=0.12)
    demand = Demand(wanted_workers=4, ready_workers_higher_tiers=0, rented_workers=2, overflow_age_s=30)
    assert decide_teardown([host], demand, lease(), TeardownConfig(), lease_open=True) == []
