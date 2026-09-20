"""Strategies: pure functions of (market, hosts, lease, configuration) → decision.

docs/spec/plugin-interfaces.md §3. No I/O, no clock, no randomness of their own — everything
they need arrives as arguments, and everything they decide is returned **with the numbers that
produced it**. That is what makes them unit-testable and explainable.

They are advisory. The supervisor clamps every bid to both ceilings and re-checks every cap
*after* a strategy returns, so a faulty or hostile strategy cannot spend past the limits.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence

from .config import BiddingConfig, OfferPolicy, ScaleConfig, TeardownConfig
from .providers.base import Offer


@dataclasses.dataclass(frozen=True)
class Demand:
    """What a lease wants, against what the pool already has."""

    wanted_workers: int
    ready_workers_higher_tiers: int
    rented_workers: int
    overflow_since: Optional[float] = None
    #: Seconds the overflow has persisted, as measured by the caller.
    overflow_age_s: float = 0.0
    hosts_pending: int = 0
    rented_hosts: int = 0

    @property
    def overflow(self) -> int:
        return max(0, self.wanted_workers - self.ready_workers_higher_tiers - self.rented_workers)


@dataclasses.dataclass(frozen=True)
class LeaseView:
    """The spending authority a decision is made under."""

    lease_id: str
    allow_rent: bool
    hours_left: float
    dollars_left: float
    bid_ceiling: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class RentDecision:
    rent: bool
    count: int
    reasons: list[str]


@dataclasses.dataclass(frozen=True)
class Bid:
    hourly: float
    reasons: list[str]


@dataclasses.dataclass(frozen=True)
class EvictionDecision:
    action: str  # "rebid" | "replace" | "destroy"
    bid: Optional[float]
    reasons: list[str]


@dataclasses.dataclass(frozen=True)
class TeardownAction:
    host_id: str
    action: str  # "drain" | "park" | "destroy"
    reasons: list[str]


@dataclasses.dataclass(frozen=True)
class HostView:
    """What tear-down and eviction need to know about a rented host."""

    host_id: str
    priority: int
    busy_workers: int
    total_workers: int
    idle_seconds: float
    bid_hourly: float
    machine_id: Optional[str] = None
    hours_held: float = 0.0


# --- when to rent, and how many (spec §5) ---


def decide_rent(demand: Demand, lease: Optional[LeaseView], cfg: ScaleConfig, offer_workers: int) -> RentDecision:
    reasons: list[str] = []
    if lease is None:
        return RentDecision(False, 0, ["no lease is open, so nothing may be rented"])
    if not lease.allow_rent:
        return RentDecision(False, 0, [f"lease {lease.lease_id} does not allow renting"])
    if demand.overflow <= 0:
        return RentDecision(
            False,
            0,
            [f"no overflow: {demand.wanted_workers} wanted, "
             f"{demand.ready_workers_higher_tiers + demand.rented_workers} already available"],
        )
    if demand.overflow_age_s < cfg.scale_up_after_s:
        return RentDecision(
            False,
            0,
            [f"overflow has persisted {demand.overflow_age_s:.0f}s, below "
             f"{cfg.scale_up_after_s:.0f}s — a short burst is not worth a model download"],
        )
    if lease.hours_left < cfg.min_useful_hours:
        return RentDecision(
            False,
            0,
            [f"lease has {lease.hours_left:.2f}h left, below the {cfg.min_useful_hours}h a new "
             "host needs to be worth starting"],
        )
    if cfg.one_at_a_time and demand.hosts_pending > 0:
        return RentDecision(
            False, 0, [f"{demand.hosts_pending} host already on its way; one at a time"]
        )

    count = max(1, math.ceil(demand.overflow / max(1, offer_workers)))
    if cfg.one_at_a_time:
        count = 1
    reasons.append(
        f"overflow {demand.overflow} workers for {demand.overflow_age_s:.0f}s; "
        f"renting {count} host(s) of {offer_workers} workers"
    )
    return RentDecision(True, count, reasons)


# --- which offers are acceptable, and in what order (spec §6.1) ---


def reject_reasons(offer: Offer, policy: OfferPolicy, model_set_gb: float = 0.0) -> list[str]:
    """Hard filters. **Never relaxed unattended** — an empty result means stay paused."""
    # Each reason is "<filter>: <what happened>". The filter name is stable, so the console
    # can count rejections by filter; the rest carries the numbers.
    reasons: list[str] = []
    if offer.gpu_memory_gb < policy.min_gpu_memory_gb:
        reasons.append(
            f"gpu memory: {offer.gpu_memory_gb}GB below the {policy.min_gpu_memory_gb}GB minimum"
        )
    if offer.disk_gb < policy.min_disk_gb:
        reasons.append(f"disk: {offer.disk_gb}GB below the {policy.min_disk_gb}GB minimum")
    if policy.max_all_in_hourly is not None and offer.all_in_hourly > policy.max_all_in_hourly:
        reasons.append(
            f"all-in ceiling: ${offer.all_in_hourly:.3f}/h above ${policy.max_all_in_hourly:.3f}"
        )
    if policy.max_download_per_gb is not None and offer.download_per_gb > policy.max_download_per_gb:
        reasons.append(
            f"download price: ${offer.download_per_gb:.4f}/GB above "
            f"${policy.max_download_per_gb:.4f}"
        )
    if offer.download_mbps < policy.min_download_mbps:
        reasons.append(
            f"download speed: {offer.download_mbps:.0f}Mbps below the "
            f"{policy.min_download_mbps:.0f}Mbps minimum"
        )
    if offer.reliability < policy.min_reliability:
        reasons.append(
            f"reliability: {offer.reliability:.3f} below the {policy.min_reliability:.3f} minimum"
        )
    if policy.verified_only and not offer.verified:
        reasons.append("verification: the provider has not verified this machine")
    if offer.machine_id in policy.avoid_machines:
        reasons.append(f"avoid list: machine {offer.machine_id} is on it")
    excluded = next(
        (name for name in policy.exclude_hardware if name.lower() in offer.hardware.lower()), None
    )
    if excluded is not None:
        reasons.append(f"excluded hardware: {offer.hardware!r} matches {excluded!r}")
    if model_set_gb and offer.disk_gb < model_set_gb:
        reasons.append(
            f"disk: {offer.disk_gb}GB cannot hold the pool's model set ({model_set_gb}GB)"
        )
    return reasons


def filter_name(reason: str) -> str:
    return reason.split(":", 1)[0]


def score(offer: Offer, bid_hourly: float, hours: float, model_set_gb: float) -> float:
    """Throughput proxy per run-dollar, where run-dollars include the model download
    amortised over the hours the lease expects to use it (spec §6.1)."""
    download = offer.download_per_gb * model_set_gb
    run_dollars = (bid_hourly + offer.storage_hourly) * max(hours, 0.01) + download
    if run_dollars <= 0:
        return 0.0
    return offer.throughput_proxy / run_dollars


def rank_offers(
    offers: Sequence[Offer], policy: OfferPolicy, bidding: BiddingConfig, hours: float, model_set_gb: float
) -> tuple[list[tuple[Offer, float]], dict[str, list[str]]]:
    """Returns the acceptable offers best-first, and why each rejected one was rejected."""
    accepted: list[tuple[Offer, float]] = []
    rejected: dict[str, list[str]] = {}
    for offer in offers:
        reasons = reject_reasons(offer, policy, model_set_gb)
        if reasons:
            rejected[offer.offer_id] = reasons
            continue
        bid = price_bid(offer, bidding)
        if bid.hourly <= 0:
            rejected[offer.offer_id] = ["bid: " + bid.reasons[-1]]
            continue
        accepted.append((offer, score(offer, bid.hourly, hours, model_set_gb)))
    accepted.sort(key=lambda pair: (-pair[1], pair[0].offer_id))
    return accepted, rejected


# --- how much to bid (spec §6.1) ---


def price_bid(offer: Offer, cfg: BiddingConfig, lease_ceiling: Optional[float] = None) -> Bid:
    """`floor_plus_premium`: the market floor plus an **absolute** premium.

    A multiplier is wrong here — floors span an order of magnitude or more, so any fixed
    multiple is either free or wasteful.
    """
    if not offer.interruptible:
        # Nothing to bid: the price is fixed and the host cannot be outbid. A ceiling it
        # exceeds is a refusal, never a clamp — offering less does not rent it (D52).
        price = offer.all_in_hourly
        ceiling = cfg.bid_ceiling if lease_ceiling is None else min(cfg.bid_ceiling, lease_ceiling)
        reasons = [f"on-demand at ${price:.3f}/h — a fixed price, not a bid; it cannot be outbid"]
        if price > ceiling:
            reasons.append(f"above the ${ceiling:.3f} ceiling, and a fixed price cannot be lowered to meet it")
            return Bid(hourly=0.0, reasons=reasons)
        return Bid(hourly=round(price, 4), reasons=reasons)

    reasons = [f"floor ${offer.min_bid_hourly:.3f} + premium ${cfg.premium:.3f}"]
    bid = offer.min_bid_hourly + cfg.premium

    ceiling = cfg.bid_ceiling if lease_ceiling is None else min(cfg.bid_ceiling, lease_ceiling)
    if bid > ceiling:
        reasons.append(f"clamped to the bid ceiling ${ceiling:.3f}")
        bid = ceiling

    if offer.on_demand_hourly:
        crossover = cfg.on_demand_crossover * offer.on_demand_hourly
        if bid > crossover:
            reasons.append(
                f"clamped to {cfg.on_demand_crossover:g}x the on-demand price "
                f"(${crossover:.3f}) — past it an interruptible host carries the eviction risk "
                "without the discount"
            )
            bid = crossover

    if bid < offer.min_bid_hourly:
        # Below the floor nothing wins. That is a refusal, not a bid: the interruptible price
        # of this machine is not far enough under its on-demand price to be worth the risk.
        reasons.append(
            f"${bid:.3f} is below the floor ${offer.min_bid_hourly:.3f}, so no bid is placed"
        )
        return Bid(hourly=0.0, reasons=reasons)

    return Bid(hourly=round(bid, 4), reasons=reasons)


# --- on eviction: re-bid in place, or replace (spec §6.2) ---


def decide_eviction(
    host: HostView,
    same_machine_offer: Optional[Offer],
    best_alternative: Optional[Offer],
    lease: Optional[LeaseView],
    cfg: BiddingConfig,
    model_set_gb: float,
) -> EvictionDecision:
    """By cost over the hours the lease still has, not by habit."""
    if lease is None or not lease.allow_rent:
        return EvictionDecision(
            "destroy", None, ["no lease allows renting, so the host is not replaced"]
        )

    hours = lease.hours_left
    reasons: list[str] = []

    rebid = price_bid(same_machine_offer, cfg, lease.bid_ceiling) if same_machine_offer else None
    if rebid is not None and same_machine_offer is not None:
        alternative_rate = best_alternative.min_bid_hourly if best_alternative else rebid.hourly
        rebid_cost = (rebid.hourly - alternative_rate) * hours
        replace_cost = (
            best_alternative.download_per_gb * model_set_gb if best_alternative else float("inf")
        )
        reasons.append(
            f"re-bid in place ${rebid.hourly:.3f}/h costs ${rebid_cost:.2f} over {hours:.2f}h; "
            f"replacing costs ${replace_cost:.2f} in model download"
        )
        if rebid_cost <= replace_cost:
            return EvictionDecision("rebid", rebid.hourly, reasons + rebid.reasons)

    if best_alternative is None:
        return EvictionDecision(
            "destroy", None, reasons + ["no acceptable alternative offer; stopping billing"]
        )
    return EvictionDecision("replace", None, reasons + [f"replacing on {best_alternative.machine_id}"])


# --- who goes, and park or destroy (spec §9) ---


def decide_teardown(
    hosts: Sequence[HostView],
    demand: Demand,
    lease: Optional[LeaseView],
    cfg: TeardownConfig,
    lease_open: bool,
) -> list[TeardownAction]:
    """Idle time, not the hourly rate, dominates cost — so idleness is checked first."""
    actions: list[TeardownAction] = []
    idle_limit_s = cfg.idle_minutes * 60

    for host in sorted(hosts, key=lambda h: (-h.priority, -h.bid_hourly, h.busy_workers)):
        if lease is None or not lease_open:
            actions.append(
                TeardownAction(host.host_id, "destroy", ["no lease is open; rented hosts are released"])
            )
            continue
        if lease.dollars_left <= 0:
            actions.append(
                TeardownAction(host.host_id, "destroy", ["the lease's dollar cap is spent"])
            )
            continue
        if host.idle_seconds >= cfg.destroy_after_minutes * 60:
            actions.append(
                TeardownAction(
                    host.host_id,
                    "destroy",
                    [f"idle {host.idle_seconds / 60:.1f} min, past the {cfg.destroy_after_minutes:g} min limit"],
                )
            )
            continue
        if host.idle_seconds >= idle_limit_s:
            action = "park" if cfg.park_when_idle else "destroy"
            actions.append(
                TeardownAction(
                    host.host_id,
                    action,
                    [f"idle {host.idle_seconds / 60:.1f} min, past the {cfg.idle_minutes} min limit"],
                )
            )
            continue
        if demand.overflow <= 0 and demand.overflow_age_s == 0:
            actions.append(
                TeardownAction(
                    host.host_id,
                    "park" if cfg.park_when_idle else "destroy",
                    ["the overflow this host was rented for is gone"],
                )
            )

    return actions
