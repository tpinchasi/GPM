"""Workloads: the arithmetic (docs/spec/workloads.md, D115).

Pure functions — no I/O, no clock, no randomness — like every strategy: everything they need
arrives as arguments, everything they decide comes back with its reasons, so a plan can be read,
tested and replayed without a cloud account.

- how many answers one host may serve at once and still meet a latency target, from what the
  request log measured on that class of machine;
- how many hosts a workload starts with, and what its budget would be when none is typed;
- on-demand or bid, by expected cost over the hours left (the rental-kind rule).
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Iterable, Mapping, Optional, Sequence

#: A workload's name is part of its URL prefix and its log lines: plain, short, unambiguous.
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

#: How many answers at one concurrency must have been measured before their p95 is believed.
#: Fewer is noise — one slow answer is a p95 of one.
MIN_SAMPLES = 20

#: The margin on a derived budget: prices move, a host is re-rented after an eviction, and a
#: budget worked out to the cent is spent before the run ends.
BUDGET_MARGIN = 1.25


def valid_name(name: str) -> bool:
    """Plain, and never `rented-…`: a workload's name sits in its hosts' labels beside host ids,
    and a name shaped like one would make one label a prefix of another's."""
    return bool(NAME.match(name or "")) and not (name or "").startswith("rented-")


def p95(values: Sequence[float]) -> float:
    """The 95th percentile, nearest-rank: the value 95% of answers came in at or under."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


@dataclasses.dataclass(frozen=True)
class AtLatency:
    """How many answers one host may serve at once and still meet the target."""

    #: None when nothing measured says: the capacity profile's number is used, marked unmeasured.
    workers: Optional[int]
    measured: bool
    reasons: list[str]
    #: p95 in seconds by concurrency, for what was measured with enough answers.
    curve: dict[int, float] = dataclasses.field(default_factory=dict)


def workers_at_latency(
    samples: Iterable[tuple[int, float]], target_s: float, ceiling: int
) -> AtLatency:
    """The most answers at once whose p95 whole-answer latency met `target_s`.

    `samples` are (concurrency, latency in seconds) for one class of machine and one model, from
    the request log. A concurrency counts only with `MIN_SAMPLES` answers. The answer never
    exceeds `ceiling` — the capacity profile's number for the card, which is what the engine was
    sized for — and is None when no measured concurrency met the target or nothing was measured:
    a target is a wish until there is a log.
    """
    by_concurrency: dict[int, list[float]] = {}
    for concurrency, latency in samples:
        if concurrency and concurrency > 0 and latency is not None and latency >= 0:
            by_concurrency.setdefault(int(concurrency), []).append(float(latency))
    curve = {c: round(p95(v), 3) for c, v in sorted(by_concurrency.items()) if len(v) >= MIN_SAMPLES}
    if not curve:
        return AtLatency(
            workers=None, measured=False,
            reasons=[f"no concurrency has {MIN_SAMPLES} measured answers yet; the card's own number is used, unmeasured"],
        )
    meeting = [c for c, value in curve.items() if value <= target_s and c <= ceiling]
    if not meeting:
        lowest = min(curve)
        return AtLatency(
            workers=None, measured=True, curve=curve,
            reasons=[f"no measured concurrency met {target_s:g}s at p95 (at {lowest} at once it was {curve[lowest]:g}s); "
                     f"the card's own number is used, and the target will not be met on this class"],
        )
    best = max(meeting)
    return AtLatency(
        workers=best, measured=True, curve=curve,
        reasons=[f"{best} at once measured {curve[best]:g}s at p95, within {target_s:g}s"
                 + (f"; {min(c for c in curve if c > best)} at once was {curve[min(c for c in curve if c > best)]:g}s"
                    if any(c > best for c in curve) else "")],
    )


def hosts_at_start(parallel: int, workers_per_host: int) -> int:
    """Hosts needed for `parallel` answers at once, at `workers_per_host` each."""
    if parallel <= 0:
        return 0
    return max(1, math.ceil(parallel / max(1, workers_per_host)))


def derived_budget(hosts: int, hours: float, hourly: float, margin: float = BUDGET_MARGIN) -> float:
    """A dollar cap when none is typed: every host for every hour at the price found, with a
    margin — rounded up to the cent. Shown and confirmed before anything is opened."""
    return math.ceil(hosts * hours * hourly * margin * 100) / 100


@dataclasses.dataclass(frozen=True)
class Split:
    """How a host holding several models divides its workers between them (S7, D118): at most
    `caps[m]` answers of model *m* at once, `workers` = the sum of the caps, and `hosts` of them
    for the workload's answers at once."""

    hosts: int
    caps: dict[str, int]
    workers: int
    #: The share of the card the split uses, Σ caps[m] / w[m]: at most 1.
    load: float


def split_together(parallel: Mapping[str, int], per_host: Mapping[str, int]) -> Optional[Split]:
    """The fewest hosts that serve `parallel[m]` answers of each model at once, every host holding
    every model, each model within its target — or None where no split fits one host.

    `per_host[m]` is how many answers of model *m* alone one host serves within *m*'s target. An
    answer of *m* takes `1/per_host[m]` of the host (the linear-mixing assumption, D118): a host is
    within every target while `Σ caps[m] / per_host[m] ≤ 1`. The caps are fixed shares — a host
    runs exactly their sum — so no model starves another, and load that moves between models is
    not absorbed: it is what each model asked for, on the same cards."""
    models = [m for m in parallel if parallel[m] > 0]
    if not models or any(per_host.get(m, 0) < 1 for m in models):
        return None
    if sum(1 / per_host[m] for m in models) > 1 + 1e-9:
        return None  # one answer of each already fills the card

    def load_at(hosts: int) -> float:
        return sum(math.ceil(parallel[m] / hosts) / per_host[m] for m in models)

    # The load only falls as hosts are added (every cap is a ceiling of parallel / hosts), and at
    # the largest `parallel` every cap is 1, which fits: so the fewest hosts that fit is found by
    # halving, in a few dozen steps whatever the numbers — never one host at a time.
    low = max(1, math.ceil(sum(parallel[m] / per_host[m] for m in models) - 1e-9))
    high = max(low, max(parallel[m] for m in models))
    while low < high:
        middle = (low + high) // 2
        if load_at(middle) <= 1 + 1e-9:
            high = middle
        else:
            low = middle + 1
    caps = {m: math.ceil(parallel[m] / low) for m in models}
    return Split(hosts=low, caps=caps, workers=sum(caps.values()), load=round(load_at(low), 4))


def split_short(parallel: Mapping[str, int], per_host: Mapping[str, int], hosts: int, most: int) -> Optional[dict[str, int]]:
    """Each model's share on `hosts` hosts, where a group has fewer than it planned (D128): what
    each model asked for spread over them, or — where that is more than a card holds — the card
    filled in the proportion the models asked for, never more than `most` answers in all (what
    the hosts' engines were launched for). None where not even one of each fits."""
    models = [m for m in parallel if parallel[m] > 0]
    if not models or hosts < 1 or any(per_host.get(m, 0) < 1 for m in models):
        return None
    caps = {m: math.ceil(parallel[m] / hosts) for m in models}
    if sum(caps[m] / per_host[m] for m in models) > 1 + 1e-9 or sum(caps.values()) > most:
        scale = min(1 / sum(parallel[m] / per_host[m] for m in models), most / sum(parallel[m] for m in models))
        caps = {m: max(1, math.floor(scale * parallel[m] + 1e-9)) for m in models}
    if sum(caps[m] / per_host[m] for m in models) > 1 + 1e-9 or sum(caps.values()) > most:
        return None
    return caps


def split_fits(caps: Mapping[str, int], per_host: Mapping[str, int]) -> bool:
    """Does a card serving `per_host[m]` of each model alone hold this split within every target?"""
    if any(per_host.get(m, 0) < 1 for m in caps):
        return False
    return sum(caps[m] / per_host[m] for m in caps) <= 1 + 1e-9


@dataclasses.dataclass(frozen=True)
class Placement:
    """One way to place a workload's models, priced (D118)."""

    name: str                    # "together" or "apart"
    expected: float              # dollars over the hours, the rental-kind rule's
    hosts: int
    reasons: list[str]


#: Placements whose expected costs are this close are equal: the estimate is not finer than that.
PLACEMENT_TIE = 0.01


def choose_placement(together: Optional[Placement], apart: Optional[Placement]) -> tuple[Optional[Placement], list[str]]:
    """The cheaper placement by expected cost, and why. Within `PLACEMENT_TIE` of each other they
    are equal, and together wins: one kind of host and one card to rent, whole hosts rounded once
    rather than per model. (It does not absorb load moving between models: each host keeps a
    fixed share for each, D118.)"""
    if together is None and apart is None:
        return None, ["no placement can be rented"]
    if together is None:
        return apart, ["apart: no card on offer holds every model within its target"]
    if apart is None:
        return together, ["together: no placement apart could be rented"]
    against = f"${together.expected:.4f} together on {together.hosts} host(s), ${apart.expected:.4f} apart on {apart.hosts}"
    if together.expected <= apart.expected * (1 + PLACEMENT_TIE) + 1e-9:
        close = together.expected > apart.expected
        return together, [f"together: {against}" + (" — within 1%, so equal, and together is simpler" if close else "")]
    return apart, [f"apart: {against}"]


def time_to_ready_hours(model_gb: float, download_mbps: float, load_s: float = 180.0) -> float:
    """How long a new host takes to serve: its download at the offer's speed, then the engine's
    load. A host with no speed stated is given a slow link rather than an instant one."""
    mbps = download_mbps if download_mbps and download_mbps > 0 else 100.0
    return (model_gb * 8000.0 / mbps + load_s) / 3600.0


@dataclasses.dataclass(frozen=True)
class KindCost:
    """What one candidate is expected to cost over the hours left, and why."""

    offer_id: str
    machine_id: str
    interruptible: bool
    hourly: float
    expected: float
    per_worker_hour: float
    reasons: list[str]
    #: The provider connection it is offered through, and — interruptible — whether the pool
    #: bids for it or pays the provider's spot price (D129, D132).
    connection: str = ""
    bidding: bool = True

    @property
    def priced(self) -> str:
        """How it is paid for, in words: a bid, spot, or on demand."""
        return ("a bid" if self.bidding else "spot") if self.interruptible else "on demand"

    @property
    def where(self) -> str:
        return f"{self.machine_id} ({self.connection})" if self.connection else self.machine_id


def expected_cost(
    *,
    offer_id: str,
    machine_id: str,
    interruptible: bool,
    hourly: float,
    hours: float,
    workers: int,
    evictions_per_hour: float,
    ready_hours: float,
    lost_capacity_hourly: float,
    share_of_workload: float,
    connection: str = "",
    bidding: bool = True,
    download: float = 0.0,
) -> KindCost:
    """Expected spend over `hours`, counting what an eviction costs a workload.

    On demand: the price times the hours. A bid adds, per expected eviction, the hours spent
    getting a replacement ready (paid, and serving nothing) and the capacity the workload lacks
    meanwhile — priced at `lost_capacity_hourly` (what that capacity would cost on demand) times
    this host's share of the workload's workers: a lone host takes the whole workload down with it,
    one of four takes a quarter.
    """
    # What fetching the models costs on this machine, once (D108): a cheap host with a dear
    # download is not cheap for a short lease.
    fetch = f", plus ${download:.2f} to fetch the models" if download else ""
    base = hourly * hours + download
    if not interruptible:
        return KindCost(
            offer_id, machine_id, False, hourly, round(base, 4), round(base / max(1, workers) / max(hours, 1e-9), 5),
            [f"on demand ${hourly:.3f}/h x {hours:g}h{fetch} = ${base:.2f}; nothing can take it away"],
            connection, True,
        )
    evictions = evictions_per_hour * hours
    per_eviction = ready_hours * hourly + ready_hours * lost_capacity_hourly * share_of_workload
    total = base + evictions * per_eviction
    return KindCost(
        offer_id, machine_id, True, hourly, round(total, 4), round(total / max(1, workers) / max(hours, 1e-9), 5),
        [f"{'bid' if bidding else 'spot'} ${hourly:.3f}/h x {hours:g}h{fetch} = ${base:.2f}, plus {evictions:.2f} expected interruptions "
         f"({evictions_per_hour:.3f}/h) at ${per_eviction:.2f} each: {ready_hours * 60:.0f} min to replace, "
         f"{share_of_workload:.0%} of the workload's capacity lost meanwhile = ${total:.2f}"],
        connection, bidding,
    )


def order_by_expected_cost(costs: Sequence[KindCost]) -> tuple[list[KindCost], list[str]]:
    """The candidates, cheapest per worker-hour first, and the one line that says why the first won."""
    ordered = sorted(costs, key=lambda c: (c.per_worker_hour, c.expected, c.offer_id))
    if not ordered:
        return [], ["no candidate to compare"]
    first = ordered[0]
    # The runner-up of another kind, or else another provider: what the first won against.
    other = next((c for c in ordered[1:] if c.priced != first.priced), None) or next(
        (c for c in ordered[1:] if c.connection != first.connection), None)
    why = f"rental kind: {first.priced} on {first.where} at ${first.per_worker_hour:.4f} per worker-hour expected"
    if other is not None:
        why += f", against {other.priced} on {other.where} at ${other.per_worker_hour:.4f}"
    return ordered, [why, *first.reasons]
