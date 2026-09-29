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
from typing import Iterable, Optional, Sequence

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
) -> KindCost:
    """Expected spend over `hours`, counting what an eviction costs a workload.

    On demand: the price times the hours. A bid adds, per expected eviction, the hours spent
    getting a replacement ready (paid, and serving nothing) and the capacity the workload lacks
    meanwhile — priced at `lost_capacity_hourly` (what that capacity would cost on demand) times
    this host's share of the workload's workers: a lone host takes the whole workload down with it,
    one of four takes a quarter.
    """
    base = hourly * hours
    if not interruptible:
        return KindCost(
            offer_id, machine_id, False, hourly, round(base, 4), round(base / max(1, workers) / max(hours, 1e-9), 5),
            [f"on demand ${hourly:.3f}/h x {hours:g}h = ${base:.2f}; nothing can take it away"],
        )
    evictions = evictions_per_hour * hours
    per_eviction = ready_hours * hourly + ready_hours * lost_capacity_hourly * share_of_workload
    total = base + evictions * per_eviction
    return KindCost(
        offer_id, machine_id, True, hourly, round(total, 4), round(total / max(1, workers) / max(hours, 1e-9), 5),
        [f"bid ${hourly:.3f}/h x {hours:g}h = ${base:.2f}, plus {evictions:.2f} expected evictions "
         f"({evictions_per_hour:.3f}/h) at ${per_eviction:.2f} each: {ready_hours * 60:.0f} min to replace, "
         f"{share_of_workload:.0%} of the workload's capacity lost meanwhile = ${total:.2f}"],
    )


def order_by_expected_cost(costs: Sequence[KindCost]) -> tuple[list[KindCost], list[str]]:
    """The candidates, cheapest per worker-hour first, and the one line that says why the first won."""
    ordered = sorted(costs, key=lambda c: (c.per_worker_hour, c.expected, c.offer_id))
    if not ordered:
        return [], ["no candidate to compare"]
    first = ordered[0]
    kind = "a bid" if first.interruptible else "on demand"
    other = next((c for c in ordered[1:] if c.interruptible != first.interruptible), None)
    why = f"rental kind: {kind} on {first.machine_id} at ${first.per_worker_hour:.4f} per worker-hour expected"
    if other is not None:
        why += (f", against {'a bid' if other.interruptible else 'on demand'} on {other.machine_id} at "
                f"${other.per_worker_hour:.4f}")
    return ordered, [why, *first.reasons]
