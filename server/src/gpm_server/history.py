"""What each machine has actually done for this pool (D69).

The market describes a machine by price, memory, a reliability score and a download speed. Live
use showed that is not enough: two machines under one hardware label differed three and a half
times in service time and four times in cost per request, and one machine was rented seven
times having reached `ready` twice. None of that is in an offer; all of it is in what the pool
already wrote down.

So this is a **view**, not new collection — the decision log joined to the request log. It is
built and judged on its own before any model is asked to advise on it, which is the order the
owner chose.
"""

from __future__ import annotations

import dataclasses
import re
import statistics
from typing import Any, Iterable, Optional

#: Events that say a rental ended badly, and what to call it.
_FAILURES = {
    "prepare_failed": "could not hold the model set",
    "host_stuck_starting": "never started",
    "host_without_startup": "came up without its start-up script",
}


@dataclasses.dataclass(frozen=True)
class MachineRecord:
    """One machine's record with this pool. Every field is a count or a measurement."""

    machine_id: str
    hardware: str
    rentals: int = 0
    reached_ready: int = 0
    failures: int = 0
    evictions: int = 0
    #: Minutes from the bid to holding the model set, where it got that far.
    minutes_to_ready: Optional[float] = None
    #: Requests served, and the median seconds each took once dispatched.
    requests: int = 0
    service_s: Optional[float] = None
    #: Tokens a second across everything it served, where the engine reported them.
    tokens_per_s: Optional[float] = None
    #: What an hour on it was bid at, last time.
    last_bid_hourly: Optional[float] = None
    #: Hours it was held, from each rental to its release (or the log's last word, for one still
    #: held) — what an eviction rate is measured over (D115).
    hours_rented: float = 0.0

    @property
    def evictions_per_hour(self) -> Optional[float]:
        """Evictions per rented hour; None until it has been held long enough to say (an hour)."""
        if self.hours_rented < 1.0:
            return None
        return self.evictions / self.hours_rented

    @property
    def reliability(self) -> Optional[float]:
        """How often a rental of this machine ended up serving. None until it has been tried."""
        if not self.rentals:
            return None
        return self.reached_ready / self.rentals

    def as_dict(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["reliability"] = self.reliability
        record["evictions_per_hour"] = self.evictions_per_hour
        return record


#: Older rentals named the machine only in the sentence. Reading it back is a one-time
#: concession to logs already written — from D69 on, the event carries it as a number.
_IN_THE_SENTENCE = re.compile(r"\bon (\S+) \(([^)]+)\)")


def _machine_in(event: dict[str, Any]) -> Optional[tuple[str, str]]:
    numbers = event.get("numbers") or {}
    if numbers.get("machine"):
        return str(numbers["machine"]), str(numbers.get("hardware") or "unknown")
    found = _IN_THE_SENTENCE.search(str(event.get("summary") or ""))
    if found is None:
        return None
    return found.group(1), found.group(2)


def build(events: Iterable[dict[str, Any]], requests: Iterable[dict[str, Any]]) -> dict[str, MachineRecord]:
    """Fold the two logs into one record per machine.

    `events` is the decision log newest-last; `requests` the request log. Nothing here reads a
    provider or a host: a pool that has been running has this already.
    """
    machine_of: dict[str, str] = {}      # host id -> machine id
    hardware: dict[str, str] = {}
    rentals: dict[str, int] = {}
    ready: dict[str, int] = {}
    failures: dict[str, int] = {}
    evictions: dict[str, int] = {}
    bid: dict[str, float] = {}
    rented_at: dict[str, float] = {}
    ended_at: dict[str, float] = {}
    ready_minutes: dict[str, list[float]] = {}
    last_seen = 0.0

    for event in events:
        last_seen = max(last_seen, float(event.get("ts") or 0.0))
        host_id = event.get("host_id")
        numbers = event.get("numbers") or {}
        kind = event.get("kind")
        if kind == "rented" and host_id:
            named = _machine_in(event)
            if named is None:
                continue
            machine, hardware_seen = named
            machine_of[host_id] = machine
            hardware.setdefault(machine, hardware_seen)
            rentals[machine] = rentals.get(machine, 0) + 1
            rented_at[host_id] = float(event.get("ts") or 0.0)
            if numbers.get("bid") is not None:
                bid[machine] = float(numbers["bid"])
            continue

        machine = machine_of.get(host_id or "")
        if machine is None:
            continue  # a rental this log does not reach back far enough to have seen
        if kind == "prepared":
            ready[machine] = ready.get(machine, 0) + 1
            started = rented_at.get(host_id or "")
            if started:
                ready_minutes.setdefault(machine, []).append(
                    (float(event.get("ts") or 0.0) - started) / 60
                )
        elif kind == "eviction":
            evictions[machine] = evictions.get(machine, 0) + 1
        elif kind in ("released", "host_gone"):
            ended_at.setdefault(host_id or "", float(event.get("ts") or 0.0))
        elif kind in _FAILURES:
            failures[machine] = failures.get(machine, 0) + 1

    served: dict[str, list[float]] = {}
    tokens: dict[str, float] = {}
    seconds: dict[str, float] = {}
    for row in requests:
        machine = machine_of.get(row.get("host_id") or "")
        if machine is None or row.get("outcome") != "ok":
            continue
        if row.get("latency_ms") is not None:
            served.setdefault(machine, []).append(float(row["latency_ms"]) / 1000)
        if row.get("tokens_out") and row.get("generate_ms"):
            tokens[machine] = tokens.get(machine, 0.0) + float(row["tokens_out"])
            seconds[machine] = seconds.get(machine, 0.0) + float(row["generate_ms"]) / 1000

    held: dict[str, float] = {}
    for host_id, started in rented_at.items():
        until = ended_at.get(host_id, last_seen)
        if until > started:
            held[machine_of[host_id]] = held.get(machine_of[host_id], 0.0) + (until - started) / 3600

    records = {}
    for machine in sorted(set(rentals) | set(served)):
        times = served.get(machine, [])
        generated = seconds.get(machine, 0.0)
        records[machine] = MachineRecord(
            machine_id=machine,
            hardware=hardware.get(machine, "unknown"),
            rentals=rentals.get(machine, 0),
            reached_ready=ready.get(machine, 0),
            failures=failures.get(machine, 0),
            evictions=evictions.get(machine, 0),
            minutes_to_ready=(
                round(statistics.mean(ready_minutes[machine]), 2) if ready_minutes.get(machine) else None
            ),
            requests=len(times),
            service_s=round(statistics.median(times), 2) if times else None,
            tokens_per_s=(
                round(tokens[machine] / generated, 1) if machine in tokens and generated > 0 else None
            ),
            last_bid_hourly=bid.get(machine),
            hours_rented=round(held.get(machine, 0.0), 3),
        )
    return records


def adjustment(record: Optional[MachineRecord], cfg) -> tuple[float, Optional[str]]:
    """What this machine's record does to its offer's score, and why (D69).

    Deterministic, bounded, and explainable in one line — the simpler mechanism the advisor has
    to beat before it is worth its non-determinism. A machine nobody has tried is not penalised:
    an unknown machine and a bad one are different things.
    """
    if record is None or not cfg.enabled or record.rentals < cfg.min_rentals:
        return 1.0, None

    reliability = record.reliability
    if reliability is not None and reliability < cfg.min_reliability:
        return cfg.penalty, (
            f"rented {record.rentals}x, reached ready {record.reached_ready}x"
            + (f", {record.failures} failed" if record.failures else "")
        )

    # Throughput is the honest comparison between machines, where the engine reported it.
    if record.tokens_per_s is not None and cfg.good_tokens_per_s:
        if record.tokens_per_s >= cfg.good_tokens_per_s:
            return cfg.bonus, f"served {record.tokens_per_s:.0f} tokens/s here before"
        if record.tokens_per_s <= cfg.poor_tokens_per_s:
            return cfg.penalty, f"served only {record.tokens_per_s:.0f} tokens/s here before"
    return 1.0, None
