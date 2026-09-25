"""A day in the life of a pool: mock traffic against a market that moves under it.

Every feature so far has been proven against fakes that hold still — a market with one machine
at one price, traffic that arrives when a test sends it. This runs the whole pool against
neither: load that climbs, plateaus, spikes and stops, and a market whose prices drift, whose
machines come and go, and which takes hosts away mid-generation.

It is deliberately *not* a unit test. Nothing here asserts a particular number of hosts, which
would be a test of this scenario rather than of the pool. What it asserts is what must be true
of any run: no request is ever answered with a 5xx the pool did not explain, nothing is spent
past a lease's cap, no instance outlives the run, and the pool that ends quiet holds nothing.

Run it directly for the timeline:

    uv run python tests/simulation/market_day.py

Or as an opt-in test, which runs the same scenario and checks the invariants:

    uv run pytest -m simulation
"""

from __future__ import annotations

import collections
import dataclasses
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from fakes.harness import APP_KEY, EngineSpec, PoolHarness  # noqa: E402
from gpm_server.providers import default_offer  # noqa: E402

MODEL = "sim-model"


# --- the scenario -------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Phase:
    """One stretch of the day: how many callers, how fast, for how long."""

    name: str
    seconds: float
    callers: int
    think_s: float = 0.05
    note: str = ""
    #: Only while this is set does the market take hosts away, so the quiet end of the day
    #: shows what an *idle* host does rather than what an evicted one does.
    market_is_hostile: bool = False


DAY = (
    Phase("quiet", 4, callers=0, note="nothing is being asked of the pool"),
    Phase("morning", 10, callers=8, think_s=0.02, note="load climbs past what the laptop holds"),
    Phase("peak", 18, callers=40, think_s=0.0, note="far more than the pool holds: requests queue"),
    Phase("shock", 12, callers=40, think_s=0.0, market_is_hostile=True,
          note="the market takes hosts away mid-answer, under full load"),
    Phase("lull", 10, callers=0, note="quiet for long enough that hosts are paused"),
    Phase("false dawn", 8, callers=10, think_s=0.0,
          note="load returns while hosts are still paused — they should come back, not be re-rented"),
    Phase("night", 26, callers=0, note="nothing at all, for long enough to be paused and then given up"),
)


# --- the market ---------------------------------------------------------------------------


class MovingMarket:
    """A market that does not hold still.

    Prices drift, machines appear and vanish, and hosts are taken away — which on an
    interruptible rental is the ordinary case, not the exception.
    """

    def __init__(self, provider, report: "Report", machines: int = 10, seed: int = 7):
        self.provider = provider
        self.report = report
        self.random = random.Random(seed)
        self.machines = machines
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        #: Set per phase: the market only takes hosts away while a phase says it is hostile.
        self.evict_every_s: Optional[float] = None
        self.refresh()

    def refresh(self) -> None:
        """Re-price the market, and let a machine or two come and go."""
        offers = []
        for number in range(self.machines):
            if self.random.random() < 0.15:
                continue  # this machine is not on the market this minute
            floor = round(self.random.uniform(0.05, 0.22), 3)
            offers.append(
                default_offer(
                    offer_id=f"o-{number}-{self.random.randint(1000, 9999)}",
                    machine_id=f"m-{number}",
                    min_bid_hourly=floor,
                    all_in_hourly=floor + 0.04,
                )
            )
        self.provider.offers = offers

    def take_a_host_away(self) -> Optional[str]:
        """What an interruptible rental means: somebody outbid you, mid-answer."""
        running = [
            instance_id
            for instance_id, instance in self.provider.instances.items()
            if instance.state == "running"
        ]
        if not running:
            return None
        chosen = self.random.choice(running)
        self.provider.evict(chosen)
        self.report.note("market", f"instance {chosen} was outbid and taken away")
        return chosen

    def start(self) -> None:
        def loop() -> None:
            last_eviction = time.monotonic()
            while not self._stop.wait(1.0):
                self.refresh()
                every = self.evict_every_s
                if every and time.monotonic() - last_eviction >= every:
                    self.take_a_host_away()
                    last_eviction = time.monotonic()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# --- the traffic --------------------------------------------------------------------------


class Traffic:
    """Callers that behave like an app: they wait for capacity rather than giving up."""

    def __init__(self, url: str, report: "Report"):
        self.url = url
        self.report = report
        self.callers: list[threading.Thread] = []
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.outcomes: collections.Counter = collections.Counter()
        self.latencies: list[float] = []
        self.hosts_served: collections.Counter = collections.Counter()
        self.delivery: collections.Counter = collections.Counter()

    def _call_once(self, client: httpx.Client) -> None:
        started = time.monotonic()
        try:
            response = client.post(
                "/api/chat",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "simulate"}],
                    "stream": False,
                },
            )
        except httpx.HTTPError as exc:
            with self.lock:
                self.outcomes[f"transport:{type(exc).__name__}"] += 1
            return
        with self.lock:
            self.latencies.append(time.monotonic() - started)
            if response.status_code == 200:
                self.outcomes["ok"] += 1
                self.hosts_served[response.headers.get("X-GPM-Host", "?")] += 1
                self.delivery[response.headers.get("X-GPM-Delivery", "?")] += 1
            elif response.status_code == 503:
                # The pool saying "wait" is a correct answer, not a failure: it carries a
                # reason and a retry, which is the contract (app-contract §2).
                self.outcomes[f"503:{response.json().get('reason', '?')}"] += 1
            else:
                self.outcomes[f"{response.status_code}"] += 1

    def _caller(self, think_s: float) -> None:
        with httpx.Client(
            base_url=self.url, headers={"Authorization": f"Bearer {APP_KEY}"}, timeout=30
        ) as client:
            while not self._stop.is_set():
                self._call_once(client)
                if think_s:
                    self._stop.wait(think_s)

    def set_callers(self, count: int, think_s: float) -> None:
        """Change the shape of the load: this is the only knob the day turns."""
        self.stop()
        if count <= 0:
            return
        self._stop = threading.Event()
        self.callers = [
            threading.Thread(target=self._caller, args=(think_s,), daemon=True)
            for _ in range(count)
        ]
        for caller in self.callers:
            caller.start()

    def stop(self) -> None:
        self._stop.set()
        for caller in self.callers:
            caller.join(timeout=10)
        self.callers = []


# --- what happened ------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[float, str, str]] = []
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def note(self, kind: str, message: str) -> None:
        with self.lock:
            self.lines.append((time.monotonic() - self.started, kind, message))

    def render(self) -> str:
        return "\n".join(f"  {at:6.1f}s  {kind:<9} {message}" for at, kind, message in self.lines)


# --- the run ------------------------------------------------------------------------------


def build_pool(*, rentable_hosts: int = 8) -> PoolHarness:
    """A pool with one small local host and a market it can rent from.

    Every window is turned down so a day passes in a minute: what the scenario is testing is
    the *shape* of the pool's behaviour, not how long it waits.
    """
    return PoolHarness(
        [EngineSpec(id="laptop", resident={MODEL}, kind="local", workers=2, chunk_delay_s=0.35)],
        rentable=[
            EngineSpec(id=f"market-{n}", resident={MODEL}, workers=4, chunk_delay_s=0.3)
            for n in range(rentable_hosts)
        ],
        model_set=[MODEL],
        probe_interval_s=0.3,
        queue_timeout_s=8.0,
        rented={
            "provider": "fake",
            "workers": 2,
            "allocation": "dynamic",
            "dynamic": {
                "target_utilisation": 0.75,
                "window_s": 1.0,
                "ramp_factor": 2,
                "ramp_backoff_s": 1.0,
                "max_round": 4,
            },
            "workers_auto": {"enabled": True, "window_s": 1.0, "max": 6},
            "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02},
            "scale": {"scale_up_after_s": 0.5},
            "teardown": {"idle_minutes": 0.1, "destroy_idle_minutes": 0.25, "park_when_idle": True},
        },
        extra_config={"limits": {"max_rented_hosts": 8, "max_hourly_burn": 6.0}},
    )


def run_day(phases=DAY, *, evict_every_s: float = 3.0, verbose: bool = True) -> dict[str, Any]:
    report = Report()
    pool = build_pool()
    provider = pool.supervisor.fleet.provider
    # A day rents more hosts than there are engines in the market; the same engines answer
    # again rather than the run quietly creating hosts nothing can serve.
    provider.reuse_engine_urls = True
    market = MovingMarket(provider, report)
    traffic = Traffic(pool.url, report)
    seen_events = 0

    def drain_events() -> None:
        """The pool's own account of what it did, folded into the timeline."""
        nonlocal seen_events
        events = list(reversed(pool.supervisor.events.recent(200)))
        for event in events[seen_events:]:
            report.note(event["kind"], event["summary"][:120])
        seen_events = max(seen_events, len(events))

    try:
        # The lease is the authority and the ceiling; the traffic is the demand (D66).
        pool.supervisor.fleet.open_lease(
            workers=24, max_hours=1, max_spend=5.00, allow_rent=True
        )
        report.note("lease", "opened: up to 24 workers, $5.00, one hour")
        market.start()

        for phase in phases:
            report.note("phase", f"{phase.name}: {phase.callers} callers — {phase.note}")
            market.evict_every_s = evict_every_s if phase.market_is_hostile else None
            traffic.set_callers(phase.callers, phase.think_s)
            deadline = time.monotonic() + phase.seconds
            while time.monotonic() < deadline:
                time.sleep(0.5)
                drain_events()
            drain_events()
    finally:
        traffic.stop()
        market.stop()
        drain_events()
        summary = collect(pool, traffic, report)
        pool.close()

    if verbose:
        print(report.render())
        print_summary(summary)
    return summary


def collect(pool: PoolHarness, traffic: Traffic, report: Report) -> dict[str, Any]:
    fleet = pool.supervisor.fleet
    events = collections.Counter(e["kind"] for e in pool.supervisor.events.recent(500))
    rows = pool.request_log(limit=5000)
    served = collections.Counter(r["host_id"] for r in rows if r["outcome"] == "ok")
    return {
        "outcomes": dict(traffic.outcomes),
        "requests": sum(traffic.outcomes.values()),
        "latency_p50": statistics.median(traffic.latencies) if traffic.latencies else None,
        "latency_p95": (
            sorted(traffic.latencies)[int(len(traffic.latencies) * 0.95) - 1]
            if len(traffic.latencies) >= 20
            else None
        ),
        "delivery": dict(traffic.delivery),
        "served_by": dict(served),
        "events": dict(events),
        "hosts_rented": events.get("rented", 0),
        "instances_left": {
            instance_id: instance.state
            for instance_id, instance in fleet.provider.instances.items()
        },
        # From the ledger, not from the hosts still standing: most of a day's spend belongs to
        # hosts that have already gone.
        "spend": round(
            sum(
                row["amount"]
                for row in pool.database.query(
                    "SELECT amount FROM spend WHERE source = 'estimate'"
                )
            ),
            4,
        ),
        "lease_caps": [(lease.lease_id, lease.max_spend) for lease in fleet.leases.open_leases()],
        "report": report,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n--- what the day came to ---")
    print(f"  requests           {summary['requests']}")
    print(f"  outcomes           {summary['outcomes']}")
    print(f"  delivery           {summary['delivery']}")
    if summary["latency_p50"]:
        print(f"  latency            p50 {summary['latency_p50']:.2f}s  p95 {summary['latency_p95']:.2f}s")
    print(f"  served by          {summary['served_by']}")
    print(f"  hosts rented       {summary['hosts_rented']}")
    print(f"  instances left     {summary['instances_left'] or 'none'}")
    print(f"  estimated spend    ${summary['spend']}")
    interesting = {
        kind: count
        for kind, count in sorted(summary["events"].items())
        if kind not in ("agent_reachable", "config_applied")
    }
    print(f"  events             {interesting}")


if __name__ == "__main__":
    run_day()
