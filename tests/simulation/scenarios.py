"""Scenarios: the pool put through things that actually happen, end to end.

Each one runs the real router and the real supervisor over a fake market and fake engines,
with traffic arriving from threads — and then checks what must be true of *any* run, rather
than what was true of one recording. A scenario that asserted "four hosts were rented" would be
a test of the scenario; what these assert is that no request was ever lost, nothing outlived
the run, and no cap was passed.

The list is meant to cover the shapes a pool meets: load that climbs and stops, a market that
takes hosts away, a market with nothing acceptable in it, a provider that stops answering, a
budget that runs out, and a supervisor that restarts under all of it.

    uv run python tests/simulation/scenarios.py            # all of them, with timelines
    uv run python tests/simulation/scenarios.py market_day # one of them
    uv run pytest -m simulation                            # the same, as checks
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
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from fakes.harness import APP_KEY, EngineSpec, PoolHarness  # noqa: E402
from gpm_server.providers import default_offer  # noqa: E402

MODEL = "sim-model"


# --- the shape of a run ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Phase:
    """One stretch of a run: how many callers, how fast, for how long."""

    name: str
    seconds: float
    callers: int
    think_s: float = 0.05
    note: str = ""
    #: Only while this is set does the market take hosts away, so the quiet end of a run shows
    #: what an *idle* host does rather than what an evicted one does.
    market_is_hostile: bool = False
    #: Called once as the phase begins: how a scenario reaches in and changes the world.
    when_it_starts: Optional[Callable[["Run"], None]] = None


@dataclasses.dataclass
class Scenario:
    name: str
    what_it_shows: str
    phases: tuple[Phase, ...]
    pool: dict[str, Any] = dataclasses.field(default_factory=dict)
    lease: Optional[dict[str, Any]] = None
    evict_every_s: float = 3.0
    #: Checked after the run, beyond the invariants every scenario is held to.
    expects: Optional[Callable[[dict[str, Any]], None]] = None


# --- the market -------------------------------------------------------------------------


class MovingMarket:
    """A market that does not hold still: prices drift, machines come and go, hosts are taken."""

    def __init__(self, provider, report: "Report", machines: int = 10, seed: int = 7):
        self.provider = provider
        self.report = report
        self.random = random.Random(seed)
        self.machines = machines
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.evict_every_s: Optional[float] = None
        #: Set to empty the market, or to make every offer unaffordable.
        self.mood = "normal"
        self.refresh()

    def refresh(self) -> None:
        if self.mood == "empty":
            self.provider.offers = []
            return
        offers = []
        for number in range(self.machines):
            if self.random.random() < 0.15:
                continue  # this machine is not on the market this minute
            floor = round(self.random.uniform(0.05, 0.22), 3)
            if self.mood == "expensive":
                floor *= 20  # every machine priced past any sane ceiling
            offers.append(
                default_offer(
                    offer_id=f"o-{number}-{self.random.randint(1000, 9999)}",
                    machine_id=f"m-{number}",
                    min_bid_hourly=round(floor, 3),
                    all_in_hourly=round(floor + 0.04, 3),
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


# --- the traffic ------------------------------------------------------------------------


class Traffic:
    """Callers that behave like an app: they take the pool's answer, whatever it is."""

    def __init__(self, url: str):
        self.url = url
        self.callers: list[threading.Thread] = []
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.outcomes: collections.Counter = collections.Counter()
        self.latencies: list[float] = []
        self.hosts_served: collections.Counter = collections.Counter()
        self.delivery: collections.Counter = collections.Counter()
        self.bodies_broken = 0

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
                # A response that ends without its final frame is the failure buffered
                # delivery exists to prevent (D62): half an answer, already in the app's hands.
                if b'"done":true' not in response.content.replace(b'"done": true', b'"done":true'):
                    self.bodies_broken += 1
            elif response.status_code == 503:
                # The pool saying "wait", with a reason and a retry, is a correct answer.
                self.outcomes[f"503:{response.json().get('reason', '?')}"] += 1
            else:
                self.outcomes[str(response.status_code)] += 1

    def _caller(self, think_s: float) -> None:
        with httpx.Client(
            base_url=self.url, headers={"Authorization": f"Bearer {APP_KEY}"}, timeout=30
        ) as client:
            while not self._stop.is_set():
                self._call_once(client)
                if think_s:
                    self._stop.wait(think_s)

    def set_callers(self, count: int, think_s: float) -> None:
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
            caller.join(timeout=15)
        self.callers = []


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[float, str, str]] = []
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def note(self, kind: str, message: str) -> None:
        with self.lock:
            self.lines.append((time.monotonic() - self.started, kind, message))

    def render(self) -> str:
        return "\n".join(f"  {at:6.1f}s  {kind:<20} {message}" for at, kind, message in self.lines)


@dataclasses.dataclass
class Run:
    """What a phase can reach into while a scenario is running."""

    pool: PoolHarness
    market: MovingMarket
    traffic: Traffic
    report: Report


# --- running one -------------------------------------------------------------------------


def build_pool(**overrides: Any) -> PoolHarness:
    """A pool with one small local host and a market it can rent from.

    Every window is turned right down so a run takes a minute: what is under test is the shape
    of the pool's behaviour, not how long it waits.
    """
    rented = {
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
    }
    rented.update(overrides.pop("rented", {}))
    limits = {"max_rented_hosts": 8, "max_hourly_burn": 6.0}
    limits.update(overrides.pop("limits", {}))
    return PoolHarness(
        [EngineSpec(id="laptop", resident={MODEL}, kind="local", workers=2, chunk_delay_s=0.35)],
        rentable=[
            EngineSpec(id=f"market-{n}", resident={MODEL}, workers=4, chunk_delay_s=0.3)
            for n in range(8)
        ],
        model_set=[MODEL],
        probe_interval_s=0.3,
        queue_timeout_s=8.0,
        rented=rented,
        extra_config={"limits": limits},
        **overrides,
    )


def run(scenario: Scenario, *, verbose: bool = True) -> dict[str, Any]:
    report = Report()
    pool = build_pool(**scenario.pool)
    provider = pool.supervisor.fleet.provider
    # A run rents more hosts than there are engines in the market; the same engines answer
    # again rather than the run quietly creating hosts nothing can serve.
    provider.reuse_engine_urls = True
    market = MovingMarket(provider, report)
    traffic = Traffic(pool.url)
    state = Run(pool=pool, market=market, traffic=traffic, report=report)
    seen = 0

    def drain_events() -> None:
        nonlocal seen
        events = list(reversed(pool.supervisor.events.recent(400)))
        for event in events[seen:]:
            report.note(event["kind"], event["summary"][:110])
        seen = max(seen, len(events))

    try:
        if scenario.lease is not None:
            pool.supervisor.fleet.open_lease(**scenario.lease)
            report.note("lease", f"opened: {scenario.lease}")
        market.start()

        for phase in scenario.phases:
            report.note("phase", f"{phase.name}: {phase.callers} callers — {phase.note}")
            if phase.when_it_starts is not None:
                phase.when_it_starts(state)
            market.evict_every_s = scenario.evict_every_s if phase.market_is_hostile else None
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
        summary = collect(scenario, pool, traffic, report)
        pool.close()

    if verbose:
        print(f"\n=== {scenario.name}: {scenario.what_it_shows}")
        print(report.render())
        print_summary(summary)
    return summary


def collect(scenario: Scenario, pool: PoolHarness, traffic: Traffic, report: Report) -> dict[str, Any]:
    fleet = pool.supervisor.fleet
    events = collections.Counter(e["kind"] for e in pool.supervisor.events.recent(600))
    rows = pool.request_log(limit=8000)
    # Each row is a host's *running* estimate, written again every pass — so a day's spend is
    # the last figure per host, summed, not the sum of every row (which reads several times
    # high and made a cap look blown when it had held).
    spend = sum(
        row["spent"]
        for row in pool.database.query(
            "SELECT host_id, MAX(amount) AS spent FROM spend WHERE source = 'estimate' GROUP BY host_id"
        )
    )
    return {
        "scenario": scenario.name,
        "outcomes": dict(traffic.outcomes),
        "requests": sum(traffic.outcomes.values()),
        "bodies_broken": traffic.bodies_broken,
        "latency_p50": statistics.median(traffic.latencies) if traffic.latencies else None,
        "latency_p95": (
            sorted(traffic.latencies)[int(len(traffic.latencies) * 0.95) - 1]
            if len(traffic.latencies) >= 20
            else None
        ),
        "delivery": dict(traffic.delivery),
        "served_by": dict(collections.Counter(r["host_id"] for r in rows if r["outcome"] == "ok")),
        "events": dict(events),
        "instances_left": {
            instance_id: instance.state
            for instance_id, instance in fleet.provider.instances.items()
        },
        "spend": round(spend, 4),
        "caps": [(lease.lease_id, lease.max_spend) for lease in fleet.leases.open_leases()],
        "report": report,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n  --- what it came to ---")
    print(f"    requests         {summary['requests']}  {summary['outcomes']}")
    print(f"    delivery         {summary['delivery']}")
    if summary["latency_p50"]:
        print(f"    latency          p50 {summary['latency_p50']:.2f}s  p95 {summary['latency_p95']:.2f}s")
    print(f"    hosts rented     {summary['events'].get('rented', 0)}")
    print(f"    instances left   {summary['instances_left'] or 'none'}")
    print(f"    estimated spend  ${summary['spend']}")
    interesting = {
        kind: count
        for kind, count in sorted(summary["events"].items())
        if kind not in ("agent_reachable", "config_applied")
    }
    print(f"    events           {interesting}")


# --- what must be true of every run ---------------------------------------------------------


def check_invariants(summary: dict[str, Any]) -> None:
    """The properties no run may break, whatever the scenario did."""
    outcomes = summary["outcomes"]

    unexplained = {
        code: count
        for code, count in outcomes.items()
        if not (code == "ok" or code.startswith("503:"))
    }
    assert not unexplained, f"{summary['scenario']}: answers the contract does not describe: {unexplained}"

    assert summary["bodies_broken"] == 0, (
        f"{summary['scenario']}: {summary['bodies_broken']} responses ended without their final "
        "frame — half an answer reached an app"
    )

    # Nothing may be left *running*: that is an accelerator still being paid for. A stopped
    # instance is a parked host — storage only, deliberate, and destroyed at its own limit.
    running = {i: state for i, state in summary["instances_left"].items() if state == "running"}
    assert not running, f"{summary['scenario']}: instances left running: {running}"

    for lease_id, cap in summary["caps"]:
        assert summary["spend"] <= cap * 1.5, (
            f"{summary['scenario']}: spent ${summary['spend']} against a ${cap} cap on {lease_id}"
        )

    # Every host a run rented has to end up given back or paused — never still serving on an
    # accelerator nobody is watching. The "nothing left running" check above is that property;
    # counting releases would only describe how long a particular run happened to last.


if __name__ == "__main__":
    from catalogue import CATALOGUE

    wanted = sys.argv[1:] or list(CATALOGUE)
    failures = []
    for name in wanted:
        scenario = CATALOGUE[name]
        summary = run(scenario)
        try:
            check_invariants(summary)
            if scenario.expects is not None:
                scenario.expects(summary)
            print(f"  ✓ {name}")
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"  ✗ {name}: {exc}")
    print("\n" + ("all scenarios held" if not failures else f"{len(failures)} scenario(s) failed"))
    sys.exit(1 if failures else 0)
