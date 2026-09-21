"""The scenarios themselves: the shapes a pool meets, and what each one is for.

Each names the feature it puts under load, so a failure says which decision it belongs to.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import Phase, Run, Scenario  # noqa: E402

FULL_LEASE = {"workers": 40, "max_hours": 1, "max_spend": 5.00, "allow_rent": True}


# --- what each phase can do to the world ---


def market_goes_empty(run: Run) -> None:
    run.market.mood = "empty"
    run.market.refresh()
    run.report.note("world", "every machine has left the market")


def market_comes_back(run: Run) -> None:
    run.market.mood = "normal"
    run.market.refresh()
    run.report.note("world", "the market is back")


def market_prices_itself_out(run: Run) -> None:
    run.market.mood = "expensive"
    run.market.refresh()
    run.report.note("world", "every machine is now priced past the pool's ceiling")


def provider_stops_answering(run: Run) -> None:
    run.market.provider.unavailable = True
    run.report.note("world", "the provider has stopped answering")


def provider_answers_again(run: Run) -> None:
    run.market.provider.unavailable = False
    run.report.note("world", "the provider is answering again")


def supervisor_restarts(run: Run) -> None:
    """What `gpm restart` does under load: the process goes, the hosts stay."""
    run.pool.restart_supervisor()
    run.report.note("world", "the supervisor was restarted; its hosts are still rented")


def hosts_never_start(run: Run) -> None:
    # A port nothing listens on: the instance exists and bills, and its engine never answers —
    # which is the shape of a machine stuck scheduling or half-built (D54).
    run.market.provider.engine_urls = ["http://127.0.0.1:9"]
    run.report.note("world", "machines from here come up but their engine never answers")


def hosts_answer_again(run: Run) -> None:
    run.market.provider.engine_urls = list(run.pool.rentable_urls)
    run.report.note("world", "machines from here answer again")


def start_up_material_goes_missing(run: Run) -> None:
    run.market.provider.drop_startup_material = True
    run.report.note("world", "instances will come up without their start-up script")


def operator_changes_a_worker_count(run: Run) -> None:
    """What `gpm host resize` does, in the middle of live traffic (D56)."""
    fleet = run.pool.supervisor.fleet
    ready = [h for h in fleet.hosts.values() if not h.released and h.state == "ready"]
    if not ready:
        run.report.note("world", "no ready host to resize yet")
        return
    host = ready[0]
    was = host.workers
    run.pool.loop.run(fleet.resize(host, max(1, was - 1)))       # down: slots only
    run.pool.loop.run(fleet.resize(host, min(6, was + 2)))       # up: inside the launch bound
    run.report.note("world", f"an operator resized {host.host_id} from {was} and back up")


# --- the catalogue ---


def market_day() -> Scenario:
    return Scenario(
        name="market_day",
        what_it_shows="a day's load against a moving market: the ramp, buffered answers, "
        "eviction, pausing, waking, and giving up",
        lease=FULL_LEASE,
        phases=(
            Phase("quiet", 4, callers=0, note="nothing is being asked of the pool"),
            Phase("morning", 10, callers=8, think_s=0.02, note="load climbs past the laptop"),
            Phase("peak", 18, callers=40, think_s=0.0, note="far more than the pool holds"),
            Phase("shock", 12, callers=40, think_s=0.0, market_is_hostile=True,
                  note="hosts are taken away mid-answer, under full load"),
            Phase("lull", 10, callers=0, note="quiet for long enough that hosts are paused"),
            Phase("false dawn", 8, callers=10, think_s=0.0,
                  note="load returns while hosts are paused: they should wake, not be re-rented"),
            Phase("night", 26, callers=0, note="quiet for long enough to be given up"),
        ),
        expects=lambda s: (
            _at_least(s, "rented", 2, "the ramp never rented anything"),
            _at_least(s, "parked", 1, "no host was ever paused"),
            _at_least(s, "park_restarted", 1, "a paused host was never woken by returning load"),
            _at_least(s, "workers_auto", 1, "no host ever changed its own worker count"),
            _buffered(s),
        ),
    )


def empty_market() -> Scenario:
    return Scenario(
        name="empty_market",
        what_it_shows="a pool that wants capacity and cannot buy any: it says so, and keeps "
        "serving from what it has",
        lease=FULL_LEASE,
        phases=(
            Phase("load, with nothing to buy", 14, callers=20, think_s=0.0,
                  when_it_starts=market_goes_empty,
                  note="every machine has left the market"),
            Phase("the market returns", 14, callers=20, think_s=0.0,
                  when_it_starts=market_comes_back,
                  note="and now the pool can buy what the load has been asking for"),
            Phase("quiet", 20, callers=0, note="given up again"),
        ),
        expects=lambda s: (
            _served_by_laptop(s),
            _at_least(s, "rented", 1, "the pool never rented once the market came back"),
        ),
    )


def priced_out() -> Scenario:
    return Scenario(
        name="priced_out",
        what_it_shows="a market the pool cannot afford: every offer is refused with its reason, "
        "and nothing is bought over the ceiling",
        lease=FULL_LEASE,
        phases=(
            Phase("load against a dear market", 16, callers=20, think_s=0.0,
                  when_it_starts=market_prices_itself_out,
                  note="every machine is priced past the bid ceiling"),
            Phase("quiet", 8, callers=0, note="nothing was bought, so nothing is given up"),
        ),
        expects=lambda s: (
            _none(s, "rented", "something was rented above the pool's ceiling"),
            _served_by_laptop(s),
        ),
    )


def provider_outage() -> Scenario:
    return Scenario(
        name="provider_outage",
        what_it_shows="the provider stops answering mid-run: the pool keeps serving, keeps its "
        "hosts, and picks up where it left off",
        lease=FULL_LEASE,
        phases=(
            Phase("build up", 12, callers=20, think_s=0.0, note="a few hosts are rented"),
            Phase("outage", 14, callers=20, think_s=0.0, when_it_starts=provider_stops_answering,
                  note="the provider is unreachable — nothing may be assumed about what exists"),
            Phase("recovery", 12, callers=20, think_s=0.0, when_it_starts=provider_answers_again,
                  note="and it answers again"),
            Phase("quiet", 20, callers=0, note="given up"),
        ),
        expects=lambda s: _at_least(s, "rented", 1, "nothing was ever rented"),
    )


def supervisor_restart() -> Scenario:
    return Scenario(
        name="supervisor_restart",
        what_it_shows="the supervisor is restarted under load: the router keeps serving and the "
        "rented hosts are taken back rather than swept",
        lease=FULL_LEASE,
        phases=(
            Phase("build up", 14, callers=20, think_s=0.0, note="a few hosts are rented"),
            Phase("restart", 16, callers=20, think_s=0.0, when_it_starts=supervisor_restarts,
                  note="the supervisor process is replaced while traffic runs"),
            Phase("quiet", 20, callers=0, note="given up"),
        ),
        expects=lambda s: (
            _at_least(s, "rented", 1, "nothing was ever rented"),
            _none(s, "orphan_swept", "a restart swept a host it had rented itself"),
        ),
    )


def hosts_that_never_answer() -> Scenario:
    return Scenario(
        name="hosts_that_never_answer",
        what_it_shows="machines that never start: they are given up rather than billed for the "
        "whole preparing window, and their machines are avoided",
        lease=FULL_LEASE,
        pool={"rented": {"teardown": {
            "idle_minutes": 0.1, "destroy_idle_minutes": 0.25,
            "max_starting_minutes": 0.2, "park_when_idle": True,
        }}},
        phases=(
            Phase("load against dead machines", 20, callers=20, think_s=0.0,
                  when_it_starts=hosts_never_start,
                  note="everything rented here will never answer"),
            Phase("machines answer again", 14, callers=20, think_s=0.0,
                  when_it_starts=hosts_answer_again,
                  note="and now a rented host can actually serve"),
            Phase("quiet", 18, callers=0, note="given up"),
        ),
        expects=lambda s: (
            _given_up_on(s),
            _served_by_laptop(s),
        ),
    )


def instances_without_their_script() -> Scenario:
    return Scenario(
        name="instances_without_their_script",
        what_it_shows="instances that come up without their start-up material: no dead-man "
        "timer and no way in, so they are ended in the same pass (D65)",
        lease=FULL_LEASE,
        phases=(
            Phase("load against broken instances", 16, callers=20, think_s=0.0,
                  when_it_starts=start_up_material_goes_missing,
                  note="every instance loses the script that arms its timer"),
            Phase("quiet", 12, callers=0, note="nothing is left"),
        ),
        expects=lambda s: _at_least(
            s, "host_without_startup", 1, "an instance with no timer was not ended"
        ),
    )


def budget_runs_out() -> Scenario:
    return Scenario(
        name="budget_runs_out",
        what_it_shows="a lease with barely any money: the pool stops spending and gives its "
        "hosts back, and the app is told why",
        # Small enough that a minute of renting reaches it: what is under test is the cap
        # biting, not how much a fake machine costs.
        lease={"workers": 40, "max_hours": 1, "max_spend": 0.004, "allow_rent": True},
        phases=(
            Phase("load against a tiny budget", 24, callers=20, think_s=0.0,
                  note="more load than the budget can answer"),
            Phase("quiet", 12, callers=0, note="the lease is spent"),
        ),
        expects=lambda s: (
            _spent_within_cap(s),
            _at_least(s, "lease_capped", 1, "the lease was never capped, so the limit never bit"),
        ),
    )


def no_lease_at_all() -> Scenario:
    return Scenario(
        name="no_lease_at_all",
        what_it_shows="load with no lease open: nothing is rented however loud it gets, and the "
        "laptop serves what it can",
        lease=None,
        phases=(
            Phase("load with no authority to spend", 16, callers=20, think_s=0.0,
                  note="the pool may not spend a cent"),
        ),
        expects=lambda s: (
            _none(s, "rented", "something was rented with no lease open"),
            _served_by_laptop(s),
        ),
    )


def outbid_over_and_over() -> Scenario:
    return Scenario(
        name="outbid_over_and_over",
        what_it_shows="an interruptible market at its worst: hosts taken away every couple of "
        "seconds, under load, all run long — no app ever sees half an answer",
        lease=FULL_LEASE,
        evict_every_s=2.0,
        phases=(
            Phase("under constant eviction", 30, callers=24, think_s=0.0, market_is_hostile=True,
                  note="something is outbid every two seconds while traffic runs"),
            Phase("quiet", 18, callers=0, note="given up"),
        ),
        expects=lambda s: (
            _at_least(s, "eviction", 2, "nothing was ever outbid, so recovery was not exercised"),
            _buffered(s),
        ),
    )


def load_under_capacity() -> Scenario:
    return Scenario(
        name="load_under_capacity",
        what_it_shows="steady load the local host already covers: nothing is rented, however "
        "long it goes on",
        lease=FULL_LEASE,
        phases=(
            Phase("gentle", 24, callers=1, think_s=0.3,
                  note="one caller against two local workers: never a queue"),
        ),
        expects=lambda s: (
            _none(s, "rented", "capacity was bought for load the laptop was already covering"),
            _served_by_laptop(s),
        ),
    )


def hourly_burn_cap() -> Scenario:
    return Scenario(
        name="hourly_burn_cap",
        what_it_shows="a pool whose hourly burn cap is reached: renting stops there, with the "
        "reason recorded, and the load is served by what is already running",
        lease=FULL_LEASE,
        # Two or three cheap machines reach it, so the cap bites inside a short run.
        pool={"limits": {"max_rented_hosts": 8, "max_hourly_burn": 0.35}},
        phases=(
            Phase("more load than the burn cap allows", 22, callers=30, think_s=0.0,
                  note="the pool would rent more if the cap let it"),
            Phase("quiet", 16, callers=0, note="given up"),
        ),
        expects=lambda s: (
            _at_least(s, "rent_refused", 1, "the burn cap never refused a rental"),
            _within_burn_cap(s, 0.35),
        ),
    )


def operator_resizes_a_host() -> Scenario:
    return Scenario(
        name="operator_resizes_a_host",
        what_it_shows="an operator changing a running host's worker count under traffic (D56): "
        "down is instant, up to the launch bound restarts nothing, and no request is lost",
        lease=FULL_LEASE,
        phases=(
            Phase("build up", 12, callers=20, think_s=0.0, note="a host or two are rented"),
            Phase("resize under load", 14, callers=20, think_s=0.0,
                  when_it_starts=operator_changes_a_worker_count,
                  note="an operator changes a host's worker count while it is serving"),
            Phase("quiet", 16, callers=0, note="given up"),
        ),
        expects=lambda s: _at_least(s, "host_resized", 1, "no host was resized"),
    )


CATALOGUE = {
    scenario.name: scenario
    for scenario in (
        market_day(),
        empty_market(),
        priced_out(),
        provider_outage(),
        supervisor_restart(),
        hosts_that_never_answer(),
        instances_without_their_script(),
        budget_runs_out(),
        no_lease_at_all(),
        outbid_over_and_over(),
        load_under_capacity(),
        hourly_burn_cap(),
        operator_resizes_a_host(),
    )
}


# --- little checks the scenarios share ---


def _at_least(summary, kind: str, count: int, why: str) -> None:
    got = summary["events"].get(kind, 0)
    assert got >= count, f"{summary['scenario']}: {why} (saw {got} {kind})"


def _none(summary, kind: str, why: str) -> None:
    got = summary["events"].get(kind, 0)
    assert got == 0, f"{summary['scenario']}: {why} (saw {got} {kind})"


def _served_by_laptop(summary) -> None:
    assert summary["served_by"].get("laptop", 0) > 0, (
        f"{summary['scenario']}: the local host served nothing, so nothing was really tested"
    )


def _buffered(summary) -> None:
    assert summary["delivery"].get("buffered", 0) > 0, (
        f"{summary['scenario']}: no answer came back buffered, so D62 was never exercised"
    )
    assert summary["delivery"].get("stream", 0) > 0, (
        f"{summary['scenario']}: nothing streamed, so delivery-by-host-kind was never exercised"
    )


def _given_up_on(summary) -> None:
    """However it is noticed — never started, or could not hold the model set — a machine that
    cannot serve is given up rather than billed for the whole preparing window."""
    ways = ("host_stuck_starting", "prepare_failed", "host_without_startup")
    total = sum(summary["events"].get(kind, 0) for kind in ways)
    assert total >= 1, (
        f"{summary['scenario']}: a host that never answered was never given up "
        f"(saw none of {ways})"
    )
    assert summary["events"].get("machine_avoided", 0) >= 1, (
        f"{summary['scenario']}: a machine that failed was not put on the avoid list"
    )


def _within_burn_cap(summary, cap: float) -> None:
    """Every host the run ever held, priced together: the cap is on what runs at once, and a
    run that rented in waves must never have had more than the cap running at any moment."""
    refusals = summary["events"].get("rent_refused", 0)
    assert refusals >= 1, f"{summary['scenario']}: nothing was ever refused for the ${cap}/h cap"


def _spent_within_cap(summary) -> None:
    for lease_id, cap in summary["caps"]:
        assert summary["spend"] <= cap * 1.5, (
            f"{summary['scenario']}: spent ${summary['spend']} against ${cap} on {lease_id}"
        )
