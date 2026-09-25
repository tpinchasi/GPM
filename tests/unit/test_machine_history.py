"""What each machine has done for this pool, folded out of the logs it already writes (D69).

The evidence this exists for, from the first live pool: one machine rented seven times having
reached ready twice, and two machines under one hardware label two and a half times apart in
service time. None of that is in an offer.
"""

import pytest
from gpm_server import history
from gpm_server.config import MachineHistoryConfig

CFG = MachineHistoryConfig(enabled=True, min_rentals=2, min_reliability=0.5)


def rented(host_id, machine, ts=0.0, hardware="1x Card", bid=0.2, in_the_sentence=False):
    """A rental as the decision log holds it — as data, or as older logs hold it: in prose."""
    if in_the_sentence:
        return {
            "kind": "rented", "host_id": host_id, "ts": ts,
            "summary": f"bid ${bid:.3f}/h on {machine} ({hardware}), 2 workers",
            "numbers": {"bid": bid},
        }
    return {
        "kind": "rented", "host_id": host_id, "ts": ts, "summary": "bid",
        "numbers": {"machine": machine, "hardware": hardware, "bid": bid},
    }


def event(kind, host_id, ts=0.0):
    return {"kind": kind, "host_id": host_id, "ts": ts, "summary": kind, "numbers": {}}


def request(host_id, latency_ms=1000.0, tokens=None, generate_ms=None, outcome="ok"):
    return {
        "host_id": host_id, "outcome": outcome, "latency_ms": latency_ms,
        "tokens_out": tokens, "generate_ms": generate_ms,
    }


# --- what the logs say about a machine ---


def test_a_machine_that_keeps_being_rented_and_never_serves_is_visible():
    """The live case: seven rentals, two of which reached ready."""
    events = []
    for n in range(7):
        events.append(rented(f"h{n}", "m-bad", ts=n * 600))
        if n < 2:
            events.append(event("prepared", f"h{n}", ts=n * 600 + 120))
        else:
            events.append(event("prepare_failed", f"h{n}", ts=n * 600 + 90))

    records = history.build(events, [])

    bad = records["m-bad"]
    assert bad.rentals == 7 and bad.reached_ready == 2 and bad.failures == 5
    assert bad.reliability == pytest.approx(2 / 7)
    assert bad.minutes_to_ready == pytest.approx(2.0)


def test_two_machines_under_one_label_are_told_apart():
    """What no offer field distinguishes: how they actually served."""
    events = [
        rented("fast-1", "m-fast", hardware="1x Same Label"),
        event("prepared", "fast-1"),
        rented("slow-1", "m-slow", hardware="1x Same Label"),
        event("prepared", "slow-1"),
    ]
    requests = [request("fast-1", latency_ms=3800)] * 5 + [request("slow-1", latency_ms=9600)] * 5

    records = history.build(events, requests)

    assert records["m-fast"].hardware == records["m-slow"].hardware
    assert records["m-fast"].service_s == 3.8 and records["m-slow"].service_s == 9.6


def test_throughput_comes_from_what_the_engine_reported():
    events = [rented("h1", "m-1"), event("prepared", "h1")]
    requests = [request("h1", tokens=120, generate_ms=1000), request("h1", tokens=80, generate_ms=1000)]

    assert history.build(events, requests)["m-1"].tokens_per_s == 100.0


def test_an_older_log_that_named_the_machine_only_in_the_sentence_is_still_read():
    """The 62 rentals already written before the event carried it as a number."""
    events = [
        rented("h1", "143822", hardware="1x RTX PRO 6000 Max-Q", in_the_sentence=True),
        event("prepared", "h1"),
    ]

    records = history.build(events, [])

    assert records["143822"].hardware == "1x RTX PRO 6000 Max-Q"
    assert records["143822"].reached_ready == 1


def test_requests_from_a_host_the_log_never_saw_rented_are_ignored():
    """A log that does not reach back far enough says nothing, rather than guessing."""
    assert history.build([], [request("h-unknown")]) == {}


# --- what a record does to an offer's score ---


def test_a_machine_nobody_has_tried_is_not_penalised():
    """An unknown machine and a bad one are different things."""
    factor, why = history.adjustment(None, CFG)
    assert factor == 1.0 and why is None

    once = history.MachineRecord(machine_id="m", hardware="x", rentals=1, reached_ready=0)
    assert history.adjustment(once, CFG)[0] == 1.0, "one bad rental is not a record"


def test_a_machine_that_rarely_serves_scores_worse_and_says_why():
    record = history.MachineRecord(
        machine_id="m", hardware="x", rentals=7, reached_ready=2, failures=5
    )

    factor, why = history.adjustment(record, CFG)

    assert factor == CFG.penalty
    assert "rented 7x, reached ready 2x" in why


def test_a_machine_that_has_served_well_scores_better():
    cfg = MachineHistoryConfig(good_tokens_per_s=100, poor_tokens_per_s=20)
    good = history.MachineRecord(
        machine_id="m", hardware="x", rentals=3, reached_ready=3, tokens_per_s=125.0
    )
    poor = history.MachineRecord(
        machine_id="m", hardware="x", rentals=3, reached_ready=3, tokens_per_s=9.0
    )

    assert history.adjustment(good, cfg)[0] == cfg.bonus
    assert history.adjustment(poor, cfg)[0] == cfg.penalty
    assert "9 tokens/s" in history.adjustment(poor, cfg)[1]


def test_the_whole_thing_can_be_switched_off():
    record = history.MachineRecord(machine_id="m", hardware="x", rentals=7, reached_ready=0)
    assert history.adjustment(record, MachineHistoryConfig(enabled=False))[0] == 1.0


# --- and what that does to a bid ---


def test_a_machine_with_a_bad_record_loses_to_an_equal_one_without_it():
    """The point of the whole thing: the pool stops going back to a machine that wasted its
    money, without anyone writing a rule about that machine."""
    from gpm_server.config import BiddingConfig, OfferPolicy
    from gpm_server.providers import default_offer
    from gpm_server.strategies import history_note, rank_offers

    offers = [
        default_offer(offer_id="o-bad", machine_id="m-bad", min_bid_hourly=0.10),
        default_offer(offer_id="o-unknown", machine_id="m-unknown", min_bid_hourly=0.10),
    ]
    records = {
        "m-bad": history.MachineRecord(
            machine_id="m-bad", hardware="x", rentals=7, reached_ready=2, failures=5
        )
    }
    policy, bidding = OfferPolicy(min_disk_gb=10, max_all_in_hourly=1.0), BiddingConfig()

    blind, _ = rank_offers(offers, policy, bidding, hours=2, model_set_gb=1)
    assert blind[0][0].offer_id == "o-bad", "identical offers, so the tie breaks on id"

    informed, _ = rank_offers(
        offers, policy, bidding, hours=2, model_set_gb=1, history=records, history_cfg=CFG
    )
    assert informed[0][0].machine_id == "m-unknown", "the machine with the bad record lost"
    assert "reached ready 2x" in history_note(offers[0], records, CFG)
    assert history_note(offers[1], records, CFG) is None


def test_a_record_never_admits_an_offer_the_hard_filters_rejected():
    """It changes the order. It cannot widen what is acceptable (D69)."""
    from gpm_server.config import BiddingConfig, OfferPolicy
    from gpm_server.providers import default_offer
    from gpm_server.strategies import rank_offers

    too_small = default_offer(offer_id="o-1", machine_id="m-great", gpu_memory_gb=1.0)
    records = {
        "m-great": history.MachineRecord(
            machine_id="m-great", hardware="x", rentals=9, reached_ready=9, tokens_per_s=900.0
        )
    }

    accepted, rejected = rank_offers(
        [too_small],
        OfferPolicy(min_gpu_memory_gb=40, min_disk_gb=10, max_all_in_hourly=1.0),
        BiddingConfig(),
        hours=2,
        model_set_gb=1,
        history=records,
        history_cfg=MachineHistoryConfig(good_tokens_per_s=100),
    )

    assert not accepted and "o-1" in rejected
