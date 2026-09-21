"""Each host finding its own worker count, as a pure function (D67, D68).

Measured on the first live pool: latency flat from three to six requests in flight on every
host, so six was never the ceiling — and two machines sharing the best one's hardware label
served at a quarter of its throughput, where six workers only meant six slow requests.
"""

from gpm_server.config import WorkersAutoConfig
from gpm_server.strategies import WorkerReading, decide_workers

CFG = WorkersAutoConfig(enabled=True)


def reading(**kwargs):
    base = dict(host_id="h1", workers=6, launch_workers=16, busy_workers=6, waiting=4)
    base.update(kwargs)
    return WorkerReading(**base)


# --- up, on evidence ---


def test_a_saturated_host_with_requests_waiting_takes_one_more():
    decision = decide_workers(reading(), CFG)
    assert decision.workers == 7 and decision.changed


def test_six_is_where_a_host_starts_not_where_it_stops():
    """The owner, correcting D67: six is the default, not the cap."""
    climbing = reading(workers=6)
    for expected in (7, 8, 9):
        assert decide_workers(climbing, CFG).workers == expected
        climbing = reading(workers=expected, busy_workers=expected)


def test_it_never_climbs_past_what_the_engine_was_launched_to_run():
    """Above that, a worker is a queue slot pretending to be capacity."""
    decision = decide_workers(reading(workers=16, launch_workers=16, busy_workers=16), CFG)
    assert decision.workers == 16 and not decision.changed


def test_a_host_nobody_is_queuing_for_is_left_alone():
    assert not decide_workers(reading(waiting=0), CFG).changed


def test_a_host_that_is_not_even_full_is_left_alone():
    assert not decide_workers(reading(busy_workers=3), CFG).changed


def test_a_step_up_that_barely_moved_the_needle_is_taken_back():
    """Two per cent on a step up is flat, and flat is the evidence that it did not help."""
    decision = decide_workers(reading(last_change=1, throughput=100.0, throughput_before=98.0), CFG)
    assert decision.workers == 5 and "did not pay" in decision.reasons[0]


# --- down, on evidence ---


def test_a_step_up_that_did_not_pay_is_taken_back():
    decision = decide_workers(
        reading(last_change=1, throughput=100.0, throughput_before=140.0), CFG
    )
    assert decision.workers == 5
    assert "did not pay" in decision.reasons[0]


def test_a_host_whose_engine_ran_out_of_memory_steps_down_at_once():
    decision = decide_workers(reading(evicted_a_model=True), CFG)
    assert decision.workers == 5
    assert "left memory" in decision.reasons[0]


def test_a_host_far_slower_than_the_pool_for_the_same_model_steps_down():
    """The live case: two machines under one hardware label, three and a half times apart."""
    decision = decide_workers(reading(service_s=20.4, pool_service_s=5.8), CFG)
    assert decision.workers == 5
    assert "over 2x the pool's" in decision.reasons[0]


def test_a_host_already_at_one_worker_is_neither_lowered_nor_raised():
    """It cannot be given less, and a struggling host is never also a candidate for more —
    that it is not worth keeping is the tear-down's decision, not this one."""
    decision = decide_workers(reading(workers=1, evicted_a_model=True), CFG)
    assert decision.workers == 1 and not decision.changed


def test_a_slow_host_is_judged_against_the_pool_not_against_a_number():
    """A pool where everything is slow has no slow host — only slower hardware."""
    assert not decide_workers(reading(service_s=20.0, pool_service_s=18.0, waiting=0), CFG).changed


# --- it says why, always ---


def test_every_change_carries_the_numbers_behind_it():
    for case in (
        reading(),
        reading(evicted_a_model=True),
        reading(service_s=20.0, pool_service_s=5.0),
        reading(last_change=1, throughput=10.0, throughput_before=99.0),
    ):
        decision = decide_workers(case, CFG)
        assert decision.changed and decision.reasons and any(c.isdigit() for c in decision.reasons[0])
