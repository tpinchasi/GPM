"""Hosts added from measured load, in rounds that grow while it lasts (D66).

Measured on the first live pool: a 12 s mean queue wait and 4,138 requests refused with
`queue_timeout` in one saturated window — answered by the owner preparing hosts by hand, one at
a time, which took over half an hour to reach six.
"""

from gpm_server.config import DynamicAllocationConfig
from gpm_server.strategies import Load, decide_ramp

CFG = DynamicAllocationConfig()


def ramp(**kwargs):
    base = dict(
        round_size=0,
        load_present=True,
        previous_round_landed=True,
        since_last_round_s=10_000.0,
        hosts_pending=0,
        cfg=CFG,
    )
    base.update(kwargs)
    return decide_ramp(**base)


# --- what the load says ---


def test_a_queue_is_load_and_so_is_a_pool_held_at_full():
    """A client that sizes itself to the capacity it can see never builds a queue, so
    saturation is watched beside the queue rather than instead of it."""
    queued = Load(busy_workers=4, ready_workers=9, waiting=7)
    full = Load(busy_workers=9, ready_workers=9, waiting=0)
    quiet = Load(busy_workers=2, ready_workers=9, waiting=0)

    assert queued.present and full.present and not quiet.present
    assert full.saturated and not queued.saturated


def test_the_pool_is_sized_to_sit_below_full_not_at_it():
    """Rent before saturation: a pool held at exactly full is a pool that queues."""
    load = Load(busy_workers=9, ready_workers=9, waiting=6)
    assert load.wanted(0.75) == 20  # (9 + 6) / 0.75
    assert load.wanted(1.0) == 15


def test_an_empty_pool_is_not_called_saturated():
    assert not Load(busy_workers=0, ready_workers=0, waiting=0).present


# --- the ramp ---


def test_the_first_round_is_one_host():
    """A bad market yields one failed bid, not five."""
    assert ramp().hosts == 1


def test_each_round_asks_for_a_multiple_of_the_last():
    assert ramp(round_size=1).hosts == 2
    assert ramp(round_size=2).hosts == 4
    assert ramp(round_size=4).hosts == 8


def test_a_round_never_grows_past_what_the_operator_allows():
    assert ramp(round_size=8).hosts == CFG.max_round
    assert ramp(round_size=64).hosts == CFG.max_round


def test_a_round_waits_for_the_last_one_to_land():
    """The difference from multiplying on a timer: a host takes minutes to become ready, and
    doubling before it has helped buys capacity the last round was about to supply."""
    decision = ramp(round_size=1, previous_round_landed=False, hosts_pending=1)
    assert decision.hosts == 0 and "still coming up" in decision.reasons[0]


def test_and_then_waits_out_the_backoff():
    """Landing is not enough on its own: the pool gives the new capacity a chance to work
    before deciding it was not enough."""
    decision = ramp(round_size=2, since_last_round_s=10.0)
    assert decision.hosts == 0
    assert "inside the" in decision.reasons[0] and "waits before growing" in decision.reasons[0]


def test_the_ramp_stops_the_moment_the_load_does():
    decision = ramp(round_size=4, load_present=False)
    assert decision.hosts == 0 and "gone" in decision.reasons[0]


def test_every_round_says_what_it_is_doing_and_why():
    for case in (ramp(), ramp(round_size=2), ramp(load_present=False), ramp(round_size=1, hosts_pending=1, previous_round_landed=False)):
        assert case.reasons and case.reasons[0]
