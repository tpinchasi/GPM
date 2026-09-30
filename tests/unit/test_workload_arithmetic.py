"""The arithmetic behind a workload (D115): sizing at a latency, the start, the budget, and the
rental-kind rule. Pure functions, so every case is a line."""

import pytest
from gpm_server import workloads as w


def test_names_are_plain():
    assert w.valid_name("research-run") and w.valid_name("a1")
    for bad in ("", "-x", "Upper", "has space", "a/b", "x" * 41):
        assert not w.valid_name(bad), bad


def test_p95_is_nearest_rank():
    assert w.p95(list(range(1, 101))) == 95
    assert w.p95([3.0]) == 3.0


def samples(concurrency, latency, n=w.MIN_SAMPLES):
    return [(concurrency, latency)] * n


def test_the_most_answers_at_once_that_met_the_target():
    got = w.workers_at_latency(samples(4, 6.0) + samples(8, 9.0) + samples(16, 25.0), target_s=10, ceiling=24)
    assert got.workers == 8 and got.measured
    assert got.curve == {4: 6.0, 8: 9.0, 16: 25.0}
    assert "8 at once measured 9s at p95, within 10s; 16 at once was 25s" in got.reasons[0]


def test_never_above_what_the_card_was_sized_for():
    assert w.workers_at_latency(samples(32, 5.0), target_s=10, ceiling=16).workers is None


def test_too_few_answers_is_not_a_measurement():
    got = w.workers_at_latency(samples(8, 5.0, n=w.MIN_SAMPLES - 1), target_s=10, ceiling=16)
    assert got.workers is None and not got.measured
    assert "unmeasured" in got.reasons[0]


def test_a_target_nothing_met_is_said():
    got = w.workers_at_latency(samples(4, 30.0), target_s=10, ceiling=16)
    assert got.workers is None and got.measured
    assert "will not be met on this class" in got.reasons[0]


def test_hosts_at_start():
    assert w.hosts_at_start(16, 8) == 2
    assert w.hosts_at_start(17, 8) == 3
    assert w.hosts_at_start(1, 8) == 1
    assert w.hosts_at_start(0, 8) == 0


def test_a_derived_budget_has_its_margin_and_is_rounded_up():
    assert w.derived_budget(2, 6, 1.0) == 15.0
    assert w.derived_budget(1, 1, 0.333) == 0.42


def test_time_to_ready_is_the_download_then_the_load():
    assert w.time_to_ready_hours(36, 800, load_s=180) == pytest.approx((36 * 8000 / 800 + 180) / 3600)
    assert w.time_to_ready_hours(36, 0) > w.time_to_ready_hours(36, 800), "no stated speed is a slow link"


def kind(interruptible, hourly, share, evictions=0.1, ready_h=0.25, lost=1.5, workers=8, hours=6, offer="o"):
    return w.expected_cost(
        offer_id=offer, machine_id=f"m-{offer}", interruptible=interruptible, hourly=hourly, hours=hours,
        workers=workers, evictions_per_hour=evictions, ready_hours=ready_h,
        lost_capacity_hourly=lost, share_of_workload=share,
    )


def test_on_demand_is_its_price_for_the_hours():
    cost = kind(False, 1.5, share=1.0)
    assert cost.expected == 9.0 and "nothing can take it away" in cost.reasons[0]


def test_a_lone_bid_pays_for_the_whole_workload_it_takes_down():
    lone = kind(True, 1.2, share=1.0)
    quarter = kind(True, 1.2, share=0.25)
    assert lone.expected > quarter.expected > 1.2 * 6


def test_the_first_host_on_demand_and_the_scale_up_as_bids_is_what_the_numbers_give():
    """The owner's rule of thumb, as an outcome rather than a rule: a lone host with a bad
    eviction rate loses to on-demand; the fourth host of four, at the same rate, does not."""
    first, why = w.order_by_expected_cost([kind(False, 1.5, 1.0, offer="od"), kind(True, 1.2, 1.0, evictions=0.3, ready_h=0.4, offer="bid")])
    assert first[0].offer_id == "od" and why[0].startswith("rental kind: on demand on m-od")
    later, _ = w.order_by_expected_cost([kind(False, 1.5, 0.25, offer="od"), kind(True, 1.2, 0.25, evictions=0.3, ready_h=0.4, offer="bid")])
    assert later[0].offer_id == "bid"


def test_cost_is_compared_per_worker_hour():
    small, _ = w.order_by_expected_cost([kind(False, 1.0, 1.0, workers=4, offer="small"), kind(False, 1.5, 1.0, workers=8, offer="big")])
    assert small[0].offer_id == "big", "8 workers for 1.5 beats 4 for 1.0"


def test_nothing_to_compare_says_so():
    assert w.order_by_expected_cost([]) == ([], ["no candidate to compare"])


# --- several models on one host (D118) ---


def test_a_together_split_keeps_every_model_within_its_target():
    """The review's example: chat serves 8 alone, embeddings 32; 16 and 4 wanted at once."""
    split = w.split_together({"chat": 16, "embed": 4}, {"chat": 8, "embed": 32})
    assert split.hosts == 3 and split.caps == {"chat": 6, "embed": 2} and split.workers == 8
    assert split.load <= 1


def test_the_caps_are_shares_not_ceilings_each():
    """Caps of w_m each would let a host run 8 + 32 at once and miss both targets."""
    split = w.split_together({"chat": 8, "embed": 32}, {"chat": 8, "embed": 32})
    assert split.caps["chat"] / 8 + split.caps["embed"] / 32 <= 1
    assert split.hosts == 2


def test_no_split_where_one_answer_of_each_already_fills_the_card():
    assert w.split_together({"a": 1, "b": 1}, {"a": 1, "b": 2}) is None
    assert w.split_together({"a": 1}, {"a": 0}) is None


def test_one_model_is_the_single_model_sizing():
    split = w.split_together({"chat": 16}, {"chat": 6})
    assert split.hosts == w.hosts_at_start(16, 6) and split.caps == {"chat": 6}


def test_a_split_fits_a_card_only_within_its_share():
    assert w.split_fits({"chat": 6, "embed": 2}, {"chat": 8, "embed": 32})
    assert not w.split_fits({"chat": 6, "embed": 2}, {"chat": 6, "embed": 32})
    assert not w.split_fits({"chat": 6}, {})


def test_the_cheaper_placement_wins_and_a_tie_goes_together():
    together = w.Placement("together", 10.0, 2, [])
    apart = w.Placement("apart", 9.0, 3, [])
    chosen, why = w.choose_placement(together, apart)
    assert chosen is apart and why[0].startswith("apart:") and "$9.0000 apart" in why[0]
    chosen, _ = w.choose_placement(w.Placement("together", 9.0, 2, []), apart)
    assert chosen.name == "together"
    chosen, why = w.choose_placement(w.Placement("together", 9.05, 2, []), apart)
    assert chosen.name == "together" and "within 1%" in why[0], "a hair apart is equal"
    chosen, why = w.choose_placement(None, apart)
    assert chosen is apart and "no card" in why[0]
    assert w.choose_placement(None, None)[0] is None
