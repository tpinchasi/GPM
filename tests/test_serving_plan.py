"""A workload's serving plan (S19, D198–D204), stage 1: the plan read and refused in words, today's
fields as the plan they mean, and how a request may be routed by one.

The routing here is the pure part — `serving_plan.route` and the dispatcher's ranking — tested
without a router or a clock. The router, the supervisor and the SDK carrying a plan are in
test_serving_plan_routing.py. Nothing here spends money or needs a GPU.
"""

import itertools

import httpx
import pytest
from gpm_server.catalog import ResolvedVariant
from gpm_server.models import Host, HostState, Worker, WorkerState
from gpm_server.provisioning_store import Grant
from gpm_server.router.dispatch import Dispatcher, Need
from gpm_server.serving_plan import (
    PlanRefused,
    ServingPlan,
    compile_policy,
    describe,
    from_bursts,
    parse_serve,
    plan_of,
    route,
)
from gpm_server.supervisor.workloads import WorkloadRefused, parse_policy

HOSTED = {"after_s": 5, "budget": {"per_day": 20}}
RENTED = {"after_min": 10, "max_hosts": 4, "budget": {"per_day": 60}}


# --- reading a plan ---


def test_the_owners_stop_gap_is_one_line():
    plan = parse_serve({"until_ready": ["shared", {"hosted": {"budget": {"per_day": 10}}}]})
    assert plan.steady == "rented"
    assert [s.type for s in plan.until_ready] == ["shared", "hosted"]
    assert plan.during == ("start", "replacement"), "start and replacements by default; scale-ups only when asked"
    assert plan.max_min == 60
    assert plan.until_ready_hosted().after_s == 0, "a stop-gap's source has no wait unless it names one"
    assert plan.when_full == (), "a role left out is empty"


def test_either_role_may_list_a_source_the_other_defines():
    plan = parse_serve({"until_ready": ["shared", "burst"],
                        "when_full": [{"name": "burst", "hosted": {"after_s": 2, "budget": {"per_day": 5}}}]})
    assert plan.until_ready_hosted() is plan.when_full_hosted() and plan.until_ready_hosted().after_s == 2
    with pytest.raises(PlanRefused, match="define it once"):
        parse_serve({"until_ready": [{"name": "x", "local": None}], "when_full": [{"name": "x", "hosted": HOSTED}]})


def test_a_named_source_is_one_source_in_both_roles_with_one_budget():
    plan = parse_serve({
        "until_ready": ["local", {"name": "cover", "hosted": {"after_s": 0, "budget": {"per_day": 10}}}],
        "when_full": ["cover", {"rented": RENTED}],
    })
    assert plan.until_ready_hosted() is plan.when_full_hosted()
    policy = compile_policy(plan)
    assert policy["hosted"]["budget"] == {"per_day": 10.0}
    assert policy["rented"]["max_hosts"] == 4
    assert parse_serve(policy["serve"]) == plan, "what is stored reads back as the same plan"


@pytest.mark.parametrize("serve, says", [
    ("shared", "serve is an object"),
    ({"until_ready": "local"}, "until_ready is a list"),
    ({"until_ready": ["rented"]}, "rented machines are what until_ready covers for"),
    ({"when_full": ["local"]}, "borrowing the pool's machines ends"),
    ({"when_full": ["shared"]}, "borrowing the pool's machines ends"),
    ({"until_ready": ["gpu-farm"]}, "neither a source"),
    ({"until_ready": [{"hosted": {}}]}, "needs a budget"),
    ({"until_ready": [{"hosted": {"budget": {"per_request": 1}}}]}, "per_request alone bounds nothing"),
    ({"until_ready": [{"local": {"workers": 2}}]}, "local takes no settings"),
    ({"until_ready": [{"name": "hosted", "local": None}]}, "is a source's type"),
    ({"until_ready": [{"name": "a/b", "local": None}]}, "without '/'"),
    ({"until_ready": ["local", "local"]}, "more than once"),
    ({"when_full": [{"rented": {"max_hosts": 2}}]}, "needs max_hosts and a budget"),
    ({"when_full": [{"rented": {**RENTED, "budget": {"per_day": 5, "per_request": 1}}}]}, "paid by the hour"),
    ({"until_ready": ["local"], "during": ["start", "sometimes"]}, "lists one or more of"),
    ({"until_ready": ["local"], "max_min": 0}, "between 1 and 1440"),
    ({"during": ["start"]}, "the plan has no until_ready"),
    ({"steady": "hosted"}, "the workload's own rented machines"),
    ({"surprise": 1}, "not ['surprise']"),
])
def test_a_plan_that_cannot_be_served_is_refused_in_words(serve, says):
    with pytest.raises(PlanRefused, match=says.replace("[", r"\[").replace("]", r"\]")):
        parse_serve(serve)


@pytest.mark.parametrize("serve, says", [
    ({"rules": []}, "rules arrive in a later version"),
    ({"substitute": [{"model": "small", "after_s": 20}]}, "a substitute model arrives"),
    ({"tz": "UTC"}, "comes with its rules"),
])
def test_what_a_later_stage_brings_is_refused_never_accepted_and_ignored(serve, says):
    with pytest.raises(PlanRefused, match=says):
        parse_serve(serve)


def test_a_create_sends_bursts_or_serve_never_both():
    assert parse_policy({"serve": {"when_full": [{"hosted": HOSTED}]}})["hosted"]["after_s"] == 5
    with pytest.raises(WorkloadRefused, match="bursts or serve, not both"):
        parse_policy({"serve": {}, "bursts": {"hosted": HOSTED}})
    with pytest.raises(WorkloadRefused, match="rented machines are what until_ready covers for"):
        parse_policy({"serve": {"until_ready": ["rented"]}})


def test_a_plan_is_said_in_words_naming_what_it_leaves_out():
    said = describe(parse_serve({"when_full": [{"hosted": HOSTED}]}))
    assert "no stop-gap (until_ready not set)" in said
    said = describe(parse_serve({"until_ready": ["local", {"hosted": {"budget": {"per_day": 10}}}]}))
    assert "until ready (start and replacement, for at most 60 min each time): local" in said
    assert "hosted APIs at once" in said and "when full: the queue" in said


# --- a program's grant names the sources its plans may use (D203) ---


def test_an_existing_grant_allows_what_it_allowed_before_plans():
    assert Grant().allowed_sources() == ("rented", "shared")
    assert Grant(may_borrow=False, hosted_per_day=5).allowed_sources() == ("rented", "hosted")
    assert Grant(sources=("rented", "local")).allowed_sources() == ("rented", "local")
    assert Grant.from_dict(Grant(sources=("local",)).as_dict()).sources == ("local",)


# --- today's fields, as the plan they mean ---


POLICIES = [
    {},
    {"hosted": {"after_s": 5.0, "budget": {"per_day": 20.0}}},
    {"hosted": {"after_s": 0.0, "budget": {"per_day": 1.0}}},
    {"rented": {"after_min": 10.0, "max_hosts": 2, "budget": {"per_day": 30.0}}},
    {"hosted": {"after_s": 3.0, "budget": {"per_month": 100.0}},
     "rented": {"after_min": 10.0, "max_hosts": 2, "budget": {"per_day": 30.0}}},
]


@pytest.mark.parametrize("policy", POLICIES)
def test_todays_fields_route_exactly_as_the_plan_they_mean(policy):
    """D198: with no `serve`, a workload has the plan its fields mean — and routing by that plan
    decides every request as today's code does: when it may borrow, and how soon a hosted API may
    take it. Every shape of policy, every moment of a workload's life, supervisor up or down."""
    for may_borrow, preparing, model_ready, share, fresh in itertools.product(
            (True, False), (True, False), (True, False), (0.25, 0.0), (True, False)):
        plan = from_bursts(policy, may_borrow)
        got = route(plan, preparing=preparing, model_ready=model_ready, created_at=0.0, now=86_400.0 * 3,
                    gap_ends_at=None, fresh=fresh, borrow_share=share)
        # Today (router/app.py): may borrow while preparing, where the pool lends and the workload
        # may; the dispatcher lends only while the model has no ready host of its own.
        today_borrows = preparing and share > 0 and may_borrow and not model_ready
        today_hosted = policy["hosted"]["after_s"] if policy.get("hosted") else None
        moment = dict(may_borrow=may_borrow, preparing=preparing, model_ready=model_ready, share=share, fresh=fresh)
        assert (got.may_borrow and (got.borrow_while_ready or not model_ready)) == today_borrows, moment
        assert got.hosted_after_s == today_hosted, moment
        assert got.borrow_kinds is None, "today's borrowing reaches any machine of the shared workload"
    assert from_bursts(policy, True).max_min is None, "today's borrowing has no time limit"
    with pytest.raises(ValueError, match="a stored plan has a max_min"):
        compile_policy(from_bursts(policy, True))


# --- routing by a plan ---


def at(plan, **facts):
    moment = dict(preparing=False, model_ready=True, created_at=0.0, now=100.0, gap_ends_at=None, fresh=True,
                  borrow_share=0.25)
    moment.update(facts)
    return route(plan, **moment)


STOP_GAP = parse_serve({"until_ready": ["local", {"hosted": {"budget": {"per_day": 10}}}]})


def test_the_stop_gap_covers_the_start_then_ends_when_the_model_is_ready():
    starting = at(STOP_GAP, preparing=True, model_ready=False)
    assert starting.covering and starting.may_borrow and starting.borrow_kinds == frozenset({"local"})
    assert starting.hosted_after_s == 0 and starting.order == ("local", "hosted")
    ready = at(STOP_GAP, preparing=True, model_ready=True)
    assert not ready.covering and not ready.may_borrow
    assert ready.hosted_after_s is None, "its only hosted source is the stop-gap's: nothing once ready"


def test_the_start_is_covered_for_at_most_max_min():
    assert at(STOP_GAP, preparing=True, model_ready=False, now=59 * 60).covering
    assert not at(STOP_GAP, preparing=True, model_ready=False, now=61 * 60).covering


def test_a_replacement_is_covered_while_the_supervisor_marks_it_and_never_on_a_stale_mark():
    marked = at(STOP_GAP, gap_ends_at=200.0)
    assert marked.covering and marked.borrow_while_ready, "the workload has ready hosts, and still borrows"
    assert not at(STOP_GAP, gap_ends_at=50.0).covering, "past the gap's end"
    assert not at(STOP_GAP, gap_ends_at=200.0, short=False).covering, "its hosts are ready again"
    stale = at(STOP_GAP, gap_ends_at=200.0, fresh=False)
    assert not stale.covering, "a supervisor that stopped opens no replacement gap"
    only_start = parse_serve({"until_ready": ["local"], "during": ["start"]})
    assert not at(only_start, gap_ends_at=200.0).covering


def test_a_stale_supervisor_keeps_a_starting_workload_on_what_spends_nothing():
    stale = at(STOP_GAP, preparing=True, model_ready=False, fresh=False)
    assert stale.may_borrow and stale.hosted_after_s is None and stale.order == ("local",)
    burst = parse_serve({"until_ready": ["shared"], "when_full": [{"hosted": HOSTED}]})
    assert at(burst, preparing=True, model_ready=False, fresh=False).hosted_after_s == 5, \
        "its when_full burst is its own, as today: only the stop-gap's shorter wait is closed"


def test_shared_reaches_any_machine_of_the_pool_and_fixed_only_the_operators_elsewhere():
    assert at(parse_serve({"until_ready": ["fixed", "shared"]}), gap_ends_at=200.0).borrow_kinds is None
    assert at(parse_serve({"until_ready": ["fixed"]}), gap_ends_at=200.0).borrow_kinds == frozenset({"fixed-remote"})
    assert not at(parse_serve({"until_ready": ["local"]}), gap_ends_at=200.0, borrow_share=0).may_borrow


# --- the dispatcher ranks by the plan's order ---


def host(host_id, kind, priority, *, workload=None, busy=False):
    variant = ResolvedVariant(tag="m1", runtime_class=None, enforces_schema=None)
    worker = Worker(worker_id=f"{host_id}/w0", state=WorkerState.BUSY if busy else WorkerState.IDLE)
    return Host(host_id=host_id, kind=kind, priority=priority, capabilities=frozenset(), client=httpx.AsyncClient(),
                workers=[worker], variants={"m1": (variant,)}, literal_variants={}, state=HostState.READY, resident=frozenset({"m1"}),
                workload=workload)


def take(dispatcher, need):
    assignment = dispatcher._take_idle(dispatcher.eligible(need), "r1", need)
    return assignment.host.host_id if assignment else None


def test_a_replacement_serves_the_workloads_own_host_first_then_the_stop_gap_in_its_order():
    own = host("own", "rented-interruptible", 20, workload="w")
    laptop = host("laptop", "local", 0)
    office = host("office", "fixed-remote", 10)
    dispatcher = Dispatcher([own, laptop, office])
    need = Need(model="m1", workload="w", may_borrow=True, borrow_share=1.0, borrow_while_ready=True,
                order=("fixed", "local"))
    assert take(dispatcher, need) == "own", "steady first, though the laptop's tier comes before it"
    assert take(dispatcher, need) == "office", "then fixed, as the plan lists it before local"
    assert take(dispatcher, need) == "laptop"


def test_a_stop_gap_borrows_only_the_kinds_it_names():
    dispatcher = Dispatcher([host("laptop", "local", 0), host("shared-rented", "rented-on-demand", 20)])
    need = Need(model="m1", workload="w", may_borrow=True, borrow_share=1.0, borrow_kinds=frozenset({"local"}),
                order=("local",))
    assert [h.host_id for h, _ in dispatcher.eligible(need)] == ["laptop"]


def test_without_a_plan_the_tiers_alone_decide_as_always():
    dispatcher = Dispatcher([host("office", "fixed-remote", 10), host("laptop", "local", 0)])
    need = Need(model="m1", workload="w", may_borrow=True, borrow_share=1.0)
    assert take(dispatcher, need) == "laptop"


def test_a_stored_policy_without_a_plan_has_none():
    assert plan_of({}) is None and plan_of({"hosted": HOSTED}) is None
    assert isinstance(plan_of(compile_policy(ServingPlan())), ServingPlan)
