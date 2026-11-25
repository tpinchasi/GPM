"""Workers the operator reserves on their own machines (S19 stage 4, D199; workloads.md §14.2).

`reserve: {local: n}` gives a workload the first call on that many workers of the operator's own
machines: its requests take them before a waiting shared request, shared requests use them while
it leaves them idle, and nothing running is stopped. `steady: local` (or `fixed`) runs a workload
there, renting nothing. Reservations stay within `workloads.max_reserved_share` of each kind, and
only the operator sets them. Nothing here spends money.
"""

import threading
import time

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, pool_harness
from gpm_client import WorkloadRefused
from gpm_server.serving_plan import PlanRefused, compile_policy, parse_serve
from gpm_server.workload_store import Workload, WorkloadStore
from test_serving_plan_routing import BIG, SPEC, grant, programs, provider, provisioner  # noqa: F401 - fixtures

MODEL = "m1"
ADMIN = "gpmx_admin_test"


# --- the plan ---


def test_steady_on_the_operators_machines_needs_their_workers_reserved():
    plan = parse_serve({"steady": "local", "reserve": {"local": 4}, "defaults": {"keep_on": "owned"}})
    assert plan.steady_kinds == {"local"} and plan.reserve == {"local": 4}
    assert parse_serve(plan.as_dict()).as_dict() == plan.as_dict()
    with pytest.raises(PlanRefused, match="add reserve"):
        parse_serve({"steady": "fixed"})
    with pytest.raises(PlanRefused, match="nothing is rented, so there is nothing to cover"):
        parse_serve({"steady": "local", "reserve": {"local": 2}, "until_ready": ["fixed"]})
    with pytest.raises(PlanRefused, match="a rented burst above steady 'local' arrives with rules"):
        parse_serve({"steady": "local", "reserve": {"local": 2},
                     "when_full": [{"rented": {"max_hosts": 1, "budget": {"per_day": 1.0}}}]})
    with pytest.raises(PlanRefused, match="reserve takes local and fixed"):
        parse_serve({"reserve": {"rented": 2}})
    with pytest.raises(PlanRefused, match="whole number of workers from 1 to 1024"):
        parse_serve({"reserve": {"local": 0}})


# --- the router ---


@pytest.fixture
def pool():
    with pool_harness(
        [EngineSpec(id="laptop", resident={MODEL}, workers=2, chunk_delay_s=1.5),
         EngineSpec(id="office", kind="fixed-remote", resident={MODEL}, workers=1, chunk_delay_s=1.5)],
        model_set=[MODEL],
    ) as harness:
        harness.reprobe()
        harness.stop_supervisor()
        yield harness


def add_workload(pool, serve, name="night"):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name=name, model=MODEL, builds={MODEL: MODEL}, latency_s=20, parallel=2, kind="roi",
        lease_id=f"lease-{name}", state="serving", workers_per_host=0, hosts_at_start=0,
        plan={"may_borrow": True}, created_at=now, updated_at=now, policy=compile_policy(parse_serve(serve)),
    ))
    _, key = store.mint_key(name)
    pool.loop.run(pool.state.registry.refresh())
    return key


def ask(pool, key, *, stream=True):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30, headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": stream})


def in_background(call):
    out = []
    thread = threading.Thread(target=lambda: out.append(call()))
    thread.start()
    return thread, out


def test_a_workload_runs_on_its_reserved_workers_and_no_more(pool):
    key = add_workload(pool, {"steady": "local", "reserve": {"local": 1}})
    first, got = in_background(lambda: ask(pool, key))
    time.sleep(0.3)
    started = time.monotonic()
    second = ask(pool, key)
    first.join()
    assert got[0].headers["x-gpm-host"] == "laptop" and got[0].headers["x-gpm-route"] == "steady/local"
    assert second.headers["x-gpm-host"] == "laptop"
    assert time.monotonic() - started > 1.0, "its one reserved worker was busy: it waited, though another was idle"
    rows = [r for r in pool.wait_for_log(2) if r["workload"] == "night"]
    assert rows and all(r["borrowed"] == 0 for r in rows), "within its reservation: not borrowing"
    # The other worker is the shared workload's, and serves it.
    assert ask(pool, APP_KEY, stream=False).headers["x-gpm-host"] in ("laptop", "office")


def test_a_reserved_worker_goes_to_its_workload_before_a_waiting_shared_request(pool):
    pool.database.execute("UPDATE hosts SET state = 'unreachable', updated_at = ? WHERE host_id = 'office'",
                          (time.time() + 1,))
    pool.loop.run(pool.state.registry.refresh())
    key = add_workload(pool, {"steady": "local", "reserve": {"local": 1}})
    # Both of the laptop's workers busy with shared requests, ending 0.7 s apart: the reserved
    # one is used while idle.
    shared_a, _ = in_background(lambda: ask(pool, APP_KEY))
    time.sleep(0.7)
    shared_b, _ = in_background(lambda: ask(pool, APP_KEY))
    time.sleep(0.2)
    waiting_shared, shared_done = in_background(lambda: (ask(pool, APP_KEY), time.monotonic())[1])
    time.sleep(0.2)  # the shared request queues first
    workload_request, workload_done = in_background(lambda: (ask(pool, key), time.monotonic())[1])
    for thread in (shared_a, shared_b, waiting_shared, workload_request):
        thread.join()
    # First come, first served would give the first freed worker to the shared request, 0.7 s
    # before the second frees: the workload's answer is the earlier one only by its reservation.
    assert workload_done[0] < shared_done[0] - 0.4, "the freed worker went to the workload it is reserved for"


# --- the supervisor ---


@pytest.fixture
def operator(programs):  # noqa: F811
    return programs


def create(pool, **body):
    return httpx.post(f"{pool.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"},
                      json={"hours": 2, "latency_s": 30, "parallel": 2, **body}, timeout=60)


def test_an_operator_runs_a_workload_on_local_workers_renting_nothing(operator):
    made = create(operator, name="night", model=BIG, serve={"steady": "local", "reserve": {"local": 2}})
    assert made.status_code == 201, made.text
    view = made.json()["workload"]
    assert view["state"] == "serving" and view["groups"] == [], "nothing to rent or prepare"
    assert "on the workers reserved for it" in " ".join(made.json()["plan"]["reasons"])
    refused = create(operator, name="day", model=BIG, serve={"steady": "local", "reserve": {"local": 1}})
    assert refused.status_code == 400 and "0 left" in refused.text, "4 workers, half reservable, 2 taken"
    nowhere = create(operator, name="elsewhere", model=BIG, serve={"steady": "fixed", "reserve": {"fixed": 1}})
    assert nowhere.status_code == 400 and "no fixed machine of this pool holds it" in nowhere.text
    changed = httpx.post(f"{operator.control_url}/pool/workloads/night/policy",
                         headers={"Authorization": f"Bearer {ADMIN}"}, json={"serve": {}})
    assert changed.status_code == 400 and "where it runs is set when it is created" in changed.text


def test_a_program_never_reserves(programs):  # noqa: F811
    key = grant(programs)
    with provisioner(programs, key) as pool_side, pytest.raises(WorkloadRefused, match="never reserves"):
        pool_side.workload(BIG, **SPEC, serve={"until_ready": ["local"], "reserve": {"local": 1}})
