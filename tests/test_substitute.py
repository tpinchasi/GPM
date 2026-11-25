"""A substitute model, opted into by a workload's plan (S19 stage 9, D202; workloads.md §14.5).

A request that has waited `after_s` with no worker of its own model free is answered by the first
substitute a source of the plan can give it — through the same eligible hosts its options allow.
The answer says so (`X-GPM-Substituted-For`, `X-GPM-Route: substitute/<model>`, the SDK's
`reply.substituted_for` and one warning per client); `X-GPM-Substitute: never` or
`substitute=False` refuses it; session consistency never gets one; a rule may keep requests from
it; its answers are left out of the workload's latency. Nothing here spends money.
"""

import logging
import threading
import time

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, pool_harness
from gpm_client import PoolClient
from gpm_server.serving_plan import PlanRefused, compile_policy, json_schema, parse_serve, readback
from gpm_server.workload_store import Workload, WorkloadStore

BIG, SMALL = "big", "small"


# --- the plan ---


def test_a_substitute_is_read_and_refused_in_words():
    plan = parse_serve({"until_ready": ["shared"], "substitute": [{"model": SMALL, "after_s": 2}],
                        "rules": [{"name": "batch waits", "when": {"priority": "batch"}, "then": {"substitute": False}}]})
    assert plan.substitutes == ((SMALL, 2.0),)
    assert parse_serve(plan.as_dict()).as_dict() == plan.as_dict()
    assert any(line.startswith("Substitute — a request that has waited") for line in readback(plan))
    assert "substitute" in json_schema()["properties"]
    for serve, says in [
        ({"substitute": []}, "a list of 1 to 4"),
        ({"substitute": [{"model": SMALL}, {"model": SMALL}]}, "named twice"),
        ({"substitute": [{"model": SMALL, "after_s": -1}]}, "after_s is between"),
        ({"rules": [{"name": "x", "when": {"priority": "batch"}, "then": {"substitute": False}}]}, "names no substitute"),
        ({"substitute": [{"model": SMALL}], "rules": [{"name": "x", "when": {"priority": "batch"},
                                                        "then": {"substitute": True}}]}, "never add one"),
        ({"defaults": {"substitute": "never"}}, "leave substitute out"),
    ]:
        with pytest.raises(PlanRefused, match=says):
            parse_serve(serve)


# --- the router ---


@pytest.fixture
def pool():
    with pool_harness(
        [EngineSpec(id="laptop", resident={BIG}, workers=2, chunk_delay_s=1.0),
         EngineSpec(id="office", kind="fixed-remote", resident={SMALL}, workers=2)],
        model_set=[BIG, SMALL], extra_config={"workloads": {"borrow_share": 1.0}},
        pool_settings={"models_per_host": "declared"},
        host_overrides={"laptop": {"models": [BIG]}, "office": {"models": [SMALL]}},
    ) as harness:
        harness.reprobe()
        harness.stop_supervisor()
        yield harness


def add_workload(pool, serve, name="night"):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name=name, model=BIG, builds={BIG: BIG}, latency_s=20, parallel=2, kind="roi", lease_id=f"lease-{name}",
        state="preparing", workers_per_host=0, hosts_at_start=0, plan={"may_borrow": True}, created_at=now,
        updated_at=now, policy=compile_policy(parse_serve(serve)),
    ))
    _, key = store.mint_key(name)
    pool.loop.run(pool.state.registry.refresh())
    return key


def ask(pool, key, headers=None):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30,
                      headers={"Authorization": f"Bearer {key}", **(headers or {})},
                      json={"model": BIG, "messages": [{"role": "user", "content": "hi"}], "stream": False})


def laptop_busy(pool):
    """Both of the laptop's workers busy with the shared workload's answers, for about a second."""
    threads = [threading.Thread(target=ask, args=(pool, APP_KEY)) for _ in range(2)]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    return threads


SERVE = {"until_ready": ["shared"], "substitute": [{"model": SMALL, "after_s": 0.2}]}


def test_a_request_that_waited_is_answered_by_the_substitute_and_says_so(pool):
    key = add_workload(pool, SERVE)
    busy = laptop_busy(pool)
    answer = ask(pool, key)
    for thread in busy:
        thread.join()
    assert answer.status_code == 200, answer.text
    assert answer.headers["x-gpm-substituted-for"] == BIG and answer.headers["x-gpm-served-model"] == SMALL
    assert answer.headers["x-gpm-route"] == f"substitute/{SMALL}" and answer.headers["x-gpm-host"] == "office"
    row = [r for r in pool.wait_for_log(3) if r["workload"] == "night"][-1]
    assert row["route"] == f"substitute/{SMALL}" and row["model_requested"] == BIG
    # Its own model free: answered by it, with no substitute.
    own = ask(pool, key)
    assert own.headers["x-gpm-served-model"] == BIG and "x-gpm-substituted-for" not in own.headers


@pytest.mark.parametrize("headers", [{"X-GPM-Substitute": "never"}, {"X-GPM-Session": "s-1"},
                                     {"X-GPM-Priority": "batch"}])
def test_a_request_refusing_it_waits_for_its_own_model(pool, headers):
    key = add_workload(pool, {**SERVE, "rules": [{"name": "batch waits", "when": {"priority": "batch"},
                                                  "then": {"substitute": False}}]})
    busy = laptop_busy(pool)
    started = time.monotonic()
    answer = ask(pool, key, headers)
    for thread in busy:
        thread.join()
    assert answer.status_code == 200 and answer.headers["x-gpm-served-model"] == BIG
    assert "x-gpm-substituted-for" not in answer.headers and time.monotonic() - started > 0.5, "it waited"


def test_an_unknown_value_is_refused(pool):
    key = add_workload(pool, SERVE)
    refused = ask(pool, key, {"X-GPM-Substitute": "sometimes"})
    assert refused.status_code == 400 and refused.json()["reason"] == "unknown_option"


def test_the_sdk_says_a_substitute_answered_once_and_can_refuse_one(pool, caplog):
    key = add_workload(pool, SERVE)
    client = PoolClient(pool.url, key)
    with pytest.raises(ValueError, match="substitute is True or False"):
        client.chat(BIG, [{"role": "user", "content": "hi"}], substitute="no")
    with caplog.at_level(logging.WARNING, logger="gpm_client"):
        for _ in range(2):
            busy = laptop_busy(pool)
            reply = client.chat(BIG, [{"role": "user", "content": "hi"}])
            for thread in busy:
                thread.join()
            assert reply.substituted_for == BIG and reply.route == f"substitute/{SMALL}"
    assert len([r for r in caplog.records if "substitute model answered" in r.getMessage()]) == 1, "once per client"
    busy = laptop_busy(pool)
    reply = client.chat(BIG, [{"role": "user", "content": "hi"}], substitute=False)
    for thread in busy:
        thread.join()
    assert reply.substituted_for is None and reply.served_model == BIG
    client.close()


# --- the supervisor ---


def test_only_a_catalogued_model_doing_the_same_may_substitute():
    from gpm_server.supervisor.workloads import WorkloadRefused
    from gpm_server.workload_store import ModelTarget

    catalog = {BIG: {"variants": [{"tag": BIG, "serves": ["chat"]}]},
               SMALL: {"variants": [{"tag": SMALL, "serves": ["chat"]}]},
               "ears": {"variants": [{"tag": "ears", "serves": ["embed"]}]}}
    with pool_harness([EngineSpec(id="laptop", resident={BIG, SMALL, "ears"}, workers=2)],
                      model_set=[BIG, SMALL, "ears"], catalog=catalog) as harness:
        check = harness.supervisor.workloads.check_policy
        targets = [ModelTarget(model=BIG, latency_s=30, parallel=1)]
        check(targets, compile_policy(parse_serve({"substitute": [{"model": SMALL}]})))
        for model, says in (("nowhere", "not a model of this pool's catalog"), (BIG, "one of the workload's own"),
                            ("ears", "does none of what 'big' does")):
            with pytest.raises(WorkloadRefused, match=says):
                check(targets, compile_policy(parse_serve({"substitute": [{"model": model}]})))


# --- the dispatcher, alone ---


def _host(host_id, kind, models, *, workload=None, busy=False):
    from gpm_server.catalog import ResolvedVariant
    from gpm_server.models import Host, HostState, Worker, WorkerState

    workers = [Worker(worker_id=f"{host_id}/w0", state=WorkerState.BUSY if busy else WorkerState.IDLE)]
    return Host(host_id=host_id, kind=kind, priority=30 if kind == "hosted" else 0, capabilities=frozenset(),
                client=httpx.AsyncClient(), workers=workers,
                variants={m: (ResolvedVariant(tag=m, runtime_class=None, enforces_schema=None),) for m in models},
                literal_variants={}, state=HostState.READY, resident=frozenset(models), workload=workload)


def _acquire(dispatcher, need, within):
    import asyncio

    async def go():
        started = time.monotonic()
        try:
            got = await dispatcher.acquire(need, request_id="r", deadline=started + within)
            return got, time.monotonic() - started
        except Exception as exc:  # noqa: BLE001 - the outcome is what is compared
            return exc, time.monotonic() - started

    return asyncio.run(go())


def test_a_spent_budget_is_said_at_once_with_a_substitute_too_and_its_hosted_moment_wakes_it():
    from gpm_server.router.dispatch import BudgetSpent, Dispatcher, Need

    now = time.monotonic()
    spent = Need(model=BIG, workload="w", hosted_after=now - 1, reserve=lambda host, v: None,
                 substitutes=((SMALL, now + 0.2),))
    outcome, took = _acquire(Dispatcher([_host("h", "hosted", [BIG, SMALL])]), spent, 2.0)
    assert isinstance(outcome, BudgetSpent) and took < 0.5, "every way left is a spent budget: said at once"
    # A substitute on a hosted API that may take it later than the substitute's own moment.
    hosts = [_host("own", "rented-on-demand", [BIG], workload="w", busy=True),
             _host("shared", "local", [SMALL], busy=True), _host("api", "hosted", [SMALL])]
    now = time.monotonic()
    waits = Need(model=BIG, workload="w", hosted_after=now + 0.5, reserve=lambda host, v: object(), may_borrow=True,
                 borrow_share=1.0, substitutes=((SMALL, now + 0.1),))
    outcome, took = _acquire(Dispatcher(hosts), waits, 3.0)
    assert outcome.host.host_id == "api" and outcome.substituted_for == BIG and took < 1.5, (outcome, took)


def test_a_call_allows_a_substitute_its_client_refuses():
    from gpm_client.client import _options

    seen = []
    transport = httpx.MockTransport(lambda request: (
        seen.append(request.headers.get("x-gpm-substitute")),
        httpx.Response(200, json={"options": {"substitute": ["never"]}} if request.url.path == "/pool/status"
                       else {"choices": [{"message": {"content": "ok"}}]}))[1])
    client = PoolClient("http://pool.invalid", "k", transport=transport, substitute=False)
    client.chat(BIG, [{"role": "user", "content": "hi"}])
    client.chat(BIG, [{"role": "user", "content": "hi"}], substitute=True)
    assert [s for s in seen if s is not None] == ["never"] and seen[-1] is None, seen
    assert _options(None, None, True) == {}
    client.close()


def test_a_rule_armed_while_a_request_waits_keeps_it_from_the_substitute(pool):
    key = add_workload(pool, {**SERVE, "substitute": [{"model": SMALL, "after_s": 0.6}], "rules": [
        {"name": "careful", "when": {"between": "00:00-23:59"}, "then": {"substitute": False}}]})
    busy = laptop_busy(pool)
    out = []
    asking = threading.Thread(target=lambda: out.append(ask(pool, key)))
    asking.start()
    time.sleep(0.2)
    WorkloadStore(pool.database).set_armed_rules("night", {"careful"})  # as the supervisor writes it
    pool.state.registry.armed = {**pool.state.registry.armed, "night": frozenset({"careful"})}
    asking.join()
    for thread in busy:
        thread.join()
    assert out[0].headers["x-gpm-served-model"] == BIG and "x-gpm-substituted-for" not in out[0].headers
