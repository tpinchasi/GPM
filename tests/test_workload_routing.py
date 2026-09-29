"""The router, by workload (D115, workloads.md §3–§4): one listener, and the key picks the
workload.

Against the real router and real fake engines, with the supervisor stopped so the host table
says exactly what each test needs: `laptop` is the shared workload's, `rented` is the workload
`research`'s. Nothing here rents anything.
"""

import time

import pytest
from fakes.harness import EngineSpec, pool_harness
from gpm_server.workload_store import Workload, WorkloadStore

MODEL = "m1"


def set_host(pool, host_id, *, workload=None, state=None):
    pool.database.execute(
        "UPDATE hosts SET workload = ?, state = COALESCE(?, state), updated_at = ? WHERE host_id = ?",
        (workload, state, time.time() + 1, host_id),
    )
    pool.loop.run(pool.state.registry.refresh())


def add_workload(pool, name="research", state="serving", model=MODEL):
    store = WorkloadStore(pool.database)
    now = time.time()
    store.create(Workload(
        name=name, model=model, builds={model: model}, latency_s=20, parallel=4, kind="roi",
        lease_id=f"lease-{name}", state=state, workers_per_host=4, hosts_at_start=1, plan={},
        created_at=now, updated_at=now,
    ))
    _, key = store.mint_key(name)
    pool.loop.run(pool.state.registry.refresh())
    return store, key


def ask(client, path="/v1/chat/completions", model=MODEL):
    return client.post(path, json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


@pytest.fixture
def pool():
    with pool_harness(
        [EngineSpec(id="laptop", resident={MODEL}, workers=4), EngineSpec(id="rented", resident={MODEL}, workers=4)],
        model_set=[MODEL],
    ) as harness:
        harness.stop_supervisor()
        set_host(harness, "rented", workload="research")
        yield harness


def test_a_workload_key_reaches_only_its_workloads_hosts(pool):
    _, key = add_workload(pool)
    with pool.client(key=key) as client:
        for _ in range(3):
            answer = ask(client)
            assert answer.status_code == 200, answer.text
            assert answer.headers["X-GPM-Host"] == "rented"
    rows = pool.wait_for_log(3)
    assert {r["workload"] for r in rows} == {"research"}
    assert all(r["borrowed"] == 0 and r["concurrency"] == 1 for r in rows)


def test_an_app_key_never_reaches_a_workloads_host(pool):
    add_workload(pool)
    with pool.client() as client:
        for _ in range(3):
            assert ask(client).headers["X-GPM-Host"] == "laptop"
    assert {r["workload"] for r in pool.wait_for_log(3)} == {None}


def test_keys_that_reach_nothing(pool):
    store, key = add_workload(pool)
    with pool.client(key="gpmw_" + "0" * 64) as client:
        assert ask(client).status_code == 401
    store.expire_keys("research")
    pool.loop.run(pool.state.registry.refresh())
    with pool.client(key=key) as client:
        assert ask(client).status_code == 401, "an expired key reaches nothing"


def test_a_rotated_out_key_works_until_its_grace_ends(pool):
    store, old = add_workload(pool)
    _, new = store.rotate_key("research", grace_s=0.3)
    pool.loop.run(pool.state.registry.refresh())
    with pool.client(key=old) as client:
        assert ask(client).status_code == 200
    time.sleep(0.4)
    with pool.client(key=old) as client:
        assert ask(client).status_code == 401, "refused at the request once its time passes"
    with pool.client(key=new) as client:
        assert ask(client).status_code == 200


def test_an_ended_workload_says_so(pool):
    store, key = add_workload(pool)
    store.set_state("research", "ending")
    pool.loop.run(pool.state.registry.refresh())
    with pool.client(key=key) as client:
        answer = ask(client)
    assert answer.status_code == 503 and answer.json()["error"] == "workload_ended"
    assert "Retry-After" not in answer.headers, "nothing to retry for"


def test_the_url_prefix_must_match_the_key(pool):
    _, key = add_workload(pool)
    add_workload(pool, name="other")
    with pool.client(key=key) as client:
        assert ask(client, "/w/research/v1/chat/completions").headers["X-GPM-Host"] == "rented"
        wrong = ask(client, "/w/other/v1/chat/completions")
    assert wrong.status_code == 403 and wrong.json()["reason"] == "wrong_workload"


def test_a_workload_serves_only_its_model(pool):
    _, key = add_workload(pool)
    with pool.client(key=key) as client:
        answer = ask(client, model="m2")
    assert answer.status_code == 404 and answer.json()["reason"] == "model_not_in_workload"


def test_a_preparing_workload_borrows_the_shared_hosts(pool):
    _, key = add_workload(pool, state="preparing")
    set_host(pool, "rented", workload="research", state="preparing")
    with pool.client(key=key) as client:
        answer = ask(client)
    assert answer.status_code == 200 and answer.headers["X-GPM-Host"] == "laptop"
    (row,) = pool.wait_for_log(1)
    assert row["workload"] == "research" and row["borrowed"] == 1


def test_a_serving_workload_never_borrows(pool):
    """Serving means one of its hosts was ready; borrowing is for the start only."""
    _, key = add_workload(pool, state="serving")
    set_host(pool, "rented", workload="research", state="preparing")
    with pool.client(key=key) as client:
        answer = ask(client)
    assert answer.status_code == 503 and answer.json()["reason"] == "workload_preparing"


def test_status_for_a_workload_key_is_that_workloads_view(pool):
    _, key = add_workload(pool)
    with pool.client(key=key) as client:
        mine = client.get("/pool/status").json()
        same = client.get("/w/research/pool/status").json()
        other = client.get("/w/other/pool/status")
    assert mine["workload"] == "research" and mine["state"] == "serving" and mine["model_set"] == [MODEL]
    assert [h["host_id"] for h in mine["hosts"]] == ["rented"] and mine["borrowing"] is False
    assert same == mine and other.status_code == 403
    assert "delivery" in mine, "the SDK sizes its budget from it, as for the shared pool"
    with pool.client() as client:
        shared = client.get("/pool/status").json()
    assert [h["host_id"] for h in shared["hosts"]] == ["laptop"], "the shared view does not show a workload's hosts"


def test_a_key_stops_at_the_leases_end_even_before_the_supervisor_notices(pool):
    store, key = add_workload(pool)
    store.set_ends_at("research", time.time() - 1)
    pool.loop.run(pool.state.registry.refresh())
    with pool.client(key=key) as client:
        answer = ask(client)
    assert answer.status_code == 503 and answer.json()["error"] == "workload_ended"


def test_a_workload_key_does_not_read_the_shared_directory(pool):
    _, key = add_workload(pool)
    with pool.client(key=key) as client:
        assert client.get("/pool/directory").status_code == 403
