"""Router and supervisor share a database and nothing else.

docs/spec/supervisor.md §1 (D14). The point of the split is that the half in the request path
never waits on the half that does slow, failure-prone work — so the router must keep serving
when the supervisor is gone, and no secret may cross between them.
"""

import time

import pytest
from fakes.harness import APP_KEY, EngineSpec, pool_harness
from gpm_server.db import Database, HostTable, SupervisorBusy, SupervisorLock
from gpm_server.supervisor import Supervisor

MODEL = "m1"


def chat():
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False}


@pytest.fixture
def pool():
    with pool_harness(
        [EngineSpec(id="local-1", resident={MODEL}, workers=2)],
        model_set=[MODEL],
    ) as harness:
        yield harness


def test_the_router_serves_from_the_table_the_supervisor_published(pool):
    with pool.client() as client:
        assert client.post("/api/chat", json=chat()).status_code == 200

    published = {row.host_id: row for row in HostTable(pool.database).all()}
    assert published["local-1"].state == "ready"
    assert published["local-1"].dial_url == pool.engines["local-1"].base_url
    assert MODEL in published["local-1"].resident


def test_routing_continues_when_the_supervisor_dies(pool):
    pool.stop_supervisor()
    time.sleep(0.5)

    with pool.client() as client:
        response = client.post("/api/chat", json=chat())
        status = client.get("/pool/status").json()

    assert response.status_code == 200
    assert response.headers["X-GPM-Host"] == "local-1"
    assert status["capacity"]["hosts_ready"] == 1


def test_a_host_removed_from_the_table_stops_taking_requests(pool):
    HostTable(pool.database).remove("local-1")

    deadline = time.monotonic() + 10
    while pool.state.hosts and time.monotonic() < deadline:
        time.sleep(0.05)

    with pool.client() as client:
        response = client.post("/api/chat", json=chat())
    assert response.status_code == 503
    assert response.json()["reason"] == "hosts_unreachable"


def test_the_router_reports_what_it_is_serving_back_to_the_supervisor(pool):
    with pool.client() as client:
        client.post("/api/chat", json=chat())

    pool.loop.run(pool.state.registry.publish_counters())
    counters = pool.supervisor.counters.all()

    assert counters["local-1"].total == 2
    assert counters["local-1"].busy == 0
    assert counters["local-1"].requests_served == 1
    assert counters["local-1"].last_request_at is not None


def test_only_one_supervisor_may_hold_a_pool(pool):
    second = Supervisor(pool.config, pool.database)
    with pytest.raises(SupervisorBusy):
        pool.loop.run(second.start())


def test_a_supervisor_that_stopped_beating_does_not_lock_the_pool_out(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        crashed = SupervisorLock(database, pool="p", owner="crashed", stale_after=0.2)
        crashed.acquire()
        time.sleep(0.3)  # it stops refreshing its heartbeat, as a crash would

        successor = SupervisorLock(database, pool="p", owner="successor", stale_after=0.2)
        successor.acquire()  # must not raise: a crash cannot lock the pool out forever
    finally:
        database.close()


def test_no_secret_crosses_between_the_processes(tmp_path, monkeypatch):
    """The router builds credentials from configuration; the database carries a URL and
    state, never a key (threat model T16/T19)."""
    monkeypatch.setenv("TEST_HOST_TOKEN", "super-secret-host-token")
    with pool_harness(
        [EngineSpec(id="local-1", resident={MODEL}, workers=1)],
        model_set=[MODEL],
        host_overrides={"local-1": {"transport": {"bearer_env": "TEST_HOST_TOKEN"}}},
    ) as harness:
        with harness.client() as client:
            assert client.post("/api/chat", json=chat()).status_code == 200

        # The engine saw the credential...
        _, _, headers = harness.engines["local-1"].fake.received[-1]
        assert headers["authorization"] == "Bearer super-secret-host-token"

        # ...and the shared database never did.
        raw = harness.database.path.read_bytes()
        assert b"super-secret-host-token" not in raw
        assert b"TEST_HOST_TOKEN" not in raw
        assert APP_KEY.encode() not in raw


def test_a_lock_held_by_a_dead_process_on_this_host_is_taken_at_once(tmp_path):
    """A crash must not lock the pool out, and a restart must not have to wait."""
    import os
    import socket
    import subprocess
    import sys

    database = Database(tmp_path / "gpm.sqlite3")
    try:
        # A process that certainly existed and certainly no longer does.
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        dead_pid = gone.pid

        crashed = SupervisorLock(database, pool="p", owner=f"{socket.gethostname()}:{dead_pid}:x")
        crashed.acquire()
        database.execute("UPDATE supervisor_lock SET pid = ? WHERE pool = 'p'", (dead_pid,))

        successor = SupervisorLock(database, pool="p", owner=f"{socket.gethostname()}:{os.getpid()}:y")
        successor.acquire()  # heartbeat is fresh, but the holder is dead: no waiting
    finally:
        database.close()


def test_a_live_holder_on_this_host_still_blocks(tmp_path):
    import os
    import socket

    database = Database(tmp_path / "gpm.sqlite3")
    try:
        holder = SupervisorLock(database, pool="p", owner=f"{socket.gethostname()}:{os.getpid()}:x")
        holder.acquire()  # this very process: alive
        with pytest.raises(SupervisorBusy):
            SupervisorLock(database, pool="p", owner=f"{socket.gethostname()}:{os.getpid()}:y").acquire()
    finally:
        database.close()


def test_stopping_the_supervisor_process_releases_its_lock(tmp_path):
    """`gpm serve` stops its child with SIGTERM; a restart straight after must succeed."""
    import signal
    import subprocess
    import sys
    import textwrap

    config_path = tmp_path / "pool.yaml"
    db_path = tmp_path / "gpm.sqlite3"
    config_path.write_text(textwrap.dedent(f"""
        pool: {{ name: stoptest, model_set: [m1], probe_interval_s: 1 }}
        auth: {{ app_keys: [k] }}
        hosts: []
        rented: {{ provider: fake, bidding: {{ bid_ceiling: 0.5 }} }}
        request_log: {db_path}
    """))
    child = subprocess.Popen(
        [sys.executable, "-m", "gpm_server.cli", "supervise", "-c", str(config_path), "--log-level", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        database = Database(db_path)
        deadline = time.monotonic() + 15
        while not database.query("SELECT * FROM supervisor_lock") and time.monotonic() < deadline:
            time.sleep(0.1)
        assert database.query("SELECT * FROM supervisor_lock"), "the supervisor never took the lock"

        child.send_signal(signal.SIGTERM)
        code = child.wait(timeout=15)

        assert code == 0, child.stderr.read().decode()
        assert database.query("SELECT * FROM supervisor_lock") == []
        database.close()
    finally:
        if child.poll() is None:
            child.kill()
