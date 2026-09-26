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


def test_the_router_follows_every_routing_field_of_a_host_it_already_knows(pool):
    """Found live: the router copied a republished row's state and loaded models, but not its
    on-disk list, residency or engine. A laptop that was disabled when the router started was
    first seen with an empty disk; when it came back on demand with its models on disk and
    nothing loaded, the router kept the empty list, and refused every request as "no eligible
    host" until it was restarted."""
    import dataclasses

    pool.stop_supervisor()  # so nothing republishes over the row this test writes
    table = HostTable(pool.database)
    row = {r.host_id: r for r in table.all()}["local-1"]
    assert pool.state.hosts[0].residency == "pinned"

    # The supervisor's next word on the host: on demand, nothing loaded, the model on disk.
    table.publish(dataclasses.replace(
        row, resident=frozenset(), available=frozenset({MODEL}), residency="on_demand", engine="ollama",
        updated_at=time.time(),
    ))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        host = pool.state.hosts[0]
        if host.residency == "on_demand" and MODEL in host.available and not host.resident:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"the router never took the republished row: {vars(pool.state.hosts[0])}")

    # On disk on an on-demand host is servable: the engine loads it on first use.
    assert MODEL in pool.state.hosts[0].servable
    with pool.client() as client:
        response = client.post("/api/chat", json=chat())
    assert response.status_code == 200, response.text

    # And the engine is followed too: a row naming one that does not serve this path takes
    # the host out of the running for it, rather than handing it a request it would 404.
    table.publish(dataclasses.replace(row, engine="vllm", updated_at=time.time()))
    deadline = time.monotonic() + 10
    while pool.state.hosts[0].engine != "vllm" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pool.state.hosts[0].engine == "vllm"


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
        rented: {{ provider: fake, offer_policy: {{ min_disk_gb: 10, max_all_in_hourly: 0.5 }} }}
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


def test_a_supervisor_told_its_lock_was_taken_says_so(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = SupervisorLock(database, pool="p", owner="first", stale_after=0.2)
        first.acquire()
        assert first.beat() is True

        time.sleep(0.3)  # it stops beating; a successor takes the pool, as it should
        SupervisorLock(database, pool="p", owner="second", stale_after=0.2).acquire()

        assert first.beat() is False, "the first still believes it holds a lock it lost"
    finally:
        database.close()


def test_a_supervisor_whose_every_pass_fails_still_holds_its_lock(pool, monkeypatch):
    """Found live: a pass that raised every time never reached the heartbeat at its end, so the
    lock went stale under a running supervisor, a second one took it, and two ran at once."""
    supervisor = pool.supervisor
    supervisor.lock.stale_after = 0.5

    async def always_fails():
        raise RuntimeError("every pass raises")

    monkeypatch.setattr(supervisor, "pass_once", always_fails)
    monkeypatch.setattr(supervisor.config.pool, "probe_interval_s", 0.1)

    async def loop_for_a_while():
        import asyncio

        task = asyncio.create_task(supervisor.run_forever())
        await asyncio.sleep(1.2)  # more than two stale windows of nothing but failing passes
        supervisor._stopping = True
        await task

    pool.loop.run(loop_for_a_while())
    supervisor._stopping = False

    second = Supervisor(pool.config, pool.database)
    with pytest.raises(SupervisorBusy):
        pool.loop.run(second.start())


def test_a_supervisor_that_lost_its_lock_stops_rather_than_run_beside_the_new_one(pool, monkeypatch):
    supervisor = pool.supervisor
    monkeypatch.setattr(supervisor.config.pool, "probe_interval_s", 0.05)
    passes = []

    async def counted():
        passes.append(1)

    monkeypatch.setattr(supervisor, "pass_once", counted)
    # Someone else now owns the pool.
    pool.database.execute(
        "UPDATE supervisor_lock SET owner = ? WHERE pool = ?", ("somebody-else", supervisor.lock.pool)
    )

    pool.loop.run(supervisor.run_forever())  # returns by itself; it does not loop on

    assert passes == [], "it ran a pass on a pool it no longer holds"
