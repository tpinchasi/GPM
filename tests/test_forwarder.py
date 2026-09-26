"""The forwarder: the pool's SSH forwards, in a process that outlives the supervisor (D110).

Found live: restarting the supervisor to deploy a fix cut nine answers in flight on a rented
host and kept it out of routing for 31 seconds, because the supervisor's forwards were its
children. Against the `ssh -N -L` stand-in (tests/fakes/stub_forwarder.py) and a fake engine:
no SSH server, no GPU, no provider.
"""

import http.client
import json
import os
import signal
import time

import pytest
from fakes.fake_ollama import FakeOllama
from fakes.harness import BackgroundLoop, ServerHandle, stub_ssh_command
from gpm_server.config import PoolConfig, TransportConfig
from gpm_server.db import Database
from gpm_server.forwarder import Forwarder, command_line, listening, lock_name, pid_alive
from gpm_server.transports.forwarded import ForwardedTunnel


@pytest.fixture
def engine():
    loop = BackgroundLoop()
    fake = FakeOllama(resident={"m1"})
    server = ServerHandle(fake.app, loop)
    try:
        yield server
    finally:
        server.stop()
        loop.stop()


@pytest.fixture
def database(tmp_path):
    db = Database(tmp_path / "gpm.sqlite3")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def cleanup():
    """Every pid a test leaves running on purpose, ended afterwards."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def transport(engine, **overrides):
    base = {"type": "tunnel", "ssh_host": "127.0.0.1", "remote_port": engine.port,
            "known_hosts": "/tmp/gpm-test-known-hosts"}
    return TransportConfig.model_validate({**base, **overrides})


def forwarder(database, tmp_path, on_restart="reattach"):
    return Forwarder(database, "pool", on_restart, tmp_path, command_builder=stub_ssh_command)


def until(predicate, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def row(database, name):
    rows = database.query("SELECT * FROM forwards WHERE name = ?", (name,))
    return rows[0] if rows else None


async def want(database, engine, name="rented-1"):
    tunnel = ForwardedTunnel(name, transport(engine), database)
    await tunnel.start(wait_s=0)
    return tunnel


# --- the supervisor's side: a row, not a process ---


async def test_a_forward_is_asked_for_and_keeps_its_port_across_supervisors(database, engine):
    first = await want(database, engine)
    assert first.local_port and first.local_url == f"http://127.0.0.1:{first.local_port}"
    assert not first.reused

    # The next supervisor asks for the same forward: same port, so the router's address holds.
    second = await want(database, engine)
    assert second.local_port == first.local_port and second.reused


async def test_a_forward_that_moved_keeps_its_port_but_is_not_taken_as_the_same(database, engine):
    first = await want(database, engine)
    moved = ForwardedTunnel("rented-1", transport(engine, ssh_port=2222), database)
    await moved.start(wait_s=0)
    assert moved.local_port == first.local_port and not moved.reused
    assert json.loads(row(database, "rented-1")["transport"])["ssh_port"] == 2222


async def test_a_supervisor_going_leaves_the_forward_and_a_released_host_removes_it(database, engine):
    tunnel = await want(database, engine)
    await tunnel.detach()
    assert row(database, "rented-1") is not None, "the next supervisor finds it"
    await tunnel.stop()
    assert row(database, "rented-1") is None, "the forwarder closes it"


# --- the forwarder ---


async def test_a_wanted_forward_is_brought_up_and_carries_traffic(database, engine, tmp_path):
    tunnel = await want(database, engine)
    keeper = forwarder(database, tmp_path, on_restart="close")
    try:
        keeper.reconcile()
        assert until(lambda: listening(tunnel.local_port))
        keeper.reconcile()
        assert row(database, "rented-1")["state"] == "up" and tunnel.up
        conn = http.client.HTTPConnection("127.0.0.1", tunnel.local_port, timeout=5)
        conn.request("GET", "/api/tags")
        assert conn.getresponse().status == 200
    finally:
        keeper.shutdown()


async def test_a_forward_no_longer_wanted_is_closed(database, engine, tmp_path):
    tunnel = await want(database, engine)
    keeper = forwarder(database, tmp_path, on_restart="close")
    try:
        keeper.reconcile()
        assert until(lambda: listening(tunnel.local_port))
        pid = keeper.running["rented-1"].pid
        await tunnel.stop()
        keeper.reconcile()
        assert until(lambda: not pid_alive(pid) or keeper.running.get("rented-1") is None)
        assert "rented-1" not in keeper.running
        assert until(lambda: not listening(tunnel.local_port))
    finally:
        keeper.shutdown()


async def test_under_reattach_a_forwarder_restart_cuts_nothing(database, engine, tmp_path, cleanup):
    """The point of the mode: an answer in flight through the forward is not cut when the
    forwarder restarts — the `ssh` outlives it, and the next forwarder takes it back."""
    tunnel = await want(database, engine)
    first = forwarder(database, tmp_path, on_restart="reattach")
    first.reconcile()
    assert until(lambda: listening(tunnel.local_port))
    first.reconcile()
    pid = first.running["rented-1"].pid
    cleanup.append(pid)

    conn = http.client.HTTPConnection("127.0.0.1", tunnel.local_port, timeout=5)
    conn.request("GET", "/api/tags")
    assert conn.getresponse().read() is not None
    before = conn.sock

    first.shutdown()  # the forwarder goes; under reattach, its forwards do not
    assert pid_alive(pid)

    second = forwarder(database, tmp_path, on_restart="reattach")
    second.take_back()
    assert second.running["rented-1"].pid == pid, "taken back, not started again"
    second.reconcile()
    assert second.running["rented-1"].pid == pid and row(database, "rented-1")["state"] == "up"

    conn.request("GET", "/api/tags")
    assert conn.getresponse().status == 200 and conn.sock is before, "the same connection carried on"
    second.shutdown()


async def test_under_close_the_forwards_end_with_the_forwarder(database, engine, tmp_path):
    tunnel = await want(database, engine)
    keeper = forwarder(database, tmp_path, on_restart="close")
    keeper.reconcile()
    assert until(lambda: listening(tunnel.local_port))
    pid = keeper.running["rented-1"].pid
    keeper.shutdown()
    assert until(lambda: not pid_alive(pid))
    assert row(database, "rented-1")["state"] == "down"


async def test_a_close_forwarder_ends_what_a_reattach_one_left_and_starts_afresh(database, engine, tmp_path, cleanup):
    tunnel = await want(database, engine)
    left = forwarder(database, tmp_path, on_restart="reattach")
    left.reconcile()
    assert until(lambda: listening(tunnel.local_port))
    left.reconcile()
    old = left.running["rented-1"].pid
    child = left.running["rented-1"].child
    cleanup.append(old)
    left.shutdown()

    fresh = forwarder(database, tmp_path, on_restart="close")
    fresh.take_back()
    # Ended — and reaped here only because this test is its parent; a real forwarder that
    # left it is gone, and the system reaps it.
    assert until(lambda: child.poll() is not None), "a close forwarder takes nothing back"
    fresh.reconcile()
    assert until(lambda: listening(tunnel.local_port))
    assert fresh.running["rented-1"].pid != old
    fresh.shutdown()


async def test_a_pid_now_used_by_something_else_is_never_taken_back_or_signalled(database, engine, tmp_path):
    """A recorded pid may since belong to an unrelated process. It must match the recorded
    command line exactly before it is adopted — or ended."""
    await want(database, engine)
    database.execute("UPDATE forwards SET pid = ?, argv = ? WHERE name = 'rented-1'",
                     (os.getpid(), json.dumps(["ssh", "-N", "something", "else"])))
    keeper = forwarder(database, tmp_path)
    keeper.take_back()
    assert "rented-1" not in keeper.running
    assert pid_alive(os.getpid())  # this test process, untouched
    assert command_line(os.getpid()) is not None


async def test_a_dropped_forward_is_brought_back_after_a_back_off(database, engine, tmp_path):
    tunnel = await want(database, engine)
    keeper = forwarder(database, tmp_path, on_restart="close")
    try:
        keeper.reconcile()
        assert until(lambda: listening(tunnel.local_port))
        first = keeper.running["rented-1"]
        first.child.kill()
        first.child.wait()
        now = time.monotonic()
        keeper.reconcile(now=now)
        assert row(database, "rented-1")["state"] == "down" and "rented-1" not in keeper.running
        keeper.reconcile(now=now + 0.1)
        assert "rented-1" not in keeper.running, "not straight back: it waits out its back-off"
        keeper.reconcile(now=now + 5)
        assert until(lambda: listening(tunnel.local_port))
        assert keeper.running["rented-1"].pid != first.pid
        assert row(database, "rented-1")["restarts"] >= 1
    finally:
        keeper.shutdown()


# --- the supervisor keeps the forwarder running ---


HOSTS = [{"id": "local-1", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}]


def pool(enabled=True, **forwarder):
    return PoolConfig.model_validate({
        "pool": {"name": "pool", "model_set": ["m1"], "probe_interval_s": 3600},
        "auth": {"app_keys": ["k"]},
        "hosts": HOSTS,
        "forwarder": {"enabled": enabled, **forwarder},
    })


def test_the_supervisor_starts_a_missing_forwarder_once_and_not_a_running_one(database):
    from gpm_server.supervisor import Supervisor

    supervisor = Supervisor(pool(), database)
    started = []
    supervisor.spawn_forwarder = lambda: started.append(time.time())
    supervisor.ensure_forwarder()
    supervisor.ensure_forwarder()
    assert len(started) == 1, "given time to come up, not started every pass"

    database.execute("INSERT INTO supervisor_lock (pool, owner, pid, heartbeat) VALUES (?, 'x', 1, ?)",
                     (lock_name("pool"), time.time()))
    supervisor._forwarder_started_at = 0.0
    supervisor.ensure_forwarder()
    assert len(started) == 1, "one is running: nothing is started"


def test_no_forwarder_is_started_where_the_pool_has_none_or_starts_its_own(database):
    from gpm_server.supervisor import Supervisor

    for config in (pool(enabled=False), pool(start_automatically=False)):
        supervisor = Supervisor(config, database)
        supervisor.spawn_forwarder = lambda: pytest.fail("started a forwarder it should not have")
        supervisor.ensure_forwarder()


def test_the_forwarder_is_off_unless_configured_and_reattaches_by_default():
    config = PoolConfig.model_validate({"pool": {"name": "p", "model_set": ["m"]}, "auth": {"app_keys": ["k"]},
                                        "hosts": HOSTS})
    assert config.forwarder.enabled is False
    assert config.forwarder.on_restart == "reattach"
    with pytest.raises(ValueError):
        pool(on_restart="sometimes")
