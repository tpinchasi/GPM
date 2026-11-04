"""The pool's own processes on every system it runs on, Windows included.

Written to pass on macOS, Linux and Windows alike; CI runs this file on Windows too. What can only
be played elsewhere — Windows' flags on a Mac, a lost console — is played with the module's own
`WINDOWS`, or by taking the console away.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
import types

import pytest
from gpm_server import cli, processes
from gpm_server.supervisor.service import Supervisor


def test_a_running_process_is_alive_and_one_that_ended_is_not():
    assert processes.alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not processes.alive(child.pid)
    assert not processes.alive(0) and not processes.alive(-1)


def test_a_process_is_known_by_its_command_line():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "gpm-marker"])
    try:
        deadline = time.monotonic() + 20
        said = None
        while time.monotonic() < deadline and not (said and "gpm-marker" in said):
            said = processes.command_line(child.pid)
        assert said and "gpm-marker" in said
    finally:
        child.kill()
        child.wait()


def test_a_child_of_the_pool_imports_from_the_path_this_process_was_given(tmp_path):
    """An embeddable Python ignores PYTHONPATH and takes packages installed with `pip --target`
    only from a path set as it runs; a child started with `-m gpm_server.cli` could not import
    the pool. The child is handed this process's path instead: here, a path whose first entry
    holds a stand-in for the pool, which only that path can reach."""
    stand_in = tmp_path / "elsewhere" / "gpm_server"
    stand_in.mkdir(parents=True)
    (stand_in / "__init__.py").write_text("")
    (stand_in / "cli.py").write_text("import sys\nprint('stand-in pool', sys.argv[1:])\n")
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    ran = subprocess.run(
        processes.pool_command("supervise", "-c", "pool.yaml", path=[str(tmp_path / "elsewhere"), *sys.path]),
        capture_output=True, text=True, timeout=60, env=environment, cwd=tmp_path,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "stand-in pool ['supervise', '-c', 'pool.yaml']"


def test_the_real_pool_answers_through_the_same_command():
    ran = subprocess.run(processes.pool_command("--version"), capture_output=True, text=True, timeout=120)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip(), "the pool's own command line ran and answered"


def test_detached_and_windowless_children_on_each_system(monkeypatch):
    monkeypatch.setattr(processes, "WINDOWS", False)
    assert processes.detached() == {"start_new_session": True} and processes.windowless() == {}
    monkeypatch.setattr(processes, "WINDOWS", True)
    assert processes.detached() == {"creationflags": processes.CREATE_NEW_PROCESS_GROUP | processes.CREATE_NO_WINDOW}
    assert processes.windowless() == {"creationflags": processes.CREATE_NO_WINDOW}


def test_under_pythonw_there_is_no_console_to_share(monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)
    assert not processes.has_console()


def test_a_supervisor_with_no_console_writes_its_own_log_and_its_last_words_are_said(tmp_path, monkeypatch, caplog):
    """Under `pythonw` the supervisor's traceback had nowhere to go: the router said only "the
    supervisor exited (code 1)"."""
    monkeypatch.setattr(processes, "has_console", lambda: False)
    monkeypatch.setattr(cli, "RESTART_FIRST_S", 3600.0)  # said, and not started again within the test
    monkeypatch.setattr(processes, "pool_command", lambda *args: [
        sys.executable, "-c", "import sys; print('ModuleNotFoundError: No module named gpm_server'); sys.exit(1)"])
    config = types.SimpleNamespace(request_log=str(tmp_path / "state" / "requests.sqlite3"))
    with caplog.at_level(logging.ERROR, logger="gpm"):
        supervised = cli._supervisor_child(config, str(tmp_path / "pool.yaml"), "info")
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not caplog.records:
                time.sleep(0.05)
        finally:
            supervised.stop()
    said = caplog.records[0].getMessage()
    assert "the supervisor exited (code 1)" in said and "Starting it again in 3600s" in said
    assert "ModuleNotFoundError: No module named gpm_server" in said
    assert (tmp_path / "state" / "gpm-supervisor.log").exists()


def test_a_supervisor_that_stops_is_started_again_and_never_once_the_router_goes(tmp_path, monkeypatch):
    """Found on a quick restart at a shop: the supervisor stopped, and nothing started it again
    until the whole pool was restarted — routing carried on, recovery did not."""
    runs = tmp_path / "runs"
    script = (
        "import pathlib, sys, time\n"
        f"runs = pathlib.Path({str(runs)!r})\n"
        "n = len(runs.read_text()) if runs.exists() else 0\n"
        "runs.write_text('x' * (n + 1))\n"
        "sys.exit(1) if n == 0 else time.sleep(60)\n"
    )
    monkeypatch.setattr(processes, "pool_command", lambda *args: [sys.executable, "-c", script])
    monkeypatch.setattr(cli, "RESTART_FIRST_S", 0.2)
    config = types.SimpleNamespace(request_log=str(tmp_path / "requests.sqlite3"))
    supervised = cli._supervisor_child(config, str(tmp_path / "pool.yaml"), "info")
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and supervised.restarts < 1:
            time.sleep(0.05)
        assert supervised.restarts == 1 and supervised.child.poll() is None, "started again, and running"
        second = supervised.child
    finally:
        supervised.stop()
    assert second.poll() is not None
    time.sleep(0.5)
    assert supervised.restarts == 1 and runs.read_text() == "xx", "never started again once stopped for good"


def test_the_supervisor_starts_where_the_event_loop_takes_no_signal_handlers(monkeypatch):
    """Windows' event loop raises NotImplementedError for `add_signal_handler`."""
    loop = asyncio.new_event_loop()
    loop_class = type(loop)
    loop.close()

    def refuse(self, *args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(loop_class, "add_signal_handler", refuse)
    ran = []

    async def run(config, database, config_path=None):
        ran.append(config_path)

    import gpm_server.state
    import gpm_server.supervisor

    monkeypatch.setattr(gpm_server.supervisor, "run", run)
    monkeypatch.setattr(gpm_server.state, "open_database", lambda config: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    assert cli._supervise(types.SimpleNamespace(config="pool.yaml")) == 0
    assert ran == ["pool.yaml"]


def install_facts(working=None, **install):
    return types.SimpleNamespace(agent=types.SimpleNamespace(reachable=True, facts={"engine": {"install": {
        "allowed": True, "engine": "ollama", "running": None, "error": None, "working": working, **install}}}))


def test_an_engine_its_agent_is_still_downloading_is_said_with_how_far_and_how_fast():
    """Found on a Windows install: 1.5 GB at 0.2 MB/s, and the pool said only "unreachable"."""
    said = Supervisor._agent_is_installing(None, install_facts(
        {"step": "downloading", "version": "0.40.13", "done": 312_000_000, "total": 1_500_000_000, "bytes_per_s": 200_000}))
    assert said == ("its agent is installing ollama 0.40.13: downloading 312.0 MB of 1.5 GB (20%) at 200 kB/s, "
                    "about 1.6 hours left")
    said = Supervisor._agent_is_installing(None, install_facts(
        {"step": "downloading", "version": "0.40.13", "done": 1_000_000, "total": None, "bytes_per_s": None}))
    assert said == "its agent is installing ollama 0.40.13: downloading 1.0 MB"


def test_an_engine_its_agent_could_not_install_yet_says_why():
    said = Supervisor._agent_is_installing(None, install_facts(error="fetching ollama-windows-amd64.zip stopped"))
    assert said == "its agent has not installed ollama yet: fetching ollama-windows-amd64.zip stopped"
    assert Supervisor._agent_is_installing(None, install_facts(running="0.40.13")) is None
    host = install_facts()
    host.agent.reachable = False
    assert Supervisor._agent_is_installing(None, host) is None


def test_the_agent_says_how_fast_a_download_goes(monkeypatch):
    step = "downloading"
    from gpm_agent import installs

    clock = [100.0]
    monkeypatch.setattr(installs.time, "monotonic", lambda: clock[0])
    installer = installs.Installer.__new__(installs.Installer)
    installer.working = None
    report = installer._progress("0.40.13")
    report(step, 0, 10_000_000)
    assert installer.working["bytes_per_s"] is None
    clock[0] += 10
    report(step, 2_000_000, 10_000_000)
    assert installer.working["bytes_per_s"] == 200_000
    report("unpacking", 0, None)
    assert installer.working["bytes_per_s"] is None, "a new step is measured afresh"


# --- whether a process lives, on every system (found on a quick restart on Windows) ---


def test_a_live_process_checked_is_still_running_afterwards():
    """On Windows `os.kill(pid, 0)` ended the process it was asked about."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert processes.alive(child.pid)
        time.sleep(0.2)
        assert child.poll() is None, "asked about, never ended"
    finally:
        child.kill()
        child.wait()
    assert not processes.alive(child.pid), "gone, and no exception for it"


@pytest.mark.skipif(os.name != "nt", reason="Windows' System process")
def test_a_process_not_ours_to_open_is_alive():
    assert processes.alive(4)


def test_the_agent_and_the_pool_agree_on_whether_a_process_lives():
    """The two packages depend on neither, so each has its copy of the check (D163): held equal here."""
    from gpm_agent import platform as host_platform
    from gpm_server import forwarder
    from gpm_server.db import _process_alive

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    for pid, expected in ((os.getpid(), True), (child.pid, False)):
        assert processes.alive(pid) is expected
        assert _process_alive(pid) is expected and forwarder.pid_alive(pid) is expected
        assert host_platform.alive(pid) is expected


def a_lock_held_by(tmp_path, pid):
    import socket

    from gpm_server.db import Database, SupervisorLock

    database = Database(tmp_path / "gpm.sqlite3")
    database.execute(
        "INSERT INTO supervisor_lock (pool, owner, pid, heartbeat) VALUES (?, ?, ?, ?)",
        ("shop", f"{socket.gethostname()}:{pid}:old", pid, time.time() - 5),
    )
    return database, SupervisorLock(database, "shop", f"{socket.gethostname()}:{os.getpid()}:new")


def test_a_supervisor_restarted_quickly_takes_the_lock_its_ended_predecessor_left(tmp_path):
    """The heartbeat is five seconds old, so whether its holder lives decides: on Windows, before
    0.37.1, asking raised OSError for an ended process, and the new supervisor crashed."""
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    database, lock = a_lock_held_by(tmp_path, ended.pid)
    try:
        lock.acquire()
        assert database.query("SELECT owner FROM supervisor_lock")[0]["owner"] == lock.owner
    finally:
        database.close()


def test_a_supervisor_still_running_keeps_the_lock_and_is_not_touched(tmp_path):
    from gpm_server.db import SupervisorBusy

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        database, lock = a_lock_held_by(tmp_path, holder.pid)
        try:
            with pytest.raises(SupervisorBusy):
                lock.acquire()
        finally:
            database.close()
        assert holder.poll() is None, "its holder still running"
    finally:
        holder.kill()
        holder.wait()


def test_the_status_says_whether_a_supervisor_is_there(tmp_path):
    from gpm_server.db import Database
    from gpm_server.router.app import _supervisor_now

    database = Database(tmp_path / "gpm.sqlite3")
    try:
        assert _supervisor_now(database, "shop") == {"running": False, "heartbeat_age_s": None}
        database.execute("INSERT INTO supervisor_lock (pool, owner, pid, heartbeat) VALUES (?, ?, ?, ?)",
                         ("shop", "here:1:x", 1, time.time() - 3))
        now = _supervisor_now(database, "shop")
        assert now["running"] is True and 2 <= now["heartbeat_age_s"] <= 10
        database.execute("UPDATE supervisor_lock SET heartbeat = ?", (time.time() - 120,))
        assert _supervisor_now(database, "shop")["running"] is False
    finally:
        database.close()
