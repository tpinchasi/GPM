"""`gpm forwarder`: the pool's SSH forwards, in a process that outlives the supervisor (D110).

The supervisor restarts with every release; a forward it runs itself ends with it, and every
answer in flight through it is cut. Found live: a supervisor restart cut nine answers and left
a rented host out of routing for 31 seconds. So the forwards live here instead, and the two
processes never call each other — the same rule as the router and the supervisor: the
supervisor writes the forwards it wants to the `forwards` table, and this reconciles.

Each pass it reads the rows and, for each one, keeps exactly one `ssh -N -L` on the row's
fixed local port — starting it, restarting a dropped one with a back-off, closing one whose
row is gone — and writes back how each is doing. It takes nothing from the network and
listens on nothing: the forwards bind loopback, and its only input is the local database.

**What a restart of the forwarder does** is the operator's choice (`forwarder.on_restart`):

- `reattach`: each `ssh` runs in a session of its own, so it outlives the forwarder. The next
  forwarder takes back each one that is still exactly the process it recorded — its pid, its
  command line, its port answering — and a restart cuts nothing.
- `close`: the forwards end with the forwarder, and the next one opens them afresh. A forward
  left by a `reattach` run is closed, not taken back.

What no mode can save is an answer in flight when an `ssh` itself dies — a dropped link, a
host rebooting. That is buffered delivery's case (D62), not this one's.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import TransportConfig
from .db import Database, SupervisorLock
from .transports.tunnel import build_ssh_command

log = logging.getLogger("gpm.forwarder")

#: Restart back-off, as the supervisor's own tunnels had it: doubling to this, and forgiven only
#: once a forward has held for a while — a forward that comes up and dies a second later would
#: otherwise reconnect every second, and a provider answers that by throttling SSH.
BACKOFF_MAX_S = 30.0
STEADY_AFTER_S = 30.0


def lock_name(pool: str) -> str:
    """The forwarder's row in the lock table: one forwarder per pool, as one supervisor."""
    return f"{pool}#forwarder"


def listening(port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, and not ours — which the command-line check then settles


def command_line(pid: int) -> Optional[str]:
    """A process's command line as the system reports it, or None where it cannot be read."""
    try:
        out = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip()
    return line or None


@dataclasses.dataclass
class Running:
    pid: int
    argv: list[str]
    #: The child object where this forwarder started it; None for one taken back after a
    #: restart, which is watched by its pid instead.
    child: Optional[subprocess.Popen] = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    up_since: Optional[float] = None


class Forwarder:
    def __init__(
        self,
        database: Database,
        pool: str,
        on_restart: str,
        state_dir: Path,
        *,
        command_builder: Callable[[TransportConfig, int], list[str]] = build_ssh_command,
        owner: Optional[str] = None,
    ):
        if on_restart not in ("reattach", "close"):
            raise ValueError(f"on_restart must be 'reattach' or 'close', not {on_restart!r}")
        self.db = database
        self.on_restart = on_restart
        self.logs = Path(state_dir) / "forwards"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.command_builder = command_builder
        self.lock = SupervisorLock(
            database, lock_name(pool),
            owner or f"{socket.gethostname()}:{os.getpid()}:{os.urandom(4).hex()}", stale_after=15.0,
        )
        self.running: dict[str, Running] = {}
        self._backoff: dict[str, float] = {}
        self._next_start: dict[str, float] = {}
        #: Forwards this forwarder has started at least once, so a start after that is counted
        #: as a restart.
        self._started: set[str] = set()
        self._stopping = False

    # --- the rows ---

    def _rows(self) -> dict[str, Any]:
        return {row["name"]: row for row in self.db.query("SELECT * FROM forwards")}

    def _report(self, name: str, state: str, running: Optional[Running] = None,
                last_error: Optional[str] = None, restarted: bool = False) -> None:
        self.db.execute(
            "UPDATE forwards SET state = ?, pid = ?, argv = ?, updated_at = ?,"
            " last_error = COALESCE(?, last_error), restarts = restarts + ? WHERE name = ?",
            (state, running.pid if running else None,
             json.dumps(running.argv) if running else None, time.time(),
             last_error, 1 if restarted else 0, name),
        )

    def _argv(self, row: Any) -> list[str]:
        return self.command_builder(self._transport(row), int(row["local_port"]))

    @staticmethod
    def _transport(row: Any) -> TransportConfig:
        return TransportConfig.model_validate(json.loads(row["transport"]))

    def _log_path(self, name: str) -> Path:
        return self.logs / (name.replace("/", "_").replace(":", "_") + ".log")

    def _last_error(self, name: str) -> Optional[str]:
        try:
            lines = [line for line in self._log_path(name).read_text(errors="replace").splitlines() if line.strip()]
        except OSError:
            return None
        return lines[-1][:300] if lines else None

    # --- processes ---

    def _is_ours(self, pid: int, argv: list[str]) -> bool:
        """Still exactly the process recorded: alive, and running the recorded command. A pid
        reused by something else since must never be taken back, or signalled."""
        return pid_alive(pid) and command_line(pid) == " ".join(argv)

    def _alive(self, running: Running) -> bool:
        if running.child is not None:
            return running.child.poll() is None
        return self._is_ours(running.pid, running.argv)

    def _start(self, name: str, argv: list[str]) -> Running:
        out = open(self._log_path(name), "ab")
        child = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=out,
            # Its own session is what lets it outlive this process; `close` keeps it in ours.
            start_new_session=self.on_restart == "reattach",
        )
        out.close()
        log.info("forward %s started (pid %d)", name, child.pid)
        return Running(pid=child.pid, argv=argv, child=child)

    def _end(self, running: Running) -> None:
        if running.child is not None:
            if running.child.poll() is None:
                running.child.terminate()
                try:
                    running.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    running.child.kill()
            return
        if self._is_ours(running.pid, running.argv):
            os.kill(running.pid, signal.SIGTERM)

    # --- starting up, and going ---

    def take_back(self) -> None:
        """At start: the forwards a previous forwarder left running. Under `reattach`, each one
        still exactly as recorded, and answering on its port, is taken back as it is; one that
        is ours but not answering is ended and started again. Under `close`, each is ended."""
        for name, row in self._rows().items():
            if not row["pid"] or not row["argv"]:
                continue
            argv = json.loads(row["argv"])
            pid = int(row["pid"])
            if not self._is_ours(pid, argv):
                continue
            left = Running(pid=pid, argv=argv)
            if self.on_restart == "reattach" and argv == self._argv(row) and listening(int(row["local_port"])):
                left.up_since = time.monotonic()
                self.running[name] = left
                log.info("forward %s taken back (pid %d)", name, pid)
            else:
                self._end(left)
                log.info("forward %s left by the last forwarder ended (pid %d)", name, pid)

    def shutdown(self) -> None:
        """Going. Under `reattach` the forwards stay, recorded for the next forwarder; under
        `close` they go with this one."""
        if self.on_restart == "close":
            for name, running in list(self.running.items()):
                self._end(running)
                self._report(name, "down")
            self.running.clear()
        self.lock.release()

    # --- the pass ---

    def reconcile(self, now: Optional[float] = None) -> None:
        """One pass: every wanted forward running and reported, every unwanted one ended."""
        now = time.monotonic() if now is None else now
        rows = self._rows()
        for name in [n for n in self.running if n not in rows]:
            self._end(self.running.pop(name))
            log.info("forward %s closed: no longer wanted", name)
        for name, row in rows.items():
            argv = self._argv(row)
            running = self.running.get(name)
            if running is not None and running.argv != argv:
                # The forward moved (a new address, a new port): the old one goes first.
                self._end(self.running.pop(name))
                running = None
            if running is not None and not self._alive(running):
                self.running.pop(name)
                held = running.up_since is not None and now - running.up_since >= STEADY_AFTER_S
                backoff = 1.0 if held else min(self._backoff.get(name, 0.5) * 2, BACKOFF_MAX_S)
                self._backoff[name] = backoff
                self._next_start[name] = now + backoff
                self._report(name, "down", last_error=self._last_error(name) or "ssh exited")
                log.warning("forward %s went down; again in %.0fs", name, backoff)
                running = None
            if running is None:
                if now < self._next_start.get(name, 0.0):
                    continue
                # The host key is pinned on first use into a file the pool owns.
                Path(self._transport(row).known_hosts).expanduser().parent.mkdir(parents=True, exist_ok=True)
                try:
                    running = self._start(name, argv)
                except OSError as exc:  # no ssh binary, most likely
                    self._report(name, "down", last_error=str(exc))
                    self._next_start[name] = now + BACKOFF_MAX_S
                    continue
                self.running[name] = running
                self._report(name, "wanted", running, restarted=name in self._started)
                self._started.add(name)
            if listening(int(row["local_port"])):
                if running.up_since is None:
                    running.up_since = now
                if row["state"] != "up" or row["pid"] != running.pid:
                    self._report(name, "up", running, last_error="")
            elif row["state"] == "up":
                self._report(name, "wanted", running)

    def run(self, interval: float = 1.0) -> None:
        self.lock.acquire()
        self.take_back()
        try:
            while not self._stopping:
                self.reconcile()
                if not self.lock.beat():
                    log.error("another forwarder took this pool's forwards; stopping")
                    return
                time.sleep(interval)
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._stopping = True
