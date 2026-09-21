"""SQLite state — the one thing the router and the supervisor share.

docs/spec/supervisor.md §1: two processes, one database in WAL mode, and they never call each
other. The supervisor publishes the host table, leases and decisions; the router reads the host
table and writes the request log and per-host counters, which is how the supervisor learns about
idleness.

**No secret is ever written here.** The router builds its own credentials from configuration;
what crosses this boundary is a URL to dial and state. **No prompt or completion text either** —
that is enforced by the shape of `RequestRecord`, not by the caller's discipline
(hosts-routing-capacity.md §6, threat model T16).

The file is created owner-readable only, and the pool refuses to open one that is group- or
world-readable (threat model T19).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import socket
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    request_id      TEXT NOT NULL,
    session_id      TEXT,
    host_id         TEXT,
    worker_id       TEXT,
    model_requested TEXT,
    model_served    TEXT,
    runtime_class   TEXT,
    queue_wait_ms   REAL,
    latency_ms      REAL,
    status_code     INTEGER,
    outcome         TEXT NOT NULL,
    reason          TEXT,
    -- What the engine reports it generated, and how long it spent generating (D67).
    tokens_out      INTEGER,
    generate_ms     REAL
);
CREATE INDEX IF NOT EXISTS request_log_session ON request_log (session_id, ts);

-- Published by the supervisor, read by the router. The router never writes it.
CREATE TABLE IF NOT EXISTS hosts (
    host_id        TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    transport_type TEXT NOT NULL,
    priority       INTEGER NOT NULL,
    dial_url       TEXT NOT NULL,
    state          TEXT NOT NULL,
    workers        INTEGER NOT NULL,
    capabilities   TEXT NOT NULL,   -- JSON array
    variants       TEXT NOT NULL,   -- JSON {logical: [[tag, runtime_class, enforces_schema], ...]}
    resident       TEXT NOT NULL,   -- JSON array of tags loaded right now
    available      TEXT NOT NULL DEFAULT '[]',       -- JSON array of tags on disk
    residency      TEXT NOT NULL DEFAULT 'pinned',   -- pinned | on_demand
    lease_id       TEXT,
    provider_ref   TEXT,            -- JSON, rented hosts only
    hourly_rate    REAL,
    last_error     TEXT,
    updated_at     REAL NOT NULL
);

-- Written by the router, read by the supervisor: what it needs to spot an idle host.
CREATE TABLE IF NOT EXISTS host_counters (
    host_id         TEXT PRIMARY KEY,
    busy            INTEGER NOT NULL DEFAULT 0,
    total           INTEGER NOT NULL DEFAULT 0,
    requests_served INTEGER NOT NULL DEFAULT 0,
    failures        INTEGER NOT NULL DEFAULT 0,
    last_request_at REAL,
    updated_at      REAL NOT NULL
);

-- A lease is the only thing that can spend (supervisor.md §2).
CREATE TABLE IF NOT EXISTS leases (
    lease_id        TEXT PRIMARY KEY,
    workers         INTEGER NOT NULL,
    max_hours       REAL NOT NULL,
    max_spend       REAL NOT NULL,
    allow_rent      INTEGER NOT NULL,
    bid_ceiling     REAL,
    state           TEXT NOT NULL,
    opened_at       REAL NOT NULL,
    closed_at       REAL,
    closed_reason   TEXT
);

-- Every decision, with the numbers that produced it (supervisor.md §3).
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,
    host_id  TEXT,
    lease_id TEXT,
    summary  TEXT NOT NULL,
    numbers  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);

-- Append-only cost events per host and lease (supervisor.md §4).
CREATE TABLE IF NOT EXISTS spend (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    lease_id TEXT,
    host_id  TEXT,
    source   TEXT NOT NULL,   -- "estimate" | "reported"
    amount   REAL NOT NULL
);

-- One row per pool. Exactly one supervisor runs at a time (supervisor.md §1).
CREATE TABLE IF NOT EXISTS supervisor_lock (
    pool       TEXT PRIMARY KEY,
    owner      TEXT NOT NULL,
    pid        INTEGER NOT NULL,
    heartbeat  REAL NOT NULL
);
"""


def _secure_path(path: Path) -> None:
    """Owner-readable only, and refuse a file someone else can already read."""
    if not path.exists():
        path.touch(mode=0o600)
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            raise PermissionError(
                f"{path} is readable by others and could not be tightened: {exc}"
            ) from exc


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _secure_path(path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    _add_missing_columns(conn)
    conn.commit()
    return conn


# Columns added after a table first shipped. `CREATE TABLE IF NOT EXISTS` leaves an existing
# file alone, so each is added on open when absent; the default is what a row written by an
# older process would have meant.
_ADDED_COLUMNS = {
    "request_log": [
        # Judging a host by throughput needs what it generated, not only how long it took:
        # latency alone cannot tell a slow machine from a long answer (D67).
        ("tokens_out", "INTEGER"),
        ("generate_ms", "REAL"),
    ],
    "hosts": [
        ("available", "TEXT NOT NULL DEFAULT '[]'"),
        ("residency", "TEXT NOT NULL DEFAULT 'pinned'"),
    ],
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns:
            if name not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


class Database:
    """A connection plus the lock that keeps writes off each other's toes.

    Both processes open the same file; SQLite in WAL mode handles the rest.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = connect(path)
        self._conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --- the request log (router) ---


@dataclasses.dataclass(frozen=True)
class RequestRecord:
    request_id: str
    outcome: str
    ts: float = dataclasses.field(default_factory=time.time)
    session_id: Optional[str] = None
    host_id: Optional[str] = None
    worker_id: Optional[str] = None
    model_requested: Optional[str] = None
    model_served: Optional[str] = None
    runtime_class: Optional[str] = None
    queue_wait_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    status_code: Optional[int] = None
    reason: Optional[str] = None
    #: What the engine says it generated, where it says so (D67).
    tokens_out: Optional[int] = None
    generate_ms: Optional[float] = None


class RequestLog:
    def __init__(self, database: Database):
        self.db = database

    def _insert(self, record: RequestRecord) -> None:
        self.db.execute(
            """
            INSERT INTO request_log (
                ts, request_id, session_id, host_id, worker_id, model_requested,
                model_served, runtime_class, queue_wait_ms, latency_ms, status_code,
                outcome, reason, tokens_out, generate_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.ts,
                record.request_id,
                record.session_id,
                record.host_id,
                record.worker_id,
                record.model_requested,
                record.model_served,
                record.runtime_class,
                record.queue_wait_ms,
                record.latency_ms,
                record.status_code,
                record.outcome,
                record.reason,
                record.tokens_out,
                record.generate_ms,
            ),
        )

    async def record(self, record: RequestRecord) -> None:
        """Write off the event loop: nothing blocking goes on the router's request path."""
        await asyncio.to_thread(self._insert, record)

    def rows(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM request_log ORDER BY id DESC LIMIT ?", (limit,))


# --- the host table (supervisor writes, router reads) ---


@dataclasses.dataclass(frozen=True)
class HostRow:
    host_id: str
    kind: str
    transport_type: str
    priority: int
    dial_url: str
    state: str
    workers: int
    capabilities: tuple[str, ...]
    #: logical name -> variants in preference order, each (tag, runtime_class, enforces_schema)
    variants: dict[str, tuple[tuple[str, str, Optional[bool]], ...]]
    resident: frozenset[str]
    available: frozenset[str] = frozenset()
    residency: str = "pinned"
    lease_id: Optional[str] = None
    provider_ref: Optional[dict[str, Any]] = None
    hourly_rate: Optional[float] = None
    last_error: Optional[str] = None
    updated_at: float = 0.0


def _row_to_host(row: sqlite3.Row) -> HostRow:
    return HostRow(
        host_id=row["host_id"],
        kind=row["kind"],
        transport_type=row["transport_type"],
        priority=row["priority"],
        dial_url=row["dial_url"],
        state=row["state"],
        workers=row["workers"],
        capabilities=tuple(json.loads(row["capabilities"])),
        variants={
            name: tuple(tuple(v) for v in variants)
            for name, variants in json.loads(row["variants"]).items()
        },
        resident=frozenset(json.loads(row["resident"])),
        available=frozenset(json.loads(row["available"])),
        residency=row["residency"],
        lease_id=row["lease_id"],
        provider_ref=json.loads(row["provider_ref"]) if row["provider_ref"] else None,
        hourly_rate=row["hourly_rate"],
        last_error=row["last_error"],
        updated_at=row["updated_at"],
    )


class HostTable:
    def __init__(self, database: Database):
        self.db = database

    def publish(self, host: HostRow) -> None:
        self.db.execute(
            """
            INSERT INTO hosts (
                host_id, kind, transport_type, priority, dial_url, state, workers,
                capabilities, variants, resident, available, residency, lease_id,
                provider_ref, hourly_rate, last_error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(host_id) DO UPDATE SET
                kind=excluded.kind, transport_type=excluded.transport_type,
                priority=excluded.priority, dial_url=excluded.dial_url,
                state=excluded.state, workers=excluded.workers,
                capabilities=excluded.capabilities, variants=excluded.variants,
                resident=excluded.resident, available=excluded.available,
                residency=excluded.residency, lease_id=excluded.lease_id,
                provider_ref=excluded.provider_ref, hourly_rate=excluded.hourly_rate,
                last_error=excluded.last_error, updated_at=excluded.updated_at
            """,
            (
                host.host_id,
                host.kind,
                host.transport_type,
                host.priority,
                host.dial_url,
                host.state,
                host.workers,
                json.dumps(list(host.capabilities)),
                json.dumps({name: [list(v) for v in vs] for name, vs in host.variants.items()}),
                json.dumps(sorted(host.resident)),
                json.dumps(sorted(host.available)),
                host.residency,
                host.lease_id,
                json.dumps(host.provider_ref) if host.provider_ref is not None else None,
                host.hourly_rate,
                host.last_error,
                time.time(),
            ),
        )

    def remove(self, host_id: str) -> None:
        self.db.execute("DELETE FROM hosts WHERE host_id = ?", (host_id,))

    def all(self) -> list[HostRow]:
        return [_row_to_host(row) for row in self.db.query("SELECT * FROM hosts")]

    def revision(self) -> float:
        """The newest publish time — a cheap "has anything changed?" for the router."""
        rows = self.db.query("SELECT COUNT(*) AS n, MAX(updated_at) AS newest FROM hosts")
        row = rows[0]
        return float(row["newest"] or 0.0) + float(row["n"] or 0)


# --- per-host counters (router writes, supervisor reads) ---


@dataclasses.dataclass(frozen=True)
class CounterRow:
    host_id: str
    busy: int
    total: int
    requests_served: int
    failures: int
    last_request_at: Optional[float]


class HostCounters:
    def __init__(self, database: Database):
        self.db = database

    def _write(self, counter: CounterRow) -> None:
        self.db.execute(
            """
            INSERT INTO host_counters (
                host_id, busy, total, requests_served, failures, last_request_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(host_id) DO UPDATE SET
                busy=excluded.busy, total=excluded.total,
                requests_served=excluded.requests_served, failures=excluded.failures,
                last_request_at=excluded.last_request_at, updated_at=excluded.updated_at
            """,
            (
                counter.host_id,
                counter.busy,
                counter.total,
                counter.requests_served,
                counter.failures,
                counter.last_request_at,
                time.time(),
            ),
        )

    async def publish(self, counters: list[CounterRow]) -> None:
        def write_all() -> None:
            for counter in counters:
                self._write(counter)

        await asyncio.to_thread(write_all)

    def all(self) -> dict[str, CounterRow]:
        rows = self.db.query("SELECT * FROM host_counters")
        return {
            row["host_id"]: CounterRow(
                host_id=row["host_id"],
                busy=row["busy"],
                total=row["total"],
                requests_served=row["requests_served"],
                failures=row["failures"],
                last_request_at=row["last_request_at"],
            )
            for row in rows
        }


# --- the supervisor's lock ---


class SupervisorBusy(Exception):
    """Another supervisor holds this pool."""


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class SupervisorLock:
    """Exactly one supervisor per pool. A lock is stale — and may be taken — when its holder
    has stopped refreshing its heartbeat for `stale_after`, or sooner when the holder was a
    process on this host that no longer exists. A crash must not lock the pool out, and
    neither must a restart."""

    def __init__(self, database: Database, pool: str, owner: str, stale_after: float = 60.0):
        self.db = database
        self.pool = pool
        self.owner = owner
        self.stale_after = stale_after

    def _held_by_someone_else(self, row: Any) -> bool:
        if row["owner"] == self.owner:
            return False
        if time.time() - row["heartbeat"] >= self.stale_after:
            return False
        # The owner string is "<hostname>:<pid>:<nonce>"; a dead pid on this host is stale.
        holder_host = str(row["owner"]).split(":", 1)[0]
        if holder_host == socket.gethostname() and not _process_alive(int(row["pid"])):
            return False
        return True

    def acquire(self) -> None:
        now = time.time()
        rows = self.db.query("SELECT * FROM supervisor_lock WHERE pool = ?", (self.pool,))
        if rows and self._held_by_someone_else(rows[0]):
            row = rows[0]
            raise SupervisorBusy(
                f"another supervisor (pid {row['pid']}) holds pool {self.pool!r}"
            )
        self.db.execute(
            """
            INSERT INTO supervisor_lock (pool, owner, pid, heartbeat) VALUES (?, ?, ?, ?)
            ON CONFLICT(pool) DO UPDATE SET
                owner=excluded.owner, pid=excluded.pid, heartbeat=excluded.heartbeat
            """,
            (self.pool, self.owner, os.getpid(), now),
        )

    def beat(self) -> None:
        self.db.execute(
            "UPDATE supervisor_lock SET heartbeat = ? WHERE pool = ? AND owner = ?",
            (time.time(), self.pool, self.owner),
        )

    def release(self) -> None:
        self.db.execute(
            "DELETE FROM supervisor_lock WHERE pool = ? AND owner = ?", (self.pool, self.owner)
        )
