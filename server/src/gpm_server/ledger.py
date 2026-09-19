"""Leases, decisions and money — the supervisor's records.

docs/spec/supervisor.md §2 (a lease is the only thing that can spend), §3 (every decision is
recorded with the numbers that produced it) and §4 (spend is reconciled, not just estimated).
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from typing import Any, Optional

from .db import Database


class LeaseRefused(Exception):
    """A lease that would spend without saying how much, or one that loosens a pool limit."""


@dataclasses.dataclass(frozen=True)
class Lease:
    lease_id: str
    workers: int
    max_hours: float
    max_spend: float
    allow_rent: bool
    state: str = "open"
    bid_ceiling: Optional[float] = None
    opened_at: float = dataclasses.field(default_factory=time.time)
    closed_at: Optional[float] = None
    closed_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def hours_left(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.max_hours - (now - self.opened_at) / 3600)

    def expired(self, now: Optional[float] = None) -> bool:
        return self.hours_left(now) <= 0


class LeaseStore:
    def __init__(self, database: Database):
        self.db = database

    def open(
        self,
        *,
        workers: int,
        max_hours: float,
        max_spend: Optional[float],
        allow_rent: bool,
        bid_ceiling: Optional[float] = None,
        pool_bid_ceiling: Optional[float] = None,
        lease_id: Optional[str] = None,
    ) -> Lease:
        """A lease that can rent **must** carry a dollar cap, and may tighten the pool's
        configured limits, never loosen them (spec §2)."""
        if allow_rent and max_spend is None:
            raise LeaseRefused(
                "a lease that can rent must state a dollar cap: pass --max-spend"
            )
        if max_spend is not None and max_spend <= 0:
            raise LeaseRefused("a dollar cap must be above zero")
        if max_hours <= 0:
            raise LeaseRefused("a lease must have a time limit above zero")
        if bid_ceiling is not None and pool_bid_ceiling is not None and bid_ceiling > pool_bid_ceiling:
            raise LeaseRefused(
                f"a lease may only tighten the bid ceiling: ${bid_ceiling:.3f} is above the "
                f"pool's ${pool_bid_ceiling:.3f}"
            )

        lease = Lease(
            lease_id=lease_id or f"lease-{uuid.uuid4().hex[:8]}",
            workers=workers,
            max_hours=max_hours,
            max_spend=max_spend if max_spend is not None else 0.0,
            allow_rent=allow_rent,
            bid_ceiling=bid_ceiling,
        )
        self.db.execute(
            """
            INSERT INTO leases (
                lease_id, workers, max_hours, max_spend, allow_rent, bid_ceiling,
                state, opened_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lease.lease_id,
                lease.workers,
                lease.max_hours,
                lease.max_spend,
                int(lease.allow_rent),
                lease.bid_ceiling,
                lease.state,
                lease.opened_at,
            ),
        )
        return lease

    def close(self, lease_id: str, reason: str = "closed by operator") -> None:
        self.db.execute(
            "UPDATE leases SET state = 'closed', closed_at = ?, closed_reason = ? "
            "WHERE lease_id = ? AND state = 'open'",
            (time.time(), reason, lease_id),
        )

    def tighten(
        self,
        lease_id: str,
        *,
        max_spend: Optional[float] = None,
        max_hours: Optional[float] = None,
        workers: Optional[int] = None,
    ) -> Lease:
        lease = self.get(lease_id)
        if lease is None:
            raise LeaseRefused(f"no lease {lease_id!r}")
        if max_spend is not None and max_spend > lease.max_spend:
            raise LeaseRefused("a lease may only be tightened: the dollar cap cannot be raised")
        if max_hours is not None and max_hours > lease.max_hours:
            raise LeaseRefused("a lease may only be tightened: the time limit cannot be raised")
        if workers is not None and workers > lease.workers:
            raise LeaseRefused("a lease may only be tightened: workers cannot be raised")
        self.db.execute(
            "UPDATE leases SET max_spend = ?, max_hours = ?, workers = ? WHERE lease_id = ?",
            (
                max_spend if max_spend is not None else lease.max_spend,
                max_hours if max_hours is not None else lease.max_hours,
                workers if workers is not None else lease.workers,
                lease_id,
            ),
        )
        return self.get(lease_id)  # type: ignore[return-value]

    def get(self, lease_id: str) -> Optional[Lease]:
        rows = self.db.query("SELECT * FROM leases WHERE lease_id = ?", (lease_id,))
        return _to_lease(rows[0]) if rows else None

    def open_leases(self) -> list[Lease]:
        return [
            _to_lease(row)
            for row in self.db.query("SELECT * FROM leases WHERE state = 'open' ORDER BY opened_at")
        ]

    def all(self) -> list[Lease]:
        return [_to_lease(row) for row in self.db.query("SELECT * FROM leases ORDER BY opened_at")]


def _to_lease(row: Any) -> Lease:
    return Lease(
        lease_id=row["lease_id"],
        workers=row["workers"],
        max_hours=row["max_hours"],
        max_spend=row["max_spend"],
        allow_rent=bool(row["allow_rent"]),
        bid_ceiling=row["bid_ceiling"],
        state=row["state"],
        opened_at=row["opened_at"],
        closed_at=row["closed_at"],
        closed_reason=row["closed_reason"],
    )


class EventLog:
    """Every decision, with the numbers behind it, kept verbatim."""

    def __init__(self, database: Database):
        self.db = database

    def record(
        self,
        kind: str,
        summary: str,
        *,
        numbers: Optional[dict[str, Any]] = None,
        host_id: Optional[str] = None,
        lease_id: Optional[str] = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO events (ts, kind, host_id, lease_id, summary, numbers) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), kind, host_id, lease_id, summary, json.dumps(numbers or {})),
        )

    def since(self, after_id: int, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM events WHERE id > ? ORDER BY id ASC LIMIT ?", (after_id, limit)
        )
        return [{**dict(row), "numbers": json.loads(row["numbers"])} for row in rows]

    def recent(self, limit: int = 100, kind: Optional[str] = None) -> list[dict[str, Any]]:
        if kind is None:
            rows = self.db.query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self.db.query(
                "SELECT * FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit)
            )
        return [{**dict(row), "numbers": json.loads(row["numbers"])} for row in rows]


class SpendLedger:
    """Append-only cost events. The pool's own estimate and the provider's figure are both
    recorded, and caps are enforced on whichever is higher (spec §4)."""

    def __init__(self, database: Database):
        self.db = database

    def record(self, *, lease_id: Optional[str], host_id: Optional[str], source: str, amount: float) -> None:
        self.db.execute(
            "INSERT INTO spend (ts, lease_id, host_id, source, amount) VALUES (?, ?, ?, ?, ?)",
            (time.time(), lease_id, host_id, source, amount),
        )

    def latest_for_lease(self, lease_id: str) -> dict[str, float]:
        """The most recent figure per host per source, summed — costs accumulate per host, so
        adding every sample would count the same dollars many times over."""
        rows = self.db.query(
            """
            SELECT source, host_id, amount FROM spend s
            WHERE lease_id = ? AND id = (
                SELECT MAX(id) FROM spend
                WHERE lease_id = s.lease_id AND host_id IS s.host_id AND source = s.source
            )
            """,
            (lease_id,),
        )
        totals = {"estimate": 0.0, "reported": 0.0}
        for row in rows:
            totals[row["source"]] = totals.get(row["source"], 0.0) + row["amount"]
        return totals
