"""Provisioning keys, their grants, and what they asked for (D117, docs/stories/S6).

The router reads the keys (as hashes) and writes requests; the supervisor writes the keys and
answers the requests. Nothing secret is written: a key is kept as its hash, a workload key a
program made as the hash it sent, a certificate as the public certificate it is.
"""

from __future__ import annotations

import dataclasses
import json
import re
import secrets
import time
from typing import Any, Optional

from .db import Database
from .keys import fingerprint, mint

#: A provisioner's name is part of its workloads' names: plain and short.
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")
#: A key hash, as the SDK sends it: sha256 of the key, in hex.
KEY_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass(frozen=True)
class Grant:
    """What a provisioning key may do. Every number is a ceiling the supervisor enforces."""

    max_open: int = 1
    max_spend: float = 10.0            # per workload
    max_spend_per_day: float = 20.0    # committed, rolling 24 hours
    max_hours: float = 8.0
    models: tuple[str, ...] = ()       # empty: none — a grant names what it allows
    kinds: tuple[str, ...] = ("roi", "on_demand", "interruptible")
    may_borrow: bool = True
    idle_end_minutes: float = 15.0     # the default for its workloads
    max_idle_end_minutes: float = 120.0
    #: "optional" or "required": whether its workloads must be reached with a client certificate.
    certs: str = "optional"

    def as_dict(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["models"] = list(self.models)
        record["kinds"] = list(self.kinds)
        return record

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Grant":
        fields = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in raw.items() if k in fields}
        for key in ("models", "kinds"):
            if key in clean:
                clean[key] = tuple(clean[key])
        return cls(**clean)


@dataclasses.dataclass(frozen=True)
class Provisioner:
    name: str
    hashed: str
    grant: Grant
    created_at: float
    expires_at: Optional[float]
    revoked_at: Optional[float]

    def usable(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return self.revoked_at is None and (self.expires_at is None or self.expires_at > now)

    def view(self) -> dict[str, Any]:
        return {"name": self.name, "grant": self.grant.as_dict(), "created_at": self.created_at,
                "expires_at": self.expires_at, "revoked_at": self.revoked_at, "usable": self.usable()}


class HashTaken(ValueError):
    """A workload key hash another provisioner already sent."""


@dataclasses.dataclass(frozen=True)
class Request:
    request_id: str
    provisioner: str
    kind: str
    key_hash: Optional[str]
    workload: Optional[str]
    body: dict[str, Any]
    state: str
    answer: Optional[dict[str, Any]]
    created_at: float
    updated_at: float

    def view(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "kind": self.kind, "state": self.state,
                "workload": self.workload, "answer": self.answer, "created_at": self.created_at}


def _provisioner(row: Any) -> Provisioner:
    return Provisioner(row["name"], row["hashed"], Grant.from_dict(json.loads(row["grant_"])),
                       row["created_at"], row["expires_at"], row["revoked_at"])


def _request(row: Any) -> Request:
    return Request(row["request_id"], row["provisioner"], row["kind"], row["key_hash"], row["workload"],
                   json.loads(row["body"]), row["state"], json.loads(row["answer"]) if row["answer"] else None,
                   row["created_at"], row["updated_at"])


class ProvisioningStore:
    def __init__(self, database: Database):
        self.db = database

    # --- keys and grants (supervisor writes) ---

    def create(self, name: str, grant: Grant, expires_at: Optional[float] = None) -> str:
        """The key, returned **once**."""
        key = mint("provisioner")
        now = time.time()
        self.db.execute(
            "INSERT INTO provisioners (name, hashed, grant_, created_at, expires_at, revoked_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (name, fingerprint(key), json.dumps(grant.as_dict()), now, expires_at, now),
        )
        return key

    def get(self, name: str) -> Optional[Provisioner]:
        rows = self.db.query("SELECT * FROM provisioners WHERE name = ?", (name,))
        return _provisioner(rows[0]) if rows else None

    def all(self) -> list[Provisioner]:
        return [_provisioner(r) for r in self.db.query("SELECT * FROM provisioners ORDER BY created_at")]

    def revoke(self, name: str) -> None:
        now = time.time()
        self.db.execute("UPDATE provisioners SET revoked_at = ?, updated_at = ? WHERE name = ? AND revoked_at IS NULL",
                        (now, now, name))

    def usable_hashes(self) -> dict[str, str]:
        """hash → provisioner name, for keys neither revoked nor expired: what the router honours."""
        now = time.time()
        rows = self.db.query(
            "SELECT name, hashed FROM provisioners WHERE revoked_at IS NULL AND (expires_at IS NULL OR expires_at > ?)",
            (now,),
        )
        return {r["hashed"]: r["name"] for r in rows}

    def revision(self) -> float:
        a = self.db.query("SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM provisioners")[0]
        return float(a["t"] or 0) + float(a["n"] or 0)

    # --- requests (router writes, supervisor answers) ---

    def ask(self, provisioner: str, kind: str, body: dict[str, Any], key_hash: Optional[str] = None,
            workload: Optional[str] = None) -> tuple[str, bool]:
        """Record a request. A create whose key hash this provisioner sent before is that same
        request: (its id, False). Otherwise (a new id, True). A hash another provisioner sent is
        refused (`HashTaken`), never answered with that provisioner's request."""
        if key_hash is not None:
            existing = self.of_key_hash(provisioner, key_hash)
            if existing is not None:
                return existing, False
        request_id = f"pr-{secrets.token_hex(8)}"
        now = time.time()
        try:
            self.db.execute(
                "INSERT INTO provisioning_requests (request_id, provisioner, kind, key_hash, workload, body, state, "
                "answer, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?)",
                (request_id, provisioner, kind, key_hash, workload, json.dumps(body), now, now),
            )
        except Exception:  # noqa: BLE001 - the same hash, written by a request racing this one
            existing = self.of_key_hash(provisioner, key_hash) if key_hash is not None else None
            if existing is not None:
                return existing, False
            raise
        return request_id, True

    def of_key_hash(self, provisioner: str, key_hash: str) -> Optional[str]:
        """The request this provisioner made with this hash; `HashTaken` where another made it."""
        rows = self.db.query("SELECT request_id, provisioner FROM provisioning_requests WHERE key_hash = ?", (key_hash,))
        if not rows:
            return None
        if rows[0]["provisioner"] != provisioner:
            raise HashTaken("this key_hash is in use; make a new workload key")
        return rows[0]["request_id"]

    def asked_since(self, provisioner: str, since: float) -> int:
        """How many requests this provisioner made since `since`: the router's rate limit."""
        return int(self.db.query(
            "SELECT COUNT(*) AS n FROM provisioning_requests WHERE provisioner = ? AND created_at >= ?",
            (provisioner, since))[0]["n"])

    def get_request(self, request_id: str) -> Optional[Request]:
        rows = self.db.query("SELECT * FROM provisioning_requests WHERE request_id = ?", (request_id,))
        return _request(rows[0]) if rows else None

    def pending(self) -> list[Request]:
        return [_request(r) for r in self.db.query(
            "SELECT * FROM provisioning_requests WHERE state = 'pending' ORDER BY created_at")]

    def pending_of(self, provisioner: str) -> int:
        return int(self.db.query(
            "SELECT COUNT(*) AS n FROM provisioning_requests WHERE provisioner = ? AND state = 'pending'",
            (provisioner,))[0]["n"])

    def answer(self, request_id: str, state: str, answer: dict[str, Any], workload: Optional[str] = None) -> None:
        self.db.execute(
            "UPDATE provisioning_requests SET state = ?, answer = ?, workload = COALESCE(?, workload), updated_at = ? "
            "WHERE request_id = ?",
            (state, json.dumps(answer), workload, time.time(), request_id),
        )

    def requests_revision(self) -> float:
        row = self.db.query("SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM provisioning_requests")[0]
        return float(row["t"] or 0) + float(row["n"] or 0)
