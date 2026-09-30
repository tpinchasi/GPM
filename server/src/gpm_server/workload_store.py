"""Workloads, their keys and their volumes, as the database holds them (D115, D116).

The supervisor writes; the router reads the keys and each workload's state, off its request
path, in the background task that already reads the host table. Nothing secret is written: a
key is stored as its hash, as app keys are (app-contract.md §3), and shown once when minted.
"""

from __future__ import annotations

import dataclasses
import json
import secrets
import time
from typing import Any, Optional

from .db import Database
from .keys import fingerprint, mint

#: The states a workload moves through, in order. `ended` is final.
STATES = ("preparing", "serving", "ending", "ended")
#: States in which a workload's key is honoured.
ANSWERING = ("preparing", "serving")


#: How long an ended workload's keys are still recognised, to be told `workload_ended` rather
#: than "unknown key"; after that they drop out of what the router compares against.
ENDED_KEYS_KEPT_S = 7 * 24 * 3600

@dataclasses.dataclass(frozen=True)
class ModelTarget:
    """One model a workload serves, with its own target (D118): the whole answer's p95 within
    `latency_s`, `parallel` answers at once."""

    model: str
    latency_s: float
    parallel: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Group:
    """A set of models every one of its hosts holds, and what scales (D118). A workload placed
    *together* has one group of every model; *apart*, one group per model. A host's group is the
    set of models it was bought for (`RentedHost.models`), so it needs no column of its own."""

    models: tuple[str, ...]
    builds: dict[str, str]
    hosts_at_start: int
    #: Workers each host runs: for several models, the sum of the caps.
    workers_per_host: int
    #: For several models on one host: at most this many answers of each at once, a fixed share
    #: of the card. Empty for one model, whose host has no other to share with.
    caps: dict[str, int] = dataclasses.field(default_factory=dict)
    cards_per_copy: int = 1

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))

    def holds(self, models: Any) -> bool:
        return set(models) == set(self.models)

    def as_dict(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["models"] = list(self.models)
        return record

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Group":
        return cls(models=tuple(raw["models"]), builds=dict(raw.get("builds") or {}),
                   hosts_at_start=int(raw.get("hosts_at_start") or 0), workers_per_host=int(raw.get("workers_per_host") or 0),
                   caps={k: int(v) for k, v in (raw.get("caps") or {}).items()},
                   cards_per_copy=int(raw.get("cards_per_copy") or 1))


def group_key(models: Any) -> tuple[str, ...]:
    return tuple(sorted(models))


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    model: str
    builds: dict[str, str]
    latency_s: float
    parallel: int
    kind: str
    lease_id: str
    state: str
    workers_per_host: int
    hosts_at_start: int
    plan: dict[str, Any]
    created_at: float
    updated_at: float
    ended_at: Optional[float] = None
    #: When its lease runs out; its keys are refused from then (workloads.md §3).
    ends_at: Optional[float] = None
    #: Who made it (D117): None for an operator, else the provisioning key's name; how long it may
    #: sit unused before it is ended; the one client certificate that reaches it, where required.
    provisioner: Optional[str] = None
    idle_end_minutes: Optional[float] = None
    cert_fingerprint: Optional[str] = None
    serving_at: Optional[float] = None
    #: Every model it serves, each with its target (D118); and how they are placed on hosts. A
    #: workload made before D118 — or with one model — is one target and one group, filled in
    #: from the fields above; those fields stay for everything that reads one model.
    targets: tuple[ModelTarget, ...] = ()
    placement: str = "together"
    groups: tuple[Group, ...] = ()

    def __post_init__(self) -> None:
        if not self.targets:
            object.__setattr__(self, "targets", (ModelTarget(self.model, self.latency_s, self.parallel),))
        if not self.groups:
            object.__setattr__(self, "groups", (Group(
                models=(self.model,), builds={self.model: self.builds.get(self.model, self.model)},
                hosts_at_start=self.hosts_at_start, workers_per_host=self.workers_per_host,
                cards_per_copy=int((self.plan or {}).get("cards_per_copy") or 1)),))

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(t.model for t in self.targets)

    def target(self, model: str) -> Optional[ModelTarget]:
        return next((t for t in self.targets if t.model == model), None)

    def group_of(self, models: Any) -> Optional[Group]:
        """The group of hosts holding exactly these models."""
        return next((g for g in self.groups if g.holds(models)), None)

    def group_serving(self, model: str) -> Optional[Group]:
        return next((g for g in self.groups if model in g.models), None)

    def cap(self, model: str) -> Optional[int]:
        """The most answers of `model` one host of its group takes at once, where it shares the
        host with other models; None where it does not."""
        group = self.group_serving(model)
        return group.caps.get(model) if group is not None and len(group.models) > 1 else None

    @property
    def active(self) -> bool:
        return self.state != "ended"

    def as_dict(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["models"] = list(self.models)
        record["targets"] = [t.as_dict() for t in self.targets]
        record["groups"] = [g.as_dict() for g in self.groups]
        return record


def _to_workload(row: Any) -> Workload:
    return Workload(
        name=row["name"], model=row["model"], builds=json.loads(row["builds"]),
        latency_s=row["latency_s"], parallel=row["parallel"], kind=row["kind"],
        lease_id=row["lease_id"], state=row["state"], workers_per_host=row["workers_per_host"],
        hosts_at_start=row["hosts_at_start"], plan=json.loads(row["plan"]),
        created_at=row["created_at"], updated_at=row["updated_at"], ended_at=row["ended_at"],
        ends_at=row["ends_at"],
        provisioner=row["provisioner"], idle_end_minutes=row["idle_end_minutes"],
        cert_fingerprint=row["cert_fingerprint"], serving_at=row["serving_at"],
        targets=tuple(ModelTarget(**t) for t in json.loads(row["targets"])) if row["targets"] else (),
        placement=row["placement"] or "together",
        groups=tuple(Group.from_dict(g) for g in json.loads(row["groups"])) if row["groups"] else (),
    )


@dataclasses.dataclass(frozen=True)
class KeyGrant:
    """What a workload key reaches, as the router needs it."""

    workload: str
    not_after: Optional[float]


@dataclasses.dataclass(frozen=True)
class Volume:
    volume_id: str
    workload: str
    machine_id: str
    size_gb: float
    hourly: float
    lease_id: Optional[str]
    created_at: float
    deleted_at: Optional[float] = None
    #: The builds it holds: its group's (D118). None for one made before groups: the workload's.
    builds: Optional[dict[str, str]] = None


class WorkloadStore:
    def __init__(self, database: Database):
        self.db = database

    # --- workloads ---

    def create(self, workload: Workload) -> None:
        self.db.execute(
            """
            INSERT INTO workloads (
                name, model, builds, latency_s, parallel, kind, lease_id, state,
                workers_per_host, hosts_at_start, plan, created_at, updated_at, ended_at, ends_at,
                provisioner, idle_end_minutes, cert_fingerprint, targets, placement, groups
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workload.name, workload.model, json.dumps(workload.builds), workload.latency_s,
                workload.parallel, workload.kind, workload.lease_id, workload.state,
                workload.workers_per_host, workload.hosts_at_start, json.dumps(workload.plan),
                workload.created_at, workload.updated_at, workload.ended_at, workload.ends_at,
                workload.provisioner, workload.idle_end_minutes, workload.cert_fingerprint,
                json.dumps([t.as_dict() for t in workload.targets]), workload.placement,
                json.dumps([g.as_dict() for g in workload.groups]),
            ),
        )

    def set_cert_fingerprint(self, name: str, fingerprint: str) -> None:
        self.db.execute("UPDATE workloads SET cert_fingerprint = ?, updated_at = ? WHERE name = ?",
                        (fingerprint, time.time(), name))

    def set_serving_at(self, name: str) -> None:
        """When its first host became ready (D117, D118)."""
        now = time.time()
        self.db.execute("UPDATE workloads SET serving_at = COALESCE(serving_at, ?), updated_at = ? WHERE name = ?",
                        (now, now, name))

    def set_ends_at(self, name: str, ends_at: float) -> None:
        self.db.execute("UPDATE workloads SET ends_at = ?, updated_at = ? WHERE name = ?", (ends_at, time.time(), name))

    def get(self, name: str) -> Optional[Workload]:
        rows = self.db.query("SELECT * FROM workloads WHERE name = ?", (name,))
        return _to_workload(rows[0]) if rows else None

    def all(self) -> list[Workload]:
        return [_to_workload(r) for r in self.db.query("SELECT * FROM workloads ORDER BY created_at")]

    def active(self) -> list[Workload]:
        return [w for w in self.all() if w.active]

    def set_state(self, name: str, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown workload state {state!r}")
        now = time.time()
        self.db.execute(
            "UPDATE workloads SET state = ?, updated_at = ?, ended_at = CASE WHEN ? = 'ended' THEN ? ELSE ended_at END, "
            "serving_at = CASE WHEN ? = 'serving' AND serving_at IS NULL THEN ? ELSE serving_at END WHERE name = ?",
            (state, now, state, now, state, now, name),
        )

    def set_plan(self, name: str, plan: dict[str, Any], workers_per_host: Optional[int] = None) -> None:
        workload = self.get(name)
        if workload is None:
            return
        self.db.execute(
            "UPDATE workloads SET plan = ?, workers_per_host = ?, updated_at = ? WHERE name = ?",
            (json.dumps(plan), workers_per_host or workload.workers_per_host, time.time(), name),
        )

    # --- keys ---

    def mint_key(self, workload: str) -> tuple[str, str]:
        """(key_id, key). The key is returned **once** and only its hash is kept."""
        key = mint("workload")
        key_id = f"wk-{secrets.token_hex(4)}"
        now = time.time()
        self.db.execute(
            "INSERT INTO workload_keys (key_id, workload, hashed, created_at, not_after, updated_at) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (key_id, workload, fingerprint(key), now, now),
        )
        return key_id, key

    def add_key_hash(self, workload: str, hashed: str) -> str:
        """A key the program made itself, kept as the hash it sent (D117): the plaintext never
        reaches the pool. Returns the key's id."""
        key_id = f"wk-{secrets.token_hex(4)}"
        now = time.time()
        self.db.execute(
            "INSERT INTO workload_keys (key_id, workload, hashed, created_at, not_after, updated_at) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (key_id, workload, hashed, now, now),
        )
        return key_id

    def workload_of_key_hash(self, hashed: str) -> Optional[str]:
        rows = self.db.query("SELECT workload FROM workload_keys WHERE hashed = ?", (hashed,))
        return rows[0]["workload"] if rows else None

    def rotate_key(self, workload: str, grace_s: float) -> tuple[str, str]:
        """A new key; every key the workload had stays valid for `grace_s` more."""
        now = time.time()
        self.db.execute(
            "UPDATE workload_keys SET not_after = ?, updated_at = ? "
            "WHERE workload = ? AND (not_after IS NULL OR not_after > ?)",
            (now + grace_s, now, workload, now + grace_s),
        )
        return self.mint_key(workload)

    def expire_keys(self, workload: str) -> None:
        now = time.time()
        self.db.execute(
            "UPDATE workload_keys SET not_after = ?, updated_at = ? WHERE workload = ? AND (not_after IS NULL OR not_after > ?)",
            (now, now, workload, now),
        )

    def keys(self, workload: Optional[str] = None) -> list[dict[str, Any]]:
        """Key ids and their validity — never a hash."""
        sql = "SELECT key_id, workload, created_at, not_after FROM workload_keys"
        rows = self.db.query(sql + (" WHERE workload = ?" if workload else "") + " ORDER BY created_at",
                             (workload,) if workload else ())
        return [dict(r) for r in rows]

    def grants(self) -> dict[str, KeyGrant]:
        """hash → what it reaches, for keys not yet past their time: of workloads not ended, and
        of those ended in the last week, which are told `workload_ended` rather than "unknown
        key". Older ones drop out, so the router's per-request comparison does not grow with
        every workload ever made."""
        now = time.time()
        rows = self.db.query(
            "SELECT k.hashed, k.workload, k.not_after FROM workload_keys k JOIN workloads w ON w.name = k.workload "
            "WHERE (k.not_after IS NULL OR k.not_after > ?) AND (w.ended_at IS NULL OR w.ended_at > ?)",
            (now, now - ENDED_KEYS_KEPT_S),
        )
        return {r["hashed"]: KeyGrant(workload=r["workload"], not_after=r["not_after"]) for r in rows}

    def revision(self) -> float:
        """Changes when any workload or key does — the router's cheap "anything new?"."""
        a = self.db.query("SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM workloads")[0]
        b = self.db.query("SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM workload_keys")[0]
        return float(a["t"] or 0) + float(a["n"] or 0) + float(b["t"] or 0) * 2 + float(b["n"] or 0)

    # --- volumes ---

    def add_volume(self, volume: Volume) -> None:
        self.db.execute(
            "INSERT INTO workload_volumes (volume_id, workload, machine_id, size_gb, hourly, lease_id, created_at, builds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (volume.volume_id, volume.workload, volume.machine_id, volume.size_gb, volume.hourly,
             volume.lease_id, volume.created_at, json.dumps(volume.builds) if volume.builds is not None else None),
        )

    def volumes(self, workload: Optional[str] = None, live_only: bool = True) -> list[Volume]:
        sql = "SELECT * FROM workload_volumes WHERE 1=1"
        params: tuple = ()
        if workload is not None:
            sql += " AND workload = ?"
            params = (workload,)
        if live_only:
            sql += " AND deleted_at IS NULL"
        return [
            Volume(r["volume_id"], r["workload"], r["machine_id"], r["size_gb"], r["hourly"], r["lease_id"],
                   r["created_at"], r["deleted_at"], json.loads(r["builds"]) if r["builds"] else None)
            for r in self.db.query(sql + " ORDER BY created_at", params)
        ]

    def volume_deleted(self, volume_id: str) -> None:
        self.db.execute("UPDATE workload_volumes SET deleted_at = ? WHERE volume_id = ?", (time.time(), volume_id))
