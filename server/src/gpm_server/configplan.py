"""Configuration: versions on disk, and what a change would cause *now*.

docs/spec/console-and-control-api.md §1. One source of truth — the file. Nothing is applied
blind: validate → **plan** → apply. Changing configuration never spends money, and loosening a
limit is harder than tightening one.

`plan_changes` is a pure function of (running config, candidate config, what is rented right
now), so what the console shows before Apply is computed, never guessed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import ConfigError, PoolConfig, load_config

#: Applied versions kept beside the file (spec §1: the last 50, with a diff and rollback).
HISTORY_LIMIT = 50


@dataclasses.dataclass(frozen=True)
class Change:
    kind: str
    detail: str
    #: The value an operator must type again to confirm. Set only for loosening.
    requires_retype: Optional[str] = None
    #: True when the change cannot take effect without restarting the router.
    needs_restart: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "requires_retype": self.requires_retype is not None,
            "value": self.requires_retype,
            "needs_restart": self.needs_restart,
        }


@dataclasses.dataclass(frozen=True)
class RentedNow:
    """What the pool is holding while the change is considered."""

    host_id: str
    bid_hourly: float


def plan_changes(
    current: PoolConfig, candidate: PoolConfig, rented: Sequence[RentedNow] = ()
) -> list[Change]:
    changes: list[Change] = []

    # --- hosts ---
    before = {host.id: host for host in current.hosts}
    after = {host.id: host for host in candidate.hosts}
    for host_id in sorted(after.keys() - before.keys()):
        changes.append(Change("host_added", f"host {host_id!r} is added; it will be probed and joins once the whole model set is resident"))
    for host_id in sorted(before.keys() - after.keys()):
        changes.append(Change("host_removed", f"host {host_id!r} is removed; it will be drained and stop receiving requests"))
    for host_id in sorted(after.keys() & before.keys()):
        old, new = before[host_id], after[host_id]
        if old.transport != new.transport:
            changes.append(Change("host_transport", f"host {host_id!r} changes transport or address; it will be drained, reconnected and re-tested"))
        if old.workers != new.workers:
            if new.workers > old.workers:
                changes.append(Change("workers_raised", f"host {host_id!r} goes from {old.workers} to {new.workers} workers; the new ones start idle at once"))
            else:
                changes.append(Change("workers_lowered", f"host {host_id!r} goes from {old.workers} to {new.workers} workers; the surplus drain after their current request"))
        if old.disabled != new.disabled:
            changes.append(Change("host_disabled", f"host {host_id!r} is {'disabled' if new.disabled else 'enabled'}"))
        if not old.transport.allow_insecure and new.transport.allow_insecure:
            changes.append(Change(
                "allow_insecure",
                f"host {host_id!r} switches on allow_insecure: its bearer key will travel in clear text",
                requires_retype=host_id,
            ))

    # --- the model set and the catalog ---
    if current.pool.model_set != candidate.pool.model_set:
        changes.append(Change(
            "model_set",
            f"the model set changes from {current.pool.model_set} to {candidate.pool.model_set}; "
            "hosts are re-prepared one at a time, and any that cannot hold the new set leave the pool",
        ))
    if current.catalog != candidate.catalog:
        changes.append(Change("catalog", "the catalog changes; affected (host, model) pairs are re-prepared and resolution changes on the next request"))

    # --- the request path ---
    if current.pool.queue_timeout_s != candidate.pool.queue_timeout_s:
        changes.append(Change("queue_timeout", f"the queue timeout changes from {current.pool.queue_timeout_s}s to {candidate.pool.queue_timeout_s}s; it applies to the next request"))
    if current.listen != candidate.listen:
        changes.append(Change(
            "listen",
            "the router's listen address or TLS changes; this needs a restart of the router process. Rented hosts are unaffected",
            needs_restart=True,
        ))

    # --- limits and money ---
    changes.extend(_limit_changes(current, candidate, rented))
    return changes


def _limit_changes(current: PoolConfig, candidate: PoolConfig, rented: Sequence[RentedNow]) -> list[Change]:
    changes: list[Change] = []
    old_limits, new_limits = current.limits, candidate.limits

    if new_limits.max_rented_hosts > old_limits.max_rented_hosts:
        changes.append(Change(
            "max_rented_hosts_raised",
            f"the maximum rented hosts goes from {old_limits.max_rented_hosts} to {new_limits.max_rented_hosts}",
            requires_retype=str(new_limits.max_rented_hosts),
        ))
    elif new_limits.max_rented_hosts < old_limits.max_rented_hosts:
        over = max(0, len(rented) - new_limits.max_rented_hosts)
        detail = f"the maximum rented hosts drops to {new_limits.max_rented_hosts}"
        if over:
            detail += f"; {over} host(s) beyond it will be drained and released"
        changes.append(Change("max_rented_hosts_lowered", detail))

    if new_limits.max_hourly_burn > old_limits.max_hourly_burn:
        changes.append(Change(
            "burn_raised",
            f"the hourly burn cap goes from ${old_limits.max_hourly_burn:.2f} to ${new_limits.max_hourly_burn:.2f}",
            requires_retype=f"{new_limits.max_hourly_burn:.2f}",
        ))
    elif new_limits.max_hourly_burn < old_limits.max_hourly_burn:
        burn = sum(host.bid_hourly for host in rented)
        detail = f"the hourly burn cap drops to ${new_limits.max_hourly_burn:.2f}"
        if burn > new_limits.max_hourly_burn:
            detail += f"; the pool is burning ${burn:.3f}/h, so nothing new will be rented until it falls"
        changes.append(Change("burn_lowered", detail))

    old_rented, new_rented = current.rented, candidate.rented
    if (old_rented is None) != (new_rented is None):
        changes.append(Change(
            "rented_section",
            "rented capacity is switched on; a lease will be able to spend" if new_rented
            else "rented capacity is removed; the pool will not be able to rent at all",
            requires_retype="rented" if new_rented else None,
        ))
    elif old_rented is not None and new_rented is not None:
        old_ceiling = old_rented.bidding.bid_ceiling
        new_ceiling = new_rented.bidding.bid_ceiling
        if new_ceiling > old_ceiling:
            changes.append(Change(
                "bid_ceiling_raised",
                f"the bid ceiling goes from ${old_ceiling:.3f} to ${new_ceiling:.3f}/h",
                requires_retype=f"{new_ceiling:.3f}",
            ))
        elif new_ceiling < old_ceiling:
            over = [host for host in rented if host.bid_hourly > new_ceiling]
            detail = f"the bid ceiling drops to ${new_ceiling:.3f}/h"
            if over:
                names = ", ".join(f"{h.host_id} at ${h.bid_hourly:.3f}" for h in over)
                detail += f"; {names} now bids above it and will be drained and released"
            changes.append(Change("bid_ceiling_lowered", detail))

        if new_rented.image != old_rented.image:
            changes.append(Change("image", f"the engine image changes to {new_rented.image}; it applies to hosts rented from now on"))
        if new_rented.teardown.idle_minutes != old_rented.teardown.idle_minutes:
            changes.append(Change("idle_minutes", f"idle release changes from {old_rented.teardown.idle_minutes} to {new_rented.teardown.idle_minutes} minutes"))
        if new_rented.teardown.deadman_minutes != old_rented.teardown.deadman_minutes:
            changes.append(Change(
                "deadman",
                f"the dead-man timer changes from {old_rented.teardown.deadman_minutes} to "
                f"{new_rented.teardown.deadman_minutes} minutes; hosts already running keep the timer they were armed with",
                requires_retype=(
                    str(new_rented.teardown.deadman_minutes)
                    if new_rented.teardown.deadman_minutes > old_rented.teardown.deadman_minutes
                    else None
                ),
            ))
    return changes


def version_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class StaleVersion(Exception):
    """The file changed since it was read. A write based on a stale version is refused."""


class ConfigStore:
    """The file, its applied history, and the two operations that change either."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.versions_dir = self.path.with_suffix(self.path.suffix + ".versions")

    def read(self) -> tuple[str, str]:
        text = self.path.read_text()
        return text, version_of(text)

    def validate(self, text: str) -> list[str]:
        """Parse the candidate exactly as the pool would. No network, no side effect."""
        temporary = self.versions_dir / ".candidate.yaml"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        temporary.write_text(text)
        try:
            load_config(temporary)
            return []
        except ConfigError as exc:
            return str(exc).splitlines()
        finally:
            temporary.unlink(missing_ok=True)

    def parse(self, text: str) -> PoolConfig:
        temporary = self.versions_dir / ".candidate.yaml"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        temporary.write_text(text)
        try:
            return load_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def apply(self, text: str, expected_version: Optional[str] = None) -> str:
        """Validate, keep the outgoing version, then replace the file atomically."""
        errors = self.validate(text)
        if errors:
            raise ConfigError("\n".join(errors))

        if expected_version is not None:
            _, actual = self.read()
            if actual != expected_version:
                raise StaleVersion(
                    "the configuration changed since it was read; reload and re-apply"
                )

        self._keep_current()
        # Written next to the file and renamed: a reader never sees a half-written config.
        temporary = self.path.with_suffix(self.path.suffix + ".new")
        temporary.write_text(text)
        os.replace(temporary, self.path)
        return version_of(text)

    def _keep_current(self) -> None:
        if not self.path.exists():
            return
        text = self.path.read_text()
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.time()
        name = f"{int(stamp * 1000)}-{version_of(text)[:12]}.yaml"
        (self.versions_dir / name).write_text(text)
        (self.versions_dir / (name + ".json")).write_text(
            json.dumps({"applied_at": stamp, "version": version_of(text)})
        )
        self._trim()

    def _trim(self) -> None:
        kept = sorted(self.versions_dir.glob("*.yaml"))
        for stale in kept[:-HISTORY_LIMIT]:
            stale.unlink(missing_ok=True)
            stale.with_suffix(".yaml.json").unlink(missing_ok=True)

    def history(self) -> list[dict[str, Any]]:
        if not self.versions_dir.exists():
            return []
        entries = []
        for path in sorted(self.versions_dir.glob("*.yaml"), reverse=True):
            meta_path = path.with_suffix(".yaml.json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            entries.append(
                {
                    "version": meta.get("version") or version_of(path.read_text()),
                    "applied_at": meta.get("applied_at", path.stat().st_mtime),
                    "file": path.name,
                }
            )
        return entries

    def text_of(self, version: str) -> Optional[str]:
        for entry in self.history():
            if entry["version"] == version:
                return (self.versions_dir / entry["file"]).read_text()
        return None

    def rollback(self, version: str) -> str:
        text = self.text_of(version)
        if text is None:
            raise ConfigError(f"no kept version {version!r}")
        return self.apply(text)
