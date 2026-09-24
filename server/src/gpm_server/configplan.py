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
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

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
class MachineNow:
    """What a host's agent last said about the machine, while the change is considered."""

    #: The platform found; None when the agent is silent or could not tell.
    capabilities: Optional[frozenset[str]]
    on_disk: frozenset[str]
    free_disk_bytes: Optional[int]


@dataclasses.dataclass(frozen=True)
class RentedNow:
    """What the pool is holding while the change is considered."""

    host_id: str
    bid_hourly: float


def plan_changes(
    current: PoolConfig,
    candidate: PoolConfig,
    rented: Sequence[RentedNow] = (),
    machines: Optional[Mapping[str, MachineNow]] = None,
) -> list[Change]:
    changes: list[Change] = []
    changes.extend(_machine_changes(candidate, machines or {}))

    # --- hosts ---
    before = {host.id: host for host in current.hosts}
    after = {host.id: host for host in candidate.hosts}
    for host_id in sorted(after.keys() - before.keys()):
        joins = "on disk" if after[host_id].residency == "on_demand" else "resident"
        changes.append(Change("host_added", f"host {host_id!r} is added; it will be probed and joins once the whole model set is {joins}"))
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
        if old.agent != new.agent:
            if new.agent is None:
                detail = f"host {host_id!r} loses its agent: the pool goes back to verifying this host only, and stops learning what the machine is"
            elif old.agent is None:
                detail = (f"host {host_id!r} gains an agent at {_where(new.agent)}: the pool will read what the machine is from it"
                          + (", pull the models its set needs there, and hold them as its residency says"
                             if new.agent.manage_models else "; it reports only, because manage_models is off"))
            else:
                detail = f"host {host_id!r}'s agent moves to {_where(new.agent)}; traffic to its engine is not interrupted"
            changes.append(Change("host_agent", detail))
        if old.residency != new.residency:
            if new.residency == "on_demand":
                detail = (f"host {host_id!r} becomes on-demand: it is routable once the model set is on disk, "
                          "the engine loads a model on first use, and an evicted model no longer takes it out of routing")
            else:
                detail = (f"host {host_id!r} becomes pinned: it is routable only while the whole model set is loaded, "
                          "and leaves routing at the next probe if it is not")
            changes.append(Change("host_residency", detail))
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

    # --- capacity profiles ---
    def profiles(config: PoolConfig) -> dict[str, tuple[int, Optional[str]]]:
        found = {}
        for profile in config.capacity_profiles:
            m = profile.match
            key = ", ".join(f"{k}={v}" for k, v in m.model_dump(exclude_none=True).items()) or "all hardware"
            found[key] = (profile.max_workers, profile.note)
        return found

    old_profiles, new_profiles = profiles(current), profiles(candidate)
    for key in sorted(old_profiles.keys() | new_profiles.keys()):
        before, after = old_profiles.get(key), new_profiles.get(key)
        if before == after:
            continue
        if after is None:
            detail = f"the capacity profile for {key} ({before[0]} workers) is removed"
        elif before is None:
            detail = f"hosts matching {key} will run {after[0]} workers" + (f" — {after[1]}" if after[1] else "")
        else:
            detail = f"hosts matching {key} go from {before[0]} to {after[0]} workers"
        changes.append(Change(
            "capacity_profile",
            detail + "; applies to hosts rented from now on — a running host keeps the "
            "parallelism its engine was started with",
        ))
    if [p.match for p in current.capacity_profiles] != [p.match for p in candidate.capacity_profiles] and \
            old_profiles.keys() == new_profiles.keys():
        changes.append(Change("capacity_profile", "the capacity profiles are reordered; the first match wins"))

    # --- limits and money ---
    changes.extend(_limit_changes(current, candidate, rented))
    # --- and everything else: a plan is never silent about part of the file ---
    changes.extend(_everything_else(current, candidate))
    return changes


#: Settings a change above already explains, in words. Everything not listed here is still
#: reported — generically, by `_everything_else` — so adding a setting to the configuration
#: can never again produce a plan that omits it.
_EXPLAINED = (
    "pool.model_set", "pool.queue_timeout_s", "catalog", "listen",
    "limits.max_rented_hosts", "limits.max_hourly_burn",
    "rented.bidding.bid_ceiling", "rented.image",
    "rented.teardown.idle_minutes", "rented.teardown.deadman_minutes",
    "rented.offer_policy.max_all_in_hourly", "rented.offer_policy.max_download_per_gb",
    "capacity_profiles",
)
_EXPLAINED_PER_HOST = ("transport", "workers", "disabled", "agent", "residency")


def _leaves(value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        found: dict[str, Any] = {}
        for key, inner in value.items():
            found.update(_leaves(inner, f"{path}.{key}" if path else str(key)))
        return found
    return {path: value}


def _everything_else(current: PoolConfig, candidate: PoolConfig) -> list[Change]:
    before, after = current.model_dump(mode="json"), candidate.model_dump(mode="json")
    if (before.get("rented") is None) != (after.get("rented") is None):
        before.pop("rented", None), after.pop("rented", None)  # said once, as `rented_section`
    hosts_before = {host["id"]: host for host in before.pop("hosts")}
    hosts_after = {host["id"]: host for host in after.pop("hosts")}

    pairs = [("", _leaves(before), _leaves(after))]
    for host_id in sorted(hosts_before.keys() & hosts_after.keys()):  # added and removed are said already
        old = {k: v for k, v in hosts_before[host_id].items() if k not in _EXPLAINED_PER_HOST}
        new = {k: v for k, v in hosts_after[host_id].items() if k not in _EXPLAINED_PER_HOST}
        pairs.append((f"host {host_id!r}: ", _leaves(old), _leaves(new)))

    changes: list[Change] = []
    for prefix, old, new in pairs:
        for path in sorted(old.keys() | new.keys()):
            if old.get(path) == new.get(path):
                continue
            if not prefix and any(path == known or path.startswith(known + ".") for known in _EXPLAINED):
                continue
            changes.append(Change("setting", f"{prefix}{path} changes from {old.get(path)!r} to {new.get(path)!r}"))
    return changes


def _where(agent: Any) -> str:
    return agent.url or f"port {agent.remote_port} on the far side of the host's SSH tunnel"


_PLATFORMS = frozenset({"apple-silicon", "cuda", "rocm"})


def _machine_changes(candidate: PoolConfig, machines: Mapping[str, MachineNow]) -> list[Change]:
    """What applying this would make each delegated machine do — stated before it happens,
    in gigabytes where the catalog says how big a build is."""
    from .catalog import variants_for_host

    changes: list[Change] = []
    sizes = {v.tag: v.size_gb for entry in candidate.catalog.values() for v in entry.variants}
    for host in candidate.hosts:
        machine = machines.get(host.id)
        if host.agent is None or machine is None:
            continue
        stated = frozenset(host.capabilities) & _PLATFORMS
        found = None if machine.capabilities is None else machine.capabilities & _PLATFORMS
        if found is not None and stated and stated != found:
            changes.append(Change(
                "capability_conflict",
                f"host {host.id!r} is configured as {sorted(stated)} but its agent found "
                f"{sorted(found) or 'no accelerator'}; the configured list would be used as written",
            ))
        if not host.agent.manage_models:
            continue
        capabilities = frozenset(host.capabilities) | (found or frozenset() if not stated else frozenset())
        variants = variants_for_host(candidate.pool.model_set, candidate.catalog, capabilities, candidate.engine)
        missing = sorted({group[0].tag for group in variants.values() if group} - machine.on_disk)
        if not missing:
            continue
        known = [sizes[tag] for tag in missing if sizes.get(tag)]
        size = f"about {sum(known):.1f} GB" if len(known) == len(missing) else (
            f"at least {sum(known):.1f} GB; some sizes are not stated in the catalog" if known else "sizes not stated in the catalog"
        )
        free = "" if machine.free_disk_bytes is None else f"; {machine.free_disk_bytes / 1e9:.0f} GB free there"
        detail = f"host {host.id!r}: its agent will download {missing} ({size}{free})"
        if known and machine.free_disk_bytes is not None and sum(known) * 1e9 > machine.free_disk_bytes:
            detail += " — which does not fit, so the agent will refuse and the host will stay out of routing"
        changes.append(Change("agent_download", detail))
    return changes


def _price(cap: Optional[float], unit: str) -> str:
    return "no limit" if cap is None else f"${cap:.3f}{unit}"


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

    # The overall cap is optional (D46): unset, the bound is hosts × the per-host ceiling.
    # Unset is the *looser* end — removing a cap is loosening, setting one is tightening.
    old_burn, new_burn = old_limits.max_hourly_burn, new_limits.max_hourly_burn
    says = lambda cap: "no overall cap" if cap is None else f"${cap:.2f}/h"  # noqa: E731
    if old_burn != new_burn:
        loosened = new_burn is None or (old_burn is not None and new_burn > old_burn)
        if loosened:
            detail = f"the overall hourly burn cap goes from {says(old_burn)} to {says(new_burn)}"
            if new_burn is None and candidate.rented is not None:
                bound = new_limits.max_rented_hosts * candidate.rented.bidding.bid_ceiling
                detail += (f"; spending is then bounded per host — {new_limits.max_rented_hosts} host(s) × "
                           f"${candidate.rented.bidding.bid_ceiling:.2f}/h = ${bound:.2f}/h at most")
            changes.append(Change(
                "burn_raised", detail,
                requires_retype="none" if new_burn is None else f"{new_burn:.2f}",
            ))
        else:
            burn = sum(host.bid_hourly for host in rented)
            detail = f"the overall hourly burn cap goes from {says(old_burn)} to {says(new_burn)}"
            if burn > new_burn:
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

        # The other two price ceilings. Raising or removing either lets the pool pay more, so
        # it is loosening, and loosening is typed again — the same rule as the bid ceiling.
        for field, what, unit in (
            ("max_all_in_hourly", "the all-in hourly price the pool will accept", "/h"),
            ("max_download_per_gb", "the download price the pool will accept", "/GB"),
        ):
            old_cap, new_cap = getattr(old_rented.offer_policy, field), getattr(new_rented.offer_policy, field)
            if old_cap == new_cap:
                continue
            loosened = new_cap is None or (old_cap is not None and new_cap > old_cap)
            changes.append(Change(
                f"{field}_{'raised' if loosened else 'lowered'}",
                f"{what} goes from {_price(old_cap, unit)} to {_price(new_cap, unit)}"
                + ("" if loosened else "; offers above it are rejected from the next pass"),
                requires_retype=("none" if new_cap is None else f"{new_cap:.3f}") if loosened else None,
            ))

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
            # The problems themselves. The heading names this temporary file, which means
            # nothing to whoever is editing the pool's real one.
            lines = str(exc).splitlines()
            if lines and lines[0].startswith(str(temporary)):
                lines = lines[1:]
            return [line.strip().removeprefix("- ") for line in lines if line.strip()] or [str(exc)]
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


# --- changing one setting without rewriting the file ---------------------------------------


class CannotEdit(Exception):
    """The file cannot be changed here without guessing. The message is for an operator."""


def _as_yaml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) or value is None:
        return "null" if value is None else repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_as_yaml(item) for item in value) + "]"
    if isinstance(value, Mapping):
        # Flow style, on one line: a section the operator never wrote is added whole rather
        # than guessed at line by line, and `_set_in_flow` can edit it again afterwards.
        return "{ " + ", ".join(f"{key}: {_as_yaml(item)}" for key, item in value.items()) + " }"
    return json.dumps(str(value))


def _block_of(lines: list[str], key: str, start: int, end: int, indent: Optional[int]) -> tuple[int, int, int]:
    """Where `key:`'s own lines are, within lines[start:end]. Returns (its line, first child,
    last child). Raises if it is missing or appears more than once, because a config the pool
    cannot read unambiguously is one it must not edit."""
    pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:(.*)$")
    found = [
        (n, m) for n in range(start, end)
        if (m := pattern.match(lines[n])) and (indent is None or len(m.group(1)) == indent)
    ]
    if not found:
        raise CannotEdit(f"{key!r} is not in the configuration; add it on the Configuration screen first")
    if len(found) > 1:
        raise CannotEdit(f"{key!r} appears {len(found)} times; change it on the Configuration screen")
    at, match = found[0]
    own_indent = len(match.group(1))
    last = at
    for n in range(at + 1, end):
        stripped = lines[n].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(lines[n]) - len(lines[n].lstrip()) <= own_indent:
            break
        last = n
    return at, at + 1, last + 1


def set_values(text: str, path: Sequence[str], values: Mapping[str, Any]) -> str:
    """Change `values` inside the mapping at `path` (e.g. `("rented", "offer_policy")`).

    Everything else in the file is left byte for byte: comments, ordering, and whether a
    mapping was written in block or flow style. The alternative — loading the YAML and dumping
    it back — silently deletes every comment an operator wrote, which is not an acceptable
    price for changing one number (D51).
    """
    if not values:
        return text
    lines = text.splitlines()
    start, end, indent = 0, len(lines), 0
    for step in path:
        at, first, last = _block_of(lines, step, start, end, indent)
        inline = lines[at].split(":", 1)[1].strip()
        if inline.startswith("{"):
            lines[at] = _set_in_flow(lines[at], values, step)
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        start, end = first, last
        indent = None  # children of this block; their own indent is whatever it is

    child_indent = next(
        (len(line) - len(line.lstrip()) for line in lines[start:end] if line.strip() and not line.strip().startswith("#")),
        None,
    )
    if child_indent is None:
        raise CannotEdit(f"{'.'.join(path)} has nothing in it to change")

    for key, value in values.items():
        pattern = re.compile(rf"^(\s*){re.escape(key)}(\s*:\s*)(.*?)(\s+#.*)?$")
        hits = [n for n in range(start, end) if pattern.match(lines[n]) and len(pattern.match(lines[n]).group(1)) == child_indent]
        if len(hits) > 1:
            raise CannotEdit(f"{key!r} appears more than once under {'.'.join(path)}")
        if hits:
            match = pattern.match(lines[hits[0]])
            # The trailing comment is the operator's and is kept, even when it no longer fits.
            # A key that held a block ends at its colon; a value needs a space after it.
            separator = match.group(2).rstrip() + " "
            lines[hits[0]] = f"{match.group(1)}{key}{separator}{_as_yaml(value)}{match.group(4) or ''}"
            # A key that held a whole indented block — a list of variants, say — has its block
            # replaced too. Rewriting only the key's own line left the old block under it: a
            # file that no longer parsed, or worse, one that parsed as something else.
            own = child_indent
            stop = hits[0] + 1
            while stop < end:
                stripped = lines[stop].strip()
                if stripped and not stripped.startswith("#") and len(lines[stop]) - len(lines[stop].lstrip()) <= own:
                    break
                stop += 1
            # Trailing blank lines and comments belong to what follows, not to this value.
            while stop > hits[0] + 1 and (not lines[stop - 1].strip() or lines[stop - 1].strip().startswith("#")):
                stop -= 1
            if stop > hits[0] + 1:
                del lines[hits[0] + 1:stop]
                end -= stop - (hits[0] + 1)
        else:
            lines.insert(end, f"{' ' * child_indent}{key}: {_as_yaml(value)}")
            end += 1
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def set_in_list_item(
    text: str, list_key: str, match_key: str, match_value: str, values: Mapping[str, Any]
) -> str:
    """Change `values` inside one item of a top-level list — the host whose `id` is `laptop`.

    The same promise as `set_values`: everything outside the keys changed stays byte for byte.
    Only block-style items are edited; a list item written in flow style on one line is refused
    rather than rewritten, because rewriting it would lose how the operator wrote it.
    """
    if not values:
        return text
    lines = text.splitlines()
    _, first, last = _block_of(lines, list_key, 0, len(lines), 0)
    starts = [n for n in range(first, last) if lines[n].lstrip().startswith("- ")]
    if not starts:
        raise CannotEdit(f"{list_key!r} holds no items to change")
    item_indent = len(lines[starts[0]]) - len(lines[starts[0]].lstrip())
    starts = [n for n in starts if len(lines[n]) - len(lines[n].lstrip()) == item_indent]
    bounds = list(zip(starts, starts[1:] + [last], strict=True))
    wanted = re.compile(rf"^\s*(?:-\s+)?{re.escape(match_key)}\s*:\s*[\"']?{re.escape(str(match_value))}[\"']?\s*(#.*)?$")

    def one_line_match(a: int) -> bool:
        # An item written `- { id: desk, ... }` is read to say *why* it is refused, never edited.
        head = lines[a].lstrip()[2:].split(" #", 1)[0].strip()
        if not head.startswith("{"):
            return False
        try:
            item = yaml.safe_load(head)
        except yaml.YAMLError:
            return False
        return isinstance(item, dict) and str(item.get(match_key)) == str(match_value)

    found = [
        (a, b) for a, b in bounds
        if one_line_match(a) or any(wanted.match(lines[n]) for n in range(a, b))
    ]
    if not found:
        raise CannotEdit(f"no item in {list_key!r} has {match_key}: {match_value}")
    if len(found) > 1:
        raise CannotEdit(f"{len(found)} items in {list_key!r} have {match_key}: {match_value}")
    begin, finish = found[0]
    if lines[begin].lstrip()[2:].lstrip().startswith("{"):
        raise CannotEdit(f"the {match_value!r} item in {list_key!r} is written on one line; change it there")
    key_indent = item_indent + 2
    # Trailing blank lines and comments belong to what follows.
    while finish > begin + 1 and (not lines[finish - 1].strip() or lines[finish - 1].strip().startswith("#")):
        finish -= 1
    for key, value in values.items():
        pattern = re.compile(rf"^(\s*)(-\s+)?{re.escape(key)}(\s*:\s*)(.*?)(\s+#.*)?$")
        hit = next(
            (n for n in range(begin, finish)
             if (m := pattern.match(lines[n]))
             and len(m.group(1)) + len(m.group(2) or "") == key_indent),
            None,
        )
        if hit is None:
            lines.insert(finish, f"{' ' * key_indent}{key}: {_as_yaml(value)}")
            finish += 1
            continue
        m = pattern.match(lines[hit])
        separator = m.group(3).rstrip() + " "
        lines[hit] = f"{m.group(1)}{m.group(2) or ''}{key}{separator}{_as_yaml(value)}{m.group(5) or ''}"
        stop = hit + 1
        while stop < finish:
            stripped = lines[stop].strip()
            if stripped and not stripped.startswith("#") and len(lines[stop]) - len(lines[stop].lstrip()) <= key_indent:
                break
            stop += 1
        if stop > hit + 1:
            del lines[hit + 1:stop]
            finish -= stop - (hit + 1)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def remove_list_item(text: str, list_key: str, match_key: str, match_value: str) -> str:
    """Delete one item of a top-level list — the host whose `id` is `laptop` — and nothing else.

    The item's own lines go; the comments around it stay, because nothing in a file says which
    item a comment was about. A list left empty is written `[]`, since an empty block means
    null to YAML and a pool's list of hosts is never null.
    """
    lines = text.splitlines()
    at, first, last = _block_of(lines, list_key, 0, len(lines), 0)
    inline = lines[at].split(":", 1)[1].split(" #", 1)[0].strip()
    if inline.startswith("["):
        raise CannotEdit(f"{list_key!r} is written on one line; remove the item there")
    starts = [n for n in range(first, last) if lines[n].lstrip().startswith("- ")]
    if not starts:
        raise CannotEdit(f"{list_key!r} holds no items")
    item_indent = len(lines[starts[0]]) - len(lines[starts[0]].lstrip())
    starts = [n for n in starts if len(lines[n]) - len(lines[n].lstrip()) == item_indent]
    bounds = list(zip(starts, starts[1:] + [last], strict=True))
    wanted = re.compile(rf"^\s*(?:-\s+)?{re.escape(match_key)}\s*:\s*[\"']?{re.escape(str(match_value))}[\"']?\s*(#.*)?$")

    def matches(a: int, b: int) -> bool:
        head = lines[a].lstrip()[2:].split(" #", 1)[0].strip()
        if head.startswith("{"):
            try:
                item = yaml.safe_load(head)
            except yaml.YAMLError:
                return False
            return isinstance(item, dict) and str(item.get(match_key)) == str(match_value)
        return any(wanted.match(lines[n]) for n in range(a, b))

    found = [(a, b) for a, b in bounds if matches(a, b)]
    if not found:
        raise CannotEdit(f"no item in {list_key!r} has {match_key}: {match_value}")
    if len(found) > 1:
        raise CannotEdit(f"{len(found)} items in {list_key!r} have {match_key}: {match_value}")
    begin, finish = found[0]
    # Only the item's own lines: trailing blank lines and comments belong to what follows.
    while finish > begin + 1 and (not lines[finish - 1].strip() or lines[finish - 1].strip().startswith("#")):
        finish -= 1
    del lines[begin:finish]
    if len(bounds) == 1:
        head = lines[at]
        comment = head[len(head.split(" #", 1)[0]):] if " #" in head else ""
        lines[at] = f"{head.split(':', 1)[0]}: []{comment}"
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _set_in_flow(line: str, values: Mapping[str, Any], name: str) -> str:
    """A one-line `key: { a: 1, b: 2 }`, changed inside its braces."""
    head, _, rest = line.partition("{")
    body, closing, tail = rest.rpartition("}")
    if not closing:
        raise CannotEdit(f"{name!r} is written across lines; change it on the Configuration screen")
    for key, value in values.items():
        pattern = re.compile(rf"(^|,)(\s*){re.escape(key)}(\s*:\s*)([^,}}]*)")
        if pattern.search(body):
            body = pattern.sub(
                lambda m, key=key, value=value: f"{m.group(1)}{m.group(2)}{key}{m.group(3)}{_as_yaml(value)}",
                body, count=1,
            )
        else:
            body = (body.rstrip() + ", " if body.strip() else " ") + f"{key}: {_as_yaml(value)} "
    return f"{head}{{{body}}}{tail}"
