"""A hosted API's key, typed into the console (S13, D171).

A hosted host is sent its requests by the router and probed by the supervisor, so — unlike a
rented provider's credential, which only the supervisor uses (D130) — its key is read by both
processes. It is kept in the credential store's `hosted/` directory, one owner-only file per host,
written only by the supervisor (from the console, write-once) and read by both; the two processes
still never call each other.

A key is used in one order: the host's own `bearer_env` (or `basic_env`), when the file names one,
always; else one typed into the console for that host, **only while the host's `base_url` is the
one it was saved for** — so neither a changed file nor a live edit can steer a typed key to an
address the admin key chose afterwards; else none, and the host waits for one.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import HostConfig, PoolConfig, PoolSettings
from .credentials import CredentialStore, CredentialStoreUnsafe, StoredCredential
from .transports import build_client

log = logging.getLogger(__name__)

#: The store's directory for hosted keys, inside the pool's credential directory.
HOSTED_DIR = "hosted"
#: A host whose key can be typed in has an id that can name a file: never a path.
HOST_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class HostedKeys(CredentialStore):
    """The store's `hosted/` directory. Creating it creates the credential directory first, owner-only,
    so the directory it sits in is never one others can reach."""

    def __init__(self, config: PoolConfig):
        self.parent = CredentialStore(config.credentials_dir())
        super().__init__(Path(self.parent.directory) / HOSTED_DIR)

    def _ensure_directory(self) -> None:
        self.parent._ensure_directory()
        super()._ensure_directory()

    def check(self) -> None:
        self.parent.check()
        super().check()


def endpoint(host: HostConfig) -> dict[str, Any]:
    """Where a host's typed key may be sent: its base URL as the file says it."""
    return {"base_url": (host.transport.base_url or "").rstrip("/")}


def names_its_own(host: HostConfig) -> bool:
    """The file names where the key comes from (`bearer_env`, `basic_env`): a typed one is never used."""
    return bool(host.transport.bearer_env or host.transport.basic_env)


def stored(config: PoolConfig, host: HostConfig) -> Optional[StoredCredential]:
    """The key typed in for this host, if there is one and it may be sent where the host now points."""
    if host.kind != "hosted" or names_its_own(host) or not HOST_ID.match(host.id):
        return None
    try:
        record = HostedKeys(config).get(host.id)
    except (OSError, ValueError, KeyError, CredentialStoreUnsafe) as exc:
        log.error("the key typed in for host %s cannot be read: %s", host.id, exc)
        return None
    if record is None or not record.bound_to(endpoint(host)):
        return None
    return record


def stamp(config: PoolConfig, host: HostConfig) -> Optional[float]:
    """When the key in use was saved — what tells a process that its client is out of date."""
    record = stored(config, host)
    return record.set_at if record is not None else None


def client(config: PoolConfig, host: HostConfig, base_url: Optional[str] = None,
           settings: Optional[PoolSettings] = None) -> httpx.AsyncClient:
    """The client a configured host is reached with: its transport, and a typed key where it takes one."""
    record = stored(config, host)
    return build_client(host.transport, settings or config.pool, base_url,
                        bearer=record.value if record is not None else None)


def describe(config: PoolConfig, host: HostConfig) -> dict[str, Any]:
    """Where a hosted host's key comes from — never the key."""
    transport = host.transport
    if transport.bearer_env or transport.basic_env:
        variable = transport.bearer_env or transport.basic_env
        return {"source": "environment" if os.environ.get(variable) else "missing", "variable": variable,
                "can_type": False, "set_at": None}
    if not HOST_ID.match(host.id):
        return {"source": "missing", "variable": None, "can_type": False, "set_at": None,
                "detail": "this host's id cannot name a key file: lower-case letters, digits, '-' and '_'"}
    try:
        record = HostedKeys(config).get(host.id)
    except (OSError, ValueError, KeyError, CredentialStoreUnsafe):
        record = None
    if record is None:
        return {"source": "missing", "variable": None, "can_type": True, "set_at": None}
    if not record.bound_to(endpoint(host)):
        return {"source": "withheld", "variable": None, "can_type": True, "set_at": record.set_at,
                "detail": "the key was saved for another address; type it again for this one"}
    return {"source": "stored", "variable": None, "can_type": True, "set_at": record.set_at}


def stamps(config: PoolConfig) -> tuple:
    """Every hosted key file's modification time, for a process watching for a new key."""
    out = []
    try:
        store = HostedKeys(config)
    except Exception:  # noqa: BLE001 — a store that cannot be found is no key at all
        return ()
    for host in config.hosts:
        if host.kind != "hosted" or not HOST_ID.match(host.id):
            continue
        try:
            out.append((host.id, (Path(store.directory) / f"{host.id}.json").stat().st_mtime_ns))
        except OSError:
            out.append((host.id, None))
    return tuple(out)
