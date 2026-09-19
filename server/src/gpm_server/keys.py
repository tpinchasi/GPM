"""App keys and admin keys.

docs/spec/app-contract.md §3. Two roles that are **never interchangeable**: an app that can
request a completion must not thereby be able to open a lease, change a ceiling or release a
host. Keys are stored **hashed**, so the file cannot hand anyone a working key; they are
created, rotated and revoked from the CLI, and two keys of a role may be valid at once so
rotation needs no downtime.

The file is owner-readable only, and the pool refuses to read one that is not (threat model T19).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import secrets
import stat
import time
from pathlib import Path
from typing import Literal, Optional

Role = Literal["app", "admin"]

#: Keys are 32 random bytes. That is far beyond what a dictionary attack can reach, so a plain
#: digest is enough here — a slow KDF defends low-entropy secrets, which these are not.
_PREFIX = {"app": "gpma", "admin": "gpmx"}


class KeyFileUnsafe(Exception):
    """The key file is readable by someone other than its owner."""


def mint(role: Role) -> str:
    return f"{_PREFIX[role]}_{secrets.token_hex(32)}"


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class KeyRecord:
    key_id: str
    role: str
    hashed: str
    label: Optional[str] = None
    created_at: float = dataclasses.field(default_factory=time.time)


class KeyStore:
    """One file per role, holding hashes and never the keys themselves."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def _check_permissions(self) -> None:
        mode = self.path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            raise KeyFileUnsafe(
                f"{self.path} is readable by others; run `chmod 600 {self.path}` "
                "before starting the pool"
            )

    def load(self) -> list[KeyRecord]:
        if not self.path.exists():
            return []
        self._check_permissions()
        records = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            records.append(KeyRecord(**raw))
        return records

    def _write(self, records: list[KeyRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written owner-only from the start: a key file must never exist world-readable, even
        # for the moment between creating it and tightening it.
        handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(handle, "w") as file:
            for record in records:
                file.write(json.dumps(dataclasses.asdict(record)) + "\n")

    def create(self, role: Role, label: Optional[str] = None) -> tuple[str, KeyRecord]:
        """Returns the key **once**. It is not recoverable afterwards."""
        key = mint(role)
        record = KeyRecord(
            key_id=f"{role}-{secrets.token_hex(4)}",
            role=role,
            hashed=fingerprint(key),
            label=label,
        )
        records = self.load()
        records.append(record)
        self._write(records)
        return key, record

    def revoke(self, key_id: str) -> bool:
        records = self.load()
        remaining = [record for record in records if record.key_id != key_id]
        if len(remaining) == len(records):
            return False
        self._write(remaining)
        return True

    def hashes(self, role: Optional[Role] = None) -> set[str]:
        return {
            record.hashed for record in self.load() if role is None or record.role == role
        }


def verify(key: Optional[str], allowed: set[str]) -> bool:
    """Constant-time against every candidate: a comparison that stops early leaks the key."""
    if not key or not allowed:
        return False
    digest = fingerprint(key)
    matched = False
    for candidate in allowed:
        matched |= secrets.compare_digest(digest, candidate)
    return matched
