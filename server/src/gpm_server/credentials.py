"""Provider credentials typed into the console, kept by the supervisor (D130).

docs/spec/providers.md §5. A credential is written once, from the browser, and never read back
out: not in a response, an event, a log, the database or the configuration file and its
versions. It lives in an owner-only directory beside the pool's state, one file per provider —
at most one connection per provider (D133), so a renamed connection keeps its credential.

A stored credential is **bound to where it is sent**: the plug-in's endpoint settings as they
were when it was saved. A configuration that now names another endpoint does not get it.

The directory and every file in it are refused at start when anyone but their owner can reach
them, as the pool's key files are (threat model T19), and nothing here follows a link.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import secrets
import stat
import time
from pathlib import Path
from typing import Any, Mapping, Optional

#: A provider's entry-point name, which names its file — never a path.
_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class CredentialStoreUnsafe(Exception):
    """The credential directory or a file in it can be reached by someone other than its owner,
    or is a link. The message says how to fix it; the pool does not start until it is."""


@dataclasses.dataclass(frozen=True)
class StoredCredential:
    provider: str
    set_at: float
    #: The plug-in's endpoint settings when it was saved: where it may be sent.
    endpoint: dict[str, Any]
    #: Never in a repr, a log line or an answer.
    value: str = dataclasses.field(repr=False)

    def bound_to(self, endpoint: Mapping[str, Any]) -> bool:
        return dict(endpoint) == self.endpoint


def endpoint_of(plugin: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Where a connection's credential goes: the values of the settings its plug-in names as
    deciding it, absent ones included, so adding one later is a change too."""
    names = getattr(plugin, "endpoint_settings", ()) or ()
    return {name: settings.get(name) for name in names}


class CredentialStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()

    # --- safety ---

    def _refuse_unsafe(self, path: Path, what: str) -> os.stat_result:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise CredentialStoreUnsafe(f"{path} is a link; the provider {what} is never read through one")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise CredentialStoreUnsafe(f"{path} belongs to another user; it must be this pool's own")
        if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            fix = "700" if what == "credential directory" else "600"
            raise CredentialStoreUnsafe(f"{path} can be reached by others; run `chmod {fix} {path}` before starting the pool")
        return info

    def _refuse_open_parent(self) -> None:
        """The directory it sits in must not let someone else swap it for their own: no parent
        others can write to, unless only an entry's owner can rename it there (the sticky bit)."""
        parent = self.directory.parent.resolve()
        mode = os.stat(parent).st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH) and not mode & stat.S_ISVTX:
            raise CredentialStoreUnsafe(f"{parent} can be written by others, so the credential directory in it could be "
                                        "replaced; choose rented.credentials_dir somewhere only this user can write")

    def check(self) -> None:
        """At start: the directory, if there is one, and every file in it, are this user's alone."""
        if not os.path.lexists(self.directory):
            return
        self._refuse_open_parent()
        self._refuse_unsafe(self.directory, "credential directory")
        for entry in self.directory.iterdir():
            self._refuse_unsafe(entry, "credential")

    def _ensure_directory(self) -> None:
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self._refuse_open_parent()
        if not os.path.lexists(self.directory):
            self.directory.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(self.directory, 0o700)
        self._refuse_unsafe(self.directory, "credential directory")

    def _path(self, provider: str) -> Path:
        if not _PROVIDER.match(provider):
            raise ValueError("not a provider name")
        return self.directory / f"{provider}.json"

    # --- reading and writing ---

    def get(self, provider: str) -> Optional[StoredCredential]:
        path = self._path(provider)
        if not os.path.lexists(path):
            return None
        self._refuse_unsafe(self.directory, "credential directory")
        self._refuse_unsafe(path, "credential")
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(handle, "r", encoding="utf-8") as file:
            raw = json.load(file)
        return StoredCredential(
            provider=provider, set_at=float(raw["set_at"]), endpoint=dict(raw.get("endpoint") or {}),
            value=str(raw["value"]),
        )

    def put(self, provider: str, value: str, endpoint: Mapping[str, Any]) -> StoredCredential:
        """Written whole or not at all: a new file, owner-only from its first byte, moved over the
        old one. A reader never sees half a credential, and a crash leaves the old one."""
        if not value or not isinstance(value, str):
            raise ValueError("an empty credential")
        self._ensure_directory()
        final = self._path(provider)
        record = StoredCredential(provider=provider, set_at=time.time(), endpoint=dict(endpoint), value=value)
        temporary = self.directory / f".{provider}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        handle = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump({"set_at": record.set_at, "endpoint": record.endpoint, "value": value}, file)
                file.flush()
                os.fsync(file.fileno())
            if os.path.lexists(final):
                self._refuse_unsafe(final, "credential")
            os.replace(temporary, final)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self._sync_directory()
        return record

    def remove(self, provider: str) -> bool:
        path = self._path(provider)
        if not os.path.lexists(path):
            return False
        os.unlink(path)  # a link is removed, never followed
        self._sync_directory()
        return True

    def _sync_directory(self) -> None:
        try:
            handle = os.open(self.directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(handle)
        except OSError:
            pass
        finally:
            os.close(handle)
