"""Named app keys for the shared pool (D193).

The shared workload is reached with an app key. Several applications may share it, and each
is then told apart by its own key and the name it was given — in the request log, and so in
the stats and on the dashboards. A named key has **exactly the app key's powers**: it requests
completions on the shared workload, reads `GET /pool/status` as an app, and reaches nothing on
the control API. Several keys may hold one name at once, so a key is rotated without a pause:
make the new one, move the app to it, revoke the old.

The supervisor writes the table; the router reads the hashes in the background (never on a
request) and resolves a key to its app. Only a key's hash is kept, as for every other key here.
"""

from __future__ import annotations

import dataclasses
import re
import secrets
import time
from typing import Any, Optional

from .db import Database
from .keys import fingerprint, mint

#: An app's name: plain and short. It is shown on the console and kept in every request's row.
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
#: Names the pool counts its own requests under (the plan chat, D204), never an app key's.
RESERVED = frozenset({"console"})


class AppKeyRefused(ValueError):
    """A name that is not a name, or a key that is not there."""


@dataclasses.dataclass(frozen=True)
class AppKey:
    key_id: str
    app: str
    created_at: float
    revoked_at: Optional[float]

    def view(self) -> dict[str, Any]:
        """What the control API answers: never the key, never its hash."""
        return {"key_id": self.key_id, "app": self.app, "created_at": self.created_at,
                "revoked_at": self.revoked_at, "usable": self.revoked_at is None}


def _key(row: Any) -> AppKey:
    return AppKey(row["key_id"], row["app"], row["created_at"], row["revoked_at"])


class AppKeyStore:
    def __init__(self, database: Database):
        self.db = database

    # --- the supervisor writes ---

    def create(self, app: str) -> tuple[str, AppKey]:
        """A new key for this app — the app made if it is new. The key is returned **once**."""
        if not isinstance(app, str) or not NAME.fullmatch(app):
            raise AppKeyRefused("an app's name is lower-case letters, digits, '-' and '_', up to 40, "
                                "starting with a letter or digit")
        if app in RESERVED:
            raise AppKeyRefused(f"{app!r} is the pool's own: the console's plan chat is counted under it")
        key = mint("app")
        now = time.time()
        record = AppKey(f"appkey-{secrets.token_hex(6)}", app, now, None)
        self.db.execute(
            "INSERT INTO app_keys (key_id, app, hashed, created_at, revoked_at, updated_at) VALUES (?, ?, ?, ?, NULL, ?)",
            (record.key_id, app, fingerprint(key), now, now),
        )
        return key, record

    def revoke(self, app: str, key_id: Optional[str] = None) -> list[str]:
        """Revoke one of an app's keys, or every one of them; the ids revoked. A key already
        revoked stays as it was."""
        now = time.time()
        if key_id is None:
            rows = self.db.query("SELECT key_id FROM app_keys WHERE app = ? AND revoked_at IS NULL", (app,))
        else:
            rows = self.db.query("SELECT key_id FROM app_keys WHERE app = ? AND key_id = ? AND revoked_at IS NULL",
                                 (app, key_id))
        ids = [r["key_id"] for r in rows]
        for found in ids:
            self.db.execute("UPDATE app_keys SET revoked_at = ?, updated_at = ? WHERE key_id = ?", (now, now, found))
        return ids

    # --- read by both ---

    def all(self) -> list[AppKey]:
        return [_key(r) for r in self.db.query("SELECT * FROM app_keys ORDER BY app, created_at")]

    def known(self, app: str) -> bool:
        return bool(self.db.query("SELECT 1 FROM app_keys WHERE app = ? LIMIT 1", (app,)))

    def usable_hashes(self) -> dict[str, str]:
        """hash → app, for every key not revoked: what the router matches a request's key against."""
        return {r["hashed"]: r["app"] for r in self.db.query(
            "SELECT hashed, app FROM app_keys WHERE revoked_at IS NULL")}

    def revision(self) -> float:
        """A cheap "has anything changed?" for the router's background read."""
        row = self.db.query("SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM app_keys")[0]
        return float(row["t"] or 0) + float(row["n"] or 0)
