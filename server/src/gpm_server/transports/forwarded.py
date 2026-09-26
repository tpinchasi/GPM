"""A forward kept by the forwarder, asked for through the database (D110).

The supervisor's side of `gpm forwarder`. It has the shape `SshTunnel` has — a fixed local URL,
`start`, `stop`, whether it is up — but runs nothing itself: it writes the forward it wants to
the `forwards` table and reads back how the forwarder is keeping it. So the forward belongs to
a process that does not restart with the supervisor, and the local port the router dials stays
the same for the host's whole life, across as many supervisor restarts as there are.

`stop` is for a forward that is no longer wanted — its host released — and removes the row, so
the forwarder closes it. A supervisor that is only shutting down calls `detach`, which leaves
the row, and the forward, for the next supervisor to find.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from ..config import TransportConfig
from ..db import Database
from .tunnel import allocate_local_port


def _fields(transport: TransportConfig) -> str:
    """The forward's fields, in a stable form: equal JSON means the same forward."""
    return json.dumps(transport.model_dump(mode="json", exclude={"local_port"}), sort_keys=True)


class ForwardedTunnel:
    def __init__(self, host_id: str, transport: TransportConfig, database: Database):
        self.host_id = host_id
        self.transport = transport
        self.db = database
        self.local_port: int = transport.local_port or 0
        self.local_url = ""
        #: True when this forward was already wanted, with the same fields, when this supervisor
        #: asked for it — a forward that outlived the last supervisor, still on its port.
        self.reused = False

    # --- the row ---

    def _row(self) -> Optional[Any]:
        rows = self.db.query("SELECT * FROM forwards WHERE name = ?", (self.host_id,))
        return rows[0] if rows else None

    def _want(self) -> None:
        fields = _fields(self.transport)
        row = self._row()
        if row is not None:
            # Same forward, still wanted: keep its port, which is what the router dials.
            self.reused = row["transport"] == fields
            if self.transport.local_port is None:
                self.local_port = int(row["local_port"])
            if not self.reused or int(row["local_port"]) != self.local_port:
                self.db.execute(
                    "UPDATE forwards SET transport = ?, local_port = ?, wanted_at = ?, state = 'wanted' "
                    "WHERE name = ?",
                    (fields, self.local_port, time.time(), self.host_id),
                )
        else:
            self.local_port = self.local_port or allocate_local_port()
            self.db.execute(
                "INSERT INTO forwards (name, transport, local_port, wanted_at) VALUES (?, ?, ?, ?)",
                (self.host_id, fields, self.local_port, time.time()),
            )
        self.local_url = f"http://127.0.0.1:{self.local_port}"

    # --- what the supervisor reads ---

    @property
    def up(self) -> bool:
        row = self._row()
        return bool(row) and row["state"] == "up"

    @property
    def last_error(self) -> Optional[str]:
        row = self._row()
        return row["last_error"] if row else None

    @property
    def restarts(self) -> int:
        row = self._row()
        return int(row["restarts"]) if row else 0

    # --- what the supervisor does ---

    async def start(self, wait_s: float = 10.0) -> bool:
        """Ask for the forward, and wait up to `wait_s` for it to listen. A forward still coming
        up is not an error: the forwarder keeps at it, as the supervisor's own tunnels did."""
        self._want()
        deadline = time.monotonic() + wait_s
        while True:
            if await self._listening():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """The forward is no longer wanted: its host has gone. The forwarder closes it."""
        self.db.execute("DELETE FROM forwards WHERE name = ?", (self.host_id,))

    async def detach(self) -> None:
        """This supervisor is going; the forward is not. The next supervisor finds it."""

    async def _listening(self) -> bool:
        if not self.local_port:
            return False
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.local_port), timeout=0.5
            )
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True
