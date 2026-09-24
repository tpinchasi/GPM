"""What the router process holds: the engine, the dispatcher, and its view of the host table.

Hosts, tunnels and probing belong to the supervisor now (docs/spec/supervisor.md §1). The
router reads what the supervisor published and keeps serving from it even when the supervisor
is gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PoolConfig
from .db import Database, HostCounters, HostTable, RequestLog
from .engines import Engine, get_engine
from .models import Host
from .router.dispatch import Dispatcher
from .router.registry import HostRegistry


def open_database(config: PoolConfig, path: Optional[str | Path] = None) -> Database:
    return Database(path or config.request_log)


class RouterState:
    def __init__(self, config: PoolConfig, database: Database):
        self.config = config
        self.db = database
        # Every engine this pool runs (D93). The pool's own is first and remains "the engine"
        # for anything that asks without naming a host.
        self.engines: dict[str, Engine] = {
            name: get_engine(name) for name in config.engines_in_use()
        }
        self.engine: Engine = self.engines[config.engine]
        self.app_hashes = config.auth.app_hashes()
        self.dispatcher = Dispatcher([])
        self.registry = HostRegistry(
            config, HostTable(database), HostCounters(database), self.dispatcher
        )
        self.log = RequestLog(database)

    @property
    def hosts(self) -> list[Host]:
        return self.dispatcher.hosts

    def engine_of(self, host: Host) -> Engine:
        """The engine on that machine, falling back to the pool's for a host published before
        engines could differ."""
        return self.engines.get(host.engine, self.engine)

    def parser_for(self, path: str) -> Optional[Engine]:
        """Which engine's rules read a request that arrived on this path (D93).

        The **path** decides, not the host — a request must be understood before the pool knows
        where it will go. Where two engines serve the same path they read it identically, by
        construction: they share one module for it. Where only one serves it, that one reads it.
        """
        for engine in self.engines.values():
            if path in engine.inference_paths():
                return engine
        return None

    def paths(self) -> set[str]:
        """Every inference path this pool's hosts serve between them."""
        found: set[str] = set()
        for engine in self.engines.values():
            found |= engine.inference_paths()
        return found

    async def aclose(self) -> None:
        await self.registry.aclose()
