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
        self.engine: Engine = get_engine(config.engine)
        self.app_hashes = config.auth.app_hashes()
        self.dispatcher = Dispatcher([])
        self.registry = HostRegistry(
            config, HostTable(database), HostCounters(database), self.dispatcher
        )
        self.log = RequestLog(database)

    @property
    def hosts(self) -> list[Host]:
        return self.dispatcher.hosts

    async def aclose(self) -> None:
        await self.registry.aclose()
