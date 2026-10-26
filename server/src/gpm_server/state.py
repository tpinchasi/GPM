"""What the router process holds: the engine, the dispatcher, and its view of the host table.

Hosts, tunnels and probing belong to the supervisor now (docs/spec/supervisor.md §1). The
router reads what the supervisor published and keeps serving from it even when the supervisor
is gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PoolConfig, Variant
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
        #: Request bytes held right now — queued, in flight, kept for a resend (D144).
        self.held_bytes = 0
        self._learn_builds(config)

    def _learn_builds(self, config: PoolConfig) -> None:
        """What each build does and takes, and which builds of each name some host could hold —
        read from configuration, once per configuration, never from the published host rows."""
        traits = config.build_traits()
        self.dispatcher.traits = lambda tag: traits.get(tag, (frozenset({"chat", "embed"}), None))
        self._reachable: dict[str, list[Variant]] = {}

    def refusal_for(self, model: str, kind: str, inputs: frozenset[str]) -> Optional[tuple[str, str]]:
        """Why no host this pool has or rents could ever serve this, or None (D143). Waiting cannot
        help such a request, so it is refused at once rather than queued into a `503`."""
        builds = self._reachable.get(model)
        if builds is None:
            by_tag = {
                variant.tag: variant
                for name in self.config.catalog
                for variant in self.config.builds_reachable(name)
            }
            builds = [by_tag[model]] if model in by_tag else self.config.builds_reachable(model)
            self._reachable[model] = builds
        doing = [build for build in builds if kind in build.kinds()]
        if not doing:
            return "kind_not_served", (
                f"no build of {model!r} that this pool's hosts could hold serves {kind}; ask the operator "
                "for a model that does"
            )
        if inputs and not any(build.accepts() is None or inputs <= build.accepts() for build in doing):
            return "input_not_accepted", (
                f"no build of {model!r} that this pool's hosts could hold accepts {sorted(inputs)}"
            )
        return None

    def take_request_bytes(self, size: int) -> bool:
        """Hold a request's bytes, unless that would pass `pool.max_held_request_mb` (D144)."""
        if self.held_bytes + size > self.config.pool.max_held_request_mb * 1_000_000:
            return False
        self.held_bytes += size
        return True

    def give_back_request_bytes(self, size: int) -> None:
        self.held_bytes = max(0, self.held_bytes - size)

    def apply(self, config: PoolConfig) -> None:
        """Serve under a configuration applied while the router runs: the model set, the
        catalog's engines, the app keys and every limit the next request reads. The listen
        address and TLS are the exception — they are the socket itself, and change only with a
        restart. Hosts come from the supervisor's table, as always."""
        self.config = config
        self.engines = {name: get_engine(name) for name in config.engines_in_use()}
        self.engine = self.engines[config.engine]
        self.app_hashes = config.auth.app_hashes()
        self.registry.apply_config(config)
        self._learn_builds(config)

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
