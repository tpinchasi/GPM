"""What the router process holds: the engine, the dispatcher, and its view of the host table.

Hosts, tunnels and probing belong to the supervisor now (docs/spec/supervisor.md §1). The
router reads what the supervisor published and keeps serving from it even when the supervisor
is gone.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

from .catalog import for_engine
from .config import PoolConfig, Variant
from .db import Database, HostCounters, HostTable, RequestLog
from .engines import Engine, get_engine, kinds_of
from .models import Host, HostState
from .router.dispatch import Dispatcher, Need
from .router.registry import HostRegistry

#: What a build the catalog does not list serves and takes: chat and embeddings, nothing stated.
_UNLISTED: tuple[frozenset[str], Optional[frozenset[str]]] = (frozenset({"chat", "embed"}), None)


@dataclasses.dataclass(frozen=True)
class _Holder:
    """A build some host could hold: a configured host by its id, or (None) one the pool rents."""

    host_id: Optional[str]
    tag: str
    kinds: frozenset[str]
    accepts: Optional[frozenset[str]]


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
        #: Request bytes held right now — being read, queued, in flight, kept for a resend (D144).
        self.held_bytes = 0
        self._learn_builds(config)

    def _learn_builds(self, config: PoolConfig) -> None:
        """What each build does and takes, which builds of each name some host could hold, and
        what each engine serves — read from configuration and the engines already loaded, once
        per configuration, never from the published host rows. A request only looks up what was
        learned here: nothing on the request path loads an engine or walks the catalog (D143)."""
        traits = config.build_traits()
        self.dispatcher.traits = lambda tag: traits.get(tag, _UNLISTED)
        #: What each engine this pool runs serves (D142), from the engine itself.
        self._engine_kinds = {name: kinds_of(engine) for name, engine in self.engines.items()}
        #: Every catalogued build by its tag — the last listed wins, as in `build_traits`.
        self._by_tag = {variant.tag: variant for entry in config.catalog.values() for variant in entry.variants}
        #: Which builds each configured host could hold, by tag: a build of a model it holds, for
        #: its engine, of a kind its engine serves. What its platform requires is not asked — an
        #: agent may find the platform later — so this errs towards a host that could.
        self._host_tags: dict[str, frozenset[str]] = {}
        for host in config.hosts:
            if host.disabled:
                continue
            engine = config.engine_of(host)
            self._host_tags[host.id] = frozenset(
                variant.tag
                for name in config.models_held_by(host)
                for variant in self._builds_of(name)
                if self._engine_takes(engine, variant)
            )
        self._rented_engine = config.rented_engine() if config.rented is not None else None
        # Every name a request can reach a host by — catalogued names, build tags, the model
        # set's own names — judged now. A name learned some other way is judged when asked,
        # from the same learned facts.
        names = set(config.catalog) | set(self._by_tag) | set(config.pool.model_set)
        self._judged = {name: self._judge(name) for name in names}
        self._holders = {name: self._holders_of(name) for name in names}

    # --- what was learned, read per request ---

    def _builds_of(self, name: str) -> list[Variant]:
        """The builds a request naming this is judged by (D143), found in the order dispatch
        resolves a name (`Dispatcher.variant_for`): a catalogued name's own builds; else the one
        build a tag names; else an uncatalogued name as its own one build, which states nothing."""
        entry = self.config.catalog.get(name)
        if entry is not None:
            return list(entry.variants)
        if name in self._by_tag:
            return [self._by_tag[name]]
        return [Variant(tag=name)]

    def _engine_takes(self, engine: Optional[str], variant: Variant) -> bool:
        return engine in self._engine_kinds and for_engine(variant, engine) and bool(
            variant.kinds() & self._engine_kinds[engine]
        )

    def _judge(self, name: str) -> tuple[tuple[frozenset[str], Optional[frozenset[str]]], ...]:
        """(kinds, accepts) of every build of the name that some engine this pool runs could
        serve — `PoolConfig.builds_reachable`'s rule, with the engines already loaded."""
        return tuple(
            (variant.kinds(), variant.accepts())
            for variant in self._builds_of(name)
            if any(self._engine_takes(engine, variant) for engine in self._engine_kinds)
        )

    def _holders_of(self, name: str) -> tuple[_Holder, ...]:
        found = [
            _Holder(host_id, variant.tag, variant.kinds(), variant.accepts())
            for variant in self._builds_of(name)
            for host_id, tags in self._host_tags.items()
            if variant.tag in tags
        ]
        if self._rented_engine is not None:
            found += [
                _Holder(None, variant.tag, variant.kinds(), variant.accepts())
                for variant in self._builds_of(name)
                if self._engine_takes(self._rented_engine, variant)
            ]
        return tuple(found)

    def refusal_for(self, model: str, kind: str, inputs: frozenset[str]) -> Optional[tuple[str, str]]:
        """Why no host this pool has or rents could ever serve this, or None (D143). Waiting cannot
        help such a request, so it is refused at once rather than queued into a `503`."""
        builds = self._judged.get(model)
        if builds is None:
            builds = self._judge(model)
        doing = [accepts for kinds, accepts in builds if kind in kinds]
        if not doing:
            return "kind_not_served", (
                f"no build of {model!r} that this pool's hosts could hold serves {kind}; ask the operator "
                "for a model that does"
            )
        if inputs and not any(accepts is None or inputs <= accepts for accepts in doing):
            return "input_not_accepted", (
                f"no build of {model!r} that this pool's hosts could hold accepts {sorted(inputs)}"
            )
        return None

    def refused_for_inputs_alone(self, need: Need, exclude: frozenset[str]) -> bool:
        """Whether what the request carries is the one reason no host takes it, and waiting cannot
        change that (D143) — so `400 input_not_accepted` rather than `503 no_eligible_host`.

        Only when ready hosts would serve it without its picture, no host the pool could rent
        would take it, and every configured host that could hold a build taking it is ready and
        either does not hold that build or reports, through its engine, that the build does not
        take it. A host that is not ready may come back holding one that does; a host this
        request already failed on may have been the one that would. Either is a `503`, which an
        app waits out.
        """
        if exclude:
            return False
        if not self.dispatcher.eligible(dataclasses.replace(need, inputs=frozenset())):
            return False
        holders = self._holders.get(need.model)
        if holders is None:
            holders = self._holders_of(need.model)
        # The pool's configured hosts are the shared workload's: a workload's request reaches
        # them only while it may borrow (D115).
        reaches_shared = need.workload is None or need.may_borrow
        for holder in holders:
            if need.kind not in holder.kinds or not (holder.accepts is None or need.inputs <= holder.accepts):
                continue
            if holder.host_id is None:
                return False
            if not reaches_shared:
                continue
            host = self.registry.by_id.get(holder.host_id)
            if host is None or host.state is not HostState.READY:
                return False
            if holder.tag not in host.servable:
                continue  # not on this host, and nothing fetches it because a request asked (D39)
            seen = host.observed_inputs.get(holder.tag)
            if seen is None or need.inputs <= (seen if holder.accepts is None else holder.accepts & seen):
                return False
        return True

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
        where it will go. The engines shipped with the pool that serve the same path read it
        identically, by construction: they share one module for it. Where a version-1 plug-in
        shares a path with one of them, the version-2 engine reads it: a version-1 reader finds
        the model and nothing of what the request carries, and the pool's guards on pictures
        and sound — inline only, sent only to a build that takes them — rest on that reading.
        """
        serving = [engine for engine in self.engines.values() if path in engine.inference_paths()]
        for engine in serving:
            if getattr(engine, "request_facts", None) is not None:
                return engine
        return serving[0] if serving else None

    def paths(self) -> set[str]:
        """Every inference path this pool's hosts serve between them."""
        found: set[str] = set()
        for engine in self.engines.values():
            found |= engine.inference_paths()
        return found

    async def aclose(self) -> None:
        await self.registry.aclose()
