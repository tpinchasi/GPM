"""The router's view of the host table the supervisor publishes.

docs/spec/supervisor.md §1. The router reads the table; it never writes it. A background task
keeps an in-memory snapshot, so the request path never touches SQLite — and if the supervisor
dies, the router keeps serving from the last snapshot it saw.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from ..catalog import ResolvedVariant
from ..config import PoolConfig
from ..db import CounterRow, HostCounters, HostRow, HostTable
from ..models import Host, HostState, Worker, WorkerState
from ..transports import build_client
from .dispatch import Dispatcher

log = logging.getLogger("gpm.registry")


def _variants(row: HostRow) -> dict[str, tuple[ResolvedVariant, ...]]:
    return {
        name: tuple(
            ResolvedVariant(tag=tag, runtime_class=runtime_class, enforces_schema=enforces)
            for tag, runtime_class, enforces in variants
        )
        for name, variants in row.variants.items()
    }


def _literal_variants(variants: dict[str, tuple[ResolvedVariant, ...]]) -> dict[str, ResolvedVariant]:
    return {variant.tag: variant for group in variants.values() for variant in group}


class HostRegistry:
    def __init__(self, config: PoolConfig, table: HostTable, counters: HostCounters, dispatcher: Dispatcher):
        self.config = config
        self.table = table
        self.counters = counters
        self.dispatcher = dispatcher
        self.by_id: dict[str, Host] = {}
        self.last_revision: Optional[float] = None
        self.last_seen_at: Optional[float] = None
        self._host_config = {host.id: host for host in config.hosts}
        self._retired: list[httpx.AsyncClient] = []

    def apply_config(self, config: PoolConfig) -> None:
        """A configuration applied while the router runs. The table still says which hosts
        there are; what changes here is how a configured host is reached — its credentials
        live only in configuration, so a host whose transport changed gets a new client."""
        before = self._host_config
        self.config = config
        self._host_config = {host.id: host for host in config.hosts}
        for host_id, host in self.by_id.items():
            old, new = before.get(host_id), self._host_config.get(host_id)
            if old is not None and new is not None and old.transport != new.transport:
                self._retired.append(host.client)
                host.client = build_client(new.transport, config.pool, str(host.client.base_url))

    # --- reading the table ---

    async def refresh(self) -> bool:
        revision = await asyncio.to_thread(self.table.revision)
        if revision == self.last_revision:
            return False
        rows = await asyncio.to_thread(self.table.all)
        await self.dispatcher.update(lambda: self._apply(rows))
        self.last_revision = revision
        self.last_seen_at = asyncio.get_running_loop().time()
        return True

    def _apply(self, rows: list[HostRow]) -> None:
        seen = set()
        for row in rows:
            seen.add(row.host_id)
            host = self.by_id.get(row.host_id)
            if host is None:
                self.by_id[row.host_id] = self._build(row)
                self.dispatcher.hosts.append(self.by_id[row.host_id])
            else:
                self._update(host, row)

        for host_id in list(self.by_id):
            if host_id in seen:
                continue
            # Gone from the table: stop routing to it at once, and drop it once its workers
            # have finished what they were already serving.
            host = self.by_id[host_id]
            host.state = HostState.DISABLED
            if host.busy == 0:
                self.dispatcher.hosts.remove(host)
                self._retired.append(host.client)
                del self.by_id[host_id]

    def _build(self, row: HostRow) -> Host:
        variants = _variants(row)
        return Host(
            host_id=row.host_id,
            kind=row.kind,
            priority=row.priority,
            capabilities=frozenset(row.capabilities),
            transport_type=row.transport_type,
            client=self._client_for(row),
            workers=[Worker(worker_id=f"{row.host_id}/w{i}") for i in range(row.workers)],
            variants=variants,
            literal_variants=_literal_variants(variants),
            state=HostState(row.state),
            resident=row.resident,
            available=row.available,
            residency=row.residency,
            engine=row.engine,
            last_error=row.last_error,
        )

    def _client_for(self, row: HostRow) -> httpx.AsyncClient:
        """Credentials come from configuration, never from the database — no secret crosses
        the process boundary. A host the pool rented is reached through its tunnel, which
        needs none."""
        host_config = self._host_config.get(row.host_id)
        if host_config is not None:
            return build_client(host_config.transport, self.config.pool, row.dial_url)
        return httpx.AsyncClient(
            base_url=row.dial_url,
            timeout=httpx.Timeout(
                connect=self.config.pool.upstream_connect_timeout_s,
                read=self.config.pool.upstream_read_timeout_s,
                write=self.config.pool.upstream_read_timeout_s,
                pool=self.config.pool.upstream_connect_timeout_s,
            ),
        )

    def _update(self, host: Host, row: HostRow) -> None:
        """Every field routing reads, copied from the row — the same set `_build` sets.

        Found live: this copied the loaded models but not the on-disk list, the residency or
        the engine. A laptop that was disabled when the router started was first seen with an
        empty disk; when it came back with its models on disk and nothing loaded, the router
        kept the empty list, and on an on-demand host that meant nothing was servable — every
        request refused as "no eligible host" for as long as the router ran.
        """
        host.priority = row.priority
        host.state = HostState(row.state)
        host.resident = row.resident
        host.available = row.available
        host.residency = row.residency
        host.engine = row.engine
        host.last_error = row.last_error
        host.capabilities = frozenset(row.capabilities)
        variants = _variants(row)
        host.variants = variants
        host.literal_variants = _literal_variants(variants)

        if str(host.client.base_url).rstrip("/") != row.dial_url.rstrip("/"):
            self._retired.append(host.client)
            host.client = self._client_for(row)

        self._resize(host, row.workers)

    def _resize(self, host: Host, wanted: int) -> None:
        """Raised: new workers start idle at once. Lowered: surplus idle workers go now, and
        busy ones are left to finish (docs/spec/console-and-control-api.md §3)."""
        current = len(host.workers)
        if wanted > current:
            host.workers.extend(
                Worker(worker_id=f"{host.host_id}/w{i}") for i in range(current, wanted)
            )
        elif wanted < current:
            surplus = current - wanted
            for worker in reversed(host.workers):
                if surplus == 0:
                    break
                if worker.state is WorkerState.IDLE:
                    host.workers.remove(worker)
                    surplus -= 1

    # --- writing counters back ---

    async def publish_counters(self) -> None:
        await self.counters.publish(
            [
                CounterRow(
                    host_id=host.host_id,
                    busy=host.busy,
                    total=host.total_workers,
                    requests_served=host.requests_served,
                    failures=host.failures,
                    last_request_at=host.last_request_at,
                )
                for host in self.by_id.values()
            ]
        )

    async def run_forever(self, interval_s: float) -> None:
        while True:
            try:
                await self.refresh()
                await self.publish_counters()
                await self._close_retired()
            except Exception:  # the router must keep serving whatever the table does
                log.exception("host table refresh failed")
            await asyncio.sleep(interval_s)

    async def _close_retired(self) -> None:
        while self._retired:
            client = self._retired.pop()
            try:
                await client.aclose()
            except Exception:
                pass

    async def aclose(self) -> None:
        await self._close_retired()
        for host in self.by_id.values():
            await host.client.aclose()
