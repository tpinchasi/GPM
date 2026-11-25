"""The router's view of the host table the supervisor publishes.

docs/spec/supervisor.md §1. The router reads the table; it never writes it. A background task
keeps an in-memory snapshot, so the request path never touches SQLite — and if the supervisor
dies, the router keeps serving from the last snapshot it saw.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx

from .. import hosted_credentials
from ..app_keys import AppKeyStore
from ..catalog import ResolvedVariant
from ..config import PoolConfig
from ..db import CounterRow, HostCounters, HostRow, HostTable
from ..models import Host, HostState, Worker, WorkerState
from ..provisioning_store import KeyReach, ProvisioningStore
from ..workload_store import KeyGrant, Workload, WorkloadStore
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


def _follow_balance(host: Host, row: HostRow) -> None:
    """A hosted API whose balance the supervisor found at $0 (D188) is out of routing as if its
    provider had said so — unless an operator said its money is back since; and back in once the
    supervisor clears it. A mark the provider's own answer made is left alone."""
    if row.balance_out is not None and not (row.funds_reset_at is not None
                                            and row.funds_reset_at >= (row.balance_out_at or 0.0)):
        if host.out_of_funds is None:
            host.out_of_funds, host.out_of_funds_at = row.balance_out, row.balance_out_at or time.time()
            host.out_of_funds_by_balance = True
    elif host.out_of_funds_by_balance:
        host.out_of_funds = host.out_of_funds_at = None
        host.out_of_funds_by_balance = False


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
        #: When each configured host's typed key was saved, as its client was built (D171).
        self._key_stamps: dict[str, Optional[float]] = {}
        #: Workloads as the supervisor published them (D115): each one's state and model, and
        #: what each workload key's hash reaches. Read in the background, never on a request.
        self.workload_store = WorkloadStore(table.db)
        self.workloads: dict[str, Workload] = {}
        self.grants: dict[str, KeyGrant] = {}
        self.last_workload_revision: Optional[float] = None
        #: Replacement gaps the supervisor marked (S19, D198), and its last heartbeat: a plan's
        #: stop-gap is opened only on a recent one (workloads.md §14.7).
        self.gaps: dict[tuple[str, str], tuple[float, float, frozenset[str], str]] = {}
        self.supervisor_beat: Optional[float] = None
        #: Each workload's plan rules whose conditions on the pass held at the supervisor's last
        #: look (S19, D200), by name.
        self.armed: dict[str, frozenset[str]] = {}
        #: Where a rule's steady switch has a workload's steady served now (S19 5b): `local` or
        #: `fixed`; absent, as its plan says.
        self.steady_serving: dict[str, str] = {}
        #: Provisioning keys (D117), as hash → provisioner, for keys neither revoked nor expired.
        self.provisioning_store = ProvisioningStore(table.db)
        self.provisioners: dict[str, KeyReach] = {}
        self.last_provisioner_revision: Optional[float] = None
        #: Named app keys for the shared pool (D193), as hash → app, for keys not revoked.
        self.app_key_store = AppKeyStore(table.db)
        self.app_keys: dict[str, str] = {}
        self.last_app_key_revision: Optional[float] = None

    def apply_config(self, config: PoolConfig) -> None:
        """A configuration applied while the router runs. The table still says which hosts
        there are; what changes here is how a configured host is reached — its credentials
        live only in configuration, so a host whose transport changed gets a new client."""
        before = self._host_config
        self.config = config
        self._host_config = {host.id: host for host in config.hosts}
        for host_id, host in self.by_id.items():
            old, new = before.get(host_id), self._host_config.get(host_id)
            if old is None or new is None:
                continue
            # A hosted API's key typed into the console (D171) is read here too: a new one, or one
            # no longer sent because the host now points elsewhere, needs a new client.
            key = hosted_credentials.stamp(config, new)
            if old.transport != new.transport or key != self._key_stamps.get(host_id):
                self._rekey(host, new)

    def _rekey(self, host: Host, host_config: Any) -> None:
        self._retired.append(host.client)
        host.client, self._key_stamps[host.host_id] = hosted_credentials.keyed_client(
            self.config, host_config, str(host.client.base_url))

    async def follow_hosted_keys(self) -> None:
        """A hosted API's key typed in, replaced or removed (D171) is used from the router's next
        read of the table, without a restart — found by comparing the key each client holds with
        the store, never with what an earlier look at the store saw. Found on CI: the file watcher
        noted "no key", a reload it started then read the key just saved, and the key's removal
        looked like no change at all, so the router kept sending a removed key."""
        hosted = {host_id: config for host_id, config in self._host_config.items()
                  if host_id in self.by_id and config.kind == "hosted"}
        if not hosted:
            return
        config = self.config
        now = await asyncio.to_thread(
            lambda: {host_id: hosted_credentials.stamp(config, c) for host_id, c in hosted.items()})
        if config is not self.config:
            return  # a reload came between; it read the keys itself, and the next look checks again
        for host_id, key in now.items():
            host = self.by_id.get(host_id)
            if host is not None and key != self._key_stamps.get(host_id):
                self._rekey(host, hosted[host_id])

    # --- reading the table ---

    async def refresh_workloads(self) -> bool:
        """Workloads and their keys, when either changed. A key past its time is still refused at
        the request, where the clock is read — a revision cannot move for time passing."""
        provisioner_revision = await asyncio.to_thread(self.provisioning_store.revision)
        if provisioner_revision != self.last_provisioner_revision:
            self.provisioners = await asyncio.to_thread(self.provisioning_store.usable_hashes)
            self.last_provisioner_revision = provisioner_revision
        app_key_revision = await asyncio.to_thread(self.app_key_store.revision)
        if app_key_revision != self.last_app_key_revision:
            self.app_keys = await asyncio.to_thread(self.app_key_store.usable_hashes)
            self.last_app_key_revision = app_key_revision
        # Every look, not only on a change: a gap opens and a heartbeat moves with nothing else new.
        gaps, self.supervisor_beat, armed, switches = await asyncio.to_thread(
            lambda: (self.workload_store.gaps(), self.workload_store.supervisor_heartbeat(self.config.pool.name),
                     self.workload_store.armed_rules(), self.workload_store.steady_switches()))
        serving = {name: s["serving"] for name, s in switches.items() if (s or {}).get("serving")}
        if gaps != self.gaps or armed != self.armed or serving != self.steady_serving:
            self.steady_serving = serving
            # Requests already waiting may now use a stop-gap, or no longer may; a rule came into
            # force or left it.
            self.gaps, self.armed = gaps, armed
            await self.dispatcher.wake()
        revision = await asyncio.to_thread(self.workload_store.revision)
        if revision == self.last_workload_revision:
            return False
        workloads = await asyncio.to_thread(self.workload_store.all)
        grants = await asyncio.to_thread(self.workload_store.grants)
        self.workloads = {w.name: w for w in workloads}
        self.grants = grants
        self.last_workload_revision = revision
        await self.dispatcher.wake()
        return True

    async def refresh(self) -> bool:
        await self.refresh_workloads()
        await self.follow_hosted_keys()
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
        built = Host(
            host_id=row.host_id,
            kind=row.kind,
            priority=row.priority,
            capabilities=frozenset(row.capabilities),
            transport_type=row.transport_type,
            client=self._client_for(row),
            workers=[Worker(worker_id=f"{row.host_id}/w{i}") for i in range(row.workers)],
            variants=variants,
            literal_variants=_literal_variants(variants),
            observed_inputs={tag: frozenset(found) for tag, found in row.observed.items()},
            state=HostState(row.state),
            resident=row.resident,
            available=row.available,
            residency=row.residency,
            engine=row.engine,
            last_error=row.last_error,
            workload=row.workload,
            burst=row.burst,
        )
        _follow_balance(built, row)
        return built

    def host_config(self, host_id: str) -> Any:
        """A configured host's settings, or None for one the pool rented."""
        return self._host_config.get(host_id)

    def _client_for(self, row: HostRow) -> httpx.AsyncClient:
        """Credentials come from configuration, never from the database — no secret crosses
        the process boundary. A host the pool rented is reached through its tunnel, which
        needs none."""
        host_config = self._host_config.get(row.host_id)
        if host_config is not None:
            client, self._key_stamps[row.host_id] = hosted_credentials.keyed_client(
                self.config, host_config, row.dial_url)
            return client
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
        host.kind = row.kind
        host.state = HostState(row.state)
        host.resident = row.resident
        host.available = row.available
        host.residency = row.residency
        host.engine = row.engine
        host.last_error = row.last_error
        host.workload = row.workload
        host.burst = row.burst
        host.capabilities = frozenset(row.capabilities)
        variants = _variants(row)
        host.variants = variants
        host.literal_variants = _literal_variants(variants)
        host.observed_inputs = {tag: frozenset(found) for tag, found in row.observed.items()}
        if (host.out_of_funds is not None and row.funds_reset_at is not None
                and row.funds_reset_at >= (host.out_of_funds_at or 0.0)):
            # An operator said its money is back, after its provider said it was gone (D164).
            host.out_of_funds = host.out_of_funds_at = None
            host.out_of_funds_by_balance = False
        _follow_balance(host, row)

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
                    borrowed=sum(1 for w in host.workers if w.borrowed and w.state is WorkerState.BUSY),
                    out_of_funds=host.out_of_funds,
                    out_of_funds_at=host.out_of_funds_at,
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
