"""The supervisor: the half of the pool that is allowed to be slow.

docs/spec/supervisor.md §1 and §3. It does the blocking, failure-prone work — tunnels, probes,
and from stage 2b providers, bidding and tear-down — and publishes the host table the router
reads. The two processes never call each other.

Stage 2a is the split itself: no provider, no lease, nothing that can spend. Every host here
comes from configuration.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from ..catalog import ResolvedVariant, variants_for_host
from ..config import AgentConfig, ConfigError, HostConfig, PoolConfig, load_config
from ..configplan import ConfigStore
from ..db import Database, HostCounters, HostRow, HostTable, SupervisorLock
from ..engines import Engine, get_engine
from ..ledger import EventLog, LeaseStore, SpendLedger
from ..models import HostState
from ..providers.base import get_provider
from ..transports import SshTunnel, build_client
from . import agents
from .renting import Fleet

log = logging.getLogger("gpm.supervisor")


@dataclasses.dataclass
class SupervisedHost:
    """What the supervisor knows about one host. The router has its own view."""

    config: HostConfig
    client: httpx.AsyncClient
    variants: dict[str, tuple[ResolvedVariant, ...]]
    dial_url: str
    tunnel: Optional[SshTunnel] = None
    state: HostState = HostState.UNREACHABLE
    resident: frozenset[str] = frozenset()
    available: frozenset[str] = frozenset()
    #: The last answer from this host's agent, if it has one (docs/spec/host-agent.md).
    agent: Optional[agents.AgentView] = None
    #: The forward to an agent that sits on the far side of this host's SSH tunnel.
    agent_tunnel: Optional[SshTunnel] = None
    last_error: Optional[str] = None
    last_probe_at: Optional[float] = None

    @property
    def host_id(self) -> str:
        return self.config.id

    @property
    def agent_endpoint(self) -> Optional[AgentConfig]:
        """The agent as it is actually dialled: where configuration gave a port on the far
        side of the tunnel, the pool's own forwarded loopback port."""
        agent = self.config.agent
        if agent is None or self.agent_tunnel is None:
            return agent
        return agent.model_copy(update={"url": self.agent_tunnel.local_url, "remote_port": None})

    @property
    def capabilities(self) -> frozenset[str]:
        """The configured list, with the platform the agent found where it names none."""
        return agents.effective_capabilities(self.config.capabilities, self.agent)

    @property
    def required_tags(self) -> frozenset[str]:
        """Its preferred variant of every model in the pool's set (spec §3)."""
        return frozenset(variants[0].tag for variants in self.variants.values() if variants)


class Supervisor:
    def __init__(
        self,
        config: PoolConfig,
        database: Database,
        provider: Optional[object] = None,
        config_path: Optional[str] = None,
    ):
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        #: Present only when the pool was started from a file — the console edits it *through*
        #: the API and holds no settings of its own.
        self.store = ConfigStore(self.config_path) if self.config_path else None
        self._config_mtime = self.config_path.stat().st_mtime if self.config_path else None
        self.db = database
        self.engine: Engine = get_engine(config.engine)
        self.table = HostTable(database)
        self.counters = HostCounters(database)
        self.lock = SupervisorLock(
            database,
            pool=config.pool.name,
            owner=f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}",
        )
        self.leases = LeaseStore(database)
        self.events = EventLog(database)
        self.spend = SpendLedger(database)
        self.hosts: dict[str, SupervisedHost] = {}
        self.passes = 0
        self._stopping = False
        for host_config in config.hosts:
            self.hosts[host_config.id] = self._build(host_config)

        #: Absent `rented` means the pool has no way to spend at all.
        self.fleet: Optional[Fleet] = None
        if config.rented is not None:
            self.fleet = Fleet(
                config,
                config.rented,
                provider or get_provider(config.rented.provider, config.rented.provider_settings),
                self.leases,
                self.events,
                self.spend,
            )
        #: Clients for hosts the pool rented, keyed by host id.
        self._rented_clients: dict[str, httpx.AsyncClient] = {}
        #: Model-set preparations in flight, keyed by host id.
        self._preparing: dict[str, asyncio.Task[None]] = {}
        #: Tests substitute a transport that answers as an agent would.
        self._agent_transport: Optional[httpx.AsyncBaseTransport] = None

    def _build(self, host_config: HostConfig) -> SupervisedHost:
        capabilities = frozenset(host_config.capabilities)
        tunnel = (
            SshTunnel(host_config.id, host_config.transport)
            if host_config.transport.type == "tunnel"
            else None
        )
        dial_url = tunnel.local_url if tunnel else (host_config.transport.base_url or "")
        return SupervisedHost(
            config=host_config,
            client=build_client(host_config.transport, self.config.pool, dial_url),
            variants=variants_for_host(
                self.config.pool.model_set,
                self.config.catalog,
                capabilities,
                self.engine.name,
            ),
            dial_url=dial_url,
            tunnel=tunnel,
            state=HostState.DISABLED if host_config.disabled else HostState.UNREACHABLE,
        )

    def _sync_agent_tunnel(self, host: SupervisedHost, previous: Optional[AgentConfig] = None) -> Optional[SshTunnel]:
        """Make the agent's forward match configuration, without touching the engine's: adding,
        moving or removing an agent must not interrupt the traffic the host is serving. Returns
        a tunnel that now needs starting."""
        agent = host.config.agent
        wanted = agent.remote_port if agent is not None else None
        running = host.agent_tunnel.transport.remote_port if host.agent_tunnel is not None else None
        if wanted == running:
            return None
        if host.agent_tunnel is not None:
            asyncio.create_task(host.agent_tunnel.stop())
            host.agent_tunnel = None
        if wanted is None:
            return None
        forward = host.config.transport.model_copy(
            update={"remote_host": "127.0.0.1", "remote_port": wanted, "local_port": None}
        )
        host.agent_tunnel = SshTunnel(f"{host.host_id}:agent", forward)
        return host.agent_tunnel

    # --- lifecycle ---

    async def start(self) -> None:
        self.lock.acquire()
        for host in self.hosts.values():
            if host.tunnel is not None:
                await host.tunnel.start()
            forward = self._sync_agent_tunnel(host)
            if forward is not None:
                await forward.start()
        await self.adopt_rented()
        await self.pass_once()

    async def adopt_rented(self) -> None:
        """Before the first pass — and so before the first sweep — take back the rented hosts
        the last supervisor published, and drop the rows of hosts the provider no longer has."""
        rows = [row for row in self.table.all() if row.kind == "rented-interruptible"]
        if not rows:
            return
        if self.fleet is None:
            for row in rows:
                self.table.remove(row.host_id)
            return
        adopted = set(await self.fleet.adopt(rows))
        for row in rows:
            if row.host_id not in adopted:
                self.table.remove(row.host_id)
        if adopted:
            log.info("adopted %d rented host(s) from the previous supervisor: %s", len(adopted), sorted(adopted))

    async def run_forever(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self.config.pool.probe_interval_s)
            if self._stopping:
                return
            try:
                await self.pass_once()
            except Exception:  # a bad pass must never take the supervisor down
                log.exception("control loop pass failed")

    async def aclose(self) -> None:
        self._stopping = True
        for host in self.hosts.values():
            if host.tunnel is not None:
                await host.tunnel.stop()
            if host.agent_tunnel is not None:
                await host.agent_tunnel.stop()
            await host.client.aclose()
        for task in self._preparing.values():
            task.cancel()
        for client in self._rented_clients.values():
            await client.aclose()
        if self.fleet is not None:
            for host_id in list(self.fleet.tunnels):
                await self.fleet.close_tunnel(host_id)
        self.lock.release()

    # --- the control loop ---

    async def pass_once(self) -> None:
        """Observe → update host states → compare demand → act → record (spec §3).

        Acting is strictly ordered inside the fleet: release what should not exist, recover
        what is broken, then acquire what is missing.
        """
        self._follow_config_file()
        await asyncio.gather(*(self._ask_agent(host) for host in self.hosts.values()))
        await asyncio.gather(*(self._probe(host) for host in self.hosts.values()))
        await self._probe_rented()
        self._publish_all()

        if self.fleet is not None:
            await self.fleet.pass_once(
                ready_workers_higher_tiers=self._ready_workers(),
                idle_seconds=self._idle_seconds(),
                busy={host_id: counter.busy for host_id, counter in self.counters.all().items()},
            )
            self._publish_rented()

        self.lock.beat()
        self.passes += 1

    # --- following the configuration file (spec §1) ---

    def _follow_config_file(self) -> None:
        """Editing the file by hand keeps working: the supervisor notices and reloads."""
        if self.config_path is None:
            return
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return
        if mtime == self._config_mtime:
            return
        self._config_mtime = mtime
        self.reload_config()

    def reload_config(self) -> None:
        """Take the file as it is now. A file that does not load leaves the running pool
        exactly as it was — a bad edit must never take a gpm down."""
        if self.config_path is None:
            return
        try:
            new = load_config(self.config_path)
        except ConfigError as exc:
            self.events.record("config_invalid", f"the configuration file does not load, so nothing changed: {exc}")
            log.error("configuration did not load; keeping the running one: %s", exc)
            return
        try:
            self._config_mtime = self.config_path.stat().st_mtime
        except OSError:
            pass
        self.apply_config(new)

    def apply_config(self, new: PoolConfig) -> None:
        old = self.config
        self.config = new
        self.app_keys_changed = old.auth != new.auth

        before = {host.id: host for host in old.hosts}
        after = {host.id: host for host in new.hosts}

        for host_id in before.keys() - after.keys():
            self._retire_host(host_id)
        for host_id, host_config in after.items():
            existing = self.hosts.get(host_id)
            rebuild = (
                existing is None
                or existing.config.transport != host_config.transport
                or existing.config.capabilities != host_config.capabilities
            )
            if rebuild:
                if existing is not None:
                    self._retire_host(host_id, remove_row=False)
                self.hosts[host_id] = self._build(host_config)
                tunnel = self.hosts[host_id].tunnel
                if tunnel is not None:
                    asyncio.create_task(tunnel.start())
            else:
                existing.config = host_config
                existing.variants = variants_for_host(
                    new.pool.model_set, new.catalog, existing.capabilities, self.engine.name
                )
                if host_config.disabled and existing.state is not HostState.DISABLED:
                    existing.state = HostState.DISABLED
                elif not host_config.disabled and existing.state is HostState.DISABLED:
                    existing.state = HostState.UNREACHABLE
            forward = self._sync_agent_tunnel(self.hosts[host_id])
            if forward is not None:
                asyncio.create_task(forward.start())

        if self.fleet is not None and new.rented is not None:
            self.fleet.config = new
            self.fleet.rented = new.rented
        self.events.record(
            "config_applied",
            f"configuration reloaded: {len(after)} configured host(s)",
            numbers={"hosts": sorted(after)},
        )
        log.info("configuration reloaded")

    def _retire_host(self, host_id: str, remove_row: bool = True) -> None:
        host = self.hosts.pop(host_id, None)
        if host is None:
            return
        if host.agent_tunnel is not None:
            asyncio.create_task(host.agent_tunnel.stop())
        if host.tunnel is not None:
            asyncio.create_task(host.tunnel.stop())
        asyncio.create_task(host.client.aclose())
        if remove_row:
            self.table.remove(host_id)

    def _ready_workers(self) -> int:
        """Capacity above the rented tier — what the overflow is measured against."""
        return sum(
            host.config.workers
            for host in self.hosts.values()
            if host.state is HostState.READY
        )

    def _idle_seconds(self) -> dict[str, float]:
        """How long each rented host has had nothing routed to it, from the counters the
        router writes (spec §1)."""
        now = time.time()
        counters = self.counters.all()
        idle: dict[str, float] = {}
        if self.fleet is None:
            return idle
        for host_id, host in self.fleet.hosts.items():
            counter = counters.get(host_id)
            if host.ready_at is None:
                idle[host_id] = 0.0  # not serving yet, so not idle either
            elif counter is None or counter.last_request_at is None:
                idle[host_id] = now - host.ready_at
            elif counter.busy > 0:
                idle[host_id] = 0.0
            else:
                idle[host_id] = max(0.0, now - max(counter.last_request_at, host.ready_at))
        return idle

    async def _ask_agent(self, host: SupervisedHost) -> None:
        """Before the probe, so a platform the agent found decides which build the probe looks
        for. A silent agent changes nothing: the host is judged by its engine, as ever."""
        if host.config.agent is None or host.config.disabled:
            return
        before = host.capabilities
        was_reachable = host.agent.reachable if host.agent else None
        models_before = host.agent.models if host.agent else None
        host.agent = await agents.ask(host.agent_endpoint, transport=self._agent_transport)
        if host.agent.reachable is not was_reachable:
            self.events.record(
                "agent_reachable" if host.agent.reachable else "agent_unreachable",
                f"host {host.host_id}: " + ("its agent answers" if host.agent.reachable else host.agent.detail or "agent silent"),
                host_id=host.host_id,
            )
        if host.capabilities != before:
            host.variants = variants_for_host(
                self.config.pool.model_set, self.config.catalog, host.capabilities, self.engine.name
            )
            self.events.record(
                "capabilities_derived",
                f"host {host.host_id}: the agent found {sorted(host.capabilities - before)}; builds re-resolved",
                host_id=host.host_id,
                numbers={"capabilities": sorted(host.capabilities)},
            )
        if host.agent.reachable and host.config.agent.manage_models:
            # After the platform is known, so the machine is asked for the right builds.
            models = await agents.hold(
                host.agent_endpoint, host.required_tags, host.config.residency, transport=self._agent_transport
            )
            host.agent = dataclasses.replace(host.agent, models=models)
            for kind, summary, numbers in agents.model_events(models_before, models):
                self.events.record(kind, f"host {host.host_id}: {summary}", host_id=host.host_id, numbers=numbers)
        elif host.agent.reachable is False and models_before is not None:
            host.agent = dataclasses.replace(host.agent, models=None)

    async def _probe_rented(self) -> None:
        """A rented host is verified exactly like any other: only `ready` hosts are routed to,
        and only when the pool's whole model set is resident."""
        if self.fleet is None:
            return
        for host_id, host in list(self.fleet.hosts.items()):
            if host.released or not host.dial_url:
                continue
            client = self._rented_clients.get(host_id)
            if client is None or str(client.base_url).rstrip("/") != host.dial_url.rstrip("/"):
                if client is not None:
                    await client.aclose()
                client = httpx.AsyncClient(
                    base_url=host.dial_url,
                    timeout=httpx.Timeout(
                        connect=self.config.pool.upstream_connect_timeout_s,
                        read=self.config.pool.upstream_read_timeout_s,
                        write=self.config.pool.upstream_read_timeout_s,
                        pool=self.config.pool.upstream_connect_timeout_s,
                    ),
                )
                self._rented_clients[host_id] = client

            health = await self.engine.health(client)
            if not health.ok:
                host.mark_preparing()
                continue
            if host.engine_seen_at is None:
                host.engine_seen_at = time.time()  # it has started; "stuck starting" is over
            try:
                resident = await self.engine.models_resident(client)
            except httpx.HTTPError:
                host.mark_preparing()
                continue
            required = self._rented_required_tags()
            host.resident = resident
            if required <= resident:
                if host.state != "ready":
                    host.ready_at = time.time()
                host.state = "ready"
                continue
            host.mark_preparing()
            # A host the pool created is the pool's to configure (spec §1.2): pull the set
            # and pin it, once, off the pass — a download takes minutes and must not stall
            # the loop. The next probes find the set resident and mark the host ready.
            if host_id not in self._preparing:
                self._preparing[host_id] = asyncio.create_task(
                    self._prepare_rented(host_id, client), name=f"prepare:{host_id}"
                )

    async def _prepare_rented(self, host_id: str, client: httpx.AsyncClient) -> None:
        assert self.fleet is not None
        host = self.fleet.hosts.get(host_id)
        if host is None:
            return
        try:
            loaded = await self.fleet.load_model_set(host, self.engine, client)
            if not loaded:
                await self.fleet.destroy(host, "could not hold the pool's model set")
        except Exception:  # noqa: BLE001 - recorded; the host stays preparing and is retried
            log.exception("preparing %s failed", host_id)
        finally:
            self._preparing.pop(host_id, None)

    def _rented_required_tags(self) -> frozenset[str]:
        rented = self.config.rented
        capabilities = frozenset(rented.capabilities) if rented else frozenset()
        variants = variants_for_host(
            self.config.pool.model_set, self.config.catalog, capabilities, self.engine.name
        )
        return frozenset(v[0].tag for v in variants.values() if v)

    def _publish_rented(self) -> None:
        if self.fleet is None or self.config.rented is None:
            return
        rented = self.config.rented
        capabilities = frozenset(rented.capabilities)
        variants = variants_for_host(
            self.config.pool.model_set, self.config.catalog, capabilities, self.engine.name
        )
        live = set()
        for host_id, host in self.fleet.hosts.items():
            if host.released or not host.dial_url:
                continue
            live.add(host_id)
            self.table.publish(
                HostRow(
                    host_id=host_id,
                    kind="rented-interruptible",
                    transport_type="http",
                    priority=20,
                    dial_url=host.dial_url,
                    state=host.state if host.state in ("ready", "preparing", "draining") else "preparing",
                    workers=host.workers,
                    capabilities=tuple(rented.capabilities),
                    variants={
                        name: tuple((v.tag, v.runtime_class, v.enforces_schema) for v in group)
                        for name, group in variants.items()
                    },
                    resident=getattr(host, "resident", frozenset()),
                    lease_id=host.lease_id,
                    provider_ref=self.fleet.published_ref(host),
                    hourly_rate=host.bid_hourly,
                )
            )
        for row in self.table.all():
            if row.kind == "rented-interruptible" and row.host_id not in live:
                self.table.remove(row.host_id)

    async def _probe(self, host: SupervisedHost) -> None:
        """The pool verifies a host's engine; it never configures it, and never triggers a
        pull or a load."""
        if host.state is HostState.DISABLED:
            return

        previous = host.state
        health = await self.engine.health(host.client)
        if not health.ok:
            host.state = HostState.UNREACHABLE
            host.last_error = health.detail
            host.resident = frozenset()
            host.available = frozenset()
        else:
            try:
                resident = await self.engine.models_resident(host.client)
                available = await self.engine.models_available(host.client)
            except httpx.HTTPError as exc:
                host.state = HostState.UNREACHABLE
                host.last_error = str(exc)
                host.resident = frozenset()
                host.available = frozenset()
            else:
                host.resident = resident
                host.available = available
                # A pinned host serves only what is loaded; an on-demand host serves what is
                # on disk and lets the engine load it on first use. Neither ever downloads.
                if host.config.residency == "on_demand":
                    missing = host.required_tags - available
                    what = "not on disk"
                else:
                    missing = host.required_tags - resident
                    what = "not resident"
                if missing:
                    host.state = HostState.PREPARING
                    host.last_error = f"model set {what}: missing {sorted(missing)}"
                else:
                    host.state = HostState.READY
                    host.last_error = None

        host.last_probe_at = time.monotonic()
        if host.state is not previous:
            log.info(
                "host %s: %s -> %s (%s)",
                host.host_id,
                previous.value,
                host.state.value,
                host.last_error or "ok",
            )

    def _publish_all(self) -> None:
        for host in self.hosts.values():
            self.table.publish(
                HostRow(
                    host_id=host.host_id,
                    kind=host.config.kind,
                    transport_type=host.config.transport.type,
                    priority=host.config.routing_priority,
                    dial_url=host.dial_url,
                    state=host.state.value,
                    workers=host.config.workers,
                    capabilities=tuple(sorted(host.capabilities)),
                    variants={
                        name: tuple(
                            (v.tag, v.runtime_class, v.enforces_schema) for v in variants
                        )
                        for name, variants in host.variants.items()
                    },
                    resident=host.resident,
                    available=host.available,
                    residency=host.config.residency,
                    last_error=host.last_error,
                )
            )

    def tunnel_status(self, host_id: str) -> Optional[dict[str, object]]:
        tunnel = self.hosts[host_id].tunnel
        if tunnel is None:
            return None
        return {
            "up": tunnel.up,
            "local_port": tunnel.local_port,
            "restarts": tunnel.restarts,
            "last_error": tunnel.last_error,
        }


async def run(config: PoolConfig, database: Database, config_path: Optional[str] = None) -> None:
    """Run a supervisor until cancelled, with the control API beside it when a key exists."""
    import uvicorn

    from .control import create_control_app

    supervisor = Supervisor(config, database, config_path=config_path)
    server: Optional[uvicorn.Server] = None
    serving: Optional[asyncio.Task[None]] = None
    try:
        await supervisor.start()

        if config.auth.admin_hashes():
            server = uvicorn.Server(
                uvicorn.Config(
                    create_control_app(supervisor, config),
                    host=config.control.host,
                    port=config.control.port,
                    ssl_certfile=config.control.tls_certfile,
                    ssl_keyfile=config.control.tls_keyfile,
                    log_level="warning",
                )
            )
            serving = asyncio.create_task(server.serve())
            log.info("control API on %s:%d", config.control.host, config.control.port)
        else:
            log.warning(
                "no admin key configured, so the control API is not served. Create one with "
                "`gpm key create --role admin` — leases cannot be opened without it."
            )

        await supervisor.run_forever()
    finally:
        if server is not None:
            server.should_exit = True
        if serving is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(serving, timeout=10)
        await supervisor.aclose()
