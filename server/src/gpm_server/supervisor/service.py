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
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

from .. import strategies
from ..catalog import ResolvedVariant, literal_variants_for_host, variants_for_host
from ..config import (
    AgentConfig,
    ConfigError,
    HostConfig,
    PoolConfig,
    load_config,
    rented_profile_models,
)
from ..configplan import ConfigStore
from ..db import Database, HostCounters, HostRow, HostTable, SupervisorLock
from ..directory import Directory
from ..engines import Engine, get_engine
from ..ledger import EventLog, LeaseStore, SpendLedger
from ..models import RENTED_KINDS, HostState
from ..providers.base import ProviderAuthError, ProviderError, get_provider
from ..transports import SshTunnel, build_client, tunnel_maker
from . import agents
from .renting import Fleet

log = logging.getLogger("gpm.supervisor")
#: Programs' requests answered per look of the watcher, and per pass (D117).
REQUESTS_PER_LOOK = 4


class ProviderCredentialMissing(Exception):
    """Rented capacity is configured and the provider will not accept this process's credential.
    Raised at start, before anything is adopted, swept or published."""


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
        # Every engine this pool runs (D93); `self.engine` stays the pool's own, for the
        # questions that are not about one host.
        self.engines: dict[str, Engine] = {
            name: get_engine(name) for name in config.engines_in_use()
        }
        self.engine: Engine = self.engines[config.engine]
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
        # The model directory (D101), refreshed from here and read by both processes.
        self.directory = Directory(database, lambda: self.config)
        self.passes = 0
        #: True while the provider could not be asked which rented hosts still exist (D61).
        self._adoption_pending = False
        self._saturated_passes = 0
        #: Saturated passes in a row, per demand (D115); None is the shared workload.
        self._saturated_by: dict[Optional[str], int] = {}
        self._stopping = False
        #: How a forward is opened: by the forwarder, where the pool has one (D110), else here.
        self.make_tunnel = tunnel_maker(config, database)
        #: When the forwarder was last started from here, so a missing one is not started every
        #: pass while it is still coming up.
        self._forwarder_started_at = 0.0
        #: Tests replace how the forwarder is started; nothing else does.
        self.spawn_forwarder = self._spawn_forwarder
        for host_config in config.hosts:
            self.hosts[host_config.id] = self._build(host_config)

        #: Absent `rented` means the pool has no way to spend at all.
        self.fleet: Optional[Fleet] = None
        if config.rented is not None:
            self.fleet = Fleet(
                config,
                config.rented,
                provider or get_provider(config.rented.connection.type, config.rented.connection.settings),
                self.leases,
                self.events,
                self.spend,
            )
            self.fleet.make_tunnel = self.make_tunnel
        #: Workloads inside the pool (D115): created from the control API, moved along each pass.
        from .provisioning import Provisioning
        from .workloads import Workloads

        self.workloads = Workloads(self)
        #: Programs asking for their own workloads (D117): their requests are answered between
        #: passes, never during one, under the same lock.
        self.provisioning = Provisioning(self)
        self.pass_lock = asyncio.Lock()
        self._request_watcher: Optional[asyncio.Task] = None
        self._requests_revision: Optional[float] = None
        #: Clients for hosts the pool rented, keyed by host id.
        self._rented_clients: dict[str, httpx.AsyncClient] = {}
        #: Model-set preparations in flight, keyed by host id.
        self._preparing: dict[str, asyncio.Task[None]] = {}
        #: Tests substitute a transport that answers as an agent would.
        self._agent_transport: Optional[httpx.AsyncBaseTransport] = None

    def _build(self, host_config: HostConfig) -> SupervisedHost:
        capabilities = frozenset(host_config.capabilities)
        tunnel = (
            self.make_tunnel(host_config.id, host_config.transport)
            if host_config.transport.type == "tunnel"
            else None
        )
        dial_url = tunnel.local_url if tunnel else (host_config.transport.base_url or "")
        return SupervisedHost(
            config=host_config,
            client=build_client(host_config.transport, self.config.pool, dial_url),
            variants=variants_for_host(
                self.config.models_held_by(host_config),
                self.config.catalog,
                capabilities,
                self.config.engine_of(host_config),
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
        host.agent_tunnel = self.make_tunnel(f"{host.host_id}:agent", forward)
        return host.agent_tunnel

    # --- lifecycle ---

    # --- the forwarder (D110) ---

    #: A forwarder whose heartbeat is older than this is taken for gone.
    FORWARDER_STALE_S = 15.0
    #: And one started from here is given this long to come up before it is started again.
    FORWARDER_GRACE_S = 30.0

    def forwarder_running(self) -> bool:
        from ..forwarder import lock_name

        rows = self.db.query(
            "SELECT heartbeat FROM supervisor_lock WHERE pool = ?", (lock_name(self.config.pool.name),)
        )
        return bool(rows) and time.time() - rows[0]["heartbeat"] < self.FORWARDER_STALE_S

    def ensure_forwarder(self) -> None:
        """Start the forwarder when the pool has one and none is running. It is started detached
        — a session of its own — so it outlives this supervisor, which is its whole purpose."""
        forwarder = self.config.forwarder
        if not forwarder.enabled or not forwarder.start_automatically or self.forwarder_running():
            return
        if time.monotonic() - self._forwarder_started_at < self.FORWARDER_GRACE_S and self._forwarder_started_at:
            return
        self._forwarder_started_at = time.monotonic()
        self.spawn_forwarder()

    def _spawn_forwarder(self) -> None:
        if self.config_path is None:
            log.warning("the forwarder is on, but this pool was not started from a file to start it with")
            return
        import subprocess
        import sys

        state_dir = Path(self.config.request_log).expanduser().resolve().parent
        out = open(state_dir / "gpm-forwarder.log", "ab")
        child = subprocess.Popen(
            [sys.executable, "-m", "gpm_server.cli", "forwarder", "-c", str(self.config_path)],
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )
        out.close()
        self.events.record("forwarder_started", f"the forwarder was not running; started it (pid {child.pid})",
                           numbers={"pid": child.pid})

    async def start(self) -> None:
        self.lock.acquire()
        if self.fleet is not None:
            self.fleet.claim_records()
        self.ensure_forwarder()
        for host in self.hosts.values():
            if host.tunnel is not None:
                await host.tunnel.start()
            forward = self._sync_agent_tunnel(host)
            if forward is not None:
                await forward.start()
        await self._refuse_without_a_credential()
        if self.config.provisioning.enabled:
            _ = self.provisioning.ca  # a bad client CA stops the start, not a program's first create
        await self.adopt_rented()
        await self.pass_once()
        if self.config.provisioning.enabled and self._request_watcher is None:
            self._request_watcher = asyncio.create_task(self._watch_requests(), name="gpm:provisioning")

    async def _watch_requests(self) -> None:
        """Answer programs' requests within a second or so, rather than a pass (D117): a cheap
        look at the table's revision, and work only when it moved. Never during a pass."""
        while not self._stopping:
            await asyncio.sleep(self.config.provisioning.request_poll_s)
            try:
                revision = await asyncio.to_thread(self.provisioning.store.requests_revision)
                if revision == self._requests_revision:
                    continue
                # Remembered as read *before* answering: a request arriving while these are
                # answered moves the revision again, and is answered on the next look. Reading it
                # after would count that request as seen, and it would wait for ever.
                self._requests_revision = revision
                # One request per hold of the lock, a few per look: a pass — and the lock's
                # heartbeat, refreshed between passes — never waits behind a queue of them.
                for _ in range(REQUESTS_PER_LOOK):
                    async with self.pass_lock:
                        if not await self.provisioning.answer_pending(limit=1):
                            break
                else:
                    self._requests_revision = None  # more may be waiting: look again next time
            except Exception:  # noqa: BLE001 - the watcher must outlive a bad request
                log.exception("answering provisioning requests failed")

    async def _refuse_without_a_credential(self) -> None:
        """A pool configured to rent must be able to ask its provider what exists. One that
        cannot — no credential, or a refused one — can neither adopt its hosts nor verify a
        destroy, so it does not start (D61). A provider that is merely unreachable is a
        different case: the supervisor starts, and adoption waits."""
        if self.fleet is None:
            return
        try:
            await self.fleet.provider.account()
        except ProviderAuthError as exc:
            raise ProviderCredentialMissing(
                f"rented capacity is configured but the provider's credential is not usable: {exc}. "
                "Set it in this process's environment, or remove the `rented` section. Nothing "
                "was changed: every rented host's record is as it was."
            ) from exc
        except ProviderError:
            return

    async def adopt_rented(self) -> None:
        """Before the first pass — and so before the first sweep — take back the rented hosts
        the last supervisor published, and drop the rows of hosts the provider no longer has."""
        rows = [row for row in self.table.all() if row.kind in RENTED_KINDS]
        if not rows:
            return
        if self.fleet is None:
            for row in rows:
                self.table.remove(row.host_id)
            return
        answer = await self.fleet.adopt(rows)
        if answer is None:
            # The provider could not be asked, so nothing is known about these hosts: their
            # rows are kept exactly as found, and nothing is swept, rented or pruned until a
            # later pass can ask (D61).
            if not self._adoption_pending:
                self.events.record(
                    "adoption_deferred",
                    f"the provider could not be asked about {len(rows)} rented host(s); their "
                    "records are kept and nothing is swept or rented until it can be",
                    numbers={"hosts": sorted(row.host_id for row in rows)},
                )
            self._adoption_pending = True
            return
        self._adoption_pending = False
        adopted = set(answer)
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
            # The heartbeat says "this process is alive and looping", so it is refreshed here
            # and not at the end of a pass. Found live: a pass that raised every time never
            # reached its heartbeat, the lock went stale under a running supervisor, a second
            # one took it, and the two ran side by side. And a supervisor that finds the lock
            # is no longer its own stops: two of them acting on one pool is the thing the lock
            # exists to prevent.
            if not self.lock.beat():
                log.error("another supervisor has taken this pool's lock; stopping this one")
                self._stopping = True
                return
            try:
                await self.pass_once()
            except Exception:  # a bad pass must never take the supervisor down
                log.exception("control loop pass failed")
            # In the background: a refresh reads hundreds of pages and must never hold a pass.
            self.directory.maybe_start()

    async def aclose(self) -> None:
        self._stopping = True
        if self._request_watcher is not None:
            self._request_watcher.cancel()
        await self.directory.aclose()
        for host in self.hosts.values():
            # Detached, not stopped: with the forwarder (D110) the forwards outlive this
            # process, and the next supervisor finds them on the same ports.
            if host.tunnel is not None:
                await host.tunnel.detach()
            if host.agent_tunnel is not None:
                await host.agent_tunnel.detach()
            await host.client.aclose()
        for task in self._preparing.values():
            task.cancel()
        for client in self._rented_clients.values():
            await client.aclose()
        if self.fleet is not None:
            await self.fleet.detach_tunnels()
        self.lock.release()

    # --- the control loop ---

    async def pass_once(self) -> None:
        """Observe → update host states → compare demand → act → record (spec §3).

        Acting is strictly ordered inside the fleet: release what should not exist, recover
        what is broken, then acquire what is missing. Programs' requests are answered first,
        under the same lock the request watcher takes (D117).
        """
        async with self.pass_lock:
            if self.config.provisioning.enabled:
                await self.provisioning.answer_pending(limit=REQUESTS_PER_LOOK)
            await self._pass()

    async def _pass(self) -> None:
        self._follow_config_file()
        self.ensure_forwarder()
        if self._adoption_pending:
            await self.adopt_rented()
        await asyncio.gather(*(self._ask_agent(host) for host in self.hosts.values()))
        await asyncio.gather(*(self._probe(host) for host in self.hosts.values()))
        await self._probe_rented()
        self._publish_all()

        if self.fleet is not None and not self._adoption_pending:
            active = await self.workloads.advance()
            await self.fleet.pass_once(
                ready_workers_higher_tiers=self._ready_workers(),
                idle_seconds=self._idle_seconds(),
                busy={host_id: counter.busy for host_id, counter in self.counters.all().items()},
                pressure=self._pressure(),
                load=self._load() if self.config.rented.allocation == "dynamic" else None,
                waiting_by_model=self._waiting_by_model(),
                workloads=active,
                # Per group of each workload's hosts (D118); a one-group workload's is its own.
                loads={(w.name, g.key): self._load(w.name, g.key if len(w.groups) > 1 else None)
                       for w in active for g in w.groups},
                pressures={(w.name, g.key): self._pressure(w.name, g.key if len(w.groups) > 1 else None)
                           for w in active for g in w.groups},
            )
            if self.config.rented and self.config.rented.workers_auto.enabled:
                await self._adjust_workers()
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
                    new.models_held_by(host_config), new.catalog, existing.capabilities,
                    new.engine_of(host_config),
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

    def _worker_readings(self) -> list:
        """What each rented host has just done, for deciding its worker count (D67).

        Read from the request log the router already writes — tokens generated, service time,
        per host and per model — so nothing is asked of a host to produce it.
        """
        from ..strategies import WorkerReading

        if self.fleet is None or not self.config.rented or not self.config.rented.workers_auto.enabled:
            return []
        window = self.config.rented.workers_auto.window_s
        now = time.time()
        counters = self.counters.all()
        rows = self.db.query(
            "SELECT host_id, model_served, tokens_out, generate_ms, latency_ms, ts "
            "FROM request_log WHERE outcome = 'ok' AND ts > ?",
            (now - window * 2,),
        )
        waiting = len(
            self.db.query(
                "SELECT 1 FROM request_log WHERE ts > ? AND (queue_wait_ms >= 1000 OR reason = 'queue_timeout') LIMIT 50",
                (now - window,),
            )
        )
        recent = [r for r in rows if r["ts"] > now - window]
        before = [r for r in rows if r["ts"] <= now - window]

        def throughput(rows_in: list, host_id: str) -> Optional[float]:
            tokens = sum(r["tokens_out"] or 0 for r in rows_in if r["host_id"] == host_id)
            return (tokens / window) if tokens else None

        # The pool's own median for a model is the yardstick: a pool where everything is slow
        # has no slow host, only slower hardware.
        by_model: dict[str, list[float]] = {}
        for row in recent:
            if row["latency_ms"] is not None:
                by_model.setdefault(row["model_served"], []).append(row["latency_ms"] / 1000)

        readings = []
        for host_id, host in self.fleet.hosts.items():
            if host.released or host.state != "ready":
                continue
            if host.workload is not None and len(host.models) > 1:
                # Runs a fixed split of its models (D118): slower than the pool's median for a
                # model by design, since it shares the card, and never to be stepped down —
                # the split's shares are what keep each model within its target.
                continue
            mine = [r for r in recent if r["host_id"] == host_id and r["latency_ms"] is not None]
            busiest = max(
                ({r["model_served"] for r in mine} or {""}),
                key=lambda model: sum(1 for r in mine if r["model_served"] == model),
            )
            service = [r["latency_ms"] / 1000 for r in mine if r["model_served"] == busiest]
            pool_service = by_model.get(busiest) or []
            counter = counters.get(host_id)
            readings.append(
                WorkerReading(
                    host_id=host_id,
                    workers=host.workers,
                    launch_workers=host.launch_workers or host.workers,
                    busy_workers=counter.busy if counter else 0,
                    waiting=waiting,
                    throughput=throughput(recent, host_id),
                    throughput_before=throughput(before, host_id),
                    service_s=statistics.median(service) if service else None,
                    pool_service_s=statistics.median(pool_service) if pool_service else None,
                    evicted_a_model=host.lost_a_model,
                    last_change=host.last_worker_change,
                )
            )
        return readings

    async def _adjust_workers(self) -> None:
        """Move each host toward the count its own measurements ask for (D67, D68)."""
        from ..strategies import decide_workers

        assert self.fleet is not None
        cfg = self.config.rented.workers_auto
        for reading in self._worker_readings():
            decision = decide_workers(reading, cfg)
            if not decision.changed:
                continue
            host = self.fleet.hosts.get(reading.host_id)
            if host is None:
                continue
            done, why = await self.fleet.resize(host, decision.workers)
            if done:
                host.last_worker_change = decision.workers - reading.workers
                host.lost_a_model = False
                self.events.record(
                    "workers_auto",
                    f"{reading.host_id}: {decision.reasons[0]}",
                    numbers={"workers": decision.workers, "was": reading.workers},
                    host_id=reading.host_id,
                )
            else:
                log.info("not resizing %s: %s", reading.host_id, why)

    def _own_busy(self, counter: Any) -> int:
        """A host's busy workers that are its own demand's: lent ones are the borrower's (D115)."""
        return (counter.busy - counter.borrowed) if counter else 0

    def _in_group(self, models: Optional[tuple]) -> tuple[str, tuple]:
        """The request-log condition for one group's models (D118): its waiting only."""
        if not models:
            return "", ()
        return f" AND model_requested IN ({','.join('?' * len(models))})", tuple(models)

    def _load(self, workload: Optional[str] = None, group: Optional[tuple] = None) -> "strategies.Load":
        """What one demand's traffic is asking of it, from what the router already writes (D66,
        D115). The shared workload's configured hosts, and its rented ones; a workload's own —
        one group of them where it has several (D118)."""
        counters = self.counters.all()
        busy = ready = 0
        if workload is None:
            for host in self.hosts.values():
                if host.state is HostState.READY:
                    ready += host.config.workers
                    busy += self._own_busy(counters.get(host.host_id))
        if self.fleet is not None:
            for rented in self.fleet.hosts_of(workload, group):
                if rented.state == "ready":
                    ready += rented.workers
                    busy += self._own_busy(counters.get(rented.host_id))
        only, models = self._in_group(group)
        window = (
            self.config.rented.dynamic.window_s
            if self.config.rented
            else 120.0
        )
        waiting = len(
            self.db.query(
                "SELECT 1 FROM request_log WHERE ts > ? AND (queue_wait_ms >= 1000 OR reason = 'queue_timeout') "
                f"AND workload IS ?{only} LIMIT 200",
                (time.time() - min(window, 60.0), workload, *models),
            )
        )
        return strategies.Load(busy_workers=busy, ready_workers=ready, waiting=waiting)

    def _waiting_by_model(self) -> dict[str, int]:
        """Which models the waiting is for (D95), over the same window and the same test as the
        pool's own decision to rent at all — so "we need more" and "more of what" cannot
        disagree about what waiting means."""
        window = self.config.rented.dynamic.window_s if self.config.rented else 120.0
        rows = self.db.query(
            "SELECT model_requested AS model, COUNT(*) AS n FROM request_log "
            "WHERE ts > ? AND (queue_wait_ms >= 1000 OR reason = 'queue_timeout') "
            "AND model_requested IS NOT NULL AND workload IS NULL GROUP BY model_requested",
            (time.time() - min(window, 60.0),),
        )
        return {row["model"]: int(row["n"]) for row in rows}

    def _pressure(self, workload: Optional[str] = None, group: Optional[tuple] = None) -> bool:
        """Is load asking for more than the ready hosts give? (D64)

        Two measurements, both written by the router off the request path: every ready worker
        busy on two passes running — a client sized to capacity builds no queue, but saturation
        shows — or a request that waited, or was refused for waiting, in the last half minute.
        """
        counters = self.counters.all()
        ready: list[tuple[str, int]] = [
            (host.host_id, host.config.workers) for host in self.hosts.values() if host.state is HostState.READY
        ] if workload is None else []
        if self.fleet is not None:
            ready += [(h.host_id, h.workers) for h in self.fleet.hosts_of(workload, group) if h.state == "ready"]
        saturated = bool(ready) and all(
            self._own_busy(counters.get(host_id)) >= workers for host_id, workers in ready
        )
        passes = self._saturated_by.get((workload, group) if group else workload, 0)
        passes = passes + 1 if saturated else 0
        self._saturated_by[(workload, group) if group else workload] = passes
        if workload is None:
            self._saturated_passes = passes
        if passes >= 2:
            return True
        only, models = self._in_group(group)
        waited = self.db.query(
            "SELECT 1 FROM request_log WHERE ts > ? AND (queue_wait_ms >= 1000 OR reason = 'queue_timeout') "
            f"AND workload IS ?{only} LIMIT 1",
            (time.time() - 30, workload, *models),
        )
        return bool(waited)

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
                self.config.models_held_by(host.config), self.config.catalog, host.capabilities,
                self.config.engine_of(host.config),
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
            if host.state == "parked":
                # Its engine is stopped because the pool stopped it (D64). Probing a parked
                # host finds nothing answering and marks it `preparing`, which the eviction
                # handler then reads as "stopped, and we did not ask" — an eviction — and the
                # host is destroyed seconds after being parked. Found by the simulation:
                # parking had never once saved a download.
                continue
            if host.state == "draining":
                # The drain owns it until its work is done or its time is up. Probed, a host
                # still holding its models was set back to `ready` — so the router sent it new
                # work, and tear-down, seeing a ready host under no lease, destroyed it with
                # answers in flight. Found live: 18 seconds into a drain at a lease's end.
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

            await self.fleet.ask_agent(host)

            health = await self.engine_for(host).health(client)
            if not health.ok:
                if self.engine_for(host).loads_by_restart:
                    await self._bring_up_through_its_agent(host_id, host, client)
                else:
                    host.mark_preparing()
                continue

            if host.agent is None:
                # Only now: an engine that answers is proof the image is up and its filesystem
                # is usable. SSH answers long before that — seen live, a host rented at 12:01:13
                # had spent all three of its attempts by 12:01:47, concluding "no python3" from
                # an image that had not finished starting, and so ran without an agent for its
                # whole life. The install is still never allowed to fail the preparation, and
                # the model download carries on regardless (D63).
                await self.fleet.install_agent(host)
                await self.fleet.ask_agent(host)
            if host.engine_seen_at is None:
                host.engine_seen_at = time.time()  # it has started; "stuck starting" is over
            try:
                resident = await self.engine_for(host).models_resident(client)
            except httpx.HTTPError:
                host.mark_preparing()
                continue

            # A rented host whose engine is serving from the processor is worthless at an
            # accelerator's price, and looks healthy by every other measure (D81). The driver
            # floor refuses that machine before it is rented; this is the same fault arriving
            # any other way, on a host already being paid for.
            on_cpu = await self.engine_for(host).serving_from_cpu(client)
            if on_cpu:
                self.fleet.avoid(host.offer.machine_id, "its engine ran on the processor")
                self.fleet.events.record(
                    "engine_without_accelerator",
                    f"{host.host_id} is serving {', '.join(sorted(on_cpu))} from the processor, "
                    "not the accelerator it is paid for; giving it up",
                    numbers={"machine": host.offer.machine_id, "models": sorted(on_cpu)},
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
                await self.fleet.destroy(host, "its engine could not use the accelerator")
                continue
            # What *this* host was bought for — not everything the pool rents for. With a model
            # to a host (D94), no single host holds the whole rented set, and judging each by it
            # would leave every one of them preparing for ever.
            required = self.fleet.tags_for(host)
            if host.state == "ready" and (required & host.resident) - resident:
                # It held the set and no longer does: the engine ran out of memory for what it
                # was asked to keep, which is the clearest evidence a host has too many
                # workers (D67). Cleared when the count is changed on the strength of it.
                host.lost_a_model = True
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

    #: How long between agent install attempts on a host whose engine cannot answer yet (D97).
    AGENT_SPACING_S = 30.0

    async def _bring_up_through_its_agent(self, host_id: str, host: Any, client: httpx.AsyncClient) -> None:
        """Prepare a host whose engine cannot answer until its models are on disk (D97).

        Every other host is proven up by its engine answering: that is when the pool installs
        its agent and starts preparing. An engine that serves only what it was started with
        answers nothing until the agent has fetched its weights — so waiting for it would wait
        for ever, and the host would be given up as never having started. Here the **agent**
        answering is the proof: it only runs once the image is up and its filesystem usable.
        """
        host.mark_preparing()
        if host.agent is None:
            now = time.time()
            if host.agent_tried_at is not None and now - host.agent_tried_at < self.AGENT_SPACING_S:
                return  # a machine still settling is given time, not every attempt at once
            host.agent_tried_at = now
            await self.fleet.install_agent(host)
            if host.agent is None:
                return
            await self.fleet.ask_agent(host)
        if host.engine_seen_at is None:
            # It has started: its agent answers from it. "Never started" is over.
            host.engine_seen_at = time.time()
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
            if await self.fleet.models_from_sibling_pending(host):
                return  # a copy from a sibling is still coming (D116); the next pass looks again
            # A host with an agent is fetched by its agent: one pull at a time, each model
            # loaded as its own download finishes (D57). Without one, the pool does it the way
            # it always has, over the engine's own API.
            through_agent = await self.fleet.load_model_set_through_agent(host)
            if through_agent is False:
                errors = [
                    f"{model.get('tag')}: {model.get('error')}"
                    for model in (host.agent_models or {}).get("models", [])
                    if (model or {}).get("error")
                ]
                if errors:
                    # The reason the machine gave, not a summary of it: "could not hold the
                    # model set" told the operator nothing about a process that had died.
                    await self.fleet.destroy(host, f"could not serve the model set — {'; '.join(errors)[:400]}")
                return  # still coming; the next pass asks again
            if through_agent:
                return
            engine = self.engine_for(host)
            if engine.loads_by_restart:
                # Nothing but the agent can fetch for this engine (D90): its own API has no pull.
                # An agent not answering yet is waited for; one the pool has stopped trying to
                # install means this host can never serve, and is given up saying why.
                if host.agent is None and host.agent_attempts >= self.config.rented.agent_attempts:
                    await self.fleet.destroy(
                        host, "no agent could be installed, and nothing else can fetch this engine's models"
                    )
                return
            # The host's own engine, not the pool's default (D93): the two can differ.
            loaded = await self.fleet.load_model_set(host, engine, client)
            if not loaded:
                await self.fleet.destroy(host, "could not hold the pool's model set")
        except Exception:  # noqa: BLE001 - recorded; the host stays preparing and is retried
            log.exception("preparing %s failed", host_id)
        finally:
            self._preparing.pop(host_id, None)

    def engine_for(self, host: Any) -> Engine:
        """The engine on that machine (D93).

        Two kinds of host reach this. A configured one carries its `HostConfig` and names its
        own engine or inherits the pool's; a rented one has no such config and runs whatever
        the rented configuration says.
        """
        config = getattr(host, "config", None)
        if config is None and self.fleet is not None:
            # What the machine was rented to run, not what the pool rents now (D98): a switch
            # of engine must not make a running host look like the other one.
            return self.fleet.engine_of(host)
        name = self.config.engine_of(config) if config is not None else self.config.rented_engine()
        if name not in self.engines:
            # An engine the file started naming after this process began — loaded on first use
            # rather than silently answered by the pool's default adapter.
            self.engines[name] = get_engine(name)
        return self.engines[name]

    def _rented_models(self) -> list[str]:
        """What a rented host is asked to hold (D89). With `all`, the pool's whole set; with
        `declared`, what the rented configuration names, or anything the pool still needs."""
        rented = self.config.rented
        if rented is not None and rented.rent_profiles:
            return rented_profile_models(rented)
        if self.config.pool.models_per_host == "all" or rented is None or rented.models is None:
            return list(self.config.pool.model_set)
        return list(rented.models)

    def _rented_required_tags(self) -> frozenset[str]:
        rented = self.config.rented
        if rented is not None and rented.rent_profiles:
            return frozenset(tag for models in rented.profiles_rented().values() for tag in models.values())
        capabilities = frozenset(rented.capabilities) if rented else frozenset()
        variants = variants_for_host(
            self._rented_models(), self.config.catalog, capabilities, self.config.rented_engine()
        )
        return frozenset(v[0].tag for v in variants.values() if v)

    def _publish_rented(self) -> None:
        if self.fleet is None or self.config.rented is None:
            return
        rented = self.config.rented
        capabilities = frozenset(rented.capabilities)
        variants = variants_for_host(
            self._rented_models(), self.config.catalog, capabilities, self.config.rented_engine()
        )
        live = set()
        for host_id, host in self.fleet.hosts.items():
            if host.released or not host.dial_url:
                continue
            live.add(host_id)
            self.table.publish(
                HostRow(
                    host_id=host_id,
                    kind="rented-interruptible" if host.interruptible else "rented-on-demand",
                    transport_type="http",
                    priority=20,
                    dial_url=host.dial_url,
                    state=host.state if host.state in ("ready", "preparing", "draining") else "preparing",
                    workers=host.workers,
                    capabilities=tuple(rented.capabilities),
                    # A workload's host advertises its own model and nothing else; a shared host never
                    # advertises a workload's (D115, the review's finding 7).
                    variants=self._rented_variants_of(host, variants if not host.workload else variants_for_host(
                        list(host.builds or {}), self.config.catalog, capabilities, self.config.rented_engine())),
                    resident=getattr(host, "resident", frozenset()),
                    engine=self.config.rented_engine(),
                    lease_id=host.lease_id,
                    provider_ref=self.fleet.published_ref(host),
                    hourly_rate=host.bid_hourly,
                    workload=host.workload,
                )
            )
        for row in self.table.all():
            if row.kind in RENTED_KINDS and row.host_id not in live:
                self.table.remove(row.host_id)

    def _rented_variants_of(self, host: Any, variants: dict) -> dict:
        """What the router may resolve each model to on this rented host: the build its profile
        named, where it was bought as one (D111) — the same model may be a different build on
        another host — and otherwise every build the catalog offers rented hosts."""
        builds = getattr(host, "builds", None) or {}
        published = {}
        for name, group in variants.items():
            if builds:
                if name not in builds:
                    continue
                group = tuple(v for v in group if v.tag == builds[name]) or tuple(
                    # A build the profile has since dropped: still what this machine holds.
                    v for v in literal_variants_for_host(
                        [name], self.config.catalog, frozenset(self.config.rented.capabilities),
                        self.config.rented_engine(),
                    ).values() if v.tag == builds[name]
                )
            published[name] = tuple((v.tag, v.runtime_class, v.enforces_schema) for v in group)
        return published

    async def _probe(self, host: SupervisedHost) -> None:
        """The pool verifies a host's engine; it never configures it, and never triggers a
        pull or a load."""
        if host.state is HostState.DISABLED:
            return

        previous = host.state
        health = await self.engine_for(host).health(host.client)
        if not health.ok:
            host.state = HostState.UNREACHABLE
            host.last_error = health.detail
            host.resident = frozenset()
            host.available = frozenset()
        else:
            try:
                resident = await self.engine_for(host).models_resident(host.client)
                available = await self.engine_for(host).models_available(host.client)
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
                    engine=self.config.engine_of(host.config),
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
