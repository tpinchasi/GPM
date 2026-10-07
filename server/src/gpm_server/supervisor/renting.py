"""Renting: the only part of the pool that can spend money.

docs/spec/supervisor.md §3–§6 and §9. Order within a pass is strict: **release what should not
exist → recover what is broken → acquire what is missing**, and acquisition is refused when any
cap would be crossed.

Every decision a strategy returns is re-checked here. The strategies are advisory; the caps are
not.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gpm_agent import volume as model_volume

from .. import agentpkg, history
from .. import workloads as workload_math
from ..config import BiddingConfig, OfferPolicy, PoolConfig, RentedConfig, TransportConfig
from ..deadman import heartbeat_command, onstart_script
from ..engines import get_engine
from ..engines.base import PullResult
from ..history import adjustment as history_adjustment
from ..ledger import EventLog, Lease, LeaseRefused, LeaseStore, SpendLedger
from ..providers.base import (
    BidLost,
    ConnectionInfo,
    Instance,
    InstanceSpec,
    InstanceState,
    Offer,
    OfferGone,
    OfferQuery,
    Provider,
    ProviderError,
    ProviderRateLimited,
    VolumeSpec,
    redacted,
)
from ..sizing import Needs, needs_for
from ..strategies import (
    Demand,
    HostView,
    LeaseView,
    Load,
    all_in_ceiling,
    decide_eviction,
    decide_ramp,
    decide_rent,
    decide_teardown,
    filter_name,
    image_for,
    price_bid,
    rank_offers,
)
from ..transports import SshTunnel, build_ssh_exec_command, run_command
from ..workload_store import Volume, WorkloadStore, group_key
from . import agents, hostagent

log = logging.getLogger("gpm.renting")


class PullTooSlow(Exception):
    """Raised from inside a download's progress report to stop it: the machine's link is far
    below what its offer advertised, and every minute of waiting is billed."""


@dataclasses.dataclass
class RentedHost:
    host_id: str
    instance: Instance
    offer: Offer
    bid_hourly: float
    lease_id: str
    created_at: float = dataclasses.field(default_factory=time.time)
    state: str = "scheduling"
    dial_url: Optional[str] = None
    estimated_spend: float = 0.0
    reported_spend: float = 0.0
    connection: Optional[ConnectionInfo] = None
    #: The models this host was bought to serve (D94). With the set spread across hosts and an
    #: engine that holds one model per process, that is a single model, chosen when the machine
    #: was bought from whatever the pool was shortest of. Empty means the whole rented set, which
    #: is what every host meant before the pool could buy one per model.
    models: tuple[str, ...] = ()
    #: The model profile it was bought as, and the build of each model that profile named
    #: (D111). Empty for a host bought without profiles: its builds are the catalog's first.
    profile: Optional[str] = None
    builds: dict[str, str] = dataclasses.field(default_factory=dict)
    #: How many cards each copy of its models spans (D114), fixed when it was bought: the
    #: machine was searched for that many, and every relaunch must ask for the same.
    cards_per_copy: int = 1
    #: The workload it was bought for (D115); None is the shared workload.
    workload: Optional[str] = None
    #: The volume it was created with, and where its models came from: "volume" (copied from its
    #: workload's model volume, D139), "sibling" (copied from a ready host of its workload), or
    #: "hub" (D116).
    volume_id: Optional[str] = None
    models_source: str = "hub"
    #: Whether what its agent said about its volume has been taken in (D139): once per host.
    volume_reported: bool = False
    copy_state: Optional[str] = None
    copy_task: Optional[asyncio.Task] = dataclasses.field(default=None, repr=False, compare=False)
    copy_started_at: Optional[float] = None
    #: The disk it was rented with — the search's minimum, raised to what its models need.
    disk_gb: Optional[float] = None
    ready_at: Optional[float] = None
    parked_at: Optional[float] = None
    #: Spend settled for its current lease, and from when the rate has run since (see estimate).
    accrued: float = 0.0
    accrued_at: Optional[float] = None
    #: What the provider had reported for this instance when it joined its current lease: that
    #: much was another lease's.
    charges_base: float = 0.0
    reported_spend_total: float = 0.0
    #: The agent the pool installed here, if any, and what it reported. A host without one is
    #: prepared the way it always was (D63).
    agent: Optional[hostagent.RentedAgent] = None
    agent_facts: Optional[dict] = None
    agent_models: Optional[dict] = None
    #: For an engine that loads by being restarted (D97): the models on disk the last time the
    #: pool asked for that restart, and when. One ask per set of downloaded models — a restart
    #: takes minutes, and asking again every pass would only ever interrupt it.
    restarted_for: Optional[frozenset] = None
    restart_asked_at: Optional[float] = None
    #: When the pool last re-bid for this host and asked the provider to start it again. While
    #: that start is in flight the instance still reads "stopped", and read as a fresh
    #: eviction the pool bids against itself (D106).
    rebid_at: Optional[float] = None
    restart_task: Optional[asyncio.Task] = dataclasses.field(default=None, repr=False, compare=False)
    #: True once this host's agent has answered a heartbeat, so an operator can see the verb
    #: working before the ssh beat beside it is retired.
    agent_beats: bool = False
    #: What this host's engine was launched to run at once. A change up to it is the pool
    #: using slots the engine already has; past it, the engine has to be relaunched (D68).
    launch_workers: int = 0
    #: Which way this host's count last moved, so a step is judged against its own evidence.
    last_worker_change: int = 0
    #: True when a model the pool requires was seen to leave this host's memory.
    lost_a_model: bool = False
    #: The engine this machine was rented to run, and the port it listens on — fixed when it
    #: was bought. The pool's rented engine can be switched while hosts are running (D98), and
    #: a machine started for one engine does not become the other: it keeps being dialled,
    #: prepared and restarted as what it is. Empty for a row from before this was recorded.
    engine: str = ""
    engine_port: int = 0
    agent_detail: Optional[str] = None

    @property
    def connection_name(self) -> str:
        """The provider connection it was rented through (D129) — not `connection`, which is how
        the pool reaches the machine."""
        return self.offer.connection
    #: How many times the pool has tried to put an agent here, so it stops trying.
    agent_attempts: int = 0
    #: When the agent was last tried on a host whose engine cannot answer yet (D97): such an
    #: install cannot wait for the engine, so it waits for the clock instead, and a machine still
    #: settling does not spend every attempt in the first half-minute.
    agent_tried_at: Optional[float] = None
    #: When this host was parked for having no traffic, the moment its idleness began — so the
    #: destroy limit is counted from its last request, not from the park (D64).
    idle_since: Optional[float] = None
    prepared: bool = False
    #: Set by a preparation, so idle release does not reap a host that has no traffic *yet*.
    hold_until: Optional[float] = None
    when_ready: str = "join"
    download_cost: float = 0.0
    last_request_at: Optional[float] = None
    idle_since: Optional[float] = None
    released: bool = False
    resident: frozenset[str] = frozenset()
    #: When this host's engine first answered. None means it has never started — which is
    #: what "stuck scheduling" looks like from here.
    engine_seen_at: Optional[float] = None
    #: While draining: when to stop waiting for its work to finish, and what to do then.
    drain_until: Optional[float] = None
    drain_then: str = "destroy"
    drain_reason: str = ""
    #: Whether this host was bid for. A non-interruptible one cannot be outbid, so a stop is
    #: a fault to recover from, never an eviction to re-bid on (D52).
    interruptible: bool = True
    #: When this host started *preparing* — not when it was created. A host taken back after
    #: a supervisor restart is marked preparing again so readiness is re-verified, and the
    #: "not ready in time" rule must measure from then, or a host that has been serving for an
    #: hour is destroyed the moment it is adopted (D50: it was, live).
    preparing_since: Optional[float] = None
    #: What preparing this host is doing right now, and how far each download has got —
    #: for the operator watching, not for any decision.
    stage: str = ""
    progress: dict = dataclasses.field(default_factory=dict)
    #: How many workers this host runs — fixed at creation, because the engine was launched
    #: with that parallelism. From the matching capacity profile, or the rented default.
    workers: int = 1

    def mark_preparing(self) -> None:
        """Enter `preparing`, starting the clock that "not ready in time" is measured against.

        Only on the way *in*: a host already preparing keeps the deadline it has. Anything that
        measures this from `created_at` instead destroys hosts that have been serving for
        hours the moment they are adopted or blink (D50 — seen live).
        """
        if self.state != "preparing":
            self.preparing_since = time.time()
        self.state = "preparing"

    def mark_scheduling(self) -> None:
        """Enter `scheduling` — waiting for the provider to start it again after a re-bid —
        starting the same clock. A host that served for hours and is then outbid has not been
        "not ready for 30 minutes"; it has been not ready since now (found live, 2026-09-26)."""
        if self.state != "scheduling":
            self.preparing_since = time.time()
        self.state = "scheduling"

    @property
    def hours_held(self) -> float:
        return (time.time() - self.created_at) / 3600

    def rate(self) -> float:
        """What it bills an hour now: its bid while it runs, only its disk while parked."""
        return self.offer.storage_hourly if self.state == "parked" else self.bid_hourly

    def estimate(self, now: Optional[float] = None) -> float:
        """What it has cost its current lease: what was settled, then the time since at the rate
        it bills now. Settled when the rate changes kind (parked, restarted) and when it moves
        to another lease, which pays only from then."""
        now = now if now is not None else time.time()
        since = self.accrued_at if self.accrued_at is not None else self.created_at
        return self.accrued + max(0.0, now - since) / 3600 * self.rate()

    def settle(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        self.accrued, self.accrued_at = self.estimate(now), now

    def move_to_lease(self, lease_id: str, now: Optional[float] = None) -> None:
        """From now on it bills `lease_id`, which owes nothing it cost before: neither the pool's
        estimate nor what the provider had already reported for it."""
        now = now if now is not None else time.time()
        self.lease_id = lease_id
        self.accrued, self.accrued_at = 0.0, now
        self.charges_base = self.reported_spend_total


@dataclasses.dataclass
class Unit:
    """One demand inside the pool (D115): the shared workload, or one group of one workload's
    hosts (D118). Its own hosts, and the state its scaling keeps between passes. Passed explicitly
    wherever a decision is about *one* demand, so no decision can act on another's hosts by
    default. A workload's lease is its groups' together."""

    workload: Optional[str] = None
    #: The models every host of the group holds, sorted; None for the shared workload.
    group: Optional[tuple[str, ...]] = None
    #: When the gap between what is wanted and what is ready opened, and when it closed.
    #: Scale-down waits on the second, deliberately longer than scale-up (spec §9).
    overflow_since: Optional[float] = None
    overflow_gone_since: Optional[float] = None
    #: Set when a host is given up for idleness: from then on the lease's standing demand does
    #: not bring capacity back by itself — measured load does (D64).
    idle_gate: bool = False
    #: The ramp: how many hosts the last round asked for, and when it landed (D66).
    ramp_round: int = 0
    ramp_landed_at: float = 0.0
    #: A workload whose budget could not carry any offer: the hosts and lease it was found with.
    #: No search is made for it again until one of them changes (D126).
    budget_bound: Optional[tuple] = None


@dataclasses.dataclass
class SearchState:
    """One connection's searching (D119, D129): asked again only after `retry_at`."""

    backoff_s: float = 0.0
    retry_at: float = 0.0
    #: The provider's own words for why it refused, kept apart from the wait they earned.
    refusal: Optional[str] = None
    #: Why its last search came back empty, when it was not the market's doing.
    error: Optional[str] = None


class ConnectionEvents:
    """The decision log, as the fleet writes it: an event that names a machine also names the
    connection it is on (D129), since a machine identifier is unique only within one. The
    machine history is built from these."""

    def __init__(self, events: EventLog, connection: str, hosts: Optional[dict] = None):
        self._events = events
        self._connection = connection
        self._hosts = hosts if hosts is not None else {}

    def record(self, kind: str, summary: str, *, numbers: Optional[dict] = None, **where: Any) -> Any:
        if numbers and numbers.get("machine") and not numbers.get("connection"):
            host = self._hosts.get(where.get("host_id") or "")
            numbers = {**numbers, "connection": host.connection_name if host is not None else self._connection}
        return self._events.record(kind, summary, numbers=numbers, **where)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._events, name)


class ConnectionNameReused(ValueError):
    """A connection's name, once one provider's, configured as another's (D133)."""


def legacy_connection(db: Any, current: str) -> str:
    """The connection a record that names none belongs to (D129): fixed the first time a pool
    with connections ran, as the one it was configured with then."""
    rows = db.query("SELECT value FROM pool_meta WHERE key = 'legacy_connection'")
    if rows:
        return str(rows[0]["value"])
    db.execute("INSERT OR IGNORE INTO pool_meta (key, value) VALUES ('legacy_connection', ?)", (current,))
    return str(db.query("SELECT value FROM pool_meta WHERE key = 'legacy_connection'")[0]["value"])


#: How long the market is not asked again after the provider says it is asked too often (D119).
OFFER_RATE_LIMIT_WAIT_S = 60.0


def describe_query(query: OfferQuery) -> list[str]:
    """What a search asks the provider to filter, in words (D123)."""
    said = []
    if query.min_gpu_memory_gb:
        said.append(f"card memory ≥ {query.min_gpu_memory_gb:g} GB")
    if query.min_gpus > 1:
        said.append(f"≥ {query.min_gpus} cards")
    if query.min_disk_gb:
        said.append(f"disk ≥ {query.min_disk_gb:g} GB")
    if query.max_all_in_hourly is not None:
        said.append(f"base price ≤ ${query.max_all_in_hourly:.3f}/h")
    if query.min_download_mbps:
        said.append(f"download ≥ {query.min_download_mbps:g} Mb/s")
    if query.max_download_per_gb is not None:
        said.append(f"download price ≤ ${query.max_download_per_gb:g}/GB")
    if query.min_reliability:
        said.append(f"reliability ≥ {query.min_reliability:g}")
    if query.verified_only:
        said.append("verified only")
    if query.exclude_hardware:
        said.append("not " + ", ".join(query.exclude_hardware))
    if query.avoid_machines:
        said.append(f"not machines {', '.join(query.avoid_machines)}")
    return said


#: How an assumed field is said (providers.md §6).
_ASSUMED_WORDS = {"download_mbps": "download speed", "reliability": "reliability", "verified": "verified",
                  "download_per_gb": "download price"}

#: How fast a host copies its models from a model volume to its own disk, for estimating what a
#: volume saves until the pool has its own times (D139): a network volume reads at about 200–400
#: MB/s by its provider's own figure, and a parallel copy reached ~1.2 GiB/s in a third-party
#: benchmark (decisions.md, Verified facts). The lower figure, so the saving is never overstated.
VOLUME_COPY_MB_PER_S = 300.0


class Fleet:
    """The rented half of the pool: what exists, what it costs, and what to do next."""

    def __init__(
        self,
        config: PoolConfig,
        rented: RentedConfig,
        provider: Provider,
        leases: LeaseStore,
        events: EventLog,
        spend: SpendLedger,
    ):
        self.config = config
        self.rented = rented
        #: Every configured connection's provider, enabled or not (D129): a disabled connection's
        #: hosts are still watched, charged and swept. A single provider is the pool's one connection.
        self.providers: dict[str, Provider] = (
            dict(provider) if isinstance(provider, dict) else {rented.connection_name: provider})
        #: The first enabled connection: whose provider `provider` is, for everything that asks of
        #: "the" provider — a single-provider pool's, and the default where none is named.
        self.connection = rented.connection_name
        #: What a record from before connections existed belongs to: the connection the pool's
        #: configuration became the first time this version ran, remembered so a later rename or
        #: a second connection does not reassign those records.
        #: Settled by `claim_records`, under the pool lock; until then, this connection's.
        self.legacy_connection = self.connection
        #: Every name a connection has been configured under, with its provider (D133). At most
        #: one connection per provider, so a record naming a connection that has since been
        #: renamed is found by its provider; one whose provider is gone is never guessed at.
        self.connection_types: dict[str, str] = {name: conn.type for name, conn in rented.providers.items()}
        #: Hosts whose connection could not be listed at a restart, or is no longer configured,
        #: by connection: kept exactly as found, and that connection is not swept, until it can be.
        self.pending_adoption: dict[str, list] = {}
        self._said_pending: dict[str, tuple] = {}
        self.leases = leases
        self.spend = spend
        self.hosts: dict[str, RentedHost] = {}
        self.events = ConnectionEvents(events, self.connection, self.hosts)
        self.label_prefix = rented.label_prefix or f"gpm/{config.pool.name}/"
        #: One demand per workload, and the shared one under None (D115).
        self.units: dict[tuple, Unit] = {(None, None): Unit()}
        #: The workloads that are not over, as the supervisor read them this pass.
        self.workloads: dict[str, Any] = {}
        #: Workloads' volumes, as the database keeps them (D116).
        self.workload_store = WorkloadStore(leases.db)
        #: What the request log says a class of machine serves at a latency, read at most every
        #: thirty seconds; and the last rental-kind choice said, per workload (D82's rule).
        self._sizing_cache: dict[tuple, tuple[float, Any]] = {}
        self._hardware_cache: Optional[tuple[float, dict[str, str]]] = None
        self._said_kind: dict[tuple, tuple] = {}
        #: The last "no offer holds this group's split" said, per group (D118).
        self._said_unfit: dict[tuple, int] = {}
        #: Whether the provider has ever returned a charge figure. The narrow cap margin is
        #: only earned once it has.
        #: The connections whose provider has actually reported a charge (spec §4): one that only
        #: *says* it can keeps the wide margin, whatever another connection does.
        self.charges_seen: set[str] = set()
        #: How a command is run on a rented host. Substituted in tests; ssh otherwise.
        self.run_on_host = self._ssh_run
        self.push_to_host = self._ssh_push
        #: Where the packed agent is kept between installs, beside the pool's own state.
        self.state_dir = Path(config.request_log).expanduser().resolve().parent
        #: Tests dial the agent through their own transport; nothing else sets this.
        self._agent_transport = None
        #: Engine adapters by name, loaded once each.
        self._engines: dict[str, Any] = {}
        #: How a forward is opened — `SshTunnel` here, or through the forwarder (D110), which the
        #: supervisor sets from the pool's configuration.
        self.make_tunnel: Any = SshTunnel
        #: A second forward per host, to the agent on its loopback (D63).
        self.agent_tunnels: dict[str, SshTunnel] = {}
        #: Forwards to rented hosts the provider cannot expose directly, keyed by host id.
        self.tunnels: dict[str, SshTunnel] = {}
        #: Machines that just failed to start or to download, and until when they are skipped.
        #: In memory only: a restart forgets, which costs at most one more try.
        self.avoided: dict[str, tuple[float, str]] = {}
        #: Models whose build the directory has not measured, from the last size worked out: the
        #: disk check and the download cost leave them out, and the preview says so (D108).
        self.model_sizes_unknown: list[str] = []
        #: Why the last offer search came back empty, when it was not the market's doing: every
        #: connection asked failed. One that failed beside others that answered is in `searching`.
        self.last_offer_error: Optional[str] = None
        #: The last "nothing passed the policy" written down, so a market that has not changed
        #: is not written down again on every pass (D82).
        self._said_nothing_passed: Optional[tuple] = None
        #: What the provider was last asked to filter (D123), in words, for what a search says.
        self.last_query_filters: list[str] = []
        #: Why the last attempt to rent rented nothing — for whoever asked, in their words.
        self.last_refusal: Optional[str] = None
        #: The last attempt rented nothing because the workload's budget carried none of its
        #: offers, and nothing else was tried (D126).
        self.last_budget_bound = False
        #: The machine history, rebuilt from the logs every few seconds (D69).
        self._history: Optional[dict] = None
        self._history_at = 0.0
        #: Each connection's searching: its back-off after "too many requests", the provider's own
        #: words for it, and why its last search came back empty (D119, D129).
        self.searching: dict[str, SearchState] = {name: SearchState() for name in self.providers}
        #: Connections not rented through while a change to them waits for a restart (D135).
        self.paused: set[str] = set()

    # --- connections (D129) ---

    @property
    def provider(self) -> Provider:
        return self.providers[self.connection]

    @provider.setter
    def provider(self, value: Provider) -> None:
        self.providers[self.connection] = value

    def searchable(self) -> list[str]:
        """The connections a search asks: the enabled ones the pool has a provider for, but for
        one whose change waits for a restart — renting there now would put hosts on an account
        the restart may replace (D136)."""
        return [name for name in self.rented.enabled_connections if name in self.providers and name not in self.paused]

    def provider_for(self, what: Any) -> Provider:
        """The provider a host, an offer or a connection's name belongs to. A record whose
        connection is no longer configured belongs to the only one, where there is only one — a
        renamed connection's hosts; otherwise it cannot be acted on, and saying so is the answer."""
        name = what if isinstance(what, str) else (
            what.connection_name if isinstance(what, RentedHost) else getattr(what, "connection", ""))
        resolved = self.resolve(name or self.legacy_connection)
        if resolved is None:
            raise ProviderError(f"provider connection {name!r} is no longer configured: restore it to manage "
                                "what the pool holds there")
        return self.providers[resolved]

    def resolve(self, name: str) -> Optional[str]:
        """The configured connection a record's connection name is now: itself, or — renamed —
        the one with its provider (D133). None where its provider is no longer configured."""
        if name in self.providers:
            return name
        kind = self.connection_types.get(name)
        return next((n for n, conn in self.rented.providers.items() if kind and conn.type == kind
                     and n in self.providers), None)

    def _at(self, offer: Offer) -> tuple[str, str]:
        """Where a machine is: a machine identifier is unique only within its connection."""
        named = offer.connection or self.legacy_connection
        return (self.resolve(named) or named, offer.machine_id)

    # Before connections, one provider's back-off; kept for whatever reads it as one.
    @property
    def _offer_retry_at(self) -> float:
        return self.searching[self.connection].retry_at

    @_offer_retry_at.setter
    def _offer_retry_at(self, value: float) -> None:
        self.searching[self.connection].retry_at = value

    @property
    def _offer_backoff_s(self) -> float:
        return self.searching[self.connection].backoff_s

    @_offer_backoff_s.setter
    def _offer_backoff_s(self, value: float) -> None:
        self.searching[self.connection].backoff_s = value

    # --- units (D115) ---

    def unit(self, workload: Optional[str] = None, group: Optional[tuple[str, ...]] = None) -> Unit:
        if workload is not None and group is None:
            group = self.first_group(workload)
        key = (workload, group)
        if key not in self.units:
            self.units[key] = Unit(workload=workload, group=group)
        return self.units[key]

    def first_group(self, workload: Optional[str]) -> Optional[tuple[str, ...]]:
        spec = self.workloads.get(workload) if workload is not None else None
        return spec.groups[0].key if spec is not None and spec.groups else None

    def held_back_rows(self, workload: Any = ...) -> list:
        """Records kept at a restart for hosts on a provider not yet asked (D61, D129): hosts the
        pool may still be paying for, so they count against its limits as held hosts do — or a
        pool with one account unreachable would rent past them on the others. Given a workload,
        only its own (None is the shared workload's)."""
        return [r for held in self.pending_adoption.values() for r in held
                if workload is ... or r.workload == workload]

    def held_back_burn(self) -> float:
        return sum(float((r.provider_ref or {}).get("bid_hourly") or r.hourly_rate or 0.0)
                   for r in self.held_back_rows())

    @property
    def adoption_retry_due(self) -> bool:
        """A configured connection still holds hosts back: worth asking again each pass. One no
        longer configured is not — asking every provider for it would never end (D129)."""
        return any(name in self.providers for name in self.pending_adoption)

    def hosts_of(self, workload: Optional[str], group: Optional[tuple[str, ...]] = None) -> list[RentedHost]:
        """The live hosts one demand holds: a workload's, or one group of them (D118)."""
        return [
            h for h in self.hosts.values()
            if not h.released and h.workload == workload and (group is None or group_key(h.models) == group)
        ]

    @staticmethod
    def group_of_host(host: "RentedHost") -> Optional[tuple[str, ...]]:
        """A workload host's group is the models it was bought for (D118)."""
        return group_key(host.models) if host.workload is not None else None

    def group_spec(self, workload: Optional[str], group: Optional[tuple[str, ...]]) -> Any:
        spec = self.workloads.get(workload) if workload is not None else None
        if spec is None:
            return None
        return next((g for g in spec.groups if group is None or g.key == group), None)

    def group_ceiling(self, lease: Optional[Lease], workload: Optional[str], group: Optional[tuple[str, ...]]) -> int:
        """The most workers a demand may reach: the lease's for the shared workload; for a group,
        the answers at once its models were asked for (D118) — the lease holds every group's."""
        spec = self.workloads.get(workload) if workload is not None else None
        chosen = self.group_spec(workload, group)
        if spec is None or chosen is None:
            return lease.workers if lease else 0
        wanted = sum(t.parallel for t in spec.targets if t.model in chosen.models)
        return min(wanted, lease.workers) if lease else wanted

    @staticmethod
    def lease_of(open_leases: Sequence[Lease], workload: Optional[str]) -> Optional[Lease]:
        """The lease that may rent for this demand: the shared workload's is the first open lease
        that allows renting and belongs to no workload, as before workloads; a workload's is its own."""
        return next((lease for lease in open_leases if lease.allow_rent and lease.workload == workload), None)

    def unit_order(self, open_leases: Sequence[Lease]) -> list[tuple[Optional[str], Optional[tuple[str, ...]]]]:
        """The shared workload, then each workload in the order its lease was opened, and each of
        its groups in order: the room the pool's caps leave is taken in that order (workloads.md
        §8, D118)."""
        opened = {lease.workload: lease.opened_at for lease in open_leases if lease.workload}
        named = sorted(self.workloads, key=lambda name: (opened.get(name, float("inf")), name))
        return [(None, None), *((name, g.key) for name in named for g in self.workloads[name].groups)]

    # --- money ---

    def margin(self) -> float:
        """A provider that cannot report charges gets a wider margin (spec §4) — and so does
        one that *says* it can but has never actually done so. Found live: the declaration
        was true of the API in general and false of every instance payload, which would have
        left the cap on estimate-only with the narrow margin, silently."""
        counted = set(self.searchable()) | {h.connection_name for h in self.hosts.values() if not h.released}
        if counted and all(name in self.providers and self.providers[name].capabilities.reports_charges
                           and name in self.charges_seen for name in counted):
            return self.rented.spend.cap_safety_margin
        return self.rented.spend.margin_without_reported_charges

    @property
    def charges_ever_reported(self) -> bool:
        return bool(self.charges_seen)

    def lease_spend(self, lease: Lease) -> tuple[float, float]:
        """(the figure caps are enforced on, the raw estimate). Caps use whichever of the
        pool's estimate and the provider's report is higher."""
        totals = self.spend.latest_for_lease(lease.lease_id)
        estimate = totals.get("estimate", 0.0)
        reported = totals.get("reported", 0.0)
        return max(estimate, reported), estimate

    def budget_left(self, lease: Lease) -> float:
        spent, _ = self.lease_spend(lease)
        return lease.max_spend * (1 - self.margin()) - spent

    async def record_spend_all(self) -> None:
        """Every live host's spend, under its own lease — open or not. A host draining after its
        lease closed still bills, and the ledger must say so (D115, the review's finding 2)."""
        for lease_id in sorted({h.lease_id for h in self.hosts.values() if not h.released and h.lease_id}):
            lease = self.leases.get(lease_id)
            if lease is not None:
                await self.record_spend(lease)
        # A workload's volume bills its storage whether or not a host has it attached (D116): its
        # cost goes on the workload's lease, under the volume's own name.
        now = time.time()
        for volume in self.workload_store.volumes():
            if volume.lease_id and volume.hourly:
                self.spend.record(
                    lease_id=volume.lease_id, host_id=f"volume:{volume.volume_id}", source="estimate",
                    amount=round(volume.hourly * max(0.0, now - volume.created_at) / 3600, 6),
                    connection=volume.connection or self.legacy_connection,
                )

    async def record_spend(self, lease: Lease) -> None:
        for host in self.hosts.values():
            if host.lease_id != lease.lease_id or host.released:
                continue
            host.estimated_spend = host.estimate()
            self.spend.record(
                lease_id=lease.lease_id,
                host_id=host.host_id,
                source="estimate",
                amount=host.estimated_spend,
                connection=host.connection_name,
            )
            try:
                charges = await self.provider_for(host).reported_charges(host.instance)
            except ProviderError:
                charges = None
            if charges is not None:
                self.charges_seen.add(host.connection_name)
                host.reported_spend_total = charges.total
                host.reported_spend = max(0.0, charges.total - host.charges_base)
                self.spend.record(
                    lease_id=lease.lease_id,
                    host_id=host.host_id,
                    source="reported",
                    amount=host.reported_spend,
                    connection=host.connection_name,
                )
                drift = host.reported_spend - host.estimated_spend
                if host.estimated_spend > 0 and drift / host.estimated_spend > self.rented.spend.drift_alert:
                    self.events.record(
                        "spend_drift",
                        f"provider reports ${host.reported_spend:.4f} against an estimate of "
                        f"${host.estimated_spend:.4f}",
                        numbers={"reported": host.reported_spend, "estimate": host.estimated_spend},
                        host_id=host.host_id,
                        lease_id=lease.lease_id,
                    )

    # --- the pass ---

    async def pass_once(
        self,
        ready_workers_higher_tiers: int,
        idle_seconds: dict[str, float],
        busy: Optional[Mapping[str, int]] = None,
        pressure: bool = False,
        load: Optional[Load] = None,
        waiting_by_model: Optional[Mapping[str, int]] = None,
        workloads: Optional[Sequence[Any]] = None,
        loads: Optional[Mapping[str, Load]] = None,
        pressures: Optional[Mapping[str, bool]] = None,
    ) -> None:
        # What each model's traffic is waiting on, for deciding which model to buy for (D95).
        # Held for this pass only: it is a measurement, not state.
        self._waiting_by_model = dict(waiting_by_model or {})
        if workloads is not None:
            self.workloads = {w.name: w for w in workloads if w.active}
        await self.finish_draining(busy or {})
        await self.beat_deadman_timers()
        await self.sweep_orphans()
        await self.expire_parked()

        await self.record_spend_all()
        for lease in self.leases.open_leases():
            await self.enforce_lease_limits(lease)
        # Read again: a lease the limits just closed must not rent anything below.
        open_leases = self.leases.open_leases()
        await self.release_unleased(open_leases, busy or {})

        await self.handle_evictions()
        # Release before acquire, across every demand (spec §3): each unit's tear-down, then each
        # unit's acquisition, in the order their leases were opened (workloads.md §8).
        units = self.unit_order(open_leases)

        def per_unit(values: Optional[Mapping], workload: str, group: Any) -> Any:
            # Keyed by (workload, group); a workload's own name is its first group's (D118).
            values = values or {}
            return values.get((workload, group), values.get(workload) if group == self.first_group(workload) else None)

        for workload, group in units:
            await self.tear_down(
                open_leases, idle_seconds, ready_workers_higher_tiers if workload is None else 0,
                workload=workload, group=group,
            )
        for workload, group in units:
            await self.acquire(
                open_leases, ready_workers_higher_tiers if workload is None else 0,
                pressure=pressure if workload is None else bool(per_unit(pressures, workload, group)),
                load=load if workload is None else per_unit(loads, workload, group),
                workload=workload, group=group,
            )

    async def release_unleased(self, open_leases: Sequence[Lease], busy: Mapping[str, int]) -> None:
        """A host whose lease is no longer open is drained (D53), whatever demand it served —
        however the lease came to close: its caps, its time, an operator, a workload ended, or
        closed while no supervisor was running. Without this a host nobody's demand looks at
        again bills until its dead-man timer fires (the review's finding 1).

        A parked host of the shared workload is left to its own limits, as before: parking is
        for keeping a machine between runs. A workload's parked host has no run to come back to."""
        open_ids = {lease.lease_id for lease in open_leases}
        for host in list(self.hosts.values()):
            if host.released or host.state == "draining" or host.lease_id in open_ids:
                continue
            if host.state == "parked":
                if host.workload is not None:
                    await self.destroy(host, "its workload's lease is no longer open")
                continue
            if host.state != "ready":
                # Serving nothing — still coming up, or being re-verified after a restart — so
                # there is nothing to finish: draining would only bill for longer.
                await self.destroy(host, "its lease is no longer open")
                continue
            # A ready host is drained even when the counters read idle: they are from the start of
            # the pass, and the router still routes to it until the next publish (D53).
            await self.drain(host, "its lease is no longer open")

    async def enforce_lease_limits(self, lease: Lease) -> bool:
        """Time or dollars reached → release everything the lease holds. Returns True when
        the lease was closed."""
        spent, estimate = self.lease_spend(lease)
        limit = lease.max_spend * (1 - self.margin())

        if spent >= limit:
            self.events.record(
                "lease_capped",
                f"spend ${spent:.4f} reached ${limit:.4f} — the cap of ${lease.max_spend:.2f} "
                f"less a {self.margin() * 100:.0f}% margin — so the lease stops before its limit",
                numbers={
                    "spent": spent,
                    "estimate": estimate,
                    "cap": lease.max_spend,
                    "enforced_at": limit,
                    "margin": self.margin(),
                },
                lease_id=lease.lease_id,
            )
            self.leases.close(lease.lease_id, "dollar cap reached")
            await self.release_lease(lease, "dollar cap reached")
            return True

        if lease.expired():
            self.events.record(
                "lease_expired",
                f"lease reached its {lease.max_hours}h limit",
                numbers={"max_hours": lease.max_hours},
                lease_id=lease.lease_id,
            )
            self.leases.close(lease.lease_id, "time limit reached")
            await self.release_lease(lease, "time limit reached")
            return True
        return False

    async def release_lease(self, lease: Lease, reason: str) -> None:
        for host in list(self.hosts.values()):
            if host.lease_id == lease.lease_id:
                # Not destroyed outright: a lease ending is not a reason to drop the requests
                # already running on its host (D53).
                await self.drain(host, reason)

    async def drain(self, host: RentedHost, reason: str, then: str = "destroy") -> None:
        """Stop sending this host new work, let what it has finish, then end it.

        The router stops choosing a host that is not `ready`, so publishing `draining` is what
        makes it stop taking new requests; the ones already on it keep their workers until they
        answer. `teardown.drain_timeout_s` bounds the wait — a host that never finishes is
        still billing, so it is ended anyway and that is said plainly.
        """
        if host.state == "draining":
            return
        if host.state not in ("ready", "preparing"):
            await self.destroy(host, reason)  # nothing is being served on it
            return
        host.state = "draining"
        host.drain_until = time.time() + self.rented.teardown.drain_timeout_s
        host.drain_then = then
        host.drain_reason = reason
        self.events.record(
            "draining",
            f"{host.host_id} is draining ({reason}): no new requests, and up to "
            f"{self.rented.teardown.drain_timeout_s:g}s for the ones it has",
            numbers={"drain_timeout_s": self.rented.teardown.drain_timeout_s},
            host_id=host.host_id,
            lease_id=host.lease_id,
        )

    async def finish_draining(self, busy: Mapping[str, int]) -> None:
        """End each draining host once its last request has answered, or time is up."""
        for host in list(self.hosts.values()):
            if host.state != "draining" or host.released:
                continue
            still = busy.get(host.host_id, 0)
            if still <= 0:
                await self._end_drain(host, f"{host.drain_reason}; its work had finished")
            elif host.drain_until is not None and time.time() >= host.drain_until:
                await self._end_drain(
                    host,
                    f"{host.drain_reason}; {still} request(s) had still not finished after "
                    f"{self.rented.teardown.drain_timeout_s:g}s, and it was still billing",
                )

    async def _end_drain(self, host: RentedHost, reason: str) -> None:
        if host.drain_then == "park":
            await self.park(host, reason)
        else:
            await self.destroy(host, reason)

    # --- leases ---

    def open_lease(self, **kwargs) -> Lease:
        """Opening a lease goes through the fleet, because only it knows what the provider can
        do — and a lease longer than the dead-man timer can cover is refused outright."""
        limit = self.max_lease_hours()
        requested = kwargs.get("max_hours")
        if limit is not None and requested is not None and requested > limit:
            raise LeaseRefused(
                f"{self._without_deadman()} (it has no instance-scoped credential), so leases are "
                f"limited to {limit}h, not {requested}h"
            )
        return self.leases.open(pool_max_all_in_hourly=self.rented.max_all_in_hourly, **kwargs)

    # --- the dead-man timer ---

    def deadman_onstart(self, models: Optional[Sequence[str]] = None, connection: Optional[str] = None) -> Optional[str]:
        """The start-up script that arms the timer, or None where the provider has no
        instance-scoped credential.

        Without one the pool would have to choose between putting the account credential on a
        machine it does not trust and having no timer at all. It does neither: it refuses long
        leases on that provider instead (spec §7).
        """
        provider = self.provider_for(connection or self.connection)
        if not provider.capabilities.self_terminate:
            return None
        return onstart_script(
            provider.self_terminate_request(self.rented.teardown.deadman_action),
            window_s=int(self.rented.teardown.deadman_minutes * 60),
            engine_port=self.engine_port,
            public_key=self.pool_public_key(),
            ssh_user=self.rented.ssh_user,
            extra=self.engine_start_command(models),
        )

    def engine_start_command(self, models: Optional[Sequence[str]] = None) -> Optional[str]:
        """How the engine is started on a host this pool creates: the operator's `engine_start`
        if there is one, and otherwise the engine's own (D97). vLLM has one — the agent's
        launcher — because starting it means reading what the agent fetched, which is not a
        line an operator should have to get exactly right."""
        if self.rented.engine_start:
            return self.rented.engine_start
        return self.engine.default_start_command(
            port=self.engine_port,
            models_dir=hostagent.MODELS_DIR,
            agent_archive=hostagent.ARCHIVE,
            proxy=self.proxy_for(models),
            options=tuple(self.rented.engine_options),
        )

    def proxy_for(self, models: Optional[Sequence[str]]) -> bool:
        """Whether a machine holding these models runs the router in front of one engine
        process per model (D96): where the pool says so, or where it holds several models under
        an engine that serves one per process (D111) — a profile of two is a router of two."""
        if self.rented.engine_proxy:
            return True
        many = len(models) > 1 if models is not None else False
        return many and bool(getattr(self.engine, "serves_one_model", False))

    def pool_public_key(self) -> Optional[str]:
        """The public half of `rented.ssh_key`, read from the `.pub` beside it."""
        if not self.rented.ssh_key:
            return None
        from pathlib import Path

        pub = Path(self.rented.ssh_key).expanduser().with_suffix(
            Path(self.rented.ssh_key).suffix + ".pub"
        )
        if not pub.exists():
            log.warning("no public key beside %s; rented hosts will only accept the account's keys", self.rented.ssh_key)
            return None
        return pub.read_text().strip()

    def machine_history(self, refresh_after_s: float = 30.0) -> dict:
        """What each machine has done for this pool (D69), from the logs it already writes.

        Cached for a few seconds: it is read on every market preview and every renting pass,
        and the logs it folds do not move between them.
        """
        now = time.monotonic()
        if self._history is not None and now - self._history_at < refresh_after_s:
            return self._history
        try:
            events = list(reversed(self.events.recent(2000)))
            requests = [
                dict(row)
                for row in self.events.db.query(
                    "SELECT host_id, outcome, latency_ms, tokens_out, generate_ms FROM request_log "
                    "WHERE host_id IS NOT NULL ORDER BY id DESC LIMIT 20000"
                )
            ]
        except Exception as exc:  # noqa: BLE001 - a view is never worth failing a bid for
            log.warning("could not build the machine history: %s", exc)
            return {}
        self._history = history.build(events, requests)
        self._history_at = now
        return self._history

    #: However many cards a machine has, one engine is not launched past this. The per-card
    #: arithmetic is sound and an eight-card machine would otherwise ask for a number no engine
    #: has been measured at.
    MOST_WORKERS_A_HOST_MAY_RUN = 64

    @staticmethod
    def _card_of(hardware: str) -> str:
        """"2x RTX PRO 6000 WS" -> "rtx pro 6000 ws". The count is a fact about the machine;
        what one card runs at once is a fact about the card (D88)."""
        return re.sub(r"^\s*\d+\s*x\s*", "", hardware or "", count=1).strip().lower()

    def workers_for(self, offer: Offer, workload: Optional[str] = None,
                    models: Optional[Sequence[str]] = None) -> tuple[int, str]:
        """How many workers a host rented from this offer would run, and why (spec §2.1) — for a
        workload, no more than the class of machine was measured to serve within its latency
        target (D115); what was not measured is said. A host of several of a workload's models
        runs its group's split: the sum of each model's share (D118)."""
        workers, why = self._workers_for_card(offer)
        spec = self.workloads.get(workload) if workload is not None else None
        if spec is None:
            return workers, why
        chosen = self.group_spec(workload, group_key(models) if models else None)
        if chosen is not None and len(chosen.models) > 1:
            split = ", ".join(f"{m} {chosen.caps.get(m, 0)}" for m in chosen.models)
            return chosen.workers_per_host, f"{why}; its group's split: {split} at once"
        model = chosen.models[0] if chosen is not None else spec.model
        target = spec.target(model) or spec.targets[0]
        at = self.model_at_latency(offer.hardware, (chosen.builds if chosen else spec.builds).get(model) or model,
                                   target.latency_s, ceiling=workers)
        if at.workers is not None and at.workers < workers:
            return at.workers, f"{why}; held to {at.workers} for the {target.latency_s:g}s target: {at.reasons[0]}"
        return workers, f"{why}; {at.reasons[0]}"

    def per_model_on(self, offer: Offer, spec: Any, models: Sequence[str], builds: dict[str, str],
                     unrefuted_is_open: bool = False) -> dict[str, int]:
        """How many answers of each model, alone, one host from this offer serves within the
        model's target: measured where the log has it, else the card's own number (D115, D118).

        `unrefuted_is_open`: a measured number that no higher measured concurrency refutes is a
        floor, not a ceiling — the card may serve more. A split host only ever runs a model up to
        its share, so its own answers could otherwise never show the card holds the split, and a
        lost host of a group could never be replaced on the card it was planned on."""
        card_workers, _ = self._workers_for_card(offer)
        found = {}
        for model in models:
            target = spec.target(model)
            latency = target.latency_s if target is not None else spec.latency_s
            at = self.model_at_latency(offer.hardware, builds.get(model) or model, latency, ceiling=card_workers)
            if at.workers is None:
                found[model] = card_workers
            elif unrefuted_is_open and not any(c > at.workers and p95 > latency for c, p95 in at.curve.items()):
                # Nothing measured above it missed the target: it is a floor, not a ceiling.
                found[model] = card_workers
            else:
                found[model] = min(card_workers, at.workers)
        return found

    def split_fits_on(self, offer: Offer, spec: Any, group: Any) -> bool:
        """Can a host from this offer run the group's split within every target? Judged by what
        has been measured to fail, not by what a split host has had the chance to show."""
        return workload_math.split_fits(
            group.caps, self.per_model_on(offer, spec, group.models, group.builds, unrefuted_is_open=True))

    def model_at_latency(self, hardware: str, tag: str, latency_s: float, ceiling: int) -> workload_math.AtLatency:
        """What the request log measured for this build on machines of this card: the whole
        answer, by how many answers the host was serving at once (D115) — from answers served
        with nothing else on the host, so a model's curve never carries another's load (D118).
        Rows from before the same-model count was logged are kept."""
        card = self._card_of(hardware)
        # The samples are cached, not the answer: one read of the log serves every target and
        # every ceiling asked of the same card and build within the window.
        key = (card, tag)
        cached = self._sizing_cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < 30.0:
            return workload_math.workers_at_latency(cached[1], latency_s, ceiling)
        host_ids = [host_id for host_id, seen in self._hardware_of_hosts().items() if self._card_of(seen) == card]
        samples: list[tuple[int, float]] = []
        if host_ids:
            marks = ",".join("?" * len(host_ids))
            rows = self.events.db.query(
                f"SELECT concurrency, latency_ms FROM request_log WHERE outcome = 'ok' AND model_served = ? "
                f"AND concurrency IS NOT NULL AND latency_ms IS NOT NULL AND host_id IN ({marks}) "
                "AND (model_concurrency IS NULL OR model_concurrency = concurrency) "
                "ORDER BY id DESC LIMIT 20000",
                (tag, *host_ids),
            )
            samples = [(int(r["concurrency"]), float(r["latency_ms"]) / 1000) for r in rows]
        # Read once in a while, not once per offer: a search weighs dozens (the review's finding 12).
        self._sizing_cache[key] = (time.monotonic(), samples)
        return workload_math.workers_at_latency(samples, latency_s, ceiling)

    def _hardware_of_hosts(self) -> dict[str, str]:
        """Each rented host's hardware, from the decision log (the `rented` events carry it)."""
        if self._hardware_cache is not None and time.monotonic() - self._hardware_cache[0] < 30.0:
            return self._hardware_cache[1]
        rows = self.events.db.query("SELECT host_id, numbers FROM events WHERE kind = 'rented' AND host_id IS NOT NULL")
        found: dict[str, str] = {}
        for row in rows:
            try:
                hardware = json.loads(row["numbers"] or "{}").get("hardware")
            except ValueError:
                hardware = None
            if hardware:
                found[row["host_id"]] = str(hardware)
        self._hardware_cache = (time.monotonic(), found)
        return found

    def _workers_for_card(self, offer: Offer) -> tuple[int, str]:
        """How many workers a host rented from this offer would run, and why (spec §2.1).

        The first capacity profile the offer matches decides; with none, the rented default,
        which is per card like a profile that names the card (D88, D107).
        """
        capabilities = set(self.rented.capabilities)
        for profile in self.config.capacity_profiles:
            match = profile.match
            if match.hardware is not None and match.hardware.strip().lower() != offer.hardware.strip().lower():
                continue
            if match.gpu is not None and match.gpu.strip().lower() != self._card_of(offer.hardware):
                continue
            if match.min_gpu_memory_gb is not None and offer.gpu_memory_gb < match.min_gpu_memory_gb:
                continue
            if match.capability is not None and match.capability not in capabilities:
                continue
            if match.gpu is not None:
                # Per card, times the cards (D88). A machine with two of them runs twice the
                # work; the second card idling is not what its price was paid for.
                cards = max(1, offer.gpus or 1)
                workers = min(profile.max_workers * cards, self.MOST_WORKERS_A_HOST_MAY_RUN)
                why = (
                    f"capacity profile for {match.gpu}: {profile.max_workers} per card "
                    f"x {cards} card(s)"
                )
                if workers < profile.max_workers * cards:
                    why += f", held at {self.MOST_WORKERS_A_HOST_MAY_RUN}"
                return workers, why + (f" ({profile.note})" if profile.note else "")
            why = f"capacity profile for {match.hardware or 'this hardware'}"
            return profile.max_workers, why + (f" ({profile.note})" if profile.note else "")
        # Per card, like a profile that names the card: a second card left idle is what its
        # price was not paid for, and the engine runs a copy on each (D107).
        cards = max(1, offer.gpus or 1)
        workers = min(self.rented.workers * cards, self.MOST_WORKERS_A_HOST_MAY_RUN)
        why = f"the rented default, {self.rented.workers} per card x {cards} card(s)"
        if workers < self.rented.workers * cards:
            why += f", held at {self.MOST_WORKERS_A_HOST_MAY_RUN}"
        return workers, why + "; no capacity profile matches this hardware"

    def launch_workers_for(self, starts_at: int, workload: Optional[str] = None) -> int:
        """What the engine is *launched* to run at once (D68).

        Under automatic adjustment a host starts at its profile's number and climbs from there,
        and climbing is only free while the engine already has the slots — so the engine is
        started at the most the machine may be asked for. Without it, the engine is launched
        with exactly what the host will be given, as before.
        """
        auto = self.rented.workers_auto
        if not auto.enabled or workload is not None:
            # A workload's host is held to what meets its latency target: climbing past it would
            # trade the target for throughput nobody asked for (D115).
            return starts_at
        return max(starts_at, auto.max)

    def _engine_named(self, name: str) -> Any:
        if name not in self._engines:
            self._engines[name] = get_engine(name)
        return self._engines[name]

    @property
    def engine(self) -> Any:
        """The engine on the machines this fleet buys **next** — not necessarily the pool's
        default (D93). Read from the configuration in force every time: it was once fixed when
        the supervisor started, so a switch to vLLM from the console rented a vLLM image with
        Ollama's start, port and download directory, and a machine that could never serve."""
        return self._engine_named(self.config.rented_engine())

    def engine_of(self, host: "RentedHost") -> Any:
        """The engine this machine was rented to run, whatever the pool buys now."""
        return self._engine_named(host.engine or self.config.rented_engine())

    def port_of(self, host: "RentedHost") -> int:
        return host.engine_port or self.engine_port

    @property
    def engine_port(self) -> int:
        """Where this pool's engine listens on a host it creates — stated, or the engine's own
        default (D92). Keeping the previous engine's port after changing engine would have the
        pool dial a closed door on every machine it rented."""
        return self.config.engine_port()

    @property
    def rented_models(self) -> list[str]:
        """The models the pool may rent **for** (D89, D94).

        With `all` that is the whole set and every host holds it. With `declared` it is what
        `rented.models` names, or the whole set where it names nothing — and each host the pool
        buys is given one of them, not all of them.
        """
        if self.rented.rent_profiles:
            from ..config import rented_profile_models

            return rented_profile_models(self.rented)
        if self.config.pool.models_per_host == "all" or self.rented.models is None:
            return list(self.config.pool.model_set)
        return list(self.rented.models)

    def builds_of(self, models: Sequence[str], builds: Optional[dict[str, str]] = None) -> dict[str, str]:
        """The build a rented host fetches for each of these models: the profile's, where one
        names it (D111), and otherwise the catalog's first for the rented engine."""
        from ..catalog import variants_for_host

        variants = variants_for_host(
            list(models), self.config.catalog, frozenset(self.rented.capabilities),
            self.config.rented_engine(),
        )
        chosen = {model: group[0].tag for model, group in variants.items() if group}
        chosen.update({model: tag for model, tag in (builds or {}).items() if model in chosen or model in models})
        return chosen

    def build_sizes(self, models: Sequence[str], builds: Optional[dict[str, str]] = None) -> dict[str, Optional[float]]:
        """Each model's build size in GB, or None where nobody has measured it: the catalog's
        own `size_gb` first, then what the directory measured (D108)."""
        from ..directory import build_sizes_gb

        chosen = self.builds_of(models, builds)
        stated = {
            v.tag: v.size_gb for entry in self.config.catalog.values() for v in entry.variants if v.size_gb
        }
        sizes = build_sizes_gb(self.leases.db, chosen) if chosen else {}
        for model, tag in chosen.items():
            if tag in stated:
                sizes[model] = stated[tag]
        return sizes

    def model_set_gb(self, models: Sequence[str], builds: Optional[dict[str, str]] = None) -> float:
        """What these models take on disk, from the sizes of the builds this pool's rented
        engine fetches (D108) — the builds the directory has measured. A build it has not is
        left out and said so in `model_sizes_unknown`, rather than guessed at."""
        sizes = self.build_sizes(models, builds)
        self.model_sizes_unknown = sorted(model for model, size in sizes.items() if size is None)
        return round(sum(size for size in sizes.values() if size), 3)

    def needs_of(
        self, models: Sequence[str], builds: Optional[dict[str, str]] = None, cards_per_copy: int = 1
    ) -> Needs:
        """The least card, cards and disk a machine holding these builds needs (D111, D114)."""
        return needs_for(self.build_sizes(models, builds), cards_per_copy)

    def policy_for(
        self, models: Sequence[str], builds: Optional[dict[str, str]] = None,
        base: Optional[OfferPolicy] = None, cards_per_copy: int = 1,
    ) -> OfferPolicy:
        """The search for a machine that will hold these models: the search in force, with its
        card-memory and disk minimums raised to what the models need where they are lower
        (D111), and its cards to whole groups of those each model is split across (D114).
        Never lowered — an operator's higher minimum is a choice, and stands. The disk
        searched for is the disk the host is rented with (D108), so it is raised as one."""
        base = base or self.rented.policy_in_force
        needs = self.needs_of(models, builds, cards_per_copy)
        raised: dict[str, float] = {}
        if needs.card_memory_gb > base.min_gpu_memory_gb:
            raised["min_gpu_memory_gb"] = needs.card_memory_gb
        if needs.disk_gb > base.min_disk_gb:
            raised["min_disk_gb"] = needs.disk_gb
        if cards_per_copy > base.gpus_multiple_of:
            raised["gpus_multiple_of"] = cards_per_copy
        return base.model_copy(update=raised) if raised else base

    @property
    def cards_per_copy_for_new_host(self) -> int:
        """How many cards each copy spans on the next machine: its profile's split (D114)."""
        return self.rented.cards_per_copy(self.profile_for_new_host())

    def next_host_policy(self, base: Optional[OfferPolicy] = None) -> OfferPolicy:
        """The search for the next machine, sized for what it would be bought for."""
        return self.policy_for(
            self.models_for_new_host(), self.builds_for_new_host(), base, self.cards_per_copy_for_new_host
        )

    @property
    def one_model_per_host(self) -> bool:
        """Does a rented host hold a single model (D94)? True where the set is spread across
        hosts and this engine serves one model per process."""
        return (
            self.config.pool.models_per_host != "all"
            and bool(getattr(self.engine, "serves_one_model", False))
        )

    @property
    def hosts_hold_part_of_the_set(self) -> bool:
        """Is each rented host bought for part of the rented set — by profile (D111), or one
        model at a time (D94)? Then which it is bought for, and which may be let go, matter."""
        return bool(self.rented.rent_profiles) or self.one_model_per_host

    def profile_for_new_host(self) -> Optional[str]:
        """Which profile the next machine is bought as (D111) — D95's rule, over profiles.

        1. **A profile holding a model no host serves** — the first such model in the order the
           operator listed profiles, and the first profile holding it. Availability first.
        2. **Otherwise the profile whose models' requests are waiting most.**
        3. **Otherwise the profile with the fewest hosts**, ties to the operator's order.
        """
        profiles = self.rented.profiles_rented()
        if not profiles:
            return None
        names = list(profiles)
        live = self.hosts_of(None)
        serving: dict[str, int] = {}
        for host in live:
            for model in self.models_of(host):
                serving[model] = serving.get(model, 0) + 1
        for model in self.rented_models:
            if not serving.get(model):
                return next(name for name in names if model in profiles[name])

        waiting = getattr(self, "_waiting_by_model", {})
        demand = {name: sum(waiting.get(model, 0) for model in profiles[name]) for name in names}
        if any(demand.values()):
            return max(names, key=lambda name: (demand[name], -names.index(name)))

        count = {name: sum(1 for host in live if host.profile == name) for name in names}
        return min(names, key=lambda name: (count[name], names.index(name)))

    def builds_for_new_host(self) -> dict[str, str]:
        """The builds the next machine is bought to fetch, where a profile names them."""
        name = self.profile_for_new_host()
        return dict(self.rented.model_profiles[name]) if name else {}

    def models_for_new_host(self) -> tuple[str, ...]:
        """Which models the next machine is bought to serve (D94, D95).

        Where a host holds one, two questions are asked in order, and the order is the point:

        1. **Is any model served by nothing?** Then buy for that one. A model with no host
           cannot be served at all, and no amount of throughput elsewhere makes up for it —
           availability comes before capacity.
        2. **Otherwise, which model's traffic is waiting most?** Measured the same way the pool
           measures whether to rent at all: requests that queued or were refused for queueing.
           Buying for coverage alone would keep adding hosts to a model nobody is asking for.

        Ties go to the order the operator listed them, so the answer is stable and explainable.
        """
        profile = self.profile_for_new_host()
        if profile is not None:
            return tuple(self.rented.model_profiles[profile])
        candidates = self.rented_models
        if not self.one_model_per_host or not candidates:
            return tuple(candidates)

        serving: dict[str, int] = {name: 0 for name in candidates}
        for host in self.hosts_of(None):
            for name in host.models:
                if name in serving:
                    serving[name] += 1

        uncovered = [name for name in candidates if serving[name] == 0]
        if uncovered:
            return (uncovered[0],)

        waiting = getattr(self, "_waiting_by_model", {})
        if any(waiting.get(name) for name in candidates):
            busiest = max(candidates, key=lambda name: (waiting.get(name, 0), -candidates.index(name)))
            return (busiest,)

        fewest = min(candidates, key=lambda name: (serving[name], candidates.index(name)))
        return (fewest,)

    def last_host_serving(self, host: RentedHost) -> bool:
        """Would taking this host leave one of its models with no host at all (D95)?

        Only where a host holds a single model: with the whole set on every host, any remaining
        host still serves everything, and the question does not arise. A host that is being
        released, or is not serving, does not count as cover.
        """
        if not self.hosts_hold_part_of_the_set or not host.models:
            return False
        for name in host.models:
            others = [
                other for other in self.hosts_of(host.workload)
                if other.host_id != host.host_id
                and other.state in ("ready", "preparing")
                and name in other.models
            ]
            if not others:
                return True
        return False

    def models_of(self, host: RentedHost) -> list[str]:
        """What this host was bought to serve, falling back to the rented set for one bought
        before the pool assigned models."""
        return list(host.models) if host.models else self.rented_models

    def image_for(self, offer) -> tuple[Optional[str], str]:
        """Which build of the engine this machine gets, and why (D92).

        With a single `image` the answer never varies, and the driver floor in the offer policy
        is what keeps an unusable machine out. With several, the pool takes the first whose
        driver floor this machine meets — so a newer, faster build is used where it can be, and
        an older machine still gets a build that runs rather than being refused outright.
        """
        images = self.rented.images
        if not images:
            return self.rented.image, "the pool's only image"
        chosen = image_for(offer, images)
        if chosen is None:
            wanted = ", ".join(f"{i.image} needs {i.min_driver}" for i in images)
            return None, (
                f"driver: this machine reports {offer.driver_version or 'nothing'}, and no build "
                f"of the engine runs on it ({wanted})"
            )
        return chosen.image, (
            f"driver {offer.driver_version} meets {chosen.min_driver}"
            + (f" — {chosen.note}" if chosen.note else "")
        )

    def instance_env(self, workers: Optional[int] = None, models: Optional[Sequence[str]] = None) -> dict[str, str]:
        """What makes the engine run this many workers at this context, holding the models this
        host is asked for — set at creation on hosts the pool creates (spec §2.2)."""
        return self.engine.launch_settings(
            workers=workers if workers is not None else self.rented.workers,
            context=self.rented.context_length,
            n_models=len(models) if models is not None else len(self.rented_models),
            # The pool reaches this engine through a forward into the machine, never across
            # the network, so it binds loopback and nothing a provider publishes leads to it.
            listen=f"127.0.0.1:{self.engine_port}",
        )

    def max_lease_hours(self) -> Optional[float]:
        """Without a dead-man timer, nothing on the host can stop it billing — so leases are
        capped short there. With several connections, a lease is as long as the one that can arm
        a timer allows; the others are not rented from for it (`lease_hours_on`)."""
        if any(self.lease_hours_on(name) is None for name in self.searchable()):
            return None
        return self.rented.teardown.max_hours_without_deadman

    def lease_hours_on(self, connection: str) -> Optional[float]:
        """The longest lease a host on this connection may serve: None where it arms a timer."""
        if self.provider_for(connection).capabilities.self_terminate:
            return None
        return self.rented.teardown.max_hours_without_deadman

    def _without_deadman(self) -> str:
        """For an operator: which provider accounts cannot arm a dead-man timer, as the start of a
        sentence. Never for an application, which is not told where the pool rents."""
        names = [n for n in self.searchable() if self.lease_hours_on(n) is not None] or [self.connection]
        if len(names) == 1:
            return f"provider account {names[0]!r} cannot arm a dead-man timer"
        return f"none of the provider accounts ({', '.join(names)}) can arm a dead-man timer"

    async def beat_deadman_timers(self) -> None:
        """Refresh the timestamp on every live host, over the connection already held.

        A failure here is not fatal: the timer is *meant* to fire when the supervisor cannot
        reach the host.
        """
        for host in list(self.hosts.values()):
            if host.released or host.connection is None:
                continue
            if host.agent is not None:
                # Through the agent where there is one — and *beside* the ssh beat, not instead
                # of it, until the verb has been seen working on a live host (D63).
                detail = await agents.beat(host.agent, transport=self._agent_transport)
                if detail is not None:
                    host.agent_detail = detail
                else:
                    host.agent_beats = True
            try:
                code, output = await self.run_on_host(host, heartbeat_command())
            except Exception as exc:  # noqa: BLE001 - never let a heartbeat take a pass down
                log.warning("heartbeat to %s failed: %s", host.host_id, exc)
                continue
            if code != 0:
                log.warning("heartbeat to %s failed: %s", host.host_id, output.strip())

    async def _open_tunnel(self, host_id: str, connection: ConnectionInfo,
                           port: Optional[int] = None) -> Optional[str]:
        """The default way to a rented host: a supervised forward, so the engine is never
        exposed (hosts-routing-capacity.md §1.3). Returns the local URL the router dials."""
        if not connection.ssh_host:
            return None
        transport = TransportConfig(
            type="tunnel",
            ssh_host=connection.ssh_host,
            ssh_port=connection.ssh_port or 22,
            ssh_user=connection.ssh_user or self.rented.ssh_user,
            ssh_key=self.rented.ssh_key,
            remote_port=port or self.engine_port,
            known_hosts=self.rented.known_hosts,
        )
        tunnel = self.make_tunnel(host_id, transport)
        self.tunnels[host_id] = tunnel
        # A host that is still booting refuses SSH for a while; the forward keeps retrying,
        # and the probe simply finds nothing listening until it is up.
        await tunnel.start(wait_s=1.0)
        return tunnel.local_url

    async def _open_agent_tunnel(self, host_id: str, connection: ConnectionInfo, port: int) -> Optional[str]:
        """The second forward: the agent listens on the host's loopback and nothing else."""
        if not connection.ssh_host:
            return None
        transport = TransportConfig(
            type="tunnel",
            ssh_host=connection.ssh_host,
            ssh_port=connection.ssh_port or 22,
            ssh_user=connection.ssh_user or self.rented.ssh_user,
            ssh_key=self.rented.ssh_key,
            remote_port=port,
            known_hosts=self.rented.known_hosts,
        )
        tunnel = self.make_tunnel(f"{host_id}/agent", transport)
        self.agent_tunnels[host_id] = tunnel
        await tunnel.start(wait_s=1.0)
        return tunnel.local_url

    async def detach_tunnels(self) -> None:
        """The supervisor is going. Forwards kept by the forwarder stay up for the next one
        (D110); forwards run here go with it, as they always did."""
        for tunnel in [*self.agent_tunnels.values(), *self.tunnels.values()]:
            await tunnel.detach()
        self.agent_tunnels.clear()
        self.tunnels.clear()

    def forward_survived(self, host_id: str) -> bool:
        """Is this host's forward one that outlived the last supervisor, still up on the port
        the router dials? Only a forward kept by the forwarder can have (D110)."""
        tunnel = self.tunnels.get(host_id)
        return bool(getattr(tunnel, "reused", False)) and bool(getattr(tunnel, "up", False))

    async def close_tunnel(self, host_id: str) -> None:
        agent_tunnel = self.agent_tunnels.pop(host_id, None)
        if agent_tunnel is not None:
            await agent_tunnel.stop()
        tunnel = self.tunnels.pop(host_id, None)
        if tunnel is not None:
            await tunnel.stop()

    async def _ssh_run(self, host: RentedHost, command: str) -> tuple[int, str]:
        connection = host.connection
        if connection is None or not connection.ssh_host:
            return 1, "no ssh connection for this host"
        return await run_command(
            build_ssh_exec_command(
                ssh_host=connection.ssh_host,
                ssh_port=connection.ssh_port or 22,
                ssh_user=connection.ssh_user,
                ssh_key=self.rented.ssh_key,
                known_hosts=self.rented.known_hosts,
                command=command,
            )
        )

    async def _ssh_push(self, host: RentedHost, data: bytes, path: str) -> tuple[int, str]:
        """Copy one file to a rented host over the connection the pool already has."""
        connection = host.connection
        if connection is None or not connection.ssh_host:
            return 1, "no ssh connection for this host"
        return await run_command(
            build_ssh_exec_command(
                ssh_host=connection.ssh_host,
                ssh_port=connection.ssh_port or 22,
                ssh_user=connection.ssh_user,
                ssh_key=self.rented.ssh_key,
                known_hosts=self.rented.known_hosts,
                command=hostagent.push_command(path),
            ),
            timeout=120.0,
            stdin=data,
        )

    async def install_agent(self, host: RentedHost) -> None:
        """Put the agent on a host once its SSH answers. Never fatal (D63).

        Tried a few times, because SSH answers before the machine has settled — and then left
        alone. A host that cannot take an agent is not going to start being able to, and
        asking it every pass costs an SSH round trip a pass and fills the log with one host's
        refusal (seen in the simulation: 153 events for two hosts).
        """
        if host.agent is not None or not self.rented.agent_on_rented_hosts:
            return
        if host.agent_attempts >= self.rented.agent_attempts:
            return
        archive = agentpkg.cached(self.state_dir)
        if archive is None:
            host.agent_detail = "the agent could not be packed on the pool's machine"
            return
        try:
            key = await hostagent.install(
                run=lambda command: self.run_on_host(host, command),
                push=lambda data, path: self.push_to_host(host, data, path),
                archive=archive,
                engine_port=self.port_of(host),
                engine=self.engine_of(host).name,
                # Where the agent fetches to, for an engine that does not fetch for itself —
                # the same directory its start command reads (D97).
                models_path=hostagent.MODELS_DIR if self.engine_of(host).loads_by_restart else None,
            )
        except hostagent.HostNotReachable as exc:
            # Not an attempt: nothing was learned about the machine, only that it is not up yet.
            host.agent_detail = f"waiting for the machine to accept SSH ({exc})"
            return
        except hostagent.AgentInstallFailed as exc:
            host.agent_attempts += 1
            host.agent_detail = str(exc)
            if host.agent_attempts >= self.rented.agent_attempts:
                # Said once, when the pool has stopped trying — not once a pass.
                self.events.record(
                    "agent_not_installed",
                    f"{host.host_id} is preparing without an agent: {exc}",
                    numbers={"attempts": host.agent_attempts},
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
            return
        except Exception as exc:  # noqa: BLE001 - an agent is never worth failing a host for
            host.agent_attempts += 1
            host.agent_detail = f"installing the agent failed: {exc}"
            log.warning("installing the agent on %s failed: %s", host.host_id, exc)
            return

        url = None
        if host.connection is not None:
            url = await self._open_agent_tunnel(host.host_id, host.connection, hostagent.AGENT_PORT)
        if url is None:
            host.agent_detail = "the agent is installed but no forward to it could be opened"
            return
        host.agent = hostagent.RentedAgent(url=url, secret=key)
        host.agent_detail = None
        self.events.record(
            "agent_installed",
            f"{host.host_id} runs the pool's agent, reached on its own forward",
            numbers={"digest": agentpkg.digest(archive)},
            host_id=host.host_id,
            lease_id=host.lease_id,
        )

    #: Lines in a machine's boot output worth repeating when a host never answered. A stuck
    #: host writes thousands of lines and almost all of them are ordinary; these are the ones
    #: that have actually explained a failure.
    _TELLING = (
        "remote port forwarding failed",   # the provider's proxy never published the host
        "no space left",
        "out of memory",
        "cannot allocate",
        "permission denied",
        "failed to start",
        "error response from daemon",
        "cuda",
    )

    async def why_it_never_started(self, host: RentedHost, tail: int = 80) -> Optional[str]:
        """What the machine itself said, for a host that never answered (D78).

        The pool has been guessing at these from the outside, and guessed wrong: two hosts were
        read as refusing the pool's SSH key when their own logs said the provider's proxy had
        refused to publish them at all — the same `Permission denied` either way, from opposite
        causes. One call distinguishes them.

        Best-effort by construction. It runs only as a host is given up, never on the request
        path, and anything it cannot get back leaves the host given up exactly as before. What
        comes back is a machine's output: recorded and shown, never executed, and never used to
        decide anything.
        """
        if host.instance is None:
            return None
        try:
            provider = self.provider_for(host)
        except ProviderError:
            return None
        if not provider.capabilities.reports_instance_logs:
            return None
        try:
            text = await provider.instance_logs(host.instance, tail=tail)
        except Exception as exc:  # noqa: BLE001 - diagnosis must never delay a tear-down
            log.debug("could not read %s's boot output: %s", host.host_id, exc)
            return None
        if not text:
            return None

        seen: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or not any(word in stripped.lower() for word in self._TELLING):
                continue
            if stripped not in seen:
                seen.append(stripped)
            if len(seen) >= 3:
                break
        if not seen:
            # Nothing recognised: the last line still beats nothing at all.
            last = [line.strip() for line in text.splitlines() if line.strip()]
            return last[-1][:200] if last else None
        repeats = len(text.splitlines())
        note = "; ".join(line[:160] for line in seen)
        return f"{note} (in {repeats} lines of boot output)"

    async def resize(self, host: RentedHost, workers: int) -> tuple[bool, str]:
        """Change how many requests a running host is given (D56). Returns (done, why).

        The two directions are not the same thing. **Lowering** is the pool using fewer of the
        slots the engine already has: the surplus workers drain and nothing in flight is
        disturbed. **Raising** needs the engine itself to run more at once, which means a
        relaunch — so it needs this host's agent, and it costs the host about a minute.
        """
        if workers < 1:
            return False, "a host serves with at least one worker"
        if workers == host.workers:
            return True, f"{host.host_id} already runs {workers} workers"
        if host.workload is not None and len(host.models) > 1:
            # Its workers are its models' shares added up (D118): fewer, and one model's share
            # starves the other's; more, and nothing ever uses them.
            return False, (f"{host.host_id} runs its workload's split of {', '.join(host.models)}: its worker "
                           "count is the sum of their shares, and is not changed by hand")

        if workers < host.workers:
            was = host.workers
            host.workers = workers
            self.events.record(
                "host_resized",
                f"{host.host_id} now takes {workers} requests at once, down from {was}; the "
                "surplus workers drain and the engine is left alone",
                numbers={"workers": workers, "was": was, "restarted": False},
                host_id=host.host_id,
                lease_id=host.lease_id,
            )
            return True, f"{host.host_id} now takes {workers} at once"

        if workers <= (host.launch_workers or 0):
            # Its engine was already launched to run this many, so the pool is simply using
            # slots that exist: instant, graceful, and nothing is restarted (D68).
            was = host.workers
            host.workers = workers
            self.events.record(
                "host_resized",
                f"{host.host_id} now takes {workers} requests at once, up from {was}; its "
                f"engine was launched for {host.launch_workers}, so nothing restarted",
                numbers={"workers": workers, "was": was, "restarted": False},
                host_id=host.host_id,
                lease_id=host.lease_id,
            )
            return True, f"{host.host_id} now takes {workers} at once"

        if host.agent is None:
            return False, (
                f"{host.host_id} has no agent, and raising its workers past the "
                f"{host.launch_workers} its engine was launched for means relaunching it — "
                "which only an agent on that host can do"
            )
        settings = agents.wanted_engine_settings(workers, len(self.tags_for(host)), host.cards_per_copy)
        status, answer = await agents.restart_engine(
            host.agent, settings, transport=self._agent_transport
        )
        if status != 200:
            return False, f"its agent refused the change: {answer.get('detail') or status}"
        if not answer.get("engine_answers"):
            # It restarted and did not come back: the host keeps the count it can actually
            # serve, and the probe finds it unready until the engine returns.
            self.events.record(
                "host_resize_failed",
                f"{host.host_id}: its engine was relaunched for {workers} workers and is not answering",
                numbers={"workers": workers},
                host_id=host.host_id,
                lease_id=host.lease_id,
            )
            return False, "the engine was relaunched and is not answering"
        was = host.workers
        host.workers = workers
        host.mark_preparing()  # it has to hold the model set again before it is routed to
        self.events.record(
            "host_resized",
            f"{host.host_id} now takes {workers} requests at once, up from {was}; its engine "
            "was relaunched by its own agent",
            numbers={"workers": workers, "was": was, "restarted": True},
            host_id=host.host_id,
            lease_id=host.lease_id,
        )
        return True, f"{host.host_id} now takes {workers} at once"

    async def ask_agent(self, host: RentedHost) -> None:
        """What the machine says about itself — load, accelerator, memory (D63)."""
        if host.agent is None:
            return
        view = await agents.ask(host.agent, transport=self._agent_transport)
        if view.reachable:
            host.agent_facts = view.facts
            host.agent_detail = None
            self._volume_news(host)
        else:
            host.agent_facts = None
            host.agent_detail = view.detail

    # --- what would happen (spec §3) ---

    async def plan(self, ready_workers_higher_tiers: int) -> list[dict]:
        """Everything the next pass would do, and nothing else. Spends nothing.

        This is the payoff of strategies being pure functions: the plan is produced by asking
        the same functions the pass would ask, not by a second implementation that can drift.
        """
        steps: list[dict] = []
        open_leases = self.leases.open_leases()

        instances = []
        for name, provider in list(self.providers.items()):
            try:
                instances += [(name, i) for i in await provider.list_instances(self.label_prefix)]
            except ProviderError as exc:
                steps.append({"step": "observe", "detail": f"{name} is not answering: {exc}"})

        known = {(h.connection_name, h.instance.instance_id) for h in self.hosts.values() if not h.released}
        for name, instance in instances:
            if (name, instance.instance_id) not in known:
                steps.append(
                    {
                        "step": "sweep",
                        "host": instance.instance_id,
                        "detail": "carries this pool's label but was never intended; would be destroyed",
                    }
                )

        for lease in open_leases:
            spent, estimate = self.lease_spend(lease)
            limit = lease.max_spend * (1 - self.margin())
            steps.append(
                {
                    "step": "lease",
                    "lease_id": lease.lease_id,
                    "spent_enforced_on": round(spent, 4),
                    "estimate": round(estimate, 4),
                    "stops_at": round(limit, 4),
                    "hours_left": round(lease.hours_left(), 3),
                    "detail": "would be closed and released" if spent >= limit else "within its caps",
                }
            )

        lease = self.lease_of(open_leases, None)
        rented_workers = sum(h.workers for h in self.hosts_of(None) if h.state == "ready")
        demand = Demand(
            wanted_workers=lease.workers if lease else 0,
            ready_workers_higher_tiers=ready_workers_higher_tiers,
            rented_workers=rented_workers,
            overflow_age_s=(time.time() - self.unit(None).overflow_since) if self.unit(None).overflow_since else 0.0,
            hosts_pending=sum(1 for h in self.hosts_of(None) if h.state in ("scheduling", "preparing")),
        )
        decision = decide_rent(
            demand, self._lease_view(lease) if lease else None, self.rented.scale, self.rented.workers
        )
        refusal = self._refuse_for_caps(demand, lease) if decision.rent else None
        step = {
            "step": "acquire",
            "overflow": demand.overflow,
            "would_rent": bool(decision.rent and refusal is None),
            "reasons": decision.reasons,
        }
        if refusal:
            step["refused_by_caps"] = refusal
        if decision.rent and refusal is None and lease is not None:
            policy = self.next_host_policy()
            offers = await self._offers(policy)
            ranked, rejected = rank_offers(
                offers, self._policy_with_avoided(policy), self.rented.bidding,
                lease.hours_left(), self.model_set_gb(self.models_for_new_host(), self.builds_for_new_host()),
                history=self.machine_history(), history_cfg=self.rented.history,
            )
            step["offers_seen"] = len(offers)
            step["offers_rejected"] = {key: value for key, value in list(rejected.items())[:10]}
            if ranked:
                best, best_score = ranked[0]
                bid = price_bid(best, self.rented.bidding, self.rented.policy_in_force, lease.max_all_in_hourly)
                step["would_bid"] = {
                    "machine": best.machine_id,
                    "hardware": best.hardware,
                    "floor": best.min_bid_hourly,
                    "bid": bid.hourly,
                    "score": round(best_score, 3),
                    "reasons": bid.reasons,
                }
                # The same check the pass makes once it knows what it would bid.
                burn_refusal = self._refuse_bid_for_burn(bid.hourly)
                if burn_refusal is not None:
                    step["would_rent"] = False
                    step["refused_by_caps"] = burn_refusal
            else:
                step["would_rent"] = False
                step["refused_by_caps"] = "no offer passed the policy"
        steps.append(step)
        return steps

    async def market_preview(
        self,
        hours: float = 4.0,
        offer_policy: Optional[dict] = None,
        bidding: Optional[dict] = None,
        kinds: Optional[str] = None,
        search: bool = True,
    ) -> dict:
        """The offer pipeline, read-only (docs/spec/console-and-control-api.md §2.2).

        `search=False` answers with the settings and the next host's needs, and asks the market
        nothing: the provider counts every offer a search returns against a daily quota, which
        a screen merely being open must not spend (D120).

        Moving a ceiling and watching "4 pass" become "0 pass" is the fastest way to learn what
        a number means — so this runs the *same* filters and the same bid strategy the
        supervisor would, and creates nothing.
        """
        # Unsaved values from the console's form, merged over what is configured. Validated
        # here, so a typo in the form is a 400 and never something the pool acts on.
        policy = self.rented.policy_in_force
        bid_config = self.rented.bidding
        if offer_policy:
            policy = OfferPolicy.model_validate({**policy.model_dump(), **offer_policy})
        if bidding:
            bid_config = BiddingConfig.model_validate({**bid_config.model_dump(), **bidding})
        # Searched as the next host would be: minimums raised to what it would hold (D111).
        typed = policy
        next_models, next_builds = self.models_for_new_host(), self.builds_for_new_host()
        next_cards = self.cards_per_copy_for_new_host
        policy = self.policy_for(next_models, next_builds, typed, next_cards)
        needs = self.needs_of(next_models, next_builds, next_cards)

        if kinds is not None and kinds not in self._KINDS:
            raise ValueError(f"kinds must be one of {sorted(self._KINDS)}")
        offers = await self._offers(policy, kinds) if search else []
        ranked, rejected = rank_offers(
            offers, self._policy_with_avoided(policy), bid_config, hours,
            self.model_set_gb(next_models, next_builds),
            history=self.machine_history(), history_cfg=self.rented.history,
        )
        if ranked:
            # The order the pool rents in: least expected cost per worker-hour (D131), not the score.
            lease_hours = max((lease.hours_left() for lease in self.leases.open_leases() if lease.workload is None),
                              default=hours or 1.0)
            ordered, _why = self.expected_costs(ranked, lease_hours, None, next_models, next_builds,
                                                policy.max_all_in_hourly)
            by_key = {(o.connection, o.offer_id, o.interruptible): (o, s) for o, s in ranked}
            cost_of = {(c.connection, c.offer_id, c.interruptible): c.per_worker_hour for c in ordered}
            ranked = [by_key[(c.connection, c.offer_id, c.interruptible)] for c in ordered] + [
                pair for pair in ranked if (pair[0].connection, pair[0].offer_id, pair[0].interruptible) not in cost_of]
        else:
            cost_of = {}
        problem = self.last_offer_error if search else None
        by_reason: dict[str, int] = {}
        for reasons in rejected.values():
            # An offer is counted against the first filter that stopped it.
            key = filter_name(reasons[0])
            by_reason[key] = by_reason.get(key, 0) + 1

        accepted = []
        for offer, offer_score in ranked[:10]:
            bid = price_bid(offer, bid_config, policy)
            workers, workers_why = self.workers_for(offer)
            accepted.append(
                {
                    # What to name to rent exactly this one, and how it would be rented.
                    "offer_id": offer.offer_id,
                    "kind": "interruptible" if offer.interruptible else "on_demand",
                    # Which provider lists it, and how it is paid for: bid, spot or on demand (D129, D132).
                    "connection": offer.connection,
                    "priced": ("bid" if offer.bidding else "spot") if offer.interruptible else "on_demand",
                    # What the pool chooses by (D131): expected dollars per worker-hour.
                    "per_worker_hour": cost_of.get((offer.connection, offer.offer_id, offer.interruptible)),
                    "machine": offer.machine_id,
                    "hardware": offer.hardware,
                    # What a host rented from this offer would run, and whether a profile says so.
                    "workers": workers,
                    "workers_from": workers_why,
                    "gpu_memory_gb": round(offer.gpu_memory_gb, 1),
                    "gpus": offer.gpus,
                    "floor": offer.min_bid_hourly,
                    "would_bid": bid.hourly,
                    "all_in": offer.all_in_hourly,
                    "on_demand": offer.on_demand_hourly,
                    "download_per_gb": offer.download_per_gb,
                    "download_mbps": offer.download_mbps,
                    "storage_hourly": round(offer.storage_hourly, 5),
                    "reliability": offer.reliability,
                    # What the provider does not report, filled with a stated default (§6).
                    "assumed": list(offer.assumed),
                    "score": round(offer_score, 3),
                }
            )
        return {
            "searched": search,
            # What this search, and the day's others, have used of the provider's quota (D121).
            "search_quota": self.search_quota(),
            "seen": len(offers),
            "passed": len(ranked),
            "rejected": len(rejected),
            "rejected_by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
            # Filters the provider applied before answering (D123): an offer failing one is never
            # returned, so it is not counted among the rejections — said here instead.
            "filtered_by_provider": self.last_query_filters if search else [],
            # None when the market really was asked. Set when it could not be, so nobody
            # reads "0 offers seen" as "there is nothing out there" (D44).
            "problem": problem,
            # A provider that could not be asked while others answered: what is shown is theirs (D129).
            "provider_errors": ({n: self.searching[n].error for n in self.searchable()
                                 if n in self.searching and self.searching[n].error} if search and not problem else {}),
            # Machines skipped for now because they just failed, and why.
            "avoided": self.avoided_now(),
            "best": accepted,
            # What the next host would be bought for takes this much disk, from its builds' sizes;
            # a build the directory has not measured is named rather than counted as nothing.
            "model_set_gb": self.model_set_gb(next_models, next_builds),
            "model_sizes_unknown": self.model_sizes_unknown,
            # What the next host is bought as and holds, what that needs, and the minimums the
            # search used because of it — beside the operator's own (D111).
            "next_host": {
                "profile": self.profile_for_new_host(),
                "models": list(next_models),
                "builds": self.builds_of(next_models, next_builds),
                "needs": needs.as_dict(),
                "searched": {"min_gpu_memory_gb": policy.min_gpu_memory_gb, "min_disk_gb": policy.min_disk_gb,
                             "gpus_multiple_of": policy.gpus_multiple_of},
                "typed": {"min_gpu_memory_gb": typed.min_gpu_memory_gb, "min_disk_gb": typed.min_disk_gb,
                          "gpus_multiple_of": typed.gpus_multiple_of},
            },
            "policy": {
                "max_all_in_hourly": policy.max_all_in_hourly,
                "disk_gb": policy.min_disk_gb,
                "premium": bid_config.premium,
                "on_demand_crossover": bid_config.on_demand_crossover,
            },
            # What this preview ran with, and what is saved — the console builds its form from
            # these, so the fields are always the pool's own, never a copy that can drift.
            "offer_policy": policy.model_dump(),
            "bidding": bid_config.model_dump(),
            "saved": {
                # What is *in force* — the named profile where one is chosen (D87), so the
                # console shows the filters the pool is really searching with.
                "offer_policy": self.rented.policy_in_force.model_dump(),
                "search_profile": self.rented.search_profile,
                "search_profiles": sorted(self.rented.search_profiles),
                "bidding": self.rented.bidding.model_dump(),
                # Which listings are searched at all (D80). Not a filter: an offer in a listing
                # this pool never asks for is not rejected, it is never seen — so it cannot
                # appear among the reasons below, and an operator looking at a fixed-price host
                # in the market had no way to learn why it was never rented.
                "mode": self.rented.mode,
                # How capacity is allocated, edited on the same screen (D74): two opt-in
                # features that spend money should not be visible only in a file.
                "allocation": self.rented.allocation,
                "dynamic": self.rented.dynamic.model_dump(),
                "workers_auto": self.rented.workers_auto.model_dump(),
                # Every lever that decides when a host is given up (D86). The whole block was
                # missing from the console, so an operator wanting to change how long a stuck
                # host bills had to edit the supervisor's own file.
                "teardown": self.rented.teardown.model_dump(),
            },
        }

    # --- preparing a host on request (spec §8) ---

    async def prepare(
        self,
        *,
        max_spend: float,
        max_hours: float,
        max_all_in_hourly: Optional[float] = None,
        when_ready: str = "join",
        engine: Optional[object] = None,
        client_for: Optional[object] = None,
        offer_id: Optional[str] = None,
        kind: Optional[str] = None,
        connection: Optional[str] = None,
    ) -> Optional[RentedHost]:
        """Get a host ready *before* a run, or keep one warm between runs.

        It is its own small lease: it cannot start without a price ceiling, a dollar cap and a
        time limit, and it borrows authority from no other open lease.
        """
        lease = self.open_lease(
            workers=self.rented.workers,
            max_hours=max_hours,
            max_spend=max_spend,
            allow_rent=True,
            max_all_in_hourly=max_all_in_hourly,
        )
        self.events.record(
            "prepare_started",
            f"preparing a host: up to ${max_spend:.2f} over {max_hours}h, then {when_ready}",
            numbers={"max_spend": max_spend, "max_hours": max_hours, "when_ready": when_ready},
            lease_id=lease.lease_id,
        )

        if kind is not None and kind not in ("interruptible", "on_demand"):
            self.leases.close(lease.lease_id, "nothing could be prepared")
            raise LeaseRefused("kind must be interruptible or on_demand")
        # A parked host is reused only when the operator did not name a particular one.
        reused = None if (offer_id or kind) else await self.restart_parked(lease)
        if reused is None:
            # A new rental counts against the pool's host limit, as every other does: without
            # this an operator could prepare past it, and the worst case it bounds (D46) was not.
            live = len([h for h in self.hosts.values() if not h.released]) + len(self.held_back_rows())
            reserved = self.reserved_hosts()
            if live + reserved >= self.config.limits.max_rented_hosts:
                self.last_refusal = (f"{live} rented hosts already"
                                     + (f", and {reserved} reserved for workloads still starting" if reserved else "")
                                     + f", at the pool's limit of {self.config.limits.max_rented_hosts}")
                self.leases.close(lease.lease_id, "nothing could be prepared")
                return None
        host = reused or await self.rent_one(
            lease, ["prepared on request" + (f": chosen offer {offer_id}" if offer_id else "")],
            offer_id=offer_id, kind=kind, connection=connection,
        )
        if host is None:
            self.leases.close(lease.lease_id, "nothing could be prepared")
            return None

        host.prepared = True
        host.hold_until = time.time() + max_hours * 3600
        host.when_ready = when_ready
        return host

    def avoid(self, machine_id: str, why: str, minutes: Optional[float] = None) -> None:
        """Do not bid on this machine again for a while. Without this the best-ranked offer —
        the machine that just failed — is simply rented again (seen live: four times running)."""
        minutes = self.rented.teardown.avoid_failed_machine_minutes if minutes is None else minutes
        if minutes <= 0:
            return
        self.avoided[machine_id] = (time.time() + minutes * 60, why)
        self.events.record(
            "machine_avoided",
            f"machine {machine_id} {why}; not bidding on it for {minutes:g} minutes",
            numbers={"machine": machine_id, "minutes": minutes},
        )

    def avoided_now(self) -> dict[str, str]:
        now = time.time()
        self.avoided = {m: (until, why) for m, (until, why) in self.avoided.items() if until > now}
        skipped = {m: why for m, (_, why) in self.avoided.items()}
        # A machine this pool already rents still appears in the interruptible listing — its
        # GPU is "available" to anyone who outbids the tenant, and the tenant is us. Bidding on
        # it either fails or evicts our own host at a higher price (seen live: both).
        for host in self.hosts.values():
            if not host.released:
                skipped.setdefault(host.offer.machine_id, f"already rented by this pool as {host.host_id}")
        return skipped

    def _policy_with_avoided(self, policy: OfferPolicy) -> OfferPolicy:
        skipped = self.avoided_now()
        if not skipped:
            return policy
        return policy.model_copy(update={"avoid_machines": sorted(set(policy.avoid_machines) | set(skipped))})

    async def _pull_with_retries(self, host: RentedHost, engine, client, tag: str):
        """One model's download, tried again when the failure is the kind that passes.

        Found live: three H200 hosts in a row were destroyed because the download stream was
        cut partway ("peer closed connection without sending complete message body"). The
        engine keeps the layers that arrived, so another attempt resumes rather than restarts
        — much cheaper than a new host, and a new host re-downloads everything. A failure the
        engine calls permanent (a tag it does not have) is not retried. Returns None if the
        host is released meanwhile; the last result otherwise.
        """
        attempts = self.rented.teardown.pull_attempts
        wait = self.rented.teardown.pull_retry_after_s
        result = None
        for attempt in range(1, attempts + 1):
            if host.released:
                return None
            host.stage = f"downloading {tag}" + (f" (attempt {attempt} of {attempts})" if attempt > 1 else "")

            samples: list[tuple[float, int]] = []

            def progress(done: int, total: int, tag=tag, attempt=attempt, samples=samples) -> None:
                now = time.monotonic()
                samples.append((now, done))
                grace = self.rented.teardown.slow_pull_grace_s
                while len(samples) > 1 and now - samples[0][0] > max(grace, 1.0):
                    samples.pop(0)
                window = now - samples[0][0]
                mbps = (done - samples[0][1]) * 8 / 1e6 / window if window > 0 else None
                host.progress[tag] = {"completed": done, "total": total, "attempt": attempt, "mbps": mbps}
                floor = self.rented.teardown.min_pull_mbps
                # Judged only over a full window: a download ramps up, and a blip is not a verdict.
                if floor > 0 and mbps is not None and window >= grace and mbps < floor and done < total:
                    raise PullTooSlow(
                        f"downloading {tag} at {mbps:.0f} Mbps, under the {floor:.0f} Mbps floor for "
                        f"{window:.0f}s — the offer advertised {host.offer.download_mbps:.0f} Mbps"
                    )

            try:
                result = await self._pull(engine, client, tag, progress)
            except PullTooSlow as slow:
                # Not retried: the same link on the same machine will be just as slow.
                self.events.record(
                    "host_too_slow", f"{host.host_id}: {slow}",
                    numbers={"tag": tag, "floor_mbps": self.rented.teardown.min_pull_mbps,
                             "advertised_mbps": host.offer.download_mbps},
                    host_id=host.host_id, lease_id=host.lease_id,
                )
                return PullResult(tag=tag, ok=False, detail=str(slow), retryable=False)
            if result.ok:
                host.progress[tag] = {"completed": result.bytes_total, "total": result.bytes_total, "attempt": attempt}
            if result.ok or not result.retryable or attempt == attempts:
                if not result.ok and attempt > 1:
                    result = dataclasses.replace(
                        result, detail=f"{result.detail} (after {attempt} attempts)"
                    )
                return result
            self.events.record(
                "pull_retry",
                f"{host.host_id}: pulling {tag} was cut ({result.detail}); trying again in "
                f"{wait:.0f}s — attempt {attempt + 1} of {attempts}, resuming what arrived",
                numbers={"tag": tag, "attempt": attempt + 1, "of": attempts, "wait_s": wait},
                host_id=host.host_id,
                lease_id=host.lease_id,
            )
            await asyncio.sleep(wait)
            wait = min(wait * 2, 120.0)
        return result

    @staticmethod
    async def _pull(engine, client, tag: str, progress):
        """Ask for progress where the engine reports it, and do not require it: an engine
        adapter written before progress existed takes two arguments and still works."""
        try:
            return await engine.pull(client, tag, on_progress=progress)
        except TypeError:
            return await engine.pull(client, tag)

    def _progress_from_agent(self, host: RentedHost, tags: list[str], held: dict) -> dict:
        """The agent's report, in the one shape everything downstream reads.

        Both preparation paths feed the same console and the same slow-download check, so they
        must say the same thing: `{tag: {completed, total, attempt, mbps}}`. Found live, the
        first time a rented host was ever prepared through its agent: the agent's own words
        were passed along as they came, and the console drew an empty bar for a download that
        was running at a gigabit.
        """
        now = time.time()
        progress: dict = {}
        for tag in tags:
            model = held.get(tag) or {}
            pulling = model.get("pulling") or {}
            size = int(model.get("size_bytes") or 0)
            if pulling:
                completed = int(pulling.get("completed_bytes") or 0)
                total = int(pulling.get("total_bytes") or 0)
            elif size and (model.get("on_disk") or model.get("loaded")):
                completed = total = size
            else:
                continue
            entry = {"completed": completed, "total": total, "attempt": 1}
            before = (host.progress or {}).get(tag) or {}
            seen_at = before.get("seen_at")
            if pulling and seen_at and now > seen_at and completed >= before.get("completed", 0):
                entry["mbps"] = (completed - before["completed"]) * 8 / 1e6 / (now - seen_at)
            entry["seen_at"] = now
            progress[tag] = entry
        return progress

    #: How long a just-installed agent is given to start answering before the pool prepares
    #: the host without it.
    agent_hold_attempts = 5
    agent_hold_retry_s = 2.0

    async def load_model_set_through_agent(self, host: RentedHost) -> Optional[bool]:
        """Let the host's own agent fetch and hold the model set (D63, stage 3).

        It works toward the whole desired state in the background, one pull at a time, and
        loads each model the moment that model's own download finishes (D57) — so the last
        phase of preparation is not a minute of an idle accelerator on a billing host.

        Returns True when the set is held, False while it is still coming, None when this host
        has no agent to ask — the caller then prepares it the way it always did.
        """
        if host.agent is None or not host.agent.manage_models:
            return None
        tags = sorted(self.tags_for(host))
        # An agent that was started a second ago may not be listening yet. Seen live: the pool
        # asked the moment the install returned, the one call failed, and the host was prepared
        # the slow way for its whole life — a minute of idle accelerator that D57 exists to
        # remove. A few seconds of patience is cheap next to that; a real silence still falls
        # back, as before.
        report = None
        for attempt in range(self.agent_hold_attempts):
            if attempt:
                await asyncio.sleep(self.agent_hold_retry_s)
            report = await agents.hold(host.agent, tags, "pinned", transport=self._agent_transport)
            if report is not None:
                break
        if report is None:
            host.agent_detail = "the agent stopped answering while the model set was loading"
            return None
        host.agent_models = report
        held = {
            model.get("tag"): model
            for model in report.get("models", [])
            if isinstance(model, dict)
        }
        failed = [
            f"{tag}: {held[tag].get('error')}"
            for tag in tags
            if tag in held and held[tag].get("error")
        ]
        if failed:
            self.events.record(
                "prepare_failed",
                f"{host.host_id}: its agent could not hold the model set — {'; '.join(failed)}",
                host_id=host.host_id,
                lease_id=host.lease_id,
            )
            return False
        host.progress = self._progress_from_agent(host, tags, held)
        if not all(held.get(tag, {}).get("loaded") for tag in tags):
            if report.get("loads_by_restart"):
                self._restart_when_downloaded(host, tags, held, report)
            return False
        host.download_cost = host.offer.download_per_gb * (
            sum(int(model.get("size_bytes") or 0) for model in held.values()) / 1e9
        )
        self.events.record(
            "prepared",
            f"{host.host_id} holds the pool's model set, fetched by its own agent",
            numbers={"tags": tags},
            host_id=host.host_id,
            lease_id=host.lease_id,
        )
        return True

    #: How long before a failed restart request is tried again. The agent answers fast when it
    #: refuses, so without a pause a machine that cannot restart would be asked every pass.
    RESTART_RETRY_S = 120.0
    #: How long after a re-bid a "stopped" instance is the pool's own restart in flight rather
    #: than a new eviction (D106). Long enough for a provider to bring a container back; short
    #: enough that a start that never happens is judged within the pass or two after.
    REBID_GRACE_S = 120.0

    def _restart_when_downloaded(
        self, host: RentedHost, tags: list[str], held: dict[str, dict], report: dict
    ) -> None:
        """Start the engine again once every model this host was bought for is on disk (D97).

        An engine that serves only what it was started with cannot be asked to load a model, so
        on a host the pool created this *is* loading: the agent fetched the weights, and the
        pool's own restart script — installed at boot, never sent over the protocol — starts
        the engine on them. Without it a machine with every model safely downloaded sat
        unserving until it was given up, and was paid for the whole time.

        Asked in the background: a restart waits for the engine to answer, which on a large
        model is minutes, and nothing slow runs in the control pass.
        """
        if host.agent is None:
            return
        # Every model complete on disk, and nothing still downloading. Not the agent's "busy":
        # asking it starts a pass and it reports in the same breath, so it is busy every time
        # the pool asks — waiting on that waited for ever. "On disk" already means complete
        # (the fetch marks a model only when every file has landed).
        if not all(held.get(tag, {}).get("on_disk") for tag in tags):
            return  # still fetching: a restart now would start the engine on half the set
        if any(held.get(tag, {}).get("pulling") for tag in tags):
            return
        if not any(held.get(tag, {}).get("awaiting_restart") for tag in tags):
            return
        on_disk = frozenset(tags)
        if host.restarted_for == on_disk:
            return  # asked already; the engine is loading them
        now = time.time()
        if host.restart_asked_at is not None and now - host.restart_asked_at < self.RESTART_RETRY_S:
            return
        if host.restart_task is not None and not host.restart_task.done():
            return
        host.restarted_for = on_disk
        host.restart_asked_at = now
        self.events.record(
            "engine_restart",
            f"{host.host_id}: every model it was bought for is on disk "
            f"({', '.join(sorted(tags))}); starting its engine on them",
            host_id=host.host_id,
            lease_id=host.lease_id,
        )
        settings = agents.wanted_engine_settings(host.launch_workers or host.workers, len(tags), host.cards_per_copy)

        async def ask() -> None:
            status, answer = await agents.restart_engine(
                host.agent, settings, transport=self._agent_transport
            )
            if status != 200:
                # Let the next pass after the pause ask again, rather than wait on a restart
                # that was never made.
                host.restarted_for = None
                self.events.record(
                    "engine_restart_failed",
                    f"{host.host_id}: its agent did not restart the engine — "
                    f"{answer.get('detail') or answer.get('error') or status}",
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )

        host.restart_task = asyncio.create_task(ask(), name=f"gpm:restart:{host.host_id}")

    async def load_model_set(self, host: RentedHost, engine, client) -> bool:
        """Fetch the set, loading each model as its own download finishes, then check they are
        resident **together** — a host that cannot hold the whole set does not join the pool.

        Loading as each lands rather than after them all (D57) is worth about a minute of idle
        accelerator on a billing host, and it holds whether or not this host runs an agent: the
        agent does the same thing on hosts that have one.
        """
        tags = sorted(self.tags_for(host))
        moved = 0
        for tag in tags:
            result = await self._pull_with_retries(host, engine, client, tag)
            if result is None:
                return False  # the host was released while its download was being retried
            if not result.ok:
                self.events.record(
                    "prepare_failed",
                    f"{host.host_id}: pulling {tag} failed: {result.detail}",
                    numbers={"tag": tag, "retryable": result.retryable},
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
                self.avoid(host.offer.machine_id, f"could not download {tag}")
                return False
            moved += result.bytes_total
            host.stage = f"loading {tag} while the rest downloads"
            try:
                await engine.load_and_pin(client, [tag])
            except Exception as exc:  # noqa: BLE001 — the whole-set check below is the verdict
                # Not fatal on its own: what decides is whether the set is resident together,
                # which is checked once everything has landed.
                log.info("%s: %s did not load as it landed (%s); trying again with the set", host.host_id, tag, exc)

        host.stage = "checking the model set is resident together"
        try:
            await engine.load_and_pin(client, tags)
        except Exception as exc:  # noqa: BLE001 - the engine says why, and the host does not join
            self.events.record(
                "prepare_failed",
                f"{host.host_id}: the model set would not stay resident together: {exc}",
                host_id=host.host_id,
                lease_id=host.lease_id,
            )
            return False

        host.stage = ""
        download_cost = host.offer.download_per_gb * (moved / 1e9)
        host.download_cost = download_cost
        self.events.record(
            "prepared",
            f"{host.host_id} holds the pool's model set; download moved {moved / 1e9:.2f}GB "
            f"at ${download_cost:.4f}",
            numbers={"bytes": moved, "download_cost": download_cost, "tags": tags},
            host_id=host.host_id,
            lease_id=host.lease_id,
        )
        return True

    @property
    def required_tags(self) -> frozenset[str]:
        """Every build the pool may be asked to put on a rented host. Where each host holds one
        model this is the union across them, not what any single machine carries."""
        return self.tags_for(None)

    def tags_for(self, host: Optional[RentedHost]) -> frozenset[str]:
        """The builds one host must hold (D94) — what it was bought for, or the whole rented set
        for a host bought before the pool assigned models."""
        if host is not None and host.builds:
            # Bought as a profile: exactly the builds it named, whatever the catalog lists first.
            return frozenset(host.builds.values())
        if host is None and self.rented.rent_profiles:
            return frozenset(
                tag for models in self.rented.profiles_rented().values() for tag in models.values()
            )
        from ..catalog import variants_for_host

        variants = variants_for_host(
            self.models_of(host) if host is not None else self.rented_models,
            self.config.catalog,
            frozenset(self.rented.capabilities),
            self.config.rented_engine(),
        )
        return frozenset(group[0].tag for group in variants.values() if group)

    # --- parking (spec §8) ---

    async def restart_parked(self, lease: Lease, workload: Optional[str] = None,
                             group: Optional[tuple[str, ...]] = None) -> Optional[RentedHost]:
        """Parked hosts are tried before new offers: no download, minutes instead of tens of
        minutes. It must still win the auction on that machine within the ceilings. Only this
        demand's own: another workload's host holds another model (D115), and another group's
        other models (D118)."""
        for host in self.hosts_of(workload, group):
            if host.state != "parked":
                continue
            if host.connection_name not in self.searchable():
                continue  # a disabled connection rents nothing, a restart included (D129)
            limit = self.lease_hours_on(host.connection_name)
            if limit is not None and (limit < lease.max_hours or host.hours_held >= limit):
                continue  # no dead-man timer there: this lease, or this host's hours, run past it
            offers = await self._offers(connections=[host.connection_name])
            same_machine = next((o for o in offers if self._at(o) == self._at(host.offer)
                                 and o.interruptible == host.offer.interruptible
                                 and o.bidding == host.offer.bidding), None)
            if same_machine is None:
                continue
            bid = price_bid(same_machine, self.rented.bidding, self.rented.policy_in_force, lease.max_all_in_hourly)
            capped = self._cap_bid(bid.hourly, lease, same_machine)
            if capped is None:
                continue
            refused = self._refuse_rebid(host, capped, lease)
            if refused is not None:
                self.events.record("park_restart_refused", f"{host.host_id} stays parked: {refused}",
                                   host_id=host.host_id, lease_id=lease.lease_id)
                continue
            try:
                provider = self.provider_for(host)
                if same_machine.bidding:
                    await provider.set_bid(host.instance, capped)
                await provider.start(host.instance)
            except ProviderError as exc:
                self.events.record(
                    "park_restart_failed",
                    f"{host.host_id} could not be restarted: {exc}; it stays parked",
                    host_id=host.host_id,
                    lease_id=lease.lease_id,
                )
                continue
            host.settle()  # parked until now, at its disk's rate
            if host.lease_id != lease.lease_id:
                host.move_to_lease(lease.lease_id)  # the new lease pays from its restart only
            host.bid_hourly = capped
            host.mark_preparing()  # its clock starts now, not when it was first rented
            host.parked_at = None
            host.idle_since = None
            self.events.record(
                "park_restarted",
                f"{host.host_id} restarted from parked at ${capped:.3f}/h — it already holds "
                "the models, so nothing is downloaded",
                numbers={"bid": capped, "machine": host.offer.machine_id},
                host_id=host.host_id,
                lease_id=lease.lease_id,
            )
            return host
        return None

    async def expire_parked(self) -> None:
        """Parking is never open-ended: past the limit the disk stops being worth its storage."""
        limit = self.rented.teardown.max_park_hours * 3600
        now = time.time()
        for host in list(self.hosts.values()):
            if host.released or host.state != "parked" or host.parked_at is None:
                continue
            idle_limit = self.rented.teardown.destroy_after_minutes * 60
            if host.idle_since is not None and now - host.idle_since >= idle_limit:
                await self.destroy(
                    host,
                    f"unused for {(now - host.idle_since) / 60:.1f} min, past the "
                    f"{self.rented.teardown.destroy_after_minutes:g} min limit",
                )
                continue
            if now - host.parked_at >= limit:
                await self.destroy(
                    host, f"parked longer than the {self.rented.teardown.max_park_hours}h limit"
                )

    # --- adopting what a previous supervisor rented (spec §1.2) ---

    def published_ref(self, host: RentedHost) -> dict:
        """Everything a successor needs to pick this host up: no secret, only facts."""
        return {
            "instance_id": host.instance.instance_id,
            "machine_id": host.offer.machine_id,
            "bid_hourly": host.bid_hourly,
            "created_at": host.created_at,
            "ready_at": host.ready_at,
            "prepared": host.prepared,
            "hold_until": host.hold_until,
            "when_ready": host.when_ready,
            "download_cost": host.download_cost,
            "workers": host.workers,
            "interruptible": host.interruptible,
            "engine_seen_at": host.engine_seen_at,
            "state": host.state,
            "parked_at": host.parked_at,
            "idle_since": host.idle_since,
            "launch_workers": host.launch_workers,
            "engine": host.engine,
            "engine_port": host.engine_port,
            # What it was bought to hold. Left out once, so a restarted supervisor read every
            # host bought for one model as holding the whole rented set.
            "models": list(host.models),
            "profile": host.profile,
            "builds": dict(host.builds),
            "cards_per_copy": host.cards_per_copy,
            "workload": host.workload,
            "disk_gb": host.disk_gb,
            # Beside the offer, not in it: a release from before connections builds the offer from
            # this dict and refuses a field it does not know — rolling back must still adopt (D129).
            "offer": {k: v for k, v in dataclasses.asdict(dataclasses.replace(host.offer, raw={})).items()
                      if k != "connection"},
            "connection": host.connection_name,
        }

    async def adopt(self, rows: list) -> Optional[list[str]]:
        """On start, list what the provider has under this pool's label, take back what the
        database says was intended, and leave the rest for the sweep.

        The provider is the source of truth for what exists; the database for what was
        intended. A restart is not a reason to destroy a good host — nor to keep billing for
        one that is gone.
        """
        # "Could not ask" is not "nothing there" (D44, D61). Dropping rows on it once told the
        # caller to forget a host — and the next supervisor, finding the instance with no record
        # of it, destroyed a healthy host as an orphan (seen live). Per connection (D129): one
        # that cannot be asked, or is no longer configured, holds back only its own hosts.
        by_connection: dict[str, list] = {}
        for row in rows:
            ref = row.provider_ref or {}
            named = ref.get("connection") or (ref.get("offer") or {}).get("connection") or self.legacy_connection
            by_connection.setdefault(self.resolve(named) or f"?{named}", []).append(row)
        pending: dict[str, list] = {}
        existing: dict[tuple[str, str], Instance] = {}
        answered = 0
        # Every configured connection is asked, rows or not: "nobody could be asked" must mean
        # exactly that, or a retry holding only one account's rows reads as the whole pool down.
        for name, provider in list(self.providers.items()):
            try:
                existing.update({(name, i.instance_id): i for i in await provider.list_instances(self.label_prefix)})
                answered += 1
            except Exception as exc:  # a plug-in's own failure, of any kind, stays its own (D129)
                log.warning("could not list %s's instances to adopt: %s", name, exc)
                if name in by_connection:
                    pending[name] = by_connection[name]
        for name, held in by_connection.items():
            if name.startswith("?"):
                pending[name[1:]] = held
        if rows and not answered:
            self.pending_adoption = pending
            return None  # nothing could be asked: every record kept, and the pass waits (D61)
        for name, held in pending.items():
            # Said when it changes, not every pass (D82).
            ids = tuple(sorted(r.host_id for r in held))
            if self._said_pending.get(name) == ids:
                continue
            self._said_pending[name] = ids
            gone = name not in self.providers
            self.events.record(
                "adoption_deferred",
                (f"{len(held)} rented host(s) are on provider connection {name!r}, which is no longer configured: "
                 "their records are kept, and nothing is done to them until it is again") if gone else
                (f"provider connection {name!r} could not be asked about {len(held)} rented host(s); their records "
                 "are kept and it is not swept until it can be — the other connections carry on"),
                numbers={"hosts": list(ids), "connection": name},
            )
        for name in [n for n in self._said_pending if n not in pending]:
            del self._said_pending[name]
        self.pending_adoption = pending
        held_back = {r.host_id for held in pending.values() for r in held}
        rows = [row for row in rows if row.host_id not in held_back]

        adopted: list[str] = []
        for row in rows:
            ref = row.provider_ref or {}
            instance_id = ref.get("instance_id")
            if not instance_id or "offer" not in ref:
                continue
            row_connection = self.resolve(
                ref.get("connection") or (ref.get("offer") or {}).get("connection") or self.legacy_connection) or ""
            instance = existing.get((row_connection, instance_id))
            if instance is None:
                self.events.record(
                    "host_gone",
                    f"{row.host_id} was intended but the provider no longer has "
                    f"{instance_id}; dropped from the table",
                    numbers={"instance": instance_id},
                    host_id=row.host_id,
                    lease_id=row.lease_id,
                )
                continue

            host = RentedHost(
                host_id=row.host_id,
                instance=instance,
                offer=Offer(**{**{k: v for k, v in ref["offer"].items() if k != "connection"},
                               "connection": row_connection}),
                bid_hourly=float(ref.get("bid_hourly") or row.hourly_rate or 0.0),
                lease_id=row.lease_id or "",
                created_at=float(ref.get("created_at") or time.time()),
                state=ref.get("state") if ref.get("state") in ("parked", "ready", "preparing") else "preparing",
                ready_at=ref.get("ready_at"),
                prepared=bool(ref.get("prepared")),
                hold_until=ref.get("hold_until"),
                models=tuple(ref.get("models") or ()),
                profile=ref.get("profile"),
                builds={str(k): str(v) for k, v in (ref.get("builds") or {}).items()},
                # A row from before splitting existed ran a copy on every card.
                cards_per_copy=int(ref.get("cards_per_copy") or 1),
                workload=ref.get("workload"),
                disk_gb=ref.get("disk_gb"),
                when_ready=ref.get("when_ready") or "join",
                download_cost=float(ref.get("download_cost") or 0.0),
                parked_at=ref.get("parked_at"),
                idle_since=ref.get("idle_since"),
                launch_workers=int(ref.get("launch_workers") or 0),
                # What its engine was launched with. A row from before profiles existed was
                # launched with the rented default of the day.
                workers=int(ref.get("workers") or self.rented.workers),
                interruptible=bool(ref.get("interruptible", True)),
                engine_seen_at=ref.get("engine_seen_at"),
                engine=str(ref.get("engine") or ""),
                engine_port=int(ref.get("engine_port") or 0),
            )
            if host.idle_since is not None:
                # Paused for idleness: load, not the lease, brings it back.
                self.unit(host.workload, self.group_of_host(host)).idle_gate = True
            was_ready = host.state == "ready"
            if was_ready:
                # Readiness is re-verified, never assumed — and the clock on "not ready in
                # time" starts now, not when the host was created (D50).
                host.state = "adopting"
            host.mark_preparing()
            try:
                host.connection = await self.provider_for(row_connection).connection(instance)
            except ProviderError as exc:
                log.warning("no connection details for %s yet: %s", row.host_id, exc)
            if host.state != "parked" and host.connection is not None:
                host.dial_url = host.connection.public_url or await self._open_tunnel(
                    row.host_id, host.connection, self.port_of(host)
                )
            if was_ready and self.forward_survived(row.host_id):
                # Its forward outlived the last supervisor and is still up on the same port, so
                # the router never stopped reaching it (D110). It keeps `ready` rather than
                # leaving routing to be re-verified: the pass's probe checks it now as it checks
                # every host every pass, and takes it out if it no longer holds its models.
                host.state = "ready"
            self.hosts[row.host_id] = host
            adopted.append(row.host_id)
            self.events.record(
                "adopted",
                f"{row.host_id} taken back after a restart: {instance_id} on "
                f"{host.offer.machine_id} at ${host.bid_hourly:.3f}/h, held "
                f"{host.hours_held:.2f}h so far",
                numbers={"instance": instance_id, "bid": host.bid_hourly, "created_at": host.created_at},
                host_id=row.host_id,
                lease_id=host.lease_id,
            )
        return adopted

    async def _left_nothing_behind(self, label: str, lease: Lease, connection: Optional[str] = None) -> bool:
        """After a bid the provider reported lost: is anything running under its label?

        Returns False when something was left and could not be ended, or when the provider
        could not be asked at all — either way the pool stops bidding rather than stacking a
        second machine on top of one that may be billing.
        """
        try:
            stray = await self.provider_for(connection or self.connection).list_instances(label)
        except ProviderError as exc:
            self.events.record(
                "bid_unverifiable",
                f"a bid failed and the provider could not be asked what it left behind ({exc}); "
                "no further bids this pass",
                numbers={"label": label},
                lease_id=lease.lease_id,
            )
            return False
        if not stray:
            return True

        ended = []
        for instance in stray:
            self.events.record(
                "bid_left_an_instance",
                f"the bid reported lost had in fact created {instance.instance_id}; destroying it",
                numbers={"instance": instance.instance_id, "label": label},
                lease_id=lease.lease_id,
            )
            if await self._destroy_instance(instance, "left behind by a failed bid", connection):
                ended.append(instance.instance_id)
        if len(ended) == len(stray):
            return True
        self.events.record(
            "bid_left_an_instance_undestroyed",
            "an instance left by a failed bid could not be destroyed; no further bids this pass",
            numbers={"label": label},
            lease_id=lease.lease_id,
        )
        return False

    # --- release what should not exist ---

    async def sweep_orphans(self) -> None:
        """The provider is the source of truth for what exists; the database for what was
        intended. Anything carrying this pool's label that we did not intend is swept."""
        known = {(self.resolve(host.connection_name), host.instance.instance_id)
                 for host in self.hosts.values() if not host.released}
        for name, provider in list(self.providers.items()):
            if name in self.pending_adoption:
                continue  # its hosts are not taken back yet: what it lists may be ours
            try:
                instances = await provider.list_instances(self.label_prefix)
            except ProviderError as exc:
                log.warning("could not list %s's instances: %s", name, exc)
                continue
            for instance in instances:
                if (name, instance.instance_id) in known:
                    continue
                self.events.record(
                    "orphan_swept",
                    f"instance {instance.instance_id} carries this pool's label but was never "
                    "intended; destroying so it stops billing",
                    numbers={"instance": instance.instance_id, "label": instance.label, "connection": name},
                )
                await self._destroy_instance(instance, "orphan", name)
        await self.sweep_volumes()

    async def sweep_volumes(self) -> None:
        """A volume under the pool's label whose workload is over, or that the pool has no record
        of, is deleted (D116). A provider that cannot be asked leaves every volume where it is:
        "could not list" is never "none there" (D61)."""
        live = {(self.resolve(v.connection or self.legacy_connection) or v.connection, v.volume_id): v
                for v in self.workload_store.volumes()}
        for name, provider in list(self.providers.items()):
            if not provider.capabilities.volumes:
                continue
            try:
                found = await provider.list_volumes(self.label_prefix)
            except ProviderError as exc:
                log.warning("could not list %s's volumes: %s", name, exc)
                continue
            settings = self.rented.providers.get(name)
            attached = {h.volume_id for h in self.hosts.values() if h.volume_id}
            for volume in found:
                record = live.get((name, volume.volume_id))
                if record is not None and record.workload in self.workloads:
                    if (record.location is None or volume.volume_id in attached or settings is None
                            or settings.keep_models):
                        continue
                    # Keeping models was turned off for this account: each of its model volumes
                    # goes once no host has it (D139).
                    await self._delete_volume(volume.volume_id, f"keeping models between hosts is off for {name}", name)
                    continue
                await self._delete_volume(volume.volume_id, "its workload is over" if record else "never recorded by this pool",
                                          name)

    async def delete_volumes(self, workload: str) -> None:
        """Every volume of a workload that is over."""
        for volume in self.workload_store.volumes(workload):
            await self._delete_volume(volume.volume_id, f"workload {workload} is over", volume.connection or None)

    async def _delete_volume(self, volume_id: str, why: str, connection: Optional[str] = None) -> None:
        try:
            provider = self.provider_for(connection or self.legacy_connection)
            await provider.delete_volume(volume_id)
            remaining = {v.volume_id for v in await provider.list_volumes(self.label_prefix)}
        except ProviderError as exc:
            log.warning("could not delete volume %s: %s", volume_id, exc)
            return
        if volume_id in remaining:
            return  # asked again next pass; a delete counts only once the listing agrees
        self.workload_store.volume_deleted(volume_id)
        self.events.record("volume_deleted", f"volume {volume_id} deleted and verified: {why}",
                           numbers={"volume": volume_id})

    # --- recover what is broken ---

    async def handle_evictions(self) -> None:
        for host in list(self.hosts.values()):
            if host.released:
                continue
            try:
                status = await self.provider_for(host).status(host.instance)
            except ProviderError as exc:
                log.warning("status for %s failed: %s", host.host_id, exc)
                continue
            if await self._spot_moved(host, status):
                continue
            if await self._past_its_timerless_hours(host):
                continue

            if status.state == InstanceState.GONE:
                self.events.record(
                    "host_gone",
                    f"{host.host_id} no longer exists at the provider",
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
                host.released = True
                self.hosts.pop(host.host_id, None)
                self._end_prepare_lease(host, "no longer exists at the provider")
                continue

            if (
                status.startup_material is False
                and host.state != "ready"
                and self.deadman_onstart(connection=host.connection_name) is not None
            ):
                # Created with a start-up script and reported back without one (seen live): no
                # dead-man timer, no key for the pool, so it can never join and nothing on it
                # would stop it billing. There is no way in to repair it. End it now (D65).
                self.avoid(host.offer.machine_id, "came up without its start-up material")
                self.events.record(
                    "host_without_startup",
                    f"{host.host_id} came up without the start-up material it was created with: "
                    "no dead-man timer and no way in for the pool; ending it",
                    numbers={"machine": host.offer.machine_id},
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
                await self.destroy(host, "came up without its start-up material")
                continue

            if status.state != InstanceState.STOPPED:
                if status.state == InstanceState.RUNNING:
                    host.rebid_at = None  # the start the last re-bid asked for has landed
                continue
            if host.state == "parked":
                # Stopped because the pool parked it. Read as an eviction, it would be re-bid
                # and restarted the moment it was parked (D64).
                continue
            if host.rebid_at is not None and time.time() - host.rebid_at < self.REBID_GRACE_S:
                # Stopped because the provider has not yet restarted it after the pool's own
                # re-bid — one pass is not long enough. Found live: sixteen seconds after a
                # re-bid the instance still read "stopped", the pool judged a second eviction,
                # took the machine's floor — now its own bid, as the top bidder — added the
                # premium, and paid ten cents an hour more to outbid itself (D106).
                continue

            if not host.interruptible:
                # Nobody outbid this one — it is not that kind of rental. Something else
                # stopped it, so there is nothing to re-bid: give it up and say so (D52).
                self.events.record(
                    "host_stopped",
                    f"{host.host_id} stopped, and it was rented on demand so it was not outbid; releasing it",
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
                await self.destroy(host, "an on-demand host stopped without being asked to")
                continue

            # Stopped, and the pool never asked for that: an eviction (spec §1.2).
            lease = self.leases.get(host.lease_id)
            if lease is None or not lease.is_open:
                await self.destroy(host, "evicted with no open lease")
                continue

            if host.rebid_at is not None:
                # The pool re-bid, the grace has run out, and the machine is still not ours: the
                # bid did not win it back. Let it go rather than bid again and again (D109).
                self.events.record(
                    "rebid_lost",
                    f"{host.host_id}: the re-bid at ${host.bid_hourly:.3f}/h did not win "
                    f"{host.offer.machine_id} back within {self.REBID_GRACE_S:.0f}s; releasing it",
                    numbers={"bid": host.bid_hourly, "machine": host.offer.machine_id},
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
                await self.destroy(host, "outbid, and the re-bid did not win it back")
                continue

            offers = await self._offers()
            same_machine = next((o for o in offers if self._at(o) == self._at(host.offer) and o.bidding), None)
            if not host.offer.bidding:
                same_machine = None  # a spot host taken back is replaced: there is no bid to win it with
            elif same_machine is None and host.interruptible:
                # Outbid means someone else holds it now, so the search does not list it: ask
                # for this machine's own price (D109).
                same_machine = await self._machine_price(host)
            alternatives = [o for o in offers if self._at(o) != self._at(host.offer)]
            best_alternative = alternatives[0] if alternatives else None

            decision = decide_eviction(
                HostView(
                    host_id=host.host_id,
                    priority=20,
                    busy_workers=0,
                    total_workers=host.workers,
                    idle_seconds=0,
                    bid_hourly=host.bid_hourly,
                    machine_id=host.offer.machine_id,
                    hours_held=host.hours_held,
                ),
                same_machine,
                best_alternative,
                self._lease_view(lease),
                self.rented.bidding,
                self.rented.policy_in_force,
            )
            # Say what happens next, not the strategy's label for it. Found live: "was outbid:
            # replace … replacing on 43532" on a prepared host, whose lease closed with it a
            # second later and rented nothing.
            capped = (
                self._cap_bid(decision.bid, lease, same_machine)
                if decision.action == "rebid" and decision.bid is not None and same_machine is not None
                else None
            )
            over_cap = self._refuse_rebid(host, capped, lease) if capped is not None else None
            if over_cap is not None:
                capped = None
            if capped is not None:
                outcome = f"re-bidding ${capped:.3f}/h to win it back"
            elif decision.action == "rebid":
                outcome = f"released; the re-bid would cross a ceiling{f' ({over_cap})' if over_cap else ''}"
            elif host.prepared and not any(
                h.lease_id == host.lease_id and h is not host and not h.released for h in self.hosts.values()
            ):
                outcome = ("released; it was a prepared host, so its lease closes with it and nothing "
                           "is rented in its place")
            else:
                # The lease stays open, as standing demand; whether it rents again is the
                # ordinary acquire path's to decide, not this one's (spec §6.2).
                outcome = "released; its lease rents a replacement if it still needs the capacity"
            self.events.record(
                "eviction",
                f"{host.host_id} was outbid: {outcome}",
                numbers={"action": decision.action, "reasons": decision.reasons, "bid": decision.bid},
                host_id=host.host_id,
                lease_id=lease.lease_id,
            )

            if decision.action == "rebid":
                if capped is None:
                    await self.destroy(host, "re-bid would cross a ceiling")
                    continue
                try:
                    provider = self.provider_for(host)
                    await provider.set_bid(host.instance, capped)
                    # Billed at the new bid from here, whether or not the start below goes through.
                    host.bid_hourly = capped
                    await provider.start(host.instance)
                    host.mark_scheduling()
                    host.rebid_at = time.time()
                except ProviderError as exc:
                    log.warning("re-bid for %s failed: %s", host.host_id, exc)
            else:
                await self.destroy(host, f"evicted; {decision.action}")

    async def _machine_price(self, host: "RentedHost") -> Optional[Offer]:
        """The current bid listing for the machine this host was rented on, asked of the machine
        itself — or None where the provider cannot say, which releases rather than guesses."""
        if not host.offer.bidding:
            return None  # a spot price is not won back with a bid
        try:
            finder = getattr(self.provider_for(host), "offer_for_machine", None)
        except ProviderError:
            return None
        if finder is None:
            return None
        try:
            offer = await self._counted(finder(host.offer.machine_id, host.offer.gpus), host.connection_name)
        except ProviderError as exc:
            log.warning("the price of %s could not be read: %s", host.offer.machine_id, exc)
            return None
        if offer is None:
            return None
        return dataclasses.replace(offer.priced_for(host.disk_gb or self.rented.disk_gb), connection=host.connection_name)

    # --- acquire what is missing ---

    async def acquire(
        self,
        open_leases: list[Lease],
        ready_workers_higher_tiers: int,
        pressure: bool = False,
        load: Optional[Load] = None,
        workload: Optional[str] = None,
        group: Optional[tuple[str, ...]] = None,
    ) -> None:
        unit = self.unit(workload, group)
        lease = self.lease_of(open_leases, workload)
        if workload is not None:
            # A workload always sizes from load, above a floor of the hosts it starts with: its
            # lease reserved them for the run (D115). Nothing measured is no load, not no floor.
            # Each group of its hosts on its own (D118).
            chosen = self.group_spec(workload, unit.group)
            if lease is None or chosen is None:
                return
            self.widen_while_short(workload, chosen)
            await self._acquire_dynamically(
                lease, 0, load or Load(busy_workers=0, ready_workers=0, waiting=0), unit=unit,
                floor=chosen.hosts_at_start, per_host=chosen.workers_per_host,
                ceiling=self.group_ceiling(lease, workload, unit.group),
            )
            return
        if self.rented.allocation == "dynamic" and lease is not None:
            await self._acquire_dynamically(lease, ready_workers_higher_tiers, load, unit=unit)
            return
        if unit.idle_gate:
            # A host was given up because nothing was using it. The lease says what may be
            # spent, not that it must be: capacity comes back when load asks for it (D64).
            if not pressure:
                unit.overflow_since = None
                return
            unit.idle_gate = False
            if lease is not None and self._refuse_for_caps_quietly(lease) is None:
                back = await self.restart_idle_parked(lease, workload)
                if back is not None:
                    return
        mine = self.hosts_of(workload)
        rented_workers = sum(h.workers for h in mine if h.state == "ready")
        pending = sum(1 for h in mine if h.state in ("scheduling", "preparing"))
        wanted = lease.workers if lease else 0
        overflow = max(0, wanted - ready_workers_higher_tiers - rented_workers)

        now = time.time()
        if overflow > 0 and unit.overflow_since is None:
            unit.overflow_since = now
        elif overflow <= 0:
            unit.overflow_since = None

        demand = Demand(
            wanted_workers=wanted,
            ready_workers_higher_tiers=ready_workers_higher_tiers,
            rented_workers=rented_workers,
            overflow_age_s=(now - unit.overflow_since) if unit.overflow_since else 0.0,
            hosts_pending=pending,
            rented_hosts=len([h for h in self.hosts.values() if not h.released]),
        )

        decision = decide_rent(
            demand,
            self._lease_view(lease) if lease else None,
            self.rented.scale,
            self.rented.workers,
        )
        if not decision.rent:
            return

        # Every cap is re-checked here, after the strategy has had its say.
        refusal = self._refuse_for_caps(demand, lease)
        if refusal is not None:
            self.events.record(
                "rent_refused", refusal, numbers={"reasons": decision.reasons},
                lease_id=lease.lease_id if lease else None,
            )
            return

        assert lease is not None
        await self.rent_one(lease, decision.reasons)

    async def _acquire_dynamically(
        self, lease: Lease, ready_workers_higher_tiers: int, load: Optional[Load],
        unit: Optional[Unit] = None, floor: Optional[int] = None, per_host: Optional[int] = None,
        ceiling: Optional[int] = None,
    ) -> None:
        """Hosts added from measured load, in rounds that grow while it lasts (D66).

        The lease is still the only spending authority, and still the ceiling: its `workers` is
        the most this pool may reach, its dollars and hours the most it may spend. What it is
        no longer is the *demand* — that is what the traffic asks for.
        """
        now = time.time()
        if load is None:
            return
        unit = unit or self.unit(None)
        live = [h for h in self.hosts_of(unit.workload, unit.group) if h.state != "draining"]
        ready = [h for h in live if h.state == "ready"]
        pending = [h for h in live if h.state in ("scheduling", "preparing")]
        rented_workers = sum(h.workers for h in ready)

        wanted = min(lease.workers if ceiling is None else ceiling, load.wanted(self.rented.dynamic.target_utilisation))
        least = self.rented.dynamic.min_hosts if floor is None else floor
        below_floor = len(live) < least
        overflow = max(0, wanted - ready_workers_higher_tiers - rented_workers)

        if overflow > 0 and unit.overflow_since is None:
            unit.overflow_since = now
        elif overflow <= 0 and not below_floor:
            unit.overflow_since = None
            if unit.ramp_round:
                # The load that started this ramp is gone: the next one starts from one again.
                self.events.record(
                    "ramp_reset",
                    "the load has cleared; the ramp starts from one host again",
                    numbers={"was": unit.ramp_round, "workload": unit.workload},
                    lease_id=lease.lease_id,
                )
                unit.ramp_round = 0
            return

        held = (now - unit.overflow_since) if unit.overflow_since else 0.0
        if not below_floor and held < self.rented.dynamic.window_s:
            return  # a burst shorter than the window is not worth a model download

        if not live and unit.ramp_round:
            # Every host the ramp bought is gone — evicted, or given up. There is no new
            # capacity to wait and see about, so the back-off has nothing to measure: the ramp
            # starts again from one, at once. Seen live: a host was outbid three minutes after
            # it was rented, and the pool then sat out its whole back-off while every request
            # was refused.
            unit.ramp_round = 0
            unit.ramp_landed_at = 0.0
        elif pending:
            unit.ramp_landed_at = 0.0  # the round has not landed while a host is still coming
        elif unit.ramp_round and not unit.ramp_landed_at:
            unit.ramp_landed_at = now

        # A paused host is capacity the pool already has: waking one adds nothing to the host
        # count, costs no download, and is the right answer before renting anything. It has to
        # be tried *before* the caps are consulted, or a pool sitting at its host limit with
        # every host paused refuses the load it could serve at once (found by the simulation).
        woken = await self.restart_idle_parked(lease, unit.workload, unit.group)
        if woken is not None:
            return

        if unit.budget_bound is not None:
            if unit.budget_bound == self._budget_situation(lease, unit.workload):
                return  # its budget carried no offer, and nothing has changed since (D126)
            unit.budget_bound = None

        if below_floor and unit.workload is not None:
            # A workload below the hosts it starts with: exactly the gap, each host re-checked on
            # its own, whatever the ramp would say — the ramp is for load above the floor (D115).
            await self._rent_round(
                lease, unit, least - len(live), [f"workload {unit.workload}: {len(live)} of the "
                                                   f"{least} host(s) it runs on at least"],
                wanted=wanted, rented_workers=rented_workers,
                ready_workers_higher_tiers=ready_workers_higher_tiers, held=held, load=load, ramp=False,
            )
            return

        ramp = decide_ramp(
            round_size=unit.ramp_round,
            load_present=load.present or below_floor,
            previous_round_landed=not pending,
            since_last_round_s=(now - unit.ramp_landed_at) if unit.ramp_landed_at else 0.0,
            hosts_pending=len(pending),
            cfg=self.rented.dynamic,
        )
        if ramp.hosts <= 0:
            return

        # Never more than the gap itself asks for: a ramp is a rate, not a target.
        by_overflow = max(1, math.ceil(overflow / max(1, per_host or self.rented.workers)))
        asked = max(1, min(ramp.hosts, by_overflow)) if not below_floor else ramp.hosts
        await self._rent_round(
            lease, unit, asked, [f"dynamic allocation: {ramp.reasons[0]}"],
            wanted=wanted, rented_workers=rented_workers,
            ready_workers_higher_tiers=ready_workers_higher_tiers, held=held, load=load, ramp=True,
        )

    async def _rent_round(
        self, lease: Lease, unit: Unit, asked: int, reasons: list[str], *, wanted: int, rented_workers: int,
        ready_workers_higher_tiers: int, held: float, load: Load, ramp: bool,
    ) -> None:
        rented_now = 0
        for _ in range(asked):
            demand = Demand(
                wanted_workers=wanted,
                ready_workers_higher_tiers=ready_workers_higher_tiers,
                rented_workers=rented_workers,
                overflow_age_s=held,
                hosts_pending=0,  # the ramp decides how many at once, not the one-at-a-time rule
                rented_hosts=len([h for h in self.hosts.values() if not h.released]),
            )
            # Every host in a round is re-checked on its own: a round is a number of attempts,
            # never a bulk purchase (D66).
            refusal = self._refuse_for_caps(demand, lease, group=unit.group)
            if refusal is not None:
                self.events.record(
                    "rent_refused", refusal,
                    numbers={"round": asked, "rented_so_far": rented_now, "workload": unit.workload},
                    lease_id=lease.lease_id,
                )
                break
            host = await self.rent_one(lease, list(reasons), workload=unit.workload, group=unit.group)
            if host is None:
                if self.last_budget_bound and unit.workload is not None:
                    # Searching again finds the same offers beyond the same budget, and every
                    # search spends the provider's daily quota: seen live, ten a minute for a
                    # host no offer could be afforded for (D126).
                    unit.budget_bound = self._budget_situation(lease, unit.workload)
                    live = len([h for h in self.hosts_of(unit.workload, unit.group) if h.state != "draining"])
                    self.events.record(
                        "workload_budget_bound",
                        f"workload {unit.workload}: its budget carries no further host; it runs on the "
                        f"{live} it has, and looks again when a host leaves or its lease is extended",
                        numbers={"workload": unit.workload, "hosts": live, "last_refusal": self.last_refusal},
                        lease_id=lease.lease_id,
                    )
                break  # a round that loses its bids does not grow the next one
            rented_now += 1

        if rented_now:
            # Said once a round, and only when a round actually bought something: a pass that
            # decides to rent and is then refused by a cap has already said so.
            self.events.record(
                "ramp_round",
                f"{reasons[0].removeprefix('dynamic allocation: ')}; rented {rented_now} of {asked}",
                numbers={
                    "round": asked,
                    "rented": rented_now,
                    "wanted_workers": wanted,
                    "ready_workers": rented_workers + ready_workers_higher_tiers,
                    "waiting": load.waiting,
                    "busy": load.busy_workers,
                    "workload": unit.workload,
                },
                lease_id=lease.lease_id,
            )
            if ramp:
                unit.ramp_round = asked
            unit.ramp_landed_at = 0.0

    def claim_records(self) -> None:
        """Fix what a record from before connections belongs to, and carry the search use counted
        before into this connection's counter (D129, D121). Called under the pool lock: a second
        supervisor started on the same database, which then exits on the lock, must not decide it."""
        db = self.leases.db
        self.legacy_connection = legacy_connection(db, self.connection)
        for row in db.query("SELECT key, value FROM pool_meta WHERE key LIKE 'connection_type:%'"):
            name, was = str(row["key"]).split(":", 1)[1], str(row["value"])
            now = self.rented.providers[name].type if name in self.rented.providers else None
            if now is not None and now != was:
                # Its records — hosts, volumes, spend — are that provider's: read as another's, the
                # pool would act on them at the wrong provider.
                raise ConnectionNameReused(
                    f"provider connection {name!r} was {was!r} and is now configured as {now!r}: give the new "
                    f"provider another name — {name!r} names {was!r}'s records (D133)")
            self.connection_types.setdefault(name, was)
        if self.legacy_connection not in self.connection_types:
            # The first pool with connections had only this one; its records are its provider's.
            self.connection_types[self.legacy_connection] = self.rented.providers[self.connection].type
        for name, conn in self.rented.providers.items():
            db.execute("INSERT OR REPLACE INTO pool_meta (key, value) VALUES (?, ?)", (f"connection_type:{name}", conn.type))
        db.execute("INSERT OR IGNORE INTO pool_meta (key, value) VALUES (?, ?)",
                   (f"connection_type:{self.legacy_connection}", self.connection_types[self.legacy_connection]))
        db.execute(
            "INSERT OR IGNORE INTO provider_search_usage (connection, day, rows, quota, exhausted_at, resets_at, "
            "updated_at) SELECT ?, day, rows, quota, exhausted_at, resets_at, updated_at FROM search_usage",
            (self.legacy_connection,))

    def add_connection(self, name: str, type_name: str, provider: Provider) -> None:
        """A connection added or rebuilt while the supervisor runs (D135): searched from the next
        pass. A name once another provider's is refused, as at a start (D133)."""
        was = self.connection_types.get(name)
        if was is not None and was != type_name:
            raise ConnectionNameReused(
                f"provider connection {name!r} names {was!r}'s records; give {type_name!r} another name (D133)")
        self.providers[name] = provider
        self.searching.setdefault(name, SearchState())
        self.connection_types[name] = type_name
        self.leases.db.execute("INSERT OR REPLACE INTO pool_meta (key, value) VALUES (?, ?)",
                               (f"connection_type:{name}", type_name))

    def held_at(self, name: str) -> list[tuple[str, str]]:
        """What the pool holds at a connection, as (record, provider's instance or volume id):
        live hosts, kept volumes and hosts held back at a restart. While any is held, the
        connection's provider is kept and its credential cannot be taken away."""
        # By provider, never by the file: one connection per provider (D133), so whatever name a
        # record carries — an old name, a rename's — it is held at that provider's connection,
        # and the answer does not change with a file the supervisor has only half applied.
        kind = self.connection_types.get(name)
        if kind is None:
            return []

        def here(named: Optional[str]) -> bool:
            return self.connection_types.get(named or self.legacy_connection) == kind

        held = [(h.host_id, h.instance.instance_id) for h in self.hosts.values()
                if not h.released and here(h.connection_name)]
        held += [(f"volume:{v.volume_id}", v.volume_id) for v in self.workload_store.volumes() if here(v.connection)]
        for pending_name, rows in self.pending_adoption.items():
            if here(pending_name):
                held += [(row.host_id, str((row.provider_ref or {}).get("instance_id") or "")) for row in rows]
        return held

    def widen_while_short(self, workload: str, chosen: Any) -> None:
        """A group of several models with fewer ready hosts than it planned runs more of each card
        (D128): its planned shares spread the models' targets over every planned host, so two of
        three hosts would serve two thirds of the target with a sixth of each card unused. Back
        to the plan when its hosts are. Never past what the engines were launched for."""
        if len(chosen.models) < 2:
            return
        spec = self.workloads.get(workload)
        live = [h for h in self.hosts_of(workload, chosen.key) if h.state != "draining"]
        ready = [h for h in live if h.state == "ready"]
        shares = dict(chosen.caps)
        if spec is not None and ready and len(ready) < chosen.hosts_at_start:
            per_model = {m: min(self.per_model_on(h.offer, spec, chosen.models, chosen.builds)[m] for h in ready)
                         for m in chosen.models}
            most = min(h.launch_workers or h.workers for h in ready)
            wider = workload_math.split_short({m: spec.target(m).parallel for m in chosen.models},
                                              per_model, len(ready), most)
            if wider is not None and sum(wider.values()) > sum(shares.values()):
                shares = wider
        if shares != chosen.shares:
            self.workload_store.set_caps_now(workload, chosen.key, {} if shares == chosen.caps else shares)
            fresh = self.workload_store.get(workload)
            if fresh is not None:
                self.workloads[workload] = fresh
            said = ", ".join(f"{m} {shares[m]}" for m in chosen.models)
            self.events.record(
                "workload_resplit",
                (f"workload {workload}: {len(ready)} of its {chosen.hosts_at_start} planned host(s) ready; each "
                 f"takes {said} at once until the rest are" if shares != chosen.caps
                 else f"workload {workload}: its planned hosts are ready; each takes {said} at once, as planned"),
                numbers={"workload": workload, "models": list(chosen.models), "shares": shares,
                         "planned": chosen.caps, "ready": len(ready)},
            )
        workers = sum(shares.values())
        for host in live:
            if host.workers != workers and workers <= (host.launch_workers or host.workers):
                host.workers = workers

    def _budget_situation(self, lease: Lease, workload: Optional[str]) -> tuple:
        """What a workload's budget for another host depends on, but for the market: its hosts
        and its lease's dollars and hours (D126)."""
        hosts = tuple(sorted(h.host_id for h in self.hosts_of(workload) if h.state != "draining"))
        return (hosts, lease.max_spend, lease.max_hours)

    def _refuse_for_caps_quietly(self, lease: Lease) -> Optional[str]:
        """The dollar check alone: a parked host is already counted among the pool's hosts."""
        left = self.budget_left(lease)
        return None if left > 0 else f"the lease has ${left:.4f} left"

    async def restart_idle_parked(self, lease: Lease, workload: Optional[str] = None,
                                  group: Optional[tuple[str, ...]] = None) -> Optional[RentedHost]:
        """Load came back while a host was paused: it returns at once, with no download and no
        wait for the scale-up window, which exists to stop a burst buying a download."""
        if not any(h.state == "parked" and h.idle_since is not None for h in self.hosts_of(workload, group)):
            return None
        return await self.restart_parked(lease, workload, group)

    def reserved_hosts(self, besides: Optional[str] = None, group: Optional[tuple[str, ...]] = None) -> int:
        """Hosts workloads still mean to rent to reach the start they were created with: room
        another demand may not take (workloads.md §8). Counted per group (D118): one group above
        its floor hides nothing of another still below its own. The shared workload yields to
        every workload's; a workload yields to those created before it, and they to none of its —
        the room is taken in the order the workloads were opened, never held back both ways. Within
        one workload the same holds in the order of its groups: a group yields to those listed
        before it, never they to it — two groups yielding to each other would both wait for ever."""
        mine = self.workloads.get(besides) if besides is not None else None
        if mine is not None and group is None:
            group = self.first_group(besides)
        my_index = next((i for i, g in enumerate(mine.groups) if g.key == group), 0) if mine is not None else 0
        total = 0
        for name, spec in self.workloads.items():
            if spec.state not in ("preparing", "serving"):
                continue
            if mine is not None and name != besides and not spec.created_at < mine.created_at:
                continue
            for index, g in enumerate(spec.groups):
                if name == besides and index >= my_index:
                    continue
                total += max(0, g.hosts_at_start - len(self.hosts_of(name, g.key)))
        return total

    def volume_burn(self) -> float:
        """What the pool's live volumes cost per hour (D116): spend the hosts' prices do not show."""
        return sum(v.hourly for v in self.workload_store.volumes())

    def _refuse_for_caps(self, demand: Demand, lease: Optional[Lease],
                         group: Optional[tuple[str, ...]] = None) -> Optional[str]:
        if lease is None:
            return "no lease is open"
        mine = self.held_back_rows(lease.workload)
        if mine:
            # Its hosts there may still be serving and billing: one rented in their place would
            # bill beside them until the provider answers.
            return (f"{len(mine)} of its hosts are on a provider not yet asked since the restart; "
                    "nothing is rented in their place until it answers")
        live = [h for h in self.hosts.values() if not h.released]
        held = len(self.held_back_rows())
        reserved = self.reserved_hosts(besides=lease.workload, group=group)
        if len(live) + held + reserved >= self.config.limits.max_rented_hosts:
            return (
                f"{len(live) + held} rented hosts already"
                + (f" ({held} on a provider not yet asked)" if held else "")
                + (f", and {reserved} reserved for workloads still starting" if reserved else "")
                + f", at the pool's limit of {self.config.limits.max_rented_hosts}"
            )
        burn = sum(h.bid_hourly for h in live) + self.volume_burn() + self.held_back_burn()
        total = self.config.limits.max_hourly_burn
        if total is not None and burn >= total:
            return f"hourly burn ${burn:.3f} is at the ${total:.2f} cap"
        left = self.budget_left(lease)
        if left <= 0:
            return f"the lease has ${left:.4f} left once the safety margin is taken off"
        return None

    def worst_case_hourly(self) -> float:
        """The most the rented hosts can burn per hour: every host at the all-in maximum, as
        many hosts as the pool may hold — or the overall cap, if one is set below that (D46)."""
        bound = self.config.limits.max_rented_hosts * self.rented.max_all_in_hourly
        total = self.config.limits.max_hourly_burn
        return bound if total is None else min(bound, total)

    def _refuse_for_workload_budget(self, offer: Offer, bid: float, lease: Lease, workload: Optional[str],
                                    also_hourly: float = 0.0) -> Optional[str]:
        """A workload's host it cannot pay for to the end of its hours (D124): what the
        workload's hosts already burn, with this one at its price, over the hours the lease has
        left, beyond what the budget has left. Seen live: a $5, 1.75-hour workload fell back to a
        $7.17/h on-demand host, which would have spent the budget in forty minutes. A model volume
        this host would make bills too, to the end, and is counted with it (D139)."""
        hourly = ((bid + offer.storage_hourly) if offer.interruptible else offer.all_in_hourly) + also_hourly
        burning = sum(h.bid_hourly for h in self.hosts_of(workload) if h.state != "draining")
        hours = lease.hours_left()
        # The dollars left, not the margin-reduced figure the cap is enforced on: the margin
        # stops a lease early, it says nothing about whether a host fits the budget.
        left = lease.max_spend - self.lease_spend(lease)[0]
        if (burning + hourly) * hours > left:
            return (f"{offer.machine_id} at ${hourly:.3f}/h: with its hosts' ${burning:.3f}/h, "
                    f"${(burning + hourly) * hours:.2f} over the {hours:.2f}h left, more than the ${left:.2f} its budget has left")
        return None

    def _refuse_bid_for_burn(self, bid: float) -> Optional[str]:
        """The cap applies to the burn this bid *would* create, not only to today's. With no
        overall cap set there is nothing to refuse here: the bid is already clamped to the
        per-host ceiling, and the host count to the pool's limit (D46)."""
        if self.config.limits.max_hourly_burn is None:
            return None
        burn = (sum(h.bid_hourly for h in self.hosts.values() if not h.released) + self.volume_burn()
                + self.held_back_burn() + bid)
        if burn > self.config.limits.max_hourly_burn:
            return (
                f"bidding ${bid:.3f}/h would take the burn to ${burn:.3f}/h, above the "
                f"${self.config.limits.max_hourly_burn:.2f} cap"
            )
        return None

    def _refuse_rebid(self, host: "RentedHost", bid: float, lease: Lease) -> Optional[str]:
        """A new bid on a host the pool already holds — won back after an eviction, or restarted
        from parked — is spending like any other: the pool's burn with this host at its new
        price, and the lease's budget, are checked as they are for a new rental."""
        left = self.budget_left(lease)
        if left <= 0:
            return f"the lease has ${left:.4f} left"
        cap = self.config.limits.max_hourly_burn
        if cap is None:
            return None
        others = sum(h.bid_hourly for h in self.hosts.values() if not h.released and h is not host)
        burn = others + self.volume_burn() + self.held_back_burn() + bid
        if burn > cap:
            return f"bidding ${bid:.3f}/h on {host.host_id} would take the burn to ${burn:.3f}/h, above the ${cap:.2f} cap"
        return None

    async def rent_one(
        self, lease: Lease, reasons: list[str],
        offer_id: Optional[str] = None, kind: Optional[str] = None,
        workload: Optional[str] = None, group: Optional[tuple[str, ...]] = None,
        connection: Optional[str] = None,
    ) -> Optional[RentedHost]:
        """Rent the best offer — or, when an operator named one, exactly that one.

        Naming an offer chooses *among* what the policy allows; it is not a way round it. The
        offer still has to pass every hard filter and every ceiling, and if it has gone, or no
        longer passes, nothing else is rented in its place: the operator asked for that host,
        not for a host (D55).
        """
        self.last_refusal = None
        self.last_budget_bound = False
        spec = self.workloads.get(workload) if workload is not None else None
        chosen_group = self.group_spec(workload, group) if spec is not None else None
        if spec is not None and chosen_group is None:
            return None
        if spec is not None:
            # A workload's host holds exactly its group's models, as the builds it was created
            # with (D115, D118).
            profile = None
            for_this_host: tuple[str, ...] = tuple(chosen_group.models)
            builds = dict(chosen_group.builds)
            cards_per_copy = chosen_group.cards_per_copy
            if kind is None and offer_id is None:
                kind = {"on_demand": "on_demand", "interruptible": "interruptible"}.get(spec.kind, "both")
        else:
            # Which profile and models this machine is bought for, decided before the search: what
            # it will hold sets the least card and disk worth looking at (D94, D111).
            profile = self.profile_for_new_host()
            for_this_host = self.models_for_new_host()
            builds = self.builds_for_new_host()
            cards_per_copy = self.rented.cards_per_copy(profile)
        policy = self.policy_for(for_this_host, builds, cards_per_copy=cards_per_copy)
        if spec is not None:
            policy = policy.model_copy(
                update={"min_reliability": max(policy.min_reliability, self.config.workloads.min_reliability)}
            )
        offers = await self._offers(policy, kinds=kind or ("both" if offer_id else None))
        # A connection with no dead-man timer serves only short leases (spec §7): for a longer
        # one, its offers are not among those this lease may rent (D129).
        too_long = {(o.connection, o.offer_id) for o in offers
                    if (self.lease_hours_on(o.connection) or float("inf")) < lease.max_hours}
        offers = [o for o in offers if (o.connection, o.offer_id) not in too_long]
        ranked, rejected = rank_offers(
            offers,
            self._policy_with_avoided(policy),
            self.rented.bidding,
            lease.hours_left(),
            self.model_set_gb(for_this_host, builds),
            history=self.machine_history(),
            history_cfg=self.rented.history,
        )
        if offer_id is not None:
            chosen = [pair for pair in ranked if pair[0].offer_id == offer_id
                      and (connection is None or pair[0].connection == connection)]
            if len({pair[0].connection for pair in chosen}) > 1:
                why = (f"offer {offer_id} is listed by more than one provider "
                       f"({', '.join(sorted({p[0].connection for p in chosen}))}); say which")
                self.last_refusal = why
                self.events.record("chosen_offer_unavailable", why, numbers={"offer": offer_id}, lease_id=lease.lease_id)
                return None
            stopped = next((name for name, oid in too_long if oid == offer_id and connection in (None, name)), None)
            if not chosen and stopped is not None:
                why = (f"{stopped} cannot arm a dead-man timer, so it serves leases of at most "
                       f"{self.lease_hours_on(stopped):g}h, and this one is {lease.max_hours:g}h")
                self.last_refusal = why
                self.events.record("chosen_offer_unavailable", why, numbers={"offer": offer_id}, lease_id=lease.lease_id)
                return None
            if not chosen:
                # Say exactly why, from the same filters — gone, or refused and by which rule.
                if offer_id in rejected:
                    why = f"that offer no longer passes: {'; '.join(rejected[offer_id])}"
                elif any(o.offer_id == offer_id for o in offers):
                    why = "that offer is no longer acceptable"
                else:
                    why = f"offer {offer_id} is gone from the market, or no longer passes the pool's filters; refresh and choose again"
                self.last_refusal = why
                self.events.record("chosen_offer_unavailable", why, numbers={"offer": offer_id}, lease_id=lease.lease_id)
                return None
            ranked = chosen  # that host, or nothing: never a substitute
        if not ranked:
            self.last_refusal = (
                f"{len(offers)} offers seen, none passed the policy" if offers
                else (self.last_offer_error or "the market returned no offers")
            )
            # A market that refuses everything refuses it again ten seconds later, and a
            # decision log filling with the same line is one an operator stops reading. Said
            # when it changes — a different count, or a different set of reasons (D82).
            # What the provider was asked to filter is part of what was said: a policy changed
            # to one the provider answers with nothing is news, though nothing came back twice.
            fingerprint = (len(offers), repr(sorted({r for rs in rejected.values() for r in rs})),
                           tuple(self.last_query_filters))
            if fingerprint != self._said_nothing_passed:
                self._said_nothing_passed = fingerprint
                asked = f" (asked of the provider: {'; '.join(self.last_query_filters)})" if self.last_query_filters else ""
                self.events.record(
                    "no_offer",
                    f"{len(offers)} offers seen, none passed the policy; staying paused rather "
                    "than relaxing a filter"
                    if offers
                    else (
                        f"the market could not be asked: {self.last_offer_error}"
                        if self.last_offer_error
                        else f"no offer matches the search{asked}"
                    ),
                    numbers={"seen": len(offers), "rejected": rejected, "asked_of_provider": self.last_query_filters},
                    lease_id=lease.lease_id,
                )
            return None

        # The market is offering something again, so the next dry spell is news once more.
        self._said_nothing_passed = None

        if chosen_group is not None and len(chosen_group.models) > 1:
            # Every host of a group of several models runs the same split (D118): a card that
            # cannot hold it within every model's target is not rented for it.
            fitting = [pair for pair in ranked if self.split_fits_on(pair[0], spec, chosen_group)]
            if not fitting:
                self.last_refusal = (f"none of {len(ranked)} accepted offers holds {', '.join(chosen_group.models)} "
                                     f"at {chosen_group.caps} within their targets")
                # Said when it changes, like "nothing passed the policy" (D82): a group stuck short
                # of its start must show in the decision log, not only in a field the next call
                # overwrites.
                said = (workload, chosen_group.key)
                if self._said_unfit.get(said) != len(ranked):
                    self._said_unfit[said] = len(ranked)
                    self.events.record("split_unfit", self.last_refusal,
                                       numbers={"workload": workload, "models": list(chosen_group.models),
                                                "caps": chosen_group.caps, "accepted": len(ranked)},
                                       lease_id=lease.lease_id)
                return None
            self._said_unfit.pop((workload, chosen_group.key), None)
            ranked = fitting

        if offer_id is None:
            # Every rental — the shared pool's and each workload's, of any kind — takes the offer
            # expected to cost least per worker-hour over the hours its lease has left, across
            # every provider: a bid, a spot price or on demand, each with what its interruptions
            # cost (D115, D131). Deterministic, with its reasons; every cap still applies.
            ranked, kind_reasons = self._by_rental_kind(
                ranked, lease, spec, for_this_host, builds,
                fixed_workers=(chosen_group.workers_per_host
                               if chosen_group is not None and len(chosen_group.models) > 1 else None))
            reasons = [*reasons, *kind_reasons[:1]]
        here = self._volumes_here(spec, builds) if spec is not None else {}
        if here and offer_id is None:
            # An offer whose host would copy its models from the workload's volume is preferred
            # by no more than the download it saves (D139).
            ranked, kept_why = self._prefer_kept_models(ranked, here, lease, for_this_host, builds)
            reasons = [*reasons, *kept_why]

        unaffordable_offers, created = 0, False
        for offer, offer_score in ranked[: self.rented.bidding.attempts]:
            bid = price_bid(offer, self.rented.bidding, self.rented.policy_in_force, lease.max_all_in_hourly)
            capped = self._cap_bid(bid.hourly, lease, offer)
            if capped is None:
                continue

            refusal = self._refuse_bid_for_burn(capped)
            if refusal is not None:
                self.last_refusal = refusal
                self.events.record(
                    "rent_refused", refusal, numbers={"bid": capped}, lease_id=lease.lease_id
                )
                continue
            if spec is not None:
                unaffordable = self._refuse_for_workload_budget(
                    offer, capped, lease, workload, also_hourly=self._new_volume_hourly(spec, offer, here, for_this_host, builds))
                if unaffordable is not None:
                    unaffordable_offers += 1
                    self.last_refusal = unaffordable
                    self.events.record("rent_refused", unaffordable,
                                       numbers={"offer": offer.offer_id, "workload": workload}, lease_id=lease.lease_id)
                    continue

            image, image_why = self.image_for(offer)
            if image is None:
                # Refused *before* the bid: renting a machine whose driver cannot run any build
                # of the engine buys a host that can never answer, and pays for it until the
                # give-up window closes (D92).
                self.events.record(
                    "offer_refused", f"{offer.machine_id}: {image_why}", lease_id=lease.lease_id
                )
                continue

            created = True
            host_id = f"rented-{uuid.uuid4().hex[:6]}"
            workers, workers_why = self.workers_for(offer, workload, tuple(for_this_host) if spec is not None else None)
            launch_workers = self.launch_workers_for(workers, workload)
            if chosen_group is not None and len(chosen_group.models) > 1:
                # Launched with room for the card's whole split, so a group left short of its
                # plan can widen its shares on this host without relaunching the engine (D128).
                whole = workload_math.split_short(
                    {m: spec.target(m).parallel for m in chosen_group.models},
                    self.per_model_on(offer, spec, chosen_group.models, chosen_group.builds), 1, 10**6)
                if whole:
                    launch_workers = max(launch_workers, sum(whole.values()))
            instance_spec = InstanceSpec(
                # A workload's name sits in the label, under the pool's one prefix, so the one
                # sweep still finds every instance the pool owns (workloads.md §1).
                label=f"{self.label_prefix}{workload + '/' if workload else ''}{host_id}",
                image=image,
                disk_gb=policy.min_disk_gb,
                # Launched at what it may be asked for, used at what it is given (D68).
                env=self.instance_env(launch_workers, for_this_host),
                # Armed before anything else runs, and carrying no account credential.
                onstart=self.deadman_onstart(for_this_host, offer.connection),
                volume=self._volume_for(spec, offer, here, for_this_host, builds) if spec is not None else None,
            )
            try:
                # No price on an on-demand rental: the provider's listed rate is what is paid. A spot
                # rental is told the most the pool pays, which a provider that takes a maximum price
                # holds it to (D132).
                instance = await self.provider_for(offer).create(offer, instance_spec, self._price_for_create(offer, capped, lease))
            except (BidLost, OfferGone) as exc:
                numbers = {"bid": capped, "offer": offer.offer_id, "score": offer_score}
                response = getattr(exc, "response", None)
                if response is not None:
                    # Kept whole beside the summary, so a refusal can be read later in the
                    # provider's own terms rather than reconstructed from the market — with
                    # any credential in it replaced first, whichever provider raised it.
                    numbers["provider_response"] = redacted(response)
                created = getattr(exc, "instance_state", None)
                if created is not None:
                    numbers["created_instance"] = redacted(created)
                self.events.record(
                    "bid_failed",
                    f"bid ${capped:.3f} on {offer.machine_id} did not take: {exc}",
                    numbers=numbers,
                    lease_id=lease.lease_id,
                )
                self.last_refusal = f"the bid on {offer.machine_id} did not take: {exc}"
                # A bid is not failed until nothing is running under its label. Bidding on the
                # next offer while a machine from this one bills is how one lease ends up
                # paying for three hosts (D43) — so this is checked here, in the pool, and not
                # left to a plug-in's own discipline. Unprovable means stop, not carry on.
                if not await self._left_nothing_behind(instance_spec.label, lease, offer.connection):
                    return None
                if isinstance(exc, BidLost):
                    # Lost: someone holds the machine above the listed least bid, and the same bid
                    # loses again. Avoided for a while, and this round stops here — firing the next
                    # create at once, and the next, tripped the provider's request limits (D124).
                    self.avoid(offer.machine_id, f"a bid of ${capped:.3f} on it lost",
                               minutes=self.rented.teardown.avoid_lost_bid_minutes)
                    return None
                continue
            except ProviderError as exc:
                self.events.record(
                    "provider_error",
                    f"create failed: {exc}",
                    numbers={"offer": offer.offer_id},
                    lease_id=lease.lease_id,
                )
                return None

            host = RentedHost(
                host_id=host_id,
                instance=instance,
                offer=offer,
                bid_hourly=capped,
                lease_id=lease.lease_id,
                workers=workers,
                launch_workers=launch_workers,
                interruptible=offer.interruptible,
                models=for_this_host,
                profile=profile,
                builds=builds,
                cards_per_copy=cards_per_copy,
                workload=workload,
                disk_gb=policy.min_disk_gb,
                engine=self.config.rented_engine(),
                engine_port=self.engine_port,
            )
            if instance.volume_id is not None and spec is not None and instance_spec.volume is not None:
                self._record_volume(host, instance, instance_spec.volume, offer, lease, for_this_host, builds)
            connection = await self.provider_for(offer).connection(instance)
            host.connection = connection
            host.dial_url = connection.public_url or await self._open_tunnel(host_id, connection, host.engine_port)
            host.mark_preparing()
            self.hosts[host_id] = host
            self.events.record(
                "rented",
                (f"bid ${capped:.3f}/h" if offer.interruptible else f"on demand at ${capped:.3f}/h, not outbiddable")
                + f" on {offer.machine_id} ({offer.hardware}), {workers} workers: {workers_why}"
                + (f"; assumed, as the provider does not report them: {', '.join(_ASSUMED_WORDS.get(a, a) for a in offer.assumed)}"
                   if offer.assumed else ""),
                numbers={
                    "workers": workers,
                    "assumed": list(offer.assumed),
                    # Named here as data, not only in the sentence: the machine history is a
                    # view over this log, and a view should not have to parse prose (D69).
                    "machine": offer.machine_id,
                    "hardware": offer.hardware,
                    "bid": capped,
                    "floor": offer.min_bid_hourly,
                    "all_in": offer.all_in_hourly,
                    "score": offer_score,
                    "reasons": reasons + bid.reasons,
                    "rejected": rejected,
                },
                host_id=host_id,
                lease_id=lease.lease_id,
            )
            return host
        self.last_budget_bound = unaffordable_offers > 0 and not created
        return None

    # --- a new host's models without the hub (D116) ---

    def _sources(self) -> list[str]:
        return list(self.config.workloads.model_sources)

    def _models_dir(self) -> str:
        return getattr(self.engine, "models_dir", "") or ""

    def _group_volumes(self, spec: Any, builds: dict[str, str]) -> list[Volume]:
        """This group's volumes (D118). One made before groups belongs to the workload's only group."""
        return [v for v in self.workload_store.volumes(spec.name)
                if (v.builds == builds) or (v.builds is None and len(spec.groups) == 1)]

    def keeps_models(self, connection: str) -> bool:
        """Is "Keep models between hosts" on for this provider account, and can it be (D139)?
        Only where its volumes reach a data center; set on any other, it does nothing."""
        named = connection or self.legacy_connection
        name = self.resolve(named) or named
        settings, provider = self.rented.providers.get(name), self.providers.get(name)
        return bool(settings is not None and settings.keep_models and provider is not None
                    and provider.capabilities.reach == "data_center" and "volume" in self._sources())

    def _volumes_here(self, spec: Any, builds: Optional[dict[str, str]] = None) -> dict[str, Volume]:
        """This group's model volumes, by the connection each is on (D139). A volume whose filler
        has gone without filling it is left empty, for the next host there to fill — named only
        now, once the pool holds that host no more."""
        if spec is None:
            return {}
        builds = builds if builds is not None else dict(spec.groups[0].builds)
        found: dict[str, Volume] = {}
        for volume in self._group_volumes(spec, builds):
            if volume.location is None:
                continue  # bound to a machine (D116): kept until its workload ends, never used
            if volume.state == "filling" and volume.filler and volume.filler not in self.hosts:
                self.workload_store.volume_state(volume.volume_id, "empty")
                self.events.record(
                    "volume_not_filled",
                    f"model volume {volume.volume_id} at {volume.location}: {volume.filler} went before filling it; "
                    "the next host there fills it",
                    numbers={"volume": volume.volume_id, "workload": spec.name},
                )
                volume = dataclasses.replace(volume, state="empty", filler=None)
            named = volume.connection or self.legacy_connection
            found[self.resolve(named) or named] = volume
        return found

    def _volume_role(self, offer: Offer, here: dict[str, Volume]) -> Optional[tuple[str, Optional[Volume]]]:
        """What a new host from this offer does with its group's volume (D139): copy from it
        ("read"), fill it ("fill"), make it ("new") — or nothing: models not kept here, the offer
        cannot land in the volume's data center, or another host is filling it."""
        if not offer.locations or not self.keeps_models(offer.connection):
            return None
        named = offer.connection or self.legacy_connection
        volume = here.get(self.resolve(named) or named)
        if volume is None:
            return ("new", None)
        if volume.location not in offer.locations:
            return None
        if volume.state == "ready":
            return ("read", volume)
        if volume.state in ("empty", "stale"):
            return ("fill", volume)
        return None

    def _volume_size(self, models: Sequence[str], builds: dict[str, str]) -> int:
        return max(10, math.ceil(self.model_set_gb(models, builds) * 1.1) + 1)

    def _new_volume_hourly(self, spec: Any, offer: Offer, here: dict[str, Volume],
                           models: Sequence[str], builds: dict[str, str]) -> float:
        """What a volume this host would make costs per hour, for the budget check (D139)."""
        role = self._volume_role(offer, here) if spec is not None else None
        if role is None or role[0] != "new":
            return 0.0
        return self._volume_size(models, builds) * (offer.volume_per_gb_hourly or 0.0)

    def _prefer_kept_models(self, ranked: list, here: dict[str, Volume], lease: Lease,
                            models: Sequence[str], builds: dict[str, str]) -> tuple[list, list[str]]:
        """Put first an offer whose host would copy its models from the volume — but only while
        what it costs more over the lease's hours is no more than the download it saves, priced at
        its own hourly rate (D139). Estimated until the pool has its own times for both ways."""
        reaching = [pair for pair in ranked if (self._volume_role(pair[0], here) or ("",))[0] == "read"]
        if not reaching:
            return ranked, []
        top, best = ranked[0][0], reaching[0][0]
        volume = self._volume_role(best, here)[1]
        gigabytes = self.model_set_gb(models, builds)
        download_s = gigabytes * 8000 / max(top.download_mbps or 0.0, 100.0)
        copy_s = gigabytes * 1000 / VOLUME_COPY_MB_PER_S
        saved_h = max(0.0, download_s - copy_s) / 3600
        worth = saved_h * best.all_in_hourly
        extra = max(0.0, best.all_in_hourly - top.all_in_hourly) * lease.hours_left()
        if reaching[0] is ranked[0]:
            return ranked, [f"its models are kept at {volume.location}: this host copies ~{gigabytes:.0f} GB "
                            f"instead of downloading (≈{saved_h * 60:.0f} min saved, estimate)"]
        if extra <= worth:
            return ([reaching[0], *[p for p in ranked if p is not reaching[0]]],
                    [f"its models are kept at {volume.location}: a host there copies ~{gigabytes:.0f} GB instead "
                     f"of downloading — ≈{saved_h * 60:.0f} min, worth ${worth:.2f}, for ${extra:.2f} more over the "
                     "lease (estimate)"])
        return ranked, [f"a host at {volume.location} would copy its models (≈{saved_h * 60:.0f} min, ${worth:.2f}) "
                        f"but costs ${extra:.2f} more over the lease: not preferred"]

    def _volume_for(self, spec: Any, offer: Offer, here: dict[str, Volume],
                    models: Sequence[str], builds: dict[str, str]) -> Optional[VolumeSpec]:
        """The volume a workload's new host is created with (D139): its group's, mounted for the
        agent to copy from, or to fill; or — for its first host where models are kept — a new one
        to fill. Mounted at the agent's own paths, which the pool never sends to the agent."""
        role = self._volume_role(offer, here) if spec is not None else None
        if role is None:
            return None
        kind, volume = role
        index = next((i for i, g in enumerate(spec.groups) if g.builds == builds), 0)
        label = f"{self.label_prefix}{spec.name}/models" + (f"-{index}" if len(spec.groups) > 1 else "")
        if kind == "read":
            return VolumeSpec(mount=str(model_volume.READ_PATH), label=label,
                              volume_id=volume.volume_id, location=volume.location)
        if kind == "fill":
            return VolumeSpec(mount=str(model_volume.FILL_PATH), label=label,
                              volume_id=volume.volume_id, location=volume.location)
        return VolumeSpec(mount=str(model_volume.FILL_PATH), label=label,
                          size_gb=float(self._volume_size(models, builds)))

    def _record_volume(self, host: RentedHost, instance: Instance, asked: VolumeSpec, offer: Offer,
                       lease: Lease, models: Sequence[str], builds: dict[str, str]) -> None:
        host.volume_id = instance.volume_id
        location = instance.location or asked.location or ""
        if asked.volume_id is None:
            size = asked.size_gb
            hourly = round(size * (offer.volume_per_gb_hourly or 0.0), 6)
            self.workload_store.add_volume(Volume(
                volume_id=instance.volume_id, workload=host.workload or "", machine_id=location, size_gb=size,
                hourly=hourly, lease_id=lease.lease_id, created_at=time.time(), builds=dict(builds),
                connection=offer.connection or self.connection, location=location, state="filling",
                filler=host.host_id,
            ))
            self.events.record(
                "volume_created",
                f"model volume {instance.volume_id} ({size:g} GB) made at {location} for workload {host.workload}'s "
                f"models, at ${hourly:.4f}/h while it exists; {host.host_id} fills it, and later hosts there copy "
                "from it",
                numbers={"volume": instance.volume_id, "location": location, "size_gb": size, "hourly": hourly,
                         "workload": host.workload},
                host_id=host.host_id, lease_id=lease.lease_id,
            )
        elif asked.mount == str(model_volume.READ_PATH):
            host.models_source = "volume"
            self.events.record(
                "volume_attached",
                f"{host.host_id} at {location} copies {', '.join(models)} from model volume {instance.volume_id}, "
                "each file checked against the hub",
                numbers={"volume": instance.volume_id, "location": location, "workload": host.workload},
                host_id=host.host_id, lease_id=lease.lease_id,
            )
        else:
            self.workload_store.volume_state(instance.volume_id, "filling", host.host_id)
            self.events.record(
                "volume_filling",
                f"{host.host_id} at {location} fills model volume {instance.volume_id} with {', '.join(models)}",
                numbers={"volume": instance.volume_id, "location": location, "workload": host.workload},
                host_id=host.host_id, lease_id=lease.lease_id,
            )

    def _models_kept(self, host: RentedHost) -> bool:
        """Are this host's models kept on its workload's data-center volume (D139)?"""
        return bool(host.volume_id and self.provider_for(host).capabilities.reach == "data_center")

    def _volume_news(self, host: RentedHost) -> None:
        """What a host's agent says it did with its volume, taken in once every model is fetched
        (D139): a filler's volume is ready, or empty again; a reader's copy is put on record —
        how much, how long, and any file that did not match — and a volume that holds nothing of
        the hub's current build is marked stale, for the next host there to fill."""
        if host.volume_id is None or host.volume_reported or not host.agent_facts or not host.workload:
            return
        sources = (host.agent_facts.get("engine") or {}).get("model_sources") or {}
        tags = [(host.builds or {}).get(model) or model for model in host.models]
        reports = [sources.get(tag) for tag in tags]
        if not reports or any(not isinstance(r, dict) for r in reports):
            return
        volume = next((v for v in self.workload_store.volumes(host.workload) if v.volume_id == host.volume_id), None)
        if volume is None or volume.location is None:
            return
        host.volume_reported = True
        where = {"volume": volume.volume_id, "location": volume.location, "workload": host.workload}
        if volume.filler == host.host_id:
            fills = [str(r.get("fill") or "") for r in reports]
            if all(f == "already there" or f.startswith("filled") for f in fills):
                self.workload_store.volume_state(volume.volume_id, "ready")
                self.events.record(
                    "volume_filled",
                    f"model volume {volume.volume_id} at {volume.location} holds {', '.join(host.models)}: later hosts "
                    "there copy from it", numbers=where, host_id=host.host_id, lease_id=host.lease_id)
            else:
                self.workload_store.volume_state(volume.volume_id, "empty")
                self.events.record(
                    "volume_not_filled",
                    f"{host.host_id} did not fill model volume {volume.volume_id}: "
                    + "; ".join(f for f in fills if not (f == "already there" or f.startswith("filled")))
                    + "; the next host there fills it", numbers=where, host_id=host.host_id, lease_id=host.lease_id)
            return
        copied = sum(int(r.get("bytes_from_volume") or 0) for r in reports)
        files = sum(int(r.get("files_from_volume") or 0) for r in reports)
        mismatched = sum(int(r.get("files_mismatched") or 0) for r in reports)
        missing = sum(int(r.get("files_missing") or 0) for r in reports)
        seconds = sum(float(r.get("seconds_from_volume") or 0.0) for r in reports)
        self.events.record(
            "models_from_volume",
            f"{host.host_id} copied {files} file(s), {copied / 1e9:.1f} GB of {', '.join(host.models)}, from the model "
            f"volume at {volume.location} in {seconds:.0f} s"
            + (f"; {missing} file(s) were not there and came from the hub" if missing else ""),
            numbers={**where, "files": files, "bytes": copied, "seconds": round(seconds, 1),
                     "missing": missing, "mismatched": mismatched},
            host_id=host.host_id, lease_id=host.lease_id,
        )
        if mismatched:
            self.events.record(
                "volume_mismatch",
                f"{host.host_id}: {mismatched} file(s) in the model volume at {volume.location} did not match the "
                "hub; it downloaded them instead", numbers={**where, "mismatched": mismatched},
                host_id=host.host_id, lease_id=host.lease_id)
        if files == 0 and missing and volume.state == "ready":
            self.workload_store.volume_state(volume.volume_id, "stale")
            self.events.record(
                "volume_stale",
                f"model volume {volume.volume_id} at {volume.location} holds nothing of the hub's current build of "
                f"{', '.join(host.models)}; the next host there fills it", numbers=where,
                host_id=host.host_id, lease_id=host.lease_id)

    async def models_from_sibling_pending(self, host: RentedHost) -> bool:
        """Is a copy of this host's models from a ready sibling still running? (D116)

        Started the first time a new workload host is prepared, where a sibling holds the same
        builds; the host's own fetch waits for it, then finds the files already there. A copy
        that fails or takes too long falls through to the hub, said in the log. False when there
        is nothing to wait for."""
        if host.workload is None or host.models_source == "volume" or host.copy_state in ("done", "failed", "none"):
            return False
        if ("sibling" not in self._sources() or not self.provider_for(host).capabilities.copies
                or not self._models_dir()):
            host.copy_state = "none"
            return False
        if host.copy_task is None:
            sibling = next(
                (other for other in self.hosts_of(host.workload)
                 if other is not host and other.state == "ready" and other.builds == host.builds
                 and other.connection_name == host.connection_name),
                None,
            )
            if sibling is None:
                host.copy_state = "none"
                return False
            host.copy_state, host.copy_started_at = "copying", time.time()

            async def copy(source=sibling.instance, destination=host.instance) -> None:
                # The provider is asked only once the task runs: one cancelled first asks nothing.
                await asyncio.wait_for(
                    self.provider_for(host).copy_between(source, destination, self._models_dir()),
                    timeout=self.config.workloads.copy_timeout_s,
                )

            host.copy_task = asyncio.create_task(copy(), name=f"gpm:copy:{host.host_id}")
            self.events.record(
                "models_copy_started",
                f"{host.host_id}: copying its models from {sibling.host_id}, which already holds them",
                numbers={"from": sibling.host_id, "workload": host.workload},
                host_id=host.host_id, lease_id=host.lease_id,
            )
            return True
        if not host.copy_task.done():
            return True
        took = time.time() - (host.copy_started_at or time.time())
        failure = None if host.copy_task.cancelled() else host.copy_task.exception()
        if isinstance(failure, (asyncio.TimeoutError, TimeoutError)):
            # Not waited for any longer, and not known to have stopped: fetching into the same
            # directory now could race it. The host is given up; the floor rents another.
            host.copy_state = "failed"
            self.events.record(
                "models_copy_failed",
                f"{host.host_id}: the copy from a sibling did not finish in "
                f"{self.config.workloads.copy_timeout_s:g}s; giving the host up rather than fetch over it",
                numbers={"seconds": round(took, 1), "workload": host.workload},
                host_id=host.host_id, lease_id=host.lease_id,
            )
            await self.destroy(host, "its models copy did not finish in time")
            return True
        if host.copy_task.cancelled() or failure is not None:
            host.copy_state = "failed"
            why = "cancelled" if failure is None else (str(failure) or type(failure).__name__)
            self.events.record(
                "models_copy_failed",
                f"{host.host_id}: the copy from a sibling did not finish ({why}); fetching from the hub",
                numbers={"seconds": round(took, 1), "workload": host.workload},
                host_id=host.host_id, lease_id=host.lease_id,
            )
            return False
        host.copy_state, host.models_source = "done", "sibling"
        self.events.record(
            "models_copied",
            f"{host.host_id}: its models were copied from a sibling in {took:.0f}s",
            numbers={"seconds": round(took, 1), "workload": host.workload},
            host_id=host.host_id, lease_id=host.lease_id,
        )
        return False

    def expected_costs(
        self, ranked: list[tuple[Offer, float]], hours: float, spec: Any,
        models: Sequence[str], builds: dict[str, str], max_all_in_hourly: Optional[float] = None,
        fixed_workers: Optional[int] = None, have: Optional[float] = None,
    ) -> tuple[list[workload_math.KindCost], list[str]]:
        """What each accepted offer is expected to cost a workload per worker-hour over `hours`
        (D115, workloads.md §5), cheapest first, with the line that says why. A bid's expected
        evictions come from the machine's own history where it has an hour or more of it, else
        the configured prior; each costs the time to get a replacement ready, paid and not
        serving, and this host's share of the workload's capacity meanwhile, priced at what that
        capacity costs on demand. Used by the creation plan and by every rental alike."""
        hours = max(hours, 1e-6)
        size = self.model_set_gb(models, builds)
        record_of = self.machine_history()
        name = spec.name if spec is not None else None
        if have is None:
            have = sum(h.workers for h in self.hosts_of(name, group_key(models) if spec is not None else None)
                       if h.state != "draining")
        on_demand = [o.all_in_hourly for o, _ in ranked if not o.interruptible]
        costs = []
        for offer, _ in ranked:
            # A group of several models runs its fixed split on whatever card it is rented (D118):
            # a bigger card is not more capacity to it, only more cost.
            workers = (fixed_workers if fixed_workers is not None
                       else self.capacity_on(offer, spec, models, builds) if spec is not None
                       else self.workers_for(offer)[0])  # the shared pool: what the card runs
            if workers <= 0:
                continue  # holds the models, but not every one within its target at once
            if offer.interruptible:
                bid = price_bid(offer, self.rented.bidding, self.rented.policy_in_force, max_all_in_hourly)
                hourly = bid.hourly + offer.storage_hourly
            else:
                hourly = offer.all_in_hourly
            record = record_of.get(offer.machine_id)
            rate = getattr(record, "evictions_per_hour", None)
            if rate is None:
                # The machine has no history: its connection's prior, where it has one — a bid
                # market's hosts are outbid, a spot market's reclaimed, at their own rates (D131).
                conn = self.rented.providers.get(offer.connection)
                rate = (conn.interruption_prior_per_hour
                        if conn is not None and conn.interruption_prior_per_hour is not None
                        else self.config.workloads.eviction_prior_per_hour)
            costs.append(workload_math.expected_cost(
                offer_id=offer.offer_id, machine_id=offer.machine_id, interruptible=offer.interruptible,
                connection=offer.connection, bidding=offer.bidding,
                hourly=hourly, hours=hours, workers=workers, evictions_per_hour=rate,
                ready_hours=workload_math.time_to_ready_hours(size, offer.download_mbps, self.config.workloads.engine_load_s),
                lost_capacity_hourly=offer.on_demand_hourly or (min(on_demand) if on_demand else hourly),
                share_of_workload=workers / max(1, have + workers),
                download=round(offer.download_per_gb * size, 4),
            ))
            # The machine's own record, as the score weighed it (D69): one that failed to come up
            # here, or served slowly, costs more per worker-hour than its price says; one that
            # served fast, less.
            factor, why = history_adjustment(record, self.rented.history)
            if factor != 1.0:
                cost = costs[-1]
                costs[-1] = dataclasses.replace(cost, per_worker_hour=round(cost.per_worker_hour / factor, 5),
                                                reasons=[*cost.reasons, f"its record here: {why} (x{1 / factor:.2f})"])
        return workload_math.order_by_expected_cost(costs)

    def capacity_on(self, offer: Offer, spec: Any, models: Sequence[str], builds: dict[str, str]) -> float:
        """Answers at once one host from this offer gives the workload, for pricing it per
        worker-hour: one model's measured number; for several on one host, the answers of the
        workload's own mix it holds within every target, `P / Σ p_m / w_m` (D118). Zero where
        one answer of each already fills the card."""
        per_model = self.per_model_on(offer, spec, models, builds)
        if len(models) == 1:
            return float(per_model[models[0]])
        parallel = {m: (spec.target(m).parallel if spec.target(m) else 1) for m in models}
        share = sum(parallel[m] / per_model[m] for m in models if per_model[m] > 0)
        if any(per_model[m] < 1 for m in models) or sum(1 / per_model[m] for m in models) > 1 + 1e-9:
            return 0.0
        return sum(parallel.values()) / share if share else 0.0

    def _by_rental_kind(
        self, ranked: list[tuple[Offer, float]], lease: Lease, spec: Any,
        models: Sequence[str], builds: dict[str, str], fixed_workers: Optional[int] = None,
    ) -> tuple[list[tuple[Offer, float]], list[str]]:
        """The accepted offers in the rental-kind rule's order, over the hours the lease has left."""
        ordered, why = self.expected_costs(ranked, lease.hours_left(), spec, models, builds, lease.max_all_in_hourly,
                                           fixed_workers=fixed_workers)
        # By id and kind: a provider may list one machine's bid and its fixed price under the same
        # id, and keyed by id alone the second overwrote the first — the rule chose a bid and the
        # pool rented the same machine on demand, at twice the price (D127).
        # And by connection: two providers may number their offers alike (D129).
        by_id = {(offer.connection, offer.offer_id, offer.interruptible): (offer, points) for offer, points in ranked}
        said = (ordered[0].connection, ordered[0].offer_id, ordered[0].interruptible) if ordered else None
        name = spec.name if spec is not None else None
        # Per group (D118): two groups choosing differently must not flip the one note.
        if self._said_kind.get((name, group_key(models))) == said:
            return [by_id[(c.connection, c.offer_id, c.interruptible)] for c in ordered], why
        self._said_kind[(name, group_key(models))] = said
        self.events.record(
            "rental_kind", why[0],
            numbers={"workload": name, "hours_left": round(lease.hours_left(), 3),
                     "candidates": [dataclasses.asdict(c) for c in ordered[:5]]},
            lease_id=lease.lease_id,
        )
        return [by_id[(c.connection, c.offer_id, c.interruptible)] for c in ordered], why

    def _price_for_create(self, offer: Offer, capped: float, lease: Lease) -> Optional[float]:
        """What `create` is told: the bid for a bid; for a spot rental, the most the pool pays for
        the machine (its all-in ceiling less storage), which a provider that takes a maximum
        price holds it to (D132); nothing for an on-demand rental, whose price is fixed."""
        if not offer.interruptible:
            return None
        if offer.bidding:
            return capped
        ceiling, _which = all_in_ceiling(offer, self.rented.policy_in_force, lease.max_all_in_hourly)
        return round(ceiling - offer.storage_hourly, 4)

    async def _past_its_timerless_hours(self, host: RentedHost) -> bool:
        """A host on a connection with no dead-man timer is ended once it has run as long as a
        lease there may (spec §7) — however its lease came to be longer: restarted from a park,
        extended, amended. Nothing on the machine could stop it billing (D129)."""
        limit = self.lease_hours_on(host.connection_name) if host.connection_name in self.providers else None
        if limit is None or host.state in ("draining", "parked") or host.hours_held < limit:
            return False
        self.events.record(
            "timerless_limit",
            f"{host.host_id} has run {host.hours_held:.2f}h on {host.connection_name}, which cannot arm a dead-man "
            f"timer; hosts there run at most {limit:g}h, so it is released",
            numbers={"hours": round(host.hours_held, 3), "limit": limit},
            host_id=host.host_id, lease_id=host.lease_id,
        )
        await self.drain(host, f"{host.connection_name} cannot arm a dead-man timer, and its {limit:g}h are up")
        return True

    async def _spot_moved(self, host: RentedHost, status: Any) -> bool:
        """A spot host whose price moved, or that the provider is about to take back (D132). Its
        spend is settled at the old price and carries on at the new; past the all-in ceiling it is
        drained and released, as no bid can bring it back under; warned of an interruption, it is
        drained at once, so what it has finishes or is sent elsewhere. True when it was drained."""
        if status.interrupting and host.state not in ("draining", "parked"):
            self.events.record(
                "interruption_warned",
                f"{host.host_id}: the provider is taking it back; draining it now",
                numbers={"machine": host.offer.machine_id},
                host_id=host.host_id, lease_id=host.lease_id,
            )
            await self.drain(host, "the provider is taking it back")
            return True
        if not host.interruptible or host.offer.bidding or status.bid_hourly is None:
            return False
        price = float(status.bid_hourly)
        if abs(price - host.bid_hourly) < 1e-9:
            return False
        was = host.bid_hourly
        host.settle()
        host.bid_hourly = price
        lease = self.leases.get(host.lease_id)
        ceiling, which = all_in_ceiling(host.offer, self.rented.policy_in_force,
                                        lease.max_all_in_hourly if lease is not None else None)
        over = price + host.offer.storage_hourly > ceiling
        self.events.record(
            "spot_price_moved",
            f"{host.host_id}'s spot price moved from ${was:.3f}/h to ${price:.3f}/h"
            + (f", past {which} (${ceiling:.3f}/h all-in); releasing it" if over else ""),
            numbers={"machine": host.offer.machine_id, "was": was, "now": price, "ceiling": ceiling},
            host_id=host.host_id, lease_id=host.lease_id,
        )
        if over and host.state not in ("draining", "parked"):
            await self.drain(host, f"its spot price passed {which}")
            return True
        refused = self._refuse_rebid(host, price, lease) if lease is not None else None
        if refused is not None and host.state not in ("draining", "parked"):
            # The same test a re-bid passes (D45): the burn cap and what the lease has left.
            self.events.record("spot_price_refused", f"{host.host_id}: at its new spot price, {refused}; releasing it",
                               numbers={"machine": host.offer.machine_id, "now": price},
                               host_id=host.host_id, lease_id=host.lease_id)
            await self.drain(host, f"at its new spot price, {refused}")
            return True
        return False

    def _cap_bid(self, bid: float, lease: Lease, offer: Offer) -> Optional[float]:
        """The supervisor's own clamp, applied after the strategy returns — a faulty or
        hostile strategy cannot spend past the ceilings. The ceiling is all-in (D108): a bid
        may rise to it less the storage the host is billed beside the bid; a fixed price is
        all-in already."""
        all_in, _which = all_in_ceiling(offer, self.rented.policy_in_force, lease.max_all_in_hourly)
        ceiling = all_in - offer.storage_hourly if offer.interruptible else all_in
        if bid > ceiling:
            self.events.record(
                "bid_clamped",
                f"a strategy returned ${bid:.3f}/h, above the ${ceiling:.3f} ceiling; clamped",
                numbers={"returned": bid, "ceiling": ceiling},
                lease_id=lease.lease_id,
            )
            bid = ceiling
        return bid if bid > 0 else None

    # --- tear down ---

    async def tear_down(
        self,
        open_leases: list[Lease],
        idle_seconds: dict[str, float],
        ready_workers_higher_tiers: int = 0,
        workload: Optional[str] = None,
        group: Optional[tuple[str, ...]] = None,
    ) -> None:
        unit = self.unit(workload, group)
        live = [h for h in self.hosts_of(workload, unit.group) if h.state != "parked"]
        if not live:
            unit.overflow_gone_since = None
            return
        lease = self.lease_of(open_leases, workload)
        spec = self.group_spec(workload, unit.group)
        now = time.time()

        # A host that is not ready yet is neither idle nor surplus: it is capacity on its
        # way, and billing while it comes. It is only given up on when it takes too long.
        # Only a host actually on its way counts: one that is draining is finishing its work
        # and ends when that is done (found live: a host drained at its lease's end was taken
        # for one "never ready", destroyed a second later with four answers in flight, and its
        # good machine avoided for an hour).
        preparing = [h for h in live if h.state in ("scheduling", "preparing")]
        ready = [h for h in live if h.state == "ready"]
        for host in preparing:
            since = host.preparing_since or host.created_at
            starting_for = (now - since) / 60
            if host.engine_seen_at is None and starting_for >= self.rented.teardown.max_starting_minutes:
                # Stuck before it ever served anything: the provider is still scheduling or
                # starting it. Waiting the full preparing window would bill three times as long.
                self.avoid(host.offer.machine_id, "never started")
                said = await self.why_it_never_started(host)
                self.events.record(
                    "host_stuck_starting",
                    f"{host.host_id} has not started after {starting_for:.0f} minutes "
                    f"(limit {self.rented.teardown.max_starting_minutes:g}); giving it up"
                    + (f". Its own boot output says: {said}" if said else ""),
                    numbers={"minutes": round(starting_for, 1), "machine": host.offer.machine_id},
                    host_id=host.host_id, lease_id=host.lease_id,
                )
                await self.destroy(host, f"never started within {self.rented.teardown.max_starting_minutes:g} minutes")
                continue
            if (now - since) / 60 >= self.rented.teardown.max_preparing_minutes:
                self.avoid(host.offer.machine_id, "never became ready")
                await self.destroy(
                    host,
                    f"not ready after {self.rented.teardown.max_preparing_minutes:g} minutes",
                )

        wanted = self.group_ceiling(lease, workload, unit.group) if lease else 0
        covered = ready_workers_higher_tiers + sum(h.workers for h in ready)
        if lease is not None and covered >= wanted and wanted > 0:
            unit.overflow_gone_since = unit.overflow_gone_since or now
        else:
            unit.overflow_gone_since = None
        gone_for = (now - unit.overflow_gone_since) if unit.overflow_gone_since else 0.0

        views = [
            HostView(
                host_id=h.host_id,
                priority=20,
                busy_workers=0,
                total_workers=h.workers,
                idle_seconds=idle_seconds.get(h.host_id, 0.0),
                bid_hourly=h.bid_hourly,
                machine_id=h.offer.machine_id,
                hours_held=h.hours_held,
            )
            for h in ready
        ]
        # "Overflow gone" is only surplus if demand stays covered *without* the host being
        # reaped, and only once it has stayed gone for the scale-down window.
        surplus_workers = covered - wanted
        # Only surplus if losing *any* ready host would still cover demand — with hosts of
        # different sizes, the largest is the one to measure against.
        largest = max((h.workers for h in ready), default=self.rented.workers)
        overflow_gone = gone_for >= self.rented.scale.scale_down_after_s and surplus_workers >= largest
        demand = Demand(
            wanted_workers=wanted,
            ready_workers_higher_tiers=ready_workers_higher_tiers,
            rented_workers=sum(h.workers for h in ready),
            overflow_age_s=0.0 if overflow_gone else 1.0,
        )
        held = {
            h.host_id for h in live if h.hold_until is not None and h.hold_until > now
        }
        reaped_for_overflow = 0
        for action in decide_teardown(
            views, demand, self._lease_view(lease) if lease else None, self.rented.teardown, lease is not None
        ):
            host = self.hosts.get(action.host_id)
            if host is None or host.released:
                continue
            if spec is not None and lease is not None and len(ready) <= spec.hosts_at_start:
                # A workload keeps the hosts it starts with for its whole lease: they were
                # reserved for the run, and reaping one only rents it again next pass (D115).
                continue
            if self.last_host_serving(host):
                # Taking it would leave a model with nowhere to go, and every request for it
                # would be refused until another machine was bought and prepared — minutes at
                # best (D95). A host doing nothing is cheaper than a model that cannot be
                # served at all; the lease's caps still bound what this costs.
                self.events.record(
                    "teardown_declined",
                    f"{host.host_id}: kept — the only host serving {', '.join(host.models)}",
                    host_id=host.host_id,
                )
                continue
            idle = action.reasons[0].startswith("idle")
            if action.host_id in held and not idle:
                continue  # prepared and inside its hold: not surplus yet. Unused is another matter (D64)
            if idle:
                unit.idle_gate = True
            if "overflow" in action.reasons[0]:
                if reaped_for_overflow >= 1:
                    continue  # one host at a time (spec §9)
                reaped_for_overflow += 1
            if action.action == "park" and self.provider_for(host).capabilities.parkable:
                if idle:
                    host.idle_since = now - idle_seconds.get(host.host_id, 0.0)
                await self.park(host, "; ".join(action.reasons))
            elif action.action == "park" and idle:
                continue  # cannot be paused here: it runs on until the destroy limit
            else:
                await self.destroy(host, "; ".join(action.reasons))

    async def down_all(self, reason: str = "gpm down --all") -> list[str]:
        """The panic button: destroy everything rented, now, verified — and close every open
        lease, so nothing is rented straight back (spec §9)."""
        released = []
        for lease in self.leases.open_leases():
            self.leases.close(lease.lease_id, reason)
        for host in list(self.hosts.values()):
            if host.released:
                continue
            await self.destroy(host, reason)
            released.append(host.host_id)
        return released

    async def park(self, host: RentedHost, reason: str) -> None:
        if self._models_kept(host):
            # Its models are on its workload's volume, so a stopped host would keep nothing the
            # next host lacks, and would bill for it (D139).
            await self.destroy(host, f"{reason}; released rather than parked: its models are kept on its model volume")
            return
        try:
            await self.provider_for(host).stop(host.instance)
        except ProviderError as exc:
            log.warning("park of %s failed: %s", host.host_id, exc)
            return
        host.settle()  # what it cost running, before it bills only its disk
        host.state = "parked"
        host.parked_at = time.time()
        break_even_hours = (
            host.download_cost / host.offer.storage_hourly if host.offer.storage_hourly else None
        )
        self.events.record(
            "parked",
            f"{host.host_id} parked: {reason}",
            numbers={
                "storage_hourly": host.offer.storage_hourly,
                "download_cost": host.download_cost,
                "break_even_hours": break_even_hours,
            },
            host_id=host.host_id,
            lease_id=host.lease_id,
        )

    async def destroy(self, host: RentedHost, reason: str) -> None:
        if host.copy_task is not None and not host.copy_task.done():
            host.copy_task.cancel()  # a copy to a host that is going is work for nobody
        ok = await self._destroy_instance(host.instance, reason, host.connection_name)
        if ok:
            host.released = True
            host.state = "released"
            self.hosts.pop(host.host_id, None)
            await self.close_tunnel(host.host_id)
            self.events.record(
                "released",
                f"{host.host_id} destroyed and verified: {reason}",
                numbers={"estimated_spend": host.estimate()},
                host_id=host.host_id,
                lease_id=host.lease_id,
            )
            self._end_prepare_lease(host, reason)

    def _end_prepare_lease(self, host: RentedHost, why: str) -> None:
        """A prepare lease exists to get *one* host ready; when that host is gone, so is the
        lease (D47).

        Left open, it is still spending authority, and its workers still count as demand — so
        the pool may rent a replacement nobody asked for, quite possibly on the machine that
        just failed. Found live: four prepare leases outlived their hosts tonight, one by 18
        minutes, each until the operator closed it by hand. An overflow lease is a different
        thing — standing demand, deliberately kept open after an eviction so capacity is
        recovered — and is left alone.
        """
        if not host.prepared:
            return
        if any(h.lease_id == host.lease_id and not h.released for h in self.hosts.values()):
            return
        lease = self.leases.get(host.lease_id)
        if lease is None or not lease.is_open:
            return  # already closed, and its reason (a cap, an expiry) is the one to keep
        self.leases.close(host.lease_id, f"its host {host.host_id} is gone: {why}")
        self.events.record(
            "lease_closed",
            f"{host.lease_id} closed with its prepared host {host.host_id} ({why}); prepare "
            "again for another — nothing is rented in its place",
            host_id=host.host_id,
            lease_id=host.lease_id,
        )

    async def _destroy_instance(self, instance: Instance, reason: str, connection: Optional[str] = None) -> bool:
        """A release counts only once the provider's listing no longer shows it."""
        try:
            provider = self.provider_for(connection or self.legacy_connection)
            await provider.destroy(instance)
        except ProviderError as exc:
            log.warning("destroy of %s failed (%s); will retry next pass", instance.instance_id, exc)
            self.events.record(
                "destroy_failed",
                f"destroy of {instance.instance_id} failed: {exc}",
                numbers={"instance": instance.instance_id},
            )
            return False
        try:
            remaining = await provider.list_instances(self.label_prefix)
        except ProviderError:
            return False
        if any(i.instance_id == instance.instance_id for i in remaining):
            self.events.record(
                "destroy_unverified",
                f"{instance.instance_id} still listed after destroy; retrying next pass",
                numbers={"instance": instance.instance_id},
            )
            return False
        return True

    # --- helpers ---

    #: Which listings each choice searches: (interruptible, on_demand).
    _KINDS = {"interruptible": (True, False), "on_demand": (False, True),
              "cheaper": (True, True), "both": (True, True)}

    def query_for(self, policy: Optional[OfferPolicy], bids: bool, fixed: bool) -> OfferQuery:
        """The provider's query for a search: the listings asked for, and — with a policy — every
        filter of it that a provider can apply without hiding an offer the pool would take (D123)."""
        broad = policy or self.rented.policy_in_force
        if policy is None:
            return OfferQuery(verified_only=broad.verified_only, min_gpus=broad.gpus_multiple_of,
                              interruptible=bids, on_demand=fixed)
        return OfferQuery(
            verified_only=policy.verified_only,
            min_gpus=policy.gpus_multiple_of,
            min_gpu_memory_gb=policy.min_gpu_memory_gb,
            min_disk_gb=policy.min_disk_gb,
            max_all_in_hourly=policy.max_all_in_hourly,
            min_download_mbps=policy.min_download_mbps,
            max_download_per_gb=policy.max_download_per_gb,
            min_reliability=policy.min_reliability,
            exclude_hardware=tuple(policy.exclude_hardware),
            avoid_machines=tuple(policy.avoid_machines),
            interruptible=bids,
            on_demand=fixed,
        )

    # --- the provider's daily search quota (D121) ---

    async def _counted(self, search: Any, connection: Optional[str] = None) -> Any:
        """Run a search, then add what it returned to the day's count — refused or not."""
        try:
            return await search
        finally:
            self.note_search_usage(connection)

    @staticmethod
    def _utc_day(now: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now))

    def note_search_usage(self, connection: Optional[str] = None) -> None:
        """Take the rows the provider counted since last asked, and its last refusal, into the
        day's row. A provider without a search quota has nothing to take."""
        connection = connection or self.connection
        take = getattr(self.providers.get(connection), "take_search_usage", None)
        if take is None:
            return
        usage = take()
        refusal = usage.get("refusal")
        if not usage.get("rows") and refusal is None:
            return
        now = time.time()
        day = self._utc_day(now)
        db = self.events.db
        key = (connection, day)
        db.execute("INSERT INTO provider_search_usage (connection, day, rows, updated_at) VALUES (?, ?, 0, ?) "
                   "ON CONFLICT(connection, day) DO NOTHING", (*key, now))
        db.execute("UPDATE provider_search_usage SET rows = rows + ?, quota = ?, updated_at = ? "
                   "WHERE connection = ? AND day = ?", (int(usage.get("rows") or 0), usage.get("limit"), now, *key))
        if refusal is not None:
            # The provider's own count beats ours: other tools on the same key spend it too.
            spent = int(refusal["limit"]) - int(refusal.get("remaining") or 0)
            db.execute(
                "UPDATE provider_search_usage SET rows = MAX(rows, ?), quota = ?, exhausted_at = ?, resets_at = ? "
                "WHERE connection = ? AND day = ?",
                (spent, int(refusal["limit"]), refusal["at"], refusal["at"] + float(refusal.get("retry_after_s") or 0), *key),
            )

    def search_quota(self, connection: Optional[str] = None) -> Optional[dict]:
        """Today's use of the provider's search quota, for the console: what the pool's own
        searches returned (other tools on the same key are not seen until the provider refuses),
        the quota, and when it resets. None for a provider without one."""
        connection = connection or self.connection
        limit = getattr(self.providers.get(connection), "daily_search_rows", None)
        if limit is None:
            return None
        now = time.time()
        rows = self.events.db.query("SELECT * FROM provider_search_usage WHERE connection = ? AND day = ?",
                                    (connection, self._utc_day(now)))
        row = rows[0] if rows else None
        quota = int((row["quota"] if row and row["quota"] else None) or limit)
        used = int(row["rows"]) if row else 0
        midnight = (int(now // 86400) + 1) * 86400
        exhausted = bool(row and row["exhausted_at"] and (row["resets_at"] or midnight) > now)
        return {
            "used": min(used, quota) if not exhausted else quota,
            "limit": quota,
            "remaining": 0 if exhausted else max(0, quota - used),
            "exhausted": exhausted,
            "resets_at": (row["resets_at"] if row and row["resets_at"] else None) or midnight,
            "source": "provider" if exhausted else "counted",
        }

    async def _offers(self, policy: Optional[OfferPolicy] = None, kinds: Optional[str] = None,
                      connections: Optional[Sequence[str]] = None) -> list[Offer]:
        """Ask the market — every enabled connection at once, or the ones named — then filter here.

        With a `policy`, the filters it implies are also asked of each provider (D123): every
        offer a search returns counts against the provider's daily quota, so what the pool would
        only throw away is better never returned. The pool still applies every filter itself,
        and what it says was rejected, and why, is what came back. Without a policy — finding the
        very machine a host is on, to re-bid or restart it — the market is asked broadly, as a
        machine already held need not pass today's search.

        Each connection has its own back-off, quota and errors (D129): one that refuses or fails
        leaves the others' offers standing, and says why in `searching`."""
        names = [n for n in (connections if connections is not None else self.searchable()) if n in self.providers]
        # The pool's mode decides what it rents *by itself*. One request — a preview, or an
        # operator preparing a particular host — may name its own (D55).
        bids, fixed = self._KINDS[kinds or self.rented.mode]
        query = self.query_for(policy, bids, fixed)
        self.last_query_filters = describe_query(query)
        found = await asyncio.gather(*(self._search_one(name, query) for name in names))
        failed = [n for n in names if self.searching[n].error]
        if names and len(failed) == len(names):
            # Nobody answered: an empty list here is indistinguishable from a market with
            # nothing in it, and "0 offers seen" is a very different thing to tell an operator
            # than "the provider would not answer" (D44).
            self.last_offer_error = (self.searching[names[0]].error if len(names) == 1
                                     else "; ".join(f"{n}: {self.searching[n].error}" for n in failed))
        else:
            self.last_offer_error = None
        # Priced for the disk the host would be rented with, so the all-in the filters and
        # the bid compare is what it would be billed (D108).
        disk_gb = (policy or self.rented.policy_in_force).min_disk_gb
        return [dataclasses.replace(offer.priced_for(disk_gb), connection=name)
                for name, offers in zip(names, found, strict=True) for offer in offers]

    async def _search_one(self, name: str, query: OfferQuery) -> list[Offer]:
        state = self.searching.setdefault(name, SearchState())
        now = time.monotonic()
        if now < state.retry_at:
            # The reason is the provider's, said once; only the remaining wait moves. Nesting
            # this produced "… — not asking again for 60s — not asking again for 44s" live.
            state.error = f"{state.refusal or 'the provider refused'} — not asking again for {state.retry_at - now:.0f}s"
            return []
        try:
            offers = await self._counted(self.providers[name].search_offers(query), name)
        except ProviderRateLimited as exc:
            # Asking again on the next pass is what earned the refusal: wait a minute instead, and
            # again after every refusal (D44, D119 — a flat minute, the owner's choice: no longer
            # doubling to fifteen).
            state.backoff_s = OFFER_RATE_LIMIT_WAIT_S
            state.retry_at = now + state.backoff_s
            log.warning("%s's offer search rate limited; not asking again for %.0fs", name, state.backoff_s)
            state.refusal = str(exc)
            state.error = f"{exc} — not asking again for {state.backoff_s:.0f}s"
            return []
        except ProviderError as exc:
            log.warning("%s's offer search failed: %s", name, exc)
            state.error = str(exc)
            return []
        except Exception as exc:  # noqa: BLE001 - a plug-in's bug must not take the other providers' offers with it
            log.exception("%s's offer search raised", name)
            state.error = f"the provider plug-in failed: {type(exc).__name__}: {exc}"
            return []
        state.backoff_s, state.retry_at, state.error = 0.0, 0.0, None
        return offers

    def _lease_view(self, lease: Optional[Lease]) -> Optional[LeaseView]:
        if lease is None:
            return None
        return LeaseView(
            lease_id=lease.lease_id,
            allow_rent=lease.allow_rent,
            hours_left=lease.hours_left(),
            dollars_left=self.budget_left(lease),
            max_all_in_hourly=lease.max_all_in_hourly,
        )
