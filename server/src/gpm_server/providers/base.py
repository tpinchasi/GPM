"""The provider plug-in interface.

docs/spec/plugin-interfaces.md §1. A provider is an **HTTP API client** with typed errors,
timeouts and retries — never a wrapper around a command-line tool, because output formats churn
between versions, errors arrive as prose, and none of that is acceptable in an unattended
service that spends money.

A provider states what it can do; the pool adapts rather than assumes.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Optional, Protocol, runtime_checkable


class ProviderError(Exception):
    """Base for everything a provider raises. Never a bare HTTP error."""


class ProviderAuthError(ProviderError):
    """The account credential was refused."""


class ProviderRateLimited(ProviderError):
    """Back off and try the next pass."""


class ProviderUnavailable(ProviderError):
    """The provider's API is not answering."""


class OfferGone(ProviderError):
    """The offer disappeared between search and create — normal in a live market."""


class BidLost(ProviderError):
    """The bid was placed and did not win."""


@dataclasses.dataclass(frozen=True)
class ProviderCapabilities:
    """What this provider can do. Each absent capability removes a behaviour rather than
    silently degrading one."""

    #: Instances are bid for and can be outbid. Without it, no bidding strategy applies.
    interruptible: bool = False
    #: An instance can be stopped with its disk kept. Without it, tear-down is always destroy.
    parkable: bool = False
    #: A stopped instance can be restarted by raising its bid.
    same_machine_rebid: bool = False
    #: An instance-scoped credential lets an instance end itself. Without it there is no
    #: dead-man timer, and the pool refuses leases longer than a short maximum.
    self_terminate: bool = False
    #: The provider says what an instance has cost. Without it, spend is estimate-only and the
    #: cap safety margin is widened.
    reports_charges: bool = False
    price_history: bool = False
    #: A public port can be mapped to the engine. Without it, rented hosts are tunnel-only.
    direct_port_mapping: bool = False
    #: The provider will hand back an instance's own boot output. Without it, a host that
    #: never answers is given up knowing only that it never answered (D78).
    reports_instance_logs: bool = False


@dataclasses.dataclass(frozen=True)
class Offer:
    """One rentable machine, carrying everything ranking and the ceilings need."""

    offer_id: str
    machine_id: str
    hardware: str
    gpus: int
    gpu_memory_gb: float
    disk_gb: float
    #: The lowest bid that currently wins this machine.
    min_bid_hourly: float
    #: The all-in hourly figure the provider quotes, including its own overheads.
    all_in_hourly: float
    on_demand_hourly: Optional[float] = None
    storage_hourly: float = 0.0
    download_per_gb: float = 0.0
    download_mbps: float = 0.0
    #: The machine's accelerator driver, as the provider reports it ("595.84"). None
    #: where a provider does not say. An engine image needs a floor, and a machine below
    #: it runs on the CPU instead of the accelerator being paid for (D81).
    driver_version: Optional[str] = None
    reliability: float = 0.0
    verified: bool = False
    #: A rough throughput proxy for ranking — never presented as a benchmark.
    throughput_proxy: float = 0.0
    #: Whether this offer is bid for and can be outbid. False is an on-demand rental: it costs
    #: `all_in_hourly`, there is nothing to bid, and nobody can take it away (D52).
    interruptible: bool = True
    raw: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)


@dataclasses.dataclass(frozen=True)
class InstanceSpec:
    """What to create. Nothing secret goes in here: the host can read its own environment."""

    label: str
    image: str
    disk_gb: float
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    onstart: Optional[str] = None
    ports: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class Instance:
    instance_id: str
    label: Optional[str] = None
    machine_id: Optional[str] = None
    raw: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)


class InstanceState(str):
    """Coarse states every provider must be able to report."""

    RUNNING = "running"
    SCHEDULING = "scheduling"
    STOPPED = "stopped"
    GONE = "gone"


@dataclasses.dataclass(frozen=True)
class SelfTerminateRequest:
    """How one instance ends itself: the call, not a command line (D71).

    The provider says *what* to ask for; the host's script decides *how* to ask, with whatever
    HTTP client that machine turns out to have. A command line naming one client is a
    dependency on a binary the engine image may not carry — the first one carries neither curl
    nor wget.

    A header value may name an environment variable the provider injects
    (`$CONTAINER_API_KEY`), expanded on the host. The account credential never appears here
    (threat model T5).
    """

    method: str
    url: str
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    body: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class InstanceStatus:
    state: str
    #: True when the provider stopped it — outbid — rather than the pool asking for it.
    #: Providers rarely say this outright; the supervisor also compares against its own
    #: record of intent (docs/spec/supervisor.md §1.2).
    stopped_by_provider: Optional[bool] = None
    bid_hourly: Optional[float] = None
    detail: Optional[str] = None
    #: Does the instance carry the start-up material it was created with? False when the
    #: provider reports it absent; None when the provider cannot say. A host without it has no
    #: dead-man timer and no way in for the pool, and can never join (D65).
    startup_material: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class ConnectionInfo:
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: str = "root"
    #: Set only where `direct_port_mapping` applies.
    public_url: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Charges:
    """What the provider says an instance has cost so far."""

    total: float
    currency: str = "USD"
    as_of: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class AccountStatus:
    credential_valid: bool
    credit_remaining: Optional[float] = None
    detail: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class OfferQuery:
    min_gpu_memory_gb: float = 0.0
    min_disk_gb: float = 0.0
    max_all_in_hourly: Optional[float] = None
    verified_only: bool = False
    limit: int = 100
    #: Which rentals to look at: interruptible ones to bid for, non-interruptible ones at the
    #: listed price, or both and let ranking choose.
    interruptible: bool = True
    on_demand: bool = False


@runtime_checkable
class Provider(Protocol):
    interface_version: ClassVar[str]
    name: ClassVar[str]
    capabilities: ProviderCapabilities

    async def list_instances(self, label_prefix: str) -> list[Instance]:
        """**Every** instance carrying the prefix, in any state, including stopped ones that
        still bill storage. The orphan sweep and crash recovery rest on this."""

    async def search_offers(self, query: OfferQuery) -> list[Offer]: ...

    async def create(self, offer: Offer, spec: InstanceSpec, bid: Optional[float]) -> Instance:
        """Either returns a running-or-scheduling instance, or raises and **leaves nothing
        behind**. A bid that loses must fail, not leave a parked instance."""

    async def set_bid(self, instance: Instance, bid: float) -> None: ...

    async def start(self, instance: Instance) -> None: ...

    async def stop(self, instance: Instance) -> None:
        """Keep the disk."""

    async def destroy(self, instance: Instance) -> None:
        """Idempotent. The pool verifies by listing again; it never trusts a return value."""

    async def status(self, instance: Instance) -> InstanceStatus: ...

    async def connection(self, instance: Instance) -> ConnectionInfo: ...

    async def reported_charges(self, instance: Instance) -> Optional[Charges]: ...

    async def instance_logs(self, instance: Instance, tail: int = 60) -> Optional[str]:
        """The instance's own boot output, newest `tail` lines, or None where there is none.

        Read when a host is given up for never answering, so the pool records *why* rather
        than only that it happened (D78). Never on the request path, never required: a
        provider without `capabilities.reports_instance_logs` simply has none, and a host is
        given up exactly as before. The text is a machine's output and is treated as data —
        recorded and shown, never executed, never parsed for anything the pool then acts on.
        """
        return None

    async def account(self) -> AccountStatus: ...

    def self_terminate_request(self, action: str = "destroy") -> SelfTerminateRequest:
        """The call an instance makes to end **itself**, with the provider's instance-scoped
        credential as the instance already holds it (D71).

        A request, not a command line: the host script picks a client that exists on that
        machine. Only meaningful where `capabilities.self_terminate` is set. It must never
        embed the account credential — this is written to a file on a machine the pool does not
        trust (threat model T5).
        """


class ProviderNotFound(Exception):
    """Configuration named a provider that is not installed."""


#: The group third-party providers register under. The ones shipped here use it too: there is
#: no privileged path for first-party plug-ins.
ENTRY_POINT_GROUP = "gpm.providers"


def available_providers() -> dict[str, Any]:
    from importlib.metadata import entry_points

    return {point.name: point for point in entry_points(group=ENTRY_POINT_GROUP)}


def get_provider(name: str, settings: dict[str, Any]) -> Provider:
    """Nothing is loaded that configuration does not name.

    A plug-in runs inside the supervisor with its full authority, including the account
    credential, so installing one is a trust decision equal to installing the framework
    (threat model T14).
    """
    found = available_providers()
    if name not in found:
        raise ProviderNotFound(
            f"unknown provider {name!r}; installed: {sorted(found) or 'none'}"
        )
    return found[name].load()(**settings)
