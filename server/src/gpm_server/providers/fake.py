"""The fake provider: the full interface, in memory, scriptable.

docs/spec/plugin-interfaces.md §1 and supervisor.md §10. **The entire supervisor test suite runs
against this** — floors that move, a bid lost between search and create, an instance parked by
the provider, an eviction at a chosen moment, a `destroy` that fails the first time, charges that
drift from the estimate. No test needs a cloud account.

It ships with the framework rather than living in the test tree: an adopter writing a provider
plug-in tests it against the same scripts.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from typing import Any, ClassVar, Optional

from .base import (
    AccountStatus,
    BidLost,
    Charges,
    ConnectionInfo,
    Instance,
    InstanceSpec,
    InstanceState,
    InstanceStatus,
    Offer,
    OfferGone,
    OfferQuery,
    ProviderAuthError,
    ProviderCapabilities,
    ProviderUnavailable,
    SelfTerminateRequest,
)


@dataclasses.dataclass
class FakeInstance:
    instance_id: str
    label: str
    machine_id: str
    bid_hourly: float
    offer: Offer
    spec: InstanceSpec
    engine_url: Optional[str] = None
    state: str = InstanceState.SCHEDULING
    stopped_by_provider: bool = False
    created_at: float = dataclasses.field(default_factory=time.time)
    running_since: Optional[float] = None
    billed: float = 0.0
    #: What the provider *says* it cost — may deliberately drift from the pool's estimate.
    charge_multiplier: float = 1.0


class FakeProvider:
    """A market you can script.

    Everything that happens to an instance happens because a test asked for it, so a drill
    reads as a sequence of events rather than a sleep.
    """

    interface_version: ClassVar[str] = "1"
    name: ClassVar[str] = "fake"

    def __init__(
        self,
        offers: Optional[list[Offer]] = None,
        capabilities: Optional[ProviderCapabilities] = None,
        engine_urls: Optional[list[str]] = None,
        **_: Any,
    ):
        #: Engines a created instance is pointed at, so a rented host can actually serve in a
        #: test. Empty means an instance with nothing behind it, which is also worth testing.
        self.engine_urls: list[str] = list(engine_urls or [])
        self.capabilities = capabilities or ProviderCapabilities(
            interruptible=True,
            parkable=True,
            same_machine_rebid=True,
            self_terminate=True,
            reports_charges=True,
            direct_port_mapping=False,
        )
        self.offers: list[Offer] = list(offers or [default_offer()])
        self.instances: dict[str, FakeInstance] = {}
        self._ids = itertools.count(1)

        # --- the script ---
        #: Offer ids whose bid will be lost at create time.
        self.lose_bid_on: set[str] = set()
        #: Offer ids that vanish between search and create.
        self.offer_gone_on: set[str] = set()
        #: How many more times destroy() should fail before working.
        self.destroy_failures = 0
        #: Where a fired dead-man timer reports to, so a test can watch it happen.
        self.self_terminate_url = "http://127.0.0.1:9/fake-terminate"
        #: Raise ProviderUnavailable from every call while set.
        self.unavailable = False
        #: Scripted: the account credential is missing or refused.
        self.credential_refused = False
        #: Scripted: the next instance comes up without the start-up material it was given.
        self.drop_startup_material = False
        #: Hand the same engines out again rather than running out of them, for a long run.
        self.reuse_engine_urls = False
        self._engine_turn = 0
        self.calls: list[str] = []

    # --- scripting helpers ---

    def set_floor(self, machine_id: str, floor: float) -> None:
        self.offers = [
            dataclasses.replace(offer, min_bid_hourly=floor)
            if offer.machine_id == machine_id
            else offer
            for offer in self.offers
        ]

    def evict(self, instance_id: str) -> None:
        """The provider outbids the pool: the instance stops without being asked to."""
        instance = self.instances[instance_id]
        instance.state = InstanceState.STOPPED
        instance.stopped_by_provider = True
        instance.running_since = None

    def bring_up(self, instance_id: str) -> None:
        instance = self.instances[instance_id]
        instance.state = InstanceState.RUNNING
        instance.running_since = time.time()

    def strand(self, label: str, machine_id: str = "m-stray") -> str:
        """An instance the pool never intended — what the orphan sweep must catch."""
        instance_id = f"i-{next(self._ids)}"
        self.instances[instance_id] = FakeInstance(
            instance_id=instance_id,
            label=label,
            machine_id=machine_id,
            bid_hourly=0.1,
            offer=default_offer(),
            spec=InstanceSpec(label=label, image="x", disk_gb=10),
            state=InstanceState.RUNNING,
            running_since=time.time(),
        )
        return instance_id

    def set_reported_charges(self, instance_id: str, multiplier: float) -> None:
        """Make the provider's figure run ahead of the pool's estimate."""
        self.instances[instance_id].charge_multiplier = multiplier

    # --- the interface ---

    def _next_engine_url(self) -> Optional[str]:
        if not self.engine_urls:
            return None
        if self.reuse_engine_urls:
            url = self.engine_urls[self._engine_turn % len(self.engine_urls)]
            self._engine_turn += 1
            return url
        return self.engine_urls.pop(0)

    def _guard(self, call: str) -> None:
        self.calls.append(call)
        if self.credential_refused:
            raise ProviderAuthError("fake provider is scripted to refuse the credential")
        if self.unavailable:
            raise ProviderUnavailable("fake provider is scripted unavailable")

    async def list_instances(self, label_prefix: str) -> list[Instance]:
        self._guard("list_instances")
        return [
            Instance(
                instance_id=i.instance_id,
                label=i.label,
                machine_id=i.machine_id,
                raw={"state": i.state},
            )
            for i in self.instances.values()
            if i.label.startswith(label_prefix)
        ]

    async def search_offers(self, query: OfferQuery) -> list[Offer]:
        self._guard("search_offers")
        found = [
            offer
            for offer in self.offers
            if offer.gpu_memory_gb >= query.min_gpu_memory_gb
            and offer.disk_gb >= query.min_disk_gb
            and (query.max_all_in_hourly is None or offer.all_in_hourly <= query.max_all_in_hourly)
            and (offer.verified or not query.verified_only)
            # Two listings, as on a real marketplace: a query gets only the ones it asked for.
            and (query.interruptible if offer.interruptible else query.on_demand)
        ]
        return found[: query.limit]

    async def create(self, offer: Offer, spec: InstanceSpec, bid: Optional[float]) -> Instance:
        self._guard("create")
        if offer.offer_id in self.offer_gone_on:
            raise OfferGone(f"offer {offer.offer_id} is no longer available")
        if offer.offer_id in self.lose_bid_on:
            # A losing bid must fail and leave nothing behind.
            raise BidLost(f"bid {bid} did not win machine {offer.machine_id}")

        instance_id = f"i-{next(self._ids)}"
        self.instances[instance_id] = FakeInstance(
            instance_id=instance_id,
            label=spec.label,
            machine_id=offer.machine_id,
            bid_hourly=bid if bid is not None else offer.all_in_hourly,
            offer=offer,
            spec=dataclasses.replace(spec, onstart=None) if self.drop_startup_material else spec,
            state=InstanceState.RUNNING,
            running_since=time.time(),
            # Handed out in turn and put back, so a long run does not exhaust the market's
            # engines and start creating hosts that can never answer.
            engine_url=self._next_engine_url(),
        )
        return Instance(instance_id=instance_id, label=spec.label, machine_id=offer.machine_id)

    async def set_bid(self, instance: Instance, bid: float) -> None:
        self._guard("set_bid")
        self.instances[instance.instance_id].bid_hourly = bid

    async def start(self, instance: Instance) -> None:
        self._guard("start")
        found = self.instances[instance.instance_id]
        found.state = InstanceState.RUNNING
        found.stopped_by_provider = False
        found.running_since = time.time()

    async def stop(self, instance: Instance) -> None:
        self._guard("stop")
        found = self.instances[instance.instance_id]
        found.state = InstanceState.STOPPED
        found.running_since = None

    async def destroy(self, instance: Instance) -> None:
        self._guard("destroy")
        if self.destroy_failures > 0:
            self.destroy_failures -= 1
            raise ProviderUnavailable("destroy failed; the pool must verify and retry")
        self.instances.pop(instance.instance_id, None)

    async def status(self, instance: Instance) -> InstanceStatus:
        self._guard("status")
        found = self.instances.get(instance.instance_id)
        if found is None:
            return InstanceStatus(state=InstanceState.GONE)
        return InstanceStatus(
            state=found.state,
            stopped_by_provider=found.stopped_by_provider,
            bid_hourly=found.bid_hourly,
            startup_material=bool(found.spec.onstart),
        )

    async def connection(self, instance: Instance) -> ConnectionInfo:
        self._guard("connection")
        found = self.instances[instance.instance_id]
        return ConnectionInfo(
            ssh_host="127.0.0.1",
            ssh_port=22000 + int(found.instance_id[2:]),
            public_url=found.engine_url,
        )

    async def reported_charges(self, instance: Instance) -> Optional[Charges]:
        self._guard("reported_charges")
        if not self.capabilities.reports_charges:
            return None
        found = self.instances.get(instance.instance_id)
        if found is None:
            return None
        hours = (time.time() - found.created_at) / 3600
        return Charges(total=hours * found.bid_hourly * found.charge_multiplier, as_of=time.time())

    async def account(self) -> AccountStatus:
        self._guard("account")
        return AccountStatus(credential_valid=True, credit_remaining=100.0)

    def self_terminate_request(self, action: str = "destroy") -> SelfTerminateRequest:
        # Instance-scoped credential only, exactly as a real provider's would be.
        return SelfTerminateRequest(
            method="DELETE" if action == "destroy" else "PUT",
            url=f"{self.self_terminate_url}?instance=$CONTAINER_ID",
            headers={"Authorization": "Bearer $CONTAINER_API_KEY"},
            body=None if action == "destroy" else '{"state": "stopped"}',
        )


def default_offer(
    offer_id: str = "o-1",
    machine_id: str = "m-1",
    min_bid_hourly: float = 0.10,
    **overrides: Any,
) -> Offer:
    base = dict(
        offer_id=offer_id,
        machine_id=machine_id,
        hardware="FakeGPU 48GB",
        gpus=1,
        gpu_memory_gb=48.0,
        disk_gb=100.0,
        min_bid_hourly=min_bid_hourly,
        all_in_hourly=min_bid_hourly + 0.05,
        on_demand_hourly=0.50,
        storage_hourly=0.005,
        download_per_gb=0.01,
        download_mbps=500.0,
        reliability=0.99,
        verified=True,
        throughput_proxy=100.0,
    )
    base.update(overrides)
    return Offer(**base)  # type: ignore[arg-type]
