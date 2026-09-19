"""Renting: the only part of the pool that can spend money.

docs/spec/supervisor.md §3–§6 and §9. Order within a pass is strict: **release what should not
exist → recover what is broken → acquire what is missing**, and acquisition is refused when any
cap would be crossed.

Every decision a strategy returns is re-checked here. The strategies are advisory; the caps are
not.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from typing import Optional

from ..config import BiddingConfig, OfferPolicy, PoolConfig, RentedConfig, TransportConfig
from ..deadman import heartbeat_command, onstart_script
from ..engines import get_engine
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
)
from ..strategies import (
    Demand,
    HostView,
    LeaseView,
    decide_eviction,
    decide_rent,
    decide_teardown,
    filter_name,
    price_bid,
    rank_offers,
)
from ..transports import SshTunnel, build_ssh_exec_command, run_command

log = logging.getLogger("gpm.renting")


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
    ready_at: Optional[float] = None
    parked_at: Optional[float] = None
    prepared: bool = False
    #: Set by a preparation, so idle release does not reap a host that has no traffic *yet*.
    hold_until: Optional[float] = None
    when_ready: str = "join"
    download_cost: float = 0.0
    last_request_at: Optional[float] = None
    idle_since: Optional[float] = None
    released: bool = False
    resident: frozenset[str] = frozenset()

    @property
    def hours_held(self) -> float:
        return (time.time() - self.created_at) / 3600

    def estimate(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        return (now - self.created_at) / 3600 * self.bid_hourly


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
        self.provider = provider
        self.leases = leases
        self.events = events
        self.spend = spend
        self.hosts: dict[str, RentedHost] = {}
        self.label_prefix = rented.label_prefix or f"gpm/{config.pool.name}/"
        self.overflow_since: Optional[float] = None
        #: When demand last became fully covered. Scale-down waits on this, deliberately
        #: longer than scale-up, so capacity does not flap (spec §9).
        self.overflow_gone_since: Optional[float] = None
        #: Whether the provider has ever returned a charge figure. The narrow cap margin is
        #: only earned once it has.
        self.charges_ever_reported = False
        #: How a command is run on a rented host. Substituted in tests; ssh otherwise.
        self.run_on_host = self._ssh_run
        self.engine = get_engine(config.engine)
        #: Forwards to rented hosts the provider cannot expose directly, keyed by host id.
        self.tunnels: dict[str, SshTunnel] = {}

    # --- money ---

    def margin(self) -> float:
        """A provider that cannot report charges gets a wider margin (spec §4) — and so does
        one that *says* it can but has never actually done so. Found live: the declaration
        was true of the API in general and false of every instance payload, which would have
        left the cap on estimate-only with the narrow margin, silently."""
        if self.provider.capabilities.reports_charges and self.charges_ever_reported:
            return self.rented.spend.cap_safety_margin
        return self.rented.spend.margin_without_reported_charges

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
            )
            try:
                charges = await self.provider.reported_charges(host.instance)
            except ProviderError:
                charges = None
            if charges is not None:
                self.charges_ever_reported = True
                host.reported_spend = charges.total
                self.spend.record(
                    lease_id=lease.lease_id,
                    host_id=host.host_id,
                    source="reported",
                    amount=charges.total,
                )
                drift = charges.total - host.estimated_spend
                if host.estimated_spend > 0 and drift / host.estimated_spend > self.rented.spend.drift_alert:
                    self.events.record(
                        "spend_drift",
                        f"provider reports ${charges.total:.4f} against an estimate of "
                        f"${host.estimated_spend:.4f}",
                        numbers={"reported": charges.total, "estimate": host.estimated_spend},
                        host_id=host.host_id,
                        lease_id=lease.lease_id,
                    )

    # --- the pass ---

    async def pass_once(self, ready_workers_higher_tiers: int, idle_seconds: dict[str, float]) -> None:
        open_leases = self.leases.open_leases()
        await self.beat_deadman_timers()
        await self.sweep_orphans()
        await self.expire_parked()

        for lease in open_leases:
            await self.record_spend(lease)
            if await self.enforce_lease_limits(lease):
                continue

        await self.handle_evictions()
        await self.tear_down(open_leases, idle_seconds, ready_workers_higher_tiers)
        await self.acquire(open_leases, ready_workers_higher_tiers)

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
                await self.destroy(host, reason)

    # --- leases ---

    def open_lease(self, **kwargs) -> Lease:
        """Opening a lease goes through the fleet, because only it knows what the provider can
        do — and a lease longer than the dead-man timer can cover is refused outright."""
        limit = self.max_lease_hours()
        requested = kwargs.get("max_hours")
        if limit is not None and requested is not None and requested > limit:
            raise LeaseRefused(
                f"{self.provider.name} offers no instance-scoped credential, so no dead-man "
                f"timer can be armed; leases there are limited to {limit}h, not {requested}h"
            )
        return self.leases.open(pool_bid_ceiling=self.rented.bidding.bid_ceiling, **kwargs)

    # --- the dead-man timer ---

    def deadman_onstart(self) -> Optional[str]:
        """The start-up script that arms the timer, or None where the provider has no
        instance-scoped credential.

        Without one the pool would have to choose between putting the account credential on a
        machine it does not trust and having no timer at all. It does neither: it refuses long
        leases on that provider instead (spec §7).
        """
        if not self.provider.capabilities.self_terminate:
            return None
        return onstart_script(
            self.provider.self_terminate_command(self.rented.teardown.deadman_action),
            window_s=int(self.rented.teardown.deadman_minutes * 60),
            engine_port=self.rented.engine_port,
            public_key=self.pool_public_key(),
            ssh_user=self.rented.ssh_user,
            extra=self.rented.engine_start,
        )

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

    def instance_env(self) -> dict[str, str]:
        """What makes the engine run this many workers at this context, holding the whole
        model set — set at creation on hosts the pool creates (spec §2.2)."""
        return self.engine.launch_settings(
            workers=self.rented.workers,
            context=self.rented.context_length,
            n_models=len(self.config.pool.model_set),
        )

    def max_lease_hours(self) -> Optional[float]:
        """Without a dead-man timer, nothing on the host can stop it billing — so leases are
        capped short there."""
        if self.provider.capabilities.self_terminate:
            return None
        return self.rented.teardown.max_hours_without_deadman

    async def beat_deadman_timers(self) -> None:
        """Refresh the timestamp on every live host, over the connection already held.

        A failure here is not fatal: the timer is *meant* to fire when the supervisor cannot
        reach the host.
        """
        for host in list(self.hosts.values()):
            if host.released or host.connection is None:
                continue
            try:
                code, output = await self.run_on_host(host, heartbeat_command())
            except Exception as exc:  # noqa: BLE001 - never let a heartbeat take a pass down
                log.warning("heartbeat to %s failed: %s", host.host_id, exc)
                continue
            if code != 0:
                log.warning("heartbeat to %s failed: %s", host.host_id, output.strip())

    async def _open_tunnel(self, host_id: str, connection: ConnectionInfo) -> Optional[str]:
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
            remote_port=self.rented.engine_port,
            known_hosts=self.rented.known_hosts,
        )
        tunnel = SshTunnel(host_id, transport)
        self.tunnels[host_id] = tunnel
        # A host that is still booting refuses SSH for a while; the forward keeps retrying,
        # and the probe simply finds nothing listening until it is up.
        await tunnel.start(wait_s=1.0)
        return tunnel.local_url

    async def close_tunnel(self, host_id: str) -> None:
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

    # --- what would happen (spec §3) ---

    async def plan(self, ready_workers_higher_tiers: int) -> list[dict]:
        """Everything the next pass would do, and nothing else. Spends nothing.

        This is the payoff of strategies being pure functions: the plan is produced by asking
        the same functions the pass would ask, not by a second implementation that can drift.
        """
        steps: list[dict] = []
        open_leases = self.leases.open_leases()

        try:
            instances = await self.provider.list_instances(self.label_prefix)
        except ProviderError as exc:
            steps.append({"step": "observe", "detail": f"the provider is not answering: {exc}"})
            instances = []

        known = {h.instance.instance_id for h in self.hosts.values() if not h.released}
        for instance in instances:
            if instance.instance_id not in known:
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

        lease = next((lease for lease in open_leases if lease.allow_rent), None)
        rented_workers = sum(
            self.rented.workers for h in self.hosts.values() if not h.released and h.state == "ready"
        )
        demand = Demand(
            wanted_workers=lease.workers if lease else 0,
            ready_workers_higher_tiers=ready_workers_higher_tiers,
            rented_workers=rented_workers,
            overflow_age_s=(time.time() - self.overflow_since) if self.overflow_since else 0.0,
            hosts_pending=sum(
                1 for h in self.hosts.values() if not h.released and h.state in ("scheduling", "preparing")
            ),
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
            offers = await self._offers()
            ranked, rejected = rank_offers(
                offers, self.rented.offer_policy, self.rented.bidding, lease.hours_left(), self.rented.model_set_gb
            )
            step["offers_seen"] = len(offers)
            step["offers_rejected"] = {key: value for key, value in list(rejected.items())[:10]}
            if ranked:
                best, best_score = ranked[0]
                bid = price_bid(best, self.rented.bidding, lease.bid_ceiling)
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
    ) -> dict:
        """The offer pipeline, read-only (docs/spec/console-and-control-api.md §2.2).

        Moving a ceiling and watching "4 pass" become "0 pass" is the fastest way to learn what
        a number means — so this runs the *same* filters and the same bid strategy the
        supervisor would, and creates nothing.
        """
        # Unsaved values from the console's form, merged over what is configured. Validated
        # here, so a typo in the form is a 400 and never something the pool acts on.
        policy = self.rented.offer_policy
        bid_config = self.rented.bidding
        if offer_policy:
            policy = OfferPolicy.model_validate({**policy.model_dump(), **offer_policy})
        if bidding:
            bid_config = BiddingConfig.model_validate({**bid_config.model_dump(), **bidding})

        offers = await self._offers(policy)
        ranked, rejected = rank_offers(offers, policy, bid_config, hours, self.rented.model_set_gb)
        by_reason: dict[str, int] = {}
        for reasons in rejected.values():
            # An offer is counted against the first filter that stopped it.
            key = filter_name(reasons[0])
            by_reason[key] = by_reason.get(key, 0) + 1

        accepted = []
        for offer, offer_score in ranked[:10]:
            bid = price_bid(offer, bid_config)
            accepted.append(
                {
                    "machine": offer.machine_id,
                    "hardware": offer.hardware,
                    "gpu_memory_gb": round(offer.gpu_memory_gb, 1),
                    "floor": offer.min_bid_hourly,
                    "would_bid": bid.hourly,
                    "all_in": offer.all_in_hourly,
                    "on_demand": offer.on_demand_hourly,
                    "download_per_gb": offer.download_per_gb,
                    "download_mbps": offer.download_mbps,
                    "storage_hourly": round(offer.storage_hourly, 5),
                    "reliability": offer.reliability,
                    "score": round(offer_score, 3),
                }
            )
        return {
            "seen": len(offers),
            "passed": len(ranked),
            "rejected": len(rejected),
            "rejected_by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
            "best": accepted,
            "policy": {
                "bid_ceiling": bid_config.bid_ceiling,
                "premium": bid_config.premium,
                "on_demand_crossover": bid_config.on_demand_crossover,
            },
        }

    # --- preparing a host on request (spec §8) ---

    async def prepare(
        self,
        *,
        max_spend: float,
        max_hours: float,
        bid_ceiling: Optional[float] = None,
        when_ready: str = "join",
        engine: Optional[object] = None,
        client_for: Optional[object] = None,
    ) -> Optional[RentedHost]:
        """Get a host ready *before* a run, or keep one warm between runs.

        It is its own small lease: it cannot start without a bid ceiling, a dollar cap and a
        time limit, and it borrows authority from no other open lease.
        """
        lease = self.open_lease(
            workers=self.rented.workers,
            max_hours=max_hours,
            max_spend=max_spend,
            allow_rent=True,
            bid_ceiling=bid_ceiling,
        )
        self.events.record(
            "prepare_started",
            f"preparing a host: up to ${max_spend:.2f} over {max_hours}h, then {when_ready}",
            numbers={"max_spend": max_spend, "max_hours": max_hours, "when_ready": when_ready},
            lease_id=lease.lease_id,
        )

        host = await self.restart_parked(lease) or await self.rent_one(lease, ["prepared on request"])
        if host is None:
            self.leases.close(lease.lease_id, "nothing could be prepared")
            return None

        host.prepared = True
        host.hold_until = time.time() + max_hours * 3600
        host.when_ready = when_ready
        return host

    async def load_model_set(self, host: RentedHost, engine, client) -> bool:
        """Pull every tag, then check they are resident **together** — a host that cannot hold
        the whole set does not join the pool."""
        tags = sorted(self.required_tags)
        moved = 0
        for tag in tags:
            result = await engine.pull(client, tag)
            if not result.ok:
                self.events.record(
                    "prepare_failed",
                    f"{host.host_id}: pulling {tag} failed: {result.detail}",
                    numbers={"tag": tag},
                    host_id=host.host_id,
                    lease_id=host.lease_id,
                )
                return False
            moved += result.bytes_total

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
        from ..catalog import variants_for_host

        variants = variants_for_host(
            self.config.pool.model_set,
            self.config.catalog,
            frozenset(self.rented.capabilities),
            self.config.engine,
        )
        return frozenset(group[0].tag for group in variants.values() if group)

    # --- parking (spec §8) ---

    async def restart_parked(self, lease: Lease) -> Optional[RentedHost]:
        """Parked hosts are tried before new offers: no download, minutes instead of tens of
        minutes. It must still win the auction on that machine within the ceilings."""
        for host in self.hosts.values():
            if host.released or host.state != "parked":
                continue
            offers = await self._offers()
            same_machine = next((o for o in offers if o.machine_id == host.offer.machine_id), None)
            if same_machine is None:
                continue
            bid = price_bid(same_machine, self.rented.bidding, lease.bid_ceiling)
            capped = self._cap_bid(bid.hourly, lease)
            if capped is None:
                continue
            try:
                await self.provider.set_bid(host.instance, capped)
                await self.provider.start(host.instance)
            except ProviderError as exc:
                self.events.record(
                    "park_restart_failed",
                    f"{host.host_id} could not be restarted: {exc}; it stays parked",
                    host_id=host.host_id,
                    lease_id=lease.lease_id,
                )
                continue
            host.bid_hourly = capped
            host.lease_id = lease.lease_id
            host.state = "preparing"
            host.parked_at = None
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
            "state": host.state,
            "parked_at": host.parked_at,
            "offer": dataclasses.asdict(dataclasses.replace(host.offer, raw={})),
        }

    async def adopt(self, rows: list) -> list[str]:
        """On start, list what the provider has under this pool's label, take back what the
        database says was intended, and leave the rest for the sweep.

        The provider is the source of truth for what exists; the database for what was
        intended. A restart is not a reason to destroy a good host — nor to keep billing for
        one that is gone.
        """
        try:
            existing = {i.instance_id: i for i in await self.provider.list_instances(self.label_prefix)}
        except ProviderError as exc:
            log.warning("could not list instances to adopt: %s", exc)
            return []

        adopted: list[str] = []
        for row in rows:
            ref = row.provider_ref or {}
            instance_id = ref.get("instance_id")
            if not instance_id or "offer" not in ref:
                continue
            instance = existing.get(instance_id)
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
                offer=Offer(**ref["offer"]),
                bid_hourly=float(ref.get("bid_hourly") or row.hourly_rate or 0.0),
                lease_id=row.lease_id or "",
                created_at=float(ref.get("created_at") or time.time()),
                state=ref.get("state") if ref.get("state") in ("parked", "ready", "preparing") else "preparing",
                ready_at=ref.get("ready_at"),
                prepared=bool(ref.get("prepared")),
                hold_until=ref.get("hold_until"),
                when_ready=ref.get("when_ready") or "join",
                download_cost=float(ref.get("download_cost") or 0.0),
                parked_at=ref.get("parked_at"),
            )
            if host.state == "ready":
                host.state = "preparing"  # readiness is re-verified, never assumed
            try:
                host.connection = await self.provider.connection(instance)
            except ProviderError as exc:
                log.warning("no connection details for %s yet: %s", row.host_id, exc)
            if host.state != "parked" and host.connection is not None:
                host.dial_url = host.connection.public_url or await self._open_tunnel(
                    row.host_id, host.connection
                )
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

    # --- release what should not exist ---

    async def sweep_orphans(self) -> None:
        """The provider is the source of truth for what exists; the database for what was
        intended. Anything carrying this pool's label that we did not intend is swept."""
        try:
            instances = await self.provider.list_instances(self.label_prefix)
        except ProviderError as exc:
            log.warning("could not list instances: %s", exc)
            return

        known = {host.instance.instance_id for host in self.hosts.values() if not host.released}
        for instance in instances:
            if instance.instance_id in known:
                continue
            self.events.record(
                "orphan_swept",
                f"instance {instance.instance_id} carries this pool's label but was never "
                "intended; destroying so it stops billing",
                numbers={"instance": instance.instance_id, "label": instance.label},
            )
            await self._destroy_instance(instance, "orphan")

    # --- recover what is broken ---

    async def handle_evictions(self) -> None:
        for host in list(self.hosts.values()):
            if host.released:
                continue
            try:
                status = await self.provider.status(host.instance)
            except ProviderError as exc:
                log.warning("status for %s failed: %s", host.host_id, exc)
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
                continue

            if status.state != InstanceState.STOPPED:
                continue

            # Stopped, and the pool never asked for that: an eviction (spec §1.2).
            lease = self.leases.get(host.lease_id)
            if lease is None or not lease.is_open:
                await self.destroy(host, "evicted with no open lease")
                continue

            offers = await self._offers()
            same_machine = next((o for o in offers if o.machine_id == host.offer.machine_id), None)
            alternatives = [o for o in offers if o.machine_id != host.offer.machine_id]
            best_alternative = alternatives[0] if alternatives else None

            decision = decide_eviction(
                HostView(
                    host_id=host.host_id,
                    priority=20,
                    busy_workers=0,
                    total_workers=self.rented.workers,
                    idle_seconds=0,
                    bid_hourly=host.bid_hourly,
                    machine_id=host.offer.machine_id,
                    hours_held=host.hours_held,
                ),
                same_machine,
                best_alternative,
                self._lease_view(lease),
                self.rented.bidding,
                self.rented.model_set_gb,
            )
            self.events.record(
                "eviction",
                f"{host.host_id} was outbid: {decision.action}",
                numbers={"reasons": decision.reasons, "bid": decision.bid},
                host_id=host.host_id,
                lease_id=lease.lease_id,
            )

            if decision.action == "rebid" and decision.bid is not None:
                capped = self._cap_bid(decision.bid, lease)
                if capped is None:
                    await self.destroy(host, "re-bid would cross a ceiling")
                    continue
                try:
                    await self.provider.set_bid(host.instance, capped)
                    await self.provider.start(host.instance)
                    host.bid_hourly = capped
                    host.state = "scheduling"
                except ProviderError as exc:
                    log.warning("re-bid for %s failed: %s", host.host_id, exc)
            else:
                await self.destroy(host, f"evicted; {decision.action}")

    # --- acquire what is missing ---

    async def acquire(self, open_leases: list[Lease], ready_workers_higher_tiers: int) -> None:
        lease = next((lease for lease in open_leases if lease.allow_rent), None)
        rented_workers = sum(
            self.rented.workers for h in self.hosts.values() if not h.released and h.state == "ready"
        )
        pending = sum(
            1 for h in self.hosts.values() if not h.released and h.state in ("scheduling", "preparing")
        )
        wanted = lease.workers if lease else 0
        overflow = max(0, wanted - ready_workers_higher_tiers - rented_workers)

        now = time.time()
        if overflow > 0 and self.overflow_since is None:
            self.overflow_since = now
        elif overflow <= 0:
            self.overflow_since = None

        demand = Demand(
            wanted_workers=wanted,
            ready_workers_higher_tiers=ready_workers_higher_tiers,
            rented_workers=rented_workers,
            overflow_age_s=(now - self.overflow_since) if self.overflow_since else 0.0,
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

    def _refuse_for_caps(self, demand: Demand, lease: Optional[Lease]) -> Optional[str]:
        if lease is None:
            return "no lease is open"
        live = [h for h in self.hosts.values() if not h.released]
        if len(live) >= self.config.limits.max_rented_hosts:
            return (
                f"{len(live)} rented hosts already, at the pool's limit of "
                f"{self.config.limits.max_rented_hosts}"
            )
        burn = sum(h.bid_hourly for h in live)
        if burn >= self.config.limits.max_hourly_burn:
            return f"hourly burn ${burn:.3f} is at the ${self.config.limits.max_hourly_burn:.2f} cap"
        left = self.budget_left(lease)
        if left <= 0:
            return f"the lease has ${left:.4f} left once the safety margin is taken off"
        return None

    def _refuse_bid_for_burn(self, bid: float) -> Optional[str]:
        """The cap applies to the burn this bid *would* create, not only to today's."""
        burn = sum(h.bid_hourly for h in self.hosts.values() if not h.released) + bid
        if burn > self.config.limits.max_hourly_burn:
            return (
                f"bidding ${bid:.3f}/h would take the burn to ${burn:.3f}/h, above the "
                f"${self.config.limits.max_hourly_burn:.2f} cap"
            )
        return None

    async def rent_one(self, lease: Lease, reasons: list[str]) -> Optional[RentedHost]:
        offers = await self._offers()
        ranked, rejected = rank_offers(
            offers,
            self.rented.offer_policy,
            self.rented.bidding,
            lease.hours_left(),
            self.rented.model_set_gb,
        )
        if not ranked:
            self.events.record(
                "no_offer",
                f"{len(offers)} offers seen, none passed the policy; staying paused rather "
                "than relaxing a filter"
                if offers
                else "the market returned no offers at all",
                numbers={"seen": len(offers), "rejected": rejected},
                lease_id=lease.lease_id,
            )
            return None

        for offer, offer_score in ranked[: self.rented.bidding.attempts]:
            bid = price_bid(offer, self.rented.bidding, lease.bid_ceiling)
            capped = self._cap_bid(bid.hourly, lease)
            if capped is None:
                continue

            refusal = self._refuse_bid_for_burn(capped)
            if refusal is not None:
                self.events.record(
                    "rent_refused", refusal, numbers={"bid": capped}, lease_id=lease.lease_id
                )
                continue

            host_id = f"rented-{uuid.uuid4().hex[:6]}"
            spec = InstanceSpec(
                label=f"{self.label_prefix}{host_id}",
                image=self.rented.image,
                disk_gb=self.rented.disk_gb,
                env=self.instance_env(),
                # Armed before anything else runs, and carrying no account credential.
                onstart=self.deadman_onstart(),
            )
            try:
                instance = await self.provider.create(offer, spec, capped)
            except (BidLost, OfferGone) as exc:
                self.events.record(
                    "bid_failed",
                    f"bid ${capped:.3f} on {offer.machine_id} did not take: {exc}",
                    numbers={"bid": capped, "offer": offer.offer_id, "score": offer_score},
                    lease_id=lease.lease_id,
                )
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
            )
            connection = await self.provider.connection(instance)
            host.connection = connection
            host.dial_url = connection.public_url or await self._open_tunnel(host_id, connection)
            host.state = "preparing"
            self.hosts[host_id] = host
            self.events.record(
                "rented",
                f"bid ${capped:.3f}/h on {offer.machine_id} ({offer.hardware})",
                numbers={
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
        return None

    def _cap_bid(self, bid: float, lease: Lease) -> Optional[float]:
        """The supervisor's own clamp, applied after the strategy returns — a faulty or
        hostile strategy cannot spend past the ceilings."""
        ceiling = self.rented.bidding.bid_ceiling
        if lease.bid_ceiling is not None:
            ceiling = min(ceiling, lease.bid_ceiling)
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
    ) -> None:
        live = [h for h in self.hosts.values() if not h.released and h.state != "parked"]
        if not live:
            self.overflow_gone_since = None
            return
        lease = next((lease for lease in open_leases if lease.allow_rent), None)
        now = time.time()

        # A host that is not ready yet is neither idle nor surplus: it is capacity on its
        # way, and billing while it comes. It is only given up on when it takes too long.
        preparing = [h for h in live if h.state != "ready"]
        ready = [h for h in live if h.state == "ready"]
        for host in preparing:
            if (now - host.created_at) / 60 >= self.rented.teardown.max_preparing_minutes:
                await self.destroy(
                    host,
                    f"not ready after {self.rented.teardown.max_preparing_minutes:g} minutes",
                )

        wanted = lease.workers if lease else 0
        covered = ready_workers_higher_tiers + len(ready) * self.rented.workers
        if lease is not None and covered >= wanted and wanted > 0:
            self.overflow_gone_since = self.overflow_gone_since or now
        else:
            self.overflow_gone_since = None
        gone_for = (now - self.overflow_gone_since) if self.overflow_gone_since else 0.0

        views = [
            HostView(
                host_id=h.host_id,
                priority=20,
                busy_workers=0,
                total_workers=self.rented.workers,
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
        overflow_gone = gone_for >= self.rented.scale.scale_down_after_s and surplus_workers >= self.rented.workers
        demand = Demand(
            wanted_workers=wanted,
            ready_workers_higher_tiers=ready_workers_higher_tiers,
            rented_workers=len(ready) * self.rented.workers,
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
            if action.host_id in held:
                continue  # prepared and still inside its hold, so it is not idle "yet"
            if "overflow" in action.reasons[0]:
                if reaped_for_overflow >= 1:
                    continue  # one host at a time (spec §9)
                reaped_for_overflow += 1
            if action.action == "park" and self.provider.capabilities.parkable:
                await self.park(host, "; ".join(action.reasons))
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
        try:
            await self.provider.stop(host.instance)
        except ProviderError as exc:
            log.warning("park of %s failed: %s", host.host_id, exc)
            return
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
        ok = await self._destroy_instance(host.instance, reason)
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

    async def _destroy_instance(self, instance: Instance, reason: str) -> bool:
        """A release counts only once the provider's listing no longer shows it."""
        try:
            await self.provider.destroy(instance)
        except ProviderError as exc:
            log.warning("destroy of %s failed (%s); will retry next pass", instance.instance_id, exc)
            self.events.record(
                "destroy_failed",
                f"destroy of {instance.instance_id} failed: {exc}",
                numbers={"instance": instance.instance_id},
            )
            return False
        try:
            remaining = await self.provider.list_instances(self.label_prefix)
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

    async def _offers(self, policy: Optional[OfferPolicy] = None) -> list[Offer]:
        """Ask the market broadly and filter here.

        Pushing the policy into the provider's query would make the market look empty
        whenever a filter bites, and the operator could never see *which* filter — but
        "4 pass, 76 rejected, and here is why" is the whole point of the offer policy.
        """
        try:
            return await self.provider.search_offers(
                OfferQuery(verified_only=(policy or self.rented.offer_policy).verified_only)
            )
        except ProviderError as exc:
            log.warning("offer search failed: %s", exc)
            return []

    def _lease_view(self, lease: Optional[Lease]) -> Optional[LeaseView]:
        if lease is None:
            return None
        return LeaseView(
            lease_id=lease.lease_id,
            allow_rent=lease.allow_rent,
            hours_left=lease.hours_left(),
            dollars_left=self.budget_left(lease),
            bid_ceiling=lease.bid_ceiling,
        )
