"""Workloads, from the supervisor's side (docs/spec/workloads.md, D115).

Creating one works out what it needs — the build, how many answers a host may serve and still
meet the latency target, how many hosts it starts with, which kind of rental, what it would
cost — and opens nothing until the budget is known: typed, or derived and sent back to confirm.
Then one lease bound to the workload, one row, and one key shown once.

Each pass moves every workload along: `preparing` until one of its own hosts is ready, then
`serving`; `ending` once its lease is no longer open, for whatever reason (its caps, its time,
an operator); `ended` once its last host is gone. A workload's key is refused from `ending` on.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from .. import workloads as math_
from ..catalog import variants_for_host
from ..ledger import LeaseRefused
from ..strategies import rank_offers
from ..workload_store import Workload, WorkloadStore

if TYPE_CHECKING:
    from .service import Supervisor


class WorkloadRefused(Exception):
    """A workload that cannot be created, extended or ended as asked — said in words."""


@dataclasses.dataclass(frozen=True)
class WorkloadRequest:
    name: str
    model: str
    latency_s: float
    parallel: int
    hours: float
    max_spend: Optional[float] = None
    profile: Optional[str] = None
    kind: str = "roi"


KINDS = ("roi", "on_demand", "interruptible")


class Workloads:
    def __init__(self, supervisor: "Supervisor"):
        self.supervisor = supervisor
        self.store = WorkloadStore(supervisor.db)
        #: One creation at a time: a plan awaits the market, and two creations in flight would
        #: both pass the caps and the name check that the other was about to fill.
        self._creating = asyncio.Lock()

    @property
    def config(self) -> Any:
        return self.supervisor.config

    @property
    def fleet(self) -> Any:
        return self.supervisor.fleet

    # --- what a workload would be ---

    def _build(self, req: WorkloadRequest) -> tuple[dict[str, str], int]:
        """The build its hosts fetch, and how many cards each copy spans: a profile's, where one
        is named, otherwise the catalog's first build rented machines can run (D98's rule)."""
        rented = self.config.rented
        entry = self.config.catalog.get(req.model)
        if req.model not in self.config.pool.model_set and not (entry and entry.workloads_only):
            raise WorkloadRefused(
                f"{req.model!r} is neither in the pool's model set nor a workloads-only model in its catalog; "
                "add it — as workloads-only, and the shared hosts never fetch it"
            )
        if req.profile:
            models = rented.model_profiles.get(req.profile)
            if models is None:
                raise WorkloadRefused(f"no model profile {req.profile!r}; known: {sorted(rented.model_profiles)}")
            if req.model not in models:
                raise WorkloadRefused(f"profile {req.profile!r} does not hold {req.model!r}")
            return {req.model: models[req.model]}, rented.cards_per_copy(req.profile)
        variants = variants_for_host(
            [req.model], self.config.catalog, frozenset(rented.capabilities), self.config.rented_engine()
        )
        group = variants.get(req.model) or ()
        if not group:
            raise WorkloadRefused(
                f"{req.model!r} has no build rented hosts can run ({self.config.rented_engine()}); add one to the catalog"
            )
        return {req.model: group[0].tag}, 1

    def _check_request(self, req: WorkloadRequest) -> None:
        if self.fleet is None or self.config.rented is None:
            raise WorkloadRefused("this pool has no rented capacity: a workload rents its own hosts")
        if not math_.valid_name(req.name):
            raise WorkloadRefused("a workload's name is lower-case letters, digits and '-', up to 40, starting with a letter or digit")
        existing = self.store.get(req.name)
        if existing is not None:
            raise WorkloadRefused(
                f"workload {req.name!r} already exists ({existing.state}); names are never reused, so logs stay unambiguous"
            )
        if req.kind not in KINDS:
            raise WorkloadRefused(f"kind is one of {', '.join(KINDS)}")
        if req.parallel < 1:
            raise WorkloadRefused("parallel is how many answers at once: at least 1")
        if req.hours <= 0:
            raise WorkloadRefused("hours must be above zero")
        if req.latency_s <= 0:
            raise WorkloadRefused("the latency target must be above zero seconds")
        if req.max_spend is not None and req.max_spend <= 0:
            raise WorkloadRefused("a dollar cap must be above zero")

    def _reserved_hosts(self) -> int:
        """Hosts other workloads still mean to rent: their start, less what they have. Counted
        against the pool's host limit, so two workloads created back to back cannot both pass
        and the second starve (the review's finding 8)."""
        reserved = 0
        for workload in self.store.active():
            if workload.state not in ("preparing", "serving"):
                continue
            have = len(self.fleet.hosts_of(workload.name))
            reserved += max(0, workload.hosts_at_start - have)
        return reserved

    async def plan(self, req: WorkloadRequest) -> dict[str, Any]:
        """Everything creating it would do, and nothing else: nothing is opened or rented."""
        self._check_request(req)
        builds, cards = self._build(req)
        now = time.time()
        sketch = Workload(
            name=req.name, model=req.model, builds=builds, latency_s=req.latency_s, parallel=req.parallel,
            kind=req.kind, lease_id="", state="preparing", workers_per_host=0, hosts_at_start=0,
            plan={"cards_per_copy": cards}, created_at=now, updated_at=now,
        )
        fleet = self.fleet
        models = (req.model,)
        policy = fleet.policy_for(models, builds, cards_per_copy=cards)
        policy = policy.model_copy(update={"min_reliability": max(policy.min_reliability, self.config.workloads.min_reliability)})
        kinds = {"on_demand": "on_demand", "interruptible": "interruptible"}.get(req.kind, "both")
        offers = await fleet._offers(policy, kinds)
        ranked, rejected = rank_offers(
            offers, fleet._policy_with_avoided(policy), self.config.rented.bidding, req.hours,
            fleet.model_set_gb(models, builds), history=fleet.machine_history(), history_cfg=self.config.rented.history,
        )
        plan: dict[str, Any] = {
            "name": req.name, "model": req.model, "build": builds[req.model], "cards_per_copy": cards,
            "latency_s": req.latency_s, "parallel": req.parallel, "hours": req.hours, "kind": req.kind,
            "offers_seen": len(offers), "offers_passed": len(ranked), "refused": None, "reasons": [],
        }
        if not ranked:
            plan["refused"] = (
                f"no machine on the market passes this workload's search ({len(offers)} seen"
                + (f"; {fleet.last_offer_error}" if fleet.last_offer_error else "") + ")"
            )
            plan["rejected_by_reason"] = _count_reasons(rejected)
            return plan

        if req.kind == "roi":
            ordered, why = fleet.expected_costs(ranked, req.hours, sketch, models, builds)
            first = next(o for o, _ in ranked if o.offer_id == ordered[0].offer_id)
            first_cost = ordered[0]
            plan["reasons"] += why[:1]
        else:
            first = ranked[0][0]
            ordered, _ = fleet.expected_costs([ranked[0]], req.hours, sketch, models, builds)
            first_cost = ordered[0]

        card_workers, card_why = fleet._workers_for_card(first)
        at = fleet.latency_sizing(first.hardware, sketch, ceiling=card_workers)
        per_host = min(card_workers, at.workers) if at.workers is not None else card_workers
        hosts = math_.hosts_at_start(req.parallel, per_host)
        ready_h = math_.time_to_ready_hours(fleet.model_set_gb(models, builds), first.download_mbps,
                                            self.config.workloads.engine_load_s)
        derived = math_.derived_budget(hosts, req.hours, first_cost.hourly)
        budget = req.max_spend if req.max_spend is not None else derived
        plan.update({
            "workers_per_host": per_host,
            "workers_measured": at.measured and at.workers is not None,
            "sizing": [card_why, *at.reasons],
            "latency_curve": {str(k): v for k, v in at.curve.items()},
            "hosts_at_start": hosts,
            "first_host": {
                "offer_id": first.offer_id, "machine": first.machine_id, "hardware": first.hardware,
                "kind": "interruptible" if first.interruptible else "on_demand",
                "hourly": round(first_cost.hourly, 4), "expected_per_worker_hour": first_cost.per_worker_hour,
                "reasons": first_cost.reasons,
            },
            "minutes_to_serve": round(ready_h * 60, 1),
            "max_spend": round(budget, 2),
            "budget_derived": req.max_spend is None,
            "derived_budget": derived,
        })
        entry = self.config.catalog.get(req.model)
        if (self.config.pool.models_per_host == "all" and not self.config.rented.rent_profiles
                and not (entry and entry.workloads_only)):
            # The model must be in the pool's set, and with every rented host holding the whole
            # set, the shared workload's rented hosts would fetch it too.
            plan["reasons"].append(
                "note: this model is in the pool's shared set, and this pool rents shared hosts for its "
                "whole set, so they fetch it too; marking it workloads-only keeps it to workloads"
            )
        burn_now = sum(h.bid_hourly for h in fleet.hosts.values() if not h.released) + fleet.volume_burn()
        plan.update({
            "hourly_total": round(hosts * first_cost.hourly, 4),
            "pool_burn_now": round(burn_now, 4),
            "pool_burn_after": round(burn_now + hosts * first_cost.hourly, 4),
            "pool_burn_cap": self.config.limits.max_hourly_burn,
            "borrow_while_starting": self._shared_serves(req.model),
        })
        plan["refused"] = self._cap_refusal(hosts, first_cost.hourly, budget, req.hours)
        return plan

    def _shared_serves(self, model: str) -> bool:
        """Can a starting workload borrow for this model: does a ready shared host serve it?"""
        if self.config.workloads.borrow_share <= 0:
            return False
        for host in self.supervisor.hosts.values():
            if host.state.value == "ready" and model in self.config.models_held_by(host.config):
                return True
        return any(
            h.state == "ready" and model in self.fleet.models_of(h) for h in self.fleet.hosts_of(None)
        ) if self.fleet is not None else False

    def full(self) -> Optional[str]:
        """Why no workload at all could be created now, without asking the market; or None."""
        open_now = [w for w in self.store.active() if w.state in ("preparing", "serving")]
        if len(open_now) >= self.config.workloads.max_open:
            return f"{len(open_now)} workloads are running, at the pool's limit of {self.config.workloads.max_open}"
        live = len([h for h in self.fleet.hosts.values() if not h.released])
        reserved = self._reserved_hosts()
        if self.config.limits.max_rented_hosts - live - reserved <= 0:
            return (f"the pool has room for 0 more hosts: {self.config.limits.max_rented_hosts} at most, "
                    f"{live} rented, {reserved} reserved by workloads still starting")
        limits = self.config.limits
        if limits.max_hourly_burn is not None:
            burn = sum(h.bid_hourly for h in self.fleet.hosts.values() if not h.released)
            if burn >= limits.max_hourly_burn:
                return f"the pool already burns ${burn:.2f}/h, at its ${limits.max_hourly_burn:.2f} cap"
        return None

    def _cap_refusal(self, hosts: int, hourly: float, budget: float, hours: float) -> Optional[str]:
        limits = self.config.limits
        longest = self.fleet.max_lease_hours()
        if longest is not None and hours > longest:
            return (f"{self.fleet.provider.name} has no dead-man timer, so a lease there runs at most "
                    f"{longest:g}h, not {hours:g}h")
        full = self.full()
        if full is not None:
            return full
        live = len([h for h in self.fleet.hosts.values() if not h.released])
        room = limits.max_rented_hosts - live - self._reserved_hosts()
        if hosts > room:
            return (
                f"it starts on {hosts} host(s), and the pool has room for {max(0, room)}: {limits.max_rented_hosts} "
                f"at most, {live} rented, {self._reserved_hosts()} reserved by workloads still starting"
            )
        if limits.max_hourly_burn is not None:
            burn = sum(h.bid_hourly for h in self.fleet.hosts.values() if not h.released)
            if burn + hosts * hourly > limits.max_hourly_burn:
                return (
                    f"its {hosts} host(s) at ${hourly:.3f}/h would take the pool's burn to "
                    f"${burn + hosts * hourly:.2f}/h, above the ${limits.max_hourly_burn:.2f} cap"
                )
        if budget < hourly * hosts:
            return f"a budget of ${budget:.2f} does not pay for one hour of its {hosts} host(s) at ${hourly:.3f}/h"
        return None

    # --- creating, changing, ending ---

    async def create(
        self, req: WorkloadRequest, confirm_max_spend: Optional[float] = None, *,
        key_hash: Optional[str] = None, provisioner: Optional[str] = None,
        idle_end_minutes: Optional[float] = None, may_borrow: bool = True,
        csr: Optional[str] = None, ca: Any = None,
    ) -> tuple[Workload, Optional[str], dict, Optional[str]]:
        """Create it: (the workload, its key — None where the program made its own and sent only
        the hash —, the plan, a signed client certificate where a signing request came)."""
        async with self._creating:
            # Planned inside the lock, so its name check and its caps see every workload created
            # before it — including one whose plan was awaiting the market a moment ago.
            plan = await self.plan(req)
            if plan["refused"]:
                raise WorkloadRefused(plan["refused"])
            budget = plan["max_spend"]
            if provisioner is not None and plan["budget_derived"]:
                raise WorkloadRefused("a program must say what it will spend: send max_spend")
            if csr is not None and ca is None:
                raise WorkloadRefused("this pool has no client CA to sign a certificate with")
            plan["may_borrow"] = bool(may_borrow)
            if plan["budget_derived"] and (confirm_max_spend is None or abs(confirm_max_spend - budget) > 0.005):
                raise WorkloadRefused(
                    f"no budget was typed; the derived one is ${budget:.2f}. Send it back as confirm_max_spend to accept it"
                )
            now = time.time()
            lease_id = f"lease-{uuid.uuid4().hex[:8]}"
            workload = Workload(
                name=req.name, model=req.model, builds={req.model: plan["build"]}, latency_s=req.latency_s,
                parallel=req.parallel, kind=req.kind, lease_id=lease_id, state="preparing",
                workers_per_host=plan["workers_per_host"], hosts_at_start=plan["hosts_at_start"], plan=plan,
                created_at=now, updated_at=now, ends_at=now + req.hours * 3600,
                provisioner=provisioner, idle_end_minutes=idle_end_minutes,
            )
            # The row first: a name taken meanwhile fails here, before any spending authority
            # exists. The lease second, under the id the row already names.
            try:
                self.store.create(workload)
            except Exception as exc:  # noqa: BLE001 - a unique name, lost to another creation
                raise WorkloadRefused(f"workload {req.name!r} could not be recorded: {exc}") from exc
            try:
                # Through the fleet: a lease longer than the dead-man timer covers is refused there.
                lease = self.fleet.open_lease(
                    workers=req.parallel, max_hours=req.hours, max_spend=budget, allow_rent=True,
                    workload=req.name, lease_id=lease_id,
                )
            except LeaseRefused as exc:
                self.store.set_state(req.name, "ended")
                raise WorkloadRefused(str(exc)) from exc
            self.store.set_ends_at(req.name, lease.opened_at + req.hours * 3600)
            certificate = None
            if csr is not None:
                from ..certs import CertRefused

                try:
                    certificate, print_ = ca.sign(csr, req.name, req.hours)
                except CertRefused as exc:
                    self.end(req.name, "its certificate could not be signed")
                    raise WorkloadRefused(str(exc)) from exc
                self.store.set_cert_fingerprint(req.name, print_)
            if key_hash is not None:
                # The program made the key and sent its hash: the plaintext never reaches the pool.
                key_id, key = self.store.add_key_hash(req.name, key_hash), None
            else:
                key_id, key = self.store.mint_key(req.name)
        self.supervisor.events.record(
            "workload_created",
            f"workload {req.name}: {req.model} at {req.parallel} at once, {req.latency_s:g}s p95, for "
            f"{req.hours:g}h — {plan['hosts_at_start']} host(s) at {plan['workers_per_host']} each, up to ${budget:.2f}",
            numbers={"workload": req.name, "key_id": key_id, "plan": plan, "provisioner": provisioner},
            lease_id=lease.lease_id,
        )
        return self.store.get(req.name), key, plan, certificate

    def get(self, name: str) -> Workload:
        workload = self.store.get(name)
        if workload is None:
            raise WorkloadRefused(f"no workload {name!r}")
        return workload

    def extend(self, name: str, *, hours: Optional[float] = None, max_spend: Optional[float] = None,
               confirm: bool = False, confirm_hours: bool = False) -> Workload:
        """More hours and/or a new dollar cap: a raise of its lease, typed twice as D49 requires."""
        workload = self.get(name)
        if workload.state not in ("preparing", "serving"):
            raise WorkloadRefused(f"workload {name!r} is {workload.state}; create a new one")
        lease = self.supervisor.leases.get(workload.lease_id)
        if lease is None or not lease.is_open:
            raise WorkloadRefused(f"workload {name!r}'s lease is closed")
        if hours is not None and hours <= 0:
            raise WorkloadRefused("add a positive number of hours")
        # Each raise is confirmed on its own (D49): typing the budget again says nothing about
        # the hours, nor the other way round.
        if hours and not confirm_hours:
            raise WorkloadRefused("adding hours raises the time limit: type the hours again to confirm")
        if hours and workload.cert_fingerprint:
            raise WorkloadRefused(f"workload {name!r} is reached with a client certificate signed for its hours; "
                                  "hours cannot be added past it — raise its budget, or make a new workload")
        if max_spend is not None and max_spend > lease.max_spend and not confirm:
            raise WorkloadRefused("raising the dollar cap must be confirmed: type the new value again")
        longest = self.fleet.max_lease_hours() if self.fleet is not None else None
        if hours and longest is not None and lease.max_hours + hours > longest:
            raise WorkloadRefused(f"{self.fleet.provider.name} has no dead-man timer, so a lease there runs at most "
                                  f"{longest:g}h, not {lease.max_hours + hours:g}h")
        try:
            changed = self.supervisor.leases.tighten(
                lease.lease_id,
                max_hours=lease.max_hours + hours if hours else None,
                max_spend=max_spend,
                loosen=True,
            )
        except LeaseRefused as exc:
            raise WorkloadRefused(str(exc)) from exc
        self.store.set_ends_at(name, changed.opened_at + changed.max_hours * 3600)
        # The hold on every host follows the lease it was extended for (D49).
        for host in self.fleet.hosts_of(name):
            if host.hold_until is not None:
                host.hold_until = changed.opened_at + changed.max_hours * 3600
        self.supervisor.events.record(
            "workload_extended",
            f"workload {name}: now {changed.max_hours:g}h and up to ${changed.max_spend:.2f}",
            numbers={"workload": name, "max_hours": changed.max_hours, "max_spend": changed.max_spend},
            lease_id=lease.lease_id,
        )
        return self.get(name)

    def end(self, name: str, reason: str = "ended by the operator") -> Workload:
        workload = self.get(name)
        if workload.state in ("ending", "ended"):
            return workload
        lease = self.supervisor.leases.get(workload.lease_id)
        if lease is not None and lease.is_open:
            self.supervisor.leases.close(lease.lease_id, f"workload {name} {reason}")
        self._ending(workload, reason)
        return self.get(name)

    def rotate_key(self, name: str) -> tuple[str, str]:
        workload = self.get(name)
        if workload.state not in ("preparing", "serving"):
            raise WorkloadRefused(f"workload {name!r} is {workload.state}; its keys reach nothing")
        if workload.provisioner is not None:
            raise WorkloadRefused(
                f"workload {name!r} was made by a program, which holds its only key; it makes a new workload instead"
            )
        grace = self.config.workloads.rotation_grace_minutes * 60
        key_id, key = self.store.rotate_key(name, grace)
        self.supervisor.events.record(
            "workload_key_rotated",
            f"workload {name}: a new key; the old one works for {grace / 60:g} more minutes",
            numbers={"workload": name, "key_id": key_id},
        )
        return key_id, key

    def _ending(self, workload: Workload, reason: str) -> None:
        # The keys are not expired: the router refuses every request of a workload that is not
        # preparing or serving, and a key it still recognises lets it say why — "this workload
        # has ended" — where an expired one could only say "unknown key" (workloads.md §3).
        self.store.set_state(workload.name, "ending")
        self.supervisor.events.record(
            "workload_ending", f"workload {workload.name}: {reason}; its hosts drain and its key reaches nothing",
            numbers={"workload": workload.name}, lease_id=workload.lease_id,
        )

    # --- every pass ---

    def _idle_past(self, workload: Workload, now: float) -> Optional[float]:
        """Minutes a program's workload has gone unused past its cutoff, or None (D117).

        Only once serving, and counted from the later of its last request and the moment it
        began serving: a program waiting for a slow preparation is not idle. A program that dies
        while its workload prepares is bounded by the lease, as an operator's is."""
        if workload.provisioner is None or workload.state != "serving":
            return None
        cutoff = (workload.idle_end_minutes or 15.0) * 60
        last = self.supervisor.db.query(
            "SELECT MAX(ts) AS t FROM request_log WHERE workload = ?", (workload.name,))[0]["t"]
        start = max(last or 0.0, workload.serving_at or workload.updated_at)
        return (now - start - cutoff) / 60 if now - start > cutoff else None

    async def advance(self) -> list[Workload]:
        """Move every workload along, and return the ones the fleet should serve this pass."""
        now = time.time()
        for workload in self.store.active():
            if self._idle_past(workload, now) is not None:
                # A program that stopped using its workload ends it — or a crashed one would keep
                # its floor, and spend its whole budget (D117).
                self.end(workload.name, f"unused for {(workload.idle_end_minutes or 15):g} minutes")
        for workload in self.store.active():
            lease = self.supervisor.leases.get(workload.lease_id)
            hosts = self.fleet.hosts_of(workload.name) if self.fleet is not None else []
            if workload.state in ("preparing", "serving") and (lease is None or not lease.is_open):
                self._ending(workload, f"its lease closed ({lease.closed_reason if lease else 'gone'})")
                continue
            if workload.state == "preparing" and any(h.state == "ready" for h in hosts):
                self.store.set_state(workload.name, "serving")
                self.supervisor.events.record(
                    "workload_serving", f"workload {workload.name}: its first host is ready; borrowing stops",
                    numbers={"workload": workload.name}, lease_id=workload.lease_id,
                )
            elif workload.state == "ending" and not hosts:
                self.store.set_state(workload.name, "ended")
                if self.fleet is not None:
                    await self.fleet.delete_volumes(workload.name)
                self.supervisor.events.record(
                    "workload_ended", f"workload {workload.name}: its last host is gone",
                    numbers={"workload": workload.name}, lease_id=workload.lease_id,
                )
        return [w for w in self.store.active() if w.state in ("preparing", "serving", "ending")]

    def view(self, workload: Workload) -> dict[str, Any]:
        """What the operator sees: the workload, its hosts, its spend against its cap, and how
        its answers compare with its target — never a key."""
        lease = self.supervisor.leases.get(workload.lease_id)
        hosts = self.fleet.hosts_of(workload.name) if self.fleet is not None else []
        spent = self.fleet.lease_spend(lease)[0] if (self.fleet is not None and lease is not None) else 0.0
        rows = self.supervisor.db.query(
            "SELECT latency_ms FROM request_log WHERE workload = ? AND outcome = 'ok' AND latency_ms IS NOT NULL "
            "ORDER BY id DESC LIMIT 2000",
            (workload.name,),
        )
        refused = self.supervisor.db.query(
            "SELECT COUNT(*) AS n FROM request_log WHERE workload = ? AND status_code >= 500", (workload.name,),
        )[0]["n"]
        borrowed = self.supervisor.db.query(
            "SELECT COUNT(*) AS n FROM request_log WHERE workload = ? AND borrowed = 1", (workload.name,),
        )[0]["n"]
        latencies = [r["latency_ms"] / 1000 for r in rows]
        return {
            **{k: v for k, v in workload.as_dict().items() if k != "plan"},
            "plan": workload.plan,
            # While it starts: are its requests served on shared hosts, or refused until then?
            "borrowing": workload.state == "preparing" and self._shared_serves(workload.model),
            "hosts": [
                {"host_id": h.host_id, "state": h.state, "workers": h.workers, "hardware": h.offer.hardware,
                 "kind": "interruptible" if h.interruptible else "on_demand", "hourly": round(h.bid_hourly, 4)}
                for h in hosts
            ],
            "lease": {
                "lease_id": workload.lease_id,
                "open": bool(lease and lease.is_open),
                "max_spend": lease.max_spend if lease else None,
                "spent": round(spent, 4),
                "hours_left": round(lease.hours_left(), 3) if lease and lease.is_open else 0.0,
            },
            "answers": {
                "count": len(latencies),
                "p95_s": round(math_.p95(latencies), 3) if latencies else None,
                "meets_target": (math_.p95(latencies) <= workload.latency_s) if latencies else None,
                "refused": int(refused),
                "borrowed": int(borrowed),
            },
            "keys": self.store.keys(workload.name),
        }


def _count_reasons(rejected: dict[str, list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reasons in rejected.values():
        for reason in reasons:
            name = reason.split(":", 1)[0]
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
