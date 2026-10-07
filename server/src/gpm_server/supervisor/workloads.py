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
import math
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .. import workloads as math_
from ..catalog import variants_for_host
from ..ledger import LeaseRefused
from ..strategies import rank_offers
from ..workload_store import Group, ModelTarget, Workload, WorkloadStore

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
    #: Several models, each with its own target (D118); empty for one: `model`, `latency_s`
    #: and `parallel` above. With several, those three are the first model's.
    targets: tuple[ModelTarget, ...] = ()
    #: auto (the cheaper), together (every host holds every model) or apart (a group per model).
    placement: str = "auto"

    @classmethod
    def of(cls, name: str, targets: Sequence[ModelTarget], hours: float, **rest: Any) -> "WorkloadRequest":
        targets = tuple(targets)
        if not targets:
            raise WorkloadRefused("a workload names at least one model")
        first = targets[0]
        return cls(name=name, model=first.model, latency_s=first.latency_s, parallel=first.parallel, hours=hours,
                   targets=targets if len(targets) > 1 else (), **rest)

    @property
    def all_targets(self) -> tuple[ModelTarget, ...]:
        return self.targets or (ModelTarget(self.model, self.latency_s, self.parallel),)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(t.model for t in self.all_targets)

    @property
    def total_parallel(self) -> int:
        return sum(t.parallel for t in self.all_targets)


KINDS = ("roi", "on_demand", "interruptible")
PLACEMENTS = ("auto", "together", "apart")
#: The most answers at once one model may ask for: far past any pool's hosts, and a bound on the
#: work sizing it takes.
MAX_PARALLEL = 100_000


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise WorkloadRefused(f"{name} must be a number above zero")
    return float(value)


def _whole(value: Any, name: str) -> int:
    number = _finite(value, name)
    if number != int(number):
        raise WorkloadRefused(f"{name} must be a whole number")
    if number > MAX_PARALLEL:
        raise WorkloadRefused(f"{name} is at most {MAX_PARALLEL}")
    return int(number)


def parse_targets(body: dict) -> tuple[list[ModelTarget], str]:
    """A request's models and placement, from untrusted JSON (D118): one model — `model`,
    `latency_s`, `parallel` — or several as `models: [{model, latency_s, parallel}, ...]`. Every
    number is refused in words, never truncated. Shared by the control API and by programs."""
    placement = body.get("placement") or "auto"
    if placement not in PLACEMENTS:
        raise WorkloadRefused(f"placement is one of {', '.join(PLACEMENTS)}")
    if body.get("models") is not None:
        if body.get("model") is not None:
            raise WorkloadRefused("send model, or models — not both")
        listed = body["models"]
        if not isinstance(listed, list) or not listed or not all(isinstance(m, dict) for m in listed):
            raise WorkloadRefused("models is a list of {model, latency_s, parallel}")
    else:
        listed = [body]
    targets = []
    for entry in listed:
        model = entry.get("model")
        if not isinstance(model, str) or not model.strip():
            raise WorkloadRefused("each model is named: model is required")
        if entry.get("latency_s") is None or entry.get("parallel") is None:
            raise WorkloadRefused(f"{model}: latency_s and parallel are required")
        targets.append(ModelTarget(model.strip(), _finite(entry["latency_s"], f"{model}: latency_s"),
                                   _whole(entry["parallel"], f"{model}: parallel")))
    return targets, placement


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
        """The build each model's hosts fetch, and how many cards each copy spans: a profile's,
        where one is named, otherwise the catalog's first build rented machines can run (D98's
        rule)."""
        rented = self.config.rented
        for model in req.models:
            entry = self.config.catalog.get(model)
            if model not in self.config.pool.model_set and not (entry and entry.workloads_only):
                raise WorkloadRefused(
                    f"{model!r} is neither in the pool's model set nor a workloads-only model in its catalog; "
                    "add it — as workloads-only, and the shared hosts never fetch it"
                )
        if req.profile:
            models = rented.model_profiles.get(req.profile)
            if models is None:
                raise WorkloadRefused(f"no model profile {req.profile!r}; known: {sorted(rented.model_profiles)}")
            missing = [m for m in req.models if m not in models]
            if missing:
                raise WorkloadRefused(f"profile {req.profile!r} does not hold {', '.join(repr(m) for m in missing)}")
            return {m: models[m] for m in req.models}, rented.cards_per_copy(req.profile)
        variants = variants_for_host(
            list(req.models), self.config.catalog, frozenset(rented.capabilities), self.config.rented_engine()
        )
        builds = {}
        for model in req.models:
            group = variants.get(model) or ()
            if not group:
                raise WorkloadRefused(
                    f"{model!r} has no build rented hosts can run ({self.config.rented_engine()}); add one to the catalog"
                )
            builds[model] = group[0].tag
        return builds, 1

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
        if req.placement not in PLACEMENTS:
            raise WorkloadRefused(f"placement is one of {', '.join(PLACEMENTS)}")
        targets = req.all_targets
        most = self.config.workloads.max_models
        if len(targets) > most:
            raise WorkloadRefused(f"a workload serves at most {most} models, not {len(targets)}")
        if len({t.model for t in targets}) != len(targets):
            raise WorkloadRefused("each model is named once")
        for t in targets:
            if not isinstance(t.parallel, int) or isinstance(t.parallel, bool) or t.parallel < 1:
                raise WorkloadRefused(f"{t.model}: parallel is how many answers at once: at least 1")
            if t.parallel > MAX_PARALLEL:
                raise WorkloadRefused(f"{t.model}: parallel is at most {MAX_PARALLEL}")
            if not (isinstance(t.latency_s, (int, float)) and math.isfinite(t.latency_s) and t.latency_s > 0):
                raise WorkloadRefused(f"{t.model}: the latency target must be above zero seconds")
        if not (math.isfinite(req.hours) and req.hours > 0):
            raise WorkloadRefused("hours must be above zero")
        if req.max_spend is not None and not (math.isfinite(req.max_spend) and req.max_spend > 0):
            raise WorkloadRefused("a dollar cap must be above zero")

    def _reserved_hosts(self) -> int:
        """Hosts other workloads still mean to rent: their start, less what they have — per group
        (D118). Counted against the pool's host limit, so two workloads created back to back
        cannot both pass and the second starve (the review's finding 8)."""
        reserved = 0
        for workload in self.store.active():
            if workload.state not in ("preparing", "serving"):
                continue
            for group in workload.groups:
                have = len(self.fleet.hosts_of(workload.name, group.key))
                reserved += max(0, group.hosts_at_start - have)
        return reserved

    async def plan(self, req: WorkloadRequest) -> dict[str, Any]:
        """Everything creating it would do, and nothing else: nothing is opened or rented.

        With several models, both placements are priced from one market search — every host
        holding every model, and a group of hosts per model — and the cheaper kept (D118)."""
        self._check_request(req)
        targets = req.all_targets
        builds, cards = self._build(req)
        now = time.time()
        sketch = Workload(
            name=req.name, model=targets[0].model, builds=builds, latency_s=targets[0].latency_s,
            parallel=req.total_parallel, kind=req.kind, lease_id="", state="preparing", workers_per_host=0,
            hosts_at_start=0, plan={"cards_per_copy": cards}, created_at=now, updated_at=now, targets=targets,
        )
        fleet = self.fleet
        models = req.models
        kinds = {"on_demand": "on_demand", "interruptible": "interruptible"}.get(req.kind, "both")
        # One search for every placement: the market is asked broadly and filtered here, so each
        # group re-prices and ranks the same offers for its own disk and card (the review's finding 7).
        # The loosest needs of any group it may price: the provider is asked to filter (D123), and
        # a search sized for every model on one card would hide the cards that hold one model.
        candidates = [models] + ([(m,) for m in models] if len(models) > 1 else [])
        needs = [fleet.policy_for(g, {m: builds[m] for m in g}, cards_per_copy=cards) for g in candidates]
        search = needs[0].model_copy(update={
            "min_gpu_memory_gb": min(p.min_gpu_memory_gb for p in needs),
            "min_disk_gb": min(p.min_disk_gb for p in needs),
        })
        offers = await fleet._offers(search, kinds)
        # Priced only on what it could rent: a provider with no dead-man timer serves no lease
        # longer than it may run, so its offers would quote a price the workload never pays (D129).
        short = [o for o in offers if (fleet.lease_hours_on(o.connection) or float("inf")) < req.hours]
        offers = [o for o in offers if o not in short]
        plan: dict[str, Any] = {
            "name": req.name, "model": targets[0].model, "build": builds[targets[0].model], "cards_per_copy": cards,
            "latency_s": targets[0].latency_s, "parallel": req.total_parallel, "hours": req.hours, "kind": req.kind,
            "models": [t.as_dict() for t in targets], "builds": builds,
            "offers_seen": len(offers), "refused": None, "reasons": [],
        }
        if short and not offers:
            limit = min(fleet.lease_hours_on(o.connection) for o in short)
            # Said to applications too: how long, never where the pool rents.
            plan["refused"] = (f"a workload here runs at most {limit:g}h, not {req.hours:g}h: where it would rent, "
                               "no dead-man timer could stop a host billing")
            plan["offers_passed"] = 0
            return plan
        options: dict[str, Any] = {}
        if len(targets) == 1 or req.placement in ("auto", "together"):
            options["together"] = self._price(req, sketch, [models], offers, builds, cards)
        if len(targets) > 1 and req.placement in ("auto", "apart"):
            options["apart"] = self._price(req, sketch, [(m,) for m in models], offers, builds, cards)
        # The pool's caps are judged per placement, before one is chosen: `auto` must not pick the
        # cheaper and then be refused, when the other would have passed (the review's finding).
        for option in options.values():
            if option.get("refused") is None:
                option["derived"] = math_.derived_budget(1, req.hours, option["hourly"])
                budget_here = req.max_spend if req.max_spend is not None else option["derived"]
                option["cap_refused"] = self._cap_refusal(option["hosts"], option["hourly"] / max(1, option["hosts"]),
                                                          budget_here, req.hours)
        placed = {name: o for name, o in options.items() if o.get("refused") is None}
        allowed = {name: o for name, o in placed.items() if o.get("cap_refused") is None} or placed
        if len(targets) > 1:
            plan["placements"] = {
                name: {"expected": o.get("expected"), "hosts": o.get("hosts"), "hourly": o.get("hourly"),
                       "refused": o.get("refused") or o.get("cap_refused")}
                for name, o in options.items()
            }
        if not placed:
            said = [f"{name}: {o['refused']}" for name, o in options.items()] if len(options) > 1 else [
                next(iter(options.values()))["refused"]]
            plan["refused"] = "; ".join(said)
            first_refused = next(iter(options.values()))
            plan["offers_passed"] = max(o.get("offers_passed", 0) for o in options.values())
            if first_refused.get("rejected_by_reason"):
                plan["rejected_by_reason"] = first_refused["rejected_by_reason"]
            return plan
        if req.placement != "auto" or len(targets) == 1:
            chosen_name, why = next(iter(allowed)), [f"{next(iter(allowed))}, as asked"]
        else:
            chosen, why = math_.choose_placement(allowed.get("together") and math_.Placement(
                "together", allowed["together"]["expected"], allowed["together"]["hosts"], []),
                allowed.get("apart") and math_.Placement("apart", allowed["apart"]["expected"], allowed["apart"]["hosts"], []))
            chosen_name = chosen.name
            skipped = [f"{name} is refused by the pool's caps ({o['cap_refused']})" for name, o in placed.items()
                       if name not in allowed or (o.get("cap_refused") and name != chosen_name)]
            if len(placed) > len(allowed) and skipped:
                why = [f"{chosen_name}: the only placement the pool's caps allow; {skipped[0]}"]
        option = placed[chosen_name]
        if len(targets) > 1:
            plan["reasons"].append(f"placement: {why[0]}")
        groups = option["groups"]
        first = groups[0]
        hosts = option["hosts"]
        derived = option["derived"]
        budget = req.max_spend if req.max_spend is not None else derived
        plan["reasons"] += [r for g in groups for r in g["kind_reasons"]]
        plan.update({
            "placement": chosen_name,
            "groups": [{k: g[k] for k in ("models", "builds", "hosts_at_start", "workers_per_host", "caps",
                                          "cards_per_copy", "first_host", "sizing", "latency_curves",
                                          "minutes_to_serve", "hourly_total", "model_volume")} for g in groups],
            "offers_passed": option["offers_passed"],
            "workers_per_host": first["workers_per_host"],
            "workers_measured": all(g["measured"] for g in groups),
            "sizing": [s for g in groups for s in g["sizing"]],
            "latency_curve": first["latency_curves"].get(targets[0].model, {}),
            "hosts_at_start": hosts,
            "first_host": first["first_host"],
            "minutes_to_serve": max(g["minutes_to_serve"] for g in groups),
            "max_spend": round(budget, 2),
            "budget_derived": req.max_spend is None,
            "derived_budget": derived,
        })
        for model in models:
            entry = self.config.catalog.get(model)
            if (self.config.pool.models_per_host == "all" and not self.config.rented.rent_profiles
                    and not (entry and entry.workloads_only)):
                # The model must be in the pool's set, and with every rented host holding the whole
                # set, the shared workload's rented hosts would fetch it too.
                plan["reasons"].append(
                    f"note: {model} is in the pool's shared set, and this pool rents shared hosts for its "
                    "whole set, so they fetch it too; marking it workloads-only keeps it to workloads"
                )
        burn_now = sum(h.bid_hourly for h in fleet.hosts.values() if not h.released) + fleet.volume_burn()
        borrow = {m: self._shared_serves(m) for m in models}
        plan.update({
            "hourly_total": round(option["hourly"], 4),
            "pool_burn_now": round(burn_now, 4),
            "pool_burn_after": round(burn_now + option["hourly"], 4),
            "pool_burn_cap": self.config.limits.max_hourly_burn,
            "borrow_while_starting": any(borrow.values()),
            "borrow_by_model": borrow,
        })
        plan["refused"] = option["cap_refused"]
        return plan

    def _price(self, req: WorkloadRequest, sketch: Workload, placement: list[tuple[str, ...]], offers: list,
               builds: dict[str, str], cards: int) -> dict[str, Any]:
        """One placement priced: each group's first host, its sizing and its hosts, and the whole
        placement's expected cost over the hours — or why it cannot be placed."""
        fleet = self.fleet
        groups = []
        for models in placement:
            group_builds = {m: builds[m] for m in models}
            policy = fleet.policy_for(models, group_builds, cards_per_copy=cards)
            policy = policy.model_copy(update={"min_reliability": max(policy.min_reliability, self.config.workloads.min_reliability)})
            ranked, rejected = rank_offers(
                [o.priced_for(policy.min_disk_gb) for o in offers], fleet._policy_with_avoided(policy),
                self.config.rented.bidding, req.hours, fleet.model_set_gb(models, group_builds),
                history=fleet.machine_history(), history_cfg=self.config.rented.history,
            )
            named = ", ".join(models)
            if not ranked:
                return {
                    "refused": (f"no machine on the market passes the search for {named} ({len(offers)} seen"
                                + (f"; {fleet.last_offer_error}" if fleet.last_offer_error else "") + ")"),
                    "rejected_by_reason": _count_reasons(rejected), "offers_passed": 0,
                }
            if len(models) > 1:
                # Every model on one host within its own target, or not at all (D118).
                passed = len(ranked)
                ranked = [pair for pair in ranked if fleet.capacity_on(pair[0], sketch, models, group_builds) > 0]
                if not ranked:
                    return {"refused": (f"{passed} machine(s) pass the search, and none holds {named} together "
                                        "within every model's target"), "offers_passed": passed}
            # As the rental will choose (D131): the least expected cost per worker-hour, of any kind
            # the request allows, across every provider.
            ordered, why = fleet.expected_costs(ranked, req.hours, sketch, models, group_builds)
            kind_reasons = why[:1]
            if not ordered:
                return {"refused": f"no machine holds {named} within every target", "offers_passed": len(ranked)}
            first = next(o for o, _ in ranked
                         if (o.connection, o.offer_id, o.interruptible)
                         == (ordered[0].connection, ordered[0].offer_id, ordered[0].interruptible))
            first_cost = ordered[0]
            if len(placement) > 1 or len(models) > 1:
                kind_reasons = [f"{named}: {r}" for r in kind_reasons]
            card_workers, card_why = fleet._workers_for_card(first)
            sizing, curves, measured = [card_why], {}, True
            per_model = {}
            for model in models:
                target = sketch.target(model)
                at = fleet.model_at_latency(first.hardware, group_builds[model], target.latency_s, ceiling=card_workers)
                per_model[model] = min(card_workers, at.workers) if at.workers is not None else card_workers
                sizing += [f"{model}: {r}" if len(sketch.targets) > 1 else r for r in at.reasons]
                curves[model] = {str(k): v for k, v in at.curve.items()}
                measured = measured and at.measured and at.workers is not None
            if len(models) == 1:
                per_host = per_model[models[0]]
                hosts = math_.hosts_at_start(sketch.target(models[0]).parallel, per_host)
                caps: dict[str, int] = {}
            else:
                split = math_.split_together({m: sketch.target(m).parallel for m in models}, per_model)
                if split is None:
                    return {"refused": f"one answer of each of {named} already fills a {first.hardware}", "offers_passed": 0}
                per_host, hosts, caps = split.workers, split.hosts, dict(split.caps)
                sizing.append("split: " + ", ".join(f"{m} {caps[m]} of {per_model[m]}" for m in models)
                              + f" at once per host, {split.load:.0%} of the card")
            ready_h = math_.time_to_ready_hours(fleet.model_set_gb(models, group_builds), first.download_mbps,
                                                self.config.workloads.engine_load_s)
            # Each host priced at its own share of the group: an eviction of one of H hosts takes
            # 1/H of the capacity away, not all of it (the review's finding). Priced as one of
            # the whole group, a placement with more hosts looked dearer than it is.
            if hosts > 1:
                # Against the same offers as before, so the capacity an eviction loses is priced at
                # the same on-demand rate; only this host's share of the group changes.
                shared, _ = fleet.expected_costs(ranked, req.hours, sketch, models, group_builds,
                                                 fixed_workers=per_host, have=per_host * (hosts - 1))
                first_cost = next((c for c in shared if (c.connection, c.offer_id, c.interruptible)
                                   == (first.connection, first.offer_id, first.interruptible)), first_cost)
            # Where the first host's provider keeps models between hosts (D139), the volume it would
            # make: said before anything is spent, in the workload's own hours and budget.
            volume = None
            if fleet.keeps_models(first.connection) and first.locations:
                size = fleet._volume_size(models, group_builds)
                volume_hourly = size * (first.volume_per_gb_hourly or 0.0)
                volume = {"connection": first.connection, "size_gb": size, "hourly": round(volume_hourly, 6),
                          "over_hours": round(volume_hourly * req.hours, 4)}
                kind_reasons = [*kind_reasons,
                                f"keeps a {size} GB model volume at {first.connection}, in the first host's data "
                                f"center — ${volume_hourly:.4f}/h, ${volume_hourly * req.hours:.2f} over "
                                f"{req.hours:g} h, in the budget; deleted when the workload ends"]
            groups.append({
                "models": list(models), "builds": group_builds, "hosts_at_start": hosts, "workers_per_host": per_host,
                "model_volume": volume,
                "caps": caps, "cards_per_copy": cards, "sizing": sizing, "latency_curves": curves, "measured": measured,
                "kind_reasons": kind_reasons, "offers_passed": len(ranked),
                "first_host": {
                    "offer_id": first.offer_id, "machine": first.machine_id, "hardware": first.hardware,
                    "kind": "interruptible" if first.interruptible else "on_demand",
                    "hourly": round(first_cost.hourly, 4), "expected_per_worker_hour": first_cost.per_worker_hour,
                    "reasons": first_cost.reasons,
                },
                "hourly_total": round(hosts * first_cost.hourly + (volume["hourly"] if volume else 0.0), 4),
                "expected": first_cost.expected * hosts + (volume["over_hours"] if volume else 0.0),
                "minutes_to_serve": round(ready_h * 60, 1),
            })
        return {
            "refused": None, "groups": groups, "hosts": sum(g["hosts_at_start"] for g in groups),
            # The hosts and their model volumes (D139): a volume bills whether or not a host has it.
            "hourly": sum(g["hosts_at_start"] * g["first_host"]["hourly"]
                          + (g["model_volume"]["hourly"] if g.get("model_volume") else 0.0) for g in groups),
            "expected": round(sum(g["expected"] for g in groups), 4),
            "offers_passed": min(g["offers_passed"] for g in groups),
        }

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
        live = len([h for h in self.fleet.hosts.values() if not h.released]) + len(self.fleet.held_back_rows())
        reserved = self._reserved_hosts()
        if self.config.limits.max_rented_hosts - live - reserved <= 0:
            return (f"the pool has room for 0 more hosts: {self.config.limits.max_rented_hosts} at most, "
                    f"{live} rented, {reserved} reserved by workloads still starting")
        limits = self.config.limits
        if limits.max_hourly_burn is not None:
            burn = (sum(h.bid_hourly for h in self.fleet.hosts.values() if not h.released) + self.fleet.volume_burn()
                    + self.fleet.held_back_burn())
            if burn >= limits.max_hourly_burn:
                return f"the pool already burns ${burn:.2f}/h, at its ${limits.max_hourly_burn:.2f} cap"
        return None

    def _cap_refusal(self, hosts: int, hourly: float, budget: float, hours: float) -> Optional[str]:
        limits = self.config.limits
        longest = self.fleet.max_lease_hours()
        if longest is not None and hours > longest:
            # Said to applications too (a program's plan): how long, never where the pool rents.
            return (f"a workload here runs at most {longest:g}h, not {hours:g}h: where it would rent, no "
                    "dead-man timer could stop a host billing")
        full = self.full()
        if full is not None:
            return full
        live = len([h for h in self.fleet.hosts.values() if not h.released]) + len(self.fleet.held_back_rows())
        room = limits.max_rented_hosts - live - self._reserved_hosts()
        if hosts > room:
            return (
                f"it starts on {hosts} host(s), and the pool has room for {max(0, room)}: {limits.max_rented_hosts} "
                f"at most, {live} rented, {self._reserved_hosts()} reserved by workloads still starting"
            )
        if limits.max_hourly_burn is not None:
            # Volumes bill too, and the fleet counts them when it rents: a plan that left them out
            # would pass here and never rent.
            burn = (sum(h.bid_hourly for h in self.fleet.hosts.values() if not h.released) + self.fleet.volume_burn()
                    + self.fleet.held_back_burn())
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
            if csr is not None:
                # Read before anything is opened: a request the pool cannot sign spends nothing.
                from ..certs import CertRefused

                try:
                    ca.public_key_of(csr)
                except CertRefused as exc:
                    raise WorkloadRefused(str(exc)) from exc
            plan["may_borrow"] = bool(may_borrow)
            if plan["budget_derived"] and (confirm_max_spend is None or abs(confirm_max_spend - budget) > 0.005):
                raise WorkloadRefused(
                    f"no budget was typed; the derived one is ${budget:.2f}. Send it back as confirm_max_spend to accept it"
                )
            now = time.time()
            lease_id = f"lease-{uuid.uuid4().hex[:8]}"
            targets = req.all_targets
            groups = tuple(Group(
                models=tuple(g["models"]), builds=dict(g["builds"]), hosts_at_start=int(g["hosts_at_start"]),
                workers_per_host=int(g["workers_per_host"]), caps=dict(g["caps"]), cards_per_copy=int(g["cards_per_copy"]),
            ) for g in plan["groups"])
            workload = Workload(
                name=req.name, model=targets[0].model, builds=dict(plan["builds"]), latency_s=min(t.latency_s for t in targets),
                parallel=req.total_parallel, kind=req.kind, lease_id=lease_id, state="preparing",
                workers_per_host=plan["workers_per_host"], hosts_at_start=plan["hosts_at_start"], plan=plan,
                created_at=now, updated_at=now, ends_at=now + req.hours * 3600,
                provisioner=provisioner, idle_end_minutes=idle_end_minutes,
                targets=targets, placement=plan["placement"], groups=groups,
            )
            # The row first: a name taken meanwhile fails here, before any spending authority
            # exists. The lease second, under the id the row already names.
            try:
                self.store.create(workload)
            except Exception as exc:  # noqa: BLE001 - a unique name, lost to another creation
                raise WorkloadRefused(f"workload {req.name!r} could not be recorded: {exc}") from exc
            try:
                # Through the fleet: a lease longer than the dead-man timer covers is refused there.
                # Its workers are every model's answers at once: the lease bounds its groups together.
                lease = self.fleet.open_lease(
                    workers=req.total_parallel, max_hours=req.hours, max_spend=budget, allow_rent=True,
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
                except Exception as exc:  # noqa: BLE001 - whatever failed, the lease must not outlive it
                    self.end(req.name, "its certificate could not be signed")
                    raise WorkloadRefused(str(exc) if isinstance(exc, CertRefused)
                                          else f"its certificate could not be signed: {type(exc).__name__}") from exc
                self.store.set_cert_fingerprint(req.name, print_)
            if key_hash is not None:
                # The program made the key and sent its hash: the plaintext never reaches the pool.
                key_id, key = self.store.add_key_hash(req.name, key_hash), None
            else:
                key_id, key = self.store.mint_key(req.name)
        self.supervisor.events.record(
            "workload_created",
            f"workload {req.name}: "
            + "; ".join(f"{t.model} at {t.parallel} at once, {t.latency_s:g}s p95" for t in req.all_targets)
            + f", for {req.hours:g}h — {plan['hosts_at_start']} host(s)"
            + (f" placed {plan['placement']}" if len(req.all_targets) > 1 else f" at {plan['workers_per_host']} each")
            + f", up to ${budget:.2f}",
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
            raise WorkloadRefused(f"{self.fleet._without_deadman()}, so a lease runs at most "
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

        Only once serving — with several models, once every group has a host (D118) — and counted
        from the later of its last request and the moment it began serving: a program waiting for
        a slow preparation is not idle, and polls for its state write nothing a cutoff could read.
        A program that dies while its workload prepares is bounded by the lease, as an operator's
        is, whether none of its groups came up or only some did."""
        if workload.provisioner is None or workload.state != "serving":
            return None
        # A request still being answered is use: the log is written when one finishes, and an
        # answer can take longer than the cutoff.
        counters = self.supervisor.counters.all()
        hosts = self.fleet.hosts_of(workload.name) if self.fleet is not None else []
        mine = [counters[h.host_id] for h in hosts if h.host_id in counters]
        if any(c.busy > 0 for c in mine):
            return None
        cutoff = (workload.idle_end_minutes or 15.0) * 60
        last = self.supervisor.db.query(
            "SELECT MAX(ts) AS t FROM request_log WHERE workload = ?", (workload.name,))[0]["t"]
        started = max((c.last_request_at or 0.0 for c in mine), default=0.0)
        start = max(last or 0.0, started, workload.serving_at or workload.updated_at)
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
            ready_groups = [g for g in workload.groups
                            if any(h.state == "ready" for h in hosts if g.holds(h.models))]
            if workload.state == "preparing" and len(ready_groups) == len(workload.groups):
                # Serving once every model has a host of its own: until then a model still coming
                # up borrows where it may (D118).
                self.store.set_state(workload.name, "serving")
                self.supervisor.events.record(
                    "workload_serving",
                    f"workload {workload.name}: "
                    + ("its first host is ready" if len(workload.groups) == 1 else "every group of its hosts is ready")
                    + "; borrowing stops",
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
            "SELECT latency_ms, model_requested FROM request_log WHERE workload = ? AND outcome = 'ok' "
            "AND latency_ms IS NOT NULL ORDER BY id DESC LIMIT 2000",
            (workload.name,),
        )
        several = len(workload.targets) > 1
        refused = self.supervisor.db.query(
            "SELECT COUNT(*) AS n FROM request_log WHERE workload = ? AND status_code >= 500", (workload.name,),
        )[0]["n"]
        borrowed = self.supervisor.db.query(
            "SELECT COUNT(*) AS n FROM request_log WHERE workload = ? AND borrowed = 1", (workload.name,),
        )[0]["n"]
        latencies = [r["latency_ms"] / 1000 for r in rows]
        by_model = {}
        for target in workload.targets:
            # Each model's own window: a busy model must not crowd a quiet one out of the answers.
            mine = [r["latency_ms"] / 1000 for r in self.supervisor.db.query(
                "SELECT latency_ms FROM request_log WHERE workload = ? AND model_requested = ? AND outcome = 'ok' "
                "AND latency_ms IS NOT NULL ORDER BY id DESC LIMIT 2000", (workload.name, target.model))] if several \
                else [r["latency_ms"] / 1000 for r in rows]
            by_model[target.model] = {
                "count": len(mine), "latency_s": target.latency_s, "parallel": target.parallel,
                "p95_s": round(math_.p95(mine), 3) if mine else None,
                "meets_target": (math_.p95(mine) <= target.latency_s) if mine else None,
                "cap_per_host": workload.cap(target.model),
            }
        ready_models = {m for h in hosts if h.state == "ready" for m in h.models}
        borrowing_models = [
            m for m in workload.models
            if workload.state == "preparing" and (workload.plan or {}).get("may_borrow", True)
            and m not in ready_models and self._shared_serves(m)
        ]
        return {
            **{k: v for k, v in workload.as_dict().items() if k != "plan"},
            "plan": workload.plan,
            # While it starts: are its requests served on shared hosts, or refused until then?
            # Per model (D118): borrowing while none of its own hosts holding it is ready.
            "borrowing_models": borrowing_models,
            "borrowing": bool(borrowing_models),
            "hosts": [
                {"host_id": h.host_id, "state": h.state, "workers": h.workers, "hardware": h.offer.hardware,
                 "kind": "interruptible" if h.interruptible else "on_demand", "hourly": round(h.bid_hourly, 4),
                 "priced": ("bid" if h.offer.bidding else "spot") if h.interruptible else "on_demand",
                 "connection": h.connection_name,
                 "models": list(h.models),
                 # Where its models came from: its workload's model volume, a sibling, the hub (D139).
                 "models_source": h.models_source}
                for h in hosts
            ],
            # Its model volumes, where models are kept between hosts (D139).
            "volumes": [
                {"volume_id": v.volume_id, "connection": v.connection, "location": v.location, "size_gb": v.size_gb,
                 "hourly": v.hourly, "state": v.state, "filler": v.filler, "models": sorted((v.builds or {}).keys()),
                 "spent": round(v.hourly * max(0.0, time.time() - v.created_at) / 3600, 4)}
                for v in self.store.volumes(workload.name) if v.location is not None
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
                # With several models, each against its own target (D118): one mixed p95 against
                # the tightest target says nothing true about either.
                "meets_target": (
                    all(m["meets_target"] for m in by_model.values() if m["meets_target"] is not None)
                    if any(m["meets_target"] is not None for m in by_model.values()) else None
                ) if several else ((math_.p95(latencies) <= workload.latency_s) if latencies else None),
                "refused": int(refused),
                "borrowed": int(borrowed),
                "by_model": by_model,
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
