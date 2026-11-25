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
import statistics
import time
import uuid
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from .. import workloads as math_
from ..catalog import variants_for_host
from ..ledger import LeaseRefused
from ..strategies import rank_offers
from ..workload_store import Group, ModelTarget, Workload, WorkloadStore, steady_of

if TYPE_CHECKING:
    from .service import Supervisor


class WorkloadRefused(Exception):
    """A workload that cannot be created, extended or ended as asked — said in words."""


def kind_matches(kind: str, interruptible: bool) -> bool:
    """Whether a rented machine is of a workload's kind (`machines=`): `roi` takes either."""
    return kind == "roi" or (kind == "interruptible") == interruptible


def parse_policy(body: dict) -> dict[str, Any]:
    """The spending policy a create or a change carries (D165), checked — `{}` where it carries
    none: `bursts.hosted` and `bursts.rented`. A workload's steady capacity is its `parallel` at
    its `latency_s`, as it always was, so `steady` is refused in words rather than accepted and
    ignored."""
    from pydantic import ValidationError

    from ..config import HostedBurst, RentedBurst

    if "steady" in body:
        raise WorkloadRefused("a workload's steady capacity is its parallel at its latency_s: send those, and "
                              "the pool works out the hosts that meet them")
    if body.get("serve") is not None:
        # A serving plan (S19, D198): today's bursts are its shorthand, so the two never mix.
        from ..serving_plan import PlanRefused, compile_policy, parse_serve

        if body.get("bursts") is not None:
            raise WorkloadRefused("send bursts or serve, not both: bursts are shorthand for a serving plan's when_full")
        try:
            return compile_policy(parse_serve(body["serve"]))
        except PlanRefused as exc:
            raise WorkloadRefused(str(exc)) from None
    bursts = body.get("bursts")
    if bursts is None:
        return {}
    if not isinstance(bursts, dict):
        raise WorkloadRefused("bursts is an object: {\"hosted\": {...}, \"rented\": {...}}")
    unknown = sorted(set(bursts) - {"hosted", "rented"})
    if unknown:
        raise WorkloadRefused(f"bursts are hosted and rented, not {unknown}")
    policy: dict[str, Any] = {}
    for name, model in (("hosted", HostedBurst), ("rented", RentedBurst)):
        raw = bursts.get(name)
        if raw is None:
            continue
        try:
            burst = model.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg")).removeprefix("Value error, ")
            raise WorkloadRefused(f"bursts.{name}{'.' + where if where else ''}: {message}") from None
        policy[name] = burst.model_dump(exclude_none=True)
    return policy


def _switch_times(rule: Any, hours: float) -> int:
    """How many times a rule's steady switch can begin within a lease's hours, at most: once per
    day for a window, an end-of-day hold or a day's spending; once per eviction window; once for
    the lease's spending; once a month for a month's."""
    days = math.ceil(hours / 24) + 1
    bounds = []
    if "between" in rule.when or rule.until == "end_of_day":
        bounds.append(days)
    if "evictions" in rule.when:
        bounds.append(math.ceil(hours / rule.when["evictions"]["within_h"]) + 1)
    if "spent_over" in rule.when:
        spent = rule.when["spent_over"]
        bounds.append(1 if spent["budget"] == "lease" else days if spent.get("period") == "day"
                      else math.ceil(hours / (24 * 28)) + 1)
    return max(1, min(bounds)) if bounds else days


def _switches(policy: Optional[Mapping[str, Any]]) -> frozenset[str]:
    """Where a policy's rules may switch steady (S19 5b): `local`, `fixed`, `rented:<kind>`."""
    from ..serving_plan import PlanRefused, plan_of

    try:
        plan = plan_of(policy or {})
    except PlanRefused:
        return frozenset()
    return frozenset(r.switch for r in (plan.rules if plan is not None else ()) if r.switch)


def describe_policy(policy: dict[str, Any]) -> str:
    """A policy in words, for the event log."""
    if policy.get("serve"):
        from ..serving_plan import PlanRefused, describe, parse_serve

        try:
            return "serving plan: " + describe(parse_serve(policy["serve"]))
        except PlanRefused:
            return "a serving plan this supervisor cannot read"
    said = []
    hosted, rented = policy.get("hosted"), policy.get("rented")
    if hosted:
        said.append(f"hosted APIs after {hosted['after_s']:g}s, within {hosted['budget']}")
    if rented:
        said.append(f"up to {rented['max_hosts']} more host(s) once a burst has lasted {rented['after_min']:g} min, "
                    f"within {rented['budget']}")
    return "; ".join(said) or "no bursts"


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
    #: Its spending policy (D165), as `parse_policy` returns it: a rented burst is paid from the
    #: same lease, so the budget a plan derives carries it.
    policy: Optional[dict] = None

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

    @property
    def on_machines(self) -> tuple[ModelTarget, ...]:
        """Its models that run on machines rented for it; the rest are served by hosted APIs
        alone (D165), and none where it runs on workers the operator reserved (S19)."""
        if steady_of(self.policy) != "rented":
            return ()
        return tuple(t for t in self.all_targets if t.on_machines)

    @property
    def hosted_alone(self) -> tuple[str, ...]:
        return tuple(t.model for t in self.all_targets if not t.on_machines)


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
        if entry.get("parallel") is None:
            raise WorkloadRefused(f"{model}: parallel is required — 0 for a model served by hosted APIs alone")
        if entry["parallel"] == 0 and not isinstance(entry["parallel"], bool):
            # Served by hosted APIs alone (D165): no machine, so no latency a machine is sized for.
            latency = entry.get("latency_s")
            if latency not in (None, 0):
                _finite(latency, f"{model}: latency_s")
            targets.append(ModelTarget(model.strip(), float(latency or 0.0), 0))
            continue
        if entry.get("latency_s") is None:
            raise WorkloadRefused(f"{model}: latency_s is required where parallel is above 0")
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
        engine = self.config.rented_engine()
        can = self.config.engine_kinds(engine)
        traits = self.config.build_traits()
        carried = [frozenset(i.capabilities) for i in rented.images]
        builds = {}
        for model in req.models:
            # Only a build the rented engine serves (D142), from an image that carries what it
            # needs (D146) — else the workload would rent hosts that could never answer it.
            group = [
                v for v in variants.get(model) or ()
                if (traits.get(v.tag, (frozenset({"chat", "embed"}), None))[0] & can)
                and (not self.config.image_needs([v.tag]) or any(self.config.image_needs([v.tag]) <= c for c in carried))
            ]
            if not group:
                raise WorkloadRefused(
                    f"{model!r} has no build rented hosts can serve ({engine}, from the images listed); add one to the catalog"
                )
            builds[model] = group[0].tag
        return builds, 1

    def _check_request(self, req: WorkloadRequest) -> None:
        if req.on_machines and (self.fleet is None or self.config.rented is None):
            raise WorkloadRefused("this pool has no rented capacity: a workload rents its own hosts — "
                                  "parallel 0 serves a model by hosted APIs alone")
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
            if not isinstance(t.parallel, int) or isinstance(t.parallel, bool) or t.parallel < 0:
                raise WorkloadRefused(f"{t.model}: parallel is how many answers at once on machines: 0 or more")
            if t.parallel > MAX_PARALLEL:
                raise WorkloadRefused(f"{t.model}: parallel is at most {MAX_PARALLEL}")
            if t.parallel and self.config.hosted_only(t.model):
                raise WorkloadRefused(
                    f"{t.model!r} has no build a machine runs: it is served by hosted APIs alone — send parallel 0, "
                    "with a hosted burst")
            if t.parallel and not (isinstance(t.latency_s, (int, float)) and math.isfinite(t.latency_s)
                                   and t.latency_s > 0):
                raise WorkloadRefused(f"{t.model}: the latency target must be above zero seconds")
        if not (math.isfinite(req.hours) and req.hours > 0):
            raise WorkloadRefused("hours must be above zero")
        if req.max_spend is not None and not (math.isfinite(req.max_spend) and req.max_spend > 0):
            raise WorkloadRefused("a dollar cap must be above zero")
        if not req.on_machines and req.max_spend is not None:
            why = ("it runs on the operator's machines, on workers reserved for it"
                   if steady_of(req.policy) != "rented" else "every model is served by hosted APIs alone")
            raise WorkloadRefused(f"it rents no machine — {why} — so it has no max_spend: its hosted burst's "
                                  "budget is what it spends")
        steady = steady_of(req.policy)
        if steady != "rented":
            # Its machines are the operator's own of that kind: one must hold each model it runs.
            from ..serving_plan import HOST_KINDS

            kinds = HOST_KINDS[steady]
            for t in req.all_targets:
                if t.on_machines and not any(h.kind in kinds and not h.disabled and t.model in self.config.models_held_by(h)
                                             for h in self.config.hosts):
                    raise WorkloadRefused(f"{t.model!r}: no {steady} machine of this pool holds it, so steady {steady} "
                                          "would never serve it")

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

    async def plan(self, req: WorkloadRequest, may_borrow: bool = True) -> dict[str, Any]:
        """Everything creating it would do, and nothing else: nothing is opened or rented — and
        its serving plan read back (§14.6, D203): `serving` holds the readback, the timeline, the
        conditions that can change it, the most it can spend, the warnings and the providers."""
        plan = await self._plan(req)
        plan["serving"] = self._serving_view(req, plan, may_borrow)
        return plan

    def _serving_view(self, req: WorkloadRequest, plan: dict[str, Any], may_borrow: bool = True) -> dict[str, Any]:
        """The plan check's reading of a serving plan (§14.6): from the plan and facts the pool
        already has — the card class's measured time to ready (D69), the configured machines and
        hosted APIs. Nothing here spends or asks the market."""
        from ..serving_plan import (
            HOST_KINDS,
            conditions,
            from_bursts,
            most_hosted,
            option_warnings,
            plan_of,
            readback,
            timeline,
        )

        serving = plan_of(req.policy or {}) or from_bursts(req.policy or {},
                                                           may_borrow and self.config.workloads.borrow_share > 0)
        now = time.time()
        hosted_alone = len(req.hosted_alone) == len(req.all_targets)
        rented = serving.steady == "rented" and bool(req.on_machines)
        ready = self._ready_minutes(plan) if rented and plan.get("groups") else None
        workers = sum(int(g.get("hosts_at_start") or 0) * int(g.get("workers_per_host") or 0)
                      for g in plan.get("groups") or [])
        if serving.steady != "rented":
            workers = int(serving.reserve.get(serving.steady, 0))
        phases = timeline(serving, start=now, hours=req.hours, ready_min=ready, hosts=int(plan.get("hosts_at_start") or 0),
                          workers=workers, hourly=float(plan.get("hourly_total") or 0.0))
        hosted = most_hosted(serving, now, req.hours)
        cap = float(plan.get("max_spend") or 0.0)
        warnings = option_warnings(serving)
        models = [t.model for t in req.on_machines]
        # A steady local or fixed source holding none of them is refused before this (check_policy).
        for source in serving.until_ready:
            kind = source.type
            if kind not in ("local", "fixed", "shared") or not models:
                continue
            if kind == "shared":
                holds = [m for m in models if self._shared_holds(m)]
            else:
                holds = [m for m in models if any(h.kind in HOST_KINDS[kind] and not h.disabled
                                                  and m in self.config.models_held_by(h) for h in self.config.hosts)]
            if not holds:
                warnings.append(f"{source.name}: no {kind} machine of this pool holds {', '.join(models)} — it serves "
                                "none of this workload's requests")
        if ready and serving.until_ready and "start" in serving.during and serving.max_min is not None \
                and serving.max_min < ready["p90"]:
            warnings.append(f"max_min {serving.max_min:g} min is shorter than the 90th-percentile time to ready of its "
                            f"machines ({ready['p90']:g} min, {ready['card']}): the stop-gap may end before they serve")
        said = readback(serving, kind=req.kind)
        if hosted_alone:
            # Every model by hosted APIs alone (D165): nothing is rented, whatever its fields mean.
            said = [f"Served by hosted APIs alone ({', '.join(req.hosted_alone)}): nothing is rented, and it serves "
                    "from the start.", *[line for line in said if not line.startswith(("Steady", "Until ready"))]]
            phases = [p for p in phases if p["phase"] == "serving"]
            for phase in phases:
                phase.update({"sources": ["hosted"], "said": "hosted APIs", "workers": 0, "hosts": 0,
                              "hourly_usd": 0.0})
        return {
            "readback": said,
            "timeline": phases,
            "tz": serving.tz,
            "conditions": conditions(serving),
            "most_it_can_spend": {
                "lease_cap": round(cap, 2), "hosted": hosted,
                "total": round(cap + sum(h["most"] for h in hosted), 2),
                "said": (f"${cap + sum(h['most'] for h in hosted):.2f} at most: the lease's cap ${cap:.2f} (every "
                         "rented machine, bursts included)"
                         + "".join(f" + {h['name']} ${h['most']:.2f}" for h in hosted)
                         + (" (each hosted budget over every UTC day and month its hours touch)" if hosted else "")),
            },
            "warnings": warnings,
            "providers": [h.id for h in self.config.hosts if h.kind == "hosted" and not h.disabled
                          and any(t.model in self.config.models_held_by(h) for t in req.all_targets)],
            "plan": serving.as_dict() if serving.max_min is not None else None,
        }

    def _shared_holds(self, model: str) -> bool:
        """Whether the shared workload's machines hold a model at all: it is in the pool's set and
        not kept to workloads."""
        entry = self.config.catalog.get(model)
        return model in self.config.pool.model_set and not (entry and entry.workloads_only)

    def _ready_minutes(self, plan: dict[str, Any]) -> dict[str, Any]:
        """How long its machines take to be ready, minutes: the median and 90th percentile over
        machines of the first host's card class this pool has rented (D69), else the estimate from
        the download and the engine's load — the slowest group's."""
        records = list(self.fleet.machine_history().values()) if self.fleet is not None else []
        worst: Optional[dict[str, Any]] = None
        for group in plan["groups"]:
            card = group["first_host"]["hardware"]
            seen = sorted(r.minutes_to_ready for r in records if r.hardware == card and r.minutes_to_ready is not None)
            if seen:
                median = statistics.median(seen)
                p90 = seen[max(0, math.ceil(0.9 * len(seen)) - 1)]
                said = {"median": round(median, 1), "p90": round(p90, 1), "measured": True, "machines": len(seen),
                        "card": card}
            else:
                estimate = float(group["minutes_to_serve"])
                said = {"median": round(estimate, 1), "p90": round(estimate, 1), "measured": False, "machines": 0,
                        "card": card}
            if worst is None or said["p90"] > worst["p90"]:
                worst = said
        return worst or {"median": float(plan.get("minutes_to_serve") or 0), "p90": float(plan.get("minutes_to_serve") or 0),
                         "measured": False, "machines": 0, "card": None}

    async def _plan(self, req: WorkloadRequest) -> dict[str, Any]:
        """Everything creating it would do, and nothing else: nothing is opened or rented.

        With several models, both placements are priced from one market search — every host
        holding every model, and a group of hosts per model — and the cheaper kept (D118). Its
        models served by hosted APIs alone (D165) take no part in that: they have no hosts."""
        self._check_request(req)
        # What creating it would refuse, the plan says: a policy its models cannot be served under.
        self.check_policy(req.all_targets, req.policy or {})
        self.check_reserve(req.policy or {}, excluding=req.name)
        if not req.on_machines:
            return self._plan_without_machines(req)
        if req.hosted_alone:
            machines = WorkloadRequest.of(
                req.name, req.on_machines, req.hours, max_spend=req.max_spend, profile=req.profile, kind=req.kind,
                placement=req.placement, policy=req.policy)
            plan = await self._plan_machines(machines)
            plan["models"] = [t.as_dict() for t in req.all_targets]
            plan["hosted_alone"] = list(req.hosted_alone)
            plan["reasons"].append(f"served by hosted APIs alone, no machine: {', '.join(req.hosted_alone)}")
            return await self._price_switches(machines, plan)
        return await self._price_switches(req, await self._plan_machines(req))

    async def _price_switches(self, req: WorkloadRequest, plan: dict[str, Any]) -> dict[str, Any]:
        """A rule's steady switch is paid for before it is allowed (S19 5b, §14.3): checked, and
        the budget priced at the dearest steady the rules can reach plus the overlap of each
        switch — both sets of hosts for the time a new one takes to be ready. A derived budget
        becomes that; a typed one must hold it."""
        from ..serving_plan import HOST_KINDS, PlanRefused, plan_of

        try:
            serving = plan_of(req.policy or {})
        except PlanRefused:
            return plan
        switches = sorted({r.switch for r in (serving.rules if serving is not None else ()) if r.switch})
        if not switches or plan.get("refused"):
            return plan
        for switch in switches:
            if switch == f"rented:{req.kind}":
                raise WorkloadRefused(f"a rule switches steady to {req.kind} machines, which it rents already "
                                      f"(machines={req.kind})")
            if switch in HOST_KINDS:
                kinds = HOST_KINDS[switch]
                for t in req.on_machines:
                    if not any(h.kind in kinds and not h.disabled and t.model in self.config.models_held_by(h)
                               for h in self.config.hosts):
                        raise WorkloadRefused(f"a rule switches steady to {switch}, and no {switch} machine of this "
                                              f"pool holds {t.model!r}")
        # Each switch to rented machines of another kind, both ways, for every time its rule can
        # begin within the hours: going, the old base bills while the new is prepared; coming
        # back, the new bills while the old kind is prepared again. The base itself is priced at
        # the dearest kind for every hour.
        here_hourly = float(plan.get("hourly_total") or 0.0)
        here_min = float(plan.get("minutes_to_serve") or 0.0)
        needed, dearest, overlaps, times = plan["derived_budget"], req.kind, 0.0, 0
        others: dict[str, dict[str, Any]] = {}
        for rule in serving.rules:
            if not (rule.switch or "").startswith("rented:"):
                continue  # the operator's machines: nothing rented while they serve, nothing to overlap
            kind = rule.switch.split(":", 1)[1]
            if kind not in others:
                other = await self._plan_machines(dataclasses.replace(req, kind=kind, max_spend=None))
                if other.get("refused"):
                    raise WorkloadRefused(f"a rule switches steady to {kind} machines, and the market has none to "
                                          f"serve it: {other['refused']}")
                others[kind] = other
                if other["derived_budget"] > needed:
                    needed, dearest = other["derived_budget"], kind
            other = others[kind]
            count = _switch_times(rule, req.hours)
            times += count
            overlaps += count * (here_hourly * float(other["minutes_to_serve"])
                                 + float(other["hourly_total"]) * here_min) / 60
        needed = round(needed + overlaps, 2)
        plan["switch_budget"] = {"needed": needed, "dearest": dearest, "switches": switches,
                                 "overlaps": round(overlaps, 2), "times": times}
        if needed > plan["derived_budget"]:
            said = (f"its rules can switch steady to {dearest} machines: priced there for every hour, with both sets of "
                    f"hosts while each is prepared, both ways, up to {times} time(s) — ${needed:.2f}")
            plan["reasons"].append(said)
            plan["derived_budget"] = needed
            if plan["budget_derived"]:
                plan["max_spend"] = needed
            elif req.max_spend is not None and req.max_spend < needed:
                plan["refused"] = f"{said}, more than the ${req.max_spend:.2f} typed"
        return plan

    def _plan_without_machines(self, req: WorkloadRequest) -> dict[str, Any]:
        """A workload whose every model is served by hosted APIs alone (D165): nothing to search
        for, size or rent, no budget but its hosted burst's, and serving from the start."""
        first = req.all_targets[0]
        open_now = [w for w in self.store.active() if w.state in ("preparing", "serving")]
        most = self.config.workloads.max_open
        return {
            "name": req.name, "model": first.model, "build": None, "cards_per_copy": 1,
            "latency_s": first.latency_s, "parallel": 0, "hours": req.hours, "kind": req.kind,
            "models": [t.as_dict() for t in req.all_targets], "builds": {}, "placement": "none", "groups": [],
            "hosted_alone": list(req.hosted_alone), "offers_seen": 0, "offers_passed": 0,
            "workers_per_host": 0, "workers_measured": False, "sizing": [], "hosts_at_start": 0,
            "first_host": None, "minutes_to_serve": 0, "max_spend": 0.0, "budget_derived": False,
            "derived_budget": 0.0, "hourly_total": 0.0, "borrow_while_starting": False, "borrow_by_model": {},
            "reasons": (["served by hosted APIs alone: nothing is rented, and it serves from the start, "
                         "paid from its hosted burst's budget"] if req.hosted_alone and len(req.hosted_alone) == len(
                             req.all_targets) else
                        [f"runs on the operator's {steady_of(req.policy)} machines, on the workers reserved for it: "
                         "nothing is rented, and it serves from the start"]),
            "refused": (f"{len(open_now)} workloads are running, at the pool's limit of {most}"
                        if len(open_now) >= most else None),
        }

    async def _plan_machines(self, req: WorkloadRequest) -> dict[str, Any]:
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
        burst = self._rented_burst(req)
        for option in options.values():
            if option.get("refused") is None:
                option["derived"] = math_.derived_budget(1, req.hours, option["hourly"])
                if burst is not None:
                    # A rented burst is paid from the same lease (D165): the most its hosts could
                    # cost over the hours — each group's at its first host's price, the whole
                    # time — or its budget over every day and month the hours touch, if less.
                    most_hourly = sum(burst.max_hosts * g["first_host"]["hourly"] for g in option["groups"])
                    option["burst_money"] = min(math_.derived_budget(1, req.hours, most_hourly),
                                                math.ceil(burst.budget.most_over(now, req.hours) * 100) / 100)
                    option["derived"] = round(option["derived"] + option["burst_money"], 2)
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
        if burst is not None:
            plan["rented_burst"] = {**burst.model_dump(exclude_none=True), "in_the_budget": option["burst_money"]}
            plan["reasons"].append(
                f"a rented burst: up to {burst.max_hosts} more host(s) per group once a burst has lasted "
                f"{burst.after_min:g} min, within {burst.budget.model_dump(exclude_none=True)}; "
                f"${option['burst_money']:.2f} of it is in the derived budget, paid from the same lease")
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

    @staticmethod
    def _rented_burst(req: WorkloadRequest) -> Any:
        from ..config import RentedBurst

        raw = (req.policy or {}).get("rented")
        return RentedBurst.model_validate(raw) if raw else None

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
        csr: Optional[str] = None, ca: Any = None, policy: Optional[dict[str, Any]] = None,
    ) -> tuple[Workload, Optional[str], dict, Optional[str]]:
        """Create it: (the workload, its key — None where the program made its own and sent only
        the hash —, the plan, a signed client certificate where a signing request came)."""
        if policy is None:
            policy = req.policy
        req = dataclasses.replace(req, policy=policy)
        async with self._creating:
            # Planned inside the lock, so its name check and its caps see every workload created
            # before it — including one whose plan was awaiting the market a moment ago. The
            # readback is the plan check's, not stored with the workload.
            plan = await self._plan(req)
            if plan["refused"]:
                raise WorkloadRefused(plan["refused"])
            self.check_policy(req.all_targets, policy or {})
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
            machines = req.on_machines
            workload = Workload(
                name=req.name, model=targets[0].model, builds=dict(plan["builds"]),
                latency_s=min(t.latency_s for t in (machines or targets)),
                # With no machine there is nothing to prepare: it serves from the start (D165).
                parallel=req.total_parallel, kind=req.kind, lease_id=lease_id,
                state="preparing" if machines else "serving",
                workers_per_host=plan["workers_per_host"], hosts_at_start=plan["hosts_at_start"], plan=plan,
                created_at=now, updated_at=now, ends_at=now + req.hours * 3600,
                provisioner=provisioner, idle_end_minutes=idle_end_minutes,
                targets=targets, placement=plan["placement"], groups=groups, policy=dict(policy or {}),
            )
            # The row first: a name taken meanwhile fails here, before any spending authority
            # exists. The lease second, under the id the row already names.
            try:
                self.store.create(workload)
            except Exception as exc:  # noqa: BLE001 - a unique name, lost to another creation
                raise WorkloadRefused(f"workload {req.name!r} could not be recorded: {exc}") from exc
            try:
                if machines:
                    # Through the fleet: a lease longer than the dead-man timer covers is refused there.
                    # Its workers are every model's answers at once: the lease bounds its groups together.
                    lease = self.fleet.open_lease(
                        workers=req.total_parallel, max_hours=req.hours, max_spend=budget, allow_rent=True,
                        workload=req.name, lease_id=lease_id,
                    )
                else:
                    # Nothing to rent: a lease that cannot rent, for its hours alone — its key's life,
                    # its end. Its spending is its hosted burst's budget (D165).
                    lease = self.supervisor.leases.open(
                        workers=0, max_hours=req.hours, max_spend=None, allow_rent=False,
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
            + "; ".join([f"{t.model} at {t.parallel} at once, {t.latency_s:g}s p95" for t in req.on_machines]
                        + ([f"{', '.join(req.hosted_alone)} by hosted APIs alone"] if req.hosted_alone else []))
            + (f", for {req.hours:g}h — {plan['hosts_at_start']} host(s)"
               + (f" placed {plan['placement']}" if len(req.on_machines) > 1 else f" at {plan['workers_per_host']} each")
               + f", up to ${budget:.2f}" if req.on_machines else f", for {req.hours:g}h — no machine"),
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

    # --- its spending policy (D165) ---

    def check_policy(self, targets: Sequence[ModelTarget], policy: dict[str, Any]) -> None:
        """A policy the workload's models can be served under (D165):

        - a hosted burst, where some hosted API this pool has serves one of its models — else
          it would be a budget nothing could spend, and the app would wait for a spill that never
          comes;
        - a model served by hosted APIs alone needs a hosted burst, and a hosted API serving it:
          it has nothing else;
        - a rented burst needs a model that runs on machines: it adds hosts beyond their base."""
        models = [t.model for t in targets]
        alone = [t.model for t in targets if not t.on_machines]
        hosted = [h for h in self.config.hosts if h.kind == "hosted" and not h.disabled]
        served = {m for h in hosted for m in self.config.models_held_by(h)}
        if alone and not policy.get("hosted"):
            raise WorkloadRefused(
                ", ".join(repr(m) for m in alone) + " is served by hosted APIs alone (parallel 0), so it needs a "
                "hosted burst: bursts.hosted with a budget")
        unserved = [m for m in alone if m not in served]
        if unserved:
            raise WorkloadRefused("no hosted API in this pool serves " + ", ".join(repr(m) for m in unserved)
                                  + ", and parallel 0 has nothing else to serve it")
        if policy.get("rented") and not any(t.on_machines for t in targets):
            raise WorkloadRefused("a rented burst adds hosts beyond a base on machines, and none of its models runs "
                                  "on one (every parallel is 0)")
        if policy.get("hosted") and not any(m in served for m in models):
            raise WorkloadRefused(
                "no hosted API in this pool serves " + ", ".join(repr(m) for m in models)
                + "; a hosted burst would have nothing to spend on"
            )
        # A plan's defaults (S19, D201): refused, per model, where no source the plan has could
        # serve a request that sends nothing.
        from ..serving_plan import PlanRefused, kept_off, plan_of

        try:
            plan = plan_of(policy)
        except PlanRefused:
            plan = None
        if plan is not None:
            # A hosted source's providers are hosted hosts of this pool that serve one of its
            # models (S19, §14.2): a name that is not would be a source that never pays.
            by_id = {h.id: h for h in hosted}
            for source in plan.hosted_sources():
                for provider in source.providers:
                    host = by_id.get(provider)
                    if host is None:
                        raise WorkloadRefused(f"source {source.name}: {provider!r} is not a hosted API of this pool; "
                                              f"its hosted APIs are {', '.join(sorted(by_id)) or 'none'}")
                    if not set(self.config.models_held_by(host)) & set(models):
                        raise WorkloadRefused(f"source {source.name}: {provider!r} serves none of "
                                              + ", ".join(repr(m) for m in models))
        if plan is not None and plan.defaults:
            off = kept_off(plan.defaults.get("priority", "interactive"), plan.defaults.get("keep_on", "any"))
            if alone and "hosted" in off:
                raise WorkloadRefused(
                    ", ".join(repr(m) for m in alone) + " is served by hosted APIs alone, and the plan's defaults keep "
                    "requests that send no option off hosted APIs: no source could serve them")
        if plan is not None and plan.substitutes:
            self._check_substitutes(models, plan)

    def _kinds_of(self, model: str) -> frozenset[str]:
        entry = self.config.catalog.get(model)
        if entry is None or not entry.variants:
            return frozenset({"chat", "embed"})  # an uncatalogued build: chat and embeddings, as every build was
        return frozenset().union(*(v.kinds() for v in entry.variants))

    def _check_substitutes(self, models: Sequence[str], plan: Any) -> None:
        """A substitute (§14.5, D202) is a catalogued model of this pool, not one of the workload's
        own, that does what each of its models does — chat for chat, transcription for
        transcription — so a request for either can be answered by it."""
        for model, _ in plan.substitutes:
            entry = self.config.catalog.get(model)
            if model not in self.config.pool.model_set and not (entry is not None and entry.workloads_only):
                raise WorkloadRefused(f"substitute {model!r} is not a model of this pool's catalog")
            if model in models:
                raise WorkloadRefused(f"substitute {model!r} is one of the workload's own models")
            theirs = self._kinds_of(model)
            for own in models:
                if not self._kinds_of(own) & theirs:
                    raise WorkloadRefused(f"substitute {model!r} does none of what {own!r} does "
                                          f"({', '.join(sorted(self._kinds_of(own)))})")

    def check_reserve(self, policy: dict[str, Any], excluding: Optional[str] = None) -> None:
        """Reserved workers stay within `workloads.max_reserved_share` of the operator's own
        workers of each kind, across every running workload (S19, §14.2): counted from the
        configured hosts, so a reservation is possible on the day the machines are all up. While
        fewer are ready the router scales every reservation down pro rata."""
        from ..serving_plan import HOST_KINDS, PlanRefused, plan_of

        try:
            plan = plan_of(policy)
        except PlanRefused:
            return
        if plan is None or not plan.reserve:
            return
        share = self.config.workloads.max_reserved_share
        for source, wanted in plan.reserve.items():
            kinds = HOST_KINDS[source]
            workers = sum(h.workers for h in self.config.hosts if h.kind in kinds and not h.disabled)
            taken = 0
            for other in self.store.active():
                if other.name == excluding or other.state not in ("preparing", "serving"):
                    continue
                try:
                    theirs = plan_of(other.policy or {})
                except PlanRefused:
                    continue
                taken += (theirs.reserve.get(source, 0) if theirs is not None else 0)
            room = int(workers * share) - taken
            if wanted > room:
                raise WorkloadRefused(
                    f"reserve.{source}: {wanted} worker(s) asked, {max(0, room)} left — reservations stay within "
                    f"{share:.0%} of the pool's {workers} {source} worker(s) (workloads.max_reserved_share), "
                    f"{taken} already reserved")

    def set_policy(self, name: str, policy: dict[str, Any], by: str = "the operator") -> Workload:
        """Replace a running workload's spending policy. Takes effect at the router's next look at
        the workload table; what it already spent today and this month still counts."""
        workload = self.get(name)
        if workload.state not in ("preparing", "serving"):
            raise WorkloadRefused(f"workload {name!r} is {workload.state}; its policy can no longer change")
        if (workload.policy or {}).get("serve") and not policy.get("serve"):
            raise WorkloadRefused(f"workload {name!r} has a serving plan; change it with serve — its bursts are the "
                                  "plan's when_full")
        if steady_of(policy) != steady_of(workload.policy):
            raise WorkloadRefused(f"workload {name!r} runs on {steady_of(workload.policy)}; where it runs is set when "
                                  "it is created (a rule may switch it, with the switch priced then)")
        added = _switches(policy) - _switches(workload.policy)
        if added:
            # A switch is checked and priced into the lease when the workload is created (§14.3).
            raise WorkloadRefused(f"workload {name!r}: a rule's steady switch is checked and priced when a workload is "
                                  f"created, and this one was created without a switch to {', '.join(sorted(added))}; "
                                  "create a workload with that rule")
        self.check_policy(workload.targets, policy)
        self.check_reserve(policy, excluding=name)
        self.store.set_policy(name, policy)
        self.supervisor.events.record(
            "workload_policy",
            f"workload {name}: {describe_policy(policy)} (set by {by})",
            numbers={"workload": name, "policy": policy, "by": by}, lease_id=workload.lease_id,
        )
        return self.get(name)

    def hosted_spend(self, name: str) -> dict[str, float]:
        """What the workload spent on hosted APIs today and this month (UTC)."""
        from ..db import RequestLog
        from ..router.budgets import period_starts

        day, month = period_starts()
        log = RequestLog(self.supervisor.db)
        return {"today": round(log.hosted_spend_of([name], day), 6),
                "this_month": round(log.hosted_spend_of([name], month), 6)}

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
        switched = self.store.steady_switches()
        for workload in self.store.active():
            lease = self.supervisor.leases.get(workload.lease_id)
            hosts = self.fleet.hosts_of(workload.name) if self.fleet is not None else []
            if (not workload.on_machines and workload.state in ("preparing", "serving") and lease is not None
                    and lease.is_open and lease.expired()):
                # Its lease rents nothing, so the fleet's pass never looks at it — and a pool
                # serving by hosted APIs alone may have no fleet at all. Its hours end here.
                self.supervisor.leases.close(lease.lease_id, "time limit reached")
                lease = self.supervisor.leases.get(workload.lease_id)
            if workload.state in ("preparing", "serving") and (lease is None or not lease.is_open):
                self._ending(workload, f"its lease closed ({lease.closed_reason if lease else 'gone'})")
                continue
            ready_groups = [g for g in workload.groups
                            if any(h.state == "ready" for h in hosts if g.holds(h.models))]
            if (switched.get(workload.name) or {}).get("serving") in ("local", "fixed"):
                ready_groups = list(workload.groups)  # served on the operator's machines by a rule (5b)
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
        current = self.store.active()
        self._mark_gaps(now, current)
        self._arm_rules(now, current)
        self._switch_steady(now, current)
        return [w for w in current if w.state in ("preparing", "serving", "ending")]

    def _mark_gaps(self, now: float, active: list[Workload]) -> None:
        """Mark each gap a serving plan covers (S19, D198), per group of a serving workload with
        an open lease: a **replacement** while its ready hosts bought for its base are fewer than
        the group planned — the fleet replacing what it lost — and, where the plan lists
        `scale_up`, a **scale-up** while a host rented for load above them is being prepared.
        From the moment it began, for at most the plan's `max_min`, once per shortfall: a
        replacement retried keeps the gap it had. Written only on a change; the router ends a gap
        itself the moment a host not ready as it opened is."""
        from ..serving_plan import PlanRefused, plan_of

        marked = self.store.gaps()
        switched = self.store.steady_switches()
        wanted: set[tuple[str, str]] = set()
        for workload in active:
            policy = workload.policy or {}
            if workload.state != "serving" or "serve" not in policy:
                continue
            if (switched.get(workload.name) or {}).get("serving") in ("local", "fixed"):
                continue  # steady is on the operator's machines by a rule: its rented hosts are meant to go
            try:
                plan = plan_of(policy)
            except PlanRefused:
                continue
            if (plan is None or not plan.until_ready or plan.max_min is None
                    or not {"replacement", "scale_up"} & set(plan.during)):
                continue
            lease = self.supervisor.leases.get(workload.lease_id)
            if lease is None or not lease.is_open:
                continue
            hosts = self.fleet.hosts_of(workload.name) if self.fleet is not None else []
            for group in workload.groups:
                mine = [h for h in hosts if group.holds(h.models)]
                held = [h for h in mine if h.state == "ready"]
                base = [h for h in held if not h.burst]
                if len(base) < group.hosts_at_start:
                    reason = "replacement" if "replacement" in plan.during else None
                    said = f"{len(base)} of {group.hosts_at_start} host(s) ready"
                elif "scale_up" in plan.during and any(h.burst and h.state in ("scheduling", "preparing")
                                                       for h in mine):
                    reason, said = "scale_up", "a host rented for load is being prepared"
                else:
                    reason = None
                if reason is None:
                    continue
                for model in group.models:
                    key = (workload.name, model)
                    wanted.add(key)
                    if key in marked and marked[key][3] == reason:
                        continue  # once per shortfall: never extended, never re-read
                    ends_at = now + plan.max_min * 60
                    self.store.open_gap(workload.name, model, now, ends_at, [h.host_id for h in held], reason)
                    self.supervisor.events.record(
                        "workload_covering",
                        f"workload {workload.name}: {said} for {model}; its plan's until_ready covers the "
                        f"{reason.replace('_', '-')} for at most {plan.max_min:g} min",
                        numbers={"workload": workload.name, "model": model, "ends_at": ends_at, "reason": reason},
                        lease_id=workload.lease_id,
                    )
        for key in marked:
            if key not in wanted:
                self.store.close_gap(*key)

    def _arm_rules(self, now: float, active: list[Workload]) -> None:
        """Judge the conditions on the pass of every running workload's plan rules — a time
        window in the plan's time zone, a budget's spend, evictions — and write the rules whose
        conditions all hold, by name, where that changed (S19, D200). A rule with no such
        condition is the router's alone to judge. A window takes effect at the first pass after
        its boundary, never before."""
        from ..serving_plan import PlanRefused, narrowed, narrowing, plan_of

        stored = self.store.armed_rules()
        suppressed: set[str] = set()
        for workload in active:
            if workload.state not in ("preparing", "serving"):
                continue
            try:
                plan = plan_of(workload.policy or {})
            except PlanRefused:
                plan = None
            if plan is None or not plan.rules:
                if stored.get(workload.name):
                    self.store.set_armed_rules(workload.name, ())
                continue
            armed = frozenset(r.name for r in plan.rules if r.on_the_pass and self._holds(workload, plan, r.when, now))
            if armed != stored.get(workload.name, frozenset()):
                self.store.set_armed_rules(workload.name, armed)
                began, ended = sorted(armed - stored.get(workload.name, frozenset())), sorted(
                    stored.get(workload.name, frozenset()) - armed)
                self.supervisor.events.record(
                    "workload_rules",
                    f"workload {workload.name}: "
                    + "; ".join([f"rule {n!r} in force" for n in began] + [f"rule {n!r} no longer" for n in ended]),
                    numbers={"workload": workload.name, "armed": sorted(armed)}, lease_id=workload.lease_id,
                )
            # A rented burst the rules in force for every request keep out is not rented (and its
            # hosts drain, as a burst its policy no longer has).
            every = [r for r in plan.rules if not ("priority" in r.when or "keep_on" in r.when)]
            plain = dataclasses.replace(plan, rules=tuple(every))
            seen = narrowed(plan, narrowing(plain, armed, "interactive", "any"))
            if plan.rented() is not None and seen.rented() is None:
                suppressed.add(workload.name)
        running = {w.name for w in active if w.state in ("preparing", "serving")}
        for name in stored:
            if name not in running and stored[name]:
                self.store.set_armed_rules(name, ())  # ended: none of its rules is in force
        if self.fleet is not None:
            self.fleet.bursts_held_back = suppressed

    def _switch_steady(self, now: float, active: list[Workload]) -> None:
        """A rule's steady switch, handed over make-before-break (S19 stage 5b, §14.3). Per
        workload, what its rules want (`want`) and where its steady is served (`serving`, the
        router's to follow): a switch to the operator's machines is served there once one of that
        kind holding its models is ready, and the fleet then gives its rented hosts up; a switch to
        rented machines of another kind — or back from the operator's — keeps serving where it
        was until the fleet's new base is ready. Written only on a change, with an event."""
        from ..serving_plan import PlanRefused, plan_of, steady_wanted

        stored = self.store.steady_switches()
        armed_all = self.store.armed_rules()
        targets: dict[str, Optional[str]] = {}
        serving_on: dict[str, Optional[str]] = {}
        for workload in active:
            if workload.state not in ("preparing", "serving"):
                continue
            try:
                plan = plan_of(workload.policy or {})
            except PlanRefused:
                plan = None
            before = stored.get(workload.name)
            switching = plan is not None and any(r.switch for r in plan.rules)
            if not switching and not before:
                continue
            # A switch whose rule was removed hands back as one that ended: make-before-break.
            want = steady_wanted(plan, armed_all.get(workload.name, frozenset()),
                                 (before or {}).get("want"), now) if switching else None
            target = want["to"] if want else None
            served = (before or {}).get("serving")
            if target in ("local", "fixed"):
                serving = target if (served == target or self._own_ready(workload, target)) else served
            elif served in ("local", "fixed") and not self._rented_base_ready(workload, target):
                serving = served  # handed back only once its rented base is ready again
            else:
                serving = None
            state = {"want": want, "serving": serving} if (want or serving) else None
            if state != before:
                self.store.set_steady_switch(workload.name, state)
                where = {None: "its rented machines"}.get(serving, f"the operator's {serving} machines")
                self.supervisor.events.record(
                    "workload_steady",
                    f"workload {workload.name}: steady "
                    + (f"switching to {target} by rule {want['rule']!r}" if want else "back as its plan says")
                    + f"; serving on {where}",
                    numbers={"workload": workload.name, "want": want, "serving": serving}, lease_id=workload.lease_id,
                )
            targets[workload.name], serving_on[workload.name] = target, serving
        if self.fleet is not None:
            self.fleet.steady_targets = targets
            self.fleet.steady_serving = serving_on

    def _own_ready(self, workload: Workload, source: str) -> bool:
        """Is a ready host of the operator's of this kind holding every model of the workload?"""
        from ..serving_plan import HOST_KINDS

        kinds = HOST_KINDS[source]
        return all(any(h.config.kind in kinds and h.state.value == "ready" and t.model in h.variants
                       for h in self.supervisor.hosts.values())
                   for t in workload.targets if t.on_machines)

    def _rented_base_ready(self, workload: Workload, target: Optional[str]) -> bool:
        """Has each group its planned hosts ready, of the kind steady now rents?"""
        hosts = self.fleet.hosts_of(workload.name) if self.fleet is not None else []
        kind = target.split(":", 1)[1] if target else workload.kind
        for group in workload.groups:
            ready = [h for h in hosts if group.holds(h.models) and h.state == "ready" and not h.burst
                     and kind_matches(kind, getattr(h, "interruptible", False))]
            if len(ready) < group.hosts_at_start:
                return False
        return True

    def _spent(self, workload: Workload, source: Any, period: str, now: float) -> Optional[float]:
        """What one of its spending sources has spent this UTC day or month: a hosted source's
        answers (the request log records the budget each was held against), a rented burst's
        hosts. None where it cannot be read (no lease or fleet)."""
        from ..router.budgets import period_starts

        since = period_starts(now)[0 if period == "day" else 1]
        if source.type == "hosted":
            if not source.role:
                # The shorthand's one hosted burst (today's fields): its answers carry no budget id.
                return float(self.supervisor.db.query(
                    "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM request_log WHERE workload = ? "
                    "AND (budget IS NULL OR budget = '') AND ts >= ?", (workload.name, since))[0]["c"])
            return float(self.supervisor.db.query(
                "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM request_log WHERE workload = ? AND budget = ? "
                "AND ts >= ?", (workload.name, source.budget_id, since))[0]["c"])
        lease = self.supervisor.leases.get(workload.lease_id)
        if lease is None or self.fleet is None:
            return None
        return self.fleet.burst_spend(lease.lease_id, since)

    def now_context(self) -> dict[str, Any]:
        """What the Now line reads of every workload at once: read once for a list of them."""
        return {"switches": self.store.steady_switches(), "armed": self.store.armed_rules(), "gaps": self.store.gaps()}

    def now_line(self, workload: Workload, hosts: Sequence[Any],
                 context: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """The Now line (§14.6, D203): which role serves, where the next request would go, how far
        preparation is and how long the stop-gap has left, each budget against its limit, and the
        rules in force — from what the supervisor last wrote for the router."""
        from ..serving_plan import PlanRefused, from_bursts, plan_of

        if workload.state not in ("preparing", "serving"):
            return {"said": f"{workload.state.capitalize()}.", "role": None, "next": None, "budgets": [], "rules": []}
        now = time.time()
        policy = workload.policy or {}
        try:
            plan = plan_of(policy) or from_bursts(policy, bool((workload.plan or {}).get("may_borrow", True))
                                                  and self.config.workloads.borrow_share > 0)
        except PlanRefused:
            return {"said": "its serving plan cannot be read by this supervisor", "role": None, "next": None,
                    "budgets": [], "rules": []}
        context = context or self.now_context()
        switch = context["switches"].get(workload.name) or {}
        armed = sorted(context["armed"].get(workload.name, frozenset()))
        gaps = {m: g for (w, m), g in context["gaps"].items() if w == workload.name and now < g[1]}
        planned = sum(g.hosts_at_start for g in workload.groups)
        base_ready = len([h for h in hosts if h.state == "ready" and not h.burst])
        served_on = switch.get("serving") or (plan.steady if plan.steady != "rented" else None)
        cover = [s.name for s in plan.until_ready]
        cap = plan.max_min if plan.max_min is not None else None
        if served_on in ("local", "fixed"):
            by = f" (rule {switch['want']['rule']!r})" if (switch.get("want") or {}).get("rule") else ""
            role, next_ = "steady", f"steady/{served_on}"
            said = f"Steady on the operator's {served_on} machines{by}."
        elif all(not t.on_machines for t in workload.targets):
            role, next_, said = "when_full", "hosted", "Served by hosted APIs alone."
        elif workload.state == "preparing":
            elapsed = (now - workload.created_at) / 60
            left_ready = max(0.0, float((workload.plan or {}).get("minutes_to_serve") or 0) - elapsed)
            ready_part = f"{base_ready} of {planned} rented machine(s) ready (about {left_ready:.0f} min)"
            if cover and "start" in plan.during and (cap is None or elapsed < cap):
                left = f"; stop-gap {cap - elapsed:.0f} min left" if cap is not None else ""
                role, next_ = "until_ready", f"until_ready/{cover[0]}"
                said = f"Covering until ready: {ready_part}{left}."
            else:
                ended = "; its stop-gap has ended" if cover and "start" in plan.during else ""
                role, next_, said = "steady", "waits for its machines", f"Preparing: {ready_part}{ended}; requests wait."
        elif gaps and cover:
            ends = max(g[1] for g in gaps.values())
            reason = next(iter(gaps.values()))[3]
            role, next_ = "until_ready", f"until_ready/{cover[0]}"
            said = (f"Covering a {reason.replace('_', '-')}: {base_ready} of {planned} rented machine(s) ready; "
                    f"stop-gap {max(0.0, (ends - now) / 60):.0f} min left.")
        else:
            role, next_ = "steady", "steady/rented"
            said = f"Steady: {base_ready} of {planned} rented machine(s) ready."
            want = switch.get("want") or {}
            if str(want.get("to", "")).startswith("rented:"):
                kind = want["to"].split(":", 1)[1]
                if self._rented_base_ready(workload, want["to"]) and not any(
                        not kind_matches(kind, getattr(h, "interruptible", False)) for h in hosts if not h.burst):
                    said = f"Steady on {kind} machines (rule {want['rule']!r}): {base_ready} of {planned} ready."
                else:
                    said = (f"Steady switching to {kind} machines (rule {want['rule']!r}); the old serve until the new "
                            f"are ready. {base_ready} ready.")
            elif want.get("to") in ("local", "fixed"):
                said += f" Switching to the operator's {want['to']} machines once one is ready (rule {want['rule']!r})."
        budgets = []
        for source in plan.sources():
            if not source.spends:
                continue
            period = "day" if (source.budget or {}).get("per_day") is not None else "month"
            limit = (source.budget or {}).get(f"per_{period}")
            used = self._spent(workload, source, period, now)
            if limit is None or used is None:
                continue
            budgets.append({"budget": source.budget_id, "name": source.name, "period": period,
                            "spent": round(used, 4), "limit": limit})
        in_force = armed + [r.name for r in plan.rules if not r.on_the_pass]
        tail = []
        if next_:
            tail.append(f"Next request: {next_}.")
        if budgets:
            tail.append("; ".join(f"{b['name']} ${b['spent']:.2f} of ${b['limit']:g} "
                                  + ("today" if b["period"] == "day" else "this month") for b in budgets) + ".")
        tail.append(f"Rules in force: {', '.join(in_force) if in_force else 'none'}"
                    + (" (by a request's options: " + ", ".join(r.name for r in plan.rules if not r.on_the_pass) + ")"
                       if any(not r.on_the_pass for r in plan.rules) else "") + ".")
        return {"said": " ".join([said, *tail]), "role": role, "next": next_, "budgets": budgets, "rules": in_force,
                "switch": switch or None}

    def _holds(self, workload: Workload, plan: Any, when: Mapping[str, Any], now: float) -> bool:
        """Whether a rule's conditions on the pass hold now."""
        from ..serving_plan import in_window

        if "between" in when and not in_window(when, plan.tz, now):
            return False
        if "spent_over" in when:
            spent = when["spent_over"]
            lease = self.supervisor.leases.get(workload.lease_id)
            if spent["budget"] == "lease":
                if lease is None or not lease.max_spend or self.fleet is None:
                    return False
                if self.fleet.lease_spend(lease)[0] < spent["share"] * lease.max_spend:
                    return False
            else:
                source = next(s for s in plan.sources() if s.spends and s.budget_id == spent["budget"])
                limit = source.budget[f"per_{spent['period']}"]
                used = self._spent(workload, source, spent["period"], now)
                if used is None or used < spent["share"] * limit:
                    return False
        if "evictions" in when:
            ev = when["evictions"]
            count = int(self.supervisor.db.query(
                "SELECT COUNT(*) AS n FROM events WHERE kind = 'eviction' AND lease_id = ? AND ts >= ?",
                (workload.lease_id, now - ev["within_h"] * 3600))[0]["n"])
            if count < ev["count"]:
                return False
        return True

    def view(self, workload: Workload, context: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """What the operator sees: the workload, its hosts, its spend against its cap, and how
        its answers compare with its target — never a key."""
        lease = self.supervisor.leases.get(workload.lease_id)
        hosts = self.fleet.hosts_of(workload.name) if self.fleet is not None else []
        spent = self.fleet.lease_spend(lease)[0] if (self.fleet is not None and lease is not None) else 0.0
        rows = self.supervisor.db.query(
            # A substitute's answers are left out (D202): they say nothing of its own model's latency.
            "SELECT latency_ms, model_requested FROM request_log WHERE workload = ? AND outcome = 'ok' "
            "AND latency_ms IS NOT NULL AND (route IS NULL OR route NOT LIKE 'substitute/%') "
            "ORDER BY id DESC LIMIT 2000",
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
            if not target.on_machines:
                # Served by hosted APIs alone (D165): no latency target to meet.
                by_model[target.model] = {"count": None, "latency_s": None, "parallel": 0, "p95_s": None,
                                          "meets_target": None, "cap_per_host": None, "hosted_alone": True}
                continue
            mine = [r["latency_ms"] / 1000 for r in self.supervisor.db.query(
                "SELECT latency_ms FROM request_log WHERE workload = ? AND model_requested = ? AND outcome = 'ok' "
                "AND latency_ms IS NOT NULL AND (route IS NULL OR route NOT LIKE 'substitute/%') "
                "ORDER BY id DESC LIMIT 2000", (workload.name, target.model))] if several \
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
            and not workload.hosted_alone(m)
            and m not in ready_models and self._shared_serves(m)
        ]
        return {
            **{k: v for k, v in workload.as_dict().items() if k != "plan"},
            "plan": workload.plan,
            # Which role serves, where the next request goes, budgets and rules (§14.6).
            "now": self.now_line(workload, hosts, context),
            # What its hosted burst has spent, against its policy's budget (D165).
            "hosted_spend": self.hosted_spend(workload.name) if workload.policy.get("hosted") else None,
            # What its rented burst's hosts cost, today and this month, and how many it holds (D165).
            "rented_burst": ({**{k: round(v, 4) for k, v in self.fleet.burst_spent(workload.lease_id).items()},
                              "hosts": len([h for h in hosts if h.burst])}
                             if workload.policy.get("rented") and self.fleet is not None else None),
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
                 "burst": h.burst,
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
                ) if several else ((math_.p95(latencies) <= workload.latency_s)
                                   if latencies and workload.on_machines else None),
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
