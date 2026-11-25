"""A workload's serving plan (S19, D198–D204; docs/spec/workloads.md §14).

A plan says where a workload's requests go and when: **steady** (where it runs), **until_ready**
(a stop-gap used only while a machine of the workload is being prepared) and **when_full** (its
bursts), tried in that order. How much it must serve stays in `parallel` and `latency_s`.

This module is pure: it reads a plan, refuses what it cannot be in words, compiles it to the
policy the rest of the pool already reads (`hosted`, `rented`, D165), writes today's fields as
the plan they mean (the shorthand), and works out — from facts the router hands it — how a
request may be routed now. Nothing here touches the database, the clock or the network.

Built in stages (the story's build order). What this stage does not do yet is refused in words,
never accepted and ignored: rules, request options and their defaults, a substitute, reserved
workers, a steady source other than rented machines, covering a scale-up, `providers`, and more
than one hosted or rented source.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

#: The words a plan may use for where a request goes.
SOURCE_TYPES = ("local", "fixed", "shared", "rented", "hosted")
#: Which source types each role takes (workloads.md §14.1). Rented machines are what until_ready
#: covers for; borrowing never outlasts the start (D115); hosted APIs are never where a workload runs.
ROLE_TAKES = {
    "steady": ("rented",),
    "until_ready": ("local", "fixed", "shared", "hosted"),
    "when_full": ("hosted", "rented"),
}
#: Moments until_ready may cover; `start` and `replacement` unless the plan says otherwise.
DURING = ("start", "replacement", "scale_up")
DEFAULT_DURING = ("start", "replacement")
DEFAULT_MAX_MIN = 60.0
#: Host kinds each machine source reaches. `shared` is any machine of the shared workload.
HOST_KINDS = {"local": ("local",), "fixed": ("fixed-remote",)}
#: Fields of the plan a later stage brings; refused in words until then.
LATER = {
    "rules": "rules arrive in a later version of the pool",
    "defaults": "request options and their defaults arrive in a later version of the pool",
    "substitute": "a substitute model arrives in a later version of the pool",
    "reserve": "reserved workers arrive in a later version of the pool",
    "tz": "a plan's time zone comes with its rules, in a later version of the pool",
}
_TOP = ("steady", "until_ready", "when_full", "during", "max_min", *LATER)
#: A stop-gap's longest gap, minutes: a day. Longer is not a stop-gap.
MAX_MAX_MIN = 24 * 60


class PlanRefused(ValueError):
    """A plan that cannot be served as written — said in words."""


@dataclasses.dataclass(frozen=True)
class Source:
    """One place a request may go. `name` is what the plan calls it; an unnamed source is called
    by its type. Settings are those of its type: a hosted source's `after_s`, `budget` and
    `reserve_max_tokens`; a rented one's `after_min`, `max_hosts` and `budget`."""

    type: str
    name: str
    after_s: Optional[float] = None
    after_min: Optional[float] = None
    max_hosts: Optional[int] = None
    budget: Optional[dict[str, Any]] = None
    reserve_max_tokens: Optional[int] = None

    @property
    def spends(self) -> bool:
        return self.type in ("hosted", "rented")

    def as_dict(self) -> Any:
        """As a plan would write it: a bare type where it carries nothing else."""
        settings = {k: v for k, v in (("after_s", self.after_s), ("after_min", self.after_min),
                                      ("max_hosts", self.max_hosts), ("budget", self.budget),
                                      ("reserve_max_tokens", self.reserve_max_tokens)) if v is not None}
        if not settings and self.name == self.type:
            return self.type
        written: dict[str, Any] = {self.type: settings}
        if self.name != self.type:
            written["name"] = self.name
        return written


@dataclasses.dataclass(frozen=True)
class ServingPlan:
    steady: str = "rented"
    until_ready: tuple[Source, ...] = ()
    during: tuple[str, ...] = DEFAULT_DURING
    #: None: no limit — only the shorthand, which is today's borrowing, has none.
    max_min: Optional[float] = DEFAULT_MAX_MIN
    when_full: tuple[Source, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        written: dict[str, Any] = {"steady": self.steady}
        if self.until_ready:
            written["until_ready"] = [s.as_dict() for s in self.until_ready]
            written["during"] = list(self.during)
            written["max_min"] = self.max_min
        if self.when_full:
            # A source until_ready defines is the same source here: written by its name.
            written["when_full"] = [s.name if any(s is u for u in self.until_ready) else s.as_dict()
                                    for s in self.when_full]
        return written

    def source_types(self) -> set[str]:
        return {self.steady, *(s.type for s in self.until_ready), *(s.type for s in self.when_full)}

    def hosted(self) -> Optional[Source]:
        return next((s for s in (*self.until_ready, *self.when_full) if s.type == "hosted"), None)

    def rented(self) -> Optional[Source]:
        return next((s for s in self.when_full if s.type == "rented"), None)

    def until_ready_hosted(self) -> Optional[Source]:
        return next((s for s in self.until_ready if s.type == "hosted"), None)

    def when_full_hosted(self) -> Optional[Source]:
        return next((s for s in self.when_full if s.type == "hosted"), None)

    def borrows(self) -> tuple[str, ...]:
        """The machine sources until_ready names, in its order."""
        return tuple(s.type for s in self.until_ready if s.type in ("local", "fixed", "shared"))

    def order(self) -> tuple[str, ...]:
        """until_ready's sources in order, by type: what the dispatcher ranks borrowed machines
        and a hosted API by while a gap is open."""
        return tuple(s.type for s in self.until_ready)


# --- reading a plan ---


def _number(value: Any, where: str, *, low: float = 0.0, high: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
        raise PlanRefused(f"{where} is a number")
    if value < low or (high is not None and value > high):
        raise PlanRefused(f"{where} is between {low:g} and {high:g}" if high is not None else f"{where} is at least {low:g}")
    return float(value)


def _budget(raw: Any, where: str, *, by_the_hour: bool) -> dict[str, Any]:
    from pydantic import ValidationError

    from .config import Budget

    try:
        budget = Budget.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        place = ".".join(str(part) for part in first.get("loc", ()))
        message = str(first.get("msg")).removeprefix("Value error, ")
        raise PlanRefused(f"{where}.budget{'.' + place if place else ''}: {message}") from None
    if by_the_hour and budget.per_request is not None:
        raise PlanRefused(f"{where}.budget.per_request limits one hosted answer; a rented machine is paid by the hour — "
                          "name per_day or per_month")
    return budget.model_dump(exclude_none=True)


def _source(raw: Any, role: str, index: int) -> Source:
    where = f"{role}[{index}]"
    name: Optional[str] = None
    if isinstance(raw, str):
        kind, settings = raw, {}
    elif isinstance(raw, dict):
        name = raw.get("name")
        kinds = [k for k in raw if k != "name"]
        if len(kinds) != 1:
            raise PlanRefused(f"{where} names one source — one of {', '.join(SOURCE_TYPES)} — and optionally its name")
        kind, settings = kinds[0], raw[kinds[0]]
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise PlanRefused(f"{where}.{kind} is an object of its settings")
    else:
        raise PlanRefused(f"{where} is a source: a word ({', '.join(SOURCE_TYPES)}) or an object")
    if kind not in SOURCE_TYPES:
        raise PlanRefused(f"{where}: {kind!r} is not a source; the sources are {', '.join(SOURCE_TYPES)}")
    if kind not in ROLE_TAKES[role]:
        why = {
            ("until_ready", "rented"): "rented machines are what until_ready covers for",
            ("when_full", "local"): "borrowing the pool's machines ends when the workload's own are ready (D115)",
            ("when_full", "fixed"): "borrowing the pool's machines ends when the workload's own are ready (D115)",
            ("when_full", "shared"): "borrowing the pool's machines ends when the workload's own are ready (D115)",
        }.get((role, kind), f"{role} takes {', '.join(ROLE_TAKES[role])}")
        raise PlanRefused(f"{where}: {kind} is not a source of {role} — {why}")
    if name is not None:
        if not isinstance(name, str) or not name or len(name) > 40 or "/" in name or not name.strip() == name:
            raise PlanRefused(f"{where}.name is a short word or two, without '/'")
        if name in SOURCE_TYPES:
            raise PlanRefused(f"{where}.name {name!r} is a source's type; name it something else")

    if kind in ("local", "fixed", "shared"):
        if settings:
            raise PlanRefused(f"{where}: {kind} takes no settings; it is bounded by workloads.borrow_share")
        return Source(type=kind, name=name or kind)
    if kind == "hosted":
        unknown = sorted(set(settings) - {"after_s", "budget", "reserve_max_tokens", "providers"})
        if unknown:
            raise PlanRefused(f"{where}.hosted takes after_s, budget and reserve_max_tokens, not {unknown}")
        if "providers" in settings:
            raise PlanRefused(f"{where}.hosted.providers arrives in a later version of the pool; "
                              "until then hosted APIs are tried in the pool's own order")
        if "budget" not in settings:
            raise PlanRefused(f"{where}.hosted needs a budget: per_day and/or per_month, in dollars")
        default_wait = 5.0 if role == "when_full" else 0.0
        after_s = _number(settings.get("after_s", default_wait), f"{where}.hosted.after_s", high=3600)
        tokens = settings.get("reserve_max_tokens", 4096)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise PlanRefused(f"{where}.hosted.reserve_max_tokens is a whole number of at least 1")
        return Source(type="hosted", name=name or "hosted", after_s=after_s,
                      budget=_budget(settings["budget"], f"{where}.hosted", by_the_hour=False),
                      reserve_max_tokens=tokens)
    # rented, in when_full
    unknown = sorted(set(settings) - {"after_min", "max_hosts", "budget", "kind"})
    if unknown:
        raise PlanRefused(f"{where}.rented takes after_min, max_hosts and budget, not {unknown}")
    if "kind" in settings:
        raise PlanRefused(f"{where}.rented.kind arrives in a later version of the pool; until then it rents the "
                          "workload's own kind of machine (machines=)")
    if "max_hosts" not in settings or "budget" not in settings:
        raise PlanRefused(f"{where}.rented needs max_hosts and a budget")
    max_hosts = settings["max_hosts"]
    if isinstance(max_hosts, bool) or not isinstance(max_hosts, int) or not 1 <= max_hosts <= 100:
        raise PlanRefused(f"{where}.rented.max_hosts is a whole number from 1 to 100")
    return Source(type="rented", name=name or "rented",
                  after_min=_number(settings.get("after_min", 10.0), f"{where}.rented.after_min", high=24 * 60),
                  max_hosts=max_hosts, budget=_budget(settings["budget"], f"{where}.rented", by_the_hour=True))


def _role(raw: Any, role: str, defined: dict[str, Source]) -> tuple[Source, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PlanRefused(f"{role} is a list of sources, in the order they are tried")
    sources: list[Source] = []
    for index, item in enumerate(raw):
        # A name another role defines: the same source, with the same budget (§14.2).
        if isinstance(item, str) and item not in SOURCE_TYPES and item in defined:
            source = defined[item]
            if source.type not in ROLE_TAKES[role]:
                raise PlanRefused(f"{role}[{index}]: {item!r} is a {source.type} source, which {role} does not take")
            sources.append(source)
            continue
        if isinstance(item, str) and item not in SOURCE_TYPES:
            raise PlanRefused(f"{role}[{index}]: {item!r} is neither a source ({', '.join(SOURCE_TYPES)}) nor the "
                              "name of one the plan defines")
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"] in defined:
            sources.append(defined[item["name"]])  # read once, before the roles: one source by its name
            continue
        sources.append(_source(item, role, index))
    names = [s.name for s in sources]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise PlanRefused(f"{role} names {repeated} more than once")
    return tuple(sources)


def parse_serve(raw: Any) -> ServingPlan:
    """A plan as sent — `serve` — read, or refused in words."""
    if not isinstance(raw, dict):
        raise PlanRefused("serve is an object: {\"until_ready\": [...], \"when_full\": [...]}")
    for field, why in LATER.items():
        if field in raw:
            raise PlanRefused(f"serve.{field}: {why}")
    unknown = sorted(set(raw) - set(_TOP))
    if unknown:
        raise PlanRefused(f"serve takes steady, until_ready, during, max_min and when_full, not {unknown}")

    steady = raw.get("steady", "rented")
    if steady != "rented":
        if steady in ("local", "fixed"):
            raise PlanRefused(f"serve.steady {steady!r} needs workers the operator reserves, which arrive in a later "
                              "version of the pool; until then a workload runs on machines rented for it")
        raise PlanRefused("serve.steady is \"rented\": the workload's own rented machines, sized by parallel and "
                          "latency_s")

    # Named sources first, from both roles: either role may list, by its name, one the other defines.
    defined: dict[str, Source] = {}
    for role in ("until_ready", "when_full"):
        listed = raw.get(role)
        for index, item in enumerate(listed if isinstance(listed, list) else []):
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                source = _source(item, role, index)
                if source.name in defined:
                    raise PlanRefused(f"two sources are named {source.name!r}; define it once and list it by its "
                                      "name wherever else it is used")
                defined[source.name] = source
    until_ready = _role(raw.get("until_ready"), "until_ready", defined)
    when_full = _role(raw.get("when_full"), "when_full", defined)

    hosted = {id(s) for s in (*until_ready, *when_full) if s.type == "hosted"}
    if len(hosted) > 1:
        raise PlanRefused("one hosted source for now — a budget for each of several arrives in a later version of "
                          "the pool; name it, and list it by its name in the other role to use it in both")
    if len([s for s in when_full if s.type == "rented"]) > 1:
        raise PlanRefused("one rented source in when_full")
    by_name: dict[str, Source] = {}
    for source in (*until_ready, *when_full):
        if by_name.setdefault(source.name, source) is not source:
            raise PlanRefused(f"two sources are named {source.name!r}; a name is the plan's word for one source")

    during: tuple[str, ...] = DEFAULT_DURING
    max_min: Optional[float] = DEFAULT_MAX_MIN
    if "during" in raw or "max_min" in raw:
        if not until_ready:
            raise PlanRefused("during and max_min say when until_ready is used, and the plan has no until_ready")
    if "during" in raw:
        listed = raw["during"]
        if not isinstance(listed, list) or not listed or any(d not in DURING for d in listed):
            raise PlanRefused(f"serve.during lists one or more of {', '.join(DURING)}")
        if "scale_up" in listed:
            raise PlanRefused("serve.during: covering a scale-up arrives in a later version of the pool")
        during = tuple(d for d in DURING if d in listed)
    if "max_min" in raw:
        max_min = _number(raw["max_min"], "serve.max_min", low=1, high=MAX_MAX_MIN)
    return ServingPlan(steady="rented", until_ready=until_ready, during=during, max_min=max_min, when_full=when_full)


# --- today's fields as a plan (the shorthand) ---


def from_bursts(policy: Mapping[str, Any], may_borrow: bool) -> ServingPlan:
    """The plan a workload with no `serve` has (workloads.md §14.1): until_ready is the shared
    workload where it may borrow, then its hosted burst, during the start and with no limit;
    when_full is its bursts. The hosted burst is one source in both roles, with one budget.

    Its meaning, never stored: such a workload is routed from its own fields, as before S19. This
    is what the test proving the two route alike compares with, and what a readback describes."""
    hosted_raw, rented_raw = policy.get("hosted"), policy.get("rented")
    hosted = None
    if hosted_raw:
        hosted = Source(type="hosted", name="hosted", after_s=float(hosted_raw.get("after_s", 5.0)),
                        budget=dict(hosted_raw["budget"]), reserve_max_tokens=int(hosted_raw.get("reserve_max_tokens", 4096)))
    until_ready = tuple(s for s in (Source(type="shared", name="shared") if may_borrow else None, hosted) if s is not None)
    rented = None
    if rented_raw:
        rented = Source(type="rented", name="rented", after_min=float(rented_raw.get("after_min", 10.0)),
                        max_hosts=int(rented_raw["max_hosts"]), budget=dict(rented_raw["budget"]))
    when_full = tuple(s for s in (hosted, rented) if s is not None)
    return ServingPlan(steady="rented", until_ready=until_ready, during=("start",), max_min=None, when_full=when_full)


# --- the policy the pool already reads ---


def compile_policy(plan: ServingPlan) -> dict[str, Any]:
    """The policy (D165) this plan carries: its hosted source as `hosted` — the budget every
    hosted answer is held against — its rented burst as `rented`, and the plan itself as `serve`,
    which the router routes by. A supervisor or router that reads only `hosted` and `rented`
    still bounds every dollar the plan can spend."""
    if plan.max_min is None:
        # Only the shorthand has no limit, and it is never stored as a plan (`from_bursts`).
        raise ValueError("a stored plan has a max_min")
    policy: dict[str, Any] = {"serve": plan.as_dict()}
    hosted = plan.hosted()
    if hosted is not None:
        # The wait a reader of `hosted` alone would use: when_full's, else until_ready's.
        wait = plan.when_full_hosted().after_s if plan.when_full_hosted() is not None else hosted.after_s
        policy["hosted"] = {"after_s": wait, "budget": hosted.budget, "reserve_max_tokens": hosted.reserve_max_tokens}
    rented = plan.rented()
    if rented is not None:
        policy["rented"] = {"after_min": rented.after_min, "max_hosts": rented.max_hosts, "budget": rented.budget}
    return policy


def plan_of(policy: Mapping[str, Any]) -> Optional[ServingPlan]:
    """The plan a stored policy carries, or None where it carries none (today's fields)."""
    raw = policy.get("serve") if policy else None
    return parse_serve(raw) if raw else None


def describe(plan: ServingPlan) -> str:
    """A plan in words, for the event log and the plan check."""
    def named(source: Source) -> str:
        if source.type == "hosted":
            wait = f" after {source.after_s:g}s" if source.after_s else " at once"
            return f"{source.name} (hosted APIs{wait}, within {source.budget})"
        if source.type == "rented":
            return (f"{source.name} (up to {source.max_hosts} more rented host(s) once a burst has lasted "
                    f"{source.after_min:g} min, within {source.budget})")
        return {"local": "local (the pool's own computer)", "fixed": "fixed (the operator's machines elsewhere)",
                "shared": "shared (any free machine of the pool)"}[source.type] + (
                    f" as {source.name}" if source.name != source.type else "")

    said = ["steady on machines rented for it"]
    if plan.until_ready:
        limit = f", for at most {plan.max_min:g} min each time" if plan.max_min is not None else ""
        said.append(f"until ready ({' and '.join(plan.during)}{limit}): " + ", then ".join(named(s) for s in plan.until_ready))
    else:
        said.append("no stop-gap (until_ready not set)")
    said.append("when full: " + ", then ".join(named(s) for s in plan.when_full) if plan.when_full else "when full: the queue")
    return "; ".join(said)


# --- how a request may be routed now ---


@dataclasses.dataclass(frozen=True)
class Routing:
    """What the router puts in a request's need from its workload's plan."""

    #: A gap is open: until_ready's sources may be used.
    covering: bool
    #: May borrow the shared workload's machines, and of which host kinds (None: any).
    may_borrow: bool
    borrow_kinds: Optional[frozenset[str]]
    #: Borrow even though the workload has a ready host of its own for the model (a replacement).
    borrow_while_ready: bool
    #: Seconds after arrival a hosted API may take the request, or None: never.
    hosted_after_s: Optional[float]
    #: The ranking of until_ready's sources while covering, by type; empty when not covering.
    order: tuple[str, ...]


def route(plan: ServingPlan, *, preparing: bool, model_ready: bool, created_at: float, now: float,
          gap_ends_at: Optional[float], fresh: bool, borrow_share: float, short: bool = True) -> Routing:
    """How a request may go now, by its workload's plan and the facts of the moment:

    - `preparing`: the workload has not yet served (its state, as the router reads it);
    - `model_ready`: one of its own ready hosts holds the request's model;
    - `gap_ends_at`: the end of a replacement gap the supervisor marked for this model, if any;
    - `fresh`: the supervisor's heartbeat is recent. A stale one opens no paid stop-gap and no
      replacement gap: only a start, on sources that spend nothing (§14.7);
    - `short`: no host holding the model has become ready since the supervisor marked the gap. A
      gap ends the moment one has, without waiting for the supervisor's next pass.

    A start gap is the router's own to judge: from the workload's creation, while the model has
    no ready host of its own, for at most `max_min`."""
    starting = (
        "start" in plan.during and preparing and not model_ready
        and (plan.max_min is None or now < created_at + plan.max_min * 60)
    )
    replacing = (
        "replacement" in plan.during and fresh and short and gap_ends_at is not None and now < gap_ends_at
    )
    covering = bool(plan.until_ready) and (starting or replacing)
    borrows = plan.borrows() if covering else ()
    kinds: Optional[frozenset[str]] = None
    if borrows and "shared" not in borrows:
        kinds = frozenset(k for b in borrows for k in HOST_KINDS[b])
    may_borrow = bool(borrows) and borrow_share > 0

    waits: list[float] = []
    cover = plan.until_ready_hosted()
    if covering and cover is not None and fresh:
        waits.append(cover.after_s or 0.0)
    burst = plan.when_full_hosted()
    if burst is not None:
        waits.append(burst.after_s if burst.after_s is not None else 5.0)
    order = plan.order() if covering else ()
    if not fresh:
        order = tuple(t for t in order if t != "hosted")
    return Routing(
        covering=covering, may_borrow=may_borrow, borrow_kinds=kinds,
        borrow_while_ready=replacing and may_borrow, hosted_after_s=min(waits) if waits else None, order=order,
    )
