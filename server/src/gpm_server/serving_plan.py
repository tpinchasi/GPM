"""A workload's serving plan (S19, D198–D204; docs/spec/workloads.md §14).

A plan says where a workload's requests go and when: **steady** (where it runs), **until_ready**
(a stop-gap used only while a machine of the workload is being prepared) and **when_full** (its
bursts), tried in that order. How much it must serve stays in `parallel` and `latency_s`.

This module is pure: it reads a plan, refuses what it cannot be in words, compiles it to the
policy the rest of the pool already reads (`hosted`, `rented`, D165), writes today's fields as
the plan they mean (the shorthand), and works out — from facts the router hands it — how a
request may be routed now. Nothing here touches the database, the clock or the network.

Built in stages (the story's build order). What this stage does not do yet is refused in words,
never accepted and ignored: rules, a substitute, and more than one rented source.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

#: The words a plan may use for where a request goes.
SOURCE_TYPES = ("local", "fixed", "shared", "rented", "hosted")
#: Which source types each role takes (workloads.md §14.1). Rented machines are what until_ready
#: covers for; borrowing never outlasts the start (D115); hosted APIs are never where a workload runs.
ROLE_TAKES = {
    "steady": ("rented", "local", "fixed"),
    "until_ready": ("local", "fixed", "shared", "hosted"),
    "when_full": ("hosted", "rented"),
}
#: Moments until_ready may cover; `start` and `replacement` unless the plan says otherwise.
DURING = ("start", "replacement", "scale_up")
DEFAULT_DURING = ("start", "replacement")
DEFAULT_MAX_MIN = 60.0
#: Host kinds each machine source reaches. `shared` is any machine of the shared workload.
HOST_KINDS = {"local": ("local",), "fixed": ("fixed-remote",)}
#: The kinds of machine a rented source may name (`machines=`'s words).
RENTED_KINDS = ("roi", "on_demand", "interruptible")
#: Fields of the plan a later stage brings; refused in words until then.
LATER = {
    "substitute": "a substitute model arrives in a later version of the pool",
}
#: A request's options (§14.4, D201): each header's values, and the one meant when a request sends
#: none and its plan names no default.
OPTIONS = {"priority": ("interactive", "batch"), "keep_on": ("no_hosted", "owned", "any")}
DEFAULT_OPTIONS = {"priority": "interactive", "keep_on": "any"}
OPTION_HEADERS = {"priority": "X-GPM-Priority", "keep_on": "X-GPM-Keep-On"}
_TOP = ("steady", "until_ready", "when_full", "during", "max_min", "defaults", "reserve", "rules", "tz", *LATER)
#: A plan's rules (§14.3, D200): a closed list of conditions and effects, at most 16.
MAX_RULES = 16
WHEN = ("priority", "keep_on", "between", "days", "spent_over", "evictions")
THEN = ("use_only", "never_use", "after_s", "steady")
#: When a rule's steady switch ends (§14.3): with the window it opened in, or at the end of the
#: day it began on, in the plan's time zone; with neither, when the rule is no longer in force.
UNTIL = ("window_end", "end_of_day")
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
#: Conditions the supervisor judges each pass; the rest are a request's, judged by the router.
PASS_CONDITIONS = ("between", "days", "spent_over", "evictions")
#: The most workers one plan may reserve of a kind (`reserve: {local: n}`).
MAX_RESERVE = 1024
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
    #: Whether the plan named it; an unnamed one is called by its type, numbered from the second
    #: of that type in its role (`hosted`, `hosted-2`), and `role` is the role that defines it.
    named: bool = False
    role: str = ""
    #: A hosted source's hosted APIs, by host id, in the order they are tried; empty: every hosted
    #: host serving the model, in the pool's order.
    providers: tuple[str, ...] = ()
    #: A rented source's kind of machine (`roi`, `on_demand`, `interruptible`); None: the kind
    #: steady rents in force — the workload's `machines=`.
    kind: Optional[str] = None

    @property
    def spends(self) -> bool:
        return self.type in ("hosted", "rented")

    @property
    def budget_id(self) -> str:
        """What its budget is kept under (§14.2): its name, or `<role>/<name>` for an unnamed one."""
        return self.name if self.named else f"{self.role}/{self.name}"

    def serves(self, host_id: str) -> bool:
        """Whether a hosted host is one of this source's providers."""
        return not self.providers or host_id in self.providers

    def as_dict(self) -> Any:
        """As a plan would write it: a bare type where it carries nothing else."""
        settings = {k: v for k, v in (("after_s", self.after_s), ("after_min", self.after_min),
                                      ("max_hosts", self.max_hosts), ("budget", self.budget),
                                      ("reserve_max_tokens", self.reserve_max_tokens),
                                      ("providers", list(self.providers) or None), ("kind", self.kind))
                    if v is not None}
        if not settings and not self.named:
            return self.type
        written: dict[str, Any] = {self.type: settings}
        if self.named:
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
    #: A request option's value where a request sends none (§14.4).
    defaults: Mapping[str, str] = dataclasses.field(default_factory=dict)
    #: Workers of the operator's own machines this workload has the first call on, by source
    #: type (`local`, `fixed`) — the operator's alone to set (§14.2).
    reserve: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: Its rules, in order (§14.3), and the time zone their windows are read in.
    rules: tuple["Rule", ...] = ()
    tz: str = "UTC"

    @property
    def steady_kinds(self) -> frozenset[str]:
        """The host kinds steady runs on where it is the operator's machines; empty: rented."""
        return frozenset(HOST_KINDS.get(self.steady, ()))

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
        if self.defaults:
            written["defaults"] = dict(self.defaults)
        if self.reserve:
            written["reserve"] = dict(self.reserve)
        if self.rules:
            written["rules"] = [r.as_dict() for r in self.rules]
        if self.tz != "UTC":
            written["tz"] = self.tz
        return written

    def sources(self) -> tuple[Source, ...]:
        """Every source of its roles once (a named one listed in both roles once)."""
        seen: list[Source] = []
        for source in (*self.until_ready, *self.when_full):
            if not any(source is s for s in seen):
                seen.append(source)
        return tuple(seen)

    def source_types(self) -> set[str]:
        return {self.steady, *(s.type for s in self.until_ready), *(s.type for s in self.when_full)}

    def hosted(self) -> Optional[Source]:
        return next((s for s in (*self.until_ready, *self.when_full) if s.type == "hosted"), None)

    def hosted_sources(self) -> tuple[Source, ...]:
        """Every hosted source once, in the plan's order (a named one listed in both roles once)."""
        seen: list[Source] = []
        for source in (*self.until_ready, *self.when_full):
            if source.type == "hosted" and not any(source is s for s in seen):
                seen.append(source)
        return tuple(seen)

    def hosted_total(self, days_in_month: int = 31) -> dict[str, float]:
        """The workload's one hosted total (§14.2): per day and per month, the sum of its hosted
        sources' budgets for that period; a source with no budget for a period counts its month's
        toward a day, or its day's times the days in the month toward a month."""
        total: dict[str, float] = {}
        sources = self.hosted_sources()
        if not sources:
            return total
        day = month = 0.0
        for source in sources:
            budget = source.budget or {}
            per_day, per_month = budget.get("per_day"), budget.get("per_month")
            day += per_day if per_day is not None else per_month
            month += per_month if per_month is not None else per_day * days_in_month
        return {"per_day": round(day, 6), "per_month": round(month, 6)}

    def rented(self) -> Optional[Source]:
        return next((s for s in self.when_full if s.type == "rented"), None)

    def until_ready_hosted(self) -> Optional[Source]:
        return next((s for s in self.until_ready if s.type == "hosted"), None)

    def when_full_hosted(self) -> Optional[Source]:
        return next((s for s in self.when_full if s.type == "hosted"), None)

    def role_hosted(self, role: str) -> tuple[Source, ...]:
        return tuple(s for s in (self.until_ready if role == "until_ready" else self.when_full) if s.type == "hosted")

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
        return Source(type=kind, name=name or kind, named=name is not None, role=role)
    if kind == "hosted":
        unknown = sorted(set(settings) - {"after_s", "budget", "reserve_max_tokens", "providers"})
        if unknown:
            raise PlanRefused(f"{where}.hosted takes after_s, budget, reserve_max_tokens and providers, not {unknown}")
        providers = settings.get("providers", [])
        if (not isinstance(providers, list) or len(providers) > 16
                or any(not isinstance(p, str) or not p or len(p) > 64 for p in providers)
                or len(set(providers)) != len(providers)):
            raise PlanRefused(f"{where}.hosted.providers is a list of hosted host ids, each once, at most 16")
        if "budget" not in settings:
            raise PlanRefused(f"{where}.hosted needs a budget: per_day and/or per_month, in dollars")
        default_wait = 5.0 if role == "when_full" else 0.0
        after_s = _number(settings.get("after_s", default_wait), f"{where}.hosted.after_s", high=3600)
        tokens = settings.get("reserve_max_tokens", 4096)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise PlanRefused(f"{where}.hosted.reserve_max_tokens is a whole number of at least 1")
        return Source(type="hosted", name=name or "hosted", after_s=after_s,
                      budget=_budget(settings["budget"], f"{where}.hosted", by_the_hour=False),
                      reserve_max_tokens=tokens, named=name is not None, role=role, providers=tuple(providers))
    # rented, in when_full
    unknown = sorted(set(settings) - {"after_min", "max_hosts", "budget", "kind"})
    if unknown:
        raise PlanRefused(f"{where}.rented takes after_min, max_hosts, budget and kind, not {unknown}")
    kind = settings.get("kind")
    if kind is not None and kind not in RENTED_KINDS:
        raise PlanRefused(f"{where}.rented.kind is one of {', '.join(RENTED_KINDS)}; left out, it rents the kind "
                          "steady rents (machines=)")
    if "max_hosts" not in settings or "budget" not in settings:
        raise PlanRefused(f"{where}.rented needs max_hosts and a budget")
    max_hosts = settings["max_hosts"]
    if isinstance(max_hosts, bool) or not isinstance(max_hosts, int) or not 1 <= max_hosts <= 100:
        raise PlanRefused(f"{where}.rented.max_hosts is a whole number from 1 to 100")
    return Source(type="rented", name=name or "rented",
                  after_min=_number(settings.get("after_min", 10.0), f"{where}.rented.after_min", high=24 * 60),
                  max_hosts=max_hosts, budget=_budget(settings["budget"], f"{where}.rented", by_the_hour=True),
                  named=name is not None, role=role, kind=kind)


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
        source = _source(item, role, index)
        if not source.named and source.spends:
            # The second unnamed spending source of a type in a role is `<type>-2`, and so on
            # (§14.2): each has a budget of its own. A machine source listed twice is refused.
            count = sum(1 for s in sources if not s.named and s.type == source.type)
            if count:
                source = dataclasses.replace(source, name=f"{source.type}-{count + 1}")
        sources.append(source)
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
        raise PlanRefused(f"serve takes steady, until_ready, during, max_min, when_full, defaults, reserve, rules "
                          f"and tz, not {unknown}")

    steady = raw.get("steady", "rented")
    if steady not in ROLE_TAKES["steady"]:
        raise PlanRefused("serve.steady is \"rented\" (the workload's own rented machines, the default), \"local\" "
                          "or \"fixed\" (the operator's own machines, with workers reserved for it)")
    reserve = _reserve(raw.get("reserve"))
    if steady != "rented" and not reserve.get(steady):
        raise PlanRefused(f"serve.steady {steady!r} runs on workers the operator reserves for it: add "
                          f"reserve: {{\"{steady}\": <workers>}}")

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

    if len([s for s in when_full if s.type == "rented"]) > 1:
        raise PlanRefused("one rented source in when_full")
    by_name: dict[str, Source] = {}
    for source in (*until_ready, *when_full):
        if source.named and by_name.setdefault(source.name, source) is not source:
            raise PlanRefused(f"two sources are named {source.name!r}; a name is the plan's word for one source")
    unnamed = {s.name for s in (*until_ready, *when_full) if not s.named}
    clash = sorted(unnamed & set(by_name))
    if clash:
        raise PlanRefused(f"{clash[0]!r} is what an unnamed source is called; name yours something else")

    during: tuple[str, ...] = DEFAULT_DURING
    max_min: Optional[float] = DEFAULT_MAX_MIN
    if "during" in raw or "max_min" in raw:
        if not until_ready:
            raise PlanRefused("during and max_min say when until_ready is used, and the plan has no until_ready")
    if "during" in raw:
        listed = raw["during"]
        if not isinstance(listed, list) or not listed or any(d not in DURING for d in listed):
            raise PlanRefused(f"serve.during lists one or more of {', '.join(DURING)}")
        during = tuple(d for d in DURING if d in listed)
    if "max_min" in raw:
        max_min = _number(raw["max_min"], "serve.max_min", low=1, high=MAX_MAX_MIN)
    if steady != "rented":
        if until_ready:
            raise PlanRefused(f"until_ready covers while the workload's rented machines are prepared; with steady "
                              f"{steady!r} nothing is rented, so there is nothing to cover")
        if any(s.type == "rented" for s in when_full):
            raise PlanRefused(f"a rented burst above steady {steady!r} arrives with rules, in a later version of the "
                              "pool; when_full may use hosted APIs")
    defaults = _defaults(raw.get("defaults"), owned_steady=steady != "rented")
    plan = ServingPlan(steady=steady, until_ready=until_ready, during=during, max_min=max_min, when_full=when_full,
                       defaults=defaults, reserve=reserve, tz=_tz(raw.get("tz", "UTC")))
    return dataclasses.replace(plan, rules=_rules(raw.get("rules"), plan))


# --- rules (§14.3, D200) ---


@dataclasses.dataclass(frozen=True)
class Rule:
    name: str
    when: Mapping[str, Any]
    then: Mapping[str, Any]
    until: Optional[str] = None

    @property
    def on_the_pass(self) -> bool:
        """Whether a condition of it is the supervisor's to judge (a time, a spend, evictions);
        one judged on a request's options alone is in force whenever they match."""
        return any(k in self.when for k in PASS_CONDITIONS)

    def matches(self, priority: str, keep_on: str) -> bool:
        return (self.when.get("priority", priority) == priority and self.when.get("keep_on", keep_on) == keep_on)

    def as_dict(self) -> dict[str, Any]:
        written = {"name": self.name, "when": dict(self.when), "then": dict(self.then)}
        if self.until is not None:
            written["until"] = self.until
        return written

    @property
    def switch(self) -> Optional[str]:
        """Where its steady switch moves the workload: `local`, `fixed`, or `rented:<kind>`."""
        steady = self.then.get("steady")
        if steady is None:
            return None
        if isinstance(steady, str):
            return steady
        return f"rented:{steady['rented'].get('kind', 'roi')}"


def _tz(raw: Any) -> str:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    if not isinstance(raw, str) or not raw:
        raise PlanRefused("serve.tz is an IANA time zone, such as America/New_York")
    try:
        ZoneInfo(raw)
    except (ZoneInfoNotFoundError, ValueError):
        raise PlanRefused(f"serve.tz {raw!r} is not an IANA time zone, such as America/New_York") from None
    return raw


def _hhmm(text: str) -> int:
    hours, minutes = text.split(":")
    if len(hours) != 2 or len(minutes) != 2 or not (hours + minutes).isdigit():
        raise ValueError(text)
    value = int(hours) * 60 + int(minutes)
    if int(hours) > 23 or int(minutes) > 59:
        raise ValueError(text)
    return value


def window_of(between: str) -> tuple[int, int]:
    """`"HH:MM-HH:MM"` as minutes of the day, start and end; the end may be before the start (the
    window crosses midnight)."""
    start, end = between.split("-")
    return _hhmm(start), _hhmm(end)


def in_window(when: Mapping[str, Any], tz: str, now: float) -> bool:
    """Whether `now` is inside a rule's `between` (and `days`), read in the plan's time zone. A
    time a DST change skips takes effect at the first minute after it (the clock is past it); in
    an hour DST repeats, every boundary took effect at its first occurrence, so the repeat is read
    as its last minute."""
    import datetime
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)
    local = datetime.datetime.fromtimestamp(now, zone)
    if local.fold:
        probe = now
        while datetime.datetime.fromtimestamp(probe, zone).fold:
            probe += 60
        local = datetime.datetime.fromtimestamp(probe - 60, zone).replace(fold=0)
    minute = local.hour * 60 + local.minute
    start, end = window_of(when["between"])
    inside = (start <= minute < end) if start < end else (minute >= start or minute < end)
    if not inside:
        return False
    if "days" in when:
        # The day the window opened: yesterday, past midnight.
        opened = local.date() - (datetime.timedelta(days=1) if start > end and minute < end else datetime.timedelta())
        if DAYS[opened.weekday()] not in when["days"]:
            return False
    return True


def source_keys(plan: ServingPlan, words: Any, where: str) -> frozenset[str]:
    """The sources a rule's words name, by their budget ids: a type names every source of it, a
    name the one so called. A word naming nothing in the plan is refused."""
    if not isinstance(words, list) or not words or any(not isinstance(w, str) for w in words):
        raise PlanRefused(f"{where} is a list of source names or types")
    keys: set[str] = set()
    for word in words:
        # A type names every source of it; a name, or a budget id (`when_full/rented`), the one.
        found = [s for s in plan.sources()
                 if (s.type == word if word in SOURCE_TYPES else word in (s.name, s.budget_id))]
        if not found and not (word in SOURCE_TYPES and word == plan.steady):
            raise PlanRefused(f"{where}: {word!r} is no source of this plan")
        keys.update(s.budget_id for s in found)
    return frozenset(keys)


def _rules(raw: Any, plan: ServingPlan) -> tuple[Rule, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PlanRefused("serve.rules is a list of {name, when, then}, tried in order")
    if len(raw) > MAX_RULES:
        raise PlanRefused(f"serve.rules: at most {MAX_RULES} rules, not {len(raw)}")
    rules: list[Rule] = []
    for index, item in enumerate(raw):
        where = f"rules[{index}]"
        if not isinstance(item, dict) or set(item) - {"name", "when", "then", "until"}:
            raise PlanRefused(f"{where} is {{name, when, then}}")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise PlanRefused(f"{where}.name is a few words naming the rule")
        if any(r.name == name for r in rules):
            raise PlanRefused(f"{where}: two rules are named {name!r}")
        where = f"rule {name!r}"
        when, then = item.get("when"), item.get("then")
        if not isinstance(when, dict) or not when:
            raise PlanRefused(f"{where}.when names at least one condition: {', '.join(WHEN)}")
        if not isinstance(then, dict) or not then:
            raise PlanRefused(f"{where}.then names at least one effect: {', '.join(THEN)}")
        rule = Rule(name=name, when=_when(when, plan, where), then=_then(then, plan, where))
        if "steady" in rule.then and ("priority" in when or "keep_on" in when):
            raise PlanRefused(f"{where}: a steady switch is per workload, never per request — its when names a "
                              "request's option")
        if ("priority" in when or "keep_on" in when) and plan.rented() is not None:
            # A burst is sized from every request's load: one kept from some requests alone would
            # be rented for load it may not serve.
            only = rule.then.get("use_only")
            never = rule.then.get("never_use")
            out = (source_keys(plan, never, where) if never else frozenset()) | (
                frozenset(s.budget_id for s in plan.sources()) - source_keys(plan, only, where) if only else frozenset())
            if plan.rented().budget_id in out:
                raise PlanRefused(f"{where}: a rule on a request's options cannot keep the rented burst out — it is "
                                  "sized from every request's load; keep it out for all requests (a condition on "
                                  "the pass), or send keep_on owned")
        until = item.get("until")
        if until is not None:
            if "steady" not in rule.then:
                raise PlanRefused(f"{where}.until holds a steady switch; this rule has none")
            if until not in UNTIL:
                raise PlanRefused(f"{where}.until is window_end or end_of_day")
            if until == "window_end" and "between" not in rule.when:
                raise PlanRefused(f"{where}.until window_end needs a window: when.between")
            rule = dataclasses.replace(rule, until=until)
        rules.append(rule)
    return tuple(rules)


def _when(when: dict, plan: ServingPlan, where: str) -> dict[str, Any]:
    unknown = sorted(set(when) - set(WHEN))
    if unknown:
        raise PlanRefused(f"{where}.when: {unknown} — the conditions are {', '.join(WHEN)}")
    for option in ("priority", "keep_on"):
        if option in when and when[option] not in OPTIONS[option]:
            raise PlanRefused(f"{where}.when.{option} is {_listed(OPTIONS[option])}")
    if "days" in when and "between" not in when:
        raise PlanRefused(f"{where}.when.days goes with between")
    if "between" in when:
        try:
            start, end = window_of(when["between"]) if isinstance(when["between"], str) else (None, None)
        except ValueError:
            start = end = None
        if start is None or start == end:
            raise PlanRefused(f"{where}.when.between is \"HH:MM-HH:MM\" (it may cross midnight), two different times")
    if "days" in when:
        days = when["days"]
        if not isinstance(days, list) or not days or any(d not in DAYS for d in days) or len(set(days)) != len(days):
            raise PlanRefused(f"{where}.when.days lists days, each once: {', '.join(DAYS)}")
    if "spent_over" in when:
        spent = when["spent_over"]
        if not isinstance(spent, dict) or set(spent) - {"budget", "period", "share"} or "budget" not in spent:
            raise PlanRefused(f"{where}.when.spent_over is {{budget: <budget id> | lease, period: day | month, share}}")
        share = spent.get("share")
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 < share <= 1:
            raise PlanRefused(f"{where}.when.spent_over.share is a share of the budget, above 0 and at most 1")
        if spent["budget"] != "lease":
            period = spent.get("period")
            if period not in ("day", "month"):
                raise PlanRefused(f"{where}.when.spent_over.period is day or month")
            source = next((s for s in plan.sources() if s.spends and s.budget_id == spent["budget"]), None)
            if source is None:
                ids = [s.budget_id for s in plan.sources() if s.spends]
                raise PlanRefused(f"{where}.when.spent_over.budget {spent['budget']!r} is no budget of this plan; its "
                                  f"budgets are {', '.join(ids + ['lease'])}")
            if (source.budget or {}).get(f"per_{period}") is None:
                raise PlanRefused(f"{where}.when.spent_over: budget {spent['budget']!r} has no per_{period} limit")
        elif plan.steady != "rented":
            raise PlanRefused(f"{where}.when.spent_over.budget lease: steady {plan.steady!r} rents nothing")
    if "evictions" in when:
        ev = when["evictions"]
        count, within = (ev.get("count"), ev.get("within_h")) if isinstance(ev, dict) else (None, None)
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1 or isinstance(within, bool)
                or not isinstance(within, (int, float)) or not 0 < within <= 24 * 7 or set(ev) - {"count", "within_h"}):
            raise PlanRefused(f"{where}.when.evictions is {{count: a whole number of at least 1, within_h: hours up to "
                              "168}")
    return dict(when)


def _switch(steady: Any, plan: ServingPlan, where: str) -> None:
    """A rule's steady switch (§14.3): to the operator's own machines with a reservation, or to
    rented machines of another kind. Where the workload runs on its rented machines, as created."""
    if plan.steady != "rented":
        raise PlanRefused(f"{where}.then.steady: a switch from steady {plan.steady!r} would rent a base the workload "
                          "was created without; create it on rented machines and switch to local or fixed")
    if steady in ("local", "fixed"):
        if not plan.reserve.get(steady):
            raise PlanRefused(f"{where}.then.steady {steady!r} runs on workers the operator reserves: add "
                              f"reserve: {{\"{steady}\": <workers>}}")
        return
    if isinstance(steady, dict) and set(steady) == {"rented"} and isinstance(steady["rented"], dict):
        kind = steady["rented"].get("kind")
        if set(steady["rented"]) - {"kind"} or kind not in RENTED_KINDS:
            raise PlanRefused(f"{where}.then.steady.rented.kind is one of {', '.join(RENTED_KINDS)}")
        return
    if steady == "rented":
        raise PlanRefused(f"{where}.then.steady: steady is rented already; to change the kind, say "
                          "{\"rented\": {\"kind\": ...}}")
    raise PlanRefused(f"{where}.then.steady is \"local\", \"fixed\" or {{\"rented\": {{\"kind\": ...}}}}")


def _then(then: dict, plan: ServingPlan, where: str) -> dict[str, Any]:
    if "steady" in then:
        _switch(then["steady"], plan, where)
    if "substitute" in then:
        raise PlanRefused(f"{where}.then.substitute: a substitute model arrives in a later version of the pool")
    unknown = sorted(set(then) - set(THEN))
    if unknown:
        raise PlanRefused(f"{where}.then: {unknown} — the effects are {', '.join(THEN)}")
    if "use_only" in then:
        source_keys(plan, then["use_only"], f"{where}.then.use_only")
        if plan.steady not in then["use_only"]:
            raise PlanRefused(f"{where}.then.use_only leaves out steady ({plan.steady}), where the workload runs; list "
                              "it, or narrow with never_use")
    if "never_use" in then:
        source_keys(plan, then["never_use"], f"{where}.then.never_use")
        if plan.steady in then["never_use"]:
            raise PlanRefused(f"{where}.then.never_use removes steady ({plan.steady}), where the workload runs; name "
                              "the source you mean by its name or budget id (an unnamed rented burst is "
                              "when_full/rented)")
    if "after_s" in then:
        waits = then["after_s"]
        if not isinstance(waits, dict) or not waits:
            raise PlanRefused(f"{where}.then.after_s is {{<hosted source name>: seconds}}")
        for name, seconds in waits.items():
            named = [s for s in plan.hosted_sources() if name in (s.name, s.budget_id)]
            if not named:
                raise PlanRefused(f"{where}.then.after_s: {name!r} is no hosted source of this plan")
            longest = max(s.after_s or 0 for s in named)
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 <= seconds < longest:
                raise PlanRefused(f"{where}.then.after_s.{name}: a shorter wait only — from 0 to below its "
                                  f"{longest:g}s")
    return dict(then)


def end_of_day(tz: str, now: float) -> float:
    """The next midnight after `now` in the time zone, as a timestamp."""
    import datetime
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)
    local = datetime.datetime.fromtimestamp(now, zone)
    tomorrow = (local + datetime.timedelta(days=1)).date()
    return datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=zone).timestamp()


def steady_wanted(plan: ServingPlan, armed: frozenset[str], held: Optional[Mapping[str, Any]],
                  now: float) -> Optional[dict[str, Any]]:
    """The steady switch in force (§14.3): the last rule in force that switches, or one begun
    before and held by its `until` — with the window it opened in, or to the end of that day.
    None: steady is as the plan says. `held` is what the last pass found, which keeps its start."""
    switching = [r for r in plan.rules if r.switch is not None and r.name in armed]
    if switching:
        rule = switching[-1]
        if held is not None and held.get("rule") == rule.name and held.get("to") == rule.switch:
            return dict(held)
        return {"rule": rule.name, "to": rule.switch, "since": now,
                "until": end_of_day(plan.tz, now) if rule.until == "end_of_day" else None}
    if held is None:
        return None
    rule = next((r for r in plan.rules if r.name == held.get("rule") and r.switch == held.get("to")), None)
    if rule is None:
        return None
    if rule.until == "end_of_day" and held.get("until") is not None and now < held["until"]:
        return dict(held)
    if rule.until == "window_end" and in_window(rule.when, plan.tz, now):
        return dict(held)
    return None


@dataclasses.dataclass(frozen=True)
class Narrowing:
    """What the rules in force do to one request (§14.3): sources it may not use, by budget id,
    and shorter waits. They only narrow, and add up across every matching rule."""

    excluded: frozenset[str] = frozenset()
    waits: Mapping[str, float] = dataclasses.field(default_factory=dict)
    by: tuple[str, ...] = ()


def narrowing(plan: ServingPlan, armed: frozenset[str], priority: str, keep_on: str) -> Narrowing:
    """The rules in force for a request: each whose conditions on the pass the supervisor last
    found true (`armed`, by name) — or that has none — and whose conditions on a request's
    options match its options."""
    every = frozenset(s.budget_id for s in plan.sources())
    allowed, excluded, waits, by = every, set(), {}, []
    for rule in plan.rules:
        if (rule.on_the_pass and rule.name not in armed) or not rule.matches(priority, keep_on):
            continue
        by.append(rule.name)
        if "use_only" in rule.then:
            allowed = allowed & source_keys(plan, rule.then["use_only"], rule.name)
        if "never_use" in rule.then:
            excluded |= source_keys(plan, rule.then["never_use"], rule.name)
        for name, seconds in (rule.then.get("after_s") or {}).items():
            # Every hosted source so called (an unnamed one may be in both roles) whose wait is longer.
            for source in plan.hosted_sources():
                if name in (source.name, source.budget_id) and (source.after_s or 0) > seconds:
                    waits[source.budget_id] = min(seconds, waits.get(source.budget_id, seconds))
    return Narrowing(excluded=frozenset((every - allowed) | excluded), waits=waits, by=tuple(by))


def narrowed(plan: ServingPlan, by: Narrowing) -> ServingPlan:
    """The plan as a request sees it under its rules: the sources they exclude gone, the waits
    they shorten shorter. Budgets keep their ids."""
    if not by.excluded and not by.waits:
        return plan
    kept: dict[int, Optional[Source]] = {}
    for source in plan.sources():
        if source.budget_id in by.excluded:
            kept[id(source)] = None
        elif source.budget_id in by.waits:
            kept[id(source)] = dataclasses.replace(source, after_s=by.waits[source.budget_id])
        else:
            kept[id(source)] = source
    def role(sources: tuple[Source, ...]) -> tuple[Source, ...]:
        return tuple(kept[id(s)] for s in sources if kept[id(s)] is not None)
    return dataclasses.replace(plan, until_ready=role(plan.until_ready), when_full=role(plan.when_full))


def _reserve(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise PlanRefused("serve.reserve is an object of workers by source: {\"local\": 4} and/or {\"fixed\": 8}")
    unknown = sorted(set(raw) - set(HOST_KINDS))
    if unknown:
        raise PlanRefused(f"serve.reserve takes local and fixed — the operator's own machines — not {unknown}")
    for kind, workers in raw.items():
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_RESERVE:
            raise PlanRefused(f"serve.reserve.{kind} is a whole number of workers from 1 to {MAX_RESERVE}")
    return dict(raw)


def _defaults(raw: Any, owned_steady: bool = False) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise PlanRefused("serve.defaults is an object: {\"priority\": ..., \"keep_on\": ...}")
    if "substitute" in raw:
        raise PlanRefused("serve.defaults.substitute: a substitute model arrives in a later version of the pool")
    unknown = sorted(set(raw) - set(OPTIONS))
    if unknown:
        raise PlanRefused(f"serve.defaults takes priority and keep_on, not {unknown}")
    for option, value in raw.items():
        if value not in OPTIONS[option]:
            raise PlanRefused(f"serve.defaults.{option} is {_listed(OPTIONS[option])}")
    if raw.get("keep_on") == "owned" and not owned_steady:
        # Steady on rented machines: `owned` by default would leave a request that sends nothing
        # no source once the workload serves.
        raise PlanRefused("serve.defaults.keep_on owned keeps every request off the workload's own rented machines, "
                          "where it runs; it needs steady local or fixed (workers the operator reserves). A request "
                          "may still send it")
    return dict(raw)


def _listed(values: tuple[str, ...]) -> str:
    return ", ".join(values[:-1]) + " or " + values[-1]


# --- a request's options (§14.4, D201) ---


def request_options(plan: Optional[ServingPlan], priority: Optional[str], keep_on: Optional[str]) -> tuple[str, str]:
    """The options a request travels with: what it sent, else its plan's default, else the first
    of each list. A value not on its list is refused (ValueError, the router's `unknown_option`):
    a misspelt option never passes as none."""
    defaults = plan.defaults if plan is not None else {}
    chosen = []
    for option, sent in (("priority", priority), ("keep_on", keep_on)):
        if sent is not None:
            sent = sent.strip().lower()
            if sent not in OPTIONS[option]:
                raise ValueError(f"{OPTION_HEADERS[option]} is {_listed(OPTIONS[option])}")
            chosen.append(sent)
        else:
            chosen.append(defaults.get(option, DEFAULT_OPTIONS[option]))
    return chosen[0], chosen[1]


def kept_off(priority: str, keep_on: str) -> frozenset[str]:
    """What the options keep a request off, of `hosted` and `rented`: `batch` and `no_hosted`
    every hosted API; `owned` those and every rented machine, the workload's own included."""
    off = set()
    if priority == "batch" or keep_on in ("no_hosted", "owned"):
        off.add("hosted")
    if keep_on == "owned":
        off.add("rented")
    return frozenset(off)


def why_kept_off(plan: Optional[ServingPlan], priority: Optional[str], keep_on: Optional[str],
                 chosen: tuple[str, str]) -> str:
    """The option that left a request nowhere to go, in words: the header, or the plan's default."""
    parts = []
    # Lifting one option helps only where the other does not keep the request off as well.
    both = chosen[0] == "batch" and chosen[1] in ("no_hosted", "owned")
    for option, sent, value in (("keep_on", keep_on, chosen[1]), ("priority", priority, chosen[0])):
        if value == DEFAULT_OPTIONS[option]:
            continue
        said = f"{OPTION_HEADERS[option]}: {value}"
        if sent is None:
            lift = "" if both else f"; send {DEFAULT_OPTIONS[option]} to lift it"
            said += f" (the plan's default{lift})"
        parts.append(said)
    return "; ".join(parts)


# --- the route an answer took ---


def route_label(plan: ServingPlan, *, hosted: bool, own: bool, burst: bool, host_kind: str, covering: bool,
                host_id: str = "") -> str:
    """`<role>/<source name>` for an answer (§14.7): the workload's own machine is steady, or
    when_full where it was bought for a burst; a borrowed one is the until_ready source it matched;
    a hosted API is until_ready's while the workload is covered and its until_ready has one, else
    when_full's."""
    if hosted:
        # The router labels a hosted answer by the source whose budget held it; this is for one
        # it could not (a host of no source's providers, which the router never reaches).
        for role in (("until_ready",) if covering else ()) + ("when_full", "until_ready"):
            for source in plan.role_hosted(role):
                if source.serves(host_id):
                    return f"{role}/{source.name}"
        return "when_full/hosted"
    if own:
        if burst:
            rented = plan.rented()
            return f"when_full/{rented.name if rented is not None else 'rented'}"
        return f"steady/{plan.steady}"
    for source in plan.until_ready:
        if source.type == "shared" or (source.type in HOST_KINDS and host_kind in HOST_KINDS[source.type]):
            return f"until_ready/{source.name}"
    return "until_ready/shared"


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
    sources = plan.hosted_sources()
    if sources:
        # The wait a reader of `hosted` alone would use: when_full's, else until_ready's.
        waits = [s.after_s for s in plan.role_hosted("when_full")] or [s.after_s for s in sources]
        if len(sources) == 1:
            budget = sources[0].budget
        else:
            # Several: the one hosted total (a month of 31 days, the most it can be), so a reader
            # of `hosted` alone still bounds every hosted dollar; per request, the largest any allows.
            budget = dict(plan.hosted_total(31))
            each = [s.budget.get("per_request") for s in sources]
            if all(v is not None for v in each):
                budget["per_request"] = max(each)
        policy["hosted"] = {"after_s": min(waits), "budget": budget,
                            "reserve_max_tokens": max(s.reserve_max_tokens for s in sources)}
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
            which = f" {', '.join(source.providers)}" if source.providers else ""
            return f"{source.name} (hosted APIs{which}{wait}, within {source.budget})"
        if source.type == "rented":
            kind = f" {source.kind}" if source.kind else ""
            return (f"{source.name} (up to {source.max_hosts} more{kind} rented host(s) once a burst has lasted "
                    f"{source.after_min:g} min, within {source.budget})")
        return {"local": "local (the pool's own computer)", "fixed": "fixed (the operator's machines elsewhere)",
                "shared": "shared (any free machine of the pool)"}[source.type] + (
                    f" as {source.name}" if source.name != source.type else "")

    said = ["steady on machines rented for it" if plan.steady == "rented" else
            f"steady on the operator's {plan.steady} machines, {plan.reserve[plan.steady]} worker(s) reserved"]
    for kind, workers in sorted(plan.reserve.items()):
        if kind != plan.steady:
            said.append(f"{workers} {kind} worker(s) reserved for its stop-gap")
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
    #: The roles whose hosted sources may pay for an answer now, in order: until_ready's while
    #: covering with a fresh supervisor, then when_full's.
    hosted_roles: tuple[str, ...] = ("when_full",)
    #: The hosted hosts those sources may use (their `providers`), or None: any hosted host.
    hosted_ids: Optional[frozenset[str]] = None
    #: Those providers in the order the sources list them.
    hosted_order: tuple[str, ...] = ()
    #: Every hosted source's wait in force, soonest first: the request looks again at each.
    hosted_waits: tuple[float, ...] = ()
    #: The plan as this request sees it under its rules (`narrowed`), which its hosted holds and
    #: route labels read; whether those rules keep it off the workload's rented burst hosts; and
    #: the rules that applied, by name.
    plan: Optional["ServingPlan"] = None
    no_burst: bool = False
    rules: tuple[str, ...] = ()


def route(plan: ServingPlan, *, preparing: bool, model_ready: bool, created_at: float, now: float,
          gap_ends_at: Optional[float], fresh: bool, borrow_share: float, short: bool = True,
          gap_reason: str = "replacement") -> Routing:
    """How a request may go now, by its workload's plan and the facts of the moment:

    - `preparing`: the workload has not yet served (its state, as the router reads it);
    - `model_ready`: one of its own ready hosts holds the request's model;
    - `gap_ends_at`: the end of a gap the supervisor marked for this model, if any, and
      `gap_reason` what it covers — `replacement`, or `scale_up` (hosts rented for load);
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
        gap_reason in plan.during and fresh and short and gap_ends_at is not None and now < gap_ends_at
    )
    covering = bool(plan.until_ready) and (starting or replacing)
    borrows = plan.borrows() if covering else ()
    kinds: Optional[frozenset[str]] = None
    if borrows and "shared" not in borrows:
        kinds = frozenset(k for b in borrows for k in HOST_KINDS[b])
    may_borrow = bool(borrows) and borrow_share > 0

    # The soonest any hosted source this moment allows may take the request; each source's own
    # wait is held to when its budget is (the router's reserve).
    waits: list[float] = []
    if covering and fresh:
        waits += [s.after_s or 0.0 for s in plan.role_hosted("until_ready")]
    waits += [s.after_s if s.after_s is not None else 5.0 for s in plan.role_hosted("when_full")]
    order = plan.order() if covering else ()
    if not fresh:
        order = tuple(t for t in order if t != "hosted")
    roles = (("until_ready",) if covering and fresh else ()) + ("when_full",)
    usable = [s for role in roles for s in plan.role_hosted(role)]
    ids: Optional[frozenset[str]] = None
    if usable and all(s.providers for s in usable):
        ids = frozenset(p for s in usable for p in s.providers)
    return Routing(
        covering=covering, may_borrow=may_borrow, borrow_kinds=kinds,
        borrow_while_ready=replacing and may_borrow, hosted_after_s=min(waits) if waits else None, order=order,
        hosted_roles=roles, hosted_ids=ids,
        hosted_order=tuple(dict.fromkeys(p for s in usable for p in s.providers)),
        hosted_waits=tuple(sorted(set(waits))),
        plan=plan,
    )
