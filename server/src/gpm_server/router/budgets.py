"""Hosted APIs' budgets, held by the router before it spends (S11, D165, D166).

A request may go to a hosted API only if its **worst case** — its prompt's size and the most it
may generate, at the build's prices — fits every budget it counts towards: per request, what is
left today, what is left this month (UTC days and months). That amount is held while the request
runs, so requests in flight together can never take a budget past its limit; when the answer
ends the hold is replaced by what it really cost, from the answer's own token counts.

What was spent is written to the request log with each answer, and read back from it when the
router starts, so a restart forgets nothing it spent. Nothing here does I/O on the request path:
the log is read once, at start, off the event loop.
"""

from __future__ import annotations

import dataclasses
import datetime
import threading
from typing import Any, Optional

from ..config import Budget


def day_and_month(when: Optional[float] = None) -> tuple[str, str]:
    """The UTC day and month a moment falls in — the periods a budget is counted in."""
    moment = datetime.datetime.fromtimestamp(when, datetime.timezone.utc) if when is not None else \
        datetime.datetime.now(datetime.timezone.utc)
    return moment.strftime("%Y-%m-%d"), moment.strftime("%Y-%m")


def period_starts(when: Optional[float] = None) -> tuple[float, float]:
    """The start of the UTC day and of the UTC month, as timestamps."""
    moment = datetime.datetime.fromtimestamp(when, datetime.timezone.utc) if when is not None else \
        datetime.datetime.now(datetime.timezone.utc)
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return day.timestamp(), day.replace(day=1).timestamp()


@dataclasses.dataclass
class Hold:
    """Dollars held for one request until its answer says what it cost — against one budget of
    its workload (`budget_id`; "" the one a workload without a serving plan has) and, always, the
    workload's hosted total. `route` is the role and source that paid (S19), where a plan says."""

    workload: Optional[str]
    amount: float
    day: str
    month: str
    settled: bool = False
    budget_id: str = ""
    route: Optional[str] = None


def days_in_month(when: Optional[float] = None) -> int:
    moment = datetime.datetime.fromtimestamp(when, datetime.timezone.utc) if when is not None else \
        datetime.datetime.now(datetime.timezone.utc)
    following = (moment.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    return (following - datetime.timedelta(days=1)).day


class HostedBudgets:
    """What each workload has spent on hosted APIs, and holds for requests in flight.

    `None` is the shared workload. A workload with no budget never reaches a hosted API. Spending
    is kept per budget (S19, §14.2: a serving plan's sources each have one; `""` is the one budget
    of a workload without a plan, and every row recorded before budgets had ids) and per workload
    in total — the one hosted total a plan's sources share, which counts everything spent under
    any id, so renaming a budget gives nothing back."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spent: dict[tuple[Optional[str], Optional[str], str], float] = {}
        self._held: dict[tuple[Optional[str], Optional[str], str], float] = {}

    #: The key the workload's total is kept under: None, which no budget id can be.
    _TOTAL: Optional[str] = None

    # --- from the request log, once ---

    def load(self, rows: list[Any]) -> None:
        """Rows of (workload, ts, cost_usd[, budget]) for this month's hosted answers."""
        with self._lock:
            self._spent.clear()
            for row in rows:
                cost = row["cost_usd"]
                if not cost:
                    continue
                budget_id = (row["budget"] if "budget" in row.keys() else None) or ""
                day, month = day_and_month(row["ts"])
                for period in (day, month):
                    for key in ((row["workload"], budget_id, period), (row["workload"], self._TOTAL, period)):
                        self._spent[key] = self._spent.get(key, 0.0) + float(cost)

    # --- the request path ---

    def _used(self, workload: Optional[str], budget_id: Optional[str], period: str) -> float:
        key = (workload, budget_id, period)
        return self._spent.get(key, 0.0) + self._held.get(key, 0.0)

    def hold(self, workload: Optional[str], budget: Optional[Budget], amount: float,
             now: Optional[float] = None, *, budget_id: str = "",
             total: Optional[dict[str, float]] = None, route: Optional[str] = None) -> Optional[Hold]:
        """Hold `amount` against every limit of `budget` (kept as `budget_id`) and of the
        workload's hosted `total`, where it has one; or None where any would be passed."""
        if budget is None:
            return None
        day, month = day_and_month(now)
        # A workload's one budget (no plan) bounds everything it spends, as it always did.
        counted = budget_id or self._TOTAL
        with self._lock:
            if budget.per_request is not None and amount > budget.per_request:
                return None
            if budget.per_day is not None and self._used(workload, counted, day) + amount > budget.per_day:
                return None
            if budget.per_month is not None and self._used(workload, counted, month) + amount > budget.per_month:
                return None
            if total is not None:
                for period, limit in ((day, total.get("per_day")), (month, total.get("per_month"))):
                    if limit is not None and self._used(workload, self._TOTAL, period) + amount > limit:
                        return None
            ids = {budget_id, self._TOTAL}
            for period in (day, month):
                for held_id in ids:
                    key = (workload, held_id, period)
                    self._held[key] = self._held.get(key, 0.0) + amount
        return Hold(workload, amount, day, month, budget_id=budget_id, route=route)

    def settle(self, hold: Hold, cost: Optional[float]) -> float:
        """Replace the hold with what the answer cost — the hold itself where it could not be
        read (a stream cut before its usage) — and return what was charged. `cost` of 0: refused
        by the provider, nothing spent."""
        charged = hold.amount if cost is None else max(0.0, cost)
        with self._lock:
            if hold.settled:
                return 0.0
            hold.settled = True
            for period in (hold.day, hold.month):
                for held_id in {hold.budget_id, self._TOTAL}:
                    key = (hold.workload, held_id, period)
                    self._held[key] = max(0.0, self._held.get(key, 0.0) - hold.amount)
                    self._spent[key] = self._spent.get(key, 0.0) + charged
        return charged

    def left(self, workload: Optional[str], budget: Optional[Any], now: Optional[float] = None,
             budget_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """What a budget allows and what is left of it, for `GET /pool/status`: the workload's
        total (`budget_id` None), or one budget's. `budget` has `per_day` / `per_month` (and
        optionally `per_request`), as a Budget or a dict."""
        if budget is None:
            return None
        limits = budget if isinstance(budget, dict) else budget.model_dump(exclude_none=True)
        day, month = day_and_month(now)
        kept = self._TOTAL if budget_id is None else budget_id
        with self._lock:
            spent_day = self._spent.get((workload, kept, day), 0.0)
            spent_month = self._spent.get((workload, kept, month), 0.0)
        answer: dict[str, Any] = {"spent_today_usd": round(spent_day, 6), "spent_this_month_usd": round(spent_month, 6)}
        if limits.get("per_request") is not None:
            answer["per_request_usd"] = limits["per_request"]
        if limits.get("per_day") is not None:
            answer["per_day_usd"] = limits["per_day"]
            answer["left_today_usd"] = round(max(0.0, limits["per_day"] - spent_day), 6)
        if limits.get("per_month") is not None:
            answer["per_month_usd"] = limits["per_month"]
            answer["left_this_month_usd"] = round(max(0.0, limits["per_month"] - spent_month), 6)
        return answer
