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
    """Dollars held for one request until its answer says what it cost."""

    workload: Optional[str]
    amount: float
    day: str
    month: str
    settled: bool = False


class HostedBudgets:
    """What each workload has spent on hosted APIs, and holds for requests in flight.

    `None` is the shared workload. A workload with no budget never reaches a hosted API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spent: dict[tuple[Optional[str], str], float] = {}
        self._held: dict[tuple[Optional[str], str], float] = {}

    # --- from the request log, once ---

    def load(self, rows: list[Any]) -> None:
        """Rows of (workload, ts, cost_usd) for this month's hosted answers."""
        with self._lock:
            self._spent.clear()
            for row in rows:
                cost = row["cost_usd"]
                if not cost:
                    continue
                day, month = day_and_month(row["ts"])
                for period in (day, month):
                    key = (row["workload"], period)
                    self._spent[key] = self._spent.get(key, 0.0) + float(cost)

    # --- the request path ---

    def _used(self, workload: Optional[str], period: str) -> float:
        return self._spent.get((workload, period), 0.0) + self._held.get((workload, period), 0.0)

    def hold(self, workload: Optional[str], budget: Optional[Budget], amount: float,
             now: Optional[float] = None) -> Optional[Hold]:
        """Hold `amount` against every limit of `budget`, or None where any would be passed."""
        if budget is None:
            return None
        day, month = day_and_month(now)
        with self._lock:
            if budget.per_request is not None and amount > budget.per_request:
                return None
            if budget.per_day is not None and self._used(workload, day) + amount > budget.per_day:
                return None
            if budget.per_month is not None and self._used(workload, month) + amount > budget.per_month:
                return None
            for period in (day, month):
                key = (workload, period)
                self._held[key] = self._held.get(key, 0.0) + amount
        return Hold(workload, amount, day, month)

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
                key = (hold.workload, period)
                self._held[key] = max(0.0, self._held.get(key, 0.0) - hold.amount)
                self._spent[key] = self._spent.get(key, 0.0) + charged
        return charged

    def left(self, workload: Optional[str], budget: Optional[Budget], now: Optional[float] = None) -> Optional[dict[str, Any]]:
        """What the budget allows and what is left of it, for `GET /pool/status`."""
        if budget is None:
            return None
        day, month = day_and_month(now)
        with self._lock:
            spent_day = self._spent.get((workload, day), 0.0)
            spent_month = self._spent.get((workload, month), 0.0)
        answer: dict[str, Any] = {"spent_today_usd": round(spent_day, 6), "spent_this_month_usd": round(spent_month, 6)}
        if budget.per_request is not None:
            answer["per_request_usd"] = budget.per_request
        if budget.per_day is not None:
            answer["per_day_usd"] = budget.per_day
            answer["left_today_usd"] = round(max(0.0, budget.per_day - spent_day), 6)
        if budget.per_month is not None:
            answer["per_month_usd"] = budget.per_month
            answer["left_this_month_usd"] = round(max(0.0, budget.per_month - spent_month), 6)
        return answer
