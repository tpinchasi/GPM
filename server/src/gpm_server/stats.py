"""Stats kept, and what the raw logs may forget (D194, D195).

The request log is one row per request and the spend ledger one row per rented host per pass:
fine for the questions the pool asks of the last minutes and hours, too much to keep for ever,
and slow to sum over a month. So the **supervisor** — never the router, whose request path
must stay free of anything slow — folds them, a batch at a time, into two hourly tables:

- `stats_requests_hourly`: per UTC hour × workload × named app (D193) × model × host × provider ×
  kind — requests and how they ended, tokens in and out, latency as a sum and a **fixed-bucket
  histogram** (so a p50 or p95 can be read over any range, merged bucket by bucket), queue wait,
  worker time, hosted cost.
- `stats_rented_hourly`: per UTC hour × host × lease × provider connection — what rented machines
  and a workload's volumes cost. The ledger holds **running totals** per host and source, not
  increments: each hour gets what the totals rose by in it.

Each table is followed by a **watermark** (the last row folded, in `pool_meta`), moved in the
same transaction as the rows it produced: a fold run twice, or by two callers at once, counts
each row once.

**Retention** (D195): the request log, the decision log and superseded spend rows are pruned after
`stats.keep_requests_days`, the hourly tables after `stats.keep_rollups_days`. Never pruned: a row
not yet folded; a request row that carries a hosted cost and is younger than any budget looks back
(32 days: a UTC month, a provisioner's rolling 30); the newest rows sizing (D156) and the machine
record (D69) read; the machine record's events, those renting restores from, and any of a lease
still open; and, per lease, host and source of the spend ledger, the newest figure, the newest
before the start of today and of this month (UTC), and the newest already folded — every figure
lease and budget enforcement reads. What stays is worked out once per prune by one grouped read per
table; candidates are then read a batch at a time behind an id cursor and deleted by id.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Iterable, Optional

from .db import Database
from .history import MACHINE_EVENT_KINDS

log = logging.getLogger("gpm.stats")

HOUR_S = 3600
DAY_S = 86400
#: The longest any budget looks back: a UTC month (D165) or a provisioner's rolling 30 days
#: (D117), with a day to spare. Request rows carrying a hosted cost are kept at least this long.
BUDGET_LOOKBACK_S = 32 * DAY_S

#: Upper edges of the latency histogram's buckets, in milliseconds; one more bucket holds the rest.
#: Fixed, so histograms from any hours, hosts or apps add up bucket by bucket.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    25, 50, 100, 200, 300, 500, 750, 1_000, 1_500, 2_000, 3_000, 5_000, 7_500, 10_000, 15_000,
    20_000, 30_000, 45_000, 60_000, 90_000, 120_000, 180_000, 300_000, 600_000,
)

#: How many of its newest rows each of the pool's own measures reads from the request log — a
#: build's latency curve at a card (D156) and the machine record (D69). The prune spares as many,
#: so retention never changes what they read; the readers take their limit from here.
MEASURED_ROWS = 20_000

#: Events never pruned: the machine record's (D69) and those renting reads back — a lease's
#: replacements for value (D75) and the machines it avoids, with when each may be tried again. An
#: event of a lease still open is kept too, whatever its kind.
KEPT_EVENT_KINDS = frozenset({*MACHINE_EVENT_KINDS, "replaced_for_value", "machine_avoided"})

_REQUESTS_MARK = "stats.request_log_folded_to"
_SPEND_MARK = "stats.spend_folded_to"

#: What one hourly row counts; every one of them adds.
_COUNTS = ("requests", "ok", "rejected", "failed", "cancelled", "status_4xx", "status_5xx", "out_of_funds",
           "borrowed", "tokens_in", "tokens_out", "latency_n", "latency_sum_ms", "busy_ms", "queue_n",
           "queue_sum_ms", "cost_usd")
_REQUEST_KEY = ("hour", "workload", "app", "model", "host_id", "host_kind", "provider", "kind")


def bucket_of(latency_ms: float) -> int:
    for i, edge in enumerate(LATENCY_BUCKETS_MS):
        if latency_ms <= edge:
            return i
    return len(LATENCY_BUCKETS_MS)


def empty_histogram() -> list[int]:
    return [0] * (len(LATENCY_BUCKETS_MS) + 1)


def merge_histograms(into: list[int], other: Iterable[int]) -> list[int]:
    for i, n in enumerate(other):
        if i < len(into):
            into[i] += int(n)
    return into


def percentile(histogram: list[int], q: float, max_ms: Optional[float] = None) -> Optional[float]:
    """The `q` quantile (0–1) read from a histogram, interpolated inside its bucket — so within
    that bucket's width of the true value, and never above the largest latency seen."""
    total = sum(histogram)
    if total <= 0:
        return None
    target = q * total
    seen = 0
    for i, n in enumerate(histogram):
        if n <= 0:
            continue
        if seen + n >= target:
            lower = LATENCY_BUCKETS_MS[i - 1] if i > 0 else 0.0
            upper = LATENCY_BUCKETS_MS[i] if i < len(LATENCY_BUCKETS_MS) else (max_ms or lower)
            if max_ms is not None:
                upper = min(upper, max(max_ms, lower))
            value = lower + (upper - lower) * max(0.0, target - seen) / n
            return round(value, 1)
        seen += n
    return max_ms


def _hour(ts: float) -> int:
    return int(ts // HOUR_S) * HOUR_S


class Stats:
    """The folding and the pruning, run by the supervisor between passes and before a stats read.

    `config` gives the pool's configuration now.
    """

    def __init__(self, database: Database, config: Callable[[], Any]):
        self.db = database
        self._config = config
        #: One fold at a time in this process; the watermark keeps two processes honest.
        self._lock = threading.Lock()
        #: One prune at a time. Never held with the fold's lock: a prune deletes only rows already
        #: folded, so neither needs the other to wait.
        self._prune_lock = threading.Lock()
        self.last_pruned_at: Optional[float] = None
        #: When a fold last began that left nothing behind it: everything logged before then is
        #: in the hourly tables.
        self.caught_up_at: Optional[float] = None

    def _legacy(self) -> str:
        """The connection a ledger row naming none belongs to (D129), where the pool has rented."""
        rows = self.db.query("SELECT value FROM pool_meta WHERE key = 'legacy_connection'")
        return str(rows[0]["value"]) if rows else ""

    # --- watermarks ---

    def _mark(self, name: str) -> int:
        rows = self.db.query("SELECT value FROM pool_meta WHERE key = ?", (name,))
        return int(rows[0]["value"]) if rows else 0

    @staticmethod
    def _mark_in(conn: Any, name: str) -> int:
        row = conn.execute("SELECT value FROM pool_meta WHERE key = ?", (name,)).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _set_mark(conn: Any, name: str, value: int) -> None:
        conn.execute("INSERT INTO pool_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (name, str(value)))

    # --- folding ---

    def roll_up(self, batch: int = 5000, batches: int = 20) -> dict[str, int]:
        """Fold what is new in the request log and the spend ledger: at most `batches` batches of
        each, so a first run over a long log is spread across passes. How many rows each took."""
        with self._lock:
            started = time.time()
            folded = {"requests": 0, "spend": 0}
            caught_up = True
            for name, fold in (("requests", self._fold_requests), ("spend", self._fold_spend)):
                for _ in range(batches):
                    n = fold(batch)
                    folded[name] += n
                    if n < batch:
                        break
                else:
                    caught_up = False
            if caught_up:
                self.caught_up_at = started
            return folded

    def _host_facts(self, host_ids: set[str]) -> dict[str, tuple[str, str]]:
        """Each host's kind — `owned`, `rented`, `hosted` — and provider: a hosted API is its own
        provider; a rented host's is the connection its spend went to (D129)."""
        config = self._config()
        configured = {h.id: h.kind for h in getattr(config, "hosts", [])}
        legacy = self._legacy() or ""
        facts: dict[str, tuple[str, str]] = {"": ("", "")}
        rest = []
        for host_id in host_ids:
            if not host_id:
                continue
            kind = configured.get(host_id)
            if kind == "hosted":
                facts[host_id] = ("hosted", host_id)
            elif kind is not None:
                facts[host_id] = ("owned", "")
            else:
                rest.append(host_id)
        if rest:
            marks = ",".join("?" * len(rest))
            # SQLite answers the bare column from the row MAX() chose: the newest connection.
            for row in self.db.query(
                    f"SELECT host_id, connection, MAX(id) AS newest FROM spend WHERE host_id IN ({marks}) GROUP BY host_id",
                    tuple(rest)):
                facts[row["host_id"]] = ("rented", row["connection"] or legacy)
            for row in self.db.query(f"SELECT host_id, kind FROM hosts WHERE host_id IN ({marks})", tuple(rest)):
                if row["host_id"] in facts:
                    continue
                kind = str(row["kind"] or "")
                facts[row["host_id"]] = (("rented", legacy) if kind.startswith("rented") else
                                         ("hosted", row["host_id"]) if kind == "hosted" else ("owned", ""))
        return facts

    def _fold_requests(self, batch: int) -> int:
        after = self._mark(_REQUESTS_MARK)
        rows = self.db.query(
            "SELECT id, ts, workload, app, model_requested, host_id, kind, outcome, status_code, reason, borrowed, "
            "tokens_in, tokens_out, latency_ms, queue_wait_ms, cost_usd FROM request_log WHERE id > ? ORDER BY id LIMIT ?",
            (after, batch))
        if not rows:
            return 0
        facts = self._host_facts({r["host_id"] or "" for r in rows})
        folded: dict[tuple, dict[str, Any]] = {}
        for r in rows:
            host_kind, provider = facts.get(r["host_id"] or "", ("", ""))
            key = (_hour(float(r["ts"])), r["workload"] or "", r["app"] or "", r["model_requested"] or "",
                   r["host_id"] or "", host_kind, provider, r["kind"] or "")
            row = folded.get(key)
            if row is None:
                row = folded[key] = {**{name: 0 for name in _COUNTS}, "latency_max_ms": 0.0, "latency_hist": empty_histogram()}
            if r["reason"] == "provider_out_of_funds":
                row["out_of_funds"] += 1
            outcome = r["outcome"]
            if outcome == "redispatched":
                continue  # an attempt sent on elsewhere: its request is counted where it ended
            status = r["status_code"]
            row["requests"] += 1
            if outcome == "ok" and (status is None or status < 400):
                row["ok"] += 1
            elif outcome == "rejected":
                row["rejected"] += 1
            elif outcome == "cancelled":
                row["cancelled"] += 1
            else:
                row["failed"] += 1
            if status is not None and 400 <= status < 500:
                row["status_4xx"] += 1
            elif status is not None and status >= 500:
                row["status_5xx"] += 1
            row["borrowed"] += 1 if r["borrowed"] else 0
            row["tokens_in"] += int(r["tokens_in"] or 0)
            row["tokens_out"] += int(r["tokens_out"] or 0)
            latency = r["latency_ms"]
            if latency is not None:
                row["busy_ms"] += float(latency)
                if outcome == "ok" and (status is None or status < 400):
                    row["latency_n"] += 1
                    row["latency_sum_ms"] += float(latency)
                    row["latency_max_ms"] = max(row["latency_max_ms"], float(latency))
                    row["latency_hist"][bucket_of(float(latency))] += 1
            if r["queue_wait_ms"] is not None:
                row["queue_n"] += 1
                row["queue_sum_ms"] += float(r["queue_wait_ms"])
            row["cost_usd"] += float(r["cost_usd"] or 0.0)
        last = int(rows[-1]["id"])
        with self.db.transaction() as conn:
            if self._mark_in(conn, _REQUESTS_MARK) != after:
                return 0  # another caller folded these first
            for key, row in folded.items():
                where = " AND ".join(f"{name} = ?" for name in _REQUEST_KEY)
                existing = conn.execute(
                    f"SELECT {', '.join(_COUNTS)}, latency_max_ms, latency_hist FROM stats_requests_hourly WHERE {where}",
                    key).fetchone()
                if existing is not None:
                    for i, name in enumerate(_COUNTS):
                        row[name] += existing[i]
                    row["latency_max_ms"] = max(row["latency_max_ms"], existing[len(_COUNTS)])
                    merge_histograms(row["latency_hist"], json.loads(existing[len(_COUNTS) + 1]))
                columns = (*_REQUEST_KEY, *_COUNTS, "latency_max_ms", "latency_hist")
                conn.execute(
                    f"INSERT OR REPLACE INTO stats_requests_hourly ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                    (*key, *(row[name] for name in _COUNTS), row["latency_max_ms"], json.dumps(row["latency_hist"])))
            self._set_mark(conn, _REQUESTS_MARK, last)
        return len(rows)

    def _fold_spend(self, batch: int) -> int:
        after = self._mark(_SPEND_MARK)
        rows = self.db.query(
            "SELECT id, ts, lease_id, host_id, source, amount, connection FROM spend WHERE id > ? ORDER BY id LIMIT ?",
            (after, batch))
        if not rows:
            return 0
        legacy = self._legacy() or ""
        # The running total each host and source had reached before this batch.
        before: dict[tuple, float] = {}
        for lease_id, host_id, source in {(r["lease_id"], r["host_id"], r["source"]) for r in rows}:
            found = self.db.query(
                "SELECT amount FROM spend WHERE lease_id IS ? AND host_id IS ? AND source = ? AND id <= ? "
                "ORDER BY id DESC LIMIT 1", (lease_id, host_id, source, after))
            before[(lease_id, host_id, source)] = float(found[0]["amount"]) if found else 0.0
        leases = {r["lease_id"] for r in rows if r["lease_id"]}
        workload_of: dict[str, str] = {}
        if leases:
            marks = ",".join("?" * len(leases))
            workload_of = {r["lease_id"]: r["workload"] or "" for r in self.db.query(
                f"SELECT lease_id, workload FROM leases WHERE lease_id IN ({marks})", tuple(leases))}
        folded: dict[tuple, list] = {}
        for r in rows:
            key = (r["lease_id"], r["host_id"], r["source"])
            amount = float(r["amount"] or 0.0)
            # A total that went down was re-based (a provider's figure read again from a new
            # base): nothing is taken back, and the next rise is counted from here.
            rose = max(0.0, amount - before.get(key, 0.0))
            before[key] = amount
            hour_key = (_hour(float(r["ts"])), r["host_id"] or "", r["lease_id"] or "", r["connection"] or legacy)
            entry = folded.setdefault(hour_key, [workload_of.get(r["lease_id"] or "", ""), 0.0, 0.0])
            entry[1 if r["source"] == "estimate" else 2] += rose
        last = int(rows[-1]["id"])
        with self.db.transaction() as conn:
            if self._mark_in(conn, _SPEND_MARK) != after:
                return 0
            for (hour, host_id, lease_id, connection), (workload, estimate, reported) in folded.items():
                conn.execute(
                    "INSERT INTO stats_rented_hourly (hour, host_id, lease_id, connection, workload, estimate_usd, reported_usd) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(hour, host_id, lease_id, connection) DO UPDATE SET "
                    "estimate_usd = estimate_usd + excluded.estimate_usd, reported_usd = reported_usd + excluded.reported_usd",
                    (hour, host_id, lease_id, connection, workload, estimate, reported))
            self._set_mark(conn, _SPEND_MARK, last)
        return len(rows)

    # --- forgetting ---

    def prune(self, now: Optional[float] = None, batch: int = 2000, batches: int = 250) -> dict[str, int]:
        """Delete what retention lets go. What must stay is worked out once per prune — a grouped
        read per table, never a lookup per row — then candidates are read a batch at a time, in id
        order behind a cursor, and deleted by id: the file's write lock is held only for each
        delete, and folding and stats reads go on beside it. How many rows went from each table."""
        from .router.budgets import period_starts

        now = time.time() if now is None else now
        settings = self._config().stats
        removed = {"request_log": 0, "events": 0, "spend": 0, "stats": 0}
        with self._prune_lock:  # one prune at a time; never the fold's lock
            if settings.keep_requests_days is not None:
                cutoff = now - settings.keep_requests_days * DAY_S
                folded = self._mark(_REQUESTS_MARK)
                removed["request_log"] = self._delete_behind_cursor(
                    "request_log",
                    "SELECT id FROM request_log WHERE id > ? AND id <= ? AND ts < ? AND (cost_usd IS NULL OR ts < ?) "
                    "ORDER BY id LIMIT ?",
                    (folded, cutoff, min(cutoff, now - BUDGET_LOOKBACK_S)), self._requests_kept(), batch, batches)
                kinds = sorted(KEPT_EVENT_KINDS)
                removed["events"] = self._delete_behind_cursor(
                    "events",
                    f"SELECT id FROM events WHERE id > ? AND ts < ? AND kind NOT IN ({','.join('?' * len(kinds))}) "
                    "AND (lease_id IS NULL OR lease_id NOT IN (SELECT lease_id FROM leases WHERE state = 'open')) "
                    "ORDER BY id LIMIT ?",
                    (cutoff, *kinds), set(), batch, batches)
                day, month = period_starts(now)
                spend_folded = self._mark(_SPEND_MARK)
                kept: set[int] = set()
                # Per lease, host and source: the newest figure, the newest before today and before
                # this month, and the newest folded — every figure enforcement and the fold read.
                for row in self.db.query(
                        "SELECT MAX(id) AS newest, MAX(CASE WHEN ts < ? THEN id END) AS before_day, "
                        "MAX(CASE WHEN ts < ? THEN id END) AS before_month, MAX(CASE WHEN id <= ? THEN id END) AS folded "
                        "FROM spend GROUP BY lease_id, host_id, source", (day, month, spend_folded)):
                    kept.update(int(v) for v in tuple(row) if v is not None)
                removed["spend"] = self._delete_behind_cursor(
                    "spend", "SELECT id FROM spend WHERE id > ? AND id <= ? AND ts < ? ORDER BY id LIMIT ?",
                    (spend_folded, cutoff), kept, batch, batches)
            if settings.keep_rollups_days is not None:
                before = _hour(now - settings.keep_rollups_days * DAY_S)
                for table in ("stats_requests_hourly", "stats_rented_hourly"):
                    removed["stats"] += self._delete_behind_cursor(
                        table, f"SELECT rowid AS id FROM {table} WHERE rowid > ? AND hour < ? ORDER BY rowid LIMIT ?",
                        (before,), set(), batch, batches, column="rowid")
        self.last_pruned_at = now
        return removed

    def _requests_kept(self) -> set[int]:
        """Request rows the pool's own measures read whatever their age, chosen as those readers
        choose them: per build, the newest answers a latency curve is made of (D156's sizing at a
        card's latency), and the newest chat answers the machine record folds (D69)."""
        kept = {int(r["id"]) for r in self.db.query(
            "SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY model_served ORDER BY id DESC) AS newest "
            "FROM request_log WHERE outcome = 'ok' AND model_served IS NOT NULL AND concurrency IS NOT NULL "
            "AND latency_ms IS NOT NULL AND host_id IS NOT NULL "
            "AND (model_concurrency IS NULL OR model_concurrency = concurrency)) WHERE newest <= ?",
            (MEASURED_ROWS,))}
        kept.update(int(r["id"]) for r in self.db.query(
            "SELECT id FROM request_log WHERE host_id IS NOT NULL AND (kind IS NULL OR kind = 'chat') "
            "ORDER BY id DESC LIMIT ?", (MEASURED_ROWS,)))
        return kept

    def _delete_behind_cursor(self, table: str, find: str, params: tuple, kept: set[int], batch: int, batches: int,
                              column: str = "id") -> int:
        """Read candidates `batch` at a time after the last one read — so a row kept is read once,
        never again — and delete those not in `kept`. `find` takes the cursor first, the batch last."""
        removed, cursor = 0, 0
        for _ in range(batches):
            ids = [int(r["id"]) for r in self.db.query(find, (cursor, *params, batch))]
            if not ids:
                break
            cursor = ids[-1]
            gone = [i for i in ids if i not in kept]
            if gone:
                self.db.execute(f"DELETE FROM {table} WHERE {column} IN ({','.join('?' * len(gone))})", tuple(gone))
                removed += len(gone)
            if len(ids) < batch:
                break
        return removed

    def run_once(self, now: Optional[float] = None, prune_every_s: float = HOUR_S) -> dict[str, Any]:
        """What the supervisor runs between passes: fold, then prune when an hour has gone by."""
        result: dict[str, Any] = {"folded": self.roll_up()}
        now = time.time() if now is None else now
        if self.last_pruned_at is None or now - self.last_pruned_at >= prune_every_s:
            result["pruned"] = self.prune(now)
        return result

    # --- reading ---

    def folded_to(self) -> Optional[float]:
        """Everything logged before this moment is counted: when the last fold that caught up began."""
        return self.caught_up_at

    def query(self, start: float, end: float, bucket: str = "hour", by: Optional[str] = None,
              filters: Optional[dict[str, Optional[str]]] = None) -> dict[str, Any]:
        """Series and totals over `[start, end)`, per `bucket` (`hour` or `day`, UTC), per `by`
        (`app`, `model`, `host`, `provider`, `kind`, or None for the pool as a whole)."""
        step = HOUR_S if bucket == "hour" else DAY_S
        first = int(start // step) * step
        buckets = list(range(first, int(end), step)) or [first]
        filters = {k: v for k, v in (filters or {}).items() if v is not None}
        # `workload=shared` is the shared workload, kept as '' in the table.
        stored = {**filters, **({"workload": ""} if filters.get("workload") == "shared" else {})}
        where, params = ["hour >= ?", "hour < ?"], [first, int(end)]
        for name in ("workload", "app", "model", "kind", "provider"):
            if name in stored:
                where.append(f"{name} = ?")
                params.append(stored[name])
        if "host" in filters:
            where.append("host_id = ?")
            params.append(filters["host"])
        request_rows = self.db.query(f"SELECT * FROM stats_requests_hourly WHERE {' AND '.join(where)}", tuple(params))

        # Rented cost is per host: it follows a filter on the workload, host or provider, and no
        # other — a machine's hour is not any one model's, kind's or named app's.
        rented_rows: list[Any] = []
        rented_applies = not any(name in filters for name in ("app", "model", "kind"))
        if rented_applies:
            rwhere, rparams = ["hour >= ?", "hour < ?"], [first, int(end)]
            if "workload" in stored:
                rwhere.append("workload = ?")
                rparams.append(stored["workload"])
            if "host" in filters:
                rwhere.append("host_id = ?")
                rparams.append(filters["host"])
            if "provider" in filters:
                rwhere.append("connection = ?")
                rparams.append(filters["provider"])
            rented_rows = self.db.query(f"SELECT * FROM stats_rented_hourly WHERE {' AND '.join(rwhere)}", tuple(rparams))

        def series_of_request(r: Any) -> tuple[tuple, dict[str, Any]]:
            if by == "app":
                workload, app = r["workload"], r["app"]
                kind = "workload" if workload else ("app" if app else "shared")
                label = workload or app or "shared pool (configured app key)"
                return ("app", workload, app), {"key": {"workload": workload or None, "app": app or None},
                                                "label": label, "type": kind}
            if by == "model":
                return ("model", r["model"]), {"key": {"model": r["model"] or None}, "label": r["model"] or "(none named)"}
            if by == "host":
                host = r["host_id"]
                return ("host", host), {"key": {"host": host or None, "host_kind": r["host_kind"] or None,
                                                "provider": r["provider"] or None},
                                        "label": host or "(no host: refused before one was chosen)", "type": r["host_kind"] or None}
            if by == "provider":
                return ("provider", r["host_kind"], r["provider"]), _provider_series(r["host_kind"], r["provider"])
            if by == "kind":
                return ("kind", r["kind"]), {"key": {"kind": r["kind"] or None}, "label": r["kind"] or "(unknown)"}
            return ("all",), {"key": {}, "label": "the whole pool"}

        def series_of_rented(r: Any) -> Optional[tuple[tuple, dict[str, Any]]]:
            if by == "app":
                if r["workload"]:
                    return ("app", r["workload"], ""), {"key": {"workload": r["workload"], "app": None},
                                                        "label": r["workload"], "type": "workload"}
                return ("app-shared-hosts",), {"key": {"shared_hosts": True},
                                               "label": "shared pool's rented hosts (every shared-pool app)",
                                               "type": "shared hosts"}
            if by == "host":
                return ("host", r["host_id"]), {"key": {"host": r["host_id"], "host_kind": "rented",
                                                        "provider": r["connection"] or None},
                                                "label": r["host_id"], "type": "rented"}
            if by == "provider":
                return ("provider", "rented", r["connection"]), _provider_series("rented", r["connection"])
            if by in ("model", "kind"):
                return None  # not split by model or kind: in the totals only
            return ("all",), {"key": {}, "label": "the whole pool"}

        series: dict[tuple, dict[str, Any]] = {}
        index = {t: i for i, t in enumerate(buckets)}

        def slot(key: tuple, about: dict[str, Any]) -> dict[str, Any]:
            found = series.get(key)
            if found is None:
                found = series[key] = {**about, "acc": [_Accumulator() for _ in buckets]}
            return found

        overall = [_Accumulator() for _ in buckets]
        for r in request_rows:
            i = index.get(int(r["hour"] // step) * step)
            if i is None:
                continue
            key, about = series_of_request(r)
            slot(key, about)["acc"][i].add_requests(r)
            overall[i].add_requests(r)
        for r in rented_rows:
            i = index.get(int(r["hour"] // step) * step)
            if i is None:
                continue
            overall[i].add_rented(r)
            found = series_of_rented(r)
            if found is not None:
                slot(*found)["acc"][i].add_rented(r)

        out = []
        for about in series.values():
            accumulators = about.pop("acc")
            total = _Accumulator.merged(accumulators)
            out.append({**about, "points": [{"t": t, **a.view()} for t, a in zip(buckets, accumulators, strict=True)],
                        "totals": total.view()})
        out.sort(key=lambda s: (-(s["totals"]["requests"]), -(s["totals"]["cost_usd"]), s["label"]))
        notes = []
        if not rented_applies:
            notes.append("rented machines' cost is per host: it is left out where the view is narrowed to an app, "
                         "model or kind")
        elif by in ("model", "kind"):
            notes.append("rented machines' cost is per host, not per model or kind: it is in the totals only")
        if by == "app":
            notes.append("a named app shares the shared pool's rented hosts with every other shared-pool app: their "
                         "cost is its own line")
        return {
            "from": start, "to": end, "bucket": bucket, "by": by, "filters": filters,
            "buckets": buckets,
            "series": out,
            "totals": _Accumulator.merged(overall).view(),
            "points": [{"t": t, **a.view()} for t, a in zip(buckets, overall, strict=True)],
            "notes": notes,
            "folded_to": self.folded_to(),
        }


def _provider_series(host_kind: str, provider: str) -> dict[str, Any]:
    if host_kind == "owned":
        label = "own machines"
    elif host_kind == "hosted":
        label = f"{provider} (hosted API)"
    elif host_kind == "rented":
        label = f"{provider or 'rented'} (rented)"
    else:
        label = "(no host)"
    return {"key": {"host_kind": host_kind or None, "provider": provider or None}, "label": label,
            "type": host_kind or None}


class _Accumulator:
    """One series' figures in one bucket, added from hourly rows."""

    def __init__(self) -> None:
        self.counts = {name: 0.0 for name in _COUNTS}
        self.latency_max = 0.0
        self.histogram = empty_histogram()
        #: Rented cost by host in this bucket: (estimate, reported), each host's the higher.
        self.rented: dict[str, list[float]] = {}

    def add_requests(self, row: Any) -> None:
        for name in _COUNTS:
            self.counts[name] += row[name] or 0
        self.latency_max = max(self.latency_max, float(row["latency_max_ms"] or 0.0))
        merge_histograms(self.histogram, json.loads(row["latency_hist"] or "[]"))

    def add_rented(self, row: Any) -> None:
        figures = self.rented.setdefault(row["host_id"], [0.0, 0.0])
        figures[0] += float(row["estimate_usd"] or 0.0)
        figures[1] += float(row["reported_usd"] or 0.0)

    @classmethod
    def merged(cls, parts: list["_Accumulator"]) -> "_Accumulator":
        total = cls()
        for part in parts:
            for name in _COUNTS:
                total.counts[name] += part.counts[name]
            total.latency_max = max(total.latency_max, part.latency_max)
            merge_histograms(total.histogram, part.histogram)
            for host, (estimate, reported) in part.rented.items():
                figures = total.rented.setdefault(host, [0.0, 0.0])
                figures[0] += estimate
                figures[1] += reported
        return total

    def view(self) -> dict[str, Any]:
        c = self.counts
        # Enforcement counts whichever of the pool's estimate and the provider's figure is higher
        # (supervisor.md §4); so does this.
        rented = sum(max(estimate, reported) for estimate, reported in self.rented.values())
        n = int(c["latency_n"])
        return {
            "requests": int(c["requests"]), "ok": int(c["ok"]),
            "errors": int(c["rejected"] + c["failed"]), "rejected": int(c["rejected"]), "failed": int(c["failed"]),
            "cancelled": int(c["cancelled"]), "status_4xx": int(c["status_4xx"]), "status_5xx": int(c["status_5xx"]),
            "out_of_funds": int(c["out_of_funds"]), "borrowed": int(c["borrowed"]),
            "tokens_in": int(c["tokens_in"]), "tokens_out": int(c["tokens_out"]),
            "p50_ms": percentile(self.histogram, 0.50, self.latency_max or None),
            "p95_ms": percentile(self.histogram, 0.95, self.latency_max or None),
            "mean_ms": round(c["latency_sum_ms"] / n, 1) if n else None,
            "max_ms": round(self.latency_max, 1) if n else None,
            "queue_mean_ms": round(c["queue_sum_ms"] / c["queue_n"], 1) if c["queue_n"] else None,
            "busy_s": round(c["busy_ms"] / 1000, 3),
            "hosted_usd": round(c["cost_usd"], 6),
            "rented_usd": round(rented, 6),
            "cost_usd": round(c["cost_usd"] + rented, 6),
        }
