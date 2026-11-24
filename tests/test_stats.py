"""Stats kept, and the raw logs pruned (D194, D195).

The supervisor folds the request log and the spend ledger into hourly tables, a batch at a time
behind a watermark, and prunes what retention lets go — never a row lease or budget enforcement
reads. Against a real database file; nothing here needs a host, an engine or a provider.
"""

import json
import time
from types import SimpleNamespace

import pytest
from gpm_server.config import PoolConfig, StatsConfig
from gpm_server.db import Database, RequestLog, RequestRecord
from gpm_server.ledger import EventLog, SpendLedger
from gpm_server.router.budgets import period_starts
from gpm_server.stats import (
    BUDGET_LOOKBACK_S,
    DAY_S,
    HOUR_S,
    LATENCY_BUCKETS_MS,
    Stats,
    bucket_of,
    empty_histogram,
    percentile,
)
from gpm_server.supervisor.renting import Fleet

NOW = time.time()
HOUR = int(NOW // HOUR_S) * HOUR_S


def config(**stats):
    return SimpleNamespace(
        hosts=[SimpleNamespace(id="mac", kind="local"), SimpleNamespace(id="openai", kind="hosted")],
        stats=StatsConfig(**stats),
    )


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


def log(db, **fields):
    defaults = {"request_id": f"r{time.monotonic_ns()}", "outcome": "ok", "status_code": 200, "kind": "chat",
                "model_requested": "m1", "host_id": "mac", "latency_ms": 400.0, "queue_wait_ms": 10.0}
    RequestLog(db)._insert(RequestRecord(**{**defaults, **fields}))


def spend(db, ts, amount, *, host="rented-1", lease="lease-a", source="estimate", connection="vast"):
    db.execute("INSERT INTO spend (ts, lease_id, host_id, source, amount, connection) VALUES (?, ?, ?, ?, ?, ?)",
               (ts, lease, host, source, amount, connection))


def hourly(db):
    return [dict(r) for r in db.query("SELECT * FROM stats_requests_hourly ORDER BY hour, workload, app, host_id")]


# --- folding the request log ---


def test_requests_fold_into_one_row_per_hour_app_model_host_and_kind(db):
    stats = Stats(db, config)
    log(db, ts=HOUR + 10, tokens_out=100, tokens_in=20)
    log(db, ts=HOUR + 20, tokens_out=50, app="shop")
    log(db, ts=HOUR + 30, outcome="rejected", status_code=503, host_id=None, latency_ms=None, reason="no_eligible_host")
    log(db, ts=HOUR + 40, outcome="failed", status_code=504, latency_ms=300_000.0)
    log(db, ts=HOUR + 50, outcome="cancelled", status_code=None, latency_ms=None)
    log(db, ts=HOUR + 60, host_id="openai", cost_usd=0.002, tokens_in=10, tokens_out=40, workload="research")
    log(db, ts=HOUR - 100)  # the hour before

    assert stats.roll_up() == {"requests": 7, "spend": 0}
    rows = {(r["hour"], r["workload"], r["app"], r["host_id"]): r for r in hourly(db)}
    shared_mac = rows[(HOUR, "", "", "mac")]
    assert shared_mac["requests"] == 3 and shared_mac["ok"] == 1 and shared_mac["failed"] == 1
    assert shared_mac["cancelled"] == 1 and shared_mac["status_5xx"] == 1
    assert shared_mac["tokens_out"] == 100 and shared_mac["tokens_in"] == 20
    assert shared_mac["host_kind"] == "owned" and shared_mac["provider"] == ""
    assert shared_mac["latency_n"] == 1, "only answers that succeeded are in the latency"
    assert shared_mac["busy_ms"] == pytest.approx(400.0 + 300_000.0), "a failure held its worker too"
    named = rows[(HOUR, "", "shop", "mac")]
    assert named["requests"] == 1 and named["tokens_out"] == 50, "a named app key's requests are its own (D193)"
    refused = rows[(HOUR, "", "", "")]
    assert refused["rejected"] == 1 and refused["status_5xx"] == 1 and refused["host_kind"] == ""
    hosted = rows[(HOUR, "research", "", "openai")]
    assert hosted["host_kind"] == "hosted" and hosted["provider"] == "openai"
    assert hosted["cost_usd"] == pytest.approx(0.002)
    assert rows[(HOUR - HOUR_S, "", "", "mac")]["requests"] == 1


def test_folding_is_incremental_and_counts_each_row_once(db):
    stats = Stats(db, config)
    for i in range(5):
        log(db, ts=HOUR + i)
    assert stats.roll_up()["requests"] == 5
    assert stats.roll_up()["requests"] == 0, "nothing new: nothing folded again"
    for i in range(3):
        log(db, ts=HOUR + 100 + i)
    # A row logged late — its time in an hour already folded — still lands in its own hour.
    log(db, ts=HOUR - HOUR_S + 5)
    assert stats.roll_up(batch=2)["requests"] == 4, "in batches, each behind the watermark"
    by_hour = {r["hour"]: r["requests"] for r in hourly(db)}
    assert by_hour == {HOUR: 8, HOUR - HOUR_S: 1}


def test_two_folders_at_once_never_count_a_row_twice(db, tmp_path):
    for i in range(50):
        log(db, ts=HOUR + i)
    other = Database(tmp_path / "gpm.sqlite3")  # a second connection, as another process would hold
    try:
        first, second = Stats(db, config), Stats(other, config)
        # The second reads the same batch, but the first moves the watermark before it writes.
        rows_seen = second.db.query("SELECT COUNT(*) AS n FROM request_log")[0]["n"]
        assert rows_seen == 50
        first.roll_up(batch=20)
        second.roll_up(batch=20)
        first.roll_up(batch=20)
    finally:
        other.close()
    assert sum(r["requests"] for r in hourly(db)) == 50


def test_the_fold_skips_a_batch_another_folder_already_took(db, monkeypatch):
    for i in range(10):
        log(db, ts=HOUR + i)
    stats = Stats(db, config)
    real = stats._mark
    calls = []

    def stale_mark(name):
        calls.append(name)
        return 0  # as read before another folder moved it

    stats.roll_up()
    monkeypatch.setattr(stats, "_mark", stale_mark)
    assert stats._fold_requests(100) == 0, "the watermark moved since it was read: nothing is written"
    monkeypatch.setattr(stats, "_mark", real)
    assert sum(r["requests"] for r in hourly(db)) == 10


# --- latency, read over any range ---


def test_the_histogram_reads_percentiles_within_a_bucket():
    histogram = empty_histogram()
    for latency in range(1, 101):  # 100 ms … 10 s, evenly
        histogram[bucket_of(latency * 100.0)] += 1
    p50, p95 = percentile(histogram, 0.5, 10_000), percentile(histogram, 0.95, 10_000)
    assert 4_500 <= p50 <= 5_500
    assert 9_000 <= p95 <= 10_000
    assert percentile(empty_histogram(), 0.95) is None
    one = empty_histogram()
    one[bucket_of(1_200)] += 1
    assert percentile(one, 0.95, 1_200) <= 1_200, "never above the largest latency seen"
    assert bucket_of(10 ** 9) == len(LATENCY_BUCKETS_MS)


def test_p95_over_a_range_merges_the_hours(db):
    stats = Stats(db, config)
    for i in range(95):
        log(db, ts=HOUR - HOUR_S + i, latency_ms=200.0)
    for i in range(5):
        log(db, ts=HOUR + i, latency_ms=20_000.0)
    stats.roll_up()
    answer = stats.query(HOUR - HOUR_S, HOUR + HOUR_S)
    assert answer["totals"]["requests"] == 100
    assert answer["totals"]["p50_ms"] <= 200
    assert 150 <= answer["totals"]["p95_ms"] <= 200, "the 95th of 100 is still the fast hour's"
    assert answer["points"][1]["p95_ms"] > 15_000, "the slow hour on its own"


# --- the spend ledger's running totals ---


def test_rented_cost_per_hour_is_what_the_running_totals_rose_by(db):
    stats = Stats(db, config)
    db.execute("INSERT INTO leases (lease_id, workers, max_hours, max_spend, allow_rent, state, opened_at, workload) "
               "VALUES ('lease-a', 1, 4, 10, 1, 'open', ?, 'research')", (NOW,))
    spend(db, HOUR - HOUR_S + 60, 0.10)
    spend(db, HOUR - HOUR_S + 1800, 0.30)
    spend(db, HOUR + 60, 0.45)
    spend(db, HOUR + 1800, 0.60)
    spend(db, HOUR + 1800, 0.20, source="reported")
    stats.roll_up()
    rows = {r["hour"]: dict(r) for r in db.query("SELECT * FROM stats_rented_hourly")}
    assert rows[HOUR - HOUR_S]["estimate_usd"] == pytest.approx(0.30)
    assert rows[HOUR]["estimate_usd"] == pytest.approx(0.30), "0.60 less the 0.30 already counted"
    assert rows[HOUR]["reported_usd"] == pytest.approx(0.20)
    assert rows[HOUR]["workload"] == "research" and rows[HOUR]["connection"] == "vast"

    # More later, folded on a later pass: counted from where the totals had reached.
    spend(db, HOUR + 3000, 0.70)
    spend(db, HOUR + 3100, 0.65)  # re-based: nothing taken back
    spend(db, HOUR + 3200, 0.75)
    stats.roll_up()
    rows = {r["hour"]: dict(r) for r in db.query("SELECT * FROM stats_rented_hourly")}
    assert rows[HOUR]["estimate_usd"] == pytest.approx(0.30 + 0.10 + 0.10)
    answer = stats.query(HOUR - HOUR_S, HOUR + HOUR_S, by="provider")
    rented = next(s for s in answer["series"] if s["key"] == {"host_kind": "rented", "provider": "vast"})
    assert rented["totals"]["rented_usd"] == pytest.approx(0.80)
    assert answer["totals"]["rented_usd"] == pytest.approx(0.80)


def test_rented_cost_is_the_higher_of_estimate_and_report(db):
    stats = Stats(db, config)
    spend(db, HOUR + 10, 0.10)
    spend(db, HOUR + 10, 0.50, source="reported")
    stats.roll_up()
    assert stats.query(HOUR, HOUR + HOUR_S)["totals"]["rented_usd"] == pytest.approx(0.50)


# --- what the stats answer ---


def test_series_by_app_name_workloads_named_keys_and_the_shared_hosts(db):
    stats = Stats(db, config)
    log(db, ts=HOUR + 1)
    log(db, ts=HOUR + 2, app="shop")
    log(db, ts=HOUR + 3, app="shop", outcome="failed", status_code=502)
    log(db, ts=HOUR + 4, workload="research")
    spend(db, HOUR + 5, 0.25, lease="lease-shared")
    stats.roll_up()
    answer = stats.query(HOUR, HOUR + HOUR_S, by="app")
    by_label = {s["label"]: s for s in answer["series"]}
    assert by_label["shop"]["type"] == "app" and by_label["shop"]["totals"]["errors"] == 1
    assert by_label["research"]["type"] == "workload"
    assert by_label["shared pool (configured app key)"]["totals"]["requests"] == 1
    shared_hosts = next(s for s in answer["series"] if s["key"] == {"shared_hosts": True})
    assert shared_hosts["totals"]["rented_usd"] == pytest.approx(0.25)
    narrowed = stats.query(HOUR, HOUR + HOUR_S, by="host", filters={"app": "shop"})
    assert narrowed["totals"]["requests"] == 2 and narrowed["totals"]["rented_usd"] == 0
    assert narrowed["notes"], "says why rented cost is left out"
    shared = stats.query(HOUR, HOUR + HOUR_S, filters={"workload": "shared"})
    assert shared["totals"]["requests"] == 3


def test_daily_buckets_are_utc_days(db):
    stats = Stats(db, config)
    day = int(NOW // DAY_S) * DAY_S
    log(db, ts=day + 10)
    log(db, ts=day - 10)
    stats.roll_up()
    answer = stats.query(day - DAY_S, day + DAY_S, bucket="day")
    assert answer["buckets"] == [day - DAY_S, day]
    assert [p["requests"] for p in answer["points"]] == [1, 1]


# --- retention (D195) ---


def test_old_rows_go_once_folded_and_not_before(db):
    stats = Stats(db, lambda: config(keep_requests_days=30, keep_rollups_days=60))
    old = NOW - 40 * DAY_S
    log(db, ts=old, kind="embed")  # not a row the machine record or sizing reads
    stats.roll_up()
    log(db, ts=old + 1, kind="embed")  # as old, but not folded yet
    log(db, ts=NOW - 60)
    EventLog(db).record("worker_adjusted", "an old decision")
    EventLog(db).record("rented", "an old rental", host_id="rented-1", lease_id="lease-a")
    db.execute("UPDATE events SET ts = ?", (old,))
    db.execute("INSERT INTO stats_requests_hourly (hour, workload, app, model, host_id, host_kind, provider, kind) "
               "VALUES (?, '', '', 'm1', 'mac', 'owned', '', 'chat')", (int(NOW - 90 * DAY_S) // HOUR_S * HOUR_S,))

    removed = stats.prune(NOW)
    assert removed["request_log"] == 1, "the unfolded row is kept until it is folded"
    assert [e["kind"] for e in EventLog(db).recent(10)] == ["rented"], "the machine record is kept"
    assert removed["stats"] == 1
    stats.roll_up()
    assert stats.prune(NOW)["request_log"] == 1
    assert len(RequestLog(db).rows()) == 1
    assert stats.query(old - HOUR_S, old + HOUR_S)["totals"]["requests"] == 2, "what was pruned is still counted"


def test_pruning_never_changes_what_budgets_read(db):
    """Hosted spend this month and over a provisioner's 30 days is read from the request log; it
    is kept for 32 days whatever the retention says."""
    stats = Stats(db, lambda: config(keep_requests_days=7, keep_rollups_days=7))
    for days in (2, 20, 29.5, 31, 40):
        log(db, ts=NOW - days * DAY_S, host_id="openai", workload="research", cost_usd=0.01, kind="embed")
    reader = RequestLog(db)
    day, month = period_starts(NOW)
    before = (reader.hosted_spend_of(["research"], NOW - 30 * DAY_S), reader.hosted_spend_of(["research"], month),
              len(reader.hosted_spend_since(month)))
    stats.roll_up()
    stats.prune(NOW)
    after = (reader.hosted_spend_of(["research"], NOW - 30 * DAY_S), reader.hosted_spend_of(["research"], month),
             len(reader.hosted_spend_since(month)))
    assert after == before
    assert len(reader.rows()) == 4, "only the row older than every budget's look-back went"
    assert min(r["ts"] for r in reader.rows()) >= NOW - BUDGET_LOOKBACK_S


def test_pruning_never_changes_what_lease_and_burst_enforcement_read(db):
    """The ledger keeps running totals: a lease's spend is each host's newest figure, a burst's
    today and this month what the totals rose by since then. Both read the same after pruning."""
    stats = Stats(db, lambda: config(keep_requests_days=7, keep_rollups_days=400))
    events = EventLog(db)
    events.record("rented", "a burst host", host_id="burst-1", lease_id="lease-a", numbers={"burst": 1})
    events.record("rented", "another", host_id="burst-2", lease_id="lease-a", numbers={"burst": 1})
    db.execute("UPDATE events SET ts = ?", (NOW - 50 * DAY_S,))
    day, month = period_starts(NOW)
    total = 0.0
    # A sample every six hours for fifty days, both sources; the second host stopped 45 days ago.
    for i in range(200):
        ts = NOW - 50 * DAY_S + i * 6 * HOUR_S
        total += 0.05
        spend(db, ts, total, host="burst-1")
        spend(db, ts, total * 0.9, host="burst-1", source="reported")
        if ts < NOW - 45 * DAY_S:
            spend(db, ts, total, host="burst-2")
    # One just before today began and this month began, nothing after the month start for a while.
    spend(db, month - 5, total + 0.01, host="burst-2")
    ledger = SpendLedger(db)
    fleet = SimpleNamespace(events=SimpleNamespace(db=db))

    def read():
        return (ledger.latest_for_lease("lease-a"), Fleet.burst_spend(fleet, "lease-a", day),
                Fleet.burst_spend(fleet, "lease-a", month))

    before = read()
    stats.roll_up()
    removed = stats.prune(NOW, batches=1000)
    assert removed["spend"] > 300, "superseded rows do go"
    assert read() == before
    stats.roll_up()
    assert stats.prune(NOW, batches=1000)["spend"] == 0
    assert read() == before, "and again on the next prune"
    assert stats.query(NOW - 51 * DAY_S, NOW + 1, bucket="day")["totals"]["rented_usd"] == pytest.approx(
        max(before[0]["estimate"], before[0]["reported"]), rel=1e-6), "the hourly table still has every dollar"


def test_retention_is_validated_with_defaults():
    assert StatsConfig().keep_requests_days == 30 and StatsConfig().keep_rollups_days == 400
    with pytest.raises(ValueError, match="keep_rollups_days is shorter"):
        StatsConfig(keep_requests_days=60, keep_rollups_days=30)
    with pytest.raises(ValueError):
        StatsConfig(keep_requests_days=6)
    assert StatsConfig(keep_requests_days=None).keep_requests_days is None, "null keeps them for ever"
    pool = PoolConfig.model_validate({
        "pool": {"name": "p", "model_set": ["m1"]}, "auth": {"app_keys": ["gpma_stats_app"]},
        "hosts": [{"id": "mac", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "stats": {"keep_requests_days": 7},
    })
    assert pool.stats.keep_requests_days == 7 and pool.stats.keep_rollups_days == 400


def test_never_pruned_when_kept_for_ever(db):
    stats = Stats(db, lambda: config(keep_requests_days=None, keep_rollups_days=None))
    log(db, ts=NOW - 1000 * DAY_S)
    stats.roll_up()
    assert stats.prune(NOW) == {"request_log": 0, "events": 0, "spend": 0, "stats": 0}
    assert json.loads(hourly(db)[0]["latency_hist"])[bucket_of(400.0)] == 1


def test_a_database_from_before_gains_the_app_column_and_the_stats_tables(tmp_path):
    """An existing pool's file is migrated on open: a request log without `app` gains it, and the
    tables the stats and named keys need are made — no row is lost."""
    import sqlite3

    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE request_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, request_id TEXT NOT NULL, "
                "session_id TEXT, host_id TEXT, worker_id TEXT, model_requested TEXT, model_served TEXT, runtime_class TEXT, "
                "queue_wait_ms REAL, latency_ms REAL, status_code INTEGER, outcome TEXT NOT NULL, reason TEXT)")
    old.execute("INSERT INTO request_log (ts, request_id, outcome, host_id, model_requested) VALUES (?, 'old', 'ok', 'mac', 'm1')",
                (NOW,))
    old.commit()
    old.close()
    path.chmod(0o600)
    db = Database(path)
    try:
        columns = {r[1] for r in db.query("PRAGMA table_info(request_log)")}
        assert {"app", "workload", "cost_usd"} <= columns
        tables = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"app_keys", "stats_requests_hourly", "stats_rented_hourly"} <= tables
        stats = Stats(db, config)
        assert stats.roll_up()["requests"] == 1
        assert stats.query(HOUR - HOUR_S, HOUR + HOUR_S)["totals"]["requests"] == 1
    finally:
        db.close()


# --- found in review ---


def test_pruning_a_long_spend_ledger_takes_seconds_not_minutes(db):
    """One host's ledger for a long lease: what is kept is worked out once per key, not by a
    lookup per row (found in review: 95 000 rows took 40 s, a host-month 25 minutes, holding the
    stats lock all the while)."""
    stats = Stats(db, lambda: config(keep_requests_days=7, keep_rollups_days=400))
    day, month = period_starts(NOW)
    rows, total = [], 0.0
    for i in range(120_000):  # every 30 s for about 42 days
        total += 0.001
        rows.append((NOW - 42 * DAY_S + i * 30, "lease-a", "rented-1", "estimate", total, "vast"))
    with db.transaction() as conn:
        conn.executemany("INSERT INTO spend (ts, lease_id, host_id, source, amount, connection) VALUES (?, ?, ?, ?, ?, ?)",
                         rows)
    ledger, fleet = SpendLedger(db), SimpleNamespace(events=SimpleNamespace(db=db))
    db.execute("INSERT INTO events (ts, kind, host_id, lease_id, summary, numbers) VALUES (?, 'rented', 'rented-1', "
               "'lease-a', 'a burst host', '{\"burst\": 1}')", (NOW - 42 * DAY_S,))
    before = (ledger.latest_for_lease("lease-a"), Fleet.burst_spend(fleet, "lease-a", day),
              Fleet.burst_spend(fleet, "lease-a", month))
    stats.roll_up(batches=1000)
    started = time.monotonic()
    removed = stats.prune(NOW)
    took = time.monotonic() - started
    assert took < 5.0, f"the prune took {took:.1f} s"
    assert removed["spend"] > 90_000
    assert (ledger.latest_for_lease("lease-a"), Fleet.burst_spend(fleet, "lease-a", day),
            Fleet.burst_spend(fleet, "lease-a", month)) == before


def test_pruning_leaves_what_sizing_and_the_machine_record_read(db):
    """A build's latency on a card (D156) and the machine record (D69) read the newest rows of the
    request log, however old: the prune spares as many as they read."""
    from gpm_server.stats import MEASURED_ROWS

    stats = Stats(db, lambda: config(keep_requests_days=7, keep_rollups_days=400))
    events = EventLog(db)
    events.record("rented", "a machine", host_id="rented-1", lease_id="lease-a",
                  numbers={"machine": "m-1", "hardware": "1x RTX TEST 48GB"})
    old = NOW - 40 * DAY_S
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO request_log (ts, request_id, host_id, model_served, outcome, status_code, latency_ms, "
            "concurrency, model_concurrency, kind, tokens_out, generate_ms) VALUES (?, ?, 'rented-1', 'tag-a', 'ok', 200, ?, ?, ?, "
            "'chat', 50, ?)",
            [(old + i, f"r{i}", 1000.0 + (i % 7) * 400, 1 + i % 6, 1 + i % 6, 900.0) for i in range(MEASURED_ROWS + 5000)])

    def stub():
        fleet = SimpleNamespace(events=events, _sizing_cache={}, _hardware_cache=None, _history=None, _history_at=0.0,
                                _card_of=Fleet._card_of)
        fleet._hardware_of_hosts = lambda: Fleet._hardware_of_hosts(fleet)
        return fleet

    def read():
        fleet = stub()
        at = Fleet.model_at_latency(fleet, "1x RTX TEST 48GB", "tag-a", 2.0, ceiling=6)
        samples = next(iter(fleet._sizing_cache.values()))[1]
        machines = {k: v.as_dict() for k, v in Fleet.machine_history(fleet).items()}
        return at, samples, machines

    before = read()
    assert len(before[1]) == MEASURED_ROWS and before[2]["m-1"]["requests"] == MEASURED_ROWS
    stats.roll_up(batches=1000)
    assert stats.prune(NOW)["request_log"] == 5000, "the older ones beyond what is read do go"
    assert read() == before


def test_events_renting_reads_back_and_those_of_open_leases_are_kept(db):
    stats = Stats(db, lambda: config(keep_requests_days=7, keep_rollups_days=400))
    db.execute("INSERT INTO leases (lease_id, workers, max_hours, max_spend, allow_rent, state, opened_at) "
               "VALUES ('open-lease', 1, 2000, 10, 1, 'open', ?), ('closed-lease', 1, 4, 10, 1, 'closed', ?)", (NOW, NOW))
    events = EventLog(db)
    events.record("replaced_for_value", "a host replaced for value", host_id="h1", lease_id="closed-lease")
    events.record("machine_avoided", "a machine avoided", host_id="h1", numbers={"avoid_until": NOW + DAY_S})
    events.record("worker_adjusted", "of an open lease", host_id="h1", lease_id="open-lease")
    events.record("worker_adjusted", "of a closed lease", host_id="h1", lease_id="closed-lease")
    events.record("worker_adjusted", "of none")
    db.execute("UPDATE events SET ts = ?", (NOW - 40 * DAY_S,))
    stats.prune(NOW)
    assert sorted(e["summary"] for e in events.recent(10)) == ["a host replaced for value", "a machine avoided",
                                                              "of an open lease"]


def test_retention_is_at_least_a_week():
    with pytest.raises(ValueError):
        StatsConfig(keep_requests_days=6, keep_rollups_days=400)
    assert StatsConfig(keep_requests_days=7).keep_requests_days == 7
