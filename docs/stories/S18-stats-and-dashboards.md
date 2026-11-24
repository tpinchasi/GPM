# S18 — Stats kept, and dashboards

> Status: **built against fakes** (2026-11-24) — D193–D197. Part of the [feature list](README.md).

## The story

The owner, 2026-11-24: "add stats storing and dashboards for showing app level, host/cost level,
provider level etc." Before this, the pool kept a row per request and per rented host per pass,
for ever, and nothing summed them over time: the Spending screen showed today and this month, and
a workload's page its last answers.

The owner chose what "an app" is: **workloads, and named keys** — a workload is an app, as today,
and several apps may share the shared pool, each with a key of its own and a name, so each is
counted apart. That takes up the per-client identity D16 and D115 had deferred.

## What was built

- **Named app keys (D193).** `app_keys` in the database, kept as hashes; `GET/POST /pool/apps`,
  `DELETE /pool/apps/{app}` and `…/keys/{key_id}`; `gpm app create/list/revoke`; a section on the
  Workloads screen that shows a new key once. Several keys may hold one name, so a key is rotated
  without a pause. The router reads the hashes in the background, resolves a key to its app, and
  writes it in the request log's new `app` column. A named key has exactly the app key's powers;
  the control API refuses it in words. The configured app keys keep working, as the unnamed app.
- **Stats kept (D194).** The supervisor folds the request log into `stats_requests_hourly` — per
  hour, workload, app, model, host, provider and kind: requests and how they ended, tokens,
  latency with a fixed-bucket histogram, worker time, queue wait, hosted cost — and the spend
  ledger's running totals into `stats_rented_hourly`, what each hour added per host, lease and
  connection. A watermark moved in the same transaction as the rows makes a fold idempotent.
  It runs every minute in a thread, on its own database connection, and before every stats read.
- **Retention (D195).** `stats: {keep_requests_days: 30, keep_rollups_days: 400}` in `pool.yaml`,
  validated with defaults. The request log, the decision log and superseded spend rows are pruned,
  in batches, once an hour. Never pruned: a row not yet folded, a hosted cost the budgets still
  read (32 days), the machine record's events, and the spend rows lease and burst enforcement
  read — the tests show both read the same before and after a prune.
- **`GET /pool/stats` (D196)** — series and totals by hour or day, by app, model, host, provider
  or kind, narrowed by any; p50 and p95 from the histograms; cost hosted, rented and both. And
  `gpm stats`.
- **A Dashboards screen (D197)** — Apps, Hosts & cost, Providers, Models; 24 h, 7 d, 30 d; total
  tiles, line charts drawn by the page as SVG with a title, legend, a tooltip per time slot and a
  table of the same numbers, in light and dark, at a phone's width.

## What it gives up

- Latency percentiles are read from buckets, so each is within its bucket's width of the true value.
- Rented machines' cost is per machine: it is not split across the apps, models or kinds that
  shared one. On the Apps tab the shared pool's rented hosts are a line of their own.
- "Utilisation" is busy worker-hours — the time requests held a worker — not a share of the
  workers a host had, which the pool does not keep per hour.
- A latency measured on a class of card (D156) is forgotten with its request rows after
  `keep_requests_days`; a pool that wants a longer memory for sizing raises it.

## Not run on a live pool

Built and tested against fakes and a real browser. A pool upgraded with a long request log folds
it in batches over its first passes; how long that takes on a large live database has not been
measured.
