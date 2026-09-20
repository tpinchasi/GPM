# Specification — Operator Console and Control API

> Reasons are in [../decisions.md](../decisions.md) (D8, D16, D21). The console is for the
> **operator**. It is not part of any app, and apps never see it.

## 1. Ground rules

- **The console is one more client of the control API.** Everything it does goes through the
  same `/pool/*` endpoints the CLI uses. There is no console-only capability, so anything
  clickable is also scriptable.
- **One source of truth for configuration: the configuration file.** The console edits it
  *through* the API — validate, then atomic write, then reload — and holds no settings of its
  own. Editing the file by hand keeps working; the supervisor notices and reloads. Every
  applied version is kept (last 50) with a diff and one-click rollback.
- **Changing configuration never spends money.** Spending is only ever a lease or a host
  preparation, each behind a confirmation that states the worst case in dollars.
- **Loosening is harder than tightening.** Lowering a ceiling applies on save. Raising a bid
  ceiling, the hourly burn cap or the maximum rented hosts; raising or removing the offer
  policy's two price ceilings (`max_all_in_hourly`, `max_download_per_gb`); or switching on
  `allow_insecure`, requires typing the new value again.
- **Nothing is applied blind.** Save runs validate → **plan** → apply: the console shows what
  the change will cause right now ("host X's bid is above the new ceiling → it will be drained
  and released") before the operator commits. **A plan is complete by construction**: what it
  cannot explain in words it still reports, setting by setting, so no part of the file can
  change without appearing in it (D42).
- **The admin key is required on every control-API request**, sent as a header, never a cookie,
  with `Host` / `Origin` checked — on loopback too, because any web page open in the same
  browser can send requests to `127.0.0.1`. The app key is refused here.
- **Secrets never reach the browser.** The console shows only "set ✓ / missing ✗" against the
  environment variable or file a secret is read from.

## 2. Screens

| Screen | Shows | Does |
|---|---|---|
| **Overview** | Hosts grouped by routing tier: state, busy / total workers (each worker and its current request on expand), build being served, cost per hour. Open leases with burn-down against their caps. Queue depth and wait. Live event feed | **Release all rented** — the panic button, always visible. Drain / release per host |
| **Hosts** | Every `local` and `fixed-remote` host: transport, priority, workers (profile ceiling, memory ceiling, the number in force), capabilities and how each was learned | Add / edit / disable. **Test connection** before saving. Restart engine |
| **Rented capacity** | Provider account (credential valid, credit left); offer policy, bid strategy, tear-down settings — each field beside its default and a one-line reason. Rented and **parked** hosts with their running and storage cost | Edit with **live market preview** — both rental kinds listed, labelled, each row with its own **Rent**. **Prepare a host**, as a bid, on demand, or as configured. Restart or destroy a parked host |
| **Rented capacity → what the pool looks for** | Every offer-policy and bidding parameter, editable in place. **Try these** runs the real pipeline against the live market with the unsaved values and saves nothing; **Save** writes them into the configuration file *in place* — comments, ordering and flow style untouched — then validates, plans, and applies, with loosening retyped as anywhere else | `PATCH /pool/config/rented` |
| **One host** | Opened from any host id. Its state and stage in words (*starting · downloading gemma4:26b 12.1 of 18.6 GB · loading into memory · ready*), what the provider says about the machine, tunnel, cost so far, the model set with what is on disk and what is loaded, per-model download progress, and that host's own slice of the decision log. Follows the host while open | — (`GET /pool/hosts/{id}`, `gpm host show <id>`) |
| **Models** | The pool's model set; the catalog of logical names and variants; per-host matrix of which build is served, its runtime class, whether it enforces schemas; capacity profiles and calibration entries | Edit the set and the catalog (changing the set re-prepares hosts — shown in plan) |
| **Leases** | Open and past leases: workers, caps, spend so far — **estimated and provider-reported side by side**, with the margin left before the cap — and the hosts charged to each | Open (worst-case confirmation), close, tighten |
| **Decisions** | The event log, filterable by host and lease | Expand any decision to see the numbers behind it — offers considered and why each was rejected, the floor, the bid, the options compared on an eviction |
| **Configuration** | The raw file, validation errors, version history with diffs | Edit as text for whatever the forms do not cover; roll back |

### 2.1 Test connection

What makes adding a host safe. Given the form as filled in, **without saving**: reach the
endpoint over the chosen transport → authenticate → list models → check the pool's whole model
set is resident together → inspect capabilities → run the **concurrency check** (*n* short
generations at once against one; near *n*× wall time means the engine is serialising, and the
console says what engine setting to change) → report the worker count that would apply. Each
step passes or fails with the actual error. A plain-`http` public address without auth fails
here, with the reason, rather than at the next restart.

### 2.2 Live market preview

Turns bidding settings into something observable. With the values **currently in the form, not
yet saved**, it runs the real offer pipeline against the live market, read-only:

```
 Offer policy (unsaved)                       Market right now — 80 offers
 ─────────────────────────────                ─────────────────────────────────────────────
 Min memory           64 GB                   4 pass · 76 rejected
 All-in ceiling       $0.66 /h
 Download ceiling     $0.010 /GB              #  hardware      bid     all-in  download
 Bid strategy         floor + $0.02           1  <card A>      $0.153  $0.211  $0.09  ◀ would rent
 Bid ceiling          $0.60 /h                2  <card B>      $0.284  $0.330  $0.21
 On-demand crossover  0.8 ×                   3  …

 [ Preview ]  [ Save ▸ plan ]                 Rejected, by reason
                                              41  memory bandwidth outside band
                                              19  all-in above ceiling
                                              11  download price above ceiling   · 5 other
```

Moving a ceiling and watching "4 pass" become "0 pass" is the fastest way to learn what a number
means. *(Strategy replay against recorded markets is post-v1.)*

### 2.3 Prepare a host

The console flow for [supervisor.md](supervisor.md) §8: choose models (the pool's set by
default) and an offer — *best by policy* or one row of the live market list; see the estimate
(download size × that host's price per gigabyte, time to ready, hourly rate, storage rate if
parked); confirm the bid ceiling, dollar cap and time limit with the worst case stated; then
watch bid → instance up → per-model download with cost so far → verify → workers sized → ready;
finally **join**, **park** or **destroy**.

## 3. What takes effect when

| Change | Effect |
|---|---|
| Routing priority, queue timeout | Next request |
| A host's worker count | Raised: new workers start idle at once. Lowered: surplus workers drain after their current request |
| Tightened ceilings and caps; idle, drain and park timers | Next control-loop pass — may drain a host, which plan shows first |
| Offer policy, bid strategy | Next bid or re-bid. Running hosts are not re-shopped |
| Transport or auth of an existing host | The host is drained, reconnected, re-tested |
| The pool's model set or context length | Hosts are re-prepared one at a time; worker counts recomputed; hosts that no longer fit leave the pool — all shown in plan |
| Catalog | Next request; affected (host, model) pairs re-prepared |
| Router listen address, TLS | Restart of the **router** process only. Rented hosts are unaffected |

## 4. Control API (admin key)

| Endpoint | Purpose |
|---|---|
| `GET /pool/status` | Hosts, workers, queue, limits in force, contract version. **Also readable with the app key**, minus cost and provider detail |
| `GET /pool/events`, `GET /pool/events/stream` | Decision log; server-sent events for the console |
| `GET/PUT /pool/config` | Versioned; a write based on a stale version is rejected |
| `POST /pool/config/validate`, `POST /pool/config/plan` | Check, and preview consequences, without applying |
| `GET/POST /pool/leases`, `DELETE /pool/leases/{id}`, `PATCH …` (tighten only) | Leases |
| `POST /pool/hosts/test` | Test connection for an unsaved host definition |
| `POST /pool/hosts/prepare`, `GET /pool/hosts/prepare/{id}` | Prepare a rented host; progress. Optional `kind` (`interruptible` · `on_demand`) rents the best offer of that kind; optional `offer_id` rents that one offer or nothing (D55). A refusal says the actual reason and that nothing was spent |
| `POST /pool/hosts/{id}/drain | release | park | restart | disable | enable` | Per-host actions |
| `POST /pool/down` | Destroy all rented hosts now, verified |
| `GET /pool/market/preview` | Run the offer pipeline read-only with supplied settings. `kinds=both` lists bid and on-demand offers together whatever `rented.mode` is; every row carries its `offer_id` and `kind` |
| `POST /pool/keys`, `DELETE /pool/keys/{id}` | Create, rotate and revoke app and admin keys |

CLI verbs map one-to-one: `gpm status | plan | serve | stop [--release] | restart | lease … |
host test | host prepare | host drain | host release | host park | down --all | key …`.

## 5. How it is built

A static page (HTML and JavaScript, no build step) served by the supervisor process at `/ui`,
with live updates over server-sent events. It has no server-side session: the admin key is held
in the page's memory for the tab's lifetime and sent as a header on each call.
