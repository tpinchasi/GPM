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
- **Several providers** are designed, not built: their connections, credentials and the market
  across them are in [providers.md](providers.md) (D129–D132).
- **Secrets never reach the browser.** The console shows only "set ✓ / missing ✗" against the
  environment variable or file a secret is read from.

## 2. Screens

| Screen | Shows | Does |
|---|---|---|
| **Overview** | Hosts grouped by routing tier: state, busy / total workers (each worker and its current request on expand), build being served, cost per hour. Open leases with burn-down against their caps. Queue depth and wait. Live event feed | **Release all rented** — the panic button, always visible. Drain / release per host |
| **Hosts** | Every `local` and `fixed-remote` host: transport, priority, workers (profile ceiling, memory ceiling, the number in force), capabilities and how each was learned | Add / edit / disable. **Test connection** before saving. Restart engine |
| **Rented capacity** | In five tabs, each at `#rented/<tab>` (D102): **Hosts** — provider account, what is rented and burning against the limits, Prepare a host, the rented and parked hosts; **Engine & models**; **Finding machines** — the search beside the live market; **Scaling** — limits and allocation; **Tear-down**. Only the last three read the market's settings, and none of them searches the market by itself: a search happens when the operator presses **Search the market** or **Try these**, and the page shows its last search, with its time, until the next (D120) — the provider counts every offer returned against a daily quota the pool needs for renting. The day's use of that quota is shown beside the search and in the Provider panel — counted from the pool's own searches, the provider's own figure once it refuses (D121). A page loaded from an older release reloads itself when the status reports a newer one, or, while the operator is in the middle of something, says so with a Reload button (D122). Across them: provider account (credential valid, credit left); offer policy, bid strategy, tear-down settings — each field beside its default and a one-line reason. Rented and **parked** hosts with their running and storage cost | Edit with **live market preview** — both rental kinds listed, labelled, each row with its own **Rent**. **Prepare a host**, as a bid, on demand, or as configured. Restart or destroy a parked host  **How capacity is decided** is edited here too (D74): the allocation mode, the ramp's settings and per-host worker adjustment, saved through the same file-plan-retype path as the offer search. |
| **Rented capacity → what the pool looks for** | **How the pool rents** — `interruptible`, `on_demand` or `cheaper` — chosen first, because it decides which listings are searched at all and an offer in a listing the pool never asks for cannot appear among the rejections below (D80). Then every offer-policy and bidding parameter, editable in place. **Try these** runs the real pipeline against the live market with the unsaved values and saves nothing; **Save** writes them into the configuration file *in place* — comments, ordering and flow style untouched — then validates, plans, and applies, with loosening retyped as anywhere else | `PATCH /pool/config/rented` |
| **Rented capacity → Profiles** | What each machine the pool rents holds (D111, D112); for an engine that can, how many cards each model is split across, the needs then per card of a group (D114). The engine rented machines run, then each **model profile** as a card: a rent tick, its name, whether a machine holds **one model** (choosing another replaces it) or **several** (a chosen few), and its rows — each a **model and the variant** a machine fetches for it, with precision and size — and the least card and disk that needs. **Every picker here chooses a model and its variant together, in one click**: *In the pool* lists the variants the pool already has that rented machines can run, with no search; a search lists every model on the model hub (or in Ollama's library, for an Ollama pool) with **its variants side by side** — the original and its quantisations — filtered by parameters, size, card fit, precision and kind. A variant new to the pool is added under the pool's name for that model, or a name typed beside it. **Machine setup** — images, engine options, start command — is folded below. **Saved as one change** with the engine, since a variant is one engine's; a variant is written into the catalog after those there, only if a profile uses it | `PATCH /pool/config/engine` `{profiles, rent, add}` · `GET /pool/models/search` · `GET /pool/models/size` |
| **Rented capacity → finding machines: the next host** | What the next machine would be bought as, what that needs, and — where it is more than the minimums set here — the card, disk and whole groups of cards the search asks for because of it (D111, D114) | — (`GET /pool/market/preview` `next_host`) |
| **Workloads** | The running workloads — state, hosts ready, latency p95 against the target, spend against the cap, hours left — each with Extend…, New key and End; a form that plans a new one — a row per model, with "Add model" and, for several, a placement (D118) — (the plan shows the start, the first host, the time to serve and the budget, typed or proposed; for several models, both placements priced, the one kept and each group's split) before Create is allowed, a proposed budget typed again to accept it, and the key shown once with copy buttons until put away (D115) | `/pool/workloads…` |
| **Hosts → out of service, remove** | Per configured host: **Take out of service** sets `disabled` in the file — the host stays, stops receiving requests, finishes those on it, and **Return to service** brings it back. **Remove…** deletes its entry after showing the plan and asking for the host's id to be typed. Taking out of service is never refused; when it leaves models with no host in service, the plan names them, and requests for them are refused until the host returns (D113). Removal goes through the file's rules: removing the only host serving a model is refused with the reason, before anything is asked. Rented hosts are released from Rented capacity instead (D99) | `PATCH /pool/config/hosts/{id}` `{disabled}` · `DELETE /pool/config/hosts/{id}` `{confirm: id}` |
| **One host** | Opened from any host id. Its state and stage in words (*starting · fetching models: 1 of 3 on disk; downloading gemma4:26b — 12.1 of 18.6 GB · every model is on disk; the engine is starting on them · loading into memory · ready*), what the provider says about the machine, tunnel, cost so far, **one state per model** — *not here yet · downloading* (with its bar and rate) *· on disk · loading · loaded · failed* (with the engine's reason) — derived from the pool's own downloads, the agent's report and the engine's list together, and that host's own slice of the decision log. An engine that is started once its weights have landed is shown as *not started yet* while they land, not as a fault. Follows the host while open | — (`GET /pool/hosts/{id}`, `gpm host show <id>`) |
| **Models** | The pool's model set; the catalog of logical names and variants; per-host matrix of which build is served, its runtime class, whether it enforces schemas; capacity profiles and calibration entries | Edit the set and the catalog (changing the set re-prepares hosts — shown in plan) |
| **Leases** | Open and past leases: workers, caps, spend so far — **estimated and provider-reported side by side**, with the margin left before the cap — and the hosts charged to each | Open (worst-case confirmation), close, tighten, and **extend** — hours, dollars **and the worker ceiling**, each raise retyped to confirm. Under dynamic allocation that ceiling is what decides whether another host is ever rented, so it is edited here rather than only in the CLI (D80) |
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
parked); confirm the all-in price ceiling, dollar cap and time limit with the worst case stated; then
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

Both processes follow the file. The supervisor applies a change on its next pass; the router re-reads the file in the background, off the request path, and serves under it from then on — the model set it reports, the app keys, the limits (D113). A file that does not load leaves both running as they were.

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
| `GET /pool/market/preview` | Run the offer pipeline read-only with supplied settings — the market is searched only with `search=true`; without it, the settings and the next host's needs only (D120, D122). `POST` likewise. `kinds=both` lists bid and on-demand offers together whatever `rented.mode` is; every row carries its `offer_id` and `kind` |
| `POST /pool/keys`, `DELETE /pool/keys/{id}` | Create, rotate and revoke app and admin keys |
| `GET /pool/builds?model=&engine=&search=&fresh=` | An engine's builds of a model on the model hub, sorted (D100); from the directory's cache unless stale or `fresh` |
| `GET /pool/directory?q=`, `POST /pool/directory/refresh` | The model directory from the cache; start a refresh in the background (D101) |
| `POST /pool/config/models` | Add models from the directory: `add: [{name, builds: {engine: build}, rent_for}]`, `engine_options` — one write, validated, planned, confirmed (D101) |
| `POST /pool/workloads/plan` | What creating a workload would do — its build, workers per host at its latency (measured or not), hosts at the start, the first host and its kind, minutes to serve, the budget typed or derived, and whether the pool's caps allow it. Opens and rents nothing (D115) |
| `POST /pool/workloads` | Create it: `{name, model, latency_s, parallel, hours, max_spend?, profile?, kind?}` — or, for several models (D118), `models: [{model, latency_s, parallel}, …]` and `placement?` (`auto`, `together`, `apart`) in place of the first three; a budget not typed must come back as `confirm_max_spend`. Opens its lease, mints its key and answers with `connection: {base_url, api_key}` — **the only time the key is shown** |
| `GET /pool/workloads`, `GET /pool/workloads/{name}` | Each workload: state, hosts, spend against its cap, hours left, answers served with their p95 against the target, borrowed and refused counts, key ids — never a key or a hash |
| `POST /pool/workloads/{name}/extend` | Hours added and/or a new budget; a raise carries `confirm`, the value typed again (D49) |
| `POST /pool/workloads/{name}/end` | Its lease closes; its hosts drain and go; its key reaches only a `workload_ended` answer |
| `POST /pool/workloads/{name}/keys` | A new key, shown once; the old ones work for `workloads.rotation_grace_minutes` |
| `PUT /pool/config/profiles` | The model profiles and which the pool rents as: `profiles: {name: {model: build}}`, `rent: [name]`, `split: {name: cards}` (optional, D114 — not sent, the file's split is kept for the profiles still named), `add: [{name, builds: [{tag, engine, size_gb}]}]` — one write, validated, planned, confirmed (D111). The same fields ride on `PATCH /pool/config/engine` in place of `placement` |
| `GET /pool/models/search?q=` | Any model whose name holds `q`: the model hub asked live, with no credential, as models each with its variants — the original and its quantisations, each with precision, the cards it runs on and, where the hub's tally is exact, its size — and Ollama's library from the directory's cache, saying whether it has ever been read (D111, D112) |
| `GET /pool/models/size?repo=` | One hub repository's weights, exactly, from its file listing; read once, then kept (D112) |

CLI verbs map one-to-one: `gpm status | plan | serve | stop [--release] | restart | lease … |
host test | host prepare | host drain | host release | host park | down --all | key …`.

## 5. How it is built

A static page (HTML and JavaScript, no build step) served by the supervisor process at `/ui`,
with live updates over server-sent events. It has no server-side session: the admin key is held
in the page's memory for the tab's lifetime and sent as a header on each call.
