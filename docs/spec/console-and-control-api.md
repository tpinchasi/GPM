# Specification — Operator Console and Control API

> Reasons are in [../decisions.md](../decisions.md) (D8, D16, D21; named app keys and dashboards D193–D197). The console is for the
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
  browser can send requests to `127.0.0.1`. The app key is refused here, and every named app key (D193) with it.
- **Several providers** rent side by side (D129–D136). The **Providers** screen is where
  their accounts are managed, in one list with the hosted model APIs (S13): a card each, with its icon, credential state, quota, holdings and
  capabilities, and Add provider — choose, name, credential, test, save — with no restart
  ([providers.md](providers.md) §7.1). A provider credential is the one secret typed into the
  console: sent once, never sent back (D130). Each host and offer is marked with its account and
  how it is paid for — **bid**, **spot** or **on demand** — and the market says when a provider
  could not be asked while the others answered.
- **Secrets never reach the browser.** The console shows only "set ✓ / missing ✗" against the
  environment variable or file a secret is read from.

## 2. Screens

| Screen | Shows | Does |
|---|---|---|
| **Overview** | Hosts grouped by routing tier: state, busy / total workers (each worker and its current request on expand), build being served, cost per hour. Open leases with burn-down against their caps. Queue depth and wait. Live event feed | **Release all rented** — the panic button, always visible. Drain / release per host |
| **Hosts** | Every `local` and `fixed-remote` host: transport, priority, workers (profile ceiling, memory ceiling, the number in force), capabilities and how each was learned; for a host whose agent installs its engine (S9), the engine's running, staged, previous and latest versions and what is being fetched or built | Add / edit / disable. **Test connection** before saving. Restart engine (answered at once; the outcome is an event, D162). **Upgrade now** and **Roll back** (D153) |
| **Providers** | Every provider the pool gets capacity from, in one list (S13, D171): a summary (how many, how many working, the hourly burn of rented machines, hosted spend today against the budget), then a card per rented account (providers.md §7.1) and per hosted API — status, address, its key's source, requests at once, spend, **balance** (provider-reported, typed or unknown, as of when, counted down — providers.md §12, D188), models and prices. The only place provider accounts are listed: `#rented/providers` leads here | **Add provider** — one first step listing both kinds, *Rent GPU machines* (the installed plug-ins) and *Hosted model APIs* (choose → key → test against `/models` → models and prices → save); on a hosted card Test, Type in / Replace key, Models and prices, **Settings** (requests at once; prompt caching in Anthropic's format), Take out of service, Remove; **Hosted budget**; **Check every account** (asks each balance too); **Type it** / **Change** on a card's balance (providers.md §7.1, §10, §12) |
| **Rented capacity** | (A pool with no account is pointed to Providers; one with an account but renting not set up shows only **Set up renting** — the two ceilings, the engine, and the models that engine has a build of, D172.) In five tabs, each at `#rented/<tab>` (D102): **Hosts** — provider account, what is rented and burning against the limits, Prepare a host, the rented and parked hosts; **Engine & models**; **Finding machines** — the search beside the live market; **Scaling** — limits and allocation; **Tear-down**. Only the last three read the market's settings, and none of them searches the market by itself: a search happens when the operator presses **Search the market** or **Try these**, and the page shows its last search, with its time, until the next (D120) — the provider counts every offer returned against a daily quota the pool needs for renting. The day's use of that quota is shown beside the search — each provider's, named, or that it has none — and in the Provider panel — counted from the pool's own searches, the provider's own figure once it refuses (D121). A page loaded from an older release reloads itself when the status reports a newer one, or, while the operator is in the middle of something, says so with a Reload button (D122). Across them: provider account (credential valid, credit left); offer policy, bid strategy, tear-down settings — each field beside its default and a one-line reason. Rented and **parked** hosts with their running and storage cost | Edit with **live market preview** — both rental kinds listed, labelled, each row with its own **Rent**. **Prepare a host**, as a bid, on demand, or as configured. Restart or destroy a parked host  **How capacity is decided** is edited here too (D74): the allocation mode, the ramp's settings and per-host worker adjustment, saved through the same file-plan-retype path as the offer search. |
| **Rented capacity → what the pool looks for** | **How the pool rents** — `interruptible`, `on_demand` or `cheaper` — chosen first, because it decides which listings are searched at all and an offer in a listing the pool never asks for cannot appear among the rejections below (D80). Then every offer-policy and bidding parameter, editable in place. **Try these** runs the real pipeline against the live market with the unsaved values and saves nothing; **Save** writes them into the configuration file *in place* — comments, ordering and flow style untouched — then validates, plans, and applies, with loosening retyped as anywhere else | `PATCH /pool/config/rented` |
| **Rented capacity → Profiles** | What each machine the pool rents holds (D111, D112); for an engine that can, how many cards each model is split across, the needs then per card of a group (D114). The engine rented machines run, then each **model profile** as a card: a rent tick, its name, whether a machine holds **one model** (choosing another replaces it) or **several** (a chosen few), and its rows — each a **model and the variant** a machine fetches for it, with precision and size — and the least card and disk that needs. **Every picker here chooses a model and its variant together, in one click**: *In the pool* lists the variants the pool already has that rented machines can run, with no search; a search lists every model on the model hub (or in Ollama's library, for an Ollama pool) with **its variants side by side** — the original and its quantisations — filtered by parameters, size, card fit, precision and kind. A variant new to the pool is added under the pool's name for that model, or a name typed beside it. **Machine setup** — images, engine options, start command — is folded below. **Saved as one change** with the engine, since a variant is one engine's; a variant is written into the catalog after those there, only if a profile uses it | `PATCH /pool/config/engine` `{profiles, rent, add}` · `GET /pool/models/search` · `GET /pool/models/size` |
| **Rented capacity → finding machines: the next host** | What the next machine would be bought as, what that needs, and — where it is more than the minimums set here — the card, disk and whole groups of cards the search asks for because of it (D111, D114) | — (`GET /pool/market/preview` `next_host`) |
| **Workloads** | The running workloads — state, hosts ready, latency p95 against the target, spend against the cap, hours left — each with Extend…, New key and End; a form that plans a new one — a row per model, with "Add model" and, for several, a placement (D118) — (the plan shows the start, the first host, the time to serve and the budget, typed or proposed; for several models, both placements priced, the one kept and each group's split) before Create is allowed, a proposed budget typed again to accept it, and the key shown once with copy buttons until put away (D115) | `/pool/workloads…` |
| **Workloads → serving plan** (D203, D204; **not built yet**) | Beside the form, a **plan builder** in the form's own words ("Answers at once", "Machines"): four toggles — *cover start-up and replacements with…*, *when full, also use…*, *keep some requests off hosted APIs…*, *at certain hours…* — with rules (and covering scale-ups) under **Advanced** and a **JSON** tab that edits both ways. Beside them, the plan's **readback**, **timeline**, the conditions that can change it, the most it can spend, and every refusal in words, as it is edited. **Code** tabs: the SDK for a program, the command line and HTTP for an operator, filled from a template with the validated plan, marking what a program's key cannot send. A running workload's row has **Change serving plan…**: current and proposed side by side, "takes effect within seconds; today's spend still counts". Its detail has the **Now** line (workloads.md §14.6). A **chat** (where `console.plan_assistant.model` names a model; otherwise a line saying how to turn it on): a description becomes the plan's JSON, shown as a change to the builder with **Undo**; "Asking <model> on this pool…" with **Cancel** while it waits; an answer still invalid after two tries opens in the JSON tab with its refusals marked. It never creates | Create (the builder's, with the budget typed again); Change serving plan; Copy code | `POST /pool/workloads/plan`, `POST /pool/workloads/{name}/policy`, `GET /pool/schema/serving-plan`, `POST /pool/workloads/plan-assistant` |
| **Workloads → apps on the shared pool** | Each named app key's app (D193): its keys by id and date — never a key — and whether any is usable | **Make key** for a new name, or **Add a key** to one there (how a key is rotated), the key and the pool's URL shown once with copy buttons until put away; **Revoke** one key, or **Revoke…** every key of an app | `GET/POST /pool/apps`, `DELETE /pool/apps/{app}[/keys/{key_id}]` |
| **Dashboards** | The stats kept (D194, D197), in four tabs at `#dashboards/<tab>`: **Apps** — each workload, each named app key, the configured key: requests, errors, p50 and p95, tokens, hosted and rented cost; **Hosts & cost** — each machine and hosted API: requests, errors, p95, busy worker-hours, cost, per hour, per 1 000 tokens out and per request; **Providers** — own machines, each rented account, each hosted API: spend over time, requests, errors, refusals for money; **Models**. Over 24 h or 7 d by the hour, 30 d by the day: total tiles, line charts drawn by the page as SVG — titled, with a legend, a tooltip per time slot and the same numbers as a table beneath — and a table per tab. At most six series a chart, the rest as "others"; a series keeps its colour whatever else is shown | Choose the range; Refresh | `GET /pool/stats` |
| **Spending** | What hosted APIs and rented bursts cost, today and this month (UTC days and months, as every budget is counted): totals; each budget — the shared workload's `pool.bursts.hosted`, each workload's hosted and rented burst — with its spend against its day's and month's limits; hosted answers per workload and provider, counted and priced. Hosted answers are charged from their own token counts at the catalog's prices; rented bursts from the spend ledger (D165, D166) | `GET /pool/hosted` | Every hosted API: format, address, its key's source (never the key), state, models offered with prices, spend; the presets; the hosted budget (providers.md §10) |
| `POST /pool/hosted/test` | Ask a provider's `/models` with a typed key — or a saved API with its own — which spends nothing |
| `POST /pool/hosted`, `PATCH /pool/hosted/{id}`, `DELETE /pool/hosted/{id}` | Add (host, pinned builds, model set; key tested and kept), change (workers, service, caching, models and prices), remove (typed again) — through validate → plan → apply |
| `PUT /pool/hosted/{id}/credential`, `DELETE …/credential` | Type in or replace a hosted API's key (tested first), or remove it; takes effect without a restart |
| `PATCH /pool/hosted-budget` | The shared workload's hosted burst: `after_s`, `per_request`, `per_day`, `per_month` |
| `GET /pool/spending` |
| **Out of money** | A hosted API whose balance reached $0 (D188) is marked the same way, said as such. A hosted API whose provider refused for money is marked on the Hosts and Spending screens with what the provider said and since when, and out of routing; **Money is back**, once the account is topped up, routes to it again (D164) | `POST /pool/hosts/{id}/funds-restored` |
| **Hosts → out of service, remove** | Per configured host: **Take out of service** sets `disabled` in the file — the host stays, stops receiving requests, finishes those on it, and **Return to service** brings it back. **Remove…** deletes its entry after showing the plan and asking for the host's id to be typed. Taking out of service is never refused; when it leaves models with no host in service, the plan names them, and requests for them are refused until the host returns (D113). Removal goes through the file's rules: removing the only host serving a model is refused with the reason, before anything is asked. Rented hosts are released from Rented capacity instead (D99) | `PATCH /pool/config/hosts/{id}` `{disabled}` · `DELETE /pool/config/hosts/{id}` `{confirm: id}` |
| **One host** | Opened from any host id. Its state and stage in words (*starting · fetching models: 1 of 3 on disk; downloading gemma4:26b — 12.1 of 18.6 GB · every model is on disk; the engine is starting on them · loading into memory · ready*), what the provider says about the machine, tunnel, cost so far, **one state per model** — *not here yet · downloading* (with its bar and rate) *· on disk · loading · loaded · failed* (with the engine's reason) — derived from the pool's own downloads, the agent's report and the engine's list together, and that host's own slice of the decision log. An engine that is started once its weights have landed is shown as *not started yet* while they land, not as a fault. Follows the host while open | — (`GET /pool/hosts/{id}`, `gpm host show <id>`) |
| **Models** | The pool's model set; the catalog of logical names and variants; per-host matrix of which build is served, its runtime class, whether it enforces schemas; capacity profiles and calibration entries | Edit the set and the catalog (changing the set re-prepares hosts — shown in plan) |
| **Models → Add model** (S15, D178–D182) | A trained model: its name; the model — a hub file, an upload, or a container image by digest; for ONNX, its inputs and outputs read from the file, each given how it arrives (`tensor`, `image`, `features`) and what it means (`tensor`, `scores` with labels, `vector`, `value`); for an image, its port and one example input; memory, card, largest batch; where it may run. Registry logins and the hub's tokens; whether containers are enabled | **Test** — the declarations held to the file — then **Add**; turning containers on asks for the pool's name typed again |
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
| `POST /pool/hosts/{id}/engine/upgrade`, `POST /pool/hosts/{id}/engine/rollback` | For a host whose agent installs its engine (D153): switch to the staged version now, through the same drain and check; or go back to the previous version. The host id typed again (`confirm`) |
| `POST /pool/hosts/{id}/funds-restored` | A hosted API's money is back (D164): the router routes to it again. 404 for a host that is not a hosted API; the event names what the provider had said |
| `GET /pool/balances` | Every provider account's balance: source (`provider`, `typed`, `unknown`), amount and its moment, spent here since, the estimate left, whether the provider can be read, and the last read's words (D188) |
| `POST /pool/balances/read` | Ask the providers that state a balance now — `{account}` for one — each at most once a minute; spends nothing |
| `PUT /pool/balances/{hosted\|rented}/{id}`, `DELETE …` | `{usd}`: what is left there now, typed, counted down from this moment; or forget it. Above $0 on a hosted API it is also the operator's word that its money is back |
| `GET /pool/spending` | Hosted spend today and this month — per workload and provider, with each hosted budget against its limits — each workload's rented burst against its budget, and each hosted API's out-of-money mark (D165, D166) |
| `GET /pool/apps`, `POST /pool/apps` `{app}`, `DELETE /pool/apps/{app}`, `DELETE /pool/apps/{app}/keys/{key_id}` | Named app keys for the shared pool (D193): each app and its keys, never a key or a hash; a new key for an app, the app made if new, answered with `connection: {base_url, api_key}` **once**; revoke all of an app's keys or one. A named key is an app key: refused here like the configured one |
| `GET /pool/stats` | The stats kept (D194, D196): `range` (`24h`, `7d`, `30d` …) or `from`/`to`; `bucket` `hour` (31 days at most) or `day` (UTC); `by` `app`, `model`, `host`, `provider` or `kind`; narrowed by `workload` (`shared`), `app`, `model`, `host`, `provider`, `kind`. Each series and the whole, per bucket and in total: requests, answered, errors, refused, failed, cancelled, 4xx, 5xx, refused for money, tokens in and out, p50, p95, mean and largest latency, queue wait, worker time, cost hosted, rented and both; `notes` where rented cost is per machine and so left out or a line of its own; the retention in force. New rows are folded first |
| `POST /pool/down` | Destroy all rented hosts now, verified |
| `GET /pool/market/preview` | Run the offer pipeline read-only with supplied settings — the market is searched only with `search=true`; without it, the settings and the next host's needs only (D120, D122). `POST` likewise. `kinds=both` lists bid and on-demand offers together whatever `rented.mode` is; every row carries its `offer_id` and `kind` |
| `POST /pool/keys`, `DELETE /pool/keys/{id}` | Create, rotate and revoke app and admin keys |
| `GET /pool/builds?model=&engine=&search=&fresh=` | An engine's builds of a model on the model hub, sorted (D100); from the directory's cache unless stale or `fresh` |
| `GET /pool/directory?q=`, `POST /pool/directory/refresh` | The model directory from the cache; start a refresh in the background (D101) |
| `POST /pool/config/models` | Add models from the directory: `add: [{name, builds: {engine: build}, rent_for}]`, `engine_options` — one write, validated, planned, confirmed (D101) |
| `POST /pool/workloads/plan` | What creating a workload would do — its build, workers per host at its latency (measured or not), hosts at the start, the first host and its kind, minutes to serve, the budget typed or derived, and whether the pool's caps allow it. Opens and rents nothing (D115). With `serve` (D203; not built yet): the readback in plain words, the timeline over the lease's hours, the conditions that can change it, the most it can spend, a warning per request option that reaches different sources, and the budget at the dearest state |
| `POST /pool/workloads` | Create it: `{name, model, latency_s, parallel, hours, max_spend?, profile?, kind?}` — or, for several models (D118), `models: [{model, latency_s, parallel}, …]` and `placement?` (`auto`, `together`, `apart`) in place of the first three; a budget not typed must come back as `confirm_max_spend`. Opens its lease, mints its key and answers with `connection: {base_url, api_key}` — **the only time the key is shown** |
| `GET /pool/schema/serving-plan` | The serving plan's JSON Schema (D203; not built yet) — the same one the package carries |
| `POST /pool/workloads/plan-assistant`, `GET /pool/workloads/plan-assistant/{id}`, `DELETE …/{id}` | The console's chat (D204; not built yet): `{description, plan?}` → an id; the answer — a plan, or the model's words with their refusals where none validated after two tries — read back by id, or cancelled. One at a time. Written to `assistant_requests`; the router's watcher claims it (one router only), sends it through its own dispatch to `console.plan_assistant.model` with `keep_on: owned`, app `console`, a `max_tokens` bound and a timeout, writes the answer and deletes the row once read. `404` where no model is named; a hosted-only model, or one that cannot give structured output, is refused when it is set |
| `GET /pool/workloads`, `GET /pool/workloads/{name}` | Each workload: state, hosts, spend against its cap, hours left, answers served with their p95 against the target, borrowed and refused counts, key ids — never a key or a hash |
| `POST /pool/workloads/{name}/extend` | Hours added and/or a new budget; a raise carries `confirm`, the value typed again (D49) |
| `POST /pool/workloads/{name}/end` | Its lease closes; its hosts drain and go; its key reaches only a `workload_ended` answer |
| `POST /pool/workloads/{name}/keys` | A new key, shown once; the old ones work for `workloads.rotation_grace_minutes` |
| `PUT /pool/config/profiles` | The model profiles and which the pool rents as: `profiles: {name: {model: build}}`, `rent: [name]`, `split: {name: cards}` (optional, D114 — not sent, the file's split is kept for the profiles still named), `add: [{name, builds: [{tag, engine, size_gb}]}]` — one write, validated, planned, confirmed (D111). The same fields ride on `PATCH /pool/config/engine` in place of `placement` |
| `GET /pool/models/search?q=` | Any model whose name holds `q`: the model hub asked live, with no credential, as models each with its variants — the original and its quantisations, each with precision, the cards it runs on and, where the hub's tally is exact, its size — and Ollama's library from the directory's cache, saying whether it has ever been read (D111, D112) |
| `POST /pool/models/uploads` | Stream an ONNX file; answers its tag, `upload/<sha256>`, and size; bounded by `models.max_upload_mb` (D182) |
| `POST /pool/models/inspect` | An ONNX file's inputs and outputs (names, element types, shapes), read from a hub file or an upload, for Add model |
| `POST /pool/models/test`, `POST /pool/models/trained` | Hold a build's declarations to its file before saving, spending nothing (each machine also runs one made-up input as the build loads); add the build — catalog entry, model set, the hosts that hold it — in one validated, planned write (D178) |
| `GET`/`PUT /pool/containers/registry-logins`, `DELETE …/{registry}` | A private registry's login (`registry`, `username`, `password`), kept by the supervisor and handed only to a provider's own create call for a rented container host; listed without passwords, never answered back (D183) |
| `PUT /pool/models/hub-credential`, `DELETE …` | The hub's write token for relayed uploads, and the read-only token rented machines are given (D182) |
| `PATCH /pool/containers` | `{enabled, confirm}` — the pool's name typed again to turn it on; recorded as an event (D180) |
| `GET /pool/models/size?repo=` | One hub repository's weights, exactly, from its file listing; read once, then kept (D112) |

CLI verbs map one-to-one: `gpm status | plan | serve | stop [--release] | restart | lease … |
host test | host prepare | host drain | host release | host park | host funds-restored | spending |
balance [list | read | set --usd N | clear] [hosted:<id> | rented:<connection>] |
down --all | key … | app create/list/revoke | stats`.

## 5. How it is built

A static page (HTML and JavaScript, no build step) served by the supervisor process at `/ui`,
with live updates over server-sent events. It has no server-side session: the admin key is sent
as a header on each call, and kept in the tab's `sessionStorage` — a reload keeps it, closing the
tab forgets it — or, when the operator ticks **Keep me signed in on this browser**, in
`localStorage` until **Sign out** (D137). The page is served with a Content-Security-Policy that
lets no script run on it but its own. A screen drawn again — every status update, every action —
keeps the operator's scroll position — the page's, and that of any box that scrolls on its own, like
the live feed — and shows no “Loading…” in between.
