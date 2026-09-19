# GPM — Status and Handoff

> The single place that says where the project stands. **Update it at the end of every working
> session** (current state, open decisions, session log) so the next session can resume without
> re-reading everything.

## Current state — 2026-09-17

**Phases 1–3 are complete** — the router and SDK, the supervisor with leases and renting
(proven on a real marketplace for $0.008), and the operator console. **Phase 4 (Release) is
part-done**: the identifiers are renamed to `gpm`, the licence is Apache-2.0, and the project
hygiene files and CI are in. **Nothing is committed yet** — the docs and code are working-tree files waiting for the owner to
review and commit.

The design was drafted, extended through ten rounds of owner requirements, put through a
critical architecture review (18 findings, every one decided), and then split from one
1,400-line file into a generic specification plus a first-adopter guide — all on 2026-09-17, in
the repository of the first adopter (Aletheia, `~/workspace/Aletheia`). It was moved here the
same day, and the first code was written the same day.

### What runs today

A router and a client SDK over **statically configured hosts**. Two packages in one `uv`
workspace: `client/` (`gpm_client`, one dependency: `httpx`) and `server/` (`gpm_server`,
the router, the Ollama engine adapter and the `pool` command).

| Built | Where |
|---|---|
| Config: hosts, transports, catalog, secrets from the environment, every rule refused at load | `server/src/gpm_server/config.py` |
| All three transports: `http`, `https`, and `tunnel` — a supervised SSH local forward that reconnects on the same local port | `server/src/gpm_server/transports/` |
| Engine plug-in interface (phase-1 subset) + the Ollama adapter | `server/src/gpm_server/engines/` |
| Routing: strict tiers, least-loaded within a tier, FCFS queue, failover-once | `server/src/gpm_server/router/dispatch.py` |
| Request path: passthrough, model rewrite only, pool dialect, cancel-on-disconnect | `server/src/gpm_server/router/app.py` |
| Readiness probe: whole-model-set-resident check, in-process for now | `server/src/gpm_server/router/probe.py` |
| Request log in SQLite (WAL), written off the request path, no prompt or completion text | `server/src/gpm_server/db.py` |
| SDK: `pool_transport()` / `PoolClient`, wait-and-retry by default, typed errors | `client/src/gpm_client/` |
| `gpm serve` (router here, supervisor in its own process) and `gpm status` | `server/src/gpm_server/cli.py` |
| **Supervisor process**: single-instance lock, control loop, probing, tunnels, publishing the host table | `server/src/gpm_server/supervisor/service.py` |
| **Provider plug-in interface** with declared capabilities, and the scriptable **fake provider** | `server/src/gpm_server/providers/` |
| **Leases** with mandatory dollar caps and tighten-only changes; decisions and a spend ledger | `server/src/gpm_server/ledger.py` |
| **Strategies** as pure functions returning their reasons: rent, offer filter/rank, bid, eviction, tear-down | `server/src/gpm_server/strategies.py` |
| **Renting**: overflow-driven, one at a time, caps re-checked after every strategy, spend reconciled against provider-reported charges, orphan sweep, verified destroy | `server/src/gpm_server/supervisor/renting.py` |

**Tests: 231 pass with no GPU and no cloud account** (`uv run pytest`, ~50 s, against a fake
engine and a fake provider), and **9 more against a real local Ollama** (`uv run pytest -m integration`), which is
where the phase-1 exit criterion is actually demonstrated: streaming, tool calls, structured
output and cancel-on-disconnect behaving the same through the router as direct to the engine,
including a byte-for-byte comparison wherever the engine is reproducible.

**One gap in the tunnel's testing:** the supervisor is exercised against a stand-in for
`ssh -N -L` — start, listen, carry traffic, die, come back on the same port — but the `ssh`
command itself is only asserted option by option. A first run against a real SSH host is worth
doing before this is relied on; it needs an SSH server, so it is not something the suite can
bring up for itself.

### Phase 2, stage by stage

| Stage | State |
|---|---|
| **2a** Two processes, one database — supervisor owns probing, tunnels and the host table; the router reads it and keeps serving when the supervisor dies | **done** |
| **2b** Provider interface, fake provider, leases, strategies, renting, money | **done** (core) |
| **2c** Dead-man timer (armed at creation, run and proven to fire), prepare-a-host, parking with reuse | **done** |
| **2d** Control API on the supervisor, admin key, hashed key files, CLI verbs, `gpm plan` | **done** |
| **2e** The interruption drill as one offline suite | **done** |
| **2f** One live Vast.ai run, capped under $2 | **done** (2026-09-19): bid → instance → timer armed and the pool's key installed by the start-up script → tunnel → model set pulled and pinned by the supervisor → `ready` → a request served through the router from the rented host → **a real, unforced eviction recovered unattended** → `gpm down --all` → account empty. Total cost of the run, both attempts: **$0.008** |

### What the read-only live check found (2026-09-19)

- **Endpoint shapes confirmed** for account and offer search. Credit on the account: $4.18.
- **A wrong assumption caught before any bid**: on bid-type offers, `dph_base` is the floor
  again, not the on-demand price. Every "would bid" came out at 0.8 × floor — below the floor,
  unwinnable. Fixed by joining the on-demand listing by machine id (verified fact recorded), and
  it produced a rule: a bid the ceilings push below the floor is a refusal, not a bid (D34).
- With that fixed, the market through a $0.25 ceiling: 100 offers seen, 8 pass. Cheapest
  acceptable ~$0.18/h on a 64 GB mining card; a known-good RTX 4090 at $0.25/h (clamped by the
  ceiling from $0.26).
- **Still unverified, needs an instance**: whether per-instance charges are reported. If not,
  the provider declares `reports_charges: false` and the cap margin widens to 25% by itself.

### Restarts are safe now

A restarted supervisor lists the provider's instances under the pool's label, **takes back the
ones the host table says it intended** (bid, machine, start time — so spend keeps counting from
the real start — lease, hold, park state), re-opens their tunnels, re-verifies readiness rather
than assuming it, drops rows for instances the provider no longer has, and only then sweeps.
`gpm restart` therefore keeps rented hosts, as spec §1.1 says it must. Covered by
`tests/test_adoption.py`, including the case where the lease closed while no supervisor ran.

### Phase 3 — the console

| Stage | State |
|---|---|
| **3a** The page at `/ui`, live over server-sent events; Overview, Hosts, Models, Leases, Decisions | **done** |
| **3b** Configuration: validate → plan → apply, versions and rollback, reload in place | **done** |
| **3c** Rented capacity: provider, limits, live market preview with unsaved values, Prepare a host | **done** |
| **3d** Test connection (including the concurrency check), CLI parity verbs | **done** |
| **3e** Exit criterion | **done** — see below |

The roadmap's exit criterion, demonstrated: **every console action has a CLI verb** making the
same call (`gpm config get|validate|plan|apply|history|rollback`, `gpm host-test`, plus the
existing lease/host/market/events/down verbs); **a mistyped ceiling is caught by plan** — live,
0.25 typed as 25 was reported as "the bid ceiling goes from $0.250 to $25.000/h", flagged as
needing the value retyped, and `gpm config apply` refused it without `--yes`; **the app key is
refused** by every control endpoint the console calls.

Also proven live against the local Ollama: applying a model-set change made the running pool
follow it (host went `ready`, the router served a request), and rollback restored it.

### Phase 4 — Release

| Item | State |
|---|---|
| Identifier rename to `gpm` (CLI, packages, `GPM_*`, `X-GPM-*`, `~/.config/gpm/`, label prefix) | **done** (D36) |
| Licence: Apache-2.0, `LICENSE` + `NOTICE` (copyright ClearViews), DCO in `CONTRIBUTING.md`, dependency licences checked | **done** (D37) |
| `SECURITY.md`, `CONTRIBUTING.md` | **done** |
| CI: tests on 3.11–3.13, gitleaks, `pip-audit`, and a job that fails if anything able to reach a real provider enters the default suite | **done** |
| Secret scan before the first commit | **done** — only test fixtures matched; the account key is in no tracked file |
| Two packages, `gpm-client` (one dependency) and `gpm-server` | **done** |
| Code of conduct, issue/PR templates, lint and type-check in CI | to do |
| Quick start over hosts you already own, renting nothing | **done** — `docs/quickstart.md` |
| The rented-provider guide; plug-in author guides | to do |
| The three adopter-internal files | **deferred by the owner**: commit now, decide before going public — which then means publishing from a fresh history |
| Threat model re-read line by line against the implementation | to do |

### Not built yet, on purpose

- The right-hand column of the roadmap's §1 (post-v1), and strategy replay.
- Worker counts come from configuration only. Capacity profiles, the memory-ceiling formula and
  evidence-based step-down need hardware inspection, which arrives with test-connection.
- `enforces_schema` is declared by the operator; nothing measures it yet. The smoke test that
  would (engine interface `smoke_test`) is phase 2.

## What GPM is, in five lines

1. Apps see **one URL and one API key**; they never know where the GPU runs.
2. A **pool** holds hosts of three kinds — local, fixed-remote, rented on a bidding marketplace —
   reached by tunnel, http or https. Capacity is **workers per host**; every host keeps the
   pool's **whole model set** loaded.
3. Routing is **strict priority**: local, then fixed, then rented; first come, first served.
4. A **lease** with a mandatory dollar cap is the only thing that can spend. A **supervisor**
   process rents, bids, recovers and tears down; a separate **router** process serves requests
   and keeps serving if the supervisor dies. A **dead-man timer** sits on every rented host.
5. An **SDK** makes apps wait-and-retry by default; an **operator console** configures hosts and
   bidding, prepares a rented host with models on request, and shows every decision with its numbers.

Read [docs/overview.md](docs/overview.md) first, then [docs/decisions.md](docs/decisions.md).

## Next actions, in order

1. **Owner: review the docs and the code, and make the first commit** (see "Before anything here
   becomes public" below — the repo is private, so committing everything is safe for now).
3. **Owner decisions still open** — the first two block phase 4, not phase 1:
   - **Name of the identifiers.** The repository is `GPM`, but the docs still use working-title
     identifiers: CLI `pool`, SDK package `gpm_client`, env prefix `GPM_*`, header
     prefix `X-GPM-*`, config dir `~/.config/gpm/`. Decide the final set (e.g. CLI
     `gpm`, `GPM_*`, `X-GPM-*`) and rename once. **The code now uses the working titles too**,
     so the rename is a mechanical pass over two packages rather than over docs alone — still
     small, but no longer free. See [docs/release-checklist.md](docs/release-checklist.md) §1.2.
   - **Licence.** Apache-2.0 is recommended; reasoning in the release checklist §1.3.
   - The questions in [docs/roadmap.md](docs/roadmap.md) §4 — chiefly default lease caps, and
     whether the supervisor runs permanently or only while a lease is open.
4. **Phase 2 — Supervise** ([docs/roadmap.md](docs/roadmap.md) §2): the supervisor process and
   the provider plug-in interface, built against the **fake provider first**. Nothing rents
   until this phase, and no test needs a cloud account.
5. Phases 3–4 as in the roadmap: Console, Release.

## What is decided (index — the detail is in [docs/decisions.md](docs/decisions.md))

| # | Decision |
|---|---|
| D1 | Pool and app are decoupled; the only contract is one URL, one key, the engine API plus a small pool dialect |
| D2 | Three host kinds; three transports (tunnel / http / https) |
| D3 | Client SDK; default wait-and-retry when no capacity; max wait configurable |
| D4 | Strict routing priority: local → fixed-remote → rented |
| D5 | Session affinity off by default |
| D6 | Logical model names resolve to the build suited to the host |
| D7 | Renting / bidding / holding / tear-down are named, configurable strategies |
| D8 | Operator console, as one more client of the control API |
| D9 | Capacity is workers per host, by hardware capability |
| D10 | A lease is the only thing that can spend; dollar caps mandatory |
| D11 | **Public framework**; Aletheia is only the first adopter |
| D12 | The first adopter's current GPU scripts are not design inputs |
| D13 | Mixed model builds within a session: record and measure, don't prevent |
| D14 | Two processes + SQLite; explicit `stop --release` vs `restart`; no release-on-exit |
| D15 | Time budget: cancel on disconnect, one capacity-retry layer, SDK-owned timeouts, deadlines |
| D16 | The pool is the isolation unit with a required API key; separate app and admin keys; multi-pool later |
| D17 | Host data-trust levels deferred (documented residual risk) |
| D18 | v1 cut line; owner musts: all three host kinds, full console, dead-man timer |
| D19 | Dead-man timer uses the provider's instance-scoped credential (verified available) |
| D20 | Build order: 1 Route, 2 Supervise, 3 Console, 4 Release |
| D21 | Prepare a rented host on request (join / park / destroy); parked hosts used first |
| D22 | Catalog-only model resolution; contract stated as engine API + versioned dialect |
| D23 | Every host holds the pool's whole model set; queue is first come, first served |
| D24 | Spend reconciled against provider-reported charges; providers driven by HTTP API, not CLI |
| D25 | Build the router and supervisor rather than adopt (Olla, LiteLLM, SkyPilot, dstack evaluated) |
| D26 | Docs split generic / adopter; provider-engine-strategy plug-ins; capability-keyed variants; threat model; release checklist |
| D27 | Repository created as **GPM** ("GPU Hosts Pool Management"), private, 2026-09-17; identifier naming still open |
| D28 | Three dialect items added while building phase 1: `503 no_eligible_host`, `504 deadline_exceeded`, `X-GPM-Wait-S` |
| D29 | `enforces_schema` is three-valued — true / false / unstated — so uncatalogued names keep passing through |
| D30 | The `tunnel` transport drives the `ssh` client as a supervised child process, not an in-process SSH library |
| D31 | The supervisor runs permanently; it rents nothing without a lease |
| D32 | Conservative shipped defaults; a lease's dollar cap is mandatory with no default |
| D33 | Offers are filtered by the pool, not in the provider's query, so every rejection keeps its reason |
| D34 | A bid the ceilings push below the floor is a refusal, not a bid |
| D35 | Configuration is versioned by content hash; stale writes refused; reload in place, except the listener |
| D36 | Identifiers: CLI `gpm`, packages `gpm-client` / `gpm-server`, `GPM_*`, `X-GPM-*`, label `gpm/<pool>/<host>` |
| D37 | Apache-2.0, copyright held by a company, DCO rather than a CLA |

## Before anything here becomes public

The repository is **private**, so everything can be committed now. Before it is made public,
decide what happens to three adopter-internal items — they describe another private codebase's
files, measurements, prices and plans (they contain **no credentials**; scanned 2026-09-17):

| Path | Contains | Suggested |
|---|---|---|
| `docs/adopters/aletheia/` | The first adopter's code analysis, file-by-file change list, real configuration values, the earlier migration workplan | Move back into the adopter's own repository, leaving a generic "writing an adopter guide" page here |
| `docs/archive/design-2026-09-17-pre-split.md` | The original design, written in the adopter's terms | Remove, or keep only its framework comparison (§4.4) as a standalone page with its sources |
| `docs/architecture-review.md` | The review; argues several findings from the adopter's scripts | Keep privately as history, or publish after a pass to genericise it — [docs/decisions.md](docs/decisions.md) already carries every outcome |

If those are removed after being committed, they remain in git history — so if there is any
doubt, decide **before the first commit**, or plan to publish from a fresh history.

## Working agreements carried over

- **Nothing that spends money, calls a paid API, or starts a long-running job runs without the
  owner's explicit approval.** Validate on the smallest possible case first.
- **Keep changes to the requested scope**; ask before widening.
- New decisions are recorded in [docs/decisions.md](docs/decisions.md) (numbered, with why and
  what was rejected) **and** in the relevant spec file. Adopter-specific content never goes in
  `docs/spec/`.
- Design arguments are made on generic grounds — what any client, provider and engine need —
  not from one adopter's scripts (D12).
- Use plain, descriptive names for components; no internal jargon or staging tags.
- The owner tends to prefer the simpler mechanism when one exists (a pool API key rather than
  per-request client ids; all models resident rather than policing swaps; record-and-measure
  rather than prevent). Offer the simple option first.

## Where the first adopter's side lives

In `~/workspace/Aletheia`: backlog entries `GPU-POOL-01` and `GPU-CLOUD-01` in
`platform-design/unified/TASKS.md`; the full day's session log in
`platform-design/unified/progress.md`; the single-host predecessor still in use there is
`prototypes/per-turn-behavioral-labeling/vast_provision.py`. A stub at
`~/workspace/Aletheia/gpm/README.md` points here.

## Session log

| Date | What happened |
|---|---|
| 2026-09-17 | Design drafted in the Aletheia repo: host kinds, router, supervisor, leases. Extended by owner requirements: pool/app decoupling and tunnel/http/https transports; client SDK with default wait-and-retry; routing priority; logical model names (Apple-optimised vs standard builds); rent / bid / hold / tear-down strategies; operator console; workers per host. |
| 2026-09-17 | Architecture review written (18 findings). Decided one by one with the owner: F1 record-and-measure; F2 two processes; F3 time budget; F4 pool API key as the isolation unit; F5 deferred; F6 v1 cut line with three owner musts; F7 phase re-order; F9 catalog-only resolution; F11 whole model set resident + FCFS; F13 spend reconciliation + provider API. Owner added: public-framework intent; current scripts are not design inputs; prepare-a-host from the console. Verified: the first provider's per-instance restricted key makes the dead-man timer possible without the account key. |
| 2026-09-17 | F14–F18 delegated and done: docs split into generic core + adopter guide; plug-in interfaces; capability-keyed variants; threat model; release checklist. Owner created this repository (GPM, private); docs moved here; this file, the README and `CLAUDE.md` written for handoff. **Not committed.** |
| 2026-09-17 | **Phase 1 built.** `uv` workspace with `client/` and `server/`; config and its refusals; engine interface + Ollama adapter; dispatch with tiers, FCFS queue and failover-once; the request path with passthrough, cancel-on-disconnect and the time budget; readiness probe; SQLite request log; the SDK's transport and `PoolClient`; `gpm serve` / `gpm status`. 104 fake-engine tests plus 9 against a real local Ollama, all green; `gpm serve` driven by hand against that Ollama as well. Two decisions the build forced: **D28** (three dialect additions) and **D29** (three-valued `enforces_schema`) — both recorded in the decision log and in the spec. **Still not committed.** |
| 2026-09-19 | **Phase 4 (Release) started.** Identifiers renamed to `gpm` throughout — 36 files, zero occurrences of the old ones left, `gpm --help` and `import gpm_client` both verified. Two things were checked rather than assumed: the PyPI name `gpm` is taken by an abandoned 2019 package (so the distributions are `gpm-client` / `gpm-server`, both free), and `gpm` is also the Linux mouse daemon. Rented instances now carry a `gpm/<pool>/` label, because the live run found another tool renting on the same account. Apache-2.0 added with `NOTICE`, `SECURITY.md`, `CONTRIBUTING.md` with a DCO, and CI that runs the suite on three Pythons, scans for secrets, audits dependencies and **fails if anything able to reach a real provider enters the default suite**. All 15 runtime dependencies checked as licence-compatible. A secret scan before the first commit found only test fixtures. **D36** and **D37** recorded. 306 tests green after the rename. |
| 2026-09-19 | **Phase 3 (Console) built.** A static page at `/ui` served by the supervisor — no framework, no build step — over the same control API the CLI uses: Overview with hosts by tier and lease burn-down, Hosts with test-connection, Rented capacity with the live market preview and Prepare a host, Models, Leases, Decisions with the numbers behind each one, and Configuration with validate → plan → apply, history and rollback. Live updates over server-sent events read with `fetch`, because the key travels as a header and never in a URL; the key is held in page memory and the page is asserted to touch no storage and no cookie. New underneath it: `configplan.py` (the plan diff as a pure function, and the version store) and `hostcheck.py` (test connection with the concurrency check). **D35** recorded. 303 tests green, and the whole flow driven live against the local Ollama. |
| 2026-09-19 | **`reports_charges` earned for Vast.ai.** Found the real charges endpoint (`/api/v0/charges/`, learned from the official CLI's source) and read the run's rows: the pool's RTX 3090 host reported $0.007 against the pool's estimate of $0.008. The plug-in now sums an instance's daily rows, fetched once per pass for all hosts and paginated; an instance with no row yet reports *nothing*, not zero, so the cap margin narrows to 10% only once a real figure has arrived — and since caps take the higher of estimate and report, a lagging daily row can never loosen one. Verified fact updated. |
| 2026-09-19 | **Re-adoption on restart built.** The gap the live run exposed is closed: rented rows now carry everything a successor needs (no secret, no raw payload), `Supervisor.start()` adopts before its first sweep, gone instances are dropped with a `host_gone` event, and an adopted host under a lease that closed meanwhile is released by the first pass. A provider can be injected into a supervisor so a test can restart one against the same market. Six tests. |
| 2026-09-19 | **Phase 2 live run complete — $0.008.** Second attempt on the fixed code: an RTX 3090 at the $0.10 ceiling; tunnel up on the pool's own key with no account keys registered; supervisor pulled and pinned the model set; `ready`; one request through the router answered from the rented host (`X-GPM-Host: rented-903fdc`). Then the market outbid us for real and the pool recovered unattended — destroy verified, RTX 5060 Ti rented at $0.098 — before `gpm down --all` ended it with the account empty. Found and fixed along the way: queue wait and latency were both measured after the engine's reply (a 46 s answer read as 46 s of wait and 0.8 ms of latency); the engine image has neither `ss` nor `netstat`, so the timer now reads `/proc/net/tcp`; the provider rate-limits per-instance fetches, now cached for a pass; and it declares no per-instance charge reporting, with the cap margin widening by itself whenever a provider never actually reports. Verified facts recorded. |
| 2026-09-19 | **First live attempt, aborted by a flapping bug — cost ≈ $0.** With the owner's go-ahead (cheapest host, $0.10 ceiling, $2 cap, 1 h): the bid took at once — an RTX A4500 at $0.079/h — and then the pool destroyed the host the pass after renting it, five times in ~2 minutes, each time "the overflow this host was rented for is gone". Cause: tear-down counted a host that was still *booting* as capacity, so 2 workers wanted against one 2-worker host in flight read as overflow zero; and scale-down had no waiting window at all, though the spec gives it 600 s precisely so capacity does not flap. Stopped with `gpm down --all` — which worked: 0 instances remain, credit unchanged. Also found first: the pinned image default did not exist on Docker Hub, and nothing in the supervisor pulled the model set onto a fresh host (only tests called it). All four fixed with regression tests on the live numbers. The drill had hidden the flap because its lease wanted 3 workers. |
| 2026-09-19 | **Operator path proven end to end, read-only.** A real `gpm supervise` with the control API up, driven by the CLI verbs against the live provider: account, market (with the histogram by filter), the app key refused, `plan` with no lease. Three bugs found by doing it rather than unit-testing it: every non-serving CLI verb crashed on a missing `--log-level`; a SIGTERM stop left the supervisor lock behind so an immediate restart was refused (now: SIGTERM shuts down cleanly, and a lock whose holder is a dead pid on this host is stale at once — both proven with real processes); and the fleet had no way to *reach* a real rented host, since only the fake hands over a public URL — rented hosts are now reached over a supervised tunnel by default, launch settings go into the instance environment, and an image-specific start command can follow the armed timer. `exclude_hardware` added to the offer policy after a mining card passed every numeric filter. 250+ tests green. |
| 2026-09-19 | **First contact with the real provider, read-only.** Account and market fetched with the owner's credential; nothing created. Found and fixed a plug-in bug the docs could not have shown (bid-row `dph_base` is the floor, not the on-demand price), which produced D34. Rejection histogram now buckets by filter. Live market preview and account check added as `gpm market` / `gpm account` and control endpoints. 239 tests green. **The capped run is next and waits on the owner's explicit go-ahead.** |
| 2026-09-17 | **Phase 2 finished offline.** Stages 2c–2e: the dead-man timer (generated on-host script, armed before anything else at creation, heartbeat each pass over ssh, and **actually executed in tests** — it fires when the pool goes silent and stays quiet while beaten, carrying only the instance-scoped credential); prepare-a-host as its own small lease; parking with parked-first reuse and a hard park limit; the control API on the supervisor behind the admin key, with the app key explicitly refused; hashed key files; the CLI verbs (`lease`, `host`, `key`, `plan`, `events`, `down --all`); and the interruption drill as one end-to-end suite. The Vast.ai plug-in is written and tested against stubbed HTTP — **it has never called the real API**. 231 tests green offline. **Still not committed.** |
| 2026-09-17 | **Phase 2 started.** Stage 2a: the two-process split — shared SQLite (host table, counters, leases, events, spend, supervisor lock), the supervisor taking over probing and tunnels, the router reading the published table and serving on from it when the supervisor dies (checked against real processes, not just tasks). Stage 2b: the provider interface with declared capabilities, the scriptable fake provider, leases with mandatory dollar caps, the five strategies as pure functions, and renting with every cap re-checked after a strategy returns. Two owner decisions recorded: **D31** (supervisor permanent) and **D32** (conservative defaults), closing roadmap questions 1 and 2. 170 tests green offline, 9 against the real Ollama. **Still not committed.** |
| 2026-09-17 | **Phase 1 finished**: the `tunnel` transport — `ssh -N -L` supervised by the router, host key pinned on first connection against a pool-owned known-hosts file, reconnect with back-off on the same local port so the URL the router dials never moves. 14 more tests, against a stand-in for `ssh`; the command itself asserted option by option. 118 green in total. **Still not committed.** |
