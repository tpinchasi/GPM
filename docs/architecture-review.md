# GPU Host Pool — Architecture Review

> Review of the single-file design as it stood on 2026-09-17 (≈1,070 lines, no code yet), with
> [the migration workplan](adopters/aletheia/gpu-cloud-migration-workplan.md) as background.
> **Every finding now has a decision** — see the note under each, and
> [decisions.md](decisions.md). Section numbers (§) throughout refer to the pre-split file, kept at
> [archive/design-2026-09-17-pre-split.md](archive/design-2026-09-17-pre-split.md); where a
> resolution note says "design.md §x", the content now lives in the split docs listed in
> [overview.md](overview.md).
> Reviewer's caveat: the review was written by the same assistant that drafted the design, in a
> deliberate second pass reading it as a critic. It is not an independent review; treat the
> findings as claims to check, not verdicts.

## Verdict

**The shape is right; the scope and a handful of seams are not.** The three-way separation — a
request path the apps talk to, a control loop that owns hosts and money, and a thin client that
hides waiting — is sound and should survive. So should the lease as the only thing that can
spend, and the decision to relocate `vast_provision.py`'s policy rather than rewrite it.

What needs to change before building:

1. One **correctness risk for the data this pool exists to produce** (F1) — *resolved
   2026-09-17 as record-and-measure; see the note under F1.*
2. Four **structural problems** that get expensive to fix after code exists (F2–F5).
3. A design that has grown to roughly a quarter's work for a problem that is, today, one
   developer, one Mac, one rented card, and runs that cost $1–11. It needs a **cut line** (F6)
   and a **different phase order** (F7).

Findings are ordered by how much they should change what gets built.

---

## Blocking — fix in the design before writing code

### F1. The default routing mixes two builds of a model *inside a single conversation*

> **Resolved 2026-09-17 — downgraded from blocking to "record and measure".** Decision: do not
> prevent mixing; record *(model served, runtime class)* for every call, join it onto the
> exported sessions, and compare behaviour per build on the first run. Accepted because a run
> costs $1–11 to regenerate and random per-turn assignment is itself a good equivalence test.
> The session-consistency rule recommended below is kept as an off-by-default switch
> (`session_build_consistency`). Written into design.md §6.1. Known limit, accepted: recording
> detects mixing but cannot repair a mixed dataset. The original finding follows unchanged.

Three decisions made separately combine badly:

- local is always the first tier (§4),
- session affinity is off (§4),
- `gemma4:e4b` silently becomes the MLX build on the Mac and the standard build elsewhere (§4.2).

With the Mac (3 workers) and one rented card (6 workers) both busy, each turn lands on the Mac
with probability about one in three. So nearly every 12-turn session has some agent turns
produced by `gemma4:26b` MLX and others by `gemma4:26b` CUDA GGUF. §6 treats mixing as a
per-*run* property that can be "separated afterwards" using the per-call host record. That works
when whole sessions differ; it does not work when the mixing is inside the session, because the
unit the behavioural pipeline analyses **is** the session. There is nothing to separate.

The design's own premise (§6, and the workplan's parity gate) is that these builds are not
output-equivalent. The default configuration therefore produces the contamination the parity
gate exists to prevent, per turn, without anyone choosing it.

**Recommendation.** Make the *session* the unit of runtime-class consistency. When a request
carries a session id, the class that served its first turn becomes an eligibility filter for
the rest of that session. This is not host affinity — any host of the same class may serve any
turn, so failover and priority keep working — and it costs one small map in the router. Make it
the default whenever a session id is present, and have the SDK send one by default. Requests
without a session id behave as designed today.

### F2. Request path and control loop share one process, and `on_exit` contradicts crash recovery

> **Resolved 2026-09-17 — accepted in full.** Two processes sharing a SQLite database; the
> router keeps serving from the last published host table if the supervisor dies; `on_exit` is
> removed in favour of `gpm stop --release` (destroy) vs. `gpm restart` / `gpm stop` (keep
> and re-adopt), with crashes left to the dead-man timer. The one-process alternative was
> dropped once the framework was declared public: isolation has to hold by construction, not by
> every contributor's discipline. Written into design.md §2.2. Original finding follows.

§2 specifies "one process". That couples two things with opposite needs:

- The **router** must never stall: a slow request path is a slow app.
- The **supervisor** does slow, blocking, failure-prone work: shelling out to the `vastai` CLI,
  SSH, waiting minutes for a bid (`wait_running` polls for up to 600 s).

A blocking provider call, or a supervisor bug, takes down routing to the *local* host too —
which needs no supervisor at all.

Separately, §7.4 says a restarted supervisor **adopts** the instances it finds, while §7.9 sets
`on_exit: release_rented` by default. Both cannot be the default. As written, restarting the
service to pick up a code change destroys a host holding 34 GB of models mid-run — and the
console (§9.3) lists changes that *require* a restart.

**Recommendation.** Two processes, or at minimum two isolated task groups with every provider
call pushed off the event loop. The router reads a host table the supervisor publishes and
keeps serving from the last known table if the supervisor dies. Replace the single `on_exit`
flag with two explicit verbs: `gpm stop --release` (operator is done: drain and destroy) and
`gpm restart` (hosts are kept and re-adopted). A *crash* is neither — that case belongs to the
dead-man timer, which is what it is for.

### F3. Timeouts and cancellation are unspecified, and the layers will fight

> **Resolved 2026-09-17 — accepted.** Written into design.md §4.3 as five rules: cancel upstream
> work on client disconnect; only the SDK retries for capacity (the router retries once, only
> for a host that died before first byte); the SDK owns connect / time-to-first-byte /
> between-bytes timeouts with the enforced invariant *router queue timeout < SDK
> time-to-first-byte*; non-SDK clients get an early `503` + `Retry-After` (queue timeout now
> 30 s, limits published in `/pool/status`); optional request deadlines so no worker starts
> work that is already too late. The §4 retry inconsistency is fixed. Note: the original text
> below argues from one of this repo's scripts; the accepted reasoning is the generic one —
> the pool cannot know an arbitrary client's timeout. Original finding follows.

There are now four places that wait or retry: the client library's own timeout, the SDK's
retry loop, the router's queue, and the router's retry-on-another-host — plus the existing
driver loops (`CONSECUTIVE_FAILURE_THRESHOLD`). The design gives each a number and never relates
them. Concretely: `langfuse_ollama.raw_call_ollama` uses a 60 s timeout, and the router queue
is 120 s. A queued request is abandoned by its client at 60 s, retried, and now occupies **two**
queue positions; when a worker frees up, it serves the request nobody is listening to. The
design never says the router cancels upstream work when the client disconnects.

Also inconsistent: §4's failure handling says "retry once on a different host", step 5 of the
same section says "the next eligible host in the same order".

**Recommendation.** Add a short "time budget" section that owns all of it:

- The SDK sets the client timeouts, split into *connect*, *time to first byte* (which includes
  queueing) and *between bytes*. No caller sets raw timeouts.
- The router **cancels the upstream request and frees the worker when the client disconnects**.
- One layer retries for capacity (the SDK). The router's cross-host retry is for a host dying,
  is attempted once, and is not a second capacity loop.
- State the invariant: `router queue timeout < SDK time-to-first-byte timeout`.

### F4. Requests are anonymous, so leases, fairness and the `no_lease` answer do not work

> **Resolved 2026-09-17 — by a different, simpler route than recommended.** Decision: requests
> stay anonymous; the **pool is the unit of isolation** and has a required API key. Budgets,
> leases, authorisation to rent and per-client policy all become properties of the pool;
> workloads that must be kept apart use different pools. **Multi-pool management** (fairness
> between workloads, per-team budgets, whether pools may share a host) is future work. Written
> into design.md §2.3, with one reviewer addition accepted into the design: the **app key**
> (inference only) and the **admin key** (control API, console) are separate, so an app that
> can request a completion cannot spend money. Not adopted: client ids, `interactive`/`batch`
> classes, and the `gpm run -- <cmd>` lease wrapper (the last is independent of this decision
> and can be revisited). Accepted limit: inside one pool the queue is first come, first served
> and the request log cannot tell apps apart. Original finding follows.

§2.1 says the app never knows a lease exists — right. But the design then needs to know, per
request, things it has no way to know:

- **Spend attribution.** With two leases open, whose budget does vast-1 burn?
- **`503 no_lease`.** "Only rented hosts could serve this and no lease is open" — no lease *for
  whom*? Any open lease currently authorises renting for anyone's overflow: a labeling run's
  lease will rent capacity for unrelated chat-UI traffic.
- **Fairness.** A person typing into the chat UI queues behind 12 batch workers. There is no
  notion of an interactive request outranking a batch one.
- **Runtime-class pins** are per client, leases were per run; the two never meet.

**Recommendation.** Requests carry a **client id** (a plain label from the app's environment —
`GPM_CLIENT=chat-agents-generation` — sent by the SDK). That is not knowledge of the pool's
internals, so §2.1 holds. Leases name the client ids they cover; spend, `no_lease` and pins are
then all resolved per client. Add two request classes, `interactive` and `batch`, with
interactive dequeued first.

Related ergonomic gap: a run script must remember to open a lease, and if it forgets, the run
quietly proceeds on the Mac alone at a tenth of the speed. Provide `gpm run --workers 12
--max-spend 5 -- <command>`, which opens the lease, runs the command, and closes the lease when
the command exits. Lease lifetime tied to process lifetime also removes abandoned leases.

### F5. Nothing says what data may go to which host

> **Deferred 2026-09-17 — not a blocker, by the owner's decision.** The current focus does not
> include routing sensitive data, so host trust levels are recorded as a future requirement
> (design.md §14) rather than designed now. It fits the pool-as-isolation-unit model when taken
> up. The one point kept live is already open question 8: the dead-man timer must not put the
> cloud account key on a rented machine. The addendum's re-weighting ("most important of all")
> is the reviewer's view and was not adopted. Original finding follows.

A rented Vast host is someone else's machine. The SSH tunnel protects the wire; it does nothing
about the machine's owner, who can read the container's memory, and therefore every prompt and
completion. For simulator output that is fine. The design, though, is a general "serve GPU
compute to every LLM client" layer, and Aletheia's purpose is analysing *customers'* agent
traffic. The first time the labeling pipeline is pointed at real traces, the pool will route
them to an anonymous community host, because nothing in the design says not to.

Two smaller items in the same family:

- The dead-man timer (§7.9) needs the host to stop itself. If that requires the *account* API
  key on the rented machine, the design hands the account to an untrusted host. §13 lists this
  as "believed available, not verified"; it should be a hard rule: **instance-scoped credential
  or no dead-man timer**.
- Anything passed via `--env` at instance creation is visible to the host.

**Recommendation.** Give every host a `trust` level (`own` / `private` / `untrusted`), defaulting
rented community hosts to `untrusted`. Give every client id a data sensitivity. Eligibility
(§4 step 1) gains one more filter. Default for a new client: `own` and `private` hosts only, so
using an untrusted host is a decision somebody made.

---

## Should fix — before the phase that touches it

### F6. Scope: the design needs a cut line

> **Resolved 2026-09-17 — cut line accepted, with three owner musts moved into v1:** all three
> host kinds (local, fixed-remote, bidding provider), the **operator console** (full, including
> editing and live market preview — only strategy replay waits), and the **dead-man timer**. The
> timer's blocking unknown was checked the same day: Vast injects a per-instance restricted key
> (`CONTAINER_API_KEY`) that can only start/stop/destroy that instance, so no account key goes
> on a rented machine. Written into design.md §11.1 and §7.9. The table below is the reviewer's
> original proposal, superseded by §11.1. Original finding follows.

The document describes a router, a supervisor, an SDK, a variant resolver with a behavioural
probe, three bidding strategies, floor sampling, per-machine memory, a replay harness, a
dead-man timer and a seven-screen console. The measured problems are three: evictions kill or
pause runs; idle hosts cost money; one card stops scaling at 6 workers.

**Recommendation — a stated v1, and a gate after it.**

| v1 (solves the three problems) | Deferred until v1 has run a full generation cycle |
|---|---|
| Router: workers, priority tiers, failover, machine-readable `503` | Behavioural variant probe — v1 uses declared/implied platform only |
| SDK: default wait-and-retry, typed errors | `volatility_aware`, `fraction_of_on_demand`, re-bid down, `rebalance` |
| Catalog-based variant rewrite (F9) | Floor sampling, machine memory, replay harness |
| Supervisor: leases, idle release, verified destroy, orphan sweep, today's bidding unchanged | Console editing, market preview, host wizard |
| Read-only status page | Queue-driven scale-up (F8), calibration runs, on-demand fallback |

The gate: after v1, write down what actually hurt. Build from that list, not from this one.

### F7. Phase order puts the biggest technical risk second and delivers nothing first

> **Resolved 2026-09-17 — accepted.** design.md §11 re-ordered: 1 Route (router + SDK over static
> hosts, exit test = passthrough fidelity), 2 Supervise (provider plug-in, leases, bidding,
> dead-man timer, fake provider first), 3 Console, 4 Release (generic core, plug-in docs, threat
> model, licence, clean repo), then everything deferred in §11.1. **Owner addition the same day:**
> a v1 requirement to *prepare a rented host on request* from the console — rent, load chosen
> models, verify, then join / park / destroy — with parked hosts restarted before new offers
> are bid on (design.md §7.11). API + CLI land in phase 2, the console flow in phase 3. Original
> finding follows.

Phase 1 ("relocate `vast_provision.py`, behaviour unchanged") ships no user-visible value.
Phase 2 carries the riskiest unknown in the whole design: whether a proxy can carry Ollama
streaming, 41 tools' worth of tool calls and `format` schemas without changing behaviour.

**Recommendation.** Swap them. Phase 1 becomes the router + workers + priority + SDK over
**static** hosts: the Mac, plus the rented host that `vast_provision.py` already keeps alive,
reached through the existing tunnel on a fixed port. That delivers decoupling, local + remote
pooling and failover immediately, tests passthrough fidelity first, and leaves the working
provisioner untouched until the supervisor is ready to absorb it.

### F8. The scale-up signal is circular

> **Settled 2026-09-17 by the v1 scope decision (F6):** queue-driven scale-up is deferred; in v1
> the lease's requested worker count is the demand signal. Original finding follows.

§7.6 rents on sustained queue depth. But the drivers size themselves from the pool
(`--workers auto` reads total workers from `/pool/status`). A client that never sends more than
capacity never builds a queue, so the pool never sees demand and never scales. The only real
demand signal is the number the lease states.

**Recommendation.** In v1 the lease's `--workers` *is* the demand; drop queue-driven scaling.
If it returns later, drivers must size from the lease's wanted workers, not current capacity.

### F9. Variant rewriting makes "the app cannot tell the difference" untrue

> **Resolved 2026-09-17 — accepted.** Only catalogued models are ever resolved; anything else is
> passed through literally — no guessed `<name>-mlx`, no pull (design.md §4.2). The "cannot tell
> the difference" claim is withdrawn and the contract is stated as the engine API plus a small,
> versioned pool dialect (design.md §2.4). Original finding follows.

§2.1 promises that pointing an app at a bare Ollama or at the pool is indistinguishable. After
§4.2 it is not: `gemma4:e4b` means the GGUF build against bare Ollama on the Mac and the MLX
build through the pool. Same app, same config, different model. The requirement (ask for the
logical name, get the right build) is reasonable; the claim needs correcting, and one part of
the mechanism is unsafe: resolving *unlisted* models by guessing `<name>-mlx`. A guess that
triggers a pull is an unreviewed download of an unreviewed artifact.

**Recommendation.** Resolve **only** catalogued models; anything not in the catalog is passed
through literally. State the contract honestly as two layers: the Ollama API, plus a small named
pool dialect (logical model names, client id, session id, machine-readable `503`). Recording
`served_model` in trace metadata should be mandatory in the SDK integration, not optional.

### F10. "Recorded as a fact" is not deliverable on the LangChain path

> **Settled 2026-09-17 by the F1 and F6 decisions:** v1 joins the pool's request log onto
> exports offline by session id and timestamp; the SDK capture hook is deferred. Original
> finding follows.

§4.1 promises each response exposes `pool_wait_s` and the serving host. Those arrive as HTTP
response headers. chat-agents reaches the pool through `ChatOllama`, which does not surface
HTTP headers — and the transport is injected *underneath* it precisely so call sites do not
change. So on the path that produces the dataset, wait time will be folded into generation
latency after all, which is the distortion the feature was meant to prevent.

**Recommendation.** The SDK's transport records per-call metadata into a context-local
collector (`with pool.capture() as calls:` around a turn), which the app reads and attaches to
its Langfuse span. It is one small integration point in chat-agents and should be listed as such
rather than implied to be free.

### F11. Workers ignore model swapping and let embeddings queue behind generations

> **Resolved 2026-09-17 — part A by a stronger rule than recommended, part B not adopted.**
> **A:** the owner's model is that every host runs one engine process holding *all* of the
> pool's models, loaded permanently. The pool declares its model set; a host is `ready` only
> with the full set resident; a host that cannot hold it does not join; a request for a model
> outside the set is refused rather than loaded. Swapping is removed by construction, so the
> reviewer's loaded-vs-present routing and pinned resident sets are unnecessary (design.md
> §5.5). **B:** the queue stays first come, first served — no separate lane for embeddings;
> recorded as a deferred requirement (design.md §14). Original finding follows.

- Ollama's parallelism is per loaded model, and the memory ceiling depends on the resident set.
  The design re-evaluates worker counts "when the model set changes" but routes on "model
  *present*". A request for a model that is on disk but not loaded makes Ollama evict one that
  a lease depends on; two clients alternating models will thrash a host. Eligibility should
  distinguish **loaded** from **present**, and a host under a lease should have a pinned
  resident set that outside requests cannot displace.
- An `/api/embed` call takes milliseconds and currently needs a worker, so it can wait minutes
  behind 12 long generations. Embeddings should bypass worker accounting or have their own lane.

### F12. A JSON state file will not carry what the design asks of it

> **Settled 2026-09-17 by the F2 decision:** SQLite (WAL) is the shared store between router and
> supervisor; configuration stays a human-editable file. Original finding follows.

`state.json` plus `events.jsonl` is asked to hold hosts, leases, spend, the request log, floor
samples, per-machine memory, probe results and fifty config versions, with at least two writers
(F2) and a console that filters and expands decisions.

**Recommendation.** SQLite in WAL mode from the start: one file, safe concurrent access,
queryable for the console and the replay harness. Keep the config as a human-editable file.

### F13. Lease caps are enforced against an estimate

> **Resolved 2026-09-17 — both parts accepted.** Spend is reconciled every control-loop pass
> against provider-reported charges; caps are enforced on the higher of the two figures less a
> safety margin (default 10 %); drift is logged and shown in the console; a provider that
> cannot report charges must declare it (design.md §7.3, `reported_charges()` in §3.2). The
> provider plug-in is an HTTP API client, not a CLI wrapper (§3.2). Original finding follows.

Spend is computed as bid × time. Real charges also include storage while stopped, ingress, and
whatever the provider rounds. A "hard stop" budget enforced on an estimate will drift.

**Recommendation.** Reconcile against provider-reported charges each control-loop pass where the
API offers them, and enforce caps with a stated margin. Also move the supervisor from shelling
out to the `vastai` CLI (answering its prompt with `input="y\n"`, parsing JSON by searching for
the first bracket) to the provider's API: acceptable in a hand-run script, fragile in a service
that runs unattended with money attached.

---

## Document defects

> **Resolved 2026-09-17 in the document split (F14).** Stale "no app change" and "keeps working
> exactly as now" claims are gone; the retry wording is single-sourced in the app contract's
> time budget; idle release has one definition (park if a lease is open, else destroy); the host
> state machine gained `parked` and `disabled`; an on-demand rented kind is listed as deferred
> rather than half-specified; the recovery options are now "re-bid in place" and "replace", so
> "tier" means routing priority only; decisions moved to a numbered log; the build-or-adopt
> conclusion is recorded as decision D25 with the deciding requirements stated.

Left over from the design growing by accretion through the day. None changes the architecture;
all will mislead an implementer.

| Where | Defect |
|---|---|
| §2, "Three decisions", item 1 | Still says "chat-agents needs no change … No client learns that a pool exists". §4.1 adds the SDK and a change in `_chat_ollama`. |
| §4 failure handling vs. §4.1 | §4: `host_health.is_connection_error` "keeps working exactly as now". §4.1: typed errors *replace* it. |
| §4 failure handling vs. §4 step 5 | "Retry once on a different host" vs. "the next eligible host in the same order". |
| §7.3 vs. §7.9 | Idle release: §7.3 requires "no open lease"; §7.9 stops-and-keeps-warm *inside* an open lease. |
| §3.3 state machine | No state for a stopped-and-kept-warm host (§7.9), none for `disabled`, and the diagram predates proactive re-bid and wait-it-out (§7.8). |
| Host kinds | `allow_on_demand` (§7.7) introduces a rented, non-interruptible host. It has no kind, no priority tier, and no row in §3.2. |
| Terminology | "Tier" means routing priority (0/10/20) *and* recovery option (tier 1 / tier 2). Rename the recovery options. |
| Structure | One 1,070-line file with decisions recorded as inline "(decided 2026-09-17)" notes. Split into overview, app contract (router + SDK), supervisor and strategies, console; pull decisions into a short numbered decision log so a reader can see what was settled and why without reading everything. |
| Build-or-adopt (§4) | The conclusion ("write the router") is now better supported than the table shows — workers, variant rewriting and `503` reasons are things no candidate does. The table should say so; as written it rests mainly on dynamic endpoints, which the stable-local-port pattern in the same section works around. |

---

## What is good and should not be lost in revision

- **The lease as the sole spending authority**, with tighten-only overrides. The strongest idea
  in the document, and the one most in keeping with how this project is actually run.
- **Relocating, not rewriting, the provisioner.** The bidding knowledge in `vast_provision.py`
  was paid for in real evictions and real ingress bills.
- **Verified destroy, orphan sweep, provider-as-source-of-truth-for-existence.** Correct model
  for infrastructure that bills when forgotten.
- **Decisions logged with the numbers behind them**, plus plan mode. This is what makes
  unattended spending reviewable after the fact.
- **The `format`-schema guard on MLX.** Measured by probe rather than assumed from a name, and
  self-correcting if the engine changes.
- **Step-down on evidence, never step-up** for worker counts.
- **The console as one more client of the control API**, with the loopback CSRF point addressed.

---

## Suggested order of work

1. Revise the design for F1–F5 and the document defects. No code.
2. Re-cut scope per F6 and re-order phases per F7.
3. Build the new phase 1 (router + workers + priority + SDK, static hosts) and run the
   passthrough fidelity test against chat-agents' real tool-calling traffic. This is the
   experiment that decides whether the rest of the design is worth building.
4. Only then let the supervisor absorb `vast_provision.py`.

---

## Addendum, 2026-09-17 — the framework is to be public

Stated after this review was written: the pool will be released publicly as a reusable
framework, with Aletheia as first adopter (design.md §0). The review assumed internal tooling
for one developer. Re-weighing the findings under the new intent:

| Finding | Change | Why |
|---|---|---|
| F2 process isolation | Decided: two processes | The "one developer can just restart it" counter-argument no longer applies |
| F3 timeouts and cancellation | **More important** | Unknown clients with unknown timeouts; the contract must state the time budget, not leave it to each adopter to discover |
| F4 client identity, fairness | **More important** | Several apps and several people sharing one pool is the normal public case, not a corner of it |
| F5 data trust per host | **Most important of all** | Strangers will route real data. Sending prompts to an anonymous rented machine must be impossible by default and explicit to enable |
| F6 scope / v1 cut line | **More important** | A public v1 has to be dependable; every deferred feature is one fewer thing to support |
| F9 no guessed model pulls | **More important** | A convention that downloads artifacts by guessed name is a supply-chain risk in someone else's hands |
| F12 SQLite, F13 provider API over CLI | Promoted from "should fix" to required | Unattended operation with other people's money |
| F1 mixed builds | Unchanged | Still an adopter's data-quality choice; record-and-measure remains the right default |

> **F14–F18 resolved 2026-09-17 — delegated by the owner to the reviewer ("fix those on your
> own").** F14: the design is split into [overview.md](overview.md), a generic
> [spec/](spec/) (no adopter names), [decisions.md](decisions.md), [roadmap.md](roadmap.md),
> and [adopters/aletheia/](adopters/aletheia/README.md). F15:
> [spec/plugin-interfaces.md](spec/plugin-interfaces.md) — provider, engine and strategy as
> separate versioned interfaces with stated guarantees, declared capabilities and a fake
> provider. F16: variants are keyed by **host capability**
> ([spec/hosts-routing-capacity.md](spec/hosts-routing-capacity.md) §4). F17:
> [release-checklist.md](release-checklist.md) — clean repository, naming criteria, a licence
> recommendation (Apache-2.0); **name and licence remain the owner's decisions**. F18:
> [threat-model.md](threat-model.md) — 19 threats with mitigations and residual risk; two
> requirements it surfaced (SSH host-key pinning; owner-only file permissions) were added to
> the spec. The deferred data-trust requirement (F5) appears there as an accepted, documented
> residual risk (T7).

**New findings the public intent creates:**

- **F14. The design is not separable from Aletheia.** Names, paths, models, Langfuse and the
  calibration constants are woven through every section. Needed: a generic core specification,
  an Aletheia adoption guide, and an example config — and the existing-code analysis (§1)
  reframed as the motivating case study.
- **F15. Extension points are implied, not specified.** The host-kind table (§3.2) is close to a
  provider interface but mixes provider concerns (rent, bid) with engine concerns (list models,
  preload, restart). Split into **provider**, **engine** and **strategy** interfaces, each
  versioned. Ollama is the first engine; OpenAI-compatible servers should be reachable by
  passthrough, not translation, so the tool-calling fidelity argument still holds.
- **F16. MLX-vs-standard is one instance of a general idea.** "Pick the right build of a model
  for the host it lands on" also covers AWQ vs. GGUF, FP8 on cards that support it, and so on.
  The catalog should map a logical name to variants keyed by **host capability**, with
  Mac/MLX as the first capability rather than the only one.
- **F17. Repository, name, licence.** Start a clean repository (this one's history carries
  internal material and local credentials), pick a neutral name before anything is published
  under `aletheia-*`, choose a licence before the first public commit.
- **F18. No threat model.** Internal tooling could get by on "bind to loopback". A public
  framework that holds a cloud API key, spends money unattended and exposes a control API needs
  a written threat model: who can reach the control API, what a compromised rented host can do,
  what a malicious web page can do, what is logged.
