# Roadmap — v1 Scope, Phases, Deferred Requirements, Open Questions

> Reasons are in [decisions.md](decisions.md) (D18, D20).

## 1. v1 scope

v1 is the left-hand column. Everything on the right waits until v1 has had real use, and is then
chosen from what actually hurt, not from this list. Items marked **must** were set by the owner.

| In v1 | After v1 |
|---|---|
| **All three host kinds** — local, fixed-remote, a bidding provider (**must**) — over `tunnel` / `http` / `https` | A second rented provider; a second engine; an on-demand (non-interruptible) rented kind |
| Router: workers per host, priority tiers, single failover, time budget, machine-readable `503`, required pool API key, first-come-first-served queue | Session affinity; `session_build_consistency`; calibration runs; evidence-based worker step-down |
| SDK: default wait-and-retry, typed errors, owned timeouts | A capture hook for wait and host metadata (the request log's offline join covers v1) |
| The pool's whole model set resident on every host; catalog-only variant resolution keyed by host capability, with the structured-output guard; capabilities declared, implied by the provider, or inspected over SSH | The behavioural capability probe; variant demotion on later failure |
| Supervisor: leases with **mandatory** dollar caps; overflow-based renting, one host at a time; `floor_plus_premium` bidding with both ceilings; eviction choice between re-bid in place and replace; idle release; drain; park-or-destroy; verified destroy; orphan sweep; spend reconciled against provider-reported charges | `volatility_aware` and `fraction_of_on_demand` bidding; proactive re-bid; re-bid downward; wait-it-out holding; thrash guard; warm-disk bonus; machine memory and avoid list; rebalancing; on-demand fallback; queue-driven scale-up |
| **Prepare a rented host on request** (**must**) — console, API and CLI; join / park / destroy; parked hosts restarted before new offers are bid on | — |
| **Dead-man timer on every rented host** (**must**) — instance-scoped credential | — |
| **Operator console** (**must**): Overview, Hosts with test-connection, Rented capacity with live market preview and Prepare a host, Models, Leases, Decisions, Configuration with validate → plan → apply and version history | Strategy replay in the console |
| Two processes; SQLite state; request log with model served and runtime class; decisions logged with their numbers; `gpm plan` | Market sampling; replay harness |
| Provider, engine and strategy plug-in interfaces, versioned, one implementation each | — |
| A **fake provider** and a test suite that needs no cloud account | — |

Live market preview stays in v1 although replay does not: it reuses the offer pipeline
read-only, costs almost nothing to build, and is what makes the console's bidding forms safe to
use. Replay needs recorded market data, which v1 does not collect.

## 2. Phases

The router comes first because whether a proxy can sit in front of an inference engine without
changing its behaviour is the largest technical unknown, and everything else depends on the
answer. Each phase ends in something usable on its own.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **1. Route** — a pool over hosts you already have | Router and SDK over **static hosts** listed in configuration — local, fixed-remote, any already-running remote endpoint — over all three transports. Workers, priority tiers, single failover, time budget, required API key, machine-readable `503`, whole-model-set readiness check, catalog-based variant resolution with the structured-output guard, request log. The engine plug-in interface with its first implementation. Router process only; SQLite from the start | **Passthrough fidelity**: streaming, tool calls, structured output and cancel-on-disconnect behave identically through the router and direct to the engine, on recorded real traffic. With two hosts up, killing one mid-run loses no request. With none up, a client pauses and resumes |
| **2. Supervise** — renting, safely | Supervisor process; the provider plug-in interface, built against the **fake provider first**, then the first real provider. Leases, renting, bidding, eviction handling, idle release, drain, park, verified destroy, orphan sweep, spend reconciliation, **dead-man timer**, `gpm host prepare`, decisions logged, `gpm plan`, control API with the admin key | Interruption drill: a forced eviction recovers unattended inside a lease and does nothing outside one. An idle rented host releases itself. A hand-made stray instance is caught by the sweep. Killing the supervisor leaves routing up, and the dead-man timer removes the rented host. A lease stops before its dollar cap when reported charges run ahead of the estimate. No test needs a cloud account |
| **3. Console** | The operator console on the phase-2 control API, all seven screens, including test-connection, live market preview and Prepare a host | Every console action is also possible from the CLI; a mistyped ceiling is caught by plan before apply; the app key is refused by the control API |
| **4. Release** | Everything in [release-checklist.md](release-checklist.md): clean repository, neutral name, licence, packaging, docs, security policy, the threat model reviewed against the implementation | A newcomer stands up a pool over one local and one remote host from the docs alone, with no reference to the first adopter |
| **Later** | The right-hand column of §1 | Chosen from what v1's real use showed to hurt |

During phase 1 a rented host, if wanted, is brought up by whatever means already exists and
listed as a static endpoint. The pool rents nothing until phase 2.

## 3. Deferred requirements

Known requirements deliberately **not** part of the current design — listed so they are picked
up on purpose rather than rediscovered.

| Requirement | Why it will matter | Status |
|---|---|---|
| **Data trust per host** — a trust level on every host (own / private / untrusted), a maximum per pool, untrusted hosts opt-in. A tunnel or TLS protects the wire, not the machine: the operator of a rented marketplace host can read every prompt and completion | Any adopter routing real data through a pool that can spill onto marketplace hosts | Deferred by the owner (D17). Fits the pool-as-isolation-unit model when taken up: a sensitive pool simply contains no untrusted hosts. **Until then it is a documented residual risk** — see [threat-model.md](threat-model.md) T7 |
| **Multi-pool management** — fairness between workloads, per-team budgets, an overview across pools | Several teams or workloads sharing hardware | Deferred (D16); open question 5 |
| **A separate lane for short calls**, and any request classes | An embedding call of milliseconds can wait minutes behind long generations; retrieval-augmented apps pay that on every turn | Not adopted (D23). Revisit with multi-pool management, or sooner if measured waits justify it |
| **Per-client identity inside a pool** — several app keys per pool, key id = client | Telling apps apart in the request log; per-client limits | Deferred (D16) |
| **A lease wrapper** — `gpm run … -- <command>` opens a lease, runs the command, closes it on exit | Removes forgotten leases, and the quiet failure where a run that forgot to open one crawls along on local hosts only | Not adopted (D16); independent of it, may return |

## 4. Open questions

1. ~~**Default lease caps**~~ — **answered (D32)**: the dollar cap is mandatory with no default;
   1 rented host, $1.00/hour burn, 10 idle minutes, 4-hour leases, a 20-minute dead-man timer
   and a 10% cap safety margin.
2. ~~**Does the supervisor run permanently or only while a lease is open?**~~ — **answered
   (D31)**: permanently. It rents nothing without a lease, and the sweep and parked-host limits
   have to mean something between runs.
3. **Does the first provider charge the bid or a clearing price?** Decides whether re-bidding
   downward is worth building.
4. **Is price history available from the provider**, or only what the pool samples itself?
5. **Multi-pool management: can two pools share a physical host?** If yes, a layer above the
   pools must split a host's workers between them; if no, a host belongs to exactly one pool.
6. **On-demand fallback** — may a lease fall back to a non-interruptible instance when bidding
   fails or crosses the on-demand crossover, or should the run always wait? Drafted as off.
7. **Name and licence** — see [release-checklist.md](release-checklist.md).
