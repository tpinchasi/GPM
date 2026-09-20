# S3 — An advisor model that picks the machine

> Status: **planned, not decided.** Part of the [feature list](README.md). Nothing here is
> specification until its decisions are recorded in [decisions.md](../decisions.md).

## The story

*As an operator, I can switch on an advisor: a small language model that looks at the offers my
rules already accepted, and at how machines have actually behaved for this pool, and chooses
which one to rent — telling me why.*

## Why

The rules judge an offer by what the market says about it: price, memory, reliability score,
download speed. Measured on the first live pool, that is not enough. Six hosts were profiled at
once, and **two machines carrying the identical hardware label differed fourfold in cost per
request** — one served 125 tokens a second, another 9. The slow ones sat on shared machines with
a load average above 250, their accelerators idle at 5–13 %. Nothing in their offers said so.

What separates a good machine from a bad one is **history**, and history is awkward to express
as thresholds: which machines served well, which were evicted within minutes, which data centre
throttled downloads at a certain hour, which seller's machines never started.

## The history already exists

Nothing new has to be collected to begin. The decision log and the request log, joined on the
host id, already give a per-machine record. Taken from the first live pool as it stood —
**28 machines, 62 rentals**:

| Machine | Hardware label | Rented | Reached ready | Evicted | Failed | Mean service time, largest model |
|---|---|---|---|---|---|---|
| A | one 140 GB-class card | 7 | 2 | 4 | 3 | 11.1 s |
| B | one 96 GB-class card, server edition | 4 | 3 | 3 | 0 | 20.4 s |
| C | the same label as B | 2 | 1 | 1 | 0 | **5.8 s** |
| D | two 80 GB-class cards | 3 | 3 | 2 | 0 | 15.3 s, over 10,029 requests |

Two things the rules could not see are plain here. **The pool went back to machine A seven times**
though it reached ready twice — each return a bid, a wait and a bill. And **B and C carry one
label and differ three and a half times over**, which no offer field distinguishes.

So stage 1 below is a *view* over data the pool already writes, not new collection — and it can
be judged against these 62 rentals before it decides anything.

## The simpler mechanism, offered first

Most of the value is in the **history**, not in the model. A machine-history table — per machine
id: times rented, time to ready, measured throughput, evictions, failures — can feed the existing
scoring function directly: a deterministic bonus for a machine that served well, a penalty for
one that did not. No model, no non-determinism, fully replayable.

**What that gives up:** weighing soft, unstructured evidence against price in a way nobody wrote
a rule for. The plan below therefore builds the history first — useful on its own, and to S1 and
S5 — and the advisor as a layer over it that can be switched off without losing the history.

## Design

**The advisor chooses; it never decides what is allowed.** It sees only offers that already
passed every hard filter, and answers with one of their ids. It cannot widen the list, touch a
price, or see a credential.

```
offers ─► hard filters ─► rule ranking ─► top N ─► ADVISOR ─► chosen offer ─► bid strategy ─► caps re-checked
                                                     │
                                  machine history ───┘
```

- **Input:** the top *N* ranked offers with the numbers the rules used, and the history rows for
  those machines. Nothing about requests, prompts, apps or keys.
- **Output:** structured and schema-checked — `{offer_id, reasons[]}`. An id not in the list, a
  malformed answer, a timeout, or an unreachable endpoint all mean the same thing: **the rule
  ranking's first choice is used**, and the log says the advisor was not heard.
- **After it answers**, the bid strategy prices the offer and the supervisor re-checks every cap
  and ceiling, exactly as for any strategy. A wrong or hostile advisor can pick a worse machine
  from an acceptable list. It cannot spend past a limit.
- **Its reasons are stored verbatim** beside the rule ranking it overrode, so the Decisions
  screen shows both and an operator can judge whether it is earning its place.

**It is not a strategy, and the spec should say so.** Strategies are pure functions — no I/O, no
randomness — and that is what makes them replayable. A model call is neither. The advisor is a
named stage *between* two pure strategies, with the fallback above, so every other stage keeps
its guarantees.

**Where the model runs.** `advisor.endpoint` is a URL and a key: any endpoint speaking the
engine's API. An operator may point it at this pool's own app-facing URL with an app key, making
the supervisor one more app of the pool. No private path is added between supervisor and router,
so the rule that they never call each other stands. The advisor call is on the supervisor's
control pass, never on a request path, and is bounded by a timeout.

**Toggleable** per pool (`advisor.enabled`), and per action: prepare-a-host gains "let the
advisor choose", and the market preview shows what it would pick and why, spending nothing.

## Configuration sketch

```yaml
rented:
  advisor:
    enabled: false
    endpoint: http://127.0.0.1:8090     # any engine-compatible URL; may be this pool's own
    key_env: GPM_ADVISOR_KEY
    model: <a small model from the catalog>
    consider_top: 8
    timeout_s: 20
```

## Safety

- Hard filters first, caps re-checked after; the advisor sits strictly between.
- Off by default. With it off, behaviour is exactly today's.
- Only offer and machine-history data is sent. If the endpoint is not this pool's own, that data
  leaves the pool — stated in the threat model when decided.
- A pool whose only capacity is the rented capacity it is trying to obtain cannot ask its own
  model. The fallback covers it: no answer, rule ranking.

## Decisions this story needs

1. A machine-history table, and its use in the deterministic score.
2. The advisor stage: its place in the pipeline, its closed input and output, its fallback.
   **Amends plugin-interfaces §3** by naming a non-pure stage and bounding it.
3. How the supervisor reaches a model without breaking the router/supervisor separation.

## Open questions for the owner

1. **History alone first?** Recommended: build and run the history-fed score, then add the
   advisor and compare the two on the same decisions before trusting it.
2. **Is "the pool's own URL with an app key" acceptable**, or must the advisor endpoint always
   be separate from the pool it advises?
3. **May the advisor decline every offer** — "none of these, wait"? Recommended: not at first. A
   stage that can stall renting needs its own bounds.

## Build stages

1. Machine history as a view over the decision log and request log — no new collection — shown
   on the Rented capacity screen. One gap to close: a rental's machine id is only in its event's
   text today, so the event gains it as a number.
2. History in the deterministic score; the preview shows the adjustment and its reason.
3. The advisor stage with a fake advisor; fallback paths; decision log.
4. A real endpoint; the side-by-side comparison in the Decisions screen.

## Tests

A fake advisor that answers well, answers with an unknown id, answers garbage, hangs, or is
unreachable: the first picks its offer, the rest fall back, none bypasses a cap. No model and no
GPU is needed by any test.

## Depends on

Nothing to start. The history gets richer with **S4** (facts from the machine) and **S5**
(measured throughput).
