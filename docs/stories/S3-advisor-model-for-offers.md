# S3 — An advisor model that picks the machine

> Status: **partly decided (D69), not built.** Part of the [feature list](README.md). Decided: the
> machine history comes first and is judged on its own; the advisor may decline every offer, inside
> bounds; and an evaluation suite gates it. **Still open: how the advisor reaches a model** — the
> owner's answer, "via tool", needs one clarification (see the end of this page).

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

## Declining, and the evaluations that gate it (D69)

**The advisor may say "none of these — wait."** Waiting is sometimes right in a bad market; the
hard filters already say an empty result means stay paused. But a stage that can stall renting is
bounded: after `max_declines` in a row, or `max_wait_s` since the first, the rule ranking's first
choice is used. Every decline is a logged decision with its reasons.

**Evaluations are part of the feature.** The advisor is not switched on for real spending until
it has been shown to be worth having, on the record the pool already holds:

| The suite | What it measures |
|---|---|
| **Replay** — each recorded renting decision is put to the advisor again, with the history as it stood *then* | Would it have chosen differently from the rules? |
| **Hindsight score** — each choice is judged by what happened to that machine: reached ready or not, time to ready, how long it lasted, measured cost per request | Were its different choices *better*? |
| **Decline audit** — for every "wait", what the market offered over the following minutes | Was waiting right, or did it just delay the same rental? |
| **Robustness** — malformed answers, unknown ids, prompt-injection text planted in an offer's free-text fields | Does every bad answer fall back to the rules? |

The baseline is the history-fed deterministic score, not the bare rules — the advisor has to beat
the simpler mechanism, or it is not worth its non-determinism. The harness and its scoring run
with a fake advisor and need no GPU; scoring a real model is an opt-in run, like every live test.

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
    max_declines: 3                     # "none of these" in a row, then the rules choose
    max_wait_s: 900
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

## The owner's answers

1. **History alone first?** Yes. Built, run and judged before any model is involved (D69).
2. **How does the advisor reach a model?** "Via tool" — **to be clarified**, see below.
3. **May it decline every offer?** It can happen, inside bounds — and good evaluations must
   validate its responses before it is trusted (D69).

## Still open: what "via tool" means

Two readings, and they are different designs:

- **(a) The advisor works through tool calls.** Instead of being handed everything in one prompt,
  the model is given a closed set of tools — look up a machine's history, list the accepted
  offers, `choose(offer_id, reasons)`, `decline(reasons)` — and answers by calling them. The
  closed tool list plays the same part as the agent's closed verb list: the model can only do
  what a tool permits, and `choose` accepts only an id from the accepted list.
- **(b) The advisor is a separate tool beside the pool** — its own small program that uses the
  pool like any other app to reach a model, rather than the supervisor calling a model itself.

Reading (a) says how the model is *used*; reading (b) says where it *runs*. They can also both
be true.

## Build stages

1. Machine history as a view over the decision log and request log — no new collection — shown
   on the Rented capacity screen. One gap to close: a rental's machine id is only in its event's
   text today, so the event gains it as a number.
2. History in the deterministic score; the preview shows the adjustment and its reason.
3. The evaluation harness: replay, hindsight score, decline audit, robustness — with a fake
   advisor, against the recorded rentals.
4. The advisor stage itself; fallback paths; bounded declining; decision log.
5. A real model, scored by the harness; switched on for spending only if it beats stage 2.

## Tests

A fake advisor that answers well, answers with an unknown id, answers garbage, hangs, or is
unreachable: the first picks its offer, the rest fall back, none bypasses a cap. No model and no
GPU is needed by any test.

## Depends on

Nothing to start. The history gets richer with **S4** (facts from the machine) and **S5**
(measured throughput).
