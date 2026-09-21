# S5 — Automatic worker adjustment per host

> Status: **built (D67, D68)**, 2026-09-21. Part of the [feature list](README.md). Off by
> default: `rented.workers_auto.enabled`. A host starts at six, climbs on evidence within what
> its engine was launched to run, and steps down when a model leaves memory, when it is far
> slower than the pool for the same model, or when its last step up did not pay. The controller
> is a pure function; the measurements come from the request log the router already writes.

## The story

*As an operator, I stop guessing how many requests each machine can run at once. Each host finds
its own number while it serves — more where the machine has room, fewer where it is struggling —
and what it learns is kept for the next time that machine is rented.*

## Why

A worker count is set once, from a hardware label, before the bid (D45). Measured on the first
live pool, that is the wrong grain in both directions:

- **Too low for good machines.** Latency was flat from three to six requests in flight on every
  host, so six was not the ceiling. The best machine ran at 89 % of its accelerator; another,
  just as fast, at 41 %. No host has ever been run above six, so the real number is unknown.
- **Too high for bad ones.** Two machines with the same label as the best one served at a
  quarter of its throughput, starved by neighbours. Six workers there only meant six slow
  requests, and because the router prefers the least-loaded host, their open slots kept pulling
  traffic toward the worst machines in the pool.

The label cannot tell these apart. Only the running host can.

## What it supersedes — stated plainly

| Today | This story |
|---|---|
| "Step down on evidence, never up. Raising a ceiling is an operator decision" ([hosts-routing-capacity.md](../spec/hosts-routing-capacity.md) §2.2) | Both directions, automatically, inside bounds |
| D56: resizing is "an operator's explicit act and never the supervisor's own pass" | The supervisor may resize, when the operator has enabled it |
| D41: a restart is "only ever an operator's explicit act" | **Left standing** — see the next section |

## Design

**Start at six, launch with room above it.** The expensive direction today is *up*: engine
parallelism is fixed at start, so raising means a relaunch and a minute without the host (D56).
The way round is to launch the engine at the most this machine can really hold — the memory
ceiling, or `workers_auto.max` where that is lower — and have the pool *use* fewer workers than
that, beginning at the profile's number, **six where no profile matches**. Workers stay real,
because they never exceed what the engine runs; and **every adjustment, up or down, becomes a
change of router slots: instant, graceful, no restart.** D41's rule is never touched, because
nothing is restarted. Six is a starting point and the evidence behind it, not a limit (D68).

**What it gives up:** memory. An engine reserves worst-case context for every parallel slot, so a
high launch bound costs accelerator memory whether or not the slots are used. It is bounded by
the memory ceiling formula, and on the machines measured the whole model set used a third of the
card. A relaunch is needed only to go above the launch bound itself — that stays an operator's
act, through S4's agent.

**The controller, per host, each window:**

| It sees | From |
|---|---|
| Requests in flight and completed; service time per model | The request log — already written |
| Tokens generated per second, in aggregate | The engine's own usage figures, read from the final frame through the engine plug-in — **new**: the log records no token counts today, and latency alone cannot separate a slow host from a long answer |
| Load average, accelerator utilisation and memory | S4's extended facts |

- **Step down** by one when throughput stayed flat or fell after the last step up, when a
  resident model was evicted, or when the host's service time is far above the pool's median for
  the same model. Surplus workers drain; nothing in flight is disturbed.
- **Step up** by one when the host is saturated, requests are waiting, and the last step up
  raised aggregate throughput by a meaningful margin — past six if the evidence carries it, never
  past the launch bound.
- **Hold** otherwise. One change per host per window; no step up straight after a step down.
- **A floor of one, and then a replacement (D75).** A host that would go below it is not resized.
  Where its cost per unit of work is far worse than the pool's median, it is given up instead:
  drained, destroyed, its machine avoided, and the replacement bought by the ordinary allocation
  path. Shrinking a bad machine to one worker still pays its hourly rate for almost nothing.

**What it learns is kept.** The settled number and measured throughput are written to the machine
history (S3), so the next rental of that machine starts from evidence, and a capacity profile's
note can cite it. The first live pool's profile — "six, because six was observed" — is exactly
the kind of number this replaces with a measured one.

**Explicitly enabled, bounded, explained.** `rented.workers_auto`, off by default. Every step is a
logged decision with the numbers before and after. The console shows each host's current count,
its launch bound, and why it last moved.

## Configuration sketch

```yaml
rented:
  workers_auto:
    enabled: false
    window_s: 120
    min_gain: 0.10             # a step up must have raised throughput by this much
    slow_host_factor: 2.0      # service time this far over the pool median steps down
    start_at: profile          # the capacity profile's number; six where none matches
    max: 16                    # a hard limit, under the machine's own memory ceiling
```

## Safety

- Never above the launch bound, so never above real engine parallelism; never below one.
- Stepping down is always safe; stepping up is limited to one per window and only on evidence.
- The controller is a pure function of its measurements: replayable, unit-testable.
- It changes how many requests a host is *given*, never what may be *spent*.

## Decisions this story rests on (D67, D68)

1. Automatic adjustment in both directions, opt-in. **Supersedes §2.2's "never up" and D56's
   "operator only"**, for pools that enable it.
2. Launching at what the machine can hold, and starting at the profile's number — six by default —
   as a starting point rather than a cap (D68). **Amends D45.**
3. Token accounting in the request log, through the engine plug-in interface.

## The owner's answers

1. **Launch with room above the starting number:** yes — it is what makes raising free.
2. **The number itself:** six is the default, not the cap. A host climbs past it on evidence,
   bounded by what the machine can hold (D68).
3. **Rented hosts only** at first; an owner's machine is shared with their own work.
4. **This replaces the one-off measurement** of what each hardware profile can hold, which was
   asked for and never completed: the controller answers it continuously, per machine.

## Build stages

1. Token accounting: the engine plug-in reads usage from the final frame; the log stores it.
2. The controller as a pure function, with recorded traces from the live pool as its test data.
3. Launch at the memory ceiling or `max`; start at six; slots move beneath the launch bound;
   decisions logged; console shows the state.
4. S4's facts as extra signals; results written to the machine history.

## Tests

Recorded and synthetic traces through the pure controller: a machine with headroom climbs and
settles, a starved machine steps down and stays there, a noisy signal does not oscillate, the
bounds hold. Against the fake engine, a step down drains and a step up takes effect with no
restart.

## Depends on

**S4** for the machine's own facts and for any raise above the launch bound. The slots-only
controller (stages 1–3) can be built and used before S4 lands.
