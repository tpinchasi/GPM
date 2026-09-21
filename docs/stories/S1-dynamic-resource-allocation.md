# S1 — Dynamic resource allocation

> Status: **built (D66)**, 2026-09-21. Part of the [feature list](README.md). Off by default:
> `rented.allocation: dynamic`. The demand becomes what the traffic asks for — busy workers plus
> what is waiting, against a target utilisation — and hosts arrive in rounds of 1, then 2, then
> 4, each round waiting for the last to land. The lease stays the only spending authority and
> becomes the ceiling. Machines are still chosen by the configured offer rules, and every host
> in a round is re-checked against every cap on its own.

## The story

*As an operator, I switch on dynamic allocation for a pool, give it a budget, and the pool adds
rented hosts when requests are waiting and gives them up when they are not — choosing machines
by the same rules I already configured.*

## Why

Today **demand is the lease**: `overflow = lease.workers − ready workers in higher tiers`
([supervisor.md](../spec/supervisor.md) §5). The operator guesses a worker count in advance, and
the pool rents toward that guess whether or not the traffic needs it. Queue-driven scaling was
left for after v1 because the signal looked circular: a client that sizes itself from current
capacity never builds a queue.

Measured on the first live pool, the queue does form. In one 45-minute window every worker on
every host was busy, requests waited **12.3 s on average** before dispatch, and **4,138 requests
were refused with `queue_timeout`**. The owner answered by preparing hosts by hand, one at a
time, watching the console. That is the loop this story automates.

## What exists already

- The offer pipeline — filter, rank, bid, with reasons kept — and capacity profiles, the avoid
  list, the give-up rules for hosts that never start (D43–D45, D54, D55).
- Leases with mandatory dollar caps; caps re-checked after every strategy returns.
- Scale-up and scale-down windows, one host at a time, idle release (D58), drain (D53).
- The router already records queue wait per request and busy workers per host, off the request
  path.

Only the **demand signal** is missing. Everything downstream of "how many workers are wanted"
stays as it is, which is what keeps this story small.

## Design

**Explicitly enabled, and still inside a lease.** `rented.allocation: lease` (today's behaviour,
the default) or `dynamic`. A dynamic pool still rents nothing without an open lease that allows
renting and carries a dollar cap. The lease stops being the *demand* and becomes only the
*authority*: its `workers` is the most the pool may reach, its dollars and hours the most it may
spend.

**The signal answers the circularity objection.** Two measurements, both already in SQLite:

| Signal | Says | Why it is not circular |
|---|---|---|
| **Utilisation** — busy ÷ total ready workers, averaged over a window | The pool is saturated | A client sized to capacity keeps every worker busy. No queue forms, but saturation itself is visible |
| **Queue pressure** — requests waiting, the oldest wait, `queue_timeout` refusals | Work is being turned away | Present whenever clients do not size themselves, which is what was measured |

`wanted workers = ceil((busy + waiting) ÷ target utilisation)`, clamped to the lease's
`workers`. A host is rented when `wanted − ready` has stayed at or above the smallest useful
host for `scale_up_after_s`; one is given up when the pool would still cover `wanted` without
its largest host for `scale_down_after_s`. Those are today's two windows and today's surplus
rule, so hysteresis comes from mechanisms that already exist.

**The machine is chosen exactly as today.** Offer policy, ranking, bidding, capacity profiles,
the avoid list, mixed interruptible and on-demand renting: all unchanged. Dynamic allocation
decides *when* and *how many*, never *which* or *how much*.

**It stays a pure strategy.** `RentStrategy.decide` gains the measured fields in its `Demand`
argument; the supervisor gathers them and passes them in. No I/O, no clock, replayable.

**Which host goes first when shrinking:** idle before busy; among the busy, the most expensive
per unit of measured throughput once S5 supplies that number, the highest hourly rate until then.
A host leaving always drains first (D53).

## The ramp — the owner's answer to "how many at once"

Not one at a time, and not the whole gap at once. The scheme a widely used cluster framework uses
for its executors: start with one, and while the backlog lasts, ask for a multiple each round.

| Round | Adds | Starts when |
|---|---|---|
| 1 | 1 host | load has held for `window_s` |
| 2 | `ramp_factor` × 1 = 2 | round 1's hosts are ready or given up, `ramp_backoff_s` has passed, and the load is still there |
| 3 | 4 | the same, after round 2 |
| … | up to `max_round` | |

The ramp **resets when the load clears**. Waiting for the previous round to land is the
difference from multiplying on a timer: a host takes minutes to become ready, and doubling before
it has helped would buy capacity the last round was about to supply. A round that loses its bids
does not grow the next one. Every host in every round is chosen by the configured offer rules and
re-checked against every cap on its own — a round is a number of attempts, not a bulk purchase.

## Configuration sketch

```yaml
rented:
  allocation: dynamic            # lease (default) | dynamic
  dynamic:
    target_utilisation: 0.75     # rent before saturation, not at it
    window_s: 120                # how long a signal must hold
    min_hosts: 0                 # a warm floor kept while a lease is open
    ramp_factor: 2               # each round adds this many times the last
    ramp_backoff_s: 300          # and waits this long after the last round has landed
    max_round: 8                 # the most one round may add
```

## What it gives up

The operator no longer states the size of the pool; they state a ceiling and a budget. A
workload with a sharp, short peak will be served late, because a host takes minutes to become
ready — dynamic allocation cannot beat the download. `min_hosts` is the answer for a workload
that cannot wait.

## Safety

- Nothing rents without a lease; every cap is re-checked after the strategy returns.
- The ramp's first round is one host, preserving "a bad market yields one failed bid, not five";
  a round that loses its bids does not grow the next.
- A host is not rented unless the lease outlasts its start-up by a useful hour (unchanged).
- Every decision is logged with the numbers behind it; `gpm plan` shows what it would do now.

## Decisions this story rests on (D66)

1. Dynamic allocation, opt-in. **Supersedes "Demand is the lease"** (supervisor §5) for pools
   that enable it, and moves *queue-driven scale-up* out of the roadmap's after-v1 column.
2. The load signal is D64's, already built for bringing a paused host back.
3. Hosts are added in a ramp; one-at-a-time becomes its first round.

## The owner's answers

1. **More than one host at once?** Yes — gradually: one, then a multiple, again and again, with
   a back-off between rounds. That is the ramp above.
2. **Default target utilisation:** 0.75, as recommended; an operator's to change.
3. **A warm floor:** `min_hosts` exists and defaults to 0, so nothing is spent without traffic
   unless an operator asks for it.

## Build stages

1. ~~The measured signal.~~ **Built** — read from the request log the router already writes.
2. ~~Measured demand and the ramp.~~ **Built**, with the decision log carrying each round's
   numbers (`ramp_round`, `ramp_reset`).
3. Shrink ordering by measured cost, and a console chart of signal against capacity — **later**;
   shrinking today is D64's pause-then-destroy, idle hosts first.

## Tests

Scripted load traces against the fake provider and fake engine: a ramp rents, a plateau holds, a
drop releases, a spike shorter than the window rents nothing, and a lease cap stops a ramp.

## Depends on

Nothing. Improves with **S5**, which supplies measured throughput per host.
