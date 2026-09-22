# What a pool costs, and when it is worth it

> Measured on a live pool over three days, 2026-09-19 to 2026-09-22: ~30 rented hosts, 4.4 M
> tokens generated, a 26B-class model at Q4 on 80 GB-class cards through the `ollama` adapter.
> These are numbers from one workload on one provider. The *method* is the durable part; treat
> the figures as an order of magnitude, not a benchmark.

A pool rents machines by the hour. A managed inference API sells tokens. Comparing them needs
one number on both sides — **cost per million tokens generated** — and getting that number
right turns out to be easy to get wrong.

## How to measure it, and how not to

**Cost per million tokens = (price per hour × wall-clock hours paid) ÷ (tokens ÷ 1e6).**

Two mistakes, both made while producing this page:

1. **Summing `generate_ms` across requests and calling it machine time.** Requests run
   *concurrently*: six workers generating for ten seconds is ten seconds of machine time, not
   sixty. Summing per-request generation time overstated cost by about the worker count — a
   6× error, reported repeatedly before anyone checked it.
2. **Summing every row of the spend ledger.** Those rows are *cumulative running estimates*,
   not increments. The total is each host's **highest** figure, summed across hosts. Summing
   the rows gave $20,456 against a true $56.62.

A third figure is worth keeping separate: **throughput measured per stream** (a single
request's tokens ÷ its own generation time) is not **aggregate throughput** (all tokens ÷
wall-clock seconds). Per-stream flatters a host that serves one request at a time. Only the
aggregate figure is comparable with a managed service, and only the aggregate figure predicts
cost.

## What was measured

Best hosts, over their serving windows:

| | aggregate tok/s | $/h | $/Mtok |
|---|---|---|---|
| best host observed | 175 | 1.467 | **2.33** |
| a good host | 118–157 | 0.95–1.47 | 2.2–2.6 |
| a poor host of the same card class | 38–74 | 1.08–1.74 | 4.6–8.3 |

**Two machines of the same advertised card differed 3–4× in throughput**, and therefore in cost
per token, at similar hourly prices. Hourly price is a poor predictor of value; the cheapest
host measured was among the worst value. This is the evidence behind ranking hosts by
`$/h ÷ tokens per second` rather than by price (D75, not yet built).

A managed API for comparable models advertises on the order of **$0.20–0.90 per million
tokens**. So a well-chosen rented host on this stack is roughly **3–10× the price** of buying
the same tokens.

## Where the gap comes from

Ranked by size, largest first.

1. **The serving stack, and it dominates.** The best host reached **175 aggregate tok/s**. The
   same card under a batching server (vLLM, SGLang, TensorRT-LLM) is commonly reported at
   **1,000–3,000 tok/s** for a model of this size and quantisation. Continuous batching reads
   the model's weights once for a large batch instead of once per few streams, so tokens per
   GPU-hour rise by most of an order of magnitude. Nothing about *procurement* competes with
   that; it is the single biggest lever.
2. **Time bought versus tokens sold.** A rented host bills whether it is saturated or idle. A
   managed service bills its own utilisation and sells tokens, and at their scale a card is
   never idle because another tenant's request is always available to batch. Idle time on a
   pool is paid time.
3. **Preparation.** Every rental downloads the model set — ~28 GB in this workload — before it
   serves anything, and pays for the machine throughout. A host that serves for twenty minutes
   spends a large fraction of its life not serving. Parking a host (keeping its disk) exists to
   amortise this and is worth using wherever a lease will return.
4. **Failed rentals.** Machines that never start, never answer, or turn out to be unusable are
   paid for until they are given up. The filters (driver floor, per-GPU price, hardware
   exclusions) and the give-up windows exist to shrink this, and are worth tuning.

## When a pool is the right answer anyway

The economics above are honest: **for steady inference of popular models at moderate volume, a
managed API is cheaper, and no amount of better renting closes that gap.** A pool earns its
keep on questions price does not answer:

- **Models nobody hosts** — a private fine-tune, or a model no API offers.
- **Data that cannot leave** — the request never reaches a third party. This is a compliance
  answer, not a cost answer.
- **Capacity you control** — no per-token pricing, no rate limits, no queue behind other
  tenants, and the ability to hold capacity warm for a known burst.
- **Price discovery and independence** — a 4× spread between machines of one card class was
  visible in a single day's market; an API offers one price and one roadmap.

## What would close the gap

In order of expected effect:

1. **A batching engine adapter.** The engine is already a plug-in
   ([plugin-interfaces.md](spec/plugin-interfaces.md) §2); `ollama` is one implementation, not
   an assumption. An adapter for a continuous-batching server is the single change with a
   plausible 5–15× effect on tokens per GPU-hour. It needs the same closed set of operations:
   health, models resident, pull, load-and-pin, launch settings.
2. **Higher utilisation per host.** Worker counts are already measured per card (D67, D68,
   D88), but "busy" today means *worker slots occupied*, not *accelerator saturated*. With a
   batching engine the right target changes from slots to batch occupancy.
3. **Fewer paid-for minutes that serve nothing.** Shorter give-up windows, parking between
   runs, and buying multi-card machines whose per-card price is lower (D85) all reduce the
   denominator problem rather than the numerator.

## Reproducing these numbers

Everything above comes from the pool's own records; no extra instrumentation was added.

- **Tokens and generation time**: `request_log.tokens_out`, `generate_ms`, written by the
  router off the request path.
- **What was paid**: the `spend` ledger, taking each host's **highest** `reported` figure.
- **Serving window**: the `prepared` and `released` events bound the hours a host was useful;
  `rented` to `prepared` is the preparation that was paid for and served nothing.
- **Aggregate throughput**: a host's tokens divided by the **span** of its requests, never by
  the sum of their generation times.
