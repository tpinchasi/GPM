# S7 — Several models in one workload

> Status: **decided (D118) and built** (2026-09-30); `docs/spec/workloads.md` §12 is the
> specification. The architecture review replaced §3–§5's sizing and caps with a fixed split per
> host carried on each request, and closed six further gaps — see D118. Part of the
> [feature list](README.md). What follows is the design as first proposed.

## The story

*As an application that uses a chat model and an embedding model, I create one workload for my
run — one key, one lease, one budget — and both models are served on GPUs of their own. I say, per
model, how fast its answers must be and how many I send at once; the pool decides whether every
host holds both models or each model gets its own hosts, whichever costs less.*

## Why

A workload holds one model (D115). An application that uses two creates two workloads: two keys,
two leases, two budgets, two things to end, and no way for the pool to see that the two could share
cards. The shared pool already puts several models on every host it rents (one engine process per
model behind the agent's proxy, D96; the machine search sums their sizes, D111) — the limit is the
workload's own shape, not the machinery under it.

## The owner's answers (2026-09-30)

1. **Placement: the pool chooses per workload** — every host holding every model (*together*), or
   each model on hosts of its own (*apart*), by expected cost.
2. **A latency target per model.**
3. **Answers at once per model.**

## Design

### 1. The request

```
gpm workload create research --hours 6 --max-spend 25 \
    --model gemma4:31b:latency=20:parallel=16 \
    --model embed-small:latency=2:parallel=4
```

- A workload names **one to four models** (`workloads.max_models`, default 4), each with its own
  `latency_s` and `parallel`. Hours, budget, kind of machine, borrowing and the idle cutoff stay
  the workload's.
- A one-model workload is exactly what D115 built: the form and the SDK keep their `model=`
  shorthand.
- `placement: together | apart | auto` (default `auto`), as `kind` can be forced today.

### 2. Groups: what a placement is

A workload's hosts are divided into **groups**; a group is a set of models every one of its hosts
holds, and it is what scales.

- *together*: one group holding every model.
- *apart*: one group per model.
- Only these two are weighed. Partial splits (A and B together, C apart) are not: for three or
  four models they multiply the market searches for little gain, and nothing asks for them yet.

A group is what the fleet already calls a unit (D115): its own hosts, its floor, its load, its
scaling. The unit key becomes (workload, group); a host carries its group beside its workload,
published to the router like every routing field.

### 3. Choosing — by expected cost, reasons said

The plan sizes both placements and keeps the cheaper, using what D115 already computes:

- **together** — a card that fits all the models (the existing per-card sum, D111/D114). On it,
  each model's measured answers-at-once at its target, `w_m`. Hosts:
  `ceil(Σ_m parallel_m / w_m)` — each answer of model *m* takes `1/w_m` of a host.
- **apart** — per model, a card that fits it alone, its own `w_m` there, and
  `ceil(parallel_m / w_m)` hosts.
- Each is priced by the rental-kind rule (on demand or bid, evictions counted) over the workload's
  hours. The cheaper wins; a tie goes to *together* — any host answers any model, and load that
  shifts between models is absorbed by the same hosts. The plan shows both totals and why.

### 4. One host, several models, each within its target

Workers on a host are shared by every model it holds. On a *together* host nothing today stops the
big model taking every worker and the small one waiting past its target. So a host of a
multi-model group carries a **per-model cap**: at most `w_m` answers of model *m* at once (sum of
caps ≥ the host's workers is fine; the host's worker count still bounds the total). The dispatcher
skips a host for a model at its cap, as it skips a busy host. The supervisor publishes the caps
with the host; the router only reads them. A one-model host has no caps: nothing changes for it,
or for the shared pool.

### 5. Measuring latency where models share a card

The request log records the concurrency an answer was served at (D115): every answer in flight on
the host. On a *together* host that mixes models, so the curve for one model also carries the
other's load. Add the **same-model concurrency** to the log, and size `w_m` from it on hosts that
hold more than one model. The mixed number stays, for hosts that hold one.

### 6. Everything else, per group

- **Floor and reservation**: each group's hosts at start; reservations summed over groups.
- **Load**: a group's busy workers, and waiting requests for its models only.
- **Borrowing while preparing**: per model — a request for model *m* borrows a shared host while
  no host of the workload holding *m* is ready, and only where the shared hosts serve *m*.
- **State**: `serving` once every group has a ready host.
- **Warm volumes and sibling copies (D116)**: per group — a volume holds its group's models; a
  copy comes from a sibling in the same group.
- **Budget, extend, end, keys, certificates, idle-end**: the workload's, unchanged.

### 7. Router

The workload's models become a list. A request for a model outside it: `404`, said in words, as
for a model outside the pool today. Eligibility is already by model resident on the host; to it are
added the per-model caps (§4) and per-model borrowing (§6).

### 8. SDK and programs (D117)

```python
pool.workload(models={"gemma4:31b": {"latency_s": 20, "parallel": 16},
                      "embed-small": {"latency_s": 2, "parallel": 4}},
              hours=6, max_spend=25)
```

The grant's model list checks every model; `placement` may be sent. The `model=` form stays.

### 9. Console

The form gets one row per model — model, latency, answers at once — and "Add model". The plan shows
both placements, their hosts, cards and cost, and which was chosen and why. The table lists a
workload's models, and its hosts by group.

## What it gives up

- **A plan takes two market searches** instead of one for more than one model (cached, as plans are).
- **Partial splits** are not considered.
- **Together is pricier per card** (every card fits every model) and **apart has no shared slack**
  between models: the choice is made once, at creation, on the numbers of that moment — not
  revisited while the workload runs.
- **Per-model caps are a new dispatcher rule** — small, but on the request path, and so the part
  most worth reviewing.

## Supersedes

D115's "a workload is one model". Everything else of D115 and D116 stands, per group.

## Build stages (once decided)

1. Workload shape (`models` list, groups), migration of existing rows, the one-model shorthand.
2. Plan: both placements, the choice and its reasons; units per group; floors, load, reservations.
3. Router and dispatcher: model list, per-model caps, per-model borrowing; same-model concurrency
   in the log and in sizing.
4. Volumes and copies per group; SDK, provisioning, CLI; console form, plan and table.

Each stage testable against the fake provider and the fake engines.
