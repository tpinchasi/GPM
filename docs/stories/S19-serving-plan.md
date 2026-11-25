# S19 — A serving plan per workload

> Status: **decided** (2026-11-24), revised after an architecture review and a UX review the
> same day, and again after a second pair on 2026-11-25: D198–D204, written into
> [spec/](../spec/) (workloads §14). Not built. Part of the [feature list](README.md).

## The story

The owner, 2026-11-24: "I want to be able to create a complex tiering logic when creating an app.
We already support that, but I want to take it to the next level. For instance, I can say that I
want rentals, but as a stop-gap till they come up I want local or serverless cloud." And: "maybe
we also need to provide a local chat that can create SDK code with the app tiering schema."

What a workload can say today has a fixed shape. It has a **base**: `parallel` answers at once
within `latency_s`, on machines rented for it. It has a **hosted burst**: after a request waits
`after_s`, it goes to a hosted API, within a budget. And it has a **rented burst**: after the
load stays above the base for `after_min`, more machines are rented, within a budget (D165).

The stop-gap the owner describes exists only by accident. While a workload has no ready host of
its own for a model, it may borrow up to 25% of the shared workload's machines (local, fixed or
rented; D115). A workload that may not borrow, or finds no free machine, goes to its hosted burst
at once, or is refused `503 workload_preparing`. Nothing lets an app say "this is my stop-gap,
use it here and only here". After an eviction, or while a scale-up rents, nothing covers the gap.

## What a plan should be able to say

The owner's example and nine more, gathered before the design:

1. **Stop-gap until rentals are up.** Rented machines are the goal. Local machines carry the
   load meanwhile, then a hosted API, and the stop-gap ends once the rented machines serve.
2. **The same gap after an eviction.** A spot machine is lost, and a hosted API covers until its
   replacement is ready.
3. **Spot with a fallback.** Rent interruptible machines; after three evictions in a day, switch
   to on-demand machines for the rest of that day.
4. **Requests kept on the pool's machines.** Some requests never go to a hosted API, and some
   stay on machines the operator owns.
5. **Priority.** Interactive requests may go to a hosted API at once. Batch requests never pay
   per token.
6. **Spending more carefully as the budget runs low.** At 80% of the month's hosted budget, stop
   using hosted APIs.
7. **Time of day.** Rented machines in working hours, the operator's own machines at night.
   Eight rented machines for a nightly batch, given up after it.
8. **A hosted failover chain.** One hosted API first; another when the first is busy or out of
   money.
9. **A guaranteed share of the operator's own machines.** This app always has the first call on
   four of their workers.
10. **A smaller model as a last resort.** Opt-in, because it changes the answers.

## The design

A workload's `serve` says **where its requests go and when**. How much the workload must serve
stays where it is today: `parallel`, `latency_s`, the kind of machine, per model in `models=`.
The owner's stop-gap is one line:

```python
provisioner.workload("llama-3.3-70b", latency_s=4, parallel=32, hours=24,
                     max_spend=200,          # machines only; hosted budgets come on top
                     serve={"until_ready": ["local", {"hosted": {"budget": {"per_day": 10}}}]})
```

`local` is the operator's to give: a program needs it in its grant, while `shared` (any free
machine of the pool) works with today's grants. Rented but never interruptible is the workload's
kind, which every rented source follows:

```python
provisioner.workload("llama-3.3-70b", latency_s=4, parallel=32, hours=24, max_spend=300,
                     machines="on_demand")   # the default, roi, may rent interruptible machines
```

A fuller plan:

```python
provisioner.workload("llama-3.3-70b", latency_s=4, parallel=32, hours=24, max_spend=300,
                     machines="interruptible", serve={
    "until_ready": [
        "shared",
        {"name": "hosted-cover", "hosted": {"providers": ["hosted-a", "hosted-b"],
                                            "budget": {"per_day": 10}}},
    ],
    "when_full": [
        {"name": "hosted-burst", "hosted": {"after_s": 5,
                                            "budget": {"per_day": 20, "per_month": 300}}},
        {"name": "rented-burst", "rented": {"after_min": 10, "max_hosts": 4,
                                            "budget": {"per_day": 60}}},
    ],
    "substitute": [{"model": "llama-3.1-8b", "after_s": 20}],
    "tz": "America/New_York",
    "rules": [
        {"name": "interactive at once",
         "when": {"priority": "interactive"}, "then": {"after_s": {"hosted-burst": 0}}},
        {"name": "careful spending",
         "when": {"spent_over": {"budget": "hosted-burst", "period": "month", "share": 0.8}},
         "then": {"never_use": ["hosted-burst"]}},
        {"name": "on-demand after evictions",
         "when": {"evictions": {"count": 3, "within_h": 24}},
         "then": {"steady": {"rented": {"kind": "on_demand"}}, "until": "end_of_day"}},
    ],
})
```

**The three roles, in one order.** A request tries **steady**, then **until_ready**, then
**when_full**, then a **substitute** if the plan names one, then the queue.

- **steady** is where the workload runs. It is the workload's own rented machines unless the
  operator gives it reserved `local` or `fixed` workers, sized as today (D115).
- **until_ready** is the stop-gap. It is used only while a machine of the workload is actually
  being prepared: at the start and while an evicted or failed host is replaced; while a scale-up
  rents only if the plan says so. It is never used for longer than `max_min` per gap. It ends as
  soon as the machines it covered for serve.
- **when_full** is used once the steady machines are all busy. It is today's bursts, as an
  ordered list.
- **A role left out is empty**, and the readback says so. With no `serve` at all, the workload
  runs exactly as today.

**Sources.**
- `local`: the computer the pool runs on.
- `fixed`: machines the operator declares at a fixed address: their own elsewhere, such as
  on-prem servers, or a server they lease.
- `shared`: any machine of the shared workload, which is today's borrowing.
- `rented`: machines rented for the workload, of the kind the steady machines rent unless it
  names one, so `machines="on_demand"` is never interruptible anywhere.
- `hosted`: hosted APIs, paid per token, in the order `providers` gives.

Each source can be given a name, and rules refer to it by that name. Every source that spends
money carries its own budget, and the pool keeps each budget apart. All the hosted budgets
together are also one total, so changing a plan to rename a budget gives nothing back.

**Rules** are written `{name, when, then}`, from a closed list; no plan can run code.
- **Conditions:** a request's priority or `keep_on`, a time window, a budget's share spent, and
  evictions.
- **Effects:** use only some sources, never use some, shorten a named source's wait, switch the
  steady source (until the window or the day ends), or turn off the substitute.
- **What a rule can never do:** add a source the plan does not name, raise a budget, or change
  renting because of one request's priority.

**Two request options**, from optional headers or SDK arguments, each on a fixed list:
- **Priority**, `X-GPM-Priority`: `interactive` or `batch`. It chooses which sources the plan's
  rules let a request use, and never moves it ahead in the queue. A `batch` request never
  reaches a hosted API, and is queued like any other request.
- **Where a request may go**, `X-GPM-Keep-On`:
  - `no_hosted`: never a hosted API.
  - `owned`: only the machines the operator declared (`local` and `fixed`).
  - `any`: whatever the plan allows.

Both hold by construction, whatever a plan says. A plan's `defaults` gives each option a request
does not send; `any` lifts the default, never past the plan. A value not on the list is refused,
so a misspelt option never passes as none.

**Every answer says how it was served.** `X-GPM-Route` (`until_ready/local`, say) and
`reply.route` in the SDK. A substituted answer also carries `X-GPM-Substituted-For`. The
workload's page has a **Now** line, for example: "Covering until ready: 0 of 2 rented machines
ready (about 6 min; stop-gap 54 min left). Next request: local. local 3 of 4 busy; hosted-cover
$2.10 of $10 today. Rules in force: none." A refusal names the rule behind it.

**Who does what.** The supervisor evaluates the plan every pass and writes each workload's
routing order into SQLite, per model and per request option, only when it changes, with a time
after which it is stale. The router reads it and judges for itself whether each machine is
ready, whether a worker is free, and whether a budget has room. An order gone stale (the
supervisor stopped) keeps the workload serving on its steady and `when_full` sources, opens no
paid stop-gap, and keeps every restriction in force. The two processes never call each other.

**Seeing it before it spends.** `gpm workload plan` and `POST /pool/workloads/plan` read the
plan back in plain words, then give:
- a **timeline** over the lease's hours, in clock time;
- the **conditions that can change it**, each with its effect;
- the **most it can spend**: the lease's cap (`max_spend`, all machines) plus every hosted
  budget over the days and months the hours touch, as one total;
- a warning wherever a request option leaves a gap. For example: "Requests with no options keep
  off hosted APIs here, so while the rented machines start they use local, then wait."

**Writing a plan.** The console's plan builder and a chat, built last (stages 7 and 8):

- **The builder** has four plain toggles: cover start-up and replacements with…; when full, also
  use…; keep some requests off hosted APIs…; at certain hours…. Rules go under Advanced, and a
  JSON tab edits both ways. Beside it are the readback and the timeline, and code tabs (the SDK
  for a program, the command line and HTTP for an operator) filled from a template, never by a
  model. A running workload gets **Change serving plan**, current and proposed side by side.
- **The chat** turns a description into the plan's JSON. It runs on a chat model the pool serves
  (never a hosted-only one) and is held to the schema by structured output. Its proposal is
  shown as a change to the builder, with Undo. It only proposes: Create is the builder's button,
  with the budget typed again.
- **The chat's request travels through the database.** The control API runs in the supervisor,
  so it writes the request to the database, and the router's watcher sends it inside the router
  and writes the answer back: a program's provisioning request (D117), in reverse.

## The owner's answers (2026-11-24)

- **Request classes**: yes, from optional headers on a fixed list ("this sounds good"). After
  the UX review, split into two: priority, and where a request may go ("ok").
- **`batch`** never uses hosted APIs and waits no longer than any other request ("ok"). Longer
  waits would be a design of their own.
- **A smaller model as a last resort**: "add the option".
- **The builder and the chat**: built together, and after the reviews, last ("a").
- **Sources stay `local`, `fixed`, `rented`, `hosted`**: "keep as is, as in future I might want
  to add on-prem compute which is compute owned by me, but not local computer". `fixed` is that
  compute.
- **Reserving the operator's own machines** stays the operator's alone; a program cannot.

After the second architecture and UX reviews (2026-11-25):
- **Rented but never interruptible**: the workload's `machines="on_demand"`. A rented source that
  names no kind follows the kind the steady machines rent in force, so a rule that switches to
  on-demand stops new interruptible machines altogether ("steady kind in force").
- **Defaults are per option**, and `X-GPM-Keep-On: any` lifts a plan's default, never past the
  plan ("per option + any").
- **The stop-gap covers the start and replacements by default**; scale-ups only where the plan
  says so ("start + replacement").
- **One hosted total** per workload, so renaming a budget gives nothing back ("one hosted
  total").
- **Shared requests behind a reservation count as the shared workload's load above its
  unreserved workers**, so it still rents for its own demand ("count above unreserved").
- **A role left out of a plan is empty**, and the readback says so ("empty, said in readback").

## What it gives up

- **A request option is the app's word.** An app that sends nothing gets the plan's defaults. A
  plan whose default is `keep_on: no_hosted` makes forgetting safe, but an app may send `any`.
  To protect requests without relying on the app, use a workload with no hosted source at all.
- **`no_hosted` still allows rented machines**, whose operators can read what reaches them (T7).
  `owned` does not, though a leased `fixed` server is only as private as its provider.
- **Request priority is not a place in the queue.** It changes which sources a request may use,
  never its turn.
- **A substituted answer is another model's.** It is opt-in, marked on every answer, logged once
  by the SDK, refusable per call, and left out of the latency figures.
- **The rules are a closed list.** A condition the list lacks waits for a decision.
- **A time window takes effect at the supervisor's next pass after its boundary**, never early.
  For machines ready at 02:00, start the window earlier.
- **The stop-gap is bounded.** `local`, `fixed` and `shared` are the shared workload's machines,
  within `borrow_share` unless the operator reserves workers. A stop-gap with nothing free is the
  queue. It is used only while a machine is actually being prepared, and for at most `max_min`
  each time; a large model whose machines take longer than that to be ready queues after it, and
  the plan check warns of it.
- **A plan is one per workload.** It applies to each of the workload's models. A model served by
  hosted APIs alone keeps its own path (D165).
- **Not for the shared workload yet.** Requests on the app key alone keep `pool.bursts.hosted`.
  The two request options do apply to them.
- **The chat needs a chat model on the pool**, and a small one will sometimes misread a
  description. The readback shows what it understood before anything is created.

## Build order

1. **The plan, its shorthand and the SDK's `serve=`** — **built** (gpm-server 0.46.0,
   gpm-client 0.10.0). The plan model and its checks. Today's fields compiled into it, proven by
   a test that routes every existing policy shape the same, with the supervisor up or stopped.
   The stop-gap for the start (judged by the router) and for a replacement (marked by the
   supervisor in `serving_gaps`, read with its heartbeat), with `max_min` and the stale fallback
   — brought forward from stage 3, so the plan's default `during` works from the first stage. A
   program's grant over sources. One hosted and one rented source. Every field of a later stage
   refused in words.
2. **The two request options** (`X-GPM-Priority`, `X-GPM-Keep-On`) — **built** (gpm-server
   0.47.0, gpm-client 0.11.0) — enforced in the router's choice of eligible hosts, with a plan's
   `defaults`. The request log's new columns, `X-GPM-Route`, stats by priority and by route (a
   table of their own, so the first stats table keeps its key), and `/pool/status` advertising
   them. The console's dashboards gain no tab for them until the console stage.
3. **Scale-ups where a plan lists them, and a budget per named source** under one hosted
   total — several hosted sources, `providers`, a rented source's `kind` — with the request log
   recording which budget paid (today's rows counted under the shorthand's budget) — **built**
   (gpm-server 0.48.0).
4. **Reserved workers** on the operator's own machines — **built** (gpm-server 0.49.0), with
   `steady: local|fixed` (no rented burst above it until rules, stage 5).
5. **Rules**: time windows, budget shares, evictions — **built** (gpm-server 0.50.0), with
   `use_only`, `never_use` and `after_s`.
5b. **A rule's steady switch**, handed over make-before-break and priced with its overlap — split
   from stage 5 (2026-11-25): it changes what a workload rents, and is built and reviewed on its
   own — **built** (gpm-server 0.51.0).
6. **The plan check's readback, timeline and limit**, the JSON Schema, the CLI (`--serve`), the
   SDK's `budgets()`, and the Now line — **built** (gpm-server 0.52.0, gpm-client 0.12.0).
7. **The console's plan builder.**
8. **The chat.**
9. **The substitute model.**

Each stage is tested against the fake provider and fake engines, as every story is.
