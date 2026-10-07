# Specification — Workloads

> Status: **built and deployed** (D115, D116, D117; 2026-09-29); **several models per workload
> built** (D118, §12; 2026-09-30), against the fake provider. The first
> provider (Vast) does not yet offer volumes or copies to the pool (§6): both stay off there until
> checked live. **Keeping models between hosts on a model volume** (D139, §6) is designed, not built.

A **workload** is a named unit inside one pool that owns one to four models (§12), a lease, its own rented
hosts, its own key, and its own sizing and machine choice. Several workloads run side by side in
the one router and the one supervisor, and are kept apart by construction: a workload's key
reaches only that workload's hosts (and, while they prepare, a bounded share of the shared
ones); a workload's lease pays only for that workload's hosts; nothing a workload does can
loosen a cap of the pool's.

The pool as configured before workloads existed is the **shared workload**: the configured
hosts, the hosts rented under leases opened with `gpm lease open`, reached with the app keys.
It is always there, and it needs no change to keep working exactly as it did.

## 1. What a workload holds

| | |
|---|---|
| `name` | Plain, unique in the pool; part of the host labels and the log |
| `profile` | The model profile its hosts are bought as (hosts-routing-capacity.md §3.1): one build per model for the rented engine, and with several models the profile holds them all. Named by the operator, or made from the catalog's first build of `--model` |
| `models` | One to four, each with its own `latency_s` and `parallel` (§12, D118). With one, the rows below are that model's |
| `latency_s` | The whole answer, at p95, measured where the router hands the response to the app — after buffering, for a host that buffers (app-contract.md §5.1). Time to first byte is not a target: buffered delivery already makes it the whole generation |
| `parallel` | How many answers at once it must serve: the lease's `workers`, the ceiling of its scaling |
| `lease` | Its spending authority: `workers = parallel`, `max_hours`, `max_spend`, `allow_rent`, opened by the create command and bound to the workload. Every rule of supervisor.md §2 applies; a lease that can rent must carry a dollar cap |
| `key` | Its own bearer key, `gpmw_…`, minted at creation, stored hashed, valid while the lease is open |
| `hosts` | The rented hosts bought for it. A workload owns no configured host; local and fixed hosts belong to the shared workload |
| `state` | `preparing` (some model without a ready host of its own; that model borrows) → `serving` → `ending` (lease closed or expired; draining) → `ended` |

## 2. Creating one

```
gpm workload plan   research-run --model gemma4:31b --latency 20 --parallel 16 --hours 6   # spends nothing
gpm workload create research-run --model gemma4:31b --latency 20 --parallel 16 --hours 6 \
                    [--max-spend 25.00 | --confirm-max-spend <the proposed one>] \
                    [--profile chat-fp8] [--kind roi|on_demand|interruptible]
```

Also `POST /pool/workloads` with the same fields, admin key only. The command does, and prints,
these steps — and does nothing until the budget is confirmed:

1. **The profile.** `--profile` names one; otherwise a profile holding `--model` with the catalog's
   first build for the rented engine. Refused where no build exists (D98's rule). The model is one
   of the pool's set, or a **workloads-only** catalog entry (`workloads_only: true`): outside the
   set, so the shared hosts never fetch it and an app key is told it is not in the pool (404),
   never that it exists. `POST /pool/config/models` adds one with `workloads_only: true`.
2. **Workers per host at the latency.** The most answers one host of a card class may serve at
   once and still meet `latency_s` at p95: read from the request log for that (hardware, model)
   where the pool has served it at that concurrency, else the card's capacity profile number with
   the target marked **unmeasured** in the plan. It is what the host runs, and automatic
   adjustment (D67) does not climb past it.
3. **Hosts at the start** `= ceil(parallel ÷ workers per host)`.
4. **The budget.** `--max-spend` typed, or derived: hosts × hours × the hourly price of the first
   host the plan would rent × 1.25 — with several groups, the sum over them of each group's
   hosts × its first host's price, × hours × 1.25 (§12) — shown, and accepted only by sending it back
   (`--confirm-max-spend`, or typed again in the console), as every loosening is.
5. **The lease** is opened, bound to the workload. Its hosts are searched for with the profile's
   card and disk minimums (D111) and the reliability floor `workloads.min_reliability`; download
   speed enters through the rental-kind rule (§5), which prices a slow machine's replacement.
6. **The key** is minted and shown **once**, with what the app owner needs:

   ```
   base_url:  https://pool.example.net:8090/v1
   api_key:   gpmw_…              (shown once; `gpm workload rotate-key <name>` mints another)
   ```

   Where the listener's certificate is not a public one, the client must also be given its CA.

7. **The plan**: hosts at the start, their kind and price, the budget, and when the workload
   can serve — the time to ready of a host of that class, from the history where there is one.

A workload whose start would cross a pool cap — `max_rented_hosts`, `max_hourly_burn`,
`workloads.max_open` — is refused at creation with the numbers, not started and starved.

## 3. The key and the entry point

- **One listener, the key picks the workload.** Every app dials the same URL. A `gpmw_` key is
  looked up by its hash in the `workload_keys` table (owned by the supervisor, read by the router
  as it reads the host table); the request is then that workload's. An app key is the shared
  workload's, as before. An unknown key, or a rotated-out one past its grace, is `401`.
- A path prefix `/w/<name>/v1/…` may be used and means the same; it is refused (`403`) when the
  key is another workload's, so a misconfigured client fails loudly rather than serves quietly.
- **The key reaches nothing else.** Not the control API, not the console, not the shared hosts
  beyond §4's borrowing. The admin key manages workloads; an app key cannot reach a workload's
  hosts. Off loopback the listener requires TLS, as always (app-contract.md §3). A client
  certificate per workload (mTLS) is a later option, not a requirement: most chat front-ends
  cannot carry one, every one can carry `base_url` + `api_key`.
- `gpm workload rotate-key <name>` mints a second key; the old one is valid for
  `workloads.rotation_grace_minutes` (default 30), so a running app is switched without a gap.
- From the lease's end — its time run out, whether or not the supervisor has noticed yet — or
  once the workload is `ending`, requests are refused `503 workload_ended`, with no
  `Retry-After`, and the app is told so in words. The key still identifies the workload so that
  it can say so.
- `GET /pool/status` with a workload key reports **that workload's** view — its hosts, its models,
  the delivery policy (the SDK sizes its budget from it, unchanged), its state and whether it is
  borrowing — never the whole pool.

The client SDK needs no change: a workload is `PoolClient(base_url, api_key)`.

## 4. Routing and borrowing

Routing (hosts-routing-capacity.md §5) gains one step before the tiers: **the request's
workload decides which hosts are eligible at all.**

- A workload's request goes to that workload's `ready` hosts that hold the model. They are all
  rented, so they sit in one tier; the least-loaded takes it; all busy → the queue, first come
  first served, as before.
- **Borrowing, only while preparing.** While the workload has no ready host of its own holding the
  requested model (per model, §12), requests for that model may go to the shared workload's ready
  hosts that serve the model — after the shared
  workload's own requests, never before them, and never more than `borrow_share` (default 0.25)
  of the shared workload's ready workers at once — **for every borrowing workload together**, so
  three workloads starting at once cannot take three quarters — rounded down, at least one where
  the share rounds to nothing. A borrower does not take a shared worker that a queued shared
  request could use; a shared request that stops waiting (served, timed out, gone) wakes the
  borrowers it held back. The shared workload does not scale up for lent work: a host's borrowed
  workers are published apart from its own. The moment one host of the workload holding the model is
  ready, borrowing for that model stops and its requests queue on the workload's own hosts: a
  workload that could always borrow would have no reason to be sized.
- The shared workload never borrows. Two workloads never see each other's hosts.
- Every request is logged with its workload; the borrowed ones are marked, so the shared
  workload's operator sees what was lent and the workload's owner sees what was borrowed.

## 5. Sizing, scale-up and the machine choice

**Start, and a floor.** A workload's `hosts at the start` is a floor for its whole lease: while
it has fewer (at creation, after an eviction, after one is given up) the gap is rented at once,
each host through the whole offer pipeline and re-checked against every cap on its own
(supervisor.md §6.1). No traffic is needed for this — a workload with no host has none to show.
Idle hosts are never reaped below the floor. This amends D66's "the first round is one host"
for workloads; the shared workload is unchanged.

**Scale-up** is dynamic allocation (supervisor.md §5.1) inside the workload's lease: measured
load — its own hosts' busy workers and its own requests that waited — above the floor, then the
ramp, up to `parallel` workers and the lease's dollars and hours. A workload host's workers are
fixed at what meets the latency on its class of machine: automatic adjustment (D67) does not
climb past it. The number is worked out per offer when a host is rented; the one the plan shows
is for the first host's class.

**The machine choice** is the offer pipeline per workload, with one added, deterministic stage
between the rule ranking and the advisor (D69's order, D70's place): **the rental-kind rule**.
For each machine the rules accepted it computes, for the hours left in the lease `H`:

```
cost_on_demand = on_demand_all_in × H
cost_bid       = bid_all_in × H  +  P_evict(H) × (time_to_ready × bid_all_in + lost_work)
```

- `P_evict(H)` is the machine's eviction rate per rented hour from the machine history (D69),
  or `workloads.eviction_prior_per_hour` where the machine has none;
- `time_to_ready` is measured for the machine or the class where the history has it, else the
  model's size over the offer's download speed plus the engine's load time;
- `lost_work` is the answers in flight on one host, priced at their mean generation time — a
  host with siblings loses a share of the workload's throughput, a lone host loses all of it.

An offer is known by its id *and* its kind: a provider may list one machine's bid and its fixed
price under one id (D127). It chooses the cheaper kind and **returns its reasons** — both numbers and every input — and
the supervisor clamps and re-checks as it does for every strategy. `--kind on_demand` or
`interruptible` at creation fixes the kind instead; `roi` is the default. The owner's rule of
thumb — on-demand for the first host, bids for the scale-up — is what the numbers usually
give, not a rule of its own: the first host carries the whole run and nothing shares its work.

The advisor (S3), when it exists, chooses among the same candidates after this stage, with the
same reasons in hand, and may decline them all inside its bounds. Nothing here changes it.

## 6. The models on a new host (D116)

Preparing a host means getting its models onto its disk. Three ways, tried in this order, each a
provider capability the fake provider implements and each recording what it did and how long
it took, so the history shows what each saved:

1. **A model volume** (D139). Where the operator has turned on **Keep models between hosts** for
   a provider account (providers.md §1) — offered only by a provider whose storage reaches a data
   center — the pool creates a volume with a workload's first host there, in that host's data
   center, labelled with the pool and workload (`gpm/<pool>/<workload>/models`; with several
   groups, one per group, `…/models-<n>`, holding that group's builds — §12). It is the workload's
   own: its storage is spend, estimated into the lease's ledger at the provider's rate while it
   exists and counted when a budget is checked, and it is deleted when the workload ends (the sweep
   deletes any volume under the pool's label whose workload is gone).
   - **The engine never runs from it.** Every host keeps its models on its own disk. The first host
     fills the volume after its own fetch, by copying its verified files up into a fresh directory
     that is renamed into place, named by its *build* — a digest of the hub's file list with each
     file's hash, so a repository changed upstream is a new build beside the old; only one host
     fills at a time, and another is named only once the provider shows the first gone. Every later
     host copies from the volume **by the hub's file list, each file checked against the hash the hub
     publishes**, and fetches from the hub whatever is missing or does not match (host-agent.md
     §2.1). The volume is a cache never trusted: a host that wrote junk into it harms no other host.
     A build is never changed in place: a new one is filled into its own directory beside it.
   - **The ranking** prefers an offer that lands in the volume's data center by no more than the
     download it saves, priced at the offer's hourly rate — from the pool's own times for the two
     ways, an estimate until it has them. A data center with nothing that fits is passed over, and
     those hosts fetch from the hub; nothing waits for it. A volume that no longer holds the hub's
     current build — its hosts report every file missing — is marked stale: not preferred, and the
     next host there fills the new build beside the old.
   - **With a volume, a host is released, not parked**: the models outlive it anyway.
   - **The workload's plan says it before anything is spent**, in the workload's own hours and
     budget, not per month: *“keeps a 40 GB model volume at runpod, in the first host's data center —
     $0.0039/h, $0.02 over 6 h, in the budget; deleted when the workload ends”*.
   - **Each host records where its models came from and how long it took** — from the volume, from
     the hub, or both with how many files did not match — on the host, in the feed and summed on the
     workload, so the operator can see whether the volume pays.
   - A provider whose volumes are bound to one machine (Vast) is not offered this: the machine is
     rarely free again, and the volume bills the whole time. D116's warm machine and
     `keep_models_on_machine` are retired (D139); the plug-in interface keeps the machine-bound kind.
2. **A copy from a sibling.** With a ready host of the workload, the provider copies the models
   directory from it to the new instance before the agent is asked to fetch. The agent's fetch
   then finds every file and the complete marker already there and holds the model at once.
   **The agent's protocol is unchanged**: the pool still names hub repositories, never a URL or a
   path; the copy is the provider's own operation between two of its instances, asked of the
   provider's API, with the pool's per-host key and never an account credential on any host.
3. **The hub**, as today.

A way that fails falls through to the next, with the failure in the event log. A copy that does
not finish within `workloads.copy_timeout_s` (at most half the preparing window) gives its host up
instead: a copy the provider may still be running is never raced by a fetch into the same place. Which way to
try is per workload (`workloads.model_sources`, default `[volume, sibling, hub]`; `warm` is read as
`volume`).

## 7. Lifecycle

| Command | |
|---|---|
| `gpm workload list` / `show <name>` | State, hosts, borrowing, spend against the cap, p95 against the target, hours left |
| `gpm workload extend <name> [--hours N] [--max-spend M] --confirm <the raised value>` | A raise of the lease, typed twice as D49 requires; the hold on every host follows |
| `gpm workload end <name>` | Closes the lease: hosts drain (`teardown.drain_timeout_s`) and are destroyed; volumes deleted; the key is refused from now |
| `gpm workload rotate-key <name>` | §3 |

The lease expiring is the same as `end`. A supervisor restart adopts a workload's hosts from the
host table as it adopts any rented host (D50); what each belongs to is in its record. **Any
host whose lease is not open** — however it closed: its caps, its time, an operator, a workload
ended, or closed while no supervisor ran — **is released on the next pass**: drained where it is
serving, destroyed where it has nothing in flight. Its spend is recorded under its own lease until
it is gone. An ended workload's keys still identify it, so its app is told `workload_ended`
rather than "unknown key".

## 8. Caps, across workloads

The pool's `max_rented_hosts` and `max_hourly_burn` bound the whole pool, every workload
counted, volumes' storage included; each workload's lease bounds itself; `workloads.max_open`
bounds how many run at once. The hosts a starting workload has yet to rent are **reserved**: the
shared workload may not take them, nor a workload created later; a workload never waits on one
created after it. Two creations at once are taken one after the other. A workload's lease is
changed only through the workload (`extend`, `end`): the generic lease API refuses it.
A machine rented by one workload is on every other's avoid list (D59). When the pool's caps
leave room for less than a workload's next host, that workload waits and says so; the shared
workload is not starved for a workload, nor a workload for the shared one: the room left is
taken in the order the leases were opened.

## 9. Configuration sketch

```yaml
workloads:
  max_open: 3
  borrow_share: 0.25              # of the shared workload's ready workers, while preparing
  rotation_grace_minutes: 30
  min_reliability: 0.95
  eviction_prior_per_hour: 0.10   # until a machine has a history of its own
  model_sources: [volume, sibling, hub]   # the volume where the provider account keeps models (D139)
```

## 10. What is refused

- A workload without a dollar cap, or whose model has no build for the rented engine.
- A workload key on the control API, the console, or another workload's path prefix.
- A latency target the pool cannot measure is **accepted** and marked unmeasured — a target
  is a wish until there is a log — but a `--parallel` above `limits.max_rented_hosts ×` the
  card's workers is refused with the arithmetic.
- Borrowing from anything but the shared workload; the shared workload borrowing at all.
- A workload's start that would cross a pool cap.

## 11. Workloads a program creates (D117)

A program holding a **provisioning key** creates its own workloads through the SDK, within a grant
an operator set (`gpm provisioner create`, or the console's Programs section):

```python
from gpm_client import WorkloadProvisioner

with WorkloadProvisioner(url, os.environ["GPM_PROVISIONING_KEY"]) as pool:
    with pool.workload("gemma4:31b", latency_s=20, parallel=16, hours=6, max_spend=25) as w:
        reply = w.client.chat("gemma4:31b", [...])
```

- **Four calls under `/pool/provisioning/`** on the router's listener: `POST requests` (a plan or a
  create), `GET requests/{id}`, `GET workloads/{name}`, `POST workloads/{name}/end`. The router only
  records a request; the supervisor answers it within `provisioning.request_poll_s` (a watcher, not
  a pass), one request per hold of the lock and a few per look, so a pass never waits behind a
  queue of them. The two processes never call each other.
- **One request at a time per key**, and at most 20 a minute (`429`): each may cost a market
  search the whole pool waits on. A create is refused before its search when the pool has no room
  for another workload at all.
- **The workload's key is made in the program**; only its SHA-256 is sent, and it is the create's
  identity: sent twice — its answer lost — it is one workload, and the SDK sends it again until it
  hears. A hash another key already sent is refused (`409 key_hash_taken`).
- **Bounded twice**: by the key's grant — workloads at once (ending ones count), dollars per
  workload, dollars committed in any 24 hours (the budgets of the workloads it made in that time or
  still has open, not what has been recorded; one that has ended counts what it spent — D125), hours, models, kinds of machine, whether it may borrow, its idle cutoff's maximum,
  an expiry — and by `provisioning.max_spend_per_day` across every key. A budget is always typed.
- **A host the budget cannot carry is never rented** (D124), and a workload whose budget carries none of
  the offers a round found stops searching until a host leaves or its lease is raised (D126). The rule:
  what the workload's hosts burn, with the new
  one at its price, over the hours its lease has left, must fit the dollars it has left.
- **Hours** stay within what the provider's dead-man timer covers (`max_hours_without_deadman`
  where it has none): a grant, a plan and an extension past it are refused, as for any lease.
- **Idle-end**: a program's workload that is serving and has had no request for its cutoff — 15
  minutes unless the create says otherwise, never above the grant's maximum — is ended. The clock
  starts at the later of its last request and the moment it began serving: a program waiting on a
  slow preparation is not idle. One that dies while its workload prepares is bounded by its lease.
- **Client certificates**: with `certs: required` on the grant, the SDK sends a signing request
  (`certs=True`, the `gpm-client[certs]` extra), the supervisor signs its public key only with the
  pool's client CA (`gpm ca create`; the key in `provisioning.client_ca_keyfile`, read by the
  supervisor alone), and the router — asking every client for a certificate against
  `listen.client_ca_certfile`, which must name the same CA, checked at start — refuses the
  workload's requests unless the connection presented the certificate whose fingerprint the
  workload holds. A CA key file others can read stops the start. The certificate is signed for the
  workload's hours, so hours are not added to such a workload. The private key stays on the
  program's machine: the SDK writes it to a file only its user can read (Python's TLS loads a
  certificate from a file), in a temporary directory removed when the workload ends; a killed
  program leaves it behind.
- A program's workload key is never rotated by the pool, which never had it; the program creates a
  new workload. Revoking a key stops new workloads at once; revoking it with `--end-workloads`
  (the console's "Revoke and end") also ends the ones it made.

## 12. Several models in one workload (D118)

A workload names one to four models (`workloads.max_models`), each with its own `latency_s` and
`parallel`; hours, budget, kind of machine, borrowing, the idle cutoff, the lease and the key are
the workload's.

```
gpm workload create research --hours 6 --max-spend 25 \
    --model gemma4:31b --latency 20 --parallel 16 \
    --model embed-small --latency 2 --parallel 4 [--placement auto|together|apart]
```

`POST /pool/workloads` takes `models: [{model, latency_s, parallel}, …]` and `placement` in place of
`model`, `latency_s`, `parallel`; the SDK takes `models={name: {latency_s, parallel}}`.

- **Groups.** A workload's hosts are divided into groups: a set of models every one of its hosts
  holds. *Together* is one group of every model; *apart* is one group per model. A host's group is
  the set of models it was bought for. A group is what scales — its own floor (its hosts at the
  start), its load and waiting requests (for its models only), its ceiling (its models' answers at
  once), its reservation, its parked hosts, its model volume and its sibling copies. The lease's
  `workers` is every model's answers at once together.
- **Placement.** `auto` prices both from one market search, re-priced and ranked per group, by the
  rental-kind rule (§5): the cheaper expected cost over the hours is kept; within 1% they are equal
  and together is kept. The plan shows both, and why. `together` or `apart` forces one; together
  is refused where no card on offer holds every model within its target.
- **The split.** A together host runs a fixed share of each model. With `w_m` how many answers of
  model *m* alone one host of that card serves within *m*'s target (§5), the plan takes the fewest
  hosts `H` for which caps `k_m = ceil(parallel_m / H)` keep `Σ k_m / w_m ≤ 1`; every host runs
  `Σ k_m` workers, and an offer whose card cannot hold the split is not rented for it. The router
  takes at most `k_m` answers of *m* on one host at once: a request past its model's share waits for
  one to finish, as a request waits for any busy worker; it is never refused for it. Load that moves
  from one model to another is not absorbed by the other's share.
- **Short of its hosts** (D128). While a group has fewer ready hosts than it planned, its shares are
  each model's target spread over the hosts it has — or, where that is more than a card holds, the
  card filled in the proportion the models asked for — never past what the hosts' engines were
  launched for; a host is launched with room for its card's whole split, so this restarts nothing.
  The planned shares return when its hosts do (`workload_resplit` in the decision log).
- **Measuring.** A model's latency curve is built from answers served with nothing else on the
  host — the request log records, beside the concurrency, how many of those answers were of the
  same model. Automatic worker adjustment leaves a split host alone.
- **Routing.** The key reaches the workload's hosts; the model picks among them. A model outside
  the workload is `404`, naming its models. A model whose group has no ready host yet borrows a
  shared host that serves it, where the workload may borrow, or is refused `503 workload_preparing`
  — even while another group serves.
- **State.** `serving` once every group has a ready host; until then the models without one borrow.
  The idle cutoff (§11) counts from then: a workload whose groups do not all come up is bounded by
  its lease.

