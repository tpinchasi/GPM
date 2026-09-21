# Specification — The Supervisor: Processes, Leases, Money, Rented Hosts

> Reasons for each rule are in [../decisions.md](../decisions.md) (D7, D14, D18–D21, D24).
> Items marked *post-v1* are specified here so the shape is known; v1 scope is in
> [../roadmap.md](../roadmap.md).

## 1. Process layout

The pool is **two processes**, because its halves have opposite needs.

| | Router | Supervisor |
|---|---|---|
| Job | In the path of every request | Rents, bids, tunnels, probes, recovers, tears down |
| Must | Never stall — its latency is the app's latency | Do slow, blocking, failure-prone work |
| Serves | The app-facing engine endpoint | The control API and the console |
| If it dies | Apps get transport errors; the SDK retries | **Routing continues** on the last published host table; local and fixed hosts are unaffected; rented hosts serve until they actually fail |

They share one **SQLite database (WAL mode)** and never call each other. The supervisor publishes
the host table, leases and decisions; the router reads the table and writes the request log and
per-host counters, which is how the supervisor learns about idleness and corrupt output. One
command starts both; each can be restarted alone. Exactly one supervisor runs per pool, enforced
by a lock. Configuration stays a human-editable file; state is the database. Key files, any
configuration that references secrets, and the database are created owner-readable only, and
the pool refuses to start if they are group- or world-readable.

### 1.1 Stopping is explicit — exiting destroys nothing

| What happened | Command | Rented hosts |
|---|---|---|
| Operator is finished | `gpm stop --release` | Drained, destroyed, verified |
| Restart for a configuration or code change | `gpm restart` / `gpm stop` | **Kept**, re-adopted on start |
| Crash, power loss, machine asleep | — nothing runs | The on-host dead-man timer (§7), then lease expiry, then the orphan sweep at next start |

There is no release-on-exit. It cannot be made reliable — a crash runs no exit code — and when
it does run it is usually wrong.

### 1.2 Source of truth

**The provider is the source of truth for what exists; the database is the source of truth for
what was intended.** On start the supervisor lists the provider's instances carrying this
pool's label first, then adopts those it intended and sweeps those it did not. A crash can
never leave a billing host that nothing knows about.

**An unanswered question is not the answer "nothing" (D61).** If the provider cannot be asked at
start-up, every rented host's record is kept exactly as found; adoption is retried on each pass,
and until it succeeds nothing is swept, rented or pruned. If rented capacity is configured and
the provider refuses the credential, or the process has none, the supervisor does not start: it
could neither adopt its hosts nor verify a destroy, and says what to set instead.

## 2. Leases — the only thing that can spend

```
gpm lease open --workers 12 --max-hours 8 --max-spend 5.00 --allow-rent
                # optional, tighten only:  --bid-ceiling 0.40
```

A lease is the unit of demand and of spending authority: wanted workers, a time limit, a dollar
limit. The models are the pool's own set. `--allow-rent` is what permits money to be spent;
without it the lease is served by local and fixed hosts only. **A lease that can rent must carry
a dollar cap.** Inside an open lease the supervisor may bid, recover and replace unattended;
with no lease, or a lease exhausted, rented hosts are drained and released. A lease may tighten
the pool's configured limits, never loosen them. An abandoned lease expires by its time limit.

**A rented host is watched while it comes up** (D54). `teardown.max_starting_minutes`: its
engine has never answered — stuck scheduling or starting — so it is ended early rather than
billed for the whole preparing window. `teardown.min_pull_mbps` over `slow_pull_grace_s`: its
download is far below what the offer advertised. Either way, and on a download that fails for
good, the **machine** is skipped for `avoid_failed_machine_minutes`, or the best-ranked offer —
the machine that just failed — is simply rented again.

**An open lease can be changed while it runs** (D49): tightened freely, or **raised** — more
hours for a host worth keeping, more dollars to pay for them, more workers — with the raise
confirmed by typing the new value again, as loosening is everywhere else. Extending the hours
also pushes out the hold on a host prepared under that lease, so the lease cannot outlive the
host it was extended for. A lease still cannot loosen a pool-level ceiling, and a closed lease
is never reopened: open a new one.

## 3. Control loop

Every pass (default 15 s):

1. **Observe** — list provider instances; probe every host (health, models resident); read
   provider status so "outbid" is told apart from "connection dropped"; read provider-reported
   charges.
2. **Update host states** ([hosts-routing-capacity.md](hosts-routing-capacity.md) §1.4).
3. **Compare** lease demand with ready and pending capacity.
4. **Act**, in strict order: release what should not exist → recover what is broken → acquire
   what is missing. Acquisition is refused when any cap would be crossed.
5. **Record** every decision as an event **with the numbers that produced it**.

`gpm plan` prints what the supervisor would do and spends nothing.

## 4. Cost controls

Ordered by value — idle time, not the hourly rate, dominates cost.

| Control | Rule |
|---|---|
| Idle release | A rented host with nothing routed for `idle_minutes` (default **2**, D58) is drained, then parked if a lease is still open, else destroyed |
| Lease expiry | Time or dollar limit reached → drain → destroy. The budget is a hard stop |
| **Spend is reconciled, not just estimated** | The pool's own figure (bid × time) misses storage while stopped, per-gigabyte download charges and provider rounding. Each pass the supervisor also reads **provider-reported charges** and records both. Caps are enforced against **whichever is higher**, less `cap_safety_margin` (default 10 %), so a lease stops *before* its limit. A gap above a threshold is logged and shown in the console. A provider that cannot report charges must declare so; the margin is then widened |
| Orphan sweep | Any provider instance carrying this pool's label that the database does not know → alert, destroy after a grace period. Covers instances that bill storage while never running |
| Verified release | A release counts only once the provider's listing no longer shows the instance; retried with back-off |
| Rate caps | Maximum rented hosts at once; maximum hourly burn; per-offer bid, all-in and download-price ceilings |
| Replace only when it is cheaper | Re-bidding in place beats replacing whenever the download cost of a new host exceeds the price difference over the hours left (§6) |
| Spend ledger | Append-only cost events per host and lease |

## 5. Rented hosts — when to rent

Everything in §5–§9 applies only to rented hosts, inside a lease that allows renting. Each
behaviour is a named, swappable strategy ([plugin-interfaces.md](plugin-interfaces.md) §3).

**Demand is the lease.** `overflow = wanted workers − ready eligible workers in higher tiers`.
*(Scaling from observed queue depth is post-v1: a client that sizes itself from current capacity
never builds a queue, so the signal is circular unless clients size from the lease instead.)*

A host is rented when all of these hold:

| Check | Default | Why |
|---|---|---|
| Overflow has persisted | 120 s | A short burst is not worth a multi-gigabyte model download |
| The lease outlasts the start-up | ≥ 1 useful hour left after the estimated time to ready | A host ready as the run ends is pure cost |
| Caps allow it | hosts, hourly burn, lease dollars ≥ start-up cost + one hour of burn | §4 |
| Nothing is already on its way | no host `scheduling` or `preparing` | **One at a time** — a bad market yields one failed bid, not five |

### 5.1 Dynamic allocation (D66)

Opt-in: `rented.allocation: dynamic`. The lease stays the only spending authority and becomes
the ceiling; the **demand** is measured — the load signal of §9 — instead of read from the lease.

Hosts are added **gradually**. The first round adds one. If the load is still there once that
round's hosts are ready or given up, and `ramp_backoff_s` has passed, the next round adds
`ramp_factor` times as many — 1, 2, 4 … up to `max_round` — and the ramp resets when the load
clears. Waiting for the previous round matters: a host takes minutes to become ready, and
doubling before it has helped buys capacity the first round was about to supply. Every host in
every round is chosen by the offer rules of §6 and re-checked against every cap on its own; a
round that loses its bids does not grow the next. Shrinking is §9's pause-then-destroy.

**The demand is measured, not declared**: `wanted = (busy workers + waiting requests) ÷
target_utilisation`, clamped to the lease's `workers`. Waiting is counted from requests that
queued or were refused for queuing; saturation is watched beside the queue, because a client
that sizes itself to the capacity it can see never builds one. Load must *hold* for `window_s`
before the first round — a burst shorter than a model download is pure cost.

```yaml
rented:
  allocation: dynamic            # lease (default) | dynamic
  dynamic:
    target_utilisation: 0.75     # rent before saturation, not at it
    window_s: 120                # how long load must hold before the first round
    ramp_factor: 2               # each round asks for this many times the last
    ramp_backoff_s: 300          # and waits this long after the last round landed
    max_round: 8
    min_hosts: 0                 # a warm floor, kept while a lease is open
```

Count: `ceil(overflow / workers of the chosen offer)`. Each host must hold the pool's whole
model set. **Parked hosts are tried first** (§8).

## 6. Rented hosts — bidding, holding

### 6.1 Bidding pipeline

1. **Hard filters** — memory, memory-bandwidth band, provider verification, reliability,
   download speed, disk, excluded hardware, per-gigabyte download-price ceiling, all-in hourly
   ceiling, "the model set fits at the pool's context length", and the machine avoid list —
   which always includes every machine the pool already rents: it is still listed, to be outbid,
   and the tenant it would outbid is the pool (D59).
   **Never relaxed unattended** — an empty result means stay paused. The filters are applied by
   the pool, not pushed into the provider's own query, so every rejected offer carries the
   reason it was rejected; a market that merely *looks* empty teaches an operator nothing.
2. **Rank** by throughput proxy per run-dollar, where run-dollars = bid + storage + model
   download amortised over the lease's expected hours. *(Post-v1: warm-disk bonus for machines
   holding a parked instance; per-machine memory of evictions and failed starts; avoid list.)*
3. **Price the bid** with a named strategy:

   | Strategy | Bid | In v1 |
   |---|---|---|
   | `floor_plus_premium` | market floor + an absolute premium. A multiplier is wrong here: floors span an order of magnitude or more, so any fixed multiple is either free or wasteful | **yes — default** |
   | `volatility_aware` | premium scaled by the observed swing of that machine's floor | post-v1 |
   | `fraction_of_on_demand` | a fixed fraction of the same offer's on-demand price | post-v1 |

   Every strategy is clamped by **two ceilings**: the configured `bid_ceiling`, and
   `on_demand_crossover` (default 0.8) × the on-demand price of an equivalent offer. Past the
   second an interruptible host has the eviction risk without the discount.
4. **Place and confirm** — ask the provider to fail rather than park a losing bid; label the
   instance `<pool>/<host_id>`; wait for running; if it sits stopped with nothing pending for
   60 s the bid lost — destroy and verify. The floor moving between search and create is normal:
   re-read the market and retry, up to three attempts.
5. **When nothing works**, in fixed order: next-best offer → wait and search again →
   *(post-v1: on-demand, only if the lease allows)* → answer requests `503 no_offer`, still
   re-checking the market.

### 6.2 On eviction

Choose by cost over the hours the lease still has, not by habit:

| Option | Cost | In v1 |
|---|---|---|
| **Re-bid in place** — same machine, disk and models kept | (new bid − best alternative's run rate) × hours left; only within both ceilings | yes |
| **Replace** on another machine | model download at that host's price + time to ready | yes |
| **Wait it out**, instance stopped, storage only, time-boxed | storage, plus the run pausing unless higher tiers cover demand | post-v1 |

*(Also post-v1: sampling each rented machine's floor every pass, proactive re-bid before being
outbid, re-bidding downward once it is verified the provider charges the bid rather than a
clearing price, and a thrash guard per machine and per hardware class.)*

## 7. Dead-man timer

If the supervisor crashes, or its machine sleeps or loses its network, lease limits stop being
enforced while a rented instance keeps billing. So **every rented host carries its own dead-man
timer**, installed by the start-up script and independent of anything off-host.

- **Condition:** no supervisor heartbeat **and** no inference request for `deadman_minutes`
  (default 20). Both: because the router is a separate process, a host still serving requests
  while the supervisor restarts is doing useful work and is left alone.
- **Heartbeat:** the supervisor refreshes a timestamp on the host each pass over the connection
  it already holds; a small on-host loop compares it, and the engine's last-request time, with
  the clock.
- **Action:** `destroy` by default — with nobody watching, a stopped instance bills storage
  indefinitely, and re-downloading models costs far less than an unattended month. `stop` is
  available.
- **Credential:** the provider's **instance-scoped** credential — one that can only act on that
  instance. **The account credential is never placed on a rented machine.** This is a declared
  provider capability (`self_terminate`); on a provider without it the pool refuses leases
  longer than a short maximum.

Lease expiry and the orphan sweep at next start are the second and third lines of defence.


**The call, not a command (D71).** The provider returns the *request* its instance makes to end
itself — method, URL, headers, body — and the start-up script writes it to a file as JSON and
reads it back when it fires: nothing about a provider's URL or header can become shell on a
machine the pool does not trust. The script resolves an HTTP client **at arm time**, while the
pool is still watching: `curl`, else `wget`, else `python3`, else one quiet package install.
Where a machine has none, it records that it has none and the timer falls back to **stopping the
container** — the accelerator stops billing, the instance shows as stopped, and a live pool
destroys it on its next pass. The first engine image carries no HTTP client at all, so a timer
that named one would have armed, looked healthy, and failed only after the pool had died.
## 8. Preparing a rented host on request

Overflow-driven renting answers "demand exceeded what I have". **Prepare a host** answers "get
one ready before I start" and "keep one warm between runs". Available from the console, the
control API and the CLI (`gpm host prepare`).

**It is its own small lease**: it cannot start without a bid ceiling, a total dollar cap and a
time limit, and its confirmation states the worst case. It borrows no authority from any other
open lease.

**Inputs:** provider; models (the pool's full set by default, resolved to the right variants
for that provider's hardware); context length; offer policy (pre-filled, editable for this
preparation); offer selection — *best by policy* or *picked from the live market list*; caps;
what to do when ready.

**Shown before anything is spent:** download size × that host's price per gigabyte; time to
ready from its advertised download speed; hourly rate while preparing; storage rate if parked.

| Step | What the operator sees |
|---|---|
| Bid | Offer chosen, bid placed, won or lost; a lost bid retries within the caps |
| Instance up | Image pulled, engine answering, dead-man timer armed. A host the provider reports as having come up **without its start-up material** is ended in the same pass and its machine avoided (D65): it has no timer and no way in |
| Models | Per-model download progress, gigabytes so far, download cost so far against the estimate. **Each model is loaded and pinned as soon as its own download finishes, while the rest are still downloading** (D57), so the loading of one overlaps the download of the next |
| Verify | Every model loads; all resident **together**; a short clean generation per model. Readiness is unchanged by D57: a host joins on the whole set, never on a partial one |
| Size | Worker count for this hardware and model set; engine parallelism set to match |
| Ready | Cost to date, hourly cost from here, storage cost if parked |

| When ready | Effect |
|---|---|
| **Join the pool** | A `ready` rented host at its normal priority, with a **hold-until** time from the preparation's limit — otherwise idle release would reap it minutes later for having no traffic yet |
| **Park it** | Stopped, disk and models kept, storage billed only. The console shows the break-even (`download cost ÷ storage cost per hour`); a parked host is destroyed automatically past `max_park_hours` — parking is never open-ended |
| **Destroy** | Ends all billing, verified |

**Parked hosts are used first.** When a lease needs rented capacity the supervisor first tries
restarting a parked host that already holds the models: no download, minutes instead of tens of
minutes. It must still win the auction on that machine within the lease's ceilings; if it
cannot, the supervisor bids on a fresh offer and the parked host stays parked until its limit.

## 9. Tearing down

| Trigger | Action |
|---|---|
| Unused for `idle_minutes` (2) | **paused** — parked where the provider can park; elsewhere it runs on (D64) |
| Still unused `destroy_idle_minutes` (5) after its last request | destroy. Unset, this limit follows `idle_minutes` at the same ratio |
| Load returns while a host is paused | it is restarted **at once** — no scale-up window, since a restart buys no download |
| Overflow gone for 600 s | same, one host at a time. Deliberately slower than the 120 s scale-up, so capacity does not flap |
| Lease closed or expired | drain all rented hosts → destroy |
| Lease budget nearly spent | stop admitting requests to rented hosts while the remaining dollars still cover the drain window, then drain → destroy |
| Evicted and not worth holding | destroy, so storage stops billing |
| Unhealthy and quarantine did not clear it | destroy; replace if the overflow remains |
| Park limit reached | destroy |
| Operator: `gpm release <host>` | drain → destroy |
| Operator: `gpm down --all` | the panic button — destroy everything now, no drain, verified |
| Operator: `gpm stop --release` | drain → destroy everything, then stop |

**The lease says what may be spent, not that it must be (D64).** Once a host has been given up
for being unused, the lease's standing demand does not bring capacity back by itself; *measured
load* does — every ready worker busy on two passes running, or a request that waited, or was
refused for waiting, in the last half minute. A prepared host's hold (§8) stops it being reaped as
surplus; it does not shelter a host nobody is using. A host the pool parked is stopped on purpose
and is never read as an eviction.

**Order:** reverse routing priority, then highest cost per worker, then fewest busy workers.

**Drain:** stop routing to the host; let busy workers finish for up to `drain_timeout_s`
(default 300); then release regardless — anything still running reaches its client as an
interrupted stream.

**Park or destroy:** destroy ends all billing and loses the disk. Parking keeps the models at
storage cost, so a restart needs no download but must win the auction again. A parked host is
kept no longer than `min(max_park_hours, break-even)`. At lease end it is always destroy.

## 10. Strategies are tested without spending

Every strategy is a pure function of (market snapshot, host state, lease, configuration) →
decision. They are unit-tested against the **fake provider**, which can replay a scripted
market — floors spiking and returning, a bid lost between search and create, an instance parked
by the provider, an eviction mid-run. **No test needs a cloud account.** *(Post-v1: market
samples recorded by a running pool become a replay set for scoring candidate strategies on
dollars spent, evictions and capacity-hours delivered.)*

## 11. Configuration sketch

```yaml
pool:
  name: default
  models: [my-model:7b, my-embedder]          # every host holds all of these, loaded
  context_length: 32768
  app_keys_file:   ~/.config/gpm/default.app-keys      # hashed
  admin_keys_file: ~/.config/gpm/default.admin-keys

router:
  listen: 127.0.0.1:11435
  queue_timeout_s: 30                          # must stay below the SDK's time to first byte
  retry_other_host: once
  session_affinity: off                        # off | within_tier | across_tiers   (post-v1)
  session_build_consistency: off               # post-v1

limits: { max_rented_hosts: 2, max_hourly_burn: 1.50 }

hosts:
  workstation:
    kind: local
    url: http://127.0.0.1:11434
    capabilities: [apple-silicon]              # declared; omit and it is inspected
    workers: auto                              # a number here can only lower the profile's ceiling
  lab-box:
    kind: fixed-remote
    transport: { type: tunnel, ssh: user@lab-box.internal }
    capabilities: [cuda]
    workers: 4
  office-gpu:
    kind: fixed-remote
    transport: { type: https, url: https://gpu.example.internal, auth: { bearer_env: OFFICE_GPU_TOKEN } }
    workers: auto

rented:
  marketplace:
    kind: rented-interruptible
    provider: <provider plug-in name>
    transport: { type: tunnel }
    image: <pinned engine image>               # never a floating tag
    disk_gb: 60
    offer_policy: { min_vram_gb: 64, max_allin_hourly: 0.66, max_download_per_gb: 0.01 }
    scale:    { scale_up_after_s: 120, scale_down_after_s: 600, min_useful_hours: 1, one_at_a_time: true }
    bidding:  { strategy: floor_plus_premium, premium: 0.02, bid_ceiling: 0.60,
                on_demand_crossover: 0.8, attempts: 3, retry_market_every_min: 10 }
    spend:    { cap_safety_margin: 0.10, drift_alert: 0.15 }
    prepare:  { max_park_hours: 72, default_when_ready: join }
    teardown: { idle_minutes: 2, destroy_idle_minutes: 5, drain_timeout_s: 300, deadman_minutes: 20, deadman_action: destroy }

capacity_profiles:                             # workers per hardware class; first match wins
  - match: { hardware: "1x RTX PRO 6000 Max-Q" }  # the offer's hardware, compared whole
    max_workers: 6
    note: "measured: latency flat from 4 to 6 in flight"   # the evidence, shown where it is used
  - { match: { capability: cuda, min_gpu_memory_gb: 80 }, max_workers: 4 }
  # no match → rented.workers

calibration:                                   # memory-ceiling constants, per model set
  "<model set>@32k": { fixed_gib: 10.0, per_worker_gib: 7.6, operate_at: 0.67 }
```

The numbers are illustrative defaults, not recommendations for any particular hardware.
