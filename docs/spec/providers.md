# Specification — Several Providers

> Status: **designed; steps 1 and 2 built** (D129–D133; 2026-10-07) — the connections'
> configuration, every record naming its connection, the supervisor renting through several at
> once, and spot prices (§9). The console (steps 3–4) and the second real provider (step 5) are
> not built: until they are, connections are managed in the configuration file, and the
> console's screens show the first enabled connection. Reasons are in
> [../decisions.md](../decisions.md). The second real
> provider is **RunPod** (its v2 API), chosen by the owner on 2026-10-07 from a comparison of
> candidates' APIs against [plugin-interfaces.md](plugin-interfaces.md) §1.

A pool may rent from several GPU providers at once, bidding or fixed-price, each through its
own **connection**, on demand or interruptible — bid for, or at the provider's spot price. The operator manages the connections in the console. One market view shows
every enabled provider's offers together, and the pool chooses among them by one rule.

## 1. Connections

```yaml
rented:
  providers:
    vast:   { type: vast,   enabled: true,  settings: {} }
    runpod: { type: runpod, enabled: false, settings: {} }
```

- **A connection** — a *provider account*, in the console — has a name (the key), a `type` (an
  installed `gpm.providers` plug-in), `enabled`, and the plug-in's own `settings`. **A pool has at
  most one connection per provider** (D133): two connections of one type could be one account,
  each sweeping the other's hosts as strays.
- **The old shape still loads.** `rented.provider: vast` with `provider_settings` is read as one
  connection named after its type; naming both shapes is refused. At least one connection must
  be enabled. The console writes the new shape the first time it saves.
- **Disabled** means no new search and no new rental on that connection. Its hosts stay until
  they are released, and are still monitored, charged, swept and bound by the dead-man timer.
- **Removing** a provider is refused by the configuration plan while the pool holds a host, a
  parked disk or a kept volume there, with the reason. Renaming a connection is not removing it:
  its records are found by its provider. A host whose provider is no longer configured — the
  file edited while the supervisor was stopped — is kept, never dropped, and nothing is done to it
  until the provider is configured again.
- **Turning a connection on** is a loosening, typed again like any (`requires_retype`).
- **Limits are the pool's** (D129). Leases, the most rented hosts and the hourly burn cap hold
  across every connection together. A connection has no limits of its own in this version.

## 2. Interruptible: a bid, or the provider's spot price (D132)

The two kinds stay what they are today: **on demand** (fixed price, never taken away) and
**interruptible** (cheaper, and can be taken away). An interruptible offer's price is set in one
of two ways:
- **by the pool, as a bid**, within its ceilings, and raised in place where the provider allows.
  Example: Vast.ai.
- **by the provider, as a spot price**: the listed price is paid, and nothing is bid. Examples:
  AWS EC2 Spot, Google Cloud Spot VMs, Azure Spot VMs.

The rest follows from that:
- **An offer says whether its price can be bid**, and the capability splits into
  `interruptible` (a host can be taken away) and `bidding` (the pool sets the price). Bid
  strategies and re-bidding apply only where the pool sets the price; a spot offer's "bid" is
  its listed price.
- **Bid and spot compete on the same terms.** The §6 rule compares them by expected cost per
  worker-hour, so a bid that would cost more than a spot price loses to it. They differ only
  where their interruption rates do, and the decision says so.
- **The all-in ceiling holds for spot as for a bid.** Where the provider takes a maximum price,
  the ceiling is sent as that maximum, and the provider stops the host above it. Where it does
  not, and the spot price can move while the host runs, the pool reads the current price each
  pass and drains and releases a host whose price has passed the ceiling.
- **A spot host's spend** is estimated at its current price, and the provider's reported charges
  reconcile it, as for any host. A price that moves within the ceiling still answers to the hourly
  burn cap and the lease's budget, as a re-bid does: past either, the host is released.
- **An interruption notice**, where a provider gives one (a capability), drains the host at once,
  so the answers in flight finish or are redispatched before it goes.
- **`rented.mode` is unchanged**: `interruptible` takes bids and spot, `on_demand` takes fixed
  prices, `cheaper` both.

## 3. Every record names its connection

Each offer, host, instance, spend row, decision-log event, machine-history row and search-usage
counter carries the connection's name. An offer is known by its connection, its id and its kind
together: an operator's chosen offer names its connection, and one id listed by two providers is
not guessed at. A decision-log event about a host names that host's connection.

Machine identifiers are each provider's own, and with one connection per provider (D133) the
machine history and the avoid list are kept by machine identifier: a collision would need two
providers to number a machine alike, and would at worst skip or rate that machine on the other.
Warm volumes and the stray sweep are kept per connection.

A record written before connections names none. It belongs to the **legacy connection**: the
one the pool's configuration became the first time a version with connections ran, kept in
`pool_meta` so a later rename or a second connection does not reassign it. The day's search
use counted before is carried into the per-connection counter once, under that name.

## 4. The supervisor with several connections

- **Searching** asks every enabled connection at once. Each has its own back-off, its own
  rate-limit handling and its own daily quota counter. A connection that errors or is out of
  quota is reported beside the results, and the others are still used.
- **Every operation on a host goes to the host's own connection**: create, bid, start, stop,
  destroy, status, connection details, charges and the dead-man timer.
- **The stray-instance sweep and crash recovery** list instances on every connection, enabled
  or not. A connection that cannot be asked at a restart holds back only its own hosts — their
  records are kept, never deleted, it is not swept, and it is asked again each pass until it
  answers — while the others carry on. A held-back host still counts against the pool's host
  limit and hourly burn cap, at its last known price, and its demand is not rented again
  elsewhere until its provider answers: it may still be serving, and billing. One whose provider
  is no longer configured is not asked about again. Only when no connection can be asked does the
  pass wait, as before (D61). A name once one provider's cannot be reused for another: the supervisor refuses
  to start, since that name's records are the first provider's (D133).
- **Capabilities are per connection.** A host on a connection that cannot be parked is always
  destroyed. A connection without an instance-scoped credential serves only short leases: a
  lease longer than `max_hours_without_deadman` is not rented from it — nor restarted from a park,
  nor priced in a plan — and a lease is refused that long only when no enabled connection can arm
  a timer. The rule is also held on the host itself: a host there is released once it has run
  that long, however its lease came to be longer (extended, amended). Bidding applies only to
  bid offers (§2).
- **One plug-in's failure stays its own**: a search that raises anything leaves the other
  connections' offers standing, and says why beside them.
- **Changing the connections takes a supervisor restart**, and the configuration plan says so.
- **Spend** is reconciled per connection, against each one's reported charges where it has them.
  A lease's caps hold on the sum.

## 5. Credentials, typed in once (D130)

- **Entered in the console, write-only.** The add and edit flows have a credential field. Its
  value is sent once to the control API, with the admin key and the usual `Host`/`Origin`
  checks. A credential sent over plain HTTP from anywhere but loopback is refused.
- **Stored by the supervisor** in an owner-only file per connection, in a secrets directory
  beside the pool's state. It is never in the configuration file, its versions or its plan; never
  in a response, an event, a log or the database; and never sent to a rented host. The router
  never reads it.
- **Shown only as a state**: *set on <date>, valid, credit $X*, *set, refused by the provider*,
  or *missing*. It is never shown again, in whole or in part. **Replace** and **Remove** are the
  only actions.
- **An environment variable still works**: `credential_env: NAME` on the connection takes the
  credential from the supervisor's environment instead, as today, for an operator who keeps
  secrets out of the pool's machine state.
- **A new credential takes effect without a restart**, on the supervisor's next pass, after a
  test against the provider's account call.

When this is built, the threat model's asset table and T16 change: the credential *may pass
through the browser once, typed in*, and is never sent back. Found by the architecture review,
and required of step 3:
- **The credential is bound to where it is sent.** A change to a connection's settings that names
  another endpoint (a `base_url`, an `api_key_env`) clears it; **Test connection** sends a stored
  credential only to the endpoint it was saved with, and fetches no URL it is given.
- **Replacing a credential cannot switch accounts** while the connection holds hosts or volumes,
  where the provider can say which account a key is.
- **Nothing echoes it**: a refused request answers without the submitted value (validation errors
  included), and a test proves it is never logged.
- **The file** lives in an owner-only directory, written atomically without following links,
  checked at start like the pool's other secrets (T19), and kept out of configuration versions.
- **A plug-in is handed its credential** rather than reading the environment itself — a versioned
  change to the provider interface (plugin-interfaces.md §1), made before the second provider.
- **Threat-model rows** for the credential in the browser's memory, an endpoint redirect through
  settings, a replacement to another account, and the test call fetching what it is told.

## 6. One choice across providers (D131)

Every rental, shared pool and workloads alike, is chosen among **all enabled connections' offers
by expected cost per worker-hour** over the hours the lease has left. This is the rental-kind
rule of D115 (workloads.md §5), extended to every offer:
- **the price**: a bid plus storage, a spot price, or the fixed all-in price;
- **for an interruptible offer, the expected interruptions**: from the machine's own history on its
  connection (for spot, the instance type's in that region), else that connection's prior for
  that kind (`interruption_prior_per_hour` per connection and kind, since providers differ);
- **what the machine serves**: the workers its card runs within the target, from the
  capacity profile or measurement;
- **what it costs to get ready**: the download size over its speed, and its download price —
  once, so it weighs most on a short lease;
- **the machine's own record here** (D69): one that failed to come up or served slowly costs
  more per worker-hour than its price says, one that served fast less.

The score's rough throughput proxy is not weighed: what a card serves comes from its capacity
profile and measured latency. The score still decides what passes the filters, and the market
preview lists offers in the order the pool rents them, each with its expected cost per
worker-hour.

`rented.mode` still says which kinds may be rented at all. A field one provider does not report
(reliability, download speed or price) takes a stated default, and the decision says so.

## 7. Console

### 7.1 Providers (Rented capacity → Providers, the first tab)

A card per connection:
- its type, and what it can do: on demand; interruptible, by bid or spot price; park; dead-man
  timer; interruption notice; reports charges;
- its credential's state and the credit left;
- the day's search use against its quota;
- what is rented on it and burning;
- **Test connection**, **Edit**, **Disable/Enable**.

**Add provider**, in this order:
1. Choose a type, from the installed plug-ins with their capabilities.
2. Name the connection.
3. Enter the credential (§5).
4. **Test connection.** Each step shows passed or failed, with the provider's error:
   - the credential is accepted, and the credit left;
   - one search of a single row;
   - any instances already carrying the pool's label;
   - the declared capabilities.
5. **Save.** The usual validate → plan → apply, with the plan saying what changes ("searches
   will now also ask runpod; no host affected").

One connection's page shows its status history, its hosts and spend, and its slice of the
decision log.

### 7.2 The market, across providers (Rented capacity → Finding machines)

- **Choosing what to search.** A chip per enabled connection and per kind (on demand, interruptible) narrows
  the search. A search still happens only when **Search the market** is pressed (D120).
- **A line per connection**: rows returned, rows that pass, its quota use, or why it was not
  asked or failed.
- **One table of offers.**
  - Each row shows its connection, hardware, kind, price, all-in, download and reliability.
    A value a connection does not report shows as "—" with the default used.
  - **"Would rent"** marks the row the §6 rule picks, across connections.
  - **Rent** on a row rents that offer from that connection.
- **Rejections by reason**, with a column per connection.
- **The offer policy stays one, pool-wide**, and is edited beside the market as now.

### 7.3 Elsewhere

- **Rented hosts.** A connection column, burn per connection, and Prepare a host from the best
  across connections or a chosen one.
- **Overview.** Hosts and burn per connection. **Release all rented** releases on every
  connection.
- **Workloads and provisioning grants.** An optional "providers allowed", all enabled by
  default; the plan names the first host's connection.
- **Leases.** Estimated and reported spend per connection.
- **Decisions.** A filter by connection.

## 8. Control API

| Endpoint | Purpose |
|---|---|
| `GET /pool/providers` | Every connection: type, enabled, capabilities, credential state (never its value), credit, quota use, hosts and burn |
| `POST /pool/providers/test` | Test an unsaved connection, credential included; saves nothing |
| `POST /pool/providers`, `PATCH /pool/providers/{name}` | Add or change a connection through validate → plan → apply; enabling is retyped like any loosening |
| `PUT /pool/providers/{name}/credential`, `DELETE …/credential` | Set or replace, or remove, the credential; answers with its state only |
| `DELETE /pool/providers/{name}` | Remove; refused while it holds a host, a parked disk or a volume |
| `GET /pool/market/preview` | Gains `providers=` to narrow the search; each row carries its connection |

## 9. Build order

Each step ships on its own and keeps a single-provider pool working as it does today.
1. The connections' configuration shape, and the connection's name on every record.
2. Interruptible offers priced by bid or spot (§2), and a supervisor with several connections, tested against
   **fake providers** with different capabilities: one that takes bids; one that offers spot at a
   price it changes, with an interruption notice; and one fixed-price only that cannot park.
3. Credentials typed in once (§5), and the Providers screen.
4. The market across providers, then the connection columns everywhere else.
5. The second real provider plug-in, **RunPod** (on demand, with park), built on its v2 API and
   checked live on the smallest case with the owner's go-ahead. Two facts are verified first (see
   decisions.md, Unverified assumptions): that it no longer offers spot, and whether its pod-scoped
   key can terminate its own pod — which decides whether its hosts get the dead-man timer. **Verda**
   is the likely third, as the first provider whose interruptible offers have a spot price.
