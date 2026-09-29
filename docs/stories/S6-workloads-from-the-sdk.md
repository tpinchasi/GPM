# S6 — Workloads created from the SDK, and client certificates

> Status: **decided (D117), built** (2026-09-29; `docs/spec/workloads.md` §11 is the specification). Part of the [feature list](README.md). The
> owner changed the app-boundary rule for provisioning keys, chose client certificates in the
> first build, and set idle-end at 15 minutes, overridable per workload. An architecture review
> found the shape sound and four gaps that made the grant's bound soft; all are folded in (§7).
> A story is a plan: nothing here is specification until its decisions are recorded.

## The story

*As an application developer, I write `with pool.workload(model="gemma4:31b", latency_s=20,
parallel=16, hours=6, max_spend=25) as w:` and my program gets its own GPUs for the run: the pool
rents them, my client is pointed at them, and they are released when the block ends — without an
operator creating anything by hand, and without my program ever holding a key that can do more
than that.*

And, where the operator wants it: *the workload is reached with a client certificate the SDK made
for it, whose private half never leaves my machine.*

## Why

Workloads (D115) are created by an operator with the admin key, and the key is handed to the
application owner by copying it out of the console or the command line. For a batch job, a test
run, a nightly evaluation, that handover is the whole friction: the job knows its model, its
latency and its hours better than anyone, and it has to wait for a person.

The owner asked for client certificates **only** together with this (2026-09-29): a certificate a
person has to mint and hand over is the same friction with a harder file format.

## The decision this needs first

CLAUDE.md, "the app boundary": **apps never trigger recovery or spending**. D16: an app that can
request a completion must not be able to spend. This story gives a program the power to spend —
within limits the operator sets — so it cannot be built without the owner changing that rule, and
recording the change as a decision. The design below keeps what the rule protects:

- **Not the app key.** A new, third kind of key, the **provisioning key** (`gpmp_`), made by the
  operator for one application, is the only thing that can ask. The app key still cannot spend;
  a workload key still cannot spend. A provisioning key cannot request a completion at all.
- **Bounded by a grant the operator writes**, checked by the supervisor, never by the SDK:
  - how many workloads at once;
  - most dollars per workload, and most per rolling day across all of them;
  - most hours;
  - which models, which kinds of machine;
  - whether it may borrow.
- **Never the control API.** It reaches four calls on the router's listener and nothing else.

If the owner prefers to keep the rule as it is, the alternative is a person-in-the-loop: the SDK
*asks* and an operator approves in the console. That keeps the rule and loses most of the point.

## Design

### 1. The provisioning key and its grant

```
gpm provisioner create nightly-evals --max-open 2 --max-spend 20 --max-spend-per-day 60 \
                        --max-hours 8 --models gemma4:31b,big --kinds roi,interruptible
```

- It prints the key once, as every key.
- The grant is written to a table the supervisor owns; `gpm provisioner list|show|revoke`.
- Revoking stops new workloads at once. `--end-workloads` also ends the ones it made.

### 2. Four calls on the router's listener — and the router still never calls the supervisor

The SDK knows one URL. Creating a workload is slow (a market search) and spends money, so the
router must not do it (CLAUDE.md: nothing slow on the request path; the router and the supervisor
never call each other). The router only **records a request**, and the supervisor answers it on
its next pass, as it already answers everything:

| Call | The router does | The supervisor does |
|---|---|---|
| `POST /pool/provisioning/requests` `{kind: "plan"\|"create", model, latency_s, parallel, hours, max_spend, machines?, key_hash, csr?}` | checks the provisioning key, writes a row — or, for a `key_hash` it has seen, answers with that row's `request_id` — and answers `202 {request_id}` | reads the row, checks every field again, the grant and the pool's caps, plans or creates, writes the outcome |
| `GET /pool/provisioning/requests/{id}` | reads the row — only for the key that made it | — |
| `GET /pool/provisioning/workloads/{name}` | that workload's state — only its owner's | — |
| `POST /pool/provisioning/workloads/{name}/end` | writes an "end" request | ends it |

Under `/pool/provisioning/` so no path is shared with the control API's, which a different key
reaches; declared before the router's catch-all inference route.

To keep a create from waiting a whole pass, the supervisor watches the request table as it watches
the configuration file: a cheap revision check every second, work only when it moved. Still no call
between the processes. Requests are answered in one task that holds the workloads' creation lock,
and a plan's market search waits for the pass to finish rather than competing with it for the
provider (whose rate-limit back-off the whole pool shares).

**Why not a scoped role on the control API**, which would answer synchronously and need no table:
the control listener is loopback-only by default, and a program on another machine would need it
exposed — the admin surface, off loopback. The router already is the one listener apps reach.

### 3. The key never leaves the program

The SDK makes the workload's key itself — 32 random bytes, locally — and sends only its **hash**
(`key_hash`). The pool stores the hash, as it stores every key. So:

- the plaintext key is never written to the database;
- it never crosses the process boundary;
- it is never "shown once" anywhere.

A lost program loses its own key and nobody else's. The same shape serves certificates, below.

### 4. Money: typed, bounded, committed, never proposed

- A request must carry `max_spend`. There is no derived-and-confirmed budget: there is no person
  to confirm it.
- **Created once.** The workload key's hash is the request's identity: a create sent twice — the SDK
  retrying a request whose answer was lost — is the same request, and the supervisor never creates
  twice for one hash. The SDK sends creates without its transport's retries, and polls instead.
- **The day counts what was committed, not what has been spent**: the sum of the `max_spend` of
  every workload the provisioner made in the last 24 hours. Spend is recorded pass by pass, and a
  program could otherwise open several before any of it showed. `ending` workloads count toward
  "at once": they bill while they drain.
- **A pool-wide day cap across every provisioner** (`provisioning.max_spend_per_day`). Without it
  the operator's real exposure is the sum of every grant, which nobody set on purpose — and it is
  what bounds a compromised router (§7).
- **A grant expires** (`--expires`), like a lease: a key forgotten in a CI secret stops.
- Every rule of D115 then applies unchanged: the pool's caps, the floor, the reservations, lease end.
- **A program that stops using its workload ends it** (idle-end): once serving, a provisioned
  workload with no request for its idle cutoff is ended — 15 minutes by default, set per workload
  in the create request (`idle_end_minutes`), never above the grant's maximum; one that
  never sends a request is ended after its time to serve plus that. Read from the request log the
  supervisor already reads — no heartbeat, no SDK thread. Without it the floor (D115) keeps a
  crashed program's hosts for the lease's whole life, and its whole budget is spent.
- `plan` requests let the SDK see the price and the start before it asks to create. They spend
  nothing and count against nothing.

### 5. The SDK

```python
from gpm_client import PoolClient

pool = PoolClient(base_url, api_key=None, provisioning_key=os.environ["GPM_PROVISIONING_KEY"])
with pool.workload(model="gemma4:31b", latency_s=20, parallel=16, hours=6, max_spend=25,
                   wait_until="serving", timeout_s=1800) as w:
    reply = w.client.chat(...)            # a PoolClient bound to the workload's own key
# leaving the block ends the workload: its lease closes, its hosts drain and go
```

- `pool.workload(...)` → makes the key, posts `create` once (no transport retries; a repeat is the
  same request anyway), polls the request, then the workload's state, and hands back a client.
  The pool names the workload `<provisioner>-<6 hex>`; names are never reused.
- `wait_until="serving"` or `"created"`. A workload still preparing borrows shared hosts where it
  may (D115), so a program can start at once.
- Leaving the block ends it. An exception inside it ends it too. A program that is killed leaves
  it to idle-end (§4).
- A provisioned workload's key is not rotated by the pool: it never had it. A program that needs
  a new key creates a new workload.
- `pool.plan_workload(...)` answers the plan without creating anything.
- The SDK keeps its one dependency (httpx): the key is `secrets.token_hex`. A certificate request
  would need a cryptography library, so it is an optional extra (`gpm-client[certs]`).

### 6. Client certificates (the owner's condition met)

With `certs: required` on a provisioner's grant, a workload it creates is reached with a client
certificate as well as its key:

- The SDK makes a key pair locally and sends a certificate-signing request (`csr`).
- The supervisor takes **only the public key** from it. Subject, workload name, client-authentication
  usage, `CA:false` and validity are the pool's own; anything else in the request is ignored —
  signing what was asked could hand a program a CA.
- The supervisor signs it with the **pool's own client CA**. The CA's private key is held by the
  supervisor only: in a file the operator names, never in the database, never in the router.
- The signed certificate is not secret; it comes back in the request's row. It is valid for the
  workload's hours and names the workload.
- The router's listener asks for a client certificate from clients that have one. A workload that
  requires one refuses a request whose certificate is missing, not signed by the pool's CA, or
  names another workload — `403`, said in words.
- The private key never leaves the program's machine. Python's TLS loads a certificate from a
  file, so the SDK writes it to one only its user can read, and removes it when the workload ends;
  a killed program leaves it behind.

**The piece to verify first:** the router must learn *which* certificate a request came with.
Uvicorn, which the router runs on today, is started with a certificate and key only, and very
likely does not pass the peer certificate to the application.
That goes in the unverified-assumptions table, and is checked before anything else of this part
is built. If it does not, the options are a different server for the listener, or a TLS-terminating
proxy in front that passes the verified name on — the second weakens "the pool verifies", and is
not recommended.

## What it gives up

- **The app-boundary rule**, in a bounded form (the decision above).
- **A request-response `create`**: an SDK call waits on the supervisor's next look. It is a second
  or two with the table watched, not a round trip.
- **Budgets proposed by the pool**, for programs: a program must say what it will spend.

## Safety

### 7. What the review changed, and the threat it adds

- **A compromised router can now cause spending.** Today it has no path to money; with S6 the
  supervisor acts on rows the router wrote after checking a key. Bounded by the pool-wide day cap
  and every grant; the supervisor treats each row as untrusted input — model in the grant, finite
  positive numbers, a hash of the right shape, the name pattern — and the threat model says so.
- **One pending request per provisioner**, and plans rate-limited and answered from a minute's
  cache: a looping program must not stall the shared pool's renting through the provider's shared
  back-off.
- **Ownership is a column**: `workloads.provisioner` and the request rows carry it, and the router's
  snapshot too, so a key sees and ends only what it made.
- **Hash-only keys move "32 random bytes" onto the client**, where the pool cannot check it. The harm
  of a weak key stays inside that one workload; accepted, and said.


- A provisioning key reaches four calls on the router and the workloads it made — nothing on the
  control API (`403`, like the agent and workload keys), and no completion.
- Every spend is bounded twice: by its grant (per workload, per day, at once) and by the pool's
  caps. The supervisor checks both after the request is read, never the router.
- No secret crosses the process boundary: key hashes and signed certificates go in the table,
  never a key or a CA key.
- Every request, granted or refused, goes in the decision log with the grant's numbers.

## The owner's answers needed

1. **Change the app-boundary rule** for provisioning keys, as bounded above? Or the approval
   variant, where an operator approves each SDK request? *Review's view: change it — only with
   the hard bounds above (idempotent create, committed-dollar day cap, a pool-wide cap, idle-end,
   grant expiry). The owner changes CLAUDE.md's text; a decision entry alone does not.*
2. **Certificates:** in the first build, or after the SDK part is used? *Review's view: after —
   a crypto dependency, a CA to guard and a server change, for a property hash-only bearer keys
   over TLS already give: the secret never leaves the program.*
3. **A killed program's workload:** *answered by the design — idle-end, not the lease's hours and
   not a heartbeat.* Is 15 minutes the right default?

## Build stages (once decided)

1. The provisioner table, the grant, `gpm provisioner …`, and the router's refusal of `gpmp_` keys
   everywhere but the four calls.
2. The request table, the supervisor's watcher, and plan and create from requests.
3. The SDK's `workload()` and `plan_workload()`, and hash-only keys.
4. The live check of the peer certificate on the router's server, then certificates end to end.

Each stage is testable against the fake provider and the fake engines.

## Depends on

D115 (workloads). Touches D16, the app contract §3 (a fourth key), and CLAUDE.md's app boundary.
