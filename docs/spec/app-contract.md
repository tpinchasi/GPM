# Specification — The App Contract

> What an application can rely on when it talks to a pool. Versioned: this is **contract v1**.
> Reasons for each rule are in [../decisions.md](../decisions.md) (D1, D3, D15, D16, D22).

## 1. The boundary

The pool serves GPU compute; an app requests it. They share **one URL, one API key, and the
contract on this page** — nothing else.

| The app knows | The app never knows |
|---|---|
| The pool's base URL and its app API key | How many hosts exist, what kind they are, where they run |
| The engine's HTTP API | Providers, bids, leases, tunnels, recovery, cost |
| The model name it wants | Which host holds it, worker counts, memory |
| Standard HTTP outcomes: success, `401`, `404`, `503` + `Retry-After`, a transport error | Why capacity came or went |

Rules that keep the boundary real:

- **No pool server code is imported by an app.** An app depends on at most the client SDK (§4).
- **The pool never calls into an app.** Excess requests queue in the pool; nothing is pushed
  back into the app's configuration.
- **Apps never trigger recovery or spending.** They wait. Leases are opened by an operator or a
  run script, through the control API, with a different key (§3).
- **The pool's location is free.** Same machine, another machine, behind TLS — a configuration
  change for the app, never a code change.

## 2. The contract: the engine's API plus a small pool dialect

Requests and responses are the **inference engine's own HTTP API, passed through untouched**.
On top of it the pool adds a short, named dialect. Nothing else is added silently.

| Dialect item | Required? | Meaning |
|---|---|---|
| `Authorization: Bearer <app key>` | **Required**, loopback included | Admits the app to this pool (§3) |
| Logical model names | Opt-in per model, by the operator | A name listed in the pool's catalog resolves to the build suited to the serving host. The same request can therefore return a different build through the pool than against a bare engine — by design, only for catalogued names. Unlisted names pass through literally |
| Model actually served | Always reported | The response body's own model field (left untouched), and `X-GPM-Served-Model`, with `X-GPM-Host` and `X-GPM-Runtime-Class` |
| `X-GPM-Wait-S` | Always reported | Seconds this request spent waiting for a worker, so queue wait is never mistaken for generation time |
| Machine-readable `503` | Always | Body carries `reason` and `retry_after_s`; header carries `Retry-After`. A client that ignores the body still behaves correctly |
| `X-GPM-Session` | Optional | Groups calls in the pool's request log. Affects routing only if the operator switched on an affinity or build-consistency option |
| `X-GPM-Deadline` | Optional | Absolute time after which the response is useless; lets the pool drop work that can no longer be used (§5) |
| `X-GPM-Runtime-Class` (request) | Optional | Pins the request to hosts whose served build is of the given class — for runs that must not mix builds |

### Status codes an app will see

| Code | Reason field | Meaning | SDK default |
|---|---|---|---|
| engine's own | — | Passed through | returned |
| `401` | — | Missing or wrong app key | raise `PoolAuthError`, never retried |
| `404` | `model_not_in_pool` | The model is not in this pool's declared set; it will not be loaded on demand | raise `PoolRequestError` |
| `503` | `queue_timeout` | All workers busy for longer than the queue limit | wait and retry |
| `503` | `preparing` | A host is up but still loading the pool's models | wait and retry |
| `503` | `recovering` | The supervisor is bringing a host back | wait and retry |
| `503` | `hosts_unreachable` | Configured hosts are not answering | wait and retry |
| `503` | `no_offer` | Renting is authorised but no acceptable offer exists right now | wait and retry |
| `503` | `no_lease` | Only rented hosts could serve this and no lease is open — waiting cannot help | **fail fast** with `PoolUnavailable` (overridable) |
| `503` | `no_eligible_host` | Hosts are ready, but none holds a build of this model that satisfies the request — its schema requirement or its pinned runtime class | wait and retry |
| `504` | `deadline_exceeded` | The request's `X-GPM-Deadline` passed before it could be served | **fail fast** — the app already said the answer would be useless |

```json
{"error": "no_capacity", "reason": "recovering", "retry_after_s": 30,
 "detail": "rented host outbid; re-bidding on the same machine"}
```

## 3. Keys

- **App key** — permits inference and `GET /pool/status`. Nothing else.
- **Admin key** — required for the control API and the console. **Never interchangeable with
  the app key**: an app able to request a completion must not thereby be able to open a lease,
  change a ceiling or release a host.
- Sent as a bearer header, never a cookie. Stored hashed. Created, rotated and revoked from the
  CLI; two app keys may be valid at once so rotation needs no downtime.
- The key is required **on loopback too**. Any web page open in a browser on the same machine
  can send requests to `127.0.0.1`; a required header closes that.
- Off loopback the listener additionally requires TLS — a bearer key over plain HTTP is a
  published key.

Requests are otherwise anonymous. The pool does not distinguish between apps holding the same
key; separation is achieved with separate pools.

## 4. Client SDK

A small Python package (`httpx` only). **Waiting for capacity is its default behaviour** — an
app opts *out*, not in.

Its core is an **HTTP transport**, not a wrapper function, so the retry behaviour sits
underneath whatever client library the app already uses, with no call-site changes:

```python
from gpm_client import PoolClient, pool_transport, async_pool_transport

# An app using a higher-level client library: inject the transport once
client = SomeEngineClient(base_url=POOL_URL,
                          http_client_kwargs={"transport": pool_transport()})

# A script: call the pool directly
pool = PoolClient()                         # GPM_URL, GPM_API_KEY from the environment
reply = pool.chat("my-model:7b", messages, format=schema, session_id=sid)
reply.content, reply.served_model
vectors = pool.embed("my-embedder", texts)
pool.wait_until_ready()                     # optional pre-flight for a batch driver
```

### Default retry policy

Every field is overridable per call, per client, or by environment variable, in that order.

| Situation | Default |
|---|---|
| `503` other than `no_lease` | Retry. Wait `Retry-After` if present, else exponential back-off 2 s → 60 s with jitter |
| Transport error before any response byte | Retry, same back-off |
| Failure **mid-stream**, after output was delivered | **Not retried** — `PoolStreamInterrupted`. The app has consumed partial output; only it can decide to redo the call |
| `401` | Not retried — `PoolAuthError` |
| Other `4xx` | Not retried — `PoolRequestError` |
| `503 no_lease` | Fail fast — `PoolUnavailable`; `wait_without_lease=True` to wait anyway |
| Total time allowed waiting — `max_wait` | **Configurable**; ships at 30 minutes. `GPM_MAX_WAIT_S`, `RetryPolicy(max_wait=…)`, or per call. `None` waits indefinitely (what batch drivers usually want) |
| Visibility while waiting | One log line per minute with the reason; optional `on_wait(reason, waited_s, next_try_s)` |

Retrying is safe because a chat or generate call has no server-side effect to duplicate: the
full conversation travels with each request, and tools are executed by the app, not the engine.

### What else it carries

- **Typed errors**: `PoolUnavailable`, `PoolStreamInterrupted`, `PoolRequestError`, `PoolAuthError`.
- **The facts of each call**: `served_model`, and — where the calling library exposes response
  headers — serving host, runtime class and `pool_wait_s`, so time spent waiting for capacity is
  never mistaken for generation time. Where headers are not reachable through the app's client
  library, the same facts are in the pool's request log, keyed by session id, for an offline
  join. (A context-local capture hook is a post-v1 item.)
- Sync and async variants of everything.

**What it never contains:** host lists, provider names, bids, tunnels, or lease management.

## 5. Time budget — who waits, who retries, who cancels

Several layers can wait on one request. Unrelated, they produce abandoned work: a client gives
up while its request is still queued and retries, the request now exists twice, and a worker
eventually runs a generation nobody is listening to. Under load this feeds itself, and on
rented hosts the waste is paid for. The pool cannot know an arbitrary client's timeout, so the
contract states the budget.

1. **Cancel on disconnect.** When the client goes away the router cancels the upstream request
   and frees the worker at once, queued or generating.
2. **One layer retries for capacity — the client side.** The router's own cross-host retry is
   for one case only: the chosen host failed before the first response byte. It is attempted
   **once**, on the next eligible host in priority order, and never for "no capacity".
3. **The SDK owns client timeouts**, split three ways; callers do not set raw ones.

   | Timeout | Covers | Default |
   |---|---|---|
   | connect | reaching the router | 5 s |
   | time to first byte | queueing + prompt processing | 300 s |
   | between bytes | a stalled stream | 60 s |

   **Invariant, enforced at configuration load:** `router queue timeout < SDK time to first
   byte`. The router always answers a queued request — with a worker or a `503` — before an SDK
   client would give up on it.
4. **Clients without the SDK get an early, clean answer.** The router's queue timeout defaults
   to 30 s, below common HTTP client defaults, then `503 queue_timeout` with `Retry-After`. The
   limits in force are published in `GET /pool/status`.
5. **Deadlines travel with the request.** The router drops a queued request whose
   `X-GPM-Deadline` has passed and does not start one that cannot produce a first byte in time.

For non-streaming requests "first byte" is the whole response, so the time-to-first-byte budget
must cover the full generation; the SDK derives it from the request's output-length limit when
one is given.

## 6. Versioning

The contract on this page has a version, reported in `GET /pool/status` and in an
`X-GPM-Contract` response header. Adding an optional dialect item is a minor change; changing
or removing one, or changing a status-code meaning, is a major change. The SDK follows semantic
versioning and states which contract versions it speaks.
