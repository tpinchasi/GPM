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

**Both shipped engines serve the OpenAI-shaped `/v1` paths** — `/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings` — and the SDK's convenience methods use them by default
(D89). That is not translation: both engines genuinely speak these paths, so the same call
reaches a pool of either unchanged. Ollama also serves its own native `/api/*` paths, and apps
written against them keep working; a request is passed through on whichever surface it arrived.

Two differences between the surfaces matter to an app:

| | `/v1/*` | Ollama's `/api/*` |
|---|---|---|
| Streaming | **Off** unless `"stream": true` | **On** unless `"stream": false` |
| Structured output | `response_format` | `format` |

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
| `X-GPM-Delivery`, `X-GPM-Attempts` (response) | Always reported | How the response reached the app — `stream`, `buffered`, or `stream-after-overflow` — and how many hosts were tried (§5.1) |
| `X-GPM-Delivery: stream` (request) | Optional, if the operator allows | Asks for tokens as they come from a host that can be interrupted, accepting that the stream may break (§5.1) |

### What else the pool could serve (D101)

`GET /pool/directory?q=` — with the app key, on the pool's one URL — returns the directory the
operator's pool keeps: every model in Ollama's library with its tags, and each one's builds on
the model hub where they have been looked up, each with the engine options its family has. Each
tag carries **`in_pool`**: whether a request may name it today. It is read-only and informational
— an app cannot add a model, and nothing it reads changes the pool; the operator adds models.
It is served from a cached copy, never by asking a third party while the app waits. The SDK's
`PoolClient.directory(query)` returns it. An optional addition: the contract version is unchanged.

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
| `503` | `no_eligible_host` | Hosts are ready, but none may take this request. `detail` says which: the request arrived on one engine's own path (`/api/chat`, say) and every ready host runs an engine that does not serve it — the detail names both and the paths that reach every engine — or none holds a build of this model that satisfies the request's schema requirement or pinned runtime class | for a path no ready host serves, call the `/v1` paths; otherwise wait and retry |
| `503` | `host_lost` | The host serving this was taken away before any of the response reached you; nothing partial was sent (§5.1) | wait and retry |
| `504` | `deadline_exceeded` | The request's `X-GPM-Deadline` passed before it could be served | **fail fast** — the app already said the answer would be useless |

```json
{"error": "no_capacity", "reason": "recovering", "retry_after_s": 30,
 "detail": "rented host outbid; re-bidding on the same machine"}
```

## 3. Keys

- **App key** — permits inference and `GET /pool/status`. Nothing else.
- **Workload key** (`gpmw_`) — permits inference on one workload's hosts and `GET /pool/status`
  for that workload's view; nothing else ([workloads.md](workloads.md) §3, D115). Minted when the
  workload is created, expires with its lease. `/w/<name>/v1/…` means the same as `/v1/…` and is
  refused `403 wrong_workload` with another workload's key. A workload that has ended answers
  `503 workload_ended`, with no `Retry-After`: there is nothing to wait for. A workload may serve
  several models (D118): the request names one of them, as always; any other is `404`, naming the
  workload's models, and `GET /pool/status` lists them in `model_set` and, while some are still
  coming up, in `borrowing_models`.
- **Provisioning key** (`gpmp_`) — lets one application create, read and end its own workloads
  under `/pool/provisioning/…` on this listener, within a grant its operator set (D117). It never
  requests a completion and never reaches the control API. The SDK's `WorkloadProvisioner` uses it;
  the workload's own key is made in the program and only its hash is sent.
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
key; separation is by workload ([workloads.md](workloads.md), D115), or by separate pools.

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
   for one case only: the chosen host failed before the first byte **reached the client**. For a
   streamed response that is the first upstream byte; for a buffered one (§5.1) it is any point
   in the generation. It is attempted `max_redispatch` times (default **once**), on the next
   eligible host in priority order, inside the request's deadline, and never for "no capacity".
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

### 5.1 Buffered delivery from hosts that can be interrupted (D62)

An interruptible host can vanish with no notice, and a streamed response is by then partly in
the app's hands. So, **by host kind**, the router may hold a response until it is whole:

| Host kind | Default | |
|---|---|---|
| `rented-interruptible` | `buffered` | The operator may set `stream` |
| `rented-on-demand`, `fixed-remote`, `local` | `stream` | These do not vanish without notice |

`GET /pool/status` publishes the policy under `delivery`, so an SDK can size its
time-to-first-byte from it; the pool's own SDK widens its budget to the published figure and
never shortens it.

A buffered response is delivered as **the same frames, verbatim and in order**, in the engine's
own streaming format: an app that asked for a stream still parses a stream, and the body is
byte-identical to a direct one. Only the timing differs — **time to first byte becomes the whole
generation**, so rule 3's note on non-streaming requests applies, and `GET /pool/status`
publishes the delivery policy for the SDK to size its budget from. The "between bytes" timeout
does not run while the pool is buffering.

If the host is lost while buffering, the app has received nothing, so the pool runs the request
again itself (rule 2). Generation has no side effects — a tool call is returned to the app, never
executed by the pool. If no host can take it, the app gets a retryable `503`.

- A response that outgrows `max_buffer_mb` (default 16) is flushed and streamed from there on,
  marked `stream-after-overflow`, rather than failed.
- Cancel-on-disconnect is unchanged: the app's connection stays open while the pool buffers.
- An engine plug-in may declare a keep-alive frame that is harmless in its stream format, sent
  while buffering so that an intermediary's idle timeout does not cut the connection. Where an
  engine declares none, nothing is sent.

## 6. Versioning

The contract on this page has a version, reported in `GET /pool/status` and in an
`X-GPM-Contract` response header. Adding an optional dialect item is a minor change; changing
or removing one, or changing a status-code meaning, is a major change. The SDK follows semantic
versioning and states which contract versions it speaks.
