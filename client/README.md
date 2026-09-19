# gpm-client

The client SDK for a **GPM pool** — a single endpoint that serves GPU inference from whatever
hosts the operator has: their laptop, a fixed remote box, or an instance rented on a bidding
marketplace for the next twenty minutes. Your application never learns which.

You need three things from whoever runs the pool: **a URL, an app key, and the names of the
models it serves.** Nothing else — not host lists, not provider names, not bids.

```sh
pip install gpm-client        # one dependency: httpx
```

## The thing worth knowing first

A pool's capacity moves. A rented host can be outbid mid-afternoon and replaced by another;
every worker can be busy; a host can be loading models. So the pool answers "not now, try in
`n` seconds" far more often than a fixed server does, and **this SDK waits and retries by
default**. You opt *out* of waiting, not in.

That single decision is why the SDK exists. Without it every app that talks to a pool grows its
own retry loop, and they all get it subtly wrong in the same places: retrying a stream that has
already delivered half its output, retrying a bad API key forever, or ignoring the
`Retry-After` the pool actually sent.

## Two ways in

### 1. Inject the transport (the main path)

The SDK's core is an **`httpx` transport**, not a wrapper function. Give it to any library that
lets you supply an `httpx` client, and the waiting happens underneath code you do not change:

```python
import httpx
from gpm_client import pool_transport

http = httpx.Client(
    base_url="http://127.0.0.1:8090",
    headers={"Authorization": f"Bearer {APP_KEY}"},
    transport=pool_transport(),
)

# From here on this is the inference engine's own API, passed through untouched.
reply = http.post("/api/chat", json={
    "model": "gemma4:26b",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": False,
})
print(reply.json()["message"]["content"])
print(reply.headers["X-GPM-Served-Model"])   # e.g. gemma4:26b-mlx
```

Use `async_pool_transport()` with `httpx.AsyncClient`. Both take an optional `RetryPolicy`.

**The request and response bodies are the engine's own API.** GPM does not translate between
API shapes — that is where tool calls and structured output get quietly lost — so whatever you
would send to the engine directly, you send here. Ask the operator which engine is behind the
pool; today that is Ollama, so `/api/chat`, `/api/generate` and `/api/embed`.

### 2. `PoolClient`, for scripts

When you just want the text and are not already inside some client library:

```python
from gpm_client import PoolClient

pool = PoolClient()                    # reads GPM_URL and GPM_API_KEY
reply = pool.chat("gemma4:26b", [{"role": "user", "content": "hello"}])

reply.content         # the text
reply.served_model    # "gemma4:26b-mlx" — the build that actually ran
reply.host            # "laptop"
reply.wait_s          # 0.042 — seconds queued, not generation time
reply.raw             # the engine's whole response body, untouched

vectors = pool.embed("nomic-embed-text:latest", ["some text"])
pool.close()          # or use it as a context manager
```

`AsyncPoolClient` is the same with `await`, `aclose()` and `async with`.

For a batch driver that should not start until there is somewhere to run:

```python
pool.wait_until_ready(timeout=600)     # raises PoolUnavailable if nothing becomes ready
```

## Configuration

| Variable | Default | What it sets |
|---|---|---|
| `GPM_URL` | `http://127.0.0.1:8080` | The pool's app-facing endpoint |
| `GPM_API_KEY` | — | Your app key. **Required**, loopback included |
| `GPM_MAX_WAIT_S` | `1800` | Total seconds the SDK will spend waiting for capacity |

`PoolClient(base_url=..., api_key=...)` overrides the first two per client.

The app key is **not** the admin key, and the pool will refuse it if you try: an application
that can ask for a completion must not thereby be able to open a lease or release a host. If
you are handed a key starting `gpmx_`, that is the wrong one — ask for a `gpma_` key.

## What the SDK does when the pool says "not now"

| What happened | What the SDK does |
|---|---|
| `503` with a `Retry-After` | Waits exactly that long, then retries |
| `503` without one | Exponential back-off, 2 s → 60 s, with jitter |
| Transport error **before** any byte arrived | Retries on the same back-off |
| Failure **mid-stream**, after output was delivered | **Stops.** Raises `PoolStreamInterrupted` |
| `503 no_lease` | **Stops.** Raises `PoolUnavailable` |
| `401` | Raises `PoolAuthError` |
| Any other `4xx` — e.g. `404 model_not_in_pool` | Raises `PoolRequestError` |
| Total wait exceeds `max_wait_s` | Raises `PoolUnavailable` |

Three of those deserve their reasoning:

**Mid-stream failure is never retried.** Your code has already consumed half an answer. Only
you know whether re-running the call is correct, so the SDK refuses to guess.

**`no_lease` fails immediately**, unlike every other `503`. A lease is the operator's spending
authority; without one, the pool is not permitted to rent capacity, so waiting cannot possibly
help. Waiting anyway would burn your `max_wait_s` for nothing. If you genuinely want to sit
there until an operator opens one, pass `RetryPolicy(wait_without_lease=True)`.

**Retrying is safe here** because a chat or generate call has no server-side effect to
duplicate — the whole conversation travels with each request, and your tools run in your
process, not the engine's.

### Tuning it

```python
from gpm_client import PoolClient, RetryPolicy, pool_transport

policy = RetryPolicy(
    max_wait_s=None,                 # None = wait indefinitely; what batch drivers usually want
    backoff_initial_s=2.0,
    backoff_max_s=60.0,
    wait_without_lease=False,
    on_wait=lambda reason, waited_s, next_try_s: print(
        f"pool says {reason}; waited {waited_s:.0f}s, retrying in {next_try_s:.0f}s"
    ),
)

pool = PoolClient(retry_policy=policy)
transport = pool_transport(policy)   # or the same policy on the transport path
```

There is no logging while it waits — if you want to see that, pass `on_wait`. Without it a long
wait is silent, which is the wrong default for anything running unattended.

## Errors

```python
from gpm_client import (
    PoolError,               # base class: catch this to catch everything
    PoolAuthError,           # 401 — wrong or missing app key
    PoolRequestError,        # other 4xx; carries .status_code, .reason, .detail
    PoolUnavailable,         # no capacity within the time you allowed; carries .reason
    PoolStreamInterrupted,   # died after partial output
)
```

`PoolUnavailable.reason` is the pool's machine-readable cause — `queue_timeout`, `preparing`,
`recovering`, `hosts_unreachable`, `no_offer`, `no_lease`, `no_eligible_host` — which is what
you want in a log line, rather than a stack trace.

## Streaming

Stream through the transport path with `httpx` directly; `PoolClient.chat` is non-streaming.

```python
with http.stream("POST", "/api/chat", json={
    "model": "gemma4:26b",
    "messages": [{"role": "user", "content": "count to five"}],
    "stream": True,
}) as response:
    for line in response.iter_lines():
        ...
```

Everything above about mid-stream failure applies: once bytes have arrived, an interruption
raises `PoolStreamInterrupted` rather than silently starting over.

**If you stop reading, the pool stops working.** Closing the response cancels the upstream
request and frees the worker immediately — which on a rented host means it stops costing the
operator money. This is worth knowing if you are tempted to leave streams open.

## What each answer tells you

Response headers, on every call:

| Header | Meaning |
|---|---|
| `X-GPM-Served-Model` | The build that actually ran. May differ from what you asked for |
| `X-GPM-Host` | Which host served it |
| `X-GPM-Runtime-Class` | e.g. `apple-mlx`, `cuda-gguf` |
| `X-GPM-Wait-S` | Seconds spent queued — so queue time is never mistaken for generation time |
| `X-GPM-Contract` | The contract version this pool speaks |

`X-GPM-Served-Model` differing from your request is **by design**. If the operator has
catalogued `gemma4:26b` as a logical name, the pool serves the build suited to the host that
got your request — the Apple-optimised one on a Mac, the standard one on a CUDA box. A name the
operator has *not* catalogued is passed through literally and is never substituted, and no
request can ever cause a model to be downloaded or loaded.

## Optional request headers

Send these yourself on the transport path; `PoolClient` exposes `session_id` only.

| Header | What it does |
|---|---|
| `X-GPM-Session` | Groups your calls in the pool's request log, for an offline join later |
| `X-GPM-Deadline` | Epoch seconds or ISO-8601. After this instant the answer is useless to you; the pool drops the work rather than paying to finish it. You get `504 deadline_exceeded`, and it is never retried |
| `X-GPM-Runtime-Class` | Pins the request to hosts serving that class of build — for a run that must not mix builds mid-way |

## Timeouts

The SDK sets them so you do not have to: **5 s** to connect, **300 s** for a read. The pool's
own queue timeout is always configured below the SDK's read timeout, so a queued request gets a
real answer — a worker or a `503` — before your client would give up on it. Override with
`PoolClient(timeout=httpx.Timeout(...))` if you must, but lowering the read timeout below the
pool's queue timeout re-introduces exactly the abandoned-work problem the contract prevents.

## What this package deliberately does not have

No host lists, no provider names, no bids, no leases, no tear-down. Those belong to the
operator and the admin key. If you find yourself wanting one of them in application code, the
thing you actually want is for the operator to change the pool's configuration.

## Versioning

`X-GPM-Contract` and `GET /pool/status` report the contract version; this SDK speaks contract
`1`. Adding an optional header is a minor change; changing the meaning of a status code is a
major one. The full normative contract is
[docs/spec/app-contract.md](../docs/spec/app-contract.md).
