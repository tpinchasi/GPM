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

### 3. A workload of your own, created from code

A **workload** is capacity reserved for one program: its own hosts, its own key, its own latency
target, and a dollar budget it cannot pass. An operator can create one for you, or, if they give
your program a **provisioning key** (`gpmp_…`), the program creates, uses and ends its own,
entirely from code.

**What you need from the operator.** A provisioning key and the limits it carries. The operator
makes it with:

```sh
gpm provisioner create my-app --models gemma4:26b,gemma4:e4b \
    --max-spend 5 --max-spend-per-day 10 --max-hours 4 --max-open 1
```

The key can create workloads only for those models, up to those limits. It cannot request a
completion and cannot reach the control API. Put it in `GPM_PROVISIONING_KEY`, and the pool's
URL in `GPM_URL`.

**Price it first, then create it.** Pricing spends no money. Creating opens a lease and starts
renting hosts:

```python
from gpm_client import WorkloadProvisioner, WorkloadRefused

with WorkloadProvisioner() as provisioner:            # reads GPM_URL and GPM_PROVISIONING_KEY
    plan = provisioner.plan_workload("gemma4:26b", latency_s=30, parallel=30, hours=2, max_spend=5)
    if plan["refused"] or not plan["within_grant"]:
        raise SystemExit(plan["refused"] or "outside this key's grant")
    print(plan["hosts_at_start"], plan["placement"], plan["reasons"])

    with provisioner.workload("gemma4:26b", latency_s=30, parallel=30, hours=2, max_spend=5) as w:
        reply = w.client.chat("gemma4:26b", [{"role": "user", "content": "hello"}])
        print(reply.content)
    # leaving the block ends the workload, and its hosts are released
```

- `latency_s`: the 95th-percentile answer time each request should meet.
- `parallel`: how many requests you will send at once. The pool sizes hosts for it.
- `hours`: the most it runs. `max_spend`: the most it spends, in dollars. Both are hard limits.
- `machines`: `"roi"` (the default) lets the pool choose by expected cost among interruptible
  machines (a bid or a provider's spot price — cheaper, and can be taken away) and on-demand ones
  (a fixed price). `"on_demand"` and `"interruptible"` fix the choice.
- `idle_end_minutes`: once serving, the workload ends after this many minutes with no request.
  It defaults to the grant's (usually 15).

A plan searches every provider the pool rents from, and providers limit searches by a daily
quota, so plan once rather than in a loop.

**Several models in one workload.** Each model gets its own target. Send requests for any of
them through the same `w.client`:

```python
w = provisioner.workload(
    models={"gemma4:26b": {"latency_s": 30, "parallel": 30},
            "gemma4:e4b": {"latency_s": 30, "parallel": 30}},
    placement="auto",          # "together": every host holds both; "apart": hosts per model;
    hours=1.75, max_spend=5,   # "auto": whichever is expected to cost less
)
```

**Waiting for it.** `workload()` returns as soon as the workload is created. Its own hosts take
minutes to rent, download the models and start. Meanwhile your requests may be answered on the
pool's shared hosts, a small share of them, if the grant allows borrowing. The SDK waits and
retries when they are busy. To wait for hosts of its own instead:

```python
w = provisioner.workload("gemma4:26b", latency_s=30, parallel=30, hours=2, max_spend=5,
                         wait_until="serving", timeout_s=1800)   # ends it again if it never serves
```

`w.state()` returns its state at any time: `preparing`, `serving`, `ending` or `ended`, whether
it is borrowing, its hosts and their busy workers.

**Without a `with` block**, for a service that keeps a workload across many calls:

```python
w = provisioner.workload("gemma4:26b", latency_s=30, parallel=30, hours=4, max_spend=10)
...                         # w.client, w.state(), w.name, w.models
w.end()                     # always end it when done; ending twice is harmless
```

**Using your own HTTP client.** `w.key` is the workload's key and `w.base_url` the pool's URL.
Use them as in section 1, with `pool_transport()`. The key is made in your program and only its
hash is sent, so the pool never holds it. **Keep it in memory only:** if the process loses it,
nothing can recover it.

**When creating is refused.** `workload()` and `plan_workload()` raise `WorkloadRefused`, with
the pool's reason in `detail`, for example:
- "this key has 1 workload(s), at its limit of 1";
- "at most $5.00 per workload";
- "programs committed $8.00 in the last 24 hours; $5.00 more would pass the pool's $10.00 a day".

For the daily limit, a workload that has ended counts what it actually spent; one still open
counts its whole budget.

**If your program dies without ending it**, the pool ends it on its own:
- once serving, after its idle cutoff with no requests;
- whatever its state, when its hours or its budget run out.

It never spends more than `max_spend`. A program that is restarted cannot take back a workload
it made, because the key existed only in the old process. Create a new one; the old one ends at
its idle cutoff.

**Client certificates.** `workload(..., certs=True)` also has the pool sign a client
certificate, and the SDK uses it for every request (`pip install 'gpm-client[certs]'`). The
private key never leaves your machine. Some grants require it.

## Configuration

| Variable | Default | What it sets |
|---|---|---|
| `GPM_URL` | `http://127.0.0.1:8080` | The pool's app-facing endpoint |
| `GPM_API_KEY` | — | Your app key. **Required**, loopback included |
| `GPM_MAX_WAIT_S` | `1800` | Total seconds the SDK will spend waiting for capacity |
| `GPM_PROVISIONING_KEY` | — | A provisioning key, for `WorkloadProvisioner` only (section 3) |

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
operator and the admin key. The one exception is section 3: a program the operator has given a
provisioning key may create and end its own workloads, within the limits the operator set. If you find yourself wanting one of them in application code, the
thing you actually want is for the operator to change the pool's configuration.

## Versioning

`X-GPM-Contract` and `GET /pool/status` report the contract version; this SDK speaks contract
`1`. Adding an optional header is a minor change; changing the meaning of a status code is a
major one. The full normative contract is
[docs/spec/app-contract.md](../docs/spec/app-contract.md).
