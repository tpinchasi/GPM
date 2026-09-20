# S2 — Reliable responses from interruptible hosts

> Status: **built (D62)**, 2026-09-20. Part of the [feature list](README.md). Delivered as
> planned, with one thing the plan had not seen: every rented host was published as
> `rented-interruptible`, on-demand ones included, so the router could not tell the kinds apart.
> On-demand rentals now publish as `rented-on-demand`. Stage 4's keep-alive frame is an engine
> hook that the first engine declares as `None` — newline-delimited JSON has no harmless frame —
> so nothing is sent and no untested machinery was built for it.

## The story

*As an app developer, I never receive half an answer. If the host serving my request is taken
away mid-generation, the pool runs it again somewhere else, and I see one complete response or
one clean, retryable error.*

## Why

An interruptible host can vanish with no notice and no graceful shutdown. A streamed response
is already partly in the client's hands when that happens: the app has consumed tokens it
cannot take back, and the only honest outcome is a broken stream.

Measured on the first live pool: **110 responses broke mid-stream — every one on a rented host,
none on the local one** — after an average of **13.3 s** of generation each, across 36
evictions. The router's one failover covers only a host that fails *before the first response
byte* ([app-contract.md](../spec/app-contract.md) §5); after it, nothing can be done.

## Design

**Hold the response until it is whole, then deliver it.** For a host whose kind can be
interrupted, the router reads the upstream stream into a buffer instead of relaying it. When the
engine finishes, the router sends the client everything at once.

**Delivered as the same frames, not as a different response.** The buffered frames are replayed
verbatim, in order, in the engine's own streaming format. A client that asked for a stream
still parses a stream; only the timing differs. Byte-for-byte fidelity of the body — tool calls,
structured output, the final usage frame — is preserved, which is the property phase 1 was built
to prove.

**Because the client has received nothing, the pool can recover by itself.** If the upstream
breaks while buffering, the router dispatches the request again on the next eligible host, up to
`max_redispatch` times and inside the request's deadline. This widens today's failover from
"before the first *upstream* byte" to "before the first *client* byte". Generation has no side
effects — a tool call is returned to the app, never executed by the pool — so running it again
is safe. If no host can take it, the client gets `503` with a retryable reason, and the SDK's
existing wait-and-retry does the rest.

**By host kind, the operator's choice, and disclosed.**

| Host kind | Default delivery | Why |
|---|---|---|
| `rented-interruptible` | **buffered** | The case above |
| rented on demand, `fixed-remote`, `local` | stream | They do not vanish without notice; streaming is what apps expect |

Every response says how it was delivered: `X-GPM-Delivery: buffered | stream` and
`X-GPM-Attempts: <n>`. An app that prefers tokens as they come and accepts the risk sends
`X-GPM-Delivery: stream` on the request, if the operator allows the override.

## What it gives up

On an interruptible host the app sees no tokens until the generation is complete: **time to
first byte becomes the whole generation time.** An interactive chat interface loses its typing
effect there. That is the honest price of the guarantee, which is why it follows host kind and
can be overridden per request.

The simpler mechanism — keep streaming, let the SDK retry a broken stream — was considered
first. It gives up the guarantee itself: by the time a stream breaks, the app has already acted
on the partial tokens.

## The time budget

The contract already covers this case: *"for non-streaming requests 'first byte' is the whole
response, so the time-to-first-byte budget must cover the full generation."* Buffered delivery
puts streamed requests under the same rule. `GET /pool/status` publishes the delivery policy so
the SDK sizes its time-to-first-byte accordingly, and the "between bytes" timeout no longer
applies while buffering.

**A silent connection may be cut by something in between** — a proxy or load balancer with an
idle timeout. The engine plug-in may declare a keep-alive frame that is harmless in its stream
format (a comment line in server-sent events, for instance). Where an engine declares none,
nothing is sent, and the operator is told that intermediaries must allow the wait.

## Bounds

- `max_buffer_mb` per response (default 16). A response that outgrows it is flushed and
  streamed from there on, marked `X-GPM-Delivery: stream-after-overflow`, rather than failed.
- Cancel-on-disconnect is unchanged: the client connection stays open while the pool buffers,
  so a client that leaves still frees the worker at once.
- Re-dispatch counts against the request's deadline, and is logged with each host tried.

## Configuration sketch

```yaml
pool:
  delivery:
    rented_interruptible: buffered    # buffered | stream
    allow_request_override: true
    max_redispatch: 1
    max_buffer_mb: 16
```

## Decisions this story rests on (D62)

1. Buffered delivery by host kind, with the defaults above.
2. Failover widened to "before the first client byte" — **amends app-contract §5 rule 2**.
3. The new dialect items. Adding optional items is a minor contract change; the behaviour change
   is operator-configured and disclosed on every response, so it is not silent.

## The owner's answers

Accepted as recommended: **on by default** for interruptible hosts, because a broken stream is
the worse surprise; **an app may override it per request**, where the operator allows; and
**`max_redispatch` is 1**, since each attempt can cost a full generation time.

## Build stages

1. Buffer and replay for one host kind; the two response headers; request log outcome
   `redispatched`.
2. Re-dispatch on upstream failure while buffering, inside the deadline.
3. Status publishes the policy; the SDK sizes its budget from it; the overflow path.
4. Engine-declared keep-alive frame.

## Tests

The fake engine already breaks streams on command. A broken stream on an interruptible host
yields one complete response from another host; on the last host it yields a clean `503`; the
replayed body is byte-identical to a direct stream; a disconnecting client frees the worker
mid-buffer; an oversized response overflows into a stream.

## Depends on

Nothing. **S1** and **S5** read latency signals, and buffered delivery changes what "time to
first byte" means on these hosts — both must use service time, not first-byte time.
