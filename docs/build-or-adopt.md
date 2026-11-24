# Build or adopt

> Checked on 2026-09-17 against each project's own documentation, before GPM was built; kept as
> the evidence for [D25](decisions.md). The projects have moved on since — re-check before
> relying on any cell.

Checked 2026-09-17 against each project's own docs. No framework covers the
whole requirement; the split is clean — some route, some provision, none do both, and **none
offers programmable bidding**.

| Requirement | Olla | LiteLLM proxy | SkyPilot / SkyServe | dstack |
|---|---|---|---|---|
| App sees one endpoint, unaware of hosts | yes | yes | yes | yes (gateway) |
| Native Ollama API kept (`format` schema, tools) | Ollama-aware, under a `/olla/ollama/` prefix; byte-for-byte `/api/chat` passthrough **not confirmed** in docs | **no** — clients must speak OpenAI `/v1/chat/completions`; translated to Ollama | n/a (plain HTTP LB to replicas) | n/a |
| Backends over `http` / `https` URL | yes | yes | replicas it launched only | replicas it launched only |
| Backends over SSH tunnel | not itself — point it at a tunnel's local port | same | no | no |
| Mix local + fixed + rented hosts in one pool | yes, as static URLs | yes, as static URLs | **no** — cloud replicas only; controller must be a cloud VM | no |
| Add/remove hosts at runtime (evicted host replaced) | **no** — static YAML, no documented reload or API | believed yes (model-management API; not re-checked) | yes, its own replicas | yes, its own replicas |
| Per-host concurrency limit | not documented | rpm/tpm limits, not worker counts | not documented | — |
| Health checks, failover, retry on another host | yes | yes | yes | yes |
| Session affinity (KV-cache reuse) | yes | not checked | no (least-load only) | not checked |
| Provisions Vast.ai | no | no | yes | yes, **on-demand only** — no interruptible |
| Interruptible/bid instances on Vast | — | — | yes (`use_spot`) | **no** |
| **Customizable bidding** (floor + premium, ceiling, raise-bid-on-same-machine to keep the disk, ingress-aware ranking, bandwidth class filter) | — | — | **no** — a static `bid_price` passed through `create_instance_kwargs`; recovery means relaunch elsewhere | — |
| Leases / dollar caps / idle release / orphan sweep | no | spend tracking per key, no infra control | autoscale-to-zero, autostop; no dollar caps | idle duration; no dollar caps |

Reading of the table:

- **The provisioning half had to be built.** A bidding policy that keeps a machine's disk by
  bidding again on the *same machine* after an interruption has no equivalent in any of them.
  SkyPilot's model is "preempted → launch a fresh replica", which on a marketplace means
  downloading the models again, at the provider's per-GB price, every time.
- **The routing half could have been adopted, with one blocker each.** Olla was the closest fit
  (Ollama-aware, affinity, health, failover) but had no documented way to change its endpoint
  list at runtime, and hosts in a pool come and go. The workable pattern would have been a stable
  local port per host (a tunnel or a small forwarder) named in Olla's static list, so that a dead
  host is just an unhealthy endpoint. Its costs: a second process, no per-host worker limit, a URL
  prefix apps would have to carry, and unverified `/api/chat` fidelity. LiteLLM was out unless
  apps moved to the OpenAI API.
- **Decision: write the router (FastAPI + httpx) and own the supervisor** — the router is the
  smaller half, and owning it buys per-host workers, exact passthrough, the request log, and
  hosts that appear and disappear at runtime (D25).

Sources: [SkyPilot on Vast](https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot),
[SkyServe](https://docs.skypilot.ai/en/latest/serving/sky-serve.html),
[SkyServe spot policy](https://docs.skypilot.co/en/stable/serving/spot-policy.html),
[dstack backends](https://dstack.ai/docs/concepts/backends/),
[Olla](https://github.com/thushan/olla), [Olla configuration](https://thushan.github.io/olla/configuration/overview/),
[LiteLLM Ollama provider](https://docs.litellm.ai/docs/providers/ollama).
