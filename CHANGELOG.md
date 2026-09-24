# Changelog

Both packages follow [semantic versioning](https://semver.org). Three things carry their own
version numbers and are reported at runtime, because other people's code depends on them:

| Versioned separately | Where it is reported |
|---|---|
| The **app contract** | `X-GPM-Contract` on every response, and `GET /pool/status` |
| The **control API** | `GET /pool/status` |
| Each **plug-in interface** | `interface_version` on the provider, engine and strategy protocols |

## Unreleased

Nothing is published yet, so these are the versions inside the repository. **Each package's
version is raised in the same change that alters it** — a consumer cannot tell what it has
otherwise — and CI refuses a pull request that edits a package's source without raising it.

| Package | Version | What it carries |
|---|---|---|
| `gpm-server` | 0.13.0 | **The rented engine and model placement are changed from the console (D98)** — one write through the file, the plan and the confirmation, with builds marked by engine and configured hosts keeping their models; a model rented for with no build for the rented engine is refused at load; a refused configuration reads as plain sentences. **A rented vLLM host is prepared end to end (D97)**: its agent is installed without waiting for the engine, the engine is started once every model it was bought for has landed, and a host is ready when it holds what it was bought for. vLLM gets a built-in start command; an `engine_start` written for another engine is refused at load; the restart script no longer fails when a start leaves no background job. **The console shows which engine runs where and what each host was bought for** (D93, D94): an Engine column and holdings on configured hosts, a Serving column on rented ones, and the Engine panel naming every engine in use and whether a router fronts several processes. **Both placement shapes are now selectable**: `rented.engine_proxy` puts several engine processes behind one port on a machine (D96), and a host is bought for the model whose traffic is waiting while the last host serving a model is never torn down (D95). **A rented host is bought for a model (D94)** — `rented.models` is the set the pool may rent *for*, each machine is given the model fewest hosts serve, and is prepared only for that one; any host asked for more models than its engine holds is refused at load. **A different engine on each host (D93)** — `engine:` per host and `rented.engine` for the machines the pool buys, catalog variants that name the engine they are for, a request read by the path it arrived on and routed only to hosts whose engine serves that path. Supersedes "a pool has one engine type". **The engine build chosen per machine (D92)** — `rented.images` lists builds newest first with the driver each needs, a machine no build runs on is refused before it is bid on, `engine_port` defaults to the engine's own, and an image built for a different engine than the one configured is refused at load. Every configuration row on the Rented capacity screen is now built by one control factory, so the search and allocation sections gained the sliders the tear-down section had. **vLLM as a second engine (D90)**, the OpenAI-shaped `/v1` surface both engines serve (D89), `Engine.occupancy()` (D91), and `pool.models_per_host` — the pool's set may be spread across hosts rather than held on each one (D89). Previously: dynamic allocation (D66), buffered delivery (D62), the host agent on rented hosts (D63), per-host worker adjustment (D67, D68), the machine history (D69), allocation edited from the console (D74), per-GPU pricing and the console levers (D85–D88), and the spending-path fixes D58–D65, D71, D73, D84 |
| `gpm-client` | 0.3.0 | **`PoolClient` and `AsyncPoolClient` call the `/v1` paths by default** (D89), so one call reaches a pool of either engine; both reply shapes are understood. Previously: the SDK widens its time budget to the figure the pool publishes (D62) |
| `gpm-agent` | 0.6.0 | **Fetches for vLLM while it is not running, and starts it (D97)**: `gpm-agent vllm-start` launches one engine, or one per model behind the router with memory split by weight size; a model is on disk only once every file has landed, and a download cut short is refused rather than accepted; the router lists only engines that answer; `init --models-path`. **Carries the router** that puts several engine processes behind one port (`gpm-agent proxy`), run as its own process by the machine's start-up — the agent itself stays off the request path (D96). **Told which engine it is minding** (`--engine`), so a host the pool rents holds models in that engine's terms rather than the one the agent happened to default to (D93). **Fetches model weights from a hub itself**, over plain HTTP and resumable by range request, for engines that have no pull of their own (D90); vLLM joins Ollama in its engine registry, with no new protocol verbs. Previously: the heartbeat verb, each model loaded as its own download finishes (D57, D83), and engine settings written where the pool installed it (D63) |

### One change a caller can see

`PoolClient.chat()` and `.embed()` now post to `/v1/chat/completions` and `/v1/embeddings`
rather than Ollama's native paths. Ollama serves both, so a pool of Ollama hosts answers either
— but the **reply shape differs**, and `format=` is sent as `response_format`. Code reading
`reply.content` is unaffected: the SDK understands both shapes. Code reading `reply.raw` should
either expect the OpenAI shape or construct the client with `api="ollama"` to keep the previous
behaviour exactly. **`pool_transport()` is unchanged** — it carries whatever the app sends.

The **app contract stays at 1**: every dialect item added is optional, and no status code
changed meaning, so an application built against the first version still works untouched.


The first release is being prepared; see `docs/release-checklist.md` for what remains.

### Added

- **The router and the client SDK** over statically configured hosts, across three transports
  (`http`, `https`, and a supervised SSH `tunnel`). Requests pass through in the engine's own
  API: streaming, tool calls, structured output and cancel-on-disconnect behave the same
  through the pool as directly against the engine.
- **The supervisor**: leases with mandatory dollar caps, overflow-driven renting one host at a
  time, `floor_plus_premium` bidding under two ceilings, eviction recovery, idle release,
  parking with reuse, the orphan sweep, verified destroy, and spend reconciled against the
  provider's own reported charges.
- **A dead-man timer on every rented host**, armed before anything else in its start-up script
  and using the provider's instance-scoped credential, so a host ends itself if the pool goes
  silent.
- **The operator console** at `/ui`: hosts by tier, lease burn-down, the live market through
  your own offer policy, every decision with the numbers behind it, and configuration with
  validate → plan → apply, history and rollback.
- **Two provider plug-ins** (a scriptable fake and Vast.ai) and one engine plug-in (Ollama),
  loaded through the same entry-point mechanism a third party's would use.

### Known limits at this point

- One engine and one marketplace provider.
- No multi-pool management, no per-client limits inside a pool, and no data-trust levels per
  host — so **the operator of a rented marketplace host can read everything sent to it**.
- Advanced bidding strategies and market replay are deliberately not included; see
  `docs/roadmap.md` §1.
