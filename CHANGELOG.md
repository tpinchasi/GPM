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
| `gpm-server` | 0.37.0 | **A host may name `engine: auto`, and a vLLM its owner installed is started once its models are on disk** (S10, D160, D161): an `auto` host's agent chooses vLLM or Ollama from the machine, and the pool publishes the engine it chose, holds that engine's builds, routes each model to them and says why on the Hosts screen (`engine_chosen`); until the agent has chosen, nothing is held. The configuration refuses `auto` without an agent, a model held there without a build for both engines, a pinned `engine_version`, and `auto` as the pool's or the rented machines' engine. A host whose owner installed the engine and gave its agent a restart command is restarted once every model it holds is on disk — never while ready, never with a request in flight, once per set (amends D41); an engine the agent runs itself is never restarted by the pool. **Control API:** `POST /pool/hosts/{id}/engine/restart` now answers `202` at once and runs the restart in the background, its outcome an event (D162). The upgrade pass waits on a switch as long as the agent says it can take (`switch_bound_s`, D153). |
| `gpm-client` | 0.7.0 | **`transcribe()`, `rerank()` and `image_part()`** (S8): speech to text as an upload, ranking documents against a query, and a picture inline in a chat message; the new refusals (`kind_not_served`, `input_not_accepted`, `media_not_inline`, `bad_upload`, `request_too_large`) raise `PoolRequestError`, and `504 generation_timeout` is not retried. Nor is the SDK's own read timeout, and `transcribe()` reads for at least 900 s; `rerank()` returns a `Ranking`, a list carrying the call's facts; file names are sent in a form the pool reads. Previously (0.6.0): **`workload(models={...}, placement=...)` and `plan_workload(models=...)`**: several models in one workload, each with its own target (D118); `Workload.models`. Previously (0.5.0): **`WorkloadProvisioner`**: `workload(...)` creates a workload with a provisioning key, makes its key locally (only the hash is sent), returns a client bound to it and ends it on leaving the block; `plan_workload(...)`; `certs=True` with the `certs` extra makes a key pair and a signing request. `pool_transport(transport=…)`. Previously (0.4.0): **`PoolClient.directory()`** and its async twin read what the pool could serve (D101). Previously: **`PoolClient` and `AsyncPoolClient` call the `/v1` paths by default** (D89), so one call reaches a pool of either engine; both reply shapes are understood. Previously: the SDK widens its time budget to the figure the pool publishes (D62) A create whose answer was lost is sent again — the same request, one workload. |
| `gpm-agent` | 0.15.0 | **The agent installs and runs vLLM on a Linux machine with an NVIDIA card, and chooses its engine under `engine: auto`** (S10, D157–D161). With `install_engines`, a version is a private Python environment filled only from the locked set the agent carries for its Python and processor — every wheel pinned and checked by its sha256, wheels only, no dependency resolved at install time — refused before anything is fetched where the machine's glibc, card (compute capability 7.5) or driver falls short; `tools/vllm-lock` makes the sets at each release (vLLM 0.32.0: Python 3.11–3.14, x86_64 and aarch64, CUDA 13, driver 580.65.06 or later). The agent starts its own launcher with that version's `vllm` (`--vllm`, accepted only inside its engines directory) once every model it holds is on disk, again when that set changes, stops it with `gpm-agent vllm-stop`, and never stops a vLLM it did not start; a model that will not start is that model's failure, not the version's. `engine: auto` chooses vLLM on Ampere or newer with the set's driver, Ollama anywhere else, says why under `engine.choice`, and hands over from the engine it ran before. The facts carry each NVIDIA card's compute capability and driver; `prerequisites` covers Python's `venv`, the set and the driver, and under `auto` the engine it would choose. From the review: the facts say how long one switch can take (`engine.install.switch_bound_s`) and which models are still loading (`models_loading`, never `awaiting_restart`); only the models the pool holds are started (`vllm-start --models-from`); the launcher signals only its own processes by their exact argument lists, within the boot it recorded, each with its process group; a crashed vLLM behind the router is started again; the audio check gets none of the agent's credentials; staging failures back off up to six hours. |

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
