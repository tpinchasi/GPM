# Changelog

The packages follow [semantic versioning](https://semver.org). Three things carry their own
version numbers and are reported at runtime, because other people's code depends on them:

| Versioned separately | Where it is reported |
|---|---|
| The **app contract** | `X-GPM-Contract` on every response, and `GET /pool/status` |
| The **control API** | `GET /pool/status` |
| Each **plug-in interface** | `interface_version` on the provider, engine and strategy protocols |

## Versions

These are the versions on `main`. Each [GitHub release](https://github.com/tpinchasi/GPM/releases)
carries the three packages' wheels and says which versions it covers; the newest is v0.45.0. **Each package's
version is raised in the same change that alters it** — a consumer cannot tell what it has
otherwise — and CI refuses a pull request that edits a package's source without raising it.

| Package | Version | What it carries |
|---|---|---|
| `gpm-server` | 0.49.0 | **Workers the operator reserves** (S19 stage 4, D199): a serving plan's `reserve: {local: n}` / `{fixed: n}` gives a workload the first call on that many of the operator's own workers — taken before a waiting shared request and beyond `borrow_share`, used by shared requests while idle, nothing running stopped — within `workloads.max_reserved_share` (0.5) of each kind, scaled pro rata to the ready ones. **`steady: local|fixed`** runs a workload on its reserved workers, renting nothing, answering `steady/local`. A program never reserves; where a workload runs is fixed at creation. *0.48.0:* **A budget per source, under one hosted total** (S19 stage 3, D198–D200): a serving plan may have several hosted sources, each paying from its own budget (its name, or `<role>/<type>` unnamed; `hosted-2` for the second in a role) under the workload's one hosted total, so renaming a budget gives nothing back; the first source of a role in force whose `providers` include the hosted API, whose wait is over and whose budget and the total hold the worst case pays, and is the answer's `X-GPM-Route`; the request log records `budget`; `/pool/status` shows each source's budget under `budgets.sources`. **`providers`** (hosted host ids, in order; checked against the pool). A rented source's **`kind`** (`roi`, `on_demand`, `interruptible`) for its burst hosts. **`during: scale_up`**: the supervisor marks a scale-up gap while a host rented for load is prepared. |
| `gpm-client` | 0.11.0 | **`priority=` and `keep_on=`** on `PoolClient` / `AsyncPoolClient` (for every call) and on each call (S19, D201), checked before sending and refused where the pool's status does not list them (`PoolOptionNotHonoured`, nothing sent); **`reply.route`** (`X-GPM-Route`) on every result; `503 no_allowed_source` raises `PoolUnavailable` at once. *0.10.0:* **`serve=`** on `WorkloadProvisioner.workload()` and `plan_workload()`, and **`Workload.set_serve()`** (S19): a serving plan in place of `bursts` (the two together are refused); `bursts=` on `plan_workload()` too. |
| `gpm-agent` | 0.19.0 | **One Ollama per card** (S17, D190–D192): on a machine with more than one NVIDIA card, where the models held fit one card, Ollama runs once per card (`CUDA_VISIBLE_DEVICES`, its own loopback port) behind the router's new Ollama mode (`proxy --engine ollama`, inference paths only); models are pulled once into the shared store (`OLLAMA_NOPRUNE` on all but the first) and pinned on every server, and count as loaded only where every server holds them. Until the pool says what it holds, the layout is planned from the models on disk, or one server. New launcher commands `ollama-start` / `ollama-stop` for the pool's restart script; facts `engine.layout`, `engine.copies`, `models_loaded_anywhere`. No new protocol verb. *0.18.1:* **vLLM 0.34.0 and 0.33.0** (D158's sets, regenerated for the release): the agent carries locked sets for vLLM 0.34.0 and 0.33.0 (Python 3.11–3.14, x86_64 and aarch64; CUDA 13, driver 580.65.06 or later, glibc 2.31 on x86_64 and 2.34 on aarch64) in place of 0.32.0's, so a machine's `latest` is 0.34.0 and a rollback has 0.33.0. A machine running 0.32.0 keeps it until it upgrades. *0.18.0:* **ONNX Runtime installed on Windows** (S16, D185): locked sets for Windows x86_64, the processor's and CUDA's (`onnxruntime-gpu` with NVIDIA's Windows wheels; least driver 580 for CUDA 13), ONNX Runtime 1.32.0 for Python 3.11–3.14; an environment's interpreter found in `Scripts\python.exe`. *0.17.2:* **Fixed: the same, across a restart of the agent.** What a failed install or check wrote is kept in the install record (`state.json`, `failure`), so an agent restarted between the failed check and the one that succeeds takes the error out too. An error written since, by a start that failed, still stays. *0.17.1:* **Fixed: an engine install kept saying a check had failed after a later one succeeded.** Where asking GitHub for the latest release failed (a lookup that failed for a moment, say) and the next check found the release already running, nothing was staged and the old error stayed in `engine.install.error` until the next release was installed. A check that succeeds now takes out the error a failed one left; a notice written by that check — a latest older than what runs, a version that failed here before — stays. *0.17.0:* **Trained models** (S15, D179–D182): the `onnx` engine — the pool's own ONNX server (numpy, Pillow, ONNX Runtime) on loopback, every ONNX build the machine holds in one process, inputs prepared and outputs read as declared, requests that arrive together batched, a made-up input run through each build as it loads; ONNX Runtime installed from locked sets the agent carries (`tools/onnx-lock`: Linux x86_64 and aarch64 with CUDA or the processor, Apple-silicon Macs; 1.32.0), or the owner's Python (`engine_binary`). The `container` engine — an image by digest pulled and run through the Docker Engine API, its port on loopback, not privileged, no new privileges, no added capability, no host directory; the machine's own registry login; `run_containers: never` (and `docker_socket`) in the agent's settings. `PUT /models` takes `specs`, checked as data; a seventh verb, `PUT /model-file` — bytes and their sha256, kept by the hash. Sets and the server are read as package resources, so they are found inside the archive rented machines get. *0.16.0:* **Ollama never reaches Ollama's cloud** (D168): the Ollama the agent runs starts with `OLLAMA_NO_CLOUD=1`; a desired set naming a `:cloud` or `-cloud` tag is refused (`400 cloud_tag`); the facts never list a cloud model a signed-in Ollama shows beside its own. *0.15.3:* **A model that fails is tried again ever less often, for as long as the pool asks** (D163, amended): 30 s, doubling, up to 10 min, reported per model (`attempts`, `retry_in_s`); a cause put right is found at the next attempt with nothing restarted. whisper.cpp's record of the server it started is read again when it could not be read, rather than taken for none for the agent's whole life; on Windows a folder refused ("Access is denied") says how to give it back (`icacls "<folder>" /reset /T`, as administrator). *0.15.2:* **Windows, run elevated** (D163, amended): the folders the agent makes — engines, versions, logs, state — inherit the profile's permissions on Windows, so a version staged by an elevated agent stays readable to the same user; the agent says when it runs elevated. **whisper.cpp says where it computes** (`engine.compute`: a card and its backend, or the processor). *0.15.1:* **An engine download says how fast it goes** (`engine.install.working.bytes_per_s`), so the pool can say a slow line from a stuck one (D163). *0.15.0:* **The agent installs and runs vLLM on a Linux machine with an NVIDIA card, and chooses its engine under `engine: auto`** (S10, D157–D161). With `install_engines`, a version is a private Python environment filled only from the locked set the agent carries for its Python and processor — every wheel pinned and checked by its sha256, wheels only, no dependency resolved at install time — refused before anything is fetched where the machine's glibc, card (compute capability 7.5) or driver falls short; `tools/vllm-lock` makes the sets at each release (vLLM 0.32.0: Python 3.11–3.14, x86_64 and aarch64, CUDA 13, driver 580.65.06 or later). The agent starts its own launcher with that version's `vllm` (`--vllm`, accepted only inside its engines directory) once every model it holds is on disk, again when that set changes, stops it with `gpm-agent vllm-stop`, and never stops a vLLM it did not start; a model that will not start is that model's failure, not the version's. `engine: auto` chooses vLLM on Ampere or newer with the set's driver, Ollama anywhere else, says why under `engine.choice`, and hands over from the engine it ran before. The facts carry each NVIDIA card's compute capability and driver; `prerequisites` covers Python's `venv`, the set and the driver, and under `auto` the engine it would choose. From the review: the facts say how long one switch can take (`engine.install.switch_bound_s`) and which models are still loading (`models_loading`, never `awaiting_restart`); only the models the pool holds are started (`vllm-start --models-from`); the launcher signals only its own processes by their exact argument lists, within the boot it recorded, each with its process group; a crashed vLLM behind the router is started again; the audio check gets none of the agent's credentials; staging failures back off up to six hours. |

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
