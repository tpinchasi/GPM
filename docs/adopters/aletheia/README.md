# First Adopter Guide — Aletheia

> How the GPU Host Pool applies to *this* repository: the simulator (`chat-agents/`) and the
> per-turn labeling prototype (`prototypes/per-turn-behavioral-labeling/`). Everything specific
> to Aletheia lives here and nowhere in the generic docs. Tracked as `GPU-POOL-01` in
> `platform-design/unified/TASKS.md`.
>
> **Paths in this guide are relative to the Aletheia repository (`~/workspace/Aletheia`), not to
> this one.** The guide currently lives here, in the private GPM repository, so the work can be
> continued in one place; it should return to the Aletheia repository before GPM is made public
> ([../../release-checklist.md](../../release-checklist.md) §1.1, and `STATUS.md`).
>
> **Note (decision D12):** the scripts described here are the framework's first *consumer*.
> They are not inputs to its design — the generic spec is argued on its own terms.

Background: [gpu-cloud-migration-workplan.md](gpu-cloud-migration-workplan.md) — the earlier
plan for moving generation and labeling onto a rented GPU (parity gate, cost model). The pool
extends it; the parity gate's question is now answered by the measurement plan in §5.

## 1. What exists today, and where it stops

| Piece | Where | What it does | Limit |
|---|---|---|---|
| Provisioner | `prototypes/per-turn-behavioral-labeling/vast_provision.py` | Offer scoring (memory bandwidth per dollar, ingress-aware), bid = floor + premium, recovery in two steps (re-bid on the same machine, else replace), verified destroy, following tunnel, per-card parallelism formula | Exactly one host, one provider, identified by one label |
| Concurrency gate | `chat-agents/agents/ollama_gate.py` | One global semaphore | One number for "the" host, pushed in from outside by `vast_provision.sync_app_gate` |
| Host-loss handling | `chat-agents/agents/host_health.py` | Probe, transport-error classifier, poisoned-session span filter for Langfuse | Treats host loss as session-fatal — there is nowhere else to send the turn |
| Driver pause / recover | `generate_sessions.py` (`wait_for_gpu_host`, `_run_recover_if_due`) | Driver processes block until the host is back; one shells out to `recover` under a file lock | Recovery is triggered by whichever client notices |
| Endpoint config | `OLLAMA_URL` / `OLLAMA_BASE_URL` in `chat-agents/.env`; **hardcoded `localhost:11434`** in `langfuse_ollama.py`, `step3_match_and_assign.py`, `run_judge_eval.py`, `run_local_slm_poc.py` | One address per client | The labeling pipeline cannot reach a rented host without edits |

The knowledge in `vast_provision.py` — bid volatility, parked-instance billing, failing rather
than parking a lost bid, ingress cost, the pinned image, the team-context SSH-key workaround,
the prediction-based parallelism ceiling — is what the pool's default strategies encode. That
file is the reference implementation for the first provider plug-in, and is retired once the
pool has run a full generation cycle.

## 2. What changes in this repository

| Change | Where | Phase |
|---|---|---|
| Inject the SDK transport into the LangChain clients — one added argument, no call-site changes. Verified: `langchain-ollama` 1.1.0 forwards `client_kwargs` to `httpx.Client` via `ollama` 0.6.2, so `transport=` is accepted | `chat-agents/agents/agent_runtime.py` (`_chat_ollama`), the two RAG modules' `OllamaEmbeddings` | 1 |
| Send the pool's app key (`GPM_API_KEY`) and keep `OLLAMA_BASE_URL` pointed at the router — the `.env` already uses port 11435, which the router takes over | `chat-agents/.env` | 1 |
| Pass the simulator's existing `session_id` as `X-GPM-Session` | `chat-agents/app.py` simulation loop | 1 |
| Replace hardcoded `localhost:11434` `urllib` calls with `PoolClient` | `langfuse_ollama.py`, `step3_match_and_assign.py`, `run_judge_eval.py`, `run_local_slm_poc.py` | 1 |
| Replace string-marker error sniffing with the SDK's typed errors; poison a session on `PoolUnavailable` and `PoolStreamInterrupted` only. The Langfuse span filter itself stays — it is app business | `chat-agents/agents/host_health.py` | 1 |
| Retire `host_healthy` pre-flight and `wait_for_gpu_host` in favour of the SDK's default waiting and `pool.wait_until_ready()` | `chat-agents/app.py`, `generate_sessions.py` | 1 |
| Retire `sync_app_gate`. `ollama_gate.py` stays as a client-side cap chat-agents owns; `generate_sessions.py --workers auto` reads the total from `/pool/status` | `vast_provision.py`, `generate_sessions.py` | 1–2 |
| Retire `_run_recover_if_due` — clients wait, the supervisor recovers | `generate_sessions.py` | 2 |
| Set `OLLAMA_NUM_PARALLEL=3` for the Mac's Ollama, or the Mac's three workers will queue inside the engine and every latency figure will be wrong. Test-connection checks this. **Done through the pool since 2026-09-26:** the laptop agent's `restart_command` is `~/.config/gpm/restart-ollama.sh`, which sets the pool's environment (`~/.config/gpm/ollama.env`: parallelism 3, three models held, context 8,192) through `launchctl` and relaunches Ollama.app; the Hosts screen's *apply settings* runs it. After a reboot, Ollama.app comes back on its own defaults — one parallel slot, and a context it picks from memory — until *apply settings* is pressed once. **Seen 2026-09-26:** Ollama 0.34.3 (the Mac's app auto-updated from 0.32.15 at the 2026-09-23 reboot) over-estimates the memory of the Gemma 4 MLX builds, so `gemma4:26b-mlx` and `gemma4:e4b-mlx` no longer stay loaded together — a request for the E4B stops the 26B's runner with 19.4 GiB reported free, at any context length — where 0.32.15 kept them together for a month. Two smaller MLX builds still coexist, and an MLX runner and a llama-server (GGUF) runner coexist. Reported elsewhere, not tested here: the MLX runner serialises requests whatever `OLLAMA_NUM_PARALLEL` says. So until Ollama changes this, a chat app that alternates these two models on the Mac wants Ollama pinned to 0.32.15, or the E4B on a GGUF build with the catalog ordered so the Mac prefers it. **Done 2026-09-26:** `/Applications/Ollama.app` is 0.32.15 again (0.34.3 kept beside it as `Ollama 0.34.3.app`), validated with both builds loaded together; the app still checks for updates, so its *restart to update* menu item must not be clicked | the Mac's Ollama environment | 1 |
| Carry **model served** and **runtime class** into the exported sessions by joining the pool's request log on session id and timestamp | `chat-agents/scripts/export_langfuse_conversations.py` | 1 |
| Retire `vast_provision.py` | — | after the pool's first full generation cycle |

## 3. Example configuration

Real values for this repository. The numbers come from `vast_provision.py` and the
2026-09-16 measurements recorded in `platform-design/unified/progress.md`.

```yaml
pool:
  name: aletheia-sim
  models: ["gemma4:26b", "gemma4:e4b", "nomic-embed-text"]     # see §4 for the labeling pool
  context_length: 32768

router: { listen: 127.0.0.1:11435, queue_timeout_s: 30 }

hosts:
  mac:
    kind: local
    url: http://127.0.0.1:11434
    capabilities: [apple-silicon]
    workers: auto                       # profile: M4 Pro → up to 3

rented:
  vast:
    kind: rented-interruptible
    provider: vast
    transport: { type: tunnel }         # team-context API key: public key injected by the start-up script
    image: vastai/ollama:0.34.1         # there is no :latest tag; an unpinned image never starts
    offer_policy: { min_vram_gb: 64, memory_bandwidth_gbs: [1200, 2000],   # above 2000 was outbid within minutes, three times
                    min_disk_gb: 60, max_all_in_hourly: 0.60,               # the disk rented, and the most per host-hour all-in
                    max_download_per_gb: 0.01, exclude_gpu_names: ["CMP"] }
    bidding:  { strategy: floor_plus_premium, premium: 0.02 }

models:                                 # catalog
  gemma4:26b: { variants: [ { tag: "gemma4:26b-mlx", requires: [apple-silicon], runtime_class: apple-mlx, enforces_schema: false },
                            { tag: "gemma4:26b", runtime_class: by-platform } ] }
  gemma4:e4b: { variants: [ { tag: "gemma4:e4b-mlx", requires: [apple-silicon], runtime_class: apple-mlx, enforces_schema: false },
                            { tag: "gemma4:e4b", runtime_class: by-platform } ] }
  nomic-embed-text: { variants: [ { tag: nomic-embed-text } ] }

capacity_profiles:
  - { match: { capability: apple-silicon, chip: "M4 Pro" },   max_workers: 3 }
  - { match: { gpu: "RTX 6000*", min_vram_gb: 80 },            max_workers: 6 }
  - { match: { any: true },                                    max_workers: formula }

calibration:
  "gemma4:26b+gemma4:e4b@32k": { fixed_gib: 10.0, per_worker_gib: 7.6, operate_at: 0.67 }
```

Where the 6 comes from: on the RTX 6000D (85.6 GB) with both gemma models resident at 32K
context, 8 and 9 parallel slots held and 10–12 evicted a model — a memory ceiling of 9. Going
from 6 to 9 gave no extra throughput and about 1.8× per-session latency, so it is run at
two-thirds: 6. Halving the context doubles the ceiling.

## 4. Things specific to this repository to get right

- **Two pools, not one.** Generation and labeling have different model sets and different
  tolerance for mixed builds. A `aletheia-sim` pool (above) for the simulator, and an
  `aletheia-labeling` pool for the pipeline, fits the pool-as-isolation-unit model (D16).
- **The labeling pool's model set** is `qwen2.5:7b-instruct` (steps 1–2), the step-3 assignment
  model, and **`nomic-embed-text-v2-moe`** — the calibration centroids were built with that tag
  and silently break with any other. Note that `vast_provision.py` currently pulls
  `nomic-embed-text` (what the chat-agents RAG uses), **not** the `-v2-moe` tag: a rented host
  as provisioned today cannot serve step 3. Declaring the model set per pool fixes this by
  construction.
- **Steps 1–2 depend on JSON-schema enforcement**, and Ollama's MLX engine ignores the `format`
  parameter. That is what `enforces_schema: false` on the MLX variants is for: schema-constrained
  calls never resolve to them. On the Mac that means either the standard build is resident as
  well, or the Mac is not eligible for those calls. Step 3 parses leniently and is unaffected.
- **The labeling pool should pin a runtime class** (`X-GPM-Runtime-Class`): its accuracy
  baselines and centroids are class-specific.
- **`qwen2.5:7b-instruct` has no MLX build**, so it runs under the same tag on the Mac (Apple
  GGUF) and on a rented card (CUDA GGUF). This is why the per-call record is *(model served,
  runtime class)* and not the model name alone.

## 5. Measurement plan for mixed builds (decision D13)

With local-first routing, no affinity, and MLX resolved on the Mac, about one call in three
lands on the Mac when both hosts are busy (3 of 9 workers). Essentially every 12-turn session
will contain turns from both the MLX and the CUDA build. That is accepted; the first
full generation run through the pool is used to find out whether it matters.

1. Generate as normal. Every call is logged with model served, runtime class and host.
2. Join the pool's request log onto the exported sessions by session id and timestamp.
3. Compare, **served-by-`apple-mlx` vs. served-by-`cuda-gguf`**, per model:
   - action-tool call rate on closing turns — the behaviour known to be fragile (small models
     narrate a fake confirmation instead of calling the tool; see the 2026-09-12 session log);
   - terminal-outcome distribution;
   - turns per session;
   - output length and reasoning-leak rate.
4. **If the classes do not differ materially:** leave routing as it is and record the result in
   `experiments/EXPERIMENTS.md` — it supersedes the workplan's parity gate for generation.
5. **If they do:** switch on `session_build_consistency` and regenerate (a run costs $1–11).
   A mixed dataset cannot be repaired after the fact.

## 6. Local set-up notes

- The Mac's own Ollama holds port 11434; the router takes 11435, which `chat-agents/.env`
  already points at. Rented-host tunnels use local ports from 11441 up.
- The Vast CLI environment at `~/.venvs/vastai/` is no longer needed once the provider plug-in
  talks to the API directly; the API key file stays at `~/.config/vastai/vast_api_key`.
- Langfuse stays local and is unrelated to the pool. The long `LANGFUSE_FLUSH_INTERVAL` that
  the poisoned-session filter depends on is unchanged.
- Standing preference: validate on `--limit 1` end to end before any full run, and get explicit
  approval before anything that spends money or starts a long job.
