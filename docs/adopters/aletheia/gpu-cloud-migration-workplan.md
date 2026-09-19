# GPU Cloud Migration — Workplan

> Draft, written 2026-09-15. Area: dev. Tracked as GPU-CLOUD-01. Moved here from
> `platform-design/unified/_drafts/` on 2026-09-17. Background for the pool's first adopter guide
> ([README.md](README.md)); the pool design itself is at [../../overview.md](../../overview.md). Supersedes nothing. No infrastructure has been provisioned.

Move simulator generation and the per-turn labeling pipeline off the local M4 Pro onto a
rented NVIDIA GPU, without losing Langfuse traces when the machine goes away.

---

## 1. The headline finding, before anything else

**Idle time, not the hourly rate, is the dominant cost risk.**

A full 834-session generation run costs between $1 and $11 of GPU time depending on model
choice. The same GPU left running idle for a month costs $245 to $780. Spot pricing saves
60–70% of a number that is already small; forgetting to stop the box wastes 20× more than
spot ever saves.

Priority order for cost control, highest impact first:

1. Aggressive auto-stop when idle
2. Right-sizing the model to the job
3. Throughput (continuous batching vs. Ollama's naive parallelism)
4. Spot/community pricing

Spot is still worth doing — it is fourth on the list, not first. Build the auto-stop before
the spot machinery.

---

## 2. What already works in our favour

Three properties of the existing code make this migration cheaper than it looks. All three
were verified by reading the current source, not assumed.

| Property | Where | Why it matters |
|---|---|---|
| Generation is checkpointed and resumable after **every** session | `prototypes/per-turn-behavioral-labeling/generate_sessions.py`, `run_batched_generation.py` | Already spot-ready. A reclaim costs at most one in-flight session. Only the checkpoint file's *location* needs to change. |
| Langfuse is **not** the system of record | `chat-agents/scripts/export_langfuse_conversations.py`, plus `new_export2/`, `export_20260914/` on disk | Traces are exported to JSON after each run. Losing the Langfuse DB is annoying, not fatal. This lowers the persistence bar substantially. |
| Inference endpoint is already a single env var | `OLLAMA_URL` in `chat-agents/agents/model_config.py` | Phase 2 is a config change, not a code change. |

The one thing that follows from property two: **retention windows become a real deadline.**
Langfuse Cloud Hobby keeps 30 days, Core keeps 90. The export must run inside that window or
the traces are gone for good. Make export a non-optional post-run step.

---

## 3. Decisions to make before spending anything

Four decisions. Each has a recommendation and the rule that would change it.

### D1 — Which model is production?

This is the decision that drives every cost number below, so settle it first.

The 2026-09-12 session log records that smaller models (`gemma4:e4b-mlx`, `qwen2.5:7b-instruct`)
reliably call lookup tools but **narrate fake confirmations on the closing turn instead of
invoking the action tool**, while `qwen3.8:27b-mlx` closed the loop reliably across all five
agents. Since the 41 action tools landed, tool-calling fidelity is a correctness requirement,
not a nice-to-have.

| If production is | VRAM needed | Card | Verified rate |
|---|---|---|---|
| `qwen2.5:7b` + `gemma4:e4b` | 24 GB | RTX 4090, RunPod Community | $0.34/hr |
| `qwen3.8:27b` class | 48 GB | L40S | $1.09/hr |

**Recommendation:** decide empirically in Phase 3 rather than by argument. Run the parity gate
on both. If the small model's tool-call miss rate is acceptable for generation, take the 4090
and pocket a 3× saving. Note that "narrates instead of calling the tool" is itself a legitimate
behavioural signal per the 2026-09-12 log, so a miss rate is not automatically a defect.

**Worth testing:** a split where the cheap model drives ordinary turns and the 27B handles only
closing turns. That is a design change, out of scope here, but it would collapse the cost gap.

### D2 — Langfuse: managed or self-hosted?

| Option | Cost/month | Event ceiling | Ops |
|---|---|---|---|
| Cloud Hobby | $0 | 50k units, then **tracing silently stops** | None |
| Cloud Core | $29 + $8/100k overage | 100k included | None |
| Self-hosted, small always-on VM | $30–60 | Unlimited | Backups are ours |

One full cycle is roughly 50k units: ~21k for generation (834 sessions × 1 trace + ~24
observations) plus the labeling pipeline at three calls per turn.

**Recommendation: Cloud Core.** Hobby's failure mode is dropping data without billing us, which
is the worst possible behaviour for this workload. Self-hosting only wins past roughly four full
cycles a month, because unit pricing punishes bulk synthetic generation. Revisit at that point.

### D3 — Provider

**Recommendation: RunPod**, for the network volume plus per-second billing plus the cheapest
verified 4090 rate. AWS `g5.xlarge` spot at $0.44/hr in us-east-1 is the tidy alternative if we
want everything in one account with IAM and budget alarms. Pin the region before creating the
volume — RunPod network volumes are region-locked.

### D4 — Ollama or vLLM?

**Recommendation: Ollama first, vLLM later, and treat later as genuinely optional.** Ollama is a
zero-code-change lift. vLLM is worth 5–15× throughput but requires per-model tool-call parser
configuration, and we now depend on 41 tools across five agents. Do not couple the migration to
that risk. See Phase 5.

---

## 4. Phases

Each phase has an exit criterion. Do not start the next phase until it is met.

### Phase 0 — Accounts and guardrails
*No GPU spend.*

- [ ] Settle D1–D4 above.
- [ ] Create the provider account; set a hard monthly budget alert at $100.
- [ ] Create the Langfuse Cloud org and project; record the new keys.
- [ ] Map every MLX model tag to its GGUF equivalent. `gemma4:e4b-mlx` and `gemma4:31b-mlx` do
      not exist on NVIDIA. Record the exact replacement tags.
- [ ] Confirm the embedding model tag is `nomic-embed-text-v2-moe` and will not change.

**Exit:** a written list of exact model tags to pull, and a budget alert that fires.

### Phase 1 — Persistent Langfuse
*Do this before renting any GPU. The persistent thing exists first.*

- [ ] Point the **local** machine at Langfuse Cloud by setting `LANGFUSE_BASE_URL`,
      `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` in `chat-agents/.env`.
- [ ] Run one single session locally against it.
- [ ] Verify the v3 SDK still works. The labeling pipeline pins `langfuse>=3,<4` while
      `chat-agents/.venv` carries v4. Both must talk to Cloud. **Test both, separately.**
- [ ] Run `export_langfuse_conversations.py` against Cloud and confirm the export shape matches
      the existing `new_export2/` files.

**Exit:** a trace created on the laptop appears in Cloud, and the export script round-trips it.
Historical local traces need no migration — `new_export2/` and `export_20260914/` already hold
them on disk.

**Rollback:** revert the four env vars. Local Langfuse is untouched throughout.

### Phase 2 — GPU node, on-demand, Ollama lift-and-shift
*On-demand deliberately, not spot. Debug without interruptions in the mix.*

- [ ] Provision the card chosen in D1. On-demand for this phase only.
- [ ] Create a **100 GB network volume**. Sizing: ~50 GB of weights if we pull everything, plus
      checkpoints, outputs and headroom. Roughly $7/month.
- [ ] Mount it at the Ollama model directory so weights survive the node.
- [ ] Pull the GGUF tags from Phase 0. Confirm each loads and answers.
- [ ] Deploy `chat-agents` on the node; point `OLLAMA_URL` at local Ollama there.
- [ ] Point `LANGFUSE_*` at Cloud from the node.
- [ ] **Approval gate:** get explicit sign-off before the first multi-session run.

**Exit:** one session runs end to end on the GPU node, calls tools, reaches a terminal outcome,
and its trace lands in Cloud.

**Rollback:** destroy the node. The volume and Langfuse both survive independently.

### Phase 3 — Parity gate
*The phase most likely to be skipped and most expensive to skip.*

Moving from Apple MLX builds to CUDA GGUF builds changes model outputs. The labeling pipeline's
calibration centroids and accuracy baselines were established against the old outputs. Prove
equivalence before trusting anything downstream.

- [ ] Run a small matched set on the GPU node. Start with `--limit 1`, per standing preference,
      then a handful.
- [ ] Compare against the local baseline on: terminal-outcome distribution, action-tool call rate,
      turn counts, and the step 1–3 labeling outputs.
- [ ] Specifically re-measure the closing-turn tool-call miss rate per model. This is the D1
      decision input.
- [ ] Confirm JSON schema enforcement now works on step 3. MLX ignored the `format` parameter,
      which forced lenient parsing in `step3_match_and_assign.py`. On CUDA the schema should bind
      again. **This is an upgrade, and it may change step 3's output shape — check, don't assume.**
- [ ] Do **not** change the embedding model. Calibration centroids were built with
      `nomic-embed-text-v2-moe`; swapping it silently invalidates them and nothing will warn us.

**Exit:** a written parity verdict. Either outputs are equivalent, or the differences are
understood and accepted, or we go back to Phase 2 with a different model tag.

### Phase 4 — Auto-stop, then spot
*Auto-stop first. It is worth more than the spot discount.*

- [ ] Add an idle watchdog: if no request has been served for N minutes, stop the node. This is
      the single highest-value cost control in the whole plan.
- [ ] Move `generation_log.json` and all pipeline outputs onto the network volume.
- [ ] Convert the node to spot/community.
- [ ] Add a boot script that mounts the volume, starts Ollama, and resumes the generation loop.
- [ ] **Interruption drill:** kill the node deliberately mid-run. Confirm the run resumes with no
      lost sessions and no duplicates.

**Exit:** a forced kill costs at most one in-flight session, and an idle node stops by itself.

**Known, accepted loss:** `app.py` calls `_langfuse.flush()` in a `finally` block, but a hard
spot kill skips it. Traces buffered at that instant are lost. The checkpoint log, not Langfuse,
is the resume source of truth, so this costs observability on a few turns and never costs work.
Shorten the flush interval to reduce the window.

### Phase 5 — vLLM throughput
*Optional. Highest payoff, highest risk. Only after Phases 1–4 are stable.*

- [ ] Stand up vLLM with its OpenAI-compatible endpoint alongside Ollama, not replacing it.
- [ ] Configure `--enable-auto-tool-choice` and the correct per-model tool-call parser. **This is
      the real risk.** All 41 action tools must still fire correctly.
- [ ] Adapt the client. Options, cheapest first: a small adapter in `agents/agent_runtime.py`, a
      LiteLLM proxy translating Ollama calls, or switching the client outright.
- [ ] Use vLLM guided decoding for steps 1–3 structured output.
- [ ] Re-run the Phase 3 parity gate against vLLM. Same bar, no exceptions.
- [ ] Benchmark honestly: tokens/hour and cost/run, Ollama vs. vLLM, same card.

**Exit:** measured throughput gain with tool-calling parity held. If parity fails, stop and stay
on Ollama. The throughput is not worth broken tool calls.

### Phase 6 — Steady state

- [ ] Write a short runbook: start, run, export, stop.
- [ ] Make Langfuse export a mandatory post-run step, inside the 90-day retention window.
- [ ] Record actual cost per run for the first month and compare against section 5.

---

## 5. Cost model

Fixed monthly:

| Item | Cost |
|---|---|
| Langfuse Cloud Core | $29 |
| 100 GB network volume | ~$7 |
| **Fixed total** | **~$36** |

Per full 834-session generation run. Assumes roughly 6M output tokens. **Estimates, to be
replaced with Phase 6 measurements.**

| Setup | Est. run time | Est. cost |
|---|---|---|
| M4 Pro today | 50+ hours | $0 |
| 4090 + Ollama, 7B | ~4 hours | $1.36 |
| 4090 + vLLM, 7B | ~40 min | $0.23 |
| L40S + Ollama, 27B | ~10 hours | $10.90 |

Idle risk, for contrast: a 4090 left running for a month is roughly $245; an L40S is roughly $780.

---

## 6. Risk register

| Risk | Severity | Mitigation | Phase |
|---|---|---|---|
| CUDA outputs differ from MLX, silently invalidating baselines | High | Parity gate, explicit verdict | 3 |
| Embedding model swapped, centroids invalidated with no warning | High | Pin `nomic-embed-text-v2-moe`, never change | 0, 3 |
| vLLM tool-call parser breaks some of the 41 action tools | High | Keep Ollama; parity gate before cutover | 5 |
| Idle node burns the budget | High | Idle watchdog before spot conversion | 4 |
| Langfuse Hobby silently stops tracing at 50k units | Medium | Take Core, not Hobby | 0 |
| Traces expire before export | Medium | Export mandatory post-run, 90-day window | 6 |
| Weights re-downloaded on every reclaim | Medium | Network volume mounted at model dir | 2 |
| Checkpoint log on ephemeral disk | Medium | Move to network volume | 4 |
| RunPod volume region-locked to the wrong region | Low | Pin region in Phase 0 | 0 |
| v3/v4 Langfuse SDK split breaks against Cloud | Low | Test both SDKs separately | 1 |

---

## 7. Open questions

- Is the 27B model needed for all turns, or only closing turns? A split model strategy would
  collapse the cost gap but is a design change, not infrastructure.
- Does the closing-turn tool-call miss rate on small models count as a defect to engineer away,
  or as behavioural signal to capture? The 2026-09-12 log leans toward the latter. This changes
  what "parity" means in Phase 3.
- At what run cadence does self-hosted Langfuse beat Cloud Core? Estimated around four full
  cycles a month; worth recomputing once real unit counts land.

---

## 8. Sources

GPU and Langfuse pricing verified 2026-09-15. Prices move; re-check before committing.

- RunPod RTX 4090 Community $0.34/hr, Secure $0.69/hr, L40S $1.09/hr —
  <https://www.synpixcloud.com/blog/rtx-4090-cloud-rental-worth-it>,
  <https://flexprice.io/blog/runprod-pricing-guide-with-gpu-costs>
- AWS `g5.xlarge` spot from $0.4419/hr us-east-1, $0.1408/hr ap-southeast-3 —
  <https://compute.doit.com/spot/us-east-1/g5.xlarge>
- AWS `g6e.xlarge` L40S ~$1.86/hr on-demand — <https://cloudprice.net/aws/ec2/instances/g6e.xlarge>
- Langfuse Hobby 50k units/30 days, Core $29 for 100k units/90 days, $8/100k overage —
  <https://costbench.com/software/ai-observability/langfuse/free-plan/>,
  <https://markaicode.com/pricing/langfuse-pricing/>
