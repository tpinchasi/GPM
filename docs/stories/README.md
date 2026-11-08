# Feature list

> The planned features, one story each: what it is for, the evidence behind it, a design, what it
> gives up, and the owner's answers. **All five are decided** and written into
> [spec/](../spec/); their pages remain as the reasoning and the build plan. **S1, S2, S4 and S5 are built, and
> S3's machine history with them; only S3's advisor remains.**
> Keep this table current: a story's status changes here first.

| # | Feature | Status | Depends on | Touches |
|---|---|---|---|---|
| [S1](S1-dynamic-resource-allocation.md) | **Dynamic resource allocation** — hosts added and given up from measured queue pressure and worker utilisation; explicitly enabled; machines still chosen by the configured rules; hosts added in a ramp — 1, 2, 4 … | **Built** (D66) | — (better with S5) | Supersedes "demand is the lease" for pools that enable it |
| [S2](S2-reliable-responses.md) | **Reliable responses** — responses from interruptible hosts are held until whole, so a lost host means a re-run, never half an answer; other host kinds keep streaming | **Built** (D62) | — | Amends the app contract's failover rule; new dialect items |
| [S3](S3-advisor-model-for-offers.md) | **An advisor model that picks the machine** — a small model chooses among offers the rules already accepted, by calling tools over the machine history; toggleable; falls back to the rules | **History built** (D69); advisor not started (D70) | — (richer with S4, S5) | Names a non-pure stage beside the strategy interface; a closed tool list, as the agent has closed verbs |
| [S4](S4-agent-on-rented-hosts.md) | **The host agent on rented hosts** — model sync and load, worker count, heartbeat, the machine's own facts | **Built** (D63, D72) | — | Supersedes the recommendation in host-agent §8; **delivers D56 and D57** |
| [S5](S5-automatic-worker-adjustment.md) | **Automatic worker adjustment per host** — each host finds its own number from load, profile and latency while it serves; starts at six and climbs on evidence, bounded by what the machine can hold | **Built** (D67, D68) | S4 (its slots-only core does not) | Supersedes "step down on evidence, never up" and D56's operator-only rule; replaces the one-off parallelism measurement |
| [S6](S6-workloads-from-the-sdk.md) | **Workloads created from the SDK, and client certificates** — a provisioning key with an operator's grant lets a program create, use and end its own workload; the key made in the program, only its hash sent; client certificates signed by the pool's CA, the private key never leaving the program's machine | **Built (D117)** — certificates in the first build | D115 | CLAUDE.md's app boundary; D16; app contract §3 |
| [S7](S7-several-models-in-one-workload.md) | **Several models in one workload** — one key, lease and budget for an app that uses two or more models; a latency target and answers-at-once per model; the pool puts every model on every host or each model on hosts of its own, whichever is expected to cost less | **Built (D118)** | D115, D116, D117 | Supersedes D115's one model per workload; a per-model cap in the dispatcher |
| [S8](S8-models-beyond-chat.md) | **Models beyond chat** — speech to text, pictures and audio inside chat sent only to builds that accept them, and rerank; each build says what it serves and what it takes in; uploads read and rewritten like JSON bodies; requests and latency measured, not tokens | **Decided (D142–D149)**, specification written (2026-10-26); build next | D141 (shipped first) | App contract §2 (version stays 1); engine interface v2; catalog `serves` and `inputs`; image capabilities; a whisper.cpp engine; rented hosts default to vLLM (amends D93) |
| [S9](S9-engines-installed-on-local-machines.md) | **Engines installed and kept current on the machines the pool manages** — the host's agent installs, runs and upgrades Ollama and whisper.cpp from their official sources, verified, following the latest release with automatic rollback, on macOS, Linux and Windows | **Decided** (2026-10-26) — D150–D154; building | S8 | The agent's closed verb list (D151); agent platform support (D154); threat model (T35) |
| [S10](S10-vllm-on-owned-machines.md) | **vLLM on the machines the operator owns** — a Linux machine with an NVIDIA card serves on vLLM: the pool starts it once its models are on disk, and its agent installs, runs and upgrades vLLM as S9 does Ollama and whisper.cpp; Windows and macOS stay on Ollama | **Built against fakes** (2026-10-30, D157–D161; gpm-server 0.37.0, gpm-agent 0.15.0): PyPI, a locked set per version (0.32.0 is CUDA 13: driver 580.65.06+), always running, `auto` falls back to Ollama; live check next | S9 | The agent's installer and runner, the vLLM launcher, the supervisor's pass for configured hosts, prerequisites, threat model T35 |
| [S11](S11-hosted-model-apis.md) | **Hosted model APIs** — OpenAI, Anthropic, Together, Ollama's cloud and any OpenAI-shaped API, reached with an API key; used when the pool's machines have no free worker or do not serve the model, paid per token from each app's own budget | **Built** (D164–D168; merged 2026-11-07, PR #89 — gpm-server 0.38.0, gpm-client 0.8.0, gpm-agent 0.16.0): each workload's spending policy — steady on machines, hosted and rented bursts each with its own budget, models served by hosted APIs alone; the Spending screen; `OLLAMA_NO_CLOUD`. Not yet run against a real provider; Claude uncached until S12 | S8 | The router (spill, metering), a new engine and host kind, the catalog's prices, credentials, budgets in the console, the agent (`OLLAMA_NO_CLOUD`), the app contract, threat model |
| [S12](S12-both-api-formats.md) | **Both API formats** — apps on OpenAI's Chat Completions and apps on Anthropic's Messages API each reach any model: untouched wherever the host speaks the request's format, translated only where none does; Claude through Anthropic's own API with caching on by default | **Built against fakes** (2026-11-07, D169–D170; gpm-server 0.39.0): untouched wherever a host speaks the format, translated where none does, refused by name unless the app asks to drop; load alone; Claude through its own API, cached for 5 minutes by default. Live check waits on provider keys | S11 | Supersedes "never translates between APIs" (D164, plugin-interfaces §2); the router's front door, two translators, metering of cache writes, the catalog's prices |
| [S13](S13-providers-on-console.md) | Providers on the console: hosted model APIs added like a rented account, and a first rented account from a pool that rents nothing | built against fakes (D171) |

## Already specified, waiting on a story

| Decision | What | Delivered by |
|---|---|---|
| D56 | Change a rented host's worker count while it runs | **S4 — done**; S5 will do it automatically |
| D57 | Load each model as its own download finishes | **S4 — done** |

## Order

**S1, S2, S4 and S5 are done.** What remains:

1. **S2** — independent, the smallest, and it fixes a failure apps see today.
2. **S4** — unblocks D56, D57 and S5, and supplies the facts S3 and S5 want.
3. **S1** — independent of the rest; its shrink ordering improves once S5 exists.
4. **S5** — its core can start before S4 finishes.
5. **S3** — last: its history table can begin any time, but the advisor is only worth judging
   once S4 and S5 are feeding that history.

## How the five fit together

```
S4 agent on rented hosts ──► facts from the machine ──┬──► S5 workers per host ──► measured throughput ──┐
                                                       │                                                   ├──► S1 which host to give up
                                                       └──► S3 machine history ◄───────────────────────────┘
S2 reliable responses — independent; changes what "time to first byte" means for S1 and S5
```

## What every story holds to

The project's standing rules apply unchanged: nothing rents without a lease and a dollar cap;
caps are re-checked after every strategy; router and supervisor stay separate; the agent's verbs
stay a closed list; every test runs with no GPU and no cloud account; every new behaviour is
explicitly enabled and explains itself in the decision log.
