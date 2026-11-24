# S17 — One Ollama server per card

> Status: **built against fakes** (2026-11-24) — D190–D192. Part of the [feature list](README.md).

## The story

The rented default is per card for every engine (D107): a machine with N cards is given N times the
workers. vLLM runs a copy of each model on each card behind the agent's router. Ollama keeps **one**
loaded instance of a model, on one card when it fits. So with mostly one model, the extra workers
queued on one card while the others sat idle. D107 left this open, and STATUS.md named the likely
fix.

The owner decided on 2026-11-24: **"One Ollama per card."** On a machine with more than one NVIDIA
card, the agent runs an Ollama server on each card, pinned with `CUDA_VISIBLE_DEVICES`, each on its
own loopback port, behind the agent's router, which shares the work between them as it does vLLM's
copies.

Four points were open. Each is now decided:

1. **Pulls and holds** (D190). Every server reads one models directory (`OLLAMA_MODELS`). A model is
   pulled once, through the first server, and held on each. Only the first server prunes the store
   as it starts, so restarting another server never removes a download's unfinished layers.
2. **A model too large for one card** (D191). A machine runs a server per card only when the models
   it holds fit the smallest card, by vLLM's own arithmetic. Otherwise one server spans every card,
   as Ollama always ran. Groups of cards are left for later.
3. **Idle unloading, keep-alive and `on_demand`** (D192). These act on each server separately.
4. **What counts as resident** (D192). A model is resident only when every server holds it, and the
   host answers only while every server does. The host's worker count is not cut when one server is
   down: each server gets its share of the host's workers.

**No verb was added** (D40).

**Unchanged:** a machine with one card, an Apple-silicon Mac, and a machine with no NVIDIA card. AMD
cards are unchanged too: their cards would be counted with `rocm-smi` and pinned with
`ROCR_VISIBLE_DEVICES`, which is a different path, and it is not built.

## What was built

- **The router's Ollama mode** (`gpm-agent proxy --engine ollama`). The upstreams are whole Ollama
  servers. A request goes to the server with the fewest requests in flight, whatever model it names.
  - It forwards a closed list of inference paths: Ollama's own, plus the OpenAI- and Anthropic-shaped
    paths Ollama serves.
  - It never forwards a pull, delete, create, copy, push or blob path.
  - It answers `/api/version`, `/api/tags`, `/api/ps`, `/` and `/health` for the whole machine.
  - It sends `/api/show` to the first server.
- **`ollama_cards`**:
  - It reads the cards from `nvidia-smi`, or from the environment's `CUDA_VISIBLE_DEVICES`.
  - It reads model sizes from Ollama's manifests in the store, and plans the layout.
  - `OllamaCards` starts, recognises and stops the servers and the router. They are recorded in
    `<models>.gpm-state/ollama-<port>/`, which includes `servers.json`.
  - If the servers per card cannot start, it falls back to one server and says why.
  - On a machine the agent runs Ollama on, it starts the servers again in the other layout when the
    models it holds change (`wants_restart`).
- **The agent.**
  - Where it runs Ollama itself, its installer drives `OllamaCards`.
  - Its Ollama model work pulls and deletes through the first server, pins on every server, and
    releases only on the servers that hold the model.
  - Its facts add `engine.layout`, `engine.copies` and `models_loaded_anywhere`.
  - The switch bound for Ollama grows to cover stopping in three steps.
- **Rented hosts.**
  - The pool's Ollama start runs `gpm-agent ollama-start --pinned` from the agent's archive when the
    archive is present and `nvidia-smi -L` lists more than one card. Otherwise it starts the one
    server, as before.
  - Once the agent answers, and before any pull, the pool asks it once to start the engine again,
    with the host's own workers, set and context (`engine_per_card`, or `engine_per_card_failed`).
  - A host whose facts already carry `engine.layout` is not asked.

## Tests

`tests/test_ollama_per_card.py` runs against a fake `ollama`, a fake `nvidia-smi`, the real router
and launcher as processes, and mock servers. It covers:

- the plan, and the cards seen;
- sizes read from the store;
- the router's closed list, its least-busy routing and the answers it gives for the whole machine;
- pull once, hold on each server, release only where held;
- the agent's runner: a server per card, a dead card restarting the layout, too large a set, one
  card, and a taken port;
- the launcher: starting, starting again, stopping, and refusing to start over a stranger's Ollama;
- the pool's start command, and its single ask.

The structural guard in `tests/test_agent.py` now covers the router's argument list.

## Not run on real hardware

This has not run on a real machine with several NVIDIA cards. Until the first one, these are in the
unverified assumptions:

- that Ollama honours `CUDA_VISIBLE_DEVICES` per process;
- that several servers share one store without a restart;
- that `OLLAMA_NOPRUNE` stops a server pruning the store;
- that the store's manifest layout is as read;
- that a rented image's shell has `nvidia-smi`.

On the first such host, look for a `card-N/logs/ollama.log` beside the models directory, each
card's memory in use, and `engine.copies` in the host's facts.
