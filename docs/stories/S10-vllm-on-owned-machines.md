# S10 — vLLM on the machines the operator owns

> Status: **decided** (2026-10-29) — D157–D161, specified; the build next. Part of the [feature list](README.md).
> Extends S9, which installs, runs and upgrades Ollama and whisper.cpp on the operator's own
> machines, to vLLM on a Linux machine with an NVIDIA card.

## The story

An operator adds a Linux machine with an NVIDIA card to the pool — a workstation under a desk, a
server in a cupboard — and names the models it should serve on vLLM. The machine's agent
installs vLLM, fetches the models, starts them and keeps vLLM current, as it already does for
Ollama and whisper.cpp. The machine sits beside a laptop's Ollama and the rented vLLM hosts, and
an application sees none of it.

## Why

- **The owner asked** (2026-10-29): a Linux or Windows PC on the pool, and vLLM on one of them.
- **vLLM is what an NVIDIA machine is worth running.** It batches many answers at once where
  Ollama serves few. It runs the same builds the rented hosts run (FP8 and NVFP4 among them), and
  it serves speech to text and ranking (D146).
- **Today it is possible, by hand, and one step is missing.**
  - A configured host may run vLLM (`engine: vllm` on the host, D93).
  - Its agent fetches the models from the model hub (D90).
  - `gpm-agent vllm-start` runs them, one at a time on each card (D97, D107, D155).
  - But nothing installs vLLM: the agent installs Ollama and whisper.cpp only.
  - And nothing starts it once the models have landed. The supervisor acts on a model that is
    on disk and waiting for a start only on the hosts it rents (`_restart_when_downloaded`). A
    configured vLLM host fetches its models, says they are waiting, and waits.
  - None of it has run end to end on a machine the operator owns.

## Scope

- **Linux on x86_64 or ARM, with an NVIDIA card of compute capability 7.5 or higher** — Turing
  (T4, RTX 20-series) onward, as vLLM's own documentation says and GPM's table of builds already
  holds (D100). vLLM publishes its builds for Linux only:
  0.32.0 on PyPI is a manylinux wheel for x86_64 and one for aarch64, about 325 MB each with a
  sha256 digest. It needs Python 3.11–3.14 and has 76 dependencies, among them an exact `torch`
  (2.13.0).
- **Not Windows.** vLLM has no Windows build; its documentation sends Windows users to WSL2,
  which GPM does not manage. A Windows PC serves on Ollama, and on whisper.cpp for speech (S9).
- **Not macOS.** vLLM has no practical Metal backend (D93); a Mac stays on Ollama.
- **Not AMD or Intel cards in the first build**: vLLM publishes its ROCm and Intel builds only on
  its own package index (`wheels.vllm.ai`), not on PyPI. Such a machine falls back to Ollama
  (answer 5), whose AMD build the agent already installs.

## Design

Decided in D157–D161 and specified in [host-agent.md](../spec/host-agent.md) §2.1 and §2.3,
[supervisor.md](../spec/supervisor.md) §3.2, [hosts-routing-capacity.md](../spec/hosts-routing-capacity.md)
§4 and the threat model (T35). In short:


1. **The pool starts a configured vLLM host once its models are on disk.** This is the missing
   step, for a vLLM the owner installed. When every model the host is told to hold is on disk
   and waiting for a start, the supervisor asks its agent to restart the engine — once per set
   on disk, as it does for a rented host — through the agent's restart command (the owner's
   script running `gpm-agent vllm-start`). It is small, and useful on its own.
2. **The agent installs vLLM** where the owner set `install_engines: true` (S9, D150), from a new
   source beside Ollama's and whisper.cpp's:
   - The versions are PyPI's `vllm` releases. `latest` is the newest with a wheel for the
     machine's processor (and one GPM has a dependency set for, answer 2).
   - Each version is a private Python environment, side by side with the others:
     `~/.local/share/gpm/engines/vllm/<port>/<version>/`.
   - The `vllm` wheel is checked against PyPI's digest; its dependencies come from the version's locked set
     (answer 2).
   - A version is kept only if its `vllm` runs. Where the host serves speech, the audio extra
     must import too (as D146 checks today).
   - The free-disk floor of S9 applies, at about 10–15 GB a version, since torch and CUDA's
     libraries come with it.
   - It is upgraded by S9's pass: staged while serving, switched when idle and drained,
     checked, rolled back (D153).
3. **The agent runs vLLM**, as it runs Ollama and whisper.cpp (D148, D150):
   - It starts its own launcher with that version's `vllm` once every model it holds is on disk,
     and starts it again when that set changes.
   - It stops it to upgrade, and never stops a vLLM it did not start.
   - The launcher gains a `--vllm` that names only a program inside the agent's own version
     directories, so still nothing the pool sends becomes part of a command (D41).
   - With the agent running the engine, the pool sends no restart: the agent knows when.
4. **Prerequisites at `init`**:
   - An NVIDIA driver new enough for the CUDA build of the `torch` vLLM pins, read from
     `nvidia-smi` and refused with the version needed.
   - Python 3.11–3.14 with `venv` (`python3-venv` on Debian and Ubuntu).
   - Whatever the audio extra needs from the system.
   - Each one is installed with `--admin`, or printed for the owner to run (D154).
5. **The machine's card**: vLLM holds its share for as long as the machine is in the pool (answer 3).
6. **`engine: auto`** (answer 5):
   - A host may name `auto` for its engine, and its agent chooses from the machine's facts.
   - It chooses vLLM only where vLLM can run every model the host holds: an NVIDIA card of
     compute capability 7.5 or higher; a driver new enough for the locked `torch`'s CUDA; each
     model's vLLM build within the card's generation (D100's floor for its precision); and the
     whole set within the card's memory (D104's plan).
   - Anywhere else — an AMD or Intel card, an older NVIDIA card, no card at all — it chooses
     Ollama.
   - The choice and its reason are in the agent's facts. The pool publishes the host's engine
     from them (D93), routes each model to that engine's build, and says why on the Hosts
     screen.
   - Each model an `auto` host holds needs a build for both engines, or the configuration is
     refused saying which.
   - The choice is made again whenever the agent starts, so a new card is seen. Changing
     engine on a running host is drained and switched like an upgrade (D153).
7. **The threat model** (T35 extended):
   - PyPI becomes a source, every file in a version's locked set checked by its hash (answer 2).
   - No `--trust-remote-code`.
   - The environment belongs to the machine's owner, and the agent runs as that owner.
   - A gated model needs the owner's `HF_TOKEN` in the agent's environment. It is read there
     only, as today.

## The owner's answers (2026-10-29)

1. **vLLM comes from PyPI**, into a private Python environment per version, kept side by side —
   not vLLM's Docker image, which would need Docker and NVIDIA's container toolkit as root.
2. **Its dependencies are a hash-locked set per vLLM version**, generated by GPM and shipped with
   the agent: every machine installs exactly the same, and a rollback is exact. `latest` means the
   newest vLLM the agent has a set for, so it follows vLLM's own releases a GPM release behind —
   not dependencies resolved from PyPI at install time.
3. **vLLM runs whenever the machine is in the pool**, as on a rented host; the operator takes the
   machine out of service to have the card back. Stopping it when idle is a later step.
4. **All of it in one build** — the pool starting a configured vLLM host once its models are on
   disk ships with the installer, not ahead of it.
5. **A machine that cannot run vLLM falls back to Ollama**, with an engine of `auto` — rather than
   being refused and left to the operator to switch by hand.

## Build stages

1. **Decisions and specification** — done 2026-10-29: D157–D161; host-agent §2.1, §2.3 and §3,
   supervisor §3.2, configuration (`engine: auto`), threat model T35.
2. **The agent installs and runs vLLM, and the pool starts a configured vLLM host** (design 1),
   in one build, against fakes:
   - the pool's start once the models are on disk, with a test driving the real supervisor and
     agent against a fake vLLM;
   - the PyPI source and versions side by side;
   - the dependency set, and the tool GPM's release runs to generate one per vLLM version (with
     a step in the release checklist);
   - the checks;
   - the launcher's `--vllm`;
   - starting and restarting;
   - the prerequisites;
   - the driver in the facts.
3. **The pool**: `engine_version` and the upgrade pass for vLLM hosts (mostly S9's), and the
   console.
4. **Live, asked first**: on a Linux machine with an NVIDIA card, owned or rented for the check
   as a bare Ubuntu machine treated as the operator's own (about $0.50). vLLM installed by the
   agent; one chat model and Whisper served, a picture answered, then an upgrade and a
   rollback.

## Unverified

- The CUDA version of PyPI's `torch` 2.13.0 build, and the NVIDIA driver it needs.
- The NVIDIA driver the CUDA 12.9 build needs on a machine with no CUDA toolkit: 525 or later by
  CUDA 12's minor-version compatibility, or 575 for 12.9 itself.
- What the audio extra needs from the system on a bare Linux.
- The disk one vLLM version takes.
- Whether a hash-locked set generated on one machine installs on every supported Linux
  (manylinux tags, glibc).

## What it gives up

- **Linux and NVIDIA only for vLLM**; every other machine falls back to Ollama.
- **Heavy installs and upgrades:** gigabytes a version, two kept; minutes to install.
- **The card is held while vLLM runs**, until the later idle-stop step (answer 3).
- **`latest` lags vLLM's own releases** until GPM releases a set for each (answer 2).

## What it touches

The agent's installer and engine runner (S9's framework, a third engine), the vLLM launcher (one
option), the supervisor's pass for configured hosts, prerequisites, the facts, the threat model
(T35), configuration and the console's Hosts screen. GPM's own release gains a step: a
dependency set for each vLLM version it supports (answer 2). No new verb: S9's *set engine version* and
the existing hold and restart verbs cover it. Nothing on the request path.
