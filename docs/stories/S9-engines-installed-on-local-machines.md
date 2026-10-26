# S9 — Engines installed and kept current on the machines the pool manages

> Status: **decided** (2026-10-26) — D150–D154; building. Part of the [feature list](README.md).
> Follows S8, which needs Ollama and whisper.cpp on local machines and, until this story, leaves
> installing them to the owner.

## The story

An operator installs GPM on a machine they own — a Mac, a Linux box, a Windows PC — and names
what it should serve. GPM installs the engines that needs (Ollama, whisper.cpp), keeps them at
the version the configuration says, and upgrades them when told — on whichever operating system
the machine runs. The engines are infrastructure, like the pool's own processes.

## Why

- **The owner's requirement** (2026-10-26): "on local, GPM installs Ollama and whisper as this is
  infra", and "GPM on local should adapt to the OS whether Win/Linux/Mac".
- **Today it is all by hand.** On the first adopter's laptop Ollama was installed by hand, held
  back by hand for a month when an update broke the model set, and upgraded by hand — three app
  swaps in one day. whisper.cpp would be one more program to install, upgrade and remember.
- **Rented hosts already work this way**: the engine arrives with the image, and the agent runs
  it (D63, D97). Machines the operator owns have no equivalent.

## The owner's answers (2026-10-26)

1. **Versions follow the latest release, automatically, with rollback.**
2. **whisper.cpp: built by GPM.** First answered "Homebrew, else built from source"; then, told
   that Homebrew's build links Homebrew's shared `ggml` — which moves with other formulas, so a
   version kept on disk cannot be gone back to — the owner chose GPM's own build.
3. **Windows, Linux and macOS, all in the first build.**
4. **Administrator rights where a platform needs them.**
5. **Ollama on a Mac runs headless**, started by the agent — not the app, whose own updater
   would compete with the pool's version. The app is quit and removed from login items once.

## Design

Decided in D150–D154 and specified in [host-agent.md](../spec/host-agent.md) §2.3 and §7,
[supervisor.md](../spec/supervisor.md) §3.1, [hosts-routing-capacity.md](../spec/hosts-routing-capacity.md)
§4 and the console's Hosts screen. In short:

- **The agent installs, runs and upgrades** the engine it manages, on a machine whose owner set
  `install_engines: true` (D150). It runs `ollama serve` itself, as it runs whisper.cpp's server,
  and never stops a program it did not start.
- **One new verb, *set engine version*** — `latest` or a version, and whether to switch now (D151).
- **Official sources only, checked** (D152): Ollama's release archives by platform (the headless
  macOS one signed by team `3MU9H2V9Y9`); whisper.cpp built from its tag's source on macOS and
  Linux, its release archives on Windows; ffmpeg from the platform's package manager at `init`.
  Digests always; signatures pinned where published; no install scripts.
- **Side by side, staged, drained, checked, rolled back** (D153): a new version is staged while
  the host serves, switched only when the host is idle and drained, kept only if the engine
  answers and holds the model set as before.
- **All three operating systems; elevation only at `init`** (D154).

## Build stages

1. **Decisions and specification** — done 2026-10-26: D150–D154, the agent's §2.3 and verb, the
   supervisor's upgrade pass, configuration, console, threat model T35.
2. **The agent**: the platform layer (paths, processes, starting with the machine) for all three;
   the installer (the release table, resolving `latest`, download with resume, digests, macOS
   signatures, safe unpacking, versions side by side, switching and going back); Ollama run by
   the agent; whisper.cpp built from source and from Windows archives; ffmpeg at `init`; the verb
   and facts — against a fake repository and fake programs.
3. **The pool**: `engine_version`, the idle wait, the drain, the switch, the check and the
   rollback; plan and events; the console's versions with *Upgrade now* and *Roll back*.
4. **On the first adopter's laptop**, asked first: the app quit, the headless Ollama installed and
   switched to, the laptop's whisper.cpp built and run by its own agent.

## Unverified

In [decisions.md](../decisions.md): the headless macOS archive serving the laptop's MLX builds;
Ollama's Windows signatures; whisper.cpp's static Metal build with PyPI's `cmake`; lingering on
Linux.

## What it touches

The agent's closed verb list (D151), its platform support and install footprint, the threat model
(T35), the configuration (`engine_version`, `install_engines`), the supervisor's pass, the plan,
and the console's Hosts screen. Nothing on the request path.
