# S9 — Engines installed and kept current on the machines the pool manages

> Status: **proposed** (2026-10-26) — the owner's requirement, a first design and open questions;
> nothing decided. Part of the [feature list](README.md). Follows S8, which needs Ollama and
> whisper.cpp on local machines and, until this story, leaves installing them to the owner.

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

## A first design

1. **The host's agent installs, never the pool directly.** The agent already runs on the machine,
   as the owner's user, and already fetches models and restarts engines. A new verb, *install
   engine*, takes an engine name from the agent's own closed list and a version — a validated
   string, or `latest` — and never a URL, a path or a command: the agent maps them to that
   engine's **official release for its operating system and architecture** from a table built
   into the agent. Adding the verb is a decision (D40).
2. **Nothing runs unverified.** The release's published digest is checked; then the platform's own
   signature — on macOS notarization and the developer team pinned in the agent, on Windows the
   Authenticode publisher pinned likewise, on Linux the checksum and a signature where one is
   published. Never a piped install script. The version before is kept for a rollback.
3. **User-level installs, no administrator rights**, where each platform allows: a user-owned
   applications folder, a prefix under the home directory, a per-user service.
4. **The version is configuration**: per host, a pinned `engine_version`, or `latest` to follow
   releases. An upgrade waits until the host is idle and drained, and rolls back by itself if the
   engine then fails its health check or no longer holds its model set as before — the check
   that would have caught the Ollama update that split the first adopter's two models. The plan
   says what will be installed, and how large, before Apply.
5. **The console** shows each host's installed and wanted version, with Upgrade and Roll back.
6. **The agent itself** needs a way onto the machine and a way to stay running across reboots —
   a per-user launch agent, systemd user unit or Windows service — which `gpm-agent init` could
   set up. Today the agent assumes a POSIX machine (its engine restart is a shell script), so
   Windows is new for it as well as for the engines.

## What each platform needs — to be verified before deciding

| | macOS | Linux | Windows |
|---|---|---|---|
| **Ollama** | The notarized app archive from its GitHub release (checked by hand 2026-10-26: digest published, team `3MU9H2V9Y9`) into the user's Applications folder | The release tarball for the architecture and GPU (CUDA, ROCm) into a user prefix, run as a user service | The release's installer or archive, per-user — *unverified* |
| **whisper.cpp** | Homebrew's build, if Homebrew is there; otherwise a build from source (Xcode's command-line tools) — *no official macOS binary found yet* | A build from source with the GPU's backend (CUDA, Vulkan), or release binaries if published — *unverified* | Release archives with CPU and CUDA builds — *unverified* |
| **ffmpeg** (for whisper.cpp's `--convert`) | Homebrew | The distribution's package, or a static build | A static build — *unverified* |
| **Keeping the agent running** | A launch agent | A systemd user unit | A service or scheduled task — *new* |

## Questions for the owner

1. **Version policy by default**: pinned, upgraded from the console — or `latest`, followed
   automatically with the rollback check?
2. **whisper.cpp where no official binary exists** (macOS at least): rely on Homebrew where
   present, build from source on the machine, or publish GPM's own signed builds?
3. **Windows**: in the first build, or macOS and Linux first with Windows after?
4. **Administrator rights**: never (recommended — user-level only), or allowed where a platform
   needs them?

## What it touches

The agent's closed verb list (a new verb, D40), the agent's install footprint and its platform
support, the threat model (the agent downloads and runs programs: a supply-chain entry), the
configuration (`engine_version`), the plan, and the console's Hosts screen. It changes nothing on
the request path.
