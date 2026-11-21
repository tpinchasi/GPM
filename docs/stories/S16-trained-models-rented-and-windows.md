# S16 — Trained models: container images rented, private hub files rented, ONNX on Windows

> Status: **built against fakes** (2026-11-21) — D183–D185. Part of the [feature list](README.md).

## The story

S15 left three things for the owner to decide. The owner decided them on 2026-11-21:

1. **Container builds are rented, private images too** (D183). An operator's image can be rented for
   like any other build. The build says the command that starts its server on a rented machine, and
   a private image's registry login is saved once, by registry, and handed only to the provider's
   own create call.
2. **A private hub file reaches a rented machine with the read-only token** (D184). The token goes
   on the machine, copied over the SSH connection the pool installs the agent with, rather than the
   pool fetching every file and sending it.
3. **The agent installs ONNX Runtime on Windows**, for the processor and NVIDIA cards (D185), as it
   does on Linux.

## What was built

- **Stage 1 (D185).** `tools/onnx-lock` gains `windows-x86_64-cpu` and `windows-x86_64-cuda`, with
  Windows' wheel tags and Windows' least drivers (R525 for CUDA 12, R580 for CUDA 13). The
  ONNX Runtime 1.32.0 sets for Python 3.11–3.14 are committed. The agent's installer takes the
  Windows sets, finds an environment's interpreter in `Scripts\python.exe`, and unpacks into its
  work directory by `TEMP` and `TMP` too. Its own tests, one installing a set with real pip into a
  real environment, run on CI's Windows job.
- **Stage 2 (D184).** The supervisor hands the fleet its read-only hub token: never the write token,
  only for the hub it was saved for. A rented machine one of whose builds is a hub file its agent
  fetches is given it as `/var/run/gpm/hub.env`, owner-only, copied on the SSH connection's input,
  read into the agent's environment as `HF_TOKEN` when it starts. It is never on a command line, in
  the agent's settings or in the provider's instance spec. Where it went is an event
  (`hub_token_placed`); the token never is.
- **Stage 3 (D183).**
  - A container build may say `start`, an argument list. Rented hosts with `rented.engine: container`
    hold one model, start from its image on its `port`, run `start` after the dead-man timer, get no
    agent, and are ready when `GET /health` answers.
  - Registry logins are kept by registry in the credential store (`/pool/containers/registry-logins`,
    the console's Private registries).
  - The provider protocol gains `capabilities.private_images` and `create(..., registry_login=)`.
    Vast sends `image_login`; RunPod saves a registry credential once per login and names it on each
    pod. An offer from a provider without the capability is refused for an image that needs a login.
  - `login` joins the field names redacted from a provider's answers.

## Not run on real hardware

A Windows PC installing the CUDA set, a rented private image on Vast or RunPod, and a rented machine
fetching from a private hub repository have run only against fakes. What the providers do with a
login once they hold it is in the unverified assumptions (D183).
