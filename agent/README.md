# gpm-agent

The optional host agent for a [GPM](../README.md) pool. Install it on a machine you add to a
pool and the pool learns **what the machine is** — accelerators, memory, free disk, engine
version, models on disk — instead of being told, and checked by nobody.

```sh
uv tool install gpm-agent   # a lasting install (`uvx` runs from a cache uv may clear)
gpm-agent init              # mints the agent key, prints it once, stores only its hash
gpm-agent serve             # loopback by default, port 8095
```

On a machine whose owner gives it the engine (S9): `gpm-agent init --install-engines
--start-at-login --prerequisites` — the agent then installs, runs and upgrades the engine,
starts with the machine, and installs ffmpeg and a compiler (on Linux with `make` and `cmake`)
where missing (`--admin` lets it use `sudo` where a platform needs it); `init` ends non-zero while
something is still needed. `gpm-agent startup status` shows whether the agent started with the
machine answers; `gpm-agent init --rotate` starts it again with the new key; `gpm-agent startup
remove` stops it, and leaves the engine it started serving until the machine restarts or you end
that process. On Windows the settings file, and any program or engines directory it names, must
be under your profile (`%USERPROFILE%`): a folder elsewhere is usually one every user of the
machine can change.

On a Linux machine with an NVIDIA card (S10), `--engine vllm` with `--install-engines` has the
agent install vLLM from PyPI — only the exact wheels, checked by hash, of the locked set it carries
for its Python (3.11–3.14) and processor — and run it once the models it holds are on disk.
vLLM 0.32.0 needs NVIDIA driver 580.65.06 or later and a card of compute capability 7.5 or higher;
`--prerequisites` checks both and installs Python's `venv`. `--engine auto` lets the agent choose
each time it starts: vLLM on an Ampere or newer card with that driver, Ollama anywhere else, with
the reasons in its facts and on the pool's Hosts screen. **Not yet run on real hardware**: built
and tested against fakes only, so the first install on a real machine is also its first check —
see the story's [unverified list](../docs/stories/S10-vllm-on-owned-machines.md#unverified).

On the pool's machine, put the key in an environment variable and name it on the host:

```yaml
hosts:
  - id: workstation
    kind: fixed-remote
    transport: { type: https, base_url: "https://workstation.example:11434", bearer_env: WORKSTATION_KEY }
    agent:     { url: "https://workstation.example:8095", bearer_env: WORKSTATION_AGENT_KEY }
```

If the pool reaches this machine through an SSH tunnel, leave the agent on loopback and give
the pool its port instead — `agent: { remote_port: 8095, bearer_env: … }` — and it is forwarded
over the same SSH connection, exposed to nothing.

**What it can be asked to do is a closed list** of four: restart the engine with *your* command
(below); report facts; hold this set of model
tags (it pulls what is missing, and pins or releases as the host's `residency` says); delete
this one tag, when an operator presses the button for it. There is no operation that takes a
command, a path or a URL from the pool, and there never will be. The pool dials the agent; the agent never dials the pool, never sees a request
or a response, and never holds a pool or provider credential.

This machine's owner has the last word, in `~/.config/gpm-agent/agent.json`: `min_free_disk_gb`
(pulls stop before free disk falls under it; default 10), `allow_delete` (whether the pool may
ever delete a model here) and `restart_command` (how the engine is
restarted — the pool may ask for it to run, never say what it is).

To let the pool restart the engine and set how it runs, add both to `agent.json` yourself:

```json
"restart_command": ["systemctl", "restart", "ollama"],
"engine_env_file": "/etc/ollama/gpm.env"
```

and point the engine's service at that file (systemd: `EnvironmentFile=/etc/ollama/gpm.env`). The
pool sends whole numbers — requests at once, models held — and the agent writes them under the
engine's own variable names. It never receives a variable name, a value as text, a path or a
command, and a restart happens only when an operator presses the button for it — or, for an engine
like vLLM that serves only what it was started with, when every model the pool asked for is on disk
and the engine serves none of them yet (D160). Without `engine_env_file`, that restart is your
command alone, with no numbers.

The full design: [docs/spec/host-agent.md](../docs/spec/host-agent.md).
