# gpm-agent

The optional host agent for a [GPM](../README.md) pool. Install it on a machine you add to a
pool and the pool learns **what the machine is** — accelerators, memory, free disk, engine
version, models on disk — instead of being told, and checked by nobody.

```sh
uvx gpm-agent init      # mints the agent key, prints it once, stores only its hash
uvx gpm-agent serve     # loopback by default, port 8095
```

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

and point the engine's service at that file (systemd: `EnvironmentFile=/etc/ollama/gpm.env`).
The pool sends whole numbers — requests at once, models held — and the agent writes them under
the engine's own variable names. It never receives a variable name, a value as text, a path or
a command, and a restart only happens when an operator presses the button for it.

The full design: [docs/spec/host-agent.md](../docs/spec/host-agent.md).
