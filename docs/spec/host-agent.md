# Specification — The Host Agent

> **Status: decided (D40). Built in the stages of §7; STATUS.md says which exist.**

A small process, `gpm-agent`, that runs **on a host** the operator adds to a pool. It tells the
pool what the machine is, makes the machine hold the pool's model set, and applies the host's
settings. It exists because the engine's API cannot say what hardware it runs on, how much
memory or disk is free, or change how the engine itself is run — and those are exactly the
things an operator otherwise does by hand, per host, and gets wrong.

## 1. What changes, and what does not

Until now: *the pool verifies an engine someone else runs; it never configures it.* With an
agent on the host, the operator has **delegated that host to the pool**, and the rule becomes:

| Host | Who configures the engine |
|---|---|
| Rented — the pool created it | The pool, as before |
| Configured, **with an agent** | The pool, through the agent, within §4's limits |
| Configured, **without an agent** | The operator. The pool verifies and reports, as before |

An agent is **optional per host**. A pool with no agents behaves exactly as it does today.

Unchanged, and not negotiable: the agent is **never on the request path** (the router dials the
engine directly); **no request ever causes a download**; the agent never holds the provider
credential, the admin key or an app key; apps cannot see or reach it.

## 2. Shape

```
supervisor ──(the host's own transport: http on loopback · https + bearer · ssh tunnel)──► gpm-agent ──► engine
   desired state  ─────────────────────────────────────────────►
   ◄───────────────────────────────────────────  facts + actual state
```

- **The pool dials the agent; the agent never dials the pool.** It is reached exactly as the
  engine is — same transport types, same rules (no unauthenticated listener off loopback, TLS
  off loopback, or the supervised SSH forward). Any host whose engine the pool can reach has a
  reachable agent by the same path, so nothing new listens on the pool's side and no host needs
  a route *to* the pool.
- **Only the supervisor talks to it**, on its control pass. The router never does.
- **Level-triggered.** Each pass the supervisor sends the host's whole *desired state* and reads
  back *facts* and *actual state*. There is no event the agent can miss: an agent that was down,
  restarted or reinstalled converges on the next pass. "The console broadcasts configuration to
  the agents" is this: the console changes configuration through validate → plan → apply, and
  the supervisor's next pass carries it to every agent.
- **The agent is all but stateless about the pool.** It persists its own settings and one
  thing more: which models *it* pinned, in an owner-only file beside them — or it would forget
  across its own restart and never release them. A remembered pin the engine has since dropped
  is forgotten, not re-asserted. Desired state lives in the pool's configuration, the single
  source of truth, and is said again every pass.

## 3. What it reports — facts

Read-only, cheap, no privileges: operating system and architecture; accelerators (kind, name,
memory each); total and available system memory; free disk where the engine keeps models;
engine name, version and whether it answers; models on disk with sizes; models loaded.

From facts the pool **derives the host's capabilities** (`apple-silicon`, `cuda`, …) instead of
trusting what someone typed, and the console can say *before* a change is applied that a model
set does not fit this host's memory or disk, by how much.

A capability typed in configuration that the facts contradict is an error shown in plan, not a
silent override in either direction.

## 4. What it may do — and what it may not

| May, when the pool asks | Why it is bounded |
|---|---|
| **Pull a tag** | Only a tag the supervisor names, which is only ever a variant in the pool's catalog for this host's capabilities. Never from a request. Refused if free disk would fall below a floor. One pull at a time |
| **Load, pin, or unload a model** | To make `residency` true: `pinned` loads and pins the whole set; `on_demand` leaves loading to the engine |
| **Report surplus models** | Models on disk that no pool configuration names are *listed*, with sizes |
| **Delete a model** | Only as an **explicit operator action in the console** — never by the supervisor's own reconciliation, so no configuration change, bug or pass can delete anything. The tag is typed again to confirm; a tag this pool requires on that host is refused; and the agent refuses all deletion when its own configuration says `allow_delete: false`, which the machine's owner sets and the pool cannot change |
| **Restart the engine, apply engine settings** (parallelism, keep-alive, models held at once) | Only through a command the **operator wrote into the agent's own configuration on that host**. The pool can ask for it to be run; it cannot say what it is |

| May not, ever | |
|---|---|
| Run a command the pool supplies | The agent has no "run this" operation. Its verbs are the closed list above |
| Read or send request or response content | It is not on the request path and has no access to it |
| Update itself | Upgrades are the operator's package manager, like the pool's own |

The closed verb list is the security property: compromise of the pool's admin key lets an
attacker make a host download *catalogued* models, restart its engine and — where the machine's
owner has not switched it off — delete models that can be pulled again. It never lets them
execute code, read traffic or reach anything that is not a model.

## 5. Keys and enrolment

A third key role, **agent key** (`gpmg_…`), never interchangeable with the other two: the agent
refuses app and admin keys by name, and the control API and router refuse an agent key.

```sh
# on the host
uvx gpm-agent init          # mints the key, prints it once, stores only its hash (0600)
uvx gpm-agent serve         # loopback by default
```

The operator adds the host in the console with its agent address; the key goes where every
other host credential goes — an environment variable on the pool's machine, named in
configuration, shown in the console only as set ✓ / missing ✗. **Test connection** gains an
agent step: reach → authenticate → facts → what would be pulled, and how many gigabytes.

**On a host reached by SSH tunnel** the agent stays on loopback over there, exposed to nothing,
and configuration gives its port rather than a URL:

```yaml
transport: { type: tunnel, ssh_host: workstation.local, remote_port: 11434 }
agent:     { remote_port: 8095, bearer_env: WORKSTATION_AGENT_KEY }
```

The pool opens a **second supervised forward** for it, with the host's own SSH settings and
pinned host key. The two forwards are independent: adding, moving or removing an agent never
touches the one the engine's traffic is on.

## 6. In the console

**Hosts** gains, per host with an agent: derived capabilities, memory and disk, engine version,
agent version and last contact; and a progress line while a pull runs. **Models** shows fit
(set size against this host's memory and disk) and surplus. Adding or changing a model set shows
in **plan** what each agent would download, in gigabytes, before apply.

A host whose agent stops answering keeps serving: the engine is verified by the probe as it is
today. The console says the agent is unreachable; nothing is inferred from its silence.

## 7. The protocol

Four verbs, all under `/agent/v1`, all behind the agent key. The list is closed, and a test
fails until it is edited on purpose to admit another.

| Verb | Body | Does |
|---|---|---|
| `GET /facts` | — | §3 |
| `PUT /models` | `{tags, residency}` — the **whole** desired state | Starts working toward it in the background, one pull at a time, and answers at once with where each tag stands (on disk, loaded, pulling with bytes, last error), free disk, the owner's floor, and the surplus. A tag that failed is left alone for a minute, not retried every pass |
| `DELETE /models` | `{tag}` | §4's deletion, inside its bounds |
| `POST /heartbeat` | — | Touches the dead-man timer's file on a host the pool created (D63). No body, and no other bound: it can only *postpone* a shutdown the pool could equally cause by going silent. A machine nobody rented carries no timer, and the verb says so |
| `POST /engine` | `{settings}` — `null`, or `{workers, models_held, context?}` as bounded whole numbers | Writes the engine's start-up environment to the file the owner named, if it changed; runs the owner's restart command; waits for the engine to answer; reports the exit code, the tail of the output, and whether the engine came back |

A tag travels in a body, never a path, and must be a plain model tag — a name, optionally
namespaced and versioned. Anything with a scheme, a space, `..` or a leading dash is refused
before the engine sees it. `manage_models: false` on a host keeps its agent to `GET /facts`.

**Engine settings are numbers, the command is the owner's (D41).** The pool says how many
requests at once and how many models held; the agent's engine adapter turns those into the
engine's own variable names, so no name or text from the pool is ever written to a file or put
in a process's environment. Where they are written (`engine_env_file`) and what restarts the
engine (`restart_command`, an argument list, never a shell line) live in the agent's settings on
that machine; without them the verb is refused and the console says what the owner would add.
A restart drops whatever the engine is doing, so it is **only ever an operator's act** — the
host id typed again — and never the supervisor's pass. The agent reports the settings back as
the same numbers, so the console can show *what the pool needs* beside *what is set* without
knowing any engine's vocabulary.

A pull is stopped by closing its connection, which is how the engine cancels one; nothing is
left installed. The agent only ever releases a pin **it** set: a model the machine's owner keeps
loaded for their own reasons is not the pool's to unload.

## 8. Build order

1. Agent with facts only; supervisor reads them; console shows them; capabilities derived.
2. Pull and load/pin/unload to satisfy the model set and `residency`; plan shows the download.
3. Operator-defined restart and engine settings.
4. The agent behind an SSH tunnel.
5. The agent on hosts the pool rents (§9). **Built** — packed, pushed, started, asked for facts,
   the preparation path (delivering D57), its own heartbeat verb, and the worker count changed
   while the host runs (delivering D56).

Still not built, and still waiting on a package index: installing the agent over SSH onto a
*configured* host that someone else owns.

Stages 1 to 4 are built. Every stage is tested against a real agent app and the fake engine: no GPU, no cloud account.

## 9. On hosts the pool rents (D63)

The offer does not state the facts that matter: machines with identical offers have differed
fourfold in throughput, for reasons visible only from the machine. So a rented host runs the same
agent, with the same closed verbs, and **the pool still dials**.

- **Put there by the pool, as a zipapp (D72).** Once the host's SSH answers, the pool copies a
  single `.pyz` — the agent and its pure-Python dependencies — and a key made for that host
  alone, and starts it with the host's own `python3` on the host's loopback. A host with no
  interpreter gets no agent, and joins without one. Nothing secret goes into the
  start-up script, which a provider stores and can read. The pool reaches the agent through a
  second forward on the SSH connection it already holds (stage 4).
- **Packed once a run, never once a state directory.** The archive is built from the agent
  installed beside the supervisor, and an archive an *earlier* run left on disk is not reused:
  a pool running today's agent would otherwise ship a file packed weeks ago. Seen live — every
  rented host was given an agent whose `init` predated the options the pool had begun sending
  it, and each one joined agentless with `unrecognized arguments`, silently, because the file
  was present and presence was taken for currency. Packing costs about a second per run.
- **It is the preparation path.** `PUT /models` works toward the model set and loads each model
  **as its own download finishes** (D57) rather than pulling everything and then loading
  everything — which left the accelerator idle through the whole last phase of a billing host,
  measured live at 78 seconds. `POST /engine` changes the worker count (D56). Neither needed a
  new verb.
- **One new verb, and more facts.** `POST /heartbeat`, with no body, touches the dead-man timer's
  heartbeat file — it runs **beside** the pool's SSH heartbeat, not instead of it, until it has
  been seen working on a live host — it can only postpone a shutdown the pool could equally cause by going silent.
  `GET /facts` adds load average, accelerator utilisation and memory, and measured generation
  rate. The verb list stays closed; the test that guards it is edited for exactly this.
- **The restart command is the pool's here.** On a host the pool created, what restarts the
  engine comes from the pool's own start-up material — what the pool creates is the pool's to
  configure — fixed at installation and never sent over the protocol. On a delegated host it
  remains the owner's (D41), unchanged.

Three rules keep the agent from becoming a new way to lose money:

1. **The dead-man timer stays an independent script**, armed first, with the instance-scoped
   credential. If the agent dies, the timer still fires.
2. **No agent is never fatal.** A host whose agent has not answered within `agent_wait_s` is
   prepared over the engine's API as before, the reason is recorded, and it joins.
3. **Installing it overlaps the first model download**, so it adds nothing to the billed path.

