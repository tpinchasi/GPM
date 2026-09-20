# S4 — The host agent on rented hosts

> Status: **decided (D63), not built.** Part of the [feature list](README.md). The owner accepted
> this story as written, with its recommended answers to the open questions; the specification
> now carries it, and this page remains as the reasoning and the build plan.

## The story

*As an operator, every host the pool rents runs the same small agent my own hosts can run. It
gets the models onto the machine and into memory, changes how many requests the engine runs at
once, tells the pool what the machine is really doing, and keeps the dead-man timer fed.*

## Why

[host-agent.md](../spec/host-agent.md) §8 recommended **against** this: the pool already
configures a rented host over a proven path, and an agent "would add billed boot time and new
ways to fail to a spending path, to learn facts the provider's offer already stated."

Live use has answered that last clause. **The offer does not state the facts that matter.**
Machines with identical offers differed fourfold in throughput; the cause — a load average above
250 on a shared machine, an accelerator idle at 5 % — is visible only from the machine. And three
features now need something on the host that the engine's API cannot provide:

| Need | Today | With the agent |
|---|---|---|
| Load each model as its download finishes (D57) | Pull all, then load all, driven remotely | The agent's `PUT /models` already works toward a desired state one pull at a time; loading as each lands belongs there |
| Change the worker count on a running host (D56) | Impossible without replacing the host | `POST /engine` with bounded numbers already exists |
| Heartbeat | An SSH command every pass, per host | One request on a connection already open |
| What the machine is doing | Unknown | Load average, accelerator utilisation and memory, in `GET /facts` |

So the agent is mostly **built**. This story is about getting it onto a rented host and answering
the two objections: boot time, and new ways to fail.

## Design

**The same agent, the same closed verbs, the pool still dials.** Nothing about the protocol's
security property changes: no verb takes a command, a path or a URL; the agent is never on the
request path; the provider's account credential never goes near the host.

**Reached through the tunnel the host already has.** The agent listens on the rented host's
loopback, and the pool forwards a second local port to it over the same SSH connection — the
arrangement already built for configured hosts behind a tunnel. Nothing new listens anywhere.

**Getting it there.** Two ways, and the choice is the main open question:

| | Pushed by the pool over SSH | Downloaded by the start-up script |
|---|---|---|
| How | Once SSH answers, the pool copies the agent and its key file, then starts it | The start-up script fetches a pinned version, checks its hash, starts it |
| Needs | The SSH path the tunnel already requires | A published package — the agent is not on an index until release |
| Version | Always the pool's own | Whatever was pinned in configuration |
| The agent key | Travels over SSH; **never in the provider's metadata** | Must be in the start-up script, which the provider stores and can read |
| Boot time | After SSH is up, in parallel with the first model download | Before the engine starts |

Recommended: **push**. It works before anything is published, cannot drift from the pool's
version, and keeps a secret out of the provider's hands. It gives up self-sufficiency: a host
whose SSH never comes up never gets an agent — but such a host can never join today either.

**One key per host**, generated at creation, stored hashed, useless for any other host, dead with
the instance.

**It must not become a new way to lose money.** Three rules:

1. **The dead-man timer stays independent.** It remains the script armed first at creation, with
   the instance-scoped credential. The agent only *touches its heartbeat file* when the pool
   says so. If the agent dies, the timer still fires.
2. **No agent is not fatal.** If the agent has not answered within `agent_wait_s`, the pool
   prepares the host the way it does today, records why, and carries on. The agent adds
   capability; it never gates joining.
3. **The download runs meanwhile.** Installing the agent overlaps the first model pull, so it
   adds no billed time on the critical path.

**A packaging fact to verify first:** a slim engine image may carry no interpreter. The agent
would then ship as a single self-contained executable per platform, built in CI, rather than as
a package needing a runtime on the host.

## New verbs — each is a decision

The protocol's verb list is closed, and a test fails until it is edited on purpose.

| Verb | Does | Bounded by |
|---|---|---|
| `POST /heartbeat` | Touches the timer's heartbeat file | No body. It can only ever *postpone* a shutdown the pool could also cause by going silent |
| `GET /facts` — extended | Adds load average, accelerator utilisation and memory, measured generation rate | Read-only; no request content |

Loading-as-landed (D57) and resizing (D56) need **no** new verb — they are `PUT /models` and
`POST /engine` doing what they already do.

On a host the pool created, the restart command in the agent's settings is written by the pool's
own start-up material rather than by a machine owner — consistent with
[hosts-routing-capacity.md](../spec/hosts-routing-capacity.md): what the pool creates is the
pool's to configure. It is still fixed at installation and never sent over the protocol.

## Decisions this story needs

1. The agent on rented hosts. **Supersedes the recommendation in host-agent §8.**
2. Push or download; the form of the artefact.
3. The `heartbeat` verb and the extended facts (each a verb decision, per D40).
4. **Amends D41** for pool-created hosts only: the restart command comes from the pool's start-up
   material. For delegated hosts D41 stands untouched.

## Open questions for the owner

1. **Push or download?** Recommended: push, for the reasons in the table.
2. **Fallback, or give the host up, when the agent does not answer?** Recommended: fall back.
3. **Should the agent replace the SSH heartbeat, or run beside it for a while?** Recommended:
   beside it until it has been seen working live.

## Build stages

1. Verify what the engine image carries; choose the artefact form; build it in CI.
2. Push, start, per-host key; second forward; facts from a rented host in the console.
3. `PUT /models` as the preparation path, with load-as-landed (**delivers D57**).
4. The heartbeat verb; SSH heartbeat kept as the fallback.
5. `POST /engine` on rented hosts (**delivers D56's mechanism**).

## Tests

The fake provider's hosts gain a real agent app over the fake engine: install succeeds, install
fails and the host still joins, the agent dies and the timer still fires, a foreign key is
refused. No GPU, no cloud account.

## Depends on

Nothing. **Delivers D56 and D57. S5 depends on this.**
