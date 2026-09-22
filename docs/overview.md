# GPU Host Pool — Overview

> Status: phases 1–3 are built — the router and SDK, the supervisor with leases and renting,
> and the operator console. See [../STATUS.md](../STATUS.md). The project is **GPM**: the CLI is
> `gpm`, the packages are `gpm-client` and `gpm-server`, the environment prefix is `GPM_*` and
> the header prefix `X-GPM-*` (D36). Decisions and their reasons are in
> [decisions.md](decisions.md); this page only states how things are.

> **What it costs, honestly:** [economics.md](economics.md) measures cost per million tokens against a managed API, where a pool wins, and what would close the gap.

## What it is

A layer that **serves GPU inference to applications without the applications knowing where the
GPU is**. It keeps a pool of model-serving hosts, routes each request to a free worker on the
best available host, and — for hosts that are rented by the hour on a bidding market — does the
renting, bidding, recovery and tear-down itself, inside spending limits an operator sets.

An application sees one URL and one API key. Everything else is the pool's business.

## What it is not

- Not an inference engine. It sits in front of one (Ollama first; others by plug-in).
- Not a model-format translator. Requests pass through to the engine in the engine's own API.
- Not a general cloud orchestrator. It manages hosts for one purpose: serving a declared set of
  models to the apps holding the pool's key.

## The ideas, in the order you need them

**Pool.** The unit of isolation. One pool has one set of hosts, one declared set of models, one
routing policy, one budget, and one API key that admits apps to it. Workloads that must be kept
apart use different pools.

**Host.** A machine running an inference engine. Three kinds:

| Kind | Example | Lifetime controlled by | Cost |
|---|---|---|---|
| `local` | The engine on the machine the pool runs on | nobody — it is just there | none |
| `fixed-remote` | A machine you own or lease at a known address | nobody by default; optional start/stop commands | flat |
| `rented-interruptible` | An instance on a bidding marketplace | the pool: bid, recover, replace, park, destroy | hourly + storage + download |

Any remote host is reached over one of three **transports** — a supervised SSH `tunnel`, plain
`http`, or `https` — chosen per host, independently of its kind.

**Worker.** The unit of capacity. A worker belongs to one host and serves one request at a time.
A host has as many workers as its hardware can usefully run for the pool's models (a laptop-class
machine might have 3, an 80 GB datacentre card 6). Pool capacity is the workers on ready hosts.

**Model set.** A pool declares the models it serves, and **every host keeps all of them loaded,
all the time**. Nothing is loaded on demand, so nothing is ever swapped out; a host that cannot
hold the whole set does not join.

**Logical model names.** An app asks for a model by one name; the pool serves the build of it
that suits the host — for example an Apple-optimised build on Apple silicon and the standard
build elsewhere — but only for names an operator listed in the catalog. The build actually
served is always reported back.

**Routing priority.** Strict tiers: `local` first, `fixed-remote` next, `rented-interruptible`
last. A lower tier is used only when every eligible worker above it is busy. Rented hosts
therefore carry only the overflow, go idle first, and are released first. Within a tier the
least-loaded host wins; across the pool the queue is first come, first served.

**Lease.** The only thing that can spend money. A lease says how many workers are wanted, for
how long at most, and for how many dollars at most. Inside an open lease the pool may bid,
recover and replace rented hosts unattended; with no lease it rents nothing. Dollar caps are
mandatory, enforced against the provider's own reported charges, with a safety margin.

**Two processes.** The **router** sits in the path of every request and must never stall. The
**supervisor** does the slow, failure-prone work: provider calls, tunnels, probes, bidding,
tear-down. They share a SQLite database and never call each other; if the supervisor dies the
router keeps serving from the last host table it published.

**Client SDK.** A small library apps use to reach the pool. Its default behaviour is to **wait
and retry when the pool has no capacity** rather than fail, with a configurable limit. Any plain
client of the engine's API also works — it just does not get the waiting behaviour.

**Operator console.** A web UI over the same control API the CLI uses: set up hosts, configure
bidding with a live view of the market, prepare a rented host with the pool's models ahead of a
run, open leases, and read every decision the pool made with the numbers behind it.

## How a request flows

```
 app ──(engine API + gpm key)──▶ router ──▶ idle worker on the highest-priority eligible host
                                    │             local ▸ fixed-remote ▸ rented
                                    │
                        none idle ──┤── queue (FCFS) ── still none ──▶ 503 + Retry-After + reason
                                    │                                        │
                                    │                         SDK waits and retries (default)
                                    ▼
                         response passes through untouched,
                         plus headers: model served, host, runtime class
```

## How money is spent

```
 operator ── opens lease (workers, max hours, max dollars) ──▶ supervisor
                                                                  │
      overflow = wanted workers − workers on local and fixed hosts
                                                                  │
      parked host with the models? ── restart it ─┐               │
      otherwise: filter offers ▸ rank ▸ price bid ▸ place ▸ verify ┘
                                                                  │
      every pass: probe, reconcile spend with the provider, act on evictions,
                  release what is idle or over budget, sweep for orphans
                                                                  │
      on every rented host: a dead-man timer that ends the instance
                            if the supervisor and all traffic go silent
```

## Where to read next

| If you want to… | Read |
|---|---|
| Point an app at a pool | [spec/app-contract.md](spec/app-contract.md) |
| Understand hosts, workers, routing and model resolution | [spec/hosts-routing-capacity.md](spec/hosts-routing-capacity.md) |
| Understand leases, bidding, recovery and tear-down | [spec/supervisor.md](spec/supervisor.md) |
| Add a provider, an engine or a strategy | [spec/plugin-interfaces.md](spec/plugin-interfaces.md) |
| Operate a pool | [spec/console-and-control-api.md](spec/console-and-control-api.md) |
| Know what is in v1 and in what order it is built | [roadmap.md](roadmap.md) |
| Know why something is the way it is | [decisions.md](decisions.md) |
| Assess the risks | [threat-model.md](threat-model.md) |
| See it applied to a real codebase | [adopters/aletheia/README.md](adopters/aletheia/README.md) |
