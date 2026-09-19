# GPM — GPU Hosts Pool Management

A layer that **serves GPU inference to applications without the applications knowing where the
GPU is**. It keeps a pool of model-serving hosts — the local machine, fixed remote machines,
and instances rented on a bidding marketplace — routes each request to a free worker on the
best available host, and does the renting, bidding, recovery and tear-down itself, inside
spending limits an operator sets.

**Status: phases 1–3 are built** — the router and client SDK, the supervisor with leases,
renting and the dead-man timer, and the operator console. It has run against a real marketplace:
a capped live run bid, prepared a host, served a request through the router, recovered from a
real eviction unattended and tore everything down, for $0.008. Phase 4 (Release) is in progress.
See [STATUS.md](STATUS.md) for exactly where things stand.

**Intent: to be released publicly as a reusable framework.** This repository is private until
the items in [docs/release-checklist.md](docs/release-checklist.md) are done.

## Running it

```sh
uv sync                                        # one workspace, two packages
cp server/examples/pool.yaml pool.yaml         # then edit: hosts, model set, catalog
uv run gpm key create --role app               # shown once; only its hash is stored
uv run gpm key create --role admin             # the control API and console need this one
uv run gpm serve -c pool.yaml                  # the router, and the supervisor beside it
uv run gpm status                              # what the pool sees
```

Then open **http://127.0.0.1:8081/ui** for the console: hosts by tier, leases with their
burn-down, the live market through your own offer policy, and every decision with the numbers
behind it. Everything it does is also a CLI verb — `gpm lease`, `gpm host`, `gpm config`,
`gpm market`, `gpm plan`, `gpm down --all`.

**Nothing rents until you say so.** A lease is the only thing that can spend, and one that may
rent cannot be opened without a dollar cap:

```sh
uv run gpm lease open --workers 4 --max-hours 2 --max-spend 5.00 --allow-rent
```

An app then knows one URL and one key:

```python
from gpm_client import PoolClient

pool = PoolClient()                            # GPM_URL, GPM_API_KEY
reply = pool.chat("my-model:7b", [{"role": "user", "content": "hello"}])
reply.content, reply.served_model
```

Every model in the pool's set must already be loaded on a host before that host is used: the
pool verifies a host's engine, it never configures it, and no request ever triggers a pull.

### Tests

```sh
uv run pytest                                  # the whole default suite: no GPU, no cloud account
uv run pytest -m integration                   # opt-in, needs a local Ollama holding the models
```

## Layout

| Path | What it is |
|---|---|
| [client/](client/) | `gpm_client` — the SDK an app depends on. One dependency: `httpx` |
| [server/](server/) | `gpm_server` — the router, the engine adapters, the `pool` command |
| [tests/](tests/) | The default suite against a fake engine and a fake provider, and the opt-in suites against a real Ollama |

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions need a Developer
Certificate of Origin sign-off (`git commit -s`); see [CONTRIBUTING.md](CONTRIBUTING.md).
Security reports go through [SECURITY.md](SECURITY.md), never a public issue.

## Docs

Start with the overview. The specification is generic and names no adopter.

| Doc | What it is |
|---|---|
| [STATUS.md](STATUS.md) | Where the project stands, what is decided, what is open, what to do next, session log |
| [docs/overview.md](docs/overview.md) | The concepts in one read: pool, hosts, workers, model set, routing priority, leases, the two processes, the SDK, the console |
| [docs/spec/app-contract.md](docs/spec/app-contract.md) | What an app relies on: the engine API plus a small pool dialect, keys, the client SDK and its default wait-and-retry, the time budget |
| [docs/spec/hosts-routing-capacity.md](docs/spec/hosts-routing-capacity.md) | Host kinds, transports and states; workers per host; the pool's model set; logical model names resolved by host capability; runtime classes; routing |
| [docs/spec/supervisor.md](docs/spec/supervisor.md) | Process layout, leases, cost controls and spend reconciliation, renting, bidding, evictions, the dead-man timer, preparing a host on request, tear-down, configuration sketch |
| [docs/spec/plugin-interfaces.md](docs/spec/plugin-interfaces.md) | The three extension points — provider, engine, strategy — with guarantees, declared capabilities and the fake provider |
| [docs/spec/console-and-control-api.md](docs/spec/console-and-control-api.md) | The operator console's rules and screens, test-connection, live market preview, the control API |
| [docs/roadmap.md](docs/roadmap.md) | v1 scope, the four build phases and their exit criteria, deferred requirements, open questions |
| [docs/decisions.md](docs/decisions.md) | Every decision, numbered: what, why, what was rejected; verified facts and unverified assumptions |
| [docs/threat-model.md](docs/threat-model.md) | Assets, actors, 19 threats with mitigations and residual risk, deliberate non-goals |
| [docs/release-checklist.md](docs/release-checklist.md) | What must be true before the first public commit and the first release |
| [docs/architecture-review.md](docs/architecture-review.md) | The critical review of the original single-file design; every finding carries its decision |
| [docs/adopters/aletheia/](docs/adopters/aletheia/README.md) | First adopter guide — **adopter-internal, see the note in STATUS.md before publishing** |
| [docs/archive/](docs/archive/design-2026-09-17-pre-split.md) | The pre-split single-file design, kept only so the review's section references resolve — **adopter-internal** |
