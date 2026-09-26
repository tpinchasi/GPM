# GPM — GPU Hosts Pool Management

A layer that **serves GPU inference to applications without the applications knowing where the
GPU is**. It keeps a pool of model-serving hosts — the local machine, fixed remote machines,
and instances rented on a bidding marketplace — routes each request to a free worker on the
best available host, and does the renting, bidding, recovery and tear-down itself, inside
spending limits an operator sets.

**Status: phases 1–4 are built** — the router and client SDK, the supervisor with leases,
renting and the dead-man timer, and the operator console. It has run against a real marketplace:
a capped live run bid, prepared a host, served a request through the router, recovered from a
real eviction unattended and tore everything down, for $0.008. Two release-checklist items remain.
See [STATUS.md](STATUS.md) for exactly where things stand.

**Intent: to be released publicly as a reusable framework.** This repository is private until
the items in [docs/release-checklist.md](docs/release-checklist.md) are done.

## Install

Needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). One workspace, three packages:
the server, the client SDK, and the optional host agent.

```sh
git clone <this repository> && cd GPM
uv sync                                        # everything, including the dev tools
uv run gpm --help
```

An app only needs the client, which depends on `httpx` and nothing else:

```sh
uv pip install ./client                        # or: pip install gpm-client, once published
```

## Configure

One file describes the pool. Start from the example and edit it:

```sh
cp server/examples/pool.yaml pool.yaml
```

The parts that matter, and what each decides:

| Block | What it sets |
|---|---|
| `pool.model_set` | The models **every** host must hold before it is used. A request for anything else is refused, never loaded on demand |
| `catalog` | Logical names (`my-model:7b`) and the per-capability builds they resolve to, so one request serves an Apple-silicon build on the laptop and a CUDA build on a rented host |
| `hosts` | The machines you already have — local, or remote over `http`, `https` or an SSH `tunnel` |
| `listen` / `control` | Where the app-facing router and the operator control API listen. Loopback unless you give TLS |
| `auth` | Where the hashed key files live (below) |
| `rented` | Everything about renting: provider, image, offer policy, bidding ceilings, teardown timings, and the opt-in features — `allocation: dynamic`, `workers_auto`, `agent_on_rented_hosts` |
| `limits` | Pool-wide ceilings: how many rented hosts at once, and optionally an overall hourly burn |

Check a change before it takes effect — `plan` says what would differ, and loosening a ceiling
has to be typed again:

```sh
uv run gpm config validate -f pool.yaml
uv run gpm config plan -f pool.yaml
uv run gpm config apply -f pool.yaml
```

### Keys

Three roles, never interchangeable. **Only hashes are stored**; each key is shown once, when it
is created.

| Key | Who holds it | What it opens | Where the pool reads it |
|---|---|---|---|
| **App key** (`gpma_…`) | Every application | The router: inference, and nothing else. **Required, loopback included** | `auth.app_keys_file` |
| **Admin key** (`gpmx_…`) | The operator | The control API and console: leases, renting, configuration. **Never reaches an app**, and an app key is refused here | `auth.admin_keys_file` |
| **Agent key** (`gpmg_…`) | One host's agent | That host's agent, and only from the pool | Named by `agent.bearer_env` on that host |

```sh
uv run gpm key create --role app                 # give this to an application
uv run gpm key create --role admin               # keep this to yourself
uv run gpm key list                              # fingerprints and when each was made
uv run gpm key revoke <fingerprint>
```

The provider's account credential is **never** written in configuration and never placed on a
rented machine: it is read from the environment of the supervisor's own process
(`VAST_API_KEY` for the first provider). What a rented host carries is the provider's
instance-scoped credential, which can only end that one instance.

## Run it

```sh
export VAST_API_KEY=…                          # only if the pool may rent
uv run gpm serve -c pool.yaml                  # the router, and the supervisor beside it
uv run gpm status                              # what the pool sees
```

`gpm serve` runs the router in this process and the supervisor in its own, sharing one SQLite
file. They never call each other: if the supervisor stops, the router keeps serving from the
last table it published.

Then open **http://127.0.0.1:8081/ui** for the console: hosts by tier, leases with their
burn-down, the live market through your own offer policy, and every decision with the numbers
behind it. Everything it does is also a CLI verb — `gpm lease`, `gpm host`, `gpm config`,
`gpm market`, `gpm plan`, `gpm down --all`.

### Deploy

**A pool that spends money runs a tagged release, not a working tree.** Build one, check it,
then point `current` at it:

```sh
deploy/gpm-deploy v0.6.1               # build it: git archive of the tag, its own venv
deploy/gpm-deploy v0.6.1 --activate    # and make it current
~/.local/share/gpm/releases/current/venv/bin/gpm --version
```

A release is a directory with its own virtual environment, installed **non-editable** from the
tag — so editing your checkout cannot change what is running. `--activate` is one symlink swap,
and rolling back is activating the previous tag, still sitting on disk beside it. A build that
fails to identify itself as its tag, or to pack its agent, is deleted rather than activated.

`gpm --version` tells you which release a process is running, or says plainly that it is a
development tree. Three live failures — a stale agent shipped to hosts, and twice a fix that
was committed but not running — all came from that state, and each was found only by a rented
host failing.

Then two processes, one directory, no services to install:

```sh
RELEASE=~/.local/share/gpm/releases/current/venv/bin
$RELEASE/gpm serve -c pool.yaml                  # both halves
$RELEASE/gpm supervise -c pool.yaml              # or: the supervisor alone, on its own machine
$RELEASE/gpm serve -c pool.yaml --router-only    # and the router alone, reading the same database
$RELEASE/gpm forwarder -c pool.yaml              # the SSH forwards, when forwarder.enabled (started by the supervisor by default)
```

During development, `uv run gpm serve -c pool.yaml` runs the working tree instead — convenient,
and it will tell you that is what it is doing.

- **Keep the database and `pool.yaml` together**, and back up the database: it holds the host
  table, the leases, the spend ledger and the decision log.
- **Only one supervisor may run per pool.** A second refuses to start, by a lock in the
  database, and says which process holds it.
- **The listeners are loopback by default.** Off loopback, the router needs TLS; a plain-HTTP
  listener with a key travelling over it is refused at load.
- **Restart when nothing is rented** where you can: a restart takes back its hosts, but the
  SSH tunnels to them are re-opened as it starts.

**Nothing rents until you say so.** A lease is the only thing that can spend, and one that may
rent cannot be opened without a dollar cap:

```sh
uv run gpm lease open --workers 4 --max-hours 2 --max-spend 5.00 --allow-rent
uv run gpm host prepare --max-spend 3 --max-hours 3 --when-ready join   # or one host, now
uv run gpm down --all                                                   # the panic button
```

## Use it from an application

An app knows one URL and one key:

```sh
export GPM_URL=http://127.0.0.1:8080
export GPM_API_KEY=gpma_…
```

```python
from gpm_client import PoolClient

pool = PoolClient()                            # reads GPM_URL and GPM_API_KEY
reply = pool.chat("my-model:7b", [{"role": "user", "content": "hello"}])
reply.content, reply.served_model              # what came back, and which build served it
```

The SDK waits and retries for capacity by default, and raises rather than waiting where waiting
cannot help. [client/README.md](client/README.md) covers the transport, the retry policy, the
errors, streaming, and what each answer tells you.

Every model in the pool's set must already be loaded on a host before that host is used: the
pool verifies a host's engine, it never configures it, and no request ever triggers a pull.

### Tests

```sh
uv run pytest                                  # the default suite: no GPU, no cloud account
uv run pytest -m simulation                    # whole-pool scenarios against a moving market
uv run pytest -m integration                   # opt-in, needs a local Ollama holding the models
```

The **[simulation](docs/simulation.md)** runs the real router and supervisor against a market
that moves under them — load that climbs and stops, machines that come and go, hosts taken away
mid-answer, a provider that goes quiet, a supervisor restarted under traffic. It is the merge
gate for a change to renting, allocation or recovery: CI runs it on a pull request that carries
the `simulation` label, and on every supported Python weekly and on request. Every pull request
gets the default suite, lint, the version check, the secret scan and the dependency audit.

## Layout

| Path | What it is |
|---|---|
| [client/](client/) | `gpm_client` — the SDK an app depends on. One dependency: `httpx`. **[How to use it](client/README.md)** |
| [agent/](agent/) | `gpm_agent` — the optional host agent: tells the pool what a machine is. [Design](docs/spec/host-agent.md) |
| [server/](server/) | `gpm_server` — the router, the engine adapters, the `pool` command |
| [tests/](tests/) | The default suite against a fake engine and a fake provider, the opt-in suite against a real Ollama, and [the simulation](docs/simulation.md) |

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
| [docs/simulation.md](docs/simulation.md) | The whole-pool simulation: what it runs, the thirteen scenarios, what every run must hold to, and the faults it has already found |
| [docs/stories/README.md](docs/stories/README.md) | The planned features, one story each: the evidence behind them, the design, what each gives up |
| [docs/roadmap.md](docs/roadmap.md) | v1 scope, the four build phases and their exit criteria, deferred requirements, open questions |
| [docs/decisions.md](docs/decisions.md) | Every decision, numbered: what, why, what was rejected; verified facts and unverified assumptions |
| [docs/threat-model.md](docs/threat-model.md) | Assets, actors, 19 threats with mitigations and residual risk, deliberate non-goals |
| [docs/release-checklist.md](docs/release-checklist.md) | What must be true before the first public commit and the first release |
| [docs/architecture-review.md](docs/architecture-review.md) | The critical review of the original single-file design; every finding carries its decision |
| [docs/adopters/aletheia/](docs/adopters/aletheia/README.md) | First adopter guide — **adopter-internal, see the note in STATUS.md before publishing** |
| [docs/archive/](docs/archive/design-2026-09-17-pre-split.md) | The pre-split single-file design, kept only so the review's section references resolve — **adopter-internal** |
