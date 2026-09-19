# Specification — Plug-in Interfaces

> The three places the framework is meant to be extended without forking: **providers**,
> **engines** and **strategies**. Each interface is versioned; a plug-in declares the interface
> version it implements and the pool refuses one it does not support. v1 ships one
> implementation of each, plus a fake provider. Signatures below are illustrative Python; the
> normative part is the operations, their guarantees, and the declared capabilities.

The split matters. A single "host kind" interface mixed two unrelated concerns — *how a machine
is obtained* and *what is running on it*. They vary independently: the same marketplace can run
any engine, and the same engine runs on a laptop, a leased server or a marketplace instance.

| Interface | Answers | Varies with |
|---|---|---|
| **Provider** | How is a machine obtained, priced, kept and given back? | The marketplace or cloud |
| **Engine** | What is serving on the machine, and how do I ask it things? | The inference server |
| **Strategy** | Given the facts, what should the pool do? | The operator's policy |

`local` and `fixed-remote` hosts have **no provider** — nothing obtains or returns them. They
have an engine like any other host.

---

## 1. Provider interface (v1)

A provider is an **HTTP API client** with typed errors, timeouts and retries — never a wrapper
around a command-line tool. Output formats of CLIs change between versions, interactive prompts
need workarounds, errors arrive as text, and a CLI drags in a runtime to install; none of that
is acceptable in an unattended service that spends money.

```python
class Provider(Protocol):
    interface_version: ClassVar[str] = "1"
    capabilities: ProviderCapabilities

    def list_instances(self, label_prefix: str) -> list[Instance]: ...
    def search_offers(self, query: OfferQuery) -> list[Offer]: ...
    def create(self, offer: Offer, spec: InstanceSpec, bid: Money | None) -> Instance: ...
    def set_bid(self, instance: Instance, bid: Money) -> None: ...
    def start(self, instance: Instance) -> None: ...
    def stop(self, instance: Instance) -> None: ...            # keep the disk
    def destroy(self, instance: Instance) -> None: ...
    def status(self, instance: Instance) -> InstanceStatus: ...
    def connection(self, instance: Instance) -> ConnectionInfo: ...
    def reported_charges(self, instance: Instance) -> Charges | None: ...
    def account(self) -> AccountStatus: ...                    # credential valid, credit left
```

### Guarantees the pool requires

| Operation | Guarantee |
|---|---|
| `list_instances` | Returns **every** instance carrying the label prefix, in any state, including stopped ones that still bill storage. The orphan sweep and crash recovery rest on this |
| `create` | Either returns a running-or-scheduling instance, or raises and **leaves nothing behind**. A bid that loses must fail, not leave a parked instance |
| `destroy` | Idempotent. The pool verifies by calling `list_instances` again; it never trusts a return value |
| `status` | Distinguishes at least: running, scheduling, stopped-by-us, **outbid / stopped-by-provider**, gone |
| `search_offers` | Each `Offer` carries what ranking needs: hardware, memory, a throughput proxy, the current minimum bid, on-demand price if any, storage price, download price per gigabyte, download speed, reliability, provider verification flag, a stable machine identifier |
| `reported_charges` | What the provider says the instance has cost so far, or `None` if it cannot say |
| Every operation | Bounded by a timeout; raises typed errors (`ProviderAuthError`, `ProviderRateLimited`, `ProviderUnavailable`, `OfferGone`, `BidLost`); never blocks indefinitely |

### Declared capabilities

A provider states what it can do; the pool adapts rather than assumes.

| Capability | If absent |
|---|---|
| `interruptible` — instances are bid for and can be outbid | No bidding strategies apply; `create` takes no bid |
| `parkable` — an instance can be stopped with its disk kept | "Park" is unavailable; tear-down is always destroy |
| `same_machine_rebid` — a stopped instance can be restarted by raising its bid | Eviction choice is "replace" only |
| `self_terminate` — an **instance-scoped** credential lets an instance end itself | No dead-man timer; the pool refuses leases longer than a short maximum on this provider |
| `reports_charges` | Spend is estimate-only; the cap safety margin is widened and the console says why |
| `price_history` | Volatility-aware bidding relies on the pool's own samples only |
| `direct_port_mapping` — a public port can be mapped to the engine | `http` / `https` transports to rented hosts are unavailable; `tunnel` only |

### The fake provider

Ships with the framework and implements the full interface in memory. It can be scripted:
offers and floors over time, a bid lost between search and create, an instance parked by the
provider, an eviction at a chosen moment, a `destroy` that fails the first time, charges that
drift from the estimate. **The entire supervisor test suite runs against it; no test needs a
cloud account.**

---

## 2. Engine interface (v1)

An engine adapter tells the pool how to talk to one kind of inference server. The router
**passes requests through in the engine's own API and never translates between APIs** — an app
written for engine X talks to a pool of engine-X hosts. (Translation is where tool-calling and
structured-output fidelity get lost; the pool does not take that risk on.) A pool has one
engine type.

```python
class Engine(Protocol):
    interface_version: ClassVar[str] = "1"
    name: ClassVar[str]

    # --- used by the router, on the request path: must be cheap and never block ---
    def inference_paths(self) -> set[str]: ...                      # paths that take a worker
    def requested_model(self, path: str, body: bytes) -> str | None: ...
    def with_model(self, path: str, body: bytes, tag: str) -> bytes: ...   # rewrite the model field only
    def wants_schema(self, path: str, body: bytes) -> bool: ...     # structured output requested?
    def is_streaming(self, path: str, body: bytes) -> bool: ...

    # --- used by the supervisor ---
    async def health(self, conn: Connection) -> Health: ...
    async def models_present(self, conn: Connection) -> set[str]: ...
    async def models_resident(self, conn: Connection) -> set[str]: ...
    async def pull(self, conn: Connection, tag: str) -> AsyncIterator[PullProgress]: ...
    async def load_and_pin(self, conn: Connection, tags: list[str]) -> None: ...
    async def smoke_test(self, conn: Connection, tag: str, schema: bool) -> SmokeResult: ...
    def launch_settings(self, workers: int, context: int, n_models: int) -> dict[str, str]: ...
    def looks_corrupt(self, text: str) -> bool: ...
```

### Guarantees the pool requires

| Operation | Guarantee |
|---|---|
| `with_model` | Changes the model field and **nothing else** — byte-for-byte otherwise. The router relies on this to keep passthrough exact |
| `models_resident` | Reports what is loaded *now*. The "whole model set on every host" rule and eviction detection rest on it |
| `load_and_pin` | Loads all the tags and configures them to stay loaded; raises if they cannot all be resident together |
| `launch_settings` | Environment or flags that make the engine run `workers` requests in parallel at `context`, holding `n_models` models — used only on hosts the pool creates |
| `smoke_test` | A short fixed generation; with `schema=True`, reports whether a structured-output schema was actually **enforced**, not merely accepted |
| `looks_corrupt` | Engine-specific signs of a degraded back-end (reserved tokens leaking, repetition collapse); feeds quarantine |

The first engine is Ollama. Servers exposing an OpenAI-compatible API (vLLM, llama.cpp's server
and others) are the natural second adapter: the same interface, different paths and body shapes.

---

## 3. Strategy interface (v1)

Strategies are **pure functions** — no I/O, no clock, no randomness of their own. Everything
they need arrives as arguments; everything they decide is returned, together with the reasons.
That is what makes them unit-testable, replayable against recorded markets, and explainable in
the console's Decisions screen.

```python
class RentStrategy(Protocol):      # when, and how many
    def decide(self, demand: Demand, capacity: Capacity, lease: Lease, cfg: ScaleConfig) -> RentDecision: ...

class OfferStrategy(Protocol):     # which offers are acceptable, and in what order
    def reject_reasons(self, offer: Offer, need: HostNeed, cfg: OfferPolicy) -> list[str]: ...
    def score(self, offer: Offer, need: HostNeed, lease: Lease) -> float: ...

class BidStrategy(Protocol):       # how much
    def price(self, offer: Offer, market: MarketView, cfg: BidConfig) -> Bid: ...

class EvictionStrategy(Protocol):  # re-bid in place, replace, or (post-v1) wait
    def decide(self, host: Host, market: MarketView, lease: Lease, cfg: HoldConfig) -> EvictionDecision: ...

class TeardownStrategy(Protocol):  # who goes, and park or destroy
    def decide(self, hosts: list[Host], demand: Demand, lease: Lease | None, cfg: TeardownConfig) -> list[TeardownAction]: ...
```

Rules every strategy obeys:

- **Returns its reasons.** Every decision object carries the numbers and comparisons that
  produced it; the supervisor stores them verbatim.
- **Cannot exceed the ceilings.** The supervisor clamps every bid to the configured ceiling and
  the on-demand crossover *after* the strategy returns, and re-checks every cap before acting.
  A faulty or hostile strategy cannot spend past the limits.
- **Cannot relax a hard filter.** `reject_reasons` from the built-in offer policy are applied
  before any custom scoring sees an offer.

---

## 4. Versioning and loading

- Each interface has an integer major version. Adding an optional operation or capability is
  backwards-compatible; changing a guarantee is a new major version.
- Plug-ins are ordinary Python packages discovered by entry point; the pool's configuration
  names the ones in use. Nothing is loaded that configuration does not name.
- A plug-in runs **inside the supervisor process with its full authority**, including the
  provider credential. Installing one is a trust decision equivalent to installing the
  framework itself — see [../threat-model.md](../threat-model.md).
