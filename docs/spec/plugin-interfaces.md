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

## 1. Provider interface (v2)

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
| `create` | Either returns a running-or-scheduling instance, or raises and **leaves nothing behind**. A bid that loses must fail, not leave a parked instance. Its price is the bid for a bid offer; for a spot offer, the most the pool pays, which a provider that takes a maximum price holds the instance to (D132); none on demand |
| `destroy` | Idempotent. The pool verifies by calling `list_instances` again; it never trusts a return value |
| `status` | Distinguishes at least: running, scheduling, stopped-by-us, **outbid / stopped-by-provider**, gone. An instance the provider has decided not to run is stopped **whatever its container is doing** — outbid while still loading its image included (D109) |
| `search_offers` | Each `Offer` carries what ranking needs: hardware, memory, a throughput proxy, the current minimum bid, on-demand price if any, storage price, download price per gigabyte, download speed, reliability, provider verification flag, a stable machine identifier |
| `reported_charges` | What the provider says the instance has cost so far, or `None` if it cannot say |
| Every operation | Bounded by a timeout; raises typed errors (`ProviderAuthError`, `ProviderRateLimited`, `ProviderUnavailable`, `OfferGone`, `BidLost`); never blocks indefinitely |
| `BidLost` | Carries the provider's answer (`response`) and, where the attempt created an instance, that instance's state as read **before** it was destroyed (`instance_state`) — with every credential-named field replaced. The pool writes both into the `bid_failed` event, and replaces credentials once more before it does |

**`self_terminate_request(action)`** returns the call one instance makes to end itself — method,
URL, headers, optional body — never a command line (D71). Header values may name an environment
variable the provider injects; the account credential never appears. The host decides how to make
the call, because what a machine has to make it with is not the provider's business.

**Optional: `offer_for_machine(machine_id, gpus)`** — the current bid listing for one machine's
slice, asked of the machine itself whether or not anyone may rent it (D109). The ordinary search
lists rentable machines only, and a machine the pool was just outbid on is held by whoever
outbid it — so without this the pool has no price to re-bid against, and lets the host go. A
provider without it is not an error: an outbid host whose machine the search cannot see is
released.

**Optional in an instance's status: `startup_material`** — true, false, or unknown. A provider
that can tell whether an instance still carries the start-up material it was created with says
so, and the pool ends a host that lost it (D65). A provider that cannot leaves it unknown, and
the pool falls back to its give-up rule for hosts that never start.

### Declared capabilities

A provider states what it can do; the pool adapts rather than assumes.

| Capability | If absent |
|---|---|
| `interruptible` — instances can be taken away: outbid, or reclaimed at a spot price. Each offer's `bidding` says whether the pool sets its price (a bid) or the provider does (a spot price, which may move while the instance runs; `status` reports it as `bid_hourly`) (D132) | Every offer is on demand; `create` takes no price |
| `interruption_notice` — the provider warns before taking an instance away, and `status` says `interrupting` | The pool learns of an interruption only when the instance stops |
| `parkable` — an instance can be stopped with its disk kept | "Park" is unavailable; tear-down is always destroy |
| `same_machine_rebid` — a stopped instance can be restarted by raising its bid | Eviction choice is "replace" only |
| `self_terminate` — an **instance-scoped** credential lets an instance end itself | No dead-man timer; the pool refuses leases longer than a short maximum on this provider |
| `reports_charges` | Spend is estimate-only; the cap safety margin is widened and the console says why |
| `price_history` | Volatility-aware bidding relies on the pool's own samples only |
| `direct_port_mapping` — a public port can be mapped to the engine | `http` / `https` transports to rented hosts are unavailable; `tunnel` only |

### Version 2: the credential handed in, and how it presents itself (D134)

Every member is optional, read where present, so a version-1 plug-in still loads and runs — it
reads its own credential, and the console says so and offers no field for one.

```python
    interface_version: ClassVar[str] = "2"
    display_name: ClassVar[str]               # "Vast.ai"; the entry-point name where absent
    icon: ClassVar[str | None]                # a small SVG (≤16 KB), drawn as an image, never markup
    icon_url: ClassVar[str | None]            # or an https address of the provider's own logo
    endpoint_settings: ClassVar[tuple[str, ...]]  # the settings that decide where the credential goes
    offered: ClassVar[bool] = True            # False keeps a test double off Add provider
    credential_env: str                       # the variable read when nothing else is given

    def set_credential(self, credential: str | None) -> None: ...
```

- **The supervisor hands the plug-in its credential** and the plug-in reads no environment itself
  (providers.md §5). A new credential takes effect from the next call; a client built with the
  old one is closed once its calls have finished.
- **A stored credential is bound to `endpoint_settings`**: a connection whose values for them
  change no longer gets it.

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
structured-output fidelity get lost; the pool does not take that risk on.)

**A pool may run a different engine on each host** (D93) — `engine:` on a host, `rented.engine`
for the machines it buys, the pool's own where neither says. What makes this safe is that both
shipped engines serve one wire API (D89), so a request is read the same way wherever it goes:

- **The path chooses how a request is read**, not the host: it must be understood before the
  pool knows where it will go. Engines serving the same path read it identically, by
  construction — they share one module for it.
- **A request only reaches a host whose engine serves its path.** Ollama serves its own API
  beside the shared one and vLLM does not, so an `/api/*` request is eligible only on Ollama
  hosts. Without this the pool would hand a request to a machine that answers 404 to it.
- **A catalog variant may name the engine it is for**, and is then offered only to hosts running
  it — the same model is a plain tag to one engine and a model-hub repository to another. A
  variant naming no engine works anywhere, which is what every catalog written before this
  means.
- **A rented host keeps the engine it was rented with** (D103). Switching `rented.engine`
  changes what the pool rents next — image, start, port, download directory — and nothing about
  a machine already running: it is probed, dialled and prepared as what it is.

```python
class Engine(Protocol):
    interface_version: ClassVar[str] = "1"
    name: ClassVar[str]
    serves_one_model: ClassVar[bool] = False    # one process, one model?
    builds_on_hub: ClassVar[bool] = False       # are its builds model-hub repositories? (D100)
    options: ClassVar[dict[str, EngineOption]] = {}   # named options of its own start (D100)

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
    async def occupancy(self, conn: Connection) -> Occupancy | None: ...
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
| `occupancy` | Requests running, requests **waiting inside the engine**, and cache used — or `None` where the engine cannot say, which means "judge me by worker slots". Read by the supervisor on its own pass, never on the request path (D91) |
| `serves_one_model` | Declared, not inferred. `True` refuses a pool whose `models_per_host` is `all` **at load**, rather than after a machine has been rented that could never become ready (D89) |

### Why occupancy exists

Worker slots are exact for an engine that serves one request per slot, and misleading for one
that batches. A batching engine admitted at a hundred slots is rarely *all* busy and queues the
overflow internally, so "all workers busy" — which drives queueing, scale-up and the
`$/h ÷ tokens per second` ranking — would read as headroom while requests piled up somewhere
the pool cannot see. An engine that knows says so; one that does not is judged as before.

### The wire API

Both shipped engines serve the **OpenAI-shaped `/v1` paths**, and the SDK speaks them by
default (D89). The router still never translates between APIs: these are the same paths and the
same bodies on both engines, which is what lets a pool of either answer the same request. Ollama
also serves its own native API, and those paths keep working; a request arrives on whichever
surface the app used and is passed through on that surface.

The first engine is Ollama. **The second is vLLM** (D90), which differs in three ways the
interface had to accommodate:

- **It serves one model per process.** `serves_one_model` is `True`, and such a pool spreads its
  set across hosts (`models_per_host: declared`).
- **It cannot fetch a model over its own API.** `pull` refuses, **unretryably** — weights arrive
  from a model hub before the engine starts, so a vLLM host is prepared through the pool's agent
  or by its owner. A retryable refusal would have the supervisor retry a download that cannot
  happen and give up on the host for the wrong reason.
- **It is launched with its model and holds it for the process's life**, so `load_and_pin`
  verifies rather than acts, and `models_resident` and `models_available` are the same set.

### Starting vLLM on a host the pool creates

An engine declares **`loads_by_restart`** when a downloaded model is served by starting it
again rather than by asking it to load (D97), and may offer a **`default_start_command`** for
hosts the pool creates. vLLM does both: with no `engine_start` configured, the pool's restart
script runs the agent's own launcher,

```
python3 /var/run/gpm/gpm-agent.pyz vllm-start --models-dir /opt/gpm/models --port 8000 [--proxy]
```

which does nothing at boot (no models yet), and once the pool asks for the restart starts vLLM on
what the agent fetched — reading numbers from the environment file the agent writes, and names
from the directories it made, so no text from the pool reaches a command (D41). An operator may
still set `engine_start`; one written for a *different* engine is refused at load.

### Named options of an engine's own start (D100)

An engine may declare **named options** — `EngineOption(label, families)` — that its own start
can switch on. vLLM offers two:

| Option | What an app gets | Families (`model_type`) |
|---|---|---|
| `tool_calling` | `tools` in a request, `tool_calls` in the reply | `gemma4`, `gpt_oss`, `qwen2`, `qwen3`, `qwen3_moe` |
| `reasoning` | a thinking model's reasoning apart from its answer | `gemma4`, `qwen3`, `qwen3_moe` |

`rented.engine_options` names the ones in force; a name the engine does not offer is refused at
load, and so is naming any beside an `engine_start` (which replaces the engine's own start). The
pool writes each **name** into the machine's start (`vllm-start … --option tool_calling`); the
agent's launcher holds the flags, and chooses them per model by the family in that model's own
`config.json`. A model whose family has no such option starts without it, and the launcher says
so. The two tables — the engine's families and the launcher's flags — are kept equal by a test.

### Finding builds on the model hub (D100)

An engine that declares **`builds_on_hub`** has builds that are model-hub repositories, and the
pool looks them up for the operator: `GET /pool/builds?model=` searches the hub for the pool's
name in both spellings publishers use, most downloaded first, and sorts the answer — the
**original** (carrying the searched name, the most linked-to first, with the same publisher's other
originals of the same family) and each one's **builds** (quantisations, by the hub's own link) offered; files for other
engines (GGUF, MLX, no safetensors) and different models (fine-tunes, merges, quantisations of
something else) left out and counted. Each build carries its precision, its weight files' size,
the cards it runs on and runs at full speed on, its family and the options that apply to it, and
whether a rented host can fetch it: a **gated** repository cannot be picked, because the hub is
asked — and a rented host fetches — with no account.

### Choosing the build per machine

An engine published once per accelerator generation is configured as a list, newest first, and
the pool takes the first build a machine's driver can run (D92):

```yaml
rented:
  images:
    - { image: "vastai/vllm:v0.29.0-cuda-13.0", min_driver: "580" }
    - { image: "vastai/vllm:v0.29.0-cuda-12.9", min_driver: "550" }
```

A machine that can run none is refused **before it is bid on**. `image:` alone still means one
build for every machine, and the driver floor in the offer policy is then what keeps an unusable
machine out.

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
  names the ones in use. Nothing is loaded that configuration does not name, or that an
  operator did not choose on the console's Add provider: another package's plug-in is listed
  there from its package metadata alone (D134).
- A plug-in runs **inside the supervisor process with its full authority**, including the
  provider credential. Installing one is a trust decision equivalent to installing the
  framework itself — see [../threat-model.md](../threat-model.md).
