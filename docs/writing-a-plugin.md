# Writing a plug-in

GPM has three extension points, and they exist so you never have to fork it:

| You want to | Write a | Because it varies with |
|---|---|---|
| Rent from another marketplace or cloud | **Provider** | How a machine is obtained, priced, kept and given back |
| Serve with another inference engine | **Engine** | What is running on the machine, and how you ask it things |
| Change when and how much the pool bids | **Strategy** | Your policy, not the framework's |

The normative definitions are in [spec/plugin-interfaces.md](spec/plugin-interfaces.md). This
page is how to actually write one.

## Before you start: how plug-ins are trusted

A provider or engine plug-in **runs inside the supervisor with its full authority**, including
the provider credential. Installing one is a trust decision equal to installing GPM itself, and
only plug-ins named in configuration are ever loaded. Strategies are different: they are pure
functions that cannot do I/O, and everything they decide is re-checked.

## A provider

A provider is an **HTTP API client** — never a wrapper around a command-line tool. CLI output
formats change between versions, errors arrive as prose, interactive prompts need workarounds,
and none of that belongs in an unattended service that spends money.

Start from `server/src/gpm_server/providers/base.py` for the protocol and
`providers/vast.py` for a worked example against a real marketplace.

### Say what you can do

```python
capabilities = ProviderCapabilities(
    interruptible=True,        # instances are bid for and can be outbid
    parkable=True,             # an instance can be stopped with its disk kept
    same_machine_rebid=True,   # a stopped instance restarts by raising its bid
    self_terminate=True,       # an instance-scoped credential lets it end itself
    reports_charges=True,      # you can say what an instance has cost so far
    direct_port_mapping=False, # a public port can be mapped to the engine
)
```

The pool **adapts to what you lack** rather than assuming. Without `interruptible`, no bidding
strategy applies. Without `parkable`, tear-down is always destroy. Without `self_terminate`
there is no dead-man timer, and the pool refuses leases longer than a short maximum on your
provider — so declare that one honestly. Without `reports_charges`, spend is estimate-only and
the cap safety margin widens by itself.

Declaring a capability you do not really have is the one way to hurt an operator: GPM narrows
the cap margin only once a charge has *actually* been reported, precisely because a provider
declared this and never delivered.

### The guarantees that matter most

- **`list_instances` returns everything** carrying the label prefix, in any state, including
  stopped instances that still bill storage. The orphan sweep and crash recovery rest on it.
- **`create` either returns an instance or leaves nothing behind.** Verify this yourself before
  reporting a lost bid — look for the label you were about to use and end what you find. A real
  marketplace answered `success: false` and created the instance anyway (D43), and the pool
  paid for two machines it did not know it had. **The pool now re-checks this by label after
  every failed bid and refuses to bid again until it can prove nothing is running**, so a
  plug-in that gets it wrong costs an operator a pass rather than a machine — but get it right.
  A losing bid must raise,
  not leave a parked instance quietly billing.
- **`destroy` is idempotent.** The pool verifies by listing again; it never trusts your return
  value.
- **Errors are typed**, never a bare HTTP error: `ProviderAuthError`, `ProviderRateLimited`,
  `ProviderUnavailable`, `OfferGone`, `BidLost`.
- **Every call is bounded by a timeout** and never blocks indefinitely.

### Read the account credential from the environment

```python
key = os.environ.get(self.api_key_env)   # never from configuration
```

Configuration files get committed. Environments do not. And whatever you put in
`self_terminate_command()` is written to a file on a machine you do not control, so it may
carry only the provider's per-instance credential.

### Test it against the same scenarios as the fake

`providers/fake.py` implements the whole interface in memory and can be scripted: floors that
move, a bid lost between search and create, an instance parked by the provider, an eviction on
cue, a `destroy` that fails the first time, charges that drift from the estimate. Your provider
should behave sensibly in each. Test against recorded HTTP — `httpx.MockTransport` is what
`tests/unit/test_vast_provider.py` uses — so the suite still needs no account.

One warning from experience: the first real call will disagree with the documentation
somewhere. In our case a field documented as the on-demand price turned out to be the bid floor
repeated, which made every bid unwinnable. Budget a tiny, hard-capped live run before trusting
your mapping.

## An engine

An engine adapter tells the pool how to talk to one kind of inference server. **The router
passes requests through in the engine's own API and never translates between APIs** — that is
where tool calling and structured output get quietly lost, and the pool does not take that risk.

Split by where each operation runs:

```python
# The request path. Cheap, synchronous, no I/O — this is in the way of every request.
def inference_paths(self) -> set[str]: ...
def requested_model(self, path: str, body: bytes) -> str | None: ...
def with_model(self, path: str, body: bytes, tag: str) -> bytes: ...
def wants_schema(self, path: str, body: bytes) -> bool: ...
def is_streaming(self, path: str, body: bytes) -> bool: ...

# The probe and prepare paths. Allowed to be slow.
async def health(self, client) -> Health: ...
async def models_resident(self, client) -> frozenset[str]: ...
async def pull(self, client, tag: str) -> PullResult: ...
async def load_and_pin(self, client, tags: list[str]) -> None: ...
def launch_settings(self, workers, context, n_models) -> dict[str, str]: ...
```

**`with_model` must change the model field and nothing else** — byte for byte otherwise. The
Ollama adapter splices the new tag into the raw bytes with a regular expression rather than
re-serialising the parsed body, because re-serialising reorders keys and renormalises numbers
and whitespace, and passthrough fidelity is the promise the whole router rests on.

**`models_resident` reports what is loaded now**, not what is present on disk. "Every host holds
the whole model set" and eviction detection both depend on that distinction.

**`load_and_pin` must raise if the tags cannot all be resident together.** A host that cannot
hold the set does not join the pool, and it is better to say so than to thrash.

## A strategy

Strategies are **pure functions**: no I/O, no clock, no randomness of their own. Everything
they need arrives as arguments and everything they decide is returned **with the numbers that
produced it**. That is what makes them unit-testable, explainable in the console's Decisions
screen, and safe for an operator to swap.

```python
def price(self, offer: Offer, market: MarketView, cfg: BidConfig) -> Bid:
    bid = offer.min_bid_hourly + cfg.premium
    return Bid(hourly=bid, reasons=[f"floor ${offer.min_bid_hourly:.3f} + ${cfg.premium:.3f}"])
```

Three rules:

- **Return your reasons.** Every decision object carries the comparisons that produced it, and
  the supervisor stores them verbatim. An operator will one day ask why you bid that.
- **You cannot exceed the ceilings.** The supervisor clamps every bid to the configured ceiling
  and the on-demand crossover *after* you return, and re-checks every cap before acting. This
  is not a courtesy — it is what makes a third-party strategy safe to run.
- **You cannot relax a hard filter.** The built-in offer policy rejects before your scoring ever
  sees an offer.

A bid your strategy returns may come back clamped below the market floor, in which case the pool
places no bid at all rather than an unwinnable one.

## Loading it

Plug-ins are ordinary Python packages discovered by entry point, and configuration names the
one in use:

```yaml
rented:
  provider: your-marketplace
  provider_settings: { region: eu-west }
```

Each interface carries a major version. Adding an optional operation or capability is
backwards-compatible; changing a guarantee is a new major version, and the pool refuses a
plug-in whose version it does not support.
