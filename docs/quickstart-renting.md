# Adding rented capacity

This adds a marketplace provider to a pool that already works
([quickstart.md](quickstart.md)). From here on the pool can spend your money, so the order of
this guide is deliberate: the limits first, then a look at the market that costs nothing, then
one small host, and only then leaving it unattended.

Everything below can be done from the console at `/ui` as well as the command line. Nothing is
console-only.

## The one rule worth learning first

**A lease is the only thing that can spend.** Configuration never spends — you can set up a
provider, tighten ceilings and browse the market all day without renting anything. A lease is
what grants the authority, and a lease that may rent **cannot be opened without a dollar cap**.

Two limits sit above every lease and cannot be loosened by one:

| Limit | Ships as | What it stops |
|---|---|---|
| `max_rented_hosts` | 1 | A bug or a bad strategy renting a fleet |
| `max_hourly_burn` | $1.00/h | The total rate across every rented host |

And two more inside the rented section:

| Limit | Ships as | What it stops |
|---|---|---|
| `offer_policy.max_all_in_hourly` | required, no default | A single host costing more than you meant — machines above it are not considered, and bids stop at it |
| `on_demand_crossover` | 0.8 | Paying near the on-demand price for a host that can still be evicted |

## 1. Give the provider its credential

The account credential is read from the environment and **never from configuration**, so it
cannot end up in a file you commit:

```sh
export VAST_API_KEY=...        # or whatever your provider's plug-in names
```

It never leaves the supervisor's process. What goes onto a rented host is the per-instance
credential the provider itself injects there, which can only stop or destroy that one instance.

## 2. Describe what you are willing to rent

Add to `pool.yaml`:

```yaml
limits:
  max_rented_hosts: 1          # start here; raise it once you trust the setup
  max_hourly_burn: 0.50

rented:
  providers:
    vast: { type: vast }       # a connection: a name, and the provider plug-in it uses
  image: ollama/ollama:0.34.2  # pinned, never a floating tag
  workers: 2                   # per card: a two-card machine runs twice this
  capabilities: [cuda]
  ssh_key: ~/.ssh/id_ed25519   # the pool installs its public half on each host it rents

  offer_policy:                # hard filters. Never relaxed unattended
    min_gpu_memory_gb: 24      # raised to what the models need where that is more
    min_disk_gb: 40            # the disk each host is rented with, and the least a machine must offer
    max_all_in_hourly: 0.30    # the most per host per hour, all-in with that disk — bids stop here
    max_download_per_gb: 0.01
    min_download_mbps: 100
    min_reliability: 0.95
    verified_only: true
    exclude_hardware: ["CMP"]  # mining cards pass every numeric filter and run nothing

  bidding:
    strategy: floor_plus_premium
    premium: 0.02              # absolute, not a multiplier: floors span an order of magnitude

  teardown:
    idle_minutes: 2            # idle time, not the hourly rate, is what actually costs you
    deadman_minutes: 20
    deadman_action: destroy
```

The models each rented machine holds, and the build of each, can be named as **model profiles**
(`rented.model_profiles` and `rent_profiles` — easiest from the console's *Rented capacity →
Engine & models* tab, which searches the model hub and Ollama's library for you). What a
profile holds sets the least card and disk searched for: the pool works it out from the
builds' sizes and raises the two minimums above where it has to, never lowering yours.

Check it before it takes effect — this is what the console's Apply button does for you:

```sh
uv run gpm config plan -f pool.yaml
```

Anything that *loosens* a limit — raising a ceiling, the burn cap or the host count — is
reported as needing the value typed again, and `gpm config apply` refuses it without `--yes`.
That is the guard against a misplaced decimal point.

## 3. Look at the market. This spends nothing

```sh
uv run gpm account       # is the credential valid, how much credit is left
uv run gpm market        # the real offer pipeline, read-only
```

`market` runs the same filters and the same bid strategy the supervisor would, and shows you
what passed, what did not, and why:

```
seen 100 | passed 4 | rejected by filter: {'all-in ceiling': 79, 'download price': 9, 'bid': 5}
  1x RTX A4500    floor 0.067  bid 0.079  on-demand 0.098  dl $0.0026/GB  rel 0.971
```

Move a ceiling in the console's form and watch "4 pass" become "0 pass". That is the fastest
way to learn what each number does, and it costs nothing.

## 4. Prepare one host, deliberately

Before trusting anything unattended, rent one host on purpose and watch it:

```sh
uv run gpm host prepare --max-spend 1.00 --max-hours 1 --when-ready join
```

Preparing is its own small lease with its own cap; it borrows authority from nothing else. It
bids, brings the instance up, arms the dead-man timer, opens a tunnel, downloads your model set,
verifies every model is resident **together**, sizes the workers, and then joins, parks or
destroys as you asked.

Watch it, then end it:

```sh
uv run gpm events          # every decision with the numbers behind it
uv run gpm status
uv run gpm down --all      # destroy everything rented, now, verified
```

`down --all` is the panic button. It is in the console's header on every screen.

## 5. Let it rent on its own

Now open a real lease. Overflow — demand your local and fixed hosts cannot absorb — is what
drives renting:

```sh
uv run gpm lease open --workers 8 --max-hours 4 --max-spend 5.00 --allow-rent
```

From here the supervisor will, unattended and within your caps: rent when overflow persists
(120 s by default, because a short burst is not worth a multi-gigabyte download), one host at a
time; recover when the market outbids you, choosing between re-bidding on the same machine and
moving elsewhere by which is cheaper over the hours the lease has left; release a host that has
been idle for ten minutes; and stop the lease before its dollar cap.

Close it when you are done — or let its time limit do it:

```sh
uv run gpm lease close <lease-id>
```

## 6. A second provider, beside the first

A pool may rent from several providers at once; each is a **provider account** under
`rented.providers`, at most one per provider:

```yaml
rented:
  providers:
    vast:   { type: vast }
    runpod: { type: runpod, keep_models: true }   # RUNPOD_API_KEY, or a credential typed in
```

Vast and RunPod ship with the framework; another provider is a plug-in (see
[writing-a-plugin.md](writing-a-plugin.md)).

- **RunPod** rents on demand only (it offers no spot), from its Secure and Community clouds
  (`settings: { clouds: [SECURE] }` for one). It does not report a machine's download speed or
  reliability, so its offers carry stated defaults (`assumed_download_mbps`,
  `assumed_reliability`), marked *assumed* wherever they are shown. Its hosts get the dead-man
  timer: a pod's own key, which RunPod puts in every pod, ends that pod and cannot list, create or
  read anything else of the account.
- **Keep models between hosts** (`keep_models: true`, or the checkbox on the account's card) is
  offered only where the provider's storage reaches a data center — RunPod's network volumes, on
  Secure Cloud. Each workload then keeps its models on a volume in its first host's data center,
  and later hosts copy from it in about a minute instead of downloading them; the volume is billed
  to the workload's lease and deleted when it ends. Hosts that start together never wait for it.

- **A credential can be typed into the console** — Rented capacity → Providers → Add provider, or
  *Type in a credential* on an account's card. It is tested, kept by the supervisor in an
  owner-only file, and never shown again. The supervisor's environment still works
  (`VAST_API_KEY`, or a connection's own `credential_env`), and a `credential_env` always wins.
- **Every search asks every enabled provider**, and every rental takes what is expected to cost
  least per worker-hour across all of them — a bid, a **spot** price (a provider's own
  interruptible price, paid as listed and never bid on), or on demand — counting what being
  taken away would cost.
- **A provider without a dead-man timer** serves only short leases
  (`teardown.max_hours_without_deadman`): a longer lease rents elsewhere.
- **Adding, turning on or off, and removing a provider take effect at once** — no restart; a change
  to an existing one's settings, or a rename, at the supervisor's next restart. The configuration plan asks
  you to type the name of a provider you turn on, and refuses to remove one the pool still holds
  hosts at — release them first.
- `enabled: false` stops searching and renting there; its hosts stay until released, and are
  still watched and charged.

## What protects you if everything goes wrong

| If | Then |
|---|---|
| The strategy returns a silly bid | The supervisor clamps it to both ceilings *after* it returns |
| The provider's charges run ahead of the pool's estimate | Caps are enforced on the higher figure, less the safety margin |
| The supervisor crashes | Routing continues; the **dead-man timer on each host** ends it after 20 minutes of no heartbeat *and* no inference |
| The whole machine dies | Same timer. Then lease expiry. Then the orphan sweep at next start |
| An instance exists that the pool never intended | The sweep destroys it — matched on the pool's own `gpm/<pool>/` label, so another tool's instances are left alone |
| A destroy silently fails | It is not counted as released until the provider's listing no longer shows it |
| You restart the supervisor | It adopts the hosts it already had instead of sweeping them |

## Costs worth knowing before your first bill

- **Idle time dominates.** A host you forgot about costs more than a slightly worse bid.
- **Downloads are charged per gigabyte** on most marketplaces, so a host that lives ten minutes
  can cost more in download than in compute. That is why ranking amortises download cost over
  the hours the lease expects, and why parking exists.
- **Parking bills storage, not compute.** The console shows the break-even: below it, keeping
  the disk is cheaper than downloading again.
- **A stopped instance still bills storage**, which is why the dead-man timer destroys rather
  than stops by default.

## The thing the design does not protect you from

The operator of a rented marketplace host can read everything sent to it — prompts and
completions included. A tunnel or TLS protects the wire, not the machine. If your data would
not be safe shown to a stranger, do not put marketplace hosts in that pool. See
[threat-model.md](threat-model.md) T7.
