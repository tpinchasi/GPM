# The simulation

Everything in the test suite proves one behaviour against fakes that hold still: a market with
one machine at one price, traffic that arrives when a test sends it. The simulation proves the
*pool*, against neither — load that climbs, plateaus, spikes and stops, and a market whose
prices drift, whose machines come and go, and which takes hosts away mid-answer.

It runs the **real router and the real supervisor**, two halves over one database as in
production, against the fake provider and fake engines. No GPU, no cloud account, no money.

```sh
uv run pytest -m simulation                             # every scenario, as checks
uv run python tests/simulation/scenarios.py             # the same, with timelines printed
uv run python tests/simulation/scenarios.py market_day  # one of them
```

**It is a merge gate for the changes it exists to catch.** CI runs it as its own job on a pull
request that carries the `simulation` label — put it on any change to the supervisor's renting,
allocation or recovery — and on the weekly and on-request runs. It takes ten minutes, so it does
not run on every pull request.
A version is not merged until both are green.

## What it is not

It is not a benchmark, and it is not a recording. No scenario asserts that four hosts were
rented or that latency was under a second — those are facts about one run on one machine, and
asserting them buys flakiness instead of confidence. What each scenario asserts is a property
that must hold of *any* run.

## What every run must hold to

Checked after every scenario, whatever it was doing:

| Invariant | Why it matters |
|---|---|
| Every answer is either the engine's, or a `503` carrying a reason | An app may be told to wait. It may not be told something the contract does not describe |
| No response ends without its final frame | Half an answer already in an app's hands is the failure buffered delivery exists to prevent (D62) |
| No instance is left **running** | An accelerator still being paid for that nobody is watching. A *stopped* instance is a parked host — storage only, deliberate, given up at its own limit |
| Nothing is spent past a lease's cap | The lease is the only spending authority (D32) |

## The scenarios

Thirteen, each naming the shape it puts the pool through. Every one ends quiet, so the giving-up
path is exercised as much as the acquiring one.

| Scenario | What it puts under load |
|---|---|
| **market_day** | A full day: quiet, a morning climb, a peak far past what the pool holds, a shock where hosts are taken away mid-answer, a lull, a false dawn where load returns while hosts are paused, and a night. Exercises the ramp (D66), buffered delivery (D62), eviction recovery, pausing and waking (D64) and per-host worker adjustment (D67, D68) in one run |
| **load_under_capacity** | Steady load the local host already covers. **Nothing may be rented**, however long it goes on |
| **empty_market** | Load with every machine gone from the market, then the market returning. The pool says why it cannot buy, keeps serving from what it has, and rents the moment it can |
| **priced_out** | Every machine priced past the all-in maximum. Nothing is bought; every refusal carries its reason (D34) |
| **outbid_over_and_over** | An interruptible market at its worst: something outbid every two seconds, under load, for half a minute. No app ever sees half an answer |
| **hosts_that_never_answer** | Machines that come up and whose engine never answers. Given up rather than billed for the whole preparing window, and their machines avoided (D54) |
| **instances_without_their_script** | Instances that come up without their start-up material — no dead-man timer, no way in. Ended in the same pass (D65) |
| **provider_outage** | The provider stops answering mid-run. The pool keeps serving, keeps its hosts, assumes nothing about what exists, and picks up when it answers again (D61) |
| **supervisor_restart** | The supervisor process is replaced under live traffic. The router keeps serving from the published table, and the new supervisor **adopts** its hosts rather than sweeping them (D61) |
| **budget_runs_out** | A lease with barely any money. The cap bites, the lease is closed, and the hosts are given back |
| **hourly_burn_cap** | More load than `limits.max_hourly_burn` allows. Renting stops at the cap with the reason recorded, and the load is served by what is already running (D46) |
| **no_lease_at_all** | Load with no lease open. Nothing is rented however loud it gets (D32) |
| **operator_resizes_a_host** | `gpm host resize` under live traffic: down is instant and graceful, up to the launch bound restarts nothing, and no request is lost (D56, D68) |

## How a scenario is written

A scenario is a name, a sentence saying what it shows, and a list of phases. A phase says how
many callers there are, how fast they call and for how long; optionally it reaches into the
world as it begins — emptying the market, stopping the provider, restarting the supervisor.

```python
Scenario(
    name="empty_market",
    what_it_shows="a pool that wants capacity and cannot buy any",
    lease={"workers": 40, "max_hours": 1, "max_spend": 5.00, "allow_rent": True},
    phases=(
        Phase("load, with nothing to buy", 14, callers=20, when_it_starts=market_goes_empty),
        Phase("the market returns", 14, callers=20, when_it_starts=market_comes_back),
        Phase("quiet", 20, callers=0),
    ),
    expects=lambda s: _at_least(s, "rented", 1, "never rented once the market came back"),
)
```

Windows are turned right down — a two-minute idle limit becomes six seconds — so a day passes
in a minute. What is under test is the *shape* of the pool's behaviour, not how long it waits.

## What it has already found

Three faults that every unit test had passed over, because each needed a whole pool running for
long enough:

1. **Parking never saved anything.** A parked host's engine is stopped, so the probe found
   nothing answering and marked the host `preparing`; the eviction handler then read "stopped,
   and we did not ask" as an eviction and destroyed it seconds after it was parked. The point
   of parking — keeping the models to avoid a download — had never once been realised.
2. **A pool at its host limit could not wake what it already had.** Paused hosts count toward
   `max_rented_hosts`, and the path that wakes them sat *behind* the cap check — so a pool with
   every host paused refused returning load instead of waking a host in seconds, with no
   download.
3. **A host that could not take the agent was asked on every pass**, costing an SSH round trip
   each time and writing 153 events for two hosts.

None of these lost a request, which is why no test caught them; all three cost money.
