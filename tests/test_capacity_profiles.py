"""Capacity profiles: how many workers a rented host runs (D45).

docs/spec/hosts-routing-capacity.md §2.1. A profile is matched on the offer *before* the bid, so
the engine is launched with the same parallelism the pool will count on — the number is real,
not just a count of queue slots. Against the fake provider; nothing here spends money.
"""

import pytest
from gpm_server.config import PoolConfig
from gpm_server.configplan import plan_changes
from gpm_server.db import Database, HostRow
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor.renting import Fleet

MODEL = "m1"
MAX_Q = "1x RTX PRO 6000 Max-Q"


def config(profiles=(), rented_workers=2):
    return PoolConfig.model_validate({
        "pool": {"name": "test", "model_set": [MODEL]},
        "auth": {"app_keys": ["k"]},
        "hosts": [{"id": "local-1", "kind": "local", "workers": 1,
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "capacity_profiles": list(profiles),
        "rented": {
            "provider": "fake", "workers": rented_workers, "model_set_gb": 10.0,
            "capabilities": ["cuda"],
            "bidding": {"bid_ceiling": 0.60, "premium": 0.02},
            "scale": {"scale_up_after_s": 0},
        },
    })


PROFILE = {"match": {"hardware": MAX_Q}, "max_workers": 6,
           "note": "2,765 requests at 6 in flight, latency flat from 4 to 6"}


@pytest.fixture
def make_fleet(tmp_path):
    databases = []

    def build(cfg, offers):
        database = Database(tmp_path / f"gpm{len(databases)}.sqlite3")
        databases.append(database)
        return Fleet(cfg, cfg.rented, FakeProvider(offers=offers),
                     LeaseStore(database), EventLog(database), SpendLedger(database))

    yield build
    for database in databases:
        database.close()


# --- matching ---


def test_a_matching_offer_runs_the_profiles_workers_and_says_why(make_fleet):
    fleet = make_fleet(config([PROFILE]), [])
    workers, why = fleet.workers_for(default_offer(hardware=MAX_Q, gpu_memory_gb=95.6))
    assert workers == 6 and "2,765 requests" in why


def test_the_hardware_name_is_matched_whole_and_ignoring_case(make_fleet):
    fleet = make_fleet(config([PROFILE]), [])
    assert fleet.workers_for(default_offer(hardware="1x rtx pro 6000 max-q"))[0] == 6
    # Two of the same card are a different machine, and must not borrow one card's number.
    assert fleet.workers_for(default_offer(hardware="2x RTX PRO 6000 Max-Q"))[0] == 2
    assert fleet.workers_for(default_offer(hardware="1x RTX PRO 6000 WS"))[0] == 2


def test_no_matching_profile_falls_back_to_the_rented_default_and_says_so(make_fleet):
    workers, why = make_fleet(config([PROFILE]), []).workers_for(default_offer(hardware="1x A100 PCIE"))
    assert workers == 2 and "no capacity profile" in why


def test_the_first_matching_profile_wins(make_fleet):
    broad = {"match": {"min_gpu_memory_gb": 80}, "max_workers": 4}
    fleet = make_fleet(config([PROFILE, broad]), [])
    assert fleet.workers_for(default_offer(hardware=MAX_Q, gpu_memory_gb=95.6))[0] == 6
    assert fleet.workers_for(default_offer(hardware="1x H100", gpu_memory_gb=94))[0] == 4
    reversed_order = make_fleet(config([broad, PROFILE]), [])
    assert reversed_order.workers_for(default_offer(hardware=MAX_Q, gpu_memory_gb=95.6))[0] == 4


# --- the number is real: the engine is launched to match, and it survives a restart ---


async def test_a_host_rented_under_a_profile_is_launched_with_that_parallelism(make_fleet):
    fleet = make_fleet(config([PROFILE]), [default_offer(hardware=MAX_Q, gpu_memory_gb=95.6)])
    fleet.leases.open(workers=12, max_hours=4, max_spend=5.0, allow_rent=True)

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (host,) = fleet.hosts.values()
    assert host.workers == 6
    (spec,) = [i.spec for i in fleet.provider.instances.values()]
    assert spec.env["OLLAMA_NUM_PARALLEL"] == "6"
    rented = next(e for e in fleet.events.recent() if e["kind"] == "rented")
    assert rented["numbers"]["workers"] == 6 and "6 workers" in rented["summary"]


async def test_a_host_outside_every_profile_keeps_the_rented_default(make_fleet):
    fleet = make_fleet(config([PROFILE]), [default_offer(hardware="1x A100 PCIE")])
    fleet.leases.open(workers=12, max_hours=4, max_spend=5.0, allow_rent=True)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()
    assert host.workers == 2
    (spec,) = [i.spec for i in fleet.provider.instances.values()]
    assert spec.env["OLLAMA_NUM_PARALLEL"] == "2"


async def test_a_restarted_supervisor_restores_each_hosts_own_count(make_fleet):
    """Its engine was launched with that parallelism, whatever the profiles say by now."""
    first = make_fleet(config([PROFILE]), [default_offer(hardware=MAX_Q, gpu_memory_gb=95.6)])
    first.leases.open(workers=12, max_hours=4, max_spend=5.0, allow_rent=True)
    await first.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = first.hosts.values()
    row = HostRow(
        host_id=host.host_id, kind="rented-interruptible", transport_type="http", priority=20,
        dial_url="http://x", state="ready", workers=host.workers, capabilities=("cuda",),
        variants={}, resident=frozenset(), lease_id=host.lease_id,
        provider_ref=first.published_ref(host), hourly_rate=host.bid_hourly,
    )

    # A successor with no profile at all, over the same provider.
    successor = make_fleet(config([]), [])
    successor.provider = first.provider
    await successor.adopt([row])
    assert successor.hosts[host.host_id].workers == 6


# --- configuration ---


def test_a_profile_change_is_explained_by_plan_and_needs_no_retype():
    changes = [c for c in plan_changes(config([]), config([PROFILE])) if c.kind == "capacity_profile"]
    assert len(changes) == 1
    assert "hardware=1x RTX PRO 6000 Max-Q" in changes[0].detail and "6 workers" in changes[0].detail
    assert "a running host keeps" in changes[0].detail
    assert changes[0].requires_retype is None  # more workers on the same host costs nothing more

    raised = {**PROFILE, "max_workers": 8}
    (change,) = [c for c in plan_changes(config([PROFILE]), config([raised])) if c.kind == "capacity_profile"]
    assert "from 6 to 8 workers" in change.detail


@pytest.mark.parametrize("bad", [
    {"match": {"hardware": MAX_Q}, "max_workers": 0},
    {"match": {"hardware": MAX_Q}, "max_workers": 999},
    {"match": {"hardwar": MAX_Q}, "max_workers": 6},   # a typo is refused, not silently ignored
    {"match": {"hardware": MAX_Q}},
])
def test_a_profile_that_makes_no_sense_is_refused_at_load(bad):
    with pytest.raises(ValueError):
        config([bad])


async def test_the_market_preview_says_what_each_offer_would_run(make_fleet):
    fleet = make_fleet(config([PROFILE]), [
        default_offer("o-1", "m-1", hardware=MAX_Q, gpu_memory_gb=95.6),
        default_offer("o-2", "m-2", hardware="1x A100 PCIE"),
    ])
    preview = await fleet.market_preview(hours=1)
    by_hardware = {o["hardware"]: o for o in preview["best"]}
    assert by_hardware[MAX_Q]["workers"] == 6 and "capacity profile" in by_hardware[MAX_Q]["workers_from"]
    assert by_hardware["1x A100 PCIE"]["workers"] == 2


def test_a_profile_can_name_the_card_whatever_the_machine_holds_of_it(make_fleet):
    """Found live (D88): the owner set "9 workers for RTX PRO 6000 WS", the pool kept running
    six, and nothing said why. The market lists hardware as "1x RTX PRO 6000 WS" and a profile's
    `hardware` is compared **whole**, so a profile naming the card alone silently never fires.
    What a card runs at once is a fact about the card, not about how many are in the box."""
    fleet = make_fleet(config([
        {"match": {"gpu": "RTX PRO 6000 WS"}, "max_workers": 9},
        {"match": {"gpu": "RTX PRO 6000 S"}, "max_workers": 6},
    ]), [])

    assert fleet.workers_for(default_offer(hardware="1x RTX PRO 6000 WS", gpus=1))[0] == 9
    assert fleet.workers_for(default_offer(hardware="1x RTX PRO 6000 S", gpus=1))[0] == 6
    # Nothing it does not name falls through to the rented default, as before.
    count, why = fleet.workers_for(default_offer(hardware="1x A100 SXM4"))
    assert "no capacity profile" in why


def test_a_whole_hardware_match_still_distinguishes_the_count(make_fleet):
    """`hardware` keeps its old meaning: a 2-card machine is a different profile if you say so."""
    fleet = make_fleet(config([
        {"match": {"hardware": "2x RTX PRO 6000 WS"}, "max_workers": 12},
        {"match": {"gpu": "RTX PRO 6000 WS"}, "max_workers": 9},
    ]), [])

    assert fleet.workers_for(default_offer(hardware="2x RTX PRO 6000 WS"))[0] == 12
    assert fleet.workers_for(default_offer(hardware="1x RTX PRO 6000 WS"))[0] == 9


def test_a_per_card_profile_is_multiplied_by_the_cards_the_machine_has(make_fleet):
    """The owner, on seeing the card match: "if I rent 2x RTX PRO 6000 WS, I expect 2x the
    worker count that 1x gets." Matching by card means the number is per card — and a second
    card left idle is exactly what its price was not paid for."""
    fleet = make_fleet(config([{"match": {"gpu": "RTX PRO 6000 WS"}, "max_workers": 7}]), [])

    one, why = fleet.workers_for(default_offer(hardware="1x RTX PRO 6000 WS", gpus=1))
    two, why_two = fleet.workers_for(default_offer(hardware="2x RTX PRO 6000 WS", gpus=2))
    four, _ = fleet.workers_for(default_offer(hardware="4x RTX PRO 6000 WS", gpus=4))

    assert (one, two, four) == (7, 14, 28)
    assert "7 per card x 2 card(s)" in why_two, why_two


def test_a_whole_hardware_profile_is_the_machine_total_not_per_card(make_fleet):
    """`hardware` keeps its old meaning exactly: the number is what that machine runs."""
    fleet = make_fleet(config([{"match": {"hardware": "2x RTX PRO 6000 WS"}, "max_workers": 10}]), [])

    assert fleet.workers_for(default_offer(hardware="2x RTX PRO 6000 WS", gpus=2))[0] == 10


def test_a_machine_with_many_cards_is_still_held_to_something_measured(make_fleet):
    """The arithmetic is sound and an eight-card machine would otherwise ask an engine for a
    number nobody has measured it at."""
    fleet = make_fleet(config([{"match": {"gpu": "H200"}, "max_workers": 12}]), [])

    workers, why = fleet.workers_for(default_offer(hardware="8x H200", gpus=8))
    assert workers == 64 and "held at 64" in why
