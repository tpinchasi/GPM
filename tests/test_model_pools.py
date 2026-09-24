"""A host per model, and the pool covering its set across them (D94).

The second of the two shapes the owner asked for: where an engine serves one model per process,
the pool buys a machine per model rather than asking one machine for the lot. Each rented host
is bought **for** a model — whichever the pool is shortest of — and prepared only for that one.

Against the fake provider; nothing here spends money.
"""

import pytest
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor.renting import Fleet

BIG, SMALL, EMBED = "gemma4:26b", "gemma4:e4b", "nomic-embed-text"

CATALOG = {
    BIG: {"variants": [{"tag": "big-ollama", "engine": "ollama"}, {"tag": "big-vllm", "engine": "vllm"}]},
    SMALL: {"variants": [{"tag": "small-ollama", "engine": "ollama"}, {"tag": "small-vllm", "engine": "vllm"}]},
    EMBED: {"variants": [{"tag": "embed-ollama", "engine": "ollama"}, {"tag": "embed-vllm", "engine": "vllm"}]},
}


def config(rented_models=(BIG, SMALL), engine="vllm", per_host="declared"):
    laptop = {
        "id": "laptop", "kind": "local", "workers": 2,
        "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"},
    }
    if per_host != "all":
        # Naming models is meaningless — and refused — where every host holds the whole set.
        laptop["models"] = [EMBED]
    return PoolConfig.model_validate({
        "pool": {"name": "t", "model_set": [BIG, SMALL, EMBED], "models_per_host": per_host},
        "auth": {"app_keys": ["k"]},
        "engine": "ollama",
        "catalog": CATALOG,
        "hosts": [laptop],
        "rented": {
            "provider": "fake", "workers": 2, "capabilities": ["cuda"], "engine": engine,
            # Likewise: naming what rented hosts hold is refused where they hold everything.
            **({} if per_host == "all" else {"models": list(rented_models)}),
            "image": f"vastai/{engine}:latest",
            "bidding": {"bid_ceiling": 2.0, "premium": 0.02},
            "scale": {"scale_up_after_s": 0},
        },
    })


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


def offers(count: int):
    return [default_offer(f"o-{i}", f"m-{i}", driver_version="590.0") for i in range(count)]


def _serving(fleet, host_id: str, model: str):
    """A host already in the fleet, bought for one model — what a later pass would see."""
    from gpm_server.supervisor.renting import RentedHost

    return RentedHost(
        host_id=host_id, instance=None, offer=default_offer(), bid_hourly=1.0,
        lease_id="lease-1", models=(model,), state="ready",
    )


# --- one model to a host ---


def test_an_engine_holding_one_model_buys_one_model_to_a_host(make_fleet):
    fleet = make_fleet(config(), [])
    assert fleet.one_model_per_host is True
    assert len(fleet.models_for_new_host()) == 1


def test_an_engine_holding_several_buys_a_host_for_all_of_them(make_fleet):
    """Nothing changes for the engine the pool shipped first."""
    fleet = make_fleet(config(engine="ollama"), [])
    assert fleet.one_model_per_host is False
    assert set(fleet.models_for_new_host()) == {BIG, SMALL}


def test_with_the_whole_set_on_every_host_nothing_is_assigned(make_fleet):
    fleet = make_fleet(config(engine="ollama", per_host="all"), [])
    assert fleet.one_model_per_host is False
    assert set(fleet.models_for_new_host()) == {BIG, SMALL, EMBED}


# --- the pool covers its set rather than buying the same model twice ---


async def test_successive_hosts_are_bought_for_different_models(make_fleet):
    """Without this every host would be bought for the same model, and the rest of the set would
    never be covered however much was spent. Driven directly, because a second purchase is also
    gated on the first landing and that is a different rule being tested elsewhere."""
    fleet = make_fleet(config(), offers(1))
    fleet.leases.open(workers=8, max_hours=4, max_spend=20.0, allow_rent=True)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (first,) = fleet.hosts.values()
    assert first.models == (BIG,)
    assert fleet.models_for_new_host() == (SMALL,), "the model with no host yet"


def test_the_shortest_model_is_the_one_bought_for(make_fleet):
    fleet = make_fleet(config(), [])

    assert fleet.models_for_new_host() == (BIG,), "nothing served yet: the first listed"

    fleet.hosts["h1"] = _serving(fleet, "h1", BIG)
    assert fleet.models_for_new_host() == (SMALL,), "the one with no host"

    fleet.hosts["h2"] = _serving(fleet, "h2", SMALL)
    # Both covered once: the tie goes to the order the operator listed them.
    assert fleet.models_for_new_host() == (BIG,)

    fleet.hosts["h3"] = _serving(fleet, "h3", BIG)
    assert fleet.models_for_new_host() == (SMALL,), "two against one"


def test_a_released_host_stops_counting_towards_its_model(make_fleet):
    fleet = make_fleet(config(), [])
    fleet.hosts["h1"] = _serving(fleet, "h1", BIG)
    assert fleet.models_for_new_host() == (SMALL,)

    fleet.hosts["h1"].released = True
    assert fleet.models_for_new_host() == (BIG,), "its model is short again"


# --- a host is prepared only for what it was bought for ---


async def test_a_host_is_prepared_for_its_own_model_and_no_other(make_fleet):
    """Asking a one-model engine for the whole set is how a machine is paid for and never
    becomes ready."""
    fleet = make_fleet(config(), offers(1))
    fleet.leases.open(workers=4, max_hours=2, max_spend=10.0, allow_rent=True)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (host,) = fleet.hosts.values()
    assert fleet.tags_for(host) == frozenset({"big-vllm"})
    # What the pool may be asked for across all its rented hosts is the union, which is a
    # different question and must not be what any one machine is prepared for.
    assert fleet.required_tags == frozenset({"big-vllm", "small-vllm"})


async def test_a_host_rented_before_models_were_assigned_keeps_the_whole_rented_set(make_fleet):
    """A supervisor upgraded under a running pool must not suddenly under-prepare a host it
    already owns."""
    fleet = make_fleet(config(engine="ollama"), offers(1))
    fleet.leases.open(workers=4, max_hours=2, max_spend=10.0, allow_rent=True)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (host,) = fleet.hosts.values()
    host.models = ()
    assert fleet.models_of(host) == [BIG, SMALL]
    assert fleet.tags_for(host) == frozenset({"big-ollama", "small-ollama"})


# --- what the operator is told ---


async def test_the_purchase_says_what_the_machine_was_bought_to_serve(make_fleet):
    fleet = make_fleet(config(), offers(1))
    fleet.leases.open(workers=4, max_hours=2, max_spend=10.0, allow_rent=True)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (host,) = fleet.hosts.values()
    assert host.models == (BIG,)


# --- buying for the model that is actually waiting (D95) ---


def test_a_model_nothing_serves_is_bought_for_first(make_fleet):
    """Availability before capacity: a model with no host cannot be served at all, and no
    amount of throughput elsewhere makes up for it."""
    fleet = make_fleet(config(), [])
    fleet.hosts["h1"] = _serving(fleet, "h1", BIG)
    fleet._waiting_by_model = {BIG: 50}   # the covered model is the busy one

    assert fleet.models_for_new_host() == (SMALL,), "the uncovered model still wins"


def test_once_every_model_is_covered_the_busiest_is_bought_for(make_fleet):
    """Coverage alone would keep adding hosts to a model nobody is asking for."""
    fleet = make_fleet(config(), [])
    fleet.hosts["h1"] = _serving(fleet, "h1", BIG)
    fleet.hosts["h2"] = _serving(fleet, "h2", SMALL)
    fleet.hosts["h3"] = _serving(fleet, "h3", SMALL)
    fleet._waiting_by_model = {SMALL: 40, BIG: 2}

    # By coverage alone this would be BIG, which has fewer hosts. The waiting says otherwise.
    assert fleet.models_for_new_host() == (SMALL,)


def test_with_nothing_waiting_it_falls_back_to_coverage(make_fleet):
    fleet = make_fleet(config(), [])
    fleet.hosts["h1"] = _serving(fleet, "h1", BIG)
    fleet.hosts["h2"] = _serving(fleet, "h2", BIG)
    fleet.hosts["h3"] = _serving(fleet, "h3", SMALL)
    fleet._waiting_by_model = {}

    assert fleet.models_for_new_host() == (SMALL,)


# --- the last host of a model is never torn down (D95) ---


def test_the_only_host_serving_a_model_is_kept(make_fleet):
    """Every request for it would be refused until another machine was bought and prepared —
    minutes at best. A host doing nothing is cheaper than a model that cannot be served."""
    fleet = make_fleet(config(), [])
    only = _serving(fleet, "h1", BIG)
    fleet.hosts["h1"] = only

    assert fleet.last_host_serving(only) is True


def test_a_host_with_a_twin_may_go(make_fleet):
    fleet = make_fleet(config(), [])
    first = _serving(fleet, "h1", BIG)
    fleet.hosts["h1"] = first
    fleet.hosts["h2"] = _serving(fleet, "h2", BIG)

    assert fleet.last_host_serving(first) is False


def test_a_host_on_its_way_out_does_not_count_as_cover(make_fleet):
    fleet = make_fleet(config(), [])
    first = _serving(fleet, "h1", BIG)
    fleet.hosts["h1"] = first
    twin = _serving(fleet, "h2", BIG)
    twin.released = True
    fleet.hosts["h2"] = twin

    assert fleet.last_host_serving(first) is True


def test_a_host_still_preparing_does_count_as_cover(make_fleet):
    """It was bought for that model and is on its way; holding the old one until it lands is
    the difference between a gap and no gap."""
    fleet = make_fleet(config(), [])
    first = _serving(fleet, "h1", BIG)
    fleet.hosts["h1"] = first
    coming = _serving(fleet, "h2", BIG)
    coming.state = "preparing"
    fleet.hosts["h2"] = coming

    assert fleet.last_host_serving(first) is False


def test_where_every_host_holds_the_whole_set_the_question_does_not_arise(make_fleet):
    """Any remaining host still serves everything, so nothing needs protecting."""
    fleet = make_fleet(config(engine="ollama", per_host="all"), [])
    host = _serving(fleet, "h1", BIG)
    fleet.hosts["h1"] = host

    assert fleet.last_host_serving(host) is False
