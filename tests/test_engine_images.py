"""The build of the engine is chosen per machine, from what its driver can run (D92).

An engine is commonly published once per accelerator generation — a newer build is smaller and
faster but needs a newer driver. A pool with one image must either refuse every older machine or
buy one and fail on it. Listed newest first, the pool takes the best that fits.

Against the fake provider; nothing here spends money.
"""

import pytest
from gpm_server.config import EngineImage, PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.strategies import image_for
from gpm_server.supervisor.renting import Fleet

MODEL = "m1"
NEW = EngineImage(image="vastai/vllm:v0.29.0-cuda-13.0", min_driver="580", note="newest")
OLD = EngineImage(image="vastai/vllm:v0.29.0-cuda-12.9", min_driver="550")


def config(images=(), image=None):
    rented = {
        "provider": "fake", "workers": 2, "capabilities": ["cuda"],
        "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 2.0}, "bidding": {"premium": 0.02},
        "scale": {"scale_up_after_s": 0},
    }
    if images:
        rented["images"] = [i.model_dump() for i in images]
    if image:
        rented["image"] = image
    # A vLLM pool, because that is the engine published once per accelerator generation. It
    # serves one model per process, so its set is spread across hosts (D89).
    engine = "vllm" if (images or (image and "vllm" in image)) else "ollama"
    pool = {"name": "test", "model_set": [MODEL]}
    if engine == "vllm":
        pool["models_per_host"] = "declared"
    return PoolConfig.model_validate({
        "pool": pool,
        "auth": {"app_keys": ["k"]},
        "engine": engine,
        "hosts": [{"id": "local-1", "kind": "local", "workers": 1, "models": [MODEL],
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}]
                 if engine == "vllm" else
                 [{"id": "local-1", "kind": "local", "workers": 1,
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "rented": rented,
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


# --- choosing ---


@pytest.mark.parametrize("driver,expected", [
    ("595.84", NEW.image),
    ("580.65", NEW.image),
    ("580", NEW.image),
    ("570.1", OLD.image),
    ("550.144", OLD.image),
    ("550", OLD.image),
])
def test_the_best_build_the_machine_can_run_is_chosen(driver, expected):
    chosen = image_for(default_offer(driver_version=driver), [NEW, OLD])
    assert chosen is not None and chosen.image == expected


def test_a_machine_too_old_for_every_build_gets_none():
    assert image_for(default_offer(driver_version="535.0"), [NEW, OLD]) is None


def test_a_machine_that_does_not_say_is_not_given_the_benefit_of_the_doubt():
    """The same reasoning as the driver floor itself (D81): an unknown driver on an 80 GB card
    turned out to be an engine serving from the processor at an accelerator's price."""
    assert image_for(default_offer(driver_version=None), [NEW, OLD]) is None
    assert image_for(default_offer(driver_version="unknown"), [NEW, OLD]) is None


def test_order_is_preference_so_the_list_is_read_newest_first():
    reversed_order = image_for(default_offer(driver_version="595.84"), [OLD, NEW])
    assert reversed_order is not None and reversed_order.image == OLD.image


# --- what the pool does with it ---


def test_a_pool_with_one_image_behaves_exactly_as_before(make_fleet):
    fleet = make_fleet(config(image="ollama/ollama:0.34.2"), [])
    chosen, why = fleet.image_for(default_offer(driver_version="535.0"))
    assert chosen == "ollama/ollama:0.34.2" and "only image" in why


def test_the_machine_is_rented_with_the_build_it_can_run(make_fleet):
    fleet = make_fleet(config(images=[NEW, OLD]), [default_offer(driver_version="560.1")])
    fleet.leases.open(workers=4, max_hours=2, max_spend=5.0, allow_rent=True)


async def test_the_instance_is_created_with_the_chosen_build(make_fleet):
    fleet = make_fleet(config(images=[NEW, OLD]), [default_offer(driver_version="560.1")])
    fleet.leases.open(workers=4, max_hours=2, max_spend=5.0, allow_rent=True)

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (spec,) = [i.spec for i in fleet.provider.instances.values()]
    assert spec.image == OLD.image


async def test_a_newer_machine_gets_the_newer_build(make_fleet):
    fleet = make_fleet(config(images=[NEW, OLD]), [default_offer(driver_version="590.0")])
    fleet.leases.open(workers=4, max_hours=2, max_spend=5.0, allow_rent=True)

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    (spec,) = [i.spec for i in fleet.provider.instances.values()]
    assert spec.image == NEW.image


async def test_a_machine_no_build_runs_on_is_never_bid_on(make_fleet):
    """This is the whole point: renting it would buy a host that can never answer and pay for
    it until the give-up window closes. The refusal names the driver and what was wanted."""
    fleet = make_fleet(config(images=[NEW, OLD]), [default_offer(driver_version="535.0")])
    fleet.leases.open(workers=4, max_hours=2, max_spend=5.0, allow_rent=True)

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert not fleet.provider.instances, "a machine no build runs on was rented"
    refusal = next(e for e in fleet.events.recent() if e["kind"] == "offer_refused")
    assert "535.0" in refusal["summary"] and "580" in refusal["summary"]


async def test_nothing_is_spent_when_no_machine_can_run_the_engine(make_fleet):
    fleet = make_fleet(config(images=[NEW]), [
        default_offer("o-1", "m-1", driver_version="550.1"),
        default_offer("o-2", "m-2", driver_version=None),
    ])
    fleet.leases.open(workers=4, max_hours=2, max_spend=5.0, allow_rent=True)

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert not fleet.provider.instances
    assert not fleet.hosts


# --- configuration ---


def test_a_build_must_say_which_driver_it_needs():
    with pytest.raises(ValueError):
        PoolConfig.model_validate({
            "pool": {"name": "t", "model_set": [MODEL], "models_per_host": "declared"},
            "auth": {"app_keys": ["k"]},
            "hosts": [{"id": "h", "kind": "local", "workers": 1, "models": [MODEL],
                       "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            "engine": "vllm",
            "rented": {"provider": "fake", "workers": 1, "capabilities": ["cuda"],
                       "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 1.0}, "bidding": {"premium": 0.0},
                       "images": [{"image": "vastai/vllm:v0.29.0-cuda-13.0"}]},
        })


def test_the_images_are_read_back_in_the_order_given():
    cfg = config(images=[NEW, OLD])
    assert [i.image for i in cfg.rented.images] == [NEW.image, OLD.image]
    assert cfg.rented.images[0].note == "newest"
