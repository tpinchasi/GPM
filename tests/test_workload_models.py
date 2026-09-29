"""A workload's new host gets its models the fastest way the provider offers (D116,
workloads.md §6): a warm machine — a volume holding them, on a machine offered again — or a copy
from a ready sibling, or the hub. Each is a provider capability; a provider without it is never
asked. Against the fake provider, which offers both; nothing here spends money."""

import asyncio
import dataclasses
import time

import pytest
from gpm_server.providers import (
    FakeProvider,
    ProviderCapabilities,
    ProviderUnavailable,
    default_offer,
)
from gpm_server.providers.base import VolumeInfo
from test_workload_fleet import make_fleet, open_workload, run

CAPABLE = ProviderCapabilities(interruptible=True, parkable=True, same_machine_rebid=True, self_terminate=True,
                               reports_charges=True, volumes=True, copies=True)


def market(*, cheaper="m-0"):
    return [default_offer("o-0", "m-0", min_bid_hourly=0.10 if cheaper == "m-0" else 0.30, storage_per_gb_hourly=0.0003),
            default_offer("o-1", "m-1", min_bid_hourly=0.10 if cheaper == "m-1" else 0.30, storage_per_gb_hourly=0.0003)]


@pytest.fixture
def db(tmp_path):
    from gpm_server.db import Database

    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


def kinds(fleet):
    return [e["kind"] for e in fleet.events.recent(200)]


# --- a warm machine ---


async def test_the_first_host_keeps_its_models_on_a_volume(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    workload, lease = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = fleet.hosts_of("research")
    (volume,) = fleet.workload_store.volumes("research")
    assert host.volume_id == volume.volume_id and volume.machine_id == host.offer.machine_id
    assert volume.lease_id == lease.lease_id and volume.hourly > 0
    spec = provider.instances[host.instance.instance_id].spec.volume
    assert spec.mount == "/root/.ollama/models" and spec.label == "gpm/test/research/models"
    assert "volume_created" in kinds(fleet)


async def test_its_next_host_on_that_machine_fetches_nothing(db):
    provider = FakeProvider(offers=market(cheaper="m-0"), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (first,) = fleet.hosts_of("research")
    assert first.offer.machine_id == "m-0"
    await fleet.destroy(first, "evicted and not won back")
    provider.offers = market(cheaper="m-1")  # the other machine is cheaper now
    await run(fleet, [workload])
    (second,) = fleet.hosts_of("research")
    assert second.offer.machine_id == "m-0", "the machine holding its models comes first"
    assert second.volume_id == first.volume_id and second.models_source == "warm"
    assert len(provider.volumes) == 1, "one volume per workload"
    assert "models_from_volume" in kinds(fleet)


async def test_a_volumes_storage_is_on_the_workloads_lease(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    workload, lease = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (volume,) = fleet.workload_store.volumes("research")
    db.execute("UPDATE workload_volumes SET created_at = ? WHERE volume_id = ?", (time.time() - 3600, volume.volume_id))
    await run(fleet, [workload])
    totals = fleet.spend.latest_for_lease(lease.lease_id)
    rows = db.query("SELECT host_id, amount FROM spend WHERE host_id = ?", (f"volume:{volume.volume_id}",))
    assert rows and rows[-1]["amount"] == pytest.approx(volume.hourly, rel=0.05)
    assert totals["estimate"] >= rows[-1]["amount"]


async def test_a_volume_is_deleted_when_its_workload_is_over(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    await fleet.delete_volumes("research")
    assert provider.volumes == {} and fleet.workload_store.volumes("research") == []
    assert "volume_deleted" in kinds(fleet)


async def test_the_sweep_deletes_a_volume_nobody_owns_and_keeps_a_live_workloads(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    provider.volumes["v-stray"] = VolumeInfo("v-stray", "m-9", "gpm/test/old/models", 10)
    await run(fleet, [workload])
    assert "v-stray" not in provider.volumes
    assert len(provider.volumes) == 1, "the live workload's volume stays"


async def test_a_provider_that_cannot_list_volumes_has_none_deleted(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    provider.volumes["v-stray"] = VolumeInfo("v-stray", "m-9", "gpm/test/old/models", 10)

    async def refuse(prefix):
        raise ProviderUnavailable("down")

    provider.list_volumes = refuse
    await fleet.sweep_volumes()
    assert "v-stray" in provider.volumes


async def test_a_provider_without_volumes_is_never_asked_for_one(db):
    provider = FakeProvider(offers=market())
    fleet = make_fleet(db, provider)
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = fleet.hosts_of("research")
    assert host.volume_id is None and provider.instances[host.instance.instance_id].spec.volume is None


async def test_warm_machines_can_be_switched_off(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    fleet.config.workloads.keep_models_on_machine = False
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    assert provider.volumes == {}


# --- a copy from a sibling ---


async def two_hosts(db, **script):
    provider = FakeProvider(offers=market(), capabilities=dataclasses.replace(CAPABLE, volumes=False))
    for name, value in script.items():
        setattr(provider, name, value)
    fleet = make_fleet(db, provider)
    workload, _ = open_workload(fleet, hosts_at_start=2)
    await run(fleet, [workload])
    first, second = fleet.hosts_of("research")
    first.state = "ready"
    return fleet, provider, first, second


async def test_a_new_host_copies_its_models_from_a_ready_sibling(db):
    fleet, provider, first, second = await two_hosts(db)
    assert await fleet.models_from_sibling_pending(second) is True
    await asyncio.sleep(0)
    await second.copy_task
    assert await fleet.models_from_sibling_pending(second) is False
    assert second.models_source == "sibling"
    assert provider.copies == [(first.instance.instance_id, second.instance.instance_id, "/root/.ollama/models")]
    assert "models_copied" in kinds(fleet)


async def test_a_copy_that_fails_falls_through_to_the_hub(db):
    fleet, provider, _, second = await two_hosts(db, copy_fails=True)
    assert await fleet.models_from_sibling_pending(second) is True
    with pytest.raises(ProviderUnavailable):
        await second.copy_task
    assert await fleet.models_from_sibling_pending(second) is False
    assert second.models_source == "hub" and second.copy_state == "failed"
    assert "models_copy_failed" in kinds(fleet)


async def test_a_copy_to_a_host_that_goes_is_cancelled(db):
    fleet, provider, _, second = await two_hosts(db, copy_delay_s=5.0)
    assert await fleet.models_from_sibling_pending(second) is True
    await fleet.destroy(second, "given up")
    await asyncio.sleep(0)
    assert second.copy_task.cancelled()


async def test_without_a_ready_sibling_nothing_is_copied(db):
    fleet, provider, first, second = await two_hosts(db)
    first.state = "preparing"
    assert await fleet.models_from_sibling_pending(second) is False
    assert provider.copies == [] and second.copy_state == "none"


async def test_the_shared_workload_never_copies(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    fleet.leases.open(workers=4, max_hours=4, max_spend=2.0, allow_rent=True)
    await run(fleet, [], passes=2)
    (host,) = fleet.hosts_of(None)
    assert await fleet.models_from_sibling_pending(host) is False
    assert host.volume_id is None and provider.volumes == {}


async def test_a_copy_that_runs_out_of_time_gives_the_host_up(db):
    """Not waited for, and not known to have stopped: a fetch into the same place could race it."""
    fleet, provider, _, second = await two_hosts(db, copy_delay_s=5.0)
    fleet.config.workloads.copy_timeout_s = 0.05
    assert await fleet.models_from_sibling_pending(second) is True
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await second.copy_task
    assert await fleet.models_from_sibling_pending(second) is True, "nothing to fetch on this host"
    assert second.released and second.instance.instance_id not in provider.instances
    assert "models_copy_failed" in kinds(fleet)


def test_a_copy_timeout_must_leave_time_to_fetch():
    from gpm_server.config import PoolConfig
    from test_workload_fleet import make_fleet as _  # noqa: F401 - same config shape

    with pytest.raises(ValueError, match="copy_timeout_s"):
        PoolConfig.model_validate({
            "pool": {"name": "t", "model_set": ["m"]}, "auth": {"app_keys": ["k"]},
            "rented": {"provider": "fake", "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 1.0},
                       "teardown": {"max_preparing_minutes": 30}},
            "workloads": {"copy_timeout_s": 1200},
        })
