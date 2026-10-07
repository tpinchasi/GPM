"""A workload's new host gets its models the fastest way the provider offers (D116, D139,
workloads.md §6): a model volume in its data center, filled by one host and copied from by the
rest — or a copy from a ready sibling, or the hub. Each is a provider capability; a provider without it is never
asked. Against the fake provider, which offers both; nothing here spends money."""

import asyncio
import dataclasses
import time

import pytest
from gpm_agent import volume as model_volume
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


# --- a model volume, in a data center (D139) ---

KEEPS = dataclasses.replace(CAPABLE, volume_reach="data_center")
RATE = 0.0001  # per GB-hour


def dc_market(*, here=0.10, elsewhere=0.30, where=("EU-1",)):
    """One offer in the volume's data center, one that can land only elsewhere."""
    return [default_offer("o-0", "m-0", min_bid_hourly=here, locations=where, volume_per_gb_hourly=RATE),
            default_offer("o-1", "m-1", min_bid_hourly=elsewhere, locations=("US-9",), volume_per_gb_hourly=RATE)]


def keeping(db, offers=None, capabilities=KEEPS, keep=True):
    provider = FakeProvider(offers=offers or dc_market(), capabilities=capabilities)
    fleet = make_fleet(db, provider)
    fleet.rented.providers["fake"].keep_models = keep
    return provider, fleet


def agent_says(fleet, host, **report):
    """The host's agent's facts, as they reach the pool, with what it did with its volume."""
    host.agent_facts = {"engine": {"model_sources": {tag: dict(report) for tag in host.builds.values()}}}
    fleet._volume_news(host)


def volume_of(fleet):
    (volume,) = fleet.workload_store.volumes("research")
    return volume


async def first_host(db, **kwargs):
    provider, fleet = keeping(db, **kwargs)
    workload, lease = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = fleet.hosts_of("research")
    return provider, fleet, workload, lease, host


async def test_the_first_host_makes_the_volume_in_its_data_center_and_fills_it(db):
    provider, fleet, _, lease, host = await first_host(db)
    volume = volume_of(fleet)
    assert host.volume_id == volume.volume_id and volume.location == "EU-1"
    assert volume.state == "filling" and volume.filler == host.host_id
    assert volume.lease_id == lease.lease_id and volume.hourly == pytest.approx(volume.size_gb * RATE)
    asked = provider.instances[host.instance.instance_id].spec.volume
    assert asked.mount == str(model_volume.FILL_PATH), "mounted at the agent's own path to fill"
    assert asked.label == "gpm/test/research/models"
    assert "volume_created" in kinds(fleet)


async def test_once_its_agent_says_it_filled_it_the_volume_is_ready(db):
    _, fleet, _, _, host = await first_host(db)
    agent_says(fleet, host, fill="filling")  # serving, and filling behind
    assert volume_of(fleet).state == "filling" and not host.volume_reported, "not ready while it fills"
    agent_says(fleet, host, fill="filled")
    assert volume_of(fleet).state == "ready" and volume_of(fleet).filler is None
    assert "volume_filled" in kinds(fleet)


async def test_a_later_host_copies_from_it_and_the_pool_records_how_it_went(db):
    provider, fleet, workload, _, first = await first_host(db)
    agent_says(fleet, first, fill="filled")
    await fleet.destroy(first, "evicted")
    await run(fleet, [workload])
    (second,) = fleet.hosts_of("research")
    asked = provider.instances[second.instance.instance_id].spec.volume
    assert asked.mount == str(model_volume.READ_PATH) and asked.volume_id == first.volume_id
    assert second.models_source == "volume" and len(provider.volumes) == 1
    agent_says(fleet, second, files_from_volume=3, bytes_from_volume=20_000_000_000, files_mismatched=0,
               files_missing=0, seconds_from_volume=41.0)
    (event,) = [e for e in fleet.events.recent(200) if e["kind"] == "models_from_volume"]
    assert "41 s" in event["summary"] and event["numbers"]["bytes"] == 20_000_000_000


async def test_files_that_did_not_match_are_said_and_a_volume_without_the_build_is_stale(db):
    _, fleet, workload, _, first = await first_host(db)
    agent_says(fleet, first, fill="filled")
    await fleet.destroy(first, "evicted")
    await run(fleet, [workload])
    (second,) = fleet.hosts_of("research")
    agent_says(fleet, second, files_from_volume=0, files_mismatched=2, files_missing=1)
    assert "volume_mismatch" in kinds(fleet)
    assert volume_of(fleet).state == "stale", "the next host there fills the new build"


async def test_a_filler_that_goes_without_filling_leaves_it_for_the_next_host_to_fill(db):
    provider, fleet, workload, _, first = await first_host(db)
    await fleet.destroy(first, "evicted before it finished")
    await run(fleet, [workload])
    (second,) = fleet.hosts_of("research")
    asked = provider.instances[second.instance.instance_id].spec.volume
    assert asked.mount == str(model_volume.FILL_PATH) and asked.volume_id == first.volume_id
    assert volume_of(fleet).filler == second.host_id and "volume_not_filled" in kinds(fleet)


async def test_a_fill_the_agent_could_not_do_empties_the_volume(db):
    _, fleet, _, _, host = await first_host(db)
    agent_says(fleet, host, fill="not filled: config.json on this host does not match the hub")
    assert volume_of(fleet).state == "empty" and "volume_not_filled" in kinds(fleet)


async def test_while_one_host_fills_it_no_other_is_given_it(db):
    provider, fleet = keeping(db)
    workload, _ = open_workload(fleet, hosts_at_start=2)
    await run(fleet, [workload])
    first, second = fleet.hosts_of("research")
    assert first.volume_id is not None and second.volume_id is None, "one writer at a time"


async def test_the_volumes_data_center_is_preferred_only_while_the_download_it_saves_is_worth_it(db):
    _, fleet, workload, _, first = await first_host(db, offers=dc_market(here=0.10, elsewhere=0.30))
    agent_says(fleet, first, fill="filled")
    await fleet.destroy(first, "done")
    fleet.provider.offers = dc_market(here=0.101, elsewhere=0.10)  # a tenth of a cent dearer: worth it
    await run(fleet, [workload])
    (second,) = fleet.hosts_of("research")
    assert second.offer.machine_id == "m-0"
    await fleet.destroy(second, "done")
    fleet.provider.offers = dc_market(here=0.16, elsewhere=0.10)  # 6 cents an hour dearer: not
    await run(fleet, [workload])
    (third,) = fleet.hosts_of("research")
    assert third.offer.machine_id == "m-1" and third.volume_id is None
    assert any("not preferred" in r for e in fleet.events.recent(50) if e["kind"] == "rented"
               for r in e["numbers"].get("reasons", []))


async def test_a_host_on_a_model_volume_is_released_not_parked(db):
    provider, fleet, _, _, host = await first_host(db)
    await fleet.park(host, "idle")
    assert host.instance.instance_id not in provider.instances, "destroyed, not stopped"
    assert host.state != "parked" and "released rather than parked" in str(fleet.events.recent(20))


async def test_a_volumes_storage_is_on_the_workloads_lease(db):
    _, fleet, workload, lease, _ = await first_host(db)
    volume = volume_of(fleet)
    db.execute("UPDATE workload_volumes SET created_at = ? WHERE volume_id = ?", (time.time() - 3600, volume.volume_id))
    await run(fleet, [workload])
    totals = fleet.spend.latest_for_lease(lease.lease_id)
    rows = db.query("SELECT host_id, amount FROM spend WHERE host_id = ?", (f"volume:{volume.volume_id}",))
    assert rows and rows[-1]["amount"] == pytest.approx(volume.hourly, rel=0.05)
    assert totals["estimate"] >= rows[-1]["amount"]


async def test_a_volume_is_deleted_when_its_workload_is_over(db):
    provider, fleet, _, _, _ = await first_host(db)
    await fleet.delete_volumes("research")
    assert provider.volumes == {} and fleet.workload_store.volumes("research") == []
    assert "volume_deleted" in kinds(fleet)


async def test_the_sweep_deletes_a_volume_nobody_owns_and_keeps_a_live_workloads(db):
    provider, fleet, workload, _, _ = await first_host(db)
    provider.volumes["v-stray"] = VolumeInfo("v-stray", "EU-1", "gpm/test/old/models", 10, location="EU-1")
    await run(fleet, [workload])
    assert "v-stray" not in provider.volumes
    assert len(provider.volumes) == 1, "the live workload's volume stays"


async def test_turning_it_off_deletes_a_volume_once_no_host_has_it(db):
    provider, fleet, workload, _, host = await first_host(db)
    fleet.rented.providers["fake"].keep_models = False
    await fleet.sweep_volumes()
    assert len(provider.volumes) == 1, "its host still has it"
    await fleet.destroy(host, "done")
    await fleet.sweep_volumes()
    assert provider.volumes == {} and "volume_deleted" in kinds(fleet)


async def test_a_provider_that_cannot_list_volumes_has_none_deleted(db):
    provider = FakeProvider(offers=market(), capabilities=CAPABLE)
    fleet = make_fleet(db, provider)
    provider.volumes["v-stray"] = VolumeInfo("v-stray", "m-9", "gpm/test/old/models", 10)

    async def refuse(prefix):
        raise ProviderUnavailable("down")

    provider.list_volumes = refuse
    await fleet.sweep_volumes()
    assert "v-stray" in provider.volumes


async def test_nothing_is_kept_unless_the_account_keeps_models(db):
    provider, fleet, _, _, host = await first_host(db, keep=False)
    assert host.volume_id is None and provider.volumes == {}


async def test_volumes_bound_to_one_machine_are_never_used(db):
    # A machine-bound volume helps only when that machine is free again (D139): not offered.
    provider, fleet, _, _, host = await first_host(db, capabilities=CAPABLE, offers=market())
    assert host.volume_id is None and provider.instances[host.instance.instance_id].spec.volume is None


async def test_an_offer_that_cannot_land_where_a_volume_can_go_rents_without_one(db):
    provider, fleet, _, _, host = await first_host(db, offers=dc_market(where=()))
    assert host.offer.machine_id == "m-0" and host.volume_id is None


async def test_the_retired_warm_machine_setting_still_loads():
    from gpm_server.config import WorkloadsConfig

    config = WorkloadsConfig.model_validate({"keep_models_on_machine": True, "model_sources": ["warm", "sibling", "hub"]})
    assert config.model_sources == ["volume", "sibling", "hub"]


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


# --- the reviews' findings ---


async def test_whatever_a_host_says_of_its_volume_never_stops_a_pass(db):
    _, fleet, _, _, host = await first_host(db)
    for junk in ({"engine": "x"}, {"engine": {"model_sources": []}},
                 {"engine": {"model_sources": {"big:q4": {"fill": "filled" * 100, "bytes_from_volume": "x"}}}},
                 {"engine": {"model_sources": {"big:q4": {"files_from_volume": float("inf"), "fill": None}}}}):
        host.agent_facts, host.volume_reported = junk, False
        fleet._volume_news(host)  # raises nothing
    assert all(len(e["summary"]) < 1000 for e in fleet.events.recent(50)), "a host's text is cut short"


async def test_a_mismatch_marks_the_volume_stale_so_it_draws_no_more_hosts(db):
    _, fleet, workload, _, first = await first_host(db)
    agent_says(fleet, first, fill="filled")
    await fleet.destroy(first, "done")
    await run(fleet, [workload])
    (second,) = fleet.hosts_of("research")
    agent_says(fleet, second, files_from_volume=2, files_mismatched=1, files_missing=0)
    assert volume_of(fleet).state == "stale"


async def test_a_filler_that_never_says_it_filled_frees_the_volume_after_its_time(db):
    _, fleet, workload, _, first = await first_host(db)
    first.ready_at = time.time() - 4 * 3600  # its fill was lost, say, when its agent was put back
    assert fleet._volumes_here(workload)["fake"].state == "empty"
    assert "did not say it filled it" in str(fleet.events.recent(10))


async def test_a_filler_held_back_at_a_restart_is_not_taken_for_gone(db):
    _, fleet, workload, _, first = await first_host(db)
    del fleet.hosts[first.host_id]
    fleet.pending_adoption = {"fake": [type("Row", (), {"host_id": first.host_id, "workload": "research"})()]}
    assert fleet._volumes_here(workload)["fake"].state == "filling", "it may still be writing"
    fleet.pending_adoption = {}
    assert fleet._volumes_here(workload)["fake"].state == "empty"


async def test_a_hosts_volume_survives_a_restart_and_old_releases_can_still_adopt_it(db):
    _, fleet, _, _, host = await first_host(db)
    ref = fleet.published_ref(host)
    assert ref["volume_id"] == host.volume_id and ref["models_source"] == host.models_source
    assert "locations" not in ref["offer"] and "assumed" not in ref["offer"], "a release before 0.32 refuses them"
    assert ref["offer_beside"]["locations"] == ["EU-1"]


async def test_a_volume_deleted_at_the_provider_has_its_record_retired(db):
    provider, fleet, workload, _, _ = await first_host(db)
    volume = volume_of(fleet)
    provider.volumes.clear()  # deleted by hand in the provider's own console
    await fleet.sweep_volumes()
    assert fleet.workload_store.volumes("research"), "too new to be sure: a listing may not show it yet"
    db.execute("UPDATE workload_volumes SET created_at = ? WHERE volume_id = ?", (time.time() - 3600, volume.volume_id))
    await fleet.sweep_volumes()
    assert fleet.workload_store.volumes("research") == [] and "volume_gone" in kinds(fleet)


async def test_a_volume_whose_price_the_provider_does_not_state_is_never_made(db):
    offers = [dataclasses.replace(o, volume_per_gb_hourly=None) for o in dc_market()]
    provider, fleet, _, _, host = await first_host(db, offers=offers)
    assert host.volume_id is None and provider.volumes == {}


async def test_two_hosts_of_one_gpu_class_are_rented_where_an_id_names_a_class(db):
    # RunPod: an offer is a GPU type in a cloud, not a machine (the code review's finding).
    classes = dataclasses.replace(KEEPS, machine_ids_are_machines=False)
    provider, fleet = keeping(db, offers=[dc_market()[0]], capabilities=classes, keep=False)
    workload, _ = open_workload(fleet, hosts_at_start=2)
    await run(fleet, [workload])
    assert [h.offer.machine_id for h in fleet.hosts_of("research")] == ["m-0", "m-0"]


async def test_what_the_agent_reports_reaches_the_pool_under_the_tags_it_pulled(db):
    # End to end in shape: the agent's own report, as its facts carry it, keyed as the pool reads it.
    from gpm_agent import engines

    _, fleet, _, _, host = await first_host(db)
    facts = engines.VllmFacts()
    facts.sources = {tag: (model_volume.Report(fill="filled"), 12.0) for tag in host.builds.values()}
    host.agent_facts = {"engine": {"model_sources": engines._sources_fact(facts.sources)}}
    fleet._volume_news(host)
    assert volume_of(fleet).state == "ready"


async def test_a_host_whose_address_comes_after_its_creation_is_reached_when_it_does(db):
    # Found live on RunPod: a pod's SSH address appears a little after it is created.
    from gpm_server.providers.base import ConnectionInfo

    provider = FakeProvider(offers=market())
    fleet = make_fleet(db, provider)
    real = provider.connection
    placed = {"yet": False}

    async def later(instance):
        return await real(instance) if placed["yet"] else ConnectionInfo()

    provider.connection = later
    workload, _ = open_workload(fleet, hosts_at_start=1)
    await run(fleet, [workload])
    (host,) = fleet.hosts_of("research")
    assert host.dial_url is None, "no address yet"
    placed["yet"] = True
    await run(fleet, [workload])
    assert host.dial_url is not None and "host_reachable" in kinds(fleet)
