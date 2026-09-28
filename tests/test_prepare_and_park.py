"""Preparing a host on request, and parking one between runs.

docs/spec/supervisor.md §8. Overflow-driven renting answers "demand exceeded what I have";
preparing answers "get one ready before I start" and "keep one warm between runs".
"""

import asyncio
import time

import httpx
import pytest
from fakes.fake_ollama import FakeOllama
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.engines import OllamaEngine
from gpm_server.ledger import EventLog, LeaseRefused, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor.renting import Fleet
from gpm_server.transports import build_client

MODEL = "m1"


def make_fleet(database, provider, **rented_overrides):
    rented = {
        "provider": "fake",
        "workers": 2,
        "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02},
        "scale": {"scale_up_after_s": 0},
    }
    rented.update(rented_overrides)
    config = PoolConfig.model_validate(
        {
            "pool": {"name": "test", "model_set": [MODEL]},
            "auth": {"app_keys": ["k"]},
            "hosts": [
                {
                    "id": "local-1",
                    "kind": "local",
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"},
                }
            ],
            "rented": rented,
        }
    )
    return Fleet(
        config,
        config.rented,
        provider,
        LeaseStore(database),
        EventLog(database),
        SpendLedger(database),
    )


@pytest.fixture
def fleet(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    made = make_fleet(database, FakeProvider())
    try:
        yield made
    finally:
        database.close()


def kinds(fleet):
    return [event["kind"] for event in fleet.events.recent()]


# --- preparing ---


async def test_preparing_opens_its_own_small_lease(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2, when_ready="join")

    assert host is not None
    lease = fleet.leases.get(host.lease_id)
    assert lease.max_spend == 1.00  # its own cap, borrowed from no other lease
    assert lease.allow_rent
    assert "prepare_started" in kinds(fleet)


async def test_a_preparation_cannot_start_without_a_dollar_cap(fleet):
    with pytest.raises(LeaseRefused, match="dollar cap"):
        await fleet.prepare(max_spend=None, max_hours=2)


async def test_a_prepared_host_inside_its_hold_is_not_reaped_as_surplus(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    assert host.hold_until > time.time()
    host.state = "ready"

    # Demand is covered without it, which would otherwise make it surplus — but it has only
    # just been asked for, and it is not idle.
    await fleet.pass_once(ready_workers_higher_tiers=99, idle_seconds={host.host_id: 0})

    assert host.state == "ready" and not host.released


# --- unused: paused, then destroyed; load brings it back (D64) ---


async def ready_prepared_host(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"
    host.ready_at = time.time()
    return host


async def test_an_unused_host_is_paused_even_inside_its_hold(fleet):
    """The lease says what may be spent, not that it must be (the owner, 2026-09-20)."""
    host = await ready_prepared_host(fleet)
    assert host.hold_until > time.time()

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={host.host_id: 3 * 60})

    assert host.state == "parked"
    assert host.idle_since == pytest.approx(time.time() - 3 * 60, abs=5)
    assert fleet.provider.instances[host.instance.instance_id].state == "stopped"


async def test_a_paused_host_still_unused_at_the_second_limit_is_destroyed(fleet):
    host = await ready_prepared_host(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={host.host_id: 3 * 60})
    assert host.state == "parked"

    host.idle_since = time.time() - 6 * 60  # five minutes since its last request have passed
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert host.released
    assert host.instance.instance_id not in fleet.provider.instances


async def test_load_brings_a_paused_host_straight_back(fleet):
    host = await ready_prepared_host(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={host.host_id: 3 * 60})
    instances_before = set(fleet.provider.instances)

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, pressure=True)

    assert host.state in ("scheduling", "preparing") and host.idle_since is None
    assert set(fleet.provider.instances) == instances_before  # restarted, nothing new rented
    assert "park_restarted" in kinds(fleet)


async def test_the_lease_alone_does_not_bring_a_paused_host_back(fleet):
    """Before D64 the lease's standing demand restarted an idle host two minutes after it was
    parked, to idle and be parked again."""
    host = await ready_prepared_host(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={host.host_id: 3 * 60})
    instances_before = set(fleet.provider.instances)

    for _ in range(3):  # scale_up_after_s is 0 in this fleet: only the gate holds it back
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}, pressure=False)

    assert host.state == "parked"
    assert "eviction" not in kinds(fleet)  # parked on purpose is not outbid
    assert set(fleet.provider.instances) == instances_before
    assert "park_restarted" not in kinds(fleet)


async def test_the_model_set_is_pulled_and_pinned_on_a_prepared_host(fleet):
    """A host is ready only when the whole set is resident **together**."""
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident=set())
    server = ServerHandle(engine_fake.app, loop)
    try:
        fleet.provider.engine_urls = [server.base_url]
        host = await fleet.prepare(max_spend=1.00, max_hours=2)

        client = build_client(
            fleet.config.hosts[0].transport, fleet.config.pool, server.base_url
        )
        try:
            loaded = await fleet.load_model_set(host, OllamaEngine(), client)
        finally:
            await client.aclose()

        assert loaded
        assert MODEL in engine_fake.resident
        prepared = [e for e in fleet.events.recent() if e["kind"] == "prepared"][0]
        assert prepared["numbers"]["tags"] == [MODEL]
    finally:
        server.stop()
        loop.stop()


async def test_an_embedding_model_in_the_set_is_pinned_through_the_endpoint_it_serves():
    """The engine refuses `generate` for an embedding model (seen live: a 400 that sent a
    rented host back as unable to hold the set)."""
    loop = BackgroundLoop()
    engine_fake = FakeOllama(available={"chat-model:1", "an-embed-model:1"})
    server = ServerHandle(engine_fake.app, loop)
    try:
        async with httpx.AsyncClient(base_url=server.base_url) as client:
            await OllamaEngine().load_and_pin(client, ["an-embed-model:1", "chat-model:1"])
        assert engine_fake.pinned == {"chat-model:1", "an-embed-model:1"}
        assert engine_fake.resident >= {"chat-model:1", "an-embed-model:1"}
    finally:
        server.stop()
        loop.stop()


async def test_a_host_that_cannot_hold_the_whole_set_does_not_join(fleet):
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident=set())
    engine_fake.refuse_pull = True
    server = ServerHandle(engine_fake.app, loop)
    try:
        fleet.provider.engine_urls = [server.base_url]
        host = await fleet.prepare(max_spend=1.00, max_hours=2)
        client = build_client(fleet.config.hosts[0].transport, fleet.config.pool, server.base_url)
        try:
            loaded = await fleet.load_model_set(host, OllamaEngine(), client)
        finally:
            await client.aclose()

        assert not loaded
        assert "prepare_failed" in kinds(fleet)
    finally:
        server.stop()
        loop.stop()


# --- parking ---


async def test_parking_keeps_the_disk_and_records_the_break_even(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.download_cost = 0.50

    await fleet.park(host, "the run ended")

    assert host.state == "parked"
    assert fleet.provider.instances[host.instance.instance_id].state == "stopped"
    parked = [e for e in fleet.events.recent() if e["kind"] == "parked"][0]
    # download cost / storage per hour — how long parking stays cheaper than re-downloading.
    assert parked["numbers"]["break_even_hours"] == pytest.approx(100.0)


async def test_a_parked_host_is_restarted_before_a_new_offer_is_bid_on(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    await fleet.park(host, "between runs")
    instances_before = set(fleet.provider.instances)

    lease = fleet.open_lease(workers=4, max_hours=2, max_spend=1.0, allow_rent=True)
    restarted = await fleet.restart_parked(lease)

    assert restarted is not None
    assert restarted.host_id == host.host_id
    assert set(fleet.provider.instances) == instances_before  # nothing new was created
    assert "park_restarted" in kinds(fleet)


async def test_a_parked_host_whose_machine_is_gone_stays_parked(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    await fleet.park(host, "between runs")

    fleet.provider.offers = [default_offer(offer_id="o-9", machine_id="m-other")]
    lease = fleet.open_lease(workers=4, max_hours=2, max_spend=1.0, allow_rent=True)

    assert await fleet.restart_parked(lease) is None
    assert fleet.hosts[host.host_id].state == "parked"


async def test_parking_is_never_open_ended(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    fleet = make_fleet(database, FakeProvider(), teardown={"max_park_hours": 1})
    try:
        host = await fleet.prepare(max_spend=1.00, max_hours=2)
        await fleet.park(host, "between runs")

        host.parked_at = time.time() - 2 * 3600  # parked longer than the limit
        await fleet.expire_parked()

        assert fleet.hosts == {}
        assert fleet.provider.instances == {}
        assert any("parked longer" in e["summary"] for e in fleet.events.recent())
    finally:
        database.close()


# --- reaching a rented host ---


async def test_a_rented_host_without_a_public_port_is_reached_over_a_tunnel(tmp_path, monkeypatch):
    """The spec's default for rented hosts: the engine is never exposed."""
    from fakes.harness import stub_ssh_command

    monkeypatch.setattr("gpm_server.transports.tunnel.build_ssh_command", stub_ssh_command)
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident={MODEL})
    server = ServerHandle(engine_fake.app, loop)
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        # No public URL from the provider: only an SSH host and port. The stand-in forward
        # goes to the fake engine's port, as a real one would go to the engine on the host.
        fleet = make_fleet(database, FakeProvider(), engine_port=server.port)
        host = await fleet.prepare(max_spend=1.00, max_hours=1)

        assert host.connection.public_url is None
        assert host.dial_url.startswith("http://127.0.0.1:")
        assert host.host_id in fleet.tunnels

        import httpx

        async with httpx.AsyncClient(base_url=host.dial_url, timeout=10) as client:
            assert (await client.get("/api/ps")).status_code == 200

        await fleet.destroy(host, "done")
        assert host.host_id not in fleet.tunnels  # the forward went down with the host
    finally:
        for tunnel in list(fleet.tunnels.values()):
            await tunnel.stop()
        server.stop()
        loop.stop()
        database.close()


async def test_the_engine_is_launched_to_match_the_workers_and_the_model_set(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=1)
    env = fleet.provider.instances[host.instance.instance_id].spec.env
    assert env["OLLAMA_NUM_PARALLEL"] == "2"        # the rented host's worker count
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "1"   # the size of the model set
    assert env["OLLAMA_KEEP_ALIVE"] == "-1"         # loaded, all the time


async def test_the_engine_on_a_rented_host_listens_on_loopback_only(fleet):
    """Found live: a rented host's engine was answering strangers on the open internet.

    The provider's image published the engine's port and the engine bound every interface, so
    anyone who found the address could list the models and use an accelerator the pool was
    paying for. The pool dials through a forward into the machine, so loopback costs it
    nothing — and there is no authentication on an engine to fall back on (D77).
    """
    host = await fleet.prepare(max_spend=1.00, max_hours=1)
    env = fleet.provider.instances[host.instance.instance_id].spec.env

    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert not any(
        value.startswith(("0.0.0.0", "::", "*")) for value in env.values()
    ), f"the engine was launched reachable from off the host: {env}"


async def test_an_engine_start_command_runs_after_the_timer_is_armed(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        fleet = make_fleet(database, FakeProvider(), engine_start="ollama serve &")
        host = await fleet.prepare(max_spend=1.00, max_hours=1)
        onstart = fleet.provider.instances[host.instance.instance_id].spec.onstart
        assert onstart.index("deadman.sh") < onstart.index("ollama serve &")
    finally:
        database.close()


# --- a cut download is retried, not the host thrown away ---


async def _prepare_over(tmp_path, cut=0, end_early=0, refuse=False, released_after=None, **teardown):
    """A real engine on a real socket: a download can only truly be cut over one."""
    database = Database(tmp_path / "retry.sqlite3")
    fleet = make_fleet(database, FakeProvider(), teardown={"pull_retry_after_s": 0, **teardown})
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident=set())
    engine_fake.pull_cut_times, engine_fake.pull_end_early_times, engine_fake.refuse_pull = cut, end_early, refuse
    server = ServerHandle(engine_fake.app, loop)
    try:
        fleet.provider.engine_urls = [server.base_url]
        host = await fleet.prepare(max_spend=1.00, max_hours=2)
        if released_after is not None:
            real_pull = OllamaEngine.pull

            async def pull_then_release(self, client, tag):
                result = await real_pull(self, client, tag)
                if engine_fake.pulls >= released_after:
                    host.released = True
                return result

            engine = OllamaEngine()
            engine.pull = pull_then_release.__get__(engine)
        else:
            engine = OllamaEngine()
        client = build_client(fleet.config.hosts[0].transport, fleet.config.pool, server.base_url)
        try:
            loaded = await fleet.load_model_set(host, engine, client)
        finally:
            await client.aclose()
        return loaded, fleet.events.recent(), engine_fake  # read before the database closes
    finally:
        server.stop()
        loop.stop()
        database.close()


async def test_a_download_cut_partway_is_retried_and_the_host_joins(tmp_path):
    """Seen live: three H200 hosts destroyed in a row, each after its model download was cut."""
    loaded, events, engine = await _prepare_over(tmp_path, cut=2)

    assert loaded and MODEL in engine.resident
    assert engine.pulls == 3
    retries = [e for e in events if e["kind"] == "pull_retry"]
    assert [e["numbers"]["attempt"] for e in reversed(retries)] == [2, 3]
    assert "resuming what arrived" in retries[0]["summary"]
    assert {"prepared"} <= {e["kind"] for e in events} and "prepare_failed" not in {e["kind"] for e in events}


async def test_a_download_that_ends_without_success_is_not_taken_as_done(tmp_path):
    loaded, events, engine = await _prepare_over(tmp_path, end_early=1)
    assert loaded and engine.pulls == 2  # the quiet early end was retried, not believed


async def test_a_model_the_registry_does_not_have_is_not_retried(tmp_path):
    loaded, events, engine = await _prepare_over(tmp_path, refuse=True)
    assert not loaded and engine.pulls == 1
    assert "pull_retry" not in {e["kind"] for e in events}
    failed = next(e for e in events if e["kind"] == "prepare_failed")
    assert failed["numbers"]["retryable"] is False


async def test_retries_are_bounded_and_say_how_many_there_were(tmp_path):
    loaded, events, engine = await _prepare_over(tmp_path, cut=99, pull_attempts=3)
    assert not loaded and engine.pulls == 3
    failed = next(e for e in events if e["kind"] == "prepare_failed")
    assert "after 3 attempts" in failed["summary"]


async def test_a_host_released_while_its_download_is_retried_is_not_pulled_again(tmp_path):
    loaded, events, engine = await _prepare_over(tmp_path, cut=99, released_after=1)
    assert not loaded and engine.pulls == 1


# --- a prepare lease ends with its host (D47) ---


def lease_of(fleet, lease_id):
    return fleet.leases.get(lease_id)


async def test_a_prepare_lease_closes_when_its_host_is_destroyed(fleet):
    """Found live: four prepare leases outlived their hosts, one by 18 minutes, each still
    spending authority whose workers the pool counted as demand."""
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    await fleet.destroy(host, "could not hold the pool's model set")

    lease = lease_of(fleet, host.lease_id)
    assert not lease.is_open
    assert host.host_id in lease.closed_reason and "could not hold" in lease.closed_reason
    assert "lease_closed" in kinds(fleet)


async def test_a_prepare_lease_closes_when_its_host_is_evicted_and_not_re_bid(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    fleet.provider.evict(host.instance.instance_id)
    fleet.provider.offers = []  # nothing to re-bid on, nothing to replace it with
    await fleet.handle_evictions()

    assert host.released
    eviction = next(e for e in fleet.events.recent() if e["kind"] == "eviction")
    assert "rebid" not in eviction["summary"]
    assert not lease_of(fleet, host.lease_id).is_open


async def test_an_evicted_prepared_host_is_said_to_be_released_not_replaced(fleet):
    """Found live: "was outbid: replace … replacing on 43532" on a prepared host whose lease
    closed with it a second later, renting nothing. And the reasons kept only "$0.982 is below
    the floor $1.200", dropping the step that said which ceiling held the bid there."""
    import dataclasses

    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    rented_on = host.offer
    # Another machine is on offer — as 43532 was.
    fleet.provider.offers = [default_offer("o-2", "m-2")]
    # Won back only at a floor past every ceiling.
    fleet.provider.held_by_others = [dataclasses.replace(rented_on, min_bid_hourly=5.0, on_demand_hourly=None)]
    fleet.provider.evict(host.instance.instance_id)
    await fleet.handle_evictions()

    eviction = next(e for e in fleet.events.recent() if e["kind"] == "eviction")
    assert "nothing is rented in its place" in eviction["summary"]
    assert "replac" not in " ".join(eviction["numbers"]["reasons"]).replace("best other offer", "")
    assert any("clamped" in r for r in eviction["numbers"]["reasons"]), "which ceiling held the bid"
    assert host.released and not lease_of(fleet, host.lease_id).is_open
    assert len(fleet.provider.instances) == 0


async def test_an_evicted_overflow_host_is_said_to_leave_its_replacement_to_the_lease(fleet):
    import dataclasses

    fleet.open_lease(workers=6, max_hours=4, max_spend=5.00, allow_rent=True)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()
    rented_on = host.offer
    fleet.provider.offers = [default_offer("o-2", "m-2")]
    fleet.provider.held_by_others = [dataclasses.replace(rented_on, min_bid_hourly=5.0, on_demand_hourly=None)]
    fleet.provider.evict(host.instance.instance_id)
    await fleet.handle_evictions()

    eviction = next(e for e in fleet.events.recent() if e["kind"] == "eviction")
    assert "its lease rents a replacement if it still needs the capacity" in eviction["summary"]


async def test_a_prepare_lease_closes_when_its_host_vanishes_at_the_provider(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    del fleet.provider.instances[host.instance.instance_id]
    await fleet.handle_evictions()
    assert "no longer exists" in lease_of(fleet, host.lease_id).closed_reason


async def test_once_closed_nothing_is_rented_in_its_place(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    await fleet.destroy(host, "evicted; destroy")
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert fleet.hosts == {}
    assert len(fleet.provider.instances) == 0


async def test_an_overflow_lease_stays_open_after_an_eviction_so_capacity_is_recovered(fleet):
    """Unchanged on purpose: an overflow lease is standing demand, not one host's."""
    lease = fleet.open_lease(workers=6, max_hours=4, max_spend=5.00, allow_rent=True)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (host,) = fleet.hosts.values()
    assert not host.prepared
    await fleet.destroy(host, "evicted; replace")
    assert lease_of(fleet, lease.lease_id).is_open


async def test_a_lease_already_closed_keeps_the_reason_it_closed_for(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    fleet.leases.close(host.lease_id, "dollar cap reached")
    await fleet.destroy(host, "dollar cap reached")
    assert lease_of(fleet, host.lease_id).closed_reason == "dollar cap reached"


async def test_a_download_reports_its_progress_while_it_runs(tmp_path):
    """So a host being prepared shows a 19 GB download moving, rather than "preparing"."""
    loaded, events, engine = await _prepare_over(tmp_path)
    assert loaded
    prepared = next(e for e in events if e["kind"] == "prepared")
    assert prepared["numbers"]["bytes"] == engine.pull_bytes


async def test_the_stage_says_which_model_is_downloading_and_then_that_it_is_loading(tmp_path):
    seen = []
    database = Database(tmp_path / "stage.sqlite3")
    fleet = make_fleet(database, FakeProvider(), teardown={"pull_retry_after_s": 0})
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident=set())
    server = ServerHandle(engine_fake.app, loop)
    try:
        fleet.provider.engine_urls = [server.base_url]
        host = await fleet.prepare(max_spend=1.00, max_hours=2)
        engine = OllamaEngine()
        real_pull = engine.pull

        async def watch(client, tag, on_progress=None):
            result = await real_pull(client, tag, on_progress=on_progress)
            seen.append((host.stage, dict(host.progress)))
            return result

        engine.pull = watch
        client = build_client(fleet.config.hosts[0].transport, fleet.config.pool, server.base_url)
        try:
            assert await fleet.load_model_set(host, engine, client)
        finally:
            await client.aclose()
    finally:
        server.stop()
        loop.stop()
        database.close()

    stage, progress = seen[0]
    assert stage == f"downloading {MODEL}"
    assert progress[MODEL]["total"] == engine_fake.pull_bytes
    assert host.stage == ""  # cleared once the set is held


# --- a download far slower than the offer promised (D54) ---


async def test_a_download_far_slower_than_promised_gives_the_host_up(tmp_path):
    """Seen live: a download crawling on a machine advertising gigabits, released by hand."""
    database = Database(tmp_path / "slow.sqlite3")
    fleet = make_fleet(database, FakeProvider(), teardown={
        "pull_retry_after_s": 0, "min_pull_mbps": 50, "slow_pull_grace_s": 0.3,
    })
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident=set())
    engine_fake.pull_bytes = 3_000_000      # 3 MB...
    engine_fake.pull_delay_s = 0.4          # ...over more than a second: about 20 Mbps
    server = ServerHandle(engine_fake.app, loop)
    try:
        fleet.provider.engine_urls = [server.base_url]
        host = await fleet.prepare(max_spend=1.00, max_hours=2)
        client = build_client(fleet.config.hosts[0].transport, fleet.config.pool, server.base_url)
        try:
            loaded = await fleet.load_model_set(host, OllamaEngine(), client)
        finally:
            await client.aclose()
        events = fleet.events.recent()
        avoided = fleet.avoided_now()
    finally:
        server.stop()
        loop.stop()
        database.close()

    assert not loaded
    slow = next(e for e in events if e["kind"] == "host_too_slow")
    assert "under the 50 Mbps floor" in slow["summary"] and "advertised" in slow["summary"]
    assert engine_fake.pulls == 1  # not retried: the same link would be just as slow
    assert host.offer.machine_id in avoided


async def test_a_fast_enough_download_is_left_alone_and_reports_its_speed(tmp_path):
    loaded, events, engine = await _prepare_over(tmp_path, min_pull_mbps=50, slow_pull_grace_s=0.0)
    assert loaded and "host_too_slow" not in {e["kind"] for e in events}


async def test_the_speed_floor_can_be_switched_off(tmp_path):
    loaded, events, engine = await _prepare_over(tmp_path, min_pull_mbps=0)
    assert loaded


async def test_a_host_that_came_up_without_its_start_up_material_is_ended_at_once(fleet):
    """Seen live: an instance created with the script and reported back without it. It refused
    SSH for six minutes, had no dead-man timer, and billed until the operator released it."""
    fleet.provider.drop_startup_material = True
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    assert host is not None

    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

    assert host.released
    assert host.instance.instance_id not in fleet.provider.instances
    assert "host_without_startup" in kinds(fleet)
    assert host.offer.machine_id in fleet.avoided_now()



# --- the agent the pool puts on a host it rents (D63) ---


async def test_a_host_that_cannot_take_an_agent_still_prepares(fleet):
    """No agent is never fatal: the host is prepared the way it always was, and the reason is
    recorded rather than left for someone to notice."""
    host = await fleet.prepare(max_spend=1.00, max_hours=2)

    async def no_interpreter(_host, command):
        return (0, "") if "command -v python3" in command else (0, "")

    fleet.run_on_host = no_interpreter
    for _ in range(fleet.rented.agent_attempts):
        await fleet.install_agent(host)

    assert host.agent is None
    assert "no python3" in (host.agent_detail or "")
    assert "agent_not_installed" in kinds(fleet)
    # Said once, when the pool gave up — not once a pass, which cost an SSH round trip each
    # time and filled the log (seen in the simulation).
    assert kinds(fleet).count("agent_not_installed") == 1
    await fleet.install_agent(host)
    assert kinds(fleet).count("agent_not_installed") == 1, "it stopped asking"
    # And the host is still perfectly usable.
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident=set())
    server = ServerHandle(engine_fake.app, loop)
    try:
        client = build_client(fleet.config.hosts[0].transport, fleet.config.pool, server.base_url)
        try:
            assert await fleet.load_model_set(host, OllamaEngine(), client)
        finally:
            await client.aclose()
    finally:
        server.stop()
        loop.stop()


async def test_the_agent_is_not_installed_when_the_operator_says_not_to(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        fleet = make_fleet(database, FakeProvider(), agent_on_rented_hosts=False)
        host = await fleet.prepare(max_spend=1.00, max_hours=2)

        called = []
        fleet.run_on_host = lambda *a: called.append(a)
        await fleet.install_agent(host)

        assert host.agent is None and not called
    finally:
        database.close()


async def test_a_host_takes_fewer_requests_at_once_without_touching_its_engine(fleet):
    """Lowering is the pool using fewer of the slots the engine already has: instant, graceful,
    and no restart — which is what makes it free (D56)."""
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"

    done, why = await fleet.resize(host, 1)

    assert done and host.workers == 1
    assert host.state == "ready", "nothing was restarted, so nothing has to be re-verified"
    resized = [e for e in fleet.events.recent() if e["kind"] == "host_resized"][0]
    assert resized["numbers"]["restarted"] is False


async def test_raising_the_count_needs_the_hosts_own_agent(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"

    done, why = await fleet.resize(host, host.workers + 4)

    assert not done and "no agent" in why
    assert host.workers == 2, "the count only changes when the engine really can"


async def test_raising_the_count_relaunches_the_engine_through_the_agent(fleet, monkeypatch):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"
    host.agent = object()
    asked = {}

    async def restart(agent, settings, transport=None):
        asked.update(settings)
        return 200, {"engine_answers": True}

    monkeypatch.setattr("gpm_server.supervisor.agents.restart_engine", restart)

    done, why = await fleet.resize(host, 6)

    assert done and host.workers == 6
    assert asked["workers"] == 6, "the engine is told the number, as a number (D41)"
    assert host.state == "preparing", "it holds the set again before anything is routed to it"


async def test_a_relaunch_hours_into_a_hosts_life_restarts_its_clock(fleet, monkeypatch):
    """Found 2026-09-26: a resize that relaunches the engine put a host that had served for
    hours back in `preparing` with the clock it was created with, so the next pass destroyed it
    as "not ready after 30 minutes" and avoided its machine."""
    fleet.rented.teardown.max_preparing_minutes = 30
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"
    host.created_at -= 3 * 3600
    host.preparing_since -= 3 * 3600
    host.agent = object()

    async def restart(agent, settings, transport=None):
        return 200, {"engine_answers": True}

    monkeypatch.setattr("gpm_server.supervisor.agents.restart_engine", restart)
    done, _why = await fleet.resize(host, 6)
    assert done and host.state == "preparing" and host.preparing_since > time.time() - 5

    await fleet.tear_down(fleet.leases.open_leases(), {})
    assert host.host_id in fleet.hosts and host.offer.machine_id not in fleet.avoided


async def test_a_host_restarted_from_parked_is_given_its_own_time(fleet):
    fleet.rented.teardown.max_preparing_minutes = 30
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    await fleet.park(host, "between runs")
    host.created_at -= 3 * 3600
    host.preparing_since = (host.preparing_since or time.time()) - 3 * 3600

    lease = fleet.open_lease(workers=4, max_hours=2, max_spend=1.0, allow_rent=True)
    restarted = await fleet.restart_parked(lease)
    assert restarted is host and host.state == "preparing" and host.preparing_since > time.time() - 5


async def test_an_engine_that_does_not_come_back_does_not_get_the_new_count(fleet, monkeypatch):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"
    host.agent = object()

    async def restart(agent, settings, transport=None):
        return 200, {"engine_answers": False}

    monkeypatch.setattr("gpm_server.supervisor.agents.restart_engine", restart)

    done, why = await fleet.resize(host, 6)

    assert not done and host.workers == 2
    assert "host_resize_failed" in kinds(fleet)


# --- a host finding its own worker count (D67, D68) ---


async def test_a_host_climbs_within_what_its_engine_was_launched_for(fleet):
    """The engine is started at the most the host may be asked for, so climbing to it is the
    pool using slots that already exist: instant, and nothing restarts (D68)."""
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"
    host.launch_workers = 16

    done, why = await fleet.resize(host, host.workers + 3)

    assert done and host.workers == 5
    assert host.state == "ready", "nothing was restarted, so nothing has to be re-verified"
    resized = [e for e in fleet.events.recent() if e["kind"] == "host_resized"][0]
    assert resized["numbers"]["restarted"] is False


async def test_past_that_it_still_takes_an_agent_and_a_relaunch(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"
    host.launch_workers = 4

    done, why = await fleet.resize(host, 8)

    assert not done and "relaunching" in why
    assert host.workers == 2


async def test_the_engine_is_launched_for_the_ceiling_only_when_auto_is_on(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        off = make_fleet(database, FakeProvider())
        assert off.launch_workers_for(6) == 6, "without auto, launched for what it is given"

        on = make_fleet(database, FakeProvider(), workers_auto={"enabled": True, "max": 16})
        assert on.launch_workers_for(6) == 16, "with auto, launched for the most it may climb to"
        assert on.launch_workers_for(24) == 24, "a profile above the ceiling is still honoured"
    finally:
        database.close()


async def test_a_parked_host_is_left_alone_rather_than_probed_into_an_eviction(fleet, tmp_path):
    """Found by the simulation: parking a host stopped its engine, the probe then found
    nothing answering and marked it `preparing`, and the eviction handler read "stopped, and
    we did not ask" as an eviction. Parked hosts were destroyed seconds after being parked, so
    parking had never once saved a download."""
    from gpm_server.supervisor import Supervisor

    supervisor = Supervisor(fleet.config, Database(tmp_path / "probe.sqlite3"), provider=fleet.provider)
    supervisor.fleet = fleet
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.state = "ready"
    host.dial_url = "http://127.0.0.1:1"  # nothing answers there, as on a stopped engine
    await fleet.park(host, "no traffic")
    assert host.state == "parked"

    await supervisor._probe_rented()

    assert host.state == "parked", "a parked host is not probed back into preparing"
    await fleet.handle_evictions()
    assert not host.released, "and so is never read as outbid"


async def test_the_agent_is_not_attempted_before_the_image_is_up(fleet, tmp_path):
    """Seen live: a host rented at 12:01:13 had spent all three of its attempts by 12:01:47,
    concluding "no python3" from an image that had not finished starting — and so ran without
    an agent for its whole life. SSH answers long before the image is usable; an engine that
    answers is the proof that it is."""
    from gpm_server.supervisor import Supervisor

    supervisor = Supervisor(fleet.config, Database(tmp_path / "early.sqlite3"), provider=fleet.provider)
    supervisor.fleet = fleet
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    host.dial_url = "http://127.0.0.1:1"  # nothing answers there yet, as on a booting image

    tried = []
    fleet.install_agent = lambda h: tried.append(h) or asyncio.sleep(0)

    await supervisor._probe_rented()

    assert not tried, "the agent was attempted before the engine had answered"
    assert host.agent_attempts == 0, "and so none of its attempts were spent"


async def test_each_model_is_loaded_as_it_lands_even_without_an_agent(fleet):
    """D57 was built into the agent — but a host without one falls back to this path, and it
    was still pulling everything before loading anything. Seen live: the first rental of the
    day had no agent, so the saving D57 exists for was not made."""
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident=set(), available={"a", "b"})
    server = ServerHandle(engine_fake.app, loop)
    try:
        fleet.config.pool.model_set = ["a", "b"]
        fleet.provider.engine_urls = [server.base_url]
        host = await fleet.prepare(max_spend=1.00, max_hours=2)
        client = build_client(fleet.config.hosts[0].transport, fleet.config.pool, server.base_url)
        try:
            await fleet.load_model_set(host, OllamaEngine(), client)
        finally:
            await client.aclose()
    finally:
        server.stop()
        loop.stop()

    ordered = [path for path, _body, _headers in engine_fake.received]
    pulls = [i for i, path in enumerate(ordered) if path == "/api/pull"]
    loads = [i for i, path in enumerate(ordered) if path in ("/api/generate", "/api/embed")]
    assert len(pulls) == 2 and loads, "both models pulled, and loading happened"
    assert min(loads) < max(pulls), "the first model was loaded before the last one downloaded"


async def test_a_rented_host_with_a_working_agent_does_not_stop_the_control_loop(fleet, monkeypatch):
    """Found live, on the first rented host whose agent ever installed at once: asking the
    agent read a field its answer does not have, the pass raised, and because the probe runs
    before preparation the host never began downloading — it sat there billing, healthy."""
    from gpm_server.supervisor import agents, hostagent

    host = await fleet.prepare(max_spend=1.00, max_hours=1)
    host.agent = hostagent.RentedAgent(url="http://127.0.0.1:1", secret="gpmg_" + "0" * 64)

    async def answers(agent, transport=None):
        return agents.AgentView(reachable=True, facts={"os": "Linux"})

    monkeypatch.setattr(agents, "ask", answers)
    await fleet.ask_agent(host)
    assert host.agent_facts == {"os": "Linux"} and host.agent_detail is None

    async def silent(agent, transport=None):
        return agents.AgentView(reachable=False, detail="no answer")

    monkeypatch.setattr(agents, "ask", silent)
    await fleet.ask_agent(host)
    assert host.agent_facts is None and host.agent_detail == "no answer"


async def test_a_freshly_installed_agent_is_given_a_moment_before_the_slow_path(fleet, monkeypatch):
    """Found live: asked the instant its install returned, the agent was not listening yet; one
    failed call and the host was prepared without it — pull, then load, then pull — for good."""
    from gpm_server.supervisor import agents, hostagent

    host = await fleet.prepare(max_spend=1.00, max_hours=1)
    host.agent = hostagent.RentedAgent(url="http://127.0.0.1:1", secret="gpmg_" + "0" * 64)
    fleet.agent_hold_retry_s = 0.01
    calls = []

    async def not_yet_then_answers(agent, tags, residency, transport=None):
        calls.append(1)
        if len(calls) < 3:
            return None
        return {"models": [{"tag": tag, "loaded": False, "pulling": {"tag": tag}} for tag in tags]}

    monkeypatch.setattr(agents, "hold", not_yet_then_answers)
    assert await fleet.load_model_set_through_agent(host) is False  # coming, through the agent
    assert len(calls) == 3

    async def never(agent, tags, residency, transport=None):
        calls.append(1)
        return None

    calls.clear()
    monkeypatch.setattr(agents, "hold", never)
    assert await fleet.load_model_set_through_agent(host) is None   # a real silence falls back
    assert len(calls) == fleet.agent_hold_attempts


async def test_a_download_through_the_agent_is_reported_the_way_the_console_reads_it(fleet, monkeypatch):
    """Found live: the agent's own words were passed straight to the console, which drew an
    empty bar for a download that was running. Both paths must speak one shape."""
    from gpm_server.supervisor import agents, hostagent

    host = await fleet.prepare(max_spend=1.00, max_hours=1)
    host.agent = hostagent.RentedAgent(url="http://127.0.0.1:1", secret="gpmg_" + "0" * 64)
    (tag,) = sorted(fleet.required_tags)

    async def halfway(agent, tags, residency, transport=None):
        return {"models": [{"tag": tag, "on_disk": False, "loaded": False, "size_bytes": None,
                            "pulling": {"tag": tag, "completed_bytes": 5_000, "total_bytes": 10_000}}]}

    monkeypatch.setattr(agents, "hold", halfway)
    assert await fleet.load_model_set_through_agent(host) is False

    shown = host.progress[tag]
    assert shown["completed"] == 5_000 and shown["total"] == 10_000 and shown["attempt"] == 1
    assert set(host.progress) == {tag}, "keyed by model tag, as the direct path is"


async def test_a_host_whose_engine_runs_on_the_processor_is_given_up(fleet):
    """Found live on an 80GB A100: the image refused the machine's driver, ollama fell back to
    the CPU, and `nvidia-smi` read 0 MiB used while a 26B model loaded at 100% CPU. It passed
    every filter, answered every probe, and was worth nothing at an accelerator's price. D81's
    driver floor refuses that machine before renting; this is the same fault arriving any other
    way, on a host already paid for."""
    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident={MODEL})
    engine_fake.on_cpu = {MODEL}          # the card is there and the engine is not using it
    server = ServerHandle(engine_fake.app, loop)
    try:
        import httpx as _httpx

        async with _httpx.AsyncClient(base_url=server.base_url, timeout=10) as client:
            on_cpu = await OllamaEngine().serving_from_cpu(client)
        assert on_cpu == frozenset({MODEL})

        engine_fake.on_cpu = set()        # and when it is using the card, nothing is reported
        async with _httpx.AsyncClient(base_url=server.base_url, timeout=10) as client:
            assert await OllamaEngine().serving_from_cpu(client) == frozenset()
    finally:
        server.stop()
        loop.stop()


async def test_an_engine_that_does_not_report_where_it_runs_is_not_accused(fleet):
    """`size_vram` absent means this engine build does not say — not that it is on the CPU.
    Guessing would destroy healthy hosts."""
    import httpx as _httpx

    loop = BackgroundLoop()
    engine_fake = FakeOllama(resident={MODEL})
    server = ServerHandle(engine_fake.app, loop)
    try:
        # An engine build that does not report `size_vram` at all.
        engine_fake._model_list = lambda tags: {"models": [{"name": t, "size": 1} for t in sorted(tags)]}
        async with _httpx.AsyncClient(base_url=server.base_url, timeout=10) as client:
            assert await OllamaEngine().serving_from_cpu(client) == frozenset()
    finally:
        server.stop()
        loop.stop()
