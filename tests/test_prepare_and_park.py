"""Preparing a host on request, and parking one between runs.

docs/spec/supervisor.md §8. Overflow-driven renting answers "demand exceeded what I have";
preparing answers "get one ready before I start" and "keep one warm between runs".
"""

import time

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
        "model_set_gb": 10.0,
        "bidding": {"bid_ceiling": 0.60, "premium": 0.02},
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


async def test_a_prepared_host_is_held_rather_than_reaped_for_having_no_traffic_yet(fleet):
    host = await fleet.prepare(max_spend=1.00, max_hours=2)
    assert host.hold_until > time.time()

    # It has served nothing at all, which would otherwise look exactly like idleness.
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={host.host_id: 99 * 60})

    assert host.host_id in fleet.hosts
    assert not fleet.hosts[host.host_id].released


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


async def test_an_engine_start_command_runs_after_the_timer_is_armed(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        fleet = make_fleet(database, FakeProvider(), engine_start="ollama serve &")
        host = await fleet.prepare(max_spend=1.00, max_hours=1)
        onstart = fleet.provider.instances[host.instance.instance_id].spec.onstart
        assert onstart.index("deadman.sh") < onstart.index("ollama serve &")
    finally:
        database.close()
