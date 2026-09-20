"""A supervisor restart takes back what it rented; it does not sweep it as an orphan.

docs/spec/supervisor.md §1.1–§1.2: `gpm restart` keeps rented hosts and re-adopts them. The
provider is the source of truth for what exists; the database for what was intended. Found as
a gap on the first live run — a restarted supervisor knew nothing, so its first sweep would
have destroyed a perfectly good host that was still billing.
"""


import pytest
from fakes.fake_ollama import FakeOllama
from fakes.harness import BackgroundLoop, ServerHandle, stub_ssh_command
from gpm_server.config import PoolConfig
from gpm_server.db import Database, HostTable
from gpm_server.providers import FakeProvider
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.service import ProviderCredentialMissing

MODEL = "m1"


def config(model_set=(MODEL,)):
    return PoolConfig.model_validate(
        {
            "pool": {"name": "restart", "model_set": list(model_set), "probe_interval_s": 3600},
            "auth": {"app_keys": ["k"]},
            "hosts": [],
            "rented": {
                "provider": "fake",
                "workers": 2,
                "bidding": {"bid_ceiling": 0.60},
                "scale": {"scale_up_after_s": 0},
                "teardown": {"park_when_idle": False},
            },
        }
    )


@pytest.fixture
def market(monkeypatch):
    """One provider and one engine that outlive any single supervisor, as the real ones do."""
    monkeypatch.setattr("gpm_server.transports.tunnel.build_ssh_command", stub_ssh_command)
    loop = BackgroundLoop()
    engine = FakeOllama(resident={MODEL})
    server = ServerHandle(engine.app, loop)
    provider = FakeProvider(engine_urls=[server.base_url, server.base_url])
    try:
        yield loop, provider, engine
    finally:
        server.stop()
        loop.stop()


def run_until(loop, supervisor, predicate, passes=6):
    for _ in range(passes):
        loop.run(supervisor.pass_once())
        if predicate():
            return True
    return predicate()


def rent_one(loop, supervisor):
    supervisor.fleet.open_lease(workers=2, max_hours=2, max_spend=2.0, allow_rent=True)
    assert run_until(loop, supervisor, lambda: any(h.state == "ready" for h in supervisor.fleet.hosts.values()))
    return next(iter(supervisor.fleet.hosts.values()))


async def silent(host, command):
    return 0, ""


def test_a_restarted_supervisor_takes_its_host_back_instead_of_sweeping_it(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        instance_id = host.instance.instance_id
        loop.run(first.aclose())  # `gpm restart`: the host is kept, the process is not

        second = Supervisor(config(), database, provider=provider)
        second.fleet.run_on_host = silent
        loop.run(second.start())  # adopts, then its first pass sweeps

        assert instance_id in provider.instances, "the restart destroyed a good host"
        assert host.host_id in second.fleet.hosts
        adopted = second.fleet.hosts[host.host_id]
        assert adopted.instance.instance_id == instance_id
        assert adopted.bid_hourly == host.bid_hourly
        assert adopted.offer.machine_id == host.offer.machine_id
        assert adopted.created_at == pytest.approx(host.created_at)  # spend keeps counting from the start
        assert adopted.lease_id == host.lease_id

        kinds = [e["kind"] for e in second.events.recent(50)]
        assert "adopted" in kinds
        assert "orphan_swept" not in kinds
        loop.run(second.aclose())
    finally:
        database.close()


def test_an_adopted_host_is_re_verified_and_serves_again(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        loop.run(first.aclose())

        second = Supervisor(config(), database, provider=provider)
        second.fleet.run_on_host = silent
        # Adoption alone never trusts the old row's "ready": the first pass re-probes.
        loop.run(second.adopt_rented())
        adopted = second.fleet.hosts[host.host_id]
        assert adopted.state == "preparing"
        loop.run(second.pass_once())
        assert adopted.state == "ready"  # verified through the tunnel, not assumed
        rows = {r.host_id: r for r in HostTable(database).all()}
        assert rows[host.host_id].state == "ready"
        loop.run(second.aclose())
    finally:
        database.close()


def test_a_row_whose_instance_is_gone_is_dropped_not_kept_billing_on_paper(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        loop.run(first.aclose())

        # While no supervisor ran, the dead-man timer (or an operator) ended the instance.
        provider.instances.pop(host.instance.instance_id)

        second = Supervisor(config(), database, provider=provider)
        second.fleet.run_on_host = silent
        loop.run(second.start())

        assert host.host_id not in second.fleet.hosts
        assert all(r.host_id != host.host_id for r in HostTable(database).all())
        assert "host_gone" in [e["kind"] for e in second.events.recent(50)]
        loop.run(second.aclose())
    finally:
        database.close()


def test_an_adopted_host_under_a_closed_lease_is_released_by_the_first_pass(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        first.leases.close(host.lease_id, "operator finished")
        loop.run(first.aclose())  # stopped before the close could take effect

        second = Supervisor(config(), database, provider=provider)
        second.fleet.run_on_host = silent
        loop.run(second.start())

        assert host.instance.instance_id not in provider.instances
        assert host.host_id not in second.fleet.hosts
        loop.run(second.aclose())
    finally:
        database.close()


def test_a_stray_instance_is_still_swept_after_adoption(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        loop.run(first.aclose())
        stray = provider.strand(label="gpm/restart/rented-nobody-knows")

        second = Supervisor(config(), database, provider=provider)
        second.fleet.run_on_host = silent
        loop.run(second.start())

        assert host.instance.instance_id in provider.instances  # kept
        assert stray not in provider.instances  # swept
        loop.run(second.aclose())
    finally:
        database.close()


def test_the_published_ref_carries_no_secret(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        supervisor = Supervisor(config(), database, provider=provider)
        supervisor.fleet.run_on_host = silent
        loop.run(supervisor.start())
        host = rent_one(loop, supervisor)
        ref = supervisor.fleet.published_ref(host)
        flat = str(ref).lower()
        assert "key" not in flat and "token" not in flat and "secret" not in flat
        assert ref["offer"]["raw"] == {}  # the provider's raw payload does not travel
        loop.run(supervisor.aclose())
    finally:
        database.close()


def test_a_supervisor_that_cannot_ask_the_provider_keeps_every_record(tmp_path, market):
    """Seen live, and it cost a host: a supervisor started where the provider could not be
    asked read "could not list" as "nothing there", dropped the host's record, and the next
    supervisor destroyed the healthy instance as an orphan."""
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        instance_id = host.instance.instance_id
        loop.run(first.aclose())

        provider.unavailable = True  # the second supervisor cannot reach the provider
        second = Supervisor(config(), database, provider=provider)
        second.fleet.run_on_host = silent
        loop.run(second.start())
        loop.run(second.pass_once())

        rows = {row.host_id for row in HostTable(database).all()}
        assert host.host_id in rows, "the record was dropped because the provider could not be asked"
        assert "adoption_deferred" in [e["kind"] for e in second.events.recent(50)]
        loop.run(second.aclose())

        provider.unavailable = False  # and the third can: it adopts, it does not sweep
        third = Supervisor(config(), database, provider=provider)
        third.fleet.run_on_host = silent
        loop.run(third.start())

        assert instance_id in provider.instances, "a healthy host was destroyed as an orphan"
        assert host.host_id in third.fleet.hosts
        assert "orphan_swept" not in [e["kind"] for e in third.events.recent(50)]
        loop.run(third.aclose())
    finally:
        database.close()


def test_adoption_resumes_in_the_same_supervisor_once_the_provider_answers(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        loop.run(first.aclose())

        provider.unavailable = True
        second = Supervisor(config(), database, provider=provider)
        second.fleet.run_on_host = silent
        loop.run(second.start())
        assert host.host_id not in second.fleet.hosts

        provider.unavailable = False
        loop.run(second.pass_once())
        assert host.host_id in second.fleet.hosts
        loop.run(second.aclose())
    finally:
        database.close()


def test_a_supervisor_with_no_usable_credential_does_not_start(tmp_path, market):
    loop, provider, _ = market
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        first = Supervisor(config(), database, provider=provider)
        first.fleet.run_on_host = silent
        loop.run(first.start())
        host = rent_one(loop, first)
        loop.run(first.aclose())

        provider.credential_refused = True
        second = Supervisor(config(), database, provider=provider)
        with pytest.raises(ProviderCredentialMissing, match="credential"):
            loop.run(second.start())

        assert host.host_id in {row.host_id for row in HostTable(database).all()}
        loop.run(second.aclose())
    finally:
        database.close()

