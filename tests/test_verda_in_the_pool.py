"""The Verda plug-in under the pool's own renting, against its stubbed API (D140): a spot offer at
the provider's price is rented without a bid, and with no dead-man timer a lease longer than
`max_hours_without_deadman` rents nothing there. No network, no credentials, no spending."""

import pytest
from gpm_server.db import Database
from test_workload_fleet import make_fleet, open_workload, run
from unit.test_verda_provider import Api, provider


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


async def test_a_short_workload_rents_a_spot_machine_at_the_providers_price(db):
    api = Api(**{"POST /v1/instances": "inst-1", "GET /v1/instances": []})
    fleet = make_fleet(db, provider(api))
    workload, lease = open_workload(fleet, hosts_at_start=1, hours=0.9)
    await run(fleet, [workload])
    (body,) = api.bodies("POST", "/v1/instances")
    assert body["is_spot"] is True and body["os_volume"]["on_spot_discontinue"] == "delete_permanently"
    (script,) = api.bodies("POST", "/v1/scripts")
    assert "deadman.sh" not in script["script"], "no timer: the provider has no credential that ends only itself"
    (host,) = fleet.hosts_of("research")
    assert host.offer.interruptible and not host.offer.bidding


async def test_a_long_lease_rents_nothing_without_a_dead_man_timer(db):
    api = Api(**{"POST /v1/instances": "inst-1", "GET /v1/instances": []})
    fleet = make_fleet(db, provider(api))
    workload, lease = open_workload(fleet, hosts_at_start=1, hours=4.0)
    await run(fleet, [workload])
    assert "POST /v1/instances" not in api.paths()
