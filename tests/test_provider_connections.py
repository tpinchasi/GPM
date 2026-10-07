"""Provider connections (D129, docs/spec/providers.md), step 1: the configuration's shape, and
every record naming the connection it belongs to — so a pool that later rents from several
providers can tell their machines, money and quotas apart. Nothing here spends money."""

import pytest
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor.renting import Fleet, legacy_connection
from gpm_server.workload_store import Workload, WorkloadStore

BIG = "big"


def pool_config(**rented):
    base = {"workers": 4, "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 2.0},
            "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0}}
    return PoolConfig.model_validate({
        "pool": {"name": "test", "model_set": ["m1", BIG], "models_per_host": "declared"},
        "auth": {"app_keys": ["k"]},
        "catalog": {BIG: {"variants": [{"tag": "big:q4", "size_gb": 20}]}},
        "hosts": [{"id": "laptop", "kind": "local", "models": ["m1"],
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "workloads": {"min_reliability": 0.0},
        "rented": {**base, **rented},
    })


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield database
    database.close()


def make_fleet(db, plugin, **rented):
    config = pool_config(**rented)
    return Fleet(config, config.rented, plugin, LeaseStore(db), EventLog(db), SpendLedger(db))


# --- the configuration ---


def test_the_single_provider_shape_loads_as_one_connection_named_after_its_type():
    rented = pool_config(provider="fake", provider_settings={}).rented
    assert rented.connection_name == "fake" and rented.connection.type == "fake"
    assert set(rented.providers) == {"fake"}


def test_an_old_plugin_name_that_is_no_connection_name_still_loads():
    """A plug-in's entry-point name has no such rule; a file that loaded before still does."""
    from gpm_server.config import _connection_name_for

    assert _connection_name_for("My.Cloud") == "my-cloud"
    assert _connection_name_for("9lives") == "p-9lives"
    assert len(_connection_name_for("x" * 40)) == 32


def test_connections_are_named_and_at_least_one_is_enabled():
    two = pool_config(providers={"a": {"type": "fake-a", "enabled": False}, "b": {"type": "fake-b"}}).rented
    assert two.connection_name == "b", "the one enabled is the one rented through"
    with pytest.raises(ValueError, match="both `provider` and `providers`"):
        pool_config(provider="fake", providers={"a": {"type": "fake"}})
    both = pool_config(providers={"a": {"type": "fake-a"}, "b": {"type": "fake-b"}}).rented
    assert both.enabled_connections == ["a", "b"] and both.connection_name == "a"
    with pytest.raises(ValueError, match="at least one provider connection must be enabled"):
        pool_config(providers={"a": {"type": "fake", "enabled": False}})
    with pytest.raises(ValueError, match="at most one connection per provider"):
        pool_config(providers={"a": {"type": "fake"}, "b": {"type": "fake", "enabled": False}})
    with pytest.raises(ValueError, match="lower-case name"):
        pool_config(providers={"Bad Name": {"type": "fake"}})
    with pytest.raises(ValueError, match="needs a provider"):
        pool_config(providers={})


# --- every record names its connection ---


async def test_offers_hosts_spend_and_machine_events_carry_their_connection(db):
    fleet = make_fleet(db, FakeProvider(offers=[default_offer("o-1", "m-1")]), providers={"acct-a": {"type": "fake"}})
    lease = fleet.leases.open(workers=4, max_hours=2, max_spend=10, allow_rent=True, workload="w")
    now = 1e9
    WorkloadStore(db).create(Workload(
        name="w", model=BIG, builds={BIG: "big:q4"}, latency_s=30, parallel=4, kind="interruptible",
        lease_id=lease.lease_id, state="preparing", workers_per_host=4, hosts_at_start=1, plan={},
        created_at=now, updated_at=now, ends_at=now + 7200))
    assert all(o.connection == "acct-a" for o in await fleet._offers())
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={},
                          workloads=[WorkloadStore(db).get("w")], loads={})
    (host,) = fleet.hosts_of("w")
    assert host.connection_name == "acct-a"
    ref = fleet.published_ref(host)
    assert ref["connection"] == "acct-a", "kept across a restart"
    assert "connection" not in ref["offer"], "an older release builds the offer from this: rolling back still adopts"
    await fleet.record_spend(lease)
    spent = db.query("SELECT DISTINCT connection FROM spend WHERE host_id = ?", (host.host_id,))
    assert [r["connection"] for r in spent] == ["acct-a"]
    rented = [e for e in fleet.events.recent(50) if e["numbers"].get("machine")]
    assert rented and all(e["numbers"]["connection"] == "acct-a" for e in rented)


def test_a_record_from_before_connections_is_the_connection_the_pool_first_had(db):
    db.execute("INSERT INTO search_usage (day, rows, quota, updated_at) VALUES ('2026-10-06', 18984, 20000, 1)")
    first = make_fleet(db, FakeProvider(), provider="fake")
    first.claim_records()
    assert first.legacy_connection == "fake"
    rows = db.query("SELECT connection, rows FROM provider_search_usage WHERE day = '2026-10-06'")
    assert [(r["connection"], r["rows"]) for r in rows] == [("fake", 18984)], "the day's search use carried over"
    # Renamed later: what was written before connections still belongs to the first name.
    renamed = make_fleet(db, FakeProvider(), providers={"main": {"type": "fake"}})
    renamed.claim_records()
    assert renamed.connection == "main" and renamed.legacy_connection == "fake"
    assert legacy_connection(db, "anything") == "fake"
    assert db.query("SELECT COUNT(*) AS n FROM provider_search_usage")[0]["n"] == 1, "carried over once"


def test_a_change_to_the_connections_is_planned_restarted_retyped_and_never_strands_a_host():
    """The supervisor rents through the accounts it started with; the plan says so, asks for an
    account turned on to be typed again, and refuses dropping a provider the pool still holds
    hosts at — nothing could stop them billing (D129, D133)."""
    from gpm_server.configplan import RentedNow, plan_changes

    old_shape = pool_config(provider="fake")
    assert not [c for c in plan_changes(old_shape, pool_config(providers={"fake": {"type": "fake"}}))
                if c.kind == "providers"], "the same connection in the new shape is no change"
    added = pool_config(providers={"fake": {"type": "fake"}, "other": {"type": "other"}})
    (change,) = [c for c in plan_changes(old_shape, added) if c.kind == "providers"]
    assert change.needs_restart and change.restarts == "supervisor"
    assert "when the supervisor restarts" in change.detail and "other will be searched" in change.detail
    assert change.requires_retype == "other", "renting somewhere new is typed again"
    assert change.refused is None
    holding = [RentedNow("rented-a", 0.4, provider="fake")]
    (dropped,) = [c for c in plan_changes(added, pool_config(providers={"other": {"type": "other"}}), holding)
                  if c.kind == "providers"]
    assert dropped.refused and "rented-a" in dropped.refused
    renamed = pool_config(providers={"main": {"type": "fake"}})
    (kept,) = [c for c in plan_changes(old_shape, renamed, holding) if c.kind == "providers"]
    assert kept.refused is None, "renamed, the provider is still there: its hosts are found by it"
