"""Workloads, keys and volumes in the database (D115, D116). A key is kept as its hash only,
shown once; a rotated-out key keeps working for its grace; an ended workload's keys reach
nothing."""

import time

import pytest
from gpm_server.db import Database
from gpm_server.keys import fingerprint, match
from gpm_server.workload_store import Volume, Workload, WorkloadStore


@pytest.fixture
def store(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield WorkloadStore(database)
    database.close()


def workload(name="research", state="preparing"):
    now = time.time()
    return Workload(
        name=name, model="gemma4:31b", builds={"gemma4:31b": "org/Big"}, latency_s=20, parallel=16,
        kind="roi", lease_id=f"lease-{name}", state=state, workers_per_host=8, hosts_at_start=2,
        plan={"reasons": ["x"]}, created_at=now, updated_at=now,
    )


def test_a_workload_round_trips(store):
    store.create(workload())
    got = store.get("research")
    assert got.builds == {"gemma4:31b": "org/Big"} and got.plan == {"reasons": ["x"]} and got.active
    assert [w.name for w in store.active()] == ["research"]


def test_ending_is_recorded_and_final(store):
    store.create(workload())
    store.set_state("research", "ended")
    got = store.get("research")
    assert got.state == "ended" and got.ended_at is not None and not got.active
    assert store.active() == []
    with pytest.raises(ValueError):
        store.set_state("research", "sleeping")


def test_a_key_is_kept_as_its_hash_and_shown_once(store, tmp_path):
    store.create(workload())
    key_id, key = store.mint_key("research")
    assert key.startswith("gpmw_") and len(key) == 5 + 64
    raw = (tmp_path / "gpm.sqlite3").read_bytes()
    assert key.encode() not in raw, "the key itself never reaches the file"
    grants = store.grants()
    assert grants[fingerprint(key)].workload == "research"
    assert match(key, {h: g.workload for h, g in grants.items()}) == "research"
    assert match("gpmw_" + "0" * 64, {h: g.workload for h, g in grants.items()}) is None
    assert all("hashed" not in k for k in store.keys("research")), "listing never shows a hash"


def test_a_rotated_key_works_for_its_grace_only(store):
    store.create(workload())
    _, old = store.mint_key("research")
    _, new = store.rotate_key("research", grace_s=0.2)
    assert {fingerprint(old), fingerprint(new)} <= set(store.grants())
    time.sleep(0.3)
    assert fingerprint(old) not in store.grants() and fingerprint(new) in store.grants()


def test_expired_keys_reach_nothing(store):
    store.create(workload())
    _, key = store.mint_key("research")
    store.expire_keys("research")
    assert fingerprint(key) not in store.grants()


def test_the_revision_moves_with_every_change(store):
    before = store.revision()
    store.create(workload())
    after_create = store.revision()
    store.mint_key("research")
    after_key = store.revision()
    assert before < after_create < after_key


def test_volumes_are_listed_until_deleted(store):
    store.add_volume(Volume("v-1", "research", "m-1", 40, 0.01, "lease-research", time.time()))
    assert [v.volume_id for v in store.volumes("research")] == ["v-1"]
    store.volume_deleted("v-1")
    assert store.volumes("research") == []
    assert store.volumes("research", live_only=False)[0].deleted_at is not None
