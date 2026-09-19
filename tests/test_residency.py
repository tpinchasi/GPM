"""Residency: a host holds the model set pinned, or on demand (D39).

docs/spec/hosts-routing-capacity.md §3. A `pinned` host is routable only while the whole set
is loaded; an `on_demand` host once it is on disk, and the engine loads a model on first use.
On neither does a request ever cause a download — the line that stays absolute.
"""

import sqlite3
import textwrap

import pytest
from fakes.harness import EngineSpec, pool_harness
from gpm_server import db as gpm_db
from gpm_server.config import ConfigError, load_config
from gpm_server.configplan import plan_changes

MODEL = "m1"


def chat(pool, model=MODEL):
    with pool.client() as http:
        return http.post(
            "/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )


# --- readiness and routing ---


def test_an_on_demand_host_is_ready_with_the_set_on_disk_and_the_engine_loads_on_first_use():
    """The owner's laptop: models on disk, nothing loaded, other work using the memory."""
    with pool_harness(
        [EngineSpec(id="laptop", resident=set(), available={MODEL}, residency="on_demand")],
        model_set=[MODEL],
    ) as pool:
        host = pool.supervisor.hosts["laptop"]
        assert host.state.value == "ready", host.last_error
        fake = pool.engines["laptop"].fake
        assert MODEL not in fake.resident  # on disk only

        response = chat(pool)

        assert response.status_code == 200, response.text
        assert response.headers["X-GPM-Served-Model"] == MODEL
        assert MODEL in fake.resident  # the engine loaded it, on use
        assert [path for path, _, _ in fake.received] == ["/api/chat"]  # and fetched nothing


def test_a_pinned_host_with_the_set_only_on_disk_stays_out_of_routing():
    """Unchanged behaviour, now stated as the default policy rather than the only one."""
    with pool_harness([EngineSpec(id="box", resident=set(), available={MODEL})], model_set=[MODEL]) as pool:
        host = pool.supervisor.hosts["box"]
        assert host.state.value == "preparing"
        assert "not resident" in host.last_error
        assert chat(pool).status_code == 503
        assert pool.engines["box"].fake.received == []


def test_an_on_demand_host_missing_a_tag_from_disk_stays_out_and_nothing_is_pulled():
    """On demand is not a licence to fetch: a tag that is not on disk keeps the host out until
    the operator pulls it, and no request reaches the engine to ask."""
    with pool_harness(
        [EngineSpec(id="laptop", resident=set(), available=set(), residency="on_demand")],
        model_set=[MODEL],
    ) as pool:
        host = pool.supervisor.hosts["laptop"]
        assert host.state.value == "preparing"
        assert "not on disk" in host.last_error
        assert chat(pool).status_code == 503
        assert pool.engines["laptop"].fake.received == []


def test_an_evicted_model_leaves_an_on_demand_host_in_routing_but_takes_a_pinned_one_out():
    """The live failure that produced D39: the engine's keep-alive expired and the pool went
    dark. Under on_demand the next request simply pays the load again."""
    with pool_harness(
        [
            EngineSpec(id="laptop", resident={MODEL}, residency="on_demand"),
            EngineSpec(id="box", resident={MODEL}, kind="fixed-remote"),
        ],
        model_set=[MODEL],
    ) as pool:
        for engine in pool.engines.values():
            engine.fake.resident.discard(MODEL)  # evicted from memory; still on disk
        pool.reprobe()

        assert pool.supervisor.hosts["laptop"].state.value == "ready"
        assert pool.supervisor.hosts["box"].state.value == "preparing"
        # And the request goes to the one host that may serve it, loading the model there.
        response = chat(pool)
        assert response.status_code == 200
        assert response.headers["X-GPM-Host"] == "laptop"


# --- configuration ---

YAML = textwrap.dedent(
    f"""
    pool:
      name: residency
      model_set: [{MODEL}]
    auth:
      app_keys: ["gpma_x"]
      admin_keys: ["gpmx_x"]
    hosts:
      - id: box
        kind: local
        workers: 2
        transport: {{type: http, base_url: "http://127.0.0.1:1"}}
    """
).strip()


def parse(text, tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(text)
    return load_config(path)


def test_residency_defaults_to_pinned_and_refuses_anything_else(tmp_path):
    assert parse(YAML, tmp_path).hosts[0].residency == "pinned"
    with pytest.raises(ConfigError):
        parse(YAML.replace("workers: 2", "workers: 2\n    residency: sometimes"), tmp_path)


def test_plan_says_what_a_residency_change_means_before_it_is_applied(tmp_path):
    current = parse(YAML, tmp_path)
    on_demand = parse(YAML.replace("workers: 2", "workers: 2\n    residency: on_demand"), tmp_path)

    to_on_demand = [c for c in plan_changes(current, on_demand) if c.kind == "host_residency"][0]
    assert "on disk" in to_on_demand.detail and "first use" in to_on_demand.detail
    assert to_on_demand.requires_retype is None  # no money moves either way

    back = [c for c in plan_changes(on_demand, current) if c.kind == "host_residency"][0]
    assert "pinned" in back.detail and "loaded" in back.detail


# --- the shared database, across versions ---


def test_a_host_table_written_by_an_older_version_gains_the_new_columns_on_open(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` leaves an existing file alone. A pool upgraded in place
    must still open its database, and a row the old version wrote must read as pinned."""
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE hosts (host_id TEXT PRIMARY KEY, kind TEXT NOT NULL, transport_type TEXT NOT NULL, "
        "priority INTEGER NOT NULL, dial_url TEXT NOT NULL, state TEXT NOT NULL, workers INTEGER NOT NULL, "
        "capabilities TEXT NOT NULL, variants TEXT NOT NULL, resident TEXT NOT NULL, lease_id TEXT, "
        "provider_ref TEXT, hourly_rate REAL, last_error TEXT, updated_at REAL NOT NULL)"
    )
    conn.execute(
        "INSERT INTO hosts VALUES ('h', 'local', 'http', 0, 'http://x', 'ready', 1, '[]', '{}', "
        f"'[\"{MODEL}\"]', NULL, NULL, NULL, NULL, 0)"
    )
    conn.commit()
    conn.close()

    database = gpm_db.Database(path)
    try:
        (row,) = gpm_db.HostTable(database).all()
    finally:
        database.close()
    assert row.residency == "pinned"
    assert row.available == frozenset()
    assert row.resident == frozenset({MODEL})
