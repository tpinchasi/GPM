from pathlib import Path

import pytest
from gpm_server.config import ConfigError, PoolConfig, SecretMissing, load_config
from gpm_server.keys import fingerprint

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "server" / "examples" / "pool.yaml"

BASE = {
    "pool": {"model_set": ["m1"]},
    "auth": {"app_keys": ["k"]},
    "hosts": [
        {
            "id": "local-1",
            "kind": "local",
            "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"},
        }
    ],
}


def config(**overrides):
    raw = {**BASE, **overrides}
    return PoolConfig.model_validate(raw)


def test_minimal_config_loads():
    parsed = config()
    assert parsed.hosts[0].routing_priority == 0
    assert parsed.hosts[0].workers == 1
    assert parsed.pool.queue_timeout_s == 30.0


def test_priority_defaults_by_kind():
    parsed = config(
        hosts=[
            {"id": "a", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}},
            {"id": "b", "kind": "fixed-remote", "transport": {"type": "https", "base_url": "https://b"}},
        ]
    )
    assert [h.routing_priority for h in parsed.hosts] == [0, 10]


def test_queue_timeout_must_stay_under_client_first_byte_budget():
    with pytest.raises(ValueError, match="queue_timeout_s"):
        config(pool={"model_set": ["m1"], "queue_timeout_s": 300, "client_time_to_first_byte_s": 300})


def test_plain_http_off_loopback_is_refused():
    with pytest.raises(ValueError, match="allow_insecure"):
        config(
            hosts=[
                {
                    "id": "a",
                    "kind": "fixed-remote",
                    "transport": {"type": "http", "base_url": "http://10.0.0.5:11434"},
                }
            ]
        )


def test_plain_http_off_loopback_allowed_when_said_deliberately():
    parsed = config(
        hosts=[
            {
                "id": "a",
                "kind": "fixed-remote",
                "transport": {
                    "type": "http",
                    "base_url": "http://10.0.0.5:11434",
                    "allow_insecure": True,
                },
            }
        ]
    )
    assert parsed.hosts[0].transport.allow_insecure


def test_transport_type_must_match_the_url_scheme():
    with pytest.raises(ValueError, match="scheme"):
        config(
            hosts=[
                {"id": "a", "kind": "fixed-remote", "transport": {"type": "https", "base_url": "http://x"}}
            ]
        )


def test_catalog_names_must_be_in_the_model_set():
    with pytest.raises(ValueError, match="model set"):
        config(catalog={"other": {"variants": [{"tag": "other"}]}})


def test_host_without_a_usable_variant_is_refused_by_name():
    with pytest.raises(ValueError, match="local-1"):
        config(catalog={"m1": {"variants": [{"tag": "m1-mlx", "requires": ["apple-silicon"]}]}})


def test_host_with_the_capability_is_accepted():
    parsed = config(
        catalog={"m1": {"variants": [{"tag": "m1-mlx", "requires": ["apple-silicon"]}]}},
        hosts=[
            {
                "id": "local-1",
                "kind": "local",
                "capabilities": ["apple-silicon"],
                "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"},
            }
        ],
    )
    assert parsed.hosts[0].capabilities == ["apple-silicon"]


def test_host_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        config(
            hosts=[
                {"id": "a", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}},
                {"id": "a", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:2"}},
            ]
        )


def test_listening_off_loopback_requires_tls():
    with pytest.raises(ValueError, match="TLS"):
        config(listen={"host": "0.0.0.0", "port": 8080})


def test_a_pool_without_an_app_key_is_refused():
    parsed = config(auth={"app_keys": []})
    with pytest.raises(ConfigError, match="app key"):
        parsed.auth.app_hashes()


def test_app_keys_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("TEST_POOL_KEYS", "one, two")
    parsed = config(auth={"app_keys_env": "TEST_POOL_KEYS"})
    assert parsed.auth.app_hashes() == {fingerprint("one"), fingerprint("two")}


def test_app_and_admin_keys_are_never_interchangeable(monkeypatch):
    parsed = config(auth={"app_keys": ["app-one"], "admin_keys": ["admin-one"]})
    assert parsed.auth.app_hashes() == {fingerprint("app-one")}
    assert parsed.auth.admin_hashes() == {fingerprint("admin-one")}
    assert not parsed.auth.app_hashes() & parsed.auth.admin_hashes()


def test_the_control_api_off_loopback_requires_tls():
    with pytest.raises(ValueError, match="TLS"):
        config(control={"host": "0.0.0.0", "port": 8081})


def test_a_missing_secret_is_named(monkeypatch):
    monkeypatch.delenv("TEST_HOST_KEY", raising=False)
    parsed = config(
        hosts=[
            {
                "id": "a",
                "kind": "fixed-remote",
                "transport": {"type": "https", "base_url": "https://x", "bearer_env": "TEST_HOST_KEY"},
            }
        ]
    )
    with pytest.raises(SecretMissing, match="TEST_HOST_KEY"):
        parsed.hosts[0].transport.auth_headers()


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError):
        config(pool={"model_set": ["m1"], "typo_here": 1})


def test_load_config_reports_a_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="no configuration file"):
        load_config(tmp_path / "absent.yaml")


def test_load_config_reads_the_shipped_example():
    """The example must parse, including its rented section — an example that does not load is
    worse than none."""
    parsed = load_config(EXAMPLE_CONFIG)
    assert parsed.pool.model_set
    assert parsed.rented.connection_name == "vast" and parsed.rented.connection.type == "vast"
    assert parsed.rented.max_all_in_hourly == 0.60
    assert parsed.rented.disk_gb == 60
    assert parsed.limits.max_rented_hosts == 1
    assert parsed.auth.app_keys_file and parsed.auth.admin_keys_file
