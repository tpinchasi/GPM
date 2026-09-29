"""The router serves under the configuration file as it is now, not as it was at start.

Found live: a model added to the set from the console was served by a rented host within
minutes, but the router kept answering `/pool/status` with the set it had read when it started,
and the app — which checks names against that list — refused the model without asking. The
supervisor had applied the edit; the router never read it. Only a restart of the router made
the two agree. Against a real router process on a real socket; no engine, no provider.
"""

import textwrap
import time

import httpx
import pytest
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.db import Database
from gpm_server.router.app import create_app

APP_KEY = "gpma_follow_app"
NEW_KEY = "gpma_follow_rotated"


def pool_yaml(tmp_path, model_set=("a",), app_keys=(APP_KEY,), queue_timeout=30) -> str:
    keys = ", ".join(f'"{k}"' for k in app_keys)
    models = ", ".join(model_set)
    return textwrap.dedent(
        f"""
        pool:
          name: follow
          model_set: [{models}]
          probe_interval_s: 3600
          host_table_poll_s: 0.1
          queue_timeout_s: {queue_timeout}
        auth:
          app_keys: [{keys}]
        hosts:
          - id: laptop
            kind: local
            transport: {{ type: http, base_url: "http://127.0.0.1:1" }}
        request_log: {tmp_path / 'gpm.sqlite3'}
        """
    ).strip()


@pytest.fixture
def router(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(pool_yaml(tmp_path))
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    server = ServerHandle(create_app(load_config(path), database, config_path=path), loop)
    try:
        yield server.base_url, path
    finally:
        server.stop()
        loop.stop()
        database.close()


def status(url, key=APP_KEY) -> httpx.Response:
    return httpx.get(f"{url}/pool/status", headers={"Authorization": f"Bearer {key}"}, timeout=5)


def edit(path, text):
    """Write, and move the file's time on so a same-second write is still seen as a change."""
    path.write_text(text)
    stamp = time.time() + 1
    import os

    os.utime(path, (stamp, stamp))


def until(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_a_model_added_to_the_set_is_listed_without_a_restart(router, tmp_path):
    url, path = router
    assert status(url).json()["model_set"] == ["a"]
    edit(path, pool_yaml(tmp_path, model_set=("a", "b")))
    assert until(lambda: status(url).json()["model_set"] == ["a", "b"])


def test_a_limit_changed_in_the_file_is_the_one_in_force(router, tmp_path):
    url, path = router
    edit(path, pool_yaml(tmp_path, queue_timeout=7))
    assert until(lambda: status(url).json()["limits"]["queue_timeout_s"] == 7)


def test_a_rotated_app_key_takes_effect_and_the_old_one_stops_working(router, tmp_path):
    url, path = router
    edit(path, pool_yaml(tmp_path, app_keys=(NEW_KEY,)))
    assert until(lambda: status(url, NEW_KEY).status_code == 200)
    assert status(url, APP_KEY).status_code == 401


def test_a_file_that_does_not_load_leaves_the_router_as_it_was(router, tmp_path):
    url, path = router
    edit(path, "pool: [this is not a pool")
    time.sleep(0.5)
    assert status(url).json()["model_set"] == ["a"]
    # And the next good edit is still followed.
    edit(path, pool_yaml(tmp_path, model_set=("a", "c")))
    assert until(lambda: status(url).json()["model_set"] == ["a", "c"])


def test_a_key_revoked_in_its_file_stops_working_without_a_restart(tmp_path):
    """`gpm key revoke` rewrites the key file and leaves the configuration alone."""
    from gpm_server.keys import KeyStore

    store = KeyStore(tmp_path / "app.keys")
    kept, _ = store.create("app")
    revoked, record = store.create("app")
    path = tmp_path / "pool.yaml"
    path.write_text(pool_yaml(tmp_path).replace(f'app_keys: ["{APP_KEY}"]',
                                                f"app_keys_file: {tmp_path / 'app.keys'}"))
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    server = ServerHandle(create_app(load_config(path), database, config_path=path), loop)
    try:
        assert status(server.base_url, revoked).status_code == 200
        store.revoke(record.key_id)
        assert until(lambda: status(server.base_url, revoked).status_code == 401)
        assert status(server.base_url, kept).status_code == 200
    finally:
        server.stop()
        loop.stop()
        database.close()
