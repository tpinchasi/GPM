"""Taking a configured host out of the pool from the console (D99).

The owner: "I want to be able to remove non-rented hosts — the laptop, for example — from the
pool." Two ways, because they are different wishes. **Out of service** is the reversible one:
`disabled` in the file, the host kept, back with one click. **Remove** takes it out of the file,
and is typed to confirm. Both are edits to the file in place (D51), through the plan and the
file's own rules — so taking out the only host that serves a model is refused, with the reason.

Against a real file; nothing here rents or spends.
"""

import textwrap

import httpx
import pytest
import yaml
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.db import Database
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_hosts_admin"


def pool_file(per_host: str = "all") -> str:
    declared = per_host == "declared"
    return textwrap.dedent(
        f"""
        pool:
          name: hosts
          model_set: [a, b]
          probe_interval_s: 3600
          models_per_host: {per_host}

        auth:
          admin_keys: ["{ADMIN_KEY}"]
          app_keys: ["gpma_hosts_app"]

        hosts:
          - id: laptop
            kind: local
            workers: 3   # measured
            {"models: [a]" if declared else ""}
            transport: {{ type: http, base_url: "http://127.0.0.1:1" }}
          - id: desk
            kind: local
            {"models: [b]" if declared else ""}
            transport: {{ type: http, base_url: "http://127.0.0.1:2" }}

        rented:
          provider: fake
          {"models: [b]" if declared else ""}
          offer_policy: {{ min_disk_gb: 10, max_all_in_hourly: 0.60 }}
        """
    ).strip()


def serve(tmp_path, text):
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(text + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    return supervisor, server, loop, database, config_path


@pytest.fixture
def pool(tmp_path):
    supervisor, server, loop, database, path = serve(tmp_path, pool_file())
    try:
        yield supervisor, server.base_url, path
    finally:
        server.stop()
        loop.stop()
        database.close()


@pytest.fixture
def declared_pool(tmp_path):
    supervisor, server, loop, database, path = serve(tmp_path, pool_file("declared"))
    try:
        yield supervisor, server.base_url, path
    finally:
        server.stop()
        loop.stop()
        database.close()


def admin(url):
    return httpx.Client(base_url=url, headers={"Authorization": f"Bearer {ADMIN_KEY}"}, timeout=30)


def hosts_in(path):
    return {h["id"]: h for h in yaml.safe_load(path.read_text())["hosts"]}


# --- out of service, and back ---


def test_a_host_taken_out_of_service_stays_in_the_file_and_stops_serving(pool):
    supervisor, url, path = pool
    with admin(url) as http:
        answer = http.patch("/pool/config/hosts/laptop", json={"disabled": True})
        assert answer.status_code == 200, answer.text
        status = http.get("/pool/status").json()

    assert hosts_in(path)["laptop"]["disabled"] is True
    assert "desk" in hosts_in(path) and "disabled" not in hosts_in(path)["desk"]
    assert {h["host_id"]: h["state"] for h in status["hosts"]}["laptop"] == "disabled"
    assert "workers: 3   # measured" in path.read_text(), "its comment is kept"


def test_it_returns_to_service_with_one_click(pool):
    supervisor, url, path = pool
    with admin(url) as http:
        assert http.patch("/pool/config/hosts/laptop", json={"disabled": True}).status_code == 200
        assert http.patch("/pool/config/hosts/laptop", json={"disabled": False}).status_code == 200
    assert hosts_in(path)["laptop"]["disabled"] is False
    assert not next(h for h in supervisor.config.hosts if h.id == "laptop").disabled


def test_the_only_host_serving_a_model_cannot_be_taken_out_of_service(declared_pool):
    """Otherwise that model would have nowhere to go, and every request for it would fail."""
    _, url, path = declared_pool
    before = path.read_text()
    with admin(url) as http:
        answer = http.patch("/pool/config/hosts/laptop", json={"disabled": True})
    assert answer.status_code == 400
    assert "would be held by none" in answer.json()["detail"] and "a" in answer.json()["detail"]
    assert path.read_text() == before


# --- removing ---


def test_removing_is_confirmed_by_typing_the_hosts_id_and_says_what_will_happen(pool):
    _, url, path = pool
    before = path.read_text()
    with admin(url) as http:
        answer = http.request("DELETE", "/pool/config/hosts/laptop", json={})
    assert answer.status_code == 400 and answer.json()["error"] == "not_confirmed"
    (removal,) = [c for c in answer.json()["changes"] if c["kind"] == "host_removed"]
    assert removal["requires_retype"] is True and removal["value"] == "laptop"
    assert "drained" in removal["detail"]
    assert path.read_text() == before, "nothing written until it is confirmed"


def test_a_confirmed_removal_takes_the_host_out_of_the_file_and_the_pool(pool):
    supervisor, url, path = pool
    with admin(url) as http:
        answer = http.request("DELETE", "/pool/config/hosts/laptop", json={"confirm": "laptop"})
        assert answer.status_code == 200, answer.text
        status = http.get("/pool/status").json()
    assert set(hosts_in(path)) == {"desk"}
    assert [h.id for h in supervisor.config.hosts] == ["desk"]
    assert [h["host_id"] for h in status["hosts"]] == ["desk"]


def test_the_wrong_id_typed_does_not_remove_it(pool):
    _, url, path = pool
    with admin(url) as http:
        answer = http.request("DELETE", "/pool/config/hosts/laptop", json={"confirm": "desk"})
    assert answer.status_code == 400
    assert "laptop" in hosts_in(path)


def test_the_last_configured_host_can_go_when_the_pool_can_still_rent(pool):
    supervisor, url, path = pool
    with admin(url) as http:
        for host in ("laptop", "desk"):
            assert http.request("DELETE", f"/pool/config/hosts/{host}", json={"confirm": host}).status_code == 200
    assert yaml.safe_load(path.read_text())["hosts"] == []
    assert supervisor.config.hosts == []


def test_removing_the_only_host_serving_a_model_is_refused_before_it_is_offered(declared_pool):
    """Refused by the file's rules first, so nobody is asked to type a name for a change that
    cannot be made."""
    _, url, path = declared_pool
    before = path.read_text()
    with admin(url) as http:
        answer = http.request("DELETE", "/pool/config/hosts/laptop", json={})
    assert answer.status_code == 400 and answer.json()["error"] == "invalid_config"
    assert "would be held by none" in answer.json()["detail"]
    assert path.read_text() == before


# --- what these are not for ---


def test_a_host_that_is_not_configured_is_not_found(pool):
    _, url, _ = pool
    with admin(url) as http:
        assert http.patch("/pool/config/hosts/nobody", json={"disabled": True}).status_code == 404
        assert http.request("DELETE", "/pool/config/hosts/nobody", json={"confirm": "nobody"}).status_code == 404


def test_a_rented_host_is_released_not_removed(pool):
    supervisor, url, _ = pool
    with admin(url) as http:
        http.post("/pool/hosts/prepare", json={"max_spend": 1.0, "max_hours": 1, "when_ready": "park"})
        (rented,) = supervisor.fleet.hosts
        answer = http.request("DELETE", f"/pool/config/hosts/{rented}", json={"confirm": rented})
    assert answer.status_code == 400 and answer.json()["error"] == "rented_host"
    assert "Rented capacity" in answer.json()["detail"]


def test_the_app_key_cannot_do_either(pool):
    _, url, _ = pool
    with httpx.Client(base_url=url, headers={"Authorization": "Bearer gpma_hosts_app"}, timeout=30) as http:
        assert http.patch("/pool/config/hosts/laptop", json={"disabled": True}).status_code in (401, 403)
        assert http.request("DELETE", "/pool/config/hosts/laptop", json={"confirm": "laptop"}).status_code in (401, 403)
