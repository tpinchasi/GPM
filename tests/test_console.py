"""The console, through the same API it uses.

docs/spec/console-and-control-api.md. The console is one more client of the control API, so
these tests drive the endpoints the page calls — which is also what proves the roadmap's exit
criterion: every console action is possible from the CLI, a mistyped ceiling is caught by plan
before apply, and the app key is refused.
"""

import textwrap
import time
from pathlib import Path

import httpx
import pytest
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.configplan import ConfigStore, RentedNow, plan_changes
from gpm_server.db import Database
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_console_admin"
APP_KEY = "gpma_console_app"
MODEL = "m1"

BASE_YAML = textwrap.dedent(
    f"""
    pool:
      name: console
      model_set: [{MODEL}]
      probe_interval_s: 3600
    auth:
      app_keys: ["{APP_KEY}"]
      admin_keys: ["{ADMIN_KEY}"]
    hosts:
      - id: local-1
        kind: local
        workers: 2
        transport: {{type: http, base_url: "http://127.0.0.1:1"}}
    limits: {{max_rented_hosts: 1, max_hourly_burn: 1.00}}
    rented:
      provider: fake
      bidding: {{bid_ceiling: 0.60}}
      scale: {{scale_up_after_s: 0}}
    """
).strip()


@pytest.fixture
def console(tmp_path):
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(BASE_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        yield supervisor, server.base_url, config_path, loop
    finally:
        server.stop()
        loop.stop()
        database.close()


def client(url, key=ADMIN_KEY):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.Client(base_url=url, headers=headers, timeout=30)


def edited(text, old, new):
    assert old in text
    return text.replace(old, new)


# --- the page ---


def test_the_page_is_served_and_holds_no_key_of_its_own(console):
    _, url, _, _ = console
    with httpx.Client(base_url=url, timeout=30) as anonymous:
        page = anonymous.get("/ui/")
        assert page.status_code == 200  # the page itself needs no key: it asks for one
        assert "gpm" in page.text.lower()
        script = anonymous.get("/ui/app.js").text

    # The key is entered into the page and kept in memory; it is never persisted.
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "document.cookie" not in script
    # ...and every call it makes carries it as a header.
    assert "Authorization: `Bearer ${ADMIN_KEY}`" in script


def test_the_page_cannot_read_anything_without_the_key(console):
    _, url, _, _ = console
    with httpx.Client(base_url=url, timeout=30) as anonymous:
        assert anonymous.get("/pool/status").status_code == 401
        assert anonymous.get("/pool/config").status_code == 401


def test_every_screens_data_call_is_refused_to_the_app_key(console):
    """An app that can request a completion must not be able to read or spend here."""
    _, url, _, _ = console
    paths = ["/pool/status", "/pool/events", "/pool/leases", "/pool/plan", "/pool/config",
             "/pool/config/history", "/pool/market/preview", "/pool/account"]
    with client(url, key=APP_KEY) as http:
        for path in paths:
            assert http.get(path).status_code == 403, path


def test_status_carries_what_the_overview_draws(console):
    supervisor, url, _, loop = console
    supervisor.fleet.open_lease(workers=4, max_hours=2, max_spend=2.0, allow_rent=True)
    loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))

    with client(url) as http:
        status = http.get("/pool/status").json()

    host = status["hosts"][0]
    assert {"host_id", "kind", "transport", "priority", "state", "workers", "busy"} <= set(host)
    lease = status["open_leases"][0]
    assert {"lease_id", "max_spend", "spent_enforced_on", "dollars_left", "stops_at"} <= set(lease)
    rented = status["rented"][0]
    assert {"bid_hourly", "storage_hourly", "estimated_spend", "hours_held", "hardware"} <= set(rented)


def test_status_carries_the_model_set_and_catalog_the_models_screen_draws(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        status = http.get("/pool/status").json()
        assert status["model_set"] == [MODEL]

        # And it follows a reload, rather than reporting what the process started with.
        changed = edited(path.read_text(), f"model_set: [{MODEL}]", "model_set: [m2]")
        http.put("/pool/config", json={"text": changed, "version": status_version(http)})
        assert http.get("/pool/status").json()["model_set"] == ["m2"]


def status_version(http):
    return http.get("/pool/config").json()["version"]


def test_limits_shown_follow_a_reload_too(console):
    _, url, path, _ = console
    with client(url) as http:
        assert http.get("/pool/status").json()["limits"]["max_hourly_burn"] == 1.00
        changed = edited(path.read_text(), "max_hourly_burn: 1.00", "max_hourly_burn: 0.25")
        http.put("/pool/config", json={"text": changed, "version": status_version(http)})
        assert http.get("/pool/status").json()["limits"]["max_hourly_burn"] == 0.25


def test_the_live_stream_sends_decisions_and_status(console):
    supervisor, url, _, _ = console
    supervisor.events.record("test_event", "something happened", numbers={"n": 1})

    with client(url) as http:
        with http.stream("GET", "/pool/events/stream", timeout=20) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            kinds, chunk = set(), ""
            for text in response.iter_text():
                chunk += text
                for frame in chunk.split("\n\n"):
                    if "event: decision" in frame:
                        kinds.add("decision")
                        assert "something happened" in frame
                    if "event: status" in frame:
                        kinds.add("status")
                if {"decision", "status"} <= kinds:
                    break
    assert kinds == {"decision", "status"}


# --- plan: nothing is applied blind ---


def test_a_mistyped_ceiling_is_caught_by_plan_and_must_be_retyped(console):
    """The roadmap's exit criterion: 6.0 typed for 0.60 is not applied silently."""
    supervisor, url, path, loop = console
    supervisor.fleet.open_lease(workers=4, max_hours=2, max_spend=2.0, allow_rent=True)
    loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))

    typo = edited(path.read_text(), "bid_ceiling: 0.60", "bid_ceiling: 6.0")
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": typo}).json()

    raised = [c for c in plan["changes"] if c["kind"] == "bid_ceiling_raised"]
    assert raised, plan
    assert raised[0]["requires_retype"] is True
    assert raised[0]["value"] == "6.000"
    assert "$0.600 to $6.000" in raised[0]["detail"]
    # And nothing has changed until Apply.
    assert supervisor.config.rented.bidding.bid_ceiling == 0.60


def test_plan_says_which_running_host_a_lowered_ceiling_would_release(console):
    supervisor, url, path, loop = console
    supervisor.fleet.open_lease(workers=4, max_hours=2, max_spend=2.0, allow_rent=True)
    loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))
    host = next(iter(supervisor.fleet.hosts.values()))

    lowered = edited(path.read_text(), "bid_ceiling: 0.60", "bid_ceiling: 0.05")
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": lowered}).json()

    change = [c for c in plan["changes"] if c["kind"] == "bid_ceiling_lowered"][0]
    assert host.host_id in change["detail"]
    assert "drained and released" in change["detail"]
    assert change["requires_retype"] is False  # tightening needs no ceremony


def test_plan_refuses_a_file_that_would_not_load(console):
    _, url, path, _ = console
    broken = path.read_text() + "\n  this: is not valid yaml: at all:\n"
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": broken}).json()
    assert plan["errors"]
    assert plan["changes"] == []


def test_plan_spends_nothing_and_creates_nothing(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        http.post("/pool/config/plan", json={"text": path.read_text()})
    assert supervisor.fleet.provider.instances == {}


# --- apply, and what it does to a running pool ---


def test_applying_a_host_makes_the_running_pool_follow(console):
    supervisor, url, path, loop = console
    added = edited(
        path.read_text(),
        '    transport: {type: http, base_url: "http://127.0.0.1:1"}',
        '    transport: {type: http, base_url: "http://127.0.0.1:1"}\n'
        '  - id: local-2\n    kind: local\n    workers: 1\n'
        '    transport: {type: http, base_url: "http://127.0.0.1:2"}',
    )
    with client(url) as http:
        version = http.get("/pool/config").json()["version"]
        assert http.put("/pool/config", json={"text": added, "version": version}).status_code == 200

    assert "local-2" in supervisor.hosts
    assert len(supervisor.config.hosts) == 2


def test_removing_a_host_retires_it_from_the_published_table(console):
    supervisor, url, path, loop = console
    loop.run(supervisor.pass_once())
    assert any(row.host_id == "local-1" for row in supervisor.table.all())

    without = edited(
        path.read_text(),
        '  - id: local-1\n    kind: local\n    workers: 2\n'
        '    transport: {type: http, base_url: "http://127.0.0.1:1"}\n',
        "",
    ).replace("hosts:\n", "hosts: []\n")
    with client(url) as http:
        version = http.get("/pool/config").json()["version"]
        assert http.put("/pool/config", json={"text": without, "version": version}).status_code == 200

    assert supervisor.hosts == {}
    assert all(row.host_id != "local-1" for row in supervisor.table.all())


def test_a_write_based_on_a_stale_version_is_refused(console):
    _, url, path, _ = console
    with client(url) as http:
        stale = http.get("/pool/config").json()["version"]
        first = edited(path.read_text(), "max_hourly_burn: 1.00", "max_hourly_burn: 0.50")
        assert http.put("/pool/config", json={"text": first, "version": stale}).status_code == 200

        second = edited(path.read_text(), "max_rented_hosts: 1", "max_rented_hosts: 1 ")
        refused = http.put("/pool/config", json={"text": second, "version": stale})

    assert refused.status_code == 409
    assert "changed since it was read" in refused.json()["detail"]


def test_a_file_that_does_not_load_never_reaches_the_running_pool(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        version = http.get("/pool/config").json()["version"]
        refused = http.put("/pool/config", json={"text": "pool: {}", "version": version})
    assert refused.status_code == 400
    assert supervisor.config.pool.name == "console"  # unchanged
    assert "local-1" in supervisor.hosts


def test_editing_the_file_by_hand_is_noticed(console):
    supervisor, url, path, loop = console
    path.write_text(edited(path.read_text(), "max_hourly_burn: 1.00", "max_hourly_burn: 0.25"))
    # The file's mtime must differ from the one recorded at start.
    import os
    os.utime(path, (time.time() + 1, time.time() + 1))

    loop.run(supervisor.pass_once())

    assert supervisor.config.limits.max_hourly_burn == 0.25
    assert "config_applied" in [e["kind"] for e in supervisor.events.recent(20)]


def test_history_keeps_what_was_applied_and_rollback_restores_it(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        original = http.get("/pool/config").json()
        changed = edited(original["text"], "max_hourly_burn: 1.00", "max_hourly_burn: 0.25")
        http.put("/pool/config", json={"text": changed, "version": original["version"]})
        assert supervisor.config.limits.max_hourly_burn == 0.25

        history = http.get("/pool/config/history").json()["versions"]
        assert history and history[0]["version"] == original["version"]

        http.post("/pool/config/rollback", json={"version": original["version"]})

    assert supervisor.config.limits.max_hourly_burn == 1.00


# --- the market preview with unsaved values ---


def test_the_market_preview_can_use_the_forms_unsaved_values(console):
    _, url, _, _ = console
    with client(url) as http:
        saved = http.get("/pool/market/preview").json()
        tightened = http.post(
            "/pool/market/preview", json={"offer_policy": {"min_gpu_memory_gb": 999}}
        ).json()

    assert saved["passed"] == 1
    assert tightened["passed"] == 0  # moving a ceiling and watching "1 pass" become "0 pass"
    assert "gpu memory" in " ".join(tightened["rejected_by_reason"])


def test_an_unsaved_policy_that_makes_no_sense_is_a_400_not_something_the_pool_acts_on(console):
    _, url, _, _ = console
    with client(url) as http:
        response = http.post("/pool/market/preview", json={"bidding": {"bid_ceiling": "not a number"}})
    assert response.status_code == 400


# --- the plan function itself, as a pure function ---


def parse(text):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pool.yaml"
        path.write_text(text)
        return load_config(path)


def test_listen_changes_are_reported_as_needing_a_restart():
    current = parse(BASE_YAML)
    candidate = parse(BASE_YAML + "\nlisten: {host: 127.0.0.1, port: 9999}\n")
    change = [c for c in plan_changes(current, candidate) if c.kind == "listen"][0]
    assert change.needs_restart
    assert "Rented hosts are unaffected" in change.detail


def test_switching_on_allow_insecure_must_be_retyped():
    current = parse(BASE_YAML)
    candidate = parse(BASE_YAML.replace(
        'transport: {type: http, base_url: "http://127.0.0.1:1"}',
        'transport: {type: http, base_url: "http://10.0.0.5:1", allow_insecure: true}',
    ))
    kinds = {c.kind: c for c in plan_changes(current, candidate)}
    assert kinds["allow_insecure"].requires_retype == "local-1"
    assert "clear text" in kinds["allow_insecure"].detail


def test_a_model_set_change_says_hosts_are_re_prepared():
    current = parse(BASE_YAML)
    candidate = parse(BASE_YAML.replace(f"model_set: [{MODEL}]", f"model_set: [{MODEL}, m2]"))
    change = [c for c in plan_changes(current, candidate) if c.kind == "model_set"][0]
    assert "re-prepared one at a time" in change.detail


def test_raising_the_host_count_must_be_retyped_but_lowering_need_not_be():
    current = parse(BASE_YAML)
    raised = parse(BASE_YAML.replace("max_rented_hosts: 1", "max_rented_hosts: 5"))
    lowered = parse(BASE_YAML.replace("max_rented_hosts: 1", "max_rented_hosts: 0"))

    up = [c for c in plan_changes(current, raised) if c.kind == "max_rented_hosts_raised"][0]
    assert up.requires_retype == "5"
    down = [c for c in plan_changes(current, lowered, [RentedNow("rented-a", 0.1)]) if c.kind == "max_rented_hosts_lowered"][0]
    assert down.requires_retype is None
    assert "1 host(s) beyond it will be drained" in down.detail


def test_no_change_is_no_changes():
    assert plan_changes(parse(BASE_YAML), parse(BASE_YAML)) == []


def test_a_worker_count_change_says_which_way_it_goes():
    current = parse(BASE_YAML)
    raised = parse(BASE_YAML.replace("workers: 2", "workers: 4"))
    lowered = parse(BASE_YAML.replace("workers: 2", "workers: 1"))
    assert "start idle at once" in [c for c in plan_changes(current, raised) if c.kind == "workers_raised"][0].detail
    assert "drain after their current request" in [c for c in plan_changes(current, lowered) if c.kind == "workers_lowered"][0].detail


# --- the store ---


def test_the_store_refuses_a_candidate_that_does_not_load(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(BASE_YAML)
    store = ConfigStore(path)
    assert store.validate(BASE_YAML) == []
    assert store.validate("pool: {}")
    assert path.read_text() == BASE_YAML  # validation never touches the file


def test_applying_is_atomic_and_keeps_the_outgoing_version(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(BASE_YAML)
    store = ConfigStore(path)

    changed = BASE_YAML.replace("max_hourly_burn: 1.00", "max_hourly_burn: 0.25")
    store.apply(changed)

    assert "0.25" in path.read_text()
    assert len(store.history()) == 1
    assert store.text_of(store.history()[0]["version"]) == BASE_YAML
    assert not list(path.parent.glob("*.new"))  # nothing half-written left behind


def test_history_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setattr("gpm_server.configplan.HISTORY_LIMIT", 3)
    path = tmp_path / "pool.yaml"
    path.write_text(BASE_YAML)
    store = ConfigStore(path)
    for burn in ("0.90", "0.80", "0.70", "0.60", "0.50"):
        store.apply(BASE_YAML.replace("max_hourly_burn: 1.00", f"max_hourly_burn: {burn}"))
        time.sleep(0.002)
    assert len(store.history()) == 3
