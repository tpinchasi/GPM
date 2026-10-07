"""Provider credentials typed into the console, and provider accounts managed from it (D130,
D134-D136; docs/spec/providers.md §5, §8).

A credential is sent once, tested against the provider, kept by the supervisor in an owner-only
file bound to the endpoint it was saved for, and handed to the plug-in — never answered, logged,
recorded or written into the configuration. Against the fake market; nothing rents or spends.
"""

import json
import logging
import os
import stat
import textwrap

import httpx
import pytest
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import ProviderConnection, load_config
from gpm_server.credentials import CredentialStore, CredentialStoreUnsafe
from gpm_server.db import Database
from gpm_server.providers import FakeProvider, default_offer, presentation
from gpm_server.providers.fake import FakeInstance
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_creds_admin"
#: Obviously not a real credential, and long enough to be found anywhere it leaks.
GOOD = "fake-credential-good-0123456789"
OTHER = "fake-credential-other-account-9876"
BAD = "fake-credential-refused-5555"
AUTH = {"Authorization": f"Bearer {ADMIN_KEY}"}


def pool_file() -> str:
    return textwrap.dedent(
        f"""
        pool:
          name: creds
          model_set: [a]
          probe_interval_s: 3600

        auth:
          admin_keys: ["{ADMIN_KEY}"]
          app_keys: ["gpma_creds_app"]

        hosts:
          - id: laptop
            kind: local
            transport: {{ type: http, base_url: "http://127.0.0.1:1" }}

        rented:
          provider: fake   # the old shape: the console's first save rewrites it
          offer_policy: {{ min_disk_gb: 10, max_all_in_hourly: 0.60 }}
        """
    ).strip()


@pytest.fixture
def pool(tmp_path, monkeypatch):
    path = tmp_path / "pool.yaml"
    path.write_text(pool_file() + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    market = supervisor.fleet.providers["fake"]
    market.accepted_credentials = {GOOD: "home", OTHER: "elsewhere"}
    # Every plug-in built for a test call or a new connection is a client of this one market.
    supervisor.accounts.factory = lambda type_name, settings: market.twin()
    classes = {"fake": FakeProvider, "other": type("OtherMarket", (FakeProvider,), {"offered": True, "display_name": "Other"})}
    installed = {name: {"package": "gpm-server", "version": "test", "loaded": True, **presentation(cls, name)}
                 for name, cls in classes.items()}
    monkeypatch.setattr("gpm_server.supervisor.providers_api.installed_plugins", lambda: installed)
    monkeypatch.setattr("gpm_server.supervisor.connections.installed_plugins", lambda: installed)
    monkeypatch.setattr("gpm_server.supervisor.providers_api.plugin_presentation", lambda name: installed[name])
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        yield supervisor, market, server.base_url, path, database
    finally:
        server.stop()
        loop.stop()
        database.close()


def everything_written(supervisor, path, database) -> str:
    """Every place the pool writes: the file and its versions, the database, the decision log."""
    parts = [path.read_text()]
    versions = path.with_suffix(path.suffix + ".versions")
    if versions.exists():
        parts += [p.read_text() for p in versions.iterdir() if p.is_file()]
    for table in ("events", "pool_meta", "hosts"):
        try:
            parts.append(json.dumps([dict(r) for r in database.query(f"SELECT * FROM {table}")], default=str))
        except Exception:  # noqa: BLE001 - a table this version does not have
            pass
    parts.append(json.dumps(supervisor.events.recent(500), default=str))
    return "\n".join(parts)


def hold_a_host(supervisor, market):
    """The pool holds an instance at its connection, as after a rental."""
    market.instances["i-held"] = FakeInstance("i-held", f"{supervisor.fleet.label_prefix}x", "m-1", 0.1,
                                              default_offer(), None)
    supervisor.fleet.held_at = lambda name: [("rented-held", "i-held")] if name == "fake" else []


# --- the store ---


def test_the_store_is_owner_only_written_whole_and_never_followed_through_a_link(tmp_path):
    store = CredentialStore(tmp_path / "creds")
    record = store.put("fake", GOOD, {"endpoint": None})
    assert GOOD not in repr(record), "never in a repr"
    assert stat.S_IMODE(os.stat(tmp_path / "creds").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(tmp_path / "creds" / "fake.json").st_mode) == 0o600
    assert [p.name for p in (tmp_path / "creds").iterdir()] == ["fake.json"], "no temporary file left"
    assert store.get("fake").value == GOOD and store.get("fake").bound_to({"endpoint": None})
    store.check()

    os.chmod(tmp_path / "creds" / "fake.json", 0o640)
    with pytest.raises(CredentialStoreUnsafe, match="chmod 600"):
        store.check()
    os.chmod(tmp_path / "creds" / "fake.json", 0o600)

    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"set_at": 0, "endpoint": {}, "value": "x"}))
    os.symlink(elsewhere, tmp_path / "creds" / "linked.json")
    with pytest.raises(CredentialStoreUnsafe, match="link"):
        store.get("linked")
    with pytest.raises(ValueError):
        store.get("../escape")


def test_a_credential_is_resolved_environment_first_then_typed_in_then_the_plugins_own(tmp_path, monkeypatch):
    path = tmp_path / "pool.yaml"
    path.write_text(pool_file() + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        supervisor = Supervisor(load_config(path), database, config_path=str(path))
        accounts, market = supervisor.accounts, supervisor.fleet.providers["fake"]
        conn = ProviderConnection(type="fake")
        assert accounts.resolve(conn, market) == ("missing", None)
        monkeypatch.setenv("GPM_FAKE_PROVIDER_KEY", "from-the-plugins-variable")
        assert accounts.resolve(conn, market) == ("environment", "from-the-plugins-variable")
        accounts.store.put("fake", GOOD, {"endpoint": None})
        assert accounts.resolve(conn, market) == ("stored", GOOD), "typed in beats the plug-in's variable"
        monkeypatch.setenv("MY_KEY", "from-credential-env")
        assert accounts.resolve(ProviderConnection(type="fake", credential_env="MY_KEY"), market) == (
            "environment", "from-credential-env"), "a connection's own variable always wins"

        moved = ProviderConnection(type="fake", settings={"endpoint": "https://elsewhere.invalid"})
        assert accounts.resolve(moved, market) == ("environment", "from-the-plugins-variable")
        assert accounts.store.get("fake") is None, "saved for another endpoint: removed, never sent"
        assert any(e["kind"] == "credential_cleared" for e in supervisor.events.recent(20))

        class Version1:
            pass
        assert accounts.resolve(conn, Version1())[0] == "plugin", "a version-1 plug-in reads its own"
    finally:
        database.close()


# --- the API ---


def test_the_providers_screen_lists_each_account_and_never_a_credential(pool):
    supervisor, market, base, path, database = pool
    market.credential = GOOD
    answer = httpx.get(f"{base}/pool/providers", headers=AUTH)
    assert answer.status_code == 200
    payload = answer.json()
    (fake,) = payload["connections"]
    assert fake["connection"] == "fake" and fake["display_name"] == "Fake market" and fake["enabled"]
    assert fake["credential"]["source"] == "missing" and fake["credential"]["can_type"]
    assert fake["account"]["state"] == "valid"
    assert {p["type"]: p["offered"] for p in payload["plugins"]} == {"fake": False, "other": True}
    assert payload["secure_transport"] is True


def test_a_typed_credential_is_tested_kept_handed_over_and_never_said_again(pool, caplog):
    supervisor, market, base, path, database = pool
    caplog.set_level(logging.DEBUG)
    refused = httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json={"credential": BAD})
    assert refused.status_code == 422 and BAD not in refused.text
    assert supervisor.accounts.store.get("fake") is None, "a refused one is not kept"

    answer = httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json={"credential": GOOD})
    assert answer.status_code == 200, answer.text
    assert answer.json()["credential"]["source"] == "stored" and answer.json()["account"]["state"] == "valid"
    assert market.credential == GOOD, "the running plug-in takes it at once — no restart"
    assert supervisor.accounts.store.get("fake").value == GOOD

    listed = httpx.get(f"{base}/pool/providers", headers=AUTH).text
    for said in (answer.text, listed, everything_written(supervisor, path, database),
                 "\n".join(r.getMessage() for r in caplog.records)):
        assert GOOD not in said and BAD not in said


def test_a_replacement_from_another_account_is_refused_while_the_pool_holds_hosts(pool):
    supervisor, market, base, path, database = pool
    market.credential = GOOD
    hold_a_host(supervisor, market)
    answer = httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json={"credential": OTHER})
    assert answer.status_code == 409 and answer.json()["error"] == "another_account"
    assert OTHER not in answer.text
    assert market.credential == GOOD and supervisor.accounts.store.get("fake") is None, "nothing changed"
    assert httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json={"credential": GOOD}).status_code == 200
    removed = httpx.delete(f"{base}/pool/providers/fake/credential", headers=AUTH)
    assert removed.status_code == 409 and removed.json()["error"] == "holds_hosts", "nothing could watch its hosts"


def test_a_credential_is_refused_over_plain_http_from_another_machine_and_never_echoed(pool):
    supervisor, market, base, path, database = pool
    app = create_control_app(supervisor, supervisor.config)
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 40000))

    async def put(body):
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            return await client.put("/pool/providers/fake/credential", headers=AUTH, json=body)

    import asyncio
    remote = asyncio.run(put({"credential": GOOD}))
    assert remote.status_code == 403 and remote.json()["error"] == "insecure_transport"
    assert GOOD not in remote.text
    for strange in ({"credential": {"nested": GOOD}}, {"credential": GOOD + "\nsecond line"}, {"credential": [GOOD]}):
        answer = httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json=strange)
        assert answer.status_code == 400 and GOOD not in answer.text, "a refusal says what was wrong, never the value"


def test_adding_an_account_is_typed_again_stores_nothing_until_confirmed_and_takes_effect_at_once(pool):
    supervisor, market, base, path, database = pool
    body = {"name": "second", "type": "other", "credential": GOOD}
    first = httpx.post(f"{base}/pool/providers", headers=AUTH, json=body)
    assert first.status_code == 400 and first.json()["error"] == "not_confirmed", first.text
    assert GOOD not in first.text
    assert supervisor.accounts.store.get("other") is None, "nothing stored before it is confirmed"

    done = httpx.post(f"{base}/pool/providers", headers=AUTH, json={**body, "confirm": "second"})
    assert done.status_code == 200, done.text
    assert "second" in supervisor.fleet.providers and "second" in supervisor.fleet.searchable(), "no restart"
    assert supervisor.fleet.providers["second"].credential == GOOD
    text = path.read_text()
    assert "providers:" in text and "provider: fake" not in text, "the file is in the connections shape"
    assert GOOD not in everything_written(supervisor, path, database)

    refused = httpx.post(f"{base}/pool/providers", headers=AUTH,
                         json={"name": "third", "type": "other", "credential": BAD, "confirm": "third"})
    assert refused.status_code in (400, 409, 422) and BAD not in refused.text


def test_a_bad_credential_on_a_new_account_adds_nothing(pool):
    supervisor, market, base, path, database = pool
    before = path.read_text()
    answer = httpx.post(f"{base}/pool/providers", headers=AUTH,
                        json={"name": "second", "type": "other", "credential": BAD, "confirm": "second"})
    assert answer.status_code == 422 and BAD not in answer.text
    assert path.read_text() == before and "second" not in supervisor.fleet.providers


def test_moving_an_account_to_another_endpoint_clears_its_credential(pool):
    """A running plug-in is never rebuilt (D135): the move waits for a restart, and there the
    credential saved for the old endpoint is deleted rather than sent to the new one."""
    supervisor, market, base, path, database = pool
    assert httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json={"credential": GOOD}).status_code == 200
    moved = httpx.patch(f"{base}/pool/providers/fake", headers=AUTH,
                        json={"settings": {"endpoint": "https://elsewhere.invalid"}})
    assert moved.status_code == 200, moved.text
    (change,) = [c for c in moved.json()["changes"] if c["kind"] == "providers"]
    assert change["needs_restart"] and "when the supervisor restarts" in change["detail"]
    assert market.credential == GOOD, "the running plug-in is as it was built"
    view = httpx.get(f"{base}/pool/providers", headers=AUTH).json()["connections"][0]
    assert "restarts" in view["pending"]

    restarted = Supervisor(load_config(path), database, config_path=str(path))
    assert restarted.accounts.store.get("fake") is None, "saved for another endpoint: deleted, never sent"
    assert restarted.fleet.providers["fake"].credential is None


def test_a_saved_account_is_tested_with_its_own_settings_and_reports_each_step(pool):
    supervisor, market, base, path, database = pool
    market.credential = GOOD
    saved = httpx.post(f"{base}/pool/providers/test", headers=AUTH, json={"connection": "fake"}).json()
    assert [s["step"] for s in saved["steps"]] == ["credential", "search", "instances", "capabilities"]
    assert saved["ok"] and all(s["status"] == "passed" for s in saved["steps"])

    hold_a_host(supervisor, market)
    unsaved = httpx.post(f"{base}/pool/providers/test", headers=AUTH,
                         json={"type": "other", "credential": GOOD}).json()
    assert {s["step"]: s["status"] for s in unsaved["steps"]}["instances"] == "warning", \
        "instances already under the pool's label would be swept as strays once added"

    refused = httpx.post(f"{base}/pool/providers/test", headers=AUTH, json={"type": "other", "credential": BAD})
    steps = refused.json()["steps"]
    assert steps[0]["status"] == "failed" and all(s["status"] == "skipped" for s in steps[1:])
    assert BAD not in refused.text


def test_removing_an_account_is_typed_and_refused_while_it_holds_anything(pool):
    supervisor, market, base, path, database = pool
    body = {"name": "second", "type": "other", "credential": GOOD, "confirm": "second"}
    assert httpx.post(f"{base}/pool/providers", headers=AUTH, json=body).status_code == 200
    asked = httpx.request("DELETE", f"{base}/pool/providers/second", headers=AUTH, json={})
    assert asked.status_code == 400 and asked.json()["error"] == "not_confirmed"
    gone = httpx.request("DELETE", f"{base}/pool/providers/second", headers=AUTH, json={"confirm": "second"})
    assert gone.status_code == 200, gone.text
    assert "second" not in supervisor.fleet.providers and supervisor.accounts.store.get("other") is None


def test_a_connection_removed_from_the_file_while_it_holds_a_host_is_kept_until_released(pool):
    supervisor, market, base, path, database = pool
    body = {"name": "second", "type": "other", "confirm": "second"}
    assert httpx.post(f"{base}/pool/providers", headers=AUTH, json=body).status_code == 200
    supervisor.fleet.held_at = lambda name: [("rented-x", "i-x")] if name == "second" else []
    text = path.read_text().replace("    second: { type: \"other\" }\n", "")
    path.write_text(text)
    supervisor.reload_config()
    assert "second" in supervisor.fleet.providers, "still watched and released as usual"
    assert "second" not in supervisor.fleet.searchable(), "never searched again"
    assert any(e["kind"] == "provider_kept" for e in supervisor.events.recent(20))


def test_a_pool_holding_nothing_starts_without_a_credential_and_waits_for_one(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(pool_file() + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    database = Database(tmp_path / "gpm.sqlite3")
    loop = BackgroundLoop()
    try:
        supervisor = Supervisor(load_config(path), database, config_path=str(path))
        supervisor.fleet.providers["fake"].accepted_credentials = {GOOD: "home"}
        loop.run(supervisor.start())
        assert any(e["kind"] == "credential_missing" for e in supervisor.events.recent(50))
        loop.run(supervisor.aclose())
    finally:
        loop.stop()
        database.close()


# --- the screen, in a real browser ---

from fakes.browser import a_browser, open_page  # noqa: E402


@pytest.mark.skipif(a_browser() is None, reason="no browser here")
@pytest.mark.timeout(180)
def test_the_providers_screen_takes_a_credential_and_keeps_nothing_of_it(pool):
    supervisor, market, base, path, database = pool
    with open_page(f"{base}/ui/#rented/providers") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.until("!!document.querySelector('.provider-card')", within=30, what="the account's card")
        assert "Needs a credential" in page.js("document.querySelector('.provider-card').textContent")

        page.js("[...document.querySelectorAll('button')].find(b => b.textContent === 'Type in a credential').click(); true")
        page.until("!!document.querySelector('#provider-dialog input[data-secret]')", within=10, what="the dialog")
        page.js(f"const field = document.querySelector('#provider-dialog input[data-secret]'); field.value = {json.dumps(GOOD)};"
                "field.dispatchEvent(new Event('input', {bubbles: true}));"
                "[...document.querySelectorAll('#provider-dialog button')].find(b => b.textContent === 'Test and save').click(); true")
        page.until("!document.getElementById('provider-dialog').open", within=20, what="the save")
        page.until("document.querySelector('.provider-card').textContent.includes('saved ')", within=20, what="the new state")
        assert market.credential == GOOD

        # Add provider: typed, then abandoned — every field it was typed into is emptied.
        page.js("[...document.querySelectorAll('button')].find(b => b.textContent === 'Add provider').click(); true")
        page.until("!!document.querySelector('.plugin-tile:not([disabled])')", within=10, what="the providers offered")
        page.js("document.querySelector('.plugin-tile:not([disabled])').click(); true")
        page.until("!!document.querySelector('#provider-dialog input[data-secret]')", within=10, what="the account step")
        page.js(f"window.__typed = document.querySelector('#provider-dialog input[data-secret]');"
                f"window.__typed.value = {json.dumps(OTHER)}; document.getElementById('provider-dialog').close(); true")
        page.until("window.__typed.value === '' && !document.getElementById('provider-dialog').children.length",
                   within=5, what="the typed credential dropped as the dialog closed")
        assert OTHER not in page.js("document.documentElement.outerHTML")
        assert GOOD not in page.js("document.documentElement.outerHTML")


def test_another_packages_plugin_is_listed_without_being_loaded(monkeypatch):
    """T14: a plug-in runs only once configuration names it or an operator chooses it — being
    installed is not enough, so the Add provider list reads another package's metadata only."""
    from gpm_server.providers import base

    class Point:
        def __init__(self, package):
            self.dist = type("Dist", (), {"name": package, "version": "1.0"})()
            self.loaded = False

        def load(self):
            self.loaded = True
            return FakeProvider

    theirs, ours = Point("someone-elses-plugin"), Point("gpm-server")
    monkeypatch.setattr(base, "available_providers", lambda: {"theirs": theirs, "ours": ours})
    listed = base.installed_plugins()
    assert listed["theirs"]["loaded"] is False and not theirs.loaded
    assert listed["ours"]["loaded"] is True and listed["ours"]["display_name"] == "Fake market"
    assert base.plugin_presentation("theirs")["display_name"] == "Fake market" and theirs.loaded, "loaded once chosen"


# --- found by the reviews of step 3 ---


def _hold_a_live_host(supervisor, market, connection="fake"):
    import dataclasses

    from gpm_server.providers.base import Instance
    from gpm_server.supervisor.renting import RentedHost

    fleet = supervisor.fleet
    label = f"{fleet.label_prefix}rented-abc"
    market.instances["i-held"] = FakeInstance("i-held", label, "m-1", 0.1, default_offer(), None, state="running")
    fleet.hosts["rented-abc"] = RentedHost(host_id="rented-abc", instance=Instance("i-held", label),
                                           offer=dataclasses.replace(default_offer(), connection=connection),
                                           bid_hourly=0.1, lease_id="l", models=("a",), state="ready")


def test_renaming_a_connection_that_holds_a_host_never_sweeps_it(pool):
    """Found by both reviews: the old name was kept beside the new — two plug-ins on one account —
    and the sweep through the new one destroyed the pool's own host as a stray (D133). A rename
    now waits for a restart, the old plug-in watching its hosts until then."""
    import asyncio

    supervisor, market, base, path, database = pool
    market.accepted_credentials = None
    path.write_text(path.read_text().replace("provider: fake   # the old shape: the console's first save rewrites it",
                                             "providers:\n    fake: { type: fake }"))
    supervisor.reload_config()
    _hold_a_live_host(supervisor, market)
    path.write_text(path.read_text().replace("    fake: { type: fake }", "    renamed: { type: fake }"))
    supervisor.reload_config()
    assert list(supervisor.fleet.providers) == ["fake"], "one plug-in per account, never two"
    assert "restarts" in supervisor.accounts.pending["renamed"]
    asyncio.run(supervisor.fleet.sweep_orphans())
    assert not [e for e in supervisor.events.recent(50) if e["kind"] == "orphan_swept"]
    assert "i-held" in market.instances and supervisor.fleet.held_at("fake")


def test_a_rename_that_also_moves_a_holding_connection_is_refused():
    """Found by the verification review: a rename in the same edit as a new credential source
    escaped the check, though it is the same connection (D136)."""
    from gpm_server.configplan import RentedNow, plan_changes
    from test_provider_connections import pool_config
    were = pool_config(providers={"vast": {"type": "fake", "credential_env": "KEY_A"}})
    become = pool_config(providers={"renamed": {"type": "fake"}})
    (change,) = [c for c in plan_changes(were, become, [RentedNow("rented-a", 0.4, provider="fake")]) if c.kind == "providers"]
    assert change.refused and "release them first" in change.refused
    (plain,) = [c for c in plan_changes(were, pool_config(providers={"renamed": {"type": "fake", "credential_env": "KEY_A"}}),
                                        [RentedNow("rented-a", 0.4, provider="fake")]) if c.kind == "providers"]
    assert plain.refused is None and plain.needs_restart, "a plain rename is allowed, at a restart"


def test_no_request_can_point_the_environments_credential_somewhere(pool, monkeypatch):
    """Found by the reviews: a test, a new connection or an edit naming an endpoint sent the
    supervisor's own secrets there — readable with the admin key alone. A credential from the
    environment now goes only to the plug-in's own default endpoint, live or after a restart
    (T27, T29, D130)."""
    supervisor, market, base, path, database = pool
    monkeypatch.setenv("GPM_FAKE_PROVIDER_KEY", GOOD)
    monkeypatch.setenv("SOME_OTHER_SECRET", "not-for-any-provider-1234")
    for body in ({"type": "other", "settings": {"endpoint": "http://127.0.0.1:9"}},
                 {"type": "other", "credential_env": "SOME_OTHER_SECRET"}):
        answer = httpx.post(f"{base}/pool/providers/test", headers=AUTH, json=body)
        assert answer.status_code == 400 and answer.json()["error"] == "bad_credential", answer.text
    plain = httpx.post(f"{base}/pool/providers/test", headers=AUTH, json={"type": "other"}).json()
    assert plain["steps"][0]["status"] == "passed", "its own variable, to its own default endpoint, is fine"

    pointed = httpx.post(f"{base}/pool/providers", headers=AUTH, json={
        "name": "second", "type": "other", "settings": {"endpoint": "http://127.0.0.1:9"}, "confirm": "second"})
    assert pointed.status_code == 200, pointed.text
    assert supervisor.fleet.providers["second"].credential is None, "nothing from the environment goes there"
    view = {c["connection"]: c for c in httpx.get(f"{base}/pool/providers", headers=AUTH).json()["connections"]}
    assert view["second"]["credential"]["source"] == "withheld"
    assert "not-for-any-provider-1234" not in json.dumps(view) and GOOD not in json.dumps(view)

    # Nor after a restart: the file is the admin key's to write, so where it points is not trusted.
    path.write_text(path.read_text().replace("    fake: { type: \"fake\" }",
                                             '    fake: { type: "fake", settings: { endpoint: "http://127.0.0.1:9" } }'))
    restarted = Supervisor(load_config(path), database, config_path=str(path),
                           provider={"fake": FakeProvider(), "second": FakeProvider()})
    assert restarted.fleet.providers["fake"].credential is None
    assert restarted.fleet.providers["fake"].credentials_handed == [None]


def test_one_provider_never_runs_twice_and_the_pool_always_has_a_default(pool):
    """Found by the last verification: a raw edit giving the running name another provider, and the
    provider a new name, left two plug-ins on one account; a failed build could leave none."""
    supervisor, market, base, path, database = pool
    path.write_text(path.read_text().replace("provider: fake   # the old shape: the console's first save rewrites it",
                                             "providers:\n    fake: { type: other }\n    b: { type: fake }"))
    supervisor.reload_config()
    supervisor.accounts.sync(supervisor.config, supervisor.config, drop=True)
    kinds = [supervisor.accounts.built[n].type for n in supervisor.fleet.providers]
    assert kinds.count("fake") == 1 and "b" not in supervisor.fleet.providers
    assert "another name" in supervisor.accounts.pending["fake"] and "fake" in supervisor.fleet.paused
    assert supervisor.fleet.connection in supervisor.fleet.providers
    assert httpx.get(f"{base}/pool/status", headers=AUTH).status_code == 200


def test_a_provider_still_releasing_its_hosts_is_not_added_again(pool):
    """Found by the last verification: removing a holding connection from the file, then adding it
    back with another account's credential, skipped the check that the new one sees its hosts."""
    supervisor, market, base, path, database = pool
    hold_a_host(supervisor, market)
    path.write_text(path.read_text().replace("provider: fake   # the old shape: the console's first save rewrites it",
                                             "providers:\n    other: { type: other }"))
    supervisor.reload_config()
    assert "fake" in supervisor.fleet.providers, "kept: it holds a host"
    answer = httpx.post(f"{base}/pool/providers", headers=AUTH,
                        json={"name": "fake", "type": "fake", "credential": OTHER, "confirm": "fake"})
    assert answer.status_code == 409 and answer.json()["error"] == "provider_still_running"
    assert supervisor.accounts.store.get("fake") is None


def test_a_connection_holding_hosts_cannot_be_pointed_elsewhere(pool):
    """Found by the security review: a new endpoint or credential source could be another account,
    where its hosts read as gone while they bill (D136)."""
    supervisor, market, base, path, database = pool
    _hold_a_live_host(supervisor, market)
    supervisor.fleet.held_at = lambda name: [("rented-abc", "i-held")] if name == "fake" else []
    moved = httpx.patch(f"{base}/pool/providers/fake", headers=AUTH, json={"settings": {"endpoint": "https://elsewhere.invalid"}})
    assert moved.status_code == 409 and "release them first" in moved.json()["detail"]
    kept = httpx.patch(f"{base}/pool/providers/fake", headers=AUTH, json={"interruption_prior_per_hour": 0.05})
    assert kept.status_code == 200, "what does not move it is still changed"


def test_a_redacted_setting_sent_back_keeps_the_files_value(pool):
    supervisor, market, base, path, database = pool
    path.write_text(path.read_text().replace("provider: fake   # the old shape: the console's first save rewrites it",
                                             "providers:\n    fake: { type: fake, settings: { api_key_env: MY_VARIABLE } }"))
    supervisor.reload_config()
    shown = httpx.get(f"{base}/pool/providers", headers=AUTH).json()["connections"][0]["settings"]
    assert shown == {"api_key_env": "[redacted]"}
    assert httpx.patch(f"{base}/pool/providers/fake", headers=AUTH, json={"settings": shown}).status_code == 200
    assert "MY_VARIABLE" in path.read_text() and "[redacted]" not in path.read_text()


def test_the_old_shape_with_its_settings_first_and_a_one_line_providers_are_edited_safely():
    """Found by the code review: settings written before `provider:` put `providers:` inside the
    next block; a one-line `providers:` took a connection's values a level too high."""
    import yaml
    from gpm_server.config import PoolConfig
    from gpm_server.configplan import CannotEdit, set_values, with_connections

    text = textwrap.dedent("""
        pool: { name: p, model_set: [a] }
        auth: { app_keys: [k] }
        rented:
          provider_settings:
            endpoint: "http://a"
          provider: fake
          offer_policy:
            min_disk_gb: 10
            max_all_in_hourly: 0.60
        """).lstrip()
    rewritten = with_connections(text, PoolConfig.model_validate(yaml.safe_load(text)))
    loaded = PoolConfig.model_validate(yaml.safe_load(rewritten))
    assert loaded.rented.providers["fake"].settings == {"endpoint": "http://a"}
    assert loaded.rented.offer_policy.max_all_in_hourly == 0.60

    one_line = text.replace('  provider_settings:\n    endpoint: "http://a"\n  provider: fake\n',
                            "  providers: { fake: { type: fake } }\n")
    with pytest.raises(CannotEdit, match="one line"):
        set_values(one_line, ("rented", "providers", "fake"), {"enabled": False})


def test_a_credential_that_is_not_plain_text_is_refused_without_a_trace(pool):
    supervisor, market, base, path, database = pool
    answer = httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json={"credential": "ключ-" + GOOD})
    assert answer.status_code == 400 and GOOD not in answer.text


def test_a_credential_directory_others_could_replace_is_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o777)
    with pytest.raises(CredentialStoreUnsafe, match="written by others"):
        CredentialStore(shared / "creds").put("fake", GOOD, {})
    os.chmod(shared, 0o1777)  # sticky: only an entry's owner can rename it
    CredentialStore(shared / "creds").put("fake", GOOD, {})


def test_no_path_around_the_plan_points_a_holding_connection_elsewhere(pool):
    """Found by the last verification: a raw save, a rollback or a followed edit skipped the plan,
    and after a restart the holding connection was another account — its host dropped while it
    billed (D136)."""
    supervisor, market, base, path, database = pool
    holding = lambda name: [("rented-held", "i-held")] if name == "fake" else []  # noqa: E731
    plain, _ = supervisor.store.read()
    moved = plain.replace("provider: fake   # the old shape: the console's first save rewrites it",
                          "providers:\n    fake: { type: fake, credential_env: OTHER_ACCOUNT_KEY }")

    # A raw save.
    supervisor.fleet.held_at = holding
    _, version = supervisor.store.read()
    raw = httpx.put(f"{base}/pool/config", headers=AUTH, json={"text": moved, "version": version})
    assert raw.status_code == 409 and raw.json()["error"] == "change_refused", raw.text
    assert "OTHER_ACCOUNT_KEY" not in path.read_text(), "nothing written"

    # An edit the supervisor follows.
    path.write_text(moved)
    supervisor.reload_config()
    assert supervisor.config.rented.providers["fake"].credential_env is None, "the running configuration is kept"
    assert any(e["kind"] == "config_refused" for e in supervisor.events.recent(20))
    path.write_text(plain)

    # A rollback to a version that says it.
    supervisor.fleet.held_at = lambda name: []
    _, version = supervisor.store.read()
    assert httpx.put(f"{base}/pool/config", headers=AUTH, json={"text": moved, "version": version}).status_code == 200
    _, version = supervisor.store.read()
    assert httpx.put(f"{base}/pool/config", headers=AUTH, json={"text": plain, "version": version}).status_code == 200
    saying = next(v["version"] for v in supervisor.store.history()
                  if "OTHER_ACCOUNT_KEY" in (supervisor.store.text_of(v["version"]) or ""))
    supervisor.fleet.held_at = holding
    back = httpx.post(f"{base}/pool/config/rollback", headers=AUTH, json={"version": saying})
    assert back.status_code == 409 and back.json()["error"] == "change_refused", back.text


def test_a_plugin_that_fails_in_its_own_way_never_takes_the_screen_down(pool):
    """Found by the last verification: a settings typo made the plug-in raise something other than
    a provider error, and every provider route answered 500 from then on."""
    supervisor, market, base, path, database = pool

    async def broken():
        raise TypeError("unsupported operand type(s) for +: 'float' and 'str'")

    market.account = broken
    supervisor.accounts.forget_account("fake")
    assert httpx.get(f"{base}/pool/providers", headers=AUTH).status_code == 200
    tested = httpx.post(f"{base}/pool/providers/test", headers=AUTH, json={"connection": "fake"})
    assert tested.status_code == 200 and tested.json()["steps"][0]["status"] == "failed"
    put = httpx.put(f"{base}/pool/providers/fake/credential", headers=AUTH, json={"credential": GOOD})
    assert put.status_code == 400 and put.json()["error"] == "plugin_failed"


def test_a_logo_address_is_offered_only_over_https():
    """The owner's choice: a provider's logo loaded from its own site (D134) — https only, and
    nothing a page could be steered with."""
    from gpm_server.providers import VastProvider

    assert presentation(VastProvider, "vast")["icon_url"] == "https://vast.ai/icon.png"
    for bad in ("http://vast.ai/icon.png", "javascript:alert(1)", 'https://x/" onerror="x', None, 3):
        plugin = type("P", (FakeProvider,), {"icon_url": bad})
        assert presentation(plugin, "p")["icon_url"] is None, bad
