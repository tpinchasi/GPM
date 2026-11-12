"""Hosted model APIs added from the console, like a rented provider (S13, D171).

A hosted API is chosen from a list (or given as another API of either format), its key typed
once and tested against its `/models` — which spends nothing — and its models offered under the
names apps ask for, each with its price, pinned to it. The key is kept by the supervisor in the
credential store, read by the router, used only while the host points where it was saved for,
and never answered, logged, recorded or written into the configuration.

Against a fake provider; nothing here spends money.
"""

import json
import os
import stat
import textwrap
import time

import httpx
import pytest
import yaml
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server import hosted_credentials
from gpm_server.config import load_config
from gpm_server.db import Database
from gpm_server.router.app import create_app
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_hosted_admin"
APP_KEY = "gpma_hosted_app"
AUTH = {"Authorization": f"Bearer {ADMIN_KEY}"}
WRONG = "fake-hosted-key-wrong-4242"
CLAUDE = {"input": 3.0, "cached_input": 0.3, "cache_write": 3.75, "output": 15.0}
GPT = {"input": 2.0, "cached_input": 0.2, "output": 8.0}


def pool_file(tmp_path) -> str:
    return textwrap.dedent(
        f"""
        pool:
          name: hosted
          model_set: [llama]
          models_per_host: declared
          probe_interval_s: 3600
          host_table_poll_s: 0.05
          bursts:
            hosted: {{ after_s: 0.1, budget: {{ per_day: 2.0 }} }}

        request_log: {tmp_path / 'gpm.sqlite3'}

        auth:
          admin_keys: ["{ADMIN_KEY}"]
          app_keys: ["{APP_KEY}"]

        catalog:
          llama: {{ variants: [{{ tag: "llama:7b" }}] }}

        hosts:
          - id: laptop
            kind: local
            transport: {{ type: http, base_url: "http://127.0.0.1:1" }}
            models: [llama]
        """
    ).strip() + "\n"


@pytest.fixture
def pool(tmp_path):
    loop = BackgroundLoop()
    fake = FakeHostedApi(models=("provider-claude", "provider-gpt"))
    provider = ServerHandle(fake.app, loop)
    path = tmp_path / "pool.yaml"
    path.write_text(pool_file(tmp_path))
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    loop.run(supervisor.start())
    control = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    router_app = create_app(supervisor.config, database=database, config_path=path)
    router = ServerHandle(router_app, loop)

    class Pool:
        pass

    p = Pool()
    p.loop, p.fake, p.provider_url, p.path, p.db, p.supervisor = loop, fake, provider.base_url, path, database, supervisor
    p.control, p.router, p.router_app, p.tmp = control.base_url, router.base_url, router_app, tmp_path
    try:
        yield p
    finally:
        router.stop()
        control.stop()
        provider.stop()
        loop.stop()
        database.close()


def api(pool, method, path, **kwargs):
    return httpx.request(method, f"{pool.control}{path}", headers=AUTH, timeout=30, **kwargs)


def settle(pool, until, timeout=30.0):
    """A supervisor pass and the router's next read, until `until()` holds — and the router has
    read the file as it is now, which it does on its own clock.

    Returns the moment `until()` holds, so the long timeout costs a passing run nothing. Found on
    CI (twice, never on a developer's machine): a wait of 10 s ran out. What each side holds is
    said on failure, so the next one names what was still waiting.
    """
    deadline = time.monotonic() + timeout
    while True:
        hosts_now = {h.id for h in load_config(pool.path).hosts}
        if {h.id for h in pool.router_app.state.pool.config.hosts} != hosts_now:
            if time.monotonic() > deadline:
                raise AssertionError(f"never settled: the router never read the file's hosts {sorted(hosts_now)}")
            time.sleep(0.05)
            continue
        pool.loop.run(pool.supervisor.pass_once())
        pool.loop.run(pool.router_app.state.pool.registry.refresh())
        if until():
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"never settled: {where_things_stand(pool)}")
        time.sleep(0.1)


def where_things_stand(pool) -> str:
    """Each side's view of every host — never a key, only whether one is held."""
    registry = pool.router_app.state.pool.registry
    router = {h.host_id: h.state.value for h in pool.router_app.state.pool.hosts}
    supervisor = {i: (h.state.value, h.last_error) for i, h in pool.supervisor.hosts.items()}
    stamps = {i: s is not None for i, s in registry._key_stamps.items()}
    on_disk = [(h.id, (pool.tmp / "provider-credentials" / "hosted" / f"{h.id}.json").exists())
               for h in pool.supervisor.config.hosts if h.kind == "hosted"]
    return (f"router hosts {router}; router holds a key for {stamps}; supervisor hosts {supervisor}; "
            f"key files {on_disk}")


def router_has_key(pool) -> bool:
    """The router has read a typed key for the hosted API — on its own clock, from the store."""
    return pool.router_app.state.pool.registry._key_stamps.get("anthropic") is not None


def chat(pool, model):
    return httpx.post(f"{pool.router}/v1/chat/completions", timeout=30, headers={"Authorization": f"Bearer {APP_KEY}"},
                      json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


def everything_written(pool) -> str:
    parts = [pool.path.read_text()]
    versions = pool.path.with_suffix(".yaml.versions")
    if versions.exists():
        parts += [p.read_text() for p in versions.iterdir() if p.is_file()]
    for table in ("events", "hosts", "request_log"):
        try:
            parts.append(json.dumps([dict(r) for r in pool.db.query(f"SELECT * FROM {table}")], default=str))
        except Exception:  # noqa: BLE001 - a table this version does not have
            pass
    parts.append(json.dumps(pool.supervisor.events.recent(500), default=str))
    return "\n".join(parts)


def add_claude(pool, **extra):
    return api(pool, "POST", "/pool/hosted", json={
        "id": "anthropic", "engine": "anthropic-api", "base_url": f"{pool.provider_url}/v1", "credential": KEY,
        "models": [{"name": "claude", "tag": "provider-claude", "price": CLAUDE}], **extra})


# --- listing and testing ---


def test_the_providers_screen_lists_the_choices_and_needs_no_rented_capacity(pool):
    answer = api(pool, "GET", "/pool/hosted")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    ids = {p["id"] for p in body["presets"]}
    assert {"anthropic", "openai", "groq", "openai-compatible", "anthropic-compatible"} <= ids
    assert all(p["base_url"].startswith("https://") for p in body["presets"] if not p["custom"])
    assert body["hosts"] == [] and body["budget"] == {"after_s": 0.1, "per_request": None, "per_day": 2.0, "per_month": None}
    assert pool.supervisor.config.rented is None, "a pool that rents nothing still adds hosted APIs"


def test_a_key_is_tested_against_the_providers_models_and_never_said_again(pool):
    url = f"{pool.provider_url}/v1"
    good = api(pool, "POST", "/pool/hosted/test", json={"engine": "openai-api", "base_url": url, "credential": KEY})
    assert good.status_code == 200 and good.json()["ok"] and good.json()["models"] == ["provider-claude", "provider-gpt"]
    bad = api(pool, "POST", "/pool/hosted/test", json={"engine": "openai-api", "base_url": url, "credential": WRONG})
    assert bad.status_code == 200 and not bad.json()["ok"]
    assert "refused the key" in json.dumps(bad.json()) and WRONG not in bad.text
    none = api(pool, "POST", "/pool/hosted/test", json={"engine": "openai-api", "base_url": url})
    assert none.status_code == 400 and none.json()["error"] == "no_credential", "no key of the pool's goes to an address typed in"
    plain = api(pool, "POST", "/pool/hosted/test", json={"engine": "openai-api", "base_url": "http://example.com/v1",
                                                          "credential": KEY})
    assert plain.status_code == 400 and plain.json()["error"] == "bad_address"
    assert KEY not in everything_written(pool)


# --- adding, using, changing, removing ---


def test_a_hosted_api_added_from_the_console_serves_with_the_key_typed_in(pool):
    planned = add_claude(pool, plan_only=True)
    assert planned.status_code == 200, planned.text
    assert any(c["kind"] == "host_added" for c in planned.json()["changes"])
    assert not (pool.tmp / "provider-credentials" / "hosted" / "anthropic.json").exists(), "a plan stores nothing"

    added = add_claude(pool)
    assert added.status_code == 200, added.text
    config = yaml.safe_load(pool.path.read_text())
    host = next(h for h in config["hosts"] if h["id"] == "anthropic")
    assert host["kind"] == "hosted" and host["engine"] == "anthropic-api" and host["capabilities"] == ["hosted:anthropic"]
    assert "bearer_env" not in host["transport"], "the key is not in the file"
    assert config["catalog"]["claude"]["variants"] == [
        {"tag": "provider-claude", "engine": "anthropic-api", "requires": ["hosted:anthropic"], "price": CLAUDE}]
    assert "claude" in config["pool"]["model_set"]
    key_file = pool.tmp / "provider-credentials" / "hosted" / "anthropic.json"
    if os.name != "nt":
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(key_file.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(key_file.parent.parent.stat().st_mode) == 0o700

    settle(pool, lambda: any(h.host_id == "anthropic" and h.state.value == "ready"
                             for h in pool.router_app.state.pool.hosts))
    answer = httpx.post(f"{pool.router}/v1/messages", timeout=30,
                        headers={"Authorization": f"Bearer {APP_KEY}", "anthropic-version": "2023-06-01"},
                        json={"model": "claude", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
    assert answer.status_code == 200, answer.text
    assert pool.fake.seen[-1]["headers"]["authorization"] == f"Bearer {KEY}", "the router sent the key typed in"

    listed = api(pool, "GET", "/pool/hosted").json()["hosts"][0]
    assert listed["credential"]["source"] == "stored" and listed["provider"] is None and listed["format"] == "Anthropic"
    assert [m["name"] for m in listed["models"]] == ["claude"] and listed["models"][0]["pinned"]
    assert KEY not in everything_written(pool) and KEY not in json.dumps(api(pool, "GET", "/pool/hosted").json())


def test_a_key_replaced_or_removed_takes_effect_without_a_restart(pool):
    assert add_claude(pool, credential=None).status_code == 200
    described = api(pool, "GET", "/pool/hosted").json()["hosts"][0]["credential"]
    assert described["source"] == "missing" and described["can_type"]
    settle(pool, lambda: any(h.host_id == "anthropic" for h in pool.router_app.state.pool.hosts))
    assert chat(pool, "claude").status_code != 200, "no key, no answer"

    refused = api(pool, "PUT", "/pool/hosted/anthropic/credential", json={"credential": WRONG})
    assert refused.status_code == 400 and refused.json()["error"] == "credential_refused" and WRONG not in refused.text
    assert api(pool, "PUT", "/pool/hosted/anthropic/credential", json={"credential": KEY}).status_code == 200
    settle(pool, lambda: router_has_key(pool) and any(h.host_id == "anthropic" and h.state.value == "ready"
                                                      for h in pool.router_app.state.pool.hosts))
    assert chat(pool, "claude").status_code == 200, "the router read the new key from the store"

    assert api(pool, "DELETE", "/pool/hosted/anthropic/credential").status_code == 200
    settle(pool, lambda: not router_has_key(pool) and all(h.state.value != "ready" for h in pool.router_app.state.pool.hosts if h.host_id == "anthropic"))
    assert api(pool, "GET", "/pool/hosted").json()["hosts"][0]["credential"]["source"] == "missing"


def test_a_key_removed_after_a_reload_read_it_is_dropped(pool):
    """Found on CI, after the test above failed four times in a month: the router's file watcher
    noted "no key", the reload it started read the key saved meanwhile, and the key's removal then
    looked like no change — the router kept sending a removed key until a restart. The router now
    compares the key its client holds with the store at each read of the table."""
    assert add_claude(pool, credential=None).status_code == 200
    settle(pool, lambda: any(h.host_id == "anthropic" for h in pool.router_app.state.pool.hosts))
    registry = pool.router_app.state.pool.registry
    config = load_config(pool.path)
    host = next(h for h in config.hosts if h.id == "anthropic")
    store = hosted_credentials.HostedKeys(config)

    async def reload_reading_the_new_key():
        store.put("anthropic", KEY, hosted_credentials.endpoint(host))
        registry.apply_config(config)
        assert router_has_key(pool)
        store.remove("anthropic")
        await registry.refresh()

    pool.loop.run(reload_reading_the_new_key())
    assert not router_has_key(pool), "the removed key was dropped at the next read of the table"
    client = registry.by_id["anthropic"].client
    assert not ({"authorization", "x-api-key"} & {k.lower() for k in client.headers}), "and is no longer sent"


def test_a_typed_key_is_never_sent_where_the_host_was_moved_by_hand(pool):
    assert add_claude(pool).status_code == 200
    raw = yaml.safe_load(pool.path.read_text())
    next(h for h in raw["hosts"] if h["id"] == "anthropic")["transport"] = {"type": "https", "base_url": "https://elsewhere.example/v1"}
    config = load_config_text(pool, yaml.safe_dump(raw))
    host = next(h for h in config.hosts if h.id == "anthropic")
    assert hosted_credentials.stored(config, host) is None
    assert hosted_credentials.describe(config, host)["source"] == "withheld"
    client = hosted_credentials.client(config, host)
    assert "authorization" not in {k.lower() for k in client.headers}


def load_config_text(pool, text):
    other = pool.tmp / "moved.yaml"
    other.write_text(text)
    return load_config(other)


def test_models_and_prices_are_changed_and_the_api_removed_with_what_only_it_served(pool):
    assert add_claude(pool).status_code == 200
    two = [{"name": "claude", "tag": "provider-claude", "price": {**CLAUDE, "output": 20.0}},
           {"name": "claude-fast", "tag": "provider-claude", "price": CLAUDE}]
    changed = api(pool, "PATCH", "/pool/hosted/anthropic", json={"models": two, "workers": 8})
    assert changed.status_code == 200, changed.text
    config = yaml.safe_load(pool.path.read_text())
    assert config["catalog"]["claude"]["variants"][0]["price"]["output"] == 20.0
    assert {"claude", "claude-fast"} <= set(config["pool"]["model_set"])
    assert next(h for h in config["hosts"] if h["id"] == "anthropic")["workers"] == 8

    unconfirmed = api(pool, "DELETE", "/pool/hosted/anthropic", json={})
    assert unconfirmed.status_code == 400 and unconfirmed.json()["error"] == "not_confirmed"
    assert "claude-fast" in unconfirmed.json()["changes"][0]["detail"]
    gone = api(pool, "DELETE", "/pool/hosted/anthropic", json={"confirm": "anthropic"})
    assert gone.status_code == 200, gone.text
    config = yaml.safe_load(pool.path.read_text())
    assert all(h["id"] != "anthropic" for h in config["hosts"])
    assert "claude" not in config["catalog"] and config["pool"]["model_set"] == ["llama"]
    assert not (pool.tmp / "provider-credentials" / "hosted" / "anthropic.json").exists()


def test_one_model_id_at_two_providers_is_priced_as_each_says(pool):
    assert add_claude(pool).status_code == 200
    other = api(pool, "POST", "/pool/hosted", json={
        "id": "reseller", "engine": "openai-api", "base_url": f"{pool.provider_url}/v1", "credential": KEY,
        "models": [{"name": "gpt", "tag": "provider-gpt", "price": GPT},
                   {"name": "claude", "tag": "provider-claude", "price": {"input": 9.0, "output": 9.0}}]})
    assert other.status_code == 200, other.text
    variants = yaml.safe_load(pool.path.read_text())["catalog"]["claude"]["variants"]
    assert [(v["requires"], v["price"]["input"]) for v in variants] == [(["hosted:anthropic"], 3.0), (["hosted:reseller"], 9.0)]


def test_what_cannot_be_priced_or_named_is_refused_in_words(pool):
    no_write = api(pool, "POST", "/pool/hosted", json={
        "id": "anthropic", "engine": "anthropic-api", "base_url": f"{pool.provider_url}/v1",
        "models": [{"name": "claude", "tag": "provider-claude", "price": {"input": 3.0, "output": 15.0}}]})
    assert no_write.status_code == 400 and "cache-write" in no_write.json()["detail"]
    unpriced = api(pool, "POST", "/pool/hosted", json={
        "id": "x", "engine": "openai-api", "base_url": f"{pool.provider_url}/v1", "models": [{"tag": "provider-gpt"}]})
    assert unpriced.status_code == 400 and unpriced.json()["error"] == "bad_price"
    bad_name = api(pool, "POST", "/pool/hosted", json={"id": "Bad Name", "engine": "openai-api",
                                                        "base_url": f"{pool.provider_url}/v1"})
    assert bad_name.status_code == 400 and bad_name.json()["error"] == "bad_name"
    taken = api(pool, "POST", "/pool/hosted", json={"id": "laptop", "engine": "openai-api",
                                                     "base_url": f"{pool.provider_url}/v1"})
    assert taken.status_code == 409


def test_a_key_is_refused_over_plain_http_from_another_machine(pool):
    assert add_claude(pool, credential=None).status_code == 200
    app = create_control_app(pool.supervisor, pool.supervisor.config)
    transport = httpx.ASGITransport(app=app, client=("10.0.0.5", 4321))

    async def send():
        async with httpx.AsyncClient(transport=transport, base_url="http://pool.example") as client:
            return await client.put("/pool/hosted/anthropic/credential", json={"credential": KEY},
                                    headers={**AUTH, "host": "pool.example"})

    answer = pool.loop.run(send())
    assert answer.status_code == 403 and KEY not in answer.text


def test_the_hosted_budget_is_set_from_the_console(pool):
    answer = api(pool, "PATCH", "/pool/hosted-budget", json={"per_day": 3.5, "per_month": 40, "after_s": 2})
    if answer.status_code == 400 and answer.json().get("error") == "not_confirmed":
        value = next(c["value"] for c in answer.json()["changes"] if c.get("requires_retype"))
        answer = api(pool, "PATCH", "/pool/hosted-budget", json={"per_day": 3.5, "per_month": 40, "after_s": 2,
                                                                  "confirm": value})
    assert answer.status_code == 200, answer.text
    burst = yaml.safe_load(pool.path.read_text())["pool"]["bursts"]["hosted"]
    assert burst["budget"] == {"per_day": 3.5, "per_month": 40.0} and burst["after_s"] == 2.0
    refused = api(pool, "PATCH", "/pool/hosted-budget", json={"per_day": None, "per_month": None})
    assert refused.status_code == 400 and refused.json()["error"] == "bad_budget"


# --- the first rented provider, from a pool that rents nothing (the Rented capacity dead end) ---


def test_a_pool_that_rents_nothing_may_add_its_first_provider_with_renting_set_up_at_once(pool, monkeypatch):
    from gpm_server.providers import FakeProvider, presentation

    market = FakeProvider()
    market.accepted_credentials = {"fake-rented-credential-0001": "home"}
    pool.supervisor.accounts.factory = lambda type_name, settings: market.twin()
    installed = {"fake": {"package": "gpm-server", "version": "test", "loaded": True, **presentation(FakeProvider, "fake")}}
    monkeypatch.setattr("gpm_server.supervisor.providers_api.installed_plugins", lambda: installed)
    monkeypatch.setattr("gpm_server.supervisor.connections.installed_plugins", lambda: installed)

    listed = api(pool, "GET", "/pool/providers")
    assert listed.status_code == 200, listed.text
    assert listed.json()["connections"] == [] and listed.json()["renting"] is False
    assert [p["type"] for p in listed.json()["plugins"]] == ["fake"]

    # The account alone is accepted and rents nothing until renting is set up (D172,
    # tests/test_renting_setup.py); `first` still sets renting up in the same request.
    body = {"name": "market", "type": "fake", "credential": "fake-rented-credential-0001"}
    body["first"] = {"min_disk_gb": 60, "max_all_in_hourly": 0.8, "engine": "ollama", "models": ["llama"]}
    answer = api(pool, "POST", "/pool/providers", json=body)
    if answer.status_code == 400 and answer.json().get("error") == "not_confirmed":
        value = next(c["value"] for c in answer.json()["changes"] if c.get("requires_retype"))
        answer = api(pool, "POST", "/pool/providers", json={**body, "confirm": value})
    assert answer.status_code == 200, answer.text
    assert not any(c.get("needs_restart") for c in answer.json()["changes"]), "renting starts at once (D173)"
    rented = yaml.safe_load(pool.path.read_text())["rented"]
    assert rented["providers"] == {"market": {"type": "fake"}}
    assert rented["engine"] == "ollama" and rented["models"] == ["llama"]
    assert rented["offer_policy"] == {"min_disk_gb": 60.0, "max_all_in_hourly": 0.8}
    assert (pool.tmp / "provider-credentials" / "fake.json").exists()
    assert "fake-rented-credential-0001" not in everything_written(pool)

    # In use at once, with no restart (D173): listed, its typed-in credential handed to it, and
    # changed like any running account.
    assert pool.supervisor.fleet is not None and "market" in pool.supervisor.fleet.providers
    running = api(pool, "GET", "/pool/providers").json()
    assert running["renting"] is True and running["restart_needed"] is False
    assert [c["connection"] for c in running["connections"]] == ["market"]
    assert running["connections"][0]["credential"]["source"] == "stored"
    # Turning off the only account is refused by the file's own rule, no longer by a restart.
    assert "must be enabled" in api(pool, "PATCH", "/pool/providers/market", json={"enabled": False}).json()["detail"]


# --- the console ---

from fakes.browser import a_browser, open_page  # noqa: E402


def _click(page, text, scope="document"):
    page.js(f"[...{scope}.querySelectorAll('button')].find(b => b.textContent.trim() === {json.dumps(text)}).click(); true")


@pytest.mark.skipif(a_browser() is None, reason="no browser here")
def test_the_providers_screen_adds_a_hosted_api_like_a_rented_account(pool):
    with open_page(f"{pool.control}/ui/#providers") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.until("document.body.textContent.includes('No provider yet')", within=30, what="the screen")

        # One Add provider, one list: machines to rent and hosted APIs side by side (the owner
        # found two halves and a second list on Rented capacity confusing).
        _click(page, "Add provider")
        page.until("[...document.querySelectorAll('.plugin-tile')].some(t => t.textContent.includes('Another Anthropic-format API'))",
                   within=10, what="the choices")
        groups = page.js("[...document.querySelectorAll('#provider-dialog h3.choice-group')].map(h => h.textContent)")
        assert [g.split(" —")[0] for g in groups] == ["Rent GPU machines", "Hosted model APIs"]
        page.js("[...document.querySelectorAll('.plugin-tile')].find(t => t.textContent.includes('Another Anthropic-format API')).click(); true")
        page.until("!!document.querySelector('#hosted-url')", within=10, what="the key step")
        page.js(f"""
            const set = (sel, v) => {{ const f = document.querySelector(sel); f.value = v; f.dispatchEvent(new Event('input', {{bubbles: true}})); }};
            set('#hosted-id', 'claude-api'); set('#hosted-url', {json.dumps(pool.provider_url + '/v1')});
            set('#provider-dialog input[data-secret]', {json.dumps(KEY)});
            document.querySelector('#provider-dialog form').requestSubmit(); true""")
        page.until("document.getElementById('provider-dialog').textContent.includes('2 model(s) offered')", within=20, what="the test")
        _click(page, "Choose models", "document.getElementById('provider-dialog')")
        page.until("!!document.querySelector('.model-row[data-tag=\"provider-claude\"]')", within=10, what="the models")
        page.js("""
            const row = document.querySelector('.model-row[data-tag="provider-claude"]');
            const box = row.querySelector('input[type=checkbox]'); box.checked = true; box.dispatchEvent(new Event('change'));
            const name = row.querySelector('input.mono'); name.value = 'claude';
            const prices = row.querySelectorAll('input.price'); [3, 0.3, 3.75, 15].forEach((v, i) => { prices[i].value = v; });
            true""")
        _click(page, "Continue", "document.getElementById('provider-dialog')")
        page.until("document.getElementById('provider-dialog').textContent.includes('hosted:') || "
                   "document.getElementById('provider-dialog').textContent.includes('host')", within=20, what="the plan")
        page.until("!document.querySelector('#provider-dialog .wizard-nav button.primary').disabled", within=20, what="the plan to load")
        _click(page, "Add it", "document.getElementById('provider-dialog')")
        page.until("!document.getElementById('provider-dialog').open", within=20, what="the save")
        page.until("!!document.querySelector('.hosted-card[data-hosted=\"claude-api\"]')", within=20, what="the new card")
        card = page.js("document.querySelector('.hosted-card').textContent")
        assert "claude" in card and "provider-claude" in card and "$3.75" in card
        assert KEY not in page.js("document.documentElement.outerHTML")

        # Requests at once, changed from the card (the owner could see "at most 4" and not change it).
        page.js("[...document.querySelectorAll('.hosted-card button')].find(b => b.textContent === 'Settings').click(); true")
        page.until("!!document.querySelector('#hosted-settings-workers')", within=10, what="the settings")
        page.js("document.querySelector('#hosted-settings-workers').value = 9;"
                "document.querySelector('#provider-dialog form').requestSubmit(); true")
        page.until("!document.getElementById('provider-dialog').open", within=20, what="the settings saved")
        page.until("document.querySelector('.hosted-card').textContent.includes('at most 9 requests')", within=20, what="the new limit")
    config = yaml.safe_load(pool.path.read_text())
    assert next(h for h in config["hosts"] if h["id"] == "claude-api")["workers"] == 9
    assert config["catalog"]["claude"]["variants"][0]["requires"] == ["hosted:claude-api"]
    assert (pool.tmp / "provider-credentials" / "hosted" / "claude-api.json").exists()
