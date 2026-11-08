"""A provider account before renting is set up (S14, D172).

Found by the owner adding Vast.ai to a pool that rented nothing: Add provider asked for the disk
and the hourly ceiling, defaulted rented machines to vLLM, and was refused because the pool's
models had no vLLM build. Now an account is only an account; renting is set up on its own, on
Rented capacity, offering only models the chosen engine has a build of. Until both ceilings are
named the pool rents nothing: no lease opens, nothing is rented.
"""

from __future__ import annotations

import json
import textwrap

import httpx
import pytest
import yaml
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import PoolConfig, load_config
from gpm_server.configplan import plan_changes
from gpm_server.db import Database
from gpm_server.ledger import LeaseRefused
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_setup_admin"
AUTH = {"Authorization": f"Bearer {ADMIN_KEY}"}
CREDENTIAL = "fake-rented-credential-0001"

BASE = textwrap.dedent(
    f"""
    pool:
      name: setup
      model_set: [chat, big]
      models_per_host: declared
      probe_interval_s: 3600

    auth:
      admin_keys: ["{ADMIN_KEY}"]
      app_keys: ["gpma_setup_app"]

    engine: ollama

    # `chat` exists only as an Ollama tag; `big` also as a hub repository for vLLM.
    catalog:
      chat: {{ variants: [{{ tag: "chat:4b", engine: ollama }}] }}
      big:
        variants:
          - {{ tag: "big:7b", engine: ollama }}
          - {{ tag: "org/big", engine: vllm }}

    hosts:
      - id: laptop
        kind: local
        transport: {{ type: http, base_url: "http://127.0.0.1:1" }}
        models: [chat, big]
    """
).strip() + "\n"

ACCOUNT_ONLY = BASE + textwrap.dedent(
    """
    rented:
      providers:
        market: { type: fake }
    """
)


def parse_config(text: str) -> PoolConfig:
    return PoolConfig.model_validate(yaml.safe_load(text))


def start(tmp_path, text, monkeypatch=None):
    path = tmp_path / "pool.yaml"
    path.write_text(text + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)

    def stop():
        server.stop()
        loop.stop()
        database.close()

    return supervisor, server.base_url, path, stop


def call(base, method, path, **kwargs):
    return httpx.request(method, f"{base}{path}", headers=AUTH, timeout=30, **kwargs)


def confirmed(base, method, path, body):
    answer = call(base, method, path, json=body)
    if answer.status_code == 400 and answer.json().get("error") == "not_confirmed":
        value = next(c["value"] for c in answer.json()["changes"] if c.get("requires_retype"))
        answer = call(base, method, path, json={**body, "confirm": value})
    return answer


@pytest.fixture
def plugins(monkeypatch):
    from gpm_server.providers import FakeProvider, presentation

    market = FakeProvider()
    market.accepted_credentials = {CREDENTIAL: "home"}
    installed = {"fake": {"package": "gpm-server", "version": "test", "loaded": True, **presentation(FakeProvider, "fake")}}
    monkeypatch.setattr("gpm_server.supervisor.providers_api.installed_plugins", lambda: installed)
    monkeypatch.setattr("gpm_server.supervisor.connections.installed_plugins", lambda: installed)
    return market


def test_an_account_is_added_alone_and_renting_is_set_up_after(tmp_path, plugins):
    supervisor, base, path, stop = start(tmp_path, BASE)
    try:
        supervisor.accounts.factory = lambda type_name, settings: plugins.twin()
        assert call(base, "GET", "/pool/renting").json()["account"] is False
        assert call(base, "POST", "/pool/renting/setup", json={}).json()["error"] == "no_account"

        # The account alone: no ceiling, no engine, no model asked — and nothing to type again,
        # because an account that cannot rent spends nothing.
        added = call(base, "POST", "/pool/providers", json={"name": "market", "type": "fake", "credential": CREDENTIAL})
        assert added.status_code == 200, added.text
        assert not any(c.get("requires_retype") for c in added.json()["changes"])
        said = " ".join(c["detail"] for c in added.json()["changes"])
        assert "nothing is rented until renting is set up" in said
        assert yaml.safe_load(path.read_text())["rented"] == {"providers": {"market": {"type": "fake"}}}

        view = call(base, "GET", "/pool/renting").json()
        assert view["account"] and not view["set_up"] and view["restart_needed"]
        assert len(view["missing"]) == 2
        # Only models the engine has a build of are offered, and the rest say why.
        assert view["engines"]["vllm"]["can"] == ["big"]
        assert "no build" in view["engines"]["vllm"]["cannot"]["chat"]
        assert view["engines"]["ollama"]["can"] == ["chat", "big"]

        setup = {"min_disk_gb": 40, "max_all_in_hourly": 0.5, "engine": "vllm", "models": ["chat"]}
        refused = call(base, "POST", "/pool/renting/setup", json=setup)
        assert refused.status_code == 400 and "chat" in refused.json()["detail"]
        assert call(base, "POST", "/pool/renting/setup", json={**setup, "models": []}).json()["error"] == "bad_setup"
        assert call(base, "POST", "/pool/renting/setup", json={**setup, "engine": "whisper-cpp"}).json()["error"] == "bad_setup"

        setup = {**setup, "engine": "ollama", "models": ["chat"]}
        plan = call(base, "POST", "/pool/renting/setup", json={**setup, "plan_only": True})
        assert plan.status_code == 200, plan.text
        set_up = next(c for c in plan.json()["changes"] if c["kind"] == "renting_set_up")
        assert set_up["requires_retype"] and "$0.50/h" in set_up["detail"] and "chat" in set_up["detail"]

        done = confirmed(base, "POST", "/pool/renting/setup", setup)
        assert done.status_code == 200, done.text
        rented = yaml.safe_load(path.read_text())["rented"]
        assert rented["engine"] == "ollama" and rented["models"] == ["chat"]
        assert rented["offer_policy"] == {"min_disk_gb": 40.0, "max_all_in_hourly": 0.5}
        assert call(base, "GET", "/pool/renting").json()["set_up"] is True
    finally:
        stop()


def test_a_pool_not_set_up_rents_nothing(tmp_path):
    supervisor, base, path, stop = start(tmp_path, ACCOUNT_ONLY)
    try:
        fleet = supervisor.fleet
        assert fleet is not None and not supervisor.config.rented.set_up
        with pytest.raises(LeaseRefused, match="renting is not set up"):
            fleet.open_lease(workers=1, max_hours=1, max_spend=1.0, allow_rent=True)
        assert fleet.open_lease(workers=1, max_hours=1, max_spend=None, allow_rent=False), "one that may not rent opens"
        assert fleet.worst_case_hourly() == 0.0
        status = call(base, "GET", "/pool/status").json()
        assert status["renting"]["set_up"] is False and len(status["renting"]["missing"]) == 2
        opened = call(base, "POST", "/pool/leases", json={"workers": 1, "max_hours": 1, "max_spend": 1.0, "allow_rent": True})
        assert opened.status_code >= 400 and "renting is not set up" in opened.text

        # Set up from the console, it rents from the next pass on: the fleet follows the file.
        done = confirmed(base, "POST", "/pool/renting/setup",
                         {"min_disk_gb": 40, "max_all_in_hourly": 0.5, "engine": "ollama", "models": ["chat"]})
        assert done.status_code == 200, done.text
        assert supervisor.fleet.rented.set_up
        lease = supervisor.fleet.open_lease(workers=1, max_hours=1, max_spend=1.0, allow_rent=True)
        assert lease.lease_id
    finally:
        stop()


def test_ceilings_once_set_are_changed_never_taken_away(tmp_path):
    with_ceilings = ACCOUNT_ONLY + "  offer_policy: { min_disk_gb: 40, max_all_in_hourly: 0.5 }\n  engine: ollama\n"
    current, without = parse_config(with_ceilings), parse_config(ACCOUNT_ONLY)
    assert current.rented.set_up and not without.rented.set_up
    refused = [c for c in plan_changes(current, without, [], [], {}) if c.refused]
    assert refused and "can be changed, not removed" in refused[0].refused


def test_a_half_set_policy_is_not_set_up_and_says_what_it_lacks(tmp_path):
    half = parse_config(ACCOUNT_ONLY + "  offer_policy: { min_disk_gb: 40 }\n")
    assert not half.rented.set_up
    assert half.rented.setup_missing() == ["the most one machine may cost an hour (offer_policy.max_all_in_hourly)"]


def test_a_search_profile_still_names_both_ceilings():
    with pytest.raises(ValueError, match="max_all_in_hourly must be set"):
        parse_config(ACCOUNT_ONLY + "  search_profiles: { cheap: { min_disk_gb: 40 } }\n")


# --- the console ---

from fakes.browser import a_browser, open_page  # noqa: E402


@pytest.mark.skipif(a_browser() is None, reason="no browser here")
@pytest.mark.timeout(180)
def test_rented_capacity_sets_up_renting_offering_only_what_the_engine_can_serve(tmp_path):
    supervisor, base, path, stop = start(tmp_path, ACCOUNT_ONLY)
    try:
        with open_page(f"{base}/ui/#rented") as page:
            page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
            page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                    "document.getElementById('key-form').requestSubmit(); true")
            page.until("!!document.querySelector('.renting-setup')", within=30, what="the set-up panel")
            assert not page.js("document.querySelectorAll('.tabs a').length"), "nothing else until renting is set up"
            # The engine that serves more of the pool's models is offered first; the other's
            # missing builds are said, not found on a refusal.
            assert page.js("document.getElementById('setup-engine').value") == "ollama"
            offered = page.js("[...document.querySelectorAll('.renting-setup input[type=checkbox]')].map(b => b.value)")
            assert offered == ["chat", "big"]
            page.js("{ const s = document.getElementById('setup-engine'); s.value = 'vllm'; s.dispatchEvent(new Event('change')); } true")
            assert page.js("[...document.querySelectorAll('.renting-setup input[type=checkbox]')].map(b => b.value)") == ["big"]
            assert "no build of it for vllm" in page.js("document.querySelector('.renting-setup').textContent")

            page.js("{ const s = document.getElementById('setup-engine'); s.value = 'ollama'; s.dispatchEvent(new Event('change')); } true")
            page.js("""{
                const set = (id, v) => { const f = document.getElementById(id); f.value = v; f.dispatchEvent(new Event('input')); };
                set('setup-disk', '40'); set('setup-hourly', '0.5');
                const box = [...document.querySelectorAll('.renting-setup input[type=checkbox]')].find(b => b.value === 'chat');
                box.checked = true; box.dispatchEvent(new Event('change'));
                [...document.querySelectorAll('.renting-setup button')].find(b => b.textContent === 'Set up renting').click(); } true""")
            page.until("document.getElementById('confirm-dialog').open", within=20, what="the confirmation")
            assert "renting is set up" in page.js("document.getElementById('confirm-body').textContent")
            page.js("{ const i = document.getElementById('confirm-input'); i.value = '0.50'; i.dispatchEvent(new Event('input'));"
                    "document.getElementById('confirm-ok').click(); } true")
            page.until("document.querySelectorAll('.tabs a').length > 0", within=30, what="the renting tabs")
        rented = yaml.safe_load(path.read_text())["rented"]
        assert rented["offer_policy"] == {"min_disk_gb": 40.0, "max_all_in_hourly": 0.5}
        assert rented["engine"] == "ollama" and rented["models"] == ["chat"]
    finally:
        stop()


@pytest.mark.skipif(a_browser() is None, reason="no browser here")
@pytest.mark.timeout(180)
def test_add_provider_asks_only_for_the_account_and_writes_no_null(tmp_path, plugins, monkeypatch):
    # Found by the owner: the account step showed the word "null" under the credential — a part
    # left out was handed to the page's own append(), which writes it as text.
    from gpm_server.providers import FakeProvider, presentation

    offered = {"fake": {"package": "gpm-server", "version": "test", "loaded": True,
                        **presentation(FakeProvider, "fake"), "offered": True}}
    monkeypatch.setattr("gpm_server.supervisor.providers_api.installed_plugins", lambda: offered)
    monkeypatch.setattr("gpm_server.supervisor.connections.installed_plugins", lambda: offered)
    supervisor, base, path, stop = start(tmp_path, BASE)
    try:
        with open_page(f"{base}/ui/#providers") as page:
            page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
            page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                    "document.getElementById('key-form').requestSubmit(); true")
            page.until("[...document.querySelectorAll('button')].some(b => b.textContent === 'Add provider')", within=30, what="Providers")
            page.until("!!providersView.data", within=30, what="the installed plug-ins")
            page.js("[...document.querySelectorAll('button')].find(b => b.textContent === 'Add provider').click(); true")
            page.until("!!document.querySelector('dialog[open] .plugin-tile')", within=20, what="the choices")
            page.js("[...document.querySelectorAll('dialog[open] .plugin-tile')].find(b => b.textContent.toLowerCase().includes('fake')).click(); true")
            page.until("!!document.getElementById('provider-name')", within=20, what="the account step")
            text = page.js("document.querySelector('dialog[open]').textContent")
            assert "null" not in text and "undefined" not in text, text
            assert not page.js("document.querySelectorAll('dialog[open] input[type=number], dialog[open] select').length"), \
                "no disk, ceiling, engine or model is asked with an account"
    finally:
        stop()
