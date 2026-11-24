"""What is left in each provider account (D188): read from the provider where its API says it,
typed by the operator elsewhere and counted down by the pool's own spend, and acted on at $0 —
a hosted API out of routing, a rented account renting nothing new.

Against fakes shaped after each provider's API reference; nothing here reaches a provider or
spends money.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import pytest
import test_hosted_providers
import yaml
from fakes.fake_hosted_api import KEY
from gpm_server import hosted_balances
from gpm_server.config import PoolConfig, load_config
from gpm_server.db import Database
from gpm_server.hosted_presets import OPENAI_ENGINE, PRESETS, HostedPreset
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, ProviderCapabilities, default_offer
from gpm_server.providers.runpod import RunPodProvider
from gpm_server.providers.verda import VerdaProvider
from gpm_server.router.registry import _follow_balance
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor import balances as balances_module
from gpm_server.supervisor.control import create_control_app
from gpm_server.supervisor.renting import Fleet
from test_hosted_providers import ADMIN_KEY, GPT, add_claude, api, chat, settle

#: The hosted pool of test_hosted_providers: a supervisor, a router and a fake provider.
pool = test_hosted_providers.pool

# --- reading a hosted provider's answer, in the shapes its reference gives ---


def test_openrouters_credits_are_bought_less_used():
    reading = hosted_balances.openrouter_credits({"data": {"total_credits": 100.5, "total_usage": 25.75}})
    assert reading.usd == pytest.approx(74.75) and reading.detail is None
    assert hosted_balances.openrouter_credits({"data": {"total_credits": "x", "total_usage": 1}}).usd is None
    assert hosted_balances.openrouter_credits({"error": {}}).usd is None
    assert hosted_balances.openrouter_credits([1, 2]).usd is None


def test_deepseeks_balance_is_its_dollar_entry_and_never_another_currency():
    answer = {"is_available": True, "balance_infos": [
        {"currency": "CNY", "total_balance": "800.00", "granted_balance": "0.00", "topped_up_balance": "800.00"},
        {"currency": "USD", "total_balance": "110.00", "granted_balance": "10.00", "topped_up_balance": "100.00"}]}
    assert hosted_balances.deepseek_balance(answer).usd == 110.0, "the amounts are strings in its reference"
    yuan = hosted_balances.deepseek_balance({"is_available": True, "balance_infos": [
        {"currency": "CNY", "total_balance": "110.00"}]})
    assert yuan.usd is None and "CNY" in yuan.detail and "not dollars" in yuan.detail
    assert hosted_balances.deepseek_balance({"balance_infos": [{"currency": "USD", "total_balance": "lots"}]}).usd is None
    assert hosted_balances.deepseek_balance({"balance_infos": [{"currency": "USD", "total_balance": "1e300"}]}).usd is None
    assert hosted_balances.deepseek_balance({}).usd is None


def test_the_presets_that_read_a_balance_read_it_from_their_own_provider():
    reading = [p for p in PRESETS if p.balance_url]
    assert {p.id for p in reading} == {"openrouter", "deepseek"}
    for preset in reading:
        assert hosted_balances.same_origin(preset.balance_url, preset.base_url), preset.id
        assert preset.balance_format in hosted_balances.READERS


async def test_a_balance_is_asked_only_where_the_hosts_key_was_saved_for():
    sent = []

    def handler(request):
        sent.append((str(request.url), request.headers.get("authorization")))
        if request.url.path == "/denied":
            return httpx.Response(403, json={"error": {"code": 403, "message": "Only management keys can perform this operation"}})
        return httpx.Response(200, json={"data": {"total_credits": 5, "total_usage": 1.25}})

    client = httpx.AsyncClient(base_url="https://provider.example/api/v1", transport=httpx.MockTransport(handler),
                               headers={"Authorization": "Bearer fake-key-1"})
    base = "https://provider.example/api/v1"
    good = await hosted_balances.read(client, base, "https://provider.example/api/v1/credits", "openrouter-credits")
    assert good.usd == 3.75
    with pytest.raises(ValueError):
        await hosted_balances.read(client, base, "https://elsewhere.example/credits", "openrouter-credits")
    with pytest.raises(ValueError):
        await hosted_balances.read(client, base, "http://provider.example/api/v1/credits", "openrouter-credits")
    denied = await hosted_balances.read(client, base, "https://provider.example/denied", "openrouter-credits",
                                        note="answered only to a management key")
    assert denied.usd is None and "403" in denied.detail and "management key" in denied.detail
    assert "Only management keys" not in denied.detail, "the pool's words, never the provider's answer"
    assert all(url.startswith("https://provider.example/") for url, _ in sent), "the key went nowhere else"
    await client.aclose()


def test_the_rented_plugins_declare_the_balance_their_apis_answer():
    assert RunPodProvider.capabilities.reports_balance and VerdaProvider.capabilities.reports_balance
    assert not ProviderCapabilities().reports_balance, "off by default: a third-party plug-in stays valid"


# --- a hosted API: typed, counted down, out of routing at $0 ---


def view(pool, account):
    answer = api(pool, "GET", "/pool/balances")
    assert answer.status_code == 200, answer.text
    return next(b for b in answer.json()["balances"] if b["account"] == account)


def router_host(pool, host_id):
    pool.loop.run(pool.router_app.state.pool.registry.refresh())
    return pool.router_app.state.pool.registry.by_id.get(host_id)


def spend_there(pool, host_id, dollars):
    pool.db.execute("INSERT INTO request_log (ts, request_id, host_id, outcome, cost_usd) VALUES (?, ?, ?, ?, ?)",
                    (time.time(), f"r-{time.monotonic_ns()}", host_id, "ok", dollars))


def test_a_typed_balance_is_counted_down_and_its_api_taken_out_of_routing_at_zero(pool):
    assert add_claude(pool).status_code == 200
    settle(pool, lambda: any(h.host_id == "anthropic" and h.state.value == "ready" for h in pool.router_app.state.pool.hosts))
    unknown = view(pool, "hosted:anthropic")
    assert unknown["source"] == "unknown" and not unknown["can_read"], "Anthropic's API says no balance"

    typed = api(pool, "PUT", "/pool/balances/hosted/anthropic", json={"usd": 1.0})
    assert typed.status_code == 200, typed.text
    assert typed.json()["source"] == "typed" and typed.json()["estimate"] == 1.0
    assert chat(pool, "claude").status_code == 200
    settle(pool, lambda: view(pool, "hosted:anthropic")["spent_since"] > 0)
    counted = view(pool, "hosted:anthropic")
    assert counted["estimate"] == pytest.approx(1.0 - counted["spent_since"])
    assert counted["summary"].startswith("about $0.99 left — $1.00 typed on"), counted["summary"]
    assert api(pool, "GET", "/pool/hosted").json()["hosts"][0]["balance"]["source"] == "typed", "on the card"

    spend_there(pool, "anthropic", 1.5)
    assert view(pool, "hosted:anthropic")["estimate"] < 0
    host = router_host(pool, "anthropic")
    assert host.out_of_funds and host.out_of_funds.startswith("its balance reached $0"), "out of routing, said why"
    exhausted = pool.supervisor.events.recent(20, kind="balance_exhausted")
    assert exhausted and "taken out of routing" in exhausted[0]["summary"]

    # Money is back: the typed balance at $0 would take it out again, so it is cleared.
    restored = api(pool, "POST", "/pool/hosts/anthropic/funds-restored")
    assert restored.status_code == 200 and "cleared" in restored.json()["balance"], restored.text
    assert view(pool, "hosted:anthropic")["source"] == "unknown"
    assert router_host(pool, "anthropic").out_of_funds is None, "routed to again"

    # At $0 typed it goes out; a topped-up figure typed puts it back.
    assert api(pool, "PUT", "/pool/balances/hosted/anthropic", json={"usd": 0}).status_code == 200
    assert router_host(pool, "anthropic").out_of_funds is not None
    assert api(pool, "PUT", "/pool/balances/hosted/anthropic", json={"usd": 25}).status_code == 200
    assert router_host(pool, "anthropic").out_of_funds is None
    assert pool.supervisor.events.recent(20, kind="balance_restored")
    cleared = api(pool, "DELETE", "/pool/balances/hosted/anthropic")
    assert cleared.status_code == 200 and cleared.json()["source"] == "unknown"


def test_a_providers_own_refusal_for_money_is_not_undone_by_a_balance(pool):
    host = SimpleNamespace(out_of_funds="spend limit", out_of_funds_at=10.0, out_of_funds_by_balance=False)
    _follow_balance(host, SimpleNamespace(balance_out=None, balance_out_at=None, funds_reset_at=None))
    assert host.out_of_funds == "spend limit", "only the operator's word clears the provider's"
    _follow_balance(host, SimpleNamespace(balance_out="its balance reached $0", balance_out_at=20.0, funds_reset_at=None))
    assert host.out_of_funds == "spend limit" and not host.out_of_funds_by_balance
    other = SimpleNamespace(out_of_funds=None, out_of_funds_at=None, out_of_funds_by_balance=False)
    _follow_balance(other, SimpleNamespace(balance_out="its balance reached $0", balance_out_at=20.0, funds_reset_at=25.0))
    assert other.out_of_funds is None, "an operator's word newer than the mark wins"


def test_a_hosted_providers_balance_is_read_with_its_key_from_its_own_address(pool, monkeypatch):
    base = f"{pool.provider_url}/v1"
    preset = HostedPreset("openrouter", "OpenRouter", base, OPENAI_ENGINE, balance_url=f"{base}/credits",
                          balance_format="openrouter-credits", balance_note="only to a management key")
    monkeypatch.setattr(balances_module, "preset_for", lambda url, engine: preset if url.rstrip("/") == base else None)
    monkeypatch.setattr(balances_module, "ASK_AGAIN_AFTER_S", 0.0)
    added = api(pool, "POST", "/pool/hosted", json={
        "id": "router-x", "engine": "openai-api", "base_url": base, "credential": KEY,
        "models": [{"name": "gpt", "tag": "provider-gpt", "price": GPT}]})
    assert added.status_code == 200, added.text
    settle(pool, lambda: "router-x" in pool.supervisor.hosts)

    read = api(pool, "POST", "/pool/balances/read", json={"account": "hosted:router-x"})
    assert read.status_code == 200, read.text
    found = next(b for b in read.json()["balances"] if b["account"] == "hosted:router-x")
    assert found["source"] == "provider" and found["amount"] == 7.5 and found["can_read"]
    asked = [s for s in pool.fake.seen if s["path"] == "/v1/credits"]
    assert asked and asked[-1]["headers"]["authorization"] == f"Bearer {KEY}"

    pool.fake.balance_status = 403
    again = api(pool, "POST", "/pool/balances/read", json={"account": "hosted:router-x"}).json()
    found = next(b for b in again["balances"] if b["account"] == "hosted:router-x")
    assert found["amount"] == 7.5, "a failed read keeps the last figure"
    assert "403" in found["read"]["detail"] and "management key" in found["read"]["detail"]
    refused = api(pool, "POST", "/pool/balances/read", json={"account": "hosted:laptop"})
    assert refused.status_code == 404
    from test_hosted_providers import everything_written

    written = everything_written(pool) + json.dumps([dict(r) for r in pool.db.query("SELECT * FROM provider_balances")])
    assert KEY not in written


def test_the_balance_endpoints_refuse_what_is_not_a_balance(pool):
    assert add_claude(pool).status_code == 200
    for bad in ({"usd": -1}, {"usd": "lots"}, {"usd": True}, {"usd": 1e12}, {}, {"usd": float("nan")}):
        answer = httpx.put(f"{pool.control}/pool/balances/hosted/anthropic", headers={
            "Authorization": f"Bearer {ADMIN_KEY}", "Content-Type": "application/json"},
            content=json.dumps(bad), timeout=30)
        assert answer.status_code == 400 and answer.json()["error"] in ("bad_balance", "bad_request"), (bad, answer.text)
    assert api(pool, "PUT", "/pool/balances/hosted/nobody", json={"usd": 1}).status_code == 404
    assert api(pool, "PUT", "/pool/balances/elsewhere/anthropic", json={"usd": 1}).status_code == 404
    assert api(pool, "POST", "/pool/balances/read", json={"account": "hosted:anthropic"}).json()["error"] == "not_reported"
    unauthorised = httpx.get(f"{pool.control}/pool/balances", headers={"Authorization": "Bearer gpma_hosted_app"})
    assert unauthorised.status_code in (401, 403), "the app key never reaches the control API"


def test_the_cli_lists_sets_and_clears_a_balance(pool, monkeypatch, capsys):
    from gpm_server.cli import main

    assert add_claude(pool).status_code == 200
    monkeypatch.setenv("GPM_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("GPM_CONTROL_URL", pool.control)
    capsys.readouterr()
    assert main(["balance", "set", "hosted:anthropic", "--usd", "12.5"]) == 0
    assert '"source": "typed"' in capsys.readouterr().out
    assert main(["balance"]) == 0 and "about $12.50 left" in capsys.readouterr().out
    assert main(["balance", "set", "hosted:anthropic"]) == 2, "set needs --usd"
    assert main(["balance", "set", "anthropic", "--usd", "1"]) == 2, "an account is named with its kind"
    assert main(["balance", "clear", "hosted:anthropic"]) == 0 and '"source": "unknown"' in capsys.readouterr().out


# --- a rented account: nothing new rented that it could not pay for ---


def rented_config(**auth) -> PoolConfig:
    return PoolConfig.model_validate(rented_dict(**auth))


def rented_dict(**auth) -> dict:
    return ({
        "pool": {"name": "test", "model_set": ["m1"]},
        "auth": {"app_keys": ["k"], **auth},
        "hosts": [{"id": "local-1", "kind": "local", "workers": 1,
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        "rented": {"provider": "fake", "engine": "ollama", "workers": 2, "bidding": {"premium": 0.02},
                   "offer_policy": {"min_gpu_memory_gb": 24, "min_disk_gb": 10, "max_all_in_hourly": 0.60},
                   "scale": {"scale_up_after_s": 0}},
    })


@pytest.fixture
def fleet(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    config = rented_config()
    made = Fleet(config, config.rented, FakeProvider(), LeaseStore(database), EventLog(database), SpendLedger(database))
    try:
        yield made
    finally:
        database.close()


async def test_an_account_that_would_not_cover_an_hour_rents_nothing_and_says_why(fleet):
    fleet.leases.open(workers=20, max_hours=4, max_spend=5.0, allow_rent=True)
    fleet.balances[fleet.connection] = 0.10  # the cheapest bid is $0.12/h
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert fleet.hosts == {} and "create" not in fleet.provider.calls
    refused = [e for e in fleet.events.recent() if e["kind"] == "rent_refused"]
    assert refused and "about $0.10 left" in refused[0]["summary"] and "topped up" in refused[0]["summary"]
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert len([e for e in fleet.events.recent() if e["kind"] == "rent_refused"]) == 1, "said once, not every pass"

    fleet.balances[fleet.connection] = 5.0
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert len(fleet.hosts) == 1
    running = next(iter(fleet.hosts.values()))
    running.state = "ready"

    # Room for the machine running, not for a second one beside it: the second is refused, and the
    # one running is never stopped for it.
    fleet.balances[fleet.connection] = 0.20
    fleet.provider.offers.append(default_offer("o-2", "m-2"))
    lease = fleet.leases.open_leases()[0]
    assert await fleet.rent_one(lease, ["asked"]) is None
    assert "an hour of what the pool would run" in fleet.last_refusal
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    assert list(fleet.hosts) == [running.host_id] and not running.released
    fleet.balances.pop(fleet.connection)
    assert fleet.balance_refusal(fleet.connection, 10.0) is None, "a balance not known refuses nothing"


async def test_a_reported_rented_balance_is_read_counted_and_shown(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    provider = FakeProvider(capabilities=ProviderCapabilities(interruptible=True, reports_charges=True,
                                                              reports_balance=True))
    provider.credit = 0.05
    path = tmp_path / "pool.yaml"
    path.write_text(yaml.safe_dump({**rented_dict(admin_keys=["gpmx_balance_admin"]),
                                    "request_log": str(tmp_path / "gpm.sqlite3")}))
    supervisor = Supervisor(load_config(path), database, provider=provider, config_path=str(path))
    try:
        await supervisor.balances.read_now()
        name = supervisor.fleet.connection
        assert supervisor.fleet.balances == {name: 0.05}
        assert supervisor.fleet.balance_refusal(name, 0.12) is not None

        app = create_control_app(supervisor, supervisor.config)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 5000)),
                                     base_url="http://127.0.0.1", headers={"Authorization": "Bearer gpmx_balance_admin"}) as client:
            listed = (await client.get("/pool/balances")).json()["balances"]
            (shown,) = [b for b in listed if b["account"] == f"rented:{name}"]
            assert shown["source"] == "provider" and shown["amount"] == 0.05 and shown["can_read"]
            card = (await client.get("/pool/providers")).json()["connections"][0]
            assert card["balance"]["source"] == "provider" and card["balance_refusal"] is None, "nothing running there"
            typed = await client.put(f"/pool/balances/rented/{name}", json={"usd": 40})
            assert typed.status_code == 200 and typed.json()["source"] == "typed", "the newer of read and typed"
        assert supervisor.fleet.balances == {name: 40.0}

        # Money the pool spends there since is counted down from the ledger's running totals.
        supervisor.spend.record(lease_id="l1", host_id="h1", source="estimate", amount=1.0, connection=name)
        supervisor.spend.record(lease_id="l1", host_id="h1", source="estimate", amount=1.5, connection=name)
        await supervisor.balances.count()
        assert supervisor.fleet.balances[name] == pytest.approx(38.5)
    finally:
        await supervisor.aclose()
        database.close()


async def test_a_plugin_that_does_not_declare_a_balance_is_never_read(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    provider = FakeProvider()  # returns a credit, but does not say it is the account's balance
    supervisor = Supervisor(rented_config(), database, provider=provider)
    try:
        await supervisor.balances.pass_once()
        assert "account" not in provider.calls
        assert supervisor.fleet.balances == {}
    finally:
        await supervisor.aclose()
        database.close()


# --- the console ---

from fakes.browser import a_browser, open_page  # noqa: E402


@pytest.mark.skipif(a_browser() is None, reason="no browser here")
def test_the_providers_card_shows_the_balance_and_takes_one_typed(pool):
    assert add_claude(pool).status_code == 200
    with open_page(f"{pool.control}/ui/#providers") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        card = "document.querySelector('.hosted-card[data-hosted=\"anthropic\"]')"
        page.until(f"!!{card} && {card}.textContent.includes('balance')", within=30, what="the card")
        assert "not known" in page.js(f"{card}.querySelector('[data-balance]').textContent")
        page.js(f"[...{card}.querySelectorAll('[data-balance] button')].find(b => b.textContent === 'Type it').click(); true")
        page.until("!!document.querySelector('#balance-usd')", within=10, what="the dialog")
        page.js("document.querySelector('#balance-usd').value = 64;"
                "document.querySelector('#provider-dialog form').requestSubmit(); true")
        page.until("!document.getElementById('provider-dialog').open", within=20, what="the save")
        page.until(f"{card}.querySelector('[data-balance]').textContent.includes('about $64.00 left')", within=20,
                   what="the estimate")
        line = page.js(f"{card}.querySelector('[data-balance]').textContent")
        assert "$64.00 set on" in line and "less $0.00 spent here since" in line and "estimate" in line
    shown = api(pool, "GET", "/pool/balances").json()["balances"][0]
    assert shown["source"] == "typed" and shown["amount"] == 64.0
