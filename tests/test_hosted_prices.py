"""A hosted API's prices, found rather than typed (D176).

Found by the owner adding Ollama's cloud: every model asked for input, output and cache prices the
owner does not decide. Now the console proposes them — from the provider's own `/models` where it
says them, else from a public price list — shown with where they came from and saved only as the
operator leaves them; a provider paid by plan is priced `plan: true`, counting nothing against a
budget. Every test here runs without the network: the price list's fetch is a fake.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import yaml
from gpm_server import hosted_prices
from gpm_server.config import PoolConfig, Price
from gpm_server.hosted_prices import (
    PriceList,
    PricingPage,
    from_listing,
    from_pricing_page,
    proposed,
)

LIST = {
    "claude-x": {"litellm_provider": "anthropic", "mode": "chat", "input_cost_per_token": 3e-06,
                 "output_cost_per_token": 1.5e-05, "cache_read_input_token_cost": 3e-07,
                 "cache_creation_input_token_cost": 3.75e-06},
    "groq/llama-y": {"litellm_provider": "groq", "mode": "chat", "input_cost_per_token": 5e-08,
                     "output_cost_per_token": 8e-08},
    "gpt-z": {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 1e-06,
              "output_cost_per_token": 4e-06},
    "embed-w": {"litellm_provider": "openai", "mode": "embedding", "input_cost_per_token": 1e-07,
                "output_cost_per_token": 0},
    "bad-v": {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": -1,
              "output_cost_per_token": 1e-06},
}


def fake_fetch(table=LIST, calls=None):
    async def fetch(url: str, most: int = 0) -> bytes:
        if calls is not None:
            calls.append(url)
        return json.dumps(table).encode()
    return fetch


# --- what a provider's own /models says ---


def test_openrouter_and_together_listings_are_read_as_dollars_per_million():
    found = from_listing([
        {"id": "or/model", "pricing": {"prompt": "0.000003", "completion": "0.000015",
                                       "input_cache_read": "0.0000003", "input_cache_write": "0.00000375"}},
        {"id": "or/free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "tg/model", "pricing": {"input": 0.88, "output": 0.88}},
        {"id": "tg/unpriced", "pricing": {"input": 0, "output": 0}},
        {"id": "router", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "absurd", "pricing": {"prompt": "5", "completion": "5"}},
        {"id": "nan", "pricing": {"prompt": "nan", "completion": "0.1"}},
        {"id": "plain"},
        "not an entry",
    ])
    assert found["or/model"] == {"input": 3.0, "output": 15.0, "cached_input": 0.3, "cache_write": 3.75}
    assert found["or/free"] == {"input": 0.0, "output": 0.0}, "a free model OpenRouter lists is free"
    assert found["tg/model"] == {"input": 0.88, "output": 0.88}
    for unsaid in ("tg/unpriced", "router", "absurd", "nan", "plain"):
        assert unsaid not in found, f"{unsaid}: not a price, so nothing is proposed"


# --- the public price list ---


def test_the_price_list_proposes_only_its_providers_chat_models():
    prices = PriceList(hosted_prices.PRICE_LIST_URL, fake_fetch())
    found = asyncio.run(prices.prices("anthropic", ["claude-x", "gpt-z"]))
    assert found == {"claude-x": {"input": 3.0, "output": 15.0, "cached_input": 0.3, "cache_write": 3.75}}, \
        "an OpenAI model is never proposed for Anthropic's API"
    assert asyncio.run(prices.prices("groq", ["llama-y"]))["llama-y"] == {"input": 0.05, "output": 0.08}
    found = asyncio.run(prices.prices("openai", ["gpt-z", "embed-w", "bad-v", "unknown"]))
    assert list(found) == ["gpt-z"], "an embedding, a negative price and an unknown id propose nothing"
    assert asyncio.run(prices.prices(None, ["claude-x"])) == {}, "a provider of no preset gets nothing from it"


def test_the_price_list_is_fetched_once_a_while_and_its_failure_said():
    calls = []
    prices = PriceList("https://example.invalid/list.json", fake_fetch(calls=calls))
    asyncio.run(prices.prices("openai", ["gpt-z"]))
    asyncio.run(prices.prices("openai", ["gpt-z"]))
    assert calls == ["https://example.invalid/list.json"]

    async def broken(url, most=0):
        raise httpx.ConnectError("no network")

    failing = PriceList("https://example.invalid/list.json", broken)
    assert asyncio.run(failing.prices("openai", ["gpt-z"])) == {}
    assert "could not be read" in failing.problem
    off = PriceList(None, fake_fetch(calls=calls))
    assert asyncio.run(off.prices("openai", ["gpt-z"])) == {} and len(calls) == 1, "turned off: never fetched"


def test_a_price_list_larger_than_any_is_refused_as_it_arrives():
    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(hosted_prices.MAX_LIST_BYTES // 65536 + 2):
                yield b" " * 65536

    def handler(request):
        return httpx.Response(200, stream=Endless())

    real = httpx.AsyncClient

    class Mocked(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    import gpm_server.hosted_prices as module

    original = module.httpx.AsyncClient
    module.httpx.AsyncClient = Mocked
    try:
        with pytest.raises(ValueError, match="larger than any price list"):
            asyncio.run(module._fetch("https://example.invalid/list.json"))
    finally:
        module.httpx.AsyncClient = original


def test_the_provider_is_asked_first_its_page_next_and_the_list_after():
    prices = PriceList(hosted_prices.PRICE_LIST_URL, fake_fetch())
    out, problem = asyncio.run(proposed("openai", ["gpt-z", "other"], {"other": {"input": 1.0, "output": 2.0}}, prices))
    assert out == {"other": {"price": {"input": 1.0, "output": 2.0}, "source": "provider"},
                   "gpt-z": {"price": {"input": 1.0, "output": 4.0}, "source": "price_list"}}
    assert problem is None

    page = PricingPage("https://provider.invalid/pricing", page_fetch())
    out, problem = asyncio.run(proposed("ollama-cloud", ["alpha:31b", "beta-2", "listed", "absent"],
                                        {"listed": {"input": 9.0, "output": 9.0}}, prices, page))
    assert out == {"listed": {"price": {"input": 9.0, "output": 9.0}, "source": "provider"},
                   "alpha:31b": {"price": {"input": 0.14, "output": 0.4, "cached_input": 0.05}, "source": "provider_page"},
                   "beta-2": {"price": {"input": 1.32, "output": 3.96}, "source": "provider_page"}}
    assert problem is None, "a provider the public list does not know is not asked of it"


# --- a provider's own pricing page ---

# Shaped as Ollama's page is (read 2026-11-08); the models and prices here are made up.
PAGE = """<html><body><h2>Model pricing</h2>
<table class="w-full"><thead><tr><th>Model</th><th>Input</th><th>Cached input</th><th>Output</th></tr></thead>
<tbody>
<tr><td><a href="/library/alpha">alpha</a></td><td>$0.14</td><td>$0.05</td><td>$0.40</td></tr>
<tr><td><a href="/library/beta-2">beta-2</a></td><td>$1.32</td><td>-</td><td>$3.96</td></tr>
<tr><td>beta-2 (Off-Peak)</td><td>$0.66</td><td>-</td><td>$1.98</td></tr>
<tr><td>gamma:120b</td><td>$0.15</td><td>$0.014</td><td>$0.60</td></tr>
<tr><td>reranker</td><td>$0.24</td><td>-</td><td>-</td></tr>
<tr><td>broken</td><td>free</td><td>-</td><td>$1</td></tr>
</tbody></table>
<table><tr><th>Plan</th><th>Price</th></tr><tr><td>Pro</td><td>$20</td></tr></table></body></html>"""


def page_fetch(text=PAGE, calls=None):
    async def fetch(url: str, most: int = 0) -> bytes:
        if calls is not None:
            calls.append((url, most))
        return text.encode()
    return fetch


def test_a_pricing_page_is_read_at_its_peak_and_only_as_the_table_it_is():
    found = from_pricing_page(PAGE)
    assert found == {"alpha": {"input": 0.14, "output": 0.4, "cached_input": 0.05},
                     "beta-2": {"input": 1.32, "output": 3.96},
                     "gamma:120b": {"input": 0.15, "output": 0.6, "cached_input": 0.014}}, \
        "the off-peak row, a model with no output price and a price that is not one are left out"
    with pytest.raises(ValueError, match="not the one expected"):
        from_pricing_page(PAGE.replace("<th>Cached input</th>", "<th>Cache</th>"))


def test_a_pricing_page_matches_an_id_by_itself_or_its_name_before_the_tag():
    calls = []
    page = PricingPage("https://provider.invalid/pricing", page_fetch(calls=calls))
    found = asyncio.run(page.prices(["alpha:31b", "gamma:120b", "gamma:20b", "delta"]))
    assert set(found) == {"alpha:31b", "gamma:120b"}, "gamma:20b is not gamma:120b, and delta is not listed"
    asyncio.run(page.prices(["alpha:31b"]))
    assert calls == [("https://provider.invalid/pricing", hosted_prices.MAX_PAGE_BYTES)], "fetched once a while, bounded"

    changed = PricingPage("https://provider.invalid/pricing", page_fetch(PAGE.replace("<th>Output</th>", "<th>Out</th>")))
    assert asyncio.run(changed.prices(["alpha:31b"])) == {} and "could not be read" in changed.problem


# --- paid by plan ---


def test_a_model_paid_by_plan_costs_nothing_per_token_and_says_nothing_else():
    plan = Price(plan=True)
    assert plan.of(10_000, 2_000, 5_000, 1_000) == 0.0 and plan.most_for_input(1_000_000) == 0.0
    from gpm_server.router.app import _worst_case

    assert _worst_case(plan, 10_000_000, 100_000) == 0.0, "a plan's request holds nothing of a budget"
    with pytest.raises(ValueError, match="paid by `plan` names no"):
        Price(plan=True, input=1.0, output=2.0)
    with pytest.raises(ValueError, match="or `plan: true`"):
        Price()


def test_the_price_list_address_is_https_or_nothing():
    base = {"pool": {"model_set": ["a"]}, "auth": {"admin_keys": ["gpmx_prices_admin"], "app_keys": ["gpma_prices_app"]},
            "catalog": {"a": {"variants": [{"tag": "a:1"}]}},
            "hosts": [{"id": "h", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}]}
    assert PoolConfig.model_validate(base).pool.price_list_url == hosted_prices.PRICE_LIST_URL
    off = {**base, "pool": {**base["pool"], "price_list_url": ""}}
    assert PoolConfig.model_validate(off).pool.price_list_url is None
    with pytest.raises(ValueError, match="https://"):
        PoolConfig.model_validate({**base, "pool": {**base["pool"], "price_list_url": "http://example.invalid/x.json"}})


# --- through the control API, against a fake provider ---

from test_hosted_providers import KEY, api, pool  # noqa: E402,F401  (the fixture)


def test_the_models_listed_come_with_the_providers_prices_and_a_plan_price_is_saved(pool):  # noqa: F811
    pool.fake.pricing = {"provider-claude": {"prompt": "0.000003", "completion": "0.000015"}}
    tested = api(pool, "POST", "/pool/hosted/test",
                 json={"engine": "openai-api", "base_url": pool.provider_url + "/v1", "credential": KEY}).json()
    assert tested["ok"] and tested["paid_by_plan"] is False
    assert tested["prices"] == {"provider-claude": {"price": {"input": 3.0, "output": 15.0}, "source": "provider"}}
    assert KEY not in json.dumps(tested)

    body = {"id": "flat", "engine": "openai-api", "base_url": pool.provider_url + "/v1", "credential": KEY,
            "models": [{"name": "flat-gpt", "tag": "provider-gpt", "price": {"plan": True}}]}
    answer = api(pool, "POST", "/pool/hosted", json=body)
    if answer.status_code == 400 and answer.json().get("error") == "not_confirmed":
        value = next(c["value"] for c in answer.json()["changes"] if c.get("requires_retype"))
        answer = api(pool, "POST", "/pool/hosted", json={**body, "confirm": value})
    assert answer.status_code == 200, answer.text
    variant = yaml.safe_load(pool.path.read_text())["catalog"]["flat-gpt"]["variants"][0]
    assert variant["price"] == {"plan": True}

    mixed = {**body, "id": "mixed", "models": [{"name": "m", "tag": "provider-gpt", "price": {"plan": True, "input": 1}}]}
    refused = api(pool, "POST", "/pool/hosted", json=mixed)
    assert refused.status_code == 400 and "names no per-token price" in refused.json()["detail"]


# --- the console ---

from fakes.browser import a_browser, open_page  # noqa: E402


@pytest.mark.skipif(a_browser() is None, reason="no browser here")
@pytest.mark.timeout(180)
def test_the_models_step_fills_the_providers_prices_and_offers_paid_by_plan(pool):  # noqa: F811
    from test_hosted_providers import ADMIN_KEY, _click

    pool.fake.pricing = {"provider-claude": {"prompt": "0.000003", "completion": "0.000015"}}
    with open_page(f"{pool.control}/ui/#providers") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.until("document.body.textContent.includes('No provider yet')", within=30, what="the screen")
        _click(page, "Add provider")
        page.until("[...document.querySelectorAll('.plugin-tile')].some(t => t.textContent.includes('Another OpenAI-format API'))",
                   within=10, what="the choices")
        page.js("[...document.querySelectorAll('.plugin-tile')].find(t => t.textContent.includes('Another OpenAI-format API')).click(); true")
        page.until("!!document.querySelector('#hosted-url')", within=10, what="the key step")
        page.js(f"""
            const set = (sel, v) => {{ const f = document.querySelector(sel); f.value = v; f.dispatchEvent(new Event('input', {{bubbles: true}})); }};
            set('#hosted-id', 'priced'); set('#hosted-url', {json.dumps(pool.provider_url + '/v1')});
            set('#provider-dialog input[data-secret]', {json.dumps(KEY)});
            document.querySelector('#provider-dialog form').requestSubmit(); true""")
        page.until("document.getElementById('provider-dialog').textContent.includes('2 model(s) offered')", within=20, what="the test")
        _click(page, "Choose models", "document.getElementById('provider-dialog')")
        page.until("!!document.querySelector('.model-row[data-tag=\"provider-claude\"]')", within=10, what="the models")

        # The provider's own prices, filled in and said to be its; labelled as prices, not limits.
        filled = page.js("[...document.querySelectorAll('.model-row[data-tag=\"provider-claude\"] input.price')].map(i => i.value)")
        assert filled[0] == "3" and filled[-1] == "15"
        row = page.js("document.querySelector('.model-row[data-tag=\"provider-claude\"]').textContent")
        assert "price from the provider's API" in row
        assert page.js("document.querySelector('.model-row input.price').placeholder") == "$/1M in"
        assert "not limits" in page.js("document.getElementById('provider-dialog').textContent")
        assert page.js("[...document.querySelectorAll('.model-row[data-tag=\"provider-gpt\"] input.price')].every(i => i.value === '')")

        # Paid by plan: no price asked, and each model saved as paid by plan.
        page.js("""{
            const plan = document.getElementById('models-plan'); plan.checked = true; plan.dispatchEvent(new Event('change'));
            const row = document.querySelector('.model-row[data-tag="provider-gpt"]');
            const box = row.querySelector('input[type=checkbox]'); box.checked = true; box.dispatchEvent(new Event('change')); } true""")
        assert page.js("document.querySelector('.model-row[data-tag=\"provider-gpt\"] .price-box').hidden") is True
        _click(page, "Continue", "document.getElementById('provider-dialog')")
        page.until("!document.querySelector('#provider-dialog .wizard-nav button.primary').disabled", within=20, what="the plan to load")
        _click(page, "Add it", "document.getElementById('provider-dialog')")
        page.until("!document.getElementById('provider-dialog').open", within=20, what="the save")
        page.until("!!document.querySelector('.hosted-card[data-hosted=\"priced\"]')", within=20, what="the new card")
        assert "paid by plan" in page.js("document.querySelector('.hosted-card[data-hosted=\"priced\"]').textContent")

        # Opened again from its card (found by the owner on Ollama's cloud): still paid by plan, no
        # stray "null", and the name apps ask by is a field one can read.
        page.js("[...document.querySelectorAll('.hosted-card[data-hosted=\"priced\"] button')]"
                ".find(b => b.textContent === 'Models and prices').click(); true")
        page.until("document.getElementById('provider-dialog').open && !!document.getElementById('models-plan')",
                   within=20, what="Models and prices")
        dialog = page.js("document.getElementById('provider-dialog').textContent")
        assert "null" not in dialog and "undefined" not in dialog, dialog
        assert page.js("document.getElementById('models-plan').checked") is True
        width = page.js("document.querySelector('.model-row[data-tag=\"provider-gpt\"] input.mono').getBoundingClientRect().width")
        assert width > 100, f"the name field is {width}px wide"
    variant = yaml.safe_load(pool.path.read_text())["catalog"]["provider-gpt"]["variants"][0]
    assert variant["price"] == {"plan": True}
