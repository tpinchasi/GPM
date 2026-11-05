"""Hosted model APIs (S11, D164–D166): a provider reached with a key is one more host, last, used
within a budget — through the real router and supervisor, against a fake provider that speaks
the OpenAI-shaped API and gives the refusals real providers give.
"""

import json
import threading
import time

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import APP_KEY, BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_server.config import PoolConfig, Price
from gpm_server.engines.hosted import HostedOpenAIEngine
from gpm_server.router.budgets import HostedBudgets

CATALOG = {
    "llama": {"variants": [
        {"tag": "llama:7b", "engine": "ollama"},
        {"tag": "provider-llama", "engine": "openai-api", "price": {"input": 1.0, "cached_input": 0.5, "output": 2.0}},
    ]},
    "claude": {"variants": [
        {"tag": "provider-claude", "engine": "openai-api", "price": {"input": 3.0, "output": 15.0}},
    ]},
}
#: The fake's usage — 1000 in, 200 of them cached, 500 out — at each build's price.
LLAMA_COST = (800 * 1.0 + 200 * 0.5 + 500 * 2.0) / 1e6
CLAUDE_COST = (800 * 3.0 + 200 * 3.0 + 500 * 15.0) / 1e6


@pytest.fixture
def provider():
    loop = BackgroundLoop()
    fake = FakeHostedApi()
    server = ServerHandle(fake.app, loop)
    yield fake, server.base_url
    server.stop()
    loop.stop()


def hosted_pool(monkeypatch, url, *, budget=None, after_s=0.3, box_delay=0.0):
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    return pool_harness(
        [EngineSpec(id="box", resident={"llama:7b"}, chunk_delay_s=box_delay)],
        model_set=["llama", "claude"], catalog=CATALOG,
        extra_hosts=[{"id": "provider", "kind": "hosted", "workers": 4, "capabilities": [],
                      "transport": {"type": "http", "base_url": f"{url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}],
        pool_settings={"bursts": {"hosted": {"after_s": after_s, "budget": budget or {"per_day": 1.0}}}},
    )


def ask(pool, model, *, stream=False, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": stream, **extra}
    return httpx.post(f"{pool.url}/v1/chat/completions", json=body, timeout=30,
                      headers={"Authorization": f"Bearer {APP_KEY}"})


def logged(pool, model):
    return [r for r in pool.request_log(50) if r["model_requested"] == model]


# --- a model only a provider serves ---


def test_a_model_no_machine_serves_goes_to_its_provider_at_once_and_says_what_it_cost(monkeypatch, provider):
    fake, url = provider
    with hosted_pool(monkeypatch, url, after_s=30) as pool:
        pool.reprobe()
        answered = ask(pool, "claude")
        assert answered.status_code == 200, answered.text
        assert answered.headers["x-gpm-host"] == "provider" and answered.headers["x-gpm-served-model"] == "provider-claude"
        assert float(answered.headers["x-gpm-cost-usd"]) == pytest.approx(CLAUDE_COST)
        assert float(answered.headers["x-gpm-wait-s"]) < 5, "no machine serves it: not made to wait"
        sent = fake.seen[-1]
        assert sent["path"] == "/v1/chat/completions"
        assert sent["headers"]["authorization"] == f"Bearer {KEY}", "the provider's key, never the app's"
        assert not any(name.startswith("x-gpm") for name in sent["headers"])
        assert sent["body"]["model"] == "provider-claude"
        assert logged(pool, "claude")[0]["cost_usd"] == pytest.approx(CLAUDE_COST)


# --- hardware first ---


def test_a_machine_with_a_free_worker_is_used_before_any_provider(monkeypatch, provider):
    fake, url = provider
    with hosted_pool(monkeypatch, url) as pool:
        pool.reprobe()
        answered = ask(pool, "llama")
        assert answered.status_code == 200 and answered.headers["x-gpm-host"] == "box"
        assert "x-gpm-cost-usd" not in answered.headers and fake.seen == []


def test_a_burst_spills_to_the_provider_only_after_waiting_for_a_machine(monkeypatch, provider):
    fake, url = provider
    with hosted_pool(monkeypatch, url, after_s=0.5, box_delay=2.0) as pool:
        pool.reprobe()
        slow = threading.Thread(target=lambda: ask(pool, "llama", stream=True))
        slow.start()  # the box's one worker, busy for a while
        time.sleep(0.2)
        started = time.monotonic()
        answered = ask(pool, "llama")
        slow.join()
        assert answered.status_code == 200 and answered.headers["x-gpm-host"] == "provider"
        assert time.monotonic() - started >= 0.5, "waited for a machine first"
        assert float(answered.headers["x-gpm-cost-usd"]) == pytest.approx(LLAMA_COST)


# --- streams ---


def test_a_stream_is_asked_for_its_usage_and_the_app_never_sees_the_extra_chunk(monkeypatch, provider):
    fake, url = provider
    with hosted_pool(monkeypatch, url) as pool:
        pool.reprobe()
        answered = ask(pool, "claude", stream=True)
        assert answered.status_code == 200
        assert fake.seen[-1]["body"]["stream_options"] == {"include_usage": True}
        frames = [f for f in answered.text.split("\n\n") if f.startswith("data: {")]
        assert frames and all(json.loads(f[6:])["choices"] for f in frames), "no usage-only chunk"
        assert answered.text.rstrip().endswith("data: [DONE]")
        assert float(answered.headers["x-gpm-cost-max-usd"]) >= CLAUDE_COST
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (logged(pool, "claude") and logged(pool, "claude")[0]["cost_usd"]):
            time.sleep(0.05)
        assert logged(pool, "claude")[0]["cost_usd"] == pytest.approx(CLAUDE_COST), "settled from the usage"


def test_a_stream_that_asked_for_its_own_usage_keeps_it(monkeypatch, provider):
    fake, url = provider
    with hosted_pool(monkeypatch, url) as pool:
        pool.reprobe()
        answered = ask(pool, "claude", stream=True, stream_options={"include_usage": True})
        usage = [f for f in answered.text.split("\n\n") if f.startswith("data: {") and '"choices": []' in f]
        assert len(usage) == 1


# --- budgets ---


def test_a_request_whose_worst_case_the_budget_cannot_hold_is_never_sent(monkeypatch, provider):
    fake, url = provider
    with hosted_pool(monkeypatch, url, budget={"per_request": 0.01}) as pool:
        pool.reprobe()
        refused = ask(pool, "claude")  # up to 4 096 tokens out at $15 a million: more than a cent
        assert refused.status_code == 503 and refused.json()["reason"] == "hosted_budget_spent"
        assert fake.seen == []
        assert ask(pool, "claude", max_tokens=100).status_code == 200, "a smaller worst case fits"


def test_a_daily_budget_spent_stops_the_spill_and_status_says_what_is_left(monkeypatch, provider):
    fake, url = provider
    with hosted_pool(monkeypatch, url, budget={"per_day": 0.03}) as pool:
        pool.reprobe()
        assert ask(pool, "claude", max_tokens=600).status_code == 200
        assert ask(pool, "claude", max_tokens=600).status_code == 200
        third = ask(pool, "claude", max_tokens=600)
        assert third.status_code == 503 and third.json()["reason"] == "hosted_budget_spent"
        status = httpx.get(f"{pool.url}/pool/status", headers={"Authorization": f"Bearer {APP_KEY}"}).json()
        hosted = status["budgets"]["hosted"]
        assert hosted["per_day_usd"] == 0.03 and hosted["spent_today_usd"] == pytest.approx(2 * CLAUDE_COST)


def test_without_a_hosted_burst_no_request_reaches_a_provider(monkeypatch, provider):
    fake, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    with pool_harness(
        [EngineSpec(id="box", resident={"llama:7b"})], model_set=["llama", "claude"], catalog=CATALOG,
        extra_hosts=[{"id": "provider", "kind": "hosted", "workers": 4,
                      "transport": {"type": "http", "base_url": f"{url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}],
    ) as pool:
        pool.reprobe()
        assert ask(pool, "claude").status_code == 503
        assert fake.seen == []


# --- what a provider's refusal means ---


def test_a_busy_provider_is_tried_again_after_its_retry_after(monkeypatch, provider):
    fake, url = provider
    fake.behaviour = ["busy", "ok"]
    with hosted_pool(monkeypatch, url) as pool:
        pool.reprobe()
        answered = ask(pool, "claude")
        assert answered.status_code == 200 and len(fake.seen) == 2
        assert float(answered.headers["x-gpm-wait-s"]) >= 0.9, "waited out its retry-after"


def test_a_provider_out_of_money_is_never_retried_as_busy(monkeypatch, provider):
    fake, url = provider
    fake.behaviour = ["out_of_funds"]
    with hosted_pool(monkeypatch, url) as pool:
        pool.reprobe()
        assert ask(pool, "claude").status_code == 503
        sent = len(fake.seen)
        assert ask(pool, "claude").status_code == 503
        assert len(fake.seen) == sent == 1, "taken out of routing: not asked again"
        assert ask(pool, "llama").headers["x-gpm-host"] == "box", "the machines carry on"


def test_a_request_the_provider_refuses_is_returned_as_said_and_costs_nothing(monkeypatch, provider):
    fake, url = provider
    fake.behaviour = ["refused"]
    with hosted_pool(monkeypatch, url) as pool:
        pool.reprobe()
        answered = ask(pool, "claude")
        assert answered.status_code == 400 and "temperature" in answered.json()["error"]["message"]
        assert logged(pool, "claude")[0]["cost_usd"] == 0.0


# --- the pieces ---


def test_hosted_only_models_are_never_asked_of_a_machine_or_rented():
    config = PoolConfig.model_validate({
        "pool": {"name": "p", "model_set": ["llama", "claude"]},
        "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]}, "catalog": CATALOG,
        "hosts": [{"id": "box", "kind": "local", "engine": "ollama",
                   "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"}},
                  {"id": "provider", "kind": "hosted", "transport": {"type": "https", "base_url": "https://x/v1"}}],
    })
    box, hosted = config.hosts
    assert config.models_held_by(box) == ["llama"] and config.models_held_by(hosted) == ["llama", "claude"]
    assert config.hosted_only("claude") and not config.hosted_only("llama")
    assert "claude" not in config._rented_models()


def test_a_hosted_build_needs_a_price_and_a_machine_build_has_none():
    base = {"pool": {"name": "p", "model_set": ["claude"]}, "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]},
            "hosts": [{"id": "provider", "kind": "hosted", "transport": {"type": "https", "base_url": "https://x/v1"}}]}
    with pytest.raises(ValueError, match="hosted builds without a price"):
        PoolConfig.model_validate({**base, "catalog": {"claude": {"variants": [{"tag": "c", "engine": "openai-api"}]}}})


def test_each_providers_way_of_saying_the_money_ran_out_is_read_as_such():
    read = HostedOpenAIEngine.refusal
    assert read(429, b'{"error": {"code": "project_spend_limit_exceeded", "message": "x"}}') == "out_of_funds"
    assert read(429, b'{"error": {"type": "rate_limit_error", "message": "slow down"}}') == "busy"
    assert read(400, b'{"error": {"code": "blocked_api_access", "message": "x"}}') == "out_of_funds"
    assert read(402, b'{"error": {"message": "limit"}}') == "out_of_funds"
    assert read(400, b'{"error": {"message": "Your credit balance is too low"}}') == "out_of_funds"
    assert read(529, b"overloaded") == "busy"
    assert read(400, b'{"error": {"message": "bad temperature"}}') == "request"


def test_a_price_counts_cached_input_apart_and_a_budget_holds_before_it_spends():
    price = Price(input=3.0, cached_input=0.3, output=15.0)
    assert price.of(1000, 200, 500) == pytest.approx((800 * 3.0 + 200 * 0.3 + 500 * 15.0) / 1e6)
    budgets = HostedBudgets()
    from gpm_server.config import Budget

    limit = Budget(per_day=1.0)
    first = budgets.hold(None, limit, 0.6)
    assert first is not None and budgets.hold(None, limit, 0.6) is None, "held money counts"
    budgets.settle(first, 0.1)
    assert budgets.hold(None, limit, 0.6) is not None, "settled for less: room again"
    assert budgets.hold(None, None, 0.0) is None, "no budget: never"
