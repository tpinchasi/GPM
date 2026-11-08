"""Both API formats, to any model (S12, D169–D170): an app on OpenAI's Chat Completions and an app
on Anthropic's Messages API each reach every model the pool serves — untouched wherever the host
speaks the request's format, translated only where it does not; what cannot be carried is refused
by name unless the app asks for it to be dropped; Claude's prompts are cached by default; usage
and cost are read from the provider's own answer.

Against a fake provider that speaks both formats and a fake Ollama; nothing here spends money."""

import json

import httpx
import pytest
from fakes.fake_hosted_api import KEY, FakeHostedApi
from fakes.harness import APP_KEY, BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_server.config import PoolConfig, Price
from gpm_server.engines import anthropic_api
from gpm_server.translate import (
    AnthropicToOpenAIStream,
    OpenAIToAnthropicStream,
    Unsupported,
    anthropic_to_openai_request,
    openai_to_anthropic_request,
)

CLAUDE_PRICE = {"input": 3.0, "cached_input": 0.3, "cache_write": 3.75, "output": 15.0}
GPT_PRICE = {"input": 2.0, "cached_input": 0.2, "output": 8.0}
CATALOG = {
    "llama": {"variants": [{"tag": "llama:7b"}]},
    "claude": {"variants": [{"tag": "provider-claude", "engine": "anthropic-api", "price": CLAUDE_PRICE}]},
    "gpt": {"variants": [{"tag": "provider-gpt", "engine": "openai-api", "price": GPT_PRICE}]},
}
#: The fake's Anthropic usage — 100 fresh, 700 read from cache, 200 written, 500 out — at Claude's price.
CLAUDE_COST = (100 * 3.0 + 700 * 0.3 + 200 * 3.75 + 500 * 15.0) / 1e6
#: The fake's OpenAI usage — 1000 in (200 cached), 500 out — at GPT's price.
GPT_COST = (800 * 2.0 + 200 * 0.2 + 500 * 8.0) / 1e6


@pytest.fixture
def provider():
    loop = BackgroundLoop()
    fake = FakeHostedApi(models=("provider-claude", "provider-gpt"))
    server = ServerHandle(fake.app, loop)
    yield fake, server.base_url
    server.stop()
    loop.stop()


def hosts(url, *, cache_prompts=True):
    common = {"kind": "hosted", "workers": 4,
              "transport": {"type": "http", "base_url": f"{url}/v1", "bearer_env": "TEST_HOSTED_KEY"}}
    return [{"id": "anthropic", "engine": "anthropic-api", "cache_prompts": cache_prompts, **common},
            {"id": "openai", "engine": "openai-api", **common}]


@pytest.fixture
def pool(monkeypatch, provider):
    _, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    with pool_harness(
        [EngineSpec(id="box", resident={"llama:7b"})], model_set=["llama", "claude", "gpt"], catalog=CATALOG,
        extra_hosts=hosts(url), pool_settings={"bursts": {"hosted": {"after_s": 0.2, "budget": {"per_day": 5.0}}}},
    ) as harness:
        harness.reprobe()
        yield harness


def messages(pool, body, *, key=APP_KEY, headers=None, stream=False):
    auth = {"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01", **(headers or {})}
    return httpx.post(f"{pool.url}/v1/messages", timeout=30, headers=auth, json={**body, **({"stream": True} if stream else {})})


def chat(pool, body, *, headers=None):
    return httpx.post(f"{pool.url}/v1/chat/completions", timeout=30,
                      headers={"Authorization": f"Bearer {APP_KEY}", **(headers or {})}, json=body)


def last_cost(pool, model):
    rows = pool.database.query("SELECT cost_usd FROM request_log WHERE model_requested = ? AND outcome = 'ok' "
                               "ORDER BY id DESC LIMIT 1", (model,))
    return rows[0]["cost_usd"] if rows else None


HI = [{"role": "user", "content": "hi"}]


# --- Anthropic's SDK, to Claude: untouched, cached by default ---


def test_an_anthropic_format_app_reaches_claude_untouched_with_its_prompt_cached(pool, provider):
    fake, _ = provider
    answer = messages(pool, {"model": "claude", "max_tokens": 64, "messages": HI})
    assert answer.status_code == 200, answer.text
    assert answer.json()["content"][0]["text"] == "hello from the provider" and "X-GPM-Translated" not in answer.headers
    sent = fake.seen[-1]
    assert sent["path"] == "/v1/messages" and sent["body"]["model"] == "provider-claude"
    assert sent["body"]["cache_control"] == {"type": "ephemeral"}, "the 5-minute cache, asked for by default"
    assert sent["headers"]["authorization"] == f"Bearer {KEY}" and "x-api-key" not in sent["headers"]
    assert float(answer.headers["x-gpm-cost-usd"]) == pytest.approx(CLAUDE_COST), "reads and writes at their prices"


def test_a_request_with_its_own_cache_marks_is_sent_as_it_came(pool, provider):
    fake, _ = provider
    body = {"model": "claude", "max_tokens": 64, "system": [{"type": "text", "text": "long", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": HI}
    assert messages(pool, body).status_code == 200
    assert "cache_control" not in fake.seen[-1]["body"], "its own marks stand; none added"


def test_the_key_as_x_api_key_reaches_the_pool_and_never_the_provider(pool, provider):
    fake, _ = provider
    answer = httpx.post(f"{pool.url}/v1/messages", timeout=30, json={"model": "claude", "max_tokens": 8, "messages": HI},
                        headers={"x-api-key": APP_KEY, "anthropic-version": "2023-06-01"})
    assert answer.status_code == 200, answer.text
    assert "x-api-key" not in fake.seen[-1]["headers"]


def test_a_claude_stream_passes_as_it_came_and_is_charged_from_its_head_and_tail(pool, provider):
    answer = messages(pool, {"model": "claude", "max_tokens": 64, "messages": HI}, stream=True)
    assert answer.status_code == 200
    assert "event: message_start" in answer.text and "event: message_stop" in answer.text
    assert last_cost(pool, "claude") == pytest.approx(CLAUDE_COST), "input counts come first, output last"


def test_counting_tokens_is_passed_through_and_costs_nothing(pool):
    answer = httpx.post(f"{pool.url}/v1/messages/count_tokens", timeout=30,
                        headers={"Authorization": f"Bearer {APP_KEY}", "anthropic-version": "2023-06-01"},
                        json={"model": "claude", "messages": HI})
    assert answer.status_code == 200 and answer.json() == {"input_tokens": 1234}
    assert float(answer.headers["x-gpm-cost-usd"]) == 0.0


def test_a_host_may_say_not_to_ask_for_the_cache(monkeypatch, provider):
    fake, url = provider
    monkeypatch.setenv("TEST_HOSTED_KEY", KEY)
    with pool_harness([EngineSpec(id="box", resident={"llama:7b"})], model_set=["llama", "claude", "gpt"], catalog=CATALOG,
                      extra_hosts=hosts(url, cache_prompts=False)[:1],
                      pool_settings={"bursts": {"hosted": {"after_s": 0.2, "budget": {"per_day": 5.0}}}}) as p:
        p.reprobe()
        assert messages(p, {"model": "claude", "max_tokens": 8, "messages": HI}).status_code == 200
        assert "cache_control" not in fake.seen[-1]["body"]


# --- OpenAI's format, to Claude: translated, cached ---


def test_an_openai_format_app_reaches_claude_translated_and_cached(pool, provider):
    fake, _ = provider
    answer = chat(pool, {"model": "claude", "messages": [{"role": "system", "content": "be brief"}, *HI]})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["object"] == "chat.completion" and body["choices"][0]["message"]["content"] == "hello from the provider"
    assert body["usage"]["prompt_tokens"] == 1000 and body["usage"]["prompt_tokens_details"]["cached_tokens"] == 700
    assert answer.headers["x-gpm-translated"] == "/v1/messages -> /v1/chat/completions"
    sent = fake.seen[-1]
    assert sent["path"] == "/v1/messages" and sent["headers"]["anthropic-version"] == anthropic_api.API_VERSION
    assert sent["body"]["system"] == [{"type": "text", "text": "be brief"}] and sent["body"]["max_tokens"] == 4096
    assert sent["body"]["cache_control"] == {"type": "ephemeral"}
    assert float(answer.headers["x-gpm-cost-usd"]) == pytest.approx(CLAUDE_COST)


def test_an_openai_stream_from_claude_arrives_as_openai_chunks_with_its_usage(pool):
    answer = chat(pool, {"model": "claude", "messages": HI, "stream": True, "stream_options": {"include_usage": True}})
    assert answer.status_code == 200
    frames = [json.loads(line[6:]) for line in answer.text.splitlines() if line.startswith("data: {")]
    text = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames if f.get("choices"))
    assert text == "hello from the provider" and answer.text.rstrip().endswith("data: [DONE]")
    assert frames[-1]["usage"]["completion_tokens"] == 500
    assert last_cost(pool, "claude") == pytest.approx(CLAUDE_COST)


def test_a_tool_call_from_claude_comes_back_as_openai_tool_calls(pool, provider):
    fake, _ = provider
    fake.behaviour = ["tool"]
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    answer = chat(pool, {"model": "claude", "messages": HI, "tools": tools, "tool_choice": "required"})
    message = answer.json()["choices"][0]
    assert message["finish_reason"] == "tool_calls"
    assert json.loads(message["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    assert fake.seen[-1]["body"]["tool_choice"] == {"type": "any"}


def test_what_cannot_be_carried_is_refused_by_name_unless_the_app_asks_it_dropped(pool, provider):
    fake, _ = provider
    refused = chat(pool, {"model": "claude", "messages": HI, "logprobs": True, "n": 2})
    assert refused.status_code == 400 and refused.json()["reason"] == "unsupported_feature"
    assert "logprobs" in refused.json()["detail"] and "n" in refused.json()["detail"]
    sent = len(fake.seen)
    dropped = chat(pool, {"model": "claude", "messages": HI, "logprobs": True}, headers={"X-GPM-Unsupported": "drop"})
    assert dropped.status_code == 200 and dropped.headers["x-gpm-dropped"] == "logprobs"
    assert len(fake.seen) == sent + 1 and "logprobs" not in fake.seen[-1]["body"]


def test_a_provider_refusal_comes_back_in_the_apps_format(pool, provider):
    fake, _ = provider
    fake.behaviour = ["refused"]
    answer = chat(pool, {"model": "claude", "messages": HI})
    assert answer.status_code == 400 and answer.json()["error"]["message"] == "max_tokens: Field required"


def test_claude_out_of_money_is_read_as_such(pool, provider):
    fake, _ = provider
    fake.behaviour = ["out_of_funds"]
    assert messages(pool, {"model": "claude", "max_tokens": 8, "messages": HI}).status_code == 503
    assert pool.state.registry.by_id["anthropic"].out_of_funds


# --- Anthropic's SDK, to an OpenAI-format provider: translated ---


def test_an_anthropic_format_app_reaches_gpt_translated(pool, provider):
    fake, _ = provider
    answer = messages(pool, {"model": "gpt", "max_tokens": 64, "system": "be brief", "messages": HI,
                             "cache_control": {"type": "ephemeral"}})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["type"] == "message" and body["content"] == [{"type": "text", "text": "hello from the provider"}]
    assert body["usage"] == {"input_tokens": 800, "output_tokens": 500, "cache_read_input_tokens": 200,
                             "cache_creation_input_tokens": 0}
    sent = fake.seen[-1]
    assert sent["path"] == "/v1/chat/completions" and sent["body"]["max_completion_tokens"] == 64
    assert sent["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert not any(k.startswith("anthropic-") for k in sent["headers"]), "its format's headers stay behind"
    assert float(answer.headers["x-gpm-cost-usd"]) == pytest.approx(GPT_COST)


def test_an_anthropic_stream_from_gpt_arrives_as_anthropic_events(pool, provider):
    answer = messages(pool, {"model": "gpt", "max_tokens": 64, "messages": HI}, stream=True)
    events = [line[7:] for line in answer.text.splitlines() if line.startswith("event: ")]
    assert events[0] == "message_start" and events[-2:] == ["message_delta", "message_stop"]
    deltas = [json.loads(line[6:]) for line in answer.text.splitlines() if line.startswith("data: ")]
    text = "".join(d["delta"].get("text", "") for d in deltas if d["type"] == "content_block_delta")
    assert text == "hello from the provider"
    assert [d for d in deltas if d["type"] == "message_delta"][0]["usage"]["output_tokens"] == 500
    assert last_cost(pool, "gpt") == pytest.approx(GPT_COST)


def test_a_tool_call_from_gpt_comes_back_as_tool_use(pool, provider):
    fake, _ = provider
    fake.behaviour = ["tool"]
    answer = messages(pool, {"model": "gpt", "max_tokens": 64, "messages": HI,
                             "tools": [{"name": "weather", "input_schema": {"type": "object"}}]})
    assert answer.json()["stop_reason"] == "tool_use"
    assert answer.json()["content"] == [{"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Paris"}}]


def test_thinking_for_gpt_is_refused_unless_dropped(pool):
    body = {"model": "gpt", "max_tokens": 64, "messages": HI, "thinking": {"type": "enabled", "budget_tokens": 1024}}
    refused = messages(pool, body)
    assert refused.status_code == 400
    assert refused.json()["type"] == "error" and refused.json()["error"]["type"] == "invalid_request_error"
    assert refused.json()["reason"] == "unsupported_feature" and "thinking" in refused.json()["error"]["message"]
    assert messages(pool, body, headers={"X-GPM-Unsupported": "drop"}).status_code == 200


# --- Anthropic's SDK, to the pool's own machines: untouched ---


def test_an_anthropic_format_app_reaches_a_machine_untouched(pool):
    answer = messages(pool, {"model": "llama", "max_tokens": 16, "messages": HI})
    assert answer.status_code == 200, answer.text
    assert answer.headers["x-gpm-host"] == "box" and answer.json()["content"][0]["text"] == "echo:hi"
    assert "X-GPM-Translated" not in answer.headers


# --- the pool's own refusals, in Anthropic's shape ---


def test_the_pools_refusals_on_anthropics_path_are_in_its_error_shape(pool):
    unknown = messages(pool, {"model": "nothing", "max_tokens": 8, "messages": HI})
    assert unknown.status_code == 404
    assert unknown.json()["type"] == "error" and unknown.json()["error"]["type"] == "not_found_error"
    assert unknown.json()["reason"] == "model_not_in_pool"
    stranger = messages(pool, {"model": "claude", "max_tokens": 8, "messages": HI}, key="wrong")
    assert stranger.status_code == 401 and stranger.json()["error"]["type"] == "authentication_error"


# --- the pieces ---


def test_an_anthropic_build_needs_its_cache_write_price():
    base = {"pool": {"name": "p", "model_set": ["claude"]}, "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]},
            "hosts": [{"id": "a", "kind": "hosted", "engine": "anthropic-api",
                       "transport": {"type": "https", "base_url": "https://x/v1"}}]}
    with pytest.raises(ValueError, match="cache_write"):
        PoolConfig.model_validate({**base, "catalog": {"claude": {"variants": [
            {"tag": "c", "engine": "anthropic-api", "price": {"input": 3, "output": 15}}]}}})
    with pytest.raises(ValueError, match="openai-api.*or.*anthropic-api"):
        PoolConfig.model_validate({**base, "hosts": [{**base["hosts"][0], "engine": "ollama"}],
                                   "catalog": {"claude": {"variants": [{"tag": "c"}]}}})


def test_an_unpriced_cache_write_is_charged_the_dearest_rate():
    price = Price(input=2.0, output=8.0)
    assert price.of(1000, 0, 0, written=1000) == pytest.approx(1000 * 5.0 / 1e6)


def test_the_cache_mark_is_added_only_where_the_request_has_none():
    body = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'
    marked, added = anthropic_api.with_cache_mark(body)
    assert added and json.loads(marked)["cache_control"] == {"type": "ephemeral"}
    assert marked.endswith(body[1:]), "every other byte as it came"
    nested = b'{"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]}'
    assert anthropic_api.with_cache_mark(nested) == (nested, False)


def test_streams_are_translated_however_their_bytes_are_split():
    upstream = (b'event: message_start\ndata: {"type":"message_start","message":{"model":"c","usage":{"input_tokens":5}}}\n\n'
                b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ab"}}\n\n'
                b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
                b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    rewriter = AnthropicToOpenAIStream(include_usage=True)
    out = b"".join(rewriter.feed(upstream[i:i + 7]) for i in range(0, len(upstream), 7))
    frames = [json.loads(line[6:]) for line in out.decode().splitlines() if line.startswith("data: {")]
    assert "".join(f["choices"][0]["delta"].get("content") or "" for f in frames if f["choices"]) == "ab"
    assert frames[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                                   "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}

    back = OpenAIToAnthropicStream()
    chunks = (b'data: {"model":"g","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
              b'data: {"model":"g","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
              b'data: [DONE]\n\n')
    out = b"".join(back.feed(chunks[i:i + 5]) for i in range(0, len(chunks), 5)) + back.finish_stream()
    kinds = [line[7:] for line in out.decode().splitlines() if line.startswith("event: ")]
    assert kinds == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                     "message_delta", "message_stop"]
    assert '"stop_reason":"max_tokens"' in out.decode()


def test_each_direction_carries_tools_and_refuses_what_it_cannot():
    out, dropped = openai_to_anthropic_request(json.dumps({
        "model": "c", "messages": [*HI, {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function",
                                                                            "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
                                   {"role": "tool", "tool_call_id": "t1", "content": "42"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}).encode())
    request = json.loads(out)
    assert dropped == [] and request["messages"][1]["content"][0] == {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}
    assert request["messages"][2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "42"}]}
    with pytest.raises(Unsupported, match="response_format"):
        openai_to_anthropic_request(b'{"model":"c","messages":[],"response_format":{"type":"json_object"}}')
    with pytest.raises(Unsupported, match="document"):
        anthropic_to_openai_request(json.dumps({"model": "g", "max_tokens": 5, "messages": [
            {"role": "user", "content": [{"type": "document", "source": {"type": "base64", "data": "x"}}]}]}).encode())
    with pytest.raises(Unsupported, match="web_search"):
        anthropic_to_openai_request(json.dumps({"model": "g", "max_tokens": 5, "messages": HI,
                                                "tools": [{"type": "web_search_20250305", "name": "web_search"}]}).encode())
