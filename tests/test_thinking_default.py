"""A build's thinking default (D187): added where the app says nothing, never over its word.

Found live: Gemma 4 on a rented machine's Ollama thinks before answering — a small `max_tokens`
came back empty — while the same model on Ollama's cloud answered at once.
"""

import json
from pathlib import Path

import pytest
from fakes.fake_ollama import inference_seen
from fakes.harness import EngineSpec, pool_harness
from gpm_server.config import Variant
from gpm_server.engines.thinking import with_thinking

CHAT = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}'


@pytest.mark.parametrize("engine, path, setting, member", [
    ("ollama", "/api/chat", "off", ("think", False)),
    ("ollama", "/api/chat", "on", ("think", True)),
    ("ollama", "/v1/chat/completions", "off", ("reasoning_effort", "none")),
    ("ollama", "/v1/chat/completions", "on", ("reasoning_effort", "medium")),
    ("openai-api", "/v1/chat/completions", "off", ("reasoning_effort", "none")),
    ("vllm", "/v1/chat/completions", "off", ("chat_template_kwargs", {"enable_thinking": False})),
    ("vllm", "/v1/chat/completions", "on", ("chat_template_kwargs", {"enable_thinking": True})),
    ("anthropic-api", "/v1/messages", "off", ("thinking", {"type": "disabled"})),
    ("anthropic-api", "/v1/messages", "on", ("thinking", {"type": "adaptive"})),
    ("vllm", "/v1/messages", "off", ("thinking", {"type": "disabled"})),
])
def test_the_engines_own_switch_is_added_and_nothing_else_changes(engine, path, setting, member):
    sent = with_thinking(engine, path, CHAT, setting)
    key, value = member
    assert json.loads(sent) == {**json.loads(CHAT), key: value}
    assert sent.endswith(CHAT[1:]), "every byte the app wrote is still there, in order"


@pytest.mark.parametrize("path, said", [
    ("/api/chat", {"think": True}),
    ("/v1/chat/completions", {"reasoning_effort": "high"}),
    ("/v1/chat/completions", {"reasoning": {"effort": "low"}}),
    ("/v1/chat/completions", {"think": True}),
    ("/v1/chat/completions", {"chat_template_kwargs": {"enable_thinking": True}}),
    ("/v1/messages", {"thinking": {"type": "enabled", "budget_tokens": 2048}}),
])
def test_the_apps_own_word_is_never_changed(path, said):
    body = json.dumps({"model": "m", "messages": [], **said}).encode()
    assert with_thinking("vllm", path, body, "off") == body
    assert with_thinking("ollama", path, body, "off") == body


@pytest.mark.parametrize("path, body", [
    ("/v1/embeddings", b'{"model": "m", "input": "x"}'),
    ("/v1/audio/transcriptions", b"--multipart--"),
    ("/v1/chat/completions", b"not json"),
])
def test_other_requests_pass_untouched(path, body):
    assert with_thinking("ollama", path, body, "off") == body


def test_unset_leaves_the_engines_default_and_on_is_not_invented_where_there_is_no_switch():
    assert with_thinking("ollama", "/v1/chat/completions", CHAT, None) == CHAT
    assert with_thinking("ollama", "/v1/messages", CHAT, "on") == CHAT, "no known way to turn it on there"


def test_an_empty_object_stays_valid():
    assert json.loads(with_thinking("ollama", "/api/chat", b"{ }", "off")) == {"think": False}


def test_thinking_is_a_chat_builds_setting():
    assert Variant(tag="g", thinking="off").thinking == "off"
    with pytest.raises(ValueError, match="thinking"):
        Variant(tag="e", serves=["embed"], thinking="off")


def test_through_the_router_the_build_default_reaches_the_engine():
    catalog = {"m1": {"variants": [{"tag": "m1-quiet", "thinking": "off"}]}}
    with pool_harness([EngineSpec(id="gpu", resident={"m1-quiet"}, workers=1)], model_set=["m1"],
                      catalog=catalog) as pool, pool.client() as client:
        assert client.post("/api/chat", json={"model": "m1", "messages": [], "stream": False}).status_code == 200
        assert client.post("/api/chat", json={"model": "m1", "messages": [], "stream": False,
                                              "think": True}).status_code == 200
        sent = [json.loads(body) for path, body, _ in inference_seen(pool.engines["gpu"].fake) if path == "/api/chat"]
    assert sent[0]["think"] is False and sent[0]["model"] == "m1-quiet"
    assert sent[1]["think"] is True, "the app's own word wins"


def test_one_build_under_two_names_has_one_thinking_default():
    """Found in review: the router finds the default by tag, so `quick` (off) and `deep` (on) on
    one build answered `quick` with thinking on."""
    import yaml
    from gpm_server.config import PoolConfig

    base = yaml.safe_load((Path(__file__).resolve().parents[1] / "server/examples/pool.yaml").read_text())
    base["pool"]["model_set"] = ["quick", "deep"]
    base["catalog"] = {"quick": {"variants": [{"tag": "g4", "thinking": "off"}]},
                       "deep": {"variants": [{"tag": "g4", "thinking": "on"}]}}
    with pytest.raises(ValueError, match="different `thinking`"):
        PoolConfig.model_validate(base)


def test_a_thinking_field_the_translation_dropped_is_still_the_apps_word():
    """An OpenAI-shaped request with `reasoning_effort`, translated for Anthropic's API with
    `X-GPM-Unsupported: drop`: the build's default must not stand in for what the app said."""
    from gpm_server.engines.thinking import says_thinking

    assert says_thinking("/v1/chat/completions", b'{"model": "m", "reasoning_effort": "none"}')
    assert not says_thinking("/v1/chat/completions", b'{"model": "m"}')
    assert not says_thinking("/v1/embeddings", b'{"model": "m", "think": true}')
