"""Test connection speaks to each engine on its own paths (D142): the concurrency check measures
a chat build with the shortest generation that engine offers, and a whisper.cpp server — which its
agent starts once the host is saved — is not failed for not answering before then (D148)."""

import asyncio

import httpx
from gpm_server import hostcheck
from gpm_server.config import CatalogEntry, PoolSettings
from gpm_server.engines import get_engine
from gpm_server.hostcheck import _chat_builds, _one_token


def test_the_shortest_generation_is_asked_on_the_engines_own_path():
    path, body = _one_token(get_engine("ollama"), "chat:7b")
    assert path == "/api/generate" and body["options"] == {"num_predict": 1}
    path, body = _one_token(get_engine("vllm"), "owner/chat")
    assert path == "/v1/completions" and body["max_tokens"] == 1
    assert _one_token(get_engine("whisper-cpp"), "owner/w/model.bin") is None


def test_only_a_chat_build_is_measured():
    catalog = {
        "speech": CatalogEntry.model_validate({"variants": [{"tag": "owner/speech", "serves": ["transcribe"]}]}),
        "ranker": CatalogEntry.model_validate({"variants": [{"tag": "owner/ranker", "serves": ["rerank"]}]}),
        "chat": CatalogEntry.model_validate({"variants": [{"tag": "owner/chat"}]}),
    }
    required = {"owner/speech", "owner/ranker", "owner/chat", "plain:7b"}
    assert _chat_builds(required, ["speech", "ranker", "chat", "plain:7b"], catalog) == ["owner/chat", "plain:7b"]
    assert _chat_builds({"owner/speech"}, ["speech"], catalog) == []


def test_a_whisper_cpp_server_not_started_yet_is_expected_before_the_host_is_saved(monkeypatch):
    not_yet = httpx.MockTransport(lambda request: httpx.Response(503))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(*a, **{**k, "transport": not_yet}))
    result = asyncio.run(hostcheck.test_connection(
        {"base_url": "http://127.0.0.1:8178", "workers": 1},
        engine=get_engine("whisper-cpp"), settings=PoolSettings(name="t", model_set=["speech"]),
        model_set=["speech"], catalog={},
    ))
    assert result["ok"], result
    steps = {step["name"]: step for step in result["steps"]}
    assert "agent starts it" in steps["reach"]["detail"]
