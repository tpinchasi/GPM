"""A machine's Ollama never reaches Ollama's cloud (D168): a signed-in Ollama serves `:cloud` and
`-cloud` tags billed per token to its account, with no key the pool sees, so no budget or lease
would count them. The configuration refuses such a tag on a machine's build; the Ollama the agent
starts, and the one the pool starts on a rented machine, run with `OLLAMA_NO_CLOUD=1`; the agent
holds none; and an Ollama the pool did not start, which may be signed in, never has one taken for a
model on its disk. Ollama's cloud is reached as a hosted API, metered like any other."""

import json

import httpx
import pytest
from gpm_agent.engines import OllamaFacts
from gpm_agent.engines import cloud_tag as agent_cloud_tag
from gpm_agent.models import ModelWork, Refused
from gpm_agent.ollama_server import OllamaServer
from gpm_server.config import PoolConfig, cloud_tag
from gpm_server.engines.ollama import OllamaEngine, _tags_in

BASE = {
    "auth": {"app_keys": ["gpma_x"], "admin_keys": ["gpmx_x"]},
    "hosts": [{"id": "box", "kind": "local", "engine": "ollama",
               "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"}}],
}


@pytest.mark.parametrize("tag, cloud", [
    ("glm-4.6:cloud", True), ("gpt-oss:120b-cloud", True), ("Qwen3-Coder:480B-Cloud", True),
    ("gemma3:27b", False), ("cloudy:7b", False), ("org/model-cloud", False), ("llama3", False),
    ("library/qwen:cloud-7b", False),
])
def test_what_a_cloud_tag_is(tag, cloud):
    assert cloud_tag(tag) is cloud and agent_cloud_tag(tag) is cloud, "the agent and the pool agree"


def test_the_configuration_refuses_ollamas_cloud_on_a_machine():
    with pytest.raises(ValueError, match="name Ollama's cloud"):
        PoolConfig.model_validate({**BASE, "pool": {"name": "p", "model_set": ["big"]},
                                   "catalog": {"big": {"variants": [{"tag": "gpt-oss:120b-cloud"}]}}})
    with pytest.raises(ValueError, match="Reach Ollama's cloud as a hosted API"):
        PoolConfig.model_validate({**BASE, "pool": {"name": "p", "model_set": ["glm-4.6:cloud"]}})
    # As a hosted API it is a host like any other, priced and metered.
    config = PoolConfig.model_validate({
        **BASE, "pool": {"name": "p", "model_set": ["big"]},
        "hosts": [*BASE["hosts"], {"id": "ollama-cloud", "kind": "hosted",
                                   "transport": {"type": "https", "base_url": "https://ollama.com/v1"}}],
        "catalog": {"big": {"variants": [{"tag": "gpt-oss:120b"},
                                         {"tag": "gpt-oss:120b-cloud", "engine": "openai-api",
                                          "price": {"input": 0.1, "output": 0.5}}]}},
    })
    assert config.catalog["big"].variants[1].hosted


def test_the_pool_starts_a_rented_machines_ollama_without_the_cloud():
    assert OllamaEngine().launch_settings(workers=4, context=8192, n_models=1)["OLLAMA_NO_CLOUD"] == "1"


def test_a_signed_in_ollamas_cloud_models_are_never_taken_for_its_own():
    listed = [
        {"name": "gemma3:27b", "model": "gemma3:27b"},
        {"name": "gpt-oss:120b-cloud", "model": "gpt-oss:120b-cloud", "remote_host": "https://ollama.com:443"},
        {"name": "glm-4.6:cloud", "model": "glm-4.6:cloud"},
        {"name": "kimi-k2:1t", "remote_model": "kimi-k2:1t"},
    ]
    assert _tags_in(listed) == frozenset({"gemma3:27b"})


def test_the_agents_ollama_runs_without_the_cloud_whatever_it_was_given(tmp_path):
    server = OllamaServer(tmp_path / "ollama", engine_url="http://127.0.0.1:11555", models_path=str(tmp_path / "m"),
                          client=httpx.AsyncClient())
    (tmp_path / "ollama").mkdir()
    environment = server.environment()
    assert environment["OLLAMA_NO_CLOUD"] == "1" and environment["OLLAMA_HOST"] == "127.0.0.1:11555"


async def test_the_agent_reports_no_cloud_model_and_holds_none():
    answers = {
        "/api/version": {"version": "0.40.13"},
        "/api/tags": {"models": [{"name": "gemma3:27b", "size": 10},
                                 {"name": "gpt-oss:120b-cloud", "size": 384, "remote_host": "https://ollama.com:443"}]},
        "/api/ps": {"models": [{"name": "glm-4.6:cloud"}, {"name": "gemma3:27b"}]},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps(answers[request.url.path])))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:11434") as client:
        facts = await OllamaFacts().describe(client)
    assert [m["tag"] for m in facts["models_on_disk"]] == ["gemma3:27b"]
    assert facts["models_loaded"] == ["gemma3:27b"]

    work = ModelWork(OllamaFacts(), None, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=False)
    with pytest.raises(Refused) as refused:
        await work.apply({"tags": ["gemma3:27b", "gpt-oss:120b-cloud"], "residency": "pinned"})
    assert refused.value.error == "cloud_tag" and "Ollama's cloud" in refused.value.detail
