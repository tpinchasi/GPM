"""Phase 1's exit criterion against a real engine on a real GPU.

Opt-in: `pytest -m integration`. Skipped unless an Ollama is reachable and holds the models
these tests name (`GPM_IT_CHAT_MODEL`, `GPM_IT_EMBED_MODEL`).

The pool never loads a model — these tests act as the operator and pin the models resident
before the pool is pointed at the host, which is exactly what an operator does for a `local`
or `fixed-remote` host.
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest
from fakes.harness import APP_KEY, PoolHarness

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CHAT_MODEL = os.environ.get("GPM_IT_CHAT_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.environ.get("GPM_IT_EMBED_MODEL", "nomic-embed-text:latest")
DETERMINISTIC = {"seed": 42, "temperature": 0}


def _available() -> set[str]:
    try:
        response = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"no Ollama at {OLLAMA}: {exc}")
    return {entry["name"] for entry in response.json().get("models", [])}


@pytest.fixture(scope="module")
def ollama_models() -> list[str]:
    present = _available()
    missing = {CHAT_MODEL, EMBED_MODEL} - present
    if missing:
        pytest.skip(f"Ollama at {OLLAMA} does not hold {sorted(missing)}")

    # The operator's job, not the pool's: load both and keep them loaded.
    with httpx.Client(base_url=OLLAMA, timeout=600) as client:
        client.post("/api/generate", json={"model": CHAT_MODEL, "keep_alive": -1}).raise_for_status()
        client.post("/api/embed", json={"model": EMBED_MODEL, "input": ["warm"], "keep_alive": -1}).raise_for_status()
    return [CHAT_MODEL, EMBED_MODEL]


@pytest.fixture(scope="module")
def pool(ollama_models: list[str]):
    """A pool of one local host: the real engine, which the pool did not start."""
    harness = PoolHarness(
        [],
        model_set=ollama_models,
        queue_timeout_s=60,
        upstream_read_timeout_s=600,
        extra_hosts=[
            {
                "id": "local-ollama",
                "kind": "local",
                "workers": 2,
                "transport": {"type": "http", "base_url": OLLAMA},
            }
        ],
    )
    try:
        yield harness
    finally:
        harness.close()


def chat_body(prompt: str, **extra) -> dict:
    return {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": DETERMINISTIC,
        "think": False,
        **extra,
    }


def test_the_host_is_ready_only_with_the_whole_model_set_resident(pool):
    with pool.client(timeout=60) as client:
        status = client.get("/pool/status").json()
    host = status["hosts"][0]
    assert host["state"] == "ready", host["last_error"]
    assert set(status["model_set"]) <= set(host["resident"])


def test_a_non_streaming_reply_keeps_the_engines_own_shape(pool):
    body = chat_body("Reply with exactly: pong", stream=False)
    with pool.client(timeout=600) as through, httpx.Client(base_url=OLLAMA, timeout=600) as direct:
        routed = through.post("/api/chat", json=body)
        straight = direct.post("/api/chat", json=body)

    assert routed.status_code == straight.status_code == 200
    assert set(routed.json()) == set(straight.json())
    assert routed.json()["model"] == CHAT_MODEL
    assert routed.json()["message"]["content"].strip()
    assert routed.headers["content-type"] == straight.headers["content-type"]
    assert routed.headers["X-GPM-Served-Model"] == CHAT_MODEL
    assert routed.headers["X-GPM-Host"] == "local-ollama"


def test_the_bytes_match_wherever_the_engine_is_deterministic(pool):
    body = chat_body("Name one primary colour. One word.", stream=False)
    with httpx.Client(base_url=OLLAMA, timeout=600) as direct:
        first = direct.post("/api/chat", json=body).json()["message"]["content"]
        second = direct.post("/api/chat", json=body).json()["message"]["content"]
    if first != second:
        pytest.skip("this engine is not reproducible at temperature 0; nothing to compare against")

    with pool.client(timeout=600) as through:
        routed = through.post("/api/chat", json=body).json()["message"]["content"]
    assert routed == first


def test_a_streamed_reply_arrives_as_frames_not_as_one_lump(pool):
    body = chat_body("Count from one to ten in words.", stream=True)
    frames, arrivals = [], []
    with pool.client(timeout=600) as through:
        with through.stream("POST", "/api/chat", json=body) as response:
            assert response.headers["content-type"].startswith("application/x-ndjson")
            assert response.headers["X-GPM-Served-Model"] == CHAT_MODEL
            for line in response.iter_lines():
                if line.strip():
                    frames.append(json.loads(line))
                    arrivals.append(time.monotonic())

    assert len(frames) > 2
    assert frames[-1]["done"] is True
    assert "".join(f["message"]["content"] for f in frames).strip()
    assert arrivals[-1] - arrivals[0] > 0.05


def test_tool_calls_come_back_intact(pool):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    body = chat_body("What is the weather in Paris? Use the tool.", stream=False, tools=tools)
    with pool.client(timeout=600) as through, httpx.Client(base_url=OLLAMA, timeout=600) as direct:
        routed = through.post("/api/chat", json=body).json()
        straight = direct.post("/api/chat", json=body).json()

    if "tool_calls" not in straight["message"]:
        pytest.skip("this model did not call the tool even directly; nothing to compare")
    calls = routed["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "get_weather"
    assert "city" in calls[0]["function"]["arguments"]


def test_structured_output_still_obeys_the_schema(pool):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
        "required": ["city", "population"],
    }
    body = chat_body("Give the city of Paris and its population.", stream=False, format=schema)
    with pool.client(timeout=600) as through:
        routed = through.post("/api/chat", json=body)

    assert routed.status_code == 200
    parsed = json.loads(routed.json()["message"]["content"])
    assert set(parsed) >= {"city", "population"}
    assert isinstance(parsed["population"], int)


def test_embeddings_pass_through(pool):
    body = {"model": EMBED_MODEL, "input": ["one", "two"]}
    with pool.client(timeout=600) as through, httpx.Client(base_url=OLLAMA, timeout=600) as direct:
        routed = through.post("/api/embed", json=body).json()
        straight = direct.post("/api/embed", json=body).json()

    assert len(routed["embeddings"]) == 2
    assert len(routed["embeddings"][0]) == len(straight["embeddings"][0])


def test_a_client_that_walks_away_frees_the_worker_at_once(pool):
    body = chat_body("Write a long essay about the sea.", stream=True)
    with pool.client(timeout=600) as through:
        with through.stream("POST", "/api/chat", json=body) as response:
            next(response.iter_lines())  # one frame, then leave

    with pool.client(timeout=60) as client:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if client.get("/pool/status").json()["capacity"]["workers_busy"] == 0:
                break
            time.sleep(0.1)
        assert client.get("/pool/status").json()["capacity"]["workers_busy"] == 0


def test_a_model_the_pool_does_not_serve_is_refused_not_pulled(pool):
    with pool.client(timeout=60) as client:
        response = client.post("/api/chat", json={"model": "tinyllama:nonexistent", "messages": []})
    assert response.status_code == 404
    assert response.json()["reason"] == "model_not_in_pool"
