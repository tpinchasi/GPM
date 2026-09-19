"""Phase 1's exit criterion: the router must not change what the engine does.

Streaming, tool calls, structured output and cancel-on-disconnect, through the router and
direct to the engine, compared byte for byte (docs/roadmap.md §2).
"""

import json
import time

import pytest
from fakes.harness import EngineSpec, pool_harness

MODEL = "m1"
CATALOG_WITH_STRICT = {
    "m1": {
        "variants": [
            {"tag": "m1-fast", "runtime_class": "fast", "enforces_schema": False},
            {"tag": "m1-strict", "runtime_class": "strict", "enforces_schema": True},
        ]
    }
}


@pytest.fixture
def pool():
    with pool_harness(
        [EngineSpec(id="local-1", resident={MODEL}, workers=2)],
        model_set=[MODEL],
    ) as harness:
        yield harness


def chat_body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hello there"}], **extra}


def test_a_non_streaming_reply_is_identical_through_the_router(pool):
    body = chat_body(stream=False)
    with pool.client() as through, pool.direct_client("local-1") as direct:
        routed = through.post("/api/chat", json=body)
        straight = direct.post("/api/chat", json=body)

    assert routed.status_code == straight.status_code == 200
    assert routed.content == straight.content
    assert routed.headers["content-type"] == straight.headers["content-type"]


def test_a_streamed_reply_is_identical_through_the_router(pool):
    body = chat_body(stream=True)
    with pool.client() as through, pool.direct_client("local-1") as direct:
        with through.stream("POST", "/api/chat", json=body) as response:
            routed = b"".join(response.iter_raw())
            routed_type = response.headers["content-type"]
        with direct.stream("POST", "/api/chat", json=body) as response:
            straight = b"".join(response.iter_raw())
            straight_type = response.headers["content-type"]

    assert routed == straight
    assert routed_type == straight_type
    assert routed.count(b"\n") > 1  # really streamed, not buffered into one frame


def test_tool_calls_survive_the_round_trip(pool):
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        }
    ]
    body = chat_body(stream=False, tools=tools)
    with pool.client() as through, pool.direct_client("local-1") as direct:
        routed = through.post("/api/chat", json=body).json()
        straight = direct.post("/api/chat", json=body).json()

    assert routed == straight
    assert routed["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_structured_output_survives_the_round_trip(pool):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    body = chat_body(stream=False, format=schema)
    with pool.client() as through, pool.direct_client("local-1") as direct:
        routed = through.post("/api/chat", json=body)
        straight = direct.post("/api/chat", json=body)

    assert routed.content == straight.content
    assert json.loads(routed.json()["message"]["content"]) == {"answer": "42"}


def test_the_forwarded_request_differs_only_in_the_model_field(pool):
    sent = json.dumps(chat_body(stream=False), separators=(", ", ": ")).encode()
    with pool.client() as through:
        through.post("/api/chat", content=sent, headers={"content-type": "application/json"})

    path, received, _ = pool.engines["local-1"].fake.received[-1]
    assert path == "/api/chat"
    assert received == sent  # no catalog entry: same tag, so byte-identical


def test_the_model_field_is_rewritten_but_nothing_else_is():
    with pool_harness(
        [EngineSpec(id="local-1", resident={"m1-fast"}, workers=1)],
        model_set=[MODEL],
        catalog=CATALOG_WITH_STRICT,
    ) as pool:
        sent = json.dumps(chat_body(stream=False), separators=(", ", ": ")).encode()
        with pool.client() as through:
            response = through.post("/api/chat", content=sent, headers={"content-type": "application/json"})

        _, received, _ = pool.engines["local-1"].fake.received[-1]
        assert received == sent.replace(b'"m1"', b'"m1-fast"')
        assert response.headers["X-GPM-Served-Model"] == "m1-fast"
        assert response.headers["X-GPM-Runtime-Class"] == "fast"


def test_a_schema_request_goes_to_the_build_that_enforces_one():
    with pool_harness(
        [EngineSpec(id="local-1", resident={"m1-fast", "m1-strict"}, workers=2)],
        model_set=[MODEL],
        catalog=CATALOG_WITH_STRICT,
    ) as pool:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        with pool.client() as through:
            plain = through.post("/api/chat", json=chat_body(stream=False))
            strict = through.post("/api/chat", json=chat_body(stream=False, format=schema))

        assert plain.headers["X-GPM-Served-Model"] == "m1-fast"
        assert strict.headers["X-GPM-Served-Model"] == "m1-strict"


def test_a_schema_request_is_refused_when_no_resident_build_enforces_one():
    with pool_harness(
        [EngineSpec(id="local-1", resident={"m1-fast"}, workers=1)],
        model_set=[MODEL],
        catalog=CATALOG_WITH_STRICT,
    ) as pool:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        with pool.client() as through:
            response = through.post("/api/chat", json=chat_body(stream=False, format=schema))

        assert response.status_code == 503
        assert response.json()["reason"] == "no_eligible_host"


def test_the_engines_own_refusal_comes_back_as_the_engines_own_refusal(pool):
    """The pool adds a dialect; it does not reinterpret what the engine says."""
    body = {"model": MODEL}  # the engine requires `messages`; the pool has no opinion
    with pool.client() as through, pool.direct_client("local-1") as direct:
        routed = through.post("/api/chat", json=body)
        straight = direct.post("/api/chat", json=body)

    assert routed.status_code == straight.status_code == 400
    assert routed.content == straight.content
    assert routed.headers["X-GPM-Host"] == "local-1"


def test_the_client_going_away_cancels_the_work_upstream():
    with pool_harness(
        [EngineSpec(id="local-1", resident={MODEL}, workers=1, chunk_delay_s=0.3, chunks=6)],
        model_set=[MODEL],
    ) as pool:
        fake = pool.engines["local-1"].fake
        with pool.client(timeout=10) as through:
            with through.stream("POST", "/api/chat", json=chat_body(stream=True)) as response:
                next(response.iter_raw())  # take the first frame, then walk away

        deadline = time.monotonic() + 5
        while fake.cancelled == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert fake.cancelled == 1
        assert fake.completed == 0

        # The worker is free again straight away, not at the end of the abandoned generation.
        with pool.client() as through:
            status = through.get("/pool/status").json()
        assert status["capacity"]["workers_busy"] == 0
