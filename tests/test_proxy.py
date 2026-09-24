"""One endpoint in front of several engine processes on one machine (D96).

The first of the two shapes the owner asked for: every model on a host, each in its own engine
process, behind one port — because the pool dials one URL per host. Against fake upstreams;
nothing here needs a GPU or vLLM.
"""

import json

import httpx
import pytest
from gpm_agent.proxy import Upstreams, create_app

BIG, SMALL = "gemma4:26b", "gemma4:e4b"


def upstream_file(tmp_path, mapping: dict) -> str:
    path = tmp_path / "upstreams.json"
    path.write_text(json.dumps(mapping))
    return str(path)


async def _frames():
    """A genuinely streamed body: the proxy forwards raw bytes without decoding them, and a
    pre-read fake would not exercise that path at all."""
    yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    yield b"data: [DONE]\n\n"


class Engines:
    """Two engine processes, each serving one model, each on its own URL."""

    def __init__(self, healthy=(BIG, SMALL)):
        self.healthy = set(healthy)
        self.seen: list[tuple[str, str, bytes]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            which = BIG if request.url.port == 8001 else SMALL
            path = request.url.path
            self.seen.append((which, path, request.content))
            if path == "/health":
                return httpx.Response(200 if which in self.healthy else 503)
            if path == "/metrics":
                return httpx.Response(200, text="vllm:num_requests_running 3.0\n")
            body = json.loads(request.content or b"{}")
            if body.get("stream"):
                # A real streaming response, not a pre-read one: the proxy forwards raw bytes
                # without decoding them, and a buffered fake would not exercise that.
                return httpx.Response(
                    200,
                    content=_frames(),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json={"served_by": which, "echo": body})

        return httpx.MockTransport(handler)


def proxy_for(tmp_path, engines: Engines, mapping=None) -> httpx.AsyncClient:
    mapping = mapping if mapping is not None else {
        BIG: "http://127.0.0.1:8001", SMALL: "http://127.0.0.1:8002",
    }
    app = create_app(
        upstream_file(tmp_path, mapping),
        client=httpx.AsyncClient(transport=engines.transport(), timeout=10),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


# --- routing by the model a request names ---


async def test_a_request_goes_to_the_engine_serving_the_model_it_names(tmp_path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG, "messages": []})

    assert answer.status_code == 200
    assert answer.json()["served_by"] == BIG


async def test_each_model_reaches_its_own_engine(tmp_path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        for model in (BIG, SMALL, BIG):
            await proxy.post("/v1/chat/completions", json={"model": model, "messages": []})

    assert [which for which, _, _ in engines.seen] == [BIG, SMALL, BIG]


async def test_the_body_arrives_byte_for_byte(tmp_path):
    """It routes; it does not translate. Translation is where tool-calling and structured
    output fidelity get lost — the same rule the pool itself obeys."""
    engines = Engines()
    body = b'{"model":"gemma4:26b","messages":[{"role":"user","content":"hi"}],"temperature":0.7}'
    async with proxy_for(tmp_path, engines) as proxy:
        await proxy.post("/v1/chat/completions", content=body,
                         headers={"content-type": "application/json"})

    (_, path, seen) = engines.seen[-1]
    assert seen == body
    assert path == "/v1/chat/completions", "the path is carried through unchanged"


async def test_a_model_this_host_does_not_serve_is_refused_by_name(tmp_path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": "gemma4:31b"})

    assert answer.status_code == 404
    detail = answer.json()["error"]["message"]
    assert "gemma4:31b" in detail and BIG in detail, detail
    assert not engines.seen, "nothing was asked of any engine"


async def test_a_body_naming_no_model_is_refused_rather_than_guessed(tmp_path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"messages": []})
    assert answer.status_code == 404


# --- streaming ---


async def test_a_stream_is_carried_through(tmp_path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        async with proxy.stream(
            "POST", "/v1/chat/completions", json={"model": BIG, "stream": True}
        ) as answer:
            frames = [line async for line in answer.aiter_lines() if line.strip()]

    assert frames[0].startswith("data: ")
    assert frames[-1] == "data: [DONE]"


# --- what the pool asks of the host as a whole ---


async def test_the_host_reports_every_model_it_serves(tmp_path):
    """The pool sees one host holding a set, which is exactly what it is."""
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.get("/v1/models")

    assert [entry["id"] for entry in answer.json()["data"]] == sorted([BIG, SMALL])


async def test_the_host_is_healthy_only_when_every_engine_is(tmp_path):
    """A machine serving two models of three is not one the pool can treat as ready: a request
    for the missing one would be refused after being routed here."""
    async with proxy_for(tmp_path, Engines(healthy=(BIG, SMALL))) as proxy:
        assert (await proxy.get("/health")).status_code == 200

    async with proxy_for(tmp_path, Engines(healthy=(BIG,))) as proxy:
        answer = await proxy.get("/health")
    assert answer.status_code == 503
    assert SMALL in answer.json()["error"]


async def test_a_host_with_no_engines_yet_is_not_healthy(tmp_path):
    async with proxy_for(tmp_path, Engines(), mapping={}) as proxy:
        assert (await proxy.get("/health")).status_code == 503


async def test_every_engines_metrics_are_offered_labelled(tmp_path):
    """Concatenated rather than summed: a sum of cache fractions across engines would mean
    nothing."""
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.get("/metrics")

    assert answer.status_code == 200
    assert answer.text.count("vllm:num_requests_running") == 2
    assert BIG in answer.text and SMALL in answer.text


# --- the map is the machine's, read as it changes ---


def test_the_map_is_reread_when_it_changes(tmp_path):
    path = tmp_path / "upstreams.json"
    path.write_text(json.dumps({BIG: "http://127.0.0.1:8001"}))
    upstreams = Upstreams(path)
    assert set(upstreams.by_model()) == {BIG}

    import os
    import time

    path.write_text(json.dumps({BIG: "http://127.0.0.1:8001", SMALL: "http://127.0.0.1:8002"}))
    os.utime(path, (time.time() + 1, time.time() + 1))
    assert set(upstreams.by_model()) == {BIG, SMALL}


def test_a_map_that_cannot_be_read_keeps_the_last_good_one(tmp_path):
    """A start-up that rewrites the file badly must not take the host's serving with it."""
    path = tmp_path / "upstreams.json"
    path.write_text(json.dumps({BIG: "http://127.0.0.1:8001"}))
    upstreams = Upstreams(path)
    assert set(upstreams.by_model()) == {BIG}

    import os
    import time

    path.write_text("not json at all")
    os.utime(path, (time.time() + 1, time.time() + 1))
    assert set(upstreams.by_model()) == {BIG}


def test_a_map_that_is_not_there_yet_is_not_an_error(tmp_path):
    """The start-up that writes it commonly runs after this process starts."""
    assert Upstreams(tmp_path / "missing.json").by_model() == {}


@pytest.mark.parametrize("bad", [
    {BIG: "not-a-url"},
    {BIG: 8001},
    {"": "http://127.0.0.1:8001"},
])
def test_an_entry_that_is_not_a_model_and_a_url_is_ignored(tmp_path, bad):
    path = tmp_path / "upstreams.json"
    path.write_text(json.dumps(bad))
    assert Upstreams(path).by_model() == {}


async def test_a_model_whose_engine_is_still_loading_is_not_listed(tmp_path):
    """The pool calls a host ready when its set appears here. Listing an engine that is still
    loading its weights would have requests routed to it minutes early."""
    async with proxy_for(tmp_path, Engines(healthy=(BIG,))) as proxy:
        answer = await proxy.get("/v1/models")
    assert [entry["id"] for entry in answer.json()["data"]] == [BIG]
