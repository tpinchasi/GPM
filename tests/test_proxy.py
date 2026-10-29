"""One endpoint in front of several engine processes on one machine (D96).

The first of the two shapes the owner asked for: every model on a host, each in its own engine
process, behind one port — because the pool dials one URL per host. Against fake upstreams;
nothing here needs a GPU or vLLM.
"""

import asyncio
import json
import logging
import secrets

import httpx
import pytest
from gpm_agent.proxy import FORWARDED_PATHS, MAX_BODY_BYTES, Upstreams, create_app

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


async def _once(data: bytes):
    """One body, sent as a stream as a real engine's is — an upload's answer is relayed raw."""
    yield data


class Engines:
    """Two engine processes, each serving one model, each on its own URL."""

    def __init__(self, healthy=(BIG, SMALL)):
        self.healthy = set(healthy)
        self.seen: list[tuple[str, str, bytes]] = []
        self.headers: list[httpx.Headers] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            which = BIG if request.url.port == 8001 else SMALL
            path = request.url.path
            self.seen.append((which, path, request.content))
            self.headers.append(request.headers)
            if path == "/health":
                return httpx.Response(200 if which in self.healthy else 503)
            if path == "/metrics":
                return httpx.Response(200, text="vllm:num_requests_running 3.0\n")
            if request.headers.get("content-type", "").startswith("multipart/form-data"):
                # An upload, answered as a transcription server answers one; never read here.
                answer = json.dumps({"served_by": which, "text": "a transcript"}).encode()
                return httpx.Response(200, content=_once(answer), headers={"content-type": "application/json"})
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


async def test_a_model_this_host_does_not_serve_is_refused(tmp_path, caplog):
    """In fixed words: the name asked for goes to the log, never into the answer (D142)."""
    engines = Engines()
    with caplog.at_level(logging.WARNING, logger="gpm.proxy"):
        async with proxy_for(tmp_path, engines) as proxy:
            answer = await proxy.post("/v1/chat/completions", json={"model": "gemma4:31b"})

    assert answer.status_code == 404
    assert answer.json()["error"]["type"] == "model_not_found"
    assert "gemma4:31b" not in answer.text
    assert "gemma4:31b" in caplog.text and BIG in caplog.text, "the operator still sees both"
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


# --- a copy of a model on each card (D107) ---


class Copies:
    """One model, a copy on each of two ports. A copy can be made to hold its requests open, to
    refuse connections, or to report itself unhealthy."""

    def __init__(self):
        self.seen: list[int] = []
        self.hold = asyncio.Event()
        self.holding: set[int] = set()
        self.refusing: set[int] = set()
        self.unhealthy: set[int] = set()

    def transport(self) -> httpx.MockTransport:
        async def handler(request: httpx.Request) -> httpx.Response:
            port = request.url.port
            if port in self.refusing:
                raise httpx.ConnectError("connection refused", request=request)
            if request.url.path == "/health":
                return httpx.Response(503 if port in self.unhealthy else 200)
            self.seen.append(port)
            if port in self.holding:
                await self.hold.wait()
            body = json.loads(request.content or b"{}")
            if body.get("stream"):
                return httpx.Response(200, content=_frames(), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"port": port})

        return httpx.MockTransport(handler)


def copies_proxy(tmp_path, copies: Copies) -> httpx.AsyncClient:
    mapping = {BIG: ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]}
    app = create_app(
        upstream_file(tmp_path, mapping),
        client=httpx.AsyncClient(transport=copies.transport(), timeout=10),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


async def test_a_request_goes_to_the_copy_with_the_fewest_in_flight(tmp_path):
    """Found live: the second card of a 2x H100 host sat idle. With a copy on each, the next
    request goes where the last one is not still running."""
    copies = Copies()
    copies.holding = {8001}
    async with copies_proxy(tmp_path, copies) as proxy:
        first = asyncio.create_task(proxy.post("/v1/chat/completions", json={"model": BIG}))
        while not copies.seen:
            await asyncio.sleep(0)
        second = await proxy.post("/v1/chat/completions", json={"model": BIG})
        copies.hold.set()
        first = await first

    assert first.json()["port"] == 8001 and second.json()["port"] == 8002


async def test_idle_copies_share_the_work_in_turn_when_requests_finish(tmp_path):
    copies = Copies()
    async with copies_proxy(tmp_path, copies) as proxy:
        for _ in range(3):
            await proxy.post("/v1/chat/completions", json={"model": BIG})
    assert copies.seen == [8001, 8001, 8001], "none in flight, so the first copy each time"


async def test_a_copy_refusing_the_connection_is_passed_over(tmp_path):
    """It never saw the request, so trying the next copy cannot run it twice."""
    copies = Copies()
    copies.refusing = {8001}
    async with copies_proxy(tmp_path, copies) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG})
        async with proxy.stream("POST", "/v1/chat/completions", json={"model": BIG, "stream": True}) as stream:
            frames = [line async for line in stream.aiter_lines() if line.strip()]
    assert answer.status_code == 200 and answer.json()["port"] == 8002
    assert frames[-1] == "data: [DONE]"


async def test_every_copy_refusing_is_a_bad_gateway(tmp_path):
    copies = Copies()
    copies.refusing = {8001, 8002}
    async with copies_proxy(tmp_path, copies) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG})
    assert answer.status_code == 502


async def test_a_model_is_served_only_when_every_copy_answers(tmp_path):
    """The host's workers were set for both cards; on one of them it is not what was bought,
    and the pool must not call it ready."""
    copies = Copies()
    copies.unhealthy = {8002}
    async with copies_proxy(tmp_path, copies) as proxy:
        listed = (await proxy.get("/v1/models")).json()["data"]
        health = await proxy.get("/health")
    assert listed == []
    assert health.status_code == 503 and "8002" in health.json()["error"]


async def test_each_copys_metrics_are_offered(tmp_path):
    engines = Engines()
    mapping = {BIG: ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]}
    async with proxy_for(tmp_path, engines, mapping=mapping) as proxy:
        answer = await proxy.get("/metrics")
    assert answer.text.count("vllm:num_requests_running") == 2


def test_a_map_of_single_urls_is_still_read(tmp_path):
    """The launcher's older shape, one URL per model."""
    path = tmp_path / "upstreams.json"
    path.write_text(json.dumps({BIG: "http://127.0.0.1:8001/"}))
    assert Upstreams(path).by_model() == {BIG: ["http://127.0.0.1:8001"]}


@pytest.mark.parametrize("bad", [{BIG: []}, {BIG: ["http://127.0.0.1:8001", "nope"]}])
def test_a_list_with_anything_but_urls_is_ignored(tmp_path, bad):
    path = tmp_path / "upstreams.json"
    path.write_text(json.dumps(bad))
    assert Upstreams(path).by_model() == {}


# --- a streamed answer says what the engine said (found in review) ---


def one_copy_proxy(tmp_path, handler) -> httpx.AsyncClient:
    app = create_app(
        upstream_file(tmp_path, {BIG: "http://127.0.0.1:8001"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


async def test_an_engines_refusal_of_a_stream_keeps_its_status(tmp_path):
    """A 400 from the engine is the app's to read as a 400; a 200 carrying an error body is
    recorded as a success by the pool."""
    refused = {"error": {"message": "max_tokens is too large", "type": "BadRequestError"}}

    async def body():
        yield json.dumps(refused).encode()

    def handler(request):
        return httpx.Response(400, content=body(), headers={"content-type": "application/json"})

    async with one_copy_proxy(tmp_path, handler) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG, "stream": True})
    assert answer.status_code == 400 and answer.json() == refused
    assert answer.headers["content-type"].startswith("application/json")


async def test_a_stream_no_copy_accepts_is_a_bad_gateway_not_an_empty_success(tmp_path):
    copies = Copies()
    copies.refusing = {8001, 8002}
    async with copies_proxy(tmp_path, copies) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG, "stream": True})
    assert answer.status_code == 502


async def test_an_engine_dying_mid_stream_breaks_the_stream_rather_than_ending_it(tmp_path):
    """A clean end would pass a truncated answer as whole — under buffered delivery (D62) the
    pool would deliver it instead of running it again."""

    async def dies():
        yield b'data: {"choices":[{"delta":{"content":"hal"}}]}\n\n'
        raise httpx.ReadError("the engine went away")

    handler = lambda r: httpx.Response(200, content=dies(), headers={"content-type": "text/event-stream"})  # noqa: E731
    async with one_copy_proxy(tmp_path, handler) as proxy:
        with pytest.raises(httpx.HTTPError):
            async with proxy.stream("POST", "/v1/chat/completions", json={"model": BIG, "stream": True}) as stream:
                async for _ in stream.aiter_raw():
                    pass


# --- the pool names the model; an upload is routed without being read (D144) ---

#: A transcription upload as the pool's router sends one: rebuilt with the router's boundary, the
#: file's bytes untouched (not valid UTF-8, as audio is not), its `model` part the header's tag.
UPLOAD_TYPE = "multipart/form-data; boundary=gpm-test-boundary"
UPLOAD = (
    b"--gpm-test-boundary\r\n"
    b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
    b"Content-Type: audio/wav\r\n\r\n"
    b"RIFF\x24\x00\x00\x00WAVEfmt \xff\xfe\x00\x80"
    b"\r\n--gpm-test-boundary\r\n"
    b'Content-Disposition: form-data; name="model"\r\n\r\n'
    + SMALL.encode()
    + b"\r\n--gpm-test-boundary\r\n"
    b'Content-Disposition: form-data; name="stream"\r\n\r\n'
    b"true"
    b"\r\n--gpm-test-boundary--\r\n"
)


async def test_an_upload_is_routed_by_the_pools_header_without_being_read(tmp_path):
    """A form names its model in a field, and reading it would mean parsing a whole recording.
    The pool's router has already chosen, and says so in a header."""
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/audio/transcriptions", content=UPLOAD,
                                  headers={"content-type": UPLOAD_TYPE, "x-gpm-model": SMALL})

    assert answer.status_code == 200 and answer.json()["served_by"] == SMALL
    (which, path, seen) = engines.seen[-1]
    assert (which, path) == (SMALL, "/v1/audio/transcriptions")
    assert seen == UPLOAD, "the form leaves byte for byte"
    assert engines.headers[-1]["content-type"] == UPLOAD_TYPE, "with its boundary"


async def test_an_upload_without_the_pools_header_is_refused_rather_than_read(tmp_path):
    """Only an older router sends no header, and an older router never sent an upload. Its
    `model` field names a model served here, and is still not looked for."""
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/audio/transcriptions", content=UPLOAD, headers={"content-type": UPLOAD_TYPE})
    assert answer.status_code == 404 and answer.json()["error"]["type"] == "model_not_found"
    assert not engines.seen


async def test_the_pools_header_wins_over_the_body(tmp_path):
    """The router writes the same tag into both; where they differ, the header is the pool's
    choice and the body only what came with it."""
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG, "messages": []},
                                  headers={"x-gpm-model": SMALL})
    assert answer.json()["served_by"] == SMALL


async def test_an_empty_header_names_no_model_rather_than_falling_back_to_the_body(tmp_path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG}, headers={"x-gpm-model": ""})
    assert answer.status_code == 404
    assert not engines.seen


async def test_the_pools_headers_never_reach_an_engine(tmp_path):
    engines = Engines()
    body = b'{"model":"gemma4:26b","messages":[]}'
    async with proxy_for(tmp_path, engines) as proxy:
        await proxy.post("/v1/chat/completions", content=body, headers={
            "content-type": "application/json", "x-gpm-model": BIG, "x-request-id": "r-1"})

    sent = engines.headers[-1]
    assert "x-gpm-model" not in sent
    assert sent["x-request-id"] == "r-1", "every other header goes on as it came"
    assert engines.seen[-1][2] == body


async def test_a_large_json_body_is_still_routed_by_its_model(tmp_path):
    """Read on a worker thread above a megabyte — a picture inline — and routed the same."""
    engines = Engines()
    picture = "x" * 1_500_000
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post("/v1/chat/completions", json={"model": BIG, "messages": [picture]})
    assert answer.json()["served_by"] == BIG


# --- only inference reaches an engine (D142) ---


def test_the_forwarded_paths_are_a_closed_list():
    """Adding one is a decision: beside these, engines serve routes that load and unload
    models."""
    assert FORWARDED_PATHS == {
        "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings",
        "/v1/audio/transcriptions", "/v1/audio/translations", "/v1/rerank",
    }


@pytest.mark.parametrize("path", sorted(FORWARDED_PATHS))
async def test_each_inference_path_is_forwarded_as_named(tmp_path, path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post(path, json={"model": BIG}, headers={"x-gpm-model": BIG})
    assert answer.status_code == 200
    assert engines.seen[-1][:2] == (BIG, path)


@pytest.mark.parametrize("path", [
    "/v1/load_lora_adapter",     # vLLM: loads an adapter it is given the path of
    "/v1/unload_lora_adapter",
    "/load",                     # whisper.cpp: loads the model file it is given the path of
    "/v1/audio/speech",          # a kind not served (D142)
    "/v1/chat/completions/x",    # by name, never by prefix
    "/v1/chat/completions/",
    "/V1/chat/completions",
    "/v1/chat/completions%3Fx",  # decoded, it only starts like an allowed path
    "/v1/models",                # this process's own, asked with GET
    "/api/pull",
])
async def test_any_other_path_is_refused_and_never_forwarded(tmp_path, path):
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post(path, json={"model": BIG}, headers={"x-gpm-model": BIG})
    assert answer.status_code == 404
    assert answer.json()["error"]["type"] == "path_not_served"
    assert not engines.seen, "no engine heard of it"


# --- an error answer says nothing the caller sent (D142) ---


def hostile() -> str:
    """Markup, and a string long and random enough that no fixed message holds it by chance."""
    return f"<script>alert(1)</script>{secrets.token_hex(64)}"


async def test_an_unknown_model_is_refused_in_the_same_words_whatever_is_asked(tmp_path, caplog):
    first, second = hostile(), hostile()
    engines = Engines()
    with caplog.at_level(logging.WARNING, logger="gpm.proxy"):
        async with proxy_for(tmp_path, engines) as proxy:
            by_header = await proxy.post("/v1/chat/completions", json={"model": first},
                                         headers={"x-gpm-model": first})
            by_body = await proxy.post("/v1/chat/completions", json={"model": second})
            by_form = await proxy.post("/v1/audio/transcriptions", content=UPLOAD,
                                       headers={"content-type": UPLOAD_TYPE, "x-gpm-model": second})

    for answer in (by_header, by_body, by_form):
        assert answer.status_code == 404
        assert first not in answer.text and second not in answer.text and "<script>" not in answer.text
    assert by_header.json() == by_body.json() == by_form.json()
    assert first in caplog.text, "the operator still sees what was asked"
    assert not engines.seen


async def test_a_refused_path_is_not_echoed(tmp_path):
    word = hostile()
    engines = Engines()
    async with proxy_for(tmp_path, engines) as proxy:
        answer = await proxy.post(f"/v1/{word}", json={"model": BIG}, headers={"x-gpm-model": BIG})
    assert answer.status_code == 404
    assert word[-128:] not in answer.text and "script" not in answer.text


async def test_an_engine_that_did_not_answer_is_reported_without_its_details(tmp_path, caplog):
    """Neither the model, the copies' addresses nor the connection's error reach the caller."""
    copies = Copies()
    copies.refusing = {8001, 8002}
    with caplog.at_level(logging.WARNING, logger="gpm.proxy"):
        async with copies_proxy(tmp_path, copies) as proxy:
            whole = await proxy.post("/v1/chat/completions", json={"model": BIG}, headers={"x-gpm-model": BIG})
            streamed = await proxy.post("/v1/chat/completions", json={"model": BIG, "stream": True},
                                        headers={"x-gpm-model": BIG})

    for answer in (whole, streamed):
        assert answer.status_code == 502 and answer.json()["error"]["type"] == "upstream_unavailable"
        assert BIG not in answer.text and "8001" not in answer.text and "refused" not in answer.text
    assert "connection refused" in caplog.text


# --- an upload's answer is relayed as it comes ---


async def _transcript_events():
    yield b'data: {"type":"transcript.text.delta","delta":"a trans"}\n\n'
    yield b'data: {"type":"transcript.text.done","text":"a transcript"}\n\n'


@pytest.mark.parametrize("status, content_type, chunks", [
    (200, "text/event-stream", _transcript_events),
    (200, "text/plain; charset=utf-8", lambda: _once(b"a transcript")),
    (400, "application/json", lambda: _once(b'{"error":{"message":"not audio","type":"invalid_request_error"}}')),
])
async def test_an_uploads_answer_is_relayed_as_it_comes_with_the_engines_status_and_type(
        tmp_path, status, content_type, chunks):
    """Whether a transcription streams is a field of its form, which is not read here — so an
    upload's answer is always relayed as the engine sends it, and in the engine's own words."""
    sent = []

    def handler(request):
        sent.append(request.content)
        return httpx.Response(status, content=chunks(), headers={"content-type": content_type})

    async with one_copy_proxy(tmp_path, handler) as proxy:
        answer = await proxy.post("/v1/audio/transcriptions", content=UPLOAD,
                                  headers={"content-type": UPLOAD_TYPE, "x-gpm-model": BIG})

    assert answer.status_code == status
    assert answer.headers["content-type"].startswith(content_type)
    assert answer.content == b"".join([chunk async for chunk in chunks()])
    assert "content-length" not in answer.headers, "relayed as it came, not held until whole"
    assert sent == [UPLOAD]


# --- a bound on what is held ---


def test_the_bound_is_above_the_pools_own_request_limit():
    """It must not refuse what the pool let through (D144)."""
    from gpm_server.config import PoolSettings

    assert MAX_BODY_BYTES > PoolSettings.model_fields["max_request_mb"].default * 1_000_000


def test_the_bound_is_above_the_most_the_pools_limit_may_be_with_room_for_a_rebuilt_form():
    """At the largest `pool.max_request_mb` allowed, the pool accepts a body of exactly that size
    and then rebuilds an upload's form to name the build — a boundary and the model's tag longer.
    Equal bounds would refuse that body here, after the pool let it through."""
    from annotated_types import Le
    from gpm_server.config import PoolSettings

    [most] = [rule.le for rule in PoolSettings.model_fields["max_request_mb"].metadata if isinstance(rule, Le)]
    assert MAX_BODY_BYTES - most * 1_000_000 >= 64 * 1024


def bounded_proxy(tmp_path, engines: Engines, limit: int) -> httpx.AsyncClient:
    app = create_app(
        upstream_file(tmp_path, {BIG: "http://127.0.0.1:8001"}),
        client=httpx.AsyncClient(transport=engines.transport(), timeout=10),
        max_body_bytes=limit,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


async def test_a_body_over_the_bound_is_refused_before_an_engine_sees_it(tmp_path):
    engines = Engines()
    upload = {"content-type": UPLOAD_TYPE, "x-gpm-model": BIG}

    async def undeclared():
        # Sent with no length, so it can only be counted as it arrives.
        for _ in range(4):
            yield b"x" * 300

    async with bounded_proxy(tmp_path, engines, limit=1000) as proxy:
        # The length it declares is enough: refused before a byte of it is read.
        declared = await proxy.post("/v1/audio/transcriptions", content=b"x" * 10,
                                    headers={**upload, "content-length": "1001"})
        counted = await proxy.post("/v1/audio/transcriptions", content=undeclared(), headers=upload)
        at_the_bound = await proxy.post("/v1/audio/transcriptions", content=b"x" * 1000, headers=upload)

    for answer in (declared, counted):
        assert answer.status_code == 413
        assert answer.json()["error"]["type"] == "request_too_large"
    assert at_the_bound.status_code == 200
    assert len(engines.seen) == 1, "only the body within the bound was forwarded"
