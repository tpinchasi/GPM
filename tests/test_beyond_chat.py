"""Models beyond chat, through the router (S8, D142–D145): speech to text, pictures inside chat,
ranking — each read once, bounded, refused when it could never be served, and sent only to a
build that can do it."""

import base64
import json
import time

import httpx
from fakes.fake_ollama import inference_seen
from fakes.fake_vllm import read_form
from fakes.harness import EngineSpec, pool_harness

WHISPER = "speech"
WHISPER_TAG = "owner/whisper-test"
RERANKER = "ranker"
RERANKER_TAG = "owner/reranker-test"
CHAT = "chatter"

CATALOG = {
    WHISPER: {"variants": [{"tag": WHISPER_TAG, "engine": "vllm", "serves": ["transcribe"]}]},
    RERANKER: {"variants": [{"tag": RERANKER_TAG, "engine": "vllm", "serves": ["rerank"]}]},
    CHAT: {"variants": [{"tag": "chatter:7b"}]},
}

PICTURE = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
AUDIO = bytes(range(256)) * 40 + b"\r\n--not-a-boundary\r\n"


def gpu_pool(**settings):
    # vLLM serves one model per process, so speech and ranking each get a machine.
    return pool_harness(
        [
            EngineSpec(id="gpu", resident={WHISPER_TAG}, engine="vllm"),
            EngineSpec(id="rank", resident={RERANKER_TAG}, engine="vllm"),
            EngineSpec(id="mac", resident={"chatter:7b"}),
        ],
        model_set=[WHISPER, RERANKER, CHAT],
        catalog=CATALOG,
        pool_settings={"models_per_host": "declared", **settings},
        host_overrides={"gpu": {"models": [WHISPER]}, "rank": {"models": [RERANKER]}, "mac": {"models": [CHAT]}},
    )


def last_logged(pool):
    """The newest request-log row: a streamed answer is recorded as its stream closes, which can
    be just after the client has the response."""
    for _ in range(100):
        rows = pool.database.query("SELECT kind, request_bytes, outcome FROM request_log ORDER BY id DESC LIMIT 1")
        if rows:
            return rows[0]
        time.sleep(0.02)
    raise AssertionError("nothing was logged")


def picture_message(url=PICTURE):
    return [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]


# --- speech to text ---

def test_a_recording_is_transcribed_by_the_build_that_serves_it_and_the_engine_reads_what_the_pool_checked():
    with gpu_pool() as pool, pool.client() as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": WHISPER, "language": "he"},
            files={"file": ("note.wav", AUDIO, "audio/wav")},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"text": f"heard {len(AUDIO)} bytes"}
        assert response.headers["X-GPM-Served-Model"] == WHISPER_TAG

        fake = pool.engines["gpu"].fake
        path, fields = fake.received[-1]
        assert path == "/v1/audio/transcriptions"
        # Read by the standard library's parser, not the pool's: the rebuilt form is clean.
        assert fields["model"] == WHISPER_TAG
        assert fields["language"] == "he"
        assert fields["file"] == {"filename": "note.wav", "data": AUDIO}
        sent = fake.headers[-1]
        assert sent["x-gpm-model"] == WHISPER_TAG
        assert sent["content-type"].startswith("multipart/form-data; boundary=gpm-")

        row = last_logged(pool)
        assert row["kind"] == "transcribe" and row["outcome"] == "ok" and row["request_bytes"] > len(AUDIO)


def test_a_transcription_asked_of_a_chat_model_is_refused_at_once():
    with gpu_pool() as pool, pool.client() as client:
        response = client.post("/v1/audio/transcriptions", data={"model": CHAT},
                               files={"file": ("a.wav", b"RIFF", "audio/wav")})
        assert response.status_code == 400
        assert response.json()["reason"] == "kind_not_served"
        assert inference_seen(pool.engines["gpu"].fake) == []


def test_a_pool_whose_engines_cannot_transcribe_says_so_rather_than_no_such_path():
    with pool_harness([EngineSpec(id="mac", resident={"chatter:7b"})], model_set=[CHAT],
                      catalog={CHAT: CATALOG[CHAT]}) as pool, pool.client() as client:
        response = client.post("/v1/audio/transcriptions", data={"model": CHAT},
                               files={"file": ("a.wav", b"RIFF", "audio/wav")})
        assert response.status_code == 400
        assert response.json()["reason"] == "kind_not_served"
        assert "transcribe" in response.json()["detail"]


def test_an_upload_naming_the_model_twice_is_refused_before_any_host_sees_it():
    boundary = "b1"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{WHISPER}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"MODEL\"\r\n\r\nother\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    with gpu_pool() as pool, pool.client() as client:
        response = client.post("/v1/audio/transcriptions", content=body,
                               headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert response.status_code == 400
        assert response.json()["reason"] == "bad_upload"
        assert "other" not in response.text
        assert inference_seen(pool.engines["gpu"].fake) == []


def test_an_image_without_the_audio_extra_answers_as_the_real_one_does():
    with gpu_pool() as pool, pool.client() as client:
        pool.engines["gpu"].fake.audio_extra = False
        response = client.post("/v1/audio/transcriptions", data={"model": WHISPER},
                               files={"file": ("a.wav", b"RIFF", "audio/wav")})
        assert response.status_code == 400
        assert "Invalid or unsupported audio file" in response.text


# --- ranking ---

def test_documents_are_ranked_by_the_reranker():
    with gpu_pool() as pool, pool.client() as client:
        response = client.post("/v1/rerank", json={"model": RERANKER, "query": "q", "documents": ["a", "b", "c"], "top_n": 2})
        assert response.status_code == 200, response.text
        assert [r["index"] for r in response.json()["results"]] == [0, 1]
        assert pool.engines["rank"].fake.received[-1][1]["model"] == RERANKER_TAG


def test_ranking_asked_of_a_speech_model_is_refused():
    with gpu_pool() as pool, pool.client() as client:
        response = client.post("/v1/rerank", json={"model": WHISPER, "query": "q", "documents": ["a"]})
        assert response.status_code == 400
        assert response.json()["reason"] == "kind_not_served"


# --- pictures ---

def two_macs(catalog_variants, **kwargs):
    """Two machines, each holding its own build of one model: the first build needs a capability
    only the first machine has, so each holds the build it would choose."""
    variants = [dict(catalog_variants[0], requires=["fast-text"]), catalog_variants[1]]
    return pool_harness(
        [
            EngineSpec(id="first", resident={"vision:7b-text"}, priority=1, capabilities=["fast-text"]),
            EngineSpec(id="second", resident={"vision:7b"}, priority=2),
        ],
        model_set=["vision"],
        catalog={"vision": {"variants": variants}},
        **kwargs,
    )


def sees(pool, host, tag):
    """That host's Ollama reports its build takes pictures, as a current one does (D143)."""
    pool.engines[host].fake.capabilities[tag] = ["vision"]
    pool.supervisor.hosts[host].observed_for = frozenset()  # as an engine upgraded in place, once its ten minutes pass
    pool.reprobe()


def test_a_picture_goes_to_the_build_that_takes_pictures_even_on_a_lower_tier():
    variants = [{"tag": "vision:7b-text", "inputs": []}, {"tag": "vision:7b", "inputs": ["image"]}]
    with two_macs(variants) as pool, pool.client() as client:
        sees(pool, "second", "vision:7b")
        with_picture = client.post("/v1/chat/completions", json={"model": "vision", "messages": picture_message()})
        text_only = client.post("/v1/chat/completions", json={"model": "vision", "messages": [{"role": "user", "content": "hi"}]})
        assert with_picture.status_code == 200, with_picture.text
        assert with_picture.headers["X-GPM-Host"] == "second"
        assert text_only.headers["X-GPM-Host"] == "first"


def test_a_picture_for_a_model_no_build_of_which_takes_pictures_is_refused():
    with two_macs([{"tag": "vision:7b-text", "inputs": []}, {"tag": "vision:7b", "inputs": []}]) as pool, pool.client() as client:
        response = client.post("/v1/chat/completions", json={"model": "vision", "messages": picture_message()})
        assert response.status_code == 400
        assert response.json()["reason"] == "input_not_accepted"


def test_a_build_nobody_has_described_takes_a_picture_where_its_engine_cannot_say():
    with pool_harness([EngineSpec(id="gpu", resident={"owner/vision"}, engine="vllm")], model_set=["vision"],
                      catalog={"vision": {"variants": [{"tag": "owner/vision", "engine": "vllm"}]}}) as pool, \
            pool.client() as client:
        response = client.post("/v1/chat/completions", json={"model": "vision", "messages": picture_message()})
        assert response.status_code == 200, response.text


def test_a_picture_every_holding_host_reports_it_cannot_take_is_refused_not_queued():
    # Nobody described the builds; both hosts' Ollama report text only. Waiting cannot help.
    with two_macs([{"tag": "vision:7b-text"}, {"tag": "vision:7b"}]) as pool, pool.client() as client:
        response = client.post("/v1/chat/completions", json={"model": "vision", "messages": picture_message()})
        assert response.status_code == 400, response.text
        assert response.json()["reason"] == "input_not_accepted"


def test_what_ollama_reports_a_build_takes_narrows_where_a_picture_goes():
    # Neither build is described in the catalog; the second host's Ollama says its build sees
    # pictures, the first's says nothing of the kind (D143).
    with two_macs([{"tag": "vision:7b-text"}, {"tag": "vision:7b"}]) as pool, pool.client() as client:
        sees(pool, "second", "vision:7b")
        response = client.post("/v1/chat/completions", json={"model": "vision", "messages": picture_message()})
        assert response.status_code == 200, response.text
        assert response.headers["X-GPM-Host"] == "second"


def test_a_picture_given_by_link_is_refused_before_any_host_is_chosen():
    with two_macs([{"tag": "vision:7b-text"}, {"tag": "vision:7b"}]) as pool, pool.client() as client:
        for messages in (
            picture_message("https://10.0.0.1/internal.png"),
            [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "x"}}]}],
            [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"url": "https://x/a.wav"}}]}],
        ):
            response = client.post("/v1/chat/completions", json={"model": "vision", "messages": messages})
            assert response.status_code == 400, messages
            assert response.json()["reason"] == "media_not_inline"
        for host in ("first", "second"):
            assert not [path for path, *_ in pool.engines[host].fake.received if "chat" in path]


def test_a_picture_in_ollamas_own_shape_is_read_and_a_path_in_its_place_refused():
    with two_macs([{"tag": "vision:7b-text", "inputs": []}, {"tag": "vision:7b", "inputs": ["image"]}]) as pool, pool.client() as client:
        sees(pool, "second", "vision:7b")
        ok = client.post("/api/chat", json={"model": "vision", "stream": False,
                                            "messages": [{"role": "user", "content": "x", "images": ["aGVsbG8="]}]})
        assert ok.status_code == 200, ok.text
        assert ok.headers["X-GPM-Host"] == "second"
        refused = client.post("/api/chat", json={"model": "vision", "stream": False,
                                                 "messages": [{"role": "user", "content": "x", "images": ["/etc/passwd:x"]}]})
        assert refused.status_code == 400 and refused.json()["reason"] == "media_not_inline"


# --- bounds ---

def test_a_body_over_the_limit_is_refused_before_it_is_read():
    with gpu_pool(max_request_mb=0.01) as pool, pool.client() as client:
        response = client.post("/v1/audio/transcriptions", data={"model": WHISPER},
                               files={"file": ("big.wav", b"x" * 20_000, "audio/wav")})
        assert response.status_code == 413
        assert response.json()["reason"] == "request_too_large"
        assert inference_seen(pool.engines["gpu"].fake) == []


def test_the_router_holds_no_more_request_bytes_than_it_allows():
    with gpu_pool(max_request_mb=0.01, max_held_request_mb=0.01) as pool, pool.client() as client:
        assert pool.state.take_request_bytes(8_000)  # another request's, still held
        response = client.post("/v1/audio/transcriptions", data={"model": WHISPER},
                               files={"file": ("a.wav", b"x" * 5_000, "audio/wav")})
        assert response.status_code == 503
        assert response.json()["reason"] == "uploads_full"
        assert int(response.headers["Retry-After"]) > 0


def test_a_compressed_request_body_is_refused():
    with gpu_pool() as pool, pool.client() as client:
        response = client.post("/v1/rerank", content=b"\x1f\x8b", headers={"Content-Encoding": "gzip"})
        assert response.status_code == 400
        assert response.json()["reason"] == "content_encoding"


def test_an_unknown_caller_is_refused_before_its_body_is_read():
    with gpu_pool(max_request_mb=0.01) as pool, pool.client(key="wrong") as client:
        response = client.post("/v1/audio/transcriptions", data={"model": WHISPER},
                               files={"file": ("big.wav", b"x" * 20_000, "audio/wav")})
        assert response.status_code == 401


def test_the_standard_library_reads_the_rebuilt_form_as_the_pool_did():
    from gpm_server.engines import multipart

    content_type, body = multipart.rebuild(
        multipart.parse("multipart/form-data; boundary=zz",
                        b"--zz\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n"
                        b"--zz\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f.wav\"\r\n"
                        b"Content-Type: audio/wav\r\n\r\n" + AUDIO + b"\r\n--zz--\r\n"),
        replace={"model": "tag"},
    )
    assert read_form(content_type, body) == {"model": "tag", "file": {"filename": "f.wav", "data": AUDIO}}


# --- whisper.cpp, run by the host's agent (D148) ---

WHISPER_FILE = "owner/whisper-ggml/model-q5_0.bin"


class WhisperAgent:
    """The host's agent as the pool hears it: what it was asked to hold, and where it stands."""

    def __init__(self):
        self.asked: list[dict] = []
        self.models: list[dict] = []
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request):
        if request.url.path.endswith("/facts"):
            return httpx.Response(200, json={
                "agent": {"version": "test", "protocol": "1"}, "os": "Darwin", "arch": "arm64",
                "capabilities": ["apple-silicon"], "engine": {"name": "whisper-cpp", "answers": True},
            })
        self.asked.append(json.loads(request.content))
        return httpx.Response(200, json={"engine_answers": True, "models": self.models, "surplus": []})


def whisper_cpp_pool(monkeypatch):
    monkeypatch.setenv("WHISPER_AGENT_KEY", "test-agent-key")
    return pool_harness(
        [EngineSpec(id="mac-whisper", resident=set(), engine="whisper-cpp"),
         EngineSpec(id="mac", resident={"chatter:7b"})],
        model_set=[WHISPER, CHAT],
        catalog={WHISPER: {"variants": [{"tag": WHISPER_FILE, "engine": "whisper-cpp", "serves": ["transcribe"]}]},
                 CHAT: CATALOG[CHAT]},
        pool_settings={"models_per_host": "declared"},
        host_overrides={
            "mac-whisper": {"models": [WHISPER],
                            "agent": {"url": "http://127.0.0.1:8096", "bearer_env": "WHISPER_AGENT_KEY"}},
            "mac": {"models": [CHAT]},
        },
    )


def test_a_whisper_cpp_host_is_ready_when_its_agent_says_the_server_holds_its_build(monkeypatch):
    with whisper_cpp_pool(monkeypatch) as pool:
        host = pool.supervisor.hosts["mac-whisper"]
        agent = WhisperAgent()
        pool.supervisor._agent_transport = agent.transport

        # Fetching: the server is not up, and that is a host being prepared, not one gone away.
        agent.models = [{"tag": WHISPER_FILE, "on_disk": False, "loaded": False,
                         "pulling": {"tag": WHISPER_FILE, "completed_bytes": 50, "total_bytes": 200}}]
        pool.reprobe()
        assert agent.asked[-1] == {"tags": [WHISPER_FILE], "residency": "pinned"}
        assert host.state.value == "preparing" and "fetching" in host.last_error and "25%" in host.last_error

        # Started with the file: the agent says so, and only then is the host ready.
        pool.engines["mac-whisper"].fake.start({WHISPER_FILE})
        agent.models = [{"tag": WHISPER_FILE, "on_disk": True, "loaded": False}]
        pool.reprobe()
        assert host.state.value == "preparing", "a server answering is not one holding the build"
        agent.models = [{"tag": WHISPER_FILE, "on_disk": True, "loaded": True}]
        pool.reprobe()
        assert host.state.value == "ready", host.last_error

        with pool.client() as client:
            response = client.post("/v1/audio/transcriptions", data={"model": WHISPER},
                                   files={"file": ("note.wav", AUDIO, "audio/wav")})
        assert response.status_code == 200, response.text
        assert response.json() == {"text": f"heard {len(AUDIO)} bytes"}
        assert pool.engines["mac-whisper"].fake.received[-1][1]["model"] == WHISPER_FILE

        # The server stops: its agent says it is starting it again, and nothing is routed there.
        pool.engines["mac-whisper"].fake.running = False
        agent.models = [{"tag": WHISPER_FILE, "on_disk": True, "loaded": False}]
        pool.reprobe()
        assert host.state.value == "preparing" and "starting the server" in host.last_error


def test_a_whisper_cpp_host_whose_agent_is_silent_is_unreachable_like_any_other(monkeypatch):
    with whisper_cpp_pool(monkeypatch) as pool:
        pool.supervisor._agent_transport = httpx.MockTransport(lambda request: httpx.Response(503))
        pool.reprobe()
        host = pool.supervisor.hosts["mac-whisper"]
        assert host.state.value == "unreachable"


def test_an_engine_that_comes_back_is_asked_again_what_its_builds_take():
    """Upgraded and restarted, Ollama may now see pictures: the first probe that finds it again
    asks, rather than serving ten minutes on what the old version said (D143)."""
    with two_macs([{"tag": "vision:7b-text"}, {"tag": "vision:7b"}]) as pool, pool.client() as client:
        refused = client.post("/v1/chat/completions", json={"model": "vision", "messages": picture_message()})
        assert refused.json()["reason"] == "input_not_accepted"

        engine = pool.engines["second"].fake
        engine.healthy = False
        pool.reprobe()
        engine.healthy = True
        engine.capabilities["vision:7b"] = ["vision"]
        pool.reprobe()
        served = client.post("/v1/chat/completions", json={"model": "vision", "messages": picture_message()})
        assert served.status_code == 200, served.text
        assert served.headers["X-GPM-Host"] == "second"
