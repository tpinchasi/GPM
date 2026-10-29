"""Beyond chat from the SDK: pictures inline, speech to text and ranking (S8 §6).

The real wait-and-retry transport over a scripted one, so what is asserted is what the pool
would receive: no pool, no GPU, no network. Where it matters, the pool's own upload parser and
configuration loader read what the SDK sends and what the examples say.
"""

import ast
import base64
import importlib.util
import io
import json
import re
from pathlib import Path

import gpm_client
import httpx
import pytest
import yaml
from gpm_client import (
    AsyncPoolClient,
    AsyncPoolTransport,
    PoolClient,
    PoolRequestError,
    PoolTransport,
    Ranked,
    Ranking,
    RetryPolicy,
    image_part,
)
from gpm_server.config import PoolConfig
from gpm_server.engines.multipart import parse

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
APP_KEY = "gpma_test_key"

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 24
GIF = b"GIF89a" + b"\x00" * 24
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 24
# Every byte value, carriage returns and line feeds among them: an upload must arrive as sent.
AUDIO = bytes(range(256)) * 8

FACTS = {
    "X-GPM-Served-Model": "speech-model-ct2",
    "X-GPM-Host": "gpu-1",
    "X-GPM-Runtime-Class": "cuda-ct2",
    "X-GPM-Wait-S": "0.25",
}


class Recorder:
    """Answers each request with the next scripted response — or raises it, where it is an
    exception — and keeps what it was sent."""

    def __init__(self, *responses: httpx.Response | Exception):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def policy(**kwargs) -> RetryPolicy:
    return RetryPolicy(max_wait_s=5, backoff_initial_s=0.001, backoff_max_s=0.002, **kwargs)


def pool(recorder: Recorder, api: str = "openai", **kwargs) -> PoolClient:
    transport = PoolTransport(policy=policy(**kwargs), transport=httpx.MockTransport(recorder))
    return PoolClient(base_url="http://pool", api_key=APP_KEY, transport=transport, api=api)


def async_pool(recorder: Recorder, api: str = "openai", **kwargs) -> AsyncPoolClient:
    transport = AsyncPoolTransport(policy=policy(**kwargs), transport=httpx.MockTransport(recorder))
    return AsyncPoolClient(base_url="http://pool", api_key=APP_KEY, transport=transport, api=api)


def form_of(request: httpx.Request) -> dict[str, tuple]:
    """The multipart form a request carried: each part's file name (None for a plain field)
    and its bytes."""
    assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    boundary = request.headers["Content-Type"].split("boundary=")[1].encode()
    fields = {}
    for part in request.content.split(b"--" + boundary)[1:-1]:
        head, _, value = part.removeprefix(b"\r\n").removesuffix(b"\r\n").partition(b"\r\n\r\n")
        disposition = head.split(b"\r\n")[0].decode()
        name = re.search(r'; name="([^"]*)"', disposition).group(1)
        filename = re.search(r'; filename="([^"]*)"', disposition)
        assert name not in fields, f"{name} sent twice"
        fields[name] = (filename.group(1) if filename else None, value)
    return fields


def transcribed(body=None, text=None) -> httpx.Response:
    if text is not None:
        return httpx.Response(200, text=text, headers=FACTS)
    return httpx.Response(200, json=body or {"text": "shalom"}, headers=FACTS)


def refusal(status: int, reason: str, detail: str = "d", **headers: str) -> httpx.Response:
    body = {"error": reason, "reason": reason, "detail": detail}
    return httpx.Response(status, json=body, headers=headers)


def ranked(*rows: dict) -> httpx.Response:
    return httpx.Response(200, json={"results": list(rows)}, headers=FACTS)


def data_url(mime: str, data: bytes) -> dict:
    encoded = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


# --- image_part --------------------------------------------------------------------------------


def test_a_picture_from_a_path_travels_inline_with_its_type(tmp_path):
    picture = tmp_path / "receipt.png"
    picture.write_bytes(PNG)
    assert image_part(picture) == data_url("image/png", PNG)
    assert image_part(str(picture)) == data_url("image/png", PNG)


@pytest.mark.parametrize(
    "data, mime",
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp")],
)
def test_bytes_with_no_name_are_typed_by_their_first_bytes(data, mime):
    assert image_part(data) == data_url(mime, data)


def test_a_binary_file_object_is_read_and_typed_by_its_name(tmp_path):
    picture = tmp_path / "photo.jpg"
    picture.write_bytes(JPEG)
    with open(picture, "rb") as opened:
        assert image_part(opened) == data_url("image/jpeg", JPEG)
    assert image_part(io.BytesIO(PNG)) == data_url("image/png", PNG)


def test_the_name_is_trusted_before_the_bytes_and_a_given_type_before_both(tmp_path):
    scan = tmp_path / "scan.jpg"
    scan.write_bytes(b"bytes no signature matches")
    assert image_part(scan) == data_url("image/jpeg", b"bytes no signature matches")
    assert image_part(PNG, mime="image/heic") == data_url("image/heic", PNG)


def test_a_picture_of_unknown_type_is_refused_before_anything_is_sent(tmp_path):
    with pytest.raises(ValueError, match="mime="):
        image_part(b"not a picture")
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"not a picture")
    with pytest.raises(ValueError, match="notes.txt"):
        image_part(notes)  # a name that guesses a type other than a picture's tells nothing


def test_a_text_mode_file_is_refused():
    with pytest.raises(TypeError, match="binary"):
        image_part(io.StringIO("text"))


def test_a_picture_part_goes_inside_an_ordinary_chat_message():
    recorder = Recorder(httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]}))
    with pool(recorder) as client:
        reply = client.chat("vision-model", [{"role": "user", "content": [
            {"type": "text", "text": "What is the total?"}, image_part(JPEG),
        ]}])
    assert reply.content == "42"
    sent = json.loads(recorder.requests[0].content)
    assert sent["messages"][0]["content"][1] == data_url("image/jpeg", JPEG)


# --- transcribe --------------------------------------------------------------------------------


def test_transcribe_uploads_a_form_and_reads_the_facts_of_the_call(tmp_path):
    recording = tmp_path / "voice-note.m4a"
    recording.write_bytes(AUDIO)
    recorder = Recorder(transcribed({"text": "shalom"}))
    with pool(recorder) as client:
        transcript = client.transcribe(
            "speech-model", recording, language="he", prompt="GPM", temperature=0.2,
            session_id="s-1",
        )

    request = recorder.requests[0]
    assert request.method == "POST" and request.url.path == "/v1/audio/transcriptions"
    assert request.headers["Authorization"] == f"Bearer {APP_KEY}"
    assert request.headers["X-GPM-Session"] == "s-1"
    assert form_of(request) == {
        "model": (None, b"speech-model"),
        "language": (None, b"he"),
        "prompt": (None, b"GPM"),
        "temperature": (None, b"0.2"),
        "file": ("voice-note.m4a", AUDIO),
    }
    assert transcript.text == "shalom"
    assert transcript.segments is None and transcript.language is None
    assert transcript.served_model == "speech-model-ct2"
    assert transcript.host == "gpu-1"
    assert transcript.runtime_class == "cuda-ct2"
    assert transcript.wait_s == 0.25
    assert transcript.raw == {"text": "shalom"}


def test_an_upload_without_a_file_name_is_called_audio():
    recorder = Recorder(transcribed(), transcribed())
    with pool(recorder) as client:
        client.transcribe("speech-model", AUDIO)
        client.transcribe("speech-model", io.BytesIO(AUDIO))
    assert [form_of(request)["file"] for request in recorder.requests] == [("audio", AUDIO)] * 2


def test_an_open_file_is_sent_under_its_own_name(tmp_path):
    recording = tmp_path / "meeting.mp3"
    recording.write_bytes(AUDIO)
    recorder = Recorder(transcribed())
    with pool(recorder) as client, open(recording, "rb") as opened:
        client.transcribe("speech-model", opened)
    assert form_of(recorder.requests[0])["file"] == ("meeting.mp3", AUDIO)


def test_verbose_json_brings_segments_and_the_language():
    body = {
        "text": " hello there",
        "language": "english",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": " hello"},
            {"start": 1.5, "end": 2.0, "text": " there"},
        ],
    }
    recorder = Recorder(transcribed(body))
    with pool(recorder) as client:
        transcript = client.transcribe("speech-model", AUDIO, response_format="verbose_json")
    assert form_of(recorder.requests[0])["response_format"] == (None, b"verbose_json")
    assert transcript.text == " hello there"
    assert transcript.language == "english"
    assert [s["end"] for s in transcript.segments] == [1.5, 2.0]
    assert transcript.raw == body


@pytest.mark.parametrize("response_format", ["text", "srt", "vtt"])
def test_a_text_format_answers_in_text(response_format):
    body = "1\n00:00:00,000 --> 00:00:01,500\nhello\n"
    recorder = Recorder(transcribed(text=body))
    with pool(recorder) as client:
        transcript = client.transcribe("speech-model", AUDIO, response_format=response_format)
    assert transcript.text == body and transcript.raw == body
    assert transcript.segments is None
    assert transcript.served_model == "speech-model-ct2" and transcript.wait_s == 0.25


def test_a_refusal_is_raised_at_once_with_its_reason():
    refused = refusal(400, "kind_not_served", "no build of chat-model transcribes")
    recorder = Recorder(refused, transcribed())
    with pool(recorder) as client:
        with pytest.raises(PoolRequestError) as raised:
            client.transcribe("chat-model", AUDIO)
    assert raised.value.status_code == 400
    assert raised.value.reason == "kind_not_served"
    assert "transcribes" in raised.value.detail
    assert len(recorder.requests) == 1


def test_a_generation_timeout_is_not_retried():
    recorder = Recorder(refusal(504, "generation_timeout"), transcribed())
    with pool(recorder) as client:
        with pytest.raises(PoolRequestError) as raised:
            client.transcribe("speech-model", AUDIO)
    assert raised.value.reason == "generation_timeout" and len(recorder.requests) == 1


def test_a_read_timeout_after_the_upload_was_sent_is_raised_not_sent_again():
    """The pool has the recording and may still be working on it (D144): re-uploading it until
    `max_wait` ran out ran it again and again."""
    recorder = Recorder(httpx.ReadTimeout("timed out"), transcribed())
    with pool(recorder) as client:
        with pytest.raises(PoolRequestError) as raised:
            client.transcribe("speech-model", AUDIO)
    assert len(recorder.requests) == 1
    assert raised.value.status_code == 504 and raised.value.reason == "read_timeout"
    assert "within 900 s" in raised.value.detail and "not sent again" in raised.value.detail
    assert isinstance(raised.value.__cause__, httpx.ReadTimeout)


def test_a_connection_refused_is_still_waited_out(monkeypatch):
    monkeypatch.setattr("gpm_client.transport.time.sleep", lambda _: None)
    recorder = Recorder(httpx.ConnectError("refused"), transcribed())
    with pool(recorder) as client:
        assert client.transcribe("speech-model", AUDIO).text == "shalom"
    assert len(recorder.requests) == 2


def read_timeout_of(request: httpx.Request) -> float | None:
    return request.extensions["timeout"]["read"]


def test_transcribe_reads_well_past_the_pools_read_time_and_can_be_told_otherwise():
    """The pool gives the engine 300 s and then answers 504; the SDK's clock also covers the
    queue, so at 300 s it always ran out first and never heard the pool's answer."""
    chat_answer = httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    recorder = Recorder(transcribed(), transcribed(), chat_answer)
    with pool(recorder) as client:
        client.transcribe("speech-model", AUDIO)
        client.transcribe("speech-model", AUDIO, read_timeout_s=60)
        client.chat("chat-model", [{"role": "user", "content": "hi"}])
    assert [read_timeout_of(r) for r in recorder.requests] == [900.0, 60.0, 300.0]
    assert recorder.requests[0].extensions["timeout"]["connect"] == 5.0


def test_transcribe_never_shortens_a_client_read_timeout_that_is_longer():
    recorder = Recorder(transcribed(), transcribed())
    transport = PoolTransport(policy=policy(), transport=httpx.MockTransport(recorder))
    for read in (1800.0, None):
        with PoolClient(base_url="http://pool", api_key=APP_KEY, transport=transport,
                        timeout=httpx.Timeout(5.0, read=read)) as client:
            client.transcribe("speech-model", AUDIO)
    assert [read_timeout_of(r) for r in recorder.requests] == [1800.0, None]


def test_a_file_name_is_sent_in_a_form_the_pools_parser_reads():
    """httpx writes a backslash as two, which the pool refuses (`400 bad_upload`): such a
    recording could never be uploaded. The name is only a label; the bytes go as they are."""
    opened = io.BytesIO(AUDIO)
    opened.name = 'C:\\calls\\"q3"\x1b\x7f\x85 הקלטה.m4a'
    recorder = Recorder(transcribed())
    with pool(recorder) as client:
        client.transcribe("speech-model", opened)

    request = recorder.requests[0]
    safe = "C:_calls__q3____ הקלטה.m4a"
    assert form_of(request)["file"] == (safe, AUDIO)
    (file,) = [part for part in parse(request.headers["Content-Type"], request.content) if part.name == "file"]
    assert file.filename == safe and file.data == AUDIO


def test_a_busy_pool_is_waited_out_and_the_upload_resent_intact(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("gpm_client.transport.time.sleep", slept.append)
    busy = refusal(503, "uploads_full", **{"Retry-After": "3"})
    recording = tmp_path / "voice-note.m4a"
    recording.write_bytes(AUDIO)
    recorder = Recorder(busy, transcribed())
    # An open file: read once, so a resend that went back to the file would find it at its end.
    with pool(recorder) as client, open(recording, "rb") as opened:
        transcript = client.transcribe("speech-model", opened, language="he")

    assert transcript.text == "shalom"
    assert slept == [3.0]
    first, second = recorder.requests
    assert first.headers["Content-Type"] == second.headers["Content-Type"]
    assert first.content == second.content
    assert form_of(second)["file"] == ("voice-note.m4a", AUDIO)


# --- rerank ------------------------------------------------------------------------------------


def test_rerank_sends_the_query_and_documents_and_returns_the_best_first():
    documents = ["reset it from the login page", "our office hours", "passwords expire yearly"]
    recorder = Recorder(ranked(
        {"index": 1, "relevance_score": 0.02, "document": {"text": documents[1]}},
        {"index": 0, "relevance_score": 0.91, "document": {"text": documents[0]}},
        {"index": 2, "relevance_score": 0.40},
    ))
    with pool(recorder) as client:
        results = client.rerank(
            "ranker", "How do I reset my password?", documents, top_n=3, session_id="s-2"
        )

    request = recorder.requests[0]
    assert request.url.path == "/v1/rerank"
    assert request.headers["X-GPM-Session"] == "s-2"
    assert json.loads(request.content) == {
        "model": "ranker",
        "query": "How do I reset my password?",
        "documents": documents,
        "top_n": 3,
    }
    assert results == [
        Ranked(index=0, score=0.91, document=documents[0]),
        Ranked(index=2, score=0.40, document=None),
        Ranked(index=1, score=0.02, document=documents[1]),
    ]


def test_rerank_without_top_n_sends_none():
    recorder = Recorder(ranked({"index": 0, "relevance_score": 1}))
    with pool(recorder) as client:
        assert client.rerank("ranker", "q", ["d"]) == [Ranked(index=0, score=1.0)]
    assert "top_n" not in json.loads(recorder.requests[0].content)


def test_a_ranking_carries_the_facts_of_the_call_and_is_still_a_list():
    body = {"model": "ranker", "results": [{"index": 0, "relevance_score": 0.5}]}
    recorder = Recorder(httpx.Response(200, json=body, headers=FACTS))
    with pool(recorder) as client:
        results = client.rerank("ranker", "q", ["d"])

    assert isinstance(results, Ranking) and isinstance(results, list)
    assert results == [Ranked(index=0, score=0.5)] and len(results) == 1
    assert results.served_model == "speech-model-ct2"
    assert results.host == "gpu-1" and results.runtime_class == "cuda-ct2"
    assert results.wait_s == 0.25
    assert results.raw == body


def test_a_ranking_names_the_model_asked_for_when_the_pool_does_not_say():
    body = {"model": "ranker", "results": []}
    recorder = Recorder(httpx.Response(200, json=body))
    with pool(recorder) as client:
        results = client.rerank("ranker", "q", [])
    assert results == [] and results.served_model == "ranker" and results.host is None


def test_transcription_and_ranking_use_the_v1_paths_whatever_the_client_shape():
    recorder = Recorder(transcribed(), ranked({"index": 0, "relevance_score": 1}))
    with pool(recorder, api="ollama") as client:
        client.transcribe("speech-model", AUDIO)
        client.rerank("ranker", "q", ["d"])
    assert [r.url.path for r in recorder.requests] == ["/v1/audio/transcriptions", "/v1/rerank"]


# --- the same, awaited -------------------------------------------------------------------------


async def test_async_transcribe_and_rerank():
    recorder = Recorder(
        transcribed({"text": "shalom", "language": "hebrew", "segments": []}),
        ranked({"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.7}),
    )
    async with async_pool(recorder, api="ollama") as client:
        transcript = await client.transcribe(
            "speech-model", AUDIO, response_format="verbose_json", session_id="s-3"
        )
        results = await client.rerank("ranker", "q", ["a", "b"], top_n=2)

    upload, ranking = recorder.requests
    assert upload.url.path == "/v1/audio/transcriptions"
    assert upload.headers["X-GPM-Session"] == "s-3"
    assert form_of(upload)["file"] == ("audio", AUDIO)
    assert transcript.text == "shalom" and transcript.language == "hebrew"
    assert transcript.segments == []
    assert transcript.host == "gpu-1" and transcript.wait_s == 0.25
    assert ranking.url.path == "/v1/rerank"
    assert [r.index for r in results] == [1, 0]
    assert results.host == "gpu-1" and results.wait_s == 0.25
    assert read_timeout_of(upload) == 900.0 and read_timeout_of(ranking) == 300.0


async def test_async_a_busy_pool_is_waited_out_and_the_upload_resent_intact():
    waits: list[tuple] = []
    busy = refusal(503, "queue_timeout", **{"Retry-After": "0"})
    recorder = Recorder(busy, transcribed(text="shalom"))
    client = async_pool(recorder, on_wait=lambda *args: waits.append(args))
    async with client:
        transcript = await client.transcribe(
            "speech-model", io.BytesIO(AUDIO), response_format="text"
        )

    assert transcript.text == "shalom"
    assert [(reason, next_try) for reason, _, next_try in waits] == [("queue_timeout", 0.0)]
    first, second = recorder.requests
    assert first.content == second.content
    assert form_of(second)["file"] == ("audio", AUDIO)


async def test_async_a_read_timeout_is_raised_not_sent_again():
    recorder = Recorder(httpx.ReadTimeout("timed out"), transcribed())
    async with async_pool(recorder) as client:
        with pytest.raises(PoolRequestError) as raised:
            await client.transcribe("speech-model", AUDIO, read_timeout_s=1200)
    assert len(recorder.requests) == 1
    assert raised.value.reason == "read_timeout" and "within 1200 s" in raised.value.detail


async def test_async_a_refusal_is_raised_with_its_reason():
    async with async_pool(Recorder(refusal(400, "kind_not_served"))) as client:
        with pytest.raises(PoolRequestError) as raised:
            await client.rerank("chat-model", "q", ["d"])
    assert raised.value.reason == "kind_not_served"


# --- the examples ------------------------------------------------------------------------------


def test_the_examples_import_only_what_the_package_exports():
    for example in sorted(EXAMPLES.glob("*.py")):
        tree = ast.parse(example.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "gpm_client":
                for alias in node.names:
                    assert alias.name in gpm_client.__all__, f"{example.name}: {alias.name}"


def run_example(name: str, argv: list[str], monkeypatch) -> list[httpx.Request]:
    """Runs an example's `main()` as its command line would, against a scripted pool."""
    received: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        received.append(request)
        if request.url.path == "/v1/audio/transcriptions":
            segments = [{"start": 0.0, "end": 1.25, "text": " shalom"}]
            return transcribed({"text": " shalom", "segments": segments})
        if request.url.path == "/v1/rerank":
            return ranked(
                {"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}
            )
        chat_answer = {"choices": [{"message": {"content": "42"}}]}
        return httpx.Response(200, json=chat_answer, headers=FACTS)

    monkeypatch.setenv("GPM_URL", "http://pool")
    monkeypatch.setenv("GPM_API_KEY", APP_KEY)
    scripted = httpx.MockTransport(answer)
    monkeypatch.setattr(
        "gpm_client.client.pool_transport",
        lambda retry_policy=None: PoolTransport(retry_policy, transport=scripted),
    )
    monkeypatch.setattr("sys.argv", [name, *argv])
    spec = importlib.util.spec_from_file_location(f"example_{Path(name).stem}", EXAMPLES / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    return received


def test_the_transcribe_example_runs(tmp_path, monkeypatch, capsys):
    recording = tmp_path / "voice-note.m4a"
    recording.write_bytes(AUDIO)
    argv = [str(recording), "--language", "he", "--timestamps"]
    (request,) = run_example("transcribe.py", argv, monkeypatch)
    form = form_of(request)
    assert form["language"] == (None, b"he") and form["response_format"] == (None, b"verbose_json")
    out = capsys.readouterr().out
    assert "1.25" in out and "shalom" in out and "speech-model-ct2 on gpu-1" in out


def test_the_rerank_example_runs(tmp_path, monkeypatch, capsys):
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for file in files:
        file.write_text(file.stem)
    run_example("rerank.py", ["reset my password", *map(str, files), "--top", "2"], monkeypatch)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1. 0.900") and lines[0].endswith("b.txt")
    assert lines[-1] == "(ranked by speech-model-ct2 on gpu-1, after 0.2s in the queue)"


def test_the_picture_example_runs(tmp_path, monkeypatch, capsys):
    picture = tmp_path / "receipt.jpg"
    picture.write_bytes(JPEG)
    argv = [str(picture), "What is the total?"]
    (request,) = run_example("ask_about_a_picture.py", argv, monkeypatch)
    sent = json.loads(request.content)
    assert sent["messages"][0]["content"][1] == data_url("image/jpeg", JPEG)
    out = capsys.readouterr().out
    assert out.startswith("42\n") and "on gpu-1, after 0.2s in the queue" in out


def test_the_operators_example_loads_once_merged_into_a_pool_of_its_own():
    """What the example leaves to the operator's own file — keys, and the provider and its
    search settings — added the smallest way; the rest must load as written."""
    example = yaml.safe_load((EXAMPLES / "pool-beyond-chat.yaml").read_text())
    example["auth"] = {"app_keys": ["gpma_example"]}
    operators_own = {"provider": "fake", "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.6}}
    example["rented"] = {**operators_own, **example["rented"]}

    config = PoolConfig.model_validate(example)
    assert config.pool.models_per_host == "declared"
    assert config.rented_engine() == "vllm"
