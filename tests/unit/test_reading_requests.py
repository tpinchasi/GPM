"""How the router reads what a request asks for and carries (D141–D144), and what it learned from
its configuration to judge it by — each read the same way whichever engine's rules apply, and
nothing on the request path looking an engine up again."""

import base64
import json

import httpx
import pytest
from gpm_server.catalog import ResolvedVariant
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.engines import (
    AmbiguousField,
    MediaNotInline,
    OllamaEngine,
    VllmEngine,
    facts_of,
    openai_api,
)
from gpm_server.models import Host, HostState, Worker
from gpm_server.router.dispatch import Need
from gpm_server.state import RouterState

PICTURE = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
INLINE = {"type": "image_url", "image_url": {"url": PICTURE}}
LINK = {"type": "image_url", "image_url": {"url": "https://10.0.0.1/internal.png"}}


def facts(path, body):
    return openai_api.json_facts(path, json.dumps(body).encode())


# --- pictures to embed or rank are read as in chat (D143, D146) ---


@pytest.mark.parametrize("body", [
    {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}, INLINE]}]},
    {"model": "m", "input": [INLINE]},
    {"model": "m", "input": INLINE},
    {"model": "m", "input": [{"content": [INLINE]}]},
    {"model": "m", "input": [[{"type": "text", "text": "x"}, INLINE]]},
    {"model": "m", "input": [{"role": "user", "content": [INLINE]}]},
])
def test_a_picture_to_embed_is_found_wherever_the_engine_would_find_it(body):
    assert facts("/v1/embeddings", body).inputs == {"image"}


@pytest.mark.parametrize("body", [
    {"model": "m", "query": {"content": [INLINE]}, "documents": ["a"]},
    {"model": "m", "query": "q", "documents": ["a", {"content": [INLINE]}]},
    {"model": "m", "query": "q", "documents": {"content": [INLINE]}},
    {"model": "m", "query": "q", "documents": [INLINE]},
])
def test_a_picture_to_rank_is_found_in_the_query_or_any_document(body):
    assert facts("/v1/rerank", body).inputs == {"image"}


@pytest.mark.parametrize("path, body", [
    ("/v1/embeddings", {"model": "m", "messages": [{"role": "user", "content": [LINK]}]}),
    ("/v1/embeddings", {"model": "m", "input": [LINK]}),
    ("/v1/embeddings", {"model": "m", "input": [{"content": [LINK]}]}),
    ("/v1/embeddings", {"model": "m", "input": [{"type": "video_url", "video_url": {"url": "x"}}]}),
    ("/v1/rerank", {"model": "m", "query": {"content": [LINK]}, "documents": ["a"]}),
    ("/v1/rerank", {"model": "m", "query": "q", "documents": ["a", {"content": [LINK]}]}),
    ("/v1/rerank", {"model": "m", "query": "q", "documents": {"content": [LINK]}}),
    ("/v1/rerank", {"model": "m", "query": "q", "documents": [{"type": "image_embeds", "image_embeds": "x"}]}),
])
def test_a_link_or_an_unknown_part_to_embed_or_rank_is_refused(path, body):
    with pytest.raises(MediaNotInline):
        facts(path, body)


def test_text_and_tokens_to_embed_or_rank_carry_nothing():
    assert facts("/v1/embeddings", {"model": "m", "input": "words"}).inputs == frozenset()
    assert facts("/v1/embeddings", {"model": "m", "input": ["a", "b"]}).inputs == frozenset()
    assert facts("/v1/embeddings", {"model": "m", "input": [[1, 2, 3], [4]]}).inputs == frozenset()
    assert facts("/v1/rerank", {"model": "m", "query": "q", "documents": ["a", "b"]}).inputs == frozenset()


# --- the fields that carry pictures follow D141's rule ---


@pytest.mark.parametrize("raw", [
    b'{"model": "m", "messages": [], "Messages": [{"role": "user", "content": "x", "images": ["aGk="]}]}',
    b'{"model": "m", "messages": [{"role": "user", "content": "x", "Images": ["aGk="]}]}',
    b'{"model": "m", "messages": [{"role": "user", "content": "x", "images": [], "images": ["aGk="]}]}',
    b'{"model": "m", "messages": [{"role": "user", "content": "x", "CONTENT": "y"}]}',
    b'{"model": "m", "prompt": "x", "IMAGES": ["aGk="]}',
])
def test_ollama_reading_keys_in_any_case_cannot_be_shown_a_picture_the_pool_did_not_see(raw):
    path = "/api/generate" if b"prompt" in raw else "/api/chat"
    with pytest.raises(AmbiguousField):
        OllamaEngine().request_facts(path, "application/json", raw)


@pytest.mark.parametrize("path, raw", [
    ("/v1/chat/completions", b'{"model": "m", "Messages": [{"role": "user", "content": "x"}]}'),
    ("/v1/chat/completions", b'{"model": "m", "messages": [], "messages": []}'),
    ("/v1/chat/completions", b'{"model": "m", "messages": [{"role": "user", "content": "x", "Content": []}]}'),
    ("/v1/responses", b'{"model": "m", "INPUT": [{"role": "user", "content": "x"}]}'),
    ("/v1/embeddings", b'{"model": "m", "Input": ["x"]}'),
    ("/v1/rerank", b'{"model": "m", "query": "q", "documents": [], "Documents": ["x"]}'),
])
def test_the_shared_paths_hold_the_fields_that_carry_pictures_to_the_same_rule(path, raw):
    with pytest.raises(AmbiguousField):
        openai_api.json_facts(path, raw)


def test_a_message_whose_role_is_spelled_otherwise_is_still_read():
    raw = json.dumps({"model": "m", "messages": [{"ROLE": "user", "content": [INLINE]}]}).encode()
    assert openai_api.json_facts("/v1/chat/completions", raw).inputs == {"image"}


def test_a_schema_whose_properties_differ_only_in_case_is_not_refused():
    raw = (b'{"model": "m", "messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_schema",'
           b' "json_schema": {"schema": {"properties": {"Content": {}, "content": {}, "IMAGES": {}}}}}}')
    assert openai_api.json_facts("/v1/chat/completions", raw).wants_schema


# --- a plug-in written before kinds still has its requests measured by kind (D145) ---


class VersionOneEngine:
    """A third-party engine written against version 1: it reads the model, and nothing more."""

    interface_version = "1"
    name = "plugin"

    def inference_paths(self):
        return set(openai_api.INFERENCE_PATHS) | {"/api/chat", "/api/generate", "/api/embed", "/api/embeddings"}

    def requested_model(self, path, body):
        return openai_api.requested_model(body)

    def is_streaming(self, path, body):
        return False

    def wants_schema(self, path, body):
        return False


@pytest.mark.parametrize("path, kind", [
    ("/v1/chat/completions", "chat"), ("/v1/responses", "chat"), ("/api/chat", "chat"), ("/api/generate", "chat"),
    ("/v1/embeddings", "embed"), ("/api/embed", "embed"), ("/api/embeddings", "embed"),
    ("/v1/rerank", "rerank"), ("/v1/audio/transcriptions", "transcribe"),
])
def test_a_version_one_engines_request_has_the_kind_its_path_says(path, kind):
    assert facts_of(VersionOneEngine(), path, "application/json", b'{"model": "m"}').kind == kind


def config(**overrides):
    base = {
        "pool": {"name": "test", "model_set": ["vision", "other"]},
        "auth": {"app_keys": ["k"]},
        "catalog": {
            # The model's name is also the tag of one of its builds: the text-only one.
            "vision": {"variants": [{"tag": "vision-mlx", "inputs": ["image"]}, {"tag": "vision", "inputs": []}]},
            "other": {"variants": [{"tag": "other:text", "inputs": []}, {"tag": "other:any"}]},
        },
        "hosts": [{"id": "mac", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"}}],
    }
    base.update(overrides)
    return PoolConfig.model_validate(base)


@pytest.fixture
def state(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    yield RouterState(config(), database)
    database.close()


def test_a_version_two_engine_reads_a_path_it_shares_with_a_version_one_plug_in(state):
    # The plug-in is the pool's own engine, listed first; vLLM hosts serve the same paths.
    state.engines = {"plugin": VersionOneEngine(), "vllm": VllmEngine(), **state.engines}
    assert state.parser_for("/v1/chat/completions").name in ("ollama", "vllm")
    assert state.parser_for("/v1/embeddings").name in ("ollama", "vllm")


# --- what the router learned from its configuration ---


def test_a_name_is_judged_as_a_catalogued_model_before_as_a_build_tag(state):
    # Dispatch reads `vision` as the model, both of whose builds it may choose between.
    assert state.refusal_for("vision", "chat", frozenset({"image"})) is None
    # A tag no name shadows is judged by that one build.
    assert state.refusal_for("other:text", "chat", frozenset({"image"}))[0] == "input_not_accepted"
    assert state.refusal_for("other:any", "chat", frozenset({"image"})) is None
    assert state.refusal_for("vision-mlx", "chat", frozenset({"image"})) is None


def test_nothing_on_the_request_path_looks_an_engine_up(state, monkeypatch):
    # Finding an engine scans every installed package's entry points: tens of milliseconds, on
    # the loop every stream in flight shares. Learned when the configuration is; never per request.
    def scan():
        raise AssertionError("an engine was looked up on the request path")

    monkeypatch.setattr("gpm_server.engines.base.available_engines", scan)
    host = Host(
        host_id="mac", kind="local", priority=1, capabilities=frozenset(), client=httpx.AsyncClient(),
        workers=[Worker(worker_id="mac/w0")], variants={}, literal_variants={},
        state=HostState.READY, resident=frozenset({"other:text"}), engine="ollama",
    )
    held = ResolvedVariant(tag="other:text", runtime_class="apple-silicon-ollama", enforces_schema=None)
    host.variants = {"other": (held,)}
    host.literal_variants = {"other:text": held}
    host.observed_inputs = {"other:text": frozenset()}
    state.dispatcher.hosts.append(host)
    state.registry.by_id["mac"] = host

    for name in ("vision", "vision-mlx", "other", "other:text", "a-name-learned-elsewhere"):
        for kind in ("chat", "embed", "transcribe", "rerank"):
            state.refusal_for(name, kind, frozenset({"image"}))
    # The mac could hold `other:any`, which states nothing; it holds only the text build.
    assert state.refused_for_inputs_alone(Need(model="other", inputs=frozenset({"image"})), frozenset())
