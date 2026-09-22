"""The vLLM adapter, and the OpenAI-shaped surface both engines now serve (D89, D90, D91).

docs/spec/plugin-interfaces.md §2. Against a fake HTTP transport; nothing here needs a GPU, a
cloud account, or vLLM itself.
"""

import httpx
import pytest
from gpm_server.engines import Occupancy, OllamaEngine, VllmEngine, openai_api
from gpm_server.engines.vllm import BATCHED_TOKENS, CONTEXT, LISTEN, WORKERS

engine = VllmEngine()
ollama = OllamaEngine()

CHAT = "/v1/chat/completions"


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://engine")


def answering(routes: dict[str, object], status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path not in routes:
            return httpx.Response(404)
        body = routes[request.url.path]
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return client_for(handler)


# --- the request path, which is the shared protocol rather than this engine's invention ---


def test_the_paths_that_take_a_worker_exclude_listing_models():
    """Listing models is metadata: it costs an engine nothing and must not consume capacity the
    pool is counting on behalf of requests that do."""
    paths = engine.inference_paths()
    assert CHAT in paths and "/v1/embeddings" in paths
    assert "/v1/models" not in paths and "/metrics" not in paths and "/health" not in paths


def test_the_model_field_is_rewritten_and_nothing_else_is():
    body = b'{"model":"logical","messages":[{"role":"user","content":"hi"}],"temperature":0.7}'
    assert engine.with_model(CHAT, body, "google/gemma-4-26B-A4B-it") == (
        b'{"model":"google/gemma-4-26B-A4B-it","messages":[{"role":"user","content":"hi"}],'
        b'"temperature":0.7}'
    )


def test_a_body_naming_no_model_is_returned_untouched():
    assert engine.with_model(CHAT, b'{"messages":[]}', "m") == b'{"messages":[]}'
    assert engine.requested_model(CHAT, b"not json at all") is None


def test_streaming_defaults_to_off_here_and_on_in_ollamas_own_api():
    """The two surfaces have **opposite** defaults, and an adapter serving both must not apply
    one to the other: the pool would hold a stream believing it a whole body, or frame a whole
    body believing it a stream."""
    assert engine.is_streaming(CHAT, b'{"model":"m"}') is False
    assert engine.is_streaming(CHAT, b'{"model":"m","stream":true}') is True
    assert ollama.is_streaming("/api/chat", b'{"model":"m"}') is True
    assert ollama.is_streaming(CHAT, b'{"model":"m"}') is False


def test_embeddings_never_stream():
    assert engine.is_streaming("/v1/embeddings", b'{"model":"m","stream":true}') is False


def test_a_schema_counts_but_loose_json_mode_does_not():
    """`json_object` asks for valid JSON and guarantees nothing about its shape; only a schema
    is something the pool may route on."""
    schema = b'{"model":"m","response_format":{"type":"json_schema","json_schema":{"name":"x"}}}'
    assert engine.wants_schema(CHAT, schema) is True
    assert engine.wants_schema(CHAT, b'{"model":"m","response_format":{"type":"json_object"}}') is False
    assert engine.wants_schema(CHAT, b'{"model":"m"}') is False


def test_token_counts_are_read_from_a_whole_body():
    body = b'{"id":"c","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":42}}'
    assert engine.usage(CHAT, body) == (42, None)


def test_token_counts_are_read_from_the_last_frame_of_a_stream():
    tail = (
        b'data: {"choices":[{"delta":{"content":"o"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert engine.usage(CHAT, tail) == (7, None)


def test_a_stream_that_was_not_asked_to_report_usage_says_so_rather_than_guessing():
    tail = b'data: {"choices":[{"delta":{"content":"o"}}]}\n\ndata: [DONE]\n\n'
    assert engine.usage(CHAT, tail) == (None, None)


def test_the_duration_is_never_claimed_because_this_protocol_does_not_report_it():
    """Returning wall-clock here would fold queue wait into generation time, and telling those
    apart is what deciding a host's worker count turns on (D67)."""
    _, milliseconds = engine.usage(CHAT, b'{"usage":{"completion_tokens":5}}')
    assert milliseconds is None


def test_a_tail_that_begins_mid_frame_does_not_fail_the_read():
    assert engine.usage(CHAT, b'ontent":"x"}}]}\n\ndata: {"usage":{"completion_tokens":3}}\n\n') == (3, None)


def test_the_keepalive_frame_is_one_the_protocol_says_to_ignore():
    """Server-sent events discard a line beginning with `:` by specification, so this is safe
    to send while a response is held whole (D62) — and it must never reach a whole-body path."""
    assert engine.keepalive_frame(CHAT) == b": keep-alive\n\n"
    assert engine.keepalive_frame("/v1/embeddings") is None
    assert ollama.keepalive_frame("/api/chat") is None
    assert ollama.keepalive_frame(CHAT) == b": keep-alive\n\n"


# --- what the pool asks the engine ---


async def test_health_is_the_engines_own_answer():
    async with answering({"/health": ""}) as client:
        assert (await engine.health(client)).ok is True
    async with answering({}) as client:
        assert (await engine.health(client)).ok is False


async def test_resident_and_available_are_the_same_set():
    """This engine is launched with its model and holds it for the life of the process: weights
    on disk it was not launched with are not servable without a restart, so counting them would
    tell the pool it can serve something it cannot."""
    models = {"data": [{"id": "google/gemma-4-26B-A4B-it"}, {"id": "nomic-embed-text"}]}
    async with answering({"/v1/models": models}) as client:
        assert await engine.models_resident(client) == frozenset(
            {"google/gemma-4-26B-A4B-it", "nomic-embed-text"}
        )
        assert await engine.models_available(client) == await engine.models_resident(client)


# --- occupancy: the number the pool's worker slots cannot express (D91) ---

METRICS = """\
# HELP vllm:num_requests_running Number of requests currently running.
vllm:num_requests_running{model_name="gemma"} 64.0
vllm:num_requests_waiting{model_name="gemma"} 11.0
vllm:kv_cache_usage_perc{model_name="gemma"} 0.87
"""


async def test_occupancy_reports_what_the_engine_is_holding():
    async with answering({"/metrics": METRICS}) as client:
        assert await engine.occupancy(client) == Occupancy(running=64, waiting=11, cache_used=0.87)


async def test_the_queue_inside_the_engine_is_what_the_pool_could_not_otherwise_see():
    """A host admitted at a hundred workers with ninety-nine in flight looks idle by slot count
    while requests pile up inside the engine. `waiting` is the whole point of asking."""
    async with answering({"/metrics": METRICS}) as client:
        occupancy = await engine.occupancy(client)
    assert occupancy is not None and occupancy.waiting == 11


async def test_an_older_builds_name_for_the_cache_is_accepted():
    older = 'vllm:num_requests_running 2.0\nvllm:num_requests_waiting 0.0\nvllm:gpu_cache_usage_perc 0.5\n'
    async with answering({"/metrics": older}) as client:
        assert (await engine.occupancy(client)).cache_used == 0.5


async def test_an_engine_that_will_not_say_returns_nothing_rather_than_a_guess():
    """None means "judge me by worker slots, as before" — it must never read as "idle"."""
    async with answering({}) as client:
        assert await engine.occupancy(client) is None
    async with answering({"/metrics": "vllm:num_requests_running 3.0\n"}) as client:
        assert await engine.occupancy(client) is None


async def test_the_engine_the_pool_shipped_first_still_declines_to_say():
    async with answering({"/api/ps": {"models": []}}) as client:
        assert await ollama.occupancy(client) is None


# --- preparing a host ---


async def test_this_engine_cannot_fetch_a_model_and_says_so_unretryably():
    """Weights arrive from a model hub before the engine starts, not over its API. Reporting a
    transient failure would have the supervisor retry a download that cannot happen, and give
    up on the host for the wrong reason."""
    async with answering({}) as client:
        result = await engine.pull(client, "google/gemma-4-26B-A4B-it")
    assert result.ok is False and result.retryable is False
    assert "agent" in result.detail


async def test_load_and_pin_verifies_rather_than_acts():
    served = {"data": [{"id": "gemma"}]}
    async with answering({"/v1/models": served}) as client:
        await engine.load_and_pin(client, ["gemma"])
        with pytest.raises(RuntimeError, match="not served by this engine"):
            await engine.load_and_pin(client, ["gemma", "nomic-embed-text"])


# --- launch settings: numbers for the host's own start command, never a command ---


def test_launch_settings_are_numbers_under_this_engines_names():
    settings = engine.launch_settings(workers=84, context=32768, n_models=3, listen="127.0.0.1:8000")
    assert settings[WORKERS] == "84" and settings[CONTEXT] == "32768"
    assert settings[LISTEN] == "127.0.0.1:8000"


def test_the_batch_is_never_smaller_than_the_sequences_the_engine_must_run():
    """Below its own floor the engine refuses to start, which would present as a host that
    never answers rather than as the arithmetic mistake it is."""
    for workers in (1, 7, 84, 512):
        settings = engine.launch_settings(workers=workers, context=4096, n_models=1)
        assert int(settings[BATCHED_TOKENS]) >= workers


def test_nothing_is_bound_wider_than_asked():
    """The pool reaches the engine through a forward into the machine, so binding every
    interface only exposes it to everyone else (D77)."""
    assert LISTEN not in engine.launch_settings(workers=8, context=4096, n_models=1)


def test_the_count_of_models_is_not_passed_on_because_it_cannot_be_honoured():
    """One process, one model. Accepting the pool's count of the set would be claiming to hold
    a number of models this engine has no way to hold."""
    settings = engine.launch_settings(workers=8, context=4096, n_models=3)
    assert not any("MODELS" in name for name in settings)


# --- the shared module is the protocol, not either engine ---


def test_both_engines_answer_the_shared_paths_identically():
    body = b'{"model":"m","stream":true}'
    for path in sorted(openai_api.INFERENCE_PATHS):
        assert ollama.is_streaming(path, body) == engine.is_streaming(path, body)
        assert ollama.wants_schema(path, body) == engine.wants_schema(path, body)
        assert ollama.keepalive_frame(path) == engine.keepalive_frame(path)


def test_ollama_keeps_serving_its_own_api_as_well():
    """Apps written against the native paths keep working; nothing translates between the two."""
    assert {"/api/chat", "/api/generate", "/api/embed"} <= ollama.inference_paths()
    assert openai_api.INFERENCE_PATHS <= ollama.inference_paths()
