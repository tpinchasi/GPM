import json

import pytest
from gpm_server.engines import AmbiguousField, EngineNotFound, OllamaEngine, VllmEngine, get_engine

engine = OllamaEngine()


def test_get_engine_by_name():
    assert isinstance(get_engine("ollama"), OllamaEngine)
    assert isinstance(get_engine("vllm"), VllmEngine)
    with pytest.raises(EngineNotFound):
        get_engine("an-engine-nobody-installed")


def test_requested_model():
    assert engine.requested_model("/api/chat", b'{"model":"m1","messages":[]}') == "m1"
    assert engine.requested_model("/api/chat", b'{"messages":[]}') is None
    assert engine.requested_model("/api/chat", b"not json") is None


def test_with_model_changes_the_model_field_and_nothing_else():
    body = b'{"messages":[{"role":"user","content":"hi \\"there\\""}],  "model" : "m1" ,"stream":true,"n":1.50}'
    rewritten = engine.with_model("/api/chat", body, "m1-mlx")
    assert rewritten == b'{"messages":[{"role":"user","content":"hi \\"there\\""}],  "model" : "m1-mlx" ,"stream":true,"n":1.50}'


def test_with_model_only_rewrites_the_first_model_field():
    body = b'{"model":"m1","options":{"model":"inner"}}'
    assert engine.with_model("/api/chat", body, "x") == b'{"model":"x","options":{"model":"inner"}}'


def test_with_model_escapes_the_replacement():
    body = b'{"model":"m1"}'
    assert engine.with_model("/api/chat", body, 'a"b') == b'{"model":"a\\"b"}'
    assert json.loads(engine.with_model("/api/chat", body, 'a"b'))["model"] == 'a"b'


def test_with_model_leaves_a_body_without_a_model_field_untouched():
    body = b'{"messages":[]}'
    assert engine.with_model("/api/chat", body, "x") == body


def test_a_pool_field_named_twice_or_in_another_case_is_refused():
    for body in (
        b'{"model":"m1","MODEL":"m2"}',
        b'{"MODEL":"m2","model":"m1"}',
        b'{"model":"m1","model":"m2"}',
        b'{"model":"m1","Model":"m1"}',
        b'{"model":"m1","STREAM":false}',
        b'{"model":"m1","\\u017ftream":true}',  # the long s, which Go folds to "s"
        b'{"model":"m1","format":{},"Format":"json"}',
        b'{"model":"m1","response_format":{},"RESPONSE_FORMAT":{}}',
    ):
        with pytest.raises(AmbiguousField):
            engine.requested_model("/api/chat", body)
    # Fields the pool does not read are the engine's business, in any case.
    assert engine.requested_model("/api/chat", b'{"model":"m1","Options":{},"options":{}}') == "m1"


def test_the_top_level_model_is_rewritten_never_a_nested_one():
    body = b'{"options":{"model":"inner"},"model":"m1"}'
    assert engine.with_model("/api/chat", body, "x") == b'{"options":{"model":"inner"},"model":"x"}'


def test_an_escaped_model_key_is_the_model_and_is_rewritten():
    # `mod\u0065l` decodes to `model` in every JSON parser, so it is read and rewritten as one.
    body = b'{"mod\\u0065l":"m1","messages":[]}'
    assert engine.requested_model("/api/chat", body) == "m1"
    assert engine.with_model("/api/chat", body, "x") == b'{"mod\\u0065l":"x","messages":[]}'


def test_bytes_outside_ascii_survive_a_rewrite():
    body = '{"messages":[{"content":"שלום 🌍"}],"model":"m1"}'.encode()
    assert engine.with_model("/api/chat", body, "x") == '{"messages":[{"content":"שלום 🌍"}],"model":"x"}'.encode()


def test_what_is_not_one_json_object_names_no_model():
    for body in (b"[1,2]", b'{"model":"m1"} trailing', b"\xef\xbb\xbf" + b'{"model":"m1"}', b'{"model":"m1",}', b"[" * 100_000):
        assert engine.requested_model("/api/chat", body) is None, body[:20]
        assert engine.with_model("/api/chat", body, "x") == body


def test_wants_schema_only_for_a_real_schema():
    assert engine.wants_schema("/api/chat", b'{"model":"m","format":{"type":"object"}}')
    # Loose JSON mode guarantees nothing about shape, so it is not schema enforcement.
    assert not engine.wants_schema("/api/chat", b'{"model":"m","format":"json"}')
    assert not engine.wants_schema("/api/chat", b'{"model":"m"}')


def test_is_streaming_defaults_to_the_engines_own_default():
    assert engine.is_streaming("/api/chat", b'{"model":"m"}')
    assert not engine.is_streaming("/api/chat", b'{"model":"m","stream":false}')
    assert not engine.is_streaming("/api/embed", b'{"model":"m"}')


def test_inference_paths():
    assert "/api/chat" in engine.inference_paths()
    assert "/api/ps" not in engine.inference_paths()
