import json

import pytest
from gpm_server.engines import EngineNotFound, OllamaEngine, VllmEngine, get_engine

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
