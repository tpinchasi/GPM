"""What a configuration may say about models beyond chat, and what it is told when it cannot work
(D142, D143, D147, D148). An operator who asks Ollama for speech hears it before anything is
rented or written."""

import pytest
from gpm_server.config import PoolConfig

WHISPER = "speech"


def config(*, catalog=None, hosts=None, rented=None, model_set=None, pool=None, **extra):
    base = {
        "pool": {"name": "test", "model_set": model_set or [WHISPER], **(pool or {})},
        "auth": {"app_keys": ["k"]},
        "catalog": catalog or {},
        "hosts": hosts if hosts is not None else [],
        **({"rented": rented} if rented is not None else {}),
        **extra,
    }
    return PoolConfig.model_validate(base)


def local(host_id="mac", **overrides):
    return {"id": host_id, "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"}, **overrides}


#: An image of vLLM said to carry its audio extra (D146).
AUDIO_IMAGES = [{"image": "example/vllm-audio:v0.29.0", "min_driver": "550", "capabilities": ["audio-decoding"]}]


def renting(**overrides):
    return {"provider": "fake", "workers": 2, "capabilities": ["cuda"],
            "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02}, **overrides}


def test_an_ollama_build_cannot_claim_speech():
    with pytest.raises(ValueError, match="cannot serve \\['transcribe'\\]"):
        config(catalog={WHISPER: {"variants": [{"tag": "whisper:turbo", "engine": "ollama", "serves": ["transcribe"]}]}},
               hosts=[local()])


def test_a_host_running_ollama_asked_for_speech_is_told_why_and_what_to_do():
    catalog = {WHISPER: {"variants": [{"tag": "owner/whisper", "serves": ["transcribe"]}]}}
    with pytest.raises(ValueError) as refused:
        config(catalog=catalog, hosts=[local()])
    message = str(refused.value)
    assert "speech" in message and "ollama" in message and "vLLM or whisper-cpp" in message


def test_rented_hosts_run_vllm_unless_told_otherwise_and_a_pool_relying_on_ollama_hears_how_to_keep_it():
    vllm_catalog = {WHISPER: {"variants": [{"tag": "owner/whisper", "engine": "vllm", "serves": ["transcribe"]}]}}
    assert config(catalog=vllm_catalog, rented=renting(images=AUDIO_IMAGES)).rented_engine() == "vllm"

    with pytest.raises(ValueError) as refused:
        config(model_set=["chat"], catalog={"chat": {"variants": [{"tag": "chat:7b", "engine": "ollama"}]}},
               rented=renting())
    assert "rented.engine: ollama" in str(refused.value)
    kept = config(model_set=["chat"], catalog={"chat": {"variants": [{"tag": "chat:7b", "engine": "ollama"}]}},
                  rented=renting(engine="ollama"))
    assert kept.rented_engine() == "ollama"


def test_a_pool_that_rents_nothing_runs_only_its_own_engine():
    pool = config(model_set=["chat"], hosts=[local()])
    assert pool.engines_in_use() == ["ollama"]


def whisper_host(**overrides):
    return local(
        "mac-whisper", engine="whisper-cpp", workers=1,
        transport={"type": "http", "base_url": "http://127.0.0.1:8178"},
        agent={"url": "http://127.0.0.1:8096", "bearer_env": "WHISPER_AGENT_KEY"},
        **overrides,
    )


WHISPER_CATALOG = {WHISPER: {"variants": [
    {"tag": "owner/whisper-ggml/model-q5_0.bin", "engine": "whisper-cpp", "serves": ["transcribe"]},
]}}


def test_a_whisper_cpp_host_with_its_agent_on_loopback_and_one_worker_loads():
    pool = config(catalog=WHISPER_CATALOG, hosts=[whisper_host()])
    assert pool.engine_of(pool.hosts[0]) == "whisper-cpp"


@pytest.mark.parametrize("change, said", [
    ({"agent": None}, "give the host an `agent`"),
    # Plain http off the machine is refused for every host; https still reaches a server with no key.
    ({"transport": {"type": "https", "base_url": "https://192.168.1.20:8178"}}, "no authentication"),
    ({"workers": 3}, "workers: 1"),
    ({"agent": {"url": "http://127.0.0.1:8096", "bearer_env": "WHISPER_AGENT_KEY", "manage_models": False}},
     "manage_models: false"),
    ({"residency": "on_demand"}, "residency: on_demand"),
])
def test_a_whisper_cpp_host_is_refused_without_its_agent_off_loopback_or_with_extra_workers(change, said):
    host = whisper_host()
    host.update(change)
    if host.get("agent") is None:
        host.pop("agent")
    with pytest.raises(ValueError, match=said):
        config(catalog=WHISPER_CATALOG, hosts=[host])


def test_serves_and_inputs_are_three_valued_and_speech_hears():
    pool = config(
        model_set=["vision", WHISPER],
        catalog={
            "vision": {"variants": [{"tag": "v:said", "inputs": ["image"]}, {"tag": "v:text", "inputs": []},
                                    {"tag": "v:unsaid"}]},
            # Held by a vLLM machine the pool would rent; a laptop running Ollama cannot.
            WHISPER: {"variants": [{"tag": "owner/whisper", "engine": "vllm", "serves": ["transcribe"]}]},
        },
        hosts=[local(models=["vision"])],
        rented=renting(models=[WHISPER], images=AUDIO_IMAGES),
        pool={"models_per_host": "declared"},
    )
    traits = pool.build_traits()
    assert traits["v:said"] == (frozenset({"chat", "embed"}), frozenset({"image"}))
    assert traits["v:text"][1] == frozenset()
    assert traits["v:unsaid"][1] is None
    assert traits["owner/whisper"] == (frozenset({"transcribe"}), frozenset({"audio"}))


def test_speech_on_rented_vllm_needs_an_image_said_to_decode_audio():
    catalog = {WHISPER: {"variants": [{"tag": "owner/whisper", "engine": "vllm", "serves": ["transcribe"]}]}}
    with pytest.raises(ValueError, match="capabilities: \\['audio-decoding'\\]"):
        config(catalog=catalog, rented=renting())
    plain = [{"image": "example/vllm:v0.29.0", "min_driver": "550"}]
    with pytest.raises(ValueError, match="example/vllm:v0.29.0"):
        config(catalog=catalog, rented=renting(images=plain))
    pool = config(catalog=catalog, rented=renting(images=plain + AUDIO_IMAGES))
    assert pool.image_needs(["owner/whisper"]) == frozenset({"audio-decoding"})
    assert pool.image_needs(["chat:7b"]) == frozenset()


def test_rented_hosts_cannot_run_whisper_cpp():
    with pytest.raises(ValueError, match="rented hosts cannot run whisper-cpp"):
        config(catalog=WHISPER_CATALOG, rented=renting(engine="whisper-cpp"))


def test_a_request_limit_above_what_a_machine_accepts_is_refused():
    with pytest.raises(ValueError, match="64"):
        config(model_set=["chat"], hosts=[local()], pool={"max_request_mb": 100})
