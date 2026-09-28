"""How the pool's model set is spread over its hosts (D89).

`all` is the original rule (D23): every host holds everything. `declared` spreads the set
across hosts — which is what an engine serving a single model per process needs, and what lets
a 0.3 GB embedding model stay off a card rented for a 26 B one. Note that `declared` does not
mean *one* model: a host may declare several, which is how a laptop keeps the whole set while
rented hosts each hold the one that justifies their price.
"""

import pytest
from gpm_server.config import PoolConfig

BIG, SMALL, EMBED = "gemma4:26b", "gemma4:e4b", "nomic-embed-text"


def config(**overrides):
    pool = {"name": "test", "model_set": [BIG, SMALL, EMBED]}
    pool.update(overrides.pop("pool", {}))
    base = {
        "pool": pool,
        "auth": {"app_keys": ["k"]},
        "engine": "ollama",
        "hosts": [{
            "id": "local-1", "kind": "local", "workers": 1,
            "transport": {"type": "http", "base_url": "http://127.0.0.1:1"},
        }],
    }
    base.update(overrides)
    return PoolConfig.model_validate(base)


def rented(**overrides):
    base = {
        "provider": "fake", "workers": 2, "capabilities": ["cuda"],
        "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02},
    }
    base.update(overrides)
    return base


# --- the default is unchanged ---


def test_by_default_every_host_holds_the_whole_set():
    """D23 stands where nothing asks otherwise: a pool that says nothing behaves as it did."""
    cfg = config()
    assert cfg.pool.models_per_host == "all"
    assert cfg.models_held_by(cfg.hosts[0]) == [BIG, SMALL, EMBED]


# --- spreading the set ---


def test_a_host_holds_what_it_declares():
    cfg = config(
        pool={"models_per_host": "declared"},
        hosts=[
            {"id": "gpu-1", "kind": "local", "workers": 1, "models": [BIG],
             "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}},
            {"id": "cheap-1", "kind": "local", "workers": 1, "models": [SMALL, EMBED],
             "transport": {"type": "http", "base_url": "http://127.0.0.1:2"}},
        ],
    )
    assert cfg.models_held_by(cfg.hosts[0]) == [BIG]
    assert cfg.models_held_by(cfg.hosts[1]) == [SMALL, EMBED]


def test_a_host_that_declares_nothing_takes_the_first_model_it_can_serve():
    """Stable across restarts and explainable in one sentence — rather than left to chance."""
    cfg = config(
        pool={"models_per_host": "declared"},
        hosts=[{"id": "gpu-1", "kind": "local", "workers": 1,
                "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        rented=rented(),
    )
    assert cfg.models_held_by(cfg.hosts[0]) == [BIG]


def test_rented_hosts_can_be_bought_for_only_the_models_that_justify_the_price():
    """The money decision: a 0.3 GB embedding model does not need the card rented for a 26 B
    one, so the pool rents for the big model and serves the rest from machines it has."""
    cfg = config(
        pool={"models_per_host": "declared"},
        hosts=[{"id": "laptop", "kind": "local", "workers": 1, "models": [SMALL, EMBED],
                "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        rented=rented(models=[BIG]),
    )
    assert cfg.rented.models == [BIG]


# --- what is refused, and when ---


def test_a_model_no_host_would_hold_is_refused_at_load():
    """The only symptom otherwise is a 503 for that one model, long after the pool looked
    healthy — and nothing would say why."""
    with pytest.raises(ValueError, match="would be held by none"):
        config(
            pool={"models_per_host": "declared"},
            hosts=[{"id": "gpu-1", "kind": "local", "workers": 1, "models": [BIG],
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        )


def test_renting_covers_what_configured_hosts_do_not():
    config(
        pool={"models_per_host": "declared"},
        hosts=[{"id": "gpu-1", "kind": "local", "workers": 1, "models": [BIG],
                "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        rented=rented(),
    )


def test_renting_for_one_model_does_not_cover_the_others():
    with pytest.raises(ValueError, match="would be held by none"):
        config(
            pool={"models_per_host": "declared"},
            hosts=[{"id": "gpu-1", "kind": "local", "workers": 1, "models": [BIG],
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            rented=rented(models=[BIG]),
        )


def test_a_host_out_of_service_is_still_its_models_home_and_says_what_has_no_host():
    """Out of service is for a while, back with one click, so the file is still whole (D113) —
    but nothing in service holds those models, and the pool says so rather than pretending."""
    cfg = config(
        pool={"models_per_host": "declared"},
        hosts=[
            {"id": "gpu-1", "kind": "local", "workers": 1, "models": [BIG],
             "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}},
            {"id": "off", "kind": "local", "workers": 1, "models": [SMALL, EMBED],
             "disabled": True,
             "transport": {"type": "http", "base_url": "http://127.0.0.1:2"}},
        ],
    )
    assert set(cfg.models_without_a_host_in_service()) == {SMALL, EMBED}


def test_naming_models_while_every_host_holds_everything_is_refused_not_ignored():
    """It would read as a restriction the pool is quietly disregarding."""
    with pytest.raises(ValueError, match="models_per_host is 'all'"):
        config(hosts=[{"id": "gpu-1", "kind": "local", "workers": 1, "models": [BIG],
                       "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}])


def test_a_model_outside_the_pools_set_is_a_typo_worth_naming():
    with pytest.raises(ValueError, match="not in the pool's set"):
        config(
            pool={"models_per_host": "declared"},
            hosts=[{"id": "gpu-1", "kind": "local", "workers": 1, "models": ["gemma4:31b"],
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            rented=rented(),
        )


def test_a_host_declaring_no_models_at_all_is_refused():
    with pytest.raises(ValueError, match="could serve nothing"):
        config(
            pool={"models_per_host": "declared"},
            hosts=[{"id": "gpu-1", "kind": "local", "workers": 1, "models": [],
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            rented=rented(),
        )


def test_a_host_asked_for_a_build_its_platform_cannot_run_is_refused():
    with pytest.raises(ValueError, match="meet no variant's requirements"):
        config(
            pool={"models_per_host": "declared"},
            catalog={BIG: {"variants": [{"tag": "gemma4:26b-mlx", "requires": ["apple-silicon"]}]}},
            hosts=[{"id": "cuda-1", "kind": "local", "workers": 1, "models": [BIG],
                    "capabilities": ["cuda"],
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            # Rented hosts rent for the others, so the configured host is the only problem.
            rented=rented(models=[SMALL, EMBED]),
        )


# --- the engine has a say (D90) ---


def test_an_engine_that_serves_one_model_cannot_be_asked_to_hold_the_whole_set():
    """Refused at load, rather than found after a machine has been rented that can never reach
    `ready` — and the message says what to do about it."""
    with pytest.raises(ValueError, match="one model per process"):
        config(engine="vllm")


def test_that_same_engine_is_fine_once_the_set_is_spread():
    cfg = config(
        engine="vllm",
        pool={"models_per_host": "declared"},
        hosts=[{"id": "gpu-1", "kind": "local", "workers": 1, "models": [BIG],
                "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
        # An image built for this engine (D92). `rented.models` is what the pool may rent
        # *for* — each host it buys is given one of them (D94).
        rented=rented(image="vastai/vllm:v0.29.0-cuda-12.9", models=[SMALL, EMBED]),
    )
    assert cfg.engine == "vllm" and cfg.models_held_by(cfg.hosts[0]) == [BIG]


def test_the_first_engine_holds_the_whole_set_as_it_always_did():
    assert config(engine="ollama").pool.models_per_host == "all"


def test_rented_hosts_with_no_build_of_a_model_they_rent_for_are_refused():
    """Otherwise the host is prepared for nothing and called ready holding nothing — a model
    never served however many machines are bought for it (D98)."""
    with pytest.raises(ValueError, match="no build of"):
        config(
            pool={"models_per_host": "declared"},
            catalog={BIG: {"variants": [{"tag": "gemma4:26b", "engine": "ollama"}]}},
            hosts=[{"id": "laptop", "kind": "local", "workers": 1, "models": [BIG, SMALL, EMBED],
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            rented=rented(engine="vllm", image="vastai/vllm:v0.29.0-cuda-12.9", models=[BIG]),
        )
