"""A pool may run more than one engine, one per host (D93).

The case it exists for: a laptop on Apple silicon runs Ollama, because vLLM has no practical
Metal backend, while the rented machines worth paying for run vLLM. What each machine runs is a
fact about the machine, and an application sees none of it.

This supersedes "a pool has one engine type" in docs/spec/plugin-interfaces.md §2.
"""

import httpx
import pytest
from gpm_server.catalog import variants_for_host
from gpm_server.config import PoolConfig
from gpm_server.db import Database, HostRow, HostTable
from gpm_server.models import Host, HostState, Worker
from gpm_server.router.dispatch import Dispatcher, Need
from gpm_server.state import RouterState

BIG, EMBED = "gemma4:26b", "nomic-embed-text"

CATALOG = {
    BIG: {"variants": [
        {"tag": "gemma4:26b", "engine": "ollama"},
        {"tag": "nvidia/Gemma-4-26B-A4B-NVFP4", "engine": "vllm"},
    ]},
    EMBED: {"variants": [{"tag": "nomic-embed-text"}]},   # no engine: usable by either
}


def config(**overrides):
    base = {
        "pool": {"name": "test", "model_set": [BIG, EMBED], "models_per_host": "declared"},
        "auth": {"app_keys": ["k"]},
        "engine": "ollama",
        "catalog": CATALOG,
        "hosts": [
            {"id": "laptop", "kind": "local", "workers": 3, "models": [BIG, EMBED],
             "capabilities": ["apple-silicon"],
             "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"}},
        ],
        "rented": {
            "provider": "fake", "workers": 2, "capabilities": ["cuda"],
            "engine": "vllm", "models": [BIG],
            "image": "vastai/vllm:v0.29.0-cuda-12.9",
            "bidding": {"bid_ceiling": 2.0, "premium": 0.02},
        },
    }
    base.update(overrides)
    return PoolConfig.model_validate(base)


# --- what each machine runs ---


def test_a_host_runs_the_pools_engine_unless_it_says_otherwise():
    cfg = config()
    assert cfg.engine_of(cfg.hosts[0]) == "ollama"
    assert cfg.rented_engine() == "vllm"


def test_the_pool_knows_every_engine_it_runs():
    """The router needs all of them: it must know which paths its hosts serve between them."""
    assert set(config().engines_in_use()) == {"ollama", "vllm"}


def test_a_host_may_name_its_own_engine():
    cfg = config(hosts=[
        {"id": "gpu-box", "kind": "local", "workers": 2, "models": [BIG], "engine": "vllm",
         "capabilities": ["cuda"], "transport": {"type": "http", "base_url": "http://127.0.0.1:8000"}},
        # The embedding model still has to be held by somebody, whatever engine runs where.
        {"id": "laptop", "kind": "local", "workers": 3, "models": [EMBED],
         "capabilities": ["apple-silicon"],
         "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"}},
    ])
    assert cfg.engine_of(cfg.hosts[0]) == "vllm"
    assert cfg.engine_of(cfg.hosts[1]) == "ollama"


# --- each host is offered only builds its engine can read ---


def test_a_build_is_offered_only_to_the_engine_it_is_for():
    """The same model is `gemma4:26b` to one engine and a hub repository to another. Handing
    either to the wrong engine produces a host that looks ready and cannot serve."""
    cfg = config()
    laptop = variants_for_host([BIG], cfg.catalog, {"apple-silicon"}, "ollama")
    rented = variants_for_host([BIG], cfg.catalog, {"cuda"}, "vllm")

    assert [v.tag for v in laptop[BIG]] == ["gemma4:26b"]
    assert [v.tag for v in rented[BIG]] == ["nvidia/Gemma-4-26B-A4B-NVFP4"]


def test_a_build_naming_no_engine_is_usable_by_any():
    """Which is what every catalog written before pools could run two engines means."""
    cfg = config()
    for engine in ("ollama", "vllm"):
        resolved = variants_for_host([EMBED], cfg.catalog, {"cuda"}, engine)
        assert [v.tag for v in resolved[EMBED]] == [EMBED]


def test_a_host_with_no_build_its_engine_can_read_is_refused_at_load():
    with pytest.raises(ValueError, match="no usable variant"):
        config(
            pool={"name": "t", "model_set": [BIG], "models_per_host": "all"},
            catalog={BIG: {"variants": [{"tag": "gemma4:26b", "engine": "ollama"}]}},
            hosts=[{"id": "gpu-box", "kind": "local", "workers": 1, "engine": "vllm",
                    "capabilities": ["cuda"],
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:8000"}}],
            rented=None,
        )


# --- routing: a request must not reach a host whose engine cannot serve its path ---


def a_host(host_id: str, engine: str, tag: str) -> Host:
    from gpm_server.catalog import ResolvedVariant

    variant = ResolvedVariant(tag=tag, runtime_class=f"x-{engine}", enforces_schema=None)
    return Host(
        host_id=host_id, kind="local", priority=0, capabilities=frozenset(),
        client=httpx.AsyncClient(), workers=[Worker(worker_id=f"{host_id}/w0")],
        variants={BIG: (variant,)}, literal_variants={}, engine=engine,
        state=HostState.READY, resident=frozenset({tag}),
    )


def test_a_request_on_one_engines_own_api_never_reaches_a_host_running_the_other():
    """Ollama serves `/api/chat`; vLLM does not. Calling such a host eligible would hand the
    request to a machine that answers 404 to it — a failure the pool caused itself."""
    dispatcher = Dispatcher([
        a_host("laptop", "ollama", "gemma4:26b"),
        a_host("rented", "vllm", "nvidia/Gemma-4-26B-A4B-NVFP4"),
    ])

    native = Need(model=BIG, engines=frozenset({"ollama"}))
    assert [h.host_id for h, _ in dispatcher.eligible(native)] == ["laptop"]


def test_a_request_on_the_shared_api_reaches_either():
    """Both engines serve `/v1/chat/completions`, which is what lets one pool hold both."""
    dispatcher = Dispatcher([
        a_host("laptop", "ollama", "gemma4:26b"),
        a_host("rented", "vllm", "nvidia/Gemma-4-26B-A4B-NVFP4"),
    ])

    shared = Need(model=BIG, engines=frozenset({"ollama", "vllm"}))
    assert {h.host_id for h, _ in dispatcher.eligible(shared)} == {"laptop", "rented"}


def test_naming_no_engine_means_any_which_is_what_a_single_engine_pool_meant():
    dispatcher = Dispatcher([a_host("laptop", "ollama", "gemma4:26b")])
    assert len(dispatcher.eligible(Need(model=BIG))) == 1


def test_each_host_is_served_the_tag_its_own_engine_knows():
    """Routing picks the host, and the host's own build is what the request is rewritten to."""
    dispatcher = Dispatcher([
        a_host("laptop", "ollama", "gemma4:26b"),
        a_host("rented", "vllm", "nvidia/Gemma-4-26B-A4B-NVFP4"),
    ])
    by_host = {h.host_id: v.tag for h, v in dispatcher.eligible(Need(model=BIG))}
    assert by_host == {"laptop": "gemma4:26b", "rented": "nvidia/Gemma-4-26B-A4B-NVFP4"}


# --- the router reads a request by the path it arrived on, not by the host ---


def test_the_parser_is_chosen_by_path(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        state = RouterState(config(), database)
        assert state.parser_for("/api/chat").name == "ollama"
        assert state.parser_for("/v1/chat/completions") is not None
        assert state.parser_for("/nonsense") is None
        # Between them the hosts serve both surfaces.
        assert {"/api/chat", "/v1/chat/completions"} <= state.paths()
    finally:
        database.close()


def test_a_hosts_engine_survives_being_published_and_read_back(tmp_path):
    """The router learns what each machine runs from the table the supervisor publishes."""
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        table = HostTable(database)
        table.publish(HostRow(
            host_id="rented-1", kind="rented-on-demand", transport_type="http", priority=20,
            dial_url="http://127.0.0.1:8000", state="ready", workers=2,
            capabilities=("cuda",), variants={}, resident=frozenset(), engine="vllm",
        ))
        (row,) = table.all()
        assert row.engine == "vllm"
    finally:
        database.close()


def test_a_table_written_before_engines_could_differ_reads_as_the_first_engine(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    try:
        table = HostTable(database)
        table.publish(HostRow(
            host_id="laptop", kind="local", transport_type="http", priority=0,
            dial_url="http://127.0.0.1:11434", state="ready", workers=3,
            capabilities=(), variants={}, resident=frozenset(),
        ))
        (row,) = table.all()
        assert row.engine == "ollama"
    finally:
        database.close()
