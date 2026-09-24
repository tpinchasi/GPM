"""Switching rented hosts' engine and placement from the console, in one write (D98).

The owner asked for it on the Rented capacity screen. It is one change because it cannot be
several: an engine switched without its builds, images or start command is a configuration that
rents machines which can never serve, and the load-time checks refuse every half-way state. The
file stays the source of truth, edited in place with every comment kept (D51).

Against a file shaped like a real pool's: comments, Apple-silicon builds, a laptop host, and an
`engine_start` written for Ollama. Nothing here rents or spends.
"""

import textwrap

import httpx
import pytest
import yaml
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.db import Database
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_editor_admin"
BIG, SMALL, EMBED = "gemma4:26b", "gemma4:e4b", "nomic-embed-text:latest"
BIG_REPO, SMALL_REPO, EMBED_REPO = (
    "nvidia/Gemma-4-26B-A4B-NVFP4", "google/gemma-4-E4B-it", "nomic-ai/nomic-embed-text-v1.5",
)
VLLM_IMAGES = [
    {"image": "vastai/vllm:v0.29.0-cuda-13.0", "min_driver": "580", "note": "newest"},
    {"image": "vastai/vllm:v0.29.0-cuda-12.9", "min_driver": "550"},
]
OLLAMA_START = "nohup ollama serve >/var/log/ollama.log 2>&1 &"

POOL_YAML = textwrap.dedent(
    f"""
    pool:
      name: editor
      model_set: ["{EMBED}", "{SMALL}", "{BIG}"]
      probe_interval_s: 3600

    auth:
      admin_keys: ["{ADMIN_KEY}"]
      app_keys: ["gpma_editor_app"]

    engine: ollama

    # Logical names. The laptop serves the MLX build, a CUDA host the standard one.
    catalog:
      {SMALL}:
        variants:
          - {{ tag: "{SMALL}-mlx", requires: [apple-silicon], runtime_class: apple-mlx }}
          - {{ tag: "{SMALL}" }}
      {BIG}:
        variants:
          - {{ tag: "{BIG}-mlx", requires: [apple-silicon], runtime_class: apple-mlx }}
          - {{ tag: "{BIG}" }}

    hosts:
      - id: laptop
        kind: local
        capabilities: [apple-silicon]
        workers: 3
        residency: on_demand   # this laptop is used for other work
        transport: {{ type: http, base_url: "http://127.0.0.1:1" }}

    limits: {{ max_rented_hosts: 1, max_hourly_burn: 2.00 }}

    rented:
      provider: fake
      # The provider's own build of the same engine.
      image: vastai/ollama:0.34.2
      # Vast's ssh launch mode replaces the image's entrypoint, so the engine is started here.
      engine_start: "{OLLAMA_START}"
      capabilities: [cuda]
      bidding: {{ bid_ceiling: 0.60 }}
    """
).strip()


@pytest.fixture
def pool(tmp_path):
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(POOL_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        yield supervisor, server.base_url, config_path
    finally:
        server.stop()
        loop.stop()
        database.close()


def patch(url, body):
    with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {ADMIN_KEY}"}, timeout=30) as http:
        return http.patch("/pool/config/engine", json=body)


VLLM_ONE_PER_HOST = {
    "rented_engine": "vllm",
    "placement": "declared",
    "rented_models": [BIG],
    "builds": {BIG: BIG_REPO},
    "images": VLLM_IMAGES,
    "engine_start": "",
}


# --- to vLLM, a model to a host ---


def test_switching_rented_hosts_to_vllm_one_model_each(pool):
    supervisor, url, path = pool
    answer = patch(url, VLLM_ONE_PER_HOST)
    assert answer.status_code == 200, answer.text

    written = yaml.safe_load(path.read_text())
    assert written["pool"]["models_per_host"] == "declared"
    assert written["rented"]["engine"] == "vllm"
    assert written["rented"]["models"] == [BIG]
    assert written["rented"]["engine_start"] is None, "vLLM's own start"
    assert [i["image"] for i in written["rented"]["images"]] == [i["image"] for i in VLLM_IMAGES]

    # And the running supervisor took it: the change is live, not only on disk.
    assert supervisor.config.rented_engine() == "vllm"
    assert supervisor.config.pool.models_per_host == "declared"


def test_the_laptop_keeps_every_model_it_held(pool):
    """Switching to a model per host would otherwise drop every configured host to the first
    model it can serve — silently, and the laptop would stop serving two of its three."""
    supervisor, url, path = pool
    assert patch(url, VLLM_ONE_PER_HOST).status_code == 200
    (laptop,) = yaml.safe_load(path.read_text())["hosts"]
    assert laptop["models"] == [EMBED, SMALL, BIG]
    assert supervisor.config.models_held_by(supervisor.config.hosts[0]) == [EMBED, SMALL, BIG]


def test_existing_builds_are_marked_as_the_pools_own_engines(pool):
    """A build written before the pool ran two engines names none, so any engine may be offered
    it — and a vLLM host would try to fetch `gemma4:e4b` from a model hub, where no such
    repository exists."""
    _, url, path = pool
    assert patch(url, VLLM_ONE_PER_HOST).status_code == 200
    catalog = yaml.safe_load(path.read_text())["catalog"]
    assert [(v["tag"], v["engine"]) for v in catalog[BIG]["variants"]] == [
        (f"{BIG}-mlx", "ollama"), (BIG, "ollama"), (BIG_REPO, "vllm"),
    ]
    assert all(v["engine"] == "ollama" for v in catalog[SMALL]["variants"])
    # The MLX build keeps what it required and how it is labelled.
    mlx = catalog[BIG]["variants"][0]
    assert mlx["requires"] == ["apple-silicon"] and mlx["runtime_class"] == "apple-mlx"


def test_every_comment_in_the_file_survives(pool):
    _, url, path = pool
    before = path.read_text()
    assert patch(url, VLLM_ONE_PER_HOST).status_code == 200
    after = path.read_text()
    comments = [line.strip() for line in before.splitlines() if line.strip().startswith("#")]
    comments += ["# this laptop is used for other work"]
    for comment in comments:
        assert comment in after, comment


def test_a_model_to_rent_for_with_no_build_for_the_engine_is_refused(pool):
    """Otherwise a host bought for it is prepared for nothing and called ready holding nothing."""
    supervisor, url, path = pool
    before = path.read_text()
    answer = patch(url, VLLM_ONE_PER_HOST | {"rented_models": [BIG, SMALL]})
    assert answer.status_code == 400
    assert "no build of" in answer.json()["detail"] and SMALL in answer.json()["detail"]
    assert path.read_text() == before, "nothing written"
    assert supervisor.config.rented_engine() == "ollama", "nothing changed"


def test_keeping_the_ollama_start_command_with_vllm_is_refused(pool):
    """It would start the wrong server inside the right image."""
    _, url, _ = pool
    answer = patch(url, VLLM_ONE_PER_HOST | {"engine_start": OLLAMA_START})
    assert answer.status_code == 400
    assert "starts 'ollama'" in answer.json()["detail"]


# --- to vLLM, every model on each host behind the router ---


def test_every_model_on_each_host_behind_the_router(pool):
    supervisor, url, path = pool
    answer = patch(url, {
        "rented_engine": "vllm", "placement": "all_proxy",
        "builds": {BIG: BIG_REPO, SMALL: SMALL_REPO, EMBED: EMBED_REPO},
        "images": VLLM_IMAGES, "engine_start": "",
    })
    assert answer.status_code == 200, answer.text
    written = yaml.safe_load(path.read_text())
    assert written["pool"]["models_per_host"] == "all"
    assert written["rented"]["engine_proxy"] is True
    assert written["rented"]["models"] is None
    # A model with no catalog entry gains one, keeping the name the laptop serves it under.
    assert [(v["tag"], v["engine"]) for v in written["catalog"][EMBED]["variants"]] == [
        (EMBED, "ollama"), (EMBED_REPO, "vllm"),
    ]
    assert supervisor.config.rented.engine_proxy is True


def test_vllm_holding_every_model_without_the_router_is_refused(pool):
    _, url, _ = pool
    answer = patch(url, {
        "rented_engine": "vllm", "placement": "all",
        "builds": {BIG: BIG_REPO, SMALL: SMALL_REPO, EMBED: EMBED_REPO},
        "images": VLLM_IMAGES, "engine_start": "",
    })
    assert answer.status_code == 400
    assert "one model per process" in answer.json()["detail"]


# --- and back ---


def test_switching_back_to_ollama_restores_the_whole_set_on_every_host(pool):
    supervisor, url, path = pool
    assert patch(url, VLLM_ONE_PER_HOST).status_code == 200
    answer = patch(url, {
        "rented_engine": "ollama", "placement": "all",
        "images": [], "image": "vastai/ollama:0.34.2", "engine_start": OLLAMA_START,
    })
    assert answer.status_code == 200, answer.text

    written = yaml.safe_load(path.read_text())
    assert written["pool"]["models_per_host"] == "all"
    assert written["rented"]["engine"] == "ollama"
    assert written["rented"]["models"] is None
    assert written["rented"]["images"] == []
    assert written["rented"]["engine_start"] == OLLAMA_START
    assert written["hosts"][0]["models"] is None, "every host holds the whole set again"
    assert supervisor.config.rented_engine() == "ollama"


# --- what is refused before anything is written ---


@pytest.mark.parametrize("body", [
    {"rented_engine": "tgi", "placement": "all"},
    {"rented_engine": "vllm", "placement": "some"},
])
def test_an_engine_or_placement_the_pool_does_not_know_is_refused(pool, body):
    _, url, path = pool
    before = path.read_text()
    assert patch(url, body).status_code == 400
    assert path.read_text() == before


def test_the_status_carries_what_the_editor_starts_from(pool):
    _, url, _ = pool
    with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {ADMIN_KEY}"}, timeout=30) as http:
        status = http.get("/pool/status").json()
    engine = status["engine"]
    assert engine["rented"] == "ollama" and engine["engine_start"] == OLLAMA_START
    assert engine["image"] == "vastai/ollama:0.34.2" and engine["rented_models"] is None
    assert set(engine["available"]) >= {"ollama", "vllm"}
    assert all("engine" in variant for variant in status["catalog"][BIG])
