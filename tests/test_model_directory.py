"""The model directory, end to end (D100, D101).

The owner: "expose a cached version both from Ollama and Hugging Face of all available models
(for vLLM also the optional config)", to apps too, and add models to the pool from it.

A real supervisor, control API and router over one pool file and one database; Ollama's site and
the model hub are stand-ins serving what the real ones served, recorded. What is proved:

- a refresh reads Ollama's library and looks the pool's models up on the hub, and the cache is
  what both processes serve — the router without making a single request of its own;
- a refresh that fails keeps the last good copy, and says so;
- a lookup is cached, asked again only when stale or when the operator asks;
- a model is added from the directory with a build per engine, through the file's own rules;
- the named vLLM options reach the machine's start as names, and only known ones do.

Nothing here rents, spends, or reaches the internet.
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import httpx
import pytest
import yaml
from fakes.fake_ollama_site import FakeOllamaSite
from fakes.fake_vllm import FakeHub
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.db import Database
from gpm_server.router.app import create_app
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY, APP_KEY = "gpmx_directory_admin", "gpma_directory_app"
BIG, SMALL = "gemma4:26b", "gemma4:e4b"
RECORDED = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "directory" / "hub_search_gemma-4-26b.json").read_text()
)
NVFP4 = "nvidia/Gemma-4-26B-A4B-NVFP4"

POOL_YAML = textwrap.dedent(
    f"""
    pool:
      name: directory
      model_set: ["{SMALL}", "{BIG}"]
      probe_interval_s: 3600
      models_per_host: declared

    auth:
      admin_keys: ["{ADMIN_KEY}"]
      app_keys: ["{APP_KEY}"]

    # Where the directory is read from, and how gently.
    directory:
      hub_requests_per_minute: 6000

    hosts:
      - id: laptop
        kind: local
        models: ["{SMALL}", "{BIG}"]   # what this machine holds
        transport: {{ type: http, base_url: "http://127.0.0.1:1" }}

    rented:
      provider: fake
      engine: vllm
      models: []
      images:
        - {{ image: "vastai/vllm:v0.29.0-cuda-12.9", min_driver: "550" }}
      offer_policy: {{ min_disk_gb: 10, max_all_in_hourly: 0.60 }}
    """
).strip()


class Pool:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.loop = BackgroundLoop()
        self.site = FakeOllamaSite()
        self.hub = FakeHub(
            {NVFP4: {"model-00001.safetensors": b"w" * 1000, "config.json": b"{}"}},
            search={"gemma-4-26b": RECORDED},
        )
        self.site_server = ServerHandle(self.site.app, self.loop)
        self.hub_server = ServerHandle(self.hub.app, self.loop)
        monkeypatch.setenv("GPM_OLLAMA_SITE", self.site_server.base_url)
        monkeypatch.setenv("HF_ENDPOINT", self.hub_server.base_url)
        self.path = tmp_path / "pool.yaml"
        self.path.write_text(POOL_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
        self.database = Database(tmp_path / "gpm.sqlite3")
        self.supervisor = Supervisor(load_config(self.path), self.database, config_path=str(self.path))
        self.control = ServerHandle(create_control_app(self.supervisor, self.supervisor.config), self.loop)
        self.router = ServerHandle(create_app(load_config(self.path), self.database), self.loop)

    def admin(self) -> httpx.Client:
        return httpx.Client(base_url=self.control.base_url, timeout=60,
                            headers={"Authorization": f"Bearer {ADMIN_KEY}"})

    def app(self) -> httpx.Client:
        return httpx.Client(base_url=self.router.base_url, timeout=30,
                            headers={"Authorization": f"Bearer {APP_KEY}"})

    def refresh(self) -> dict:
        with self.admin() as http:
            assert http.post("/pool/directory/refresh").status_code == 202
            deadline = time.monotonic() + 60
            while True:
                found = http.get("/pool/directory").json()
                if not found["refreshing"]:
                    return found
                assert time.monotonic() < deadline, "the refresh did not finish"
                time.sleep(0.1)

    def close(self) -> None:
        for server in (self.router, self.control, self.hub_server, self.site_server):
            server.stop()
        self.loop.run(self.supervisor.directory.aclose())
        self.loop.stop()
        self.database.close()


@pytest.fixture
def pool(tmp_path, monkeypatch):
    made = Pool(tmp_path, monkeypatch)
    try:
        yield made
    finally:
        made.close()


def model(found: dict, name: str) -> dict:
    return next(m for m in found["models"] if m["name"] == name)


def tag(found: dict, name: str) -> dict:
    base = name.split(":")[0]
    return next(t for t in model(found, base)["tags"] if t["name"] == name)


# --- refreshing, and what is served ---


def test_a_refresh_reads_ollamas_library_and_the_pools_models_on_the_hub(pool):
    found = pool.refresh()

    assert {m["name"] for m in found["models"]} == {"gemma4", "nomic-embed-text"}, (
        "llama3's tags page was not recorded, so it is skipped; glm-5.1 is only on Ollama's cloud"
    )
    assert found["refreshed"]["ollama"]["ok"] == 1 and "could not be read" in found["refreshed"]["ollama"]["detail"]
    big = tag(found, BIG)
    assert big["in_pool"] and big["size_gb"] == 19.0 and big["context"] == "256K"
    assert not tag(found, "gemma4:31b")["in_pool"]

    builds = big["vllm"]["builds"]
    assert builds[0]["repo"] == "google/gemma-4-26B-A4B-it"
    nvfp4 = next(b for b in builds if b["repo"] == NVFP4)
    assert nvfp4["precision"] == "NVFP4" and nvfp4["full_speed_on"] == "Blackwell"
    assert nvfp4["options"] == ["reasoning", "tool_calling"], "the optional config, per build"
    assert nvfp4["size_estimated"] is False, "sized from its own file listing"


def test_only_the_pools_models_are_looked_up_on_the_hub_unless_asked_for_all(pool):
    pool.refresh()
    searched = {r.removeprefix("search ") for r in pool.hub.requests if r.startswith("search ")}
    assert searched == {"gemma4-26b", "gemma-4-26b", "gemma4-e4b", "gemma-4-e4b"}


def test_the_hub_is_asked_with_no_credential(pool, monkeypatch):
    """What an anonymous request can see is what a rented host can fetch."""
    monkeypatch.setenv("HF_TOKEN", "hf_should_never_be_sent")
    pool.refresh()
    assert pool.hub.headers_seen
    assert all("authorization" not in h for h in pool.hub.headers_seen)


def test_apps_read_the_same_directory_from_the_router_without_it_asking_anyone(pool):
    pool.refresh()
    pool.site.down = pool.hub.down = True
    before = len(pool.site.requests) + len(pool.hub.requests)
    with pool.app() as http:
        answer = http.get("/pool/directory", params={"q": "gemma"})
    assert answer.status_code == 200
    found = answer.json()
    assert [m["name"] for m in found["models"]] == ["gemma4"]
    assert tag(found, BIG)["vllm"]["builds"], "the builds and their options reach apps too"
    assert len(pool.site.requests) + len(pool.hub.requests) == before


def test_the_sdk_reads_it(pool):
    from gpm_client import PoolClient

    pool.refresh()
    with PoolClient(base_url=pool.router.base_url, api_key=APP_KEY) as client:
        found = client.directory("nomic")
    assert [m["name"] for m in found["models"]] == ["nomic-embed-text"]


def test_an_app_key_cannot_refresh_or_change_anything(pool):
    with httpx.Client(base_url=pool.control.base_url, timeout=30,
                      headers={"Authorization": f"Bearer {APP_KEY}"}) as http:
        assert http.post("/pool/directory/refresh").status_code in (401, 403)
        assert http.post("/pool/config/models", json={"add": [{"name": "x"}]}).status_code in (401, 403)
    with pool.app() as http:
        assert http.post("/pool/directory/refresh").status_code in (401, 403, 404, 405)


def test_a_failed_refresh_keeps_the_last_copy_and_says_so(pool):
    pool.refresh()
    pool.site.down = True
    found = pool.refresh()
    assert {m["name"] for m in found["models"]} == {"gemma4", "nomic-embed-text"}
    run = found["refreshed"]["ollama"]
    assert run["ok"] == 0 and "the last copy is kept" in run["detail"]
    assert run["last_success"] is not None


def test_nothing_is_refreshed_on_its_own_unless_the_file_says_how_often(pool):
    """The pool makes no outbound request nobody asked for."""
    assert pool.supervisor.config.directory.refresh_hours == 0
    assert not pool.supervisor.directory.due()


# --- looking one model up ---


def test_a_lookup_is_cached_and_asked_again_only_when_asked(pool):
    with pool.admin() as http:
        first = http.get("/pool/builds", params={"model": BIG}).json()
        asked = len(pool.hub.requests)
        second = http.get("/pool/builds", params={"model": BIG}).json()
        assert len(pool.hub.requests) == asked and second["cached"] is True
        http.get("/pool/builds", params={"model": BIG, "fresh": "true"})
    assert len(pool.hub.requests) > asked
    assert first["original"] == "google/gemma-4-26B-A4B-it"


def test_a_model_outside_the_pool_can_be_looked_up_and_shows_in_the_directory(pool):
    with pool.admin() as http:
        http.get("/pool/builds", params={"model": "gemma4:31b", "search": "gemma-4-26b"})
        found = http.get("/pool/directory").json()
    assert any(e["name"] == "gemma4:31b" for e in found["looked_up_elsewhere"])


def test_a_hub_that_is_down_is_said_plainly(pool):
    pool.hub.down = True
    with pool.admin() as http:
        answer = http.get("/pool/builds", params={"model": BIG, "fresh": "true"})
    assert answer.status_code == 502 and answer.json()["error"] == "hub_unavailable"


@pytest.mark.parametrize("params", [{"model": "gemma4:26b", "search": "a&b"}, {"model": "bad name!"}])
def test_only_names_reach_the_hubs_query(pool, params):
    with pool.admin() as http:
        assert http.get("/pool/builds", params=params).status_code == 400


def test_an_engine_whose_builds_are_its_own_librarys_names_is_not_looked_up(pool):
    with pool.admin() as http:
        answer = http.get("/pool/builds", params={"model": BIG, "engine": "ollama"})
    assert answer.status_code == 400 and answer.json()["error"] == "no_hub"


# --- adding models from it ---


def test_a_model_is_added_with_a_build_for_each_engine_and_rented_for(pool):
    before = pool.path.read_text()
    with pool.admin() as http:
        answer = http.post("/pool/config/models", json={
            "add": [{"name": "gemma4:31b", "builds": {"ollama": "gemma4:31b", "vllm": NVFP4}, "rent_for": True}],
            "engine_options": ["tool_calling"],
        })
    assert answer.status_code == 200, answer.text
    written = yaml.safe_load(pool.path.read_text())
    assert written["pool"]["model_set"] == [SMALL, BIG, "gemma4:31b"]
    assert written["catalog"]["gemma4:31b"]["variants"] == [
        {"tag": "gemma4:31b", "engine": "ollama"}, {"tag": NVFP4, "engine": "vllm"},
    ]
    assert written["rented"]["models"] == ["gemma4:31b"]
    assert written["rented"]["engine_options"] == ["tool_calling"]
    for comment in ("# Where the directory is read from", "# what this machine holds"):
        assert comment in before and comment in pool.path.read_text()
    assert "gemma4:31b" in pool.supervisor.config.pool.model_set, "the running pool took it"


def test_a_model_can_be_added_for_workloads_only(pool):
    """Outside the shared set: a workload may be created for it, and no shared host fetches it (D115)."""
    with pool.admin() as http:
        answer = http.post("/pool/config/models", json={
            "add": [{"name": "gemma4:31b", "builds": {"vllm": NVFP4}, "workloads_only": True}],
        })
    assert answer.status_code == 200, answer.text
    written = yaml.safe_load(pool.path.read_text())
    assert "gemma4:31b" not in written["pool"]["model_set"]
    assert written["catalog"]["gemma4:31b"]["workloads_only"] is True
    assert "gemma4:31b" not in pool.supervisor.config.pool.model_set
    with pool.admin() as http:
        again = http.post("/pool/config/models", json={
            "add": [{"name": SMALL, "builds": {"vllm": NVFP4}, "workloads_only": True}],
        })
    assert again.status_code == 400 and "already in the shared set" in again.json()["detail"]


def test_a_model_nobody_could_hold_is_refused_in_words_and_nothing_is_written(pool):
    before = pool.path.read_text()
    with pool.admin() as http:
        answer = http.post("/pool/config/models", json={
            "add": [{"name": "gemma4:31b", "builds": {"vllm": NVFP4}, "rent_for": False}],
        })
    assert answer.status_code == 400 and answer.json()["error"] == "invalid_config"
    assert "gemma4:31b" in answer.json()["detail"]
    assert pool.path.read_text() == before


@pytest.mark.parametrize("item", [
    {"name": "gemma4:31b", "builds": {"tgi": "x/y"}},
    {"name": "gemma4:31b", "builds": {"vllm": "x/y; rm -rf /"}},
    {"name": "../../etc", "builds": {}},
])
def test_what_is_not_a_known_engine_or_a_name_is_refused(pool, item):
    before = pool.path.read_text()
    with pool.admin() as http:
        assert http.post("/pool/config/models", json={"add": [item]}).status_code == 400
    assert pool.path.read_text() == before


# --- the named options ---


def test_an_option_the_engine_does_not_offer_is_refused(pool):
    with pool.admin() as http:
        answer = http.post("/pool/config/models", json={
            "add": [{"name": BIG, "builds": {"vllm": NVFP4}, "rent_for": True}],
            "engine_options": ["trust_remote_code"],
        })
    assert answer.status_code == 400 and "does not offer" in answer.json()["detail"]


def test_the_options_reach_the_machines_start_as_names(pool):
    with pool.admin() as http:
        assert http.post("/pool/config/models", json={
            "add": [{"name": BIG, "builds": {"vllm": NVFP4}, "rent_for": True}],
            "engine_options": ["tool_calling", "reasoning"],
        }).status_code == 200
    start = pool.supervisor.fleet.engine_start_command()
    assert "vllm-start" in start and "--option tool_calling --option reasoning" in start


def test_options_beside_a_start_command_of_ones_own_are_refused(pool):
    """The options belong to the engine's own start; a start command replaces it, and they
    would be silently ignored."""
    with pool.admin() as http:
        answer = http.patch("/pool/config/engine", json={
            "rented_engine": "vllm", "placement": "declared", "rented_models": [BIG],
            "builds": {BIG: NVFP4}, "engine_start": "vllm serve /models --port 8000",
            "engine_options": ["tool_calling"],
        })
    assert answer.status_code == 400 and "replaces it" in answer.json()["detail"]


def test_switching_the_rented_engine_away_drops_options_it_does_not_offer(pool):
    with pool.admin() as http:
        assert http.post("/pool/config/models", json={
            "add": [{"name": BIG, "builds": {"vllm": NVFP4}, "rent_for": True}],
            "engine_options": ["tool_calling"],
        }).status_code == 200
        answer = http.patch("/pool/config/engine", json={
            "rented_engine": "ollama", "placement": "all", "images": [],
            "image": "vastai/ollama:0.34.2", "engine_start": "nohup ollama serve &",
        })
    assert answer.status_code == 200, answer.text
    assert yaml.safe_load(pool.path.read_text())["rented"]["engine_options"] == []


def test_the_status_offers_each_engines_options_and_says_which_are_on(pool):
    with pool.admin() as http:
        engine = http.get("/pool/status").json()["engine"]
    assert engine["offers"]["vllm"]["builds_on_hub"] is True
    assert set(engine["offers"]["vllm"]["options"]) == {"tool_calling", "reasoning"}
    assert "gemma4" in engine["offers"]["vllm"]["options"]["tool_calling"]["families"]
    assert engine["offers"]["ollama"] == {"builds_on_hub": False, "splits_across_cards": False, "options": {}}
    assert engine["engine_options"] == []


def test_the_hub_is_asked_no_faster_than_the_file_says(pool):
    """A refresh's sweep and the operator's clicks share one pace."""
    pool.supervisor.config.directory.hub_requests_per_minute = 600  # one every 0.1 s
    started = time.monotonic()

    async def three():
        for _ in range(3):
            await pool.supervisor.directory._paced()

    pool.loop.run(three())
    assert time.monotonic() - started >= 0.2
