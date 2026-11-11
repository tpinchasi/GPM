"""Rented hosts start from the engine's newest release, kept current (D177).

Found by the owner renting gemma4:31b: the pool's shipped Ollama image was 0.34.2, older than
the model, and an image named by hand started nothing on Vast. Now a pool that names no image
rents from the newest stable release of the engine's official image, found in the registry; a new
release is on trial until a host from it serves, and dropped after two hosts from it never answer.
Every test here runs without the network: the registry is a fake.
"""

from __future__ import annotations

import asyncio
import json
import textwrap

from gpm_server import engine_images
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.engine_images import ImageTracker, Track, newest
from gpm_server.ledger import EventLog

OLLAMA, VLLM, VLLM_OLD = engine_images.TRACKS["ollama"][0], *engine_images.TRACKS["vllm"]


def tags(*names, arch="amd64"):
    return [{"name": n, "digest": f"sha256:{i:064x}", "images": [{"architecture": arch}]} for i, n in enumerate(names, 1)]


def registry(listing: dict):
    calls = []

    async def fetch(url: str) -> bytes:
        calls.append(url)
        repo = url.split("/repositories/")[1].split("/tags")[0]
        return json.dumps({"results": listing[repo]}).encode()

    return fetch, calls


def tracker(tmp_path, fetch):
    db = Database(tmp_path / "gpm.sqlite3")
    return ImageTracker(db, EventLog(db), fetch), db


def test_only_release_tags_are_taken_newest_by_number():
    listed = tags("latest", "rocm", "0.40.15-rc2", "0.40.15-rocm", "0.40.9", "0.40.15", "0.40.14", "nightly")
    assert newest(listed, OLLAMA)[0] == "0.40.15", "never a floating tag, a candidate or another accelerator's build"
    assert newest(listed, OLLAMA, {"0.40.15"})[0] == "0.40.14", "a dropped release is not taken again"
    vllm = tags("nightly", "v0.33.0", "v0.33.0-x86_64", "v0.33.0-efa", "v0.32.0", "v0.32.0-cu129", "cu134-nightly")
    assert newest(vllm, VLLM)[0] == "v0.33.0"
    assert newest(vllm, VLLM_OLD)[0] == "v0.32.0-cu129", "the older driver's line keeps its own newest"
    assert newest(tags("0.41.0", arch="arm64") + tags("0.40.15"), OLLAMA)[0] == "0.40.15", "x86-64 builds only"
    assert newest(tags("0.9.0", "0.10.0"), OLLAMA)[0] == "0.10.0", "by number, never as text"


def test_a_new_release_is_taken_on_trial_proven_by_a_host_that_serves(tmp_path):
    fetch, calls = registry({"ollama/ollama": tags("0.40.14", "0.40.15")})
    images, db = tracker(tmp_path, fetch)
    try:
        assert images.due(["ollama"])
        asyncio.run(images.refresh(["ollama"]))
        assert images.images("ollama", []) == [("ollama/ollama:0.40.15", None, "Ollama's own image — new, on trial")]
        assert not images.due(["ollama"]) and len(calls) == 1, "asked again only after REFRESH_S"
        images.outcome("ollama/ollama:0.40.15", served=True)
        assert images.view("ollama")[0]["on_trial"] is False
        # Kept across a restart: the database holds it.
        again = ImageTracker(db, EventLog(db), fetch)
        assert again.images("ollama", [])[0][0] == "ollama/ollama:0.40.15" and not again.due(["ollama"])
    finally:
        db.close()


def test_a_release_whose_hosts_never_answer_is_dropped_for_the_one_before(tmp_path):
    listing = {"ollama/ollama": tags("0.40.14")}
    fetch, _ = registry(listing)
    images, db = tracker(tmp_path, fetch)
    try:
        asyncio.run(images.refresh(["ollama"]))
        images.outcome("ollama/ollama:0.40.14", served=True)
        listing["ollama/ollama"] = tags("0.40.14", "0.40.15")
        images.state["ollama"]["checked_at"] = 0
        asyncio.run(images.refresh(["ollama"]))
        assert images.images("ollama", [])[0][0] == "ollama/ollama:0.40.15"
        images.outcome("ollama/ollama:0.40.15", served=False)
        assert images.images("ollama", [])[0][0] == "ollama/ollama:0.40.15", "one bad machine does not drop a release"
        images.outcome("ollama/ollama:0.40.15", served=False)
        assert images.images("ollama", [])[0][0] == "ollama/ollama:0.40.14"
        kinds = [e["kind"] for e in EventLog(db).recent(10)]
        assert "engine_image_dropped" in kinds
        images.state["ollama"]["checked_at"] = 0
        asyncio.run(images.refresh(["ollama"]))
        assert images.images("ollama", [])[0][0] == "ollama/ollama:0.40.14", "never taken again on its own"
        images.outcome("ollama/ollama:0.40.13", served=False)  # not the one on trial: changes nothing
        assert images.view("ollama")[0]["dropped"] == ["0.40.15"]
    finally:
        db.close()


def test_a_registry_that_cannot_be_asked_keeps_what_is_in_use(tmp_path):
    async def away(url):
        raise ConnectionError("no network")

    images, db = tracker(tmp_path, away)
    try:
        asyncio.run(images.refresh(["ollama"]))
        shipped = [("ollama/ollama:0.40.15", None, "shipped")]
        assert images.images("ollama", shipped) == shipped and "could not be asked" in images.problem
        assert not images.due(["ollama"]), "asked again after RETRY_S, not every pass"
    finally:
        db.close()


def test_an_image_named_by_the_operator_pins_and_latest_tracks():
    base = textwrap.dedent("""
        pool: { model_set: [m], models_per_host: declared }
        auth: { admin_keys: ["gpmx_images_admin"], app_keys: ["gpma_images_app"] }
        engine: ollama
        catalog: { m: { variants: [{ tag: "m:1" }] } }
        hosts: [{ id: h, kind: local, transport: { type: http, base_url: "http://127.0.0.1:1" }, models: [m] }]
        rented:
          providers: { market: { type: fake } }
          engine: ollama
    """)
    import yaml

    def rented(extra: str):
        return PoolConfig.model_validate(yaml.safe_load(base + extra)).rented

    assert rented("").tracks_latest and rented("  image: latest\n").tracks_latest
    assert not rented('  image: "ollama/ollama:0.40.13"\n').tracks_latest
    assert not rented('  images: [{ image: "ollama/ollama:0.40.13", min_driver: "550" }]\n').tracks_latest


def test_a_line_with_a_driver_floor_is_matched_to_the_machine():
    class Offer:
        driver_version = "550.120"

    from gpm_server.config import EngineImage
    from gpm_server.strategies import image_for

    lines = [EngineImage(image="vllm/vllm-openai:v0.33.0", min_driver="580"),
             EngineImage(image="vllm/vllm-openai:v0.32.0-cu129", min_driver="550")]
    assert image_for(Offer(), lines).image == "vllm/vllm-openai:v0.32.0-cu129"
    assert isinstance(VLLM, Track) and VLLM.min_driver == "580" and OLLAMA.min_driver is None


def test_ollama_is_started_by_the_pool_on_any_image():
    from gpm_server.engines import get_engine

    start = get_engine("ollama").default_start_command(port=11434, models_dir="/m", agent_archive="/a", proxy=False)
    assert "ollama serve" in start and start.rstrip().endswith("&") and "127.0.0.1:11434" in start
