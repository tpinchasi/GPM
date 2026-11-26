"""The sizes of the pool's own builds (D207): read where they are published, and measured where
they run.

The owner: "we need this info for any new model we use and has this info public" — and, for one
that does not, "find out in the first run ... store the mem size in our db with model hash, so it
will be used next time, but if new tag comes, it will recalc".

What is proved, against stand-ins for Ollama's site and the hub (no network):

- a build in the pool's set whose size the catalog does not state has its one page read — never
  the whole library — cached, read again only once stale, and not at all when the file states
  it, when the operator turns it off, for a hosted build or for a name the library cannot list;
- a page that cannot be read is not asked again for hours;
- what a host's engine says a build took, on disk and loaded, is kept by digest; written only
  when it changes; used as the build's size and memory next time; and set aside when the
  library shows the tag pushed again, until a host holds the new digest;
- a measured memory figure replaces the estimate in what a machine must have.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fakes.fake_ollama_site import FakeOllamaSite
from fakes.fake_vllm import FakeHub
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server import directory as directory_module
from gpm_server.config import DirectoryConfig, Variant
from gpm_server.db import Database
from gpm_server.directory import (
    HUB_SIZE,
    OLLAMA_TAGS,
    Directory,
    build_memory_gb,
    build_sizes_gb,
    record_measurements,
)
from gpm_server.sizing import needs_for

BIG = "gemma4:31b"  # 20.0 GB on the recorded page, digest 6316f0629137
REPO = "acme/Some-Model-FP8"
DIGEST = "6316f0629137" + "a" * 52
NEW_DIGEST = "0123456789ab" + "b" * 52


def config(models: dict[str, list[Variant]], **directory) -> SimpleNamespace:
    return SimpleNamespace(
        directory=DirectoryConfig(**directory),
        pool=SimpleNamespace(model_set=list(models)),
        catalog={name: SimpleNamespace(variants=variants) for name, variants in models.items()},
        rented=None,
    )


class Sites:
    def __init__(self, tmp_path, monkeypatch):
        self.loop = BackgroundLoop()
        self.site = FakeOllamaSite()
        self.hub = FakeHub({REPO: {"model-00001.safetensors": b"w" * 3_000_000, "config.json": b"{}"}})
        self.site_server = ServerHandle(self.site.app, self.loop)
        self.hub_server = ServerHandle(self.hub.app, self.loop)
        monkeypatch.setenv("GPM_OLLAMA_SITE", self.site_server.base_url)
        monkeypatch.setenv("HF_ENDPOINT", self.hub_server.base_url)
        self.db = Database(tmp_path / "gpm.sqlite3")
        self.config = config({BIG: [Variant(tag=BIG, engine="ollama")]})
        self.directory = Directory(self.db, lambda: self.config)

    def look_up(self) -> int:
        return self.loop.run(self.directory.look_up_sizes())

    def close(self) -> None:
        self.loop.run(self.directory.aclose())
        for server in (self.site_server, self.hub_server):
            server.stop()
        self.loop.stop()
        self.db.close()


@pytest.fixture
def sites(tmp_path, monkeypatch, allow_size_lookups):
    made = Sites(tmp_path, monkeypatch)
    try:
        yield made
    finally:
        made.close()


# --- read where published ---


def test_a_build_with_no_size_has_its_one_page_read_and_the_size_is_known(sites):
    assert build_sizes_gb(sites.db, {BIG: BIG}) == {BIG: None}
    assert sites.directory.size_lookups() == [("ollama", "gemma4")]
    assert sites.look_up() == 1
    assert sites.site.requests == ["/library/gemma4/tags"], "one page, never the library"
    assert build_sizes_gb(sites.db, {BIG: BIG}) == {BIG: 20.0}


def test_it_is_cached_and_read_again_only_once_stale(sites):
    sites.look_up()
    assert sites.directory.size_lookups() == []
    sites.db.execute("UPDATE model_directory SET fetched_at = fetched_at - ? WHERE source = ?",
                     (8 * 24 * 3600, OLLAMA_TAGS))
    assert sites.directory.size_lookups() == [("ollama", "gemma4")], "a week on, to see a tag pushed again"


def test_two_builds_of_one_model_read_its_page_once(sites):
    sites.config = config({BIG: [Variant(tag=BIG, engine="ollama")], "small": [Variant(tag="gemma4:e4b")]})
    assert sites.directory.size_lookups() == [("ollama", "gemma4")]
    sites.look_up()
    assert build_sizes_gb(sites.db, {BIG: BIG, "small": "gemma4:e4b"}) == {BIG: 20.0, "small": 9.6}


def test_a_bare_name_is_its_latest(sites):
    sites.config = config({"plain": [Variant(tag="gemma4", engine="ollama")]})
    sites.look_up()
    assert build_sizes_gb(sites.db, {"plain": "gemma4"}) == {"plain": 9.6}


def test_a_hub_build_has_its_listing_read(sites):
    sites.config = config({"model": [Variant(tag=REPO, engine="vllm")]})
    assert sites.directory.size_lookups() == [("hub", REPO)]
    sites.look_up()
    assert sites.hub.requests == [f"list {REPO}"], "its file listing, nothing else"
    assert sites.directory.store.get(HUB_SIZE, REPO) is not None, "kept: read once"
    assert sites.directory.size_lookups() == []


@pytest.mark.parametrize("variant", [
    Variant(tag=BIG, engine="ollama", size_gb=19.0),                    # the file says
    Variant(tag="provider-model", engine="openai-api", requires=["hosted:someone"]),  # nothing to download
    Variant(tag="someone/their-model:7b", engine="ollama"),             # not in the library
    Variant(tag="hf.co/acme/model:Q4_K_M", engine="ollama"),           # not in the library
    Variant(tag="acme/whisper-models/ggml-large.bin", engine="vllm"),   # a file inside a repository
    Variant(tag="upload/" + "d" * 64, engine="vllm"),                   # the pool's own store: never sent out
])
def test_nothing_is_read_for_a_build_that_needs_no_lookup_or_cannot_have_one(sites, variant):
    sites.config = config({"model": [variant]})
    assert sites.directory.size_lookups() == []


def test_the_operator_can_turn_it_off(sites):
    sites.config = config({BIG: [Variant(tag=BIG, engine="ollama")]}, look_up_sizes=False)
    assert sites.directory.size_lookups() == []


def test_a_page_that_cannot_be_read_is_not_asked_again_for_hours(sites):
    sites.site.down = True
    assert sites.look_up() == 0
    assert sites.directory.size_lookups() == []
    sites.directory._size_failed["ollama:gemma4"] -= directory_module.SIZE_RETRY_S + 1
    assert sites.directory.size_lookups() == [("ollama", "gemma4")]


def test_a_model_the_library_does_not_have_is_not_asked_again_for_hours(sites):
    sites.config = config({"mine": [Variant(tag="not-published:1b", engine="ollama")]})
    assert sites.look_up() == 0
    assert sites.site.requests == ["/library/not-published/tags"]
    assert sites.directory.size_lookups() == []


def test_the_supervisors_check_starts_it_in_the_background_once_a_minute(sites):
    async def check() -> tuple[bool, bool]:
        first = sites.directory.maybe_look_up_sizes()
        again = sites.directory.maybe_look_up_sizes()
        await sites.directory._sizes_task
        return first, again

    assert sites.loop.run(check()) == (True, False)
    assert build_sizes_gb(sites.db, {BIG: BIG}) == {BIG: 20.0}


# --- measured where they run ---


def engine_report(digest: str = DIGEST, *, loaded: bool = True, disk: int = 19_900_000_000,
                  memory: int = 22_400_000_000) -> dict:
    return {
        "name": "ollama", "answers": True,
        "models_on_disk": [{"tag": BIG, "size_bytes": disk, "digest": digest}],
        "models_loaded": [BIG] if loaded else [],
        "models_loaded_sizes": [{"tag": BIG, "digest": digest, "size_bytes": memory,
                                 "vram_bytes": memory, "context": 8192}] if loaded else [],
    }


def test_what_a_host_measured_is_kept_by_digest_and_used_next_time(tmp_path):
    db = Database(tmp_path / "gpm.sqlite3")
    assert record_measurements(db, engine_report(), "rented-1") == 1
    row = db.query("SELECT * FROM build_measurements")[0]
    assert (row["tag"], row["digest"], row["disk_gb"], row["memory_gb"], row["context"]) == \
        (BIG, DIGEST, 19.9, 22.4, 8192)
    assert build_sizes_gb(db, {BIG: BIG}) == {BIG: 19.9}
    assert build_memory_gb(db, {BIG: BIG}) == {BIG: 22.4}


def test_a_report_that_says_nothing_new_writes_nothing(tmp_path):
    db = Database(tmp_path / "gpm.sqlite3")
    record_measurements(db, engine_report(), "rented-1")
    assert record_measurements(db, engine_report(), "rented-1") == 0
    assert record_measurements(db, engine_report(loaded=False), "rented-1") == 0, "unloaded keeps its memory"
    assert build_memory_gb(db, {BIG: BIG}) == {BIG: 22.4}


def test_an_agent_that_reports_no_digests_leaves_nothing(tmp_path):
    db = Database(tmp_path / "gpm.sqlite3")
    old = {"name": "ollama", "answers": True, "models_on_disk": [{"tag": BIG, "size_bytes": 1}], "models_loaded": [BIG]}
    assert record_measurements(db, old, "h") == 0
    assert record_measurements(db, {"name": "ollama", "answers": False}, "h") == 0
    assert record_measurements(db, None, "h") == 0


def test_a_tag_pushed_again_is_measured_again(sites):
    record_measurements(sites.db, engine_report(digest="ffffffffffff" + "c" * 52), "rented-1")
    assert build_memory_gb(sites.db, {BIG: BIG}) == {BIG: 22.4}, "nothing published to compare with yet"
    sites.look_up()  # the library now shows 31b at 6316f0629137: another build
    assert build_memory_gb(sites.db, {BIG: BIG}) == {BIG: None}, "the old build's memory is set aside"
    assert build_sizes_gb(sites.db, {BIG: BIG}) == {BIG: 20.0}, "the published size stands in"
    record_measurements(sites.db, engine_report(digest=DIGEST, memory=21_000_000_000), "rented-2")
    assert build_memory_gb(sites.db, {BIG: BIG}) == {BIG: 21.0}


# --- what a machine must have ---


def test_a_measured_figure_replaces_the_estimate():
    estimated = needs_for({BIG: 20.0}, 1)
    measured = needs_for({BIG: 20.0}, 1, {BIG: 21.0})
    assert estimated.card_memory_gb == 29.0  # (20 × 1.1 + 3.2) / 0.9, rounded up
    assert measured.card_memory_gb == 24.0   # 21 / 0.9, rounded up
    assert measured.disk_gb == estimated.disk_gb


def test_each_model_is_measured_or_estimated_on_its_own():
    needs = needs_for({"a": 10.0, "b": 4.0}, 2, {"a": 12.0})
    assert needs.card_memory_gb == pytest.approx(-(-(12.0 / 2 + 4.0 * 1.1 / 2 + 3 * 1024**3 / 1e9) // 0.9), abs=1)


# --- what the agent reports ---


async def test_the_agent_reports_each_builds_digest_and_what_it_takes_loaded():
    """Ollama's own answers, as its `/api/tags` and `/api/ps` give them."""
    from gpm_agent.engines import OllamaFacts

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": BIG, "size": 19_900_000_000, "digest": DIGEST}]})
        return httpx.Response(200, json={"models": [{"name": BIG, "digest": DIGEST, "size": 22_400_000_000,
                                                     "size_vram": 22_400_000_000, "context_length": 8192}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer), base_url="http://engine") as client:
        described = await OllamaFacts().describe(client)
    assert described["models_on_disk"] == [{"tag": BIG, "size_bytes": 19_900_000_000, "digest": DIGEST}]
    assert described["models_loaded_sizes"] == [{"tag": BIG, "digest": DIGEST, "size_bytes": 22_400_000_000,
                                                 "vram_bytes": 22_400_000_000, "context": 8192}]


def test_what_the_agent_reports_is_what_the_pool_keeps(tmp_path):
    db = Database(tmp_path / "gpm.sqlite3")
    report = engine_report()
    assert record_measurements(db, report, "laptop") == 1
    assert build_memory_gb(db, {BIG: BIG}) == {BIG: 22.4}


def test_a_build_with_no_tag_of_its_own_is_measured_under_its_latest(tmp_path):
    """Ollama reports `owner/model` as `owner/model:latest`; the catalog may write either."""
    db = Database(tmp_path / "gpm.sqlite3")
    report = engine_report()
    for key in ("models_on_disk", "models_loaded_sizes"):
        report[key][0]["tag"] = "someone/their-model:latest"
    record_measurements(db, report, "laptop")
    assert build_sizes_gb(db, {"m": "someone/their-model"}) == {"m": 19.9}
    assert build_memory_gb(db, {"m": "someone/their-model"}) == {"m": 22.4}


def test_two_hosts_holding_one_build_keep_the_most_it_took_and_stop_writing(tmp_path):
    db = Database(tmp_path / "gpm.sqlite3")
    record_measurements(db, engine_report(memory=22_400_000_000), "big-context")
    assert record_measurements(db, engine_report(memory=21_000_000_000), "small-context") == 0
    assert record_measurements(db, engine_report(memory=22_400_000_000), "big-context") == 0
    assert build_memory_gb(db, {BIG: BIG}) == {BIG: 22.4}


def test_an_old_build_measured_later_does_not_hide_the_current_one(sites):
    sites.look_up()  # the library shows 31b at 6316f0629137
    record_measurements(sites.db, engine_report(digest=DIGEST, memory=21_000_000_000), "new-host")
    record_measurements(sites.db, engine_report(digest="ffffffffffff" + "c" * 52), "old-host")
    assert build_memory_gb(sites.db, {BIG: BIG}) == {BIG: 21.0}


def test_any_failure_waits_its_hours(sites, monkeypatch):
    async def broken(self, name):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(Directory, "read_tags_page", broken)
    assert sites.look_up() == 0
    assert sites.directory.size_lookups() == []


def test_nothing_is_started_once_closed(sites):
    sites.loop.run(sites.directory.aclose())
    assert sites.directory.maybe_look_up_sizes() is False
