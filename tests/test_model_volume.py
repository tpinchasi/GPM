"""A model volume on the agent's side (D139): one host fills it, later hosts copy from it, and
nothing in it is trusted — every file is checked against the hash the hub publishes, nothing but
the hub's own file list comes across, and no link in it is followed.

Against a fake hub over httpx's mock transport and temporary directories standing in for the
volume's two mount paths; no network, no provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

import httpx
import pytest
from gpm_agent import engines, modelhub
from gpm_agent import volume as model_volume

REPO = "acme/tiny-model"
WEIGHTS = b"w" * 300_000
FILES = {
    "config.json": b'{"model_type": "tiny"}',
    "model.safetensors": WEIGHTS,
    "tokenizer/tokenizer.json": b'{"vocab": {}}',
}
LFS = {"model.safetensors"}


def git_sha1(body: bytes) -> str:
    return hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()


class Hub:
    """The hub's tree listing with each file's published hash, and its downloads, counted."""

    def __init__(self, files=FILES, hashed=True):
        self.files, self.hashed = dict(files), hashed
        self.downloads: list[str] = []

    def entry(self, path, body):
        row = {"type": "file", "path": path, "size": len(body)}
        if not self.hashed:
            return row
        if path in LFS:
            row["oid"] = "f" * 40  # the pointer's own hash, which must not be used
            row["lfs"] = {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}
        else:
            row["oid"] = git_sha1(body)
        return row

    def client(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.startswith("/api/models/"):
                return httpx.Response(200, json=[self.entry(p, b) for p, b in self.files.items()])
            name = request.url.path.split("/resolve/main/", 1)[-1]
            self.downloads.append(name)
            body = self.files[name]
            asked = request.headers.get("Range")
            start = int(asked.removeprefix("bytes=").rstrip("-")) if asked else 0
            return httpx.Response(206 if asked else 200, content=body[start:])

        return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://hub")


@pytest.fixture(autouse=True)
def hub_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hub")


def fetch(hub, models, **kwargs):
    report = model_volume.Report()

    async def run():
        async with hub.client() as client:
            async for _ in modelhub.fetch(REPO, models, client=client, report=report, **kwargs):
                pass
        await model_volume.fills_finished()  # a fill runs behind the host

    asyncio.run(run())
    return report


def filled(tmp_path):
    """A volume filled by a first host, as the pool's first host on a volume leaves it."""
    volume = tmp_path / "volume"
    volume.mkdir()
    report = fetch(Hub(), tmp_path / "first", volume_fill=volume)
    assert report.fill == "filled"
    return volume


def build_dir(volume):
    (build,) = [p for p in (volume / "acme__tiny-model").iterdir()]
    return build


def on_disk(models):
    root = models / "acme__tiny-model"
    return {p: (root / p).read_bytes() for p in FILES}


# --- the listing keeps what the hub publishes ---


def test_the_listing_keeps_each_files_published_hash():
    async def run():
        async with Hub().client() as client:
            return await modelhub.listing(REPO, client=client)

    files = {f.path: f for f in asyncio.run(run())}
    weights = files["model.safetensors"]
    assert weights.sha256 == hashlib.sha256(WEIGHTS).hexdigest() and weights.git_sha1 is None, \
        "a large file's content hash is under `lfs`, not its pointer's `oid`"
    assert files["config.json"].git_sha1 == git_sha1(FILES["config.json"])


def test_a_listed_path_that_climbs_out_is_refused():
    hub = Hub({**FILES, "../../etc/cron.d/x": b"evil"})
    with pytest.raises(modelhub.HubRefused, match="will not write"):
        fetch(hub, "/nonexistent-models")


def test_a_build_is_named_by_its_files_and_their_hashes():
    a = [model_volume.Expected("x", 1, sha256="a" * 64)]
    assert model_volume.build_id(a) == model_volume.build_id(list(a))
    assert model_volume.build_id(a) != model_volume.build_id([model_volume.Expected("x", 1, sha256="b" * 64)])
    assert model_volume.build_id(a + [model_volume.Expected("y", 1)]) is None, "an unchecked file: no build"


# --- filling ---


def test_the_first_host_fills_the_volume_with_one_whole_build(tmp_path):
    volume = filled(tmp_path)
    build = build_dir(volume)
    assert not build.name.startswith("."), "renamed into place; no draft left behind"
    assert {p: (build / p).read_bytes() for p in FILES} == FILES
    marker = json.loads((build / model_volume.BUILD_MARKER).read_text())
    assert marker["build"] == build.name and set(marker["files"]) == set(FILES)


def test_the_host_has_its_model_before_the_fill_ends_and_fills_while_it_serves(tmp_path, monkeypatch):
    # The fill runs after the fetch, not inside it: a host is ready once its models are on its
    # own disk, and does not stand paid and idle while it copies them up (the owner, D139).
    volume = tmp_path / "volume"
    volume.mkdir()
    release = asyncio.Event()
    real = model_volume.fill

    async def slow_fill(*args):
        await release.wait()
        await real(*args)

    monkeypatch.setattr(model_volume, "fill", slow_fill)
    report = model_volume.Report()

    async def run():
        async with Hub().client() as client:
            async for _ in modelhub.fetch(REPO, tmp_path / "first", client=client, report=report, volume_fill=volume):
                pass
        assert modelhub.is_complete(tmp_path / "first" / "acme__tiny-model"), "ready before the fill"
        assert report.fill == "filling"
        release.set()
        await model_volume.fills_finished()

    asyncio.run(run())
    assert report.fill == "filled" and build_dir(volume).is_dir()


def test_a_build_already_on_the_volume_is_left_as_it_is(tmp_path):
    volume = filled(tmp_path)
    report = fetch(Hub(), tmp_path / "second", volume_fill=volume)
    assert report.fill == "already there"
    assert len(list((volume / "acme__tiny-model").iterdir())) == 1


def test_a_filler_whose_own_copy_is_wrong_writes_nothing(tmp_path):
    models = tmp_path / "first"
    fetch(Hub(), models)
    (models / "acme__tiny-model" / "model.safetensors").write_bytes(b"x" * len(WEIGHTS))  # same size
    volume = tmp_path / "volume"
    volume.mkdir()
    report = fetch(Hub(), models, volume_fill=volume)
    assert report.fill.startswith("not filled") and "does not match the hub" in report.fill
    assert list((volume / "acme__tiny-model").iterdir()) == [], "no build and no draft"


def test_a_link_planted_in_the_volume_is_not_followed_by_a_fill(tmp_path):
    elsewhere = tmp_path / "host-files"
    elsewhere.mkdir()
    volume = tmp_path / "volume"
    volume.mkdir()
    os.symlink(elsewhere, volume / "acme__tiny-model")
    report = fetch(Hub(), tmp_path / "first", volume_fill=volume)
    assert report.fill.startswith("not filled")
    assert list(elsewhere.iterdir()) == [], "nothing written through the link"
    assert on_disk(tmp_path / "first") == FILES, "the host has its model either way"


def test_files_the_hub_publishes_no_hash_for_are_never_filled_or_read(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    report = fetch(Hub(hashed=False), tmp_path / "first", volume_fill=volume)
    assert report.fill.startswith("not filled") and "no hash" in report.fill


# --- reading ---


def test_a_later_host_copies_its_model_from_the_volume_and_downloads_nothing(tmp_path):
    volume = filled(tmp_path)
    hub = Hub()
    report = fetch(hub, tmp_path / "second", volume_read=volume)
    assert hub.downloads == []
    assert report.files_from_volume == len(FILES) and report.bytes_from_volume == sum(map(len, FILES.values()))
    assert report.files_mismatched == 0 and report.files_missing == 0
    assert on_disk(tmp_path / "second") == FILES
    assert modelhub.is_complete(tmp_path / "second" / "acme__tiny-model")


def test_a_file_someone_changed_on_the_volume_is_taken_from_the_hub(tmp_path):
    volume = filled(tmp_path)
    (build_dir(volume) / "model.safetensors").write_bytes(b"p" * len(WEIGHTS))  # poisoned, same size
    hub = Hub()
    report = fetch(hub, tmp_path / "second", volume_read=volume)
    assert report.files_mismatched == 1 and hub.downloads == ["model.safetensors"]
    assert on_disk(tmp_path / "second") == FILES, "the hub's bytes, not the volume's"
    assert not list((tmp_path / "second").rglob("*.gpm-volume-part")), "no draft left"


def test_only_the_files_the_hub_lists_come_across(tmp_path):
    volume = filled(tmp_path)
    (build_dir(volume) / "sitecustomize.py").write_text("import os; os.system('evil')")
    fetch(Hub(), tmp_path / "second", volume_read=volume)
    assert not (tmp_path / "second" / "acme__tiny-model" / "sitecustomize.py").exists()


def test_a_link_in_the_volume_is_not_followed_by_a_reader(tmp_path):
    volume = filled(tmp_path)
    secret = tmp_path / "secret"
    secret.write_bytes(b"s" * len(FILES["config.json"]))
    target = build_dir(volume) / "config.json"
    target.unlink()
    os.symlink(secret, target)
    hub = Hub()
    report = fetch(hub, tmp_path / "second", volume_read=volume)
    assert report.files_missing == 1 and hub.downloads == ["config.json"]
    assert on_disk(tmp_path / "second") == FILES


def test_a_volume_without_this_build_leaves_everything_to_the_hub(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    hub = Hub()
    report = fetch(hub, tmp_path / "second", volume_read=volume)
    assert sorted(hub.downloads) == sorted(FILES) and report.files_missing == len(FILES)


def test_a_changed_repository_is_a_new_build_and_the_old_one_is_not_read(tmp_path):
    volume = filled(tmp_path)
    hub = Hub({**FILES, "model.safetensors": b"v2" * 150_000})
    report = fetch(hub, tmp_path / "second", volume_read=volume)
    assert report.files_from_volume == 0 and sorted(hub.downloads) == sorted(FILES)


def test_the_volumes_paths_are_the_agents_own():
    # The pool imports these to mount the volume; it never sends either to the agent (D40, D41).
    assert str(model_volume.READ_PATH).startswith("/opt/gpm/") and str(model_volume.FILL_PATH).startswith("/opt/gpm/")
    assert model_volume.READ_PATH != model_volume.FILL_PATH


# --- Ollama: blobs named by their own hash ---


BLOBS = {"config": b'{"model_format": "gguf"}', "weights": b"g" * 200_000}


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def manifest():
    return {
        "config": {"digest": f"sha256:{digest(BLOBS['config'])}", "size": len(BLOBS["config"])},
        "layers": [{"digest": f"sha256:{digest(BLOBS['weights'])}", "size": len(BLOBS["weights"])}],
    }


def test_a_tags_blobs_come_from_the_registrys_own_manifest():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=manifest())

    async def run(tag):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await model_volume.ollama_blobs(tag, client)

    blobs = asyncio.run(run("gemma4:26b"))
    assert seen == ["/v2/library/gemma4/manifests/26b"]
    assert {b.path for b in blobs} == {f"sha256-{digest(v)}" for v in BLOBS.values()}
    assert asyncio.run(run("hf.co/someone/model:q4")) is None, "another registry: the engine does it all"


class Ollama:
    """An engine whose pull lands the tag's blobs in its store unless they are there already."""

    def __init__(self, store):
        self.store, self.downloaded = store, []

    def client(self):
        def handler(request):
            for body in BLOBS.values():
                name = f"sha256-{digest(body)}"
                if not (self.store / name).exists():
                    self.downloaded.append(name)
                    self.store.mkdir(parents=True, exist_ok=True)
                    (self.store / name).write_bytes(body)
            return httpx.Response(200, content=b'{"status": "success"}\n')

        return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://engine")


def pull_on(tmp_path, monkeypatch, host, *, read=None, fill=None):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / host))
    monkeypatch.setattr(model_volume, "READ_PATH", read or tmp_path / "not-mounted")
    monkeypatch.setattr(model_volume, "FILL_PATH", fill or tmp_path / "not-mounted")

    async def blobs(tag, client):
        return [model_volume.Expected(f"sha256-{digest(v)}", len(v), sha256=digest(v)) for v in BLOBS.values()]

    monkeypatch.setattr(model_volume, "ollama_blobs", blobs)
    engine, facts = Ollama(tmp_path / host / "blobs"), engines.OllamaFacts()

    async def run():
        async with engine.client() as client:
            async for _ in facts.pull(client, "gemma4:26b"):
                pass
        await model_volume.fills_finished()

    asyncio.run(run())
    return engine, engines._sources_fact(facts.sources).get("gemma4:26b")


def test_an_ollama_host_fills_the_volume_and_the_next_copies_from_it(tmp_path, monkeypatch):
    volume = tmp_path / "volume"
    volume.mkdir()
    _, first = pull_on(tmp_path, monkeypatch, "first", fill=volume)
    assert first["fill"] == "filled (2 new)"
    assert sorted(p.name for p in (volume / "ollama" / "blobs").iterdir()) == sorted(
        f"sha256-{digest(v)}" for v in BLOBS.values()), "no draft left behind"

    engine, second = pull_on(tmp_path, monkeypatch, "second", read=volume)
    assert engine.downloaded == [], "the engine's pull found every blob in its store"
    assert second["files_from_volume"] == 2 and second["files_mismatched"] == 0


def test_a_blob_that_does_not_hash_to_its_name_is_left_for_the_engine(tmp_path, monkeypatch):
    volume = tmp_path / "volume"
    volume.mkdir()
    pull_on(tmp_path, monkeypatch, "first", fill=volume)
    weights = volume / "ollama" / "blobs" / f"sha256-{digest(BLOBS['weights'])}"
    weights.write_bytes(b"p" * len(BLOBS["weights"]))
    engine, second = pull_on(tmp_path, monkeypatch, "second", read=volume)
    assert second["files_mismatched"] == 1 and engine.downloaded == [weights.name]
    assert (tmp_path / "second" / "blobs" / weights.name).read_bytes() == BLOBS["weights"]


def test_an_ollama_host_without_a_volume_pulls_as_before(tmp_path, monkeypatch):
    engine, sources = pull_on(tmp_path, monkeypatch, "only")
    assert len(engine.downloaded) == 2 and sources is None, "nothing to report without a volume"
