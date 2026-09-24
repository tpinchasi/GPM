"""The agent's vLLM side: fetching weights from a hub, and reporting what the engine serves.

docs/spec/host-agent.md §4. Against a fake hub and a fake engine; nothing here reaches the
network, needs a GPU, or installs vLLM.
"""

import httpx
import pytest
from gpm_agent import modelhub
from gpm_agent.engines import EngineRefused, VllmFacts, engine_facts

REPO = "nvidia/Gemma-4-26B-A4B-NVFP4"


class Settings:
    def __init__(self, models_path):
        self.models_path = str(models_path)


def facts(tmp_path) -> VllmFacts:
    return VllmFacts(Settings(tmp_path))


# --- what a repository id may be, and where it may land ---


def test_a_repository_lands_in_one_directory_named_after_it(tmp_path):
    assert modelhub.directory_for(tmp_path, REPO) == tmp_path / "nvidia__Gemma-4-26B-A4B-NVFP4"


@pytest.mark.parametrize("hostile", [
    "../../etc/passwd",
    "nvidia/../../../root/.ssh",
    "/absolute/path",
    "nvidia/model/../..",
    "no-slash-at-all",
    "",
    "owner/name;rm -rf /",
])
def test_a_name_that_is_not_a_repository_id_is_refused(hostile):
    """The name becomes a path on the machine. The pool may say which repository, never where —
    a name that climbed out of the models directory would be the pool naming a path, which the
    agent's protocol exists to prevent (D40)."""
    with pytest.raises(modelhub.HubRefused):
        modelhub.directory_for("/models", hostile)


# --- which files are worth moving ---


def test_the_same_weights_in_two_formats_are_fetched_once():
    """A repository commonly ships `.safetensors` for engines that load them and `.bin` for
    older tooling. Taking both would double the largest download the pool ever makes."""
    files = [
        modelhub.RemoteFile("model.safetensors", 1000),
        modelhub.RemoteFile("pytorch_model.bin", 1000),
        modelhub.RemoteFile("config.json", 10),
    ]
    kept = {f.path for f in modelhub.wanted(files)}
    assert kept == {"model.safetensors", "config.json"}


def test_a_repository_with_only_the_older_format_still_fetches_it():
    files = [modelhub.RemoteFile("pytorch_model.bin", 1000), modelhub.RemoteFile("config.json", 10)]
    assert {f.path for f in modelhub.wanted(files)} == {"pytorch_model.bin", "config.json"}


def test_readmes_and_pictures_are_not_weights():
    files = [
        modelhub.RemoteFile("README.md", 9000),
        modelhub.RemoteFile("preview.png", 50000),
        modelhub.RemoteFile("config.json", 10),
    ]
    assert {f.path for f in modelhub.wanted(files)} == {"config.json"}


# --- fetching ---


class FakeHub:
    """A hub that serves one repository, and can be told to cut a transfer part-way."""

    def __init__(self, files: dict[str, bytes], cut_after: int | None = None):
        self.files = files
        self.cut_after = cut_after
        self.ranges: list[str | None] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.startswith("/api/models/"):
                return httpx.Response(200, json=[
                    {"type": "file", "path": name, "size": len(body)}
                    for name, body in self.files.items()
                ])
            name = path.split("/resolve/main/", 1)[-1]
            if name not in self.files:
                return httpx.Response(404)
            body = self.files[name]
            asked = request.headers.get("Range")
            self.ranges.append(asked)
            start = int(asked.removeprefix("bytes=").rstrip("-")) if asked else 0
            body = body[start:]
            if self.cut_after is not None and len(body) > self.cut_after:
                body = body[: self.cut_after]
            return httpx.Response(206 if asked else 200, content=body)

        return httpx.MockTransport(handler)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport(), follow_redirects=True)


WEIGHTS = {"config.json": b"{}", "model.safetensors": b"w" * 4096}


async def test_a_fetch_writes_every_file_and_reports_real_bytes(tmp_path):
    hub = FakeHub(WEIGHTS)
    seen = []
    async with hub.client() as client:
        async for done, total in modelhub.fetch(REPO, tmp_path, client=client):
            seen.append((done, total))

    into = modelhub.directory_for(tmp_path, REPO)
    assert (into / "model.safetensors").read_bytes() == b"w" * 4096
    assert (into / "config.json").read_bytes() == b"{}"
    # A denominator from the first report, not a bar filling to an unknown end.
    assert all(total == 4098 for _, total in seen)
    assert seen[-1][0] == 4098


async def test_a_file_already_complete_is_not_fetched_again(tmp_path):
    into = modelhub.directory_for(tmp_path, REPO)
    into.mkdir(parents=True)
    (into / "model.safetensors").write_bytes(b"w" * 4096)
    (into / "config.json").write_bytes(b"{}")

    hub = FakeHub(WEIGHTS)
    async with hub.client() as client:
        async for _ in modelhub.fetch(REPO, tmp_path, client=client):
            pass
    assert hub.ranges == []  # nothing was asked for at all


async def test_a_cut_transfer_resumes_where_it_stopped(tmp_path):
    """A 19 GB download cut at 18 GB must cost a minute on the next attempt, not an hour — the
    pool retries a cut download (D57) and a rented host pays for every second of it."""
    hub = FakeHub(WEIGHTS, cut_after=1000)
    async with hub.client() as client:
        # A cut is a failure the next attempt resumes — never a finished download (D97).
        with pytest.raises(modelhub.HubRefused, match="resume"):
            async for _ in modelhub.fetch(REPO, tmp_path, client=client):
                pass
    partial = modelhub.directory_for(tmp_path, REPO) / "model.safetensors"
    assert partial.stat().st_size == 1000, "what arrived is kept"

    hub.cut_after = None
    async with hub.client() as client:
        async for _ in modelhub.fetch(REPO, tmp_path, client=client):
            pass
    assert partial.read_bytes() == b"w" * 4096
    assert "bytes=1000-" in hub.ranges


async def test_a_hub_that_ignores_the_range_restarts_the_file_rather_than_corrupting_it(tmp_path):
    """Appending the beginning of a file to the middle of it would produce a file of the right
    length and the wrong contents — the worst possible outcome, because nothing would notice."""
    into = modelhub.directory_for(tmp_path, REPO)
    into.mkdir(parents=True)
    (into / "model.safetensors").write_bytes(b"w" * 1000)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=[{"type": "file", "path": "model.safetensors", "size": 4096}])
        return httpx.Response(200, content=b"w" * 4096)  # 200, not 206: the range was ignored

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async for _ in modelhub.fetch(REPO, tmp_path, client=client):
            pass
    assert (into / "model.safetensors").read_bytes() == b"w" * 4096


async def test_a_file_longer_than_the_hub_says_is_replaced_not_resumed(tmp_path):
    into = modelhub.directory_for(tmp_path, REPO)
    into.mkdir(parents=True)
    (into / "model.safetensors").write_bytes(b"x" * 9999)

    hub = FakeHub({"model.safetensors": b"w" * 4096})
    async with hub.client() as client:
        async for _ in modelhub.fetch(REPO, tmp_path, client=client):
            pass
    assert (into / "model.safetensors").read_bytes() == b"w" * 4096


async def test_a_repository_the_hub_does_not_have_is_refused_plainly(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(modelhub.HubRefused, match="no repository"):
            await modelhub.listing(REPO, client=client)


async def test_a_gated_repository_says_what_is_missing(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(modelhub.HubRefused, match="HF_TOKEN"):
            await modelhub.listing(REPO, client=client)


def test_the_hub_credential_comes_from_the_machine_never_from_the_pool(monkeypatch):
    """The pool never sends a credential and the agent never accepts one: a token is the
    machine owner's, read from the process they started."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert modelhub._headers() == {}
    monkeypatch.setenv("HF_TOKEN", "secret")
    assert modelhub._headers() == {"Authorization": "Bearer secret"}


# --- what the agent reports about a vLLM engine ---


def engine_answering(routes: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path not in routes:
            return httpx.Response(404)
        return httpx.Response(200, json=routes[request.url.path])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://engine")


SERVING = {"/v1/models": {"data": [{"id": REPO}]}, "/version": {"version": "0.11.0"}}


def downloaded(tmp_path, repo=REPO, size=2048):
    """A model the fetch finished: its files, and the marker it writes last."""
    into = modelhub.directory_for(tmp_path, repo)
    into.mkdir(parents=True, exist_ok=True)
    (into / "model.safetensors").write_bytes(b"w" * size)
    (into / modelhub.COMPLETE_MARKER).write_text(f"{size}\n")
    return into


async def test_on_disk_and_serving_are_reported_as_different_sets(tmp_path):
    """The state a vLLM host spends its whole preparation in — weights present, engine not yet
    serving them — must be visible, not folded into one number."""
    downloaded(tmp_path)

    async with engine_answering({"/v1/models": {"data": []}, "/version": {"version": "0.11.0"}}) as client:
        described = await facts(tmp_path).describe(client)

    assert described["answers"] is True
    assert described["models_loaded"] == []
    assert described["models_on_disk"] == [{"tag": REPO, "size_bytes": 2048}]


async def test_a_stopped_engine_still_reports_what_is_on_disk(tmp_path):
    """On a machine that has just booted this engine is not running — it has nothing to serve
    yet — and what is on disk is exactly what decides what happens next. Reporting nothing
    while it was down is what kept the agent from ever starting the download (D97)."""
    downloaded(tmp_path)
    async with engine_answering({}) as client:
        described = await facts(tmp_path).describe(client)
    assert described["answers"] is False
    assert described["models_loaded"] == []
    assert described["models_on_disk"] == [{"tag": REPO, "size_bytes": 2048}]


async def test_a_download_in_progress_is_not_a_model_on_disk(tmp_path):
    """Files without the fetch's completion marker are a download still under way. Counting
    them would have the pool restart the engine on half a model."""
    into = modelhub.directory_for(tmp_path, REPO)
    into.mkdir(parents=True)
    (into / "model.safetensors").write_bytes(b"w" * 1000)
    async with engine_answering({}) as client:
        described = await facts(tmp_path).describe(client)
    assert described["models_on_disk"] == []


async def test_a_tag_the_engine_serves_is_already_held(tmp_path):
    """Permanently, by construction: this engine holds its model for the life of the process,
    which is exactly what the pool wanted from a pin."""
    async with engine_answering(SERVING) as client:
        await facts(tmp_path).hold(client, REPO, pinned=True)


async def test_a_tag_on_disk_but_not_served_says_a_restart_is_what_is_missing(tmp_path):
    downloaded(tmp_path)

    async with engine_answering({"/v1/models": {"data": []}}) as client:
        with pytest.raises(EngineRefused, match="restarted"):
            await facts(tmp_path).hold(client, REPO, pinned=True)


async def test_a_tag_neither_served_nor_on_disk_says_that_instead(tmp_path):
    async with engine_answering({"/v1/models": {"data": []}}) as client:
        with pytest.raises(EngineRefused, match="not on disk"):
            await facts(tmp_path).hold(client, REPO, pinned=True)


async def test_releasing_is_not_something_this_engine_offers(tmp_path):
    """One model for the life of the process: there is no lifetime to hand back, and failing
    here would fail a host for doing nothing wrong."""
    async with engine_answering(SERVING) as client:
        await facts(tmp_path).hold(client, REPO, pinned=False)


async def test_deleting_removes_the_weights_from_disk(tmp_path):
    into = modelhub.directory_for(tmp_path, REPO)
    (into / "nested").mkdir(parents=True)
    (into / "nested" / "model.safetensors").write_bytes(b"w")

    async with engine_answering(SERVING) as client:
        await facts(tmp_path).delete(client, REPO)
        assert not into.exists()
        # Idempotent: a tag already gone is not an error.
        await facts(tmp_path).delete(client, REPO)


async def test_deleting_refuses_a_name_that_is_not_a_repository(tmp_path):
    async with engine_answering(SERVING) as client:
        with pytest.raises(EngineRefused):
            await facts(tmp_path).delete(client, "../../etc")


# --- the numbers the pool sends, in this engine's names ---


def test_the_pools_numbers_become_this_engines_names(tmp_path):
    environment = facts(tmp_path).launch_environment(workers=84, models_held=3, context=32768)
    assert environment["GPM_VLLM_MAX_NUM_SEQS"] == "84"
    assert environment["GPM_VLLM_MAX_MODEL_LEN"] == "32768"
    assert int(environment["GPM_VLLM_MAX_NUM_BATCHED_TOKENS"]) >= 84


def test_the_count_of_models_is_not_written_because_it_cannot_be_honoured(tmp_path):
    environment = facts(tmp_path).launch_environment(workers=8, models_held=3)
    assert not any("MODELS" in name or "LOADED" in name for name in environment)


def test_the_numbers_read_back_are_the_numbers_written(tmp_path):
    engine = facts(tmp_path)
    written = engine.launch_environment(workers=84, models_held=1, context=32768)
    assert engine.settings_from_environment(written) == {
        "workers": 84, "models_held": 1, "context": 32768,
    }
    assert engine.settings_from_environment(None) is None


def test_the_registry_builds_both_engines_the_same_way(tmp_path):
    assert isinstance(engine_facts("vllm", Settings(tmp_path)), VllmFacts)
    assert engine_facts("ollama", Settings(tmp_path)).name == "ollama"
    assert engine_facts("an-engine-nobody-wrote", Settings(tmp_path)) is None
