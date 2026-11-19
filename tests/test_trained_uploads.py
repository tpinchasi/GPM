"""Trained models' files uploaded to the pool (S15, D182): kept by their hash, and sent by the pool
to each machine that holds them through the agent's one verb that takes a file — bytes and their
sha256, never a name, a path or a URL."""

import asyncio
import base64
import hashlib
import json

import httpx
import pytest
from fakes.harness import BackgroundLoop, EngineSpec, ServerHandle, pool_harness
from gpm_agent.app import create_app
from gpm_agent.settings import Settings, fingerprint
from gpm_server.uploads import UploadRefused, UploadStore

AGENT_KEY = "gpmg_" + "e" * 64
#: Starts as an ONNX file does (field 1, the IR version), and is otherwise just bytes.
FILE = b"\x08\x08" + bytes(range(256)) * 50
DIGEST = hashlib.sha256(FILE).hexdigest()
TAG = f"upload/{DIGEST}"


# --- the agent's verb ---


def agent(tmp_path, engine="onnx", **settings):
    from test_agent import MAC

    settings.setdefault("engine_url", "http://127.0.0.1:8190")
    chosen = Settings(key_hash=fingerprint(AGENT_KEY), engine=engine, models_path=str(tmp_path / "models"), **settings)
    return create_app(chosen, probes=MAC)


async def send(app, body, digest=DIGEST, key=AGENT_KEY, length=None):
    headers = {"Authorization": f"Bearer {key}", "X-GPM-SHA256": digest, "Content-Length": str(len(body) if length is None else length)}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
        return await http.put("/agent/v1/model-file", content=body, headers=headers)


async def test_a_file_is_kept_by_its_hash_and_reported_on_disk(tmp_path):
    app = agent(tmp_path)
    answer = await send(app, FILE)
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"tag": TAG, "on_disk": True}
    kept = tmp_path / "models" / f"upload__{DIGEST}" / "model.onnx"
    assert kept.read_bytes() == FILE
    on_disk = app.state.work.engine._on_disk()
    assert on_disk == [{"tag": TAG, "size_bytes": len(FILE)}]
    assert (await send(app, FILE)).status_code == 200, "sent again: kept as it was"


@pytest.mark.parametrize("body, digest, length, status, said", [
    (FILE, "0" * 64, None, 400, "do not hash"),
    (FILE, DIGEST, len(FILE) + 10, 400, "ended at"),
    (FILE, "../../etc/passwd", None, 400, "64 lower-case hex"),
    (FILE, DIGEST.upper(), None, 400, "64 lower-case hex"),
])
async def test_a_file_that_is_not_what_it_says_is_not_kept(tmp_path, body, digest, length, status, said):
    app = agent(tmp_path)
    answer = await send(app, body, digest=digest, length=length)
    assert answer.status_code == status and said in answer.json()["detail"], answer.text
    models = tmp_path / "models"
    assert not models.exists() or [p.name for p in models.iterdir()] == []


async def test_only_the_agent_key_sends_a_file_and_only_to_an_engine_that_takes_one(tmp_path):
    assert (await send(agent(tmp_path), FILE, key="gpmx_" + "a" * 64)).status_code == 403
    refused = await send(agent(tmp_path, engine="ollama", engine_url="http://127.0.0.1:11434"), FILE)
    assert refused.status_code == 409 and "takes no files" in refused.json()["detail"]


async def test_a_file_that_would_take_the_disk_below_the_owners_floor_is_refused(tmp_path):
    answer = await send(agent(tmp_path, min_free_disk_gb=99.99999), FILE)
    assert answer.status_code == 507 and "floor" in answer.json()["detail"]


# --- the pool's store ---


async def chunks(*parts):
    for part in parts:
        yield part


def test_the_store_keeps_a_file_by_its_hash_and_refuses_what_is_not_onnx_or_too_large(tmp_path):
    store = UploadStore(tmp_path / "uploads", max_bytes=len(FILE))
    digest, size = asyncio.run(store.keep(chunks(FILE[:100], FILE[100:])))
    assert (digest, size) == (DIGEST, len(FILE)) and store.path_of(digest).read_bytes() == FILE
    assert (tmp_path / "uploads").stat().st_mode & 0o777 == 0o700
    with pytest.raises(UploadRefused, match="not an ONNX file"):
        asyncio.run(store.keep(chunks(b"PK\x03\x04 a zip")))
    with pytest.raises(UploadRefused, match="larger"):
        asyncio.run(store.keep(chunks(FILE, b"more")))
    assert store.path_of("../x") is None and sorted(p.name for p in (tmp_path / "uploads").iterdir()) == [f"{DIGEST}.onnx"]


def test_a_file_is_uploaded_through_the_control_api(tmp_path):
    from gpm_server.config import load_config
    from gpm_server.db import Database
    from gpm_server.supervisor import Supervisor
    from gpm_server.supervisor.control import create_control_app

    path = tmp_path / "pool.yaml"
    path.write_text(
        "pool: {name: uploads, model_set: [chatter]}\n"
        f"request_log: {tmp_path / 'gpm.sqlite3'}\n"
        "models: {max_upload_mb: 1}\n"
        "auth: {admin_keys: [gpmx_uploads_admin], app_keys: [gpma_uploads_app]}\n"
        "hosts:\n  - {id: mac, kind: local, transport: {type: http, base_url: 'http://127.0.0.1:1'}}\n"
    )
    loop, database = BackgroundLoop(), Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    control = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        with httpx.Client(base_url=control.base_url, timeout=30, headers={"Authorization": "Bearer gpmx_uploads_admin"}) as api:
            kept = api.post("/pool/models/uploads", content=FILE)
            assert kept.status_code == 200, kept.text
            assert kept.json() == {"tag": TAG, "size_bytes": len(FILE)}
            assert api.post("/pool/models/uploads", content=b"not onnx").json()["error"] == "upload_refused"
            assert api.post("/pool/models/uploads", content=b"\x08" + b"x" * (2 ** 20 + 1)).status_code == 413
            assert httpx.post(f"{control.base_url}/pool/models/uploads", content=FILE).status_code == 401
        assert (tmp_path / "model-uploads" / f"{DIGEST}.onnx").read_bytes() == FILE
    finally:
        control.stop()
        loop.stop()
        database.close()


# --- the supervisor sends it ---


async def settled(pool):
    await asyncio.gather(*pool.supervisor._sending.values())


class Receiving:
    """The host's agent: holds what it is asked, and keeps what it is sent."""

    def __init__(self):
        self.received: list[tuple[dict, bytes]] = []
        self.models: list[dict] = []
        self.transport = httpx.MockTransport(self.answer)

    def answer(self, request):
        if request.url.path.endswith("/facts"):
            return httpx.Response(200, json={"agent": {"version": "t", "protocol": "1"}, "os": "Linux", "arch": "x86_64",
                                             "capabilities": [], "engine": {"name": "onnx", "answers": True}})
        if request.url.path.endswith("/model-file"):
            self.received.append((dict(request.headers), request.read()))
            return httpx.Response(200, json={"tag": TAG, "on_disk": True})
        return httpx.Response(200, json={"engine_answers": True, "models": self.models, "surplus": []})


def test_the_pool_sends_an_uploaded_build_to_the_host_that_holds_it_once(monkeypatch):
    from test_trained_models import SPEC

    monkeypatch.setenv("UPLOAD_AGENT_KEY", "test-agent-key")
    build = {"tag": TAG, "engine": "onnx", "spec": SPEC}
    with pool_harness(
        [EngineSpec(id="box", resident=set(), engine="onnx"), EngineSpec(id="mac", resident={"chatter:7b"})],
        model_set=["pets", "chatter"], catalog={"pets": {"variants": [build]}, "chatter": {"variants": [{"tag": "chatter:7b"}]}},
        pool_settings={"models_per_host": "declared"},
        host_overrides={"box": {"models": ["pets"], "agent": {"url": "http://127.0.0.1:8096", "bearer_env": "UPLOAD_AGENT_KEY"}},
                        "mac": {"models": ["chatter"]}},
    ) as pool:
        asyncio.run(pool.supervisor.uploads().keep(chunks(FILE)))
        agent_stub = Receiving()
        pool.supervisor._agent_transport = agent_stub.transport
        agent_stub.models = [{"tag": TAG, "on_disk": False, "loaded": False, "error": "has not arrived yet"}]
        pool.reprobe()
        pool.loop.run(settled(pool))
        headers, body = agent_stub.received[-1]
        assert body == FILE and headers["x-gpm-sha256"] == DIGEST and headers["content-length"] == str(len(FILE))
        assert headers["authorization"] == "Bearer test-agent-key"
        kinds = [row["kind"] for row in pool.database.query("SELECT kind FROM events ORDER BY id")]
        assert "upload_sent" in kinds

        agent_stub.models = [{"tag": TAG, "on_disk": True, "loaded": True}]
        pool.reprobe()
        pool.loop.run(settled(pool))
        assert len(agent_stub.received) == 1, "on disk: not sent again"
        assert json.loads(json.dumps(agent_stub.models))  # the report stays plain data


# --- relayed through the operator's hub repository ---


class FakeHub:
    """The model hub's API as the relay uses it, keeping what it was sent."""

    def __init__(self, mode="lfs", multipart=False):
        self.mode, self.multipart = mode, multipart
        self.stored = b""
        self.parts: dict[int, bytes] = {}
        self.commits: list[list[dict]] = []
        self.tokens_seen_at: list[str] = []
        self.transport = httpx.MockTransport(self.answer)

    async def answer(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "authorization" in request.headers:
            self.tokens_seen_at.append(request.url.host + path)
        if path == "/api/whoami-v2":
            token = request.headers.get("authorization", "")
            if token not in ("Bearer hf_fake_write", "Bearer hf_fake_read"):
                return httpx.Response(401, json={"error": "Invalid credentials"})
            role = "write" if token.endswith("write") else "read"
            return httpx.Response(200, json={"name": "operator", "auth": {"accessToken": {"role": role}}})
        if path == "/api/models/operator/models/preupload/main":
            return httpx.Response(200, json={"files": [{"path": json.loads(await request.aread())["files"][0]["path"], "uploadMode": self.mode}]})
        if path == "/operator/models.git/info/lfs/objects/batch":
            header = {"chunk_size": "6500", "00001": "https://storage.example/p1", "00002": "https://storage.example/p2"} if self.multipart else {}
            return httpx.Response(200, json={"objects": [{"oid": DIGEST, "size": len(FILE), "actions": {
                "upload": {"href": "https://storage.example/object", "header": header},
                "verify": {"href": "https://huggingface.co/operator/models.git/info/lfs/objects/verify"}}}]})
        if request.url.host == "storage.example" and path.startswith("/p"):
            self.parts[int(path[2:])] = await request.aread()
            return httpx.Response(200, headers={"etag": f'"e{path[2:]}"'})
        if request.url.host == "storage.example" and path == "/object":
            if request.method == "POST":
                payload = json.loads(await request.aread())
                assert [p["etag"] for p in payload["parts"]] == ['"e1"', '"e2"']
                self.stored = b"".join(self.parts[i] for i in sorted(self.parts))
            else:
                self.stored = await request.aread()
            return httpx.Response(200)
        if path.endswith("/lfs/objects/verify"):
            return httpx.Response(200) if hashlib.sha256(self.stored).hexdigest() == DIGEST else httpx.Response(404)
        if path == "/api/models/operator/models/commit/main":
            self.commits.append([json.loads(line) for line in (await request.aread()).decode().splitlines()])
            return httpx.Response(200, json={"commitOid": "abc"})
        return httpx.Response(404, json={"error": f"no {path}"})


@pytest.fixture
def control_pool(tmp_path):
    from gpm_server.config import load_config
    from gpm_server.db import Database
    from gpm_server.supervisor import Supervisor
    from gpm_server.supervisor.control import create_control_app

    path = tmp_path / "pool.yaml"
    path.write_text(
        "pool: {name: uploads, model_set: [chatter]}\n"
        f"request_log: {tmp_path / 'gpm.sqlite3'}\n"
        "auth: {admin_keys: [gpmx_uploads_admin], app_keys: [gpma_uploads_app]}\n"
        "hosts:\n  - {id: mac, kind: local, transport: {type: http, base_url: 'http://127.0.0.1:1'}}\n"
    )
    loop, database = BackgroundLoop(), Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    control = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    client = httpx.Client(base_url=control.base_url, timeout=30, headers={"Authorization": "Bearer gpmx_uploads_admin"})
    try:
        yield supervisor, client
    finally:
        client.close()
        control.stop()
        loop.stop()
        database.close()


@pytest.mark.parametrize("mode, multipart", [("lfs", False), ("lfs", True), ("regular", False)])
def test_an_upload_is_relayed_to_the_operators_hub_repository_and_becomes_a_hub_build(control_pool, mode, multipart):
    supervisor, api = control_pool
    hub = FakeHub(mode, multipart)
    supervisor._hub_transport = hub.transport
    assert api.post("/pool/models/uploads", content=FILE).status_code == 200
    unset = api.post(f"/pool/models/uploads/{DIGEST}/relay", json={"repo": "operator/models"})
    assert unset.status_code == 409 and unset.json()["error"] == "no_hub_credential"
    saved = api.put("/pool/models/hub-credential", json={"write_token": "hf_fake_write", "read_token": "hf_fake_read"})
    assert saved.status_code == 200 and "hf_fake" not in saved.text, saved.text
    relayed = api.post(f"/pool/models/uploads/{DIGEST}/relay", json={"repo": "operator/models", "file": "pets.onnx"})
    assert relayed.status_code == 200, relayed.text
    assert relayed.json() == {"tag": "operator/models/pets.onnx"}
    header, operation = hub.commits[-1]
    assert header["key"] == "header"
    if mode == "lfs":
        assert hub.stored == FILE and operation == {"key": "lfsFile", "value": {"path": "pets.onnx", "algo": "sha256", "oid": DIGEST, "size": len(FILE)}}
    else:
        assert operation["key"] == "file" and base64.b64decode(operation["value"]["content"]) == FILE
    assert not [seen for seen in hub.tokens_seen_at if seen.startswith("storage.example")], "the token never goes to the storage the hub hands out"


def test_a_token_the_hub_refuses_is_not_kept_and_a_bad_repository_is_refused(control_pool):
    supervisor, api = control_pool
    supervisor._hub_transport = FakeHub().transport
    refused = api.put("/pool/models/hub-credential", json={"write_token": "hf_wrong"})
    assert refused.status_code == 400 and refused.json()["error"] == "credential_refused"
    assert not (supervisor.config.credentials_dir() / "hub" / "write.json").exists()
    read_only = api.put("/pool/models/hub-credential", json={"write_token": "hf_fake_read"})
    assert read_only.status_code == 400 and "cannot write" in read_only.json()["detail"]
    api.post("/pool/models/uploads", content=FILE)
    api.put("/pool/models/hub-credential", json={"write_token": "hf_fake_write"})
    for repo in ("operator", "operator/../x", "https://evil.example/x"):
        assert api.post(f"/pool/models/uploads/{DIGEST}/relay", json={"repo": repo}).status_code == 502
    assert api.post(f"/pool/models/uploads/{'f' * 64}/relay", json={"repo": "operator/models"}).status_code == 404
    assert api.delete("/pool/models/hub-credential").json() == {"removed": ["write"]}


def test_a_read_token_that_can_write_is_refused(control_pool):
    """Found in review: the token a rented machine would carry must not be able to write (D182)."""
    supervisor, api = control_pool
    supervisor._hub_transport = FakeHub().transport
    refused = api.put("/pool/models/hub-credential", json={"read_token": "hf_fake_write"})
    assert refused.status_code == 400 and "read token can write" in refused.json()["detail"]
    assert api.put("/pool/models/hub-credential", json={"read_token": "hf_fake_read"}).status_code == 200


def test_a_file_an_agent_refuses_is_not_sent_again_every_pass(monkeypatch):
    """Found in review: a refused send streamed the whole file again on every pass."""
    from test_trained_models import SPEC

    monkeypatch.setenv("UPLOAD_AGENT_KEY", "test-agent-key")
    build = {"tag": TAG, "engine": "onnx", "spec": SPEC}
    with pool_harness(
        [EngineSpec(id="box", resident=set(), engine="onnx"), EngineSpec(id="mac", resident={"chatter:7b"})],
        model_set=["pets", "chatter"], catalog={"pets": {"variants": [build]}, "chatter": {"variants": [{"tag": "chatter:7b"}]}},
        pool_settings={"models_per_host": "declared"},
        host_overrides={"box": {"models": ["pets"], "agent": {"url": "http://127.0.0.1:8096", "bearer_env": "UPLOAD_AGENT_KEY"}},
                        "mac": {"models": ["chatter"]}},
    ) as pool:
        asyncio.run(pool.supervisor.uploads().keep(chunks(FILE)))
        sends = []

        def answer(request):
            if request.url.path.endswith("/facts"):
                return httpx.Response(200, json={"agent": {"version": "t", "protocol": "1"}, "os": "Linux", "arch": "x86_64",
                                                 "capabilities": [], "engine": {"name": "onnx", "answers": True}})
            if request.url.path.endswith("/model-file"):
                sends.append(1)
                return httpx.Response(507, json={"error": "disk_floor", "detail": "below the floor"})
            return httpx.Response(200, json={"engine_answers": True, "models": [{"tag": TAG, "on_disk": False}], "surplus": []})

        pool.supervisor._agent_transport = httpx.MockTransport(answer)
        for _ in range(4):
            pool.reprobe()
            pool.loop.run(settled(pool))
        assert sends == [1], "refused once, then waited for"
        failed = [r["summary"] for r in pool.database.query("SELECT kind, summary FROM events ORDER BY id") if r["kind"] == "upload_failed"]
        assert len(failed) == 1 and "below the floor" in failed[0] and "tried again in 60s" in failed[0]


def test_tokens_are_kept_all_or_nothing(control_pool):
    supervisor, api = control_pool
    supervisor._hub_transport = FakeHub().transport
    refused = api.put("/pool/models/hub-credential", json={"write_token": "hf_fake_write", "read_token": "hf_fake_write"})
    assert refused.status_code == 400
    assert not (supervisor.config.credentials_dir() / "hub" / "write.json").exists(), "the good one is not kept alone"


def test_a_file_missing_from_the_store_is_sent_as_soon_as_it_is_uploaded_again(monkeypatch):
    from test_trained_models import SPEC

    monkeypatch.setenv("UPLOAD_AGENT_KEY", "test-agent-key")
    build = {"tag": TAG, "engine": "onnx", "spec": SPEC}
    with pool_harness(
        [EngineSpec(id="box", resident=set(), engine="onnx"), EngineSpec(id="mac", resident={"chatter:7b"})],
        model_set=["pets", "chatter"], catalog={"pets": {"variants": [build]}, "chatter": {"variants": [{"tag": "chatter:7b"}]}},
        pool_settings={"models_per_host": "declared"},
        host_overrides={"box": {"models": ["pets"], "agent": {"url": "http://127.0.0.1:8096", "bearer_env": "UPLOAD_AGENT_KEY"}},
                        "mac": {"models": ["chatter"]}},
    ) as pool:
        agent_stub = Receiving()
        pool.supervisor._agent_transport = agent_stub.transport
        agent_stub.models = [{"tag": TAG, "on_disk": False}]
        pool.reprobe()
        pool.loop.run(settled(pool))
        assert agent_stub.received == [] and any(
            r["kind"] == "upload_failed" for r in pool.database.query("SELECT kind FROM events"))
        asyncio.run(pool.supervisor.uploads().keep(chunks(FILE)))
        pool.reprobe()
        pool.loop.run(settled(pool))
        assert [body for _, body in agent_stub.received] == [FILE], "uploaded again: sent at once, not after the wait"
