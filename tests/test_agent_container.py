"""The host agent's `container` engine (S15, D180, D181), against a stand-in for the Docker Engine
API: an image pulled by digest with the machine's own login, run narrowly with its port on loopback,
and never where the machine's owner said never. No Docker is needed."""

import asyncio
import base64
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from gpm_agent.container_engine import ContainerFacts, machine_login
from gpm_agent.models import ModelWork, Refused
from gpm_agent.settings import Settings, SettingsError

IMAGE = "registry.example/me/detector@sha256:" + "a" * 64
PUBLIC = "docker.io/library/echo@sha256:" + "b" * 64


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Predictor(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self._send({"ok": True} if self.path == "/health" else {"data": []})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"model": body["model"], "outputs": {"boxes": [[1, 2, 3, 4]]}})

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


class FakeDocker:
    """The Docker Engine API as the agent uses it, keeping what it was asked."""

    def __init__(self, private=()):
        self.images: dict[str, int] = {}
        self.containers: dict[str, dict] = {}
        self.private = set(private)
        self.created: list[dict] = []
        self.pull_headers: list[dict] = []
        self.servers: dict[str, ThreadingHTTPServer] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.answer)

    def answer(self, request: httpx.Request) -> httpx.Response:
        path, params = request.url.path, request.url.params
        if path == "/version":
            return httpx.Response(200, json={"Version": "27.0.0"})
        if path == "/images/create":
            ref = params["fromImage"]
            self.pull_headers.append(dict(request.headers))
            if ref.split("@")[0] in self.private and "x-registry-auth" not in request.headers:
                return httpx.Response(200, content=json.dumps({"error": "pull access denied"}) + "\n")
            self.images[ref] = 1000
            lines = [{"id": "l1", "progressDetail": {"current": 500, "total": 1000}},
                     {"id": "l1", "progressDetail": {"current": 1000, "total": 1000}}]
            return httpx.Response(200, content="\n".join(json.dumps(line) for line in lines) + "\n")
        if path.startswith("/images/") and path.endswith("/json"):
            ref = httpx.URL(path).path[len("/images/"):-len("/json")]
            from urllib.parse import unquote

            ref = unquote(ref)
            return httpx.Response(200, json={"Size": self.images[ref]}) if ref in self.images else httpx.Response(404)
        if path.startswith("/images/") and request.method == "DELETE":
            from urllib.parse import unquote

            self.images.pop(unquote(path[len("/images/"):]), None)
            return httpx.Response(200, json=[])
        if path == "/containers/json":
            return httpx.Response(200, json=list(self.containers.values()))
        if path == "/containers/create":
            body = json.loads(request.content)
            self.created.append(body)
            cid = f"c{len(self.created)}"
            self.containers[cid] = {"Id": cid, "Labels": body["Labels"], "State": "created", "Status": "Created",
                                    "Image": body["Image"], "HostConfig": body["HostConfig"]}
            return httpx.Response(201, json={"Id": cid})
        if path.endswith("/start"):
            cid = path.split("/")[2]
            container = self.containers[cid]
            binding = next(iter(container["HostConfig"]["PortBindings"].values()))[0]
            server = ThreadingHTTPServer((binding["HostIp"], int(binding["HostPort"])), Predictor)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers[cid] = server
            container["State"], container["Status"] = "running", "Up"
            return httpx.Response(204)
        if path.endswith("/stop"):
            cid = path.split("/")[2]
            server = self.servers.pop(cid, None)
            if server is not None:
                server.shutdown()
                server.server_close()
            self.containers[cid]["State"] = "exited"
            return httpx.Response(204)
        if path.startswith("/containers/") and request.method == "DELETE":
            self.containers.pop(path.split("/")[2], None)
            return httpx.Response(204)
        return httpx.Response(404, json={"message": f"no {request.method} {path}"})

    def close(self):
        for server in self.servers.values():
            server.shutdown()
            server.server_close()


@pytest.fixture
def docker():
    fake = FakeDocker(private={"registry.example/me/detector"})
    yield fake
    fake.close()


def engine_on(docker, tmp_path, **settings):
    config = tmp_path / "docker"
    config.mkdir(exist_ok=True)
    login = base64.b64encode(b"me:not-a-real-password").decode()
    (config / "config.json").write_text(json.dumps({"auths": {"registry.example": {"auth": login}}}))
    chosen = Settings(key_hash="x", engine="container", engine_url=f"http://127.0.0.1:{free_port()}", **settings)
    chosen.check()
    return ContainerFacts(chosen, docker_transport=docker.transport(), docker_config=config / "config.json")


def spec(**needs):
    return ContainerFacts.checked_spec(IMAGE, {"engine": "container", "port": 8080, "needs": needs})


def run(coroutine):
    return asyncio.run(coroutine)


def test_an_image_is_pulled_by_digest_with_the_machines_own_login_and_run_narrowly(docker, tmp_path):
    engine = engine_on(docker, tmp_path)
    engine.set_specs({IMAGE: spec(memory_gb=2, card=True)})

    async def scenario():
        async with httpx.AsyncClient(base_url=engine.settings.engine_url, timeout=10) as client:
            progress = [step async for step in engine.pull(client, IMAGE)]
            assert progress[-1] == (1000, 1000)
            await engine.hold(client, IMAGE, pinned=True)
            facts = await engine.describe(client)
            answer = await client.post("/v1/predict", json={"model": IMAGE, "inputs": {"frame": "..."}})
        return facts, answer

    facts, answer = run(scenario())
    said = json.loads(base64.urlsafe_b64decode(docker.pull_headers[-1]["x-registry-auth"]))
    assert said == {"username": "me", "password": "not-a-real-password", "serveraddress": "registry.example"}
    host = docker.created[-1]["HostConfig"]
    assert host["PortBindings"] == {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": engine.settings.engine_url.rsplit(":", 1)[1]}]}
    assert host["Privileged"] is False and host["NetworkMode"] == "bridge" and host["CapAdd"] == []
    assert "no-new-privileges:true" in host["SecurityOpt"] and host["Binds"] == [] and host["Mounts"] == []
    assert host["Memory"] == 2 * 1024 ** 3 and host["DeviceRequests"][0]["Capabilities"] == [["gpu"]]
    assert docker.created[-1]["Env"] == [], "nothing of the agent's environment reaches the image"
    assert facts["models_loaded"] == [IMAGE] and facts["models_on_disk"] == [{"tag": IMAGE, "size_bytes": 1000}]
    assert answer.json()["outputs"] == {"boxes": [[1, 2, 3, 4]]}


def test_a_build_with_no_card_gets_none_and_one_without_a_login_pulls_without(docker, tmp_path):
    engine = engine_on(docker, tmp_path)
    tag = PUBLIC
    engine.set_specs({tag: ContainerFacts.checked_spec(tag, {"engine": "container", "port": 9000})})

    async def scenario():
        async with httpx.AsyncClient(base_url=engine.settings.engine_url, timeout=10) as client:
            async for _ in engine.pull(client, tag):
                pass
            await engine.hold(client, tag, pinned=True)
            await engine.keep_only(client, [])
            return await engine.describe(client)

    facts = run(scenario())
    assert "x-registry-auth" not in docker.pull_headers[-1]
    assert "DeviceRequests" not in docker.created[-1]["HostConfig"] and "Memory" not in docker.created[-1]["HostConfig"]
    assert facts["models_loaded"] == [] and docker.containers == {}, "keep_only with nothing wanted removes it"


def test_a_private_image_without_a_login_is_refused_in_dockers_words(docker, tmp_path):
    engine = engine_on(docker, tmp_path)
    engine.docker_config = tmp_path / "none.json"

    async def scenario():
        async with httpx.AsyncClient(base_url=engine.settings.engine_url) as client:
            with pytest.raises(Exception, match="pull access denied"):
                async for _ in engine.pull(client, IMAGE):
                    pass

    run(scenario())


def test_where_the_owner_said_never_nothing_is_pulled_or_run(docker, tmp_path):
    engine = engine_on(docker, tmp_path, run_containers="never")
    engine.set_specs({IMAGE: spec()})

    async def scenario():
        async with httpx.AsyncClient(base_url=engine.settings.engine_url) as client:
            with pytest.raises(Exception, match="run_containers: never"):
                async for _ in engine.pull(client, IMAGE):
                    pass
            with pytest.raises(Exception, match="run_containers: never"):
                await engine.hold(client, IMAGE, pinned=True)
            return await engine.describe(client)

    facts = run(scenario())
    assert docker.pull_headers == [] and docker.created == [] and facts["run_containers"] == "never"


def test_the_owners_setting_is_follow_or_never():
    with pytest.raises(SettingsError, match="run_containers"):
        Settings(key_hash="x", engine="container", engine_url="http://127.0.0.1:8191", run_containers="always").check()
    with pytest.raises(SettingsError, match="127.0.0.1"):
        Settings(key_hash="x", engine="container", engine_url="http://10.0.0.5:8191").check()


@pytest.mark.parametrize("tag", ["registry.example/me/detector:latest", "me/detector", "registry.example/me/../x@sha256:" + "a" * 64])
def test_only_an_image_by_digest_is_a_container_build(docker, tmp_path, tag):
    engine = engine_on(docker, tmp_path)
    work = ModelWork(engine, httpx.AsyncClient(base_url="http://127.0.0.1:1"),
                     free_disk_bytes=lambda: 10**12, min_free_disk_gb=1, allow_delete=False)

    async def scenario():
        with pytest.raises(Refused) as refused:
            await work.apply({"tags": [tag], "residency": "pinned"})
        assert refused.value.error == "bad_tag"
        accepted = await work.apply({"tags": [IMAGE], "residency": "pinned",
                                     "specs": {IMAGE: {"engine": "container", "port": 8080}}})
        assert accepted is not None
        with pytest.raises(Refused, match="one model at a time"):
            await work.apply({"tags": [IMAGE, PUBLIC], "residency": "pinned"})

    run(scenario())


def test_a_login_is_read_for_the_images_registry_only(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"auths": {"registry.example": {"auth": base64.b64encode(b"u:p").decode()}}}))
    assert machine_login("registry.example/me/detector", config) is not None
    assert machine_login("other.example/me/detector", config) is None
    assert machine_login("library/echo", config) is None
