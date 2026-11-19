"""Add model on the console (S15, D178–D182): an ONNX file's inputs and outputs read from it, a build's
declarations held to its file before anything is saved, and the model added in one write."""

import io

import httpx
import pytest
import yaml
from fakes.harness import BackgroundLoop, ServerHandle

onnx = pytest.importorskip("onnx")
from gpm_server import onnxfile  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

SPEC = {
    "inputs": {"image": {"as": "image", "input": "pixel_values", "size": [4, 4], "channels": "rgb", "layout": "nchw", "scale": 255}},
    "outputs": {"label": {"as": "scores", "output": "logits", "softmax": True, "labels": ["red", "green", "blue"]}},
}


def classifier_bytes() -> bytes:
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["logits"], axes=[2, 3], keepdims=0)], "c",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["N", 3, 4, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


MODEL = classifier_bytes()


@pytest.fixture
def pool(tmp_path, monkeypatch):
    from gpm_server.config import load_config
    from gpm_server.db import Database
    from gpm_server.supervisor import Supervisor
    from gpm_server.supervisor.control import create_control_app

    monkeypatch.setenv("BOX_AGENT_KEY", "test-agent-key")
    path = tmp_path / "pool.yaml"
    path.write_text(
        "pool: {name: trained, model_set: [chatter], models_per_host: declared}\n"
        f"request_log: {tmp_path / 'gpm.sqlite3'}\n"
        "auth: {admin_keys: [gpmx_trained_admin], app_keys: [gpma_trained_app]}\n"
        "hosts:\n"
        "  - id: mac\n    kind: local\n    models: [chatter]\n    transport: {type: http, base_url: 'http://127.0.0.1:1'}\n"
        "  - id: box\n    kind: local\n    engine: onnx\n    transport: {type: http, base_url: 'http://127.0.0.1:8190'}\n"
        "    agent: {url: 'http://127.0.0.1:8096', bearer_env: BOX_AGENT_KEY}\n"
    )
    loop, database = BackgroundLoop(), Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    control = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    api = httpx.Client(base_url=control.base_url, timeout=30, headers={"Authorization": "Bearer gpmx_trained_admin"})
    try:
        yield supervisor, api, path
    finally:
        api.close()
        control.stop()
        loop.stop()
        database.close()


def test_the_reader_finds_what_the_onnx_package_finds():
    found = onnxfile.read(io.BytesIO(MODEL))
    assert found == {"inputs": [{"name": "pixel_values", "type": "float32", "shape": ["N", 3, 4, 4]}],
                     "outputs": [{"name": "logits", "type": "float32", "shape": ["N", 3]}]}
    with pytest.raises(onnxfile.NotOnnx):
        onnxfile.read(io.BytesIO(b"\x08\x08 not a model at all"))


def test_a_model_is_uploaded_inspected_tested_and_added_in_one_write(pool):
    supervisor, api, path = pool
    tag = api.post("/pool/models/uploads", content=MODEL).json()["tag"]
    seen = api.post("/pool/models/inspect", json={"upload": tag}).json()
    assert seen["inputs"][0]["shape"] == ["N", 3, 4, 4] and seen["size_bytes"] == len(MODEL)

    build = {"tag": tag, "engine": "onnx", "spec": SPEC, "max_batch": 8}
    tested = api.post("/pool/models/test", json={"build": build}).json()
    assert tested["ok"] and tested["problems"] == [], tested

    added = api.post("/pool/models/trained", json={"name": "colours", "build": build, "hosts": ["box"]})
    assert added.status_code == 200, added.text
    written = yaml.safe_load(path.read_text())
    assert written["catalog"]["colours"]["variants"][0]["spec"]["outputs"]["label"]["labels"] == ["red", "green", "blue"]
    assert "colours" in written["pool"]["model_set"]
    assert next(h for h in written["hosts"] if h["id"] == "box")["models"] == ["colours"]
    assert supervisor.config.catalog["colours"].variants[0].engine == "onnx"
    assert api.post("/pool/models/trained", json={"name": "colours", "build": build}).status_code == 409


def test_where_the_declarations_and_the_file_disagree_it_is_said_before_saving(pool):
    _, api, _ = pool
    tag = api.post("/pool/models/uploads", content=MODEL).json()["tag"]
    wrong = {"inputs": {"image": {**SPEC["inputs"]["image"], "size": [8, 8], "channels": "gray"}},
             "outputs": {"label": {"as": "scores", "output": "logits", "labels": ["a", "b"]},
                         "extra": {"as": "vector", "output": "embedding"}}}
    tested = api.post("/pool/models/test", json={"build": {"tag": tag, "engine": "onnx", "spec": wrong}}).json()
    assert not tested["ok"]
    said = " | ".join(tested["problems"])
    assert "3 channels, the declaration 1" in said and "4×4 pictures" in said
    assert "3 scores, the declaration 2 labels" in said and "'embedding', which the file has not" in said


def test_a_build_on_the_wrong_engines_host_or_a_container_while_they_are_off_is_refused(pool):
    _, api, _ = pool
    tag = api.post("/pool/models/uploads", content=MODEL).json()["tag"]
    build = {"tag": tag, "engine": "onnx", "spec": SPEC}
    wrong_host = api.post("/pool/models/trained", json={"name": "colours", "build": build, "hosts": ["mac"]})
    assert wrong_host.status_code == 400 and wrong_host.json()["error"] == "wrong_engine"
    image = {"tag": "registry.example/me/x@sha256:" + "a" * 64, "engine": "container", "port": 8080}
    assert "containers are off" in api.post("/pool/models/test", json={"build": image}).json()["problems"][0]
    refused = api.post("/pool/models/trained", json={"name": "boxes", "build": image})
    assert refused.status_code == 400 and "containers are off" in refused.text
    assert api.post("/pool/models/test", json={"build": {"tag": "x", "engine": "vllm"}}).status_code == 400


def test_a_hub_file_is_read_through_the_hub(pool):
    supervisor, api, _ = pool

    async def hub(request):
        if request.url.path == "/owner/colours/resolve/main/model.onnx":
            return httpx.Response(200, content=MODEL)
        return httpx.Response(404)

    supervisor._hub_transport = httpx.MockTransport(hub)
    seen = api.post("/pool/models/inspect", json={"hub": "owner/colours/model.onnx"})
    assert seen.status_code == 200 and seen.json()["outputs"][0]["name"] == "logits"
    assert api.post("/pool/models/inspect", json={"hub": "owner/missing/model.onnx"}).status_code == 404
    assert api.post("/pool/models/inspect", json={"hub": "../etc/passwd.onnx"}).status_code == 400
