"""The pool's own ONNX server (D179), on real ONNX models run by ONNX Runtime on the processor.

The models are built here, a few operators each, so every test runs with no GPU and no hub.
"""

import base64
import concurrent.futures
import io
import json
import threading

import httpx
import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
Image = pytest.importorskip("PIL.Image")

from gpm_agent import onnx_server  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

CPU = ["CPUExecutionProvider"]


def save(model, path):
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def classifier(path):
    """Pictures [N, 3, 4, 4] → three scores: each channel's mean (red, green, blue)."""
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["logits"], axes=[2, 3], keepdims=0)],
        "classifier",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["N", 3, 4, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 3])],
    )
    return save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def regression(path):
    """Features [N, 2] → one value: 2·age + 3·income."""
    weights = helper.make_tensor("w", TensorProto.FLOAT, [2, 1], [2.0, 3.0])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])], "regression",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])], [weights],
    )
    return save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def identity(path):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["t"], ["v"])], "identity",
        [helper.make_tensor_value_info("t", TensorProto.FLOAT, ["N", 3])],
        [helper.make_tensor_value_info("v", TensorProto.FLOAT, ["N", 3])],
    )
    return save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


IMAGE_SPEC = {
    "inputs": {"image": {"as": "image", "input": "pixel_values", "size": [4, 4], "channels": "rgb",
                         "layout": "nchw", "scale": 255}},
    "outputs": {"label": {"as": "scores", "output": "logits", "softmax": True, "top_k": 2,
                          "labels": ["red", "green", "blue"]},
                "raw": {"as": "tensor", "output": "logits"}},
}
FEATURES_SPEC = {
    "inputs": {"row": {"as": "features", "input": "x", "names": ["age", "income"]}},
    "outputs": {"price": {"as": "value", "output": "y"}},
}
TENSOR_SPEC = {"inputs": {"t": {"as": "tensor", "input": "t"}}, "outputs": {"v": {"as": "vector", "output": "v"}}}


def picture(colour, size=(16, 16), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def server(tmp_path):
    plan = {
        "owner/pets/model.onnx": {"path": classifier(tmp_path / "c.onnx"), "spec": IMAGE_SPEC, "max_batch": 8},
        "upload/" + "b" * 64: {"path": regression(tmp_path / "r.onnx"), "spec": FEATURES_SPEC},
        "owner/id/model.onnx": {"path": identity(tmp_path / "i.onnx"), "spec": TENSOR_SPEC, "max_batch": 4},
    }
    running = onnx_server.serve(plan, 0, CPU)
    thread = threading.Thread(target=running.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(base_url=f"http://127.0.0.1:{running.server_address[1]}", timeout=30) as client:
        yield client
    running.shutdown()


def predict(client, model, inputs):
    return client.post("/v1/predict", json={"model": model, "inputs": inputs})


def test_it_answers_on_loopback_and_lists_what_it_loaded(server):
    assert server.get("/health").json() == {"ok": True}
    rows = server.get("/v1/models").json()["data"]
    assert {row["id"] for row in rows if row["loaded"]} == {"owner/pets/model.onnx", "upload/" + "b" * 64, "owner/id/model.onnx"}


@pytest.mark.parametrize("colour, best", [((250, 10, 10), "red"), ((10, 250, 10), "green"), ((10, 10, 250), "blue")])
def test_a_picture_is_prepared_as_declared_and_its_best_scores_labelled(server, colour, best):
    answer = predict(server, "owner/pets/model.onnx", {"image": picture(colour, size=(37, 23))})
    assert answer.status_code == 200, answer.text
    outputs = answer.json()["outputs"]
    assert [row["label"] for row in outputs["label"]][0] == best and len(outputs["label"]) == 2
    assert max(outputs["raw"]) == pytest.approx(max(colour) / 255, abs=0.02), "scaled by 255, as declared"


def test_a_jpeg_and_a_data_url_are_read_too(server):
    answer = predict(server, "owner/pets/model.onnx", {"image": "data:image/jpeg;base64," + picture((250, 10, 10), fmt="JPEG")})
    assert answer.status_code == 200, answer.text
    assert answer.json()["outputs"]["label"][0]["label"] == "red"


def test_features_are_put_in_the_models_order(server):
    answer = predict(server, "upload/" + "b" * 64, {"row": {"income": 10, "age": 1}})
    assert answer.status_code == 200, answer.text
    assert answer.json()["outputs"] == {"price": pytest.approx(2 * 1 + 3 * 10)}


def test_a_tensor_is_passed_as_numbers_and_a_vector_read_back(server):
    answer = predict(server, "owner/id/model.onnx", {"t": [1, 2.5, -3]})
    assert answer.json()["outputs"] == {"v": [1.0, 2.5, -3.0]}


def test_requests_that_arrive_together_are_answered_each_with_its_own(server):
    def one(i):
        return predict(server, "owner/id/model.onnx", {"t": [i, i, i]}).json()["outputs"]["v"]

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        answers = list(pool.map(one, range(40)))
    assert answers == [[float(i)] * 3 for i in range(40)]


@pytest.mark.parametrize("model, inputs, said", [
    ("owner/pets/model.onnx", {}, "missing"),
    ("owner/pets/model.onnx", {"image": "https://example.com/cat.png"}, "never as a link"),
    ("owner/pets/model.onnx", {"image": "not base64!"}, "not valid base64"),
    ("owner/pets/model.onnx", {"image": base64.b64encode(b"not a picture").decode()}, "could not be read"),
    ("owner/pets/model.onnx", {"image": picture((1, 2, 3)), "extra": 1}, "does not take"),
    ("upload/" + "b" * 64, {"row": {"age": 1}}, "missing"),
    ("upload/" + "b" * 64, {"row": {"age": "old", "income": 1}}, "is a number"),
    ("owner/id/model.onnx", {"t": [[1, 2, 3]]}, "dimensions"),
    ("owner/id/model.onnx", {"t": [1, [2], 3]}, "not"),
])
def test_what_a_request_gets_wrong_is_said_and_answered_400(server, model, inputs, said):
    answer = predict(server, model, inputs)
    assert answer.status_code == 400, answer.text
    assert said in answer.json()["error"]["message"]


def test_a_model_not_served_here_is_404_and_a_body_that_is_not_json_400(server):
    assert predict(server, "owner/other/model.onnx", {}).status_code == 404
    assert server.post("/v1/predict", content=b"{not json").status_code == 400


def test_a_picture_that_decodes_to_too_many_pixels_is_refused(server, monkeypatch):
    monkeypatch.setattr(onnx_server, "MAX_PIXELS", 100)
    answer = predict(server, "owner/pets/model.onnx", {"image": picture((1, 2, 3), size=(64, 64))})
    assert answer.status_code == 400 and "too large" in answer.json()["error"]["message"]


def test_a_build_whose_declarations_do_not_match_its_file_is_listed_as_not_loaded(tmp_path):
    wrong = {"inputs": {"image": {**IMAGE_SPEC["inputs"]["image"], "input": "pixels"}}, "outputs": IMAGE_SPEC["outputs"]}
    labels = {"inputs": IMAGE_SPEC["inputs"],
              "outputs": {"label": {"as": "scores", "output": "logits", "labels": ["one", "two"]}}}
    plan = {"a/b/wrong.onnx": {"path": classifier(tmp_path / "w.onnx"), "spec": wrong},
            "a/b/labels.onnx": {"path": classifier(tmp_path / "l.onnx"), "spec": labels}}
    running = onnx_server.serve(plan, 0, CPU)
    threading.Thread(target=running.serve_forever, daemon=True).start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{running.server_address[1]}", timeout=30) as client:
            rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
            assert not rows["a/b/wrong.onnx"]["loaded"] and "pixels" in rows["a/b/wrong.onnx"]["error"]
            assert predict(client, "a/b/wrong.onnx", {}).status_code == 503
            # Loaded means answered one made-up input: the label count is found wrong at load.
            assert not rows["a/b/labels.onnx"]["loaded"] and "3 scores and 2 labels" in rows["a/b/labels.onnx"]["error"]
            assert predict(client, "a/b/labels.onnx", {"image": picture((9, 9, 9))}).status_code == 503
    finally:
        running.shutdown()


def test_it_runs_as_a_program_from_a_plan_file(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"owner/id/model.onnx": {"path": identity(tmp_path / "i.onnx"), "spec": TENSOR_SPEC}}))
    import socket
    import subprocess
    import sys
    import time

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([sys.executable, "-I", onnx_server.__file__, "--port", str(port), "--plan", str(plan),
                                "--processor-only"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                answer = httpx.post(f"http://127.0.0.1:{port}/v1/predict", timeout=5,
                                    json={"model": "owner/id/model.onnx", "inputs": {"t": [4, 5, 6]}})
                break
            except httpx.ConnectError:
                assert time.monotonic() < deadline, "the server never came up"
                time.sleep(0.1)
        assert answer.json()["outputs"]["v"] == [4.0, 5.0, 6.0]
    finally:
        process.terminate()
        process.wait(timeout=10)
