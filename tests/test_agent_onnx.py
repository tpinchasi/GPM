"""The host agent's `onnx` engine (S15, D179): it fetches an ONNX build from the hub, runs the pool's
own server on loopback with every build it holds, and reports what is loaded and why not.

The server is the real one, run with this test's own Python — which has ONNX Runtime — as an owner
would name it (`engine_binary`); the hub is a stand-in serving a real model file.
"""

import asyncio
import json
import socket
import sys

import httpx
import pytest

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from gpm_agent.models import ModelWork, Refused  # noqa: E402
from gpm_agent.onnx_engine import OnnxFacts, build_of  # noqa: E402
from gpm_agent.settings import Settings  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

DOUBLER, ADDER = "owner/doubler/model.onnx", "owner/adder/model.onnx"


def model_bytes(op: str) -> bytes:
    nodes = [helper.make_node(op, ["t", "t"], ["v"])]
    graph = helper.make_graph(nodes, op, [helper.make_tensor_value_info("t", TensorProto.FLOAT, ["N", 2])],
                              [helper.make_tensor_value_info("v", TensorProto.FLOAT, ["N", 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


FILES = {"owner/doubler": {"model.onnx": model_bytes("Add")}, "owner/adder": {"model.onnx": model_bytes("Mul")}}
SPEC = {"engine": "onnx", "spec": {"inputs": {"t": {"as": "tensor", "input": "t"}},
                                   "outputs": {"v": {"as": "vector", "output": "v"}}}, "max_batch": 4}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def hub() -> httpx.MockTransport:
    def answer(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for repo, files in FILES.items():
            if path == f"/api/models/{repo}/tree/main":
                return httpx.Response(200, json=[{"type": "file", "path": n, "size": len(b)} for n, b in files.items()])
            if path.startswith(f"/{repo}/resolve/main/"):
                body = files.get(path.rsplit("/", 1)[-1])
                return httpx.Response(200, content=body) if body is not None else httpx.Response(404)
        return httpx.Response(404)

    return httpx.MockTransport(answer)


@pytest.fixture
def machine(tmp_path):
    settings = Settings(key_hash="x", engine="onnx", engine_url=f"http://127.0.0.1:{free_port()}",
                        models_path=str(tmp_path / "models"), engine_binary=sys.executable)
    settings.check()
    engine = OnnxFacts(settings, hub_transport=hub())
    yield engine
    asyncio.run(engine._stop())


def client(engine: OnnxFacts) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=engine.settings.engine_url, timeout=30)


async def fetched(engine, *tags):
    async with client(engine) as c:
        for tag in tags:
            async for _ in engine.pull(c, tag):
                pass


async def ask(engine, tag, values):
    async with client(engine) as c:
        answer = await c.post("/v1/predict", json={"model": tag, "inputs": {"t": values}})
    return answer.json()


def test_a_build_is_fetched_served_and_reported_loaded(machine):
    async def scenario():
        await fetched(machine, DOUBLER)
        machine.set_specs({DOUBLER: OnnxFacts.checked_spec(DOUBLER, SPEC)})
        async with client(machine) as c:
            await machine.hold(c, DOUBLER, pinned=True)
            facts = await machine.describe(c)
        assert facts["answers"] and facts["models_loaded"] == [DOUBLER]
        assert [m["tag"] for m in facts["models_on_disk"]] == [DOUBLER]
        assert (await ask(machine, DOUBLER, [1.5, 2]))["outputs"] == {"v": [3.0, 4.0]}

    asyncio.run(scenario())


def test_one_server_holds_every_build_and_is_started_again_when_the_set_changes(machine):
    async def scenario():
        await fetched(machine, DOUBLER, ADDER)
        machine.set_specs({tag: OnnxFacts.checked_spec(tag, SPEC) for tag in (DOUBLER, ADDER)})
        async with client(machine) as c:
            await machine.hold(c, DOUBLER, pinned=True)
            first = machine._current()["pid"]
            await machine.hold(c, ADDER, pinned=True)
            both = machine._current()
            assert both["pid"] != first and both["tags"] == sorted([DOUBLER, ADDER])
            assert sorted((await machine.describe(c))["models_loaded"]) == sorted([DOUBLER, ADDER])
            assert (await ask(machine, ADDER, [3, 4]))["outputs"] == {"v": [9.0, 16.0]}
            await machine.hold(c, ADDER, pinned=True)
            assert machine._current()["pid"] == both["pid"], "the same set: the same server"
            await machine.keep_only(c, [ADDER])
            assert machine._current()["tags"] == [ADDER]
            await machine.keep_only(c, [])
            assert machine._current() is None, "nothing to serve: stopped"

    asyncio.run(scenario())


def test_a_build_that_does_not_match_its_declarations_is_that_builds_failure(machine):
    async def scenario():
        await fetched(machine, DOUBLER)
        wrong = {**SPEC, "spec": {"inputs": {"t": {"as": "tensor", "input": "x"}}, "outputs": SPEC["spec"]["outputs"]}}
        machine.set_specs({DOUBLER: OnnxFacts.checked_spec(DOUBLER, wrong)})
        async with client(machine) as c:
            with pytest.raises(Exception, match="names 'x', which the file has not"):
                await machine.hold(c, DOUBLER, pinned=True)
            facts = await machine.describe(c)
        assert facts["models_loaded"] == [] and "'x'" in facts["models_failed"][DOUBLER]

    asyncio.run(scenario())


def test_nothing_runs_without_a_program_and_it_says_what_to_set(tmp_path):
    settings = Settings(key_hash="x", engine="onnx", engine_url=f"http://127.0.0.1:{free_port()}",
                        models_path=str(tmp_path / "models"))
    engine = OnnxFacts(settings, hub_transport=hub())

    async def scenario():
        await fetched(engine, DOUBLER)
        engine.set_specs({DOUBLER: OnnxFacts.checked_spec(DOUBLER, SPEC)})
        async with client(engine) as c:
            with pytest.raises(Exception, match="install_engines"):
                await engine.hold(c, DOUBLER, pinned=True)

    asyncio.run(scenario())


def test_an_uploaded_build_waits_for_the_pool_to_send_it(machine):
    tag = "upload/" + "c" * 64
    assert build_of(tag).uploaded

    async def scenario():
        async with client(machine) as c:
            with pytest.raises(Exception, match="has not arrived"):
                async for _ in machine.pull(c, tag):
                    pass

    asyncio.run(scenario())


@pytest.mark.parametrize("specs, said", [
    ({DOUBLER: {"engine": "container", "port": 1}}, "not an ONNX build"),
    ({DOUBLER: {**SPEC, "max_batch": 0}}, "max_batch"),
    ({DOUBLER: {**SPEC, "command": "rm -rf /"}}, "not an ONNX build"),
    ({ADDER: SPEC}, "not in the desired set"),
    ("not an object", "specs are an object"),
])
def test_declarations_the_agent_is_sent_are_checked_as_data(machine, specs, said):
    work = ModelWork(machine, httpx.AsyncClient(base_url=machine.settings.engine_url),
                     free_disk_bytes=lambda: 10**12, min_free_disk_gb=1, allow_delete=False)

    async def scenario():
        with pytest.raises(Refused) as refused:
            await work.apply({"tags": [DOUBLER], "residency": "pinned", "specs": specs})
        assert refused.value.error == "bad_specs" and said in refused.value.detail

    asyncio.run(scenario())


def test_an_engine_without_declarations_refuses_them(tmp_path):
    from gpm_agent.engines import engine_facts

    ollama = engine_facts("ollama", Settings(key_hash="x", engine="ollama"))
    work = ModelWork(ollama, httpx.AsyncClient(base_url="http://127.0.0.1:1"),
                     free_disk_bytes=lambda: 10**12, min_free_disk_gb=1, allow_delete=False)

    async def scenario():
        with pytest.raises(Refused) as refused:
            await work.apply({"tags": ["llama:7b"], "residency": "pinned", "specs": {"llama:7b": SPEC}})
        assert "takes no declarations" in refused.value.detail

    asyncio.run(scenario())


def test_the_registry_knows_the_onnx_engine():
    from gpm_agent.engines import engine_facts

    assert isinstance(engine_facts("onnx", Settings(key_hash="x", engine="onnx", engine_url="http://127.0.0.1:8190")), OnnxFacts)
    assert json.dumps(OnnxFacts.checked_spec(DOUBLER, SPEC))  # plain data, all the way down
