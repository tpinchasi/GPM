"""Trained models (S15): one path, `POST /v1/predict`, served by an ONNX file (D179) or a container
image (D180) that the host's agent runs, and declared in the catalog (D178).

Every test runs with no GPU, no Docker and no model hub: the server is a fake, and the agent is a
stand-in that says what the machine holds.
"""

import json
import time

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, pool_harness
from gpm_server.config import PoolConfig

CLASSIFIER = "pets"
ONNX_FILE = "owner/pets-classifier/model.onnx"
DETECTOR = "boxes"
IMAGE = "registry.example/me/detector@sha256:" + "a" * 64
CHAT = "chatter"

SPEC = {
    "inputs": {"image": {"as": "image", "input": "pixel_values", "size": [224, 224], "channels": "rgb",
                         "layout": "nchw", "scale": 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
    "outputs": {"label": {"as": "scores", "output": "logits", "softmax": True, "top_k": 2,
                          "labels": ["cat", "dog", "bird"]}},
}
ONNX_BUILD = {"tag": ONNX_FILE, "engine": "onnx", "max_batch": 8, "spec": SPEC,
              "needs": {"card": True}}
CONTAINER_BUILD = {"tag": IMAGE, "engine": "container", "port": 8080, "needs": {"memory_gb": 4, "card": True}}


class Agent:
    """The host's agent as the pool hears it: what it was asked to hold, and where it stands."""

    def __init__(self, engine: str):
        self.engine = engine
        self.asked: list[dict] = []
        self.models: list[dict] = []
        self.transport = httpx.MockTransport(self._answer)

    def _answer(self, request):
        if request.url.path.endswith("/facts"):
            return httpx.Response(200, json={
                "agent": {"version": "test", "protocol": "1"}, "os": "Linux", "arch": "x86_64",
                "capabilities": [], "engine": {"name": self.engine, "answers": True},
            })
        self.asked.append(json.loads(request.content))
        return httpx.Response(200, json={"engine_answers": True, "models": self.models, "surplus": []})


def trained_pool(monkeypatch, *, engine="onnx", build=ONNX_BUILD, model=CLASSIFIER, containers=False):
    monkeypatch.setenv("TRAINED_AGENT_KEY", "test-agent-key")
    return pool_harness(
        [EngineSpec(id="box", resident=set(), engine=engine),
         EngineSpec(id="mac", resident={"chatter:7b"})],
        model_set=[model, CHAT],
        catalog={model: {"variants": [build]}, CHAT: {"variants": [{"tag": "chatter:7b"}]}},
        pool_settings={"models_per_host": "declared"},
        host_overrides={
            "box": {"models": [model], "agent": {"url": "http://127.0.0.1:8096", "bearer_env": "TRAINED_AGENT_KEY"}},
            "mac": {"models": [CHAT]},
        },
        extra_config={"containers": {"enabled": containers}},
    )


def make_ready(pool, agent, tag):
    pool.supervisor._agent_transport = agent.transport
    pool.engines["box"].fake.start({tag})
    agent.models = [{"tag": tag, "on_disk": True, "loaded": True}]
    pool.reprobe()
    assert pool.supervisor.hosts["box"].state.value == "ready", pool.supervisor.hosts["box"].last_error


def last_logged(pool):
    for _ in range(100):
        rows = pool.database.query("SELECT kind, outcome FROM request_log ORDER BY id DESC LIMIT 1")
        if rows:
            return rows[0]
        time.sleep(0.02)
    raise AssertionError("nothing was logged")


# --- asking a trained model ---


def test_an_onnx_model_is_asked_on_its_path_and_its_agent_is_told_what_it_takes(monkeypatch):
    with trained_pool(monkeypatch) as pool:
        agent = Agent("onnx")
        pool.supervisor._agent_transport = agent.transport
        agent.models = [{"tag": ONNX_FILE, "on_disk": True, "loaded": False}]
        pool.reprobe()
        host = pool.supervisor.hosts["box"]
        assert host.state.value == "preparing", "a server answering is not one holding the build"
        said = agent.asked[-1]
        assert said["tags"] == [ONNX_FILE] and said["residency"] == "pinned"
        assert said["specs"][ONNX_FILE] == {"engine": "onnx", "spec": SPEC, "max_batch": 8, "needs": {"card": True}}

        make_ready(pool, agent, ONNX_FILE)
        with pool.client() as client:
            response = client.post("/v1/predict", json={"model": CLASSIFIER, "inputs": {"image": "aGVsbG8="}})
        assert response.status_code == 200, response.text
        assert response.json() == {"model": ONNX_FILE, "outputs": {"seen": ["image"]}}
        path, body = pool.engines["box"].fake.received[-1]
        assert path == "/v1/predict" and body["model"] == ONNX_FILE, "the build, as for any engine"
        assert body["inputs"] == {"image": "aGVsbG8="}, "inputs pass as the app sent them"
        assert last_logged(pool)["kind"] == "predict"

        from gpm_client import PoolClient
        with PoolClient(base_url=pool.url + "/v1", api_key=APP_KEY) as sdk:
            prediction = sdk.predict(CLASSIFIER, {"image": b"hello"})
        assert prediction["seen"] == ["image"] and prediction.served_model == ONNX_FILE and prediction.host == "box"
        assert pool.engines["box"].fake.received[-1][1]["inputs"] == {"image": "aGVsbG8="}, "bytes travel as base64"


def test_a_container_model_is_asked_the_same_way(monkeypatch):
    with trained_pool(monkeypatch, engine="container", build=CONTAINER_BUILD, model=DETECTOR, containers=True) as pool:
        agent = Agent("container")
        make_ready(pool, agent, IMAGE)
        assert agent.asked[-1]["specs"] == {IMAGE: {"engine": "container", "port": 8080, "needs": {"memory_gb": 4, "card": True}}}
        with pool.client() as client:
            response = client.post("/v1/predict", json={"model": DETECTOR, "inputs": {"frame": [1, 2, 3]}})
        assert response.status_code == 200, response.text
        assert pool.engines["box"].fake.received[-1][1] == {"model": IMAGE, "inputs": {"frame": [1, 2, 3]}}


def test_a_chat_model_is_never_asked_to_predict_nor_a_trained_one_to_chat(monkeypatch):
    with trained_pool(monkeypatch) as pool:
        make_ready(pool, Agent("onnx"), ONNX_FILE)
        with pool.client() as client:
            wrong_path = client.post("/v1/predict", json={"model": CHAT, "inputs": {"x": 1}})
            wrong_model = client.post("/v1/chat/completions",
                                      json={"model": CLASSIFIER, "messages": [{"role": "user", "content": "hi"}]})
        assert wrong_path.status_code == 400 and wrong_path.json()["reason"] == "kind_not_served"
        assert wrong_model.status_code == 400 and wrong_model.json()["reason"] == "kind_not_served"
        assert pool.engines["box"].fake.received == []


def test_a_model_named_twice_is_refused_before_any_host_sees_it(monkeypatch):
    with trained_pool(monkeypatch) as pool:
        make_ready(pool, Agent("onnx"), ONNX_FILE)
        with pool.client() as client:
            response = client.post("/v1/predict", content=b'{"model": "pets", "Model": "chatter", "inputs": {}}',
                                   headers={"Content-Type": "application/json"})
        assert response.status_code == 400, response.text
        assert pool.engines["box"].fake.received == []


def test_a_trained_model_waits_while_its_build_is_still_arriving(monkeypatch):
    with trained_pool(monkeypatch) as pool:
        agent = Agent("onnx")
        pool.supervisor._agent_transport = agent.transport
        agent.models = [{"tag": ONNX_FILE, "on_disk": False, "loaded": False,
                         "pulling": {"tag": ONNX_FILE, "completed_bytes": 10, "total_bytes": 40}}]
        pool.reprobe()
        host = pool.supervisor.hosts["box"]
        assert host.state.value == "preparing" and "25%" in host.last_error
        with pool.client() as client:
            response = client.post("/v1/predict", json={"model": CLASSIFIER, "inputs": {}},
                                   headers={"X-GPM-Max-Wait": "0"})
        assert response.status_code == 503


# --- what configuration accepts ---


def config(catalog, hosts=None, **extra):
    base = {
        "pool": {"name": "test", "model_set": list(catalog), "models_per_host": "declared"},
        "auth": {"app_keys": ["k"]},
        "catalog": catalog,
        "hosts": hosts or [
            {"id": "box", "kind": "local", "engine": next(iter(catalog.values()))["variants"][0].get("engine", "ollama"),
             "models": list(catalog), "transport": {"type": "http", "base_url": "http://127.0.0.1:8190"},
             "agent": {"url": "http://127.0.0.1:8096", "bearer_env": "AGENT_KEY"}},
        ],
    }
    base.update(extra)
    return PoolConfig.model_validate(base)


def test_a_valid_trained_catalog_loads():
    cfg = config({CLASSIFIER: {"variants": [ONNX_BUILD]}})
    assert cfg.catalog[CLASSIFIER].variants[0].kinds() == frozenset({"predict"})
    cfg = config({DETECTOR: {"variants": [CONTAINER_BUILD]}}, containers={"enabled": True})
    assert cfg.trained_build_specs([IMAGE])[IMAGE]["port"] == 8080


@pytest.mark.parametrize("change, said", [
    ({"spec": None}, "says what its inputs and outputs are"),
    ({"port": 9000}, "`port` is a container build's"),
    ({"tag": "owner/pets-classifier/model.bin"}, "`owner/repo/file.onnx`"),
    ({"serves": ["chat"]}, "serves `predict` alone"),
    ({"price": {"input": 1, "output": 1}}, "are not a onnx build's"),
])
def test_an_onnx_build_that_is_not_one_is_refused_in_words(change, said):
    build = {**ONNX_BUILD, **change}
    build = {k: v for k, v in build.items() if v is not None}
    with pytest.raises(ValueError, match=said.replace("(", r"\(").replace("`", "`")):
        config({CLASSIFIER: {"variants": [build]}})


def test_an_image_input_says_its_size_and_one_mean_per_channel():
    for broken, said in [({"size": None}, "needs"), ({"mean": [0.5]}, "one value per channel"),
                         ({"size": [0, 224]}, "1 to 8192"), ({"names": ["a"]}, "`names` is for")]:
        spec = json.loads(json.dumps(SPEC))
        spec["inputs"]["image"].update(broken)
        spec["inputs"]["image"] = {k: v for k, v in spec["inputs"]["image"].items() if v is not None}
        with pytest.raises(ValueError, match=said):
            config({CLASSIFIER: {"variants": [{**ONNX_BUILD, "spec": spec}]}})


def test_features_and_scores_say_what_they_need():
    spec = {"inputs": {"row": {"as": "features", "input": "x"}}, "outputs": {"y": {"as": "value", "output": "y"}}}
    with pytest.raises(ValueError, match="needs `names`"):
        config({CLASSIFIER: {"variants": [{**ONNX_BUILD, "spec": spec}]}})
    spec = {"inputs": {"row": {"as": "features", "input": "x", "names": ["age", "income"]}},
            "outputs": {"y": {"as": "value", "output": "y", "labels": ["a"]}}}
    with pytest.raises(ValueError, match="for `as: scores` only"):
        config({CLASSIFIER: {"variants": [{**ONNX_BUILD, "spec": spec}]}})


def test_a_container_build_runs_only_where_containers_are_on():
    with pytest.raises(ValueError, match="containers are off"):
        config({DETECTOR: {"variants": [CONTAINER_BUILD]}})


@pytest.mark.parametrize("tag", ["registry.example/me/detector:latest", "registry.example/me/detector",
                                 "registry.example/me/../x@sha256:" + "a" * 64])
def test_a_container_image_is_named_by_digest_only(tag):
    with pytest.raises(ValueError, match="by digest"):
        config({DETECTOR: {"variants": [{**CONTAINER_BUILD, "tag": tag}]}}, containers={"enabled": True})


def test_a_container_build_names_its_port_and_takes_no_spec():
    with pytest.raises(ValueError, match="names the port"):
        config({DETECTOR: {"variants": [{k: v for k, v in CONTAINER_BUILD.items() if k != "port"}]}},
               containers={"enabled": True})
    with pytest.raises(ValueError, match="an ONNX build's"):
        config({DETECTOR: {"variants": [{**CONTAINER_BUILD, "spec": SPEC}]}}, containers={"enabled": True})


def test_only_trained_engines_serve_predict_and_their_fields_are_theirs():
    with pytest.raises(ValueError, match="only `engine: onnx` and `engine: container`"):
        config({CHAT: {"variants": [{"tag": "owner/repo", "engine": "vllm", "serves": ["predict"]}]}},
               hosts=[{"id": "gpu", "kind": "local", "engine": "vllm", "models": [CHAT],
                       "transport": {"type": "http", "base_url": "http://127.0.0.1:8000"}}])
    with pytest.raises(ValueError, match="are for a trained model's build"):
        config({CHAT: {"variants": [{"tag": "chatter:7b", "max_batch": 4}]}},
               hosts=[{"id": "mac", "kind": "local", "models": [CHAT],
                       "transport": {"type": "http", "base_url": "http://127.0.0.1:11434"}}])


def test_a_trained_models_host_carries_an_agent_and_is_reached_on_loopback():
    hosts = [{"id": "box", "kind": "local", "engine": "onnx", "models": [CLASSIFIER],
              "transport": {"type": "http", "base_url": "http://127.0.0.1:8190"}}]
    with pytest.raises(ValueError, match="give the host an `agent`"):
        config({CLASSIFIER: {"variants": [ONNX_BUILD]}}, hosts=hosts)
    hosts[0]["agent"] = {"url": "https://10.0.0.5:8096", "bearer_env": "AGENT_KEY"}
    hosts[0]["transport"] = {"type": "https", "base_url": "https://10.0.0.5:8190"}
    with pytest.raises(ValueError, match="no authentication"):
        config({CLASSIFIER: {"variants": [ONNX_BUILD]}}, hosts=hosts)


def test_a_trained_models_host_holds_only_builds_marked_for_its_engine():
    hosts = [{"id": "box", "kind": "local", "engine": "onnx", "models": [CHAT],
              "transport": {"type": "http", "base_url": "http://127.0.0.1:8190"},
              "agent": {"url": "http://127.0.0.1:8096", "bearer_env": "AGENT_KEY"}}]
    with pytest.raises(ValueError, match="no build marked `engine: onnx`"):
        config({CHAT: {"variants": [{"tag": "chatter:7b"}]}}, hosts=hosts)


# --- turning containers on (D180) ---


def test_turning_containers_on_is_confirmed_by_the_pools_name_and_recorded(tmp_path):
    from fakes.harness import BackgroundLoop, ServerHandle
    from gpm_server.config import load_config
    from gpm_server.db import Database
    from gpm_server.supervisor import Supervisor
    from gpm_server.supervisor.control import create_control_app

    path = tmp_path / "pool.yaml"
    path.write_text(
        "pool: {name: trained, model_set: [chatter]}\n"
        f"request_log: {tmp_path / 'gpm.sqlite3'}\n"
        "auth: {admin_keys: [gpmx_trained_admin], app_keys: [gpma_trained_app]}\n"
        "hosts:\n"
        "  - {id: mac, kind: local, transport: {type: http, base_url: 'http://127.0.0.1:1'}}\n"
    )
    loop, database = BackgroundLoop(), Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    control = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        with httpx.Client(base_url=control.base_url, timeout=30,
                          headers={"Authorization": "Bearer gpmx_trained_admin"}) as api:
            unconfirmed = api.patch("/pool/containers", json={"enabled": True})
            assert unconfirmed.status_code == 400 and unconfirmed.json()["error"] == "not_confirmed", unconfirmed.text
            assert not supervisor.config.containers.enabled
            current = api.get("/pool/config").json()
            raw = api.put("/pool/config", json={"text": current["text"] + "containers: {enabled: true}\n",
                                                "version": current["version"]})
            assert raw.status_code == 400 and raw.json()["error"] == "not_confirmed", "raw text is held to it too"
            confirmed = api.patch("/pool/containers", json={"enabled": True, "confirm": "trained"})
            assert confirmed.status_code == 200, confirmed.text
            assert supervisor.config.containers.enabled
            assert "containers:\n  enabled: true" in path.read_text()
            off = api.patch("/pool/containers", json={"enabled": False})
            assert off.status_code == 200 and not supervisor.config.containers.enabled, "off needs no confirming"
        kinds = [row["kind"] for row in database.query("SELECT kind FROM events ORDER BY id")]
        assert "containers_enabled" in kinds and "containers_disabled" in kinds
    finally:
        control.stop()
        loop.stop()
        database.close()


def test_one_file_is_declared_one_way_and_an_onnx_builds_needs_are_a_card_alone():
    """Found in review: two models naming one file with different declarations — the agent is told
    one per file, so one of them would be served with the other's."""
    raw = {**ONNX_BUILD, "spec": {"inputs": SPEC["inputs"], "outputs": {"raw": {"as": "tensor", "output": "logits"}}}}
    with pytest.raises(ValueError, match="declare it differently"):
        config({CLASSIFIER: {"variants": [ONNX_BUILD]}, "pets-raw": {"variants": [raw]}})
    config({CLASSIFIER: {"variants": [ONNX_BUILD]}, "pets-too": {"variants": [dict(ONNX_BUILD)]}})  # alike: fine
    with pytest.raises(ValueError, match="`card` alone"):
        config({CLASSIFIER: {"variants": [{**ONNX_BUILD, "needs": {"memory_gb": 1}}]}})


def test_a_rollback_to_a_version_with_containers_on_is_confirmed_by_the_pools_name(tmp_path):
    """Found in review: rolling back to an earlier version turned containers on unasked (D180)."""
    from fakes.harness import BackgroundLoop, ServerHandle
    from gpm_server.config import load_config
    from gpm_server.db import Database
    from gpm_server.supervisor import Supervisor
    from gpm_server.supervisor.control import create_control_app

    path = tmp_path / "pool.yaml"
    path.write_text(
        "pool: {name: trained, model_set: [chatter]}\n"
        f"request_log: {tmp_path / 'gpm.sqlite3'}\n"
        "auth: {admin_keys: [gpmx_trained_admin], app_keys: [gpma_trained_app]}\n"
        "hosts:\n  - {id: mac, kind: local, transport: {type: http, base_url: 'http://127.0.0.1:1'}}\n"
    )
    loop, database = BackgroundLoop(), Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(path), database, config_path=str(path))
    control = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        with httpx.Client(base_url=control.base_url, timeout=30, headers={"Authorization": "Bearer gpmx_trained_admin"}) as api:
            assert api.patch("/pool/containers", json={"enabled": True, "confirm": "trained"}).status_code == 200
            with_containers = api.get("/pool/config").json()["version"]
            assert api.patch("/pool/containers", json={"enabled": False}).status_code == 200
            asked = api.post("/pool/config/rollback", json={"version": with_containers})
            assert asked.status_code == 400 and asked.json()["error"] == "not_confirmed"
            assert not supervisor.config.containers.enabled
            assert api.post("/pool/config/rollback", json={"version": with_containers, "confirm": "trained"}).status_code == 200
            assert supervisor.config.containers.enabled
    finally:
        control.stop()
        loop.stop()
        database.close()


def test_gpm_config_apply_with_yes_confirms_turning_containers_on(tmp_path, monkeypatch):
    """Found in review: the CLI never sent `confirm`, so it could not turn containers on at all."""
    import types

    import httpx as real_httpx
    from gpm_server import cli

    candidate = tmp_path / "pool.yaml"
    candidate.write_text("containers: {enabled: true}\n")
    plan = {"changes": [{"kind": "containers_enabled", "detail": "containers are turned on", "requires_retype": True,
                         "value": "shop"}]}
    monkeypatch.setattr(real_httpx, "get", lambda *a, **k: types.SimpleNamespace(json=lambda: {"version": "v1"}))
    monkeypatch.setattr(real_httpx, "post", lambda *a, **k: types.SimpleNamespace(json=lambda: plan))
    sent = []
    monkeypatch.setattr(cli, "_control", lambda args, method, path, body=None: sent.append((method, path, body)) or 0)
    args = types.SimpleNamespace(action="apply", file=str(candidate), yes=False, url="http://pool")
    assert cli._config(args) == 1 and sent == [], "not without --yes"
    args.yes = True
    assert cli._config(args) == 0
    assert sent == [("PUT", "/pool/config", {"text": "containers: {enabled: true}\n", "version": "v1", "confirm": "shop"})]
