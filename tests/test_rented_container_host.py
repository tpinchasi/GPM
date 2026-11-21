"""A rented machine for a container build (S16, D183): one image, started by the build's own
`start` after the dead-man timer, no agent put on it, ready when its server answers; a private
image's registry login handed only to a provider's own create call.

The real supervisor, fake provider and router; the image's server is the fake predict server."""

import json
import time

import pytest
from fakes.harness import EngineSpec, pool_harness
from gpm_server.config import PoolConfig
from gpm_server.registry_logins import RegistryLogin, RegistryLogins, registry_of
from test_trained_add import pool  # noqa: F401 - the control API over a pool from a file
from test_trained_models import CHAT, DETECTOR, IMAGE

BUILD = {"tag": IMAGE, "engine": "container", "port": 8080, "start": ["python", "-m", "detector", "--port", "8080"]}
RENTED = {"provider": "fake", "engine": "container", "workers": 2, "models": [DETECTOR],
          "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02},
          "scale": {"scale_up_after_s": 0}, "teardown": {"idle_minutes": 10}}


def config(build=BUILD, *, containers=True, **rented):
    return PoolConfig.model_validate({
        "pool": {"name": "t", "model_set": [DETECTOR], "models_per_host": "declared"},
        "auth": {"app_keys": ["gpma_t"]},
        "containers": {"enabled": containers},
        "catalog": {DETECTOR: {"variants": [build]}},
        "hosts": [],
        "rented": {**RENTED, **rented},
    })


# --- what the file says ---


def test_a_rented_container_host_starts_from_the_builds_image_on_its_port():
    cfg = config()
    assert cfg.rented_container_build().tag == IMAGE
    assert cfg.rented_image() == IMAGE and cfg.engine_port() == 8080


@pytest.mark.parametrize(("build", "rented", "said"), [
    ({k: v for k, v in BUILD.items() if k != "start"}, {}, "says no `start`"),
    (BUILD, {"image": "other/image:1"}, "remove rented.image"),
    (BUILD, {"engine_start": "serve"}, "remove rented.engine_start"),
    (BUILD, {"engine_port": 9000}, "remove rented.engine_port"),
    ({**BUILD, "start": ["serve\nrm -rf /"]}, {}, "one line of text"),
])
def test_what_a_rented_container_host_cannot_be_is_refused_in_words(build, rented, said):
    with pytest.raises(ValueError, match=said):
        config(build, **rented)


def test_rented_container_hosts_hold_one_model_and_need_containers_on():
    with pytest.raises(ValueError, match="containers are off"):
        config(containers=False)
    two = {"pool": {"name": "t", "model_set": [DETECTOR, "other"], "models_per_host": "declared"},
           "auth": {"app_keys": ["gpma_t"]}, "containers": {"enabled": True}, "hosts": [],
           "catalog": {DETECTOR: {"variants": [BUILD]},
                       "other": {"variants": [{**BUILD, "tag": "registry.example/me/other@sha256:" + "b" * 64}]}},
           "rented": {**RENTED, "models": [DETECTOR, "other"]}}
    with pytest.raises(ValueError, match="one image each, so they hold one model"):
        PoolConfig.model_validate(two)


# --- the logins ---


def test_a_login_is_kept_by_registry_and_never_said_back(tmp_path):
    logins = RegistryLogins(tmp_path / "registries")
    logins.put("GHCR.io", "me", "not-a-real-password")
    assert logins.listed() == [{"registry": "ghcr.io", "username": "me"}]
    assert logins.for_image("ghcr.io/me/model@sha256:" + "a" * 64) == RegistryLogin("ghcr.io", "me", "not-a-real-password")
    assert logins.for_image("docker.io/me/model@sha256:" + "a" * 64) is None
    assert "not-a-real-password" not in repr(logins.for_image("ghcr.io/me/x@sha256:" + "a" * 64))
    with pytest.raises(ValueError, match="registry's host name"):
        logins.put("https://ghcr.io/v2", "me", "x")
    assert logins.remove("ghcr.io") and logins.listed() == [] and not logins.remove("ghcr.io")


@pytest.mark.parametrize(("image", "registry"), [
    ("registry.example/me/detector@sha256:" + "a" * 64, "registry.example"),
    ("localhost:5000/me/x@sha256:" + "a" * 64, "localhost:5000"),
    ("me/detector@sha256:" + "a" * 64, "docker.io"),
])
def test_an_images_registry_is_read_from_its_name(image, registry):
    assert registry_of(image) == registry


# --- renting one ---


def rented_pool(**overrides):
    return pool_harness(
        [EngineSpec(id="mac", resident={"chatter:7b"})],
        host_overrides={"mac": {"models": [CHAT]}},
        rentable=[EngineSpec(id="market-1", resident=set(), engine="container")],
        model_set=[DETECTOR, CHAT],
        catalog={DETECTOR: {"variants": [BUILD]}, CHAT: {"variants": [{"tag": "chatter:7b"}]}},
        rented=RENTED,
        pool_settings={"models_per_host": "declared"},
        extra_config={"containers": {"enabled": True}},
        **overrides,
    )


def until_ready(rig, fleet, within=40):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        rig.reprobe()
        live = [h for h in fleet.hosts.values() if not h.released]
        if live and all(h.state == "ready" for h in live):
            return live
        time.sleep(0.1)
    raise AssertionError(f"not ready: {[(h.host_id, h.state, h.agent_detail) for h in fleet.hosts.values()]}; "
                         f"{[e['summary'] for e in rig.supervisor.events.recent(15)]}")


def test_a_private_image_is_rented_with_its_login_started_by_its_own_start_and_serves(monkeypatch):
    with rented_pool() as rig:
        supervisor, fleet = rig.supervisor, rig.supervisor.fleet
        supervisor.registry_logins().put("registry.example", "me", "not-a-real-password")
        server = rig.rentable["market-1"].fake
        server.start([IMAGE])  # the image's own server, as its `start` brings it up
        fleet.open_lease(workers=2, max_hours=2, max_spend=2.00, allow_rent=True)
        (host,) = until_ready(rig, fleet)

        provider = fleet.provider_for(host.offer)
        (instance,) = [i for i in provider.instances.values()]
        assert instance.spec.image == IMAGE
        login = provider.registry_logins[instance.instance_id]
        assert (login.registry, login.username, login.password) == ("registry.example", "me", "not-a-real-password")
        assert "not-a-real-password" not in json.dumps(instance.spec.env) + (instance.spec.onstart or "")
        onstart = instance.spec.onstart
        assert "nohup python -m detector --port 8080 >/var/log/gpm-container.log 2>&1 &" in onstart
        assert onstart.index("deadman.sh") < onstart.index("python -m detector"), "the timer is armed first"
        assert host.agent is None and "without the pool's agent" in (host.agent_detail or "")

        with rig.client() as client:
            answer = client.post("/v1/predict", json={"model": DETECTOR, "inputs": {"frame": "aGk="}})
        assert answer.status_code == 200, answer.text
        assert answer.headers["X-GPM-Host"] == host.host_id
        summaries = json.dumps(supervisor.events.recent(100))
        assert "not-a-real-password" not in summaries


def test_a_provider_that_takes_no_login_is_not_rented_from_for_a_private_image():
    with rented_pool() as rig:
        supervisor, fleet = rig.supervisor, rig.supervisor.fleet
        supervisor.registry_logins().put("registry.example", "me", "not-a-real-password")
        for provider in fleet.providers.values() if hasattr(fleet, "providers") else [fleet.provider_for(None)]:
            provider.capabilities = type(provider.capabilities)(**{**provider.capabilities.__dict__, "private_images": False})
        fleet.open_lease(workers=2, max_hours=2, max_spend=2.00, allow_rent=True)
        deadline = time.monotonic() + 5
        said = []
        while time.monotonic() < deadline and not said:
            rig.reprobe()
            said = [e["summary"] for e in supervisor.events.recent(50) if "takes no registry login" in e["summary"]]
            time.sleep(0.1)
        assert said and "registry.example needs one" in said[0]
        assert all(h.released for h in fleet.hosts.values())


def test_a_public_image_is_rented_with_no_login_at_all():
    with rented_pool() as rig:
        fleet = rig.supervisor.fleet
        rig.rentable["market-1"].fake.start([IMAGE])
        fleet.open_lease(workers=2, max_hours=2, max_spend=2.00, allow_rent=True)
        (host,) = until_ready(rig, fleet)
        assert fleet.provider_for(host.offer).registry_logins == {}


# --- the logins through the control API ---


def test_the_control_api_keeps_a_login_and_never_answers_its_password(pool):  # noqa: F811 - the fixture
    supervisor, api, _ = pool
    saved = api.put("/pool/containers/registry-logins",
                    json={"registry": "ghcr.io", "username": "me", "password": "not-a-real-password"})
    assert saved.status_code == 200 and "not-a-real-password" not in saved.text
    listed = api.get("/pool/containers/registry-logins")
    assert listed.json() == {"logins": [{"registry": "ghcr.io", "username": "me"}]}
    assert api.put("/pool/containers/registry-logins", json={"registry": "ghcr.io"}).status_code == 400
    assert supervisor.registry_login_for("ghcr.io/me/x@sha256:" + "a" * 64).password == "not-a-real-password"
    assert api.delete("/pool/containers/registry-logins/ghcr.io").status_code == 200
    assert api.delete("/pool/containers/registry-logins/ghcr.io").status_code == 404
    assert "not-a-real-password" not in json.dumps(supervisor.events.recent(50))
