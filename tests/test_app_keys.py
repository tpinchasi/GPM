"""Named app keys for the shared pool (D193), and the stats read over the control API (D196).

Several apps share the shared workload, each with a key of its own and a name the request log
carries — so each shows apart on the dashboards. A named key has exactly the app key's powers:
it requests completions, and reaches nothing on the control API. Against the real router and
control API, and fake engines; nothing here rents anything.
"""

import time

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, ServerHandle, pool_harness
from gpm_server.app_keys import AppKeyRefused, AppKeyStore
from gpm_server.supervisor.control import create_control_app

MODEL = "m1"
ADMIN_KEY = "gpmx_app_keys_admin"


@pytest.fixture
def pool():
    with pool_harness([EngineSpec(id="laptop", resident={MODEL}, workers=4)], model_set=[MODEL],
                      extra_config={"auth": {"app_keys": [APP_KEY], "admin_keys": [ADMIN_KEY]}}) as harness:
        server = ServerHandle(create_control_app(harness.supervisor, harness.config), harness.loop)
        harness.control_url = server.base_url
        try:
            yield harness
        finally:
            server.stop()


def control(pool, key=ADMIN_KEY):
    return httpx.Client(base_url=pool.control_url, headers={"Authorization": f"Bearer {key}"}, timeout=30)


def ask(client):
    return client.post("/v1/chat/completions", json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})


def follow(pool):
    pool.loop.run(pool.state.registry.refresh())


def test_a_named_key_reaches_the_shared_pool_and_its_requests_carry_its_name(pool):
    with control(pool) as admin:
        made = admin.post("/pool/apps", json={"app": "shop"})
    assert made.status_code == 201, made.text
    key = made.json()["connection"]["api_key"]
    assert key.startswith("gpma_") and made.json()["shown_once"] is True
    follow(pool)
    with pool.client(key=key) as client:
        assert ask(client).status_code == 200
    with pool.client() as client:
        assert ask(client).status_code == 200, "the configured app key works as it always did"
    rows = pool.wait_for_log(2)
    assert sorted((r["app"] or "") for r in rows) == ["", "shop"]
    assert all(r["workload"] is None for r in rows), "both are the shared workload's"

    with control(pool) as admin:
        listed = admin.get("/pool/apps").json()
    assert [a["app"] for a in listed["apps"]] == ["shop"]
    text = str(listed)
    assert key not in text and key[5:] not in text, "a key is never answered back"
    from gpm_server.keys import fingerprint
    assert fingerprint(key) not in text, "nor its hash"


def test_a_named_key_is_refused_on_the_control_api(pool):
    with control(pool) as admin:
        key = admin.post("/pool/apps", json={"app": "shop"}).json()["connection"]["api_key"]
    with control(pool, key=key) as client:
        for method, path in (("GET", "/pool/status"), ("GET", "/pool/stats"), ("POST", "/pool/apps")):
            answer = client.request(method, path, json={"app": "x"} if method == "POST" else None)
            assert answer.status_code == 403, (path, answer.text)
            assert answer.json()["error"] == "app_key_refused"
    with control(pool, key=APP_KEY) as client:
        assert client.get("/pool/apps").status_code == 403, "the configured app key, as before"


def test_a_revoked_key_reaches_nothing_and_rotation_needs_no_pause(pool):
    with control(pool) as admin:
        old = admin.post("/pool/apps", json={"app": "shop"}).json()
        new = admin.post("/pool/apps", json={"app": "shop"}).json()
    follow(pool)
    for made in (old, new):
        with pool.client(key=made["connection"]["api_key"]) as client:
            assert ask(client).status_code == 200, "two keys of one app work at once"
    with control(pool) as admin:
        answer = admin.delete(f"/pool/apps/shop/keys/{old['key']['key_id']}")
        assert answer.json()["revoked"] == [old["key"]["key_id"]]
    follow(pool)
    with pool.client(key=old["connection"]["api_key"]) as client:
        assert ask(client).status_code == 401
    with pool.client(key=new["connection"]["api_key"]) as client:
        assert ask(client).status_code == 200
    with control(pool) as admin:
        assert admin.delete("/pool/apps/shop").json()["revoked"] == [new["key"]["key_id"]]
        assert admin.delete("/pool/apps/nobody").status_code == 404
        assert admin.get("/pool/apps").json()["apps"][0]["usable"] is False
    follow(pool)
    with pool.client(key=new["connection"]["api_key"]) as client:
        assert ask(client).status_code == 401
    kinds = [e["kind"] for e in pool.supervisor.events.recent(20)]
    assert kinds.count("app_key_created") == 2 and kinds.count("app_key_revoked") == 2


def test_an_app_name_is_a_plain_name(pool):
    store = AppKeyStore(pool.database)
    for bad in ("", "Shop", "a b", "x" * 41, None, "-lead"):
        with pytest.raises(AppKeyRefused):
            store.create(bad)
    with control(pool) as admin:
        assert admin.post("/pool/apps", json={"app": "no spaces"}).status_code == 400
        assert admin.post("/pool/apps", content=b"not json").status_code == 400


def test_the_stats_endpoint_shows_each_app_apart(pool):
    with control(pool) as admin:
        key = admin.post("/pool/apps", json={"app": "shop"}).json()["connection"]["api_key"]
    follow(pool)
    with pool.client(key=key) as client:
        for _ in range(3):
            assert ask(client).status_code == 200
    with pool.client() as client:
        assert ask(client).status_code == 200
        assert client.post("/v1/chat/completions", json={"model": "unknown", "messages": []}).status_code >= 400
    pool.wait_for_log(5)
    with control(pool) as admin:
        answer = admin.get("/pool/stats", params={"range": "24h", "by": "app"})
        assert answer.status_code == 200, answer.text
        stats = answer.json()
        by_label = {s["label"]: s["totals"] for s in stats["series"]}
        assert by_label["shop"]["requests"] == 3 and by_label["shop"]["errors"] == 0
        assert by_label["shop"]["p95_ms"] is not None
        assert by_label["shared pool (configured app key)"]["requests"] == 2
        assert by_label["shared pool (configured app key)"]["errors"] == 1
        assert stats["bucket"] == "hour" and len(stats["buckets"]) in (24, 25)
        assert stats["retention"] == {"keep_requests_days": 30, "keep_rollups_days": 400}

        by_host = admin.get("/pool/stats", params={"range": "7d", "by": "host", "app": "shop"}).json()
        assert [s["label"] for s in by_host["series"]] == ["laptop"]
        assert by_host["series"][0]["type"] == "owned"
        assert admin.get("/pool/stats", params={"range": "30d", "bucket": "day"}).json()["totals"]["requests"] == 5

        for params in ({"by": "colour"}, {"range": "forever"}, {"bucket": "minute"},
                       {"range": "60d", "bucket": "hour"}, {"from": str(time.time()), "to": str(time.time() - 10)}):
            assert admin.get("/pool/stats", params=params).status_code == 400, params
