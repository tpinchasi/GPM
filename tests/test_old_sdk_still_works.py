"""An application built against the previous SDK keeps working after the pool is upgraded.

The `/v1` paths were **added** beside Ollama's native API, not put in place of it (D89), and the
app contract did not change version because everything added is optional. So an app still on
`gpm-client` 0.2.x — posting `/api/chat`, reading `data["message"]["content"]` — must see
exactly what it saw before.

Nothing else pins that. These tests deliberately do **not** import the SDK: they issue the calls
the old one issued, by hand, so that upgrading the SDK cannot make them pass. A future tidy-up
of `inference_paths()` that dropped the native paths would break a live application silently,
and this is what would catch it.
"""

import httpx
from fakes.harness import APP_KEY, EngineSpec, pool_harness

MODEL = "m1"
MESSAGES = [{"role": "user", "content": "hello there"}]


def old_client(pool) -> httpx.Client:
    """What `PoolClient` 0.2.0 built: a base URL, a bearer key, nothing else."""
    return httpx.Client(
        base_url=pool.url, headers={"Authorization": f"Bearer {APP_KEY}"}, timeout=30
    )


def one_host():
    return pool_harness([EngineSpec(id="local-1", resident={MODEL}, workers=2)], model_set=[MODEL])


def test_the_old_chat_call_returns_the_old_shape():
    """`PoolClient.chat()` 0.2.0 read `data["message"]["content"]`. It still finds it there."""
    with one_host() as pool, old_client(pool) as client:
        response = client.post(
            "/api/chat", json={"model": MODEL, "messages": MESSAGES, "stream": False}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"] == "echo:hello there"
    assert response.headers["X-GPM-Served-Model"] == MODEL
    assert response.headers["X-GPM-Wait-S"] is not None


def test_the_old_embed_call_returns_the_old_key():
    """0.2.0 read `data["embeddings"]`; the OpenAI surface answers under `data` instead, and an
    app that never moved must not meet that shape."""
    with one_host() as pool, old_client(pool) as client:
        response = client.post("/api/embed", json={"model": MODEL, "input": ["one", "two"]})

    assert response.status_code == 200
    assert len(response.json()["embeddings"]) == 2


def test_streaming_still_defaults_to_on_for_the_native_path():
    """The two surfaces have opposite defaults. An app that omitted `stream` got a stream, and
    changing that under it would break every caller that iterates frames."""
    with one_host() as pool, old_client(pool) as client:
        with client.stream("POST", "/api/chat", json={"model": MODEL, "messages": MESSAGES}) as reply:
            frames = [line for line in reply.iter_lines() if line.strip()]

    assert len(frames) > 1
    assert frames[0].startswith("{")  # newline-delimited JSON, not server-sent events


def test_the_app_contract_did_not_change_version():
    """Every dialect item added is optional and no status code changed meaning, so an app built
    against the first version is still built against the version it gets."""
    with one_host() as pool, old_client(pool) as client:
        response = client.post(
            "/api/chat", json={"model": MODEL, "messages": MESSAGES, "stream": False}
        )
        status = client.get("/pool/status")

    assert response.headers["X-GPM-Contract"] == "1"
    assert status.status_code == 200
    assert any(host.get("state") == "ready" for host in status.json()["hosts"])


def test_both_surfaces_answer_the_same_pool_at_the_same_time():
    """The point of the addition: an app that moved and one that did not are both served, by the
    same hosts, with no translation between them."""
    with one_host() as pool, old_client(pool) as client:
        native = client.post(
            "/api/chat", json={"model": MODEL, "messages": MESSAGES, "stream": False}
        )
        openai = client.post(
            "/v1/chat/completions", json={"model": MODEL, "messages": MESSAGES}
        )

    assert native.status_code == openai.status_code == 200
    assert native.json()["message"]["content"] == "echo:hello there"
    assert openai.json()["choices"][0]["message"]["content"] == "echo:hello there"
