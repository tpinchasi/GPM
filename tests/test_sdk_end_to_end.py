"""The SDK against a real router: an app knows one URL, one key, and waits by default."""

import threading
import time

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, pool_harness
from gpm_client import (
    PoolAuthError,
    PoolClient,
    PoolRequestError,
    PoolUnavailable,
    RetryPolicy,
    pool_transport,
)

MODEL = "m1"
MESSAGES = [{"role": "user", "content": "hello there"}]
FAST_RETRY = dict(backoff_initial_s=0.05, backoff_max_s=0.1)
# The shipped default waits 30 minutes; a test must fail instead of stalling.
BOUNDED = RetryPolicy(max_wait_s=20, **FAST_RETRY)


def one_host(resident={MODEL}, **kwargs):
    return pool_harness([EngineSpec(id="local-1", resident=set(resident), workers=1, **kwargs)], model_set=[MODEL])


def test_a_script_gets_its_reply_and_the_facts_of_the_call():
    with one_host() as pool:
        with PoolClient(base_url=pool.url, api_key=APP_KEY, retry_policy=BOUNDED) as client:
            reply = client.chat(MODEL, MESSAGES, session_id="s-1")

        assert reply.content == "echo:hello there"
        assert reply.served_model == MODEL
        assert reply.host == "local-1"
        assert reply.runtime_class == "unknown-ollama"
        assert reply.wait_s >= 0


def test_embeddings():
    with one_host() as pool:
        with PoolClient(base_url=pool.url, api_key=APP_KEY, retry_policy=BOUNDED) as client:
            vectors = client.embed(MODEL, ["one", "two"])
        assert len(vectors) == 2


def test_the_sdk_waits_for_capacity_instead_of_failing():
    with one_host(resident=set()) as pool:  # the host is still preparing
        # The pool asks for a 10 s wait while a host prepares; max_wait clamps it.
        policy = RetryPolicy(max_wait_s=1.5, **FAST_RETRY)
        waits: list[str] = []
        policy.on_wait = lambda reason, waited, next_try: waits.append(reason)

        def make_it_ready() -> None:
            time.sleep(0.5)
            pool.engines["local-1"].fake.resident.add(MODEL)
            pool.reprobe()

        threading.Thread(target=make_it_ready, daemon=True).start()
        with PoolClient(base_url=pool.url, api_key=APP_KEY, retry_policy=policy) as client:
            reply = client.chat(MODEL, MESSAGES)

        assert reply.content == "echo:hello there"
        assert "preparing" in waits


def test_waiting_gives_up_at_max_wait():
    with one_host(resident=set()) as pool:
        policy = RetryPolicy(max_wait_s=0.3, **FAST_RETRY)
        with PoolClient(base_url=pool.url, api_key=APP_KEY, retry_policy=policy) as client:
            with pytest.raises(PoolUnavailable) as raised:
                client.chat(MODEL, MESSAGES)
        assert raised.value.reason == "preparing"


def test_wait_until_ready_is_a_usable_preflight():
    with one_host(resident=set()) as pool:
        with PoolClient(base_url=pool.url, api_key=APP_KEY, retry_policy=BOUNDED) as client:
            with pytest.raises(PoolUnavailable):
                client.wait_until_ready(timeout=0.2)

            pool.engines["local-1"].fake.resident.add(MODEL)
            pool.reprobe()
            client.wait_until_ready(timeout=5)


def test_a_wrong_key_is_not_retried():
    with one_host() as pool:
        with PoolClient(base_url=pool.url, api_key="wrong", retry_policy=BOUNDED) as client:
            with pytest.raises(PoolAuthError):
                client.chat(MODEL, MESSAGES)


def test_a_model_outside_the_pool_is_not_retried():
    with one_host() as pool:
        with PoolClient(base_url=pool.url, api_key=APP_KEY, retry_policy=BOUNDED) as client:
            with pytest.raises(PoolRequestError) as raised:
                client.chat("not-in-the-pool", MESSAGES)
        assert raised.value.status_code == 404


def test_the_transport_drops_under_a_client_library_without_call_site_changes():
    """The documented integration path: inject the transport, keep your own client."""
    with one_host() as pool:
        client = httpx.Client(
            base_url=pool.url,
            headers={"Authorization": f"Bearer {APP_KEY}"},
            transport=pool_transport(RetryPolicy(max_wait_s=10, **FAST_RETRY)),
        )
        with client:
            response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES, "stream": False})
        assert response.status_code == 200
        assert response.headers["X-GPM-Served-Model"] == MODEL


def test_the_transport_streams_without_buffering():
    with one_host(chunk_delay_s=0.05, chunks=4) as pool:
        client = httpx.Client(
            base_url=pool.url,
            headers={"Authorization": f"Bearer {APP_KEY}"},
            transport=pool_transport(RetryPolicy(max_wait_s=10, **FAST_RETRY)),
            timeout=20,
        )
        arrivals: list[float] = []
        with client:
            with client.stream("POST", "/api/chat", json={"model": MODEL, "messages": MESSAGES}) as response:
                for _ in response.iter_lines():
                    arrivals.append(time.monotonic())

        assert len(arrivals) >= 4
        assert arrivals[-1] - arrivals[0] > 0.1  # delivered as produced, not in one lump
