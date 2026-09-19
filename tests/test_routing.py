"""Strict priority tiers, readiness, failover-once, and the machine-readable answers."""

import concurrent.futures
import json

import pytest
from fakes.harness import APP_KEY, EngineSpec, pool_harness

MODEL = "m1"


def chat(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False, **extra}


def two_tiers(**kwargs):
    return pool_harness(
        [
            EngineSpec(id="local-1", resident={MODEL}, kind="local", workers=1),
            EngineSpec(id="remote-1", resident={MODEL}, kind="fixed-remote", workers=1),
        ],
        model_set=[MODEL],
        **kwargs,
    )


def test_local_is_used_before_fixed_remote():
    with two_tiers() as pool, pool.client() as client:
        response = client.post("/api/chat", json=chat())
        assert response.headers["X-GPM-Host"] == "local-1"


def test_the_next_tier_takes_the_overflow():
    with pool_harness(
        [
            EngineSpec(id="local-1", resident={MODEL}, kind="local", workers=1, chunk_delay_s=0.4),
            EngineSpec(id="remote-1", resident={MODEL}, kind="fixed-remote", workers=1),
        ],
        model_set=[MODEL],
    ) as pool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_of_threads:
            def call():
                with pool.client(timeout=20) as client:
                    return client.post("/api/chat", json=chat())

            first = pool_of_threads.submit(call)
            second = pool_of_threads.submit(call)
            hosts = {first.result().headers["X-GPM-Host"], second.result().headers["X-GPM-Host"]}
        assert hosts == {"local-1", "remote-1"}


def test_a_host_that_dies_before_answering_is_retried_once_on_the_next_host():
    with two_tiers() as pool:
        pool.stop_engine("local-1")  # the router still believes it is ready
        with pool.client() as client:
            response = client.post("/api/chat", json=chat())
        assert response.status_code == 200
        assert response.headers["X-GPM-Host"] == "remote-1"


def test_when_every_host_fails_the_answer_says_so():
    with two_tiers() as pool:
        pool.stop_engine("local-1")
        pool.stop_engine("remote-1")
        with pool.client() as client:
            response = client.post("/api/chat", json=chat())
        assert response.status_code == 503
        assert response.json()["reason"] == "hosts_unreachable"
        assert response.headers["Retry-After"]


def test_a_host_missing_the_model_set_is_not_routed_to():
    with pool_harness(
        [EngineSpec(id="local-1", resident=set(), workers=1)],
        model_set=[MODEL],
    ) as pool:
        with pool.client() as client:
            status = client.get("/pool/status").json()
            response = client.post("/api/chat", json=chat())

        assert status["hosts"][0]["state"] == "preparing"
        assert response.status_code == 503
        assert response.json()["reason"] == "preparing"


def test_a_host_becomes_routable_once_the_model_set_is_resident():
    with pool_harness(
        [EngineSpec(id="local-1", resident=set(), workers=1)],
        model_set=[MODEL],
    ) as pool:
        pool.engines["local-1"].fake.resident.add(MODEL)
        pool.reprobe()
        with pool.client() as client:
            assert client.post("/api/chat", json=chat()).status_code == 200


def test_an_unreachable_engine_is_marked_and_skipped():
    with two_tiers() as pool:
        pool.stop_engine("local-1")
        pool.reprobe()
        with pool.client() as client:
            status = client.get("/pool/status").json()
            response = client.post("/api/chat", json=chat())

        states = {h["host_id"]: h["state"] for h in status["hosts"]}
        assert states == {"local-1": "unreachable", "remote-1": "ready"}
        assert response.headers["X-GPM-Host"] == "remote-1"


def test_a_model_outside_the_pools_set_is_refused_not_loaded():
    with two_tiers() as pool, pool.client() as client:
        response = client.post("/api/chat", json={"model": "never-heard-of-it", "messages": []})
        assert response.status_code == 404
        assert response.json()["reason"] == "model_not_in_pool"
        assert pool.engines["local-1"].fake.received == []


def test_the_app_key_is_required_even_on_loopback():
    with two_tiers() as pool:
        with pool.client(key=None) as client:
            assert client.post("/api/chat", json=chat()).status_code == 401
            assert client.get("/pool/status").status_code == 401
        with pool.client(key="wrong") as client:
            assert client.post("/api/chat", json=chat()).status_code == 401


def test_a_path_the_engine_does_not_serve_is_not_forwarded():
    with two_tiers() as pool, pool.client() as client:
        response = client.post("/api/pull", json={"model": MODEL})
        assert response.status_code == 404
        assert response.json()["reason"] == "unknown_path"
        assert pool.engines["local-1"].fake.received == []


def test_the_app_key_never_reaches_the_engine():
    with two_tiers() as pool, pool.client() as client:
        client.post("/api/chat", json=chat())
        _, _, headers = pool.engines["local-1"].fake.received[-1]
    assert "authorization" not in {k.lower() for k in headers}


def test_pool_dialect_headers_are_on_every_answer():
    with two_tiers() as pool, pool.client() as client:
        response = client.post("/api/chat", json=chat())
    assert response.headers["X-GPM-Served-Model"] == MODEL
    assert response.headers["X-GPM-Host"] == "local-1"
    assert response.headers["X-GPM-Runtime-Class"] == "unknown-ollama"
    assert response.headers["X-GPM-Contract"] == "1"
    assert float(response.headers["X-GPM-Wait-S"]) >= 0


def test_status_reports_capacity_and_the_limits_in_force():
    with two_tiers() as pool, pool.client() as client:
        status = client.get("/pool/status").json()
    assert status["capacity"] == {"hosts_ready": 2, "workers_total": 2, "workers_busy": 0}
    assert status["limits"]["queue_timeout_s"] == 30.0
    assert status["model_set"] == [MODEL]
    assert status["contract_version"] == "1"


def test_every_request_is_logged_with_what_it_was_served_by():
    with two_tiers() as pool, pool.client() as client:
        client.post("/api/chat", json=chat(), headers={"X-GPM-Session": "s-42"})
        row = pool.wait_for_log()[0]
    assert row["session_id"] == "s-42"
    assert row["host_id"] == "local-1"
    assert row["model_requested"] == MODEL
    assert row["model_served"] == MODEL
    assert row["outcome"] == "ok"
    assert row["status_code"] == 200
    assert row["latency_ms"] >= 0


def test_a_refusal_is_logged_too():
    with two_tiers() as pool, pool.client() as client:
        client.post("/api/chat", json={"model": "nope", "messages": []})
        row = pool.wait_for_log()[0]
    assert row["outcome"] == "rejected"
    assert row["reason"] == "model_not_in_pool"


def test_an_embedding_request_takes_its_turn_like_any_other():
    with two_tiers() as pool, pool.client() as client:
        response = client.post("/api/embed", json={"model": MODEL, "input": ["one", "two"]})
    assert response.status_code == 200
    assert len(response.json()["embeddings"]) == 2


def test_wait_is_time_to_a_worker_not_time_to_the_engines_answer():
    """Live finding: a non-streaming reply that took the engine 46 s was reported as 46 s of
    queue wait. Waiting for capacity must never be mistaken for generation time."""
    with pool_harness(
        [EngineSpec(id="local-1", resident={MODEL}, kind="local", workers=1, chunk_delay_s=0.6)],
        model_set=[MODEL],
    ) as pool, pool.client(timeout=20) as client:
        response = client.post("/api/chat", json=chat())
        assert response.status_code == 200
        assert float(response.headers["X-GPM-Wait-S"]) < 0.3  # a worker was free at once
        row = pool.wait_for_log()[0]
        assert row["latency_ms"] > 500  # the engine, not the queue, took the time
