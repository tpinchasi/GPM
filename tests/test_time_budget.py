"""Who waits, who retries, who cancels — docs/spec/app-contract.md §5.

Unrelated waiting layers produce abandoned, billed work, so the contract states the budget and
the router holds to it: it always answers a queued request before an SDK client gives up.
"""

import concurrent.futures
import time

from fakes.harness import EngineSpec, pool_harness

MODEL = "m1"


def chat(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False, **extra}


def busy_pool(queue_timeout_s=0.5, delay=1.0):
    return pool_harness(
        [EngineSpec(id="local-1", resident={MODEL}, workers=1, chunk_delay_s=delay)],
        model_set=[MODEL],
        queue_timeout_s=queue_timeout_s,
    )


def test_a_queued_request_is_answered_before_the_queue_limit_not_left_hanging():
    with busy_pool() as pool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
            def call(headers=None):
                with pool.client(timeout=20) as client:
                    return client.post("/api/chat", json=chat(), headers=headers or {})

            first = threads.submit(call)
            time.sleep(0.1)
            started = time.monotonic()
            second = threads.submit(call)
            queued = second.result()
            waited = time.monotonic() - started

        assert first.result().status_code == 200
        assert queued.status_code == 503
        assert queued.json()["reason"] == "queue_timeout"
        assert queued.json()["retry_after_s"] == 5
        assert queued.headers["Retry-After"] == "5"
        assert waited < 2.0


def test_a_deadline_that_has_already_passed_is_not_started():
    with pool_harness([EngineSpec(id="local-1", resident={MODEL}, workers=1)], model_set=[MODEL]) as pool:
        with pool.client() as client:
            response = client.post(
                "/api/chat", json=chat(), headers={"X-GPM-Deadline": str(time.time() - 1)}
            )
        assert response.status_code == 504
        assert response.json()["reason"] == "deadline_exceeded"
        assert pool.engines["local-1"].fake.received == []


def test_a_deadline_shorter_than_the_queue_limit_ends_the_wait_early():
    with busy_pool(queue_timeout_s=10, delay=1.0) as pool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
            def call(headers=None):
                with pool.client(timeout=20) as client:
                    return client.post("/api/chat", json=chat(), headers=headers or {})

            threads.submit(call)
            time.sleep(0.1)
            started = time.monotonic()
            queued = threads.submit(
                call, {"X-GPM-Deadline": str(time.time() + 0.3)}
            ).result()
            waited = time.monotonic() - started

        assert queued.status_code == 504
        assert queued.json()["reason"] == "deadline_exceeded"
        assert waited < 3.0


def test_an_iso_deadline_is_understood():
    from datetime import datetime, timedelta, timezone

    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    with pool_harness([EngineSpec(id="local-1", resident={MODEL}, workers=1)], model_set=[MODEL]) as pool:
        with pool.client() as client:
            response = client.post("/api/chat", json=chat(), headers={"X-GPM-Deadline": past})
        assert response.status_code == 504


def test_an_unreadable_deadline_is_ignored_rather_than_failing_the_call():
    with pool_harness([EngineSpec(id="local-1", resident={MODEL}, workers=1)], model_set=[MODEL]) as pool:
        with pool.client() as client:
            response = client.post("/api/chat", json=chat(), headers={"X-GPM-Deadline": "soon-ish"})
        assert response.status_code == 200


def test_a_client_that_leaves_the_queue_frees_its_place_at_once():
    with busy_pool(queue_timeout_s=30, delay=0.8) as pool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
            def hold():
                with pool.client(timeout=20) as client:
                    return client.post("/api/chat", json=chat())

            holder = threads.submit(hold)
            time.sleep(0.1)

            with pool.client(timeout=0.3) as client:
                try:
                    client.post("/api/chat", json=chat())
                except Exception:
                    pass  # the queued caller gave up and its connection closed

            assert holder.result().status_code == 200

        rows = pool.wait_for_log(count=2)
        cancelled = [r for r in rows if r["outcome"] == "cancelled"]
        assert cancelled and cancelled[0]["reason"] == "client_disconnected"
