"""Responses from a host that can be taken away are held until whole (D62).

Measured on the first live pool: 110 responses broke mid-stream, every one on a rented host,
after 13 s of generation each. A streamed response is already in the app's hands when the host
goes; a held one is not, so the pool can run it again itself.
"""

import json

import pytest
from fakes.harness import EngineSpec, pool_harness

MODEL = "m1"


def chat(stream=True, **extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "stream": stream, **extra}


def two_hosts(delivery=None, **kwargs):
    return pool_harness(
        [
            EngineSpec(id="h1", resident={MODEL}, kind="local", workers=1),
            EngineSpec(id="h2", resident={MODEL}, kind="fixed-remote", workers=1),
        ],
        model_set=[MODEL],
        delivery=delivery,
        **kwargs,
    )


# --- the delivery decision ---


def test_by_default_a_host_that_cannot_vanish_still_streams():
    with two_hosts() as pool, pool.client() as client:
        response = client.post("/api/chat", json=chat())
        assert response.status_code == 200
        assert response.headers["X-GPM-Delivery"] == "stream"
        assert response.headers["X-GPM-Attempts"] == "1"


def test_a_buffered_response_is_byte_for_byte_what_the_engine_sent():
    """The frames are the engine's own, in order: a client that asked for a stream still
    parses a stream. Only the timing differs."""
    with two_hosts() as pool, pool.client() as client:
        streamed = client.post("/api/chat", json=chat()).content

    with two_hosts(delivery={"local": "buffered", "fixed_remote": "buffered"}) as pool, pool.client() as client:
        held = client.post("/api/chat", json=chat())

    assert held.headers["X-GPM-Delivery"] == "buffered"
    assert held.content == streamed


def test_an_app_may_ask_for_tokens_as_they_come():
    with two_hosts(delivery={"local": "buffered", "fixed_remote": "buffered"}) as pool, pool.client() as client:
        response = client.post("/api/chat", json=chat(), headers={"X-GPM-Delivery": "stream"})
        assert response.headers["X-GPM-Delivery"] == "stream"


def test_the_operator_can_refuse_that_override():
    with two_hosts(
        delivery={"local": "buffered", "fixed_remote": "buffered", "allow_request_override": False}
    ) as pool, pool.client() as client:
        response = client.post("/api/chat", json=chat(), headers={"X-GPM-Delivery": "stream"})
        assert response.headers["X-GPM-Delivery"] == "buffered"


# --- what buffering buys: the pool runs a lost request again itself ---


def test_a_host_lost_mid_generation_costs_a_re_run_not_half_an_answer():
    with two_hosts(delivery={"local": "buffered", "fixed_remote": "buffered"}) as pool:
        pool.engines["h1"].fake.cut_stream_times = 1
        with pool.client(timeout=20) as client:
            response = client.post("/api/chat", json=chat())

        assert response.status_code == 200
        assert response.headers["X-GPM-Host"] == "h2", "it should have been run again elsewhere"
        assert response.headers["X-GPM-Attempts"] == "2"
        # Whole: the last frame is there, which is what a broken stream never has.
        assert json.loads(response.content.splitlines()[-1])["done"] is True


def test_a_streamed_response_still_breaks_when_its_host_goes():
    """The honest comparison: without buffering there is nothing the pool can do."""
    with two_hosts() as pool:
        pool.engines["h1"].fake.cut_stream_times = 1
        with pool.client(timeout=20) as client:
            with pytest.raises(Exception):
                response = client.post("/api/chat", json=chat())
                response.read()


def test_when_no_host_is_left_the_app_gets_a_clean_answer_not_a_broken_stream():
    with two_hosts(delivery={"local": "buffered", "fixed_remote": "buffered"}) as pool:
        pool.engines["h1"].fake.cut_stream_times = 5
        pool.engines["h2"].fake.cut_stream_times = 5
        with pool.client(timeout=20) as client:
            response = client.post("/api/chat", json=chat())

        assert response.status_code == 503
        body = response.json()
        assert body["reason"] == "host_lost"
        assert "nothing partial was sent" in body["detail"]


def test_re_dispatch_is_bounded_by_the_operators_number():
    with two_hosts(delivery={"local": "buffered", "fixed_remote": "buffered", "max_redispatch": 0}) as pool:
        pool.engines["h1"].fake.cut_stream_times = 1
        with pool.client(timeout=20) as client:
            response = client.post("/api/chat", json=chat())
        assert response.status_code == 503
        assert response.json()["reason"] == "host_lost"


# --- bounds ---


def test_a_response_too_big_to_hold_is_streamed_rather_than_failed():
    with two_hosts(
        delivery={"local": "buffered", "fixed_remote": "buffered", "max_buffer_mb": 0.000001}
    ) as pool, pool.client() as client:
        response = client.post("/api/chat", json=chat())
        assert response.status_code == 200
        assert response.headers["X-GPM-Delivery"] == "stream-after-overflow"
        # Nothing is lost when it spills: the prefix is replayed before the rest.
        assert json.loads(response.content.splitlines()[-1])["done"] is True


def test_an_error_from_the_engine_is_not_held():
    with two_hosts(delivery={"local": "buffered", "fixed_remote": "buffered"}) as pool, pool.client() as client:
        response = client.post("/api/chat", json={"model": "not-in-this-pool", "messages": []})
        assert response.status_code == 404


# --- what the pool tells an app, and what an SDK does with it ---


def test_the_pool_publishes_how_it_delivers():
    with two_hosts(delivery={"local": "buffered"}) as pool, pool.client() as client:
        policy = client.get("/pool/status").json()["delivery"]

    assert policy["by_kind"]["local"] == "buffered"
    assert policy["by_kind"]["rented-interruptible"] == "buffered"  # the shipped default
    assert policy["by_kind"]["rented-on-demand"] == "stream"
    assert policy["max_redispatch"] == 1


def test_an_sdk_widens_its_budget_to_what_the_pool_asks_for():
    """A held response arrives whole, so "first byte" is the end of the generation. A budget
    sized for a streamed first token would give up mid-answer."""
    import httpx
    from gpm_client.client import _budget_from_status

    short = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0)
    wider = _budget_from_status({"limits": {"client_time_to_first_byte_s": 300.0}}, short)
    assert wider is not None and wider.read == 300.0
    assert wider.connect == 5.0  # nothing else is touched

    already = httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=600.0)
    assert _budget_from_status({"limits": {"client_time_to_first_byte_s": 300.0}}, already) is None
    assert _budget_from_status({}, short) is None  # a pool that says nothing changes nothing
