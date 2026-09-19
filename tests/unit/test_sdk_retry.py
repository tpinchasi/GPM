"""The SDK's default behaviour: wait and retry for capacity, fail fast on everything else."""

import httpx
import pytest
from gpm_client import (
    PoolAuthError,
    PoolRequestError,
    PoolStreamInterrupted,
    PoolTransport,
    PoolUnavailable,
    RetryPolicy,
)

FAST = dict(backoff_initial_s=0.001, backoff_max_s=0.002)


class ScriptedTransport(httpx.BaseTransport):
    """Replies with the next scripted item: a Response, or an exception to raise."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script
        if isinstance(item, Exception):
            raise item
        return item


def no_capacity(reason: str, retry_after: float | None = None) -> httpx.Response:
    body = {"error": "no_capacity", "reason": reason, "retry_after_s": retry_after, "detail": "d"}
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return httpx.Response(503, json=body, headers=headers)


def ok(text: str = "hello") -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": text}})


def client(script: list, policy: RetryPolicy | None = None) -> tuple[httpx.Client, ScriptedTransport]:
    inner = ScriptedTransport(script)
    transport = PoolTransport(policy=policy or RetryPolicy(max_wait_s=5, **FAST), transport=inner)
    return httpx.Client(transport=transport, base_url="http://pool"), inner


def test_a_503_is_waited_out_and_the_call_succeeds():
    http, inner = client([no_capacity("queue_timeout"), no_capacity("preparing"), ok()])
    response = http.post("/api/chat", json={"model": "m"})
    assert response.status_code == 200
    assert inner.calls == 3


def test_retry_after_is_honoured(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("gpm_client.transport.time.sleep", slept.append)
    http, _ = client([no_capacity("queue_timeout", retry_after=7), ok()], RetryPolicy(max_wait_s=60, **FAST))
    http.post("/api/chat", json={"model": "m"})
    assert slept == [7.0]


def test_no_lease_fails_fast_because_waiting_cannot_help():
    http, inner = client([no_capacity("no_lease"), ok()])
    with pytest.raises(PoolUnavailable) as raised:
        http.post("/api/chat", json={"model": "m"})
    assert raised.value.reason == "no_lease"
    assert inner.calls == 1


def test_no_lease_can_be_waited_out_on_request():
    policy = RetryPolicy(max_wait_s=5, wait_without_lease=True, **FAST)
    http, inner = client([no_capacity("no_lease"), ok()], policy)
    assert http.post("/api/chat", json={"model": "m"}).status_code == 200
    assert inner.calls == 2


def test_a_401_is_never_retried():
    http, inner = client([httpx.Response(401, json={"error": "unauthorized"}), ok()])
    with pytest.raises(PoolAuthError):
        http.post("/api/chat", json={"model": "m"})
    assert inner.calls == 1


def test_a_404_is_never_retried_and_carries_its_reason():
    http, inner = client([httpx.Response(404, json={"error": "model_not_in_pool", "reason": "model_not_in_pool", "detail": "no"})])
    with pytest.raises(PoolRequestError) as raised:
        http.post("/api/chat", json={"model": "m"})
    assert raised.value.status_code == 404
    assert raised.value.reason == "model_not_in_pool"
    assert inner.calls == 1


def test_a_transport_error_before_any_byte_is_retried():
    http, inner = client([httpx.ConnectError("refused"), ok()])
    assert http.post("/api/chat", json={"model": "m"}).status_code == 200
    assert inner.calls == 2


def test_a_long_retry_after_never_overruns_max_wait(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("gpm_client.transport.time.sleep", slept.append)
    policy = RetryPolicy(max_wait_s=2, **FAST)
    http, _ = client([no_capacity("preparing", retry_after=600), ok()], policy)
    http.post("/api/chat", json={"model": "m"})
    assert len(slept) == 1 and 1.9 < slept[0] <= 2.0


def test_waiting_stops_at_max_wait():
    policy = RetryPolicy(max_wait_s=0.05, **FAST)
    http, _ = client([no_capacity("queue_timeout")] * 50, policy)
    with pytest.raises(PoolUnavailable) as raised:
        http.post("/api/chat", json={"model": "m"})
    assert raised.value.reason == "queue_timeout"


def test_the_caller_can_watch_the_wait():
    seen: list[tuple[str, float, float]] = []
    policy = RetryPolicy(max_wait_s=5, on_wait=lambda *args: seen.append(args), **FAST)
    http, _ = client([no_capacity("preparing"), ok()], policy)
    http.post("/api/chat", json={"model": "m"})
    assert seen and seen[0][0] == "preparing"


def test_a_failure_after_output_was_delivered_is_not_retried():
    def broken():
        yield b'{"partial":'
        raise httpx.ReadError("connection lost")

    http, inner = client([httpx.Response(200, stream=_SyncStream(broken())), ok()])
    with http.stream("POST", "/api/chat", json={"model": "m"}) as response:
        with pytest.raises(PoolStreamInterrupted):
            for _ in response.iter_raw():
                pass
    assert inner.calls == 1


def test_the_body_is_resent_on_retry():
    http, inner = client([no_capacity("queue_timeout"), ok()])
    http.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert inner.calls == 2


class _SyncStream(httpx.SyncByteStream):
    def __init__(self, iterator):
        self._iterator = iterator

    def __iter__(self):
        yield from self._iterator
