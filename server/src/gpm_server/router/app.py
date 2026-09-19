"""The router: the only process in the request path. Nothing slow or blocking belongs here.

It passes requests through in the engine's own API, rewriting the model field and nothing else,
and adds the small pool dialect described in docs/spec/app-contract.md §2.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import anyio
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import PoolConfig
from ..contract import CONTRACT_VERSION
from ..db import Database, RequestRecord
from ..keys import verify
from ..models import HostState
from ..state import RouterState, open_database
from .dispatch import Assignment, Need, NoEligibleHost, NoReadyHost, QueueTimeout

log = logging.getLogger("gpm.router")

#: Never forwarded upstream: hop-by-hop headers, the app's own credentials, and anything the
#: transport recomputes.
_DROP_UPSTREAM = frozenset(
    {
        "host",
        "authorization",
        "content-length",
        "accept-encoding",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
    }
)

#: Never copied back to the client: hop-by-hop headers and ones the server regenerates.
_DROP_DOWNSTREAM = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "date",
        "server",
    }
)

_RETRY_AFTER_S = {
    "queue_timeout": 5,
    "preparing": 10,
    "hosts_unreachable": 30,
    "no_eligible_host": 30,
}


class ClientGone(Exception):
    """The client disconnected while its request was still queued."""


def _authorised(state: RouterState, request: Request) -> bool:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    return verify(token, state.app_hashes)


def _no_capacity(reason: str, detail: Optional[str] = None) -> JSONResponse:
    retry_after = _RETRY_AFTER_S.get(reason, 30)
    return JSONResponse(
        status_code=503,
        content={
            "error": "no_capacity",
            "reason": reason,
            "retry_after_s": retry_after,
            "detail": detail,
        },
        headers={"Retry-After": str(retry_after), "X-GPM-Contract": CONTRACT_VERSION},
    )


def _error(status_code: int, error: str, reason: Optional[str] = None, detail: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "reason": reason, "detail": detail},
        headers={"X-GPM-Contract": CONTRACT_VERSION},
    )


def _parse_deadline(raw: Optional[str]) -> Optional[float]:
    """`X-GPM-Deadline` as epoch seconds or an ISO-8601 instant, returned on the monotonic
    clock. An unparseable value is ignored rather than failing the request."""
    if not raw:
        return None
    try:
        wall = float(raw)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        wall = parsed.timestamp()
    return time.monotonic() + (wall - time.time())


def _upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _DROP_UPSTREAM and not key.lower().startswith("x-gpm-")
    }
    # Identity encoding keeps the response bytes — and its Content-Length — exactly as the
    # engine produced them, which is what passthrough fidelity means.
    headers["accept-encoding"] = "identity"
    return headers


def _downstream_headers(upstream: httpx.Response, assignment: Assignment, wait_s: float) -> dict[str, str]:
    headers = {
        key: value for key, value in upstream.headers.items() if key.lower() not in _DROP_DOWNSTREAM
    }
    headers["X-GPM-Served-Model"] = assignment.variant.tag
    headers["X-GPM-Host"] = assignment.host.host_id
    headers["X-GPM-Runtime-Class"] = assignment.variant.runtime_class
    headers["X-GPM-Wait-S"] = f"{wait_s:.3f}"
    headers["X-GPM-Contract"] = CONTRACT_VERSION
    return headers


async def _wait_for_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.2)


async def _acquire(state: RouterState, request: Request, need: Need, request_id: str, deadline: float, exclude: frozenset[str]) -> Assignment:
    """Wait for a worker, but stop waiting the moment the client goes away."""
    acquire = asyncio.create_task(
        state.dispatcher.acquire(need, request_id=request_id, deadline=deadline, exclude=exclude)
    )
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    done, _ = await asyncio.wait({acquire, disconnect}, return_when=asyncio.FIRST_COMPLETED)
    if acquire in done:
        # Cancelled, never awaited: the watcher sits inside Starlette's own pre-cancelled
        # scope, which swallows the cancellation, so awaiting it here would never return.
        disconnect.cancel()
        return acquire.result()
    acquire.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await acquire
    raise ClientGone()


def _unready_reason(state: RouterState) -> str:
    if any(host.state is HostState.PREPARING for host in state.hosts):
        return "preparing"
    return "hosts_unreachable"


def create_app(
    config: PoolConfig,
    database: Optional[Database] = None,
    db_path: Optional[str | Path] = None,
) -> FastAPI:
    owns_database = database is None
    database = database or open_database(config, db_path)
    state = RouterState(config, database)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Pick up whatever the supervisor has already published before taking requests.
        await state.registry.refresh()
        task = asyncio.create_task(
            state.registry.run_forever(config.pool.host_table_poll_s)
        )
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await state.aclose()
            if owns_database:
                database.close()

    app = FastAPI(
        title="GPM router",
        version=CONTRACT_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.pool = state

    @app.get("/pool/status")
    async def pool_status(request: Request) -> Response:
        if not _authorised(state, request):
            return _error(401, "unauthorized", detail="missing or invalid app key")
        hosts = []
        for host in state.hosts:
            entry = {
                "host_id": host.host_id,
                "kind": host.kind,
                "transport": host.transport_type,
                "priority": host.priority,
                "state": host.state.value,
                "capabilities": sorted(host.capabilities),
                "workers": {"total": host.total_workers, "busy": host.busy},
                "resident": sorted(host.resident),
                "last_error": host.last_error,
            }
            hosts.append(entry)
        return JSONResponse(
            {
                "pool": config.pool.name,
                "contract_version": CONTRACT_VERSION,
                "engine": state.engine.name,
                "model_set": config.pool.model_set,
                "limits": {
                    "queue_timeout_s": config.pool.queue_timeout_s,
                    "client_time_to_first_byte_s": config.pool.client_time_to_first_byte_s,
                },
                "capacity": {
                    "hosts_ready": sum(1 for h in state.hosts if h.state is HostState.READY),
                    "workers_total": sum(h.total_workers for h in state.hosts),
                    "workers_busy": sum(h.busy for h in state.hosts),
                },
                "hosts": hosts,
            },
            headers={"X-GPM-Contract": CONTRACT_VERSION},
        )

    @app.post("/{full_path:path}")
    async def inference(request: Request, full_path: str) -> Response:
        engine = state.engine
        path = "/" + full_path
        request_id = uuid.uuid4().hex
        session_id = request.headers.get("x-gpm-session")
        started = time.monotonic()

        if not _authorised(state, request):
            return _error(401, "unauthorized", detail="missing or invalid app key")
        if path not in engine.inference_paths():
            return _error(404, "not_found", "unknown_path", f"{path} is not an inference path of engine {engine.name!r}")

        body = await request.body()
        requested = engine.requested_model(path, body)

        async def record(outcome: str, **fields: object) -> None:
            await state.log.record(
                RequestRecord(
                    request_id=request_id,
                    outcome=outcome,
                    session_id=session_id,
                    model_requested=requested,
                    **fields,  # type: ignore[arg-type]
                )
            )

        if requested is None:
            await record("rejected", status_code=400, reason="model_missing")
            return _error(400, "bad_request", "model_missing", "the request body names no model")

        in_pool = requested in config.pool.model_set or state.dispatcher.knows_model(requested)
        if not in_pool:
            await record("rejected", status_code=404, reason="model_not_in_pool")
            return _error(
                404,
                "model_not_in_pool",
                "model_not_in_pool",
                f"{requested!r} is not in this pool's model set; it will not be loaded on demand",
            )

        need = Need(
            model=requested,
            wants_schema=engine.wants_schema(path, body),
            runtime_class_pin=request.headers.get("x-gpm-runtime-class"),
        )

        request_deadline = _parse_deadline(request.headers.get("x-gpm-deadline"))
        if request_deadline is not None and request_deadline <= time.monotonic():
            await record("rejected", status_code=504, reason="deadline_exceeded")
            return _error(504, "deadline_exceeded", "deadline_exceeded", "the request's deadline had already passed")
        queue_deadline = started + config.pool.queue_timeout_s
        if request_deadline is not None:
            queue_deadline = min(queue_deadline, request_deadline)

        excluded: set[str] = set()
        assignment: Optional[Assignment] = None
        upstream: Optional[httpx.Response] = None
        wait_s = 0.0
        dispatched_at = started

        while True:
            try:
                assignment = await _acquire(state, request, need, request_id, queue_deadline, frozenset(excluded))
                # The wait is the time to a worker, measured here — not after the engine has
                # answered, which for a non-streaming call includes the whole generation.
                wait_s = time.monotonic() - started
            except ClientGone:
                await record("cancelled", reason="client_disconnected", queue_wait_ms=(time.monotonic() - started) * 1000)
                return Response(status_code=499)
            except NoReadyHost:
                reason = _unready_reason(state)
                await record("rejected", status_code=503, reason=reason)
                return _no_capacity(reason, "no host is ready to serve this request")
            except NoEligibleHost:
                await record("rejected", status_code=503, reason="no_eligible_host")
                return _no_capacity(
                    "no_eligible_host",
                    "hosts are ready but none holds a build of this model that satisfies the request",
                )
            except QueueTimeout:
                if request_deadline is not None and time.monotonic() >= request_deadline:
                    await record("rejected", status_code=504, reason="deadline_exceeded")
                    return _error(504, "deadline_exceeded", "deadline_exceeded", "the request's deadline passed while queued")
                await record("rejected", status_code=503, reason="queue_timeout", queue_wait_ms=(time.monotonic() - started) * 1000)
                return _no_capacity("queue_timeout", "every eligible worker stayed busy past the queue limit")

            assignment.host.last_request_at = time.time()
            forwarded = engine.with_model(path, body, assignment.variant.tag)
            upstream_request = assignment.host.client.build_request(
                "POST", path, content=forwarded, headers=_upstream_headers(request)
            )
            # Latency runs from the moment the engine is asked. For a non-streaming call the
            # headers arrive only after the whole generation, so measuring from after `send`
            # would report a 46-second answer as under a millisecond (seen live).
            dispatched_at = time.monotonic()
            try:
                upstream = await assignment.host.client.send(upstream_request, stream=True)
                break
            except httpx.HTTPError as exc:
                # Failed before the first response byte: retry once, on the next eligible host
                # in priority order. Never for "no capacity" (docs/spec/app-contract.md §5).
                host_id = assignment.host.host_id
                assignment.host.failures += 1
                await state.dispatcher.release(assignment)
                log.warning("host %s failed before first byte: %s", host_id, exc)
                excluded.add(host_id)
                if len(excluded) > 1:
                    await record("failed", status_code=503, reason="hosts_unreachable", host_id=host_id)
                    return _no_capacity("hosts_unreachable", f"two hosts failed before answering: {exc}")

        assert assignment is not None and upstream is not None
        held = assignment

        async def stream() -> AsyncIterator[bytes]:
            outcome = "ok"
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception:
                outcome = "stream_error"
                raise
            finally:
                if outcome == "ok":
                    held.host.requests_served += 1
                else:
                    held.host.failures += 1
                # The client may already be gone; closing the upstream request frees the
                # worker and stops work nobody is listening to.
                with anyio.CancelScope(shield=True):
                    await upstream.aclose()
                    await state.dispatcher.release(held)
                    await record(
                        outcome,
                        host_id=held.host.host_id,
                        worker_id=held.worker.worker_id,
                        model_served=held.variant.tag,
                        runtime_class=held.variant.runtime_class,
                        queue_wait_ms=wait_s * 1000,
                        latency_ms=(time.monotonic() - dispatched_at) * 1000,
                        status_code=upstream.status_code,
                    )

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=_downstream_headers(upstream, assignment, wait_s),
        )

    return app
