"""The router: the only process in the request path. Nothing slow or blocking belongs here.

It passes requests through in the engine's own API, rewriting the model field and nothing else,
and adds the small pool dialect described in docs/spec/app-contract.md §2.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Mapping, Optional

import anyio
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import ConfigError, PoolConfig, load_config
from ..contract import CONTRACT_VERSION
from ..db import Database, RequestRecord
from ..directory import read_directory
from ..keys import match, verify
from ..models import HostState
from ..provisioning_store import HashTaken
from ..state import RouterState, open_database
from .dispatch import Assignment, Need, NoEligibleHost, NoReadyHost, QueueTimeout

#: How much of a response's end is held to read the engine's own counts out of it (D67).
_USAGE_TAIL_BYTES = 4096
#: Requests a provisioning key may make a minute (D117): each may cost a market search.
REQUESTS_PER_MINUTE = 20

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
    # A workload whose own hosts are still coming up, and which may not borrow (D115).
    "workload_preparing": 30,
}


@dataclasses.dataclass(frozen=True)
class Identity:
    """Who a request is from, as its key says (D115, D117): the shared workload (an app key), one
    workload (its `gpmw_` key), or a program's provisioning key (`gpmp_`), which asks for
    workloads and never for a completion."""

    workload: Optional[str] = None
    provisioner: Optional[str] = None


def _identity(state: RouterState, request: Request) -> Optional[Identity]:
    """The request's workload, by its key — or None when the key reaches nothing. An app key is
    the shared workload's; a workload key is its workload's while it has not passed its time."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    if verify(token, state.app_hashes):
        return Identity()
    now = time.time()
    reach = match(token, state.registry.provisioners)
    if reach is not None:
        if reach.expires_at is not None and reach.expires_at <= now:
            return None
        return Identity(provisioner=reach.name)
    grant = match(token, state.registry.grants)
    if grant is None or (grant.not_after is not None and grant.not_after <= now):
        return None
    return Identity(workload=grant.workload)


def _peer_certificate_matches(request: Request, fingerprint: Optional[str]) -> bool:
    """Is this connection's client certificate — verified against the pool's client CA in the
    handshake — the one the workload's was signed as? Compared by fingerprint: the router never
    parses a certificate (D117). A workload with no fingerprint needs none."""
    if not fingerprint:
        return True
    der = getattr(request.state, "peer_cert", None) if hasattr(request, "state") else None
    return bool(der) and hashlib.sha256(der).hexdigest() == fingerprint


def _workload_prefix(path: str) -> tuple[Optional[str], str]:
    """`/w/<name>/rest` → (name, `/rest`); any other path → (None, path)."""
    if not path.startswith("/w/"):
        return None, path
    name, _, rest = path[3:].partition("/")
    return name, "/" + rest


class ClientGone(Exception):
    """The client disconnected while its request was still queued."""


def _authorised(state: RouterState, request: Request) -> bool:
    return _identity(state, request) is not None


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


def _downstream_headers(
    upstream: httpx.Response,
    assignment: Assignment,
    wait_s: float,
    delivery: str = "stream",
    attempts: int = 1,
) -> dict[str, str]:
    headers = {
        key: value for key, value in upstream.headers.items() if key.lower() not in _DROP_DOWNSTREAM
    }
    headers["X-GPM-Served-Model"] = assignment.variant.tag
    headers["X-GPM-Host"] = assignment.host.host_id
    headers["X-GPM-Runtime-Class"] = assignment.variant.runtime_class
    headers["X-GPM-Wait-S"] = f"{wait_s:.3f}"
    headers["X-GPM-Contract"] = CONTRACT_VERSION
    headers["X-GPM-Delivery"] = delivery
    headers["X-GPM-Attempts"] = str(attempts)
    return headers


def _delivery_for(state: RouterState, host_kind: str, request: Request) -> str:
    """Stream as the engine generates, or hold the response until it is whole? (D62)

    By host kind, because it is the kind that says whether the host can vanish mid-generation.
    An app may ask for tokens as they come, where the operator allows it.
    """
    cfg = state.config.pool.delivery
    wanted = cfg.for_kind(host_kind)
    asked = (request.headers.get("x-gpm-delivery") or "").strip().lower()
    if asked in ("stream", "buffered") and cfg.allow_request_override:
        return asked
    return wanted


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


def _ineligible_detail(
    path: str, serving: frozenset[str], ready: frozenset[str], paths_by_engine: Mapping[str, set[str]],
) -> str:
    """Why hosts are ready and none may take this request, said as specifically as the pool
    can tell.

    The commonest case has a specific answer: the request arrived on one engine's own API and
    every ready host runs another (D93). Found live: an app on Ollama's `/api/chat` against a
    pool whose only ready host ran vLLM was told "none holds a build of this model" ninety
    times, and the operator spent ten minutes and a parked host on a message that did not say
    the path was the problem — or that the `/v1` paths would have reached every engine.
    """
    if serving and ready and not (serving & ready):
        shared = sorted(
            set.intersection(*(set(paths_by_engine.get(name, set())) for name in serving | ready))
        ) if (serving | ready) <= set(paths_by_engine) else []
        served_by = ", ".join(sorted(serving))
        running = ", ".join(sorted(ready))
        detail = f"{path} is served by {served_by}; the hosts ready now run {running}, which does not serve it"
        if shared:
            detail += f" — {', '.join(shared)} would reach every engine this pool runs"
        return detail
    return "hosts are ready but none holds a build of this model that satisfies the request"


def _unready_reason(state: RouterState) -> str:
    """Why the shared workload has no ready host. A workload's hosts coming up is not the
    shared workload preparing."""
    if any(host.state is HostState.PREPARING and host.workload is None for host in state.hosts):
        return "preparing"
    return "hosts_unreachable"


async def follow_config_file(state: RouterState, path: Path) -> None:
    """Serve under the configuration file as it is now, not as it was when the router started.

    The supervisor applies an edit — from the console or by hand — the moment the file changes,
    and the router must follow, or the two disagree. Found live: a model added to the set was
    served by a rented host within minutes, but the router kept answering `/pool/status` with
    the set it had read at start, and the app, which checks names against that list, refused
    the model without ever asking. The file is read here, off the request path; one that does
    not load leaves the running configuration exactly as it was, as it does in the supervisor.
    """
    try:
        seen = path.stat().st_mtime
    except OSError:
        seen = None
    while True:
        await asyncio.sleep(state.config.pool.host_table_poll_s)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime == seen:
            continue
        seen = mtime
        try:
            new = await asyncio.to_thread(load_config, path)
        except ConfigError as exc:
            log.error("configuration did not load; keeping the running one: %s", exc)
            continue
        if new.listen != state.config.listen:
            log.warning("the listen address or TLS changed in %s; that takes a router restart", path)
        state.apply(new)
        log.info("configuration reloaded from %s", path)


def create_app(
    config: PoolConfig,
    database: Optional[Database] = None,
    db_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
) -> FastAPI:
    owns_database = database is None
    database = database or open_database(config, db_path)
    state = RouterState(config, database)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Pick up whatever the supervisor has already published before taking requests.
        await state.registry.refresh()
        tasks = [asyncio.create_task(state.registry.run_forever(state.config.pool.host_table_poll_s))]
        if config_path is not None:
            tasks.append(asyncio.create_task(follow_config_file(state, Path(config_path))))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
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
        who = _identity(state, request)
        if who is None:
            return _error(401, "unauthorized", detail="missing or invalid app key")
        if who.provisioner is not None:
            return _error(403, "forbidden", "provisioning_key", "a provisioning key asks for workloads; it reads no pool")
        if who.workload is not None:
            return _workload_status(who.workload)
        hosts = []
        for host in state.hosts:
            if host.workload is not None:
                continue  # a workload's hosts are its own business, not the shared view's
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
                "pool": state.config.pool.name,
                "contract_version": CONTRACT_VERSION,
                "engine": state.engine.name,
                "model_set": state.config.pool.model_set,
                "limits": {
                    "queue_timeout_s": state.config.pool.queue_timeout_s,
                    "client_time_to_first_byte_s": state.config.pool.client_time_to_first_byte_s,
                },
                # How a response reaches the app, per host kind (D62). An SDK sizes its
                # time-to-first-byte from this: a held response arrives whole, so "first byte"
                # is the end of the generation, not the start.
                "delivery": {
                    "by_kind": {
                        kind: state.config.pool.delivery.for_kind(kind)
                        for kind in ("local", "fixed-remote", "rented-interruptible", "rented-on-demand")
                    },
                    "allow_request_override": state.config.pool.delivery.allow_request_override,
                    "max_redispatch": state.config.pool.delivery.max_redispatch,
                },
                "capacity": {
                    "hosts_ready": sum(1 for h in state.hosts if h.state is HostState.READY and h.workload is None),
                    "workers_total": sum(h.total_workers for h in state.hosts if h.workload is None),
                    "workers_busy": sum(h.busy for h in state.hosts if h.workload is None),
                },
                "hosts": hosts,
            },
            headers={"X-GPM-Contract": CONTRACT_VERSION},
        )

    def _workload_now(name: str) -> tuple[str, Optional[str]]:
        """(state, model) as the router sees it now: `ending` from the lease's end time, whether
        or not the supervisor has noticed yet."""
        workload = state.registry.workloads.get(name)
        if workload is None:
            return "ended", None
        workload_state = workload.state
        if workload_state in ("preparing", "serving") and workload.ends_at is not None and workload.ends_at <= time.time():
            workload_state = "ending"
        return workload_state, workload.model

    # --- programs asking for their own workloads (D117) ---
    #
    # The router only records what was asked and reads back what the supervisor answered: the
    # two never call each other, and nothing slow runs here. Every field is checked again by the
    # supervisor; what is checked here only keeps obvious junk out of the table.

    provisioning = state.registry.provisioning_store

    def _provisioner_of(request: Request) -> tuple[Optional[str], Optional[JSONResponse]]:
        who = _identity(state, request)
        if who is None:
            return None, _error(401, "unauthorized", detail="missing or invalid key")
        if who.provisioner is None:
            return None, _error(403, "forbidden", "not_a_provisioning_key", "these calls take a provisioning key")
        if not state.config.provisioning.enabled:
            # Nothing would answer a request: say so now, rather than leave it pending for ever.
            return None, _error(403, "forbidden", "provisioning_disabled", "this pool does not take workloads from programs")
        return who.provisioner, None

    @app.post("/pool/provisioning/requests")
    async def provisioning_request(request: Request) -> Response:
        provisioner, refused = _provisioner_of(request)
        if refused is not None:
            return refused
        raw = await request.body()
        if len(raw) > 16_384:
            return _error(413, "too_large", "request_too_large", "a request is a few fields and a signing request")
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return _error(400, "bad_request", "not_json", "the body is JSON")
        if not isinstance(body, dict) or body.get("kind") not in ("plan", "create"):
            return _error(400, "bad_request", "bad_kind", "kind is plan or create")
        key_hash = body.get("key_hash") if body["kind"] == "create" else None
        if body["kind"] == "create" and not (isinstance(key_hash, str) and len(key_hash) == 64):
            return _error(400, "bad_request", "key_hash", "a create carries key_hash: the sha256 of the workload key, in hex")
        try:
            # A create sent again — its answer lost — is the same request, pending or not.
            existing = key_hash and await asyncio.to_thread(provisioning.of_key_hash, provisioner, key_hash)
            if existing:
                return JSONResponse({"request_id": existing, "new": False}, status_code=202,
                                    headers={"X-GPM-Contract": CONTRACT_VERSION})
            # One request at a time per key, and a few a minute: every one may cost the
            # supervisor a market search, which the whole pool waits on (D117).
            if await asyncio.to_thread(provisioning.pending_of, provisioner):
                return _error(429, "busy", "request_pending", "this key has a request waiting; ask again when it is answered")
            if await asyncio.to_thread(provisioning.asked_since, provisioner, time.time() - 60) >= REQUESTS_PER_MINUTE:
                return _error(429, "busy", "too_many_requests",
                              f"at most {REQUESTS_PER_MINUTE} requests a minute per provisioning key")
            request_id, new = await asyncio.to_thread(provisioning.ask, provisioner, body["kind"], body, key_hash)
        except HashTaken as exc:
            return _error(409, "conflict", "key_hash_taken", str(exc))
        return JSONResponse({"request_id": request_id, "new": new}, status_code=202,
                            headers={"X-GPM-Contract": CONTRACT_VERSION})

    @app.get("/pool/provisioning/requests/{request_id}")
    async def provisioning_answer(request: Request, request_id: str) -> Response:
        provisioner, refused = _provisioner_of(request)
        if refused is not None:
            return refused
        asked = await asyncio.to_thread(provisioning.get_request, request_id)
        if asked is None or asked.provisioner != provisioner:
            return _error(404, "not_found", "no_such_request", "no request of this key by that id")
        return JSONResponse(asked.view(), headers={"X-GPM-Contract": CONTRACT_VERSION})

    @app.get("/pool/provisioning/workloads/{name}")
    async def provisioning_workload(request: Request, name: str) -> Response:
        provisioner, refused = _provisioner_of(request)
        if refused is not None:
            return refused
        workload = state.registry.workloads.get(name)
        if workload is None or workload.provisioner != provisioner:
            return _error(404, "not_found", "no_such_workload", "no workload of this key by that name")
        workload_state, _ = _workload_now(name)
        mine = [h for h in state.hosts if h.workload == name]
        return JSONResponse({
            "workload": name, "state": workload_state, "model": workload.model, "ends_at": workload.ends_at,
            "hosts_ready": sum(1 for h in mine if h.state is HostState.READY), "hosts": len(mine),
        }, headers={"X-GPM-Contract": CONTRACT_VERSION})

    @app.post("/pool/provisioning/workloads/{name}/end")
    async def provisioning_end(request: Request, name: str) -> Response:
        provisioner, refused = _provisioner_of(request)
        if refused is not None:
            return refused
        workload = state.registry.workloads.get(name)
        if workload is None or workload.provisioner != provisioner:
            return _error(404, "not_found", "no_such_workload", "no workload of this key by that name")
        # An end already waiting is the same end; otherwise the same limits as every request.
        waiting = await asyncio.to_thread(provisioning.pending_end, provisioner, name)
        if waiting is not None:
            return JSONResponse({"request_id": waiting}, status_code=202, headers={"X-GPM-Contract": CONTRACT_VERSION})
        if await asyncio.to_thread(provisioning.asked_since, provisioner, time.time() - 60) >= REQUESTS_PER_MINUTE:
            return _error(429, "busy", "too_many_requests", f"at most {REQUESTS_PER_MINUTE} requests a minute per provisioning key")
        request_id, _ = await asyncio.to_thread(provisioning.ask, provisioner, "end", {}, None, name)
        return JSONResponse({"request_id": request_id}, status_code=202, headers={"X-GPM-Contract": CONTRACT_VERSION})

    def _delivery() -> dict:
        return {
            "by_kind": {
                kind: state.config.pool.delivery.for_kind(kind)
                for kind in ("local", "fixed-remote", "rented-interruptible", "rented-on-demand")
            },
            "allow_request_override": state.config.pool.delivery.allow_request_override,
            "max_redispatch": state.config.pool.delivery.max_redispatch,
        }

    def _workload_status(name: str) -> Response:
        """One workload's view, for its own key: its hosts, its state, whether it is borrowing
        — never the rest of the pool (workloads.md §3)."""
        workload_state, model = _workload_now(name)
        mine = [h for h in state.hosts if h.workload == name]
        ready = [h for h in mine if h.state is HostState.READY]
        plan = getattr(state.registry.workloads.get(name), "plan", None) or {}
        borrowing = (
            workload_state == "preparing" and not ready and state.config.workloads.borrow_share > 0
            and bool(plan.get("may_borrow", True))
        )
        return JSONResponse(
            {
                "pool": state.config.pool.name,
                "workload": name,
                "state": workload_state,
                "contract_version": CONTRACT_VERSION,
                "model_set": [model] if model else [],
                "borrowing": borrowing,
                "limits": {
                    "queue_timeout_s": state.config.pool.queue_timeout_s,
                    "client_time_to_first_byte_s": state.config.pool.client_time_to_first_byte_s,
                },
                "delivery": _delivery(),
                "capacity": {
                    "hosts_ready": len(ready),
                    "workers_total": sum(h.total_workers for h in mine),
                    "workers_busy": sum(h.busy for h in mine),
                },
                "hosts": [
                    {"host_id": h.host_id, "kind": h.kind, "state": h.state.value,
                     "workers": {"total": h.total_workers, "busy": h.busy}}
                    for h in mine
                ],
            },
            headers={"X-GPM-Contract": CONTRACT_VERSION},
        )

    @app.get("/w/{name}/pool/status")
    async def workload_pool_status(request: Request, name: str) -> Response:
        who = _identity(state, request)
        if who is None:
            return _error(401, "unauthorized", detail="missing or invalid app key")
        if who.workload != name:
            return _error(403, "forbidden", "wrong_workload", "this key is not for that workload")
        return _workload_status(name)

    @app.get("/pool/directory")
    async def pool_directory(request: Request, q: Optional[str] = None) -> Response:
        """What this pool could serve, beyond what it serves now (D101): the model directory the
        supervisor keeps, read from the shared file. No outbound request is made here, ever —
        this process carries inference traffic — and nothing here changes the pool: `in_pool`
        says which names a request may use today.
        """
        who = _identity(state, request)
        if who is None:
            return _error(401, "unauthorized", detail="missing or invalid app key")
        if who.workload is not None or who.provisioner is not None:
            # The directory says what the shared pool could serve; a workload serves one model.
            return _error(403, "forbidden", "workload_key", "a workload key reaches its workload's model only")
        found = await asyncio.to_thread(read_directory, database, q, tuple(state.config.pool.model_set))
        return JSONResponse(found, headers={"X-GPM-Contract": CONTRACT_VERSION})

    @app.post("/{full_path:path}")
    async def inference(request: Request, full_path: str) -> Response:
        named, path = _workload_prefix("/" + full_path)
        # The **path** chooses how the request is read, not the host: it has to be understood
        # before the pool knows where it will go (D93). Engines that serve the same path read it
        # identically, by construction — they share one module for it.
        engine = state.parser_for(path) or state.engine
        request_id = uuid.uuid4().hex
        session_id = request.headers.get("x-gpm-session")
        started = time.monotonic()

        who = _identity(state, request)
        if who is None:
            return _error(401, "unauthorized", detail="missing or invalid app key")
        if who.provisioner is not None:
            return _error(403, "forbidden", "provisioning_key",
                          "a provisioning key creates workloads; it never requests a completion — use the workload's key")
        if named is not None and named != who.workload:
            # A client pointed at one workload with another's key fails loudly, rather than
            # quietly serving under whichever the key happens to reach.
            return _error(403, "forbidden", "wrong_workload", f"this key is not for workload {named!r}")
        workload_state, workload_model = (None, None)
        if who.workload is not None:
            workload_state, workload_model = _workload_now(who.workload)
            spec = state.registry.workloads.get(who.workload)
            if spec is not None and not _peer_certificate_matches(request, spec.cert_fingerprint):
                return _error(
                    403, "forbidden", "client_certificate_required",
                    f"workload {who.workload!r} is reached with its client certificate; this connection did not present it",
                )
            if workload_state not in ("preparing", "serving"):
                return _error(
                    503, "workload_ended", "workload_ended",
                    f"workload {who.workload!r} has ended; its key no longer reaches anything. "
                    "Ask its operator for a new workload.",
                )
        if path not in state.paths():
            served = ", ".join(sorted(state.engines))
            return _error(
                404, "not_found", "unknown_path",
                f"{path} is not an inference path of any engine this pool runs ({served})",
            )

        body = await request.body()
        requested = engine.requested_model(path, body)

        async def record(outcome: str, **fields: object) -> None:
            await state.log.record(
                RequestRecord(
                    request_id=request_id,
                    outcome=outcome,
                    session_id=session_id,
                    model_requested=requested,
                    workload=who.workload,
                    **fields,  # type: ignore[arg-type]
                )
            )

        if requested is None:
            await record("rejected", status_code=400, reason="model_missing")
            return _error(400, "bad_request", "model_missing", "the request body names no model")

        if who.workload is not None:
            in_pool = requested == workload_model
            if not in_pool:
                await record("rejected", status_code=404, reason="model_not_in_workload")
                return _error(
                    404, "model_not_in_pool", "model_not_in_workload",
                    f"workload {who.workload!r} serves {workload_model!r}, not {requested!r}",
                )
        in_pool = (
            who.workload is not None  # its own model, checked above
            or requested in state.config.pool.model_set or state.dispatcher.knows_model(requested)
        )
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
            # Only hosts whose engine serves this path (D93): a request on one engine's own API
            # would be a 404 on a host running the other, after the pool had called it eligible.
            engines=frozenset(
                name for name, candidate in state.engines.items() if path in candidate.inference_paths()
            ),
            workload=who.workload,
            # Only while the workload prepares: one that could always borrow would never need
            # sizing (D115).
            may_borrow=(workload_state == "preparing" and state.config.workloads.borrow_share > 0
                        and bool((getattr(state.registry.workloads.get(who.workload), "plan", None) or {}).get("may_borrow", True))),
            borrow_share=state.config.workloads.borrow_share,
        )

        request_deadline = _parse_deadline(request.headers.get("x-gpm-deadline"))
        if request_deadline is not None and request_deadline <= time.monotonic():
            await record("rejected", status_code=504, reason="deadline_exceeded")
            return _error(504, "deadline_exceeded", "deadline_exceeded", "the request's deadline had already passed")
        queue_deadline = started + state.config.pool.queue_timeout_s
        if request_deadline is not None:
            queue_deadline = min(queue_deadline, request_deadline)

        excluded: set[str] = set()
        assignment: Optional[Assignment] = None
        upstream: Optional[httpx.Response] = None
        wait_s = 0.0
        dispatched_at = started
        delivery = "stream"
        raw: Optional[AsyncIterator[bytes]] = None
        buffered: list[bytes] = []
        attempts = 0
        redispatched = 0
        max_buffer_bytes = int(state.config.pool.delivery.max_buffer_mb * 1_000_000)

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
                if who.workload is not None:
                    await record("rejected", status_code=503, reason="workload_preparing")
                    return _no_capacity(
                        "workload_preparing",
                        f"workload {who.workload!r} has no host ready yet"
                        + ("" if need.may_borrow else ", and it may not borrow the shared hosts"),
                    )
                reason = _unready_reason(state)
                await record("rejected", status_code=503, reason=reason)
                return _no_capacity(reason, "no host is ready to serve this request")
            except NoEligibleHost:
                await record("rejected", status_code=503, reason="no_eligible_host")
                return _no_capacity(
                    "no_eligible_host",
                    _ineligible_detail(
                        path,
                        need.engines,
                        frozenset(host.engine for host in state.hosts if host.state is HostState.READY),
                        {name: candidate.inference_paths() for name, candidate in state.engines.items()},
                    ),
                )
            except QueueTimeout:
                if request_deadline is not None and time.monotonic() >= request_deadline:
                    await record("rejected", status_code=504, reason="deadline_exceeded")
                    return _error(504, "deadline_exceeded", "deadline_exceeded", "the request's deadline passed while queued")
                await record("rejected", status_code=503, reason="queue_timeout", queue_wait_ms=(time.monotonic() - started) * 1000)
                return _no_capacity("queue_timeout", "every eligible worker stayed busy past the queue limit")

            assignment.host.last_request_at = time.time()
            concurrency = assignment.concurrency
            host_engine = state.engine_of(assignment.host)
            forwarded = host_engine.with_model(path, body, assignment.variant.tag)
            upstream_request = assignment.host.client.build_request(
                "POST", path, content=forwarded, headers=_upstream_headers(request)
            )
            # Latency runs from the moment the engine is asked. For a non-streaming call the
            # headers arrive only after the whole generation, so measuring from after `send`
            # would report a 46-second answer as under a millisecond (seen live).
            dispatched_at = time.monotonic()
            attempts += 1
            try:
                upstream = await assignment.host.client.send(upstream_request, stream=True)
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
                continue

            delivery = _delivery_for(state, assignment.host.kind, request)
            if delivery != "buffered" or upstream.status_code >= 400:
                # Streamed as the engine generates it, which is also how an error answer goes
                # back: there is nothing to gain by holding a 4xx the engine already decided.
                delivery = "stream"
                break

            # Held until whole (D62). Until the first byte reaches the client, a host lost
            # mid-generation costs a re-run rather than a broken stream.
            buffered = []
            size = 0
            overflowed = False
            raw = upstream.aiter_raw()
            try:
                async for chunk in raw:
                    buffered.append(chunk)
                    size += len(chunk)
                    if size > max_buffer_bytes:
                        # Bigger than the pool will hold: stream the rest rather than fail it.
                        overflowed = True
                        break
            except asyncio.CancelledError:
                with anyio.CancelScope(shield=True):
                    await upstream.aclose()
                    await state.dispatcher.release(assignment)
                    await record("cancelled", reason="client_disconnected", host_id=assignment.host.host_id,
                                 queue_wait_ms=wait_s * 1000)
                raise
            except Exception as exc:  # noqa: BLE001 — every upstream failure means the same here
                host_id = assignment.host.host_id
                assignment.host.failures += 1
                with anyio.CancelScope(shield=True):
                    await upstream.aclose()
                    await state.dispatcher.release(assignment)
                log.warning("host %s failed while buffering: %s", host_id, exc)
                excluded.add(host_id)
                await record(
                    "redispatched", host_id=host_id, worker_id=assignment.worker.worker_id,
                    model_served=assignment.variant.tag, runtime_class=assignment.variant.runtime_class,
                    queue_wait_ms=wait_s * 1000, latency_ms=(time.monotonic() - dispatched_at) * 1000,
                    reason="upstream_lost_while_buffering",
                )
                if redispatched >= state.config.pool.delivery.max_redispatch or (
                    request_deadline is not None and time.monotonic() >= request_deadline
                ):
                    await record("failed", status_code=503, reason="host_lost", host_id=host_id)
                    return _no_capacity(
                        "host_lost",
                        "the host serving this request was lost before any of it reached you; "
                        "nothing partial was sent",
                    )
                redispatched += 1
                # The client has received nothing, so the pool runs it again itself.
                queue_deadline = time.monotonic() + state.config.pool.queue_timeout_s
                if request_deadline is not None:
                    queue_deadline = min(queue_deadline, request_deadline)
                continue

            if overflowed:
                delivery = "stream-after-overflow"
            break

        assert assignment is not None and upstream is not None
        held = assignment
        prefix = b"".join(buffered)
        rest = raw if prefix else upstream.aiter_raw()

        if delivery == "buffered":
            # Whole, and byte-identical to what the engine sent: the frames are the engine's
            # own, in order, so a client that asked for a stream still parses a stream.
            held.host.requests_served += 1
            with anyio.CancelScope(shield=True):
                await upstream.aclose()
                await state.dispatcher.release(held)
                tokens_out, generate_ms = engine.usage(path, prefix[-_USAGE_TAIL_BYTES:])
                await record(
                    "ok",
                    host_id=held.host.host_id,
                    worker_id=held.worker.worker_id,
                    model_served=held.variant.tag,
                    runtime_class=held.variant.runtime_class,
                    queue_wait_ms=wait_s * 1000,
                    latency_ms=(time.monotonic() - dispatched_at) * 1000,
                    status_code=upstream.status_code,
                    tokens_out=tokens_out,
                    generate_ms=generate_ms,
                    borrowed=held.borrowed,
                    concurrency=concurrency,
                )
            return Response(
                content=prefix,
                status_code=upstream.status_code,
                headers=_downstream_headers(upstream, assignment, wait_s, delivery, attempts),
            )

        async def stream() -> AsyncIterator[bytes]:
            outcome = "ok"
            # The counts live in the final frame, so a few kilobytes of tail is all it takes —
            # and nothing of what was generated is kept beyond the moment it is read (D67).
            tail = bytearray()
            try:
                if prefix:
                    # What was buffered before the response outgrew the pool's limit; the rest
                    # continues from the same iterator, never a second pass over the stream.
                    yield prefix
                async for chunk in rest:
                    tail.extend(chunk)
                    del tail[:-_USAGE_TAIL_BYTES]
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
                    tokens_out, generate_ms = engine.usage(path, bytes(tail))
                    await record(
                        outcome,
                        host_id=held.host.host_id,
                        worker_id=held.worker.worker_id,
                        model_served=held.variant.tag,
                        runtime_class=held.variant.runtime_class,
                        queue_wait_ms=wait_s * 1000,
                        latency_ms=(time.monotonic() - dispatched_at) * 1000,
                        status_code=upstream.status_code,
                        tokens_out=tokens_out,
                        generate_ms=generate_ms,
                        borrowed=held.borrowed,
                        concurrency=concurrency,
                    )

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=_downstream_headers(upstream, assignment, wait_s, delivery, attempts),
        )

    return app
