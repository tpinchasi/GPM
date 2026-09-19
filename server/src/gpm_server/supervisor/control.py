"""The control API — everything that spends money or changes the pool.

docs/spec/console-and-control-api.md. It is served by the **supervisor**, not the router, and
the **admin key is required on every request**, as a header and never a cookie, with `Host` and
`Origin` checked — on loopback too, because any web page open in the same browser can post to
127.0.0.1 (threat model T1). **The app key is refused here** (T3).

Changing configuration never spends money. Spending is a lease or a host preparation, each
behind a call that states the worst case.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Optional, Sequence
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..catalog import variants_for_host
from ..config import ConfigError, PoolConfig
from ..configplan import ConfigStore, MachineNow, RentedNow, StaleVersion, plan_changes
from ..contract import CONTRACT_VERSION
from ..hostcheck import test_connection
from ..keys import verify
from ..ledger import LeaseRefused
from . import agents
from .service import Supervisor

log = logging.getLogger("gpm.control")


def _error(status_code: int, error: str, detail: Optional[str] = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


def _same_origin(request: Request, allowed_hosts: set[str]) -> bool:
    """A browser attaches Origin on cross-site requests; a mismatched one is a page on another
    site driving this API."""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    host = urlparse(origin).hostname or ""
    return host in allowed_hosts


def create_control_app(supervisor: Supervisor, config: PoolConfig) -> FastAPI:
    admin_hashes = config.auth.admin_hashes()
    app_hashes = config.auth.app_hashes()
    allowed_hosts = {"127.0.0.1", "localhost", "::1", config.control.host}

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(
        title="GPM control API",
        version=CONTRACT_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authorised(request: Request) -> Optional[JSONResponse]:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            return _error(401, "unauthorized", "the control API requires the admin key")
        if verify(token, app_hashes) and not verify(token, admin_hashes):
            # Said plainly, because this is a mistake an operator will otherwise repeat.
            return _error(
                403,
                "app_key_refused",
                "that is the app key. An app that can request a completion must not be able "
                "to spend; use the admin key here.",
            )
        if token.startswith("gpmg_"):
            # The third role (docs/spec/host-agent.md §5): it admits the pool to one host's
            # agent, and must never admit anything to the pool.
            return _error(403, "agent_key_refused", "that is an agent key; use the admin key here.")
        if not verify(token, admin_hashes):
            return _error(401, "unauthorized", "unknown admin key")
        if not _same_origin(request, allowed_hosts):
            return _error(403, "bad_origin", "this request came from another site")
        return None

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # The page itself holds no data and no secret; the key is entered into it, and every
        # call it then makes carries the key as a header like any other client.
        if request.url.path == "/ui" or request.url.path.startswith("/ui/"):
            response = await call_next(request)
            # Without this a browser keeps the page's script for as long as it likes and a
            # plain reload does not ask again — so after an upgrade, or a fix, the operator
            # is still running the old console and nothing tells them. `no-cache` means "ask
            # every time"; the ETag makes the answer a cheap 304 when nothing changed.
            response.headers["cache-control"] = "no-cache"
            return response
        refusal = authorised(request)
        if refusal is not None:
            return refusal
        return await call_next(request)

    static_dir = Path(__file__).resolve().parent.parent / "console" / "static"
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="console")

    # --- reading ---

    def served_on(
        variants: Mapping[str, Sequence[Any]], resident: frozenset[str], available: frozenset[str]
    ) -> dict[str, Any]:
        """Per logical model, the variant the pool resolved for this host — and whether *that
        tag* is loaded, and on disk. The logical name itself is never what the engine holds."""
        return {
            name: {
                "tag": group[0].tag,
                "runtime_class": group[0].runtime_class,
                "enforces_schema": group[0].enforces_schema,
                "resident": group[0].tag in resident,
                "available": group[0].tag in available,
            }
            for name, group in variants.items()
            if group
        }

    def machines_now() -> dict[str, MachineNow]:
        """What each answering agent last said, for a plan to reason from."""
        found: dict[str, MachineNow] = {}
        for host in supervisor.hosts.values():
            view = host.agent
            if view is None or not view.reachable or view.facts is None:
                continue
            engine = view.facts.get("engine") or {}
            found[host.host_id] = MachineNow(
                capabilities=view.derived_capabilities,
                on_disk=frozenset(m["tag"] for m in engine.get("models_on_disk", [])),
                free_disk_bytes=(view.facts.get("disk") or {}).get("free_bytes"),
            )
        return found

    def status_payload() -> dict[str, Any]:
        fleet = supervisor.fleet
        counters = supervisor.counters.all()
        hosts: list[dict[str, Any]] = []
        for host in supervisor.hosts.values():
            counter = counters.get(host.host_id)
            entry: dict[str, Any] = {
                "host_id": host.host_id,
                "kind": host.config.kind,
                "transport": host.config.transport.type,
                "priority": host.config.routing_priority,
                "state": host.state.value,
                "workers": host.config.workers,
                "busy": counter.busy if counter else 0,
                "requests_served": counter.requests_served if counter else 0,
                # What builds are resolved against: the configured list, plus the platform the
                # agent found where the list names none.
                "capabilities": sorted(host.capabilities),
                "configured_capabilities": sorted(host.config.capabilities),
                "resident": sorted(host.resident),
                "available": sorted(host.available),
                "residency": host.config.residency,
                "served": served_on(host.variants, host.resident, host.available),
                "last_error": host.last_error,
            }
            tunnel = supervisor.tunnel_status(host.host_id)
            if tunnel is not None:
                entry["tunnel"] = tunnel
            if host.config.agent is not None:
                view = host.agent
                entry["agent"] = {
                    "url": host.config.agent.url or f"127.0.0.1:{host.config.agent.remote_port} on the host, over its SSH tunnel",
                    "tunnel": (
                        {"up": host.agent_tunnel.up, "local_port": host.agent_tunnel.local_port,
                         "restarts": host.agent_tunnel.restarts, "last_error": host.agent_tunnel.last_error}
                        if host.agent_tunnel is not None else None
                    ),
                    "key": "set" if host.config.agent.key() else "missing",  # never the key
                    "reachable": bool(view and view.reachable),
                    "detail": view.detail if view else "not asked yet",
                    "asked_at": view.asked_at if view else None,
                    "facts": view.facts if view and view.reachable else None,
                    "manages_models": host.config.agent.manage_models,
                    "wanted_engine_settings": agents.wanted_engine_settings(host.config.workers, len(host.required_tags)),
                    "models": view.models if view and view.reachable else None,
                    "capability_conflict": agents.capability_conflict(host.config.capabilities, view),
                }
            hosts.append(entry)

        rented = []
        if fleet is not None:
            rented_variants = variants_for_host(
                supervisor.config.pool.model_set,
                supervisor.config.catalog,
                frozenset(fleet.rented.capabilities),
                supervisor.engine.name,
            )
            for host in fleet.hosts.values():
                counter = counters.get(host.host_id)
                rented.append(
                    {
                        "host_id": host.host_id,
                        "state": host.state,
                        "resident": sorted(getattr(host, "resident", frozenset())),
                        # The pool built this host and pinned the set on it; a loaded tag is
                        # on disk by definition, and the rest is not probed separately.
                        "residency": "pinned",
                        "served": served_on(
                            rented_variants,
                            getattr(host, "resident", frozenset()),
                            getattr(host, "resident", frozenset()),
                        ),
                        "machine": host.offer.machine_id,
                        "hardware": host.offer.hardware,
                        "bid_hourly": host.bid_hourly,
                        "storage_hourly": round(host.offer.storage_hourly, 5),
                        "estimated_spend": round(host.estimate(), 4),
                        "reported_spend": round(host.reported_spend, 4),
                        "lease_id": host.lease_id,
                        "parked_at": host.parked_at,
                        "hours_held": round(host.hours_held, 3),
                        "workers": fleet.rented.workers,
                        "busy": counter.busy if counter else 0,
                    }
                )

        leases = []
        for lease in supervisor.leases.open_leases():
            entry = {
                "lease_id": lease.lease_id,
                "workers": lease.workers,
                "max_hours": lease.max_hours,
                "max_spend": lease.max_spend,
                "allow_rent": lease.allow_rent,
                "hours_left": round(lease.hours_left(), 3),
            }
            if fleet is not None:
                spent, estimate = fleet.lease_spend(lease)
                entry |= {
                    "spent_enforced_on": round(spent, 4),
                    "estimated_spend": round(estimate, 4),
                    "dollars_left": round(fleet.budget_left(lease), 4),
                    "stops_at": round(lease.max_spend * (1 - fleet.margin()), 4),
                }
            leases.append(entry)

        return {
                "pool": supervisor.config.pool.name,
                "contract_version": CONTRACT_VERSION,
                "as_of": time.time(),
                # The Models screen draws these, and they change when configuration is
                # reloaded — so they are read from the supervisor, not from the config the
                # app was built with.
                "model_set": list(supervisor.config.pool.model_set),
                "catalog": {
                    name: [
                        {
                            "tag": variant.tag,
                            "requires": list(variant.requires),
                            "runtime_class": variant.runtime_class,
                            "enforces_schema": variant.enforces_schema,
                        }
                        for variant in entry.variants
                    ]
                    for name, entry in supervisor.config.catalog.items()
                },
                "hosts": hosts,
                "rented": rented,
                "open_leases": leases,
                "limits": {
                    "max_rented_hosts": supervisor.config.limits.max_rented_hosts,
                    "max_hourly_burn": supervisor.config.limits.max_hourly_burn,
                },
                "provider": (
                    {
                        "name": fleet.provider.name,
                        "capabilities": vars(fleet.provider.capabilities),
                        "cap_safety_margin": fleet.margin(),
                    }
                    if fleet is not None
                    else None
                ),
            }

    @app.get("/pool/status")
    async def status() -> JSONResponse:
        return JSONResponse(status_payload())

    @app.get("/pool/events/stream")
    async def events_stream(request: Request, after: int = 0) -> StreamingResponse:
        """Server-sent events: every new decision as it is recorded, and a status frame every
        few seconds, so the console never polls. Read with `fetch`, since the browser's own
        EventSource cannot send the admin key as a header."""

        async def frames() -> AsyncIterator[bytes]:
            last = after
            next_status = 0.0
            yield b"retry: 3000\n\n"
            while not await request.is_disconnected():
                for event in supervisor.events.since(last):
                    last = event["id"]
                    yield f"id: {last}\nevent: decision\ndata: {json.dumps(event)}\n\n".encode()
                if time.monotonic() >= next_status:
                    yield f"event: status\ndata: {json.dumps(status_payload())}\n\n".encode()
                    next_status = time.monotonic() + 3.0
                await asyncio.sleep(1.0)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/pool/events")
    async def events(limit: int = 100, kind: Optional[str] = None) -> JSONResponse:
        return JSONResponse({"events": supervisor.events.recent(limit=limit, kind=kind)})

    @app.get("/pool/leases")
    async def leases() -> JSONResponse:
        fleet = supervisor.fleet
        out = []
        for lease in supervisor.leases.all():
            entry = {
                "lease_id": lease.lease_id,
                "workers": lease.workers,
                "max_hours": lease.max_hours,
                "max_spend": lease.max_spend,
                "allow_rent": lease.allow_rent,
                "state": lease.state,
                "hours_left": round(lease.hours_left(), 3),
                "closed_reason": lease.closed_reason,
            }
            if fleet is not None:
                spent, estimate = fleet.lease_spend(lease)
                entry |= {
                    "spent_enforced_on": round(spent, 4),
                    "estimated_spend": round(estimate, 4),
                    "dollars_left": round(fleet.budget_left(lease), 4),
                }
            out.append(entry)
        return JSONResponse({"leases": out})

    @app.get("/pool/plan")
    async def plan() -> JSONResponse:
        """What the supervisor would do right now. Spends nothing (spec §3)."""
        if supervisor.fleet is None:
            return JSONResponse({"plan": [], "detail": "this pool cannot rent"})
        return JSONResponse({"plan": await supervisor.fleet.plan(supervisor._ready_workers())})

    @app.get("/pool/market/preview")
    async def market_preview(hours: float = 4.0) -> JSONResponse:
        """Read-only: the live market through the pool's own filters. Spends nothing."""
        if supervisor.fleet is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity configured")
        return JSONResponse(await supervisor.fleet.market_preview(hours=hours))

    @app.post("/pool/market/preview")
    async def market_preview_unsaved(request: Request, hours: float = 4.0) -> JSONResponse:
        """The same pipeline with the values **currently in the form, not yet saved** — which
        is what makes moving a ceiling and watching "4 pass" become "0 pass" possible."""
        if supervisor.fleet is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity configured")
        body = await request.json()
        try:
            return JSONResponse(
                await supervisor.fleet.market_preview(
                    hours=hours,
                    offer_policy=body.get("offer_policy"),
                    bidding=body.get("bidding"),
                )
            )
        except (TypeError, ValueError) as exc:
            return _error(400, "bad_policy", str(exc))

    # --- configuration: never spends money ---

    def store() -> Optional[ConfigStore]:
        return supervisor.store

    def rented_now() -> list[RentedNow]:
        if supervisor.fleet is None:
            return []
        return [
            RentedNow(host_id=h.host_id, bid_hourly=h.bid_hourly)
            for h in supervisor.fleet.hosts.values()
            if not h.released
        ]

    @app.get("/pool/config")
    async def get_config() -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        text, version = store().read()
        return JSONResponse({"text": text, "version": version, "path": str(store().path)})

    @app.post("/pool/config/validate")
    async def validate_config(request: Request) -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        errors = store().validate((await request.json()).get("text", ""))
        return JSONResponse({"ok": not errors, "errors": errors})

    @app.post("/pool/config/plan")
    async def plan_config(request: Request) -> JSONResponse:
        """What this change would cause, right now. Applies nothing, spends nothing."""
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        text = (await request.json()).get("text", "")
        errors = store().validate(text)
        if errors:
            return JSONResponse({"errors": errors, "changes": []})
        candidate = store().parse(text)
        changes = plan_changes(supervisor.config, candidate, rented_now(), machines_now())
        return JSONResponse({"errors": [], "changes": [change.as_dict() for change in changes]})

    @app.put("/pool/config")
    async def put_config(request: Request) -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        body = await request.json()
        try:
            version = store().apply(body.get("text", ""), body.get("version"))
        except StaleVersion as exc:
            return _error(409, "stale_version", str(exc))
        except ConfigError as exc:
            return _error(400, "invalid_config", str(exc))
        supervisor.reload_config()
        return JSONResponse({"version": version})

    @app.get("/pool/config/history")
    async def config_history() -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        return JSONResponse({"versions": store().history()})

    @app.post("/pool/config/rollback")
    async def rollback_config(request: Request) -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        try:
            version = store().rollback((await request.json()).get("version", ""))
        except ConfigError as exc:
            return _error(400, "invalid_config", str(exc))
        supervisor.reload_config()
        return JSONResponse({"version": version})

    @app.post("/pool/hosts/test")
    async def test_host(request: Request) -> JSONResponse:
        """Test connection for an **unsaved** host definition. Saves nothing, spends nothing."""
        body = await request.json()
        try:
            result = await test_connection(
                body,
                engine=supervisor.engine,
                settings=supervisor.config.pool,
                model_set=supervisor.config.pool.model_set,
                catalog=supervisor.config.catalog,
            )
        except (TypeError, ValueError) as exc:
            return _error(400, "bad_host", str(exc))
        return JSONResponse(result)

    @app.get("/pool/account")
    async def account() -> JSONResponse:
        """Is the provider credential valid, and how much credit is left. Spends nothing."""
        if supervisor.fleet is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity configured")
        status = await supervisor.fleet.provider.account()
        return JSONResponse(
            {
                "provider": supervisor.fleet.provider.name,
                "credential_valid": status.credential_valid,
                "credit_remaining": status.credit_remaining,
            }
        )

    # --- spending ---

    @app.post("/pool/leases")
    async def open_lease(request: Request) -> JSONResponse:
        body = await request.json()
        fleet = supervisor.fleet
        try:
            if fleet is not None:
                lease = fleet.open_lease(
                    workers=int(body["workers"]),
                    max_hours=float(body.get("max_hours", 4)),
                    max_spend=body.get("max_spend"),
                    allow_rent=bool(body.get("allow_rent", False)),
                    bid_ceiling=body.get("bid_ceiling"),
                )
            else:
                lease = supervisor.leases.open(
                    workers=int(body["workers"]),
                    max_hours=float(body.get("max_hours", 4)),
                    max_spend=body.get("max_spend"),
                    allow_rent=bool(body.get("allow_rent", False)),
                )
        except LeaseRefused as exc:
            return _error(400, "lease_refused", str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            return _error(400, "bad_request", str(exc))
        return JSONResponse(
            {
                "lease_id": lease.lease_id,
                "worst_case": {
                    "dollars": lease.max_spend,
                    "hours": lease.max_hours,
                    "max_rented_hosts": config.limits.max_rented_hosts,
                    "max_hourly_burn": config.limits.max_hourly_burn,
                },
            },
            status_code=201,
        )

    @app.delete("/pool/leases/{lease_id}")
    async def close_lease(lease_id: str) -> JSONResponse:
        supervisor.leases.close(lease_id, "closed through the control API")
        return JSONResponse({"lease_id": lease_id, "state": "closed"})

    @app.patch("/pool/leases/{lease_id}")
    async def tighten_lease(lease_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        try:
            lease = supervisor.leases.tighten(
                lease_id,
                max_spend=body.get("max_spend"),
                max_hours=body.get("max_hours"),
                workers=body.get("workers"),
            )
        except LeaseRefused as exc:
            return _error(400, "lease_refused", str(exc))
        return JSONResponse({"lease_id": lease.lease_id, "max_spend": lease.max_spend})

    @app.post("/pool/hosts/prepare")
    async def prepare_host(request: Request) -> JSONResponse:
        body = await request.json()
        if supervisor.fleet is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity configured")
        try:
            host = await supervisor.fleet.prepare(
                max_spend=float(body["max_spend"]),
                max_hours=float(body.get("max_hours", 1)),
                bid_ceiling=body.get("bid_ceiling"),
                when_ready=body.get("when_ready", "join"),
            )
        except LeaseRefused as exc:
            return _error(400, "lease_refused", str(exc))
        if host is None:
            return _error(503, "nothing_prepared", "no acceptable offer; nothing was spent")
        return JSONResponse(
            {"host_id": host.host_id, "lease_id": host.lease_id, "bid_hourly": host.bid_hourly},
            status_code=201,
        )

    @app.post("/pool/hosts/{host_id}/engine/restart")
    async def restart_host_engine(host_id: str, request: Request) -> JSONResponse:
        """Restart a host's engine through its agent, optionally after writing the settings the
        pool needs it to run with (D41). An operator's explicit act: a restart drops whatever
        the engine is doing, so the supervisor's own pass never calls this."""
        host = supervisor.hosts.get(host_id)
        if host is None or host.config.agent is None:
            return _error(404, "no_agent", f"no configured host {host_id!r} with an agent")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        if body.get("confirm") != host_id:
            return _error(400, "not_confirmed", "type the host id again as `confirm`: requests in flight on that host will fail over or fail")
        settings = (
            agents.wanted_engine_settings(host.config.workers, len(host.required_tags))
            if body.get("apply_settings") else None
        )
        status_code, answer = await agents.restart_engine(host.agent_endpoint, settings, transport=supervisor._agent_transport)
        if status_code == 200:
            supervisor.events.record(
                "agent_engine_restarted",
                f"host {host_id}: engine restarted on the operator's instruction"
                + (" with the pool's settings" if settings else "")
                + ("" if answer.get("engine_answers") else " — and it is NOT answering"),
                host_id=host_id,
                numbers={**(settings or {}), "exit_code": answer.get("restart_exit_code"), "settings_written": answer.get("settings_written")},
            )
        return JSONResponse(status_code=status_code, content=answer)

    @app.post("/pool/hosts/{host_id}/models/delete")
    async def delete_host_model(host_id: str, request: Request) -> JSONResponse:
        """Delete one model from a host's disk, through its agent (D40). An operator's explicit
        act and nothing else: the supervisor's own pass never calls this. Three refusals stand
        between the button and the disk — this one, and two the agent makes for itself."""
        host = supervisor.hosts.get(host_id)
        if host is None or host.config.agent is None:
            return _error(404, "no_agent", f"no configured host {host_id!r} with an agent")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        tag, confirm = body.get("tag"), body.get("confirm")
        if not isinstance(tag, str) or not tag:
            return _error(400, "bad_request", "name the tag to delete")
        if confirm != tag:
            return _error(400, "not_confirmed", "type the tag again as `confirm`: deleting a model cannot be undone")
        named = {variant.tag for group in host.variants.values() for variant in group}
        if tag in named:
            return _error(
                409, "model_required",
                f"{tag} is a build this pool's catalog names for {host_id}; take it out of the model set first",
            )
        status_code, answer = await agents.delete_model(host.agent_endpoint, tag, transport=supervisor._agent_transport)
        if status_code == 200:
            supervisor.events.record(
                "agent_model_deleted", f"host {host_id}: {tag} deleted from disk on the operator's instruction", host_id=host_id,
            )
            return JSONResponse({"host_id": host_id, "deleted": tag})
        return JSONResponse(status_code=status_code, content=answer)

    @app.post("/pool/hosts/{host_id}/{action}")
    async def host_action(host_id: str, action: str) -> JSONResponse:
        fleet = supervisor.fleet
        if fleet is None or host_id not in fleet.hosts:
            return _error(404, "unknown_host", f"no rented host {host_id!r}")
        host = fleet.hosts[host_id]
        if action == "park":
            await fleet.park(host, "operator asked")
        elif action in ("release", "drain"):
            await fleet.destroy(host, "operator asked")
        else:
            return _error(400, "unknown_action", f"{action!r} is not park, drain or release")
        return JSONResponse({"host_id": host_id, "action": action})

    @app.post("/pool/down")
    async def down() -> JSONResponse:
        """The panic button: destroy everything rented, now, verified."""
        fleet = supervisor.fleet
        if fleet is None:
            for lease in supervisor.leases.open_leases():
                supervisor.leases.close(lease.lease_id, "gpm down --all")
            return JSONResponse({"released": []})
        return JSONResponse({"released": await fleet.down_all()})

    return app
