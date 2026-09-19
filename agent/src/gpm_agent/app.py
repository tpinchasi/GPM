"""The agent's HTTP surface: a closed list of verbs, every one behind the agent key.

docs/spec/host-agent.md §4. Four verbs: report facts, hold this set of model tags, delete this
model tag, restart the engine. None takes a command, a path or a URL from the caller, and none
ever will: the restart runs the owner's command, and its settings are bounded integers.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import PROTOCOL_VERSION, __version__
from .engine_control import EngineControl, read_applied
from .engines import engine_facts
from .facts import Probes, disk, gather
from .models import ModelWork, Refused
from .settings import FOREIGN_PREFIXES, Settings


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)


def create_app(
    settings: Settings,
    *,
    probes: Optional[Probes] = None,
    engine_transport: Optional[httpx.AsyncBaseTransport] = None,
    state_path: Optional[Path] = None,
) -> Starlette:
    probes = probes or Probes()
    engine = engine_facts(settings.engine)

    # Made here rather than at start-up so the app answers the same however it is served.
    engine_client = httpx.AsyncClient(base_url=settings.engine_url, timeout=10, transport=engine_transport)

    work = (
        ModelWork(
            engine, engine_client,
            free_disk_bytes=lambda: disk(probes, settings.models_path)["free_bytes"],
            min_free_disk_gb=settings.min_free_disk_gb,
            allow_delete=settings.allow_delete,
            state_path=state_path,
        )
        if engine is not None else None
    )

    control = EngineControl(settings, engine, engine_client) if engine is not None else None

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if work is not None:
                await work.aclose()
            await engine_client.aclose()

    def refusal(request: Request) -> Optional[JSONResponse]:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            return _error(401, "unauthorized", "the agent requires its agent key")
        foreign = FOREIGN_PREFIXES.get(token.split("_", 1)[0])
        if foreign and not settings.accepts(token):
            # Said plainly: a pool key pasted here is a mistake worth naming, and neither
            # of the pool's keys may ever drive a host.
            return _error(403, "wrong_key_role", f"that is {foreign}; the agent accepts only its own agent key")
        if not settings.accepts(token):
            return _error(401, "unauthorized", "unknown agent key")
        return None

    async def facts(request: Request) -> JSONResponse:
        refused = refusal(request)
        if refused is not None:
            return refused
        body: dict[str, Any] = {
            "agent": {"version": __version__, "protocol": PROTOCOL_VERSION},
            **gather(probes, settings.models_path),
            "engine": (
                await engine.describe(engine_client)
                if engine is not None
                else {"name": settings.engine, "answers": False, "detail": "this agent does not know that engine"}
            ),
            # What the machine's owner allows, so the console can say so before anyone asks.
            "allows": {
                "delete": settings.allow_delete,
                "restart": bool(control and control.can_restart),
                "engine_settings": bool(control and control.can_apply_settings),
            },
            # What the engine was last started with, as written here — None if never.
            "engine_environment": read_applied(settings.engine_env_file),
            "engine_settings": engine.settings_from_environment(read_applied(settings.engine_env_file)) if engine else None,
            "min_free_disk_bytes": int(settings.min_free_disk_gb * 1e9),
        }
        return JSONResponse(body)

    async def _models(request: Request, act: str) -> JSONResponse:
        refused = refusal(request)
        if refused is not None:
            return refused
        if work is None:
            return _error(501, "unknown_engine", f"this agent does not know the engine {settings.engine!r}")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        try:
            return JSONResponse(await (work.apply(body) if act == "apply" else work.delete(body)))
        except Refused as no:
            return _error(no.status, no.error, no.detail)

    async def engine_restart(request: Request) -> JSONResponse:
        refused = refusal(request)
        if refused is not None:
            return refused
        if control is None:
            return _error(501, "unknown_engine", f"this agent does not know the engine {settings.engine!r}")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        try:
            return JSONResponse(await control.act(body))
        except Refused as no:
            return _error(no.status, no.error, no.detail)

    async def hold_models(request: Request) -> JSONResponse:
        return await _models(request, "apply")

    async def delete_model(request: Request) -> JSONResponse:
        return await _models(request, "delete")

    base = f"/agent/v{PROTOCOL_VERSION}"
    app = Starlette(
        routes=[
            Route(f"{base}/facts", facts, methods=["GET"]),
            # The tag travels in the body, never the path: tags contain ':' and '/'.
            Route(f"{base}/models", hold_models, methods=["PUT"]),
            Route(f"{base}/models", delete_model, methods=["DELETE"]),
            # Restart the engine with the owner's command; optionally after writing settings.
            Route(f"{base}/engine", engine_restart, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.work = work
    return app
