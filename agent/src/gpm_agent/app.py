"""The agent's HTTP surface: a closed list of verbs, every one behind the agent key.

docs/spec/host-agent.md §4, §7. Six verbs: report facts, hold this set of model tags, delete
this model tag, restart the engine, postpone the dead-man timer, and set the engine's version.
None takes a command, a path or a URL from the caller, and none ever will: the restart runs the
owner's command or the agent's own server, its settings are bounded integers, and a version is a
version — the agent's own table says where it comes from (D151).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import PROTOCOL_VERSION, __version__, installs
from . import platform as host_platform
from .engine_control import EngineControl
from .engines import engine_facts
from .facts import Probes, disk, gather
from .models import ModelWork, Refused
from .ollama_server import OllamaServer
from .settings import FOREIGN_PREFIXES, INSTALLABLE_ENGINES, Settings

log = logging.getLogger("gpm_agent")

#: How often the agent looks at a server it runs itself, to start it again if it has stopped —
#: between the pool's passes, which do the same, so a machine the pool is not asking still serves.
KEEP_EVERY_S = 30.0


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)


def default_installer(settings: Settings, engine_client: httpx.AsyncClient) -> Optional[installs.Installer]:
    """The installer for this machine's engine, where its owner allows installs (D150) — else None."""
    if not settings.install_engines or settings.engine not in INSTALLABLE_ENGINES:
        return None
    source = installs.source_for(settings.engine, installs.this_machine())
    if source is None:
        return None
    engines = Path(settings.engines_path).expanduser() if settings.engines_path else host_platform.engines_dir()
    engine_dir = engines / settings.engine
    server = OllamaServer(
        engine_dir, engine_url=settings.engine_url, models_path=settings.models_path, client=engine_client,
    )
    return installs.Installer(source, engine_dir, server)


def create_app(
    settings: Settings,
    *,
    probes: Optional[Probes] = None,
    engine_transport: Optional[httpx.AsyncBaseTransport] = None,
    state_path: Optional[Path] = None,
    installer: Optional[installs.Installer] = None,
) -> Starlette:
    """The agent. `installer` replaces the one its settings would make — for a test, which points
    it at a fake repository and fake programs."""
    probes = probes or Probes()
    engine = engine_facts(settings.engine, settings)

    # Made here rather than at start-up so the app answers the same however it is served.
    engine_client = httpx.AsyncClient(base_url=settings.engine_url, timeout=10, transport=engine_transport)
    if engine is not None and installer is None:
        installer = default_installer(settings, engine_client)

    work = (
        ModelWork(
            engine, engine_client,
            free_disk_bytes=lambda: disk(probes, settings.models_path)["free_bytes"],
            min_free_disk_gb=settings.min_free_disk_gb,
            allow_delete=settings.allow_delete,
            state_path=state_path,
            before_pass=installer.ensure_running if installer is not None else None,
        )
        if engine is not None else None
    )

    control = EngineControl(settings, engine, engine_client, installer) if engine is not None else None

    async def keep_running() -> None:
        """Start the engine this agent runs when the agent starts, and again whenever it stops."""
        while True:
            try:
                await installer.ensure_running()
            except Exception:  # noqa: BLE001 - a bad look must never end the looking
                log.exception("starting the engine again failed")
            await asyncio.sleep(KEEP_EVERY_S)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        keeper = asyncio.create_task(keep_running(), name="gpm-agent:keep") if installer is not None else None
        try:
            yield
        finally:
            if keeper is not None:
                keeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keeper
            if installer is not None:
                # The server is left running: it outlives the agent, which knows it again on return.
                await installer.aclose()
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
        described = (
            await engine.describe(engine_client)
            if engine is not None
            else {"name": settings.engine, "answers": False, "detail": "this agent does not know that engine"}
        )
        if engine is not None and settings.engine in INSTALLABLE_ENGINES:
            # Where the engine's versions stand (§2.3) — read from memory, so facts stay cheap.
            described["install"] = installer.state() if installer is not None else installs.not_allowed(settings.engine)
        if installer is not None:
            # Something answering on the engine's port that this agent did not start: reported,
            # and left alone (D150).
            described["foreign"] = bool(described.get("answers")) and not installer.server.ours()
        applied = control.applied() if control is not None else None
        body: dict[str, Any] = {
            "agent": {"version": __version__, "protocol": PROTOCOL_VERSION},
            **gather(probes, settings.models_path),
            "engine": described,
            # What the machine's owner allows, so the console can say so before anyone asks.
            "allows": {
                "delete": settings.allow_delete,
                "restart": bool(control and control.can_restart),
                "engine_settings": bool(control and control.can_apply_settings),
            },
            # What the engine was last started with, as written here — None if never.
            "engine_environment": applied,
            "engine_settings": engine.settings_from_environment(applied) if engine else None,
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

    async def engine_version(request: Request) -> JSONResponse:
        """Work toward an engine version, and switch to it when asked (D151). Answers at once."""
        refused = refusal(request)
        if refused is not None:
            return refused
        if engine is None:
            return _error(501, "unknown_engine", f"this agent does not know the engine {settings.engine!r}")
        if installer is None:
            return _error(
                403, "installs_not_allowed",
                "this machine's owner has not set install_engines: true in the agent's settings "
                "(`gpm-agent init --install-engines`); the pool cannot change that",
            )
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        try:
            version, switch = installs.validated(body)
            return JSONResponse(await installer.want(version, switch))
        except Refused as no:
            return _error(no.status, no.error, no.detail)

    async def hold_models(request: Request) -> JSONResponse:
        return await _models(request, "apply")

    async def delete_model(request: Request) -> JSONResponse:
        return await _models(request, "delete")

    async def heartbeat(request: Request) -> JSONResponse:
        """Postpone the dead-man timer, on a host the pool created (D63).

        No body, and nothing to get wrong: it touches one file. A machine nobody rented has no
        timer, so there is nothing here to beat and the verb says so.
        """
        refused = refusal(request)
        if refused is not None:
            return refused
        if not settings.heartbeat_file:
            return _error(409, "no_timer", "this machine carries no dead-man timer")
        path = Path(settings.heartbeat_file).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        except OSError as exc:
            return _error(500, "heartbeat_failed", f"the timer's file could not be touched: {exc}")
        return JSONResponse({"beat": True, "at": time.time()})

    base = f"/agent/v{PROTOCOL_VERSION}"
    app = Starlette(
        routes=[
            Route(f"{base}/facts", facts, methods=["GET"]),
            # The tag travels in the body, never the path: tags contain ':' and '/'.
            Route(f"{base}/models", hold_models, methods=["PUT"]),
            Route(f"{base}/models", delete_model, methods=["DELETE"]),
            # Restart the engine with the owner's command; optionally after writing settings.
            Route(f"{base}/engine", engine_restart, methods=["POST"]),
            # Postpones a shutdown the pool could equally cause by going silent (D63).
            Route(f"{base}/heartbeat", heartbeat, methods=["POST"]),
            # A version from the agent's own table, never a URL or a path (D151).
            Route(f"{base}/engine/version", engine_version, methods=["PUT"]),
        ],
        lifespan=lifespan,
    )
    app.state.work = work
    app.state.installer = installer
    return app
