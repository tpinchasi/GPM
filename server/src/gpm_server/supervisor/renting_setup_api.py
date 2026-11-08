"""Setting up renting, once a provider account is in (S14, D172).

A provider account is added on the Providers screen and is only an account: where machines could
be rented, and the credential. Renting is set up here, on its own: the disk each machine is
rented with and the most one may cost an hour (D108), the engine rented machines run, and which
of the pool's models they are rented for — offering only models the engine has a build of. Until
both ceilings are named the pool rents nothing: no lease opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..configplan import CannotEdit, set_values

if TYPE_CHECKING:
    from .service import Supervisor

#: The engines a rented machine starts from an image the pool knows (D147); whisper.cpp is a
#: machine owner's engine, never a rented one (D148).
RENTED_ENGINES = ("vllm", "ollama")


async def _body(request: Request) -> Optional[dict[str, Any]]:
    try:
        body = await request.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def setup_values(body: dict) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The ceilings and the engine as asked: numbers in range, an engine rented machines run."""
    try:
        disk, hourly = float(body.get("min_disk_gb")), float(body.get("max_all_in_hourly"))
    except (TypeError, ValueError):
        return None, "say the disk each machine is rented with (GB) and the most one may cost an hour, all-in ($)"
    if not 1 <= disk <= 100_000 or not 0 < hourly <= 1000:
        return None, "the disk is between 1 and 100000 GB, and the hourly ceiling between $0 and $1000"
    engine = str(body.get("engine") or "")
    if engine not in RENTED_ENGINES:
        return None, "rented machines run vLLM or Ollama"
    models = body.get("models")
    if models is not None and (not isinstance(models, list) or not all(isinstance(m, str) and m for m in models)):
        return None, "models is a list of the pool's model names"
    return {"min_disk_gb": disk, "max_all_in_hourly": hourly, "engine": engine, "models": models}, None


def register(
    app: FastAPI,
    supervisor: "Supervisor",
    *,
    error: Callable[..., JSONResponse],
    validate_plan_apply: Callable[[str, str, dict], JSONResponse],
    plan_of: Callable[[str], list[Any]],
) -> None:

    def view() -> dict[str, Any]:
        config = supervisor.config
        rented = config.rented
        if rented is None:
            return {"account": False, "set_up": False, "missing": [], "restart_needed": False}
        policy = rented.policy_in_force
        engines = {}
        for engine in RENTED_ENGINES:
            can, cannot = config.models_rentable_with(engine)
            engines[engine] = {"can": can, "cannot": cannot}
        return {
            "account": True,
            "set_up": rented.set_up,
            "missing": rented.setup_missing(),
            # The renting machinery is built when the pool starts (S13).
            "restart_needed": supervisor.fleet is None,
            "current": {
                "min_disk_gb": policy.min_disk_gb or None,
                "max_all_in_hourly": policy.max_all_in_hourly,
                "engine": rented.engine,
                "models": rented.models,
            },
            # With `all`, every rented machine holds the whole set; with profiles, they decide.
            "models_per_host": config.pool.models_per_host,
            "profiles": bool(rented.rent_profiles),
            "engines": engines,
        }

    @app.get("/pool/renting")
    async def renting() -> JSONResponse:
        return JSONResponse(view())

    @app.post("/pool/renting/setup")
    async def set_up(request: Request) -> JSONResponse:
        if supervisor.store is None:
            return error(400, "no_config_file", "this pool was not started from a file")
        rented = supervisor.config.rented
        if rented is None:
            return error(409, "no_account", "add a provider account first, on the Providers screen")
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        values, problem = setup_values(body)
        if problem is not None:
            return error(400, "bad_setup", problem)
        config = supervisor.config
        can, cannot = config.models_rentable_with(values["engine"])
        models = values["models"]
        write: dict[str, Any] = {"engine": values["engine"]}
        if config.pool.models_per_host == "declared" and not rented.rent_profiles:
            if not models:
                return error(400, "bad_setup", "choose at least one model rented machines are for")
            unknown = [m for m in models if m not in config.pool.model_set]
            if unknown:
                return error(400, "bad_setup", f"not in the pool's model set: {unknown}")
            without = [m for m in models if m not in can]
            if without:
                return error(400, "bad_setup", "; ".join(f"{m}: {cannot.get(m, 'cannot be rented')}" for m in without))
            write["models"] = list(dict.fromkeys(models))
        elif models:
            return error(400, "bad_setup", "this pool says what rented machines hold elsewhere — every machine holds "
                         "the whole set, or its model profiles decide — so no models are chosen here")
        text, version = supervisor.store.read()
        try:
            text = set_values(text, ("rented",), write)
            ceilings = {"min_disk_gb": values["min_disk_gb"], "max_all_in_hourly": values["max_all_in_hourly"]}
            try:
                text = set_values(text, ("rented", "offer_policy"), ceilings)
            except CannotEdit:
                # No offer_policy yet: an account added before renting was set up (D172).
                text = set_values(text, ("rented",), {"offer_policy": ceilings})
        except CannotEdit as exc:
            return error(409, "cannot_edit", str(exc))
        if body.get("plan_only"):
            errors = supervisor.store.validate(text)
            if errors:
                return error(400, "invalid_config", "; ".join(errors))
            return JSONResponse({"changes": [c.as_dict() for c in plan_of(text)]})
        async with supervisor.pass_lock:
            response = validate_plan_apply(text, version, body)
        if response.status_code == 200:
            supervisor.events.record("renting_set_up", "renting set up from the console", numbers={
                "min_disk_gb": values["min_disk_gb"], "max_all_in_hourly": values["max_all_in_hourly"],
                "engine": values["engine"], "models": write.get("models")})
        return response
