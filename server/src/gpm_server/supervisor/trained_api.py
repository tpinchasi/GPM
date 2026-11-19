"""Add model on the console (S15, D178–D182): read a trained model's file, check its declarations
against it, and add it to the pool in one validated, planned write.

- `POST /pool/models/inspect` — an ONNX file's inputs and outputs, read from an upload or a hub file
  without loading it (`onnxfile`), for the form to fill in;
- `POST /pool/models/test` — the build as the configuration reads it, and, for an ONNX build, where
  its declarations and its file disagree; spends nothing and runs nothing. The machine's own check —
  one made-up input run as the build loads — comes before the host serves it;
- `POST /pool/models/trained` — the catalog entry, the model set and the hosts that hold it.
"""

from __future__ import annotations

import re
import tempfile
from typing import Any, Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import hubbuilds, onnxfile
from ..config import CONTAINER_ENGINE, ONNX_ENGINE, TRAINED_ENGINES, Variant
from ..configplan import CannotEdit, set_in_list_item, set_values
from ..uploads import digest_of

_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_HUB_FILE = re.compile(r"^(?!.*\.\.)([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)/([A-Za-z0-9][A-Za-z0-9._-]*\.onnx)$")


def register(app: FastAPI, supervisor: Any, *, error: Callable[..., JSONResponse],
             validate_plan_apply: Callable[..., JSONResponse]) -> None:

    def hub_token() -> str | None:
        from ..credentials import CredentialStore

        store = CredentialStore(supervisor.config.credentials_dir() / "hub")
        for role in ("write", "read"):
            try:
                found = store.get(role)
            except Exception:  # noqa: BLE001 — a store that cannot be read is no token
                found = None
            if found is not None and found.bound_to({"hub": hubbuilds.hub_url()}):
                return found.value
        return None

    async def read_file(source: dict[str, Any]) -> dict[str, Any]:
        """An upload's or a hub file's inputs and outputs, with its size."""
        if isinstance(source.get("upload"), str):
            digest = digest_of(source["upload"]) or source["upload"]
            path = supervisor.uploads().path_of(digest)
            if path is None:
                raise LookupError(f"upload/{digest} is not in the pool's store")
            with open(path, "rb") as handle:
                return {**onnxfile.read(handle), "size_bytes": path.stat().st_size}
        found = _HUB_FILE.match(str(source.get("hub") or ""))
        if found is None:
            raise ValueError("send `upload` (its sha256) or `hub` (owner/repo/file.onnx)")
        repo, file = found.groups()
        limit = supervisor.config.models.max_upload_mb * 2**20
        token = hub_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with tempfile.TemporaryFile() as handle:
            async with httpx.AsyncClient(base_url=hubbuilds.hub_url(), transport=supervisor._hub_transport,
                                         follow_redirects=True, timeout=httpx.Timeout(30.0, read=600.0)) as hub:
                async with hub.stream("GET", f"/{repo}/resolve/main/{file}", headers=headers) as response:
                    if response.status_code != 200:
                        raise LookupError(f"the hub answered {response.status_code} for {repo}/{file}")
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"the file is larger than models.max_upload_mb ({limit // 2**20} MB)")
                        handle.write(chunk)
            return {**onnxfile.read(handle), "size_bytes": size}

    @app.post("/pool/models/inspect")
    async def inspect(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            found = await read_file(body if isinstance(body, dict) else {})
        except LookupError as exc:
            return error(404, "not_found", str(exc))
        except onnxfile.NotOnnx as exc:
            return error(400, "not_onnx", str(exc))
        except (ValueError, httpx.HTTPError) as exc:
            return error(400, "bad_request", str(exc))
        return JSONResponse(found)

    def build_of(body: Any) -> Variant:
        if not isinstance(body, dict) or not isinstance(body.get("build"), dict):
            raise ValueError("send `build`: the catalog build, with its engine (onnx or container)")
        build = Variant.model_validate(body["build"])
        if build.engine not in TRAINED_ENGINES:
            raise ValueError("a trained model's build is `engine: onnx` or `engine: container`")
        return build

    @app.post("/pool/models/test")
    async def test(request: Request) -> JSONResponse:
        """The build read as the configuration reads it, and its declarations held to its file."""
        body = await request.json()
        try:
            build = build_of(body)
        except ValueError as exc:
            return error(400, "bad_build", str(exc))
        problems: list[str] = []
        found: dict[str, Any] = {}
        if build.engine == CONTAINER_ENGINE and not supervisor.config.containers.enabled:
            problems.append("containers are off: turn them on before adding a container build (D180)")
        if build.engine == ONNX_ENGINE:
            source = {"upload": build.tag} if digest_of(build.tag) else {"hub": build.tag}
            try:
                found = await read_file(source)
            except (LookupError, ValueError, onnxfile.NotOnnx, httpx.HTTPError) as exc:
                problems.append(f"the file could not be read: {exc}")
            else:
                problems += onnxfile.problems(found, build.spec.model_dump(by_alias=True) if build.spec else {})
        return JSONResponse({"ok": not problems, "problems": problems, **found})

    @app.post("/pool/models/trained")
    async def add(request: Request) -> JSONResponse:
        """One write: the model's catalog entry, the model set, and the hosts that hold it."""
        if supervisor.store is None:
            return error(400, "no_config_file", "this pool was not started from a file")
        body = await request.json()
        name = str((body or {}).get("name") or "").strip()
        if not _MODEL_NAME.match(name):
            return error(400, "bad_model", f"not a model name: {name!r}")
        try:
            build = build_of(body)
        except ValueError as exc:
            return error(400, "bad_build", str(exc))
        config = supervisor.config
        if name in config.catalog or name in config.pool.model_set:
            return error(409, "model_exists", f"{name!r} is already in the pool; change it on the Models screen")
        hosts = body.get("hosts") or []
        known = {h.id: h for h in config.hosts}
        for host_id in hosts:
            host = known.get(host_id)
            if host is None:
                return error(404, "unknown_host", f"no configured host {host_id!r}")
            if config.engine_of(host) != build.engine:
                return error(400, "wrong_engine", f"host {host_id!r} runs {config.engine_of(host)}, not {build.engine}")
        written = build.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        written["tag"], written["engine"] = build.tag, build.engine
        text, version = supervisor.store.read()
        try:
            try:
                text = set_values(text, ("catalog",), {name: {"variants": [written]}})
            except CannotEdit:
                text = set_values(text, (), {"catalog": {name: {"variants": [written]}}})
            text = set_values(text, ("pool",), {"model_set": [*config.pool.model_set, name]})
            for host_id in hosts:
                text = set_in_list_item(text, "hosts", "id", host_id, {"models": [*(known[host_id].models or []), name]})
        except CannotEdit as exc:
            return error(409, "cannot_edit", str(exc))
        return validate_plan_apply(text, version, body)
