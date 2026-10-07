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
import re
import socket
import time
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Optional, Sequence
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..catalog import variants_for_host
from ..config import ConfigError, PoolConfig, Variant
from ..configplan import (
    CannotEdit,
    ConfigStore,
    MachineNow,
    RentedNow,
    StaleVersion,
    plan_changes,
    remove_list_item,
    set_in_list_item,
    set_values,
)
from ..contract import CONTRACT_VERSION
from ..directory import HUB, DirectoryStore, build_sizes_gb, read_directory
from ..engines import EngineNotFound, available_engines, get_engine
from ..hostcheck import test_connection
from ..hubbuilds import HubUnavailable, valid_search
from ..keys import verify
from ..ledger import LeaseRefused
from ..sizing import needs_for
from . import agents
from .service import Supervisor

log = logging.getLogger("gpm.control")

#: A lease's price ceiling was a bid ceiling; it is all-in now, and named for it (D108). Refused
#: by name, so a caller still sending the old field hears why rather than having it ignored.
_BID_CEILING_GONE = (
    "bid_ceiling is gone (D108): a lease tightens the pool's all-in maximum per host-hour with "
    "max_all_in_hourly"
)



def _priced(offer) -> str:
    """How a host or offer is paid for, in the console's words: bid, spot or on demand (D132)."""
    return ("bid" if offer.bidding else "spot") if offer.interruptible else "on_demand"

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


def _held_on_disk(agent_models: Optional[dict[str, Any]]) -> frozenset[str]:
    """The tags a host's agent reports as on its disk — the fact the engine's own list cannot
    give for an engine that is launched with its models."""
    if not agent_models:
        return frozenset()
    return frozenset(
        entry["tag"] for entry in agent_models.get("models") or []
        if isinstance(entry, dict) and entry.get("on_disk") and isinstance(entry.get("tag"), str)
    )


def _silent_engine(why: str, rented: Any, *, loads_by_restart: bool) -> dict[str, Any]:
    """An engine that did not answer, said truthfully (D105).

    For an engine that is started once its weights have landed, silence while the agent fetches
    is the plan, not a fault — and "not answering: ReadError" beside a download bar was read as
    one. It is expected silence only once the agent is there: before that the machine may
    still be booting, and the old words are the right ones.
    """
    if loads_by_restart and rented is not None and getattr(rented, "agent", None) is not None:
        if getattr(rented, "restart_asked_at", None) is None:
            return {
                "answers": False, "expected": True,
                "detail": "not started yet — this engine is started once every model it was bought for is on disk",
            }
        return {
            "answers": False, "expected": True,
            "detail": f"starting on the downloaded models, asked {time.time() - rented.restart_asked_at:.0f}s ago; not answering yet",
        }
    return {"answers": False, "detail": why}


def _model_states(
    required: set[str], *, progress: dict[str, Any], agent_models: Optional[dict[str, Any]],
    resident: frozenset[str], available: frozenset[str], loads_by_restart: bool, restart_asked: bool,
) -> list[dict[str, Any]]:
    """One state per model the host must hold, from every source the pool has (D105).

    In order of certainty: the engine serving it; the agent reporting it failed; the agent
    reporting it loaded; a download in progress (the pool's own, or the agent's); on disk by
    either account; nothing. Each carries the words for that state, so the console does not
    have to invent them.
    """
    by_tag: dict[str, dict[str, Any]] = {}
    for entry in (agent_models or {}).get("models") or []:
        if isinstance(entry, dict) and isinstance(entry.get("tag"), str):
            by_tag[entry["tag"]] = entry
    states: list[dict[str, Any]] = []
    for tag in sorted(required):
        agent = by_tag.get(tag, {})
        pulled = progress.get(tag) or {}
        pulling = agent.get("pulling") or {}
        done = pulled.get("completed") if pulled else pulling.get("completed_bytes")
        total = pulled.get("total") if pulled else pulling.get("total_bytes")
        downloading = bool(pulling) or (bool(pulled) and total and (done or 0) < total)
        state: dict[str, Any] = {"tag": tag}
        if tag in resident or agent.get("loaded"):
            state.update(state="loaded", detail="in memory, serving")
        elif agent.get("error"):
            state.update(state="failed", detail=str(agent["error"]))
        elif downloading:
            words = f"{(done or 0) / 1e9:.1f} of {total / 1e9:.1f} GB" if total else "download starting"
            if pulled.get("mbps") is not None:
                words += f" at {pulled['mbps']:.0f} Mbps"
            state.update(state="downloading", detail=words, completed=done or 0, total=total)
        elif agent.get("on_disk") or tag in available:
            if loads_by_restart:
                state.update(
                    state="loading" if restart_asked else "on disk",
                    detail=("downloaded; the engine is starting on it" if restart_asked
                            else "downloaded; the engine starts once every model has landed"),
                )
            else:
                state.update(state="loading", detail="downloaded; loading into memory")
        else:
            state.update(state="not here yet", detail="not on this machine yet")
        states.append(state)
    return states


def _stage_of(detail: dict[str, Any]) -> str:
    """Where preparing this host has got to, in words, from what is known about it.

    Derived rather than stored, so it cannot drift from the facts it describes.
    """
    if detail.get("state") == "ready":
        return "ready — serving requests"
    if detail.get("state") == "parked":
        return "parked — stopped, disk kept, billing storage only"
    provider = (detail.get("provider") or {}).get("detail") or ""
    engine = detail.get("engine") or {}
    stage = detail.get("stage") or ""
    models = detail.get("models") or []

    failed = [m for m in models if m.get("state") == "failed"]
    if failed:
        return f"{failed[0]['tag']} failed to start: {failed[0].get('detail', '')}"
    if not engine.get("answers"):
        if engine.get("expected"):
            # The engine is silent on purpose: say what the machine is doing instead.
            landed = sum(1 for m in models if m.get("state") in ("on disk", "loading", "loaded"))
            downloading = [m for m in models if m.get("state") == "downloading"]
            if downloading:
                return (f"fetching models: {landed} of {len(models)} on disk; "
                        f"downloading {downloading[0]['tag']} — {downloading[0].get('detail', '')}")
            if models and landed == len(models):
                return "every model is on disk; the engine is starting on them"
            return f"fetching models: {landed} of {len(models)} on disk"
        if "pulling" in provider.lower():
            return f"the provider is still starting the machine: {provider.strip()}"
        return "waiting for the engine to answer — the machine is starting, or the tunnel is not up yet"
    if stage:
        progress = (detail.get("progress") or {}).get(stage.split()[-1] if stage.startswith("downloading") else "")
        if progress and progress.get("total"):
            done, total = progress["completed"] / 1e9, progress["total"] / 1e9
            return f"{stage} — {done:.1f} of {total:.1f} GB"
        return stage
    if engine.get("missing_from_disk"):
        return f"models still to download: {', '.join(engine['missing_from_disk'])}"
    if engine.get("not_loaded"):
        return f"downloaded; loading into memory: {', '.join(engine['not_loaded'])}"
    return "the model set is loaded; waiting for the next probe to mark it ready"


def engine_offers() -> dict[str, Any]:
    """What each installed engine's own start can switch on, and whether its builds can be
    looked up on a model hub (D100) — the editor's checkboxes, from the engines themselves."""
    offers: dict[str, Any] = {}
    for name in sorted(available_engines()):
        try:
            engine = get_engine(name)
        except EngineNotFound:
            continue
        offers[name] = {
            "builds_on_hub": bool(getattr(engine, "builds_on_hub", False)),
            # Whether a profile may split its models across a group of cards (D114).
            "splits_across_cards": bool(getattr(engine, "splits_across_cards", False)),
            "options": {
                key: {"label": option.label, "families": list(option.families)}
                for key, option in (getattr(engine, "options", None) or {}).items()
            },
        }
    return offers


#: A name the pool may call a model by, as a request or the catalog names it.
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _with_builds(text: str, config: PoolConfig, name: str, builds: dict[str, str], workloads_only: bool = False) -> str:
    """`text` with the catalog entry for `name` holding `builds` — one per engine — and every
    build it already had for other engines (D98, D101).

    Two things are done because getting them wrong is silent: a build written before the pool
    ran two engines names none, so it is marked as the pool's own engine's — otherwise a vLLM
    host could be handed an Ollama tag to fetch from a model hub; and a model with no entry keeps
    being served under its own name by the engine it was written for.
    """
    entry = config.catalog.get(name)
    before = [v.model_dump(exclude_defaults=True) for v in entry.variants] if entry else []
    if entry is not None:
        variants = [v | {"engine": v.get("engine") or config.engine} for v in before]
    elif workloads_only:
        variants = []  # no shared host serves it, so no build of the pool's own engine is implied
    else:
        variants = [{"tag": name, "engine": config.engine}]
    for engine, tag in builds.items():
        variants = [v for v in variants if v.get("engine") != engine]
        variants.append({"tag": tag, "engine": engine})
    written = {"variants": variants, **({"workloads_only": True} if workloads_only else {})}
    if entry is not None and variants == before and entry.workloads_only == workloads_only:
        return text  # nothing about this model changes
    if entry is None and not workloads_only and variants == [{"tag": name, "engine": config.engine}]:
        return text  # served under its own name by the pool's engine: no entry needed
    if entry is not None:
        return set_values(text, ("catalog", name), written)
    try:
        return set_values(text, ("catalog",), {name: written})
    except CannotEdit:
        return set_values(text, (), {"catalog": {name: written}})


#: A model profile's name, as an operator would write it.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}$")


def _with_variants(text: str, config: PoolConfig, name: str, variants: list[dict[str, Any]]) -> str:
    """`text` with each of `variants` in the catalog entry for `name`, **after** those already
    there (D111): a profile names its build outright, and adding one must not move what any
    other host resolves the model to. A build already listed is kept as it is."""
    entry = config.catalog.get(name)
    before = [v.model_dump(exclude_defaults=True) for v in entry.variants] if entry else []
    if entry is not None:
        current = [v | {"engine": v.get("engine") or config.engine} for v in before]
    elif any(v["tag"] == name for v in variants):
        current = []
    else:
        # Served under its own name by the pool's engine until now: that stays its first build.
        current = [{"tag": name, "engine": config.engine}] if name in config.pool.model_set else []
    tags = {v["tag"] for v in current}
    for variant in variants:
        if variant["tag"] not in tags:
            current.append(variant)
            tags.add(variant["tag"])
    if entry is not None and current == before:
        return text
    if entry is not None:
        return set_values(text, ("catalog", name), {"variants": current})
    try:
        return set_values(text, ("catalog",), {name: {"variants": current}})
    except CannotEdit:
        return set_values(text, (), {"catalog": {name: {"variants": current}}})


def _host_can_serve(config: PoolConfig, host: Any, name: str, added: list[dict[str, Any]]) -> bool:
    """Could this configured host serve `name`, with the builds being added for it?"""
    engine = config.engine_of(host)
    caps = set(host.capabilities)
    entry = config.catalog.get(name)
    listed = [v.model_dump() for v in entry.variants] if entry else []
    return any(
        set(v.get("requires") or []) <= caps and (not v.get("engine") or v.get("engine") == engine)
        for v in listed + added
    )


class _KeyHashes:
    """The app and admin key hashes, read again when the configuration or a key file changes —
    so `gpm key revoke` takes effect on the next request, not the next restart. A file that
    cannot be read keeps the last good set rather than admitting nothing or everything."""

    def __init__(self, auth: Any):
        self._auth = auth
        self._stamp: Any = None
        self._hashes: tuple[set[str], set[str]] = (set(), set())

    def _mtimes(self, auth: Any) -> tuple:
        stamps = []
        for path in auth.key_files():
            try:
                stamps.append(path.stat().st_mtime_ns)
            except OSError:
                stamps.append(None)
        return (id(auth), tuple(stamps))

    def current(self) -> tuple[set[str], set[str]]:
        auth = self._auth()
        stamp = self._mtimes(auth)
        if stamp != self._stamp:
            try:
                self._hashes = (auth.app_hashes(), auth.admin_hashes())
            except Exception:
                if self._stamp is None:
                    raise
                log.exception("the key files could not be read again; keeping the keys read before")
                return self._hashes
            self._stamp = stamp
        return self._hashes


def _server_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("gpm-server")
    except PackageNotFoundError:
        return "unknown"


def create_control_app(supervisor: Supervisor, config: PoolConfig) -> FastAPI:
    # The configuration given here until the supervisor applies a newer one from its file.
    started_with = supervisor.config
    keys = _KeyHashes(lambda: config.auth if supervisor.config is started_with else supervisor.config.auth)
    keys.current()  # a key in both roles, or a missing app key, stops the start
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
        app_hashes, admin_hashes = keys.current()
        if verify(token, app_hashes):
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
        if token.startswith("gpmp_"):
            # A program's provisioning key (D117): it asks the router for workloads, and reaches
            # nothing here.
            return _error(403, "provisioning_key_refused",
                          "that is a provisioning key: it asks the pool's listener for workloads. Use the admin key here.")
        if token.startswith("gpmw_"):
            # A workload's key reaches its workload's hosts and nothing else (D115).
            return _error(
                403, "workload_key_refused",
                "that is a workload key: it reaches its workload's inference only. Use the admin key here.",
            )
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
            # The page may keep the admin key in the browser (D137), so nothing but its own script
            # ever runs on it: no inline script, no other origin's, no framing. Images may come
            # from a provider's own site (its logo, D134) — an image runs nothing.
            response.headers["content-security-policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'; form-action 'self'"
            )
            response.headers["referrer-policy"] = "no-referrer"
            return response
        refusal = authorised(request)
        if refusal is not None:
            return refusal
        return await call_next(request)

    static_dir = Path(__file__).resolve().parent.parent / "console" / "static"
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="console")

    # --- reading ---

    def _pinned(variants: Mapping[str, Sequence[Any]], builds: Mapping[str, str]) -> Mapping[str, Sequence[Any]]:
        """Only the builds a profile named, on a host bought as one (D111)."""
        if not builds:
            return variants
        return {
            name: [v for v in group if v.tag == builds[name]] for name, group in variants.items() if name in builds
        }

    def _profiles_view() -> list[dict[str, Any]]:
        """Every model profile, in the file's order: what it holds, each build's size where it
        is known, and the least card and disk a machine holding it needs (D111)."""
        rented = supervisor.config.rented
        fleet = supervisor.fleet
        if rented is None:
            return []
        view = []
        for name, models in rented.model_profiles.items():
            sizes = fleet.build_sizes(list(models), models) if fleet is not None else {}
            cards = rented.cards_per_copy(name)
            needs = needs_for(sizes, cards) if fleet is not None else None
            view.append({
                "name": name,
                "rented": name in rented.rent_profiles,
                "models": [{"model": model, "build": build, "size_gb": sizes.get(model)} for model, build in models.items()],
                "needs": needs.as_dict() if needs else None,
                "cards_per_copy": cards,
                "proxy": fleet.proxy_for(list(models)) if fleet is not None else None,
            })
        return view

    def _pool_variants() -> list[dict[str, Any]]:
        """Every variant the pool's catalog lists, for every engine, with its size and precision
        where anything has measured them — what a profile picks from, a model and its variant
        in one row (D111). A model with no catalog entry is served under its own name by the
        pool's engine, and is listed that way."""
        config = supervisor.config
        store = DirectoryStore(supervisor.db)
        found = []
        for model in config.pool.model_set:
            entry = config.catalog.get(model)
            variants = entry.variants if entry else [Variant(tag=model, engine=config.engine)]
            looked_up = store.get(HUB, model)
            hub = {b.get("repo"): b for b in (looked_up[0].get("builds", []) if looked_up else [])}
            for variant in variants:
                size = variant.size_gb or build_sizes_gb(supervisor.db, {model: variant.tag}).get(model)
                found.append({
                    "model": model,
                    "tag": variant.tag,
                    "engine": variant.engine or config.engine,
                    "requires": list(variant.requires),
                    "size_gb": size,
                    "precision": (hub.get(variant.tag) or {}).get("precision"),
                })
        return found

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
                # What this machine runs, and which of the pool's models it holds (D89, D93).
                "engine": supervisor.config.engine_of(host.config),
                "holds": supervisor.config.models_held_by(host.config),
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
                fleet.rented_models,
                supervisor.config.catalog,
                frozenset(fleet.rented.capabilities),
                supervisor.config.rented_engine(),
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
                            # A host bought for its own models (a workload's group, a profile)
                            # serves those, whether or not the shared set names them.
                            _pinned(
                                variants_for_host(list(host.builds), supervisor.config.catalog,
                                                  frozenset(fleet.rented.capabilities), supervisor.config.rented_engine())
                                if host.builds else rented_variants,
                                host.builds),
                            getattr(host, "resident", frozenset()),
                            getattr(host, "resident", frozenset()),
                        ),
                        # The model profile it was bought as, and the builds that named (D111).
                        "profile": host.profile,
                        "builds": dict(host.builds),
                        # How many cards each copy of its models spans (D114).
                        "cards_per_copy": host.cards_per_copy,
                        # What this machine runs, and what it was bought to serve (D93, D94).
                        # Neither was visible anywhere, so a pool buying the wrong thing looked
                        # exactly like one buying the right thing.
                        "engine": supervisor.config.rented_engine(),
                        "bought_for": list(host.models),
                        "image": host.instance.spec.image if getattr(host, "instance", None) and getattr(host.instance, "spec", None) else None,
                        "machine": host.offer.machine_id,
                        "hardware": host.offer.hardware,
                        "bid_hourly": host.bid_hourly,
                        # A pool may hold both at once (D55): which can be outbid, which cannot.
                        "interruptible": host.interruptible,
                        # Which provider account, and how it is paid for: bid, spot or on demand (D129, D132).
                        "connection": host.connection_name,
                        "priced": _priced(host.offer),
                        "storage_hourly": round(host.offer.storage_hourly, 5),
                        "estimated_spend": round(host.estimate(), 4),
                        "reported_spend": round(host.reported_spend, 4),
                        "lease_id": host.lease_id,
                        "parked_at": host.parked_at,
                        "hours_held": round(host.hours_held, 3),
                        "workers": host.workers,
                        "busy": counter.busy if counter else 0,
                    }
                )

        leases = []
        for lease in supervisor.leases.open_leases():
            entry = {
                "lease_id": lease.lease_id,
                "workload": lease.workload,
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
                # The release running: a console page loaded from an older one reloads (D122).
                "server_version": _server_version(),
                "contract_version": CONTRACT_VERSION,
                "as_of": time.time(),
                # The Models screen draws these, and they change when configuration is
                # reloaded — so they are read from the supervisor, not from the config the
                # app was built with.
                "model_set": list(supervisor.config.pool.model_set),
                # Models only workloads serve (D115): offered for a workload, never shared.
                "workload_models": sorted(
                    name for name, entry in supervisor.config.catalog.items() if entry.workloads_only
                ),
                # How many models one workload may serve (D118): the console's form stops there.
                "workload_max_models": supervisor.config.workloads.max_models,
                "catalog": {
                    name: [
                        {
                            "tag": variant.tag,
                            "requires": list(variant.requires),
                            "runtime_class": variant.runtime_class,
                            "enforces_schema": variant.enforces_schema,
                            # Which engine this build is for, or null for any (D93) — what the
                            # engine editor shows and fills in.
                            "engine": variant.engine,
                            # What it downloads, where the file says — a profile's needs (D111).
                            "size_gb": variant.size_gb,
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
                    # What the configured limits actually allow, whichever of them binds (D46).
                    "worst_case_hourly": supervisor.fleet.worst_case_hourly() if supervisor.fleet else 0.0,
                    "per_host_ceiling": supervisor.config.rented.max_all_in_hourly if supervisor.config.rented else None,
                },
                "provider": (
                    {
                        "name": fleet.provider.name,
                        # The connection it is configured as (D129).
                        "connection": fleet.connection,
                        "capabilities": vars(fleet.provider.capabilities),
                        "cap_safety_margin": fleet.margin(),
                        # Today's use of its daily search quota, where it has one (D121).
                        "search_quota": fleet.search_quota(),
                    }
                    if fleet is not None
                    else None
                ),
                # Every connection (D129): the console's Providers screen reads these. `provider`
                # above is the first enabled one's, for what reads one.
                "providers": (
                    [
                        {
                            "connection": name,
                            "type": p.name,
                            "enabled": name in fleet.rented.enabled_connections,
                            "capabilities": vars(p.capabilities),
                            "search_quota": fleet.search_quota(name),
                            "search_error": fleet.searching[name].error if name in fleet.searching else None,
                            "rented": sum(1 for h in fleet.hosts.values() if not h.released and h.connection_name == name),
                            "hourly": round(sum(h.bid_hourly for h in fleet.hosts.values()
                                                if not h.released and h.connection_name == name), 4),
                        }
                        for name, p in list(fleet.providers.items())
                    ]
                    if fleet is not None
                    else None
                ),
                # What would actually be started on a machine this pool rents (D92). Together
                # these decide whether a rental can work at all, and until now neither was
                # visible anywhere: an engine and an image for a different engine looked
                # exactly like a correct pool until a host was bought and never answered.
                "engine": {
                    "name": supervisor.config.engine,
                    # A pool may run more than one (D93); the rented machines are commonly the
                    # ones that differ, and that is where the money goes.
                    "rented": supervisor.config.rented_engine(),
                    "in_use": supervisor.config.engines_in_use(),
                    "proxy": bool(supervisor.config.rented and supervisor.config.rented.engine_proxy),
                    "models_per_host": supervisor.config.pool.models_per_host,
                    # What the engine editor starts from (D98).
                    "rented_models": (
                        list(supervisor.config.rented.models)
                        if supervisor.config.rented and supervisor.config.rented.models is not None
                        else None
                    ),
                    # The model profiles, each with what a machine holding it needs (D111).
                    "profiles": _profiles_view(),
                    # Every variant a profile can pick from, a model and its variant together.
                    "pool_variants": _pool_variants(),
                    "rented_capabilities": list(supervisor.config.rented.capabilities) if supervisor.config.rented else [],
                    "rent_profiles": list(supervisor.config.rented.rent_profiles) if supervisor.config.rented else [],
                    "engine_start": supervisor.config.rented.engine_start if supervisor.config.rented else None,
                    "image": supervisor.config.rented.image if supervisor.config.rented else None,
                    "available": sorted(available_engines()),
                    # What each engine's own start can switch on, and whether its builds can be
                    # looked up on a model hub — what the editor offers as checkboxes (D100).
                    "offers": engine_offers(),
                    "engine_options": (
                        list(supervisor.config.rented.engine_options) if supervisor.config.rented else []
                    ),
                    "port": supervisor.config.engine_port(),
                    "images": (
                        [
                            {"image": i.image, "min_driver": i.min_driver, "note": i.note}
                            for i in supervisor.config.rented.images
                        ]
                        or [{"image": supervisor.config.rented.image, "min_driver": None, "note": None}]
                    )
                    if supervisor.config.rented is not None
                    else [],
                },
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
    async def events(limit: int = 100, kind: Optional[str] = None, host_id: Optional[str] = None) -> JSONResponse:
        return JSONResponse({"events": supervisor.events.recent(limit=limit, kind=kind, host_id=host_id)})

    @app.get("/pool/leases")
    async def leases() -> JSONResponse:
        fleet = supervisor.fleet
        out = []
        for lease in supervisor.leases.all():
            entry = {
                "lease_id": lease.lease_id,
                "workload": lease.workload,
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

    def _named(connections: Optional[str]) -> Optional[list[str]]:
        """`connections=a,b`: search only these providers (providers.md §7.2)."""
        return [n for n in connections.split(",") if n] if connections is not None else None

    @app.get("/pool/market/preview")
    async def market_preview(hours: float = 4.0, kinds: Optional[str] = None, search: bool = False,
                             connections: Optional[str] = None) -> JSONResponse:
        """Read-only: the live market through the pool's own filters. Spends nothing. The market
        is asked only with `search=true`; without it, the settings only (D120, D122) — a page or
        script that asks by habit never spends the provider's daily search quota."""
        if supervisor.fleet is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity configured")
        try:
            return JSONResponse(await supervisor.fleet.market_preview(
                hours=hours, kinds=kinds, search=search, connections=_named(connections)))
        except ValueError as exc:
            return _error(400, "bad_kinds", str(exc))

    @app.post("/pool/market/preview")
    async def market_preview_unsaved(request: Request, hours: float = 4.0, kinds: Optional[str] = None,
                                     search: bool = False, connections: Optional[str] = None) -> JSONResponse:
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
                    kinds=kinds,
                    search=search,
                    connections=_named(connections),
                )
            )
        except (TypeError, ValueError) as exc:
            return _error(400, "bad_policy", str(exc))

    # --- configuration: never spends money ---

    def store() -> Optional[ConfigStore]:
        return supervisor.store

    def known_connections() -> dict[str, str]:
        """Every name a connection has had, with its provider (D133)."""
        return dict(supervisor.fleet.connection_types) if supervisor.fleet is not None else {}

    def rented_now() -> list[RentedNow]:
        if supervisor.fleet is None:
            return []
        return [
            RentedNow(host_id=h.host_id, bid_hourly=h.bid_hourly, storage_hourly=h.offer.storage_hourly,
                      provider=supervisor.fleet.connection_types.get(h.connection_name, ""))
            for h in supervisor.fleet.hosts.values()
            if not h.released
        ] + [
            # A host kept back at a restart is as much the pool's as one it watches (D129).
            RentedNow(host_id=row.host_id, bid_hourly=0.0, provider=supervisor.fleet.connection_types.get(name, ""))
            for name, rows in supervisor.fleet.pending_adoption.items() for row in rows
        ] + [
            RentedNow(host_id=f"volume:{v.volume_id}", bid_hourly=v.hourly,
                      provider=supervisor.fleet.connection_types.get(v.connection or supervisor.fleet.legacy_connection, ""))
            for v in supervisor.fleet.workload_store.volumes()
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
        changes = plan_changes(supervisor.config, candidate, rented_now(), machines_now(), known_connections())
        return JSONResponse({"errors": [], "changes": [change.as_dict() for change in changes]})

    @app.put("/pool/config")
    async def put_config(request: Request) -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        body = await request.json()
        refused = _moves_holding(body.get("text", ""))
        if refused is not None:
            return _error(409, "change_refused", refused)
        try:
            version = store().apply(body.get("text", ""), body.get("version"))
        except StaleVersion as exc:
            return _error(409, "stale_version", str(exc))
        except ConfigError as exc:
            return _error(400, "invalid_config", str(exc))
        async with supervisor.pass_lock:  # a provider change never lands mid-pass (D135)
            supervisor.reload_config()
        return JSONResponse({"version": version})

    @app.patch("/pool/config/rented")
    async def set_rented_search(request: Request) -> JSONResponse:
        """Change what the Rented capacity screen edits: the offer search, the bidding, and how
        capacity is allocated — D51, extended to allocation by D74.

        The file stays the source of truth and the rules are the file's: the change is written
        into it in place — comments, ordering and flow style untouched — then validated,
        planned, and applied only if nothing that loosens a limit is unconfirmed.
        """
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        body = await request.json()
        text, version = store().read()
        try:
            # `allocation` sits directly under `rented`; the rest are its own blocks, and one a
            # pool never wrote is added whole rather than guessed at line by line.
            # `mode` decides which listings are even searched, so a pool left on the default
            # never sees an on-demand offer at all — live, an operator watching a fixed-price
            # host sit there could not say why it was never rented, and the only way to change
            # it was the raw configuration file (D80).
            straight = {key: body[key] for key in ("allocation", "mode", "search_profile") if body.get(key) is not None}
            # Save the values on the screen under a name, and switch to it (D87).
            save_as = body.get("save_profile_as")
            if save_as:
                wanted = {k: v for k, v in (body.get("offer_policy") or {}).items() if v is not None}
                base = supervisor.config.rented.policy_in_force.model_dump()
                profile = {k: v for k, v in {**base, **wanted}.items() if v is not None}
                if _has_section(text, "search_profiles"):
                    text = set_values(text, ("rented", "search_profiles"), {str(save_as): profile})
                else:
                    # A block this pool never wrote is added whole, once — the same rule the
                    # allocation blocks follow, rather than guessing at it line by line.
                    straight["search_profiles"] = {str(save_as): profile}
                straight["search_profile"] = str(save_as)
            for section in ("dynamic", "workers_auto", "teardown"):
                wanted = body.get(section) or {}
                if wanted and not _has_section(text, section):
                    straight[section] = wanted
            if straight:
                text = set_values(text, ("rented",), straight)
            # The screen shows the search in force, so its edits go to the search in force (D108):
            # with a profile named, into that profile. Written into `offer_policy` instead, they
            # sat in the file unused while the screen went on showing the profile's values —
            # found live, as a max all-in that "kept showing 0.85".
            rented_config = supervisor.config.rented
            in_force = body.get("search_profile", rented_config.search_profile if rented_config else None)
            edits = {k: v for k, v in (body.get("offer_policy") or {}).items() if v is not None}
            if edits and in_force and not save_as:
                profile = rented_config.search_profiles.get(in_force) if rented_config else None
                if profile is None:
                    return _error(400, "bad_request", f"there is no search profile named {in_force!r}")
                merged = {k: v for k, v in {**profile.model_dump(), **edits}.items() if v is not None}
                text = set_values(text, ("rented", "search_profiles"), {str(in_force): merged})
            for section in ("offer_policy", "bidding", "dynamic", "workers_auto", "teardown"):
                # Saving under a name puts those filter values in the *profile*; writing them
                # to the live policy as well would edit the very thing being saved away from.
                # With a profile in force, the edits went to it above.
                if section == "offer_policy" and (save_as or in_force):
                    continue
                wanted = body.get(section) or {}
                if wanted and section not in straight:
                    text = set_values(text, ("rented", section), wanted)
        except CannotEdit as exc:
            return _error(409, "cannot_edit", str(exc))
        return _validate_plan_apply(text, version, body)

    @app.patch("/pool/config/engine")
    async def set_engine(request: Request) -> JSONResponse:
        """Change which engine rented hosts run and how models are placed, in one write (D98).

        These cannot be changed one at a time: an engine switched without its builds, images or
        start command is a configuration that rents machines which can never serve, and the
        load-time checks refuse every half-way state. So the whole of it arrives together, is
        written into the file in place — comments and layout kept (D51) — then validated,
        planned and confirmed exactly as every other change is.

        Two things are done for the operator because getting them wrong is silent:

        - a build written before the pool ran two engines names none, so any engine may be
          offered it; it is marked as the pool's own engine's, so a vLLM host is never handed an
          Ollama tag to fetch from a model hub;
        - a configured host that held the whole set keeps it when the set is spread across hosts
          — rather than silently dropping to the first model it can serve.
        """
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        body = await request.json()
        config = supervisor.config
        rented_engine = body.get("rented_engine")
        placement = body.get("placement")
        if rented_engine not in available_engines():
            return _error(400, "bad_engine", f"rented_engine must be one of {sorted(available_engines())}")
        # Left out, the placement is left as it is — model profiles say what hosts hold (D111).
        if placement is not None and placement not in ("all", "all_proxy", "declared"):
            return _error(400, "bad_placement", "placement must be 'all', 'all_proxy' or 'declared'")
        if config.rented is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity to configure")

        text, version = store().read()
        try:
            per_host = None
            rented: dict[str, Any] = {"engine": rented_engine}
            if placement is not None:
                per_host = "declared" if placement == "declared" else "all"
                text = set_values(text, ("pool",), {"models_per_host": per_host})
                rented["engine_proxy"] = placement == "all_proxy"
                rented["models"] = list(body.get("rented_models") or []) if per_host == "declared" else None
            if "engine_start" in body:
                # Blank means "the engine's own start" (D97), which the file says as null.
                rented["engine_start"] = (body["engine_start"] or "").strip() or None
            if body.get("image"):
                rented["image"] = body["image"]
            if "images" in body:
                rented["images"] = [
                    {k: v for k, v in item.items() if k in ("image", "min_driver", "note") and v not in (None, "")}
                    for item in body.get("images") or []
                ]
            # Named options of the engine's own start (D100). Unsent, the ones the new engine
            # also offers are kept, so switching engine never leaves a name it would refuse.
            offered = get_engine(rented_engine).options
            options = body.get("engine_options")
            if options is None:
                options = [o for o in config.rented.engine_options if o in offered]
            if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
                return _error(400, "bad_options", "engine_options is a list of option names")
            if list(options) != list(config.rented.engine_options):
                rented["engine_options"] = list(dict.fromkeys(options))
            text = set_values(text, ("rented",), rented)

            builds = body.get("builds") or {}
            for name in config.pool.model_set:
                build = (builds.get(name) or "").strip()
                text = _with_builds(text, config, name, {rented_engine: build} if build else {})

            for host in config.hosts if per_host is not None else ():
                if per_host == "declared" and host.models is None:
                    held = config.models_held_by(host)
                    text = set_in_list_item(text, "hosts", "id", host.id, {"models": held})
                elif per_host == "all" and host.models is not None:
                    text = set_in_list_item(text, "hosts", "id", host.id, {"models": None})
            # The model profiles, with the engine their builds are for (D111): switched apart,
            # each half would be refused until the other was saved.
            if "profiles" in body:
                if placement is not None:
                    return _error(400, "bad_request", "send `placement` or `profiles`, not both: profiles say what hosts hold")
                text, refused = _write_profiles(text, body)
                if refused is not None:
                    return refused
        except CannotEdit as exc:
            return _error(409, "cannot_edit", str(exc))
        return _validate_plan_apply(text, version, body)

    def _write_profiles(text: str, body: dict) -> tuple[str, Optional[JSONResponse]]:
        """`text` with the model profiles in `body` written in, or why they cannot be (D111).

        `profiles` is the whole set, {name: {model: build}}; `rent` the names rented as, in the
        order the pool prefers them; `add` models and builds new to the pool, each
        {name, builds: [{tag, engine, size_gb}]}, found by the operator's search.

        Two things are done for the operator because getting them wrong is silent: a build is
        added to the catalog *after* the ones already there, so what every other host resolves
        to does not move; and where a configured host held the whole set and cannot serve a
        model added here, the set is spread across hosts (`declared`) with each such host
        keeping exactly what it held.
        """
        config = supervisor.config
        profiles = body.get("profiles")
        rent = body.get("rent")
        if not isinstance(profiles, dict) or not isinstance(rent, list):
            return text, _error(400, "bad_request", "send `profiles`: {name: {model: build}}, and `rent`: a list of their names")
        if not rent:
            return text, _error(400, "bad_request", "tick at least one profile for the pool to rent hosts as")
        for name, models in profiles.items():
            if not _PROFILE_NAME.match(str(name)):
                return text, _error(400, "bad_profile", f"not a profile name: {name!r} — letters, digits, spaces, '.', '_' and '-'")
            if not isinstance(models, dict) or not models:
                return text, _error(400, "bad_profile", f"profile {name!r} holds no model")
            for model, build in models.items():
                if not _MODEL_NAME.match(str(model)) or not _MODEL_NAME.match(str(build or "")):
                    return text, _error(400, "bad_profile", f"profile {name!r}: not a model and build: {model!r} → {build!r}")
        unknown = [name for name in rent if name not in profiles]
        if unknown:
            return text, _error(400, "bad_profile", f"rent names profiles that are not sent: {unknown}")
        # How many cards each profile's copies span (D114). Not sent, it is kept for the profiles
        # still here, so a profile removed takes its split with it rather than leave the file
        # naming a profile that no longer exists.
        split_sent = body.get("split")
        if split_sent is None:
            split = {name: cards for name, cards in config.rented.split_across_cards.items() if name in profiles}
        elif not isinstance(split_sent, dict):
            return text, _error(400, "bad_request", "send `split`: {profile name: cards per copy}")
        else:
            split = {}
            for name, cards in split_sent.items():
                if name not in profiles:
                    return text, _error(400, "bad_profile", f"split names a profile that is not sent: {name!r}")
                if isinstance(cards, bool) or cards not in (1, 2, 4, 8):
                    return text, _error(400, "bad_profile", f"profile {name!r}: a model can be split across 1, 2, 4 or 8 cards")
                if cards > 1:
                    split[str(name)] = int(cards)

        installed = set(available_engines())
        model_set = list(config.pool.model_set)
        added: dict[str, list[dict[str, Any]]] = {}
        for item in body.get("add") or []:
            name = str((item or {}).get("name") or "").strip()
            if not _MODEL_NAME.match(name):
                return text, _error(400, "bad_model", f"not a model name: {name!r}")
            for build in (item or {}).get("builds") or []:
                tag, engine = str(build.get("tag") or "").strip(), str(build.get("engine") or "")
                if not _MODEL_NAME.match(tag):
                    return text, _error(400, "bad_build", f"not a build name: {tag!r}")
                if engine not in installed:
                    return text, _error(400, "bad_engine", f"no engine {engine!r}; installed: {sorted(installed)}")
                size = build.get("size_gb")
                variant: dict[str, Any] = {"tag": tag, "engine": engine}
                if isinstance(size, (int, float)) and size > 0:
                    variant["size_gb"] = round(float(size), 3)
                added.setdefault(name, []).append(variant)
            if name not in model_set:
                model_set.append(name)
        # Every model a profile holds is in the pool's set: a profile is how it gets served.
        for models in profiles.values():
            for model in models:
                if model not in model_set:
                    model_set.append(model)

        for name, variants in added.items():
            text = _with_variants(text, config, name, variants)
        new_models = [name for name in model_set if name not in config.pool.model_set]
        if new_models:
            text = set_values(text, ("pool",), {"model_set": model_set})
        # A configured host told to hold the whole set cannot hold a model built only for the
        # rented engine; spread the set, and let it keep exactly what it held.
        if config.pool.models_per_host == "all" and new_models:
            cannot = [
                host for host in config.hosts
                if not host.disabled and any(
                    not _host_can_serve(config, host, name, added.get(name, [])) for name in new_models
                )
            ]
            if cannot:
                text = set_values(text, ("pool",), {"models_per_host": "declared"})
                for host in config.hosts:
                    if host.models is None:
                        text = set_in_list_item(text, "hosts", "id", host.id, {"models": list(config.pool.model_set)})
        rented: dict[str, Any] = {
            "model_profiles": {str(k): {str(m): str(b) for m, b in v.items()} for k, v in profiles.items()},
            "rent_profiles": [str(name) for name in rent],
        }
        if split or config.rented.split_across_cards:
            rented["split_across_cards"] = split
        # One way to say what rented hosts hold (D111); the router in front of several engine
        # processes follows each profile's size, not a switch for the whole pool.
        if config.rented.models is not None:
            rented["models"] = None
        if config.rented.engine_proxy:
            rented["engine_proxy"] = False
        return set_values(text, ("rented",), rented), None

    @app.put("/pool/config/profiles")
    async def set_profiles(request: Request) -> JSONResponse:
        """Write the model profiles and which of them the pool rents as, in one change (D111) —
        written into the file in place, then validated, planned and confirmed like every other
        change. The file's own rules decide what is allowed: a build the catalog does not list
        for its model, or one rented hosts cannot run, is refused in words. The console saves
        these with the engine, through `PATCH /pool/config/engine`, since a profile's builds
        are one engine's."""
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        body = await request.json()
        if supervisor.config.rented is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity to configure")
        text, version = store().read()
        try:
            text, refused = _write_profiles(text, body)
        except CannotEdit as exc:
            return _error(409, "cannot_edit", str(exc))
        if refused is not None:
            return refused
        return _validate_plan_apply(text, version, body)

    @app.get("/pool/models/search")
    async def search_models(q: str) -> JSONResponse:
        """Find a model to put in a profile, by any part of its name (D111): the model hub,
        asked live, and Ollama's library from the directory's cache. Read-only; the hub is asked
        with no credential, at the directory's pace. The library is read only when the operator
        asks (`POST /pool/directory/refresh`); this says whether it ever has been."""
        term = (q or "").strip()
        if not valid_search(term):
            return _error(400, "bad_search", "search for a model's name: letters, digits, '.', '_', '-' and '/'")
        library = await asyncio.to_thread(
            read_directory, supervisor.db, term, tuple(supervisor.config.pool.model_set)
        )
        hub: list[dict[str, Any]] = []
        hub_problem = None
        try:
            hub = await supervisor.directory.search_hub(term)
        except HubUnavailable as exc:
            hub_problem = str(exc)
        return JSONResponse({
            "query": term,
            "hub": hub,
            "hub_problem": hub_problem,
            "library": library.get("models", []),
            "library_read": bool(library.get("refreshed", {}).get("ollama")),
            "library_refreshing": supervisor.directory.running,
            "in_pool": list(supervisor.config.pool.model_set),
        })

    @app.get("/pool/models/size")
    async def model_size(repo: str) -> JSONResponse:
        """One hub repository's weights, exactly, from its file listing (D111) — a search only
        has the hub's estimate, and a profile's minimums are worked out from this."""
        if not valid_search(repo):
            return _error(400, "bad_repo", "name a repository: letters, digits, '.', '_', '-' and '/'")
        try:
            size = await supervisor.directory.exact_size(repo)
        except HubUnavailable as exc:
            return _error(502, "hub_unavailable", str(exc))
        return JSONResponse({"repo": repo, "size_gb": size})

    @app.get("/pool/builds")
    async def builds_on_hub(model: str, engine: Optional[str] = None, search: Optional[str] = None,
                            fresh: bool = False) -> JSONResponse:
        """An engine's builds of one of the pool's models, found on the model hub and sorted
        (D100): the original and its quantisations, each with its precision, size, the cards it
        runs on and whether a rented host can fetch it; files the engine cannot load and
        different models under similar names left out, and counted.

        Read-only, and asked of the hub with no credential. Served from the model directory's
        cache (D101) unless it is stale or `fresh` is asked; a lookup is cached for everyone. The
        operator's choice is saved through the engine editor like any other build.
        """
        engine = engine or supervisor.config.rented_engine()
        if not _MODEL_NAME.match(model or ""):
            return _error(400, "bad_model", "name a model: letters, digits, '.', '_', ':', '-' and '/'")
        try:
            adapter = get_engine(engine)
        except EngineNotFound:
            return _error(400, "bad_engine", f"no engine {engine!r}; installed: {sorted(available_engines())}")
        if not getattr(adapter, "builds_on_hub", False):
            return _error(
                400, "no_hub",
                f"{engine!r} builds are named in its own library, not found on a model hub",
            )
        if search is not None and not valid_search(search):
            return _error(400, "bad_search", "search for a model's name: letters, digits, '.', '_', '-' and '/'")
        try:
            found = await supervisor.directory.lookup(model, search or None, fresh=fresh)
        except HubUnavailable as exc:
            return _error(502, "hub_unavailable", str(exc))
        return JSONResponse(found | {"engine": engine})

    @app.get("/pool/directory")
    async def directory(q: Optional[str] = None) -> JSONResponse:
        """The model directory (D101): Ollama's library and the hub builds looked up so far,
        from the cache — nothing here asks either of them anything."""
        # Asked before reading, not after: a refresh that ends between the two would otherwise be
        # reported finished beside what it had half written (found by CI).
        refreshing = supervisor.directory.running
        found = await asyncio.to_thread(
            read_directory, supervisor.db, q, tuple(supervisor.config.pool.model_set)
        )
        settings = supervisor.config.directory
        return JSONResponse(found | {
            "refreshing": refreshing,
            "settings": settings.model_dump(),
            "offers": engine_offers(),
            "model_set": list(supervisor.config.pool.model_set),
            "placement": supervisor.config.pool.models_per_host,
            "rented_models": (
                list(supervisor.config.rented.models)
                if supervisor.config.rented and supervisor.config.rented.models is not None else None
            ),
            "engine_options": list(supervisor.config.rented.engine_options) if supervisor.config.rented else [],
            "rent_profiles": list(supervisor.config.rented.rent_profiles) if supervisor.config.rented else [],
        })

    @app.post("/pool/directory/refresh")
    async def refresh_directory() -> JSONResponse:
        """Refresh the directory now, in the background. It reads Ollama's library and looks up
        builds on the hub at the stated pace; the status says how far it has got."""
        started = supervisor.directory.start()
        return JSONResponse(status_code=202, content={
            "started": started,
            "detail": "refreshing" if started else "a refresh is already running",
        })

    @app.post("/pool/config/models")
    async def add_models(request: Request) -> JSONResponse:
        """Add models to the pool from the directory, with a build for each engine (D101).

        One write, like every other editor: the model set, each model's builds, the hosts it is
        rented for and the engine's options, then validated, planned and confirmed. A model no
        host can hold is refused by the file's own rules, in words, before anything is written.
        """
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        body = await request.json()
        config = supervisor.config
        wanted = body.get("add")
        if not isinstance(wanted, list) or not wanted:
            return _error(400, "bad_request", "send `add`: a list of models, each with its `name` and `builds`")
        installed = set(available_engines())
        model_set = list(config.pool.model_set)
        rent_for = list(config.rented.models or []) if config.rented and config.rented.models is not None else None
        text, version = store().read()
        try:
            for item in wanted:
                name = str((item or {}).get("name") or "").strip()
                if not _MODEL_NAME.match(name):
                    return _error(400, "bad_model", f"not a model name: {name!r}")
                builds = {str(k): str(v).strip() for k, v in ((item or {}).get("builds") or {}).items() if str(v).strip()}
                unknown = set(builds) - installed
                if unknown:
                    return _error(400, "bad_engine", f"no engine {sorted(unknown)}; installed: {sorted(installed)}")
                for tag in builds.values():
                    if not _MODEL_NAME.match(tag):
                        return _error(400, "bad_build", f"not a build name: {tag!r}")
                only = bool(item.get("workloads_only"))
                if only and name in model_set:
                    return _error(400, "bad_model", f"{name!r} is already in the shared set; it cannot also be workloads-only")
                if not only and name not in model_set:
                    model_set.append(name)
                if not only and item.get("rent_for") and rent_for is not None and name not in rent_for:
                    rent_for.append(name)
                # Outside the shared set when workloads-only: the shared hosts never fetch it (D115).
                text = _with_builds(text, config, name, builds, workloads_only=only)
            if model_set != list(config.pool.model_set):
                text = set_values(text, ("pool",), {"model_set": model_set})
            rented: dict[str, Any] = {}
            if rent_for is not None and rent_for != list(config.rented.models or []):
                rented["models"] = rent_for
            if "engine_options" in body and config.rented is not None:
                options = body.get("engine_options")
                if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
                    return _error(400, "bad_options", "engine_options is a list of option names")
                if list(options) != list(config.rented.engine_options):
                    rented["engine_options"] = list(dict.fromkeys(options))
            if rented:
                text = set_values(text, ("rented",), rented)
        except CannotEdit as exc:
            return _error(409, "cannot_edit", str(exc))
        return _validate_plan_apply(text, version, body)

    def _configured_host(host_id: str) -> Optional[JSONResponse]:
        """None when `host_id` is a host this pool's file configures; otherwise why not."""
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        if any(host.id == host_id for host in supervisor.config.hosts):
            return None
        if supervisor.fleet is not None and host_id in supervisor.fleet.hosts:
            return _error(
                400, "rented_host",
                f"{host_id!r} is rented: it is released from the Rented capacity screen, not removed",
            )
        return _error(404, "unknown_host", f"no configured host {host_id!r}")

    @app.patch("/pool/config/hosts/{host_id}")
    async def set_host_service(host_id: str, request: Request) -> JSONResponse:
        """Take a configured host out of service, or return it (`disabled` in the file).

        The host stays in the file and comes back with one click; requests already on it finish.
        The file's own rules still hold — taking out the only host that serves a model is
        refused, with the reason, rather than leaving that model with nowhere to go.
        """
        refused = _configured_host(host_id)
        if refused is not None:
            return refused
        body = await request.json()
        if not isinstance(body.get("disabled"), bool):
            return _error(400, "bad_request", "send `disabled`: true to take the host out of service, false to return it")
        text, version = store().read()
        try:
            text = set_in_list_item(text, "hosts", "id", host_id, {"disabled": body["disabled"]})
        except CannotEdit as exc:
            return _error(409, "cannot_edit", str(exc))
        return _validate_plan_apply(text, version, body)

    @app.delete("/pool/config/hosts/{host_id}")
    async def remove_host(host_id: str, request: Request) -> JSONResponse:
        """Remove a configured host from the pool's file.

        Typed to confirm, like anything that cannot be undone with one click: the host's entry
        leaves the file. Requests already on it finish; its tunnels close. The file keeps its
        own history, so the previous version can be rolled back to.
        """
        refused = _configured_host(host_id)
        if refused is not None:
            return refused
        try:
            body = await request.json()
        except ValueError:
            body = {}
        text, version = store().read()
        try:
            text = remove_list_item(text, "hosts", "id", host_id)
        except CannotEdit as exc:
            return _error(409, "cannot_edit", str(exc))
        # What the file would say first: a removal the rules refuse is never offered to confirm.
        errors = store().validate(text)
        if errors:
            return _error(400, "invalid_config", "; ".join(errors))
        if str(body.get("confirm")) != host_id:
            changes = plan_changes(supervisor.config, store().parse(text), rented_now(), machines_now(), known_connections())
            listed = [c.as_dict() for c in changes]
            for change in listed:
                if change["kind"] == "host_removed":
                    change["requires_retype"], change["value"] = True, host_id
            return JSONResponse(status_code=400, content={
                "error": "not_confirmed",
                "detail": f"removing a host is confirmed by typing its id; send `confirm`: {host_id!r}",
                "changes": listed,
            })
        return _validate_plan_apply(text, version, body)

    def _validate_plan_apply(text: str, version: str, body: dict) -> JSONResponse:
        """The path every edit takes: validated, planned, and applied only if nothing that
        loosens a limit is unconfirmed."""
        errors = store().validate(text)
        if errors:
            return _error(400, "invalid_config", "; ".join(errors))
        candidate = store().parse(text)
        changes = plan_changes(supervisor.config, candidate, rented_now(), machines_now(), known_connections())
        refused = [c for c in changes if c.refused]
        if refused:
            return _error(409, "change_refused", "; ".join(c.refused for c in refused))
        loosening = [c for c in changes if c.requires_retype]
        if loosening and str(body.get("confirm")) not in {str(c.requires_retype) for c in loosening}:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "not_confirmed",
                    "detail": "this loosens a limit; send `confirm` with the new value",
                    "changes": [c.as_dict() for c in changes],
                },
            )
        try:
            new_version = store().apply(text, version)
        except (StaleVersion, ConfigError) as exc:
            return _error(409 if isinstance(exc, StaleVersion) else 400, "not_applied", str(exc))
        supervisor.reload_config()
        return JSONResponse({"version": new_version, "changes": [c.as_dict() for c in changes]})

    # --- provider accounts (D130, D134-D136) ---
    from .providers_api import register as register_providers

    register_providers(
        app, supervisor, error=_error, validate_plan_apply=_validate_plan_apply,
        plan_of=lambda text: plan_changes(supervisor.config, store().parse(text), rented_now(), machines_now(),
                                          known_connections()),
    )

    def _has_section(text: str, name: str) -> bool:
        """Is this block already in the file? A missing one is written whole, once."""
        import re as _re

        return bool(_re.search(rf"^\s+{_re.escape(name)}\s*:", text, _re.M))

    def _moves_holding(text: str) -> Optional[str]:
        """The one check a raw save or a rollback cannot skip: a holding connection pointed
        elsewhere (D136). Anything else the file gets wrong, its own load refuses."""
        if store().validate(text):
            return None
        return supervisor.accounts.refuse_moves(store().parse(text))

    @app.get("/pool/config/history")
    async def config_history() -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        return JSONResponse({"versions": store().history()})

    @app.post("/pool/config/rollback")
    async def rollback_config(request: Request) -> JSONResponse:
        if store() is None:
            return _error(400, "no_config_file", "this pool was not started from a file")
        wanted = (await request.json()).get("version", "")
        refused = _moves_holding(store().text_of(wanted) or "")
        if refused is not None:
            return _error(409, "change_refused", refused)
        try:
            version = store().rollback(wanted)
        except ConfigError as exc:
            return _error(400, "invalid_config", str(exc))
        async with supervisor.pass_lock:  # a provider change never lands mid-pass (D135)
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
                "connection": supervisor.fleet.connection,
                "credential_valid": status.credential_valid,
                "credit_remaining": status.credit_remaining,
            }
        )

    # --- spending ---

    @app.post("/pool/leases")
    async def open_lease(request: Request) -> JSONResponse:
        body = await request.json()
        fleet = supervisor.fleet
        if "bid_ceiling" in body:
            return _error(400, "bad_request", _BID_CEILING_GONE)
        try:
            if fleet is not None:
                lease = fleet.open_lease(
                    workers=int(body["workers"]),
                    max_hours=float(body.get("max_hours", 4)),
                    max_spend=body.get("max_spend"),
                    allow_rent=bool(body.get("allow_rent", False)),
                    max_all_in_hourly=body.get("max_all_in_hourly"),
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
                    "worst_case_hourly": supervisor.fleet.worst_case_hourly() if supervisor.fleet else 0.0,
                },
            },
            status_code=201,
        )

    # --- workloads (D115, docs/spec/workloads.md) ---

    def _workload_request(body: dict) -> Any:
        from .workloads import WorkloadRefused, WorkloadRequest, parse_targets

        for required in ("name", "hours"):
            if body.get(required) in (None, ""):
                raise KeyError(required)
        hours, spend = body.get("hours"), body.get("max_spend")
        for name, value in (("hours", hours), ("max_spend", spend)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise WorkloadRefused(f"{name} must be a number")
        targets, placement = parse_targets(body)
        return WorkloadRequest.of(
            str(body.get("name") or "").strip(), targets, float(hours),
            max_spend=float(spend) if spend is not None else None,
            profile=(str(body["profile"]).strip() or None) if body.get("profile") else None,
            kind=str(body.get("kind") or "roi"), placement=placement,
        )

    async def _workload_call(fn):
        from .workloads import WorkloadRefused

        try:
            return await fn()
        except WorkloadRefused as exc:
            return _error(400, "workload_refused", str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            return _error(400, "bad_request", f"missing or malformed field: {exc}")

    def _connection(name: str, key: Optional[str]) -> dict:
        """What the app owner puts in the chat client: one URL, one key (workloads.md §3)."""
        listen = supervisor.config.listen
        tls = bool(listen.tls_certfile and listen.tls_keyfile)
        scheme = "https" if tls else "http"
        host = listen.host if listen.host not in ("0.0.0.0", "::") else socket.gethostname()
        return {
            "base_url": f"{scheme}://{host}:{listen.port}/v1",
            "path_prefix_url": f"{scheme}://{host}:{listen.port}/w/{name}/v1",
            "api_key": key,
            "tls": tls,
            "shown_once": key is not None,
        }

    # --- provisioning keys (D117) ---

    @app.post("/pool/provisioners")
    async def create_provisioner(request: Request) -> JSONResponse:
        """A provisioning key and its grant; the key is in this answer **once**."""
        from ..provisioning_store import Grant
        from .workloads import WorkloadRefused

        body = await request.json()
        try:
            fields = {k: body[k] for k in (
                "max_open", "max_spend", "max_spend_per_day", "max_hours", "models", "kinds", "may_borrow",
                "idle_end_minutes", "max_idle_end_minutes", "certs") if k in body}
            grant = Grant.from_dict(fields)
            key = supervisor.provisioning.grant(str(body.get("name") or ""), grant, body.get("expires_hours"))
        except WorkloadRefused as exc:
            return _error(400, "provisioner_refused", str(exc))
        except (TypeError, ValueError) as exc:
            return _error(400, "bad_request", str(exc))
        view = supervisor.provisioning.store.get(body["name"]).view()
        return JSONResponse({"provisioner": view, "key": key, "shown_once": True}, status_code=201)

    @app.get("/pool/provisioners")
    async def list_provisioners() -> JSONResponse:
        provisioning = supervisor.provisioning
        return JSONResponse({
            "provisioners": [{**p.view(), **provisioning.usage(p.name)} for p in provisioning.store.all()],
            "max_spend_per_day": supervisor.config.provisioning.max_spend_per_day,
            "pool_url": _connection("-", None)["base_url"].removesuffix("/v1"),
        })

    @app.delete("/pool/provisioners/{name}")
    async def revoke_provisioner(name: str, end_workloads: bool = False) -> JSONResponse:
        from .workloads import WorkloadRefused

        try:
            ended = supervisor.provisioning.revoke(name, end_workloads=end_workloads)
        except WorkloadRefused as exc:
            return _error(404, "not_found", str(exc))
        return JSONResponse({"provisioner": name, "revoked": True, "ended": ended})

    @app.post("/pool/workloads/plan")
    async def workload_plan(request: Request) -> JSONResponse:
        """What creating this workload would do: its sizing, start, kind, budget and caps.
        Nothing is opened, minted or rented."""
        body = await request.json()

        async def run():
            return JSONResponse({"plan": await supervisor.workloads.plan(_workload_request(body))})

        return await _workload_call(run)

    @app.post("/pool/workloads")
    async def create_workload(request: Request) -> JSONResponse:
        """Create it: one lease bound to it, and its key, **shown once** in this answer. A budget
        that was not typed must be sent back as `confirm_max_spend` after reading the plan."""
        body = await request.json()

        async def run():
            confirm = body.get("confirm_max_spend")
            workload, key, plan, _ = await supervisor.workloads.create(
                _workload_request(body), confirm_max_spend=float(confirm) if confirm is not None else None
            )
            return JSONResponse(
                {"workload": supervisor.workloads.view(workload), "plan": plan,
                 "connection": _connection(workload.name, key)},
                status_code=201,
            )

        return await _workload_call(run)

    @app.get("/pool/workloads")
    async def list_workloads() -> JSONResponse:
        return JSONResponse({"workloads": [supervisor.workloads.view(w) for w in supervisor.workloads.store.all()]})

    @app.get("/pool/workloads/{name}")
    async def show_workload(name: str) -> JSONResponse:
        async def run():
            workload = supervisor.workloads.get(name)
            return JSONResponse({"workload": supervisor.workloads.view(workload),
                                 "connection": _connection(name, None)})

        return await _workload_call(run)

    @app.post("/pool/workloads/{name}/extend")
    async def extend_workload(name: str, request: Request) -> JSONResponse:
        """Hours added and/or a new dollar cap. Each raise carries its value typed again (D49):
        `confirm` for the dollar cap, `confirm_hours` for the hours added."""
        body = await request.json()

        async def run():
            hours = body.get("hours")
            spend = body.get("max_spend")
            confirm = body.get("confirm")
            confirm_hours = body.get("confirm_hours")
            workload = supervisor.workloads.extend(
                name, hours=float(hours) if hours is not None else None,
                max_spend=float(spend) if spend is not None else None,
                confirm=confirm is not None and spend is not None and float(confirm) == float(spend),
                confirm_hours=confirm_hours is not None and hours is not None and float(confirm_hours) == float(hours),
            )
            return JSONResponse({"workload": supervisor.workloads.view(workload)})

        return await _workload_call(run)

    @app.post("/pool/workloads/{name}/end")
    async def end_workload(name: str) -> JSONResponse:
        async def run():
            workload = supervisor.workloads.end(name)
            return JSONResponse({"workload": supervisor.workloads.view(workload)})

        return await _workload_call(run)

    @app.post("/pool/workloads/{name}/keys")
    async def rotate_workload_key(name: str) -> JSONResponse:
        """A new key, shown once; the old ones keep working for the rotation grace."""
        async def run():
            key_id, key = supervisor.workloads.rotate_key(name)
            return JSONResponse({"key_id": key_id, "connection": _connection(name, key),
                                 "old_keys_valid_minutes": supervisor.config.workloads.rotation_grace_minutes})

        return await _workload_call(run)

    def _workload_lease(lease_id: str) -> Optional[JSONResponse]:
        """A workload's lease is changed through its workload, which keeps the router's end time
        and the workload's state in step with it (D115)."""
        lease = supervisor.leases.get(lease_id)
        if lease is not None and lease.workload:
            return _error(
                409, "workload_lease",
                f"lease {lease_id} belongs to workload {lease.workload!r}; use /pool/workloads/{lease.workload}/extend or /end",
            )
        return None

    @app.delete("/pool/leases/{lease_id}")
    async def close_lease(lease_id: str) -> JSONResponse:
        refused = _workload_lease(lease_id)
        if refused is not None:
            return refused
        supervisor.leases.close(lease_id, "closed through the control API")
        return JSONResponse({"lease_id": lease_id, "state": "closed"})

    @app.patch("/pool/leases/{lease_id}")
    async def amend_lease(lease_id: str, request: Request) -> JSONResponse:
        """Tighten an open lease, or extend it (D49).

        Tightening needs nothing. Raising a limit — more dollars, more hours, more workers —
        needs `confirm` carrying the new value, the same rule the console applies to loosening
        anywhere else. Extending the hours also pushes out the hold on a host prepared under
        this lease, or the lease would outlive the host it was extended for.
        """
        refused = _workload_lease(lease_id)
        if refused is not None:
            return refused
        body = await request.json()
        before = supervisor.leases.get(lease_id)
        if before is None:
            return _error(404, "unknown_lease", f"no lease {lease_id!r}")

        wanted = {k: body.get(k) for k in ("max_spend", "max_hours", "workers")}
        raised = {
            k: v for k, v in wanted.items()
            if v is not None and v > getattr(before, k)
        }
        if raised and str(body.get("confirm")) not in {str(v) for v in raised.values()}:
            return _error(
                400, "not_confirmed",
                "raising " + ", ".join(sorted(raised)) + " must be confirmed: send `confirm` "
                "with the new value. Tightening needs no confirmation.",
            )
        try:
            lease = supervisor.leases.tighten(lease_id, **wanted, loosen=bool(raised))
        except LeaseRefused as exc:
            return _error(400, "lease_refused", str(exc))

        held = []
        if raised.get("max_hours") and supervisor.fleet is not None:
            # The host was held only as long as the lease was going to last.
            for host in supervisor.fleet.hosts.values():
                if host.lease_id == lease_id and host.prepared and not host.released:
                    host.hold_until = lease.opened_at + lease.max_hours * 3600
                    held.append(host.host_id)
        if raised:
            supervisor.events.record(
                "lease_extended",
                f"{lease_id} extended: "
                + ", ".join(f"{k} {getattr(before, k)} → {v}" for k, v in sorted(raised.items()))
                + (f"; {', '.join(held)} held for the longer lease" if held else ""),
                numbers={**raised, "was": {k: getattr(before, k) for k in raised}},
                lease_id=lease_id,
            )
        return JSONResponse({
            "lease_id": lease.lease_id, "max_spend": lease.max_spend,
            "max_hours": lease.max_hours, "workers": lease.workers,
            "hours_left": round(lease.hours_left(), 3), "hosts_held": held,
        })

    @app.post("/pool/hosts/prepare")
    async def prepare_host(request: Request) -> JSONResponse:
        body = await request.json()
        if supervisor.fleet is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity configured")
        if "bid_ceiling" in body:
            return _error(400, "bad_request", _BID_CEILING_GONE)
        try:
            # Under the pass's lock: a pass running meanwhile would find the new instance in the
            # provider's listing before the fleet knows it, and sweep it as an orphan.
            async with supervisor.pass_lock:
                host = await supervisor.fleet.prepare(
                    max_spend=float(body["max_spend"]),
                    max_hours=float(body.get("max_hours", 1)),
                    max_all_in_hourly=body.get("max_all_in_hourly"),
                    when_ready=body.get("when_ready", "join"),
                    # Optional: exactly this offer, and how to rent it (D55).
                    offer_id=str(body["offer_id"]) if body.get("offer_id") is not None else None,
                    kind=body.get("kind"),
                    connection=str(body["connection"]) if body.get("connection") else None,
                )
        except LeaseRefused as exc:
            return _error(400, "lease_refused", str(exc))
        if host is None:
            # The real reason, not a fixed sentence: a burn cap, a filter, a lost bid and a
            # vanished offer all used to read "no acceptable offer".
            why = supervisor.fleet.last_refusal or "no acceptable offer"
            return _error(503, "nothing_prepared", f"{why}; nothing was spent")
        return JSONResponse(
            {"host_id": host.host_id, "lease_id": host.lease_id, "bid_hourly": host.bid_hourly},
            status_code=201,
        )

    @app.get("/pool/hosts/{host_id}")
    async def host_detail(host_id: str) -> JSONResponse:
        """Everything about one host in one answer: what it is, what stage preparing it has
        reached, what its engine holds right now, and its own slice of the decision log.

        This is what an operator otherwise has to assemble by hand from three places while a
        host sits at "preparing" with nothing to look at.
        """
        configured = supervisor.hosts.get(host_id)
        rented = supervisor.fleet.hosts.get(host_id) if supervisor.fleet else None
        if configured is None and rented is None:
            return _error(404, "unknown_host", f"no host {host_id!r}")

        detail: dict[str, Any] = {"host_id": host_id, "events": supervisor.events.recent(limit=60, host_id=host_id)}
        required: set[str] = set()
        client = None

        if configured is not None:
            required = set(configured.required_tags)
            client = configured.client
            detail.update(
                kind=configured.config.kind, state=configured.state.value, workers=configured.config.workers,
                residency=configured.config.residency, last_error=configured.last_error,
                transport=configured.config.transport.type, dial_url=configured.dial_url,
                tunnel=supervisor.tunnel_status(host_id),
            )
        else:
            # What *this* host was bought for — its workload's group, its profile, or the rented
            # set — exactly as the supervisor prepares it (service: `fleet.tags_for`). Read from
            # the shared set, a workload's host serving gemma4:26b and e4b was shown as still to
            # download the shared pool's gemma4:31b (found live, 2026-10-06).
            required = set(supervisor.fleet.tags_for(rented))
            client = supervisor._rented_clients.get(host_id)
            detail.update(
                kind="rented-interruptible" if rented.interruptible else "rented-on-demand",
                state=rented.state, workers=rented.workers,
                residency="pinned", hardware=rented.offer.hardware, machine=rented.offer.machine_id,
                instance=rented.instance.instance_id, bid_hourly=rented.bid_hourly,
                interruptible=rented.interruptible,
                connection=rented.connection_name, priced=_priced(rented.offer),
                hours_held=round(rented.hours_held, 3), lease_id=rented.lease_id,
                estimated_spend=round(rented.estimate(), 4), reported_spend=round(rented.reported_spend, 4),
                stage=rented.stage, progress=rented.progress, prepared=rented.prepared,
                # What the machine says about itself, where the pool put an agent on it (D63).
                agent={
                    "installed": rented.agent is not None,
                    "beats": rented.agent_beats,
                    "detail": rented.agent_detail,
                    "facts": rented.agent_facts,
                    "models": rented.agent_models,
                },
                tunnel=(
                    {"up": t.up, "local_port": t.local_port, "restarts": t.restarts, "last_error": t.last_error}
                    if (t := supervisor.fleet.tunnels.get(host_id)) else None
                ),
            )
            try:  # what the provider itself says — "Pulling from ollama/ollama", and the like
                status = await supervisor.fleet.provider_for(rented).status(rented.instance)
                detail["provider"] = {"state": str(status.state), "detail": status.detail}
            except Exception as exc:  # noqa: BLE001 - a detail view never fails over a detail
                detail["provider"] = {"state": "unknown", "detail": str(exc)}

        detail["required_tags"] = sorted(required)
        # This machine's own engine, not the pool's default (D93): asked with the wrong
        # adapter, a vLLM host read as "not answering: 405 /api/ps" while it was.
        engine = supervisor.engine_for(rented) if rented is not None else supervisor.engine_for(configured)
        loads_by_restart = bool(getattr(engine, "loads_by_restart", False))
        agent_models = getattr(rented if rented is not None else configured, "agent_models", None)
        resident: frozenset[str] = frozenset()
        available: frozenset[str] = frozenset()
        if client is not None:
            try:
                resident = await engine.models_resident(client)
                available = await engine.models_available(client)
                # What the agent holds on disk counts too. An engine launched with its models
                # (vLLM) can only name what it is serving, so while its processes are still
                # loading it says nothing is here — and the operator watching three fetched
                # models read "still to download" beside a 16 GB file that had landed.
                on_disk = available | resident | _held_on_disk(agent_models)
                detail["engine"] = {
                    "answers": True,
                    "loaded": sorted(resident),
                    "on_disk": sorted(on_disk),
                    "missing_from_disk": sorted(required - on_disk),
                    "not_loaded": sorted(required - resident),
                }
            except Exception as exc:  # noqa: BLE001 - say it is not answering, do not 500
                detail["engine"] = _silent_engine(
                    str(exc) or type(exc).__name__, rented, loads_by_restart=loads_by_restart
                )
        else:
            detail["engine"] = {"answers": False, "detail": "the pool has no connection to this host yet"}

        # One state per model, from everything the pool knows (D105): its own downloads, the
        # agent's report, the engine's list. The engine alone cannot say it — for one launched
        # with its models it knows nothing until it is up.
        detail["models"] = _model_states(
            required,
            progress=detail.get("progress") or {},
            agent_models=agent_models,
            resident=resident,
            available=available,
            loads_by_restart=loads_by_restart,
            restart_asked=rented is not None and rented.restart_asked_at is not None,
        )
        detail["stage_detail"] = _stage_of(detail)
        return JSONResponse(detail)

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

    @app.get("/pool/machines")
    async def machines() -> JSONResponse:
        """What each machine has done for this pool (D69) — a view over the logs, not a store."""
        fleet = supervisor.fleet
        if fleet is None:
            return _error(400, "cannot_rent", "this pool has no rented capacity configured")
        records = fleet.machine_history(refresh_after_s=0)
        return JSONResponse(
            {
                "machines": [
                    record.as_dict()
                    for record in sorted(
                        records.values(), key=lambda r: (-r.rentals, r.machine_id)
                    )
                ],
                "policy": fleet.rented.history.model_dump(),
            }
        )

    @app.post("/pool/hosts/{host_id}/resize")
    async def resize_host(host_id: str, request: Request) -> JSONResponse:
        """How many requests this host takes at once, changed while it runs (D56).

        Lowering is immediate and graceful. Raising relaunches the engine through the host's
        own agent, so it costs that host about a minute and — like every restart — is an
        operator's explicit act, with the host id typed again (D41).
        """
        fleet = supervisor.fleet
        if fleet is None or host_id not in fleet.hosts:
            return _error(404, "unknown_host", f"no rented host {host_id!r}")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        workers = body.get("workers")
        if not isinstance(workers, int) or isinstance(workers, bool):
            return _error(400, "bad_request", "`workers` must be a whole number")
        host = fleet.hosts[host_id]
        if workers > host.workers and body.get("confirm") != host_id:
            return _error(
                400,
                "not_confirmed",
                "raising the count relaunches this host's engine: requests in flight on it will "
                "fail over or fail. Type the host id again as `confirm`.",
            )
        done, why = await fleet.resize(host, workers)
        if not done:
            return _error(400, "cannot_resize", why)
        supervisor._publish_rented()
        return JSONResponse({"host_id": host_id, "workers": host.workers, "detail": why})

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
