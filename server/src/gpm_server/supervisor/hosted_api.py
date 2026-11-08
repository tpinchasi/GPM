"""The control API's hosted model APIs: list, test, add, change, remove, the key, the budget (S13, D171).

docs/spec/providers.md §10. A hosted API is a hosted host (D164) and the catalog builds it serves,
each pinned to it with `requires`, matched against the host's own capability — so a model id two
providers share is priced and sent as its own provider says. Every change goes through the file's
own path — validated, planned, applied only when nothing that loosens a limit is unconfirmed —
and takes effect at once. The key is the one thing not in the file: it is tested against the
provider's `/models`, which spends nothing, then kept by the supervisor in the credential store
for the router to read (hosted_credentials).

**Nothing here answers with a key**, or with anything that could hold one: refusals are the
pool's own words, and no key is accepted over plain HTTP from anywhere but this machine.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import hosted_credentials
from ..config import ANTHROPIC_ENGINE, HOSTED_ENGINES, HostConfig, Price, TransportConfig
from ..configplan import (
    CannotEdit,
    append_list_item,
    remove_key,
    remove_list_item,
    set_in_list_item,
    set_values,
)
from ..engines import get_engine
from ..hosted_presets import PRESETS, preset_for
from ..transports import build_client
from .providers_api import _body, _safe_transport

if TYPE_CHECKING:
    from .service import Supervisor

#: What a model offered through a hosted API may serve (D142); chat unless said.
KINDS = ("chat", "embed", "transcribe", "rerank")
#: A catalog name the console writes: what an app asks for.
MODEL_NAME_MAX = 128


def pin(host_id: str) -> str:
    """The capability that ties a catalog build to one hosted API: `requires: [hosted:<id>]`."""
    return f"hosted:{host_id}"


def format_of(engine: str) -> str:
    return "Anthropic" if engine == ANTHROPIC_ENGINE else "OpenAI"


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def address_problem(base_url: str) -> Optional[str]:
    """A provider's base URL: https, or http on this machine only (a test double, a local proxy)."""
    try:
        parts = urlsplit(base_url)
    except ValueError:
        return "not a URL"
    if parts.scheme not in ("https", "http") or not parts.hostname:
        return "a provider's address is an https:// URL"
    if parts.username or parts.password:
        return "a provider's address carries no user name or password: the key goes in the key field"
    if parts.query or parts.fragment:
        return "a provider's address is its API base, with no query or fragment"
    if parts.scheme == "http" and not _loopback(parts.hostname):
        return "a provider is reached over https; plain http only on this machine"
    return None


def transport_for(base_url: str) -> TransportConfig:
    base = base_url.rstrip("/")
    return TransportConfig(type="http" if base.startswith("http://") else "https", base_url=base)


def price_of(raw: Any) -> tuple[Optional[dict[str, float]], Optional[str]]:
    """A price as the screen sends it, checked as the catalog would check it."""
    if not isinstance(raw, dict):
        return None, "every model needs its price: input and output per million tokens, or per minute"
    cleaned = {}
    for key in ("input", "cached_input", "cache_write", "output", "per_minute"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None, f"{key} is a number of dollars"
        if number < 0:
            return None, f"{key} cannot be negative"
        cleaned[key] = number
    try:
        Price(**cleaned)
    except ValueError as exc:
        return None, str(exc).splitlines()[-1] if str(exc) else "not a price"
    return cleaned, None


def register(
    app: FastAPI,
    supervisor: "Supervisor",
    *,
    error: Callable[..., JSONResponse],
    validate_plan_apply: Callable[[str, str, dict], JSONResponse],
    plan_of: Callable[[str], list[Any]],
) -> None:

    def file_or_refusal() -> Optional[JSONResponse]:
        if supervisor.store is None:
            return error(400, "no_config_file", "this pool was not started from a file")
        return None

    def key_text(body: dict) -> tuple[Optional[str], Optional[JSONResponse]]:
        value = body.get("credential")
        if value is None or value == "":
            return None, None
        if not isinstance(value, str) or len(value) > 4096 or not value.isascii() or not value.strip().isprintable():
            return None, error(400, "bad_credential", "a key is one line of plain ASCII text, at most 4096 characters")
        return value.strip() or None, None

    def hosted_host(host_id: str) -> Optional[HostConfig]:
        return next((h for h in supervisor.config.hosts if h.id == host_id and h.kind == "hosted"), None)

    def models_of(host: HostConfig) -> list[dict[str, Any]]:
        """The catalog builds this host serves: pinned to it, or (written by hand) open to any host
        of its engine. The console changes only the pinned ones."""
        out = []
        caps = set(host.capabilities)
        for name, entry in supervisor.config.catalog.items():
            for variant in entry.variants:
                if variant.engine != host.engine or not set(variant.requires) <= caps:
                    continue
                out.append({
                    "name": name, "tag": variant.tag, "pinned": pin(host.id) in variant.requires,
                    "price": variant.price.model_dump(exclude_none=True) if variant.price else None,
                    "serves": sorted(variant.kinds()), "in_model_set": name in supervisor.config.pool.model_set,
                })
        return out

    def spend_by_host() -> dict[str, dict[str, float]]:
        from ..router.budgets import period_starts

        day, month = period_starts()
        rows = supervisor.db.query(
            "SELECT host_id, SUM(CASE WHEN ts >= ? THEN cost_usd ELSE 0 END) AS today, SUM(cost_usd) AS this_month "
            "FROM request_log WHERE ts >= ? AND cost_usd IS NOT NULL AND cost_usd > 0 GROUP BY host_id", (day, month))
        return {r["host_id"]: {"today": round(r["today"] or 0.0, 6), "this_month": round(r["this_month"] or 0.0, 6)}
                for r in rows}

    def view(host: HostConfig, counters: dict, spend: dict) -> dict[str, Any]:
        running = supervisor.hosts.get(host.id)
        counter = counters.get(host.id)
        preset = preset_for(host.transport.base_url or "", host.engine or "")
        return {
            "id": host.id, "engine": host.engine, "format": format_of(host.engine or ""),
            "base_url": host.transport.base_url, "preset": preset.id if preset else None,
            "provider": preset.name if preset else None,
            "workers": host.workers, "disabled": host.disabled, "cache_prompts": host.cache_prompts,
            "credential": hosted_credentials.describe(supervisor.config, host),
            "state": running.state.value if running else "unknown",
            "last_error": running.last_error if running else None,
            "offered": sorted(running.available) if running else [],
            "out_of_funds": counter.out_of_funds if counter else None,
            "out_of_funds_at": counter.out_of_funds_at if counter else None,
            "models": models_of(host), "pinned_by": pin(host.id) in host.capabilities,
            "spend": spend.get(host.id, {"today": 0.0, "this_month": 0.0}),
        }

    def budget_view() -> Optional[dict[str, Any]]:
        burst = supervisor.config.pool.bursts.hosted
        if burst is None:
            return None
        return {"after_s": burst.after_s, **burst.budget.model_dump()}

    @app.get("/pool/hosted")
    async def list_hosted(request: Request) -> JSONResponse:
        counters = supervisor.counters.all()
        spend = await asyncio.to_thread(spend_by_host)
        return JSONResponse({
            "hosts": [view(h, counters, spend) for h in supervisor.config.hosts if h.kind == "hosted"],
            "presets": [p.as_dict() for p in PRESETS],
            "budget": budget_view(),
            "secure_transport": _safe_transport(request),
        })

    async def offered(engine: str, transport: TransportConfig, key: Optional[str],
                      saved: Optional[HostConfig] = None) -> tuple[list[dict], list[str], bool]:
        """Ask a provider's `/models` — which spends nothing — with the key given, or a saved host's
        own. Steps as the console shows them, the model ids, and whether every step passed."""
        steps: list[dict] = []
        if saved is not None:
            client = hosted_credentials.client(supervisor.config, saved)
        else:
            client = build_client(transport, supervisor.config.pool, transport.base_url, bearer=key)
        try:
            ids = await get_engine(engine)._offered(client)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                steps.append({"step": "key", "ok": False, "detail": f"the provider refused the key ({status})"})
            elif status == 404:
                steps.append({"step": "address", "ok": False,
                              "detail": "nothing answers /models there: is this the provider's API base?"})
            else:
                steps.append({"step": "address", "ok": False, "detail": f"/models answered {status}"})
            return steps, [], False
        except (httpx.HTTPError, ValueError, KeyError, AttributeError) as exc:
            steps.append({"step": "address", "ok": False,
                          "detail": f"the provider could not be reached: {type(exc).__name__}"})
            return steps, [], False
        finally:
            await client.aclose()
        steps.append({"step": "address", "ok": True, "detail": "the provider answered"})
        steps.append({"step": "key", "ok": True, "detail": "the key was accepted"})
        steps.append({"step": "models", "ok": bool(ids), "detail": f"{len(ids)} model(s) offered"
                      if ids else "the provider offers no models to this key"})
        return steps, sorted(ids), bool(ids)

    @app.post("/pool/hosted/test")
    async def test_hosted(request: Request) -> JSONResponse:
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        key, bad = key_text(body)
        if bad is not None:
            return bad
        if key is not None and not _safe_transport(request):
            return error(403, "insecure_transport", "a key is accepted only over HTTPS, or from this machine")
        if body.get("host"):
            host = hosted_host(str(body["host"]))
            if host is None:
                return error(404, "unknown_host", "no hosted API by that name")
            if key is not None:  # a replacement, tried where this host points before it is kept
                steps, ids, ok = await offered(host.engine or "", host.transport, key)
            else:
                steps, ids, ok = await offered(host.engine or "", host.transport, None, saved=host)
            return JSONResponse({"ok": ok, "steps": steps, "models": ids})
        engine, base_url = str(body.get("engine") or ""), str(body.get("base_url") or "").strip()
        if engine not in HOSTED_ENGINES:
            return error(400, "bad_engine", "choose OpenAI's format or Anthropic's")
        problem = address_problem(base_url)
        if problem:
            return error(400, "bad_address", problem)
        if key is None:
            # An unsaved address is tried only with a key typed for it: none of the pool's own goes there.
            return error(400, "no_credential", "type the provider's key to test it")
        steps, ids, ok = await offered(engine, transport_for(base_url), key)
        return JSONResponse({"ok": ok, "steps": steps, "models": ids})

    def models_wanted(raw: Any, engine: str) -> tuple[list[dict], Optional[JSONResponse]]:
        if not isinstance(raw, list):
            return [], error(400, "bad_models", "models is a list")
        out, names = [], set()
        for item in raw:
            if not isinstance(item, dict):
                return [], error(400, "bad_models", "each model is an object")
            name, tag = str(item.get("name") or "").strip(), str(item.get("tag") or "").strip()
            if not tag or len(tag) > 256 or not tag.isprintable():
                return [], error(400, "bad_models", "each model names the provider's id for it")
            name = name or tag
            if len(name) > MODEL_NAME_MAX or not name.isprintable() or name != name.strip() or ":" == name[-1:]:
                return [], error(400, "bad_models", f"{name!r} is not a name an app can ask for")
            if name in names:
                return [], error(400, "bad_models", f"{name!r} is offered twice")
            names.add(name)
            price, problem = price_of(item.get("price"))
            if problem:
                return [], error(400, "bad_price", f"{name}: {problem}")
            if engine == ANTHROPIC_ENGINE and "input" in price and "cache_write" not in price:
                return [], error(400, "bad_price", f"{name}: an Anthropic-format model needs its cache-write price, "
                                 "since the pool asks for its prompts to be cached (D170)")
            serves = item.get("serves") or ["chat"]
            if not isinstance(serves, list) or not set(serves) <= set(KINDS):
                return [], error(400, "bad_models", f"{name}: serves is a list of {', '.join(KINDS)}")
            out.append({"name": name, "tag": tag, "price": price, "serves": list(dict.fromkeys(serves))})
        return out, None

    def with_models(text: str, host: HostConfig, wanted: list[dict]) -> str:
        """The file with this host's pinned builds replaced by `wanted`: entries left with no build
        are removed, from the model set too; new names join it. Builds not pinned to it stay."""
        mark = pin(host.id)
        catalog = supervisor.config.catalog
        model_set = list(supervisor.config.pool.model_set)
        by_name = {w["name"]: w for w in wanted}
        touched = set(by_name) | {name for name, entry in catalog.items()
                                  if any(mark in v.requires for v in entry.variants)}
        for name in sorted(touched):
            entry = catalog.get(name)
            kept = [v.model_dump(exclude_defaults=True, exclude_none=True)
                    for v in (entry.variants if entry else []) if mark not in v.requires]
            if name in by_name:
                w = by_name[name]
                build: dict[str, Any] = {"tag": w["tag"], "engine": host.engine, "requires": [mark], "price": w["price"]}
                if w["serves"] != ["chat"]:
                    build["serves"] = w["serves"]
                kept.append(build)
            if kept:
                fields = entry.model_dump(exclude_defaults=True, exclude_none=True) if entry else {}
                fields["variants"] = kept
                text = set_values(text, ("catalog",), {name: fields})
                if name not in model_set:
                    model_set.append(name)
            elif entry is not None:
                text = remove_key(text, ("catalog",), name)
                if name in model_set:
                    model_set.remove(name)
        if model_set != list(supervisor.config.pool.model_set):
            text = set_values(text, ("pool",), {"model_set": model_set})
        return text

    async def apply_with_key(text: str, version: str, body: dict, host_id: str, base_url: str,
                             key: Optional[str], engine: str) -> JSONResponse:
        """Plan first — a change that must be typed again stores nothing — then test and store the
        key, so the host is built with it the moment it is applied; undone if the apply does not happen."""
        errors = supervisor.store.validate(text)
        if errors:
            return error(400, "invalid_config", "; ".join(errors))
        plan = plan_of(text)
        if body.get("plan_only"):
            return JSONResponse({"changes": [c.as_dict() for c in plan]})
        refusal = [c.refused for c in plan if c.refused]
        if refusal:
            return error(409, "change_refused", "; ".join(refusal))
        loosening = [c for c in plan if c.requires_retype]
        if loosening and str(body.get("confirm")) not in {str(c.requires_retype) for c in loosening}:
            return validate_plan_apply(text, version, body)  # answers not_confirmed with the plan
        store = hosted_credentials.HostedKeys(supervisor.config)
        before = store.get(host_id) if key is not None else None
        if key is not None:
            steps, _, ok = await offered(engine, transport_for(base_url), key)
            if not ok:
                return error(400, "credential_refused", next((s["detail"] for s in steps if not s["ok"]), "refused"))
            store.put(host_id, key, {"base_url": base_url.rstrip("/")})
        async with supervisor.pass_lock:
            response = validate_plan_apply(text, version, body)
        if response.status_code != 200 and key is not None:
            if before is not None:
                store.put(host_id, before.value, before.endpoint)
            else:
                store.remove(host_id)
        return response

    @app.post("/pool/hosted")
    async def add_hosted(request: Request) -> JSONResponse:
        refused = file_or_refusal()
        if refused is not None:
            return refused
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        key, bad = key_text(body)
        if bad is not None:
            return bad
        if key is not None and not _safe_transport(request):
            return error(403, "insecure_transport", "a key is accepted only over HTTPS, or from this machine")
        host_id = str(body.get("id") or "")
        if not hosted_credentials.HOST_ID.match(host_id):
            return error(400, "bad_name", "a name of lower-case letters, digits, '-' and '_', at most 40 characters")
        if any(h.id == host_id for h in supervisor.config.hosts):
            return error(409, "exists", f"there is already a host named {host_id!r}")
        engine, base_url = str(body.get("engine") or ""), str(body.get("base_url") or "").strip().rstrip("/")
        if engine not in HOSTED_ENGINES:
            return error(400, "bad_engine", "choose OpenAI's format or Anthropic's")
        problem = address_problem(base_url)
        if problem:
            return error(400, "bad_address", problem)
        try:
            workers = int(body.get("workers") or 4)
        except (TypeError, ValueError):
            return error(400, "bad_workers", "workers is a whole number")
        if not 1 <= workers <= 1000:
            return error(400, "bad_workers", "workers is between 1 and 1000: how many requests it may have in flight")
        wanted, bad = models_wanted(body.get("models") or [], engine)
        if bad is not None:
            return bad
        transport = transport_for(base_url)
        entry: dict[str, Any] = {"id": host_id, "kind": "hosted", "engine": engine, "workers": workers,
                                 "capabilities": [pin(host_id)],
                                 "transport": {"type": transport.type, "base_url": transport.base_url}}
        if engine == ANTHROPIC_ENGINE and body.get("cache_prompts") is False:
            entry["cache_prompts"] = False
        text, version = supervisor.store.read()
        try:
            text = append_list_item(text, "hosts", entry)
            host = HostConfig.model_validate(entry)
            text = with_models(text, host, wanted)
        except (CannotEdit, ValueError) as exc:
            return error(409, "cannot_edit", str(exc))
        response = await apply_with_key(text, version, body, host_id, base_url, key, engine)
        if response.status_code == 200:
            supervisor.events.record(
                "hosted_added", f"hosted API {host_id!r} ({base_url}) added from the console with "
                f"{len(wanted)} model(s)" + (", its key typed in" if key is not None else ", with no key yet"),
                numbers={"host": host_id, "models": [w["name"] for w in wanted]})
        return response

    @app.patch("/pool/hosted/{host_id}")
    async def change_hosted(host_id: str, request: Request) -> JSONResponse:
        refused = file_or_refusal()
        if refused is not None:
            return refused
        host = hosted_host(host_id)
        if host is None:
            return error(404, "unknown_host", "no hosted API by that name")
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        text, version = supervisor.store.read()
        fields: dict[str, Any] = {}
        if "workers" in body:
            try:
                fields["workers"] = int(body["workers"])
            except (TypeError, ValueError):
                return error(400, "bad_workers", "workers is a whole number")
            if not 1 <= fields["workers"] <= 1000:
                return error(400, "bad_workers", "workers is between 1 and 1000")
        if "disabled" in body:
            fields["disabled"] = bool(body["disabled"])
        if "cache_prompts" in body and host.engine == ANTHROPIC_ENGINE:
            fields["cache_prompts"] = bool(body["cache_prompts"])
        try:
            if "models" in body:
                wanted, bad = models_wanted(body["models"], host.engine or "")
                if bad is not None:
                    return bad
                if pin(host_id) not in host.capabilities:
                    fields["capabilities"] = [*host.capabilities, pin(host_id)]
                text = with_models(text, host, wanted)
            if fields:
                text = set_in_list_item(text, "hosts", "id", host_id, fields)
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
            supervisor.events.record("hosted_changed", f"hosted API {host_id!r} changed from the console",
                                     numbers={"host": host_id, "fields": sorted(set(fields) | ({"models"} & set(body)))})
        return response

    @app.delete("/pool/hosted/{host_id}")
    async def remove_hosted(host_id: str, request: Request) -> JSONResponse:
        refused = file_or_refusal()
        if refused is not None:
            return refused
        host = hosted_host(host_id)
        if host is None:
            return error(404, "unknown_host", "no hosted API by that name")
        body = await _body(request) or {}
        text, version = supervisor.store.read()
        try:
            text = with_models(text, host, [])
            text = remove_list_item(text, "hosts", "id", host_id)
        except CannotEdit as exc:
            return error(409, "cannot_edit", str(exc))
        errors = supervisor.store.validate(text)
        if errors:
            return error(400, "invalid_config", "; ".join(errors))
        if str(body.get("confirm") or "") != host_id:
            gone = [m["name"] for m in models_of(host) if m["pinned"]]
            return JSONResponse(status_code=400, content={
                "error": "not_confirmed", "detail": "type the hosted API's name to remove it",
                "changes": [{"detail": f"removes hosted API {host_id!r}" + (f" and the models only it serves: {', '.join(gone)}"
                                                                             if gone else ""),
                             "requires_retype": True, "value": host_id}],
            })
        async with supervisor.pass_lock:
            response = validate_plan_apply(text, version, {**body, "confirm": host_id})
        if response.status_code == 200:
            hosted_credentials.HostedKeys(supervisor.config).remove(host_id) if hosted_credentials.HOST_ID.match(host_id) else None
            supervisor.events.record("hosted_removed", f"hosted API {host_id!r} removed from the console",
                                     numbers={"host": host_id})
        return response

    @app.put("/pool/hosted/{host_id}/credential")
    async def set_key(host_id: str, request: Request) -> JSONResponse:
        if not _safe_transport(request):
            return error(403, "insecure_transport", "a key is accepted only over HTTPS, or from this machine")
        host = hosted_host(host_id)
        if host is None:
            return error(404, "unknown_host", "no hosted API by that name")
        if hosted_credentials.names_its_own(host):
            return error(409, "key_from_environment", f"this host takes its key from {host.transport.bearer_env or host.transport.basic_env} "
                         "in the pool's environment; remove that from the file to type one here")
        if not hosted_credentials.HOST_ID.match(host_id):
            return error(409, "cannot_type", "this host's id cannot name a key file; rename it on the Configuration screen")
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        key, bad = key_text(body)
        if bad is not None:
            return bad
        if key is None:
            return error(400, "no_credential", "type the provider's key")
        steps, _, ok = await offered(host.engine or "", host.transport, key)
        if not ok:
            return error(400, "credential_refused", next((s["detail"] for s in steps if not s["ok"]), "refused"))
        async with supervisor.pass_lock:
            hosted_credentials.HostedKeys(supervisor.config).put(host_id, key, hosted_credentials.endpoint(host))
            supervisor.rekey_hosted(host_id)
        supervisor.events.record("hosted_key_set", f"the key of hosted API {host_id!r} was typed in on the console",
                                 numbers={"host": host_id})
        return JSONResponse({"credential": hosted_credentials.describe(supervisor.config, host), "steps": steps})

    @app.delete("/pool/hosted/{host_id}/credential")
    async def remove_key_of(host_id: str) -> JSONResponse:
        host = hosted_host(host_id)
        if host is None:
            return error(404, "unknown_host", "no hosted API by that name")
        if not hosted_credentials.HOST_ID.match(host_id):
            return error(409, "cannot_type", "this host has no key typed in")
        async with supervisor.pass_lock:
            removed = hosted_credentials.HostedKeys(supervisor.config).remove(host_id)
            supervisor.rekey_hosted(host_id)
        if removed:
            supervisor.events.record("hosted_key_removed", f"the key of hosted API {host_id!r} was removed on the console",
                                     numbers={"host": host_id})
        return JSONResponse({"credential": hosted_credentials.describe(supervisor.config, host)})

    @app.patch("/pool/hosted-budget")
    async def set_budget(request: Request) -> JSONResponse:
        """The shared workload's hosted burst (D165): how long a request waits for a machine before
        a hosted API takes it, and its dollars per request, day and month."""
        refused = file_or_refusal()
        if refused is not None:
            return refused
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        current = supervisor.config.pool.bursts.hosted
        burst: dict[str, Any] = current.model_dump(exclude_defaults=True) if current else {}
        budget: dict[str, Any] = dict(burst.get("budget") or {})
        for key in ("per_request", "per_day", "per_month"):
            if key in body:
                value = body[key]
                if value in (None, "", 0):
                    budget.pop(key, None)
                    continue
                try:
                    budget[key] = float(value)
                except (TypeError, ValueError):
                    return error(400, "bad_budget", f"{key} is a number of dollars")
        if "after_s" in body:
            try:
                burst["after_s"] = float(body["after_s"])
            except (TypeError, ValueError):
                return error(400, "bad_budget", "after_s is a number of seconds")
        if not budget.get("per_day") and not budget.get("per_month"):
            return error(400, "bad_budget", "a budget names dollars per day or per month, or both")
        burst["budget"] = budget
        text, version = supervisor.store.read()
        bursts = {"hosted": burst}
        try:
            try:
                text = set_values(text, ("pool", "bursts"), bursts)
            except CannotEdit:
                text = set_values(text, ("pool",), {"bursts": bursts})
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
            supervisor.events.record("hosted_budget_set", "the hosted budget was changed on the console",
                                     numbers={k: v for k, v in budget.items()} | {"after_s": burst.get("after_s", 5.0)})
        return response

