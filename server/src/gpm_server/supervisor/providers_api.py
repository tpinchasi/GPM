"""The control API's provider accounts: list, test, add, change, remove, and the credential.

docs/spec/providers.md §5, §8. Every change to a connection goes through the configuration
file's own path — validated, planned, applied only when nothing that loosens a limit is
unconfirmed — and takes effect at once (D135). A credential is the one thing that does not go
in the file: it is tested against the provider, then kept by the supervisor (D130).

**Nothing here answers with a credential**, or with anything that could hold one: refusals are
the pool's own words, and a provider's message is scrubbed of the credential before it is
passed on. No credential is accepted over plain HTTP from anywhere but this machine.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import CONNECTION_NAME
from ..configplan import CannotEdit, remove_key, set_values, with_connections
from ..credentials import endpoint_of
from ..providers import (
    OfferQuery,
    ProviderError,
    ProviderNotFound,
    installed_plugins,
    plugin_presentation,
    takes_credential,
)
from ..providers.base import redacted
from .connections import scrub

if TYPE_CHECKING:
    from .service import Supervisor

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _safe_transport(request: Request) -> bool:
    """A credential crosses the wire only encrypted, or not at all off this machine."""
    client = request.client.host if request.client else ""
    return request.url.scheme == "https" or client in _LOOPBACK


async def _body(request: Request) -> Optional[dict[str, Any]]:
    try:
        body = await request.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def register(
    app: FastAPI,
    supervisor: "Supervisor",
    *,
    error: Callable[..., JSONResponse],
    validate_plan_apply: Callable[[str, str, dict], JSONResponse],
    plan_of: Callable[[str], list[Any]],
) -> None:
    accounts = supervisor.accounts

    def fleet_or_refusal() -> Optional[JSONResponse]:
        if supervisor.fleet is None or supervisor.config.rented is None:
            return error(400, "cannot_rent", "this pool has no rented capacity configured")
        if supervisor.store is None:
            return error(400, "no_config_file", "this pool was not started from a file")
        return None

    def credential_text(body: dict) -> tuple[Optional[str], Optional[JSONResponse]]:
        value = body.get("credential")
        if value is None or value == "":
            return None, None
        if not isinstance(value, str) or len(value) > 4096 or not value.isascii() or not value.strip().isprintable():
            # Said without the value, whatever it was.
            return None, error(400, "bad_credential", "a credential is one line of plain ASCII text, at most 4096 characters")
        return value.strip() or None, None

    def configured(name: str) -> tuple[Any, Any]:
        """(how its running plug-in was built, the plug-in), or (None, None) for a connection the
        file names that is not running — a rename or a change waiting for a restart. Credentials
        follow the running plug-in, never the file (D134)."""
        provider = supervisor.fleet.providers.get(name)
        built = accounts.built.get(name)
        if name not in supervisor.config.rented.providers or provider is None or built is None:
            return None, None
        return built, provider

    def connection_view(name: str) -> dict[str, Any]:
        fleet = supervisor.fleet
        conn = supervisor.config.rented.providers.get(name)
        provider = fleet.providers.get(name)
        type_name = conn.type if conn is not None else fleet.connection_types.get(name, "?")
        live = [h for h in fleet.hosts.values() if not h.released and fleet.resolve(h.connection_name) == name]
        held = fleet.held_at(name)
        view = accounts.presentation_of(type_name, provider)
        if provider is None:
            # Not running yet: what its class says of itself, for the card's price and note.
            try:
                known = plugin_presentation(type_name)
                view.update({k: known.get(k) for k in ("volume_price_per_gb_month", "volume_note")})
            except Exception:  # noqa: BLE001 — an unknown plug-in: nothing more to say
                pass
        view.update({
            "connection": name,
            "configured": conn is not None,
            "enabled": bool(conn is not None and conn.enabled),
            "settings": redacted(dict(conn.settings)) if conn is not None else {},
            "interruption_prior_per_hour": conn.interruption_prior_per_hour if conn is not None else None,
            # Keep models between hosts (D139): on or off, and whether this provider can at all —
            # only where its volumes reach a data center.
            "keep_models": bool(conn is not None and conn.keep_models),
            # From the running plug-in, else its class: an account not running yet still says truly
            # what its provider can keep.
            "volume_reach": _reach(provider) if provider is not None else _reach_of_type(type_name),
            "model_volumes": sum(1 for v in fleet.workload_store.volumes()
                                 if v.location is not None and fleet.resolve(v.connection or fleet.legacy_connection) == name),
            "credential_env": conn.credential_env if conn is not None else None,
            "credential": accounts.describe(name, provider) if provider is not None and name in accounts.built else None,
            "pending": accounts.pending.get(name),
            "account": accounts._accounts.get(name, (0, None))[1],
            "search_quota": fleet.search_quota(name) if provider is not None else None,
            "search_error": fleet.searching[name].error if name in fleet.searching else None,
            "rented": sum(1 for h in live if h.state != "parked"),
            "parked": sum(1 for h in live if h.state == "parked"),
            # Hosts and their model volumes' storage, which bills whether or not a host has one.
            "hourly": round(sum(h.bid_hourly for h in live) + sum(
                v.hourly for v in fleet.workload_store.volumes()
                if v.location is not None and fleet.resolve(v.connection or fleet.legacy_connection) == name), 4),
            "volumes": sum(1 for record, _ in held if record.startswith("volume:")),
            "held_back": len(fleet.pending_adoption.get(name, [])),
            "holds": len(held),
        })
        return view

    @app.get("/pool/providers")
    async def list_providers(request: Request) -> JSONResponse:
        """Every connection and every installed plug-in. Never a credential: only where it comes
        from and when it was set. `fresh=true` asks each enabled provider's account now."""
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        fleet = supervisor.fleet
        names = list(supervisor.config.rented.providers) + [n for n in fleet.providers if n not in supervisor.config.rented.providers]
        fresh = request.query_params.get("fresh") == "true"
        for name in names:
            if name in fleet.providers and (fresh or name not in accounts._accounts):
                await accounts.account(name, fresh=fresh)
        return JSONResponse({
            "connections": [connection_view(name) for name in names],
            "plugins": accounts.plugins(),
            "secure_transport": _safe_transport(request),
            "label_prefix": fleet.label_prefix,
        })

    @app.get("/pool/providers/plugins/{type_name}")
    async def describe_plugin(type_name: str) -> JSONResponse:
        """An installed plug-in an operator chose on Add provider: loaded now, by that choice —
        never merely for being installed (T14) — and how it presents itself."""
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        try:
            return JSONResponse(plugin_presentation(type_name))
        except ProviderNotFound:
            return error(404, "unknown_provider", "no such provider plug-in is installed")
        except Exception:  # noqa: BLE001 - a plug-in that fails to load says only that
            return error(500, "plugin_failed", f"the {type_name[:40]} plug-in failed to load")

    @app.post("/pool/providers/test")
    async def test_provider(request: Request) -> JSONResponse:
        """Test a connection — saved, saved with a new credential, or not saved at all. Saves
        nothing, rents nothing. A stored credential is only ever sent where it was saved for:
        a saved connection is tested with its own settings, never with settings sent here."""
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        credential, bad = credential_text(body)
        if bad is not None:
            return bad
        if credential is not None and not _safe_transport(request):
            return error(403, "insecure_transport", "a credential is accepted only over HTTPS, or from this machine")
        fleet = supervisor.fleet
        name = body.get("connection")
        fresh = None
        if name is not None:
            conn, running = configured(str(name))
            if conn is None:
                return error(404, "unknown_connection", f"no running provider connection {str(name)[:40]!r}")
            if credential is None:
                provider = running
            else:
                if not takes_credential(running):
                    return error(409, "plugin_reads_its_own", "this plug-in reads its own credential")
                try:
                    provider = fresh = accounts.factory(conn.type, dict(conn.settings))
                except Exception:  # noqa: BLE001 - settings the plug-in refuses
                    return error(400, "bad_settings", "the plug-in could not be built from this connection's settings")
                provider.set_credential(credential)
        else:
            type_name = str(body.get("type") or "")
            if type_name not in installed_plugins():
                return error(400, "unknown_provider", "choose an installed provider")
            settings = body.get("settings") or {}
            if not isinstance(settings, dict):
                return error(400, "bad_request", "settings are a mapping")
            try:
                provider = fresh = accounts.factory(type_name, dict(settings))
            except Exception as exc:  # noqa: BLE001 - a plug-in refusing its settings says why
                return error(400, "bad_settings", scrub(exc, credential))
            # The environment's credential is never sent where a request points, nor one a request
            # names (T29): without one typed in, only the plug-in's own, to its own default endpoint.
            pointed = bool(settings) or bool(body.get("credential_env"))
            if not takes_credential(provider):
                if settings:
                    await fresh.aclose() if hasattr(fresh, "aclose") else None
                    return error(409, "plugin_reads_its_own",
                                 "this plug-in reads its own credential: save the account, then test it")
            elif credential is not None:
                provider.set_credential(credential)
            elif pointed:
                await fresh.aclose() if hasattr(fresh, "aclose") else None
                return error(400, "bad_credential", "type the credential in: one from the supervisor's environment is "
                                                    "not sent where a request points")
            else:
                from ..config import ProviderConnection

                provider.set_credential(accounts.resolve_environment(ProviderConnection(type=type_name), provider))
        try:
            steps = await _steps(provider, fleet.label_prefix, credential, unsaved=name is None)
        finally:
            if fresh is not None and hasattr(fresh, "aclose"):
                await fresh.aclose()
        return JSONResponse({"steps": steps, "ok": all(s["status"] != "failed" for s in steps)})

    @app.post("/pool/providers")
    async def add_provider(request: Request) -> JSONResponse:
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        credential, bad = credential_text(body)
        if bad is not None:
            return bad
        if credential is not None and not _safe_transport(request):
            return error(403, "insecure_transport", "a credential is accepted only over HTTPS, or from this machine")
        name, type_name = str(body.get("name") or ""), str(body.get("type") or "")
        if not CONNECTION_NAME.match(name):
            return error(400, "bad_name", "a lower-case name of letters, digits, '-' and '_', starting with a letter, at most 32 characters")
        if name in supervisor.config.rented.providers:
            return error(409, "exists", f"there is already a provider connection named {name!r}")
        if type_name not in installed_plugins():
            return error(400, "unknown_provider", "choose an installed provider")
        kept = [n for n, conn in accounts.built.items() if conn.type == type_name and n in supervisor.fleet.providers]
        if kept:
            # Still releasing what it holds: a new account for the provider now could be another
            # account, whose listing would make the pool lose those hosts (D133, D136).
            return error(409, "provider_still_running",
                         f"{kept[0]!r} is still this provider's connection, releasing what it holds; add it back once "
                         "its hosts are released, or restore it in the file")
        settings = body.get("settings") or {}
        if not isinstance(settings, dict):
            return error(400, "bad_request", "settings are a mapping")
        entry: dict[str, Any] = {"type": type_name}
        if body.get("enabled") is False:
            entry["enabled"] = False
        if settings:
            entry["settings"] = settings
        if body.get("interruption_prior_per_hour") is not None:
            entry["interruption_prior_per_hour"] = body["interruption_prior_per_hour"]
        if body.get("credential_env"):
            if credential is not None:
                return error(400, "bad_request", "a connection takes its credential from credential_env or from the console, not both")
            entry["credential_env"] = body["credential_env"]
        text, version = supervisor.store.read()
        try:
            text = set_values(with_connections(text, supervisor.config), ("rented", "providers"), {name: entry})
        except CannotEdit as exc:
            return error(409, "cannot_edit", str(exc))
        return await _apply_with_credential(text, version, body, name, type_name, settings, credential)

    @app.patch("/pool/providers/{name}")
    async def change_provider(name: str, request: Request) -> JSONResponse:
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        conn = supervisor.config.rented.providers.get(name)
        if conn is None:
            return error(404, "unknown_connection", f"no provider connection {name[:40]!r}")
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        wanted: dict[str, Any] = {}
        if "enabled" in body:
            if not isinstance(body["enabled"], bool):
                return error(400, "bad_request", "`enabled` is true or false")
            wanted["enabled"] = body["enabled"]
        if "settings" in body:
            if not isinstance(body["settings"], dict):
                return error(400, "bad_request", "settings are a mapping")
            # What the screen was shown redacted comes back redacted: the file keeps its own.
            wanted["settings"] = _unredacted(body["settings"], dict(conn.settings))
        for key in ("interruption_prior_per_hour", "credential_env"):
            if key in body:
                wanted[key] = body[key]
        if "keep_models" in body:
            if not isinstance(body["keep_models"], bool):
                return error(400, "bad_request", "`keep_models` is true or false")
            provider = supervisor.fleet.providers.get(name)
            reach = _reach(provider) if provider is not None else _reach_of_type(conn.type)
            if body["keep_models"] and reach != "data_center":
                return error(409, "cannot_keep_models", (
                    f"{name} keeps a volume on one machine only: it would help only when that same machine is "
                    "free again, which is rare, and it is billed the whole time" if reach == "machine"
                    else f"{name} keeps no storage between hosts"))
            wanted["keep_models"] = body["keep_models"]
        if not wanted:
            return error(400, "bad_request", "nothing to change")
        text, version = supervisor.store.read()
        try:
            text = set_values(with_connections(text, supervisor.config), ("rented", "providers", name), wanted)
        except CannotEdit as exc:
            return error(409, "cannot_edit", str(exc))
        async with supervisor.pass_lock:  # never mid-pass: a pass may be renting there
            return validate_plan_apply(text, version, body)

    @app.delete("/pool/providers/{name}")
    async def remove_provider(name: str, request: Request) -> JSONResponse:
        """Typed to confirm. Refused while the pool holds anything there (the plan's rule)."""
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        conn = supervisor.config.rented.providers.get(name)
        if conn is None:
            return error(404, "unknown_connection", f"no provider connection {name[:40]!r}")
        body = await _body(request) or {}
        text, version = supervisor.store.read()
        try:
            text = remove_key(with_connections(text, supervisor.config), ("rented", "providers"), name)
        except CannotEdit as exc:
            return error(409, "cannot_edit", str(exc))
        errors = supervisor.store.validate(text)
        if errors:
            return error(400, "invalid_config", "; ".join(errors))
        if str(body.get("confirm")) != name:
            listed = [c.as_dict() for c in plan_of(text)]
            for change in listed:
                if change["kind"] == "providers" and not change["refused"]:
                    change["requires_retype"], change["value"] = True, name
            return JSONResponse(status_code=400, content={
                "error": "not_confirmed",
                "detail": f"removing a provider connection is confirmed by typing its name; send `confirm`: {name!r}",
                "changes": listed,
            })
        async with supervisor.pass_lock:  # never mid-pass: a pass may be renting there
            response = validate_plan_apply(text, version, body)
        if response.status_code == 200:
            accounts.store.remove(conn.type)
            accounts.cleared.pop(conn.type, None)
            async with supervisor.pass_lock:
                accounts.sync(supervisor.config, supervisor.config, drop=True)
        return response

    @app.put("/pool/providers/{name}/credential")
    async def set_credential(name: str, request: Request) -> JSONResponse:
        """Set or replace a connection's credential: tested against the provider first, and — while
        the pool holds anything there — refused unless it sees what the pool holds, so a
        replacement cannot quietly switch accounts (D136)."""
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        if not _safe_transport(request):
            return error(403, "insecure_transport", "a credential is accepted only over HTTPS, or from this machine")
        conn, provider = configured(name)
        if conn is None:
            return error(404, "unknown_connection", f"no provider connection {name[:40]!r}")
        if not takes_credential(provider):
            return error(409, "plugin_reads_its_own", "this plug-in reads its own credential; it cannot be typed in")
        if conn.credential_env:
            return error(409, "credential_from_environment",
                         f"this connection always takes its credential from {conn.credential_env}; remove credential_env to type one in")
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        credential, bad = credential_text(body)
        if bad is not None:
            return bad
        if credential is None:
            return error(400, "bad_credential", "send `credential`")
        # Under the pass's lock: a host rented meanwhile, on the old credential, would not be among
        # those the new one is checked against, and a pass mid-rental must not change accounts.
        async with supervisor.pass_lock:
            problem = await _vet(conn, credential, supervisor.fleet.held_at(name))
            if problem is not None:
                return problem
            accounts.store.put(conn.type, credential, endpoint_of(provider, conn.settings))
            accounts.cleared.pop(conn.type, None)
            _hand_live(name, conn, credential)
        supervisor.events.record("credential_set", f"provider connection {name!r}'s credential was set from the console",
                                 numbers={"connection": name})
        return JSONResponse({"credential": accounts.describe(name, provider),
                             "account": await accounts.account(name, fresh=True)})

    @app.delete("/pool/providers/{name}/credential")
    async def remove_credential(name: str) -> JSONResponse:
        refused = fleet_or_refusal()
        if refused is not None:
            return refused
        conn, provider = configured(name)
        if conn is None:
            return error(404, "unknown_connection", f"no provider connection {name[:40]!r}")
        held = supervisor.fleet.held_at(name)
        if held:
            return error(409, "holds_hosts",
                         f"the pool holds {len(held)} host(s) or volume(s) at {name}: without a credential nothing could "
                         "watch or release them. Replace the credential instead")
        if not accounts.store.remove(conn.type):
            return error(404, "no_stored_credential", "no credential was typed in for this connection")
        accounts.rehand(name, provider)
        accounts.forget_account(name)
        supervisor.events.record("credential_removed", f"provider connection {name!r}'s typed-in credential was removed",
                                 numbers={"connection": name})
        return JSONResponse({"credential": accounts.describe(name, provider)})

    # --- shared steps ---

    def _hand_live(name: str, conn: Any, credential: str) -> None:
        """The running plug-in takes it for its next call, and the connection searches again at
        once: a refusal earned by the old credential is not held against the new one."""
        provider = supervisor.fleet.providers[name]
        provider.set_credential(credential)
        state = supervisor.fleet.searching.get(name)
        if state is not None:
            state.backoff_s = state.retry_at = 0.0
            state.refusal = state.error = None
        accounts.forget_account(name)

    async def _vet(conn: Any, credential: str, held: list[tuple[str, str]],
                   settings: Optional[dict] = None) -> Optional[JSONResponse]:
        """The provider takes it, and it sees every instance and volume the pool holds there."""
        try:
            fresh = accounts.factory(conn.type, dict(settings if settings is not None else conn.settings))
        except Exception:  # noqa: BLE001 - settings the plug-in refuses
            return error(400, "bad_settings", "the plug-in could not be built from these settings")
        try:
            fresh.set_credential(credential)
            try:
                status = await fresh.account()
            except ProviderError as exc:
                return error(422, "credential_refused", f"the provider did not accept it: {scrub(exc, credential)}")
            except Exception as exc:  # noqa: BLE001 - a plug-in's own failure, never a 500
                return error(400, "plugin_failed", f"the plug-in failed: {type(exc).__name__} — check this connection's settings")
            if not status.credential_valid:
                return error(422, "credential_refused", "the provider did not accept it")
            if held:
                try:
                    seen = {i.instance_id for i in await fresh.list_instances(supervisor.fleet.label_prefix)}
                    if any(record.startswith("volume:") for record, _ in held) and fresh.capabilities.volumes:
                        seen |= {v.volume_id for v in await fresh.list_volumes(supervisor.fleet.label_prefix)}
                except Exception as exc:  # noqa: BLE001 - provider errors and plug-in failures alike
                    return error(409, "cannot_check_account",
                                 f"it was accepted, but the pool could not check it sees what it holds there: {scrub(exc, credential)}")
                unseen = [record for record, ident in held if ident not in seen]
                if unseen:
                    return error(409, "another_account",
                                 f"this credential does not see {len(unseen)} of the {len(held)} host(s) or volume(s) the "
                                 f"pool holds there ({', '.join(unseen[:5])}): it is another account's. Release them "
                                 "first, or use this account's credential")
        finally:
            if hasattr(fresh, "aclose"):
                await fresh.aclose()
        return None

    async def _apply_with_credential(text: str, version: str, body: dict, name: str, type_name: str,
                                     settings: dict, credential: Optional[str]) -> JSONResponse:
        """Plan first — a change that must be typed again stores nothing — then test and store
        the credential, so the connection is handed it the moment it is applied; undone if the
        apply does not happen."""
        errors = supervisor.store.validate(text)
        if errors:
            return error(400, "invalid_config", "; ".join(errors))
        plan = plan_of(text)
        if body.get("plan_only"):
            # What saving would do, for the console to show before it is saved: nothing stored.
            return JSONResponse({"changes": [c.as_dict() for c in plan]})
        refusal = [c.refused for c in plan if c.refused]
        if refusal:
            return error(409, "change_refused", "; ".join(refusal))
        loosening = [c for c in plan if c.requires_retype]
        if loosening and str(body.get("confirm")) not in {str(c.requires_retype) for c in loosening}:
            return validate_plan_apply(text, version, body)  # answers not_confirmed with the plan
        stored_before = accounts.store.get(type_name)
        if credential is not None:
            probe = accounts.factory(type_name, dict(settings))
            try:
                if not takes_credential(probe):
                    return error(409, "plugin_reads_its_own", "this plug-in reads its own credential; it cannot be typed in")
                endpoint = endpoint_of(probe, settings)
            finally:
                if hasattr(probe, "aclose"):
                    await probe.aclose()
            from ..config import ProviderConnection

            problem = await _vet(ProviderConnection(type=type_name, settings=settings), credential, [], settings)
            if problem is not None:
                return problem
            accounts.store.put(type_name, credential, endpoint)
        async with supervisor.pass_lock:  # never mid-pass: a pass may be renting there
            response = validate_plan_apply(text, version, body)
        if response.status_code != 200 and credential is not None:
            if stored_before is not None:
                accounts.store.put(type_name, stored_before.value, stored_before.endpoint)
            else:
                accounts.store.remove(type_name)
        elif response.status_code == 200:
            supervisor.events.record("provider_added", f"provider connection {name!r} ({type_name}) added from the console"
                                     + (", with a credential typed in" if credential is not None else ""),
                                     numbers={"connection": name})
        return response


def _unredacted(sent: Any, kept: Any) -> Any:
    """Settings as the screen sends them back, with every value it was shown as `[redacted]` —
    at any depth — taken from the file again, and dropped where the file has none."""
    if not isinstance(sent, dict):
        return sent
    out = {}
    for key, value in sent.items():
        was = kept.get(key) if isinstance(kept, dict) else None
        if value == "[redacted]":
            if was is not None:
                out[key] = was
            continue
        out[key] = _unredacted(value, was) if isinstance(value, dict) else value
    return out


def _reach(provider: Any) -> Optional[str]:
    """How far a provider's kept storage reaches, as its plug-in declares it (D139)."""
    capabilities = provider.capabilities
    return capabilities.reach or capabilities.volume_reach


def _reach_of_type(type_name: str) -> Optional[str]:
    try:
        capabilities = plugin_presentation(type_name).get("capabilities") or {}
    except Exception:  # noqa: BLE001 — an unknown or broken plug-in: nothing to say
        return None
    if capabilities.get("volumes"):
        return capabilities.get("volume_reach") or "machine"
    return capabilities.get("volume_reach")


async def _steps(provider: Any, label_prefix: str, credential: Optional[str], unsaved: bool) -> list[dict[str, Any]]:
    """What Test connection reports, step by step: passed, failed, a warning, or skipped."""
    steps: list[dict[str, Any]] = []

    def said(step: str, status: str, detail: str) -> None:
        steps.append({"step": step, "status": status, "detail": scrub(detail, credential)})

    try:
        account = await provider.account()
        if account.credential_valid:
            credit = f"; ${account.credit_remaining:,.2f} credit left" if account.credit_remaining is not None else ""
            said("credential", "passed", "the provider accepts the credential" + credit)
        else:
            said("credential", "failed", "the provider does not accept the credential")
    except Exception as exc:  # noqa: BLE001 - a plug-in's failure is a failed step
        said("credential", "failed", str(exc))
    if steps[-1]["status"] == "failed":
        for step in ("search", "instances", "capabilities"):
            said(step, "skipped", "needs the credential")
        return steps

    capabilities = provider.capabilities
    started = time.monotonic()
    try:
        rows = await provider.search_offers(OfferQuery(limit=1, interruptible=capabilities.interruptible, on_demand=True))
        said("search", "passed", f"one search of a single row answered in {time.monotonic() - started:.1f}s "
                                 f"({len(rows)} offer{'s' if len(rows) != 1 else ''})")
    except Exception as exc:  # noqa: BLE001 - a plug-in's failure is a failed step
        said("search", "failed", str(exc))
    try:
        found = await provider.list_instances(label_prefix)
        if found and unsaved:
            said("instances", "warning",
                 f"{len(found)} instance(s) already carry this pool's label ({label_prefix}) with no record here: "
                 "once added, the pool's sweep destroys them as strays. Use an account no other pool of this name uses")
        else:
            said("instances", "passed", f"{len(found)} of this pool's hosts on this account" if found
                 else "no hosts from this pool on this account")
    except Exception as exc:  # noqa: BLE001 - a plug-in's failure is a failed step
        said("instances", "failed", str(exc))
    words = {"interruptible": "interruptible", "parkable": "park", "same_machine_rebid": "raise a bid in place",
             "self_terminate": "dead-man timer", "reports_charges": "reports charges", "price_history": "price history",
             "direct_port_mapping": "direct port", "reports_instance_logs": "boot logs", "volumes": "volumes",
             "copies": "copies between hosts", "interruption_notice": "interruption warning"}
    can = [words.get(name, name.replace("_", " ")) for name, on in vars(capabilities).items() if on]
    said("capabilities", "passed", ", ".join(can) or "none declared")
    return steps
