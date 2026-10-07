"""Provider accounts while the supervisor runs: each connection's credential, and connections
added, changed or removed without a restart.

docs/spec/providers.md §1, §5. A credential is resolved in one order: the connection's own
`credential_env`, when it names one, always; else one typed into the console, if it was saved
for the endpoint the connection now uses; else the plug-in's own variable. The plug-in is
handed it (D134) — it reads nothing itself.

What is resolved is said by its *source* only. The value goes to the plug-in and nowhere else.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..config import PoolConfig, ProviderConnection
from ..credentials import CredentialStore, endpoint_of
from ..providers import (
    ProviderAuthError,
    ProviderError,
    get_provider,
    installed_plugins,
    presentation,
    takes_credential,
)
from .renting import ConnectionNameReused

if TYPE_CHECKING:
    from .service import Supervisor

log = logging.getLogger("gpm.providers")

#: How long a connection's account answer is shown before it is asked again. The account call
#: is free, but every provider limits how often it is made.
ACCOUNT_FRESH_S = 60.0
#: How long a replaced plug-in is kept open, so a call already using it finishes.
RETIRE_AFTER_S = 60.0


class ProviderAccounts:
    def __init__(self, supervisor: "Supervisor"):
        self.supervisor = supervisor
        self.store = CredentialStore(supervisor.config.credentials_dir())
        #: How a plug-in is built. Tests hand in one that returns their own market's client.
        self.factory: Callable[[str, dict], Any] = get_provider
        self._accounts: dict[str, tuple[float, dict[str, Any]]] = {}
        #: Where a stored credential was cleared because the endpoint it was saved for changed,
        #: by provider: said on the Providers screen until a new one is typed in.
        self.cleared: dict[str, float] = {}
        #: How each running plug-in was built — its provider, settings and credential source.
        #: **Every credential decision reads this, never the file**: the file may already say
        #: what the running plug-in will only do after a restart.
        self.built: dict[str, ProviderConnection] = {}
        #: What the file says that waits for a restart, by connection, in words.
        self.pending: dict[str, str] = {}
        #: Removed connections still holding something, already said.
        self._said_kept: set[str] = set()

    # --- resolving ---

    def _environment(self, conn: ProviderConnection, provider: Any) -> Optional[str]:
        return conn.credential_env or getattr(provider, "credential_env", None)

    @staticmethod
    def _may_use_environment(name: str, conn: ProviderConnection, provider: Any) -> bool:
        """A credential from the supervisor's environment goes **only to the plug-in's own default
        endpoint** — never to one a setting names. The file is the admin key's to write, live or
        before a restart, so an endpoint it names may be anyone's: a connection pointed there
        takes a credential typed in, bound to that endpoint, or none (D130, D134)."""
        return all(value is None for value in endpoint_of(provider, conn.settings).values())

    def resolve(self, conn: ProviderConnection, provider: Any,
                may_use_environment: bool = True) -> tuple[str, Optional[str]]:
        """(source, value): `environment`, `stored`, `missing`, `withheld` (one in the environment
        that this connection may not take: its endpoint is not the plug-in's default), or `plugin` for a version-1 plug-in
        that reads its own. A stored credential saved for another endpoint is removed here,
        never sent (D130)."""
        if not takes_credential(provider):
            return "plugin", None
        if not conn.credential_env:
            stored = self.store.get(conn.type)
            if stored is not None:
                if stored.bound_to(endpoint_of(provider, conn.settings)):
                    return "stored", stored.value
                self.store.remove(conn.type)
                self.cleared[conn.type] = time.time()
                self.supervisor.events.record(
                    "credential_cleared",
                    f"the {conn.type} credential typed into the console was saved for another endpoint, so it was "
                    "removed rather than sent there; type it in again",
                    numbers={"provider": conn.type},
                )
        variable = self._environment(conn, provider)
        value = os.environ.get(variable) if variable else None
        if not value:
            return "missing", None
        return ("environment", value) if may_use_environment else ("withheld", None)

    def resolve_environment(self, conn: ProviderConnection, provider: Any) -> Optional[str]:
        """For a connection not saved yet, with no settings: the plug-in's own variable only."""
        name = self._environment(conn, provider)
        return os.environ.get(name) if name else None

    def install(self, name: str, conn: ProviderConnection, provider: Any, *, at_start: bool = False) -> str:
        """Record how this plug-in was built, and hand it its credential."""
        self.built[name] = conn
        return self.rehand(name, provider)

    def rehand(self, name: str, provider: Any) -> str:
        """Hand a running plug-in its credential again — after one typed in was removed, say."""
        conn = self.built[name]
        source, value = self.resolve(conn, provider, self._may_use_environment(name, conn, provider))
        if takes_credential(provider):
            provider.set_credential(value)
        if source == "withheld":
            self.supervisor.events.record(
                "credential_withheld",
                f"provider connection {name!r} names its own endpoint, so no credential from the supervisor's "
                "environment is sent there; type one in on the Providers screen", numbers={"connection": name})
        return source

    def describe(self, name: str, provider: Any) -> dict[str, Any]:
        """The running plug-in's credential state, never its value: where it comes from, when it
        was set, and what in the file waits for a restart."""
        conn = self.built[name]
        said: dict[str, Any]
        if not takes_credential(provider):
            said = {"source": "plugin", "can_type": False,
                    "detail": "this plug-in reads its own credential, from its own setting or variable"}
        elif conn.credential_env:
            present = bool(os.environ.get(conn.credential_env))
            usable = self._may_use_environment(name, conn, provider)
            said = {"source": ("environment" if usable else "withheld") if present else "missing",
                    "variable": conn.credential_env, "can_type": False,
                    "detail": f"always taken from {conn.credential_env} in the supervisor's environment (credential_env)"}
        else:
            stored = self.store.get(conn.type)
            if stored is not None and stored.bound_to(endpoint_of(provider, conn.settings)):
                said = {"source": "stored", "set_at": stored.set_at, "can_type": True}
            else:
                variable = self._environment(conn, provider)
                present = bool(variable and os.environ.get(variable))
                usable = self._may_use_environment(name, conn, provider)
                said = {"source": ("environment" if usable else "withheld") if present else "missing",
                        "variable": variable, "can_type": True, "cleared_at": self.cleared.get(conn.type)}
        if name in self.pending:
            said["pending"] = self.pending[name]
        return said

    def refuse_moves(self, candidate: PoolConfig) -> Optional[str]:
        """Why a file may not be applied, or None: it points a connection that holds hosts or
        volumes at another endpoint or credential source — another account, maybe, where they
        would read as gone while they bill (D136). Checked however the file arrives: the plan,
        a raw save, a rollback, an edit the supervisor follows."""
        fleet = self.supervisor.fleet
        if fleet is None or candidate.rented is None:
            return None
        for name, built in self.built.items():
            if name not in fleet.providers or not fleet.held_at(name):
                continue
            after = next((conn for conn in candidate.rented.providers.values() if conn.type == built.type), None)
            if after is not None and (after.settings, after.credential_env) != (built.settings, built.credential_env):
                return (f"provider connection {name!r} holds hosts or volumes, and this changes where its requests go or "
                        "which credential it uses — it could become another account, where they would read as gone while "
                        "they bill. Release them first (D136)")
        return None

    # --- the account ---

    async def account(self, name: str, fresh: bool = False) -> dict[str, Any]:
        """Whether the provider takes the credential, and the credit left — from the last minute
        unless asked fresh."""
        fleet = self.supervisor.fleet
        if fleet is None or name not in fleet.providers:
            return {"state": "unknown"}
        if not fresh and name in self._accounts and time.time() - self._accounts[name][0] < ACCOUNT_FRESH_S:
            return self._accounts[name][1]
        try:
            status = await fleet.providers[name].account()
            said = {"state": "valid" if status.credential_valid else "refused",
                    "credit": status.credit_remaining}
        except ProviderAuthError as exc:
            said = {"state": "refused", "detail": str(exc)}
        except ProviderError as exc:
            said = {"state": "unreachable", "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a plug-in's own failure is reported, never a 500
            said = {"state": "unreachable", "detail": f"the plug-in failed: {type(exc).__name__}"}
        said["checked_at"] = time.time()
        self._accounts[name] = (time.time(), said)
        return said

    def forget_account(self, name: str) -> None:
        self._accounts.pop(name, None)

    # --- connections changing while it runs (D135) ---

    def sync(self, old: PoolConfig, new: PoolConfig, *, drop: bool = False) -> None:
        """Bring the running plug-ins toward the file. A **new** connection is built and searched
        at once, and one turned on or off is searched or not. A running connection is never
        rebuilt: a change to its settings or credential source, or a rename, waits for a restart,
        and it is not rented through meanwhile. With `drop` — under the pass's lock, never during
        a pass — a connection gone from the file is dropped once the pool holds nothing there."""
        fleet = self.supervisor.fleet
        if fleet is None or new.rented is None:
            return
        become = new.rented.providers
        running_type = {n: self.built[n].type for n in fleet.providers if n in self.built}
        self.pending = {}
        renamed_from: set[str] = set()
        for name, conn in become.items():
            built = self.built.get(name)
            if name in fleet.providers and built is not None:
                if built.type != conn.type:
                    self.pending[name] = (f"names {built.type!r}, which is still running under it: give {conn.type!r} "
                                          "another name (D133)")
                elif (built.settings, built.credential_env) != (conn.settings, conn.credential_env):
                    self.pending[name] = "its new settings or credential source take effect when the supervisor restarts"
                continue
            # One provider, one plug-in (D133): a running one of this provider under another name is
            # this connection renamed, or one still releasing what it holds — never a second.
            twin = next((n for n, kind in running_type.items() if n != name and kind == conn.type), None)
            if twin is not None:
                if twin not in become:
                    renamed_from.add(twin)
                self.pending[name] = (f"renamed from {twin!r}: takes effect when the supervisor restarts" if twin not in become
                                      else f"{twin!r} is already this provider's connection")
                continue
            try:
                provider = self.factory(conn.type, dict(conn.settings))
            except Exception as exc:  # noqa: BLE001 - one plug-in's failure stays its own (D129)
                self.pending[name] = f"could not be set up: {exc}"
                self.supervisor.events.record("provider_refused", f"provider connection {name!r} could not be set up: {exc}",
                                              numbers={"connection": name})
                continue
            if not takes_credential(provider) and conn.settings:
                # A version-1 plug-in reads its own credential: with settings named while running
                # it could be pointed anywhere, so it is built at a restart.
                self.retire(provider)
                self.pending[name] = "its plug-in reads its own credential: it starts when the supervisor restarts"
                continue
            try:
                fleet.add_connection(name, conn.type, provider)
            except ConnectionNameReused as exc:
                self.retire(provider)
                self.pending[name] = str(exc)
                self.supervisor.events.record("provider_refused", str(exc), numbers={"connection": name})
                continue
            self.install(name, conn, provider)
            running_type[name] = conn.type
            self.forget_account(name)
            self.supervisor.events.record(
                "provider_connected",
                f"provider connection {name!r} ({conn.type}) added"
                + (": searched from the next pass" if conn.enabled else ": not searched until it is turned on"),
                numbers={"connection": name})
        fleet.paused = {name for name in self.pending if name in fleet.providers}
        for name in [n for n in fleet.providers if n not in become]:
            held = fleet.held_at(name)
            if name in renamed_from or held or not drop or len(fleet.providers) == 1:
                # Kept: still holding, or a rename waiting, or not now — and never the last one,
                # which the pool's default needs.
                if held and name not in self._said_kept:
                    self._said_kept.add(name)
                    self.supervisor.events.record(
                        "provider_kept",
                        f"provider connection {name!r} left the file while the pool holds {len(held)} host(s) or "
                        "volume(s) there: they are still watched and released as usual; nothing new is rented there",
                        numbers={"connection": name, "held": [h for h, _ in held]})
                continue
            self._said_kept.discard(name)
            self.retire(fleet.providers.pop(name))
            fleet.searching.pop(name, None)
            self.built.pop(name, None)
            self.forget_account(name)
        first = new.rented.connection_name
        target = first if first in fleet.providers else (
            fleet.connection if fleet.connection in fleet.providers else next(iter(fleet.providers), None))
        if target is not None and fleet.connection != target:
            fleet.connection = target
            fleet.events._connection = target

    def retire(self, provider: Any) -> None:
        """Closed once whatever was using it has finished."""
        close = getattr(provider, "aclose", None)
        if close is None:
            return
        try:
            asyncio.get_running_loop().call_later(RETIRE_AFTER_S, lambda: asyncio.ensure_future(close()))
        except RuntimeError:
            pass

    # --- what the console offers ---

    def plugins(self) -> list[dict[str, Any]]:
        """Installed plug-ins, another package's unloaded (T14): {type, package, version, loaded, …}."""
        return [{"type": name, "display_name": name, "offered": True, **facts}
                for name, facts in installed_plugins().items()]

    def presentation_of(self, type_name: str, provider: Any) -> dict[str, Any]:
        return presentation(provider, type_name) if provider is not None else {"type": type_name}


def scrub(text: Any, secret: Optional[str]) -> str:
    """A provider's message with the credential taken out, should it ever echo one."""
    said = str(text)
    if secret and len(secret) >= 4:
        said = said.replace(secret, "[credential]")
    return said
