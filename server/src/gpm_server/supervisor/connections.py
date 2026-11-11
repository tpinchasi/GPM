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
        #: Why a connection does not run as the file says — a plug-in that could not be built, a name
        #: another provider's — by connection, in words; not rented through meanwhile.
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
        was set, and what in the file could not be applied."""
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
        """Why a file may not be applied, or None: it changes a running connection's address — fixed
        once the account is added (D175) — or points one that holds hosts or volumes at another
        credential source, another account maybe, where they would read as gone while they bill
        (D136). Checked however the file arrives: the plan, a raw save, a rollback, an edit the
        supervisor follows."""
        fleet = self.supervisor.fleet
        if fleet is None or candidate.rented is None:
            return None
        for name, built in self.built.items():
            if name not in fleet.providers:
                continue
            after = next((conn for conn in candidate.rented.providers.values() if conn.type == built.type), None)
            running = fleet.providers[name]
            if after is not None and endpoint_of(running, after.settings) != endpoint_of(running, built.settings):
                # Where an account's requests go is part of which account it is: fixed once it is
                # added, held or not (D175). Another address is another account, added as one.
                keys = ", ".join(getattr(running, "endpoint_settings", ()) or ())
                return (f"provider connection {name!r}'s address ({keys}) is fixed once the account is added: to use "
                        "another, remove the account and add it again (D175)")
            if not fleet.held_at(name):
                continue
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
        """Bring the running plug-ins toward the file, all at once (D135, D174): a **new**
        connection is built and searched, one turned on or off is searched or not, one with new
        settings or credential source is built again and handed its credential by the rules in
        force (D134), and a renamed one carries on under its new name. What would move a
        connection that holds something to another account never gets here: the plan and the file
        refuse it (D136). With `drop` — under the pass's lock, never during a pass — a connection
        gone from the file is dropped once the pool holds nothing there."""
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
                    self._rebuild(name, name, conn)
                continue
            # One provider, one plug-in (D133): a running one of this provider under another name is
            # this connection renamed, or one still releasing what it holds — never a second.
            twin = next((n for n, kind in running_type.items() if n != name and kind == conn.type), None)
            if twin is not None:
                if twin in become:
                    self.pending[name] = f"{twin!r} is already this provider's connection"
                elif self._rebuild(twin, name, conn):
                    running_type.pop(twin, None)
                    running_type[name] = conn.type
                else:
                    renamed_from.add(twin)
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

    def _rebuild(self, was: str, name: str, conn: ProviderConnection) -> bool:
        """A running connection with new settings or credential source, or renamed from `was`,
        or both, while the pool runs (D174). Its plug-in is built again where what it was built
        from changed, and handed its credential by the rules in force — one from the supervisor's
        environment only to the plug-in's own endpoint, one typed in only to the endpoint it was
        saved for (D130, D134) — so the running plug-in always follows the file. A rename keeps
        the running plug-in and everything counted under it; records under the old name are its
        provider's, and resolve to the new one (D133). A plug-in that cannot be built leaves the
        running one as it was, not rented through, and says why. True once done."""
        fleet = self.supervisor.fleet
        built = self.built[was]
        running = fleet.providers[was]
        provider = running
        if (built.settings, built.credential_env) != (conn.settings, conn.credential_env):
            try:
                provider = self.factory(conn.type, dict(conn.settings))
            except Exception as exc:  # noqa: BLE001 - one plug-in's failure stays its own (D129)
                self.pending[name] = f"its new settings could not be set up: {exc}"
                self.supervisor.events.record("provider_refused", f"provider connection {name!r} could not be set up "
                                              f"with its new settings: {exc}; it is not rented through until they are "
                                              "put right", numbers={"connection": name})
                return False
            if not takes_credential(provider) and conn.settings:
                # A version-1 plug-in reads its own credential: given settings from a file the admin
                # key writes, it could send it anywhere — built only by whoever starts the supervisor.
                self.retire(provider)
                self.pending[name] = "its plug-in reads its own credential: new settings take effect when the supervisor restarts"
                return False
        if was != name:
            try:
                fleet.rename_connection(was, name)
            except ConnectionNameReused as exc:
                if provider is not running:
                    self.retire(provider)
                self.pending[name] = str(exc)
                self.supervisor.events.record("provider_refused", str(exc), numbers={"connection": name})
                return False
            self.built.pop(was, None)
            if was in self._accounts:
                self._accounts[name] = self._accounts.pop(was)
        fleet.providers[name] = provider
        self.install(name, conn, provider)
        if provider is not running:
            self.retire(running)
            self.forget_account(name)
        said = ([f"renamed from {was!r}"] if was != name else []) + (["built again with its new settings"]
                                                                     if provider is not running else [])
        self.supervisor.events.record("provider_changed", f"provider connection {name!r} {' and '.join(said)}, at once",
                                      numbers={"connection": name, "was": was})
        return True

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
