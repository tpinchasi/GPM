"""What is left in each provider account, and what the pool does when it runs out (D188).

docs/spec/providers.md §12. Every account the pool spends through — a hosted API, a rented
account — has a balance that is one of three things:

- **provider-reported**: read from the provider's own API where it states one (a hosted preset's
  `balance_url`, a rented plug-in declaring `capabilities.reports_balance`), by the supervisor, every
  15 minutes and when an operator asks — never by the router, never on a request's way;
- **typed**: what the operator typed on the account's card, as of that moment;
- **unknown**.

A typed balance stands until the operator clears it: while one is set, the provider's reading is
shown beside it and not acted on — a provider answering a wrong $0 cannot keep an account out
against the operator's word. The one in force is counted down by the pool's own spend records since its
moment — hosted answers' costs from the request log, rented hosts' and volumes' estimates from the
spend ledger — so the card says *about $63.20 left — $64.00 typed on 8 Nov, less $0.80 spent
here since*. It is an estimate: what other users of the same key or account spend is not counted.

At $0 a hosted API is taken out of routing through the out-of-money state the router already has
(D164) — the supervisor writes it in the host table, the router reads it; the two never call
each other — until the operator says the money is back or types a new balance. A rented account
rents nothing new once its balance would not cover an hour of what the pool would run there
(`Fleet.balance_refusal`); machines already running are never stopped for it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import TYPE_CHECKING, Any, Optional

from .. import hosted_balances, hosted_credentials
from ..hosted_presets import preset_for

if TYPE_CHECKING:
    from .service import Supervisor

log = logging.getLogger("gpm.balances")

#: How often a provider is asked what is left: a slow schedule, well inside every provider's limits.
READ_EVERY_S = 900.0
#: How often the countdown is worked out from the spend records.
COUNT_EVERY_S = 30.0
#: An operator's "read now" is not sent to one provider more often than this.
ASK_AGAIN_AFTER_S = 60.0
#: The most a typed balance may be.
MOST_TYPED = 1_000_000.0
KINDS = ("hosted", "rented")


def _day(ts: float) -> str:
    return time.strftime("%d %b %H:%M UTC", time.gmtime(ts))


class BalanceStore:
    """The `provider_balances` table: written by the supervisor alone."""

    def __init__(self, db: Any):
        self.db = db

    def all(self) -> dict[str, dict[str, Any]]:
        return {r["account"]: dict(r) for r in self.db.query("SELECT * FROM provider_balances")}

    def _upsert(self, account: str, **fields: Any) -> None:
        now = time.time()
        names = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        updates = ", ".join(f"{k} = excluded.{k}" for k in fields)
        self.db.execute(
            f"INSERT INTO provider_balances (account, {names}, updated_at) VALUES (?, {marks}, ?) "
            f"ON CONFLICT(account) DO UPDATE SET {updates}, updated_at = excluded.updated_at",
            (account, *fields.values(), now))

    def type_in(self, account: str, usd: Optional[float]) -> None:
        self._upsert(account, typed_usd=usd, typed_at=time.time() if usd is not None else None)

    def asked(self, account: str, usd: Optional[float], detail: Optional[str]) -> None:
        """A provider was asked: a number replaces the last one; a failure keeps it and says why."""
        now = time.time()
        if usd is not None:
            self._upsert(account, read_usd=usd, read_at=now, read_detail=detail, asked_at=now)
        else:
            self._upsert(account, read_detail=detail, asked_at=now)

    def forget(self, account: str) -> None:
        """The account is gone, or is another account under the same name: nothing is known."""
        self.db.execute("DELETE FROM provider_balances WHERE account = ?", (account,))

    def stamp(self, account: str, fingerprint: str) -> None:
        self._upsert(account, fingerprint=fingerprint)

    def forget_read(self, account: str) -> None:
        """The last reading is no longer believed: asked again at once."""
        self._upsert(account, read_usd=None, read_at=None, asked_at=None,
                     read_detail="asked again: the operator said its money is back")


class Balances:
    def __init__(self, supervisor: "Supervisor"):
        self.supervisor = supervisor
        self.store = BalanceStore(supervisor.db)
        #: Each account's view, as last worked out.
        self.views: dict[str, dict[str, Any]] = {}
        self._counted_at = 0.0
        self._reading: Optional[asyncio.Task] = None

    # --- which accounts, and whether each can be read ---

    def accounts(self) -> dict[str, tuple[str, str]]:
        """Every account the pool spends through now: {account: (kind, id)}."""
        out: dict[str, tuple[str, str]] = {}
        for host in self.supervisor.config.hosts:
            if host.kind == "hosted":
                out[f"hosted:{host.id}"] = ("hosted", host.id)
        fleet = self.supervisor.fleet
        if fleet is not None:
            for name in list(fleet.providers):  # copied: this may run on a worker thread
                out[f"rented:{name}"] = ("rented", name)
        return out

    def fingerprint(self, account: str) -> Optional[str]:
        """What makes this the same account: for a hosted API its address, format and key's source
        and moment; for a rented one its plug-in, settings and credential's source and moment. None
        where it cannot be said (the account is not running): left as it is."""
        kind, _, ident = account.partition(":")
        try:
            if kind == "hosted":
                host = next((h for h in self.supervisor.config.hosts if h.id == ident and h.kind == "hosted"), None)
                if host is None:
                    return None
                key = hosted_credentials.describe(self.supervisor.config, host)
                return json.dumps([host.transport.base_url, host.engine, key.get("source"), key.get("variable"),
                                   key.get("set_at")])
            accounts = self.supervisor.accounts
            built = accounts.built.get(ident)
            fleet = self.supervisor.fleet
            provider = fleet.providers.get(ident) if fleet is not None else None
            if built is None or provider is None:
                return None
            said = accounts.describe(ident, provider)
            return json.dumps([built.type, built.credential_env, built.settings, said.get("source"),
                               said.get("variable"), said.get("set_at")], sort_keys=True, default=str)
        except Exception:  # noqa: BLE001 - a credential store that cannot be read says nothing here
            return None

    def _hosted_reader(self, host_id: str) -> Optional[Any]:
        """The preset that says where this host's provider answers its balance, or None. Only a
        host at exactly a preset's address has one, so the address is never the file's to choose."""
        host = next((h for h in self.supervisor.config.hosts if h.id == host_id and h.kind == "hosted"), None)
        if host is None or host.disabled:
            return None
        preset = preset_for(host.transport.base_url or "", host.engine or "")
        if preset is None or not preset.balance_url or preset.balance_format not in hosted_balances.READERS:
            return None
        return preset

    def can_read(self, account: str) -> bool:
        kind, _, ident = account.partition(":")
        if kind == "hosted":
            return self._hosted_reader(ident) is not None
        fleet = self.supervisor.fleet
        provider = fleet.providers.get(ident) if fleet is not None else None
        return bool(provider is not None and getattr(getattr(provider, "capabilities", None), "reports_balance", False))

    # --- reading from the provider ---

    async def _read(self, account: str) -> None:
        kind, _, ident = account.partition(":")
        try:
            if kind == "hosted":
                usd, detail = await self._read_hosted(ident)
            else:
                usd, detail = await self._read_rented(ident)
        except Exception as exc:  # noqa: BLE001 - one provider's failure is said, never raised
            usd, detail = None, f"could not be read: {type(exc).__name__}"
        if usd is not None and (not math.isfinite(usd) or abs(usd) > hosted_balances.MOST_BELIEVED):
            usd, detail = None, "the provider answered a balance that cannot be right"
        await asyncio.to_thread(self.store.asked, account, usd, (detail or "")[:300] or None)

    async def _read_hosted(self, host_id: str) -> tuple[Optional[float], Optional[str]]:
        preset = self._hosted_reader(host_id)
        host = self.supervisor.hosts.get(host_id)
        if preset is None or host is None:
            return None, "this provider's API says no balance"
        reading = await hosted_balances.read(host.client, host.config.transport.base_url or "", preset.balance_url,
                                             preset.balance_format, preset.balance_note)
        return reading.usd, reading.detail

    async def _read_rented(self, name: str) -> tuple[Optional[float], Optional[str]]:
        said = await self.supervisor.accounts.account(name, fresh=True)
        credit = said.get("credit")
        if said.get("state") == "valid" and isinstance(credit, (int, float)):
            return float(credit), None
        return None, str(said.get("detail") or "the provider gave no balance")

    def _due(self) -> list[str]:
        rows = self.store.all()
        now = time.time()
        return [a for a in self.accounts() if self.can_read(a)
                and now - float((rows.get(a) or {}).get("asked_at") or 0.0) >= READ_EVERY_S]

    async def _read_all(self, accounts: list[str]) -> None:
        await asyncio.gather(*(self._read(a) for a in accounts))
        self._counted_at = 0.0  # count again at the next pass, with what was read

    async def read_now(self, account: Optional[str] = None) -> None:
        """An operator asked: each readable account, or one, not asked in the last minute."""
        rows = await asyncio.to_thread(self.store.all)
        now = time.time()
        wanted = [a for a in ([account] if account else list(self.accounts())) if self.can_read(a)
                  and now - float((rows.get(a) or {}).get("asked_at") or 0.0) >= ASK_AGAIN_AFTER_S]
        if wanted:
            await self._read_all(wanted)
        await self.count()

    # --- what the operator types ---

    async def type_in(self, account: str, usd: Optional[float]) -> None:
        await asyncio.to_thread(self.store.type_in, account, usd)
        kind, _, ident = account.partition(":")
        if usd is not None:
            self.supervisor.events.record(
                "balance_typed", f"{account}: the operator typed ${usd:,.2f} as what is left there, as of now; "
                "the pool counts it down by what it spends there", host_id=ident if kind == "hosted" else None,
                numbers={"account": account, "usd": usd})
            if kind == "hosted" and usd > 0:
                # A topped-up balance is the operator's word that the money is back (D164).
                await asyncio.to_thread(self.supervisor.table.reset_funds, ident)
        else:
            self.supervisor.events.record("balance_cleared", f"{account}: the typed balance was cleared",
                                          host_id=ident if kind == "hosted" else None, numbers={"account": account})
        await self.count()

    def funds_restored(self, host_id: str) -> Optional[str]:
        """The operator says a hosted API's money is back: a balance at $0 that would take it out
        again is no longer believed — a typed one is cleared, a provider's asked again."""
        account = f"hosted:{host_id}"
        view = self.views.get(account) or {}
        said = None
        if view.get("estimate") is not None and view["estimate"] <= 0:
            if view.get("source") == "typed":
                self.store.type_in(account, None)
                said = "its typed balance, at $0, was cleared: type the new one to count it down again"
                self.supervisor.events.record("balance_cleared", f"{account}: the typed balance, at $0, was cleared "
                                              "as the operator said its money is back", host_id=host_id,
                                              numbers={"account": account})
            if view.get("read") and view["read"].get("usd") is not None:
                # A provider's figure would take over once the typed one is gone: asked again first.
                self.store.forget_read(account)
                said = (f"{said}; " if said else "") + "its provider's balance is asked again"
        self.supervisor.table.mark_balance(host_id, None)
        self._counted_at = 0.0
        return said

    # --- the countdown ---

    def _hosted_spent_since(self, host_id: str, since: float) -> float:
        rows = self.supervisor.db.query(
            "SELECT COALESCE(SUM(cost_usd), 0) AS spent FROM request_log WHERE host_id = ? AND ts >= ? "
            "AND cost_usd IS NOT NULL AND cost_usd > 0", (host_id, since))
        return float(rows[0]["spent"] or 0.0)

    def _rented_spent_since(self, name: str, since: float) -> float:
        """The ledger's estimates are each host's (and volume's) running total for its lease: what
        it cost since is its newest less its newest before."""
        fleet = self.supervisor.fleet
        # Every name its records were written under: renamed, a connection's records stay its
        # provider's (D133), and a record from before connections is the legacy one's.
        written = [r["connection"] for r in self.supervisor.db.query(
            "SELECT DISTINCT connection FROM spend WHERE ts >= ?", (since,))]
        legacy = fleet.legacy_connection if fleet is not None else name
        names = sorted({n for n in written if n is not None and (n == name or (fleet is not None and fleet.resolve(n) == name))}
                       | {name})
        with_null = 1 if None in written and (legacy == name or (fleet is not None and fleet.resolve(legacy) == name)) else 0
        marks = ", ".join("?" for _ in names)
        where = f"(connection IN ({marks}) OR (connection IS NULL AND ? = 1)) AND source = 'estimate'"
        after = self.supervisor.db.query(
            f"SELECT lease_id, host_id, MAX(amount) AS amount FROM spend WHERE {where} AND ts >= ? "
            "GROUP BY lease_id, host_id", (*names, with_null, since))
        if not after:
            return 0.0
        before = {(r["lease_id"], r["host_id"]): float(r["amount"] or 0.0) for r in self.supervisor.db.query(
            f"SELECT lease_id, host_id, MAX(amount) AS amount FROM spend WHERE {where} AND ts < ? "
            "GROUP BY lease_id, host_id", (*names, with_null, since))}
        return sum(max(0.0, float(r["amount"] or 0.0) - before.get((r["lease_id"], r["host_id"]), 0.0)) for r in after)

    def _view(self, account: str, kind: str, ident: str, row: dict[str, Any]) -> dict[str, Any]:
        typed = row.get("typed_usd")
        read = row.get("read_usd")
        typed_at, read_at = row.get("typed_at") or 0.0, row.get("read_at") or 0.0
        if typed is not None:
            # The operator's word stands until they clear it; a provider's reading is shown beside it.
            source, amount, as_of = "typed", float(typed), typed_at
        elif read is not None:
            source, amount, as_of = "provider", float(read), read_at
        else:
            source, amount, as_of = "unknown", None, None
        spent = estimate = None
        if amount is not None:
            spent = (self._hosted_spent_since(ident, as_of) if kind == "hosted"
                     else self._rented_spent_since(ident, as_of))
            estimate = round(amount - spent, 4)
        view = {
            "account": account, "kind": kind, "id": ident, "source": source,
            "amount": amount, "as_of": as_of, "spent_since": round(spent, 4) if spent is not None else None,
            "estimate": estimate, "can_read": self.can_read(account),
            "typed": {"usd": typed, "at": row.get("typed_at")} if typed is not None else None,
            "read": {"usd": read, "at": row.get("read_at"), "asked_at": row.get("asked_at"),
                     "detail": row.get("read_detail")} if (read is not None or row.get("asked_at")) else None,
            "summary": self.summary(source, amount, as_of, spent, estimate),
        }
        if kind == "hosted":
            preset = self._hosted_reader(ident)
            view["note"] = preset.balance_note if preset is not None and preset.balance_note else None
        return view

    @staticmethod
    def summary(source: str, amount: Optional[float], as_of: Optional[float], spent: Optional[float],
                estimate: Optional[float]) -> str:
        if source == "unknown" or amount is None or as_of is None:
            return "not known: its provider's API says none, or none has been typed"
        how = "typed" if source == "typed" else "reported by the provider"
        return (f"about ${estimate:,.2f} left — ${amount:,.2f} {how} on {_day(as_of)}, "
                f"less ${spent or 0.0:,.2f} spent here since")

    def _count(self) -> dict[str, dict[str, Any]]:
        rows = self.store.all()
        for account in self.accounts():
            now = self.fingerprint(account)
            if now is None:
                continue
            was = (rows.get(account) or {}).get("fingerprint")
            if account in rows and was is None:
                self.store.stamp(account, now)  # written before it was first counted: this account's
                rows[account]["fingerprint"] = now
                continue
            if account in rows and was != now:
                # Another account under the same name — moved, re-keyed, removed and added again:
                # nothing said of the old one is believed of it.
                self.store.forget(account)
                rows.pop(account, None)
            if account not in rows:
                self.store.stamp(account, now)
                rows[account] = {"fingerprint": now}
        return {account: self._view(account, kind, ident, rows.get(account, {}))
                for account, (kind, ident) in self.accounts().items()}

    async def count(self) -> dict[str, dict[str, Any]]:
        """Work the countdown out now, and act on it."""
        views = await asyncio.to_thread(self._count)
        self.views = views
        self._counted_at = time.monotonic()
        self._act(views)
        return views

    def _act(self, views: dict[str, dict[str, Any]]) -> None:
        marked = {r.host_id: r.balance_out for r in self.supervisor.table.all()}
        fleet = self.supervisor.fleet
        if fleet is not None:
            fleet.balances = {v["id"]: v["estimate"] for v in views.values()
                              if v["kind"] == "rented" and v["estimate"] is not None}
        for view in views.values():
            if view["kind"] != "hosted" or view["id"] not in marked:
                continue
            host_id = view["id"]
            out = view["estimate"] is not None and view["estimate"] <= 0
            was = marked[host_id] is not None
            if out and not was:
                words = f"its balance reached $0: {view['summary']}"
                self.supervisor.table.mark_balance(host_id, words)
                self.supervisor.events.record(
                    "balance_exhausted", f"{host_id}: {words}; taken out of routing until it is topped up — "
                    "Money is back, or a new balance typed", host_id=host_id,
                    numbers={"estimate": view["estimate"], "source": view["source"]})
            elif not out and was:
                self.supervisor.table.mark_balance(host_id, None)
                self.supervisor.events.record(
                    "balance_restored", f"{host_id}: its balance is above $0 again ({view['summary']}); routed to again",
                    host_id=host_id, numbers={"estimate": view["estimate"], "source": view["source"]})

    # --- each pass ---

    async def pass_once(self) -> None:
        """Start the providers' reads when they are due, in the background — a slow provider never
        holds a pass — and count down every half minute."""
        if self._reading is None or self._reading.done():
            due = await asyncio.to_thread(self._due)
            if due:
                self._reading = asyncio.create_task(self._read_all(due), name="gpm:balances")
        if time.monotonic() - self._counted_at >= COUNT_EVERY_S:
            await self.count()

    async def fresh(self) -> dict[str, dict[str, Any]]:
        """The figures last worked out, counted again only when older than half a minute — a page
        opened often costs no more than the pass does."""
        if not self.views or time.monotonic() - self._counted_at >= COUNT_EVERY_S:
            return await self.count()
        return self.views

    def forget(self, account: str) -> None:
        """An account removed: what was read or typed of it goes with it."""
        self.store.forget(account)
        self.views.pop(account, None)
        self._counted_at = 0.0

    def view_of(self, account: str) -> dict[str, Any]:
        """The last view of one account, for a card: never a provider call."""
        found = self.views.get(account)
        if found is not None:
            return found
        kind, _, ident = account.partition(":")
        return {"account": account, "kind": kind, "id": ident, "source": "unknown", "amount": None, "as_of": None,
                "spent_since": None, "estimate": None, "can_read": self.can_read(account), "typed": None,
                "read": None, "summary": self.summary("unknown", None, None, None, None)}

    async def aclose(self) -> None:
        if self._reading is not None:
            self._reading.cancel()
