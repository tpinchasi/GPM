"""The control API's provider balances (D188): read them, ask the providers now, type one in.

docs/spec/console-and-control-api.md §4. Admin key only, as every control route. Nothing here
answers with anything a provider sent but the number and the pool's own words; nothing here spends.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .balances import KINDS, MOST_TYPED, READ_EVERY_S
from .providers_api import _body

if TYPE_CHECKING:
    from .service import Supervisor


def register(app: FastAPI, supervisor: "Supervisor", *, error: Callable[..., JSONResponse]) -> None:
    balances = supervisor.balances

    def known(kind: str, ident: str) -> Optional[JSONResponse]:
        if kind not in KINDS:
            return error(404, "unknown_kind", "a balance is a hosted API's (hosted) or a rented account's (rented)")
        if f"{kind}:{ident}" not in balances.accounts():
            return error(404, "unknown_account", f"no {'hosted API' if kind == 'hosted' else 'provider account'} {ident!r}")
        return None

    def listing(views: dict) -> dict:
        return {"balances": [views[a] for a in balances.accounts() if a in views], "read_every_s": READ_EVERY_S}

    @app.get("/pool/balances")
    async def list_balances() -> JSONResponse:
        """Every account's balance: its source (provider-reported, typed, unknown), as of when,
        what the pool has spent there since, and so about what is left."""
        return JSONResponse(listing(await balances.count()))

    @app.post("/pool/balances/read")
    async def read_balances(request: Request) -> JSONResponse:
        """Ask the providers that state a balance now — each at most once a minute; spends nothing."""
        body = await _body(request) or {}
        account = body.get("account")
        if account is not None:
            kind, _, ident = str(account).partition(":")
            refused = known(kind, ident)
            if refused is not None:
                return refused
            if not balances.can_read(f"{kind}:{ident}"):
                return error(409, "not_reported", "this provider's API says no balance: type it on its card")
            account = f"{kind}:{ident}"
        await balances.read_now(account)
        return JSONResponse(listing(balances.views))

    @app.put("/pool/balances/{kind}/{ident}")
    async def type_balance(kind: str, ident: str, request: Request) -> JSONResponse:
        """What the operator says is left in the account now, in dollars: counted down from here."""
        refused = known(kind, ident)
        if refused is not None:
            return refused
        body = await _body(request)
        if body is None:
            return error(400, "bad_request", "send a JSON object")
        raw = body.get("usd")
        try:
            usd = float(raw)
        except (TypeError, ValueError):
            return error(400, "bad_balance", "usd is a number of dollars")
        if isinstance(raw, bool) or not math.isfinite(usd) or not 0 <= usd <= MOST_TYPED:
            return error(400, "bad_balance", f"a balance is between $0 and ${MOST_TYPED:,.0f}")
        await balances.type_in(f"{kind}:{ident}", round(usd, 4))
        return JSONResponse(balances.view_of(f"{kind}:{ident}"))

    @app.delete("/pool/balances/{kind}/{ident}")
    async def clear_balance(kind: str, ident: str) -> JSONResponse:
        refused = known(kind, ident)
        if refused is not None:
            return refused
        await balances.type_in(f"{kind}:{ident}", None)
        return JSONResponse(balances.view_of(f"{kind}:{ident}"))
