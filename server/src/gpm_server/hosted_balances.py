"""What is left in a hosted API's account, read from the provider where its API says it (D188).

A preset names the address and how its answer reads (`hosted_presets.HostedPreset.balance_url`,
`balance_format`); nothing else is ever asked. The read is the supervisor's, on a slow schedule and
when an operator asks — never the router's, never on a request's way. It is sent with the host's own
key, and only to an address on the same scheme, host and port as the host's `base_url`, which is
the one the key was saved for: the key reaches its own provider and nothing else.

Only what each provider's API reference states is read, in dollars. An answer in another currency,
or in a shape the reference does not give, is no balance — said, never guessed at. Nothing a
provider answers is kept but the number and the pool's own words about it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import httpx

#: The most a balance can be and still be believed: anything larger is a misread.
MOST_BELIEVED = 10_000_000.0


@dataclasses.dataclass(frozen=True)
class BalanceReading:
    #: Dollars left in the account, or None where the answer gave none.
    usd: Optional[float]
    #: Why there is none, or what to know about the one there is — the pool's words.
    detail: Optional[str] = None


def _dollars(value: Any) -> Optional[float]:
    """A number, or a number written as a string (DeepSeek's are), believed only if finite."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or abs(number) > MOST_BELIEVED:
        return None
    return number


def openrouter_credits(payload: Any) -> BalanceReading:
    """`GET /api/v1/credits` → `{"data": {"total_credits": 100.5, "total_usage": 25.75}}`, in
    dollars: what is left is the one less the other."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return BalanceReading(None, "the answer had no data")
    bought, used = _dollars(data.get("total_credits")), _dollars(data.get("total_usage"))
    if bought is None or used is None:
        return BalanceReading(None, "the answer had no total_credits or total_usage")
    return BalanceReading(round(bought - used, 6))


def deepseek_balance(payload: Any) -> BalanceReading:
    """`GET /user/balance` → `{"is_available": true, "balance_infos": [{"currency": "USD",
    "total_balance": "110.00", ...}]}`: the dollar entry's `total_balance` (granted and topped up
    together). An account kept in another currency has no dollar balance to count down."""
    infos = payload.get("balance_infos") if isinstance(payload, dict) else None
    if not isinstance(infos, list):
        return BalanceReading(None, "the answer had no balance_infos")
    entries = [i for i in infos if isinstance(i, dict)]
    usd = next((i for i in entries if str(i.get("currency") or "").upper() == "USD"), None)
    if usd is None:
        others = sorted({str(i.get("currency") or "?").upper()[:8] for i in entries})
        return BalanceReading(None, f"the balance is kept in {', '.join(others) or 'no currency'}, not dollars")
    amount = _dollars(usd.get("total_balance"))
    if amount is None:
        return BalanceReading(None, "the dollar entry had no total_balance")
    return BalanceReading(amount)


#: How each preset's answer reads, by its `balance_format`.
READERS: dict[str, Callable[[Any], BalanceReading]] = {
    "openrouter-credits": openrouter_credits,
    "deepseek-balance": deepseek_balance,
}


def same_origin(url: str, base_url: str) -> bool:
    """One scheme, host and port: where a key saved for `base_url` may also be sent."""
    try:
        a, b = urlsplit(url), urlsplit(base_url)
        return bool(a.hostname) and (a.scheme, a.hostname, a.port) == (b.scheme, b.hostname, b.port)
    except ValueError:
        return False


async def read(client: httpx.AsyncClient, base_url: str, balance_url: str, balance_format: str,
               note: str = "") -> BalanceReading:
    """Ask the provider. Raises only `ValueError` for an address the key may not go to; every
    other failure is a reading with no balance and the pool's words for why."""
    reader = READERS.get(balance_format)
    if reader is None:
        raise ValueError(f"no reader for {balance_format!r}")
    if not same_origin(balance_url, base_url):
        raise ValueError("a balance is read only from the host's own provider")
    try:
        response = await client.get(balance_url, timeout=20.0)
    except httpx.HTTPError as exc:
        return BalanceReading(None, f"the provider could not be reached: {type(exc).__name__}")
    if response.status_code in (401, 403):
        return BalanceReading(None, f"the provider would not answer the balance to this key ({response.status_code})"
                              + (f": {note}" if note else ""))
    if response.status_code != 200:
        return BalanceReading(None, f"the balance address answered {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        return BalanceReading(None, "the provider answered something that is not JSON")
    return reader(payload)
