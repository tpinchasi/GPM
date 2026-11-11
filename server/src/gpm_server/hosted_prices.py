"""A hosted API's prices, found rather than typed (D176).

The provider sets its prices; the operator should not have to copy them. So when the console
lists a provider's models it fills each one's price from, in order:

1. **the provider's own `/models`**, where it says them — OpenRouter (USD per token, as strings)
   and Together AI (USD per million tokens) do;
2. **the provider's own pricing page**, where its preset names one whose table the pool can read
   (Ollama's cloud: model, input, cached input, output, USD per million tokens) — the peak row,
   never an off-peak one;
3. **a public price list** — LiteLLM's community-kept model cost map by default — for a provider
   whose API and page say none (Anthropic, OpenAI, Groq, DeepSeek, Fireworks, DeepInfra);
4. nothing: the operator types it, as before.

What is found is only a proposal. It is shown with where it came from, and nothing is priced until
the operator saves it — a price list kept by a third party can be wrong, a page can change, and a
price of zero would let an app spend past its budget unseen. A provider sold as a plan is saved as
`plan: true` where the operator says so: no price per token, the plan's own limit the cap.

Everything read here is checked as untrusted: a number that is not a finite, non-negative number
of dollars under `MOST_PER_MILLION` is dropped, never guessed at.
"""

from __future__ import annotations

import html
import json
import math
import time
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Optional

import httpx

#: LiteLLM's model cost map, as LiteLLM itself fetches it (unverified as the raw form of the
#: file its docs link: decisions.md, "Unverified assumptions").
PRICE_LIST_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
#: The list is a few megabytes; anything past this is not it.
MAX_LIST_BYTES = 32 * 1024 * 1024
#: How long a fetched list is used before it is fetched again.
LIST_FRESH_S = 6 * 3600.0
#: A price per million tokens past this is a mistake in the source, not a price.
MOST_PER_MILLION = 10_000.0

#: The `litellm_provider` names each preset's models are listed under (unverified but for
#: Anthropic's: decisions.md, "Unverified assumptions"). A preset not here gets nothing from it.
LIST_PROVIDERS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "openai": ("openai",),
    "groq": ("groq",),
    "together": ("together_ai",),
    "openrouter": ("openrouter",),
    "deepseek": ("deepseek",),
    "fireworks": ("fireworks_ai",),
    "deepinfra": ("deepinfra",),
}
#: Presets paid by a plan — a subscription with its own limit — rather than per token. None now:
#: Ollama's cloud, once one, prices every model per token (its pricing page, read 2026-11-08).
PLAN_PRESETS: frozenset[str] = frozenset()
#: A pricing page is a few dozen kilobytes; anything past this is not one.
MAX_PAGE_BYTES = 2 * 1024 * 1024
#: The columns a pricing page's table must have, in order, to be read at all — a page that
#: changes them proposes nothing, rather than prices read from the wrong column.
PAGE_COLUMNS = ("model", "input", "cached input", "output")


def _dollars(value: Any, per_token: bool) -> Optional[float]:
    """Dollars per million tokens, or None for anything that is not a sane price."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if per_token:
        number *= 1e6
    if not math.isfinite(number) or number < 0 or number > MOST_PER_MILLION:
        return None
    return round(number, 6)


def _price(input_: Optional[float], output: Optional[float], cached: Optional[float] = None,
           write: Optional[float] = None) -> Optional[dict[str, float]]:
    if input_ is None or output is None:
        return None
    price = {"input": input_, "output": output}
    if cached is not None:
        price["cached_input"] = cached
    if write is not None:
        price["cache_write"] = write
    return price


def from_listing(entries: Any) -> dict[str, dict[str, float]]:
    """Prices a provider's own `/models` says, by model id: OpenRouter's `pricing` (`prompt`,
    `completion`, `input_cache_read`, `input_cache_write`: USD per token, as strings) and
    Together's (`input`, `output`: USD per million tokens, both zero where it does not say)."""
    found: dict[str, dict[str, float]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            continue
        if "prompt" in pricing or "completion" in pricing:
            price = _price(_dollars(pricing.get("prompt"), True), _dollars(pricing.get("completion"), True),
                           _dollars(pricing.get("input_cache_read"), True) if "input_cache_read" in pricing else None,
                           _dollars(pricing.get("input_cache_write"), True) if "input_cache_write" in pricing else None)
        elif "input" in pricing and "output" in pricing:
            price = _price(_dollars(pricing.get("input"), False), _dollars(pricing.get("output"), False),
                           _dollars(pricing.get("cached_input"), False) if "cached_input" in pricing else None)
            if price is not None and price["input"] == 0 and price["output"] == 0:
                price = None  # Together lists zeros for what it does not price per token
        else:
            price = None
        if price is not None:
            found[entry["id"]] = price
    return found


def _from_list_entry(entry: Any, providers: tuple[str, ...]) -> Optional[dict[str, float]]:
    if not isinstance(entry, dict) or entry.get("litellm_provider") not in providers:
        return None
    if entry.get("mode", "chat") != "chat":
        return None
    return _price(_dollars(entry.get("input_cost_per_token"), True),
                  _dollars(entry.get("output_cost_per_token"), True),
                  _dollars(entry.get("cache_read_input_token_cost"), True)
                  if "cache_read_input_token_cost" in entry else None,
                  _dollars(entry.get("cache_creation_input_token_cost"), True)
                  if "cache_creation_input_token_cost" in entry else None)


async def _fetch(url: str, most: int = MAX_LIST_BYTES) -> bytes:
    """The bytes at `url`, refused past `most` — read as they arrive, never whole first."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > most:
                    raise ValueError("the price list is larger than any price list")
                chunks.append(chunk)
            return b"".join(chunks)


class PriceList:
    """The public price list, fetched when a provider's models are listed and kept a while.
    `fetch` is replaced in tests: every test runs without the network."""

    def __init__(self, url: Optional[str], fetch: Optional[Callable[[str], Awaitable[bytes]]] = None):
        self.url = url or None
        self.fetch = fetch or _fetch
        self._table: Optional[dict] = None
        self._at = 0.0
        #: Why the list could not be read the last time it was asked, in words; None when it was.
        self.problem: Optional[str] = None

    async def table(self) -> Optional[dict]:
        if self.url is None:
            return None
        if self._table is not None and time.time() - self._at < LIST_FRESH_S:
            return self._table
        try:
            data = json.loads(await self.fetch(self.url))
            if not isinstance(data, dict):
                raise ValueError("not a price list")
        except Exception as exc:  # noqa: BLE001 - a list that cannot be read proposes nothing
            self.problem = f"the public price list could not be read ({type(exc).__name__})"
            return self._table  # an older copy, where there is one, is better than none
        self._table, self._at, self.problem = data, time.time(), None
        return data

    async def prices(self, preset_id: Optional[str], ids: list[str]) -> dict[str, dict[str, float]]:
        """Prices the list has for these model ids of this preset's provider."""
        providers = LIST_PROVIDERS.get(preset_id or "")
        if not providers or not ids:
            return {}
        table = await self.table()
        if not table:
            return {}
        found: dict[str, dict[str, float]] = {}
        for model_id in ids:
            for key in (model_id, *(f"{p}/{model_id}" for p in providers)):
                price = _from_list_entry(table.get(key), providers)
                if price is not None:
                    found[model_id] = price
                    break
        return found


class _Table(HTMLParser):
    """The first `<table>` of a page, as rows of cell texts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._depth = 0
        self._done = False
        self._cell: Optional[list[str]] = None

    def handle_starttag(self, tag, attrs):
        if self._done:
            return
        if tag == "table":
            self._depth += 1
        elif self._depth and tag == "tr":
            self.rows.append([])
        elif self._depth and tag in ("td", "th") and self.rows:
            self._cell = []

    def handle_endtag(self, tag):
        if self._done:
            return
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "table" and self._depth:
            self._depth -= 1
            self._done = self._depth == 0

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def from_pricing_page(page: str) -> dict[str, dict[str, float]]:
    """A pricing page's table — model, input, cached input, output, in dollars per million
    tokens — by model name. Read only when its headings are exactly those; an off-peak row is
    left out (a provider is priced at its peak, D164), and so is a model with no output price."""
    parser = _Table()
    parser.feed(page)
    rows = [r for r in parser.rows if r]
    if not rows or tuple(cell.lower() for cell in rows[0]) != PAGE_COLUMNS:
        raise ValueError("the pricing page's table is not the one expected")
    found: dict[str, dict[str, float]] = {}

    def cell(text: str) -> Optional[float]:
        text = html.unescape(text).strip()
        return None if text in ("", "-", "—") else _dollars(text.removeprefix("$").replace(",", ""), False)

    for row in rows[1:]:
        if len(row) != len(PAGE_COLUMNS):
            continue
        name = row[0].strip()
        if not name or "(" in name or len(name) > 200:
            continue  # "(Off-Peak)" and anything else annotated: not the price
        price = _price(cell(row[1]), cell(row[3]), cell(row[2]))
        if price is not None:
            found[name] = price
    return found


class PricingPage:
    """A provider's own pricing page (D176), fetched when its models are listed and kept a while.
    `fetch` is replaced in tests."""

    def __init__(self, url: str, fetch: Optional[Callable[..., Awaitable[bytes]]] = None):
        self.url = url
        self.fetch = fetch or _fetch
        self._prices: Optional[dict[str, dict[str, float]]] = None
        self._at = 0.0
        self.problem: Optional[str] = None

    async def prices(self, ids: list[str]) -> dict[str, dict[str, float]]:
        """Prices for these model ids: by the id itself, else by its name before the tag — the
        page prices `gemma4`, the API offers `gemma4:31b`."""
        if self._prices is None or time.time() - self._at >= LIST_FRESH_S:
            try:
                raw = await self.fetch(self.url, MAX_PAGE_BYTES)
                self._prices, self._at, self.problem = from_pricing_page(raw.decode("utf-8", "replace")), time.time(), None
            except Exception as exc:  # noqa: BLE001 - a page that cannot be read proposes nothing
                self.problem = f"the provider's pricing page could not be read ({type(exc).__name__})"
        table = self._prices or {}
        found = {}
        for model_id in ids:
            price = table.get(model_id) or table.get(model_id.split(":", 1)[0])
            if price is not None:
                found[model_id] = price
        return found


async def proposed(preset_id: Optional[str], ids: list[str], listed: dict[str, dict[str, float]],
                   price_list: Optional[PriceList], page: Optional[PricingPage] = None,
                   ) -> tuple[dict[str, dict[str, Any]], Optional[str]]:
    """Each model's proposed price with where it came from — `provider`, `provider_page`,
    `price_list` or `plan` — and why a source said nothing, where it could not be read."""
    if preset_id in PLAN_PRESETS:
        return {model_id: {"price": {"plan": True}, "source": "plan"} for model_id in ids}, None
    out = {model_id: {"price": price, "source": "provider"} for model_id, price in listed.items() if model_id in ids}
    problems = []
    rest = [model_id for model_id in ids if model_id not in out]
    if rest and page is not None:
        for model_id, price in (await page.prices(rest)).items():
            out[model_id] = {"price": price, "source": "provider_page"}
        problems.append(page.problem)
        rest = [model_id for model_id in ids if model_id not in out]
    if rest and price_list is not None and LIST_PROVIDERS.get(preset_id or ""):
        for model_id, price in (await price_list.prices(preset_id, rest)).items():
            out[model_id] = {"price": price, "source": "price_list"}
        problems.append(price_list.problem)
    return out, "; ".join(p for p in problems if p) or None
