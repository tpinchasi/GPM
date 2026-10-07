"""Vast.ai — the first real provider.

An HTTP API client with typed errors and bounded timeouts, never a CLI wrapper
(docs/spec/plugin-interfaces.md §1). The account credential is read from the environment and
**never leaves this process**: what goes on a rented host is the per-instance key the provider
itself injects there (`CONTAINER_API_KEY`), which can only act on that one instance.

Endpoint shapes come from Vast.ai's public API documentation. Where the documentation is silent
— notably on telling "outbid" apart from "stopped by us" — the pool does not guess: the database
is the source of truth for intent, the provider for existence (supervisor.md §1.2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, ClassVar, Optional

import httpx

from .base import (
    AccountStatus,
    BidLost,
    Charges,
    ConnectionInfo,
    Instance,
    InstanceSpec,
    InstanceState,
    InstanceStatus,
    Offer,
    OfferGone,
    OfferQuery,
    ProviderAuthError,
    ProviderCapabilities,
    ProviderError,
    ProviderRateLimited,
    ProviderUnavailable,
    SelfTerminateRequest,
    redacted,
)

log = logging.getLogger("gpm.vast")

_BASE_URL = "https://console.vast.ai"
#: The provider's own month, in hours: its `dph_total` for 150 GB on a machine at $0.20/GB-month
#: carries exactly $0.041667/h of storage (checked against the live market, 2026-09-25).
_HOURS_PER_MONTH = 720.0


def _state_line(entry: dict[str, Any]) -> str:
    """An instance's state as the provider reports it, on one line: where it is, where the
    provider means it to be, and the provider's own words."""
    parts = [f"actual {entry.get('actual_status') or '?'}", f"intended {entry.get('intended_status') or '?'}"]
    for name in ("cur_state", "next_state"):
        if entry.get(name):
            parts.append(f"{name} {entry[name]}")
    line = ", ".join(parts)
    message = str(entry.get("status_msg") or "").strip()
    return f"{line}: {message[:200]}" if message else line


def _compact(answer: Any, limit: int = 500) -> str:
    """An answer as one line of JSON, cut to a length a log line can carry."""
    text = json.dumps(answer, separators=(",", ":"), sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _gb(megabytes: Optional[float]) -> float:
    return float(megabytes or 0) / 1024.0


class VastProvider:
    interface_version: ClassVar[str] = "2"
    name: ClassVar[str] = "vast"
    display_name: ClassVar[str] = "Vast.ai"
    icon: ClassVar[Optional[str]] = None
    #: The provider's own logo, as its site declares it (`<link rel="icon">`, checked 2026-10-07):
    #: loaded by the console's page, which falls back to a lettermark if it does not load (D134).
    icon_url: ClassVar[Optional[str]] = "https://vast.ai/icon.png"
    #: Where the credential is sent. A stored credential is bound to it: changed, it is cleared.
    endpoint_settings: ClassVar[tuple[str, ...]] = ("base_url",)

    capabilities = ProviderCapabilities(
        interruptible=True,
        parkable=True,
        same_machine_rebid=True,
        #: The provider injects CONTAINER_API_KEY into every container, restricted to starting,
        #: stopping or destroying that instance. This is what makes the dead-man timer possible
        #: without the account credential (D19).
        self_terminate=True,
        #: Verified live: the instance payload carries no accumulated charge, but
        #: `/api/v0/charges/` reports per-instance charges by day, with an itemised breakdown.
        #: The pool still only narrows the cap margin once a figure has actually arrived.
        reports_charges=True,
        price_history=False,
        direct_port_mapping=True,
        #: `PUT /instances/request_logs/{id}/` hands back a URL the boot output is fetched
        #: from. It says outright what a host that never answered was doing (D78).
        reports_instance_logs=True,
    )

    def __init__(
        self,
        api_key_env: str = "VAST_API_KEY",
        base_url: str = _BASE_URL,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
        **_: Any,
    ):
        self.api_key_env = api_key_env
        self.base_url = base_url
        #: The credential the supervisor handed in (D134); until it does, the environment's, as
        #: before — a provider built by hand, outside a pool, still works.
        self._credential: Optional[str] = None
        self._handed = False
        self._client = client
        self._timeout = timeout
        #: The offer search is limited by a daily quota of offer rows returned, not by request
        #: rate (D121): 20,000 a day, reset at 00:00 UTC, as the provider's own refusal states
        #: it. Rows the searches returned since the pool last took the count, and the last
        #: refusal's own numbers.
        self.daily_search_rows = 20_000
        self._rows_untaken = 0
        self._quota_refusal: Optional[dict[str, Any]] = None
        #: Instance payloads seen recently, by id. Status, charges and connection details all
        #: come from the same payload, and the provider rate-limits repeated fetches of it;
        #: within one control-loop pass the answer does not change.
        self._instances: dict[str, tuple[float, dict[str, Any]]] = {}
        self.instance_cache_s = 10.0
        #: The account's charge rows, fetched once per pass and shared by every instance.
        self._charges: Optional[tuple[float, dict[str, float]]] = None
        #: How far back to look. A parked host bills storage for as long as it is kept.
        self.charges_window_days = 45
        #: A guard on the paginated instance listing: better to refuse than to act on half of
        #: what is running.
        self.max_instance_pages = 50

    # --- plumbing ---

    def _quiet(self, text: str) -> str:
        """The provider's own words, with the credential taken out should an answer echo it."""
        key = self._credential if self._handed else os.environ.get(self.api_key_env)
        return text.replace(key, "[credential]") if key and len(key) >= 4 else text

    @property
    def credential_env(self) -> str:
        return self.api_key_env

    def set_credential(self, credential: Optional[str]) -> None:
        """Use this credential from the next call on (D134). A client built with the old one is
        dropped — closed once whatever is using it has finished, never mid-call."""
        if self._handed and (credential or None) == self._credential:
            return
        self._credential = credential or None
        self._handed = True
        old, self._client = self._client, None
        if old is not None:
            try:
                asyncio.get_running_loop().call_later(60, lambda: asyncio.ensure_future(old.aclose()))
            except RuntimeError:
                pass

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            key = self._credential if self._handed else os.environ.get(self.api_key_env)
            if not key:
                raise ProviderAuthError(
                    "no credential is set for this provider: type one in on the Providers screen, or set "
                    f"{self.api_key_env} in the supervisor's environment"
                )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{method} {path}: {exc}") from exc

        if 300 <= response.status_code < 400:
            # Never quietly read a redirect's own body as the answer (D43) — the one path this
            # bit us on had a body that happened to parse as valid, contentless JSON.
            raise ProviderUnavailable(
                f"{method} {path}: unexpected redirect to {response.headers.get('location', '?')}"
            )
        if response.status_code in (401, 403):
            raise ProviderAuthError(f"{method} {path}: the account credential was refused")
        if response.status_code == 429:
            try:
                said = response.json()
            except ValueError:
                said = {}
            if isinstance(said, dict) and said.get("error") == "search_quota_exceeded":
                # The day's offer rows are spent: the provider says how many, and for how long.
                limit = int(said.get("limit") or self.daily_search_rows)
                wait = float(said.get("retry_after") or 0)
                self.daily_search_rows = limit
                self._quota_refusal = {"limit": limit, "remaining": int(said.get("remaining") or 0),
                                       "retry_after_s": wait, "at": time.time()}
                raise ProviderRateLimited(
                    f"the provider's daily search quota of {limit:,} offers is used up; it resets in {wait / 3600:.1f}h"
                )
            raise ProviderRateLimited(f"{method} {path}: rate limited")
        if response.status_code == 410:
            # On an offer, "gone" is ordinary in a live market: it went between search and
            # create. On any other path it means the *endpoint* is gone — this API answers 410
            # `deprecated_endpoint` for the retired v0 instance listing (D43) — and calling
            # that a vanished offer would send the caller down the "try the next offer" path
            # when the truth is "this client is broken".
            if "/asks/" in path:
                raise OfferGone(f"{method} {path}: no longer available")
            raise ProviderUnavailable(
                f"{method} {path}: the provider says this endpoint is gone ({self._quiet(response.text[:120])})"
            )
        if response.status_code >= 400:
            raise ProviderUnavailable(f"{method} {path}: {response.status_code} {self._quiet(response.text[:200])}")
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderUnavailable(f"{method} {path}: response was not JSON") from exc

    # --- the daily search quota (D121) ---

    async def _search(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """One offer search, its rows counted against the day's quota."""
        payload = await self._call("POST", "/api/v0/bundles/", json=body)
        entries = payload.get("offers", payload if isinstance(payload, list) else [])
        self._rows_untaken += len(entries)
        return entries

    def take_search_usage(self) -> dict[str, Any]:
        """Rows returned since last asked, and the last quota refusal — then forgotten here: the
        pool keeps the day's count, which outlives this process."""
        usage = {"rows": self._rows_untaken, "refusal": self._quota_refusal, "limit": self.daily_search_rows}
        self._rows_untaken, self._quota_refusal = 0, None
        return usage

    # --- the interface ---

    async def search_offers(self, query: OfferQuery) -> list[Offer]:
        found: list[Offer] = []
        if query.interruptible:
            raw = await self._listing("bid", query)
            on_demand = await self._on_demand_prices(raw)
            found.extend(self._to_offer(entry, on_demand) for entry in raw)
        if query.on_demand:
            # Priced at what it says, and not outbiddable: its own price is also its ceiling.
            raw = await self._listing("on-demand", query)
            found.extend(self._to_offer(entry, {}, interruptible=False) for entry in raw)
        return found

    async def offer_for_machine(self, machine_id: str, gpus: int) -> Optional[Offer]:
        """What it takes now to bid for this machine's `gpus`-card slice — asked of the machine
        itself, rentable or not (D109). The ordinary search sees only machines anyone may rent,
        and a machine the pool was just outbid on is held by whoever outbid it, so it is exactly
        the one the search cannot see."""
        try:
            wanted = int(machine_id)
        except ValueError:
            return None
        entries = await self._search({"machine_id": {"eq": wanted}, "type": "bid", "limit": 20})
        entry = next((e for e in entries if int(e.get("num_gpus") or 1) == gpus), None)
        if entry is None:
            return None
        return self._to_offer(entry, await self._on_demand_prices([entry]))

    async def _listing(self, kind: str, query: OfferQuery) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"limit": query.limit, "type": kind, "rentable": {"eq": True}}
        if query.verified_only:
            body["verified"] = {"eq": True}
        if query.min_gpu_memory_gb:
            body["gpu_ram"] = {"gte": query.min_gpu_memory_gb * 1024}
        if query.min_disk_gb:
            body["disk_space"] = {"gte": query.min_disk_gb}
        if query.min_gpus > 1:
            body["num_gpus"] = {"gte": query.min_gpus}
        # The rest of the pool's filters, as conditions they imply (D123): every offer returned
        # counts against the provider's daily quota, so what the pool would only throw away is
        # better never returned. The pool still filters each offer itself.
        if query.max_all_in_hourly is not None:
            # The pool's all-in price is `dph_base` plus storage for its own disk, so it is never
            # below `dph_base`: an offer whose base is above the ceiling cannot pass.
            body["dph_base"] = {"lte": query.max_all_in_hourly}
        if query.min_download_mbps:
            body["inet_down"] = {"gte": query.min_download_mbps}
        if query.max_download_per_gb is not None:
            body["inet_down_cost"] = {"lte": query.max_download_per_gb}
        if query.min_reliability:
            body["reliability"] = {"gte": query.min_reliability}
        if query.exclude_hardware:
            body["gpu_name"] = {"notin": list(query.exclude_hardware)}
        avoided = [int(m) for m in query.avoid_machines if str(m).isdigit()]
        if avoided:
            body["machine_id"] = {"notin": avoided}
        return await self._search(body)

    async def _on_demand_prices(self, bid_offers: list[dict[str, Any]]) -> dict[str, float]:
        """The on-demand price of each machine, from the on-demand listing.

        On a bid-type offer the provider's `dph_base` is the floor again, not the on-demand
        rate (verified against the live market: identical to `min_bid` on every row). The
        crossover ceiling needs the real on-demand price, so it is fetched for exactly the
        machines in hand and joined by machine id.
        """
        machine_ids = sorted({str(entry.get("machine_id")) for entry in bid_offers if entry.get("machine_id")})
        if not machine_ids:
            return {}
        entries = await self._search({
            "limit": max(len(machine_ids) * 4, 100),
            "type": "on-demand",
            "machine_id": {"in": [int(m) for m in machine_ids]},
        })
        prices: dict[str, float] = {}
        for entry in entries:
            machine = str(entry.get("machine_id"))
            price = entry.get("dph_total")
            if price is None:
                continue
            # A machine can list several GPU slices; the cheapest on-demand rate is the one
            # a bid on it should be measured against.
            prices[machine] = min(prices.get(machine, float("inf")), float(price))
        return prices

    def _to_offer(
        self, entry: dict[str, Any], on_demand: Optional[dict[str, float]] = None,
        interruptible: bool = True,
    ) -> Offer:
        disk_gb = float(entry.get("disk_space") or 0)
        storage_monthly_per_gb = float(entry.get("storage_cost") or 0)
        return Offer(
            offer_id=str(entry.get("id")),
            machine_id=str(entry.get("machine_id")),
            hardware=f"{entry.get('num_gpus', 1)}x {entry.get('gpu_name', 'unknown')}",
            gpus=int(entry.get("num_gpus") or 1),
            gpu_memory_gb=_gb(entry.get("gpu_ram")),
            disk_gb=disk_gb,
            # On an on-demand listing there is no bid: the price is the price.
            min_bid_hourly=float(entry.get("min_bid") or 0) if interruptible else float(entry.get("dph_total") or 0),
            all_in_hourly=float(entry.get("dph_total") or 0),
            on_demand_hourly=(
                (on_demand or {}).get(str(entry.get("machine_id")))
                if interruptible else float(entry.get("dph_total") or 0)
            ),
            interruptible=interruptible,
            # The storage inside `dph_total` is for a few GB the provider chose, not the listing's
            # disk; each offer is repriced for the disk the pool rents (D108). Quoted per GB per
            # month; the pool reasons in dollars per hour.
            storage_hourly=float(entry.get("storage_total_cost") or 0),
            storage_per_gb_hourly=storage_monthly_per_gb / _HOURS_PER_MONTH,
            download_per_gb=float(entry.get("inet_down_cost") or 0),
            download_mbps=float(entry.get("inet_down") or 0),
            # What the engine image will find when it looks for the card (D81).
            driver_version=str(entry["driver_version"]) if entry.get("driver_version") else None,
            reliability=float(entry.get("reliability") or entry.get("reliability2") or 0),
            verified=str(entry.get("verification", "")).lower() == "verified",
            # `dlperf` is the provider's own throughput index; it ranks, it does not benchmark.
            throughput_proxy=float(entry.get("dlperf") or entry.get("num_gpus") or 1),
            raw=entry,
        )

    async def create(self, offer: Offer, spec: InstanceSpec, bid: Optional[float]) -> Instance:
        env = " ".join(f"-e {key}={value}" for key, value in spec.env.items())
        env = " ".join(filter(None, [env] + [f"-p {port}:{port}" for port in spec.ports]))
        body: dict[str, Any] = {
            "client_id": "me",
            "image": spec.image,
            "disk": spec.disk_gb,
            "label": spec.label,
            "runtype": "ssh",
        }
        if bid is not None:
            body["price"] = bid
        if env:
            body["env"] = env
        if spec.onstart:
            body["onstart"] = spec.onstart

        payload = await self._call("PUT", f"/api/v0/asks/{offer.offer_id}/", json=body)
        answer = redacted(payload)
        success = bool(payload.get("success", True))
        contract = payload.get("new_contract")
        if success and contract is not None:
            return Instance(instance_id=str(contract), label=spec.label, machine_id=offer.machine_id)

        # Not a host. Three shapes, told apart because they mean different things:
        # - a refusal proper carries `error` and `msg` and no contract (the documented shape);
        # - `success: false` **with** a contract is a bid that was created and did not start —
        #   seen live, repeatedly, on a contested machine: the listing's `min_bid` is the host's
        #   floor, not the standing top bid, so floor plus premium can lose on creation;
        # - `success: true` with no contract, which this client does not trust.
        words = payload.get("msg") or payload.get("error")
        if not success and contract is not None:
            detail = words or "created but not started — the bid did not win the machine"
        elif not success:
            detail = words or f"refused, answering {_compact(answer)}"
        else:
            detail = "returned no instance"

        # What the provider says about the instance the attempt created, read **before** it is
        # destroyed: afterwards the provider keeps no record of it (seen live: `instances: None`).
        state: Optional[dict[str, Any]] = None
        state_line = ""
        if contract is not None:
            try:
                entry = await self._instance(str(contract))
            except ProviderError as exc:
                state_line = f"its state could not be read: {exc}"
            else:
                state = redacted(entry) if entry else None
                state_line = _state_line(entry) if entry else "the provider has no record of it"

        # "Either an instance or nothing behind" is this method's contract, so before reporting
        # the bid lost, look for what the label would have been and end it.
        stray = await self._destroy_stray(spec.label)
        if stray:
            detail += f" — but created {stray}"
            if state_line:
                detail += f" ({state_line})"
            detail += ", which has been destroyed"
        raise BidLost(f"bid ${bid} on {offer.machine_id}: {detail}", response=answer, instance_state=state)

    async def _destroy_stray(self, label: str) -> Optional[str]:
        """An instance this exact label names, ended. Returns its id if there was one.

        The label is chosen before the call and used once, so it identifies the attempt even
        when the response does not.

        A failure to list or destroy propagates as a `ProviderError` rather than a `BidLost`,
        which is the difference between "that bid did not take, try the next offer" and "I
        cannot prove nothing is running, stop" — and the caller acts on exactly that.
        """
        for instance in await self.list_instances(label):
            await self.destroy(instance)
            return instance.instance_id
        return None

    def _remember(self, entry: dict[str, Any]) -> None:
        if entry.get("id") is not None:
            self._instances[str(entry["id"])] = (time.monotonic(), entry)

    async def _instance(self, instance_id: str) -> dict[str, Any]:
        cached = self._instances.get(instance_id)
        if cached is not None and time.monotonic() - cached[0] < self.instance_cache_s:
            return cached[1]
        payload = await self._call("GET", f"/api/v0/instances/{instance_id}/")
        entry = payload.get("instances") or payload.get("instance") or {}
        if entry:
            self._remember(entry)
        return entry

    async def list_instances(self, label_prefix: str) -> list[Instance]:
        """Every instance on the account carrying this prefix, in any state.

        On `/api/v1/`, not v0: v0's collection endpoint answers `410 deprecated_endpoint`, and
        the bare `/api/v0/instances` before it answers a 301 whose body is valid JSON with no
        "instances" key — which is how this read as "nothing exists" while three instances
        were billing (D43). v1 paginates, and a half-read page here means a missed orphan, so
        the pages are followed to the end.
        """
        entries: list[dict[str, Any]] = []
        params: dict[str, Any] = {"owner": "me"}
        for _ in range(self.max_instance_pages):
            payload = await self._call("GET", "/api/v1/instances/", params=params)
            entries.extend(payload.get("instances", []))
            token = payload.get("next_token")
            if not token:
                break
            params = {**params, "start_token": token}
        else:
            raise ProviderUnavailable(
                f"the instance listing did not end after {self.max_instance_pages} pages; "
                "refusing to act on a partial list of what is running"
            )

        found = []
        for entry in entries:
            self._remember(entry)
            label = entry.get("label") or ""
            if not label.startswith(label_prefix):
                continue
            found.append(
                Instance(
                    instance_id=str(entry.get("id")),
                    label=label,
                    machine_id=str(entry.get("machine_id")),
                    raw=entry,
                )
            )
        return found

    async def status(self, instance: Instance) -> InstanceStatus:
        entry = await self._instance(instance.instance_id)
        if not entry:
            return InstanceStatus(state=InstanceState.GONE)

        actual = str(entry.get("actual_status") or "").lower()
        intended = str(entry.get("intended_status") or "").lower()
        if intended == "stopped":
            # The provider has decided not to run it, whatever its container is doing now.
            # Found live (D109): outbid while its image was still loading, an instance read
            # `loading` for minutes with intended, current and next state all `stopped` — and
            # read as "still starting", the pool neither re-bid nor released it.
            state = InstanceState.STOPPED
        elif actual == "running":
            state = InstanceState.RUNNING
        elif actual in ("loading", "created", "scheduling"):
            state = InstanceState.SCHEDULING
        elif actual in ("exited", "stopped", "offline"):
            state = InstanceState.STOPPED
        else:
            state = InstanceState.SCHEDULING

        # The provider does not say who stopped it. "We asked for running and it is not" is the
        # honest signal; the supervisor compares it against its own record of intent.
        stopped_by_provider = state == InstanceState.STOPPED and intended == "running"
        return InstanceStatus(
            state=state,
            stopped_by_provider=stopped_by_provider,
            bid_hourly=float(entry["dph_total"]) if entry.get("dph_total") else None,
            detail=entry.get("status_msg"),
            # Seen live: an instance created with a start-up script and reported back without
            # one. The field is always present on this API, so its emptiness is an answer.
            startup_material=bool(entry["onstart"]) if "onstart" in entry else None,
        )

    async def set_bid(self, instance: Instance, bid: float) -> None:
        await self._call(
            "PUT", f"/api/v0/instances/bid_price/{instance.instance_id}/", json={"price": bid}
        )

    async def start(self, instance: Instance) -> None:
        await self._call(
            "PUT", f"/api/v0/instances/{instance.instance_id}/", json={"state": "running"}
        )

    async def stop(self, instance: Instance) -> None:
        await self._call(
            "PUT", f"/api/v0/instances/{instance.instance_id}/", json={"state": "stopped"}
        )

    async def destroy(self, instance: Instance) -> None:
        await self._call("DELETE", f"/api/v0/instances/{instance.instance_id}/")
        self._instances.pop(instance.instance_id, None)

    async def connection(self, instance: Instance) -> ConnectionInfo:
        entry = await self._instance(instance.instance_id)
        return ConnectionInfo(
            ssh_host=entry.get("ssh_host"),
            ssh_port=int(entry["ssh_port"]) if entry.get("ssh_port") else None,
            ssh_user="root",
        )

    async def instance_logs(self, instance: Instance, tail: int = 60) -> Optional[str]:
        """This instance's boot output (D78).

        Two steps, as the provider defines it: ask for the logs, then fetch them from the URL
        it names. That URL is the provider's own and is read for text only — nothing in it is
        executed, and nothing the machine wrote decides what the pool does next.
        """
        try:
            asked = await self._call(
                "PUT", f"/api/v0/instances/request_logs/{instance.instance_id}/", json={"tail": tail}
            )
        except ProviderError:
            return None
        url = (asked or {}).get("result_url") if isinstance(asked, dict) else None
        if not url or not str(url).startswith("https://"):
            return None
        # Read with a client of its own, carrying no credential: the URL is on the provider's log
        # storage, not its API, and the account key never leaves this process (T5).
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout), follow_redirects=False) as plain:
            # The provider writes the file after answering, so a first read can find nothing there.
            for attempt in range(6):
                if attempt:
                    await asyncio.sleep(1.0)
                try:
                    response = await plain.get(url)
                except httpx.HTTPError:
                    return None
                if response.status_code == 200 and response.text.strip():
                    lines = response.text.splitlines()
                    return "\n".join(lines[-tail:])
        return None

    async def _charges_by_instance(self) -> dict[str, float]:
        cached = self._charges
        if cached is not None and time.monotonic() - cached[0] < self.instance_cache_s:
            return cached[1]
        now = int(time.time())
        totals: dict[str, float] = {}
        after: Optional[str] = None
        for _ in range(20):  # pages; far more than a pool's hosts could fill
            params: dict[str, Any] = {
                "select_filters": json.dumps(
                    {
                        "day": {"gte": now - self.charges_window_days * 86400, "lte": now + 86400},
                        "type": {"in": ["instance"]},
                    }
                ),
                "latest_first": "true",
                "limit": 100,
                "format": "table",
            }
            if after:
                params["after_token"] = after
            payload = await self._call("GET", "/api/v0/charges/", params=params)
            for row in payload.get("results", []):
                source = str(row.get("source") or "")
                if source.startswith("instance-"):
                    totals[source[len("instance-"):]] = totals.get(source[len("instance-"):], 0.0) + float(
                        row.get("amount") or 0.0
                    )
            after = payload.get("next_token")
            if not after:
                break
        self._charges = (time.monotonic(), totals)
        return totals

    async def reported_charges(self, instance: Instance) -> Optional[Charges]:
        """What the provider has charged this instance so far, summed over its daily rows.

        No row at all means nothing has been reported yet — returned as None, not zero, so the
        cap margin does not narrow on the strength of a figure that has not arrived. Caps are
        enforced on the higher of the estimate and this, so a lagging report can never loosen
        them.
        """
        totals = await self._charges_by_instance()
        if instance.instance_id not in totals:
            return None
        return Charges(total=totals[instance.instance_id], as_of=time.time())

    async def account(self) -> AccountStatus:
        payload = await self._call("GET", "/api/v0/users/current/")
        return AccountStatus(
            credential_valid=True,
            credit_remaining=float(payload["credit"]) if payload.get("credit") is not None else None,
        )

    def self_terminate_request(self, action: str = "destroy") -> SelfTerminateRequest:
        """Made *on the instance*, with the key the provider put there — never the account key."""
        return SelfTerminateRequest(
            method="DELETE" if action == "destroy" else "PUT",
            url=f"{self.base_url}/api/v0/instances/$CONTAINER_ID/",
            headers={
                "Authorization": "Bearer $CONTAINER_API_KEY",
                "Content-Type": "application/json",
            },
            body=None if action == "destroy" else '{"state": "stopped"}',
        )
