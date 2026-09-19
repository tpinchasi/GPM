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
    ProviderRateLimited,
    ProviderUnavailable,
)

log = logging.getLogger("gpm.vast")

_BASE_URL = "https://console.vast.ai"
_HOURS_PER_MONTH = 730.0


def _gb(megabytes: Optional[float]) -> float:
    return float(megabytes or 0) / 1024.0


class VastProvider:
    interface_version: ClassVar[str] = "1"
    name: ClassVar[str] = "vast"

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
        self._client = client
        self._timeout = timeout
        #: Instance payloads seen recently, by id. Status, charges and connection details all
        #: come from the same payload, and the provider rate-limits repeated fetches of it;
        #: within one control-loop pass the answer does not change.
        self._instances: dict[str, tuple[float, dict[str, Any]]] = {}
        self.instance_cache_s = 10.0
        #: The account's charge rows, fetched once per pass and shared by every instance.
        self._charges: Optional[tuple[float, dict[str, float]]] = None
        #: How far back to look. A parked host bills storage for as long as it is kept.
        self.charges_window_days = 45

    # --- plumbing ---

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ProviderAuthError(
                    f"{self.api_key_env} is not set; the account credential is read from the "
                    "environment and never from configuration"
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

        if response.status_code in (401, 403):
            raise ProviderAuthError(f"{method} {path}: the account credential was refused")
        if response.status_code == 429:
            raise ProviderRateLimited(f"{method} {path}: rate limited")
        if response.status_code == 410:
            # The offer went between search and create — ordinary in a live market.
            raise OfferGone(f"{method} {path}: no longer available")
        if response.status_code >= 400:
            raise ProviderUnavailable(f"{method} {path}: {response.status_code} {response.text[:200]}")
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderUnavailable(f"{method} {path}: response was not JSON") from exc

    # --- the interface ---

    async def search_offers(self, query: OfferQuery) -> list[Offer]:
        body: dict[str, Any] = {
            "limit": query.limit,
            "type": "bid",  # interruptible: the only kind the pool bids on
            "rentable": {"eq": True},
        }
        if query.verified_only:
            body["verified"] = {"eq": True}
        if query.min_gpu_memory_gb:
            body["gpu_ram"] = {"gte": query.min_gpu_memory_gb * 1024}
        if query.min_disk_gb:
            body["disk_space"] = {"gte": query.min_disk_gb}

        payload = await self._call("POST", "/api/v0/bundles", json=body)
        raw_offers = payload.get("offers", payload if isinstance(payload, list) else [])
        on_demand = await self._on_demand_prices(raw_offers)
        return [self._to_offer(entry, on_demand) for entry in raw_offers]

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
        payload = await self._call(
            "POST",
            "/api/v0/bundles",
            json={
                "limit": max(len(machine_ids) * 4, 100),
                "type": "on-demand",
                "machine_id": {"in": [int(m) for m in machine_ids]},
            },
        )
        prices: dict[str, float] = {}
        for entry in payload.get("offers", payload if isinstance(payload, list) else []):
            machine = str(entry.get("machine_id"))
            price = entry.get("dph_total")
            if price is None:
                continue
            # A machine can list several GPU slices; the cheapest on-demand rate is the one
            # a bid on it should be measured against.
            prices[machine] = min(prices.get(machine, float("inf")), float(price))
        return prices

    def _to_offer(self, entry: dict[str, Any], on_demand: Optional[dict[str, float]] = None) -> Offer:
        disk_gb = float(entry.get("disk_space") or 0)
        storage_monthly_per_gb = float(entry.get("storage_cost") or 0)
        return Offer(
            offer_id=str(entry.get("id")),
            machine_id=str(entry.get("machine_id")),
            hardware=f"{entry.get('num_gpus', 1)}x {entry.get('gpu_name', 'unknown')}",
            gpus=int(entry.get("num_gpus") or 1),
            gpu_memory_gb=_gb(entry.get("gpu_ram")),
            disk_gb=disk_gb,
            min_bid_hourly=float(entry.get("min_bid") or 0),
            all_in_hourly=float(entry.get("dph_total") or 0),
            on_demand_hourly=(on_demand or {}).get(str(entry.get("machine_id"))),
            # Quoted per gigabyte per month; the pool reasons in dollars per hour.
            storage_hourly=storage_monthly_per_gb * disk_gb / _HOURS_PER_MONTH,
            download_per_gb=float(entry.get("inet_down_cost") or 0),
            download_mbps=float(entry.get("inet_down") or 0),
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
        if not payload.get("success", True):
            raise BidLost(f"bid ${bid} on {offer.machine_id}: {payload.get('msg', 'refused')}")
        contract = payload.get("new_contract")
        if contract is None:
            raise BidLost(f"bid ${bid} on {offer.machine_id} returned no instance")
        return Instance(instance_id=str(contract), label=spec.label, machine_id=offer.machine_id)

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
        payload = await self._call("GET", "/api/v0/instances", params={"owner": "me"})
        found = []
        for entry in payload.get("instances", []):
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
        if actual == "running":
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

    def self_terminate_command(self, action: str = "destroy") -> str:
        """Run *on the instance*, with the key the provider put there — never the account key."""
        method = "DELETE" if action == "destroy" else "PUT"
        body = "" if action == "destroy" else " -d '{\"state\": \"stopped\"}'"
        return (
            f'curl -sS -X {method} -H "Authorization: Bearer $CONTAINER_API_KEY" '
            f'-H "Content-Type: application/json"{body} '
            f'"{self.base_url}/api/v0/instances/$CONTAINER_ID/"'
        )
