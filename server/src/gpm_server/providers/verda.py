"""Verda (formerly DataCrunch) — GPU virtual machines, on demand or spot, through its REST API.

An HTTP API client with typed errors and bounded timeouts, never a CLI wrapper
(docs/spec/plugin-interfaces.md §1). The account credential — a Cloud API client's id and secret —
is handed in by the supervisor (or, outside a pool, read from the environment), exchanged here for
a short-lived token, and **never leaves this process**.

Endpoint and field shapes come from the provider's published OpenAPI document for
`https://api.verda.com/v1` (read 2026-10-08). What they leave open is said where it is decided:

- **A virtual machine, not a container.** The provider boots an OS image and runs a start-up
  script; it has no "run this image" field. So the script runs the engine image as **one
  container** with its own SSH daemon, and publishes only that daemon's port: the container is
  the host, exactly as a pod is elsewhere, and the engine's port is never published — every port
  of the machine is open to the internet, with no provider firewall.
- **No machine before renting.** The catalog lists instance types; availability is per type and
  location. An offer is one type, in one location, on demand or spot, and its `machine_id` is
  `location:type`.
- **Spot is priced by the provider**, never bid for (D132): the listed spot price is paid, and a
  spot instance may be taken back at any moment, without notice. One taken back is deleted.
- **No dead-man timer.** The provider's credentials are project-wide: none an instance could hold
  that ends only itself, and a machine powered off from inside still bills. So no timer is armed,
  and the pool rents here only for short leases (providers.md §4).
- **No park.** A shut-down instance keeps billing at its full rate; only a delete stops it.
- **The disk is the OS volume**, billed beside the instance at the NVMe rate, and deleted with it
  — never left detached, which is the provider's default and keeps billing.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import logging
import math
import os
import re
import shlex
import time
from typing import Any, ClassVar, Optional

import httpx

from .base import (
    AccountStatus,
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
    VolumeInfo,
)
from .hostscript import SSH_DAEMON

log = logging.getLogger("gpm.verda")

_BASE_URL = "https://api.verda.com/v1"
#: An NVMe volume — the OS disk — per GB per month (`GET /volume-types`, 2026-10-08).
_DISK_PER_GB_MONTH = 0.20
#: The shorter month, so the hourly figure is never below what is billed.
_HOURS_PER_MONTH = 720.0
#: The tag every instance this plug-in makes carries, for whoever reads the account. The pool finds
#: its own by the label in the description, which takes any text — never by the tag alone.
_TAG = "gpm"
#: Statuses of an instance that is no more: taken back (a spot instance) or not found.
_GONE = frozenset({"discontinued", "notfound"})
#: Statuses of one still coming up.
_COMING = frozenset({"provisioning", "ordered", "new", "validating", "unknown"})
#: A create refused for want of capacity, told from a rule broken by its words only.
_NO_CAPACITY = re.compile(
    r"capacity|not available|unavailable|no (?:free |available )?(?:gpu|machine|instance|host)s?\b|out of stock|sold out",
    re.IGNORECASE,
)


def _problem(response: httpx.Response) -> str:
    """The provider's own words: its `{code, message}`."""
    try:
        said = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(said, dict):
        words = " ".join(str(said.get(k) or "") for k in ("code", "message")).strip()
        return words[:300] or f"HTTP {response.status_code}"
    return json.dumps(said)[:200]


def _says_no_capacity(response: httpx.Response) -> bool:
    """A refusal for want of capacity: the provider's own code, or its words. Any other 503 — a
    gateway's, after the provider may have taken the request — is an answer not known."""
    try:
        said = response.json()
    except ValueError:
        said = None
    if isinstance(said, dict) and said.get("code") == "service_unavailable":
        return True
    return bool(_NO_CAPACITY.search(_problem(response)))


def _hostname(label: str) -> str:
    """A host name the provider takes, from the pool's label: letters, digits and hyphens."""
    name = re.sub(r"[^a-z0-9-]+", "-", label.lower()).strip("-")
    return (name or "gpm-host")[:60].rstrip("-")


def _older_than(stamp: Any, cutoff: datetime.datetime) -> bool:
    """Whether the provider's creation time is before `cutoff`; unreadable is never old."""
    try:
        when = datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return when < cutoff


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class VerdaProvider:
    interface_version: ClassVar[str] = "2"
    name: ClassVar[str] = "verda"
    display_name: ClassVar[str] = "Verda"
    icon: ClassVar[Optional[str]] = None
    icon_url: ClassVar[Optional[str]] = "https://verda.com/favicon.ico"
    endpoint_settings: ClassVar[tuple[str, ...]] = ("base_url",)
    offered: ClassVar[bool] = True
    credential_format: ClassVar[str] = "A Cloud API client's id and secret, as CLIENT_ID:CLIENT_SECRET."
    volume_note: ClassVar[str] = "Not yet: a Verda volume is a raw disk the host would have to format and mount."

    capabilities = ProviderCapabilities(
        #: Spot instances, at the provider's price — each offer says it is not bid for — taken back
        #: without notice.
        interruptible=True,
        #: A shut-down instance keeps billing at its full rate.
        parkable=False,
        same_machine_rebid=False,
        #: No credential narrower than the whole project.
        self_terminate=False,
        #: A balance, and each instance's rate; no charges per instance.
        reports_charges=False,
        price_history=False,
        #: Every port is open: the engine is reached only through the SSH tunnel.
        direct_port_mapping=False,
        reports_instance_logs=False,
        volumes=False,
        machine_ids_are_machines=False,
        copies=False,
        interruption_notice=False,
    )

    def __init__(
        self,
        api_key_env: str = "VERDA_CREDENTIAL",
        base_url: str = _BASE_URL,
        timeout: float = 30.0,
        os_image: str = "ubuntu-22.04-cuda-12.4-docker",
        ssh_port: int = 2222,
        assumed_download_mbps: float = 1000.0,
        assumed_reliability: float = 0.99,
        max_disk_gb: float = 2000.0,
        client: Optional[httpx.AsyncClient] = None,
        **_: Any,
    ):
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        #: The OS image the machine boots: one with Docker and the GPU's container runtime.
        self.os_image = os_image
        #: Where the host container's SSH daemon is published on the machine — the only port that is.
        self.ssh_port = int(ssh_port)
        #: Neither is reported by the provider; each offer says it was assumed.
        self.assumed_download_mbps = float(assumed_download_mbps)
        self.assumed_reliability = float(assumed_reliability)
        self.max_disk_gb = float(max_disk_gb)
        self._credential: Optional[str] = None
        self._handed = False
        self._client = client
        self._timeout = timeout
        #: The access token and when it stops being used — a minute before the provider's expiry.
        self._token: Optional[tuple[str, float]] = None
        #: The provider's `Retry-After`, honoured per operation: its limits are per method and path,
        #: so a search refused must not hold up a delete.
        self._quiet_until: dict[str, float] = {}
        self._instances: dict[str, tuple[float, dict[str, Any]]] = {}
        #: Longer than a probe interval, so one listing serves every status read of a pass: the
        #: provider allows 120 listings a minute, and 60 calls of any other one.
        self.instance_cache_s = 15.0
        #: Stored start-up scripts and OS disks left behind — a spot machine taken back, an
        #: on-demand one the provider ended — are looked for at most this often, and only those
        #: older than `tidy_after_s`, so nothing of a create still in flight is touched.
        self.tidy_interval_s = 600.0
        self.tidy_after_s = 1800.0
        self._tidied = 0.0
        #: Raised by every new credential: a token asked for under an older one is not kept.
        self._generation = 0
        #: The catalog changes when the provider changes its prices, not between passes.
        self._types: Optional[tuple[float, list[dict[str, Any]]]] = None
        self.types_cache_s = 600.0
        self._ssh_keys: Optional[tuple[float, list[str]]] = None
        self.ssh_keys_cache_s = 600.0

    # --- plumbing ---

    def _key(self) -> Optional[str]:
        return self._credential if self._handed else os.environ.get(self.api_key_env)

    def _pair(self) -> tuple[str, str]:
        key = self._key()
        if not key:
            raise ProviderAuthError(
                "no credential is set for this provider: type one in on the Providers screen, or set "
                f"{self.api_key_env} in the supervisor's environment"
            )
        client_id, sep, secret = key.strip().partition(":")
        if not sep or not client_id or not secret:
            raise ProviderAuthError(f"the credential is not in the form CLIENT_ID:CLIENT_SECRET ({self.credential_format})")
        return client_id, secret

    def _quiet(self, text: str) -> str:
        """The provider's own words, with the credential — and the token — taken out."""
        for secret in (self._key(), *(self._key() or "").partition(":")[::2], (self._token or ("", 0))[0]):
            if secret and len(secret) >= 4:
                text = text.replace(secret, "[credential]")
        return text

    @property
    def credential_env(self) -> str:
        return self.api_key_env

    def set_credential(self, credential: Optional[str]) -> None:
        """Use this credential from the next call on (D134): the token made with the old one is
        dropped, and so is the client — closed once whatever uses it has finished."""
        if self._handed and (credential or None) == self._credential:
            return
        self._credential = credential or None
        self._handed = True
        self._generation += 1
        # Nothing of the old account is used again: its token, its SSH keys, its instances.
        self._token = None
        self._ssh_keys = None
        self._instances.clear()
        old, self._client = self._client, None
        if old is not None:
            try:
                asyncio.get_running_loop().call_later(60, lambda: asyncio.ensure_future(old.aclose()))
            except RuntimeError:
                pass

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Accept": "application/json", "User-Agent": "gpm"},
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _operation(method: str, path: str) -> str:
        """The operation a rate limit is kept for: the method and the path's first part."""
        return f"{method} /{path.strip('/').split('/')[0]}"

    def _rate_limited(self, method: str, path: str, response: httpx.Response) -> ProviderRateLimited:
        try:
            wait = max(0.0, float(response.headers.get("retry-after") or 0))
        except ValueError:
            wait = 0.0
        key = self._operation(method, path)
        self._quiet_until[key] = max(self._quiet_until.get(key, 0.0), time.monotonic() + wait)
        exc = ProviderRateLimited(
            f"{method} {path}: rate limited ({self._quiet(_problem(response))})"
            + (f"; the provider asks for {wait:.0f}s before the next call" if wait else "")
        )
        exc.retry_after_s = wait  # type: ignore[attr-defined]
        return exc

    async def _access_token(self) -> str:
        """A token from the client's id and secret, kept until a minute before it expires."""
        if self._token is not None and time.monotonic() < self._token[1]:
            return self._token[0]
        generation = self._generation
        client_id, secret = self._pair()
        path = "/oauth2/token"
        try:
            response = await self.client.post(path, json={
                "grant_type": "client_credentials", "client_id": client_id, "client_secret": secret})
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"POST {path}: {self._quiet(str(exc))}") from exc
        if response.status_code in (400, 401, 403):
            raise ProviderAuthError(f"POST {path}: the account credential was refused ({self._quiet(_problem(response))})")
        if response.status_code == 429:
            raise self._rate_limited("POST", path, response)
        if response.status_code >= 300:
            raise ProviderUnavailable(f"POST {path}: {response.status_code} {self._quiet(_problem(response))}")
        try:
            said = response.json()
            token, expires = str(said["access_token"]), float(said.get("expires_in") or 3600)
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderUnavailable(f"POST {path}: the answer carries no access token") from exc
        if generation == self._generation:
            # Not when the credential changed while this was asked: that token is the old account's.
            self._token = (token, time.monotonic() + max(60.0, expires - 60.0))
        return token

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """One request with a current token, its transport failures and the provider's typed
        answers raised; any other status handed back for the caller to read. A token refused is
        made again once — it may have been revoked before its time."""
        wait = self._quiet_until.get(self._operation(method, path), 0.0) - time.monotonic()
        if wait > 0:
            exc = ProviderRateLimited(f"{method} {path}: the provider asked for {wait:.0f}s more before the next call")
            exc.retry_after_s = wait  # type: ignore[attr-defined]
            raise exc
        for attempt in (1, 2):
            token = await self._access_token()
            try:
                response = await self.client.request(
                    method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"{method} {path}: {self._quiet(str(exc))}") from exc
            if response.status_code == 401 and attempt == 1:
                self._token = None
                continue
            break
        if 300 <= response.status_code < 400:
            raise ProviderUnavailable(
                f"{method} {path}: unexpected redirect to {response.headers.get('location', '?')}"
            )
        if response.status_code == 401:
            raise ProviderAuthError(f"{method} {path}: the account credential was refused")
        if response.status_code == 429:
            raise self._rate_limited(method, path, response)
        return response

    def _fail(self, method: str, path: str, response: httpx.Response) -> ProviderError:
        words = self._quiet(_problem(response))
        if response.status_code == 403:
            return ProviderAuthError(f"{method} {path}: the credential does not allow this ({words})")
        if response.status_code >= 500:
            return ProviderUnavailable(f"{method} {path}: {response.status_code} {words}")
        return ProviderError(f"{method} {path}: {response.status_code} {words}")

    async def _call(self, method: str, path: str, *, missing_ok: bool = False, **kwargs: Any) -> Any:
        """A JSON answer, or a typed error. With `missing_ok`, a 404 is None."""
        response = await self._send(method, path, **kwargs)
        if response.status_code == 404 and missing_ok:
            return None
        if response.status_code >= 400:
            raise self._fail(method, path, response)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderUnavailable(f"{method} {path}: response was not JSON") from exc

    async def _list(self, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        """A listing, or an error: an answer that is not a list is never read as an empty one (D61)."""
        rows = await self._call("GET", path, **kwargs)
        if not isinstance(rows, list):
            raise ProviderUnavailable(f"GET {path}: the answer is not a list; refusing to read it as none")
        return [r for r in rows if isinstance(r, dict)]

    # --- offers ---

    async def _instance_types(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if self._types is not None and now - self._types[0] < self.types_cache_s:
            return self._types[1]
        rows = await self._list("/instance-types", params={"currency": "usd"})
        self._types = (now, rows)
        return rows

    async def search_offers(self, query: OfferQuery) -> list[Offer]:
        """One offer per instance type, location and kind that can be rented now.

        Availability is asked once per kind — spot and on demand are stocked apart — and prices
        are the catalog's, the same in every location."""
        kinds = [spot for spot, wanted in ((True, query.interruptible), (False, query.on_demand)) if wanted]
        if not kinds:
            return []
        # Prices in dollars only: a row in another currency read as dollars would loosen every cap.
        types = {str(t.get("instance_type")): t for t in await self._instance_types()
                 if t.get("instance_type") and str(t.get("currency") or "usd").lower() == "usd"}
        found: list[Offer] = []
        for spot in kinds:
            for entry in await self._list("/instance-availability", params={"is_spot": "true" if spot else "false"}):
                location = str(entry.get("location_code") or "")
                for type_name in entry.get("availabilities") or []:
                    row = types.get(str(type_name))
                    offer = self._to_offer(row, location, spot, query) if row and location else None
                    if offer is not None:
                        found.append(offer)
        found.sort(key=lambda o: (o.all_in_hourly, o.offer_id))
        return found[: max(0, int(query.limit))]

    def _to_offer(self, row: dict[str, Any], location: str, spot: bool, query: OfferQuery) -> Optional[Offer]:
        type_name = str(row["instance_type"])
        gpus = int((row.get("gpu") or {}).get("number_of_gpus") or 0)
        if gpus < max(1, int(query.min_gpus)):
            return None  # a CPU type, or fewer cards than asked for
        memory = float((row.get("gpu_memory") or {}).get("size_in_gigabytes") or 0) / gpus
        if memory < query.min_gpu_memory_gb:
            return None
        listed = _number(row.get("price_per_hour"))
        price = _number(row.get("spot_price")) if spot else listed
        if not price or price <= 0 or not listed:
            return None
        name = str(row.get("name") or row.get("model") or type_name)
        hardware = f"{gpus}x {name}"
        if name in query.exclude_hardware or type_name in query.exclude_hardware or hardware in query.exclude_hardware:
            return None
        machine = f"{location}:{type_name}"
        if machine in query.avoid_machines:
            return None
        per_gb_hourly = _DISK_PER_GB_MONTH / _HOURS_PER_MONTH
        storage = per_gb_hourly * float(query.min_disk_gb or 0)
        all_in = price + storage
        if query.max_all_in_hourly is not None and all_in > query.max_all_in_hourly:
            return None
        return Offer(
            offer_id=f"{'spot' if spot else 'on-demand'}:{machine}",
            machine_id=machine,
            hardware=hardware,
            gpus=gpus,
            gpu_memory_gb=memory,
            disk_gb=self.max_disk_gb,
            # Spot or on demand, the price is the provider's: nothing is bid. A spot price is the
            # machine's alone, as a bid would be — the pool adds the storage beside it (D132).
            min_bid_hourly=price if spot else all_in,
            all_in_hourly=all_in,
            on_demand_hourly=listed + storage,
            storage_hourly=storage,
            storage_per_gb_hourly=per_gb_hourly,
            download_per_gb=0.0,
            download_mbps=self.assumed_download_mbps,
            driver_version=None,
            reliability=self.assumed_reliability,
            # The provider's own data centers: read as verified, an interpretation said to be assumed.
            verified=True,
            throughput_proxy=float(gpus),
            interruptible=spot,
            bidding=False,
            assumed=("download_mbps", "reliability", "verified"),
            raw={**row, "location_code": location},
        )

    # --- creating ---

    def vm_script(self, spec: InstanceSpec) -> str:
        """The machine's start-up script: the engine image run as the one container that is the
        host, with its SSH daemon published on `ssh_port` and nothing else. The pool's own start-up
        — its key, its engine — runs inside, after the daemon, as it does on a pod."""
        if not spec.image or spec.image.startswith("-") or any(c.isspace() for c in spec.image):
            raise ProviderError(f"not an image name: {spec.image!r}")
        inside = f"{SSH_DAEMON}\n{spec.onstart or ''}\nwhile :; do sleep 3600; done\n"
        # Written out from base64: no line of the operator's start-up can end it early and run on
        # the machine itself.
        encoded = base64.b64encode(inside.encode()).decode()
        env = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in sorted(spec.env.items()))
        return "\n".join([
            "#!/bin/bash",
            "# GPM: the engine image as the one container that is the host; only its SSH port published.",
            "set -u",
            "mkdir -p /opt/gpm",
            f"echo '{encoded}' | base64 -d > /opt/gpm/host-start.sh",
            "for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 5; done",
            "docker rm -f gpm-host >/dev/null 2>&1 || true",
            f"docker run -d --name gpm-host --restart unless-stopped --gpus all --ipc host "
            f"-p {self.ssh_port}:22 -v /opt/gpm/host-start.sh:/opt/gpm-host-start.sh:ro {env} "
            f"--entrypoint /bin/sh {shlex.quote(spec.image)} /opt/gpm-host-start.sh "
            ">>/var/log/gpm-host.log 2>&1",
        ])

    async def _account_ssh_keys(self) -> list[str]:
        """The account's SSH keys, by id: the provider asks for one to open the machine. The pool
        reaches its host — the container — with its own key, put there by the start-up script."""
        now = time.monotonic()
        if self._ssh_keys is not None and now - self._ssh_keys[0] < self.ssh_keys_cache_s:
            return self._ssh_keys[1]
        keys = [str(k["id"]) for k in await self._list("/ssh-keys") if k.get("id")]
        self._ssh_keys = (now, keys)
        return keys

    async def create(self, offer: Offer, spec: InstanceSpec, bid: Optional[float]) -> Instance:
        """The machine, created and starting, or an error with nothing left behind. A spot rental's
        `bid` is the most the pool would pay; the provider takes no maximum, so it is not sent —
        the pool reads the instance's rate each pass instead (D132)."""
        kind, _, machine = offer.offer_id.partition(":")
        location, _, type_name = machine.partition(":")
        if kind not in ("spot", "on-demand") or not location or not type_name:
            raise ProviderError(f"not an offer from this provider: {offer.offer_id!r}")
        if spec.volume is not None:
            raise ProviderError("Verda volumes are not supported by this plug-in yet")
        keys = await self._account_ssh_keys()
        if not keys:
            raise ProviderError(
                "the Verda account has no SSH key, and the provider creates no machine without one: add any "
                "key in its console (the pool reaches its hosts with its own)"
            )
        script_id = await self._add_script(spec.label, self.vm_script(spec))
        spot = kind == "spot"
        body: dict[str, Any] = {
            "instance_type": type_name,
            "image": self.os_image,
            "ssh_key_ids": keys,
            "startup_script_id": script_id,
            "hostname": _hostname(spec.label),
            "description": spec.label,
            "tags": [{"key": _TAG}],
            "location_code": location,
            # Deleted with the instance — and when a spot instance is taken back — never left
            # detached, billing, which is the provider's default.
            "os_volume": {"name": _hostname(spec.label)[:50] + "-os", "size": max(1, math.ceil(spec.disk_gb)),
                          "on_spot_discontinue": "delete_permanently"},
            "is_spot": spot,
            "contract": "SPOT" if spot else "PAY_AS_YOU_GO",
        }
        place = f"{offer.hardware} in {location}" + (" (spot)" if spot else "")
        path = "/instances"
        try:
            try:
                response = await self._send("POST", path, json=body)
            except ProviderUnavailable as exc:
                # No answer is not a refusal: the instance may exist. Nothing may be left behind.
                raise await self._after_unclear(spec.label, exc) from exc
            if response.status_code in (400, 409, 422) or (response.status_code == 503 and _says_no_capacity(response)):
                words = self._quiet(_problem(response))
                if _says_no_capacity(response):
                    raise OfferGone(f"{place}: no capacity now ({words})")
                raise ProviderError(f"POST {path}: the provider refused the request ({words})")
            if response.status_code == 402:
                raise ProviderError(
                    f"insufficient balance: the provider will not create an instance ({self._quiet(_problem(response))})")
            if response.status_code == 403:
                raise ProviderError(
                    f"{place}: this account may not create it ({self._quiet(_problem(response))}) — "
                    "a GPU quota, which the provider raises on request, refuses this way")
            if response.status_code >= 500:
                raise await self._after_unclear(spec.label, self._fail("POST", path, response))
            if response.status_code >= 400:
                raise self._fail("POST", path, response)
            instance_id = self._created_id(response)
            if instance_id is None:
                raise await self._after_unclear(spec.label, ProviderUnavailable(f"POST {path}: the answer names no instance"))
        except (Exception, asyncio.CancelledError):
            await self._delete_script(script_id)
            raise
        return Instance(instance_id=instance_id, label=spec.label, machine_id=offer.machine_id,
                        raw={"id": instance_id, "startup_script_id": script_id}, location=location)

    @staticmethod
    def _created_id(response: httpx.Response) -> Optional[str]:
        """The instance's id: the answer is the id itself, as JSON text or bare."""
        try:
            said = response.json()
        except ValueError:
            said = response.text.strip()
        if isinstance(said, dict):
            said = said.get("id")
        said = str(said or "").strip()
        return said or None

    async def _add_script(self, label: str, script: str) -> str:
        """The start-up script, stored with the provider: it holds no secret — the pool's public
        key and the engine's settings — and is deleted with its instance."""
        said = await self._call("POST", "/scripts", json={"name": label, "script": script})
        script_id = said.get("id") if isinstance(said, dict) else said
        if not script_id:
            raise ProviderUnavailable("POST /scripts: the answer names no script")
        return str(script_id)

    async def _delete_script(self, script_id: Optional[str]) -> None:
        if not script_id:
            return
        try:
            response = await self._send("DELETE", f"/scripts/{script_id}")
            if response.status_code >= 400 and response.status_code != 404:
                log.warning("start-up script %s was not deleted: %s", script_id, self._quiet(_problem(response)))
        except ProviderError as exc:
            log.warning("start-up script %s was not deleted: %s", script_id, exc)

    async def _after_unclear(self, label: str, cause: ProviderError) -> ProviderError:
        """After a create whose outcome is not known: end whatever the label names, then report
        the cause. If that cannot be proved, say so — "could not check" is never "nothing there"."""
        try:
            ended = []
            for instance in await self.list_instances(label):
                if instance.label == label:
                    await self.destroy(instance)
                    ended.append(instance.instance_id)
        except ProviderError as exc:
            return ProviderUnavailable(f"{cause}; and whether an instance was left behind could not be checked: {exc}")
        if ended:
            return type(cause)(f"{cause} — but created {', '.join(ended)}, which has been deleted")
        return cause

    # --- state ---

    def _remember(self, entry: dict[str, Any]) -> None:
        if entry.get("id") is not None:
            self._instances[str(entry["id"])] = (time.monotonic(), entry)

    async def _instance(self, instance_id: str) -> dict[str, Any]:
        cached = self._instances.get(instance_id)
        if cached is not None and time.monotonic() - cached[0] < self.instance_cache_s:
            return cached[1]
        entry = await self._call("GET", f"/instances/{instance_id}", missing_ok=True)
        if entry is None:
            self._instances.pop(instance_id, None)
            return {}  # the provider does not know it: gone
        if not isinstance(entry, dict) or not entry.get("id"):
            # Could not read is never gone (D61): a host taken for gone is dropped without a delete.
            raise ProviderUnavailable(f"GET /instances/{instance_id}: the answer is not an instance")
        self._remember(entry)
        return entry

    async def list_instances(self, label_prefix: str) -> list[Instance]:
        """Every instance this plug-in made whose label carries the prefix, in any state that
        still bills. One taken back or not found is gone and is not listed."""
        found = []
        # Every instance, filtered here by its label: one whose tag did not stick still bills, and
        # must not be invisible to the sweep.
        for entry in await self._list("/instances"):
            label = str(entry.get("description") or "")
            if not label.startswith(label_prefix) or str(entry.get("status") or "") in _GONE:
                continue
            self._remember(entry)
            location = str(entry.get("location") or "") or None
            machine = f"{location}:{entry['instance_type']}" if location and entry.get("instance_type") else None
            found.append(Instance(instance_id=str(entry.get("id")), label=label, machine_id=machine, raw=entry,
                                  location=location))
        await self._tidy(label_prefix, found)
        return found

    async def _tidy(self, label_prefix: str, live: list[Instance]) -> None:
        """Delete what a machine the pool never deleted leaves behind: its stored start-up script
        and, where the provider ended an on-demand machine itself, its detached OS disk, which
        bills. Only this pool's, only what no live instance uses, only what is old enough that no
        create in flight can own it. Best effort, never failing the listing it rides on."""
        now = time.monotonic()
        if now - self._tidied < self.tidy_interval_s:
            return
        self._tidied = now
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=self.tidy_after_s)
        used = {str(i.raw.get("startup_script_id")) for i in live if i.raw.get("startup_script_id")}
        try:
            for script in await self._list("/scripts", params={"name": label_prefix, "pageSize": 100}):
                if (str(script.get("name") or "").startswith(label_prefix) and str(script.get("id")) not in used
                        and _older_than(script.get("created_at"), cutoff)):
                    await self._delete_script(str(script["id"]))
            disk_prefix = _hostname(label_prefix) + "-"
            for volume in await self._list("/volumes", params={"status": "detached"}):
                name = str(volume.get("name") or "")
                if (volume.get("is_os_volume") and volume.get("status") == "detached" and name.startswith(disk_prefix)
                        and name.endswith("-os") and _older_than(volume.get("created_at"), cutoff)):
                    response = await self._send("DELETE", f"/volumes/{volume['id']}", json={"is_permanent": True})
                    if response.status_code >= 400 and response.status_code != 404:
                        log.warning("OS disk %s left behind was not deleted: %s", volume["id"],
                                    self._quiet(_problem(response)))
                    else:
                        log.info("deleted OS disk %s (%s), left detached by a machine that is gone", volume["id"], name)
        except ProviderError as exc:
            log.warning("what gone machines left behind was not tidied this time: %s", exc)

    async def status(self, instance: Instance) -> InstanceStatus:
        entry = await self._instance(instance.instance_id)
        if not entry:
            return InstanceStatus(state=InstanceState.GONE)
        said = str(entry.get("status") or "")
        if said == "running":
            state = InstanceState.RUNNING
        elif said in _GONE:
            state = InstanceState.GONE
        elif said in _COMING:
            state = InstanceState.SCHEDULING
        else:  # offline, error, no_capacity, installation_failed, deleting, or a word not known yet
            state = InstanceState.STOPPED
        return InstanceStatus(
            state=state,
            # A spot instance discontinued was taken back by the provider; nothing else says who.
            stopped_by_provider=True if said == "discontinued" else None,
            # What it bills now: a spot price is read again every pass (D132).
            bid_hourly=_number(entry.get("price_per_hour")),
            detail=f"status {said or '?'}",
            # Not read from `startup_script_id`: whether a running instance still names its script
            # is not known, and a False here ends the host (D65). Unverified until a live check.
            startup_material=None,
        )

    async def connection(self, instance: Instance) -> ConnectionInfo:
        """SSH to the host container's daemon, on the machine's public address."""
        entry = await self._instance(instance.instance_id)
        if entry.get("ip"):
            return ConnectionInfo(ssh_host=str(entry["ip"]), ssh_port=self.ssh_port, ssh_user="root")
        return ConnectionInfo()

    async def set_bid(self, instance: Instance, bid: float) -> None:
        raise ProviderError("Verda sets its own prices: there is no bid to change")

    async def start(self, instance: Instance) -> None:
        raise ProviderError("a Verda instance is never parked: a shut-down one keeps billing in full")

    async def stop(self, instance: Instance) -> None:
        raise ProviderError("a Verda instance is never parked: a shut-down one keeps billing in full")

    async def destroy(self, instance: Instance) -> None:
        """Delete the instance and its OS volume for good, then its start-up script. Idempotent:
        an instance the provider no longer knows is already gone."""
        entry = await self._instance(instance.instance_id)
        if not entry or str(entry.get("status") or "") in _GONE:
            await self._delete_script(instance.raw.get("startup_script_id") or entry.get("startup_script_id"))
            return
        if str(entry.get("status") or "") == "deleting":
            return  # already asked; asked again it is refused, every pass
        body: dict[str, Any] = {"action": "delete", "id": instance.instance_id, "delete_permanently": True}
        if entry.get("os_volume_id"):
            body["volume_ids"] = [str(entry["os_volume_id"])]
        path = "/instances"
        response = await self._send("PUT", path, json=body)
        self._instances.pop(instance.instance_id, None)
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise self._fail("PUT", path, response)
        try:
            results = response.json()
        except ValueError:
            results = []
        for result in results if isinstance(results, list) else []:
            if isinstance(result, dict) and result.get("status") == "error" and int(result.get("statusCode") or 0) != 404:
                raise ProviderError(f"PUT {path}: instance {instance.instance_id} was not deleted "
                                    f"({self._quiet(str(result.get('error') or 'error'))})")
        await self._delete_script(entry.get("startup_script_id"))

    async def instance_logs(self, instance: Instance, tail: int = 60) -> Optional[str]:
        return None

    # --- volumes: not yet ---

    async def list_volumes(self, label_prefix: str) -> list[VolumeInfo]:
        return []

    async def delete_volume(self, volume_id: str) -> None:
        raise ProviderError("Verda volumes are not supported by this plug-in yet")

    # --- money ---

    async def reported_charges(self, instance: Instance) -> Optional[Charges]:
        return None

    async def account(self) -> AccountStatus:
        """The credential checked by making a token; the balance from the account call."""
        said = await self._call("GET", "/balance")
        amount = _number(said.get("amount")) if isinstance(said, dict) else None
        currency = str(said.get("currency") or "usd").lower() if isinstance(said, dict) else "usd"
        if currency != "usd":
            return AccountStatus(credential_valid=True, credit_remaining=None,
                                 detail=f"the balance is in {currency.upper()}, not dollars: {amount}")
        return AccountStatus(credential_valid=True, credit_remaining=amount,
                             detail=None if amount is not None else "the balance was not in the answer")

    def self_terminate_request(self, action: str = "destroy") -> SelfTerminateRequest:
        raise ProviderError("a Verda instance holds no credential that ends only itself: no dead-man timer here")
