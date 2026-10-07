"""RunPod — on-demand pods, through the provider's REST API (v2).

An HTTP API client with typed errors and bounded timeouts, never a CLI wrapper
(docs/spec/plugin-interfaces.md §1). The account credential is handed in by the supervisor (or,
outside a pool, read from the environment) and **never leaves this process**: what a pod holds is
the pod-scoped key the provider injects there itself (`RUNPOD_API_KEY` *on the pod*).

Endpoint and field shapes come from the provider's published OpenAPI document for
`https://api.runpod.io/v2` and its documentation pages (read 2026-10-07). What they leave open
is said where it is decided:

- **No machine before renting.** The catalog lists GPU types per cloud, not machines, so an
  offer is one GPU type, in one cloud, at one card count, and its `machine_id` is
  `CLOUD:gpu-type:count`. Machine history therefore rates a GPU type in a cloud, not a host.
- **No bidding.** Interruptible pods are gone from the API: every offer is on demand, at the
  catalog price, and `create` takes no price.
- **The container disk is wiped when a pod stops** (pricing and storage pages). What must survive
  a park — the models — goes on the pod's persistent volume, mounted at `volume_path`; the disk
  the pool asks for is split between the two, so the storage billed is the storage priced.
- **The balance is not in v2.** It is read from the provider's GraphQL API (`myself {
  clientBalance }`), only there, and a failure to read it never marks the credential invalid.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import logging
import math
import os
import re
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
    redacted,
)

log = logging.getLogger("gpm.runpod")

_BASE_URL = "https://api.runpod.io/v2"
_GRAPHQL_URL = "https://api.runpod.io/graphql"
_CLOUDS = ("SECURE", "COMMUNITY")
#: Container and volume disk on a running pod, per GB per month (pricing page, 2026-10-07). A
#: stopped pod bills its volume disk at $0.20 and its container disk not at all.
_DISK_PER_GB_MONTH = 0.10
#: The provider bills storage per second at a monthly rate and does not say how many hours its
#: month has. 720 is the shorter month, so the hourly figure is never below what is billed.
_HOURS_PER_MONTH = 720.0
#: The provider's own floor on a persistent volume (`PersistentMount.size`).
_MIN_VOLUME_GB = 10
#: RunPod runs the image as it is: unlike a provider that boots its own SSH daemon into every
#: instance, it starts one only where the image does, and the pool reaches its hosts over SSH.
#: So the start-up script first makes sure one runs — installed from the image's own package
#: manager when missing, key login only (the pool's key is put in place by the script itself) —
#: in the background, so a slow install never holds up the engine. Unverified on a live pod.
_SSH_DAEMON = (
    "( if ! command -v sshd >/dev/null 2>&1 && [ ! -x /usr/sbin/sshd ]; then "
    "{ apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; } "
    "|| apk add --no-cache openssh-server; fi; "
    "mkdir -p /run/sshd && ssh-keygen -A && "
    "/usr/sbin/sshd -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no "
    "-o PermitRootLogin=prohibit-password ) >/var/log/gpm-sshd.log 2>&1 &"
)
#: A 400 on create means either "this request breaks a rule" or "no capacity", told apart only by
#: its human-readable `detail` (the create operation's own documentation). These are the words a
#: capacity refusal is read from; anything else is a rule the request broke, which no other
#: offer would fix. Unverified against a live refusal.
_NO_CAPACITY = re.compile(
    r"capacity|no longer available|not available|unavailable|no (?:free |available )?(?:gpu|machine|instance|host)s?\b"
    r"|out of stock|sold out|could not be placed|cannot be placed|could not find|no .*\bavailable",
    re.IGNORECASE,
)


def _problem(response: httpx.Response) -> str:
    """An RFC 9457 problem's own words: its `detail`, with each validation failure listed."""
    try:
        said = response.json()
    except ValueError:
        return response.text[:200]
    if not isinstance(said, dict):
        return json.dumps(said)[:200]
    words = str(said.get("detail") or said.get("title") or "").strip()
    errors = said.get("errors")
    if isinstance(errors, list) and errors:
        words = f"{words} ({'; '.join(str(e) for e in errors[:5])})" if words else "; ".join(str(e) for e in errors[:5])
    return words[:300] or f"HTTP {response.status_code}"


class RunPodProvider:
    interface_version: ClassVar[str] = "2"
    name: ClassVar[str] = "runpod"
    display_name: ClassVar[str] = "RunPod"
    icon: ClassVar[Optional[str]] = None
    #: The provider's own logo, as its site declares it: loaded by the console's page, which falls
    #: back to a lettermark if it does not load (D134).
    icon_url: ClassVar[Optional[str]] = (
        "https://cdn.prod.website-files.com/69ce570adca53340abab8376/"
        "69e08b4ced16e020149fc61d_favicon_www_runpod_io_256x256.png"
    )
    #: Where the credential is sent. A stored credential is bound to them: changed, it is cleared.
    endpoint_settings: ClassVar[tuple[str, ...]] = ("base_url", "graphql_url")
    offered: ClassVar[bool] = True

    capabilities = ProviderCapabilities(
        #: Spot ("interruptible") pods are no longer offered: every pod is on demand.
        interruptible=False,
        #: `stop` releases the GPU and keeps the volume disk; `start` boots the pod back on the
        #: same machine — and fails if its GPU has been taken meanwhile.
        parkable=True,
        same_machine_rebid=False,
        #: Checked live (2026-10-07): the pod-scoped key RunPod puts in every pod (`RUNPOD_API_KEY`)
        #: deleted its own pod through v2 (`DELETE /v2/pods/$RUNPOD_POD_ID`, 204), and was refused
        #: listing the account's pods, its volumes and creating a pod (403).
        self_terminate=True,
        #: `GET /v2/billing/pods`: per-pod amounts in time buckets.
        reports_charges=True,
        price_history=False,
        direct_port_mapping=True,
        #: `GET /v2/pods/{id}/logs`: the container's and the system's output, as server-sent events.
        reports_instance_logs=True,
        volumes=False,
        copies=False,
        interruption_notice=False,
    )

    def __init__(
        self,
        api_key_env: str = "RUNPOD_API_KEY",
        base_url: str = _BASE_URL,
        graphql_url: str = _GRAPHQL_URL,
        timeout: float = 30.0,
        clouds: tuple[str, ...] | list[str] = _CLOUDS,
        assumed_download_mbps: float = 1000.0,
        assumed_reliability: float = 0.99,
        max_disk_gb: float = 1000.0,
        container_disk_gb: float = 20.0,
        volume_path: Optional[str] = "/opt/gpm/models",
        client: Optional[httpx.AsyncClient] = None,
        **_: Any,
    ):
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.graphql_url = graphql_url
        self.clouds = tuple(str(c).upper() for c in clouds if str(c).upper() in _CLOUDS)
        #: Neither is reported by the provider; each offer says it was assumed.
        self.assumed_download_mbps = float(assumed_download_mbps)
        self.assumed_reliability = float(assumed_reliability)
        #: The catalog states no disk limit per GPU type; this is the most an offer claims.
        self.max_disk_gb = float(max_disk_gb)
        #: Of the disk the pool asks for, what the container keeps — wiped at every stop. The rest
        #: is the persistent volume at `volume_path`, which a park keeps.
        self.container_disk_gb = float(container_disk_gb)
        self.volume_path = volume_path or None
        if self.volume_path is None:
            # Everything on the container disk, which a stop wipes: nothing to park with.
            self.capabilities = dataclasses.replace(type(self).capabilities, parkable=False)
        #: The credential the supervisor handed in (D134); until it does, the environment's — a
        #: provider built by hand, outside a pool, still works.
        self._credential: Optional[str] = None
        self._handed = False
        self._client = client
        self._timeout = timeout
        #: Pod payloads seen recently, by id: status and connection come from the same payload,
        #: and within one control-loop pass the answer does not change.
        self._instances: dict[str, tuple[float, dict[str, Any]]] = {}
        self.instance_cache_s = 10.0
        #: The account's pod billing, fetched once per pass and shared by every pod.
        self._charges: Optional[tuple[float, dict[str, float]]] = None
        self.charges_window_days = 45
        #: A guard on the paginated pod listing: better to refuse than to act on half of it.
        self.max_instance_pages = 50
        #: The provider's `Retry-After`, honoured: until then no call is made.
        self._quiet_until = 0.0
        #: How long a log read listens: the stream stays open after its backfill.
        self.logs_window_s = 5.0

    # --- plumbing ---

    def _key(self) -> Optional[str]:
        return self._credential if self._handed else os.environ.get(self.api_key_env)

    def _quiet(self, text: str) -> str:
        """The provider's own words, with the credential taken out should an answer echo it."""
        key = self._key()
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
            key = self._key()
            if not key:
                raise ProviderAuthError(
                    "no credential is set for this provider: type one in on the Providers screen, or set "
                    f"{self.api_key_env} in the supervisor's environment"
                )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _rate_limited(self, method: str, path: str, response: httpx.Response) -> ProviderRateLimited:
        try:
            wait = max(0.0, float(response.headers.get("retry-after") or 0))
        except ValueError:
            wait = 0.0
        self._quiet_until = max(self._quiet_until, time.monotonic() + wait)
        exc = ProviderRateLimited(
            f"{method} {path}: rate limited ({self._quiet(_problem(response))})"
            + (f"; the provider asks for {wait:.0f}s before the next call" if wait else "")
        )
        exc.retry_after_s = wait  # type: ignore[attr-defined]
        return exc

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """One request, its transport failures and the provider's typed answers raised; any other
        status is handed back for the caller to read."""
        wait = self._quiet_until - time.monotonic()
        if wait > 0:
            exc = ProviderRateLimited(f"{method} {path}: the provider asked for {wait:.0f}s more before the next call")
            exc.retry_after_s = wait  # type: ignore[attr-defined]
            raise exc
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{method} {path}: {self._quiet(str(exc))}") from exc
        if 300 <= response.status_code < 400:
            # Never quietly read a redirect's own body as the answer (D43).
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

    # --- offers ---

    async def search_offers(self, query: OfferQuery) -> list[Offer]:
        """One offer per GPU type and cloud that can be rented now with the cards asked for.

        Asked once per cloud: the catalog's `availability` is computed for one cloud and one
        card count (`cloud`, `count`), so a single answer cannot speak for the other cloud.
        """
        if not query.on_demand:
            return []  # nothing here can be bid for
        gpus = max(1, int(query.min_gpus))
        found: list[Offer] = []
        for cloud in self.clouds:
            if query.verified_only and cloud != "SECURE":
                continue
            payload = await self._call(
                "GET", "/catalog/gpus",
                params={"include": "AVAILABILITY", "product": "POD", "cloud": cloud, "count": gpus},
            )
            rows = payload.get("gpus") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ProviderUnavailable("GET /catalog/gpus: the answer carries no `gpus` list")
            for row in rows:
                offer = self._to_offer(row, cloud, gpus, query)
                if offer is not None:
                    found.append(offer)
        found.sort(key=lambda o: (o.all_in_hourly, o.offer_id))
        return found[: max(0, int(query.limit))]

    def _to_offer(self, row: dict[str, Any], cloud: str, gpus: int, query: OfferQuery) -> Optional[Offer]:
        side = cloud.lower()
        gpu_id = str(row.get("id") or "")
        name = str(row.get("name") or gpu_id)
        if not gpu_id or not row.get(side):
            return None
        if int((row.get("maxCount") or {}).get(side) or 0) < gpus:
            return None
        if str(row.get("availability") or "NONE").upper() == "NONE":
            return None  # absent is read as none: availability was asked for
        price = float((row.get("price") or {}).get(side) or 0)
        memory = float(row.get("memory") or 0)
        if price <= 0 or memory < query.min_gpu_memory_gb:
            return None
        hardware = f"{gpus}x {name}"
        if name in query.exclude_hardware or gpu_id in query.exclude_hardware:
            return None
        machine = f"{cloud}:{gpu_id}:{gpus}"
        if machine in query.avoid_machines:
            return None
        per_gb_hourly = _DISK_PER_GB_MONTH / _HOURS_PER_MONTH
        storage = per_gb_hourly * float(query.min_disk_gb or 0)
        all_in = price * gpus + storage
        if query.max_all_in_hourly is not None and all_in > query.max_all_in_hourly:
            return None
        return Offer(
            offer_id=machine,
            machine_id=machine,
            hardware=hardware,
            gpus=gpus,
            gpu_memory_gb=memory,
            disk_gb=self.max_disk_gb,
            # On demand: the price is the price, and its own ceiling.
            min_bid_hourly=all_in,
            all_in_hourly=all_in,
            on_demand_hourly=all_in,
            storage_hourly=storage,
            storage_per_gb_hourly=per_gb_hourly,
            # "No fees for data ingress or egress" (pricing and billing pages).
            download_per_gb=0.0,
            download_mbps=self.assumed_download_mbps,
            # The catalog reports CUDA versions per GPU type (in `raw`), not a driver.
            driver_version=None,
            reliability=self.assumed_reliability,
            # Secure cloud is the provider's own data-centre hardware; community is hosted by
            # others. Read as "verified" — an interpretation, so it is said to be assumed.
            verified=cloud == "SECURE",
            throughput_proxy=float(gpus),
            interruptible=False,
            bidding=False,
            assumed=("download_mbps", "reliability", "verified"),
            raw=row,
        )

    # --- creating ---

    def _split_disk(self, disk_gb: float) -> tuple[int, Optional[int]]:
        """The disk asked for, as (container disk, persistent volume) in whole GB."""
        total = max(1, math.ceil(disk_gb))
        if self.volume_path is None:
            return total, None
        volume = max(_MIN_VOLUME_GB, total - math.ceil(self.container_disk_gb))
        return max(1, total - volume), volume

    async def create(self, offer: Offer, spec: InstanceSpec, bid: Optional[float]) -> Instance:
        if bid is not None:
            raise ProviderError("RunPod rents on demand only: a pod is created at its listed price, never with a bid")
        if spec.volume is not None:
            raise ProviderError("RunPod volumes are not supported by this provider")
        cloud, _, rest = offer.offer_id.partition(":")
        gpu_id, _, count = rest.rpartition(":")
        if cloud not in _CLOUDS or not gpu_id or not count.isdigit():
            raise ProviderError(f"not an offer from this provider: {offer.offer_id!r}")

        container_gb, volume_gb = self._split_disk(spec.disk_gb)
        ports = [f"{int(p)}/tcp" for p in spec.ports]
        if "22/tcp" not in ports:
            ports.append("22/tcp")  # the pool reaches the host over SSH
        body: dict[str, Any] = {
            "name": spec.label,
            "image": spec.image,
            "cloud": cloud,
            "gpu": {"id": gpu_id, "count": int(count)},
            "disk": container_gb,
            "ports": ports,
        }
        if volume_gb is not None:
            body["mounts"] = {"persistent": {"size": volume_gb, "path": self.volume_path}}
        if spec.env:
            body["env"] = {str(k): str(v) for k, v in spec.env.items()}
        if spec.onstart:
            # The start-up script is the container's command. Its last line keeps the container
            # alive: a pod whose command returns is `EXITED`, and the script's work (the timer,
            # the engine) runs in the background.
            body["entrypoint"] = ["/bin/sh", "-c"]
            body["cmd"] = [f"{_SSH_DAEMON}\n{spec.onstart}\nwhile :; do sleep 3600; done"]

        path = "/pods"
        try:
            response = await self._send("POST", path, json=body)
        except ProviderUnavailable as exc:
            # No answer is not a refusal: the pod may exist. Nothing may be left behind.
            raise await self._after_unclear(spec.label, exc) from exc

        if response.status_code == 400:
            words = self._quiet(_problem(response))
            if _NO_CAPACITY.search(words):
                raise OfferGone(f"{offer.hardware} in {cloud} cloud: no capacity now ({words})")
            raise ProviderError(f"POST {path}: the provider refused the request ({words})")
        if response.status_code == 402:
            raise ProviderError(
                f"insufficient balance: the provider will not create a pod ({self._quiet(_problem(response))})"
            )
        if response.status_code == 403:
            # The create operation's own documentation: this account cannot use that GPU pool —
            # skip it and keep going. A credential without write access answers the same way.
            raise OfferGone(
                f"{offer.hardware} in {cloud} cloud: this account may not create it "
                f"({self._quiet(_problem(response))}; a read-only credential is also refused this way)"
            )
        if response.status_code >= 500:
            raise await self._after_unclear(spec.label, self._fail("POST", path, response))
        if response.status_code >= 400:
            raise self._fail("POST", path, response)

        try:
            pod = response.json()
        except ValueError:
            pod = None
        if not isinstance(pod, dict) or not pod.get("id"):
            raise await self._after_unclear(spec.label, ProviderUnavailable(f"POST {path}: the answer names no pod"))
        pod_id = str(pod["id"])
        status = str(pod.get("status") or "").upper()
        if status in ("ERROR", "TERMINATED", "EXITED"):
            # Created and not starting: ended now, rather than handed back half-made.
            try:
                await self.destroy(Instance(pod_id, label=spec.label))
            except ProviderError as exc:
                raise ProviderUnavailable(
                    f"pod {pod_id} was created {status.lower()} and could not be destroyed: {exc}"
                ) from exc
            raise ProviderError(f"pod {pod_id} was created {status.lower()}, and has been destroyed")
        self._remember(pod)
        return Instance(instance_id=pod_id, label=spec.label, machine_id=offer.machine_id, raw=pod)

    async def _after_unclear(self, label: str, cause: ProviderError) -> ProviderError:
        """After a create whose outcome is not known: end whatever the label names, then report
        the cause. If that cannot be proved, say so — "could not check" is never "nothing there"."""
        try:
            ended = await self._destroy_by_label(label)
        except ProviderError as exc:
            return ProviderUnavailable(f"{cause}; and whether a pod was left behind could not be checked: {exc}")
        if ended:
            return type(cause)(f"{cause} — but created {', '.join(ended)}, which has been destroyed")
        return cause

    async def _destroy_by_label(self, label: str) -> list[str]:
        ended = []
        for instance in await self.list_instances(label):
            if instance.label == label:
                await self.destroy(instance)
                ended.append(instance.instance_id)
        return ended

    # --- state ---

    def _remember(self, entry: dict[str, Any]) -> None:
        if entry.get("id") is not None:
            self._instances[str(entry["id"])] = (time.monotonic(), entry)

    async def _pod(self, instance_id: str) -> dict[str, Any]:
        cached = self._instances.get(instance_id)
        if cached is not None and time.monotonic() - cached[0] < self.instance_cache_s:
            return cached[1]
        entry = await self._call("GET", f"/pods/{instance_id}", missing_ok=True)
        if not isinstance(entry, dict) or not entry.get("id"):
            self._instances.pop(instance_id, None)
            return {}
        self._remember(entry)
        return entry

    @staticmethod
    def _machine_of(entry: dict[str, Any]) -> Optional[str]:
        gpu = entry.get("gpu") or {}
        if not entry.get("cloud") or not gpu.get("id"):
            return None
        return f"{entry['cloud']}:{gpu['id']}:{int(gpu.get('count') or 1)}"

    async def list_instances(self, label_prefix: str) -> list[Instance]:
        """Every pod on the account whose name carries this prefix, running or stopped.

        The listing has no name filter, so every page is read and filtered here; a listing that
        does not end, or a page that is not a list of pods, is refused rather than half believed.
        A `TERMINATED` pod is gone and is not listed: the pool verifies a destroy by listing again.
        """
        entries: list[dict[str, Any]] = []
        params: dict[str, Any] = {"limit": 1000}
        for _ in range(self.max_instance_pages):
            payload = await self._call("GET", "/pods", params=params)
            pods = payload.get("pods") if isinstance(payload, dict) else None
            if not isinstance(pods, list):
                raise ProviderUnavailable("GET /pods: the answer carries no `pods` list; refusing to read it as none")
            entries.extend(p for p in pods if isinstance(p, dict))
            page = payload.get("pagination") or {}
            if not page.get("hasNextPage"):
                break
            cursor = page.get("nextCursor")
            if not cursor:
                raise ProviderUnavailable("GET /pods: more pages are said to follow, with no cursor to them")
            params = {**params, "cursor": cursor}
        else:
            raise ProviderUnavailable(
                f"the pod listing did not end after {self.max_instance_pages} pages; "
                "refusing to act on a partial list of what is running"
            )

        found = []
        for entry in entries:
            label = str(entry.get("name") or "")
            if not label.startswith(label_prefix) or str(entry.get("status") or "").upper() == "TERMINATED":
                continue
            self._remember(entry)
            found.append(Instance(instance_id=str(entry.get("id")), label=label,
                                  machine_id=self._machine_of(entry), raw=entry))
        return found

    async def status(self, instance: Instance) -> InstanceStatus:
        entry = await self._pod(instance.instance_id)
        if not entry:
            return InstanceStatus(state=InstanceState.GONE)
        said = str(entry.get("status") or "").upper()
        if said == "RUNNING":
            state = InstanceState.RUNNING
        elif said in ("EXITED", "ERROR"):
            state = InstanceState.STOPPED
        elif said == "TERMINATED":
            state = InstanceState.GONE
        else:  # PROVISIONING, STARTING, or a word this client does not know yet
            state = InstanceState.SCHEDULING
        cost = entry.get("cost")
        return InstanceStatus(
            state=state,
            # The provider keeps no record of intent: whether the pool or the provider stopped it
            # is the supervisor's own record to say (supervisor.md §1.2).
            stopped_by_provider=None,
            bid_hourly=float(cost) if cost else None,
            detail=f"status {said or '?'}",
            # `args` is the container's command as stored: the start-up script, where one was given.
            startup_material=bool(entry["args"]) if "args" in entry else None,
        )

    async def connection(self, instance: Instance) -> ConnectionInfo:
        """SSH straight to the pod's published `22/tcp` — never the provider's SSH proxy, which
        carries an interactive shell only and no forwarding."""
        entry = await self._pod(instance.instance_id)
        direct = (entry.get("ssh") or {}).get("direct") or {}
        if direct.get("host") and direct.get("port"):
            return ConnectionInfo(ssh_host=str(direct["host"]), ssh_port=int(direct["port"]),
                                  ssh_user=str(direct.get("username") or "root"))
        for port in (entry.get("runtime") or {}).get("ports") or []:
            if port.get("private") == 22 and port.get("public") and port.get("ip"):
                return ConnectionInfo(ssh_host=str(port["ip"]), ssh_port=int(port["public"]), ssh_user="root")
        return ConnectionInfo()

    async def set_bid(self, instance: Instance, bid: float) -> None:
        raise ProviderError("RunPod rents on demand only: there is no bid to change")

    async def _action(self, instance: Instance, action: str) -> httpx.Response:
        path = f"/pods/{instance.instance_id}/action"
        response = await self._send("POST", path, json={"action": action})
        self._instances.pop(instance.instance_id, None)
        return response

    async def start(self, instance: Instance) -> None:
        path = f"/pods/{instance.instance_id}/action"
        response = await self._action(instance, "start")
        if response.status_code in (400, 409, 422):
            raise ProviderError(
                f"POST {path}: the pod could not be started ({self._quiet(_problem(response))}) — "
                "its machine may no longer have a free GPU"
            )
        if response.status_code >= 400:
            raise self._fail("POST", path, response)

    async def stop(self, instance: Instance) -> None:
        path = f"/pods/{instance.instance_id}/action"
        response = await self._action(instance, "stop")
        if response.status_code == 409:
            # Not a valid action now — fine if that is because it is already stopped.
            if (await self.status(instance)).state == InstanceState.STOPPED:
                return
        if response.status_code >= 400:
            raise self._fail("POST", path, response)

    async def destroy(self, instance: Instance) -> None:
        """Idempotent: a pod the provider no longer knows is already gone."""
        path = f"/pods/{instance.instance_id}"
        response = await self._send("DELETE", path)
        self._instances.pop(instance.instance_id, None)
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise self._fail("DELETE", path, response)

    async def instance_logs(self, instance: Instance, tail: int = 60) -> Optional[str]:
        """The pod's own output, newest `tail` lines (D78), from its server-sent event stream.

        The stream stays open after its backfill, so it is read for a bounded time. Each event's
        `data:` is the provider's `{"source", "line", "ts"}`; the line is kept, as text.
        """
        lines: list[str] = []
        path = f"/pods/{instance.instance_id}/logs"

        async def read() -> bool:
            async with self.client.stream(
                "GET", path, params={"tail": max(0, min(int(tail), 5000))},
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    return False
                async for raw in response.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    try:
                        event = json.loads(data)
                    except ValueError:
                        lines.append(data)
                        continue
                    if isinstance(event, dict):
                        text = str(event.get("line") or "")
                        lines.append(f"[system] {text}" if event.get("source") == "system" else text)
                    else:
                        lines.append(data)
            return True

        try:
            if time.monotonic() < self._quiet_until:
                return None
            ok = await asyncio.wait_for(read(), timeout=self.logs_window_s)
        except asyncio.TimeoutError:
            ok = True  # the backfill came, and the stream stayed open — as it does
        except (httpx.HTTPError, ProviderError):
            return None
        if not ok or not lines:
            return None
        return self._quiet("\n".join(lines[-tail:]))

    # --- money ---

    async def _charges_by_pod(self) -> dict[str, float]:
        cached = self._charges
        if cached is not None and time.monotonic() - cached[0] < self.instance_cache_s:
            return cached[1]
        start = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start -= datetime.timedelta(days=self.charges_window_days)
        payload = await self._call(
            "GET", "/billing/pods",
            params={"bucketSize": "day", "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ")},
        )
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ProviderUnavailable("GET /billing/pods: the answer carries no `records` list")
        totals: dict[str, float] = {}
        for row in records:
            pod = str((row or {}).get("podId") or "")
            if pod:
                totals[pod] = totals.get(pod, 0.0) + float(row.get("totalAmount") or 0.0)
        self._charges = (time.monotonic(), totals)
        return totals

    async def reported_charges(self, instance: Instance) -> Optional[Charges]:
        """What the provider has billed this pod so far, over its daily records.

        One request serves every pod in a pass. No record at all is None, not zero, so the cap
        margin does not narrow on a figure that has not arrived.
        """
        totals = await self._charges_by_pod()
        if instance.instance_id not in totals:
            return None
        return Charges(total=totals[instance.instance_id], as_of=time.time())

    async def account(self) -> AccountStatus:
        """The credential checked with a cheap v2 read; the balance from GraphQL, where it can be
        read — its failure leaves the credit unknown, never the credential invalid."""
        await self._call("GET", "/pods", params={"limit": 1})
        credit, why = await self._balance()
        return AccountStatus(credential_valid=True, credit_remaining=credit,
                             detail=None if credit is not None else f"balance not read: {why}")

    async def _balance(self) -> tuple[Optional[float], str]:
        try:
            response = await self.client.post(
                self.graphql_url, json={"query": "query { myself { clientBalance } }"},
            )
        except httpx.HTTPError as exc:
            return None, self._quiet(str(exc))[:200]
        if response.status_code != 200:
            return None, f"GraphQL answered {response.status_code}"
        try:
            said = response.json()
            balance = ((said.get("data") or {}).get("myself") or {}).get("clientBalance")
        except (ValueError, AttributeError):
            return None, "GraphQL answered something that is not JSON"
        if balance is None:
            errors = said.get("errors") if isinstance(said, dict) else None
            words = json.dumps(redacted(errors))[:200] if errors else "no clientBalance in the answer"
            return None, self._quiet(words)
        try:
            return float(balance), ""
        except (TypeError, ValueError):
            return None, "clientBalance is not a number"

    def self_terminate_request(self, action: str = "destroy") -> SelfTerminateRequest:
        """Made *on the pod*, with the pod-scoped key the provider put there — never the account
        key. Both names are the pod's own environment, expanded on the host."""
        if action == "destroy":
            return SelfTerminateRequest(
                method="DELETE",
                url=f"{self.base_url}/pods/$RUNPOD_POD_ID",
                headers={"Authorization": "Bearer $RUNPOD_API_KEY"},
            )
        return SelfTerminateRequest(
            method="POST",
            url=f"{self.base_url}/pods/$RUNPOD_POD_ID/action",
            headers={"Authorization": "Bearer $RUNPOD_API_KEY", "Content-Type": "application/json"},
            body='{"action": "stop"}',
        )
