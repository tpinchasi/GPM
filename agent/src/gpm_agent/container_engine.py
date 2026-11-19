"""The `container` engine, as the host's agent runs it (S15, D180).

A container build is an operator's image, named by digest, that serves the predict contract —
`POST /v1/predict`, `GET /v1/models`, `GET /health` — on the port its build declares. The agent
pulls it and runs it through the **Docker Engine API on the machine's own socket** (an API client,
never a command line), one build per machine, with its port published on loopback at the agent's
`engine_url` only, and runs it narrowly: no host network, not privileged, no new privileges, no
added capability, no directory of the host, the build's memory as its limit, a card only where the
build needs one.

A container runs its image's own code on this machine. It runs only while the pool's
`containers.enabled` is on (the pool sends no container build otherwise) and never where this
machine's owner set `run_containers: never`. A private image is pulled with the machine's own
login, read from Docker's configuration here (`auths`), never sent by the pool (D181).
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, AsyncIterator, Collection, Optional
from urllib.parse import quote, urlsplit

import httpx

from .settings import loopback_engine_url_problem

#: An image by digest: a registry path, then `@sha256:` and 64 hex — the same rule as the pool's.
DIGEST_REF = re.compile(
    r"^(?!.*\.\.)(?P<name>[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?(?::[0-9]{1,5})?"
    r"(?:/[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?)+)@(?P<digest>sha256:[0-9a-f]{64})$"
)
#: What marks a container as one this agent made, and which build it runs.
LABEL, TAG_LABEL = "gpm.agent", "gpm.build"
ANSWER_TIMEOUT_S = 3.0
STOP_GRACE_S = 10
_SPEC_KEYS = {"engine", "port", "needs"}


def _refused(why: str) -> Exception:
    from .engines import EngineRefused

    return EngineRefused(why)


def default_socket() -> Optional[Path]:
    """Docker's socket where the system keeps it: Linux's, else Docker Desktop's in the home."""
    for found in (Path("/var/run/docker.sock"), Path.home() / ".docker" / "run" / "docker.sock"):
        if found.exists():
            return found
    return None


def registry_of(name: str) -> str:
    """The registry an image name is pulled from, as Docker's configuration keys it."""
    first = name.split("/", 1)[0]
    return first if ("." in first or ":" in first or first == "localhost") and "/" in name else "https://index.docker.io/v1/"


def machine_login(name: str, config: Optional[Path] = None) -> Optional[str]:
    """The `X-Registry-Auth` header for the image's registry, from this machine's own Docker login
    (`auths` in `~/.docker/config.json`) — None where it has none. A credential helper is never run."""
    path = config or Path(os.environ.get("DOCKER_CONFIG") or Path.home() / ".docker") / "config.json"
    try:
        auths = json.loads(path.read_text()).get("auths") or {}
    except (OSError, ValueError, AttributeError):
        return None
    registry = registry_of(name)
    entry = auths.get(registry) or auths.get(f"https://{registry}") or {}
    encoded = entry.get("auth") if isinstance(entry, dict) else None
    if not isinstance(encoded, str):
        return None
    try:
        user, _, password = base64.b64decode(encoded).decode().partition(":")
    except (ValueError, UnicodeDecodeError):
        return None
    said = json.dumps({"username": user, "password": password, "serveraddress": registry})
    return base64.urlsafe_b64encode(said.encode()).decode()


class ContainerFacts:
    name = "container"
    fetches_without_engine = True
    loads_by_restart = False
    splits_across_cards = False
    #: One build per machine, published at the agent's `engine_url`: several images are several
    #: hosts in the pool's file, as for whisper.cpp.
    serves_one_model = True

    def __init__(self, settings: Any = None, *, docker_transport: Optional[httpx.AsyncBaseTransport] = None,
                 docker_config: Optional[Path] = None) -> None:
        self.settings = settings
        self.docker_transport = docker_transport
        self.docker_config = docker_config
        self.specs: dict[str, dict[str, Any]] = {}

    # --- the pool's words ---

    @staticmethod
    def valid_tag(tag: Any) -> bool:
        return isinstance(tag, str) and len(tag) <= 300 and bool(DIGEST_REF.match(tag))

    @staticmethod
    def checked_spec(tag: str, said: Any) -> dict[str, Any]:
        if not isinstance(said, dict) or not set(said) <= _SPEC_KEYS or said.get("engine") != "container":
            raise ValueError(f"{tag}: not a container build's declarations")
        port = said.get("port")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{tag}: port is a whole number from 1 to 65535")
        needs = said.get("needs") or {}
        memory = needs.get("memory_gb") if isinstance(needs, dict) else None
        if not isinstance(needs, dict) or not set(needs) <= {"memory_gb", "card"} or (
                memory is not None and not (isinstance(memory, (int, float)) and 0 < memory <= 4096)):
            raise ValueError(f"{tag}: needs are memory_gb and card")
        return {"port": port, "memory_gb": memory, "card": bool(needs.get("card"))}

    def set_specs(self, specs: dict[str, dict[str, Any]]) -> None:
        self.specs = dict(specs)

    # --- Docker ---

    def _refuse_unless_allowed(self) -> None:
        if getattr(self.settings, "run_containers", "follow") == "never":
            raise _refused("this machine's owner has set run_containers: never — it runs no container")

    def _docker(self) -> httpx.AsyncClient:
        if self.docker_transport is not None:
            return httpx.AsyncClient(transport=self.docker_transport, base_url="http://docker", timeout=60)
        named = getattr(self.settings, "docker_socket", None)
        socket = Path(named).expanduser() if named else default_socket()
        if socket is None or not socket.exists():
            raise _refused("Docker's socket is not here: install Docker, or set docker_socket in this agent's settings")
        return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket)), base_url="http://docker",
                                 timeout=httpx.Timeout(60.0, read=600.0))

    def _host_port(self) -> int:
        url = getattr(self.settings, "engine_url", "")
        problem = loopback_engine_url_problem(url)
        if problem:
            raise _refused(f"engine_url {url!r} {problem}; a container's port is published on 127.0.0.1 only")
        return int(urlsplit(url).port or 0)

    async def _ours(self, docker: httpx.AsyncClient) -> list[dict[str, Any]]:
        filters = json.dumps({"label": [f"{LABEL}=1"]})
        response = await docker.get("/containers/json", params={"all": "true", "filters": filters})
        if response.status_code != 200:
            raise _refused(f"Docker answered {response.status_code} listing containers")
        return [c for c in response.json() if (c.get("Labels") or {}).get(LABEL) == "1"]

    async def _image_here(self, docker: httpx.AsyncClient, tag: str) -> Optional[int]:
        response = await docker.get(f"/images/{quote(tag, safe='')}/json")
        if response.status_code != 200:
            return None
        return int(response.json().get("Size") or 0)

    # --- what the agent reports ---

    async def describe(self, client: httpx.AsyncClient) -> dict[str, Any]:
        answers = await self._answers(client)
        facts: dict[str, Any] = {"name": self.name, "answers": answers, "version": None, "models_on_disk": [],
                                 "models_loaded": [], "models_failed": {},
                                 "run_containers": getattr(self.settings, "run_containers", "follow")}
        try:
            async with self._docker() as docker:
                version = await docker.get("/version")
                facts["version"] = version.json().get("Version") if version.status_code == 200 else None
                for tag in self.specs:
                    size = await self._image_here(docker, tag)
                    if size is not None:
                        facts["models_on_disk"].append({"tag": tag, "size_bytes": size})
                for container in await self._ours(docker):
                    tag = container["Labels"].get(TAG_LABEL)
                    if container.get("State") == "running":
                        if answers:
                            facts["models_loaded"].append(tag)
                    elif container.get("State") in ("exited", "dead"):
                        facts["models_failed"][tag] = f"its container stopped: {container.get('Status', 'exited')}"
        except (httpx.HTTPError, ValueError, Exception) as exc:  # noqa: BLE001 — what Docker could not say
            facts["docker"] = str(exc)[:300]
        return facts

    async def _answers(self, client: httpx.AsyncClient) -> bool:
        try:
            return (await client.get("/health", timeout=ANSWER_TIMEOUT_S)).status_code == 200
        except httpx.HTTPError:
            return False

    # --- the things the agent may do to a model ---

    async def pull(self, client: httpx.AsyncClient, tag: str) -> AsyncIterator[tuple[int, int]]:
        """Pull the image by its digest, yielding (bytes done, bytes total) over its layers."""
        self._refuse_unless_allowed()
        found = DIGEST_REF.match(tag) if isinstance(tag, str) else None
        if found is None:
            raise _refused(f"{tag!r} is not an image by digest")
        headers = {}
        login = machine_login(found.group("name"), self.docker_config)
        if login is not None:
            headers["X-Registry-Auth"] = login
        layers: dict[str, tuple[int, int]] = {}
        async with self._docker() as docker:
            async with docker.stream("POST", "/images/create", params={"fromImage": tag}, headers=headers) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")[:300]
                    raise _refused(f"Docker could not pull {tag}: {body}")
                async for line in response.aiter_lines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("error"):
                        raise _refused(f"Docker could not pull {tag}: {str(event['error'])[:300]}")
                    detail = event.get("progressDetail") or {}
                    if event.get("id") and isinstance(detail.get("total"), int):
                        layers[event["id"]] = (int(detail.get("current") or 0), detail["total"])
                        yield sum(d for d, _ in layers.values()), sum(t for _, t in layers.values())
            if await self._image_here(docker, tag) is None:
                raise _refused(f"Docker says it pulled {tag}, and does not have it")

    async def hold(self, client: httpx.AsyncClient, tag: str, *, pinned: bool) -> None:
        self._refuse_unless_allowed()
        async with self._docker() as docker:
            if not pinned:
                await self._remove(docker, lambda t: t == tag)
                return
            spec = self.specs.get(tag)
            if spec is None:
                raise _refused(f"the pool has not said which port {tag} answers on")
            ours = await self._ours(docker)
            running = [c for c in ours if c["Labels"].get(TAG_LABEL) == tag and c.get("State") == "running"]
            if running and await self._answers(client):
                return
            await self._remove(docker, lambda t: True)
            if await self._answers(client):
                raise _refused(f"something else already answers on port {self._host_port()}, where this agent "
                               "publishes the container")
            await self._run(docker, tag, spec)

    async def _run(self, docker: httpx.AsyncClient, tag: str, spec: dict[str, Any]) -> None:
        port = f"{spec['port']}/tcp"
        host: dict[str, Any] = {
            # Published on loopback alone: the container's server has no authentication of its own.
            "PortBindings": {port: [{"HostIp": "127.0.0.1", "HostPort": str(self._host_port())}]},
            "NetworkMode": "bridge", "Privileged": False, "CapAdd": [], "SecurityOpt": ["no-new-privileges:true"],
            "Binds": [], "Mounts": [], "RestartPolicy": {"Name": "unless-stopped"}, "Init": True,
        }
        if spec.get("memory_gb"):
            host["Memory"] = int(spec["memory_gb"] * 1024 ** 3)
        if spec.get("card"):
            host["DeviceRequests"] = [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]
        body = {"Image": tag, "ExposedPorts": {port: {}}, "HostConfig": host,
                "Labels": {LABEL: "1", TAG_LABEL: tag}, "Env": []}
        name = "gpm-" + tag.rsplit(":", 1)[-1][:12]
        created = await docker.post("/containers/create", params={"name": name}, json=body)
        if created.status_code not in (200, 201):
            raise _refused(f"Docker could not create a container from {tag}: {created.text[:300]}")
        started = await docker.post(f"/containers/{created.json()['Id']}/start")
        if started.status_code not in (204, 304):
            raise _refused(f"Docker could not start the container of {tag}: {started.text[:300]}")

    async def _remove(self, docker: httpx.AsyncClient, which: Any) -> None:
        for container in await self._ours(docker):
            if which(container["Labels"].get(TAG_LABEL)):
                cid = container["Id"]
                await docker.post(f"/containers/{cid}/stop", params={"t": str(STOP_GRACE_S)})
                await docker.delete(f"/containers/{cid}", params={"force": "true"})

    async def keep_only(self, client: httpx.AsyncClient, tags: Collection[str]) -> None:
        wanted = set(tags)
        async with self._docker() as docker:
            await self._remove(docker, lambda t: t not in wanted)

    async def delete(self, client: httpx.AsyncClient, tag: str) -> None:
        async with self._docker() as docker:
            await self._remove(docker, lambda t: t == tag)
            response = await docker.delete(f"/images/{quote(tag, safe='')}")
            if response.status_code not in (200, 404):
                raise _refused(f"Docker could not remove {tag}: {response.text[:300]}")

    def launch_environment(self, workers: int, models_held: int, context: Optional[int] = None,
                           cards_per_copy: int = 1) -> dict[str, str]:
        return {}

    def settings_from_environment(self, environment: Optional[dict[str, str]]) -> Optional[dict[str, Optional[int]]]:
        return None
