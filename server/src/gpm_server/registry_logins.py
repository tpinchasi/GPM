"""Logins to private container registries, for renting a container build's image (D183).

Kept by the supervisor in its credential store, one owner-only file for all of them, keyed by the
registry they are for. Handed to a provider only in its own create call, and only by a plug-in that
declares `capabilities.private_images`: the login then sits with the provider, never on the machine's
disk by the pool's hand, in `pool.yaml`, an event or a browser. A machine of the operator's own pulls
with its own Docker login and never sees these.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Optional

from .credentials import CredentialStore

#: The one credential holding every login, and what it is bound to.
_NAME = "registry-logins"
_BOUND = {"kind": "registry-logins"}
#: Docker Hub, as an image name with no registry of its own is pulled from.
DOCKER_HUB = "docker.io"
#: A registry's host, with a port where it has one: what an image name starts with.
REGISTRY = re.compile(r"^(?=.{1,253}(?::\d{1,5})?$)(?:localhost|[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+)(?::\d{1,5})?$")


@dataclasses.dataclass(frozen=True)
class RegistryLogin:
    """A login to one registry. Its password never appears in its repr."""

    registry: str
    username: str
    password: str = dataclasses.field(repr=False)


def registry_of(image: str) -> str:
    """The registry an image is pulled from: its first part where that names a host, else Docker Hub."""
    first, _, rest = image.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        return first.lower()
    return DOCKER_HUB


class RegistryLogins:
    """The pool's registry logins, in its credential store."""

    def __init__(self, credentials_dir: str | Path):
        self.store = CredentialStore(Path(credentials_dir))

    def _all(self) -> dict[str, dict[str, str]]:
        found = self.store.get(_NAME)
        if found is None or not found.bound_to(_BOUND):
            return {}
        try:
            data = json.loads(found.value)
        except ValueError:
            return {}
        return {r: v for r, v in data.items() if isinstance(v, dict) and isinstance(v.get("username"), str)
                and isinstance(v.get("password"), str)} if isinstance(data, dict) else {}

    def listed(self) -> list[dict[str, str]]:
        """Each login's registry and user name — never its password."""
        return [{"registry": r, "username": v["username"]} for r, v in sorted(self._all().items())]

    def for_image(self, image: str) -> Optional[RegistryLogin]:
        registry = registry_of(image)
        found = self._all().get(registry)
        return RegistryLogin(registry, found["username"], found["password"]) if found else None

    def put(self, registry: str, username: str, password: str) -> None:
        registry = registry.strip().lower()
        if not REGISTRY.match(registry):
            raise ValueError(f"{registry!r} is not a registry's host name (such as ghcr.io, or docker.io)")
        if not username or not password or any(c in username for c in " \t\r\n") or "\n" in password:
            raise ValueError("a registry login is a user name without spaces and a password, both given")
        logins = self._all()
        logins[registry] = {"username": username, "password": password}
        self.store.put(_NAME, json.dumps(logins, sort_keys=True), _BOUND)

    def remove(self, registry: str) -> bool:
        logins = self._all()
        if registry.strip().lower() not in logins:
            return False
        del logins[registry.strip().lower()]
        if logins:
            self.store.put(_NAME, json.dumps(logins, sort_keys=True), _BOUND)
        else:
            self.store.remove(_NAME)
        return True
