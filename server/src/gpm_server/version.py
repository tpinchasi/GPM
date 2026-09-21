"""What this process is actually running (D79).

A pool that spends money must be able to say which code is spending it. Seen live, three times
in one week: a fix was on disk and the running supervisor was not running it — once a stale
packed agent, twice code loaded before the fix was written — and each time the only way anyone
found out was a rented host failing. The cause under all three was the same: the live pool ran
from a development tree, as an editable install, so whatever was being edited was also what
was deployed.

So a deployed pool runs a *release*: a tag, installed non-editable, somewhere nobody edits
(`deploy/gpm-deploy`). This module is how the running code describes itself, so the supervisor
can record it at start and the console can show it — and so a pool that is running out of a
development tree says so, loudly, instead of looking like any other.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

#: Written into a release's directory by `gpm-deploy`: the tag and commit it was built from.
RELEASE_FILE = "RELEASE.json"


@dataclasses.dataclass(frozen=True)
class Running:
    version: str
    #: The git tag this install was deployed from, where it was deployed at all.
    tag: Optional[str]
    commit: Optional[str]
    #: True when the package is imported from a working tree: editing a file changes the pool.
    editable: bool
    #: Where the code is imported from.
    location: str

    @property
    def is_release(self) -> bool:
        return self.tag is not None and not self.editable

    def describe(self) -> str:
        if self.is_release:
            return f"release {self.tag} ({self.version}) from {self.location}"
        if self.editable:
            return (
                f"a DEVELOPMENT TREE at {self.location} ({self.version}): every file edited "
                "there changes this pool — deploy a tagged release with `gpm-deploy <tag>`"
            )
        return f"{self.version} from {self.location}, not deployed from a tag"

    def as_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "is_release": self.is_release}


def _editable(distribution: metadata.Distribution) -> bool:
    """PEP 610: an editable install records itself in `direct_url.json`."""
    try:
        raw = distribution.read_text("direct_url.json")
    except OSError:
        return False
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("dir_info", {}).get("editable"))
    except ValueError:
        return False


def _release_beside(start: Path) -> dict[str, Any]:
    """`RELEASE.json`, looked for upward from where the interpreter lives: a release is a
    directory holding its own venv, and the file sits at its top."""
    for directory in [start, *start.parents][:6]:
        found = directory / RELEASE_FILE
        if found.is_file():
            try:
                return json.loads(found.read_text())
            except (OSError, ValueError):
                return {}
    return {}


def running(distribution: str = "gpm-server") -> Running:
    try:
        dist = metadata.distribution(distribution)
        version, editable = dist.version, _editable(dist)
    except metadata.PackageNotFoundError:
        version, editable = "unknown", True  # imported straight off a path: a working tree
    location = Path(__file__).resolve().parent
    release = {} if editable else _release_beside(Path(sys.prefix).resolve())
    return Running(
        version=version,
        tag=release.get("tag"),
        commit=release.get("commit"),
        editable=editable,
        location=str(location.parent if editable else Path(sys.prefix).resolve()),
    )
