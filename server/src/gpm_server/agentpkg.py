"""The host agent, packed as one file to put on a machine the pool rents (D72).

A zipapp: the agent and its pure-Python dependencies in a single `.pyz` the host's own
`python3` runs. It is built from what is already installed beside the supervisor — no network,
no package index, no build step at release time — so what lands on a host is the agent this
pool is running, not whatever a registry has today.

A host with no interpreter gets no agent and joins without one (D63): the agent is an
improvement, never a requirement.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import zipapp
from importlib import metadata
from pathlib import Path
from typing import Optional

log = logging.getLogger("gpm.agentpkg")

#: What the agent needs to run. Walked transitively, so this names only the direct ones.
ROOT_DISTRIBUTION = "gpm-agent"

_MAIN = "gpm_agent.cli:main"


class AgentPackageUnavailable(Exception):
    """The agent cannot be packed here — so no host gets one, and every host still joins."""


def _requirements(distribution: str, seen: set[str]) -> set[str]:
    """Every distribution `distribution` needs, transitively, skipping the optional ones.

    A requirement marked for an extra ("httpx[http2]") or a platform this machine is not is not
    needed to *run* the agent, and packing it would only make the file bigger.
    """
    name = distribution.lower().replace("_", "-")
    if name in seen:
        return seen
    seen.add(name)
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        # Conditional on a Python this one is not, or on an extra nobody asked for: this
        # interpreter runs the agent without it, so the host's can too. A dependency that is
        # genuinely needed shows up when the built file is run, below.
        log.debug("%s is not installed here, so it is not packed", name)
        return seen
    for raw in dist.requires or []:
        requirement = raw.split(";")[0].strip()
        if "; extra" in raw or "extra ==" in raw:
            continue
        stop = min((requirement.find(c) for c in "<>=!~ [(" if requirement.find(c) != -1), default=len(requirement))
        _requirements(requirement[:stop], seen)
    return seen


def _top_level_paths(distribution: str) -> list[Path]:
    """The importable directories and modules one installed distribution owns.

    Resolved by *importing* rather than by reading site-packages, because a development install
    keeps its source in the working tree and an installed one does not — and the agent that
    goes to a host should be the agent this supervisor is running either way.
    """
    import importlib.util

    names: set[str] = set()
    try:
        dist = metadata.distribution(distribution)
    except metadata.PackageNotFoundError:
        return []  # conditional on another Python; nothing of it to pack
    top_level = dist.read_text("top_level.txt")
    if top_level:
        names.update(line.strip() for line in top_level.splitlines() if line.strip())
    for file in dist.files or []:
        head = str(file).split("/")[0]
        if head.endswith((".dist-info", ".pth", ".egg-info")) or head.startswith(".."):
            continue
        names.add(head[:-3] if head.endswith(".py") else head)
    names.add(distribution.replace("-", "_"))

    found: list[Path] = []
    for name in sorted(names):
        if not name.isidentifier():
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            found.append(Path(list(spec.submodule_search_locations)[0]))
        elif spec.origin and spec.origin.endswith(".py"):
            found.append(Path(spec.origin))
    return found


def build(destination: Path, *, root: str = ROOT_DISTRIBUTION) -> Path:
    """Pack the agent and what it needs into `destination`, and return it.

    The file is content-addressed by what went into it, so an unchanged pool pushes an
    unchanged file and a host can be asked whether it already has this one.
    """
    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for distribution in sorted(_requirements(root, set())):
            for path in _top_level_paths(distribution):
                target = staging / path.name
                if target.exists():
                    continue
                if path.is_dir():
                    shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                else:
                    shutil.copy2(path, target)
        if not (staging / "gpm_agent").exists():
            raise AgentPackageUnavailable("the agent package itself was not found")
        destination.parent.mkdir(parents=True, exist_ok=True)
        zipapp.create_archive(staging, target=destination, main=_MAIN)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _check_it_runs(destination)
    return destination


def _check_it_runs(archive: Path) -> None:
    """Start the packed agent here before any host is asked to.

    Packing copies what this machine has; a dependency it turns out to need and does not have
    would otherwise be discovered on a rented machine, mid-preparation, as a host that never
    reports. One subprocess is cheaper than that.
    """
    import subprocess
    import sys

    try:
        done = subprocess.run(
            [sys.executable, str(archive), "--help"],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentPackageUnavailable(f"the packed agent could not be started: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).decode(errors="replace").strip().splitlines()
        raise AgentPackageUnavailable(
            "the packed agent does not run here: " + (detail[-1] if detail else "no output")
        )


def digest(path: Path) -> str:
    """What identifies this build to a host: the file's own hash, short enough to read."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cached(state_dir: Path, *, root: str = ROOT_DISTRIBUTION) -> Optional[Path]:
    """The packed agent, built once per supervisor run. None where it cannot be built."""
    destination = state_dir / "gpm-agent.pyz"
    if destination.exists():
        return destination
    try:
        return build(destination, root=root)
    except (AgentPackageUnavailable, OSError) as exc:
        log.warning("the host agent cannot be packed here, so rented hosts will run without one: %s", exc)
        return None
