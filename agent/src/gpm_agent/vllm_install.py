"""vLLM on a machine its owner gives the agent (S10, D157–D161).

vLLM publishes its builds for Linux only, on PyPI as one manylinux wheel per processor, and needs
an NVIDIA card of compute capability 7.5 or higher (its own documentation). The agent installs a
version into a private Python environment made with its own interpreter, side by side with the
others (D153), and only from that version's **locked set** (D158): a list GPM's release generates —
every wheel, `vllm`'s and each dependency's, with its sha256, for one Python minor version and one
processor — and the least NVIDIA driver the set's CUDA needs. The sets ship inside the agent, in
`vllm_sets/`, so what PyPI serves later cannot change what a machine installs.

This module holds what both sides need: where a version's program lives, the sets, and whether
this machine can run vLLM at all — which `engine: auto` (D161) chooses by.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

#: vLLM's own floor: compute capability 7.5, Turing onward (docs.vllm.ai, GPU installation).
MIN_CAPABILITY = (7, 5)

#: Where `engine: auto` chooses vLLM (D161): Ampere onward, where the BF16, FP8 and FP4 builds
#: catalogs carry for vLLM run (D100). A Turing card runs vLLM, but only FP16 and integer builds.
AUTO_MIN_CAPABILITY = (8, 0)

#: The processors vLLM publishes a wheel for, as Python's `platform.machine()` names them.
PLATFORMS = ("x86_64", "aarch64")

#: The sets GPM's release generated, one JSON file each (`LockedSet`).
SETS_DIR = Path(__file__).parent / "vllm_sets"


def program_in(version_dir: Path) -> Path:
    """A version's `vllm` program, inside its private environment."""
    return Path(version_dir) / "venv" / "bin" / "vllm"


@dataclasses.dataclass(frozen=True)
class LockedSet:
    """One vLLM version's dependencies, locked, for one Python minor version and one processor."""

    vllm: str
    #: The Python minor version the set was resolved for, e.g. "3.12".
    python: str
    #: The processor, one of `PLATFORMS`.
    platform: str
    #: The least NVIDIA driver the set's CUDA needs, as `nvidia-smi` prints a version.
    driver_floor: str
    #: pip's requirements, every line pinned (`==`) and hashed (`--hash=sha256:…`).
    requirements: str
    #: The wheels' total size, for the free-disk floor.
    size_bytes: int

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LockedSet":
        return cls(
            vllm=str(data["vllm"]), python=str(data["python"]), platform=str(data["platform"]),
            driver_floor=str(data["driver_floor"]), requirements=str(data["requirements"]),
            size_bytes=int(data["size_bytes"]),
        )


def capability(text: Any) -> Optional[tuple[int, int]]:
    """A compute capability as `nvidia-smi` prints it ("8.6") — or None."""
    found = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(text or ""))
    return (int(found.group(1)), int(found.group(2))) if found else None


def driver_key(text: Any) -> Optional[tuple[int, ...]]:
    """A driver version ("575.51.03") as numbers to compare — or None."""
    parts = str(text or "").strip().split(".")
    return tuple(int(p) for p in parts) if parts and all(p.isdigit() for p in parts) else None


def machine_can_run(
    accelerators: Iterable[Mapping[str, Any]],
    *,
    os_name: str,
    arch: str,
    minimum: tuple[int, int] = MIN_CAPABILITY,
    driver_floor: Optional[str] = None,
) -> list[str]:
    """Every reason this machine cannot run vLLM, in words — none where it can.

    `accelerators` are the facts' (`kind`, `name`, `compute_capability`, `driver`); `minimum` is
    vLLM's floor, or `AUTO_MIN_CAPABILITY` where `auto` chooses; `driver_floor` the locked set's.
    """
    reasons: list[str] = []
    if os_name != "Linux":
        reasons.append(f"vLLM runs on Linux only, and this machine runs {os_name}")
    if arch not in PLATFORMS:
        reasons.append(f"vLLM publishes no build for the {arch or 'unknown'} processor")
    cards = [a for a in accelerators if a.get("kind") == "cuda"]
    if not cards:
        others = sorted({str(a.get("kind")) for a in accelerators if a.get("kind")})
        reasons.append("no NVIDIA card" + (f" (found: {', '.join(others)})" if others else ""))
        return reasons
    floor = ".".join(map(str, minimum))
    for card in cards:
        name = card.get("name") or "an NVIDIA card"
        found = capability(card.get("compute_capability"))
        if found is None:
            reasons.append(f"{name}: its compute capability could not be read")
        elif found < minimum:
            reasons.append(f"{name} has compute capability {found[0]}.{found[1]}; this needs {floor} or higher")
        if driver_floor is not None:
            have, need = driver_key(card.get("driver")), driver_key(driver_floor)
            if have is None:
                reasons.append(f"{name}: its driver's version could not be read")
            elif need is not None and have < need:
                reasons.append(f"{name}'s driver {card.get('driver')} is older than {driver_floor}, which vLLM's CUDA needs")
    return reasons


def load_sets(sets_dir: Path = SETS_DIR) -> list[LockedSet]:
    """Every set the agent carries — S10's installer fills this in (stage 2)."""
    found = []
    for path in sorted(Path(sets_dir).glob("*.json")) if Path(sets_dir).is_dir() else []:
        found.append(LockedSet.from_json(json.loads(path.read_text())))
    return found


def sets_for(python: str, platform: str, sets: Optional[Sequence[LockedSet]] = None) -> list[LockedSet]:
    """The sets for this Python minor version and processor, newest vLLM first."""
    from .installs import version_key  # the one ordering of versions every engine uses

    chosen = [s for s in (load_sets() if sets is None else sets) if s.python == python and s.platform == platform]
    return sorted(chosen, key=lambda s: version_key(s.vllm), reverse=True)


def newest_driver_floor(python: str, platform: str, sets: Optional[Sequence[LockedSet]] = None) -> Optional[str]:
    """The driver the newest set for this machine needs — None where the agent carries none."""
    found = sets_for(python, platform, sets)
    return found[0].driver_floor if found else None
