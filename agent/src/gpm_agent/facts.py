"""What this machine is — the facts an engine's API cannot report.

docs/spec/host-agent.md §3. Read-only, cheap and unprivileged. Every probe of the outside world
goes through `Probes`, so the whole module is tested without the hardware it describes: a test
supplies what `nvidia-smi` would have printed.

A probe that fails reports *nothing* for that fact, never a guess: an absent accelerator list
means "could not tell", and the pool does not derive a capability from silence.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import subprocess
from typing import Any, Callable, Optional, Sequence

_TIMEOUT_S = 5


def _run(command: Sequence[str]) -> Optional[str]:
    """Output of a fixed, argument-list command, or None. Never a shell, never input from
    the pool: every command this module runs is a literal written here."""
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(
            list(command), capture_output=True, text=True, timeout=_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


@dataclasses.dataclass
class Probes:
    """The outside world, injectable."""

    system: Callable[[], str] = platform.system
    machine: Callable[[], str] = platform.machine
    run: Callable[[Sequence[str]], Optional[str]] = _run
    read: Callable[[str], Optional[str]] = _read
    disk_usage: Callable[[str], Any] = shutil.disk_usage


def _int(text: Optional[str]) -> Optional[int]:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def memory(probes: Probes) -> dict[str, Optional[int]]:
    """Total and available system memory, in bytes."""
    if probes.system() == "Darwin":
        total = _int(probes.run(["sysctl", "-n", "hw.memsize"]))
        available = None
        stats = probes.run(["vm_stat"])
        if stats:
            page = 4096
            first = stats.splitlines()[0]
            if "page size of" in first:
                page = _int(first.split("page size of")[1].split("bytes")[0]) or page
            counts = {}
            for line in stats.splitlines()[1:]:
                name, _, value = line.partition(":")
                counts[name.strip()] = _int(value.strip().rstrip("."))
            # Free, plus what the system would give up without paging anything out.
            reclaimable = [counts.get(k) for k in ("Pages free", "Pages inactive", "Pages purgeable")]
            if all(v is not None for v in reclaimable):
                available = sum(reclaimable) * page
        return {"total_bytes": total, "available_bytes": available}

    info = probes.read("/proc/meminfo")
    if not info:
        return {"total_bytes": None, "available_bytes": None}
    fields = {}
    for line in info.splitlines():
        name, _, rest = line.partition(":")
        kilobytes = _int(rest.strip().split(" ")[0])
        fields[name] = kilobytes * 1024 if kilobytes is not None else None
    return {"total_bytes": fields.get("MemTotal"), "available_bytes": fields.get("MemAvailable")}


def accelerators(probes: Probes) -> Optional[list[dict[str, Any]]]:
    """Accelerators found, or None when this machine gave no way to tell."""
    found: list[dict[str, Any]] = []
    could_tell = False

    if probes.system() == "Darwin" and probes.machine() == "arm64":
        could_tell = True
        name = (probes.run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple silicon").strip()
        # Unified memory: the accelerator's memory is the system's.
        found.append({
            "kind": "apple-silicon", "name": name,
            "memory_bytes": _int(probes.run(["sysctl", "-n", "hw.memsize"])), "unified": True,
        })

    nvidia = probes.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if nvidia is not None:
        could_tell = True
        for line in nvidia.splitlines():
            name, _, mebibytes = line.rpartition(",")
            size = _int(mebibytes)
            if name.strip():
                found.append({
                    "kind": "cuda", "name": name.strip(),
                    "memory_bytes": size * 1024 * 1024 if size is not None else None, "unified": False,
                })

    rocm = probes.run(["rocm-smi", "--showproductname", "--csv"])
    if rocm is not None:
        could_tell = True
        for line in rocm.splitlines()[1:]:
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) > 1 and cells[1]:
                found.append({"kind": "rocm", "name": cells[1], "memory_bytes": None, "unified": False})

    if probes.system() == "Linux":
        could_tell = True  # on Linux, none of the above answering means there is none we can use
    return found if could_tell else None


def disk(probes: Probes, path: str) -> dict[str, Any]:
    """Free space where the engine keeps its models. The nearest existing parent is measured,
    so a models directory that does not exist yet still reports the volume it would land on."""
    probe_at = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(probe_at) and os.path.dirname(probe_at) != probe_at:
        probe_at = os.path.dirname(probe_at)
    try:
        usage = probes.disk_usage(probe_at)
    except OSError:
        return {"path": path, "total_bytes": None, "free_bytes": None}
    return {"path": path, "total_bytes": usage.total, "free_bytes": usage.free}


def derive_capabilities(found: Optional[list[dict[str, Any]]]) -> Optional[list[str]]:
    """The capability names a pool's catalog keys variants on, from what was found — or None
    when nothing could be told, which is not the same as "none"."""
    if found is None:
        return None
    return sorted({entry["kind"] for entry in found})


def gather(probes: Probes, models_path: str) -> dict[str, Any]:
    found = accelerators(probes)
    return {
        "os": probes.system(),
        "arch": probes.machine(),
        "memory": memory(probes),
        "accelerators": found,
        "capabilities": derive_capabilities(found),
        "disk": disk(probes, models_path),
    }
