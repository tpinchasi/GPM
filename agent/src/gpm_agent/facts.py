"""What this machine is — the facts an engine's API cannot report.

docs/spec/host-agent.md §3. Read-only, cheap and unprivileged. Every probe of the outside world
goes through `Probes`, so the whole module is tested without the hardware it describes: a test
supplies what `nvidia-smi` would have printed. One probe is the engine image's rather than the
machine's — whether vLLM's interpreter imports its audio extra (D146) — and is tested against
stand-ins for that interpreter instead.

A probe that fails reports *nothing* for that fact, never a guess: an absent accelerator list
means "could not tell", and the pool does not derive a capability from silence.
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import platform
import re
import shutil
import subprocess
from typing import Any, Callable, Optional, Sequence

from .platform import CREATE_NO_WINDOW, program_environment

_TIMEOUT_S = 5


def _run(
    command: Sequence[str],
    *,
    system: Callable[[], str] = platform.system,
    which: Callable[[str], Optional[str]] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
    timeout_s: float = _TIMEOUT_S,
) -> Optional[str]:
    """Output of a fixed, argument-list command, or None. Never a shell, never input from
    the pool: every command this module runs is a literal written here. `timeout_s` is longer only
    where an agent of `auto` asks again, as it starts, about cards it could not read (`app`).

    On Windows with no console window: the agent started at logon runs under `pythonw.exe`, which
    has none, so each console program it starts — `nvidia-smi`, on every facts call — would open
    a window of its own on the owner's screen."""
    if which(command[0]) is None:
        return None
    options: dict[str, Any] = {"creationflags": CREATE_NO_WINDOW} if system() == "Windows" else {}
    try:
        done = run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False, **options
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


class MemoryStatus(ctypes.Structure):
    """Windows' `MEMORYSTATUSEX`, as `GlobalMemoryStatusEx` fills it: two 32-bit numbers, then
    seven 64-bit ones, 64 bytes in all. Of the 64-bit ones only the first two are read: physical
    memory in total, and available."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _windows_memory(kernel32: Any = None) -> Optional[tuple[int, int]]:
    """Total and available physical memory, in bytes, from Windows' own `GlobalMemoryStatusEx` —
    or None anywhere else, or when the call fails. A library call, not a command: Windows has no
    `/proc/meminfo`, and nothing it ships prints these numbers in a form meant to be read."""
    if kernel32 is None:
        windll = getattr(ctypes, "windll", None)  # only Windows' ctypes has it
        if windll is None:
            return None
        kernel32 = windll.kernel32
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    try:
        if not kernel32.GlobalMemoryStatusEx(ctypes.pointer(status)):
            return None
    except OSError:
        return None
    return status.ullTotalPhys, status.ullAvailPhys


@dataclasses.dataclass
class Probes:
    """The outside world, injectable."""

    system: Callable[[], str] = platform.system
    machine: Callable[[], str] = platform.machine
    run: Callable[[Sequence[str]], Optional[str]] = _run
    read: Callable[[str], Optional[str]] = _read
    disk_usage: Callable[[str], Any] = shutil.disk_usage
    #: Total and available memory on Windows, or None.
    windows_memory: Callable[[], Optional[tuple[int, int]]] = _windows_memory


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

    if probes.system() == "Windows":
        both = probes.windows_memory()
        if both is None:
            return {"total_bytes": None, "available_bytes": None}
        return {"total_bytes": both[0], "available_bytes": both[1]}

    info = probes.read("/proc/meminfo")
    if not info:
        return {"total_bytes": None, "available_bytes": None}
    fields = {}
    for line in info.splitlines():
        name, _, rest = line.partition(":")
        kilobytes = _int(rest.strip().split(" ")[0])
        fields[name] = kilobytes * 1024 if kilobytes is not None else None
    return {"total_bytes": fields.get("MemTotal"), "available_bytes": fields.get("MemAvailable")}


#: What an NVIDIA card's compute capability and its driver's version look like as `nvidia-smi`
#: prints them ("8.6", "575.51.03"); anything else is not taken for one.
_CAPABILITY = re.compile(r"\d+\.\d+")
_DRIVER = re.compile(r"\d+(\.\d+)+")

#: The query for each card's name and memory, the one every NVIDIA driver answers.
NVIDIA_CARDS = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
#: Each card's compute capability and the driver's version (D157, D161), asked apart: a driver
#: too old to know `compute_cap` refuses the whole query it is in, and must not cost the card its
#: name and memory. Where it refuses, the driver's version is asked alone (`NVIDIA_DRIVER`).
NVIDIA_DETAILS = ["nvidia-smi", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader"]
NVIDIA_DRIVER = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]


def _nvidia_details(probes: Probes, count: int) -> list[dict[str, str]]:
    """Each card's compute capability and driver's version, in the order `nvidia-smi` lists the
    cards — nothing for a card whose line does not read as one, and nothing at all where the
    lines are not one a card, since which card each belongs to would then be a guess."""
    said = probes.run(NVIDIA_DETAILS)
    if said is not None:
        lines = [line for line in said.splitlines() if line.strip()]
        if len(lines) != count:
            return [{} for _ in range(count)]
        details = []
        for line in lines:
            capability, _, driver = (part.strip() for part in line.partition(","))
            found = {}
            if _CAPABILITY.fullmatch(capability):
                found["compute_capability"] = capability
            if _DRIVER.fullmatch(driver):
                found["driver"] = driver
            details.append(found)
        return details
    said = probes.run(NVIDIA_DRIVER)
    lines = [line.strip() for line in (said or "").splitlines() if line.strip()]
    if len(lines) != count:
        return [{} for _ in range(count)]
    return [{"driver": line} if _DRIVER.fullmatch(line) else {} for line in lines]


def nvidia_cards(probes: Probes) -> Optional[list[dict[str, Any]]]:
    """The NVIDIA cards `nvidia-smi` lists — each with its compute capability and the driver's
    version where those could be read — or None where it did not answer: no driver, or none
    that works."""
    said = probes.run(NVIDIA_CARDS)
    if said is None:
        return None
    cards = []
    for line in said.splitlines():
        name, _, mebibytes = line.rpartition(",")
        size = _int(mebibytes)
        if name.strip():
            cards.append({
                "kind": "cuda", "name": name.strip(),
                "memory_bytes": size * 1024 * 1024 if size is not None else None, "unified": False,
            })
    if cards:
        for card, details in zip(cards, _nvidia_details(probes, len(cards)), strict=True):
            card.update(details)
    return cards


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

    nvidia = nvidia_cards(probes)
    if nvidia is not None:
        could_tell = True
        found.extend(nvidia)

    # On Windows NVIDIA's cards are all this asks about — `nvidia-smi` ships with the driver there
    # too — and nothing else is claimed: there a machine without NVIDIA's answer is one this could
    # not tell about, since it may hold a card of another make, not one with none.
    rocm = None if probes.system() == "Windows" else probes.run(["rocm-smi", "--showproductname", "--csv"])
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


# --- what the engine's image carries (D146) ---

#: vLLM's `audio` extra, by the names vLLM's own audio code imports it under. At v0.29.0 the extra
#: is `av`, `scipy`, `soundfile`, `soxr` and `mistral_common[audio]` (its `setup.py`), the last of
#: which is `soundfile` and `soxr` again; `vllm/multimodal/audio.py` and
#: `vllm/multimodal/media/audio.py` import `av`, `scipy.signal`, `soxr` and `soundfile`, and put a
#: placeholder in place of any that is missing — so vLLM starts without them and refuses every
#: recording, saying why only in its log.
AUDIO_EXTRA_MODULES = ("av", "scipy.signal", "soundfile", "soxr")

#: Long enough for a cold disk under a model download to import scipy and PyAV; past it, the check
#: is given up and the answer is "not known", never "missing".
AUDIO_CHECK_TIMEOUT_S = 120.0

#: What may stand after `#!` for the check to run it: a Python interpreter, by its file name.
_PYTHON = re.compile(r"python[0-9.]*")


def interpreter_of(command: str, path: Optional[str] = None) -> Optional[str]:
    """The Python interpreter a console script runs under, from its `#!` line — or None when the
    command is not found or is not a Python script this can name.

    Two forms, the two that installers write: the interpreter's own absolute path
    (`#!/opt/venv/bin/python3`), and `#!/usr/bin/env python3`, found on the same PATH as the
    command. Anything else — a binary, a shell wrapper — is not guessed at.
    """
    found = shutil.which(command, path=path)
    if found is None:
        return None
    try:
        with open(found, "rb") as script:
            first = script.readline(1024)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    words = first[2:].decode(errors="replace").split()
    if len(words) == 2 and os.path.basename(words[0]) == "env":
        words = [shutil.which(words[1], path=path) or ""]
    if len(words) != 1 or not os.path.isabs(words[0]) or not _PYTHON.fullmatch(os.path.basename(words[0])):
        return None
    return words[0]


def audio_extra(
    command: str,
    *,
    modules: Sequence[str] = AUDIO_EXTRA_MODULES,
    timeout_s: float = AUDIO_CHECK_TIMEOUT_S,
    path: Optional[str] = None,
) -> Optional[tuple[bool, Optional[str]]]:
    """Whether the interpreter behind `command` imports vLLM's audio extra: `(True, None)`, or
    `(False, the error's last line)` — or None when the check could not run at all.

    The one command here not written out in full: its interpreter is read from the command's own
    first line, on this machine. It is run **isolated** (`-I`: no `PYTHON*` variables, no user
    site, nothing from the working directory) as an argument list, never a shell, so what imports
    is what the image installed for it and nothing else; the modules are this module's constant.
    Nothing the pool says reaches it. Slow by this module's standards — seconds, on a cold disk —
    so the agent runs it once, off its event loop.
    """
    interpreter = interpreter_of(command, path)
    if interpreter is None:
        return None
    try:
        done = subprocess.run(
            [interpreter, "-I", "-c", "import " + ", ".join(modules)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s, check=False,
            # What it imports was downloaded: none of the agent's own credentials (`HF_TOKEN`, a
            # provider's key) reach it — only what any program the agent starts is given.
            env=program_environment({}),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode == 0:
        return True, None
    lines = [line.strip() for line in done.stderr.splitlines() if line.strip()]
    return False, (lines[-1][:200] if lines else f"the import exited with {done.returncode}")
