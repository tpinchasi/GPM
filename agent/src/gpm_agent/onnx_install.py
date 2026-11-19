"""What runs an ONNX build, installed from a locked set (S15, D179) — as vLLM's are (D158).

A version is a private Python environment with exactly the wheels of one set — ONNX Runtime, numpy
and Pillow, and on a Linux machine with an NVIDIA card CUDA's and cuDNN's libraries as wheels too —
and the pool's own ONNX server (`onnx_server.py`) copied beside it. Sets are generated at GPM's
release by `tools/onnx-lock` for Linux (x86_64 and aarch64) and Apple-silicon Macs; on Windows, an
owner gives the agent a Python of their own (`engine_binary`), which this does not install.

The machine decides the set: the CUDA one where an NVIDIA card's driver meets its floor, else the
processor's. Nothing is fetched before the machine is found able to run it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import platform as py_platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import httpx

from . import platform as host_platform
from .vllm_install import (
    _HASH_MISMATCH,
    CHECK_TIMEOUT_S,
    INSTALL_LOG,
    INSTALL_TIMEOUT_S,
    PIP_SETTINGS,
    PROCESSORS,
    REQUIREMENTS,
    UNPACKED_MARGIN,
    VENV,
    VENV_TIMEOUT_S,
    WORK,
    _last_words,
    driver_key,
    index_arguments,
    machine_can_run,
    machine_glibc,
    move_environment,
    nvidia_cards,
    pip_argv,
    this_python,
    venv_argv,
)

SETS_DIR = Path(__file__).parent / "onnx_sets"
#: The server, as copied into each version: run by that version's interpreter, isolated.
SERVER = "onnx_server.py"
#: ONNX Runtime's CUDA build runs on cards from Maxwell on (compute capability 5.2); the floor the
#: pool holds it to is Pascal's, the oldest its current CUDA still names.
MIN_CAPABILITY = (6, 0)


@dataclasses.dataclass(frozen=True)
class OnnxSet:
    onnxruntime: str
    python: str
    #: "linux" or "macos".
    system: str
    #: "x86_64" or "aarch64".
    platform: str
    #: "cuda" or "cpu".
    accelerator: str
    requirements: str
    size_bytes: int
    glibc: Optional[str] = None
    macos: Optional[str] = None
    driver_floor: Optional[str] = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "OnnxSet":
        def said(key: str) -> Optional[str]:
            value = data.get(key)
            return str(value) if value is not None else None

        return cls(
            onnxruntime=str(data["onnxruntime"]), python=str(data["python"]), system=str(data["system"]),
            platform=str(data["platform"]), accelerator=str(data["accelerator"]),
            requirements=str(data["requirements"]), size_bytes=int(data["size_bytes"]),
            glibc=said("glibc"), macos=said("macos"), driver_floor=said("driver_floor"),
        )


def load_sets(sets_dir: Path = SETS_DIR) -> list[OnnxSet]:
    found = []
    for path in sorted(Path(sets_dir).glob("*.json")) if Path(sets_dir).is_dir() else []:
        found.append(OnnxSet.from_json(json.loads(path.read_text())))
    return found


def program_in(version_dir: Path) -> Path:
    """A version's interpreter: the server is run by it, never by the agent's."""
    return Path(version_dir) / VENV / "bin" / "python"


def server_in(version_dir: Path) -> Path:
    return Path(version_dir) / SERVER


def server_argv(python: str, server: str, port: int, plan: Path) -> list[str]:
    """How the server is started — a fixed argument list, never a shell line: an interpreter (the
    installed version's, or the owner's `engine_binary`), isolated, running the pool's own server
    file, on loopback at the agent's port, with the agent's own plan file."""
    return [str(python), "-I", str(server), "--port", str(int(port)), "--plan", str(plan)]


def check_argv(venv: Path) -> list[str]:
    return [str(venv / "bin" / "python"), "-I", "-c",
            "import numpy, PIL, onnxruntime; print('onnxruntime', onnxruntime.__version__)"]


def _installs() -> Any:
    from . import installs

    return installs


def macos_version() -> Optional[str]:
    found = py_platform.mac_ver()[0]
    return found or None


class OnnxRuntimeSource:
    """ONNX Runtime's versions made into version directories, from the sets the agent carries."""

    engine = "onnx"
    #: Each wheel is checked against the digest its set records, and only that.
    signature = "not checked"

    def __init__(
        self,
        target: Any,
        *,
        python: Optional[str] = None,
        sets: Optional[Sequence[OnnxSet]] = None,
        cards: Optional[Callable[[], Optional[list[dict[str, Any]]]]] = None,
        glibc: Optional[Callable[[], Optional[str]]] = None,
        macos: Optional[Callable[[], Optional[str]]] = None,
        run: Optional[Any] = None,
        interpreter: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.target = target
        self.python = python or this_python()
        self.system = {host_platform.LINUX: "linux", host_platform.MACOS: "macos"}.get(target.os)
        arch = target.arch or ""
        self.processor = PROCESSORS.get(arch, arch if arch in ("x86_64", "aarch64") else None)
        self.sets = sets
        self.cards = cards or nvidia_cards
        self.glibc = glibc or machine_glibc
        self.macos = macos or macos_version
        self.run = run or host_platform.run_command
        self.interpreter = interpreter or sys.executable
        self.environ = environ

    def program(self, version_dir: Path) -> Path:
        return program_in(version_dir)

    # --- versions ---

    def _carried(self) -> list[OnnxSet]:
        return list(load_sets() if self.sets is None else self.sets)

    def _mine(self) -> list[OnnxSet]:
        version_key = _installs().version_key
        chosen = [s for s in self._carried()
                  if s.python == self.python and s.system == self.system and s.platform == self.processor]
        return sorted(chosen, key=lambda s: version_key(s.onnxruntime), reverse=True)

    def _refuse_the_machine(self) -> None:
        refused = _installs().InstallRefused
        if self.system is None:
            raise refused(
                "the agent installs ONNX Runtime on Linux and on Apple-silicon Macs; on this machine, give it a "
                "Python with onnxruntime, numpy and pillow installed (`engine_binary`)"
            )
        if self.processor is None:
            raise refused(f"ONNX Runtime's sets the agent carries are for x86_64 and aarch64, not {self.target.arch}")

    def _no_set(self, version: Optional[str]) -> str:
        here = f"Python {self.python} on {self.system} {self.processor}"
        carried = self._carried()
        if version is not None and not any(s.onnxruntime == version for s in carried):
            return f"ONNX Runtime {version} is not a version this agent carries a locked set for"
        if version is not None:
            return f"ONNX Runtime {version} has no locked set for {here}"
        pythons = sorted({s.python for s in carried})
        return f"this agent carries no ONNX Runtime set for {here}: its sets are for Python {', '.join(pythons) or 'none'}"

    @staticmethod
    def _release(version: str) -> Any:
        return _installs().Release(version=version, tag=f"v{version}", assets=())

    async def latest(self, http: httpx.AsyncClient) -> Any:
        self._refuse_the_machine()
        found = self._mine()
        if not found:
            raise _installs().InstallRefused(self._no_set(None))
        return self._release(found[0].onnxruntime)

    async def release(self, http: httpx.AsyncClient, version: str) -> Any:
        installs = _installs()
        if not installs.VERSION.fullmatch(version):
            raise installs.InstallRefused(f"{version!r} is not a version")
        self._refuse_the_machine()
        if not any(s.onnxruntime == version for s in self._mine()):
            raise installs.InstallRefused(self._no_set(version))
        return self._release(version)

    # --- which set, on this machine ---

    def _reasons_against(self, locked: OnnxSet, cards: Optional[list[dict[str, Any]]]) -> list[str]:
        reasons: list[str] = []
        if locked.glibc is not None:
            have = self.glibc()
            if have is None:
                reasons.append(f"its wheels need glibc {locked.glibc} or later, and this machine's C library is not glibc")
            elif (driver_key(have) or ()) < (driver_key(locked.glibc) or ()):
                reasons.append(f"its wheels need glibc {locked.glibc} or later, and this machine has {have}")
        if locked.macos is not None:
            have = self.macos()
            if have is not None and (driver_key(have) or ()) < (driver_key(locked.macos) or ()):
                reasons.append(f"its wheels need macOS {locked.macos} or later, and this Mac runs {have}")
        if locked.accelerator == "cuda":
            if cards is None:
                reasons.append("nvidia-smi does not answer here: no NVIDIA card, or no driver for one")
            else:
                reasons += machine_can_run(cards, os_name="Linux", arch=self.processor or "", minimum=MIN_CAPABILITY,
                                           driver_floor=locked.driver_floor or "")
        return reasons

    async def choose(self, version: str) -> tuple[OnnxSet, list[str]]:
        """The set this machine takes for `version` — CUDA's where it can run it — and why not CUDA's."""
        candidates = [s for s in self._mine() if s.onnxruntime == version]
        if not candidates:
            raise _installs().InstallRefused(self._no_set(version))
        cards = await asyncio.to_thread(self.cards) if any(s.accelerator == "cuda" for s in candidates) else None
        not_cuda: list[str] = []
        for locked in sorted(candidates, key=lambda s: s.accelerator != "cuda"):
            reasons = self._reasons_against(locked, cards)
            if not reasons:
                return locked, not_cuda
            if locked.accelerator == "cuda":
                not_cuda = reasons
                continue
            raise _installs().InstallRefused(f"ONNX Runtime {version} is not installed here: {'; '.join(reasons)}")
        raise _installs().InstallRefused(f"ONNX Runtime {version} has no set this machine can run: {'; '.join(not_cuda)}")

    # --- staging a version ---

    async def stage(self, http: httpx.AsyncClient, release: Any, into: Path, progress: Any, room: Any = None) -> None:
        installs = _installs()
        refused = installs.InstallRefused
        version = release.version
        self._refuse_the_machine()
        locked, not_cuda = await self.choose(version)
        index = index_arguments(os.environ if self.environ is None else self.environ)
        into = Path(os.path.abspath(into))
        final = into.with_name(into.name[: -len(installs.PARTIAL)]) if into.name.endswith(installs.PARTIAL) else into
        engine_dir = into.parent
        work = engine_dir / WORK
        await asyncio.to_thread(shutil.rmtree, work, True)
        if room is not None:
            room(locked.size_bytes * UNPACKED_MARGIN, (
                f"its locked set's wheels, {installs.size_words(locked.size_bytes)}, held while they install, "
                f"and the environment they make: up to {UNPACKED_MARGIN} times that"
            ))
        host_platform.private_dir(work, parents=True)
        log = engine_dir / "logs" / INSTALL_LOG
        host_platform.private_dir(log.parent, parents=True, exist_ok=True)
        said = f" (not CUDA's: {'; '.join(not_cuda)})" if not_cuda else ""
        log.write_text(f"--- gpm-agent: installing ONNX Runtime {version}, {locked.accelerator} set, "
                       f"for Python {self.python} on {self.system} {self.processor}{said}\n")
        venv = into / VENV
        try:
            progress("downloading", 0, locked.size_bytes)
            made = await self.run(venv_argv(self.interpreter, venv), env=host_platform.program_environment({}),
                                  timeout_s=VENV_TIMEOUT_S, log=log)
            if made.code != 0:
                raise refused(
                    f"a private Python environment for ONNX Runtime {version} could not be made with this agent's "
                    f"Python ({_last_words(made)}): it needs Python's venv module (python3-venv on Debian and Ubuntu)"
                )
            requirements = into / REQUIREMENTS
            requirements.write_text(locked.requirements)
            installed = await self.run(
                pip_argv(venv, requirements, index),
                env=host_platform.program_environment({"TMPDIR": str(work), **PIP_SETTINGS}),
                timeout_s=INSTALL_TIMEOUT_S, low_priority=True, log=log,
            )
            if installed.code != 0:
                if _HASH_MISMATCH in installed.output:
                    raise refused(f"a file fetched for ONNX Runtime {version} does not match the digest its set "
                                  f"records; nothing of it is kept (the log: {log})")
                raise refused(f"installing ONNX Runtime {version}'s set failed ({_last_words(installed)}); the log is {log}")
        finally:
            await asyncio.to_thread(shutil.rmtree, work, True)

        progress("verifying", 0, None)
        source = Path(__file__).with_name(SERVER)
        await asyncio.to_thread(shutil.copyfile, source, server_in(into))
        await asyncio.to_thread(move_environment, venv, final / VENV)
        checked = await self.run(check_argv(venv), env=host_platform.program_environment({}),
                                 timeout_s=CHECK_TIMEOUT_S, log=log)
        if checked.code != 0:
            raise refused(f"ONNX Runtime {version}, installed, does not import: {_last_words(checked)}; nothing of it is kept")
        if f"onnxruntime {version}" not in checked.output:
            raise refused(f"ONNX Runtime {version}, installed, says it is another version ({_last_words(checked)})")
