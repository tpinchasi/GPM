"""vLLM on a machine its owner gives the agent (S10, D157–D161).

vLLM publishes its builds for Linux only, on PyPI as one manylinux wheel per processor, and needs
an NVIDIA card of compute capability 7.5 or higher (its own documentation). The agent installs a
version into a private Python environment made with its own interpreter, side by side with the
others (D153), and only from that version's **locked set** (D158): a list GPM's release generates —
every wheel, `vllm`'s and each dependency's, with its sha256, for one Python minor version and one
processor — and the least NVIDIA driver the set's CUDA needs. The sets ship inside the agent, in
`vllm_sets/`, so what PyPI serves later cannot change what a machine installs.

This module holds what both sides need: where a version's program lives, the sets, and whether
this machine can run vLLM at all — which `engine: auto` (D161) chooses by — and the installer's
source for vLLM (`VllmSource`, through `installs.source_for`).

What keeps an install inside the agent's bounds (T35):

- **Only what a set names, each file checked.** pip installs the set's requirements and nothing
  else: wheels only (`--only-binary :all:`), so no build script runs; nothing they depend on
  beyond the list (`--no-deps`); every file checked against the sha256 the set records
  (`--require-hashes`), so a file replaced on PyPI, or on the way, refuses the version. pip runs
  isolated from the owner's configuration — `-I` and `--isolated`, which leave out pip's
  environment variables and the user's own file, and `PIP_CONFIG_FILE` set to the null device,
  which leaves out every other file (`/etc/pip.conf`, `/etc/xdg/pip/pip.conf`, the environment's
  own) — with the agent's short list of environment names (`platform.program_environment`), at
  low priority. An index address carrying a user name or password is refused: the command pip
  runs is written to the install log.
- **Refused before anything is fetched** where the machine cannot run it: not Linux, a processor
  or Python with no set (a free-threaded Python among them), a C library older than the set's
  wheels need, no NVIDIA card of compute capability 7.5 or higher, or a driver below the set's
  floor — and where the engines' volume has no room for it above the owner's floor.
- **Kept only if it runs**: the version's own `vllm --version` must answer with the version, and
  its first lines must name the interpreter of the environment where the version will live.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shutil
import sys
import sysconfig
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

from . import platform as host_platform

if TYPE_CHECKING:  # the installer's types; the module itself is imported only where used (below)
    import httpx

    from .installs import Progress, Release, Room, Target

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
    #: The oldest glibc every wheel in the set runs on ("2.31"), where the set says — the
    #: manylinux level it was resolved for. None: not said, and not checked.
    glibc: Optional[str] = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LockedSet":
        glibc = data.get("glibc")
        return cls(
            vllm=str(data["vllm"]), python=str(data["python"]), platform=str(data["platform"]),
            driver_floor=str(data["driver_floor"]), requirements=str(data["requirements"]),
            size_bytes=int(data["size_bytes"]), glibc=str(glibc) if glibc is not None else None,
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
    return reasons + card_reasons(accelerators, minimum=minimum, driver_floor=driver_floor)


def card_reasons(
    accelerators: Iterable[Mapping[str, Any]],
    *,
    minimum: tuple[int, int] = MIN_CAPABILITY,
    driver_floor: Optional[str] = None,
) -> list[str]:
    """Every reason this machine's cards cannot run vLLM, in words — none where they can: no
    NVIDIA card, a card below `minimum` or whose compute capability could not be read, and where
    `driver_floor` is given, a driver below it. Every card counts, whichever the owner lets vLLM
    use (`CUDA_VISIBLE_DEVICES`)."""
    reasons: list[str] = []
    accelerators = list(accelerators)
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
    """Every set the agent carries: `tools/vllm-lock` writes them for each release. None are read
    where it runs from an archive — the agent the pool ships to the hosts it rents, which never
    install an engine."""
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


# --- the installer's source for vLLM (D157, D158) ---

#: PyPI's simple index, and where a test or a mirror points instead — the machine owner's
#: environment, never the pool's word, as `GPM_GITHUB_API` is for GitHub. A value that is not an
#: http(s) address names a directory of wheels, read with no index at all (`--no-index
#: --find-links`). Wherever the files come from, each must match the digest its set records.
#: A directory must hold the set's own wheels and no other build of the same release: pip
#: chooses among a directory's files by their tags before it checks a digest, so a better-tagged
#: wheel beside the one a set names refuses the version. An index lists each file with its digest,
#: and pip chooses only among the files the set names. An address with a user name or password is
#: refused: the command pip runs, and so the address, is written to the install log.
DEFAULT_INDEX = "https://pypi.org/simple"
INDEX_ENV = "GPM_PYPI_INDEX"

#: How much room a version asks for on the engines' volume, as a multiple of its wheels' size.
#: pip, under `--require-hashes`, downloads every wheel of the set into its temporary directory —
#: `TMPDIR`, a work directory beside the versions — while it resolves, before it installs any,
#: and keeps them all until it exits; it keeps no cache (`--no-cache-dir`). So at the most the
#: volume holds every wheel, the environment they unpack to, and the `.pyc` pip compiles beside
#: each Python file. vLLM 0.32.0's sets, read from each wheel's own listing of its files: 4.02 GB
#: of wheels unpack to 8.51 GB on x86_64, 0.40 GB of it Python source, whose `.pyc` take about as
#: much again — some 12.9 GB, 3.2 times the wheels (3.1 on aarch64: 4.07 GB to 8.18 GB). Four
#: times leaves about three-quarters of the wheels' size to spare — 3 GB at 0.32.0 — as sets grow,
#: and is what an archive counts (`installs.UNPACK_MARGIN`). Until a real install measures one,
#: an estimate (decisions.md, Unverified assumptions).
UNPACKED_MARGIN = 4

#: Making the environment: seconds (ensurepip installs pip into it from the wheel Python carries).
VENV_TIMEOUT_S = 10 * 60.0
#: Installing a set: gigabytes from PyPI, tens of minutes on a slow line. Three hours means
#: something is wrong.
INSTALL_TIMEOUT_S = 3 * 3600.0
#: The version's own `vllm --version`, which imports torch: seconds, a minute or two from a cold disk.
CHECK_TIMEOUT_S = 5 * 60.0

#: Inside a version: its environment, and the set it was installed from — the record of exactly
#: what is in it.
VENV = "venv"
REQUIREMENTS = "requirements.txt"
#: Beside the versions: pip's temporary directory while it installs — its `TMPDIR`, so the wheel
#: in flight lands on the volume the room was asked of — removed when it ends, and cleared by the
#: next install where an agent stopped mid-way left it; and the last install's output.
WORK = "pip-work"
INSTALL_LOG = "install.log"

#: The installer's names for processors (`platform.architecture`), as vLLM's wheels and
#: `platform.machine()` name them.
PROCESSORS = {"amd64": "x86_64", "arm64": "aarch64"}
_SYSTEM_NAMES = {host_platform.LINUX: "Linux", host_platform.MACOS: "macOS", host_platform.WINDOWS: "Windows"}

#: What pip says when a file is not the one the requirements' digest names.
_HASH_MISMATCH = "DO NOT MATCH THE HASHES"

#: How this module runs a command: `platform.run_command`, or a test's stand-in for it.
AsyncRunner = Callable[..., Any]


def _installs() -> Any:
    """The installer's module, imported where it is used, as `sets_for` imports it: the installer
    reaches the engines' modules through its own imports, and they reach this one."""
    from . import installs

    return installs


def this_python() -> str:
    """The agent's own Python minor version ("3.12"): every environment is made with it, and a set
    must be for it. A free-threaded build is "3.13t": its extension modules are built for another
    ABI (`cp313t`), and every set's wheels for the regular one, so no set is for it."""
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    return f"{sys.version_info.major}.{sys.version_info.minor}" + ("t" if free_threaded else "")


def python_named(python: str) -> str:
    """A Python minor version as the owner is told it, saying what a free-threaded one's `t` is."""
    if python.endswith("t"):
        return f"Python {python} (free-threaded, which vLLM's wheels are not built for)"
    return f"Python {python}"


def machine_glibc() -> Optional[str]:
    """This machine's glibc version ("2.35") — None where its C library is another, or where it
    is not Linux."""
    try:
        said = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        return None
    name, _, version = (said or "").partition(" ")
    return version if name == "glibc" and driver_key(version) else None


def nvidia_cards() -> Optional[list[dict[str, Any]]]:
    """This machine's NVIDIA cards as the facts read them from `nvidia-smi` — None where it does
    not answer."""
    from . import facts

    return facts.nvidia_cards(facts.Probes())


def index_arguments(environ: Mapping[str, str]) -> list[str]:
    """Where pip fetches the wheels from: PyPI, or what the owner's environment names instead."""
    named = (environ.get(INDEX_ENV) or "").strip() or DEFAULT_INDEX
    netloc = urllib.parse.urlsplit(named).netloc
    if "@" in netloc:
        # Said without the credential: this refusal reaches the facts, and the pool's console.
        shown = named.replace(netloc, netloc.rpartition("@")[2], 1)
        raise _installs().InstallRefused(
            f"{INDEX_ENV} names an index with a user name or password in its address ({shown}): the agent "
            "writes each command it runs to its install log, so the credential would be kept there in plain "
            "text; name an index that needs none, such as a mirror on this machine or its network"
        )
    if named.startswith(("https://", "http://")):
        return ["--index-url", named]
    directory = Path(named)
    if not directory.is_absolute() or not directory.is_dir():
        raise _installs().InstallRefused(
            f"{INDEX_ENV} names {named!r}, which is neither an http(s) index nor a directory of wheels"
        )
    return ["--no-index", "--find-links", str(directory)]


def venv_argv(python: str, venv: Path) -> list[str]:
    """A private environment for one version, made by the agent's own interpreter, isolated (`-I`)."""
    return [python, "-I", "-m", "venv", str(venv)]


#: What pip is given beside the agent's short list of names: the null device for its
#: configuration file, which makes pip read no configuration file at all — not the machine's
#: (`/etc/pip.conf`, `/etc/xdg/pip/pip.conf`) nor the environment's own `pip.conf`, which
#: `--isolated` still reads (pip's `Configuration._load_config_files`). ensurepip does the same.
PIP_SETTINGS = {"PIP_CONFIG_FILE": os.devnull}


def pip_argv(venv: Path, requirements: Path, index: Sequence[str]) -> list[str]:
    """pip, in the version's environment, installing exactly the set: wheels only, so no build
    script runs; nothing beyond the list; every file checked against the digest it records; none
    of the owner's pip settings (`--isolated`, and `PIP_SETTINGS` in its environment), and no
    cache left behind."""
    return [
        str(venv / "bin" / "python"), "-I", "-m", "pip", "install", "--isolated",
        "--require-hashes", "--only-binary=:all:", "--no-deps", "--no-input",
        "--disable-pip-version-check", "--no-cache-dir", "--progress-bar=off",
        *index, "-r", str(requirements),
    ]


def check_argv(venv: Path) -> list[str]:
    """The version's own `vllm --version`, run by its environment's interpreter, isolated — the
    program the agent will run, read as the script it is, wherever its first line points."""
    return [str(venv / "bin" / "python"), "-I", str(venv / "bin" / "vllm"), "--version"]


def _last_words(said: Any) -> str:
    lines = [line.strip() for line in said.output.splitlines() if line.strip()]
    how = f"exit code {said.code}" if said.code is not None else "it did not run to the end"
    return f"{how}: {lines[-1][:300] if lines else 'it said nothing'}"


def _head(data: bytes) -> tuple[bytes, bytes]:
    """A script's first three lines, where its interpreter is named, and the rest."""
    head = b"\n".join(data.split(b"\n", 3)[:3])
    return head, data[len(head):]


def move_environment(venv: Path, to: Path) -> None:
    """Make an environment made at `venv` right for `to`, where it will live once whole.

    A version is made in `<version>.partial` and renamed into place once whole (`installs`), and
    an environment names its own place in a few files: the first lines of the programs pip wrote
    (`#!<venv>/bin/python`, or for a long path a line of `/bin/sh` naming it), and its `activate`
    scripts and `pyvenv.cfg`. Each is rewritten to name `to`, so the version's `vllm` runs from
    where it lives. Nothing else names the place: what pip installs is located from the
    interpreter's own, and Python's own programs in `bin` are links to the agent's interpreter.
    """
    if venv == to:
        return
    old, new = os.fsencode(str(venv)), os.fsencode(str(to))
    # Both as Python names its own interpreter, and so as pip writes it: absolute, with no `..`.
    old, new = os.path.abspath(old), os.path.abspath(new)
    for path in [*sorted((venv / "bin").iterdir()), venv / "pyvenv.cfg"]:
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as handle:
            script = handle.read(2) == b"#!"
        if not script and not (path.name.startswith("activate") or path.name == "pyvenv.cfg"):
            continue  # a program, not a script: nothing in it is ours to change
        data = path.read_bytes()
        if script:
            head, rest = _head(data)
            changed = head.replace(old + b"/", new + b"/") + rest
        else:
            changed = data.replace(old, new)
        if changed != data:
            path.write_bytes(changed)


#: The second line of pip's form for a long interpreter path: `'''exec' <python> "$0" "$@"`, the
#: path in double quotes where it holds a space.
_SH_EXEC = re.compile(rb"""'''exec' ("[^"]*"|\S+) "\$0" "\$@"$""")


def interpreter_named(program: Path) -> Optional[bytes]:
    """The interpreter a script pip wrote names in its first lines — `#!<python>`, or `#!/bin/sh`
    and a line running `<python>` — or None where it names none this reads."""
    with Path(program).open("rb") as handle:
        lines = _head(handle.read(64 * 1024))[0].split(b"\n")
    if not lines[0].startswith(b"#!"):
        return None
    first = lines[0][2:].strip()
    if first != b"/bin/sh":
        return first
    found = _SH_EXEC.match(lines[1].rstrip()) if len(lines) > 1 else None
    return found.group(1).strip(b'"') if found else None


def _says_version(output: str, version: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(version)}(?![\w.])", output) is not None


class VllmSource:
    """vLLM's versions made into version directories: a private environment each, filled from
    PyPI with exactly the wheels its locked set names (D157, D158).

    `latest` is the newest version the agent carries a set for, for its own Python and this
    machine's processor — no question is asked of PyPI for it — and a version it carries no set
    for is refused, saying so. Nothing is fetched before the machine is found able to run it.
    """

    engine = "vllm"
    #: PyPI's wheels carry no signature the agent could pin: each file is checked against the
    #: digest its set records, and only that.
    signature = "not checked"

    def __init__(
        self,
        target: Target,
        *,
        python: Optional[str] = None,
        sets: Optional[Sequence[LockedSet]] = None,
        cards: Optional[Callable[[], Optional[list[dict[str, Any]]]]] = None,
        glibc: Optional[Callable[[], Optional[str]]] = None,
        run: Optional[AsyncRunner] = None,
        interpreter: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.target = target
        #: The Python minor version and processor a set must be for: the agent's own interpreter
        #: makes the environment, on this machine.
        self.python = python or this_python()
        self.processor = PROCESSORS.get(target.arch or "")
        #: The sets to choose from — None: the ones the agent carries (`SETS_DIR`), read when asked.
        self.sets = sets
        self.cards = cards or nvidia_cards
        self.glibc = glibc or machine_glibc
        self.run = run or host_platform.run_command
        self.interpreter = interpreter or sys.executable
        #: Where `INDEX_ENV` is read — None: the environment the owner started the agent in.
        self.environ = environ

    def program(self, version_dir: Path) -> Path:
        return program_in(version_dir)

    # --- versions: the sets the agent carries ---

    def _mine(self) -> list[LockedSet]:
        return sets_for(self.python, self.processor or "", self.sets)

    def _refuse_the_machine(self) -> None:
        """Refuse what no set can change: the system and the processor."""
        refused = _installs().InstallRefused
        if self.target.os != host_platform.LINUX:
            system = _SYSTEM_NAMES.get(self.target.os, self.target.os)
            raise refused(
                f"vLLM publishes its builds for Linux only, and this machine runs {system}: "
                "it serves on Ollama (`engine: auto` chooses it)"
            )
        if self.processor is None:
            raise refused(f"vLLM publishes no build for the {self.target.arch or 'unknown'} processor")

    def _no_set(self, version: Optional[str]) -> str:
        """Why there is no set to install, in words for the machine's owner."""
        carried = list(load_sets() if self.sets is None else self.sets)
        here = f"{python_named(self.python)} on {self.processor}"
        if version is not None and any(s.vllm == version for s in carried):
            others = sorted({f"Python {s.python} on {s.platform}" for s in carried if s.vllm == version})
            return f"vLLM {version} has no locked set for {here}, only for {', '.join(others)}"
        versions = sorted({s.vllm for s in carried}, key=_installs().version_key, reverse=True)
        if version is not None:
            return (
                f"vLLM {version} is not a version this agent carries a locked set for "
                f"({'it carries ' + ', '.join(versions) if versions else 'it carries none'}); "
                "newer versions come with a release of the agent"
            )
        if not carried:
            return "this agent carries no locked set for any vLLM version"
        pythons = sorted({s.python for s in carried}, key=lambda p: tuple(int(n) for n in p.split(".")))
        return (
            f"this agent carries no vLLM set for {here}: its sets are for Python "
            f"{', '.join(pythons)} on {', '.join(sorted({s.platform for s in carried}))}; run the agent "
            "with one of those"
        )

    def _set_for(self, version: str) -> LockedSet:
        found = next((s for s in self._mine() if s.vllm == version), None)
        if found is None:
            raise _installs().InstallRefused(self._no_set(version))
        return found

    @staticmethod
    def _release(locked: LockedSet) -> Release:
        # No files of its own: what a version is, is its set.
        return _installs().Release(version=locked.vllm, tag=f"v{locked.vllm}", assets=())

    async def latest(self, http: httpx.AsyncClient) -> Release:
        """The newest version the agent carries a set for, here."""
        self._refuse_the_machine()
        found = self._mine()
        if not found:
            raise _installs().InstallRefused(self._no_set(None))
        return self._release(found[0])

    async def release(self, http: httpx.AsyncClient, version: str) -> Release:
        installs = _installs()
        if not installs.VERSION.fullmatch(version):
            raise installs.InstallRefused(f"{version!r} is not a version")
        self._refuse_the_machine()
        return self._release(self._set_for(version))

    # --- staging a version ---

    async def reasons_against(self, locked: LockedSet) -> list[str]:
        """Every reason this machine cannot run `locked`, in words — none where it can: its C
        library, its NVIDIA cards and their driver (`machine_can_run`, at vLLM's own floor)."""
        reasons: list[str] = []
        if locked.glibc is not None:
            have = self.glibc()
            need = driver_key(locked.glibc)
            if have is None:
                reasons.append(f"its wheels need glibc {locked.glibc} or later, and this machine's C library is not glibc")
            elif need is not None and (driver_key(have) or ()) < need:
                reasons.append(f"its wheels need glibc {locked.glibc} or later, and this machine has {have}")
        cards = await asyncio.to_thread(self.cards)
        if cards is None:
            reasons.append("nvidia-smi does not answer here: no NVIDIA card, or no driver for one")
        else:
            reasons += machine_can_run(
                cards, os_name="Linux", arch=self.processor or "", minimum=MIN_CAPABILITY,
                driver_floor=locked.driver_floor,
            )
        return reasons

    async def stage(
        self, http: httpx.AsyncClient, release: Release, into: Path, progress: Progress, room: Optional[Room] = None,
    ) -> None:
        installs = _installs()
        refused = installs.InstallRefused
        version = release.version
        self._refuse_the_machine()
        locked = self._set_for(version)
        reasons = await self.reasons_against(locked)
        if reasons:
            raise refused(
                f"vLLM {version} is not installed here: {'; '.join(reasons)} — `gpm-agent prerequisites` "
                "at a terminal here says what the machine needs"
            )
        index = index_arguments(os.environ if self.environ is None else self.environ)

        # As Python names its own interpreter, and so as pip writes it in each program's first
        # line: absolute, with no `..` — or the environment would not be found in them to move.
        into = Path(os.path.abspath(into))
        # Where the version will live once whole: the installer renames `<version>.partial` into
        # place, and the environment must name that place, not this one.
        final = into.with_name(into.name[: -len(installs.PARTIAL)]) if into.name.endswith(installs.PARTIAL) else into
        engine_dir = into.parent
        work = engine_dir / WORK
        # An install stopped half-way left its wheels here: gone before the room is asked for.
        await asyncio.to_thread(shutil.rmtree, work, True)
        if room is not None:
            room(locked.size_bytes * UNPACKED_MARGIN, (
                f"its locked set's wheels, {installs.size_words(locked.size_bytes)}, held while they install, "
                f"and the environment they make: up to {UNPACKED_MARGIN} times that"
            ))
        work.mkdir(mode=0o700, parents=True)
        log = engine_dir / "logs" / INSTALL_LOG
        log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        log.write_text(f"--- gpm-agent: installing vLLM {version} for Python {self.python} on {self.processor}\n")
        venv = into / VENV
        try:
            progress("downloading", 0, locked.size_bytes)
            made = await self.run(
                venv_argv(self.interpreter, venv), env=host_platform.program_environment({}),
                timeout_s=VENV_TIMEOUT_S, log=log,
            )
            if made.code != 0:
                raise refused(
                    f"a private Python environment for vLLM {version} could not be made with this agent's "
                    f"Python ({_last_words(made)}): it needs Python's venv module (python3-venv on Debian and "
                    "Ubuntu) — `gpm-agent prerequisites` at a terminal here installs it, or says how"
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
                    raise refused(
                        f"a file fetched for vLLM {version} does not match the digest its locked set records; "
                        f"nothing of it is kept (the log: {log})"
                    )
                raise refused(f"installing vLLM {version}'s locked set failed ({_last_words(installed)}); the whole log is {log}")
        finally:
            await asyncio.to_thread(shutil.rmtree, work, True)

        progress("verifying", 0, None)
        if not program_in(into).is_file():
            raise refused(f"vLLM {version}'s locked set installed no `vllm` program")
        await asyncio.to_thread(move_environment, venv, final / VENV)
        # The check below runs the program with the environment's interpreter, as a script, so
        # it would pass whatever the first line names: that line is what the launcher runs by.
        named = await asyncio.to_thread(interpreter_named, program_in(into))
        if named != os.fsencode(str(final / VENV / "bin" / "python")):
            said = os.fsdecode(named) if named is not None else "no interpreter"
            raise refused(
                f"vLLM {version}, installed, has a `vllm` that names {said} to run it, not the interpreter of "
                f"the environment it will live in ({final / VENV / 'bin' / 'python'}); nothing of it is kept"
            )
        checked = await self.run(
            check_argv(venv), env=host_platform.program_environment({}), timeout_s=CHECK_TIMEOUT_S, log=log,
        )
        if checked.code != 0:
            raise refused(f"vLLM {version}, installed, does not run: {_last_words(checked)}; nothing of it is kept")
        if not _says_version(checked.output, version):
            raise refused(f"vLLM {version}, installed, says it is another version ({_last_words(checked)}); nothing of it is kept")
