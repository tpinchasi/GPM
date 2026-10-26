"""whisper.cpp, built on this machine from its release's source, or taken from its own Windows
archives (D152, D153).

whisper.cpp publishes no program for macOS, and its Linux archives are built for one processor
and no card. Homebrew's `whisper-server` links Homebrew's shared `ggml`, which moves with other
formulas, so a version kept on disk could not be gone back to. So on macOS and Linux each version
is built here, statically, from the source of the commit its tag names — the owner's choice
(2026-10-26) — and on Windows it comes from the archives whisper.cpp publishes.

What keeps a build inside the agent's bounds:

- **The source is the tag's commit.** The tag is read from the repository (`git/ref/tags/v…`,
  through an annotated tag to its commit) and the archive is fetched by that commit, through the
  same API (`GPM_GITHUB_API`). GitHub writes the commit's name into the archive it makes of a
  commit; one that names another is refused. No release publishes a digest for such an archive:
  the commit is what pins it.
- **A fixed build.** `cmake` runs with argument lists written here (`configure_argv`,
  `build_argv`): a Release build of the server alone, static — nothing it links can move under a
  kept version — with Metal on a Mac, CUDA where NVIDIA's compiler is installed, and the CPU
  otherwise; on half the machine's cores and at low priority, so the owner's own work comes
  first; with the agent's credentials withheld. Nothing from the source is run but its build.
- **Only the program is kept.** `whisper-server` is copied from the build into the version's
  `bin/`, checked to run (`--help`), and the build's directory — beside the versions, never in
  the models directory — removed. Its log stays, the last build's only.
- **`cmake` from PyPI where the machine has none**: Kitware's official wheels, installed into a
  private environment under the engines directory (`installs.TOOLS`), pinned to one version and
  to the digests PyPI publishes for its wheels, which pip checks (`--require-hashes`).
- **A compiler is the owner's to install** — Xcode's command-line tools on macOS, a C++ compiler
  and `make` on Linux. Without one a version is refused, naming `gpm-agent prerequisites`. Git is
  not needed.

On Windows: the archive whisper.cpp publishes for this machine — CUDA 12.4 or 11.8 by the NVIDIA
driver, else the CPU with OpenBLAS on x64, the CPU on arm64 (`installs.WHISPER_CPP`) — from the
version's own release where it carries one (1.9.2 and before), else from the build tag released
with it: the one whose tag names the same commit, read from the repository, never guessed by its
date. Checked against the digest GitHub publishes for it; whisper.cpp signs none of its programs.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from . import platform as host_platform
from .installs import (
    DOWNLOADS,
    VERSION,
    Asset,
    BuiltRelease,
    GitHubReleases,
    InstallRefused,
    Progress,
    Release,
    Target,
    _api,
    _headers,
    _keep_only,
    unpack,
)
from .platform import LINUX, MACOS, WINDOWS
from .whisper_cpp import program_runs

#: How this module runs a command: `platform.run_command`, or a test's stand-in for it.
AsyncRunner = Callable[..., Awaitable[host_platform.Ran]]

#: Beside the versions: the source and its build while it runs, removed when it ends — and left
#: by an agent stopped mid-build, then cleared by the next.
WORK = "build"
#: The last build's output, beside the versions.
BUILD_LOG = "build.log"

#: GitHub's archive of whisper.cpp's source is about 10 MB (v1.9.5: 9.6 MB). Nothing near this
#: is it, and nothing larger is written to the owner's disk.
MAX_SOURCE_BYTES = 512 * 1024 * 1024

#: The longest a configure or a build may take. A CUDA build for one card on half the cores of a
#: small machine takes tens of minutes; two hours means something is wrong.
BUILD_TIMEOUT_S = 2 * 3600.0

#: The longest a quick look takes: `xcode-select -p`, `nvidia-smi`, `cmake --version`.
LOOK_TIMEOUT_S = 30.0

#: Making cmake's private environment and installing its wheel: a 30–55 MB download.
TOOL_TIMEOUT_S = 15 * 60.0

#: Where NVIDIA's installers put the CUDA compiler, for a machine whose PATH does not name it.
CUDA_NVCC = "/usr/local/cuda/bin/nvcc"

#: Every build: Release, static (whisper and ggml linked into the program), the examples — the
#: server is one — and no tests, and no SDL2, which only the microphone examples use.
CONFIGURE_FLAGS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DWHISPER_BUILD_TESTS=OFF",
    "-DWHISPER_BUILD_EXAMPLES=ON",
    "-DWHISPER_SDL2=OFF",
)
#: On a Mac: Metal, with its shaders built into the program rather than read from beside it.
METAL_FLAGS = ("-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON")
#: Checked against whisper.cpp v1.9.5's CMakeLists (and ggml's beside it), 2026-10-26: these are
#: its options' names, and its server's target is `whisper-server`, written to `<build>/bin/`.

#: `cmake` from PyPI: Kitware's official wheels (https://pypi.org/project/cmake/), version 4.4.4,
#: uploaded 2026-10-04. Each digest was read on 2026-10-26 from PyPI's JSON API and found the same
#: in its simple index. pip installs a file only if its sha256 is listed here
#: (`--require-hashes`), so a file replaced or altered on the way is refused. One universal
#: wheel serves both kinds of Mac; Linux's are for glibc (manylinux2014). A newer cmake is taken
#: by changing this table, digests and all, in a release of the agent.
CMAKE_VERSION = "4.4.4"
CMAKE_WHEELS = {
    "cmake-4.4.4-py3-none-macosx_10_10_universal2.whl":
        "883962a72b1b3a16ff445427f24c8caca9797dd9e563bbe208cbcd89e877f9d3",
    "cmake-4.4.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl":
        "f773a0544c66370408f451acd487c2c643b0dfebb48c5d64f3638bc1bb820238",
    "cmake-4.4.4-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl":
        "b3029f586853e01ddf2824c1ff124a1c5284f8ae897994758e9c4eeb45903b58",
}
#: The machines those wheels are for.
CMAKE_MACHINES = {(MACOS, "arm64"), (MACOS, "amd64"), (LINUX, "amd64"), (LINUX, "arm64")}
REQUIREMENTS = "gpm-requirements.txt"
#: Written last into cmake's environment, once it is whole and runs.
TOOL_COMPLETE = ".gpm-complete"

#: The tags this module puts in an address: a version's (`v1.9.5`) or a build's (`b5454`).
_TAG = re.compile(r"^(v\d+(\.\d+){1,3}(-[0-9A-Za-z.]+)?|b\d{1,9})$")
_BUILD_TAG = re.compile(r"^b\d{1,9}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
#: An annotated tag may name another tag; whisper.cpp's name their commit directly.
_TAG_DEPTH = 3
#: Releases are listed a hundred at a time, newest first, at most this many pages.
LIST_PAGE = 100
LIST_PAGES = 5
#: Build tags whose commit is read before giving up: the one the release says it was made from
#: first, then those listed nearest the version.
MAX_CANDIDATES = 4
_DRIVER = re.compile(r"^\s*(\d+)\.(\d+)")

#: Credentials the agent may hold. Neither a build nor pip has a use for them.
_WITHHELD = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN")


# --- the commands, each a fixed argument list ---


def configure_argv(cmake: str, source: Path, build: Path, os_name: str, nvcc: Optional[str]) -> list[str]:
    """`cmake`'s configure step: the source and build directories the agent made, and the flags
    above — Metal on a Mac, CUDA where `nvcc` was found (named to cmake, which may not find it on
    PATH), the CPU otherwise."""
    argv = [cmake, "-S", str(source), "-B", str(build), *CONFIGURE_FLAGS]
    if os_name == MACOS:
        argv += METAL_FLAGS
    elif nvcc:
        argv += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
    return argv


def build_argv(cmake: str, build: Path, jobs: int) -> list[str]:
    """`cmake`'s build step: the server's target alone, on `jobs` cores."""
    return [cmake, "--build", str(build), "--config", "Release", "--target", "whisper-server", "-j", str(jobs)]


def jobs_for(cpu_count: Optional[int]) -> int:
    """Half the machine's cores, at least one: the owner's machine stays theirs while it builds."""
    return max(1, (cpu_count or 1) // 2)


def venv_argv(python: str, venv: Path) -> list[str]:
    """A private environment for cmake, made by the agent's own interpreter, isolated (`-I`)."""
    return [python, "-I", "-m", "venv", str(venv)]


def pip_argv(venv: Path) -> list[str]:
    """pip, in that environment, installing exactly the requirements file written here: wheels
    only, nothing they depend on, every file checked against a listed digest."""
    return [
        str(venv / "bin" / "python"), "-I", "-m", "pip", "install",
        "--require-hashes", "--only-binary=:all:", "--no-deps", "--no-input", "--disable-pip-version-check",
        "-r", str(venv / REQUIREMENTS),
    ]


def requirements_text() -> str:
    """The pin as pip reads it: one version, and the digest of every wheel it may take."""
    hashes = "".join(f" \\\n    --hash=sha256:{digest}" for digest in sorted(CMAKE_WHEELS.values()))
    return f"cmake=={CMAKE_VERSION}{hashes}\n"


def _environment(**more: str) -> dict[str, str]:
    """The agent's own environment, less its credentials, with `more`."""
    environment = {name: value for name, value in os.environ.items() if name not in _WITHHELD}
    environment.update(more)
    return environment


def _last_words(said: host_platform.Ran) -> str:
    lines = [line.strip() for line in said.output.splitlines() if line.strip()]
    how = f"exit code {said.code}" if said.code is not None else "it did not run to the end"
    return f"{how}: {lines[-1][:300] if lines else 'it said nothing'}"


# --- the source archive ---


def archive_commit(path: Path) -> Optional[str]:
    """The commit a GitHub source archive says it holds: `git archive` writes it as the archive's
    first, global header (`comment`). None where the archive says none."""
    try:
        with tarfile.open(path, mode="r:gz") as handle:
            handle.next()
            comment = handle.pax_headers.get("comment")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise InstallRefused(f"the source archive could not be read: {exc or type(exc).__name__}") from exc
    return comment if isinstance(comment, str) else None


def unpack_source(archive: Path, commit: str, into: Path) -> Path:
    """Unpack the source of `commit` into `into`, safely (`installs.unpack`), and return its one
    top directory."""
    named = archive_commit(archive)
    if named != commit:
        raise InstallRefused(
            f"the source archive GitHub sent says it is of commit {named or '(none named)'}, not "
            f"{commit}; it is not built"
        )
    into.mkdir(mode=0o700)
    unpack(archive, "source.tar.gz", into)
    tops = list(into.iterdir())
    if len(tops) != 1 or tops[0].is_symlink() or not tops[0].is_dir() or not (tops[0] / "CMakeLists.txt").is_file():
        raise InstallRefused("the source archive does not hold one directory with whisper.cpp's CMakeLists.txt")
    return tops[0]


def _windows_programs(version_dir: Path, program: str) -> list[Path]:
    """Every `<program>.exe` in an unpacked Windows archive, wherever it put them."""
    name = f"{program}.exe".lower()
    return sorted(
        path for path in version_dir.rglob("*")
        if path.name.lower() == name and path.is_file() and not path.is_symlink()
    )


def _version_release(body: Any) -> bool:
    tag = body.get("tag_name") if isinstance(body, dict) else None
    return (
        isinstance(tag, str) and tag.startswith("v") and bool(VERSION.fullmatch(tag[1:]))
        and not body.get("draft")
    )


class WhisperCppSource(GitHubReleases):
    """whisper.cpp's versions made into version directories: built here on macOS and Linux, from
    its own archives on Windows."""

    signature = "not checked"

    def __init__(
        self,
        table: BuiltRelease,
        target: Target,
        *,
        tools_dir: Optional[Path] = None,
        run: Optional[AsyncRunner] = None,
        which: Optional[Callable[[str], Optional[str]]] = None,
        cpu_count: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        super().__init__(table.repo)
        self.engine = table.engine
        self.table, self.target = table, target
        #: Where cmake from PyPI is kept, beside the engines' directories; None: never fetched.
        self.tools_dir = tools_dir
        self.run = run or host_platform.run_command
        self.which = which or shutil.which
        self.cpu_count = cpu_count or os.cpu_count

    def program(self, version_dir: Path) -> Path:
        if self.target.os == WINDOWS:
            usual = version_dir / "Release" / f"{self.table.program}.exe"
            if usual.is_file():
                return usual
            found = _windows_programs(version_dir, self.table.program) if version_dir.is_dir() else []
            return found[0] if len(found) == 1 else usual
        return version_dir / "bin" / self.table.program

    # --- versions ---

    async def latest(self, http: httpx.AsyncClient) -> Release:
        """The newest version that is not a pre-release. whisper.cpp also releases build tags, not
        always as pre-releases (`b4938`, beside a 1.9.3 that was one), so GitHub's latest can be
        one of them; the versions are then read from the listing, newest first."""
        body = await self._get(http, f"/repos/{self.repo}/releases/latest", f"{self.repo} has no release on GitHub")
        if _version_release(body):
            return self._release(body)
        listed = await self._get(
            http, f"/repos/{self.repo}/releases?per_page={LIST_PAGE}&page=1", f"{self.repo} has no releases on GitHub",
        )
        for entry in listed if isinstance(listed, list) else []:
            if _version_release(entry) and not entry.get("prerelease"):
                return self._release(entry)
        raise InstallRefused(f"none of {self.repo}'s latest {LIST_PAGE} releases is a version that is not a pre-release")

    async def commit_of(self, http: httpx.AsyncClient, tag: str) -> str:
        """The commit `tag` names, read from the repository — through an annotated tag to it."""
        if not _TAG.fullmatch(tag):
            raise InstallRefused(f"{tag!r} is not a tag this agent reads")
        body = await self._get(http, f"/repos/{self.repo}/git/ref/tags/{tag}", f"{self.repo} has no tag {tag}")
        for _ in range(_TAG_DEPTH):
            pointed = body.get("object") if isinstance(body, dict) else None
            kind = pointed.get("type") if isinstance(pointed, dict) else None
            sha = pointed.get("sha") if isinstance(pointed, dict) else None
            if not isinstance(sha, str) or not _COMMIT.fullmatch(sha):
                raise InstallRefused(f"GitHub's answer for {self.repo}'s tag {tag} names no commit")
            if kind == "commit":
                return sha
            if kind != "tag":
                raise InstallRefused(f"{self.repo}'s tag {tag} names a {kind}, not a commit")
            body = await self._get(http, f"/repos/{self.repo}/git/tags/{sha}", f"{self.repo}'s tag {tag} could not be read")
        raise InstallRefused(f"{self.repo}'s tag {tag} leads through too many tags to reach a commit")

    async def stage(self, http: httpx.AsyncClient, release: Release, into: Path, progress: Progress) -> None:
        if self.target.os == WINDOWS:
            await self._stage_archive(http, release, into, progress)
        elif self.target.os in (MACOS, LINUX):
            await self._stage_build(http, release, into, progress)
        else:
            raise InstallRefused(f"this agent installs whisper.cpp on macOS, Linux and Windows, not on {self.target.os}")

    # --- macOS and Linux: built here ---

    async def _stage_build(self, http: httpx.AsyncClient, release: Release, into: Path, progress: Progress) -> None:
        engine_dir = into.parent
        await self._need_compiler()
        commit = await self.commit_of(http, release.tag)
        cmake = await self._cmake(progress)
        work = engine_dir / WORK
        await asyncio.to_thread(shutil.rmtree, work, True)  # an earlier build stopped half-way
        work.mkdir(mode=0o700, parents=True)
        log = engine_dir / "logs" / BUILD_LOG
        program = self.program(into)
        try:
            archive = work / "source.tar.gz"
            await self._fetch_source(http, commit, archive, progress)
            progress("unpacking", 0, None)
            source = await asyncio.to_thread(unpack_source, archive, commit, work / "src")
            archive.unlink()

            progress("building", 0, None)
            build = work / "out"
            nvcc = self._nvcc() if self.target.os == LINUX else None
            log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            log.write_text(f"--- gpm-agent: building whisper.cpp {release.tag}, commit {commit}\n")
            # Git, if the machine has it, is asked by the build for the commit's name: it must not
            # find a repository above the work directory and take that one's.
            environment = _environment(GIT_CEILING_DIRECTORIES=str(work))
            said = await self.run(
                configure_argv(cmake, source, build, self.target.os, nvcc),
                env=environment, timeout_s=BUILD_TIMEOUT_S, low_priority=True, log=log,
            )
            if said.code == 0:
                said = await self.run(
                    build_argv(cmake, build, jobs_for(self.cpu_count())),
                    env=environment, timeout_s=BUILD_TIMEOUT_S, low_priority=True, log=log,
                )
            if said.code != 0:
                raise InstallRefused(
                    f"building whisper.cpp {release.tag} failed ({_last_words(said)}); the whole log is {log}"
                )
            built = build / "bin" / self.table.program
            if built.is_symlink() or not built.is_file():
                raise InstallRefused(f"building whisper.cpp {release.tag} made no {self.table.program}; the log is {log}")
            program.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, built, program)
            program.chmod(0o755)
        finally:
            await asyncio.to_thread(shutil.rmtree, work, True)
        problem = await program_runs(program, self.run)
        if problem:
            raise InstallRefused(f"whisper.cpp {release.tag}, built here, does not run: {problem}")

    async def _need_compiler(self) -> None:
        """Refuse, before anything is fetched, on a machine that cannot compile."""
        if self.target.os == MACOS:
            if (await self.run(["xcode-select", "-p"], timeout_s=LOOK_TIMEOUT_S)).code == 0:
                return
            missing = "Xcode's command-line tools"
        else:
            if self.which("c++") and self.which("make"):
                return
            missing = "a C++ compiler (c++) and make"
        raise InstallRefused(
            f"whisper.cpp is built on this machine, which has no {missing}: run `gpm-agent prerequisites` "
            "at a terminal here — it installs them, or says how — and the next attempt builds it"
        )

    def _nvcc(self) -> Optional[str]:
        """NVIDIA's CUDA compiler, where it is installed: CUDA is built in only then."""
        found = self.which("nvcc")
        if found:
            return found
        return CUDA_NVCC if os.path.isfile(CUDA_NVCC) and os.access(CUDA_NVCC, os.X_OK) else None

    async def _cmake(self, progress: Progress) -> str:
        """`cmake` on this agent's PATH, else the one this agent installed from PyPI — installed
        now if it has not been."""
        found = self.which("cmake")
        if found:
            return found
        if (self.target.os, self.target.arch) not in CMAKE_MACHINES or self.tools_dir is None:
            raise InstallRefused(
                "cmake is not on this agent's PATH, and this agent fetches it from PyPI only for macOS and "
                "for Linux on x86_64 or arm64: install cmake with the system's package manager"
            )
        venv = self.tools_dir / f"cmake-{CMAKE_VERSION}"
        cmake = venv / "bin" / "cmake"
        if (venv / TOOL_COMPLETE).is_file() and cmake.is_file():
            return str(cmake)
        progress("downloading", 0, None)
        self.tools_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for older in self.tools_dir.glob("cmake-*"):
            # Another pin's, or this one's left half-made.
            await asyncio.to_thread(shutil.rmtree, older, True)
        environment = _environment()
        made = await self.run(venv_argv(sys.executable, venv), env=environment, timeout_s=TOOL_TIMEOUT_S)
        if made.code != 0:
            raise InstallRefused(
                f"cmake is not on this agent's PATH, and a private environment for PyPI's could not be made "
                f"({_last_words(made)}): install cmake, or Python's venv module (python3-venv)"
            )
        (venv / REQUIREMENTS).write_text(requirements_text())
        installed = await self.run(pip_argv(venv), env=environment, timeout_s=TOOL_TIMEOUT_S)
        if installed.code != 0:
            raise InstallRefused(f"installing cmake {CMAKE_VERSION} from PyPI failed ({_last_words(installed)})")
        checked = await self.run([str(cmake), "--version"], env=environment, timeout_s=LOOK_TIMEOUT_S)
        if checked.code != 0:
            raise InstallRefused(f"cmake {CMAKE_VERSION} from PyPI does not run here ({_last_words(checked)})")
        (venv / TOOL_COMPLETE).write_text(f"cmake {CMAKE_VERSION}\n")
        return str(cmake)

    async def _fetch_source(self, http: httpx.AsyncClient, commit: str, sink: Path, progress: Progress) -> None:
        """GitHub's archive of the source at `commit`, through the API by the commit's name. It is
        made as it is asked for, so it has no size known in advance and no digest; a cut transfer
        is fetched again whole by the next attempt."""
        url = f"{_api()}/repos/{self.repo}/tarball/{commit}"
        progress("downloading", 0, None)
        try:
            async with http.stream("GET", url, headers=_headers()) as response:
                if response.status_code != 200:
                    raise InstallRefused(f"fetching whisper.cpp's source at {commit[:12]} returned {response.status_code}")
                declared = response.headers.get("content-length")
                total = int(declared) if declared and declared.isdigit() else None
                have = 0
                with sink.open("wb") as out:
                    async for chunk in response.aiter_bytes(1 << 20):
                        have += len(chunk)
                        if have > MAX_SOURCE_BYTES:
                            raise InstallRefused(
                                f"whisper.cpp's source at {commit[:12]} is larger than {MAX_SOURCE_BYTES >> 20} MB; it is not this"
                            )
                        out.write(chunk)
                        progress("downloading", have, total)
        except httpx.HTTPError as exc:
            raise InstallRefused(
                f"fetching whisper.cpp's source stopped: {exc or type(exc).__name__}; the next attempt fetches it again"
            ) from exc

    # --- Windows: the project's own archives ---

    async def _stage_archive(self, http: httpx.AsyncClient, release: Release, into: Path, progress: Progress) -> None:
        names = self.table.windows[await self._windows_kind()]
        asset = next((found for name in names if (found := release.asset(name)) is not None), None)
        where = release.tag
        if asset is None:
            where, asset = await self._build_tag_for(http, release, names)
        if asset.sha256 is None:
            raise InstallRefused(f"{self.repo} {where} publishes no digest for {asset.name}; nothing unchecked is installed")
        downloads = into.parent / DOWNLOADS
        _keep_only(downloads, {asset.sha256})
        progress("downloading", 0, asset.size)
        archive = await self.download(http, asset, downloads, lambda have: progress("downloading", have, asset.size))
        progress("unpacking", 0, 1)
        await asyncio.to_thread(unpack, archive, asset.name, into)
        found = _windows_programs(into, self.table.program)
        if len(found) != 1:
            raise InstallRefused(
                f"{where}'s {asset.name} holds {len(found) or 'no'} {self.table.program}.exe; exactly one is installed"
            )
        problem = await program_runs(found[0], self.run)
        if problem:
            raise InstallRefused(f"whisper.cpp {release.tag} ({asset.name}) does not run here: {problem}")

    async def _windows_kind(self) -> str:
        """Which of the table's Windows archives this machine takes."""
        if self.target.arch == "arm64":
            return "cpu-arm64"
        if self.target.arch != "amd64":
            raise InstallRefused(f"whisper.cpp publishes no Windows build for {self.target.arch or 'this processor'}")
        driver = await self._nvidia_driver()
        if driver is not None:
            floors = sorted(self.table.driver_floors.items(), key=lambda item: item[1], reverse=True)
            for kind, floor in floors:
                if driver >= floor:
                    return kind
        return "blas"

    async def _nvidia_driver(self) -> Optional[tuple[int, int]]:
        """The NVIDIA driver's version, (major, minor) — None without an NVIDIA card and driver."""
        said = await self.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout_s=LOOK_TIMEOUT_S)
        if said.code != 0:
            return None
        for line in said.output.splitlines():
            matched = _DRIVER.match(line)
            if matched:
                return int(matched.group(1)), int(matched.group(2))
        return None

    async def _build_tag_for(
        self, http: httpx.AsyncClient, release: Release, names: tuple[str, ...]
    ) -> tuple[str, Asset]:
        """The build tag released with `release` — the one whose tag names the same commit — and
        its archive among `names`. Candidates are read in the order most likely to be it; none is
        taken for its date or its place in the listing alone."""
        commit = await self.commit_of(http, release.tag)
        listed = await self._listing(http, release.tag)
        version_at = next((i for i, entry in enumerate(listed) if entry.get("tag_name") == release.tag), 0)
        candidates: list[tuple[bool, int, str, Asset]] = []
        for index, entry in enumerate(listed):
            tag = entry.get("tag_name")
            if not isinstance(tag, str) or not _BUILD_TAG.fullmatch(tag) or entry.get("draft"):
                continue
            assets = {asset.name: asset for asset in self.assets_of(entry)}
            asset = next((assets[name] for name in names if name in assets), None)
            if asset is not None:
                # Said by the release, not by the repository: it orders the reading, nothing more.
                says_so = entry.get("target_commitish") == commit
                candidates.append((not says_so, abs(index - version_at), tag, asset))
        candidates.sort(key=lambda candidate: candidate[:2])
        for _, _, tag, asset in candidates[:MAX_CANDIDATES]:
            if await self.commit_of(http, tag) == commit:
                return tag, asset
        raise InstallRefused(
            f"{self.repo} {release.tag} carries no {names[0]}, and no build tag released with it names its "
            f"commit {commit[:12]}; it is not installed on this machine"
        )

    async def _listing(self, http: httpx.AsyncClient, tag: str) -> list[dict[str, Any]]:
        """The releases, newest first, as far as the version `tag` and the page after it if it is
        listed near a page's end — a build tag is released beside its version."""
        listed: list[dict[str, Any]] = []
        for page in range(1, LIST_PAGES + 1):
            body = await self._get(
                http, f"/repos/{self.repo}/releases?per_page={LIST_PAGE}&page={page}", f"{self.repo} has no releases",
            )
            entries = [entry for entry in body if isinstance(entry, dict)] if isinstance(body, list) else []
            listed.extend(entries)
            at = next((i for i, entry in enumerate(entries) if entry.get("tag_name") == tag), None)
            if len(entries) < LIST_PAGE or (at is not None and at < LIST_PAGE - MAX_CANDIDATES):
                break
        return listed
