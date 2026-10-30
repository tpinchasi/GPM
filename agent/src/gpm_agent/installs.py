"""Installing the engine this agent runs, from its official releases (D150–D153, T35).

On a machine whose owner set `install_engines: true`, the agent installs the engine it manages,
keeps the versions it has installed side by side, and switches between them when the pool asks.
What keeps that inside the agent's bounds:

- **Where from is the agent's own table, never the pool's word.** The pool names a version —
  `latest` or digits and dots — and nothing else; `RELEASES` says which repository publishes the
  engine and which of a release's files is the one for this operating system, processor and
  accelerator. No URL, path, file name or command is ever taken from a request.
- **Every file is checked before it is used.** Against the digest its release publishes for it
  (GitHub's per-asset `sha256`), and also against the release's own list of digests where it
  publishes one; a file with no published digest is refused. Where the platform signs programs
  and the table pins the publisher, every program unpacked must carry that signature (Ollama on
  macOS: a Developer ID certificate of team `3MU9H2V9Y9`, anchored at Apple's root, checked as one
  code requirement); elsewhere the facts say the signature was not checked.
  A source archive, for which no release publishes a digest, is fetched by the commit its tag
  names — read from the repository — and must say it is that commit; the tools a build needs and
  the machine lacks come pinned to their own published digests (`whisper_build`).
- **Archives unpack only into their own directory.** No absolute path, no `..`, no link that
  leads out, no device file, no control character in a name; anything else refuses the whole
  version. Nothing in an archive is ever run but the engine's own program, and no install script
  runs at all.
- **Forward, and back only one step.** A version older than the one running is refused unless it
  is the one before it (a rollback), and a `latest` older than the one running is ignored.
- **Never past the owner's free-disk floor.** Before anything is fetched, what it will take on the
  engines' volume — an archive and what it unpacks to, or a build — must leave `min_free_disk_gb`.
- **Versions side by side** (D153), each in its own directory under the engine's directory: the
  one running, the one before it, and one staged — downloaded, checked and unpacked while the
  engine keeps serving, in a `<version>.partial` directory renamed into place only when whole.
  Anything older is removed. A switch stops the running server, starts the new one, and waits for
  it to answer; one that does not is stopped and the one before started again, by itself.

The work runs in the background, one job at a time; the verb answers at once with where things
stand (`Installer.state`), and the pool reads how it went on its next passes.

A **source** is what one engine's releases are made into a version directory by (`Source`).
Ollama's are archives (`ArchiveSource`, with the table's `OLLAMA` row); whisper.cpp's are built
from the release's source on macOS and Linux and taken from its own archives on Windows
(`whisper_build.WhisperCppSource`, with the table's `WHISPER_CPP` row); vLLM's are private
Python environments filled from PyPI with the locked set the agent carries for each version
(`vllm_install.VllmSource`, with the table's `VLLM` row) — through the same four operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import stat
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Any, Callable, Mapping, Optional, Protocol

import httpx

from . import platform as host_platform
from .models import Refused
from .platform import LINUX, MACOS, WINDOWS

log = logging.getLogger("gpm_agent.installs")

#: A version in an engine's own form: two to four groups of digits, and an optional pre-release
#: suffix (`0.40.11`, `0.12.0-rc1`, `1.9.5`). It becomes a directory name, so nothing else is —
#: always matched whole (`fullmatch`), so not even a trailing newline.
VERSION = re.compile(r"^\d+(\.\d+){1,3}(-[0-9A-Za-z.]+)?$")
LATEST = "latest"

_SUFFIX_PART = re.compile(r"\d+|[^\d.]+")


def version_key(version: str) -> tuple[Any, ...]:
    """How versions are ordered: by their numbers, part by part (`0.9.0` before `0.10.0`, `1.2`
    the same as `1.2.0`), and a pre-release before its release (`0.12.0-rc2` before `0.12.0`),
    pre-releases among themselves by their numbers and words in turn (`rc2` before `rc10`)."""
    numbers, _, suffix = version.partition("-")
    parts = [int(part) for part in numbers.split(".")]
    parts += [0] * (4 - len(parts))
    if not suffix:
        return (tuple(parts), 1, ())
    words = tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in _SUFFIX_PART.findall(suffix))
    return (tuple(parts), 0, words)


def older(version: str, than: str) -> bool:
    return version_key(version) < version_key(than)


#: How long the latest release, once read, is taken as still the latest. The pool says `latest`
#: on every pass; GitHub allows a machine without a token sixty questions an hour.
LATEST_EVERY_S = 6 * 3600.0

#: How long a switched-to server is given to answer before the switch is called a failure. Ollama
#: answers in about a second, loading nothing; whisper.cpp's server answers once it has loaded the
#: file it serves, which takes seconds — two minutes means something is wrong with either.
SWITCH_TIMEOUT_S = 120.0

#: How long a version that could not be fetched or staged is left alone before another attempt —
#: long enough not to spend the hour's questions on one failure, short enough to recover.
RETRY_AFTER_S = 300.0

#: GitHub's API, and where a test or a mirror points instead — the machine owner's environment,
#: never the pool's word. A token there (`GITHUB_TOKEN`) raises the hourly limit; it is read from
#: the environment the owner started the agent in, sent only to that API, and never logged.
DEFAULT_API = "https://api.github.com"
API_ENV = "GPM_GITHUB_API"
TOKEN_ENV = "GITHUB_TOKEN"

STATE_FILE = "state.json"
#: Where downloads wait, by their digest, so a cut transfer resumes on the next attempt.
DOWNLOADS = "downloads"
PARTIAL = ".partial"
#: Beside the engines' own directories: tools the agent fetched to build an engine with (`cmake`
#: from PyPI, where the machine has none), kept for the next build.
TOOLS = ".tools"

#: How much room an archive needs on the engines' volume, as a multiple of its size: the archive
#: itself, kept until its version is whole (1×), and what it unpacks to. Programs and libraries
#: compress to between a half and a third of their size in the archives engines publish, so
#: they unpack to up to three times it (3×).
UNPACK_MARGIN = 4

_CHUNK = 1 << 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(?:\./)?(\S.*?)\s*$")
#: Control characters — a newline among them — in a name an archive gives: never written. A
#: name reaches commands' output and the agent's own messages, where one could pass for a line.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class InstallRefused(Exception):
    """A version will not be installed here, and why — in words for an operator."""


def size_words(size: int) -> str:
    """A size in words: gigabytes, or megabytes or kilobytes under one."""
    if size >= 10**9:
        return f"{size / 1e9:.1f} GB"
    if size >= 10**6:
        return f"{size / 1e6:.0f} MB"
    return f"{size / 1e3:.0f} kB"


#: How a source asks, before it fetches anything, for room on the engines' volume: the bytes it
#: will write at most, and what they are, in words. Refuses (`InstallRefused`) where there is not
#: room for them above the floor the machine's owner keeps free.
Room = Callable[[int, str], None]


# --- what a release is ---


@dataclasses.dataclass(frozen=True)
class Asset:
    """One file of a release, as GitHub lists it: its number, its size, and the sha256 GitHub
    publishes for it — None where it publishes none, which refuses the file."""

    name: str
    id: int
    size: int
    sha256: Optional[str]


@dataclasses.dataclass(frozen=True)
class Release:
    version: str
    tag: str
    assets: tuple[Asset, ...]

    def asset(self, name: str) -> Optional[Asset]:
        return next((asset for asset in self.assets if asset.name == name), None)


#: Progress as a source reports it: the step, how far it is, and out of how much if known.
Progress = Callable[[str, int, Optional[int]], None]


class Source(Protocol):
    """How one engine's releases become a version directory. The installer resolves, keeps
    versions side by side and switches; a source knows only its engine's releases.

    `stage` fills `into` — a fresh, empty directory — with the version, or raises
    `InstallRefused` saying why it would not. It may keep what it downloads beside `into`, in
    `into.parent / DOWNLOADS`, so a cut transfer resumes on the next attempt. It reports its steps
    through `progress`: `downloading`, `verifying`, `unpacking` or `building`. Before it fetches
    anything it asks `room`, where given, for what it will write.
    """

    engine: str
    #: "checked" where every program this source installs here is checked against a pinned
    #: signature, "not checked" where only digests are.
    signature: str

    async def latest(self, http: httpx.AsyncClient) -> Release: ...

    async def release(self, http: httpx.AsyncClient, version: str) -> Release: ...

    async def stage(
        self, http: httpx.AsyncClient, release: Release, into: Path, progress: Progress, room: Optional[Room] = None,
    ) -> None: ...

    def program(self, version_dir: Path) -> Path: ...


# --- the machine a build is chosen for ---


@dataclasses.dataclass(frozen=True)
class Target:
    """What a build is chosen by: the operating system, the processor, and an accelerator that
    needs its own files (`rocm` for an AMD card), if any."""

    os: str
    arch: Optional[str]
    accelerator: Optional[str] = None


def _amd_card(os_name: str) -> bool:
    """Whether this machine has an AMD card the engine would use. Linux: the ROCm kernel driver's
    device. Windows: AMD's HIP runtime, which its graphics driver installs."""
    if os_name == LINUX:
        return Path("/dev/kfd").exists()
    if os_name == WINDOWS:
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        with contextlib.suppress(OSError):
            return any(system32.glob("amdhip64*.dll"))
    return False


def this_machine() -> Target:
    os_name = host_platform.system()
    return Target(os_name, host_platform.architecture(), "rocm" if _amd_card(os_name) else None)


# --- GitHub's releases ---


def _api() -> str:
    return (os.environ.get(API_ENV) or DEFAULT_API).rstrip("/")


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "gpm-agent"}
    token = os.environ.get(TOKEN_ENV)
    if token:
        # Only ever to the API itself: httpx drops it when a download is redirected elsewhere.
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checksums(text: str) -> dict[str, str]:
    """A release's list of digests, `sha256sum`'s own form: `<hex>  <name>`, with or without a
    leading `./` or `*`."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        matched = _CHECKSUM_LINE.match(line.strip())
        if matched:
            found[matched.group(2)] = matched.group(1).lower()
    return found


class GitHubReleases:
    """An engine's releases on GitHub: reading them, and fetching one file of one, checked."""

    def __init__(self, repo: str) -> None:
        self.repo = repo

    async def _get(self, http: httpx.AsyncClient, path: str, missing: str) -> Any:
        try:
            response = await http.get(f"{_api()}{path}", headers=_headers())
        except httpx.HTTPError as exc:
            raise InstallRefused(f"could not reach GitHub for {self.repo}: {exc or type(exc).__name__}") from exc
        if response.status_code == 404:
            raise InstallRefused(missing)
        if response.status_code in (403, 429):
            raise InstallRefused(
                f"GitHub refused to answer about {self.repo} ({response.status_code}): the hourly "
                "limit for this machine may be spent, and it resets within the hour"
                + ("" if os.environ.get(TOKEN_ENV) else "; a GITHUB_TOKEN in the agent's environment raises it")
            )
        if response.status_code != 200:
            raise InstallRefused(f"GitHub answered {response.status_code} about {self.repo}")
        try:
            return response.json()
        except ValueError as exc:
            raise InstallRefused(f"GitHub's answer about {self.repo} was not JSON") from exc

    def _release(self, body: Any) -> Release:
        tag = body.get("tag_name") if isinstance(body, dict) else None
        if not isinstance(tag, str) or not tag.startswith("v") or not VERSION.fullmatch(tag[1:]):
            raise InstallRefused(f"{self.repo}'s release {tag!r} is not named v<version>; it is not installed")
        if body.get("draft"):
            raise InstallRefused(f"{self.repo} {tag} is a draft, not a release")
        return Release(version=tag[1:], tag=tag, assets=self.assets_of(body))

    @staticmethod
    def assets_of(body: Any) -> tuple[Asset, ...]:
        """A release's files as GitHub lists them, each with the digest it publishes, if any."""
        assets = []
        for entry in (body.get("assets") if isinstance(body, dict) else None) or []:
            if not isinstance(entry, dict):
                continue
            name, number, size, digest = entry.get("name"), entry.get("id"), entry.get("size"), entry.get("digest")
            if not isinstance(name, str) or not isinstance(number, int) or not isinstance(size, int):
                continue
            sha256 = None
            if isinstance(digest, str) and digest.startswith("sha256:") and _HEX64.fullmatch(digest[7:].lower()):
                sha256 = digest[7:].lower()
            assets.append(Asset(name, number, size, sha256))
        return tuple(assets)

    async def latest(self, http: httpx.AsyncClient) -> Release:
        """The newest release that is not a pre-release, as the repository marks it."""
        return self._release(await self._get(
            http, f"/repos/{self.repo}/releases/latest", f"{self.repo} has no release on GitHub",
        ))

    async def release(self, http: httpx.AsyncClient, version: str) -> Release:
        if not VERSION.fullmatch(version):
            raise InstallRefused(f"{version!r} is not a version")
        found = self._release(await self._get(
            http, f"/repos/{self.repo}/releases/tags/v{version}", f"{self.repo} has no release v{version}",
        ))
        if found.version != version:
            raise InstallRefused(f"GitHub answered for v{version} with {found.tag}")
        return found

    async def download(
        self, http: httpx.AsyncClient, asset: Asset, downloads: Path, moved: Callable[[int], None]
    ) -> Path:
        """One file of a release, checked against its published digest, at `downloads/<sha256>`.

        Resumable: a cut transfer is kept as `<sha256>.part` and continued with a range request.
        Fetched through the API by the file's number, so every address is the API's own. A file
        whose digest does not match is removed, not kept for the next attempt.
        """
        if asset.sha256 is None:
            raise InstallRefused(
                f"{self.repo} publishes no digest for {asset.name}; nothing unchecked is installed"
            )
        downloads.mkdir(mode=0o700, parents=True, exist_ok=True)
        final = downloads / asset.sha256
        if final.is_file():
            if await asyncio.to_thread(_sha256, final) == asset.sha256:
                moved(asset.size)
                return final
            final.unlink()
        part = downloads / f"{asset.sha256}.part"
        already = part.stat().st_size if part.is_file() else 0
        if already > asset.size:
            part.unlink()
            already = 0
        if already < asset.size or asset.size == 0:
            headers = _headers("application/octet-stream")
            if already:
                headers["Range"] = f"bytes={already}-"
            url = f"{_api()}/repos/{self.repo}/releases/assets/{asset.id}"
            try:
                async with http.stream("GET", url, headers=headers) as response:
                    resumed = response.status_code == 206 and response.headers.get(
                        "content-range", f"bytes {already}-"
                    ).startswith(f"bytes {already}-")
                    if response.status_code not in (200, 206) or (response.status_code == 206 and not resumed):
                        raise InstallRefused(f"fetching {asset.name} returned {response.status_code}")
                    if not resumed:
                        already = 0  # the whole file, from its start
                    have = already
                    with part.open("ab" if already else "wb") as sink:
                        async for chunk in response.aiter_bytes(_CHUNK):
                            have += len(chunk)
                            if have > asset.size:
                                raise InstallRefused(f"{asset.name} is longer than its release says")
                            sink.write(chunk)
                            moved(have)
            except httpx.HTTPError as exc:
                raise InstallRefused(
                    f"fetching {asset.name} stopped: {exc or type(exc).__name__}; the next attempt resumes it"
                ) from exc
        landed = part.stat().st_size
        if landed != asset.size:
            raise InstallRefused(
                f"{asset.name} stopped at {landed} of {asset.size} bytes; the next attempt resumes it"
            )
        found = await asyncio.to_thread(_sha256, part)
        if found != asset.sha256:
            part.unlink()
            raise InstallRefused(
                f"{asset.name} does not match the digest its release publishes "
                f"(sha256 {found[:12]}…, published {asset.sha256[:12]}…); it was removed"
            )
        os.replace(part, final)
        return final


# --- unpacking, safely ---


def _member(name: str) -> Optional[PurePosixPath]:
    """An archive member's path inside the version directory, or a refusal. None for the
    archive's own top (`./`)."""
    if not name or _CONTROL.search(name) or "\\" in name:
        raise InstallRefused(f"the archive names a file {name!r} this agent will not write")
    path = PurePosixPath(name)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", name):
        raise InstallRefused(f"the archive names an absolute path, {name!r}")
    parts = [part for part in path.parts if part not in ("", ".")]
    if ".." in parts:
        raise InstallRefused(f"the archive names a path that climbs out of its directory, {name!r}")
    if any(":" in part for part in parts):
        raise InstallRefused(f"the archive names a file {name!r} this agent will not write")
    return PurePosixPath(*parts) if parts else None


class _Unpacking:
    """Writes one version directory from archive members, refusing anything that leads out of it."""

    def __init__(self, into: Path) -> None:
        self.into = into
        self.root = os.path.realpath(into)
        self.links: list[Path] = []

    def _inside(self, path: str) -> bool:
        return path == self.root or path.startswith(self.root + os.sep)

    def _place(self, member: PurePosixPath) -> Path:
        """Where a member goes: under the directory, through nothing that leads out of it."""
        target = self.into.joinpath(*member.parts)
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if not self._inside(os.path.realpath(target.parent)):
            raise InstallRefused(f"the archive writes {member} through a link that leads out of its directory")
        return target

    def directory(self, member: PurePosixPath) -> None:
        target = self._place(member)
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise InstallRefused(f"the archive names {member} both as a directory and as something else")
        target.mkdir(mode=0o755, exist_ok=True)

    def file(self, member: PurePosixPath, source: IO[bytes], executable: bool) -> None:
        target = self._place(member)
        if target.is_dir() and not target.is_symlink():
            raise InstallRefused(f"the archive names {member} both as a directory and as a file")
        if os.path.lexists(target):
            target.unlink()  # a later archive's copy replaces an earlier one's; never written through
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(target, flags, 0o755 if executable else 0o644)
        with os.fdopen(descriptor, "wb") as sink:
            shutil.copyfileobj(source, sink, _CHUNK)

    def link(self, member: PurePosixPath, pointing_to: str) -> None:
        if not pointing_to or _CONTROL.search(pointing_to) or posixpath.isabs(pointing_to) or "\\" in pointing_to:
            raise InstallRefused(f"the archive's link {member} points to {pointing_to!r}, outside its directory")
        lexical = posixpath.normpath(posixpath.join(str(member.parent), pointing_to))
        if lexical == ".." or lexical.startswith("../"):
            raise InstallRefused(f"the archive's link {member} points to {pointing_to!r}, outside its directory")
        target = self._place(member)
        if not self._inside(os.path.realpath(os.path.join(target.parent, pointing_to))):
            raise InstallRefused(f"the archive's link {member} leads out of its directory")
        if os.path.lexists(target):
            if target.is_dir() and not target.is_symlink():
                raise InstallRefused(f"the archive names {member} both as a directory and as a link")
            target.unlink()
        try:
            os.symlink(pointing_to, target)
        except OSError as exc:
            raise InstallRefused(f"the archive's link {member} could not be made here: {exc}") from exc
        self.links.append(target)

    def hard_link(self, member: PurePosixPath, to: PurePosixPath) -> None:
        """A tar hard link names an earlier member: copied, so it is only ever a file inside."""
        source = self.into.joinpath(*to.parts)
        if source.is_symlink() or not source.is_file() or not self._inside(os.path.realpath(source)):
            raise InstallRefused(f"the archive's link {member} names {to}, which is not a file it holds")
        with source.open("rb") as handle:
            self.file(member, handle, bool(source.stat().st_mode & 0o111))

    def finish(self) -> None:
        """Every link, now that everything is written, still leads somewhere inside: a link made
        early can be pointed out of the directory by one made later."""
        for link in self.links:
            if not self._inside(os.path.realpath(link)):
                raise InstallRefused(f"the archive's link {link.relative_to(self.into)} leads out of its directory")


def _zstd_stream(path: Path) -> IO[bytes]:
    """A `.zst` file, decompressed as it is read: Python's own module from 3.14, else the
    `zstandard` package where it is installed. The agent depends on neither — it is also shipped
    as a pure-Python archive to machines the pool rents, which never install an engine."""
    try:
        from compression import zstd  # type: ignore[import-not-found]

        return zstd.open(path, "rb")
    except ImportError:
        pass
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:
        raise InstallRefused(
            f"{path.name} is a zstd archive, which this Python cannot read: run the agent on "
            "Python 3.14 or later, or install the `zstandard` package beside it"
        ) from None
    return zstandard.ZstdDecompressor().stream_reader(path.open("rb"), closefd=True)


def _untar(handle: tarfile.TarFile, into: _Unpacking) -> None:
    for entry in handle:
        member = _member(entry.name)
        if member is None:
            continue
        if entry.isdir():
            into.directory(member)
        elif entry.isfile():
            source = handle.extractfile(entry)
            if source is None:
                raise InstallRefused(f"the archive's {member} could not be read")
            with source:
                into.file(member, source, bool(entry.mode & 0o111))
        elif entry.issym():
            into.link(member, entry.linkname)
        elif entry.islnk():
            to = _member(entry.linkname)
            if to is None:
                raise InstallRefused(f"the archive's link {member} names its own top")
            into.hard_link(member, to)
        else:
            raise InstallRefused(f"the archive holds {member}, a device or special file")


def unpack(archive: Path, name: str, into: Path) -> None:
    """Unpack `archive` — whose release calls it `name`, which says its format — into `into`,
    over what is there. `.tgz`/`.tar.gz`, `.tar.zst` and `.zip`; anything else is refused."""
    writer = _Unpacking(into)
    try:
        if name.endswith((".tgz", ".tar.gz")):
            with tarfile.open(archive, mode="r|gz") as handle:
                _untar(handle, writer)
        elif name.endswith(".tar.zst"):
            with _zstd_stream(archive) as stream, tarfile.open(fileobj=stream, mode="r|") as handle:
                _untar(handle, writer)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(archive) as handle:
                for entry in handle.infolist():
                    member = _member(entry.filename)
                    if member is None:
                        continue
                    kind = stat.S_IFMT(entry.external_attr >> 16)
                    if entry.is_dir():
                        writer.directory(member)
                    elif kind in (0, stat.S_IFREG):
                        with handle.open(entry) as source:
                            writer.file(member, source, bool((entry.external_attr >> 16) & 0o111))
                    elif kind == stat.S_IFLNK:
                        raise InstallRefused(f"the archive holds a link, {member}; a zip's links are not followed here")
                    else:
                        raise InstallRefused(f"the archive holds {member}, a device or special file")
        else:
            raise InstallRefused(f"{name} is not an archive this agent unpacks")
    except InstallRefused:
        raise
    except Exception as exc:  # noqa: BLE001 - a damaged archive, in any of three formats' words
        raise InstallRefused(f"{name} could not be unpacked: {exc or type(exc).__name__}") from exc
    writer.finish()


# --- signatures ---

#: The first four bytes of every kind of Mach-O file, in either byte order: thin, 32 and 64 bits
#: (`MH_MAGIC`, `MH_MAGIC_64`), and universal, with 32- and 64-bit offsets (`FAT_MAGIC`,
#: `FAT_MAGIC_64`). A Java class file begins as a 32-bit universal binary does; one in an engine's
#: archive is taken for one and must be signed too — the safe side, where telling them apart by
#: the count that follows would let a universal binary of many slices go unchecked.
_MACH_O = frozenset({
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",  # MH_MAGIC
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",  # MH_MAGIC_64
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",  # FAT_MAGIC
    b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",  # FAT_MAGIC_64
})


def _is_mach_o(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(4)
    except OSError:
        return False
    return head in _MACH_O


def signed_files(version_dir: Path, os_name: str) -> list[Path]:
    """The programs and libraries in a version that must carry the pinned signature."""
    found = []
    for path in sorted(version_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if os_name == MACOS and _is_mach_o(path):
            found.append(path)
        elif os_name == WINDOWS and path.suffix.lower() in (".exe", ".dll"):
            found.append(path)
    return found


# --- the table: where each engine comes from (D152) ---


@dataclasses.dataclass(frozen=True)
class ArchiveRelease:
    """An engine its project publishes as archives on its GitHub releases, one per platform."""

    engine: str
    repo: str
    #: (operating system, processor) → the archive holding the engine.
    archives: Mapping[tuple[str, str], str]
    #: (operating system, processor, accelerator) → an archive unpacked over the first, with what
    #: that accelerator needs.
    extras: Mapping[tuple[str, str, str], str]
    #: The release's own list of digests, cross-checked against each file's where published.
    checksums: Optional[str]
    #: Operating system → the publisher every program there must be signed by. A system not
    #: named is checked by digest alone, and its facts say the signature was not checked.
    signers: Mapping[str, str]
    #: Operating system → where the program is in the unpacked archive, most likely first.
    programs: Mapping[str, tuple[str, ...]]


#: Ollama's releases (verified against v0.40.11, 2026-10-26). macOS: the headless archive, not the
#: app — its programs signed by Ollama's Developer ID team. Windows' publisher is not pinned until
#: it has been read from a release's programs (D152's unverified row): digest only until then.
OLLAMA = ArchiveRelease(
    engine="ollama",
    repo="ollama/ollama",
    archives={
        (MACOS, "arm64"): "ollama-darwin.tgz",
        (MACOS, "amd64"): "ollama-darwin.tgz",
        (LINUX, "amd64"): "ollama-linux-amd64.tar.zst",
        (LINUX, "arm64"): "ollama-linux-arm64.tar.zst",
        (WINDOWS, "amd64"): "ollama-windows-amd64.zip",
        (WINDOWS, "arm64"): "ollama-windows-arm64.zip",
    },
    extras={
        (LINUX, "amd64", "rocm"): "ollama-linux-amd64-rocm.tar.zst",
        (WINDOWS, "amd64", "rocm"): "ollama-windows-amd64-rocm.zip",
    },
    checksums="sha256sum.txt",
    signers={MACOS: "3MU9H2V9Y9"},
    programs={
        MACOS: ("ollama", "bin/ollama"),
        LINUX: ("bin/ollama", "ollama"),
        WINDOWS: ("ollama.exe", "bin/ollama.exe"),
    },
)

@dataclasses.dataclass(frozen=True)
class BuiltRelease:
    """An engine whose version tags carry its source and no program for some systems: built on
    the machine there (macOS, Linux), and taken on Windows from the archives its project
    publishes — on the version's own release, or on a build tag of the same commit."""

    engine: str
    repo: str
    #: The one program a version needs, as the build names its target.
    program: str
    #: Windows: what the machine has → the archive for it, by every name the project has given
    #: it, newest first.
    windows: Mapping[str, tuple[str, ...]]
    #: The NVIDIA driver each CUDA archive needs at least, (major, minor), as `nvidia-smi` says it.
    driver_floors: Mapping[str, tuple[int, int]]


#: whisper.cpp (verified against v1.9.5 and its build tag b5454, and v1.9.4 and b5130,
#: 2026-10-26). Its version tags carry no files since 1.9.3; Windows archives come with a build
#: tag of the same commit, and were renamed between b5130 (`whisper-cublas-12.4.0-bin-x64.zip`)
#: and b5454 (`whisper-bin-win-cuda-12.4.0-x64.zip`). None is signed. CUDA 12.4 runs on NVIDIA's
#: 550 drivers and later; 11.8, by CUDA 11's compatibility across minor versions, on 452.39 and
#: later. Without such a card: the CPU build with OpenBLAS on x64, the plain CPU build on arm64.
WHISPER_CPP = BuiltRelease(
    engine="whisper-cpp",
    repo="ggml-org/whisper.cpp",
    program="whisper-server",
    windows={
        "cuda-12.4": ("whisper-bin-win-cuda-12.4.0-x64.zip", "whisper-cublas-12.4.0-bin-x64.zip"),
        "cuda-11.8": ("whisper-bin-win-cuda-11.8.0-x64.zip", "whisper-cublas-11.8.0-bin-x64.zip"),
        "blas": ("whisper-blas-bin-x64.zip",),
        "cpu-arm64": ("whisper-bin-win-cpu-arm64.zip",),
    },
    driver_floors={"cuda-12.4": (550, 0), "cuda-11.8": (452, 39)},
)

@dataclasses.dataclass(frozen=True)
class LockedRelease:
    """An engine installed from PyPI's wheels, only as one of the locked sets the agent carries
    says (D158): its versions are the sets', never PyPI's listing."""

    engine: str
    #: The project on PyPI.
    project: str


#: vLLM (D157, D158): a private Python environment per version, from its locked set
#: (`vllm_install.VllmSource`).
VLLM = LockedRelease(engine="vllm", project="vllm")

RELEASES: dict[str, ArchiveRelease | BuiltRelease | LockedRelease] = {
    "ollama": OLLAMA, "whisper-cpp": WHISPER_CPP, "vllm": VLLM,
}


class ArchiveSource(GitHubReleases):
    """An engine from its release archives: the right ones for this machine, fetched, checked
    against their digests, unpacked safely, and their programs' signatures checked where pinned."""

    def __init__(self, table: ArchiveRelease, target: Target, *, run: Optional[host_platform.Runner] = None) -> None:
        super().__init__(table.repo)
        self.engine = table.engine
        self.table, self.target, self.run = table, target, run
        self.signer = table.signers.get(target.os)
        self.signature = "checked" if self.signer else "not checked"

    def archive_names(self) -> list[str]:
        key = (self.target.os, self.target.arch or "")
        main = self.table.archives.get(key)
        if main is None:
            raise InstallRefused(
                f"{self.table.repo} publishes no {self.engine} for {self.target.os} on "
                f"{self.target.arch or 'this processor'}"
            )
        extra = self.table.extras.get((*key, self.target.accelerator or ""))
        return [main, extra] if extra else [main]

    def program(self, version_dir: Path) -> Path:
        candidates = self.table.programs.get(self.target.os) or (self.engine,)
        for candidate in candidates:
            path = version_dir.joinpath(*candidate.split("/"))
            if path.is_file():
                return path
        return version_dir.joinpath(*candidates[0].split("/"))

    async def stage(
        self, http: httpx.AsyncClient, release: Release, into: Path, progress: Progress, room: Optional[Room] = None,
    ) -> None:
        assets = []
        for name in self.archive_names():
            asset = release.asset(name)
            if asset is None:
                raise InstallRefused(f"{self.table.repo} {release.tag} has no {name}")
            if asset.sha256 is None:
                raise InstallRefused(
                    f"{self.table.repo} {release.tag} publishes no digest for {name}; nothing unchecked is installed"
                )
            assets.append(asset)
        if room is not None:
            size = sum(asset.size for asset in assets)
            room(size * UNPACK_MARGIN, (
                f"{' and '.join(asset.name for asset in assets)}, {size_words(size)}, "
                f"and up to {UNPACK_MARGIN - 1} times that unpacked"
            ))
        downloads = into.parent / DOWNLOADS
        listed = release.asset(self.table.checksums) if self.table.checksums else None
        _keep_only(downloads, {a.sha256 for a in [*assets, listed] if a is not None and a.sha256})

        if listed is not None:
            # The release's own list must agree with what GitHub says of each file: two records
            # of the same digest, written at different times by different means.
            sums = parse_checksums((await self.download(http, listed, downloads, lambda _: None)).read_text(errors="replace"))
            for asset in assets:
                if sums.get(asset.name) != asset.sha256:
                    raise InstallRefused(
                        f"{release.tag}'s {self.table.checksums} does not agree with GitHub's digest for {asset.name}"
                        if asset.name in sums else
                        f"{release.tag}'s {self.table.checksums} does not list {asset.name}"
                    )

        total, before, archives = sum(a.size for a in assets), 0, []
        for asset in assets:
            offset = before
            progress("downloading", offset, total)
            archives.append(await self.download(
                http, asset, downloads, lambda have, offset=offset: progress("downloading", offset + have, total),
            ))
            before += asset.size

        for index, (asset, archive) in enumerate(zip(assets, archives, strict=True)):
            progress("unpacking", index, len(assets))
            await asyncio.to_thread(unpack, archive, asset.name, into)
        program = self.program(into)
        if not program.is_file():
            raise InstallRefused(f"{release.tag}'s {assets[0].name} holds no {self.engine} program where expected")
        if self.target.os != WINDOWS:
            program.chmod(program.stat().st_mode | 0o755)

        if self.signer:
            files = signed_files(into, self.target.os)
            if program not in files:
                raise InstallRefused(f"{release.tag}'s {program.name} is not a program this platform signs")
            for index, path in enumerate(files):
                progress("verifying", index, len(files))
                problem = await asyncio.to_thread(
                    host_platform.signature_problem, path, self.signer, os_name=self.target.os, run=self.run,
                )
                if problem:
                    raise InstallRefused(f"{release.tag} is not installed: {problem}")


def _keep_only(downloads: Path, digests: set[str]) -> None:
    """Forget downloads that are not of these files — an abandoned version's, or a finished one's."""
    if not downloads.is_dir():
        return
    for path in downloads.iterdir():
        if path.name.removesuffix(".part") not in digests:
            with contextlib.suppress(OSError):
                path.unlink()


def source_for(
    engine: str, target: Target, *, run: Optional[host_platform.Runner] = None, tools_dir: Optional[Path] = None,
) -> Optional[Source]:
    """The source for `engine` on `target`, from the table; None for an engine it does not name.
    `tools_dir` is where a source that builds keeps the tools it fetched (`TOOLS`)."""
    table = RELEASES.get(engine)
    if isinstance(table, ArchiveRelease):
        return ArchiveSource(table, target, run=run)
    if isinstance(table, BuiltRelease):
        # Imported here: the build's module uses this one's releases, downloads and unpacking.
        from .whisper_build import WhisperCppSource

        return WhisperCppSource(table, target, tools_dir=tools_dir)
    if isinstance(table, LockedRelease):
        from .vllm_install import VllmSource

        return VllmSource(target)
    return None


# --- the record of what is installed ---


@dataclasses.dataclass
class InstallState:
    """`<engines>/<engine>/state.json`: which version runs, which is staged, which came before,
    the latest seen and when, what the pool last asked for, and the last error. Versions that
    failed to start here are remembered, and not tried again by `latest` (D153)."""

    running: Optional[str] = None
    staged: Optional[str] = None
    previous: Optional[str] = None
    latest: Optional[str] = None
    latest_checked_at: Optional[float] = None
    wanted: Optional[str] = None
    error: Optional[str] = None
    failed: dict[str, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> InstallState:
        """The record, or a blank one. Every version in it is checked again as a version: it names
        a directory, and a hand-edited file must not name another."""
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        if not isinstance(body, dict):
            return cls()

        def version(name: str) -> Optional[str]:
            value = body.get(name)
            return value if isinstance(value, str) and VERSION.fullmatch(value) else None

        checked_at = body.get("latest_checked_at")
        failed = body.get("failed") if isinstance(body.get("failed"), dict) else {}
        wanted = body.get("wanted")
        return cls(
            running=version("running"), staged=version("staged"), previous=version("previous"),
            latest=version("latest"),
            latest_checked_at=float(checked_at) if isinstance(checked_at, (int, float)) else None,
            wanted=wanted if isinstance(wanted, str) and (wanted == LATEST or VERSION.fullmatch(wanted)) else None,
            error=body.get("error") if isinstance(body.get("error"), str) else None,
            failed={k: str(v) for k, v in failed.items() if isinstance(k, str) and VERSION.fullmatch(k)},
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        draft = path.with_name(path.name + ".new")
        descriptor = os.open(draft, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(dataclasses.asdict(self), handle, indent=2)
        os.replace(draft, path)


def not_allowed(engine: str) -> dict[str, Any]:
    """`engine.install` where the machine's owner has not allowed installs."""
    return {
        "allowed": False, "engine": engine, "running": None, "staged": None, "previous": None,
        "latest": None, "latest_checked_at": None, "wanted": None, "working": None, "error": None,
        "signature": None, "switch_bound_s": None,
    }


def validated(body: Any) -> tuple[str, bool]:
    """The version wanted and whether to switch now — the whole of what the pool can say (D151)."""
    if not isinstance(body, dict) or set(body) != {"version", "switch"}:
        raise Refused(400, "bad_request", "the body is {\"version\": ..., \"switch\": ...} and nothing else")
    version, switch = body["version"], body["switch"]
    if not isinstance(version, str) or len(version) > 64 or not (version == LATEST or VERSION.fullmatch(version)):
        raise Refused(400, "bad_version", "version is `latest` or a version: digits and dots, and an optional suffix")
    if not isinstance(switch, bool):
        raise Refused(400, "bad_request", "switch is true or false")
    return version, switch


# --- the installer ---


class Server(Protocol):
    """The engine's server as the installer drives it (`ollama_server.OllamaServer`,
    `whisper_cpp.InstalledServer`). `stop` may raise on an error from the system — a process
    this user may not signal, a record that cannot be written — which the installer says as the
    reason its switch or restart did not happen, never lets end its work unsaid."""

    foreign_reason: str

    def ours(self, *, fresh: bool = False) -> bool: ...

    def running_version(self) -> Optional[str]: ...

    async def foreign(self) -> bool: ...

    async def start(self, program: Path, version: str, timeout_s: float) -> Optional[str]: ...

    async def stop(self) -> None: ...


def free_bytes_at(path: Path) -> Optional[int]:
    """Free space on the volume `path` is on — measured at its nearest existing parent, so a
    directory not made yet reports where it would land — or None where it cannot be read."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


class Installer:
    """One engine's versions on this machine, worked toward in the background, one job at a time.

    `want` records what the pool asked for and starts the work it needs — resolving `latest`,
    staging a version, or a switch — and answers at once with `state`. The pool reads how the
    work went on its next passes: a staged version, a version running, or an error.

    The pool moves an engine forward, and back only to the version it ran before (T35): a version
    older than the one running is refused unless it is that one, and a `latest` older than the one
    running — a release withdrawn — is ignored, saying so. Nothing is fetched that would leave the
    engines' volume with less free than the floor the machine's owner keeps (`min_free_disk_gb`).
    """

    def __init__(
        self,
        source: Source,
        engine_dir: Path,
        server: Server,
        *,
        http_transport: Optional[httpx.AsyncBaseTransport] = None,
        clock: Callable[[], float] = time.time,
        switch_timeout_s: float = SWITCH_TIMEOUT_S,
        switch_bound_s: Optional[float] = None,
        min_free_disk_gb: float = 0.0,
        free_disk_bytes: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        self.source, self.engine_dir, self.server = source, engine_dir, server
        self.http_transport = http_transport
        self.clock = clock
        self.switch_timeout_s = switch_timeout_s
        #: The longest one switch or start of this engine takes until it has an outcome, said in
        #: `engine.install` for the pool to wait as long (`app.switch_bound_s`, from the server's
        #: own times). Where whoever made this installer does not say: the start's wait twice —
        #: the version, and the one before it after a failure — and a minute for the stops.
        self.switch_bound_s = float(switch_bound_s) if switch_bound_s is not None else 2 * switch_timeout_s + 60.0
        #: The machine owner's floor, and how free space on the engines' volume is read.
        self.floor_bytes = int(min_free_disk_gb * 1e9)
        self.free_disk_bytes = free_disk_bytes or (lambda: free_bytes_at(self.engine_dir))
        self.record = InstallState.read(engine_dir / STATE_FILE)
        self.working: Optional[dict[str, Any]] = None
        self._job: Optional[asyncio.Task[None]] = None
        #: One of the pool's requests at a time: each is answered before the next is looked at, so
        #: two that arrive together never both find nothing in hand and both start work.
        self._asking = asyncio.Lock()
        #: One start or stop of the server at a time — a switch, a restart, or bringing it back.
        self._lock = asyncio.Lock()
        #: When each version asked for may be tried again after a failure, by what was asked.
        self._retry_at: dict[str, float] = {}
        #: When a server that would not start may be started again.
        self._start_retry_at = 0.0
        #: What the installer last said in the record's error about what it was asked — a version
        #: it will not go back to, a latest older than what runs — removed once that no longer holds.
        self._told: Optional[str] = None

    # --- what the pool sees ---

    def state(self) -> dict[str, Any]:
        """`engine.install` (host-agent.md §2.3): from memory, so the facts stay cheap."""
        record = self.record
        return {
            "allowed": True,
            "engine": self.source.engine,
            "running": record.running,
            "staged": record.staged,
            "previous": record.previous,
            "latest": record.latest,
            "latest_checked_at": record.latest_checked_at,
            "wanted": record.wanted,
            "working": dict(self.working) if self.working else None,
            "error": record.error,
            "signature": self.source.signature,
            "switch_bound_s": self.switch_bound_s,
        }

    @property
    def busy(self) -> bool:
        return self._job is not None and not self._job.done()

    @property
    def starting(self) -> bool:
        """Whether the server is being stopped or started by this installer right now — a switch,
        a restart, or bringing it back — so that nothing else starts it meanwhile."""
        return self._lock.locked()

    def program(self, version: str) -> Path:
        return self.source.program(self.engine_dir / version)

    # --- what the pool asks ---

    async def want(self, version: str, switch: bool) -> dict[str, Any]:
        """Work toward `version` — `latest` or a version — and, with `switch`, start the staged or
        previous version now. Answers at once; the work goes on in the background."""
        async with self._asking:
            return await self._want(version, switch)

    async def _want(self, version: str, switch: bool) -> dict[str, Any]:
        record = self.record
        if version != LATEST and self._older_than_running(version):
            said = (
                f"{self.source.engine} {version} is older than the {record.running} running here and is not "
                f"the version it ran before ({record.previous or 'none is kept'}): the pool moves an engine "
                "forward, and back only to the version before it; an older one is the machine owner's to install"
            )
            self._tell(said)
            raise Refused(409, "older_than_running", said)
        if switch and self.busy:
            raise Refused(409, "busy", f"{self.source.engine} is already being {self._doing()}; ask again when it is done")
        if record.wanted != version:
            record.wanted = version
            self._save()
        if self.busy:
            return self.state()  # the next pass asks again, and is taken up then
        if await self.server.foreign():
            raise Refused(409, "foreign_engine", self.server.foreign_reason)
        notice = self._latest_is_older() if version == LATEST and self._latest_fresh() else None
        self._tell(notice)
        if switch:
            target = self._switch_target(version)
            if target is None:
                return self.state()  # already running it
            self.working = {"step": "switching", "version": target, "done": 0, "total": None}
            self._job = asyncio.create_task(self._switch_job(target), name="gpm-agent:switch")
            return self.state()
        if notice or not self._needs_work(version) or self.clock() < self._retry_at.get(version, 0.0):
            return self.state()
        self.working = {"step": "resolving", "version": version, "done": 0, "total": None}
        self._job = asyncio.create_task(self._stage_job(version), name="gpm-agent:stage")
        return self.state()

    def _doing(self) -> str:
        step = (self.working or {}).get("step")
        return "switched" if step == "switching" else "installed"

    def _older_than_running(self, version: str) -> bool:
        """Is `version` older than the one running here, and not the one before it — a downgrade
        past what a rollback reaches, which the pool cannot ask for (T35)?"""
        record = self.record
        return record.running is not None and version != record.previous and older(version, record.running)

    def _latest_fresh(self) -> bool:
        checked = self.record.latest_checked_at
        return checked is not None and self.clock() - checked < LATEST_EVERY_S

    def _latest_is_older(self) -> Optional[str]:
        """What to say when the latest release, as last read, is older than the version running —
        a release withdrawn, or a version asked for by name that is newer: it is ignored."""
        record = self.record
        if record.latest and record.running and older(record.latest, record.running):
            return (
                f"the latest release of {self.source.engine}, {record.latest}, is older than the "
                f"{record.running} running here; it is ignored, and {record.running} keeps running"
            )
        return None

    def _tell(self, notice: Optional[str]) -> None:
        """Put `notice` in the record's error, or take out the one put there before, if it is
        still what the error says — never an error the work itself recorded."""
        record = self.record
        if notice is None:
            if self._told is not None and record.error == self._told:
                record.error = None
                self._save()
            self._told = None
            return
        self._told = notice
        if record.error != notice:
            record.error = notice
            self._save()

    def _switch_target(self, version: str) -> Optional[str]:
        """The version a switch would start, None where it already runs — or a refusal."""
        record = self.record
        if version == LATEST:
            if record.staged:
                return record.staged
            if record.running and record.latest and not older(record.running, record.latest):
                return None  # the latest runs, or one newer than it
            raise Refused(409, "not_staged", "no newer version is staged here yet; it is staged first, then switched to")
        if version == record.running:
            return None
        if version in (record.staged, record.previous):
            return version
        raise Refused(
            409, "not_staged",
            f"{self.source.engine} {version} is not staged here: a version is staged first, then switched to "
            f"(staged: {record.staged or 'none'}, previous: {record.previous or 'none'})",
        )

    def _needs_work(self, version: str) -> bool:
        """Whether asking for `version` means anything to do now, known without asking GitHub."""
        record = self.record
        on_disk = {record.running, record.staged, record.previous} - {None}
        if version == LATEST:
            if not self._latest_fresh():
                return True
            version = record.latest or ""
        if version in record.failed:
            self._not_again(version, latest=record.wanted == LATEST)
            return False
        return version not in on_disk or record.running is None

    def _not_again(self, version: str, *, latest: bool) -> None:
        """Say why a version that failed to start here is not being installed (D153)."""
        said = (
            f"{self.source.engine} {version} did not start here before ({self.record.failed[version]}); "
            + ("it is not tried again until a newer version is released" if latest
               else "it is not tried again: ask for another version")
        )
        if self.record.error != said:
            self.record.error = said
            self._save()

    async def restartable(self) -> None:
        """Refuse, saying why, a restart that cannot happen now."""
        if self.busy:
            raise Refused(409, "busy", f"{self.source.engine} is being {self._doing()}; ask again when it is done")
        if self.record.running is None:
            raise Refused(409, "not_installed", f"no version of {self.source.engine} is installed here yet")
        if await self.server.foreign():
            raise Refused(409, "foreign_engine", self.server.foreign_reason)

    async def restart(self) -> Optional[str]:
        """Start the running version afresh — new settings, an operator's restart. None when it
        answers again; otherwise why not."""
        await self.restartable()
        async with self._lock:
            # Asked again, and the version read, only now that nothing else can start or stop the
            # server: a switch that took it first has changed what runs, and is not undone here.
            if self.busy:
                raise Refused(409, "busy", f"{self.source.engine} is being {self._doing()}; ask again when it is done")
            running = self.record.running
            if running is None:
                raise Refused(409, "not_installed", f"no version of {self.source.engine} is installed here yet")
            stopped = await self._stop()
            if stopped is None:
                problem = await self.server.start(self.program(running), running, self.switch_timeout_s)
            else:
                problem = f"it could not be stopped: {stopped}"
            if problem:
                self._start_retry_at = self.clock() + RETRY_AFTER_S / 5
                self.record.error = f"{self.source.engine} {running} did not start again: {problem}"
            else:
                self.record.error = None
            self._save()
        return problem

    async def ensure_running(self) -> None:
        """Start the running version if this agent's server is not up: when the agent starts, and
        on the pool's passes after it has stopped. Never over something it did not start.

        Whether it is up is looked at afresh, never taken from a look a moment ago: a server that
        has stopped since, its number gone to another program, would otherwise pass for running."""
        if self.record.running is None or self.busy or self._lock.locked():
            return
        if self.server.ours(fresh=True):
            if self.server.running_version() in (None, self.record.running):
                return
            # Another version of its own runs: a switch this agent was stopped in the middle of,
            # before the version it was trying had been checked. The version on record goes back.
            async with self._lock:
                stopped = await self._stop()
                if stopped is not None:
                    self.record.error = f"{self.source.engine} could not be stopped to start {self.record.running} again: {stopped}"
                    self._save()
                    return
        if self.clock() < self._start_retry_at or await self.server.foreign():
            return
        async with self._lock:
            running = self.record.running
            if running is None or self.busy or self.server.ours(fresh=True):
                return
            problem = await self.server.start(self.program(running), running, self.switch_timeout_s)
            if problem:
                # A server that will not start is left a minute before the next attempt.
                self._start_retry_at = self.clock() + RETRY_AFTER_S / 5
                self.record.error = f"{self.source.engine} {running} stopped and did not start again: {problem}"
                self._save()

    async def finished(self) -> None:
        """Wait for the work in hand, if any."""
        if self._job is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._job)

    async def aclose(self) -> None:
        if self.busy and self._job is not None:
            self._job.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._job

    # --- the work ---

    def _save(self) -> None:
        self.record.write(self.engine_dir / STATE_FILE)

    def _progress(self, version: str) -> Progress:
        def report(step: str, done: int, total: Optional[int]) -> None:
            self.working = {"step": step, "version": version, "done": int(done), "total": total}

        return report

    def _room(self, version: str, needed: int, what: str) -> None:
        """Refuse a version whose files would leave the engines' volume with less free than the
        machine owner's floor — before anything of it is fetched."""
        free = self.free_disk_bytes()
        if free is not None and free - needed < self.floor_bytes:
            raise InstallRefused(
                f"{self.source.engine} {version} is not installed: it needs about {size_words(needed)} on the "
                f"engines' volume ({what}), which has {size_words(free)} free, and that would leave less than the "
                f"{size_words(self.floor_bytes)} this machine's owner keeps free (min_free_disk_gb)"
            )

    async def _stop(self) -> Optional[str]:
        """Stop the server: None once it is, else the error from the system that kept it from
        stopping — said as the reason a switch or a restart did not happen, never raised."""
        try:
            await self.server.stop()
        except Exception as exc:  # noqa: BLE001 - a process not this user's, a record not written: said
            return str(exc) or type(exc).__name__
        return None

    async def _stage_job(self, wanted: str) -> None:
        try:
            async with httpx.AsyncClient(
                transport=self.http_transport, timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True,
            ) as http:
                release = await self._resolve(http, wanted)
                if release is not None:
                    await self._stage(http, release)
            self._retry_at.pop(wanted, None)
            target = self.record.staged
            if self.record.running is None and target:
                # Nothing runs yet, so there is nothing to drain: the first version starts now.
                self.working = {"step": "switching", "version": target, "done": 0, "total": None}
                await self._switch(target)
        except Exception as no:  # noqa: BLE001 - said, and tried again later; never the agent's end
            # Beside the refusals: the engines directory itself — a full disk, a permission — or
            # something unforeseen, said in its own words rather than lost with the task.
            if not isinstance(no, (InstallRefused, OSError)):
                log.exception("installing %s %s failed", self.source.engine, wanted)
            self._retry_at[wanted] = self.clock() + RETRY_AFTER_S
            self.record.error = str(no) or type(no).__name__
            with contextlib.suppress(OSError):
                self._save()
        finally:
            self.working = None

    async def _resolve(self, http: httpx.AsyncClient, wanted: str) -> Optional[Release]:
        """The release to stage for `wanted`, or None where there is nothing to stage."""
        record = self.record
        if wanted == LATEST:
            release = None
            if not self._latest_fresh():
                release = await self.source.latest(http)
                record.latest, record.latest_checked_at = release.version, self.clock()
                # Versions that failed here are not tried again once a newer one is out (D153).
                record.failed = {v: why for v, why in record.failed.items() if v == release.version}
                self._save()
            version = record.latest
            notice = self._latest_is_older()
            self._tell(notice)
            if notice:
                return None
        else:
            release, version = None, wanted
            if self._older_than_running(version):
                raise InstallRefused(f"{self.source.engine} {version} is older than the {record.running} running here")
        if version is None:
            return None
        if version in record.failed:
            self._not_again(version, latest=wanted == LATEST)
            return None
        if version in (record.running, record.staged, record.previous):
            return None
        return release if release is not None else await self.source.release(http, version)

    async def _stage(self, http: httpx.AsyncClient, release: Release) -> None:
        version = release.version
        report = self._progress(version)
        # The owner's alone: it holds programs the agent runs as its owner.
        self.engine_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        partial = self.engine_dir / f"{version}{PARTIAL}"
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(partial)
        partial.mkdir(mode=0o755)

        def room(needed: int, what: str) -> None:
            self._room(version, needed, what)

        try:
            await self.source.stage(http, release, partial, report, room)
            final = self.engine_dir / version
            if final.exists():
                shutil.rmtree(final)
            os.replace(partial, final)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        _keep_only(self.engine_dir / DOWNLOADS, set())
        self.record.staged = version
        self.record.error = None
        self._prune()
        self._save()

    async def _switch_job(self, target: str) -> None:
        try:
            await self._switch(target)
        finally:
            self.working = None

    async def _switch(self, target: str) -> None:
        """Stop what runs, start `target`, and keep it only if it answers as that version — or
        stop it and start again the version that ran before, at once, by itself (D153). A server
        that will not stop is a switch that did not happen, said so, with what ran started again
        if it stopped after all."""
        engine = self.source.engine
        async with self._lock:
            record = self.record
            before = record.running
            stopped = await self._stop()
            if stopped is not None:
                # Not the new version's fault: it stays staged, to be switched to again.
                error = f"{engine} {target} was not started: {before or 'the server'} could not be stopped: {stopped}"
                if before and not self.server.ours(fresh=True):
                    back = await self.server.start(self.program(before), before, self.switch_timeout_s)
                    error += (
                        f"; {before} runs again" if back is None
                        else f"; and {before}, started again, did not answer: {back}"
                    )
                record.error = error
                self._save()
                return
            if await self.server.foreign():
                # Something else took the port as this agent's server stopped: not the new
                # version's fault, and nothing of this agent's is started over it.
                record.error = f"{engine} {target} was not started: {self.server.foreign_reason}"
                self._save()
                return
            problem = await self.server.start(self.program(target), target, self.switch_timeout_s)
            if problem is None:
                if record.staged == target:
                    record.staged = None
                if record.previous == target:
                    record.previous = None
                if before and before != target:
                    record.previous = before
                record.running, record.error = target, None
                self._start_retry_at = 0.0
            else:
                stopped = await self._stop()
                record.failed[target] = problem
                if record.staged == target:
                    record.staged = None
                if record.previous == target:
                    record.previous = None
                error = f"{engine} {target} did not start: {problem}"
                if stopped is not None:
                    error += f"; and it could not be stopped: {stopped}"
                if before and before != target:
                    back = await self.server.start(self.program(before), before, self.switch_timeout_s)
                    error += (
                        f"; {before} runs again" if back is None
                        else f"; and {before}, started again, did not answer either: {back}"
                    )
                elif not before:
                    record.running = None
                record.error = error
            self._prune()
            self._save()

    def _prune(self) -> None:
        """Keep the running version, the previous one and the staged one; remove the rest, and
        any half-staged directory not being staged now."""
        record = self.record
        keep = {record.running, record.previous, record.staged} - {None}
        staging = (self.working or {}).get("version")
        if not self.engine_dir.is_dir():
            return
        for path in self.engine_dir.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            name = path.name
            if VERSION.fullmatch(name) and name not in keep:
                shutil.rmtree(path, ignore_errors=True)
            elif name.endswith(PARTIAL) and VERSION.fullmatch(name[: -len(PARTIAL)]) and name[: -len(PARTIAL)] != staging:
                shutil.rmtree(path, ignore_errors=True)
