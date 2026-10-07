"""A model volume: models kept between hosts, copied from and verified, never trusted (D139).

A host the pool created may be given a volume — storage at the provider that outlives the host
and that later hosts of the same workload can mount. It is mounted at one of two paths, which are
**this agent's own constants**, never named by the pool (D40, D41):

- at `READ_PATH`, a host copies its models from the volume before going to the hub;
- at `FILL_PATH`, a host fetches as usual and then copies what it fetched up into the volume.

**The engine never runs from the volume.** Every host keeps its models on its own disk: a network
volume reads more slowly than a copy to local disk followed by a load, and no provider offers a
read-only mount — so whatever is on the volume may have been written by any host that ever had it,
and is treated as untrusted:

- **Reading copies by the hub's own file list**, never by listing the volume, so nothing but the
  files the hub names comes across; and **each file is checked against the hash the hub publishes**
  (sha256 for a large-file-storage file, the git blob sha1 for the rest). A file missing, of the
  wrong size, mismatched, or with no published hash is left for the hub. A host that wrote junk into
  a volume therefore harms no other host.
- **Filling writes a whole build into a fresh directory** and renames it into place, so a reader
  never sees half a build; every file is checked against the hub's hash as it is copied, so a fill
  never spreads a file this host should not have. No path in the volume is followed through a link:
  every directory is opened relative to the one above it with links refused.

A **build** is a repository's files as the hub lists them, named by a digest of that list with
each file's hash: two hosts that list the same files agree on its name without asking anyone, and
a repository that changes upstream is a different build in a directory of its own. A build is
never changed in place.
"""

from __future__ import annotations

import asyncio
import dataclasses
import errno
import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import httpx

#: Where the volume is mounted on a host that reads from it, and on the one host that fills it.
#: The pool's side mounts them here (it imports these, so the two cannot drift); the pool never
#: sends either to the agent.
READ_PATH = Path("/opt/gpm/model-volume")
FILL_PATH = Path("/opt/gpm/model-volume-fill")

#: Written last inside a filled build: what it holds, for an operator looking at the volume. A
#: reader does not need it — the hashes decide — and never trusts it.
BUILD_MARKER = ".gpm-build.json"

#: Files copied at once. A network volume answers one stream slowly and many quickly.
PARALLEL = 4

_CHUNK = 8 << 20
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class VolumeRefused(Exception):
    """A path in the volume this module will not use: a link, or a name that is not a plain one."""


@dataclasses.dataclass(frozen=True)
class Expected:
    """What the hub says one file is: its path in the repository, size and published hash."""

    path: str
    size: int
    #: The content's sha256, published for large-file-storage files.
    sha256: Optional[str] = None
    #: The git blob sha1 — of `blob <size>\0` and the content — published for every other file.
    git_sha1: Optional[str] = None

    @property
    def checkable(self) -> bool:
        return bool(self.sha256 or self.git_sha1)


@dataclasses.dataclass
class Report:
    """What one fetch took from where, for the agent's facts and the pool's history."""

    files_from_volume: int = 0
    bytes_from_volume: int = 0
    #: Files the volume had but that did not match the hub's hash or size.
    files_mismatched: int = 0
    #: Files the volume did not have, or that the hub publishes no hash for.
    files_missing: int = 0
    seconds_from_volume: float = 0.0
    #: "filled", "already there", or why not.
    fill: Optional[str] = None

    def as_fact(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def build_id(files: Iterable[Expected]) -> Optional[str]:
    """The build's name: a digest of every file's path, size and hash. None if any file has no
    published hash — such a build cannot be checked, so it is neither filled nor read."""
    listed = sorted(files, key=lambda f: f.path)
    if not listed or not all(f.checkable for f in listed):
        return None
    digest = hashlib.sha256()
    for f in listed:
        digest.update(f"{f.path}\0{f.size}\0{f.sha256 or ''}\0{f.git_sha1 or ''}\n".encode())
    return digest.hexdigest()[:32]


def plain_parts(path: str) -> list[str]:
    """A repository path as its components, refusing anything but plain names."""
    parts = path.split("/")
    for part in parts:
        if part in ("", ".", "..") or "\\" in part or "\0" in part:
            raise VolumeRefused(f"{path!r} is not a plain relative path")
    return parts


def _hasher(expected: Expected) -> Callable[[], "hashlib._Hash"]:
    if expected.sha256:
        return hashlib.sha256
    return lambda: hashlib.sha1(f"blob {expected.size}\0".encode())


def _digest_matches(expected: Expected, digest: str) -> bool:
    return digest == (expected.sha256 or expected.git_sha1)


def _open_dir(root: Path, parts: Sequence[str], *, create: bool = False) -> int:
    """A directory under `root`, opened one component at a time with links refused, so nothing a
    volume holds can lead outside it. The caller closes the descriptor."""
    fd = os.open(root, _DIR_FLAGS)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):  # a link, or a file, where a directory was
                    raise VolumeRefused(f"{part!r} in the volume is a link") from exc
                raise
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _copy_checked(source_fd: int, sink: Path, expected: Expected) -> bool:
    """Copy into `sink`'s draft beside it, hashing as it goes; keep it only if it is the file the
    hub names. Returns whether it was kept."""
    draft = sink.with_name(sink.name + ".gpm-volume-part")
    hasher = _hasher(expected)()
    moved = 0
    try:
        with os.fdopen(source_fd, "rb", closefd=True) as source, open(draft, "wb") as out:
            while chunk := source.read(_CHUNK):
                moved += len(chunk)
                if moved > expected.size:
                    break
                hasher.update(chunk)
                out.write(chunk)
        if moved != expected.size or not _digest_matches(expected, hasher.hexdigest()):
            draft.unlink(missing_ok=True)
            return False
        os.replace(draft, sink)
        return True
    except BaseException:
        draft.unlink(missing_ok=True)
        raise


def _read_one(volume: Path, repo_dir: str, build: str, expected: Expected, into: Path) -> str:
    """One file from the volume to local disk: "copied", "mismatched" or "missing"."""
    if not expected.checkable:
        return "missing"
    parts = plain_parts(expected.path)
    try:
        directory = _open_dir(volume, [repo_dir, build, *parts[:-1]])
    except (FileNotFoundError, NotADirectoryError, VolumeRefused):
        return "missing"
    try:
        try:
            source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except OSError:
            return "missing"
    finally:
        os.close(directory)
    target = into.joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return "copied" if _copy_checked(source, target, expected) else "mismatched"


async def read(volume: Path, repo_dir: str, files: Sequence[Expected], into: Path,
               report: Report) -> list[Expected]:
    """Copy what the volume holds of this build to `into`, checked file by file. Returns the
    files that landed; anything else is the hub's to fetch. Files already whole on local disk are
    left as they are."""
    build = build_id(files)
    if build is None:
        report.files_missing += len(files)
        return []
    return await _copy_in(volume, repo_dir, build, files, into, report)


async def _copy_in(volume: Path, first: str, second: str, files: Sequence[Expected], into: Path,
                   report: Report) -> list[Expected]:
    started = time.monotonic()
    gate = asyncio.Semaphore(PARALLEL)
    landed: list[Expected] = []

    async def one(expected: Expected) -> None:
        target = into.joinpath(*plain_parts(expected.path))
        if target.is_file() and target.stat().st_size == expected.size:
            return
        async with gate:
            outcome = await asyncio.to_thread(_read_one, volume, first, second, expected, into)
        if outcome == "copied":
            report.files_from_volume += 1
            report.bytes_from_volume += expected.size
            landed.append(expected)
        elif outcome == "mismatched":
            report.files_mismatched += 1
        else:
            report.files_missing += 1

    await asyncio.gather(*(one(f) for f in files))
    report.seconds_from_volume += time.monotonic() - started
    return landed


def _write_into(directory: int, parts: Sequence[str], local: Path, expected: Expected) -> bool:
    """One local file into the fresh build directory, checked against the hub as it is copied."""
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, _DIR_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
    except BaseException:
        os.close(directory)
        raise
    try:
        out = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=directory)
    finally:
        os.close(directory)
    hasher = _hasher(expected)()
    moved = 0
    with open(local, "rb") as source, os.fdopen(out, "wb") as sink:
        while chunk := source.read(_CHUNK):
            moved += len(chunk)
            hasher.update(chunk)
            sink.write(chunk)
    return moved == expected.size and _digest_matches(expected, hasher.hexdigest())


def _remove_tree(parent: int, name: str) -> None:
    """Remove a directory this fill made, without following anything in it."""
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
    except OSError:
        return
    try:
        for entry in os.listdir(fd):
            try:
                os.unlink(entry, dir_fd=fd)
            except IsADirectoryError:
                _remove_tree(fd, entry)
            except PermissionError:  # macOS answers a directory unlink with EPERM
                _remove_tree(fd, entry)
            except OSError:
                pass
    finally:
        os.close(fd)
    try:
        os.rmdir(name, dir_fd=parent)
    except OSError:
        pass


def _fill(volume: Path, repo_dir: str, files: Sequence[Expected], into: Path) -> str:
    build = build_id(files)
    if build is None:
        return "not filled: the hub publishes no hash for every file"
    parent = _open_dir(volume, [repo_dir], create=True)
    try:
        try:
            os.close(os.open(build, _DIR_FLAGS, dir_fd=parent))
            return "already there"
        except FileNotFoundError:
            pass
        draft = f".fill-{secrets.token_hex(6)}"
        os.mkdir(draft, 0o755, dir_fd=parent)
        try:
            for expected in files:
                parts = plain_parts(expected.path)
                local = into.joinpath(*parts)
                directory = os.open(draft, _DIR_FLAGS, dir_fd=parent)
                if not _write_into(directory, parts, local, expected):
                    raise VolumeRefused(f"{expected.path} on this host does not match the hub")
            marker = os.open(f"{draft}/{BUILD_MARKER}", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o644, dir_fd=parent)
            with os.fdopen(marker, "w") as out:
                json.dump({"build": build, "files": {f.path: f.size for f in files}}, out)
            try:
                os.rename(draft, build, src_dir_fd=parent, dst_dir_fd=parent)
            except OSError:
                # Another host filled the same build meanwhile: theirs stands.
                _remove_tree(parent, draft)
                return "already there"
            return "filled"
        except BaseException:
            _remove_tree(parent, draft)
            raise
    finally:
        os.close(parent)


async def fill(volume: Path, repo_dir: str, files: Sequence[Expected], into: Path, report: Report) -> None:
    """Copy a build this host fetched up into the volume. Never fails the fetch: the host has its
    model either way, and the reason is in its facts."""
    try:
        report.fill = await asyncio.to_thread(_fill, volume, repo_dir, files, into)
    except (OSError, VolumeRefused) as exc:
        report.fill = f"not filled: {exc}"


# --- Ollama: content-addressed blobs (D139) ---
#
# Ollama keeps a model as blobs named by their own sha256 (`blobs/sha256-<hex>`) beside a manifest
# that lists them. So the volume holds blobs only, under `ollama/blobs/`, and the check needs no
# hub: a blob is kept only when it hashes to the digest it is named by. Which blobs a tag needs
# comes from the registry's manifest, never from the volume — and the engine's own pull then
# fetches the manifest and whatever is still missing. Unverified until checked on a live host:
# that Ollama keeps this layout and that its pull skips a blob already on disk.

#: The registry Ollama pulls from, and where Ollama keeps its blobs on this machine — the
#: engine's own default unless the machine's owner moved it.
OLLAMA_REGISTRY = "https://registry.ollama.ai"
_OLLAMA_NAME = re.compile(r"^(?:([a-z0-9][a-z0-9._-]*)/)?([a-z0-9][a-z0-9._-]*)(?::([A-Za-z0-9._-]+))?$")
_DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")


def ollama_blobs_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS") or "~/.ollama/models").expanduser() / "blobs"


async def ollama_blobs(tag: str, client: httpx.AsyncClient) -> Optional[list[Expected]]:
    """The blobs a tag is made of, from the registry's own manifest. None for a tag from any
    other registry, or when the registry does not say — the engine's pull then does it all."""
    match = _OLLAMA_NAME.match(tag or "")
    if not match:
        return None
    namespace, name, version = match.group(1) or "library", match.group(2), match.group(3) or "latest"
    try:
        response = await client.get(
            f"{OLLAMA_REGISTRY}/v2/{namespace}/{name}/manifests/{version}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
        )
        if response.status_code != 200:
            return None
        manifest = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    blobs = []
    for layer in [manifest.get("config"), *(manifest.get("layers") or [])]:
        digest = _DIGEST.match(str((layer or {}).get("digest") or ""))
        if not digest or not isinstance(layer.get("size"), int):
            return None
        blobs.append(Expected(path=f"sha256-{digest.group(1)}", size=layer["size"], sha256=digest.group(1)))
    return blobs or None


async def read_blobs(volume: Path, blobs: Sequence[Expected], into: Path, report: Report) -> None:
    """Copy the blobs the volume holds into the engine's own store, each checked against its name."""
    into.mkdir(parents=True, exist_ok=True)
    await _copy_in(volume, "ollama", "blobs", blobs, into, report)


def _fill_blobs(volume: Path, blobs: Sequence[Expected], local: Path) -> str:
    directory = _open_dir(volume, ["ollama", "blobs"], create=True)
    added = 0
    try:
        for blob in blobs:
            try:
                os.stat(blob.path, dir_fd=directory, follow_symlinks=False)
                continue  # there already; a reader checks it whoever wrote it
            except FileNotFoundError:
                pass
            draft = f".fill-{secrets.token_hex(6)}-{blob.path}"
            if not _write_into(os.dup(directory), [draft], local / blob.path, blob):
                os.unlink(draft, dir_fd=directory)
                raise VolumeRefused(f"{blob.path} on this host does not match its name")
            os.rename(draft, blob.path, src_dir_fd=directory, dst_dir_fd=directory)
            added += 1
    finally:
        os.close(directory)
    return f"filled ({added} new)" if added else "already there"


async def fill_blobs(volume: Path, blobs: Sequence[Expected], report: Report) -> None:
    """Copy a tag's blobs from the engine's store up into the volume. Never fails the pull."""
    try:
        report.fill = await asyncio.to_thread(_fill_blobs, volume, blobs, ollama_blobs_dir())
    except (OSError, VolumeRefused) as exc:
        report.fill = f"not filled: {exc}"
