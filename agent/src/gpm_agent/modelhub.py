"""Fetching model weights from a model hub, over plain HTTP.

Engines differ in where their weights come from. Ollama pulls through its own API, so the agent
asks the engine and watches. vLLM has no pull at all: it is started with a directory and serves
what is in it, so somebody must put the weights there first — and on a host the pool created,
that somebody is this agent.

**An HTTP client, not a command.** No `huggingface-cli`, no `git lfs`, no subprocess: the agent
ships as a zipapp with httpx and must work in whatever image the engine came in. Downloading
over the hub's own HTTP API also means real byte counts for progress, and resuming a cut
transfer with a range request instead of starting a 19 GB file again.

Nothing here takes a URL, a path or a command from the pool. The pool names a **repository**;
this module decides what that means and where it lands.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

#: The hub this fetches from. Settable for a mirror or an air-gapped copy, by the machine's
#: owner in the machine's own environment — never by the pool.
DEFAULT_ENDPOINT = "https://huggingface.co"

#: A repository id: `owner/name`, in the character set the hub allows. Checked because it
#: becomes a path on this machine, and a name that climbs out of the models directory would be
#: a pool naming a path — exactly what the agent's protocol refuses to allow (D40).
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Files that are never weights and are not worth the transfer.
SKIP_SUFFIXES = (".md", ".gitattributes", ".png", ".jpg", ".jpeg", ".gif", ".pdf")

#: Weight formats to prefer when a repository ships more than one. `safetensors` is what the
#: engine loads; a `.bin` beside it is the same weights in an older format, and fetching both
#: would double a download for nothing.
PREFERRED = ".safetensors"

_CHUNK = 1 << 20

#: Written into a model's directory once **every** file has landed, and removed when a fetch
#: starts. Anything reading the models directory — what the agent reports as on disk, what the
#: engine is started with — goes by this, because a directory with some of its files is a
#: download in progress, and an engine started on half a model fails in ways that look like the
#: model's fault (D97).
COMPLETE_MARKER = ".gpm-complete"


def is_complete(directory: Path) -> bool:
    """Is this model all here? Its complete marker, and files adding up to the size the marker
    records. The marker alone is not enough once models can arrive by other means than this
    fetch — a provider's copy from another machine (D116) may bring the marker before the weights,
    and serving half a model is worse than serving none. A marker that cannot be read — empty or
    half-written by such a copy — vouches for nothing: every marker this module writes has a size."""
    marker = directory / COMPLETE_MARKER
    if not marker.is_file():
        return False
    try:
        expected = int(marker.read_text().strip())
    except (OSError, ValueError):
        return False
    have = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file() and not f.name.startswith(".gpm-"))
    return have >= expected


class HubRefused(Exception):
    """The hub said no, or the request was not one this module will make."""


@dataclasses.dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


def directory_for(models_dir: str | Path, repo: str) -> Path:
    """Where a repository lands on this machine. One directory per repository, named after it
    with the owner's slash flattened, so nothing a pool can say escapes the models directory."""
    if not REPO_ID.match(repo or ""):
        raise HubRefused(
            f"{repo!r} is not a model repository id: expected `owner/name`, in letters, digits, "
            f"dots, dashes and underscores"
        )
    return Path(models_dir).expanduser() / repo.replace("/", "__")


def _headers() -> dict[str, str]:
    """A token if this machine's owner put one in its environment, and nothing otherwise.

    Gated repositories need one. It is read here, from the process the owner started, and is
    never accepted from the pool, written to disk, or reported back in any answer.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def wanted(files: list[RemoteFile]) -> list[RemoteFile]:
    """The files worth fetching: configuration and tokenizer, plus one weight format.

    A repository commonly ships the same weights twice — `.safetensors` for engines that load
    them and `.bin` for older tooling. Taking both would double the largest download the pool
    ever makes, and the engine would read only one.
    """
    keep = [f for f in files if not f.path.lower().endswith(SKIP_SUFFIXES)]
    if any(f.path.endswith(PREFERRED) for f in keep):
        keep = [f for f in keep if not f.path.endswith((".bin", ".pt", ".pth", ".h5", ".msgpack"))]
    return keep


async def listing(repo: str, *, client: Optional[httpx.AsyncClient] = None) -> list[RemoteFile]:
    """Every file in the repository, with its size — so progress has a denominator before the
    first byte moves, rather than a bar that fills to an unknown end."""
    directory_for(models_dir=".", repo=repo)  # validates the id and nothing else
    own = client is None
    client = client or httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    try:
        response = await client.get(
            f"{_endpoint()}/api/models/{repo}/tree/main",
            params={"recursive": "true"},
            headers=_headers(),
        )
        if response.status_code == 401 or response.status_code == 403:
            raise HubRefused(
                f"the hub refused access to {repo} ({response.status_code}); a gated repository "
                f"needs HF_TOKEN in this machine's environment"
            )
        if response.status_code == 404:
            raise HubRefused(f"the hub has no repository {repo}")
        if response.status_code != 200:
            raise HubRefused(f"listing {repo} returned {response.status_code}")
        entries = response.json()
    except httpx.HTTPError as exc:
        raise HubRefused(f"could not list {repo}: {exc or type(exc).__name__}") from exc
    finally:
        if own:
            await client.aclose()
    if not isinstance(entries, list):
        raise HubRefused(f"the hub answered for {repo} in a shape this agent does not understand")
    return [
        RemoteFile(path=e["path"], size=int(e.get("size") or 0))
        for e in entries
        if isinstance(e, dict) and e.get("type") == "file" and isinstance(e.get("path"), str)
    ]


async def fetch(
    repo: str, models_dir: str | Path, *, client: Optional[httpx.AsyncClient] = None
) -> AsyncIterator[tuple[int, int]]:
    """Fetch a repository into `models_dir`, yielding (bytes done, bytes total) as it moves.

    Resumable by construction: a file already the right size is skipped, and a partial one is
    continued with a range request. A 19 GB download cut at 18 GB costs a minute on the next
    attempt, not an hour — which matters because the pool retries a cut download (D57) and a
    rented host is paying for every second of it.
    """
    into = directory_for(models_dir, repo)
    files = wanted(await listing(repo, client=client))
    if not files:
        raise HubRefused(f"{repo} holds no files this agent would fetch")
    total = sum(f.size for f in files)
    into.mkdir(parents=True, exist_ok=True)
    # Not complete until this fetch says so, whatever an earlier one left behind.
    (into / COMPLETE_MARKER).unlink(missing_ok=True)

    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True)
    done = 0
    try:
        for entry in files:
            target = into / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            already = target.stat().st_size if target.exists() else 0
            if entry.size and already == entry.size:
                done += already
                yield done, total
                continue
            if already > entry.size:
                # Longer than the hub says it should be: not a resumable transfer, a wrong file.
                target.unlink()
                already = 0
            async for moved in _one_file(client, repo, entry, target, already):
                yield done + moved, total
            landed = target.stat().st_size
            if entry.size and landed != entry.size:
                # A connection that closes early ends the stream without an error, and the loop
                # above ends with it. Taking that as done would mark half a file complete and
                # start an engine on it. Refused instead: what arrived is kept, and the next
                # attempt resumes from there (D97).
                raise HubRefused(
                    f"{entry.path} of {repo} stopped at {landed} of {entry.size} bytes; "
                    f"the next attempt will resume it"
                )
            done += entry.size or landed
            yield done, total
        (into / COMPLETE_MARKER).write_text(f"{total}\n")
    except httpx.HTTPError as exc:
        raise HubRefused(f"fetching {repo} stopped: {exc or type(exc).__name__}") from exc
    finally:
        if own:
            await client.aclose()


async def _one_file(
    client: httpx.AsyncClient, repo: str, entry: RemoteFile, target: Path, already: int
) -> AsyncIterator[int]:
    headers = dict(_headers())
    if already:
        headers["Range"] = f"bytes={already}-"
    url = f"{_endpoint()}/{repo}/resolve/main/{entry.path}"
    async with client.stream("GET", url, headers=headers, timeout=None) as response:
        if already and response.status_code == 200:
            # The hub ignored the range and is sending the whole file; start it over rather
            # than append the beginning of a file to the middle of it.
            already = 0
        elif response.status_code not in (200, 206):
            raise HubRefused(f"fetching {entry.path} from {repo} returned {response.status_code}")
        moved = already
        with target.open("ab" if already else "wb") as sink:
            async for chunk in response.aiter_bytes(_CHUNK):
                sink.write(chunk)
                moved += len(chunk)
                yield moved
