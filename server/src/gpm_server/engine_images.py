"""The engine images rented hosts start from, kept current (D177).

A pool that names no image — or `rented.image: latest` — rents each new host from the **newest
stable release of its engine's official image**, found in the registry by the supervisor, off
every request path, every `REFRESH_S`. Only the engine's own repository, by a table here; only
release tags (digits and dots, as each engine writes them): never a nightly, a release candidate,
another accelerator's build or a floating tag. What is chosen is a version tag, the same pinned
form an operator writes (threat model T10), recorded with the digest the registry gave for it.

A new version must prove itself. Until a host started from it has served, it is on trial: a host
from it whose engine never answers counts against it, and after `FAILURES_TO_DROP` such hosts it
is dropped — the pool goes back to the version before and never takes the dropped one again on
its own, saying so. Hosts already running keep their image; rented hosts are short-lived, so the
pool turns over by itself.

A pool that names an image (`rented.image`, `rented.images`) is held to it: nothing here runs.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

#: How often the registry is asked; and how long after a failed ask it is asked again.
REFRESH_S = 6 * 3600.0
RETRY_S = 15 * 60.0
#: Hosts from a version on trial whose engine never answered, before the version is dropped. Two,
#: so one bad machine — a dead card, a provider's hiccup — does not drop a good release.
FAILURES_TO_DROP = 2
#: A registry answer is a few hundred kilobytes; anything past this is not one.
MAX_ANSWER_BYTES = 4 * 1024 * 1024
#: Where tags are listed. Docker Hub's public API needs no account.
REGISTRY = "https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100&ordering=last_updated"
#: The key the tracker's state is kept under in `pool_meta`.
META_KEY = "engine_images"


@dataclasses.dataclass(frozen=True)
class Track:
    """One image line of an engine: its official repository, which of its tags are releases,
    and the accelerator driver a build of that line needs (None: the offer policy's floor)."""

    key: str
    repo: str
    #: Release tags of this line, the version's numbers as groups.
    pattern: str
    min_driver: Optional[str] = None
    note: str = ""


#: The official image lines of each engine. Newest driver line first: a machine gets the first
#: whose driver floor it meets (D92).
TRACKS: dict[str, tuple[Track, ...]] = {
    "ollama": (
        Track("ollama", "ollama/ollama", r"^(\d+)\.(\d+)\.(\d+)$", note="Ollama's own image"),
    ),
    "vllm": (
        Track("vllm", "vllm/vllm-openai", r"^v(\d+)\.(\d+)\.(\d+)$", min_driver="580",
              note="vLLM's own image, CUDA 13"),
        Track("vllm-cu129", "vllm/vllm-openai", r"^v(\d+)\.(\d+)\.(\d+)-cu129$", min_driver="550",
              note="vLLM's own image, CUDA 12.9, for older drivers"),
    ),
}


def tracked(engine: str) -> tuple[Track, ...]:
    return TRACKS.get(engine, ())


def newest(tags: Any, track: Track, dropped: set[str] = frozenset()) -> Optional[tuple[str, str]]:
    """The newest release tag of this line the registry lists for x86-64, with its digest — never
    one dropped. Versions compare by their numbers, never by date or by the order listed."""
    pattern = re.compile(track.pattern)
    best: Optional[tuple[tuple[int, ...], str, str]] = None
    for tag in tags if isinstance(tags, list) else []:
        if not isinstance(tag, dict) or not isinstance(tag.get("name"), str):
            continue
        name = tag["name"]
        match = pattern.match(name)
        if not match or name in dropped or len(name) > 128:
            continue
        images = tag.get("images")
        if isinstance(images, list) and images and not any(
                isinstance(i, dict) and i.get("architecture") == "amd64" for i in images):
            continue
        version = tuple(int(n) for n in match.groups())
        digest = tag.get("digest") if isinstance(tag.get("digest"), str) else ""
        if best is None or version > best[0]:
            best = (version, name, digest)
    return (best[1], best[2]) if best else None


async def _fetch(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0), follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_ANSWER_BYTES:
                    raise ValueError("the registry's answer is larger than any list of tags")
                chunks.append(chunk)
            return b"".join(chunks)


class ImageTracker:
    """Which image each tracked line rents from now, and why; kept in the database so a restart
    neither forgets a dropped version nor asks the registry again at once."""

    def __init__(self, db: Any, events: Any, fetch: Optional[Callable[[str], Awaitable[bytes]]] = None):
        self.db = db
        self.events = events
        self.fetch = fetch or _fetch
        self.state: dict[str, dict[str, Any]] = self._load()
        #: Why the registry could not be asked, the last time it could not; None when it could.
        self.problem: Optional[str] = None
        self._next_try = 0.0

    # --- state ---

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            rows = self.db.query("SELECT value FROM pool_meta WHERE key = ?", (META_KEY,))
            data = json.loads(rows[0]["value"]) if rows else {}
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - a state that cannot be read is started afresh
            return {}

    def _save(self) -> None:
        self.db.execute("INSERT OR REPLACE INTO pool_meta (key, value) VALUES (?, ?)",
                        (META_KEY, json.dumps(self.state, sort_keys=True)))

    def line(self, track: Track) -> dict[str, Any]:
        return self.state.setdefault(track.key, {"repo": track.repo, "tag": None, "digest": "", "previous": None,
                                                  "proven": False, "failures": 0, "dropped": [], "changed_at": None,
                                                  "checked_at": None})

    # --- asking the registry ---

    def due(self, engines: list[str]) -> bool:
        now = time.time()
        if now < self._next_try:
            return False
        checked = [self.line(t).get("checked_at") or 0 for e in engines for t in tracked(e)]
        return any(now - c >= REFRESH_S for c in checked)

    async def refresh(self, engines: list[str]) -> None:
        """Ask the registry for each line's newest release and take it where it is new. A failure
        keeps what each line has, and is asked again after `RETRY_S`."""
        listings: dict[str, Any] = {}
        try:
            for engine in engines:
                for track in tracked(engine):
                    if track.repo not in listings:
                        body = json.loads(await self.fetch(REGISTRY.format(repo=track.repo)))
                        listings[track.repo] = body.get("results") if isinstance(body, dict) else None
        except Exception as exc:  # noqa: BLE001 - the registry being away changes nothing
            self.problem = f"the registry could not be asked ({type(exc).__name__}); the images in use are kept"
            self._next_try = time.time() + RETRY_S
            return
        self.problem = None
        now = time.time()
        for engine in engines:
            for track in tracked(engine):
                state = self.line(track)
                state["checked_at"] = now
                found = newest(listings.get(track.repo), track, set(state["dropped"]))
                if found is None or found[0] == state["tag"]:
                    continue
                tag, digest = found
                was = state["tag"]
                state.update({"previous": was, "tag": tag, "digest": digest, "proven": False, "failures": 0,
                              "changed_at": now})
                self.events.record(
                    "engine_image_new",
                    f"rented hosts of {engine} now start from {track.repo}:{tag}"
                    + (f" (was {was})" if was else "") + "; on trial until a host started from it serves",
                    numbers={"engine": engine, "line": track.key, "image": f"{track.repo}:{tag}", "digest": digest,
                             "previous": was})
        self._save()

    # --- what is rented from ---

    def images(self, engine: str, fallback: list[tuple[str, Optional[str], str]]) -> list[tuple[str, Optional[str], str]]:
        """(image, min_driver, note) for each line with a version, newest driver first; `fallback`
        — the engine's shipped images — for a pool whose registry has never answered."""
        out = []
        for track in tracked(engine):
            state = self.line(track)
            if state["tag"]:
                out.append((f"{track.repo}:{state['tag']}", track.min_driver,
                            track.note + ("" if state["proven"] else " — new, on trial")))
        return out or fallback

    def outcome(self, image: Optional[str], served: bool) -> None:
        """A host started from `image` served, or its engine never answered. Only a version on
        trial is judged by it."""
        if not image:
            return
        for engine, tracks in TRACKS.items():
            for track in tracks:
                state = self.state.get(track.key)
                if not state or not state.get("tag") or image != f"{track.repo}:{state['tag']}" or state["proven"]:
                    continue
                if served:
                    state["proven"] = True
                    self.events.record("engine_image_proven",
                                       f"{image} served on a rented host; no longer on trial",
                                       numbers={"engine": engine, "image": image})
                else:
                    state["failures"] += 1
                    if state["failures"] >= FAILURES_TO_DROP:
                        dropped = state["tag"]
                        state["dropped"] = sorted(set(state["dropped"]) | {dropped})
                        state.update({"tag": state["previous"], "previous": None, "proven": bool(state["previous"]),
                                      "failures": 0, "digest": "", "changed_at": time.time()})
                        self.events.record(
                            "engine_image_dropped",
                            f"{track.repo}:{dropped}: {FAILURES_TO_DROP} hosts started from it never answered, so "
                            + (f"rented hosts go back to {track.repo}:{state['tag']}" if state["tag"]
                               else "rented hosts go back to the engine's shipped image")
                            + "; it is not taken again on its own",
                            numbers={"engine": engine, "image": image, "back_to": state["tag"]})
                self._save()
                return

    def view(self, engine: str) -> list[dict[str, Any]]:
        """What the console and the status show: each line's image, the one before, and since when."""
        return [{"line": t.key, "repo": t.repo, "image": f"{t.repo}:{s['tag']}" if s["tag"] else None,
                 "digest": s["digest"], "previous": f"{t.repo}:{s['previous']}" if s["previous"] else None,
                 "on_trial": bool(s["tag"]) and not s["proven"], "dropped": s["dropped"], "changed_at": s["changed_at"],
                 "checked_at": s["checked_at"], "min_driver": t.min_driver, "note": t.note}
                for t in tracked(engine) for s in [self.line(t)]]
