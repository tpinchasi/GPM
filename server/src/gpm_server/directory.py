"""The model directory: what the pool could serve, cached from where the models are published
(D101).

Two sources, because the pool runs two kinds of engine:

- **Ollama's library** — every model, its sizes and capabilities, and each tag's download size,
  context window and inputs. Ollama publishes this only as web pages, so they are read as pages:
  if their layout changes, a refresh fails, says so, and the last good copy is kept and served.
- **The model hub** — for each model, its vLLM builds, sorted by `hubbuilds` (D100), each with
  the engine options that apply to its family. The hub cannot be asked for "every model", so it
  is asked per model: for the pool's own set, for every size in Ollama's library when the
  operator chooses that, and for anything the operator looks up. Slowly — at a stated number of
  requests a minute — because it is someone else's service.

The supervisor refreshes; both processes read. The router serves the cached copy to apps without
a single outbound request (D101): nothing slow reaches a process that carries inference traffic.
Nothing here spends, pulls or loads — a directory entry is information until the operator adds
it to the pool's model set through the file.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import httpx

from . import hubbuilds
from .db import Database

log = logging.getLogger("gpm.directory")

DEFAULT_OLLAMA_SITE = "https://ollama.com"

#: How many of Ollama's pages are read at once. It is a public site; this is a directory refresh,
#: not a crawl race.
OLLAMA_CONCURRENCY = 4

OLLAMA, HUB = "ollama", "hub"


def ollama_site() -> str:
    return (os.environ.get("GPM_OLLAMA_SITE") or DEFAULT_OLLAMA_SITE).rstrip("/")


# --- reading Ollama's pages ---


class PageChanged(ValueError):
    """A page no longer has the shape it was read by. The last good copy stays."""


@dataclass
class LibraryModel:
    name: str
    description: str
    capabilities: list[str]
    sizes: list[str]
    #: Offered on Ollama's own cloud. A model that is *only* that has nothing to download.
    cloud: bool
    pulls: Optional[str] = None
    tag_count: Optional[int] = None
    updated: Optional[str] = None

    @property
    def local(self) -> bool:
        """Something a host can download. An embedding model lists no sizes and is still one;
        a model with the cloud badge and no sizes is only on Ollama's cloud."""
        return bool(self.sizes) or not self.cloud


@dataclass
class LibraryTag:
    name: str
    tag: str
    digest: Optional[str]
    size_gb: Optional[float]
    context: Optional[str]
    inputs: list[str]
    #: "mlx" (Apple silicon only), "cloud" (nothing to download), or None for the standard build.
    runtime: Optional[str] = None
    #: This tag is what `latest` points to.
    is_latest: bool = False


_LIBRARY_ITEM = re.compile(r'<li\b[^>]*>\s*<a href="/library/(?P<name>[A-Za-z0-9._-]+)"[^>]*>(?P<body>.*?)</li>', re.S)
_DESCRIPTION = re.compile(r'<p class="max-w-lg[^"]*">(?P<text>.*?)</p>', re.S)
_BADGE = re.compile(r'<span\s+class="[^"]*\b(?P<kind>bg-indigo-50|bg-cyan-50|bg-\[#ddf4ff\])[^"]*">(?P<text>[^<]+)</span>')
_COUNTER = re.compile(r'<span\s*>(?P<value>[^<]+)</span>\s*<span class="hidden sm:flex">&nbsp;(?P<what>Pulls|Tags)</span>')
_UPDATED = re.compile(r'<span class="flex items-center" title="(?P<when>[^"]+)">')

_TAG_ROW = re.compile(r'<a href="/library/(?P<name>[A-Za-z0-9._-]+):(?P<tag>[^"]+)" class="md:hidden[^"]*">(?P<body>.*?)</a>', re.S)
_TAG_TEXT = re.compile(
    r"^(?P<full>\S+)\s+(?P<flags>(?:(?:latest|MLX)\s+)*)(?P<digest>[0-9a-f]{12})\s*•\s*(?P<size>[^•]+?)\s*•"
    r"\s*(?P<context>\S+)\s+context window\s*•\s*(?P<inputs>[^•]+?)\s+input"
)
_SIZE = re.compile(r"^(?P<n>[\d.]+)\s*(?P<unit>KB|MB|GB|TB)$")
_UNITS = {"KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_library(page: str) -> list[LibraryModel]:
    """Every model on Ollama's library page."""
    models = []
    for item in _LIBRARY_ITEM.finditer(page):
        body = item.group("body")
        description = _DESCRIPTION.search(body)
        badges = [(b.group("kind"), _text(b.group("text"))) for b in _BADGE.finditer(body)]
        counters = {c.group("what"): _text(c.group("value")) for c in _COUNTER.finditer(body)}
        updated = _UPDATED.search(body)
        tags = counters.get("Tags", "")
        models.append(LibraryModel(
            name=item.group("name"),
            description=_text(description.group("text")) if description else "",
            capabilities=[t for k, t in badges if k == "bg-indigo-50"],
            sizes=[t for k, t in badges if k == "bg-[#ddf4ff]"],
            cloud=any(k == "bg-cyan-50" for k, _ in badges),
            pulls=counters.get("Pulls"),
            tag_count=int(tags) if tags.isdigit() else None,
            updated=updated.group("when") if updated else None,
        ))
    if not models:
        raise PageChanged("Ollama's library page lists no models in the layout this reads")
    return models


def parse_tags(model: str, page: str) -> list[LibraryTag]:
    """Every tag on one model's tags page, with what its row says about it."""
    tags: list[LibraryTag] = []
    seen: set[str] = set()
    for row in _TAG_ROW.finditer(page):
        if row.group("name") != model or row.group("tag") in seen:
            continue
        seen.add(row.group("tag"))
        text = _text(row.group("body"))
        found = _TAG_TEXT.match(text)
        if not found:
            continue
        size = _SIZE.match(found.group("size").strip())
        flags = found.group("flags").split()
        tag = row.group("tag")
        cloud = size is None or "cloud" in tag
        tags.append(LibraryTag(
            name=f"{model}:{tag}",
            tag=tag,
            digest=found.group("digest"),
            size_gb=round(float(size.group("n")) * _UNITS[size.group("unit")], 3) if size else None,
            context=found.group("context"),
            inputs=[i.strip() for i in found.group("inputs").split(",") if i.strip()],
            runtime="cloud" if cloud else ("mlx" if "MLX" in flags else None),
            is_latest="latest" in flags,
        ))
    if not tags:
        raise PageChanged(f"the tags page of {model!r} lists no tags in the layout this reads")
    return tags


def size_tags(model: LibraryModel, tags: list[LibraryTag]) -> list[LibraryTag]:
    """The tags that are a size of the model (`26b`, `e4b`) rather than one of its encodings —
    what a hub build is looked up for. Each size's encodings are the same model. A model that
    lists no sizes has one, under `latest`."""
    wanted = set(model.sizes) or {"latest"}
    return [t for t in tags if t.tag in wanted and t.runtime is None]


# --- the cache (supervisor writes, both read) ---


class DirectoryStore:
    def __init__(self, database: Database):
        self.db = database

    def put(self, source: str, name: str, data: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO model_directory (source, name, data, fetched_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(source, name) DO UPDATE SET data = excluded.data, fetched_at = excluded.fetched_at",
            (source, name, json.dumps(data, separators=(",", ":")), time.time()),
        )

    def get(self, source: str, name: str) -> Optional[tuple[dict[str, Any], float]]:
        rows = self.db.query(
            "SELECT data, fetched_at FROM model_directory WHERE source = ? AND name = ?", (source, name)
        )
        return (json.loads(rows[0]["data"]), rows[0]["fetched_at"]) if rows else None

    def all(self, source: str) -> dict[str, tuple[dict[str, Any], float]]:
        rows = self.db.query("SELECT name, data, fetched_at FROM model_directory WHERE source = ?", (source,))
        return {r["name"]: (json.loads(r["data"]), r["fetched_at"]) for r in rows}

    def remove_missing(self, source: str, keep: set[str]) -> None:
        for name in set(self.all(source)) - keep:
            self.db.execute("DELETE FROM model_directory WHERE source = ? AND name = ?", (source, name))

    def record_run(self, source: str, **fields: Any) -> None:
        current = self.runs().get(source, {})
        merged = current | fields
        self.db.execute(
            "INSERT INTO directory_refresh (source, started_at, finished_at, ok, detail, count, last_success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(source) DO UPDATE SET started_at = excluded.started_at, "
            "finished_at = excluded.finished_at, ok = excluded.ok, detail = excluded.detail, "
            "count = excluded.count, last_success = excluded.last_success",
            (source, merged.get("started_at"), merged.get("finished_at"), merged.get("ok"),
             merged.get("detail"), merged.get("count"), merged.get("last_success")),
        )

    def runs(self) -> dict[str, dict[str, Any]]:
        return {r["source"]: dict(r) for r in self.db.query("SELECT * FROM directory_refresh")}


def read_directory(database: Database, query: Optional[str] = None,
                   model_set: tuple[str, ...] = ()) -> dict[str, Any]:
    """The directory as both processes serve it: Ollama's models, each size's hub builds where
    they have been looked up, and names looked up on the hub that Ollama does not list.

    Pure reading. `query` filters by substring of a model's name or description.
    """
    store = DirectoryStore(database)
    library = store.all(OLLAMA)
    hub = store.all(HUB)
    wanted = (query or "").strip().lower()
    in_pool = set(model_set)
    models = []
    listed: set[str] = set()
    for name, (entry, fetched_at) in sorted(library.items(), key=lambda kv: kv[1][0].get("rank", 1 << 30)):
        if wanted and wanted not in name.lower() and wanted not in entry.get("description", "").lower():
            continue
        tags = []
        for tag in entry.get("tags", []):
            listed.add(tag["name"])
            builds = hub.get(tag["name"])
            tags.append(tag | {
                "in_pool": tag["name"] in in_pool,
                "vllm": (builds[0] | {"looked_up_at": builds[1]}) if builds else None,
            })
        models.append({k: v for k, v in entry.items() if k not in ("tags", "rank")}
                      | {"tags": tags, "fetched_at": fetched_at})
    elsewhere = []
    for name, (builds, fetched_at) in sorted(hub.items()):
        if name in listed or (wanted and wanted not in name.lower()):
            continue
        elsewhere.append({"name": name, "in_pool": name in in_pool,
                          "vllm": builds | {"looked_up_at": fetched_at}})
    return {"refreshed": store.runs(), "models": models, "looked_up_elsewhere": elsewhere}


def build_sizes_gb(database: Database, builds: dict[str, str]) -> dict[str, Optional[float]]:
    """Each build's size on disk, by the model it is a build of — or None where the directory
    has not measured it (D108). `builds` maps a pool model to the tag its engine fetches.

    Read from what the directory already holds: a hub build's weight files, as listed when the
    pool's builds were looked up, and an Ollama tag's size from its library page. Nothing is
    fetched here; a build never looked up is unknown, and said to be.
    """
    store = DirectoryStore(database)
    hub = store.all(HUB)
    library_tags = {
        tag.get("name"): tag.get("size_gb")
        for entry, _ in store.all(OLLAMA).values() for tag in entry.get("tags", [])
    }
    sizes: dict[str, Optional[float]] = {}
    for model, tag in builds.items():
        found: Optional[float] = None
        looked_up = hub.get(model)
        if looked_up:
            found = next(
                (b.get("size_gb") for b in looked_up[0].get("builds", []) if b.get("repo") == tag), None
            )
        if found is None:
            found = library_tags.get(tag)
        sizes[model] = float(found) if found else None
    return sizes


# --- refreshing (the supervisor only) ---


def hub_engine_options() -> dict[str, Any]:
    """The named options of every installed engine whose builds live on the hub (D100)."""
    from .engines import EngineNotFound, available_engines, get_engine

    options: dict[str, Any] = {}
    for name in available_engines():
        try:
            engine = get_engine(name)
        except EngineNotFound:
            continue
        if getattr(engine, "builds_on_hub", False):
            options.update(getattr(engine, "options", None) or {})
    return options


def with_options(found: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Each build with the engine options its family has — what the operator may tick for it."""
    for build in found.get("builds", []):
        build["options"] = sorted(
            name for name, option in options.items() if build.get("family") in option.families
        )
    return found


class Directory:
    """Refreshes the cache. One refresh at a time; lookups the operator asks for go ahead of it."""

    def __init__(self, database: Database, settings: Callable[[], Any], *,
                 options: Optional[Callable[[], dict[str, Any]]] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self.store = DirectoryStore(database)
        #: The pool's current `directory` settings and model set, read at use: the file may change.
        self._settings = settings
        self._options = options or hub_engine_options
        self._transport = transport
        self._task: Optional[asyncio.Task] = None
        self._hub_gate = asyncio.Lock()
        self._last_hub_request = 0.0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _client(self, base_url: str) -> httpx.AsyncClient:
        # No credential on either: public listings, read as anyone would read them.
        return httpx.AsyncClient(base_url=base_url, timeout=30.0, transport=self._transport,
                                 follow_redirects=True, headers={"User-Agent": "gpm-model-directory"})

    # -- Ollama --

    async def refresh_ollama(self) -> int:
        started = time.time()
        self.store.record_run(OLLAMA, started_at=started, finished_at=None, ok=None, detail="refreshing")
        try:
            async with self._client(ollama_site()) as client:
                answer = await client.get("/library")
                answer.raise_for_status()
                models = parse_library(answer.text)
                gate = asyncio.Semaphore(OLLAMA_CONCURRENCY)
                problems: list[str] = []

                async def one(rank: int, model: LibraryModel) -> Optional[str]:
                    if not model.local:
                        return None  # only on Ollama's cloud: nothing a host could download
                    async with gate:
                        try:
                            page = await client.get(f"/library/{model.name}/tags")
                            page.raise_for_status()
                            tags = parse_tags(model.name, page.text)
                        except (httpx.HTTPError, PageChanged) as exc:
                            problems.append(f"{model.name}: {exc}")
                            return None
                    entry = asdict(model) | {"rank": rank, "tags": [asdict(t) for t in tags]}
                    self.store.put(OLLAMA, model.name, entry)
                    return model.name

                kept = await asyncio.gather(*(one(i, m) for i, m in enumerate(models)))
            names = {n for n in kept if n}
            if names:
                self.store.remove_missing(OLLAMA, names)
            detail = f"{len(names)} models" + (f"; {len(problems)} could not be read" if problems else "")
            if problems:
                log.warning("directory: %s", "; ".join(problems[:5]))
            self.store.record_run(OLLAMA, finished_at=time.time(), ok=1, detail=detail,
                                  count=len(names), last_success=time.time())
            return len(names)
        except (httpx.HTTPError, PageChanged) as exc:
            # The last good copy stays, and is still what everyone is served.
            self.store.record_run(OLLAMA, finished_at=time.time(), ok=0,
                                  detail=f"not refreshed: {exc or type(exc).__name__}; the last copy is kept")
            raise

    # -- the hub --

    async def _paced(self) -> None:
        """Hold hub requests to the stated rate, whoever asked for them — a refresh's sweep and
        an operator's click share it."""
        per_minute = max(1, int(self._settings().directory.hub_requests_per_minute))
        async with self._hub_gate:
            wait = self._last_hub_request + 60.0 / per_minute - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_hub_request = time.monotonic()

    async def lookup(self, name: str, search: Optional[str] = None, *, fresh: bool = False) -> dict[str, Any]:
        """One model's builds on the hub: from the cache unless it is stale or `fresh` is asked.

        A search the operator typed is cached under the model's name too — it is their answer
        to "which builds are this model's", and the next person should see it.
        """
        max_age = float(self._settings().directory.hub_max_age_hours) * 3600
        cached = self.store.get(HUB, name)
        if cached and not fresh and not search and time.time() - cached[1] < max_age:
            return cached[0] | {"looked_up_at": cached[1], "cached": True}
        async with self._client(hubbuilds.hub_url()) as client:
            found = await hubbuilds.find_builds(name, search, client=client, pace=self._paced)
        entry = with_options(found.as_dict(), self._options())
        self.store.put(HUB, name, entry)
        return entry | {"looked_up_at": time.time(), "cached": False}

    async def search_hub(self, term: str) -> list[dict[str, Any]]:
        """Models on the hub whose name holds `term` (D111), at the directory's pace. Not cached:
        a search is an operator looking, and the builds of whatever they choose are."""
        async with self._client(hubbuilds.hub_url()) as client:
            found = await hubbuilds.search_models(term, client=client, pace=self._paced)
        return [asdict(model) for model in found]

    def hub_names(self) -> list[str]:
        """What a refresh looks up on the hub, by the operator's choice."""
        settings = self._settings()
        choice = settings.directory.hub_builds
        names = list(settings.pool.model_set) if choice in ("pool", "all") else []
        if choice == "all":
            for _, (entry, _) in sorted(self.store.all(OLLAMA).items()):
                model = LibraryModel(**{k: entry[k] for k in LibraryModel.__dataclass_fields__})
                tags = [LibraryTag(**t) for t in entry.get("tags", [])]
                names += [t.name for t in size_tags(model, tags)]
        return list(dict.fromkeys(names))

    async def refresh_hub(self) -> int:
        names = self.hub_names()
        self.store.record_run(HUB, started_at=time.time(), finished_at=None, ok=None,
                              detail=f"looking up {len(names)} models")
        done, failed = 0, []
        for name in names:
            try:
                await self.lookup(name)
                done += 1
            except (hubbuilds.HubUnavailable, ValueError) as exc:
                failed.append(f"{name}: {exc}")
        ok = not failed or done > 0
        detail = f"{done} models looked up" + (f"; {len(failed)} failed" if failed else "")
        self.store.record_run(HUB, finished_at=time.time(), ok=1 if ok else 0, detail=detail,
                              count=done, **({"last_success": time.time()} if ok else {}))
        return done

    # -- both --

    async def refresh(self) -> None:
        settings = self._settings().directory
        if settings.ollama_library:
            try:
                await self.refresh_ollama()
            except (httpx.HTTPError, PageChanged):
                log.warning("directory: Ollama's library was not refreshed", exc_info=True)
        if settings.hub_builds != "none":
            await self.refresh_hub()

    def start(self) -> bool:
        """Start a refresh in the background; False when one is already running."""
        if self.running:
            return False
        self._task = asyncio.create_task(self._guarded())
        return True

    async def _guarded(self) -> None:
        try:
            await self.refresh()
        except Exception:  # a refresh must never take the supervisor down
            log.exception("directory refresh failed")

    def due(self) -> bool:
        hours = float(self._settings().directory.refresh_hours)
        if hours <= 0 or self.running:
            return False
        runs = self.store.runs()
        sources = [OLLAMA] if self._settings().directory.ollama_library else []
        if self._settings().directory.hub_builds != "none":
            sources.append(HUB)
        last = min((runs.get(s, {}).get("started_at") or 0) for s in sources) if sources else time.time()
        return time.time() - last >= hours * 3600

    def maybe_start(self) -> None:
        if self.due():
            self.start()

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
