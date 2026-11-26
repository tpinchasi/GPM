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

Beside the refresh, which crawls and is off unless the operator sets how often: the **sizes of
the pool's own builds** (D207). For every build in the pool's model set whose size the catalog
does not state, the supervisor reads just that build's page — one Ollama tags page, or one hub
file listing — on its own, cached, and again weekly to see a tag pushed again. And what a host's
engine reports a build took, on disk and loaded, is kept by digest (`build_measurements`): it is
the size the pool goes by next time, and a new digest of the same tag is measured again.

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
#: One hub repository's exact weights, read from its file listing when a variant is chosen (D111).
HUB_SIZE = "hub-size"
#: One Ollama model's tags page, read for the sizes of the pool's own builds (D207) — kept apart
#: from the refresh's copy of the library, which a refresh replaces whole.
OLLAMA_TAGS = "ollama-tags"
#: How long a build whose size could not be read waits before it is asked again.
SIZE_RETRY_S = 6 * 3600
#: How often the supervisor checks whether any of the pool's builds has no size.
SIZES_CHECK_S = 60.0

#: A name in Ollama's own library: one word, no owner. `owner/model` and `hf.co/...` are not
#: listed there, so nothing is looked up for them; a host that holds one measures it.
_LIBRARY_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def ollama_site() -> str:
    return (os.environ.get("GPM_OLLAMA_SITE") or DEFAULT_OLLAMA_SITE).rstrip("/")


def library_name(tag: str) -> Optional[tuple[str, str]]:
    """An Ollama build's model and tag in the library (`gemma4:31b` → `gemma4`, `31b`; a bare
    name is its `latest`), or None for a name the library does not list."""
    name, _, version = tag.partition(":")
    if not _LIBRARY_NAME.match(name) or "/" in version:
        return None
    return name, version or "latest"


def _full_tag(tag: str) -> str:
    """A build's name as Ollama itself reports it: a name with no tag of its own — `gemma4`,
    `owner/model`, `hf.co/org/repo` — is its `latest`."""
    return tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"


def _hub_repo(tag: str) -> bool:
    """A hub repository's own name, `owner/name` — not a file inside one (`owner/repo/x.bin`),
    nor a build uploaded to the pool's own store (`upload/<sha256>`), which is never sent out."""
    owner, _, name = tag.partition("/")
    return bool(owner and name) and "/" not in name and ":" not in tag and owner != "upload" \
        and hubbuilds.valid_search(tag)


def _bare_digest(digest: Any) -> str:
    return str(digest or "").removeprefix("sha256:").lower()


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


def _published_tags(store: "DirectoryStore") -> dict[str, dict[str, Any]]:
    """Every Ollama tag the directory has read, by its full name — from a refresh of the library
    or from the pool's own builds' pages (D207), the more recently read where both have it."""
    found: dict[str, tuple[dict[str, Any], float]] = {}
    for source in (OLLAMA, OLLAMA_TAGS):
        for entry, fetched_at in store.all(source).values():
            for tag in entry.get("tags", []) or []:
                name = tag.get("name")
                if name and (name not in found or fetched_at > found[name][1]):
                    found[name] = (tag, fetched_at)
    return {name: tag for name, (tag, _) in found.items()}


def _measurements(database: Database) -> dict[str, list[dict[str, Any]]]:
    """Every measurement, by tag, newest first (D207)."""
    found: dict[str, list[dict[str, Any]]] = {}
    for row in database.query("SELECT * FROM build_measurements ORDER BY measured_at DESC"):
        found.setdefault(row["tag"], []).append(dict(row))
    return found


def _measured(rows: Optional[list[dict[str, Any]]], published: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The measurement of the build published now. Where the library shows the tag's digest,
    only that digest's: another is a build the tag no longer names (pushed again since), and the
    new one is to be measured. Where it shows none, the most recently measured digest."""
    if not rows:
        return None
    shown = _bare_digest((published or {}).get("digest"))
    if not shown:
        return rows[0]
    return next((r for r in rows if _bare_digest(r.get("digest")).startswith(shown)), None)


def build_sizes_gb(database: Database, builds: dict[str, str]) -> dict[str, Optional[float]]:
    """Each build's size on disk, by the model it is a build of — or None where nothing has
    measured it (D108). `builds` maps a pool model to the tag its engine fetches.

    Read from what is already held: a hub build's weight files, as listed when the pool's
    builds were looked up; what a host measured the build's files to be, by digest (D207); an
    Ollama tag's size from its library page; one repository's exact listing. Nothing is
    fetched here; a build never looked up or held is unknown, and said to be.
    """
    size_of = sizer(database)
    return {model: size_of(model, tag) for model, tag in builds.items()}


def sizer(database: Database) -> Callable[[str, str], Optional[float]]:
    """`build_sizes_gb` for one build at a time, with what the pool holds read once."""
    store = DirectoryStore(database)
    hub = store.all(HUB)
    published = _published_tags(store)
    measured = _measurements(database)

    def size_of(model: str, tag: str) -> Optional[float]:
        found: Optional[float] = None
        looked_up = hub.get(model)
        if looked_up:
            found = next(
                (b.get("size_gb") for b in looked_up[0].get("builds", []) if b.get("repo") == tag), None
            )
        shown = published.get(_full_tag(tag)) or published.get(tag)
        seen = _measured(measured.get(_full_tag(tag)) or measured.get(tag), shown)
        if found is None and seen:
            found = seen.get("disk_gb")
        if found is None:
            found = (shown or {}).get("size_gb")
        if found is None:
            exact = store.get(HUB_SIZE, tag)
            found = exact[0].get("size_gb") if exact else None
        return float(found) if found else None

    return size_of


def build_memory_gb(database: Database, builds: dict[str, str]) -> dict[str, Optional[float]]:
    """What each build was seen to take loaded, by model — weights, cache and working space, as
    the engine of a host that held that very build reported it (D207) — or None where no host
    has, or the tag has been pushed again since."""
    published = _published_tags(DirectoryStore(database))
    measured = _measurements(database)
    found: dict[str, Optional[float]] = {}
    for model, tag in builds.items():
        shown = published.get(_full_tag(tag)) or published.get(tag)
        seen = _measured(measured.get(_full_tag(tag)) or measured.get(tag), shown)
        memory = seen.get("memory_gb") if seen else None
        found[model] = float(memory) if memory else None
    return found


def _gb(value: Any) -> Optional[float]:
    return round(value / 1e9, 3) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def record_measurements(database: Database, engine: Any, host_id: Optional[str]) -> int:
    """Keep what a host's engine says about each build it holds (D207): its digest, its files'
    size and, where loaded, what it takes in memory and at what context. One row per build and
    digest, written only when something in it changed. Where hosts hold the same build at
    different contexts or cards, the most memory any was seen to take is kept: the figure a
    machine is sized from, and one that does not swing between hosts on every pass. Returns how
    many rows were written.

    `engine` is the engine part of what the host's agent reports; one that names no digests
    (an older agent, or an engine that keeps none) leaves nothing to keep.
    """
    if not isinstance(engine, dict) or not engine.get("answers"):
        return 0
    name = str(engine.get("name") or "")
    loaded = {
        m.get("tag"): m for m in engine.get("models_loaded_sizes") or []
        if isinstance(m, dict) and isinstance(m.get("tag"), str)
    }
    written = 0
    for held in engine.get("models_on_disk") or []:
        if not isinstance(held, dict):
            continue
        tag, digest = held.get("tag"), held.get("digest")
        if not isinstance(tag, str) or not tag or not isinstance(digest, str) or not digest:
            continue
        tag = _full_tag(tag)
        disk = _gb(held.get("size_bytes"))
        running = loaded.get(tag) or loaded.get(held.get("tag")) or {}
        same = _bare_digest(running.get("digest") or digest) == _bare_digest(digest)
        memory = _gb(running.get("size_bytes")) if running and same else None
        context = running.get("context") if running and same and isinstance(running.get("context"), int) else None
        rows = database.query(
            "SELECT disk_gb, memory_gb, context FROM build_measurements WHERE engine = ? AND tag = ? AND digest = ?",
            (name, tag, digest),
        )
        if rows:
            was = rows[0]
            # A build not loaded right now, or seen taking less elsewhere, keeps the most it took.
            if memory is None or (was["memory_gb"] or 0) >= memory:
                memory, context = was["memory_gb"], was["context"]
            if (was["disk_gb"], was["memory_gb"], was["context"]) == (disk, memory, context):
                continue
        database.execute(
            "INSERT INTO build_measurements (engine, tag, digest, disk_gb, memory_gb, context, host_id, measured_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(engine, tag, digest) DO UPDATE SET "
            "disk_gb = excluded.disk_gb, memory_gb = excluded.memory_gb, context = excluded.context, "
            "host_id = excluded.host_id, measured_at = excluded.measured_at",
            (name, tag, digest, disk, memory, context, host_id, time.time()),
        )
        written += 1
    return written


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
        #: The lookup of the pool's own builds' sizes (D207), apart from a refresh.
        self._sizes_task: Optional[asyncio.Task] = None
        self._sizes_checked_at = 0.0
        #: A build whose size could not be read, and when: not asked again for SIZE_RETRY_S while
        #: this supervisor runs.
        self._size_failed: dict[str, float] = {}
        self._closed = False

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
        """Models on the hub whose name holds `term`, each with its variants (D111), at the
        directory's pace. Not cached: a search is an operator looking."""
        async with self._client(hubbuilds.hub_url()) as client:
            found = await hubbuilds.search_variants(term, client=client, pace=self._paced)
        options = self._options()
        groups = []
        for group in found:
            entry = asdict(group)
            with_options({"builds": entry["variants"]}, options)
            groups.append(entry)
        return groups

    async def exact_size(self, repo: str) -> Optional[float]:
        """One repository's weights, exactly, at the directory's pace — read once, then kept."""
        cached = self.store.get(HUB_SIZE, repo)
        if cached is not None:
            return cached[0].get("size_gb")
        async with self._client(hubbuilds.hub_url()) as client:
            size = await hubbuilds.exact_size_gb(repo, client=client, pace=self._paced)
        if size is not None:
            self.store.put(HUB_SIZE, repo, {"size_gb": size})
        return size

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

    # -- the sizes of the pool's own builds (D207) --

    def size_lookups(self) -> list[tuple[str, str]]:
        """What to read for the sizes of the pool's own builds: `("ollama", model)` for an Ollama
        model's tags page, `("hub", repo)` for one repository's file listing.

        Only builds whose size the catalog does not state. An Ollama build's page is read when
        neither this lookup nor a refresh of the library has read it in `hub_max_age_hours` —
        which is also how a tag pushed again is seen: its digest changes. A hub build's listing
        is read once, when nothing the pool holds knows its size. Never a hosted build, which
        has nothing to download, nor one that failed to read in the last SIZE_RETRY_S.
        """
        config = self._settings()
        settings = config.directory
        if not getattr(settings, "look_up_sizes", True):
            return []
        from .catalog import Variant
        from .engines import EngineNotFound, get_engine

        max_age = float(settings.hub_max_age_hours) * 3600
        now = time.time()
        models = list(config.pool.model_set)
        if config.rented is not None and config.rented.models:
            models += [m for m in config.rented.models if m not in models]
        wanted: list[tuple[str, str]] = []
        hub_builds: list[tuple[str, str]] = []  # (model, repo), sized together below
        for model in models:
            entry = config.catalog.get(model)
            for variant in (entry.variants if entry else [Variant(tag=model)]):
                if variant.size_gb or any(r.startswith("hosted:") for r in variant.requires):
                    continue
                kind: Optional[str] = None
                if variant.engine:
                    try:
                        engine = get_engine(variant.engine)
                    except EngineNotFound:
                        continue
                    if getattr(engine, "builds_on_hub", False):
                        kind = "hub"
                    elif variant.engine == "ollama":
                        kind = "ollama"
                elif library_name(variant.tag):
                    kind = "ollama"
                else:
                    kind = "hub"
                if kind == "ollama" and library_name(variant.tag):
                    name = library_name(variant.tag)[0]
                    if ("ollama", name) in wanted or now - self._size_failed.get(f"ollama:{name}", 0.0) < SIZE_RETRY_S:
                        continue
                    # Read recently, by this lookup or by a refresh of the whole library.
                    read = [r for r in (self.store.get(OLLAMA_TAGS, name), self.store.get(OLLAMA, name)) if r]
                    if any(now - fetched_at < max_age for _, fetched_at in read):
                        continue
                    wanted.append(("ollama", name))
                elif kind == "hub" and _hub_repo(variant.tag):
                    if now - self._size_failed.get(f"hub:{variant.tag}", 0.0) < SIZE_RETRY_S:
                        continue
                    if self.store.get(HUB_SIZE, variant.tag) is None:
                        hub_builds.append((model, variant.tag))
        if hub_builds:
            # One read of what the pool holds for them all: a build something already sizes is
            # not asked for.
            size_of = sizer(self.store.db)
            for model, repo in hub_builds:
                if not size_of(model, repo) and ("hub", repo) not in wanted:
                    wanted.append(("hub", repo))
        return wanted

    async def read_tags_page(self, name: str) -> list[LibraryTag]:
        """One Ollama model's tags page, kept under OLLAMA_TAGS (D207). A page that cannot be
        read keeps what was read before."""
        async with self._client(ollama_site()) as client:
            page = await client.get(f"/library/{name}/tags")
            page.raise_for_status()
            tags = parse_tags(name, page.text)
        self.store.put(OLLAMA_TAGS, name, {"tags": [asdict(t) for t in tags]})
        return tags

    async def look_up_sizes(self) -> int:
        """Read what `size_lookups` names, one after another; returns how many were read."""
        done = 0
        for kind, name in self.size_lookups():
            key = f"{kind}:{name}"
            try:
                if kind == "ollama":
                    await self.read_tags_page(name)
                    found = True
                else:
                    found = await self.exact_size(name) is not None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — any failure waits its hours, not a minute
                log.info("directory: the size of %s could not be read: %s", name, exc)
                found = False
            if found:
                self._size_failed.pop(key, None)
                done += 1
            else:
                self._size_failed[key] = time.time()
        return done

    def maybe_look_up_sizes(self) -> bool:
        """Start the lookup in the background when one of the pool's builds wants it — checked
        once a minute, never while one is running. True when one was started."""
        if self._closed or (self._sizes_task is not None and not self._sizes_task.done()):
            return False
        if time.monotonic() - self._sizes_checked_at < SIZES_CHECK_S and self._sizes_checked_at:
            return False
        self._sizes_checked_at = time.monotonic()
        try:
            if not self.size_lookups():
                return False
        except Exception:  # noqa: BLE001 — a pass must never stop over a lookup
            log.exception("directory: the pool's builds could not be checked for sizes")
            return False
        self._sizes_task = asyncio.create_task(self._sizes_guarded(), name="gpm:directory-sizes")
        return True

    async def _sizes_guarded(self) -> None:
        try:
            await self.look_up_sizes()
        except Exception:  # a lookup must never take the supervisor down
            log.exception("directory: looking up the sizes of the pool's builds failed")

    async def aclose(self) -> None:
        self._closed = True
        for task in (self._task, self._sizes_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
