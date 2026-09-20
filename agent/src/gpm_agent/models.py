"""Making this machine hold the pool's model set (docs/spec/host-agent.md §4).

Level-triggered: the pool sends the whole desired state every pass, and this module works
toward it in the background — one pull at a time — and answers at once with where things
stand. Nothing here remembers the pool between restarts; the next pass says it all again.

What it will not do is as much the design as what it will: pull anything but a syntactically
plain tag, pull past the owner's free-disk floor, delete what the pool still requires, or
delete at all where the machine's owner said no.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .engines import EngineRefused, OllamaFacts

#: A model tag: a name, optionally namespaced, optionally versioned. Never a URL, a path with
#: dots leading out of anywhere, or anything with a scheme — those are refused before the
#: engine ever sees them.
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*(:[A-Za-z0-9][A-Za-z0-9._-]*)?$")
#: How long a tag that failed to pull is left alone before another attempt.
RETRY_AFTER_S = 60.0


class Refused(Exception):
    def __init__(self, status: int, error: str, detail: str):
        super().__init__(detail)
        self.status, self.error, self.detail = status, error, detail


def valid_tag(tag: Any) -> bool:
    return isinstance(tag, str) and len(tag) <= 200 and ".." not in tag and bool(_TAG.match(tag))


@dataclasses.dataclass
class Desired:
    tags: tuple[str, ...] = ()
    residency: str = "on_demand"


class ModelWork:
    def __init__(
        self,
        engine: OllamaFacts,
        client: httpx.AsyncClient,
        *,
        free_disk_bytes: Callable[[], Optional[int]],
        min_free_disk_gb: float,
        allow_delete: bool,
        state_path: Optional[Path] = None,
    ):
        self.engine, self.client = engine, client
        self.free_disk_bytes = free_disk_bytes
        self.floor_bytes = int(min_free_disk_gb * 1e9)
        self.allow_delete = allow_delete
        self.desired = Desired()
        self.pulling: Optional[dict[str, Any]] = None
        self.errors: dict[str, tuple[float, str]] = {}
        #: Tags this process pinned. Only these are ever released: a model the machine's owner
        #: pinned for their own reasons is not the pool's to unpin.
        #: Remembered across the agent's own restarts, in a file beside its settings — or the
        #: agent would forget what it pinned and never release it. It is the one thing about a
        #: pool this agent keeps; everything else is said again on the next pass.
        self.state_path = state_path
        self.pinned_by_agent: set[str] = self._remembered()
        self._task: Optional[asyncio.Task[None]] = None

    def _remembered(self) -> set[str]:
        if self.state_path is None:
            return set()
        try:
            pins = json.loads(self.state_path.read_text()).get("pinned_by_agent", [])
        except (OSError, ValueError, AttributeError):
            return set()  # unreadable: forget, and err toward leaving models loaded
        return {tag for tag in pins if valid_tag(tag)}

    def _remember(self) -> None:
        if self.state_path is None:
            return
        draft = self.state_path.with_name(self.state_path.name + ".new")
        descriptor = os.open(draft, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump({"pinned_by_agent": sorted(self.pinned_by_agent)}, handle)
        os.replace(draft, self.state_path)

    # --- what the pool asks ---

    async def apply(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise Refused(400, "bad_request", "the desired state must be an object")
        tags, residency = body.get("tags"), body.get("residency")
        if not isinstance(tags, list) or not all(valid_tag(tag) for tag in tags):
            raise Refused(400, "bad_tag", "every tag must be a plain model tag: no URL, no path, no scheme")
        if residency not in ("pinned", "on_demand"):
            raise Refused(400, "bad_residency", "residency must be pinned or on_demand")
        self.desired = Desired(tuple(sorted(set(tags))), residency)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._work(), name="gpm-agent:models")
        return await self.snapshot()

    async def delete(self, body: Any) -> dict[str, Any]:
        tag = body.get("tag") if isinstance(body, dict) else None
        if not valid_tag(tag):
            raise Refused(400, "bad_tag", "name one plain model tag to delete")
        if not self.allow_delete:
            raise Refused(403, "owner_forbids_delete", "this machine's owner has set allow_delete: false; the pool cannot change that")
        if tag in self.desired.tags:
            raise Refused(409, "model_required", f"{tag} is in the model set the pool requires here; remove it from the pool first")
        if self.pulling and self.pulling["tag"] == tag:
            raise Refused(409, "model_busy", f"{tag} is being pulled right now")
        state = await self.engine.describe(self.client)
        if tag not in {m["tag"] for m in state.get("models_on_disk", [])}:
            raise Refused(404, "not_on_disk", f"{tag} is not on this machine")
        try:
            await self.engine.delete(self.client, tag)
        except (EngineRefused, httpx.HTTPError) as exc:
            raise Refused(502, "engine_refused", str(exc)) from exc
        self.pinned_by_agent.discard(tag)
        self._remember()
        return await self.snapshot()

    async def snapshot(self) -> dict[str, Any]:
        state = await self.engine.describe(self.client)
        on_disk = {m["tag"]: m.get("size_bytes") for m in state.get("models_on_disk", [])}
        loaded = set(state.get("models_loaded", []))
        return {
            "engine_answers": bool(state.get("answers")),
            "residency": self.desired.residency,
            "busy": self._task is not None and not self._task.done(),
            "free_disk_bytes": self.free_disk_bytes(),
            "min_free_disk_bytes": self.floor_bytes,
            "models": [
                {
                    "tag": tag,
                    "on_disk": tag in on_disk,
                    "size_bytes": on_disk.get(tag),
                    "loaded": tag in loaded,
                    "pinned_by_agent": tag in self.pinned_by_agent,
                    "pulling": dict(self.pulling) if self.pulling and self.pulling["tag"] == tag else None,
                    "error": self.errors[tag][1] if tag in self.errors else None,
                }
                for tag in self.desired.tags
            ],
            # On disk, named by no pool configuration this agent has been sent. Listed, never
            # touched: deleting one is an operator's explicit act.
            "surplus": [
                {"tag": tag, "size_bytes": size, "loaded": tag in loaded}
                for tag, size in sorted(on_disk.items())
                if tag not in self.desired.tags
            ],
        }

    # --- working toward it ---

    async def _work(self) -> None:
        state = await self.engine.describe(self.client)
        if not state.get("answers"):
            return
        on_disk = {m["tag"] for m in state["models_on_disk"]}
        loaded = set(state["models_loaded"])

        for tag in self.desired.tags:
            if tag not in on_disk and self._may_retry(tag):
                if not await self._pull(tag):
                    continue
                on_disk.add(tag)
            # Loaded the moment its own download finishes, while the rest are still coming
            # (D57). The alternative — pull everything, then load everything — leaves the
            # accelerator idle for the whole of the last phase, on a host that is billing.
            if (
                self.desired.residency == "pinned"
                and tag in on_disk
                and not (tag in loaded and tag in self.pinned_by_agent)
                and self._may_retry(tag)
            ):
                await self._hold(tag, pinned=True)
        # A pin does not outlive the engine: a remembered tag that is no longer loaded was
        # unpinned by an engine restart, and is not this agent's to release any more.
        stale = {tag for tag in self.pinned_by_agent if tag not in loaded and tag not in self.desired.tags}
        if stale:
            self.pinned_by_agent -= stale
            self._remember()
        # Release what this agent pinned and should no longer hold: everything under
        # on_demand, and anything that left the set. Only if loaded — releasing loads.
        for tag in sorted(self.pinned_by_agent):
            if self.desired.residency == "on_demand" or tag not in self.desired.tags:
                if tag in loaded:
                    await self._hold(tag, pinned=False)
                self.pinned_by_agent.discard(tag)
                self._remember()

    def _may_retry(self, tag: str) -> bool:
        failed = self.errors.get(tag)
        return failed is None or time.monotonic() - failed[0] >= RETRY_AFTER_S

    def _fail(self, tag: str, why: str) -> None:
        self.errors[tag] = (time.monotonic(), why)

    async def _pull(self, tag: str) -> bool:
        free = self.free_disk_bytes()
        if free is not None and free < self.floor_bytes:
            self._fail(tag, f"not pulled: {free / 1e9:.1f} GB free is already under this machine's {self.floor_bytes / 1e9:.0f} GB floor")
            return False
        self.pulling = {"tag": tag, "completed_bytes": 0, "total_bytes": None}
        try:
            async for completed, total in self.engine.pull(self.client, tag):
                self.pulling.update(completed_bytes=completed, total_bytes=total)
                free = self.free_disk_bytes()
                if free is not None and free - (total - completed) < self.floor_bytes:
                    # Leaving the stream cancels the pull; nothing is half-installed as a model.
                    raise EngineRefused(
                        f"stopped: finishing this {total / 1e9:.1f} GB pull would leave less than "
                        f"this machine's {self.floor_bytes / 1e9:.0f} GB free-disk floor"
                    )
        except (EngineRefused, httpx.HTTPError, ValueError) as exc:
            self._fail(tag, str(exc) or type(exc).__name__)
            return False
        finally:
            self.pulling = None
        self.errors.pop(tag, None)
        return True

    async def _hold(self, tag: str, *, pinned: bool) -> None:
        try:
            await self.engine.hold(self.client, tag, pinned=pinned)
        except (EngineRefused, httpx.HTTPError) as exc:
            self._fail(tag, str(exc) or type(exc).__name__)
            return
        self.errors.pop(tag, None)
        if pinned:
            self.pinned_by_agent.add(tag)
            self._remember()

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
