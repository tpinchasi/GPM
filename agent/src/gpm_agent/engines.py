"""What the engine on this machine holds — asked of the engine itself.

Kept separate from the pool's engine adapters on purpose: the agent must install on a host
without the pool's server package, so it carries the little it needs. One class per engine;
the agent's settings name which.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx

from . import modelhub, vllm_state


class OllamaFacts:
    name = "ollama"
    #: This engine fetches through its own API, so there is nothing to fetch with until it runs.
    fetches_without_engine = False
    #: A model is loaded by asking the running engine to load it.
    loads_by_restart = False
    #: Ollama places a model on the machine's cards itself; it cannot be told to split one (D114).
    splits_across_cards = False

    def __init__(self, settings: Any = None) -> None:
        #: Unused here — this engine answers every question about itself over its own API.
        #: Accepted so that every engine in the registry is built the same way.
        self.settings = settings

    async def describe(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Version, models on disk with sizes, models loaded. An engine that does not answer
        is reported as not answering — with nothing else claimed about it."""
        try:
            version = (await client.get("/api/version")).raise_for_status().json().get("version")
            on_disk = (await client.get("/api/tags")).raise_for_status().json().get("models", [])
            loaded = (await client.get("/api/ps")).raise_for_status().json().get("models", [])
        except (httpx.HTTPError, ValueError) as exc:
            return {"name": self.name, "answers": False, "detail": str(exc) or type(exc).__name__}
        return {
            "name": self.name,
            "answers": True,
            "version": version,
            "models_on_disk": sorted(
                ({"tag": m.get("name"), "size_bytes": m.get("size")} for m in on_disk if m.get("name")),
                key=lambda m: m["tag"],
            ),
            "models_loaded": sorted(m.get("name") for m in loaded if m.get("name")),
        }

    def launch_environment(
        self, workers: int, models_held: int, context: Optional[int] = None, cards_per_copy: int = 1
    ) -> dict[str, str]:
        """The engine's own names for the pool's numbers. Built here, from integers — the pool
        never sends a variable name or a value as text."""
        # `cards_per_copy` is never more than one here: the agent refuses it first (D114).
        environment = {"OLLAMA_NUM_PARALLEL": str(workers), "OLLAMA_MAX_LOADED_MODELS": str(models_held)}
        if context is not None:
            environment["OLLAMA_CONTEXT_LENGTH"] = str(context)
        return environment

    def settings_from_environment(self, environment: Optional[dict[str, str]]) -> Optional[dict[str, Optional[int]]]:
        """The same numbers read back, so the pool compares like with like and never has to
        know this engine's variable names."""
        if not environment:
            return None

        def number(name: str) -> Optional[int]:
            value = environment.get(name, "")
            return int(value) if value.isdigit() else None

        return {
            "workers": number("OLLAMA_NUM_PARALLEL"),
            "models_held": number("OLLAMA_MAX_LOADED_MODELS"),
            "context": number("OLLAMA_CONTEXT_LENGTH"),
        }

    # --- the three things the agent may do to a model (docs/spec/host-agent.md §4) ---

    async def pull(self, client: httpx.AsyncClient, tag: str) -> AsyncIterator[tuple[int, int]]:
        """Fetch one tag, yielding (bytes completed, bytes total) as layers arrive. Raises
        `EngineRefused` with the engine's own words if it will not."""
        layers: dict[str, tuple[int, int]] = {}
        async with client.stream("POST", "/api/pull", json={"model": tag, "stream": True}, timeout=None) as response:
            if response.status_code != 200:
                raise EngineRefused(f"pull returned {response.status_code}")
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("error"):
                    raise EngineRefused(str(frame["error"]))
                if frame.get("total"):
                    layers[frame.get("digest") or frame.get("status", "")] = (
                        int(frame.get("completed") or 0), int(frame["total"]),
                    )
                    yield sum(done for done, _ in layers.values()), sum(size for _, size in layers.values())

    async def hold(self, client: httpx.AsyncClient, tag: str, *, pinned: bool) -> None:
        """Pinned: load it and keep it loaded. Not pinned: hand its lifetime back to the engine.
        Either form *loads* the model if it is not loaded, so the caller releases only what is
        loaded now — releasing a cold model would do the opposite of what was meant."""
        keep_alive: Any = -1 if pinned else "5m"
        body = {"model": tag, "keep_alive": keep_alive}
        # An embedding model refuses `generate` with a 400, so it is held through the endpoint
        # it does serve. Asking the engine which kind a tag is, is an **optimisation, never a
        # requirement** (D84): an engine busy downloading the next model may take its time over
        # a metadata call, and a host must not be condemned for that. Found live — this call
        # inherited the client's ten-second default, timed out while an 18.6GB download ran
        # beside it, and the pool destroyed a healthy host for "could not hold the model set".
        for path in self._paths_to_try(await self._kind_of(client, tag)):
            # A **deadline**, not `timeout=None`. Seen live: a machine whose engine had fallen
            # back to the processor took a 26B model past sixteen minutes, the agent waited in
            # this call the whole time, and because the loop is one tag after another the rest
            # of the model set never started. It answered every poll cheerfully while nothing
            # moved. A load that outlasts this is reported as an error the pool can act on.
            response = await client.post(path, json=body, timeout=LOAD_TIMEOUT_S)
            if response.status_code == 200:
                return
            # 400 is this engine's way of saying "wrong endpoint for this kind of model"; any
            # other refusal is about the model itself and trying elsewhere would only hide it.
            if response.status_code != 400:
                break
        raise EngineRefused(
            f"the engine would not {'pin' if pinned else 'release'} {tag}: {response.status_code}"
        )

    async def _kind_of(self, client: httpx.AsyncClient, tag: str) -> Optional[str]:
        """"embedding", "completion", or **None when the engine did not say in time**.

        Never fatal: the caller tries both endpoints when it does not know (D84).
        """
        try:
            shown = await client.post("/api/show", json={"model": tag}, timeout=SHOW_TIMEOUT_S)
        except httpx.HTTPError:
            return None
        if shown.status_code != 200:
            return None
        try:
            capabilities = shown.json().get("capabilities") or []
        except ValueError:
            return None
        return "embedding" if "embedding" in capabilities else "completion"

    @staticmethod
    def _paths_to_try(kind: Optional[str]) -> tuple[str, ...]:
        """What to load through, most likely first. Unknown means try both rather than guess."""
        if kind == "embedding":
            return ("/api/embed",)
        if kind == "completion":
            return ("/api/generate",)
        return ("/api/generate", "/api/embed")

    async def delete(self, client: httpx.AsyncClient, tag: str) -> None:
        response = await client.request("DELETE", "/api/delete", json={"model": tag})
        if response.status_code != 200:
            raise EngineRefused(f"the engine would not delete {tag}: {response.status_code}")


#: The longest a single model may take to load before it is called a failure. Generous — a
#: large model on a healthy accelerator is a matter of a minute or two — and finite, which is
#: the point: without it one stuck load stalls a host's whole preparation silently.
LOAD_TIMEOUT_S = 600.0

#: Asking what kind a model is. Generous, because the engine may be saturated downloading the
#: next model at the time (D83 put a load beside a download); and not fatal when it expires.
SHOW_TIMEOUT_S = 120.0

#: vLLM's batch, from the pool's worker count: the same two numbers the pool's own start
#: command uses, kept here in the same words because the restart the pool asks for writes them
#: from here. Below the engine's own default the pool's figure could only ever hurt (found
#: live: 1,536 tokens for six workers, and a multimodal model that refused to start on it).
TOKENS_PER_SEQUENCE = 256
MIN_BATCHED_TOKENS = 8192


class EngineRefused(Exception):
    """The engine said no. The message is the engine's, passed on to the operator."""


class VllmFacts:
    """vLLM, which differs from the first engine in three ways the agent has to know about.

    **Weights do not arrive through the engine.** There is no pull endpoint; the engine is
    started with a directory and serves what is in it. So a fetch here is an HTTP transfer from
    a model hub into the machine's models directory, which is what `modelhub` does.

    **A model becomes servable only when the engine restarts.** Downloading is not loading.
    `hold` therefore reports honestly whether the engine is serving the tag, and says what is
    missing when it is not, rather than pretending a fetched model is a held one.

    **The pool's text never reaches the start command.** The engine needs to know which model to
    serve, and that name came from the pool — so it is never written into the engine's
    environment. The host's own start command reads the models directory instead and serves
    what it finds, taking the served name from the directory the agent created (D41 stands:
    the pool sends numbers, the owner's command supplies everything else).
    """

    name = "vllm"
    #: Weights come from a hub, not through the engine — and the engine cannot run until they
    #: are here. An agent that waited for this engine to answer before fetching would wait
    #: forever on a machine that has just booted (D97).
    fetches_without_engine = True
    #: A model is "loaded" by starting the engine again with it on disk. The agent fetches and
    #: reports; the pool, seeing a model downloaded and not yet served, asks for the restart.
    loads_by_restart = True
    #: A model may be split across a group of cards (tensor parallelism, D114); the launcher
    #: reads how many from the environment written here.
    splits_across_cards = True

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings

    @property
    def _models_dir(self) -> str:
        return getattr(self.settings, "models_path", None) or "~/gpm-models"

    async def describe(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """What this engine is serving, and what is on disk waiting for a restart.

        The two are different sets here, and reporting them as one would hide the state a vLLM
        host spends its whole preparation in: weights present, engine not yet serving them.
        """
        # What is on disk is known whether or not the engine runs — and on a machine that has
        # just booted it does not, because it has nothing to serve yet. Reporting nothing then
        # would hide the one fact that decides what happens next.
        on_disk = self._on_disk()
        try:
            version = (await client.get("/version")).json().get("version")
        except (httpx.HTTPError, ValueError):
            version = None
        try:
            served = (await client.get("/v1/models")).raise_for_status().json().get("data", [])
        except (httpx.HTTPError, ValueError) as exc:
            return {
                "name": self.name, "answers": False, "detail": str(exc) or type(exc).__name__,
                "models_on_disk": on_disk, "models_loaded": [], "models_failed": self._failed([]),
            }
        loaded = sorted(entry["id"] for entry in served if isinstance(entry, dict) and entry.get("id"))
        return {
            "name": self.name,
            "answers": True,
            "version": version,
            "models_on_disk": on_disk,
            "models_loaded": loaded,
            # A process the launcher started that has since exited without serving its model
            # is a failure, not a model still loading — with the reason from its own log.
            "models_failed": self._failed(loaded),
        }

    def _failed(self, loaded: list[str]) -> dict[str, str]:
        return vllm_state.failed_engines(Path(self._models_dir).expanduser(), loaded)

    def _on_disk(self) -> list[dict[str, Any]]:
        root = Path(self._models_dir).expanduser()
        found = []
        if not root.is_dir():
            return found
        for directory in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if not modelhub.is_complete(directory):
                continue  # a download (or a copy) in progress is not a model on disk
            # The model's own files; the fetch's bookkeeping is not part of what was downloaded.
            size = sum(
                f.stat().st_size for f in directory.rglob("*")
                if f.is_file() and not f.name.startswith(".")
            )
            found.append({"tag": directory.name.replace("__", "/", 1), "size_bytes": size})
        return found

    async def pull(self, client: httpx.AsyncClient, tag: str) -> AsyncIterator[tuple[int, int]]:
        """Fetch from the model hub, not from the engine — so the engine's client is unused."""
        try:
            async for completed, total in modelhub.fetch(tag, self._models_dir):
                yield completed, total
        except modelhub.HubRefused as no:
            raise EngineRefused(str(no)) from no

    async def hold(self, client: httpx.AsyncClient, tag: str, *, pinned: bool) -> None:
        """Serving a model here is a property of how the engine was started, not something that
        can be asked of it while it runs. So this reports rather than acts: a tag the engine
        serves is held — permanently, by construction, which is what the pool wanted — and a
        tag it does not serve needs the engine restarted, which only an operator causes (D41).
        """
        if not pinned:
            # Nothing to release: this engine holds one model for the life of the process, and
            # handing its lifetime back is not something it offers.
            return
        try:
            served = (await client.get("/v1/models")).raise_for_status().json().get("data", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise EngineRefused(f"the engine did not say what it serves: {exc}") from exc
        if tag in {e.get("id") for e in served if isinstance(e, dict)}:
            return
        on_disk = {m["tag"] for m in self._on_disk()}
        raise EngineRefused(
            f"the engine is not serving {tag}: it serves "
            f"{sorted(e.get('id') for e in served if isinstance(e, dict)) or 'nothing'}. "
            + (
                "The weights are on disk; this engine picks up a model only when it is restarted."
                if tag in on_disk
                else "The weights are not on disk either."
            )
        )

    async def delete(self, client: httpx.AsyncClient, tag: str) -> None:
        """Remove the weights from disk. The engine is not asked: it has no delete, and a model
        it is currently serving is held open by the process until that process restarts."""
        try:
            directory = modelhub.directory_for(self._models_dir, tag)
        except modelhub.HubRefused as no:
            raise EngineRefused(str(no)) from no
        if not directory.is_dir():
            return
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            raise EngineRefused(f"could not remove {tag} from disk: {exc}") from exc

    def launch_environment(
        self, workers: int, models_held: int, context: Optional[int] = None, cards_per_copy: int = 1
    ) -> dict[str, str]:
        """The pool's numbers under this engine's names — numbers only, as always.

        `models_held` is deliberately not passed on: one process serves one model here, so a
        count of the pool's whole set is a number this engine could not honour.
        """
        environment = {
            "GPM_VLLM_MAX_NUM_SEQS": str(workers),
            # Never below the engine's own default. The pool's start command has had this floor
            # since a multimodal model refused to start on six workers' 1,536 tokens — but the
            # restart the pool asks for once the weights have landed writes *these* numbers,
            # and without the same floor here it undid the fix (found live: the same refusal,
            # "max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (1536)",
            # on the next host). The engine's own floor is `max_num_seqs`, which is lower.
            "GPM_VLLM_MAX_NUM_BATCHED_TOKENS": str(max(MIN_BATCHED_TOKENS, workers * TOKENS_PER_SEQUENCE)),
        }
        if context is not None:
            environment["GPM_VLLM_MAX_MODEL_LEN"] = str(context)
        if cards_per_copy > 1:
            # Written only when a model is split, so a host that splits nothing keeps exactly the
            # file it had (D107 is the case of one card per copy).
            environment["GPM_VLLM_CARDS_PER_COPY"] = str(cards_per_copy)
        return environment

    def settings_from_environment(
        self, environment: Optional[dict[str, str]]
    ) -> Optional[dict[str, Optional[int]]]:
        if not environment:
            return None

        def number(name: str) -> Optional[int]:
            value = environment.get(name, "")
            return int(value) if value.isdigit() else None

        applied = {
            "workers": number("GPM_VLLM_MAX_NUM_SEQS"),
            # Always one, and said rather than left blank: the pool compares what it asked for
            # with what was applied, and a missing number reads as "not applied yet".
            "models_held": 1,
            "context": number("GPM_VLLM_MAX_MODEL_LEN"),
        }
        split = number("GPM_VLLM_CARDS_PER_COPY")
        if split and split > 1:
            # Only where written, as the pool only asks for it then (D114).
            applied["cards_per_copy"] = split
        return applied


_ENGINES = {"ollama": OllamaFacts, "vllm": VllmFacts}


def engine_facts(name: str, settings: Any = None) -> Optional[Any]:
    factory = _ENGINES.get(name)
    return factory(settings) if factory else None
