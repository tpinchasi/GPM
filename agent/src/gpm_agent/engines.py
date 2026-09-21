"""What the engine on this machine holds — asked of the engine itself.

Kept separate from the pool's engine adapters on purpose: the agent must install on a host
without the pool's server package, so it carries the little it needs. One class per engine;
the agent's settings name which.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx


class OllamaFacts:
    name = "ollama"

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

    def launch_environment(self, workers: int, models_held: int, context: Optional[int] = None) -> dict[str, str]:
        """The engine's own names for the pool's numbers. Built here, from integers — the pool
        never sends a variable name or a value as text."""
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


class EngineRefused(Exception):
    """The engine said no. The message is the engine's, passed on to the operator."""


_ENGINES = {"ollama": OllamaFacts}


def engine_facts(name: str) -> Optional[OllamaFacts]:
    factory = _ENGINES.get(name)
    return factory() if factory else None
