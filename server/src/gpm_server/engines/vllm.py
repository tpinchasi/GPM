"""The vLLM engine adapter — a continuous-batching server behind the same interface (D90).

What is different about this engine, and why the adapter looks the way it does:

- **It speaks the OpenAI-shaped API and nothing else**, so every request-path rule comes from
  `openai_api` rather than being written again here.
- **It cannot fetch a model over HTTP.** There is no pull endpoint; weights arrive from a model
  hub before the server starts. So `pull` here refuses, with the reason, and a vLLM host is
  prepared through the pool's agent, which fetches them (docs/spec/host-agent.md §4).
- **It is launched with its model, and holds it for the process's life.** There is nothing to
  load or evict, so `load_and_pin` verifies rather than acts.
- **It can say how full it is**, which the pool's worker slots cannot express for a batching
  engine (D91).
"""

from __future__ import annotations

import re
import shlex
from typing import ClassVar, Optional

import httpx

from . import openai_api
from .base import EngineOption, Health, Occupancy, PullResult

#: What the pool's integers are called in this engine's start-up environment. The engine takes
#: command-line flags, not variables, so a host that runs it reads these in its own start
#: command — the pool writes numbers into an environment file and never a command (D41).
WORKERS = "GPM_VLLM_MAX_NUM_SEQS"
BATCHED_TOKENS = "GPM_VLLM_MAX_NUM_BATCHED_TOKENS"
CONTEXT = "GPM_VLLM_MAX_MODEL_LEN"
LISTEN = "GPM_VLLM_LISTEN"

#: Scheduling headroom: a batch step may hold this many tokens for each request in flight.
#: Below the engine's own floor (`max_num_batched_tokens` must be at least `max_num_seqs`) it
#: refuses to start, so the multiplier is never less than one.
TOKENS_PER_SEQUENCE = 256

_METRIC = re.compile(r"^(?P<name>vllm:[a-z_]+)(?:\{[^}]*\})?\s+(?P<value>[0-9.eE+-]+)\s*$", re.M)


class VllmEngine:
    interface_version: ClassVar[str] = "1"
    name: ClassVar[str] = "vllm"
    #: One process, one model. Several models on one machine means several processes, which
    #: means something in front of them choosing between ports — and that is a component the
    #: pool does not yet ship, so the combination is refused at load rather than at 3 a.m.
    serves_one_model: ClassVar[bool] = True
    default_port: ClassVar[int] = 8000
    image_words: ClassVar[tuple[str, ...]] = ("vllm",)
    #: It serves what it was started with; a downloaded model is picked up by starting it again.
    loads_by_restart: ClassVar[bool] = True
    #: Its builds are model-hub repositories, which the pool can look up (D100).
    builds_on_hub: ClassVar[bool] = True
    #: What the agent's launcher can switch on, and for which model families (`model_type` in a
    #: model's own configuration). The flags live in the launcher, on the machine; these are the
    #: same names and families, for the operator to choose from (D100).
    options: ClassVar[dict[str, EngineOption]] = {
        "tool_calling": EngineOption(
            label="tool calling — apps may send `tools` and get `tool_calls` back",
            families=("gemma4", "gpt_oss", "qwen2", "qwen3", "qwen3_moe"),
        ),
        "reasoning": EngineOption(
            label="reasoning — a thinking model's reasoning returned apart from its answer",
            families=("gemma4", "qwen3", "qwen3_moe"),
        ),
    }

    def default_start_command(
        self, *, port: int, models_dir: str, agent_archive: str, proxy: bool,
        options: tuple[str, ...] = (),
    ) -> Optional[str]:
        """The agent's own launcher, from its archive (D97).

        At boot neither the archive nor any model is on the machine yet — the agent is pushed
        later and fetches afterwards — so this does nothing then, successfully. The pool calls
        the same script again once the models are on disk, and it starts vLLM for them: one
        process, or one per model behind the router when `proxy` is set.
        """
        command = (
            f"python3 {shlex.quote(agent_archive)} vllm-start "
            f"--models-dir {shlex.quote(models_dir)} --port {int(port)}"
        )
        if proxy:
            command += " --proxy"
        for option in options:
            if option not in self.options:
                raise ValueError(f"vllm has no option {option!r}")
            command += f" --option {shlex.quote(option)}"
        return f"if [ -f {shlex.quote(agent_archive)} ]; then {command}; fi"

    # --- request path: the shared protocol, not this engine's invention ---

    def inference_paths(self) -> set[str]:
        return set(openai_api.INFERENCE_PATHS)

    def requested_model(self, path: str, body: bytes) -> Optional[str]:
        model = openai_api.decode(body).get("model")
        return model if isinstance(model, str) else None

    def with_model(self, path: str, body: bytes, tag: str) -> bytes:
        """Splice the new name into the raw bytes, leaving every other byte untouched — the
        same rule and the same regular expression as every other adapter, because passthrough
        fidelity is a property of the pool, not of an engine."""
        import json

        from .ollama import _MODEL_FIELD  # one expression, deliberately not copied

        rewritten, count = _MODEL_FIELD.subn(rb"\g<1>" + json.dumps(tag).encode(), body, count=1)
        return rewritten if count else body

    def wants_schema(self, path: str, body: bytes) -> bool:
        return openai_api.wants_schema(path, body)

    def is_streaming(self, path: str, body: bytes) -> bool:
        return openai_api.is_streaming(path, body)

    def usage(self, path: str, tail: bytes) -> tuple[Optional[int], Optional[float]]:
        return openai_api.usage(path, tail)

    def keepalive_frame(self, path: str) -> Optional[bytes]:
        return openai_api.keepalive_frame(path)

    # --- probe path ---

    async def health(self, client: httpx.AsyncClient) -> Health:
        """This engine answers `/health` with an empty 200 once it is serving — and not before.
        Start-up compiles graphs and loads weights, which on a large model is minutes, so a
        host that is not answering yet is not necessarily a host that is broken."""
        try:
            response = await client.get("/health")
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=str(exc) or type(exc).__name__)
        if response.status_code != 200:
            return Health(ok=False, detail=f"/health returned {response.status_code}")
        return Health(ok=True)

    async def occupancy(self, client: httpx.AsyncClient) -> Optional[Occupancy]:
        """What the engine is holding, read from the metrics it publishes (D91).

        `num_requests_waiting` is the number the pool cannot get any other way: requests this
        engine has accepted and not started. Without it a host admitted at a hundred workers
        looks idle at ninety-nine in flight, and every decision that rests on "busy" — ranking,
        scaling, tear-down — reads the wrong signal.
        """
        try:
            response = await client.get("/metrics")
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        found = {m.group("name"): m.group("value") for m in _METRIC.finditer(response.text)}

        def number(name: str) -> Optional[float]:
            try:
                return float(found[name])
            except (KeyError, ValueError):
                return None

        running, waiting = number("vllm:num_requests_running"), number("vllm:num_requests_waiting")
        if running is None or waiting is None:
            return None
        # Named for the cache in both current and older builds; whichever is present wins.
        cache = number("vllm:kv_cache_usage_perc")
        if cache is None:
            cache = number("vllm:gpu_cache_usage_perc")
        return Occupancy(running=int(running), waiting=int(waiting), cache_used=cache)

    async def serving_from_cpu(self, client: httpx.AsyncClient) -> Optional[frozenset[str]]:
        """None: this engine reports no per-model placement to ask. The driver floor and the
        per-machine choice of build (D81, D92) are what keep it off a card it cannot use."""
        return None

    async def models_resident(self, client: httpx.AsyncClient) -> frozenset[str]:
        """Everything this engine serves is resident: it is launched with its model and holds it
        for the life of the process. There is no on-demand load and nothing is ever evicted, so
        "resident" and "served" are the same set."""
        return await self._served(client)

    async def models_available(self, client: httpx.AsyncClient) -> frozenset[str]:
        """The same set, for the same reason: weights on disk this engine was not launched with
        are not servable without a restart, so counting them would tell the pool it can serve
        something it cannot."""
        return await self._served(client)

    async def _served(self, client: httpx.AsyncClient) -> frozenset[str]:
        response = await client.get("/v1/models")
        response.raise_for_status()
        entries = response.json().get("data", [])
        return frozenset(
            entry["id"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        )

    # --- preparing a host the pool created ---

    async def pull(self, client: httpx.AsyncClient, tag: str, on_progress=None) -> PullResult:
        """This engine has no pull: weights come from a model hub before it starts, not over its
        own API. Saying so plainly — and never retryably — is the whole of the implementation.

        On hosts the pool creates, the agent fetches them instead; a host without an agent is
        prepared by its owner. Reporting this as a transient failure would have the supervisor
        retry a download that cannot happen, and give up on the host for the wrong reason.
        """
        return PullResult(
            tag=tag,
            ok=False,
            detail=(
                "this engine cannot fetch models over its own API; a vLLM host is prepared "
                "through the pool's agent, or by its owner before the engine starts"
            ),
            retryable=False,
        )

    async def load_and_pin(self, client: httpx.AsyncClient, tags: list[str]) -> None:
        """Verify, rather than act: this engine holds what it was launched with, and a tag it
        was not launched with cannot be made resident without restarting it."""
        served = await self._served(client)
        missing = set(tags) - served
        if missing:
            raise RuntimeError(
                f"{sorted(missing)} is not served by this engine; it holds {sorted(served) or 'nothing'}. "
                "A vLLM process serves the model it was started with — the pool's model set is "
                "spread across hosts, not held on each one"
            )

    def launch_settings(
        self, workers: int, context: int, n_models: int, listen: Optional[str] = None
    ) -> dict[str, str]:
        """Numbers for the host's own start command to read (D41).

        `n_models` is not passed on: this engine serves exactly one model per process, so the
        pool's count of the set would be a number it could not honour. What holds the set
        together is the pool spreading it across hosts, not this engine holding all of it.
        """
        settings = {
            WORKERS: str(workers),
            CONTEXT: str(context),
            # The engine refuses to start with a batch smaller than the number of sequences it
            # is told to run, so this is a floor before it is a tuning knob.
            BATCHED_TOKENS: str(max(workers, workers * TOKENS_PER_SEQUENCE)),
        }
        if listen:
            # The same reasoning as every other engine the pool launches: it is reached through
            # a forward into the machine, so binding anything wider only exposes it (D77).
            settings[LISTEN] = listen
        return settings
