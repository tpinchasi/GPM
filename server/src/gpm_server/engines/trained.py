"""Trained models (S15): the engines `onnx` (D179) and `container` (D180).

A trained model — a classifier, a detector, a forecaster — is asked on one path, `POST
/v1/predict`, with `{"model": ..., "inputs": {...}}`, and answers `{"model": ..., "outputs":
{...}}` (D178). Both engines are run by the host's agent, which puts the build on the machine and
starts what serves it on loopback, so — as for whisper.cpp (D148) — the agent says what the
machine holds, and the server answers `GET /health` once it is up.

The router reads a predict request as it reads any JSON body: the model field, once, and nothing
else. What `inputs` holds is the model's, as its build declares it.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Optional

import httpx

from . import openai_api
from .base import PREDICT, EngineOption, Health, Occupancy, PullResult, RequestFacts

#: The one path a trained model is asked on (D178).
PREDICT_PATH = "/v1/predict"
PREDICT_PATHS = frozenset({PREDICT_PATH})
#: The fields the pool reads, each held to D141's once-and-in-one-spelling rule.
_READ = frozenset({"model", "inputs"})


def predict_facts(body: bytes) -> RequestFacts:
    found = openai_api.members(body, keep_written=True)
    if found is None:
        return RequestFacts(model=None, kind=PREDICT)
    return RequestFacts(model=openai_api.model_of(found, _READ), kind=PREDICT)


class TrainedModelEngine:
    """What the two engines share: the path, the reader, and a server the agent runs."""

    interface_version: ClassVar[str] = "2"
    name: ClassVar[str] = ""
    kinds: ClassVar[frozenset[str]] = frozenset({PREDICT})
    #: One server on the machine holds every build of this engine the machine is told to hold.
    serves_one_model: ClassVar[bool] = False
    models_dir: ClassVar[str] = ""
    image_words: ClassVar[tuple[str, ...]] = ()
    loads_by_restart: ClassVar[bool] = False
    splits_across_cards: ClassVar[bool] = False
    options: ClassVar[dict[str, EngineOption]] = {}
    #: The server cannot be trusted to say what it holds: the agent that started it says.
    residency_from_agent: ClassVar[bool] = True
    max_workers: ClassVar[Optional[int]] = None
    build_tag: ClassVar[re.Pattern[str]]
    #: What a build of this engine is called, said when the configuration names another.
    build_tag_hint: ClassVar[str] = ""

    def default_start_command(
        self, *, port: int, models_dir: str, agent_archive: str, proxy: bool,
        options: tuple[str, ...] = (),
    ) -> Optional[str]:
        return None  # started by the host's agent, never by a start-up script

    # --- request path ---

    def inference_paths(self) -> set[str]:
        return set(PREDICT_PATHS)

    def kind_of(self, path: str) -> Optional[str]:
        return PREDICT if path in PREDICT_PATHS else None

    def request_facts(self, path: str, content_type: str, body: bytes) -> RequestFacts:
        return predict_facts(body)

    def rewrite_model(self, path: str, content_type: str, body: bytes, tag: str) -> tuple[str, bytes]:
        return content_type, openai_api.with_model(body, tag)

    def requested_model(self, path: str, body: bytes) -> Optional[str]:
        return predict_facts(body).model

    def with_model(self, path: str, body: bytes, tag: str) -> bytes:
        return openai_api.with_model(body, tag)

    def wants_schema(self, path: str, body: bytes) -> bool:
        return False

    def is_streaming(self, path: str, body: bytes) -> bool:
        return False

    def usage(self, path: str, tail: bytes) -> tuple[Optional[int], Optional[float]]:
        return None, None  # no tokens: measured as a request and its latency (D145)

    def keepalive_frame(self, path: str) -> Optional[bytes]:
        return None

    # --- supervisor ---

    async def health(self, client: httpx.AsyncClient) -> Health:
        try:
            response = await client.get("/health")
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=f"the {self.name} server is not answering: {exc.__class__.__name__}")
        if response.status_code != 200:
            return Health(ok=False, detail=f"the {self.name} server answered {response.status_code}")
        return Health(ok=True)

    async def serving_from_cpu(self, client: httpx.AsyncClient) -> Optional[frozenset[str]]:
        return None

    async def occupancy(self, client: httpx.AsyncClient) -> Optional[Occupancy]:
        return None

    async def models_resident(self, client: httpx.AsyncClient) -> frozenset[str]:
        return frozenset()  # the agent says (`residency_from_agent`)

    async def models_available(self, client: httpx.AsyncClient) -> frozenset[str]:
        return frozenset()

    async def pull(self, client: httpx.AsyncClient, tag: str) -> PullResult:
        return PullResult(tag=tag, ok=False, detail=f"{self.name} builds are put on the machine by its agent")

    async def load_and_pin(self, client: httpx.AsyncClient, tags: list[str]) -> None:
        return None

    def launch_settings(self, workers: int, context: int, n_models: int, **_: Any) -> dict[str, str]:
        return {}


_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9.-])?"


class OnnxEngine(TrainedModelEngine):
    """An ONNX file, run by the pool's own server on ONNX Runtime (D179)."""

    name: ClassVar[str] = "onnx"
    default_port: ClassVar[int] = 8190
    builds_on_hub: ClassVar[bool] = True
    #: A `.onnx` file at the root of a hub repository, or a file uploaded to the pool (D182).
    build_tag: ClassVar[re.Pattern[str]] = re.compile(
        rf"^(?:(?!.*__)(?!.*\.\.){_NAME}/{_NAME}/[A-Za-z0-9][A-Za-z0-9._-]*\.onnx|upload/[0-9a-f]{{64}})$"
    )
    build_tag_hint: ClassVar[str] = (
        "`owner/repo/file.onnx` — a hub repository and one `.onnx` file at its root — or an upload, "
        "`upload/<sha256>` (D179, D182)"
    )


class ContainerEngine(TrainedModelEngine):
    """An operator's container image answering the predict contract (D180)."""

    name: ClassVar[str] = "container"
    default_port: ClassVar[int] = 8191
    builds_on_hub: ClassVar[bool] = False
    #: An image by digest only — a registry path and `@sha256:` — never a tag that can move.
    build_tag: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?!.*\.\.)[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?(?::[0-9]{1,5})?"
        r"(?:/[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?)+@sha256:[0-9a-f]{64}$"
    )
    build_tag_hint: ClassVar[str] = (
        "an image by digest, `registry.example/owner/image@sha256:<64 hex>` — a tag alone can move (D180)"
    )
