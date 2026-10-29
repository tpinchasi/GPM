"""The whisper.cpp engine adapter (D148): Whisper on a machine's GPU, Apple's included.

whisper.cpp's server answers OpenAI's transcription path when started with
`--inference-path /v1/audio/transcriptions`, and nothing else the pool forwards. One server runs one
model, given at start, and transcribes one recording at a time. It cannot say which model it holds
and has no authentication, so the host carries an agent: the agent fetches the build, starts the
server on loopback with it, and reports what it is serving.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Optional

import httpx

from . import openai_api
from .base import TRANSCRIBE, EngineOption, Health, Occupancy, PullResult, RequestFacts


class WhisperCppEngine:
    interface_version: ClassVar[str] = "2"
    name: ClassVar[str] = "whisper-cpp"
    kinds: ClassVar[frozenset[str]] = frozenset({TRANSCRIBE})
    serves_one_model: ClassVar[bool] = True
    default_port: ClassVar[int] = 8178
    models_dir: ClassVar[str] = ""
    image_words: ClassVar[tuple[str, ...]] = ("whisper",)
    #: The agent starts the server with each new build itself, when it is asked to hold it; the
    #: pool never asks for a restart (D148).
    loads_by_restart: ClassVar[bool] = False
    splits_across_cards: ClassVar[bool] = False
    options: ClassVar[dict[str, EngineOption]] = {}
    #: Its builds are files in model-hub repositories (`owner/repo/file.bin`).
    builds_on_hub: ClassVar[bool] = True
    #: The server cannot say what it holds; the host's agent, which started it, says (D148).
    residency_from_agent: ClassVar[bool] = True
    #: The server holds a global lock: one recording at a time, so one worker.
    max_workers: ClassVar[Optional[int]] = 1
    #: What one of its builds is called: one `.bin` file at the root of a hub repository,
    #: `owner/repo/file.bin` — as the host's agent reads it, so a tag it would refuse is refused
    #: when the configuration loads instead (D148).
    build_tag: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?!.*__)(?!.*\.\.)"
        r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9.-])?/"   # the owner, not ending in `_`
        r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9.-])?/"   # the repository, likewise
        r"[A-Za-z0-9][A-Za-z0-9._-]*\.bin$"
    )

    def default_start_command(
        self, *, port: int, models_dir: str, agent_archive: str, proxy: bool,
        options: tuple[str, ...] = (),
    ) -> Optional[str]:
        return None  # started by the host's agent, never by a start-up script

    # --- request path ---

    #: Its server answers transcription only: it is started with that one path, and translation
    #: is vLLM's (D142).
    PATHS: ClassVar[frozenset[str]] = frozenset({"/v1/audio/transcriptions"})

    def inference_paths(self) -> set[str]:
        return set(self.PATHS)

    def kind_of(self, path: str) -> Optional[str]:
        return TRANSCRIBE if path in self.PATHS else None

    def request_facts(self, path: str, content_type: str, body: bytes) -> RequestFacts:
        return openai_api.upload_facts(content_type, body)

    def rewrite_model(self, path: str, content_type: str, body: bytes, tag: str) -> tuple[str, bytes]:
        return openai_api.rewrite_model(path, content_type, body, tag)

    # Version 1's readers need the content type an upload is framed by, which they are not
    # given; the router uses the version-2 readers above.
    def requested_model(self, path: str, body: bytes) -> Optional[str]:
        return None

    def with_model(self, path: str, body: bytes, tag: str) -> bytes:
        return body

    def wants_schema(self, path: str, body: bytes) -> bool:
        return False

    def is_streaming(self, path: str, body: bytes) -> bool:
        return False

    def usage(self, path: str, tail: bytes) -> tuple[Optional[int], Optional[float]]:
        return None, None  # no output tokens: measured as a request and its latency (D145)

    def keepalive_frame(self, path: str) -> Optional[bytes]:
        return None

    # --- supervisor ---

    async def health(self, client: httpx.AsyncClient) -> Health:
        try:
            response = await client.get("/")
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=f"whisper-server not answering: {exc.__class__.__name__}")
        if response.status_code != 200:
            return Health(ok=False, detail=f"whisper-server answered {response.status_code}")
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
        return PullResult(tag=tag, ok=False, detail="whisper.cpp builds are fetched by the host's agent")

    async def load_and_pin(self, client: httpx.AsyncClient, tags: list[str]) -> None:
        return None

    def launch_settings(self, workers: int, context: int, n_models: int, **_: Any) -> dict[str, str]:
        return {}
