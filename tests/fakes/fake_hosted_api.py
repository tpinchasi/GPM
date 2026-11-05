"""A hosted model API as the pool meets one (S11): OpenAI-shaped `/v1`, a Bearer key, token usage
in every answer and, when asked, at the end of a stream — and the refusals providers give.

What it is told (`behaviour`) it does to the next requests; what it was sent it remembers
(`seen`), so a test can say exactly what left the pool for a provider.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

KEY = "test-hosted-key"


class FakeHostedApi:
    def __init__(self, models: tuple[str, ...] = ("provider-llama", "provider-claude")) -> None:
        self.models = models
        self.seen: list[dict[str, Any]] = []
        #: The next answers: "ok", "busy", "out_of_funds", "refused" — the last one repeats.
        self.behaviour: list[str] = ["ok"]
        self.usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cached": 200}
        self.app = Starlette(routes=[
            Route("/v1/models", self._models, methods=["GET"]),
            Route("/v1/chat/completions", self._chat, methods=["POST"]),
        ])

    def _authorised(self, request: Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {KEY}"

    async def _models(self, request: Request) -> Response:
        if not self._authorised(request):
            return JSONResponse({"error": {"message": "bad key"}}, status_code=401)
        return JSONResponse({"object": "list", "data": [{"id": m, "object": "model"} for m in self.models]})

    def _next(self) -> str:
        return self.behaviour.pop(0) if len(self.behaviour) > 1 else self.behaviour[0]

    def _usage(self) -> dict[str, Any]:
        u = self.usage
        return {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                "total_tokens": u["prompt_tokens"] + u["completion_tokens"],
                "prompt_tokens_details": {"cached_tokens": u["cached"]}}

    async def _chat(self, request: Request) -> Response:
        body = await request.body()
        self.seen.append({"headers": dict(request.headers), "body": json.loads(body), "path": request.url.path})
        if not self._authorised(request):
            return JSONResponse({"error": {"message": "bad key"}}, status_code=401)
        what = self._next()
        if what == "busy":
            return JSONResponse({"error": {"message": "Rate limit reached", "type": "rate_limit"}},
                                status_code=429, headers={"retry-after": "1"})
        if what == "out_of_funds":
            return JSONResponse({"error": {"message": "You exceeded your project's spend limit",
                                           "code": "project_spend_limit_exceeded"}}, status_code=429)
        if what == "refused":
            return JSONResponse({"error": {"message": "'temperature' must be 1 for this model",
                                           "type": "invalid_request_error"}}, status_code=400)
        asked = json.loads(body)
        model = asked.get("model")
        if not asked.get("stream"):
            return JSONResponse({
                "id": "chatcmpl-1", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello from the provider"},
                             "finish_reason": "stop"}],
                "usage": self._usage(),
            })
        with_usage = (asked.get("stream_options") or {}).get("include_usage") is True

        async def frames():
            for word in ("hello", " from", " the", " provider"):
                chunk = {"id": "c", "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            if with_usage:
                yield f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'choices': [], 'usage': self._usage()})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")
