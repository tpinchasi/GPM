"""A hosted model API as the pool meets one (S11): OpenAI-shaped `/v1`, a Bearer key, token usage
in every answer and, when asked, at the end of a stream — and the refusals providers give. It
speaks Anthropic's Messages format too (`/v1/messages`, S12): its answers, its stream of typed
events, its usage with cache reads and writes, its refusals.

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
        #: What its `/models` says each costs, as a provider that lists prices does (D176).
        self.pricing: dict[str, dict[str, Any]] = {}
        self.seen: list[dict[str, Any]] = []
        #: The next answers: "ok", "busy", "out_of_funds", "refused" — the last one repeats.
        self.behaviour: list[str] = ["ok"]
        self.usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cached": 200}
        self.app = Starlette(routes=[
            Route("/v1/models", self._models, methods=["GET"]),
            Route("/v1/chat/completions", self._chat, methods=["POST"]),
            Route("/v1/messages", self._messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", self._count, methods=["POST"]),
        ])
        #: Anthropic's counts: input after the last cache mark, read from and written to the cache.
        self.anthropic_usage = {"input_tokens": 100, "cache_read_input_tokens": 700,
                                "cache_creation_input_tokens": 200, "output_tokens": 500}

    def _authorised(self, request: Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {KEY}"

    async def _models(self, request: Request) -> Response:
        if not self._authorised(request):
            return JSONResponse({"error": {"message": "bad key"}}, status_code=401)
        return JSONResponse({"object": "list", "data": [
            {"id": m, "object": "model", **({"pricing": self.pricing[m]} if m in self.pricing else {})} for m in self.models]})

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
        if what == "tool":
            call = {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
            if not asked.get("stream"):
                return JSONResponse({
                    "id": "chatcmpl-1", "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                                 "finish_reason": "tool_calls"}],
                    "usage": self._usage(),
                })

            async def tool_frames():
                first = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "weather", "arguments": ""}}
                for delta in ({"tool_calls": [first]},
                              {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]},
                              {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}):
                    chunk = {"id": "c", "object": "chat.completion.chunk", "model": model,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                end = {"id": "c", "object": "chat.completion.chunk", "model": model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                yield f"data: {json.dumps(end)}\n\n".encode()
                if (asked.get("stream_options") or {}).get("include_usage") is True:
                    yield f"data: {json.dumps({'id': 'c', 'choices': [], 'usage': self._usage()})}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(tool_frames(), media_type="text/event-stream")
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

    # --- Anthropic's Messages format (S12) ---

    @staticmethod
    def _event(kind: str, data: dict[str, Any]) -> bytes:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()

    async def _count(self, request: Request) -> Response:
        body = await request.body()
        self.seen.append({"headers": dict(request.headers), "body": json.loads(body), "path": request.url.path})
        return JSONResponse({"input_tokens": 1234})

    async def _messages(self, request: Request) -> Response:
        body = await request.body()
        self.seen.append({"headers": dict(request.headers), "body": json.loads(body), "path": request.url.path})
        if not self._authorised(request):
            return JSONResponse({"type": "error", "error": {"type": "authentication_error", "message": "bad key"}},
                                status_code=401)
        if not request.headers.get("anthropic-version"):
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "anthropic-version: header is required"}},
                                status_code=400)
        what = self._next()
        if what == "busy":
            return JSONResponse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
                                status_code=529, headers={"retry-after": "1"})
        if what == "out_of_funds":
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "Your credit balance is too low to access the API"}},
                                status_code=400)
        if what == "refused":
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "max_tokens: Field required"}},
                                status_code=400)
        asked = json.loads(body)
        model = asked.get("model")
        usage = dict(self.anthropic_usage)
        tool = what == "tool"
        content = ([{"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Paris"}}] if tool
                   else [{"type": "text", "text": "hello from the provider"}])
        stop = "tool_use" if tool else "end_turn"
        if not asked.get("stream"):
            return JSONResponse({"id": "msg_1", "type": "message", "role": "assistant", "model": model,
                                 "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage})

        async def events():
            start = {k: v for k, v in usage.items()}
            start["output_tokens"] = 1
            yield self._event("message_start", {"message": {"id": "msg_1", "type": "message", "role": "assistant",
                                                            "model": model, "content": [], "stop_reason": None,
                                                            "stop_sequence": None, "usage": start}})
            yield b'event: ping\ndata: {"type": "ping"}\n\n'
            if tool:
                yield self._event("content_block_start", {"index": 0, "content_block": {
                    "type": "tool_use", "id": "toolu_1", "name": "weather", "input": {}}})
                for part in ('{"city": ', '"Paris"}'):
                    yield self._event("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta",
                                                                                    "partial_json": part}})
            else:
                yield self._event("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
                for word in ("hello", " from", " the", " provider"):
                    yield self._event("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": word}})
            yield self._event("content_block_stop", {"index": 0})
            yield self._event("message_delta", {"delta": {"stop_reason": stop, "stop_sequence": None},
                                                "usage": {"output_tokens": usage["output_tokens"]}})
            yield self._event("message_stop", {})

        return StreamingResponse(events(), media_type="text/event-stream")
