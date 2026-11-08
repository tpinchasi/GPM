"""Between OpenAI's Chat Completions and Anthropic's Messages formats (S12).

The pool passes a request through untouched wherever the chosen host speaks its format. Only
where it does not — Claude asked in OpenAI's format, GPT in Anthropic's — is the request
translated on the way in, and the answer, stream included, on the way out. Each direction is a
closed mapping: a field it knows is carried, or refused by name.

**Nothing is dropped without a word.** A request carrying what the other format cannot say —
Anthropic's thinking, PDF documents, citations or server tools to an OpenAI-format host; an
OpenAI field with no Anthropic meaning — is refused (`Unsupported`), unless the app asked for
what cannot be carried to be dropped (`X-GPM-Unsupported: drop`); then the dropped fields are
named in the answer's `X-GPM-Dropped`. A cache mark is not refused: its meaning — cache this — is
what an OpenAI-format provider does by itself.

Usage and cost are read from the provider's own answer before it is translated back; the
translated answer carries the counts in the app's format.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from .engines import anthropic_api

CHAT_PATH = "/v1/chat/completions"
MESSAGES_PATH = anthropic_api.MESSAGES_PATH

#: Output a request that names no limit may generate, where the target format requires one
#: (Anthropic's `max_tokens` is required).
DEFAULT_MAX_TOKENS = 4096


class Unsupported(ValueError):
    """What a request carries that the other format cannot say."""

    def __init__(self, fields: list[str]):
        super().__init__(", ".join(fields))
        self.fields = fields


def peer(path: str) -> Optional[str]:
    """The path a request on this one is translated to, where one exists."""
    return {CHAT_PATH: MESSAGES_PATH, MESSAGES_PATH: CHAT_PATH}.get(path)


def _dumps(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sse(event: Optional[str], data: Any) -> bytes:
    head = f"event: {event}\n".encode() if event else b""
    return head + b"data: " + (data if isinstance(data, bytes) else _dumps(data)) + b"\n\n"


class _Missing:
    def __init__(self, lenient: bool):
        self.lenient = lenient
        self.fields: list[str] = []

    def add(self, name: str) -> None:
        if name not in self.fields:
            self.fields.append(name)

    def done(self) -> list[str]:
        if self.fields and not self.lenient:
            raise Unsupported(self.fields)
        return self.fields


# ====================================================================================
# OpenAI's Chat Completions -> Anthropic's Messages (an OpenAI-format app, Claude's API)
# ====================================================================================

#: OpenAI request fields carried, and those with no Anthropic meaning that are harmless when
#: they say the default.
_OA_CARRIED = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stop",
               "stream", "stream_options", "tools", "tool_choice", "parallel_tool_calls", "user", "metadata"}
_OA_DEFAULTS = {"n": 1, "presence_penalty": 0, "frequency_penalty": 0, "logprobs": False}


def _image_source(url: str) -> dict[str, Any]:
    if url.startswith("data:") and ";base64," in url:
        media, data = url[len("data:"):].split(";base64,", 1)
        return {"type": "base64", "media_type": media, "data": data}
    return {"type": "url", "url": url}


def _oa_content(content: Any, missing: _Missing, where: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content if isinstance(content, list) else []:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if isinstance(url, str):
                blocks.append({"type": "image", "source": _image_source(url)})
        else:
            missing.add(f"{where} content part {kind!r}")
    return blocks


def openai_to_anthropic_request(body: bytes, *, lenient: bool = False,
                                default_max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[bytes, list[str]]:
    """An OpenAI chat request as an Anthropic Messages request, and what was dropped."""
    request = json.loads(body)
    missing = _Missing(lenient)
    for key, value in request.items():
        if key in _OA_CARRIED:
            continue
        if key in _OA_DEFAULTS and value in (None, _OA_DEFAULTS[key]):
            continue
        missing.add(key)
    system: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)  # one turn: Anthropic's turns alternate
        else:
            messages.append({"role": role, "content": list(blocks)})

    for message in request.get("messages") or []:
        role = message.get("role")
        if role in ("system", "developer"):
            system += [b for b in _oa_content(message.get("content"), missing, role) if b["type"] == "text"]
        elif role == "user":
            add("user", _oa_content(message.get("content"), missing, "user"))
        elif role == "assistant":
            blocks = _oa_content(message.get("content"), missing, "assistant")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except ValueError:
                    arguments = {}
                blocks.append({"type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                               "name": function.get("name", ""), "input": arguments})
            add("assistant", blocks)
        elif role == "tool":
            content = message.get("content")
            result = content if isinstance(content, str) else [
                b for b in _oa_content(content, missing, "tool") if b["type"] in ("text", "image")]
            add("user", [{"type": "tool_result", "tool_use_id": message.get("tool_call_id", ""), "content": result}])
        else:
            missing.add(f"message role {role!r}")
    out: dict[str, Any] = {"model": request.get("model")}
    if system:
        out["system"] = system
    out["messages"] = messages
    limit = request.get("max_completion_tokens") or request.get("max_tokens")
    out["max_tokens"] = limit if isinstance(limit, int) and limit > 0 else default_max_tokens
    for key in ("temperature", "top_p"):
        if request.get(key) is not None:
            out[key] = request[key]
    stop = request.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if request.get("stream") is True:
        out["stream"] = True
    tools = request.get("tools")
    if tools:
        out["tools"] = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or tool.get("type") != "function":
                missing.add(f"tool type {tool.get('type') if isinstance(tool, dict) else tool!r}")
                continue
            out["tools"].append({"name": function.get("name", ""), "description": function.get("description", ""),
                                 "input_schema": function.get("parameters") or {"type": "object", "properties": {}}})
    choice = request.get("tool_choice")
    if choice is not None:
        if choice == "auto":
            out["tool_choice"] = {"type": "auto"}
        elif choice == "none":
            out["tool_choice"] = {"type": "none"}
        elif choice == "required":
            out["tool_choice"] = {"type": "any"}
        elif isinstance(choice, dict) and isinstance(choice.get("function"), dict):
            out["tool_choice"] = {"type": "tool", "name": choice["function"].get("name", "")}
        else:
            missing.add("tool_choice")
    if request.get("parallel_tool_calls") is False:
        out.setdefault("tool_choice", {"type": "auto"})["disable_parallel_tool_use"] = True
    if isinstance(request.get("user"), str):
        out["metadata"] = {"user_id": request["user"]}
    return _dumps(out), missing.done()


_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
           "refusal": "content_filter", "pause_turn": "stop", "model_context_window_exceeded": "length"}


def _oa_usage(usage: dict[str, Any]) -> dict[str, Any]:
    read = int(usage.get("cache_read_input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0) + read + written
    completion = int(usage.get("output_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": written}}


def anthropic_to_openai_response(body: bytes) -> bytes:
    """A whole Anthropic Messages answer as an OpenAI chat completion."""
    answer = json.loads(body)
    text = "".join(block.get("text", "") for block in answer.get("content") or [] if block.get("type") == "text")
    calls = [{"id": block.get("id"), "type": "function",
              "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})}}
             for block in answer.get("content") or [] if block.get("type") == "tool_use"]
    message: dict[str, Any] = {"role": "assistant", "content": text or (None if calls else "")}
    if calls:
        message["tool_calls"] = calls
    return _dumps({
        "id": "chatcmpl-" + str(answer.get("id") or uuid.uuid4().hex), "object": "chat.completion",
        "created": int(time.time()), "model": answer.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": _FINISH.get(answer.get("stop_reason"), "stop")}],
        "usage": _oa_usage(answer.get("usage") or {}),
    })


class _SSE:
    """Server-sent events, a frame at a time, however the bytes were split."""

    def __init__(self) -> None:
        self.pending = b""

    def frames(self, chunk: bytes) -> list[tuple[Optional[str], bytes]]:
        self.pending += chunk.replace(b"\r\n", b"\n")
        found = []
        while b"\n\n" in self.pending:
            raw, self.pending = self.pending.split(b"\n\n", 1)
            event, data = None, []
            for line in raw.split(b"\n"):
                if line.startswith(b"event:"):
                    event = line[len(b"event:"):].strip().decode()
                elif line.startswith(b"data:"):
                    data.append(line[len(b"data:"):].strip())
            if data:
                found.append((event, b"\n".join(data)))
        return found


class AnthropicToOpenAIStream:
    """An Anthropic Messages stream, rewritten as OpenAI chat-completion chunks as it arrives."""

    def __init__(self, include_usage: bool):
        self.sse = _SSE()
        self.include_usage = include_usage
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.model: Optional[str] = None
        self.created = int(time.time())
        self.usage: dict[str, Any] = {}
        self.tools: dict[int, int] = {}  # content block index -> tool call index
        self.finish: Optional[str] = None

    def _chunk(self, delta: dict[str, Any], finish: Optional[str] = None) -> bytes:
        return _sse(None, {"id": self.id, "object": "chat.completion.chunk", "created": self.created,
                           "model": self.model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def feed(self, chunk: bytes) -> bytes:
        out = bytearray()
        for _, data in self.sse.frames(chunk):
            try:
                event = json.loads(data)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message") or {}
                self.model = message.get("model")
                self.usage.update(message.get("usage") or {})
                out += self._chunk({"role": "assistant", "content": ""})
            elif kind == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    at = self.tools[event.get("index", 0)] = len(self.tools)
                    out += self._chunk({"tool_calls": [{"index": at, "id": block.get("id"), "type": "function",
                                                        "function": {"name": block.get("name"), "arguments": ""}}]})
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    out += self._chunk({"content": delta.get("text", "")})
                elif delta.get("type") == "input_json_delta" and event.get("index", 0) in self.tools:
                    out += self._chunk({"tool_calls": [{"index": self.tools[event["index"]],
                                                        "function": {"arguments": delta.get("partial_json", "")}}]})
            elif kind == "message_delta":
                self.usage.update({k: v for k, v in (event.get("usage") or {}).items() if v is not None})
                reason = (event.get("delta") or {}).get("stop_reason")
                if reason:
                    self.finish = _FINISH.get(reason, "stop")
            elif kind == "message_stop":
                out += self._chunk({}, self.finish or "stop")
                if self.include_usage:
                    out += _sse(None, {"id": self.id, "object": "chat.completion.chunk", "created": self.created,
                                       "model": self.model, "choices": [], "usage": _oa_usage(self.usage)})
                out += b"data: [DONE]\n\n"
            elif kind == "error":
                out += _sse(None, {"error": _openai_error_of(event.get("error") or {})})
        return bytes(out)

    def finish_stream(self) -> bytes:
        return b""


# ====================================================================================
# Anthropic's Messages -> OpenAI's Chat Completions (an Anthropic-SDK app, an OpenAI-format host)
# ====================================================================================

_AO_CARRIED = {"model", "messages", "system", "max_tokens", "temperature", "top_p", "stop_sequences", "stream",
               "tools", "tool_choice", "metadata", "cache_control"}


def _ao_blocks(content: Any, missing: _Missing, where: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                                     list[dict[str, Any]]]:
    """(content parts, tool calls, tool results) of one Anthropic turn's content."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}], [], []
    parts, calls, results = [], [], []
    for block in content if isinstance(content, list) else []:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"}})
            elif source.get("type") == "url":
                parts.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
            else:
                missing.add(f"{where} image source {source.get('type')!r}")
        elif kind == "tool_use":
            calls.append({"id": block.get("id"), "type": "function",
                          "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})}})
        elif kind == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                text = "".join(b.get("text", "") for b in inner if isinstance(b, dict) and b.get("type") == "text")
                if any(isinstance(b, dict) and b.get("type") != "text" for b in inner):
                    missing.add(f"{where} tool_result content other than text")
            else:
                text = inner if isinstance(inner, str) else ""
            results.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": text})
        else:
            missing.add(f"{where} content block {kind!r}")
    return parts, calls, results


def _plain(parts: list[dict[str, Any]]) -> Any:
    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] for part in parts)
    return parts


def anthropic_to_openai_request(body: bytes, *, lenient: bool = False) -> tuple[bytes, list[str]]:
    """An Anthropic Messages request as an OpenAI chat request, and what was dropped."""
    request = json.loads(body)
    missing = _Missing(lenient)
    for key in request:
        if key not in _AO_CARRIED:
            missing.add(key)
    messages: list[dict[str, Any]] = []
    system = request.get("system")
    if system:
        text = system if isinstance(system, str) else "".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
        messages.append({"role": "system", "content": text})
    for message in request.get("messages") or []:
        role = message.get("role")
        parts, calls, results = _ao_blocks(message.get("content"), missing, role or "message")
        if role == "user":
            messages.extend(results)  # a tool's answer comes before what the user says next
            if parts:
                messages.append({"role": "user", "content": _plain(parts)})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": _plain(parts) if parts else None}
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
        else:
            missing.add(f"message role {role!r}")
    out: dict[str, Any] = {"model": request.get("model"), "messages": messages}
    if isinstance(request.get("max_tokens"), int):
        out["max_completion_tokens"] = request["max_tokens"]
    for key in ("temperature", "top_p"):
        if request.get(key) is not None:
            out[key] = request[key]
    if request.get("stop_sequences"):
        out["stop"] = list(request["stop_sequences"])
    if request.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}  # the counts this stream is charged by (D166)
    tools = request.get("tools")
    if tools:
        out["tools"] = []
        for tool in tools:
            if not isinstance(tool, dict) or "input_schema" not in tool or tool.get("type") not in (None, "custom"):
                missing.add(f"tool {tool.get('type') or tool.get('name') if isinstance(tool, dict) else tool!r}")
                continue
            out["tools"].append({"type": "function", "function": {
                "name": tool.get("name", ""), "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}}})
    choice = request.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "none":
            out["tool_choice"] = "none"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name", "")}}
        else:
            missing.add("tool_choice")
        if choice.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False
    metadata = request.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("user_id"), str):
        out["user"] = metadata["user_id"]
    return _dumps(out), missing.done()


_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "function_call": "tool_use",
         "content_filter": "refusal"}


def _an_usage(usage: dict[str, Any]) -> dict[str, Any]:
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cached = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    return {"input_tokens": max(0, prompt - cached - written), "output_tokens": int(usage.get("completion_tokens") or 0),
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": written}


def openai_to_anthropic_response(body: bytes) -> bytes:
    """A whole OpenAI chat completion as an Anthropic Messages answer."""
    answer = json.loads(body)
    choice = (answer.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            arguments = {}
        content.append({"type": "tool_use", "id": call.get("id"), "name": function.get("name"), "input": arguments})
    return _dumps({
        "id": "msg_" + str(answer.get("id") or uuid.uuid4().hex), "type": "message", "role": "assistant",
        "model": answer.get("model"), "content": content,
        "stop_reason": _STOP.get(choice.get("finish_reason"), "end_turn"), "stop_sequence": None,
        "usage": _an_usage(answer.get("usage") or {}),
    })


class OpenAIToAnthropicStream:
    """OpenAI chat-completion chunks, rewritten as an Anthropic Messages stream as they arrive."""

    def __init__(self) -> None:
        self.sse = _SSE()
        self.started = False
        self.model: Optional[str] = None
        self.block: Optional[int] = None      # the content block open now
        self.block_kind: Optional[str] = None
        self.blocks = 0
        self.tools: dict[int, int] = {}       # tool call index -> content block index
        self.stop: Optional[str] = None
        self.usage: dict[str, Any] = {}
        self.ended = False

    def _start(self, model: Optional[str]) -> bytes:
        self.started, self.model = True, model
        return _sse("message_start", {"type": "message_start", "message": {
            "id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}})

    def _close(self) -> bytes:
        if self.block is None:
            return b""
        at, self.block, self.block_kind = self.block, None, None
        return _sse("content_block_stop", {"type": "content_block_stop", "index": at})

    def _open(self, kind: str, block: dict[str, Any]) -> bytes:
        out = self._close()
        self.block, self.block_kind = self.blocks, kind
        self.blocks += 1
        return out + _sse("content_block_start", {"type": "content_block_start", "index": self.block,
                                                  "content_block": block})

    def _end(self) -> bytes:
        if self.ended:
            return b""
        self.ended = True
        out = b"" if self.started else self._start(self.model)
        out += self._close()
        out += _sse("message_delta", {"type": "message_delta",
                                      "delta": {"stop_reason": self.stop or "end_turn", "stop_sequence": None},
                                      "usage": _an_usage(self.usage)})
        return out + _sse("message_stop", {"type": "message_stop"})

    def feed(self, chunk: bytes) -> bytes:
        out = bytearray()
        for _, data in self.sse.frames(chunk):
            if data == b"[DONE]":
                out += self._end()
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if "error" in event and "choices" not in event:
                out += _sse("error", {"type": "error", "error": _anthropic_error_of(event.get("error") or {})})
                continue
            if not self.started:
                out += self._start(event.get("model"))
            if isinstance(event.get("usage"), dict):
                self.usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    if self.block_kind != "text":
                        out += self._open("text", {"type": "text", "text": ""})
                    out += _sse("content_block_delta", {"type": "content_block_delta", "index": self.block,
                                                        "delta": {"type": "text_delta", "text": delta["content"]}})
                for call in delta.get("tool_calls") or []:
                    at = call.get("index", 0)
                    function = call.get("function") or {}
                    if at not in self.tools:
                        out += self._open("tool_use", {"type": "tool_use", "id": call.get("id"),
                                                       "name": function.get("name"), "input": {}})
                        self.tools[at] = self.block
                    if function.get("arguments"):
                        out += _sse("content_block_delta", {"type": "content_block_delta", "index": self.tools[at],
                                                            "delta": {"type": "input_json_delta",
                                                                      "partial_json": function["arguments"]}})
                if choice.get("finish_reason"):
                    self.stop = _STOP.get(choice["finish_reason"], "end_turn")
        return bytes(out)

    def finish_stream(self) -> bytes:
        """A stream that ended without `[DONE]` still ends as this format says one does."""
        return self._end() if self.started else b""


# ====================================================================================
# Errors, in the app's format
# ====================================================================================


def _openai_error_of(error: dict[str, Any]) -> dict[str, Any]:
    return {"message": error.get("message", ""), "type": error.get("type", "api_error"), "code": error.get("type")}


def _anthropic_error_of(error: dict[str, Any]) -> dict[str, Any]:
    return {"type": error.get("type") or error.get("code") or "api_error", "message": error.get("message", "")}


def error_into(source_path: str, body: bytes) -> bytes:
    """A provider's error answer, in the format the app asked in."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return body
    if not isinstance(parsed, dict):
        return body
    error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {"message": str(parsed)}
    if source_path == CHAT_PATH:
        return _dumps({"error": _openai_error_of(error)})
    return _dumps({"type": "error", "error": _anthropic_error_of(error)})


# ====================================================================================
# One request's translation, as the router uses it
# ====================================================================================


class Translation:
    """How one request is carried to a host that does not speak its format, and back."""

    def __init__(self, source_path: str, *, lenient: bool, default_max_tokens: int = DEFAULT_MAX_TOKENS):
        self.source_path = source_path
        self.target_path = peer(source_path) or source_path
        self.lenient = lenient
        self.default_max_tokens = default_max_tokens
        self.dropped: list[str] = []
        self.include_usage = False

    def request(self, body: bytes) -> bytes:
        if self.source_path == CHAT_PATH:
            asked = json.loads(body)
            options = asked.get("stream_options")
            self.include_usage = isinstance(options, dict) and options.get("include_usage") is True
            out, self.dropped = openai_to_anthropic_request(body, lenient=self.lenient,
                                                            default_max_tokens=self.default_max_tokens)
            return out
        out, self.dropped = anthropic_to_openai_request(body, lenient=self.lenient)
        return out

    def response(self, body: bytes) -> bytes:
        if self.source_path == CHAT_PATH:
            return anthropic_to_openai_response(body)
        return openai_to_anthropic_response(body)

    def stream(self) -> Any:
        if self.source_path == CHAT_PATH:
            return AnthropicToOpenAIStream(self.include_usage)
        return OpenAIToAnthropicStream()

    def error(self, body: bytes) -> bytes:
        return error_into(self.source_path, body)


def unsupported(source_path: str, body: bytes) -> list[str]:
    """What this request carries that its translation could not, without translating it for a
    host — so the pool knows before choosing one whether a translating host may take it."""
    try:
        if source_path == CHAT_PATH:
            return openai_to_anthropic_request(body, lenient=True)[1]
        return anthropic_to_openai_request(body, lenient=True)[1]
    except (ValueError, TypeError, AttributeError):
        return ["the body"]
