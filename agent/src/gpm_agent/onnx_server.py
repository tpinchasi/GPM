"""The pool's own server for ONNX builds (S15, D179).

Run by the host's agent with the Python of the private environment it installed ONNX Runtime into
— never the agent's own — on loopback only. It serves the builds a plan file names (written by the
agent: each build's file on disk, its declarations and its largest batch), answers

- `GET /health` — 200 once it is up;
- `GET /v1/models` — each planned build, whether it is loaded, and why not;
- `POST /v1/predict` — `{"model": tag, "inputs": {...}}` → `{"model": tag, "outputs": {...}}`,

and prepares each input and reads each output as the build declares (D179): a picture decoded,
resized and normalised; named features put in the model's order; scores softmaxed, labelled and cut
to the best *k*. Requests for one build that arrive together run as one batch.

Only the runtime's own operators run: no custom-operator library is ever registered, so a file
runs no code. This module imports nothing from the agent: it is copied into the private
environment and run there with `python -I`.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

log = logging.getLogger("gpm-onnx-server")

#: The most a request body may hold; the pool's router refuses more long before (24 MB default).
MAX_BODY = 64 * 1024 * 1024
#: A picture's pixels, at most — a small file that decodes to gigabytes is refused (a "bomb").
MAX_PIXELS = 64 * 1024 * 1024
#: How long the first request of a batch waits for others to join it.
BATCH_WINDOW_S = 0.005

#: ONNX Runtime's element type names, and numpy's.
_DTYPES = {
    "tensor(float)": "float32", "tensor(float16)": "float16", "tensor(double)": "float64",
    "tensor(int64)": "int64", "tensor(int32)": "int32", "tensor(int16)": "int16", "tensor(int8)": "int8",
    "tensor(uint8)": "uint8", "tensor(uint16)": "uint16", "tensor(bool)": "bool",
}


class BadInput(ValueError):
    """The request's own fault, said in words: answered 400."""


class ModelFault(RuntimeError):
    """The build does not match what it declares (a label count, a shape): answered 500."""


# --- preparing inputs (D179) ---


def _np():
    import numpy

    return numpy


def _picture(value: Any, decl: dict[str, Any], dtype: str):
    np = _np()
    from PIL import Image

    if not isinstance(value, str):
        raise BadInput("a picture is a base64 string")
    if value.startswith(("http://", "https://", "file:")):
        raise BadInput("a picture is sent inline, as base64, never as a link")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadInput("a picture is not valid base64") from exc
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            mode = "L" if decl["channels"] == "gray" else "RGB"
            picture = opened.convert(mode)
    except Image.DecompressionBombError as exc:
        raise BadInput("the picture is too large to decode") from exc
    except Exception as exc:  # noqa: BLE001 — anything Pillow cannot read is the request's
        raise BadInput(f"the picture could not be read: {exc.__class__.__name__}") from exc
    width, height = decl["size"]
    picture = picture.resize((int(width), int(height)), Image.Resampling.BILINEAR)
    array = np.asarray(picture, dtype="float32")
    if array.ndim == 2:
        array = array[:, :, None]
    if decl["channels"] == "bgr":
        array = array[:, :, ::-1]
    if decl.get("scale"):
        array = array / float(decl["scale"])
    if decl.get("mean") is not None:
        array = array - np.asarray(decl["mean"], dtype="float32")
    if decl.get("std") is not None:
        array = array / np.asarray(decl["std"], dtype="float32")
    if decl["layout"] == "nchw":
        array = np.transpose(array, (2, 0, 1))
    if dtype in ("uint8", "int8"):
        array = np.clip(np.rint(array), 0, 255 if dtype == "uint8" else 127)
    return np.ascontiguousarray(array.astype(dtype))


def _features(value: Any, decl: dict[str, Any], dtype: str):
    np = _np()
    if not isinstance(value, dict):
        raise BadInput("features are an object of each feature's name to its value")
    names = decl["names"]
    missing = [n for n in names if n not in value]
    unknown = sorted(set(value) - set(names))
    if missing or unknown:
        raise BadInput(f"features missing {missing[:10]}, unknown {unknown[:10]}" if unknown else f"features missing {missing[:10]}")
    try:
        row = [float(value[n]) for n in names]
    except (TypeError, ValueError) as exc:
        raise BadInput("every feature is a number") from exc
    return np.asarray(row, dtype=dtype)


def _tensor(value: Any, dtype: str, rank: Optional[int]):
    np = _np()
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise BadInput(f"the tensor is not numbers of type {dtype}") from exc
    if array.dtype == object:
        raise BadInput("the tensor's rows are not all the same length")
    if rank is not None and array.ndim != rank - 1:
        raise BadInput(f"the tensor has {array.ndim} dimensions; one example of this input has {rank - 1}")
    return array


class Build:
    """One ONNX build, loaded: its session, what it declares, and how to batch it."""

    def __init__(self, tag: str, path: str, spec: dict[str, Any], max_batch: int, providers: list[str]):
        import onnxruntime

        self.tag, self.spec, self.max_batch = tag, spec, max(1, int(max_batch or 1))
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(path, sess_options=options, providers=providers)
        self.inputs = {meta.name: meta for meta in self.session.get_inputs()}
        self.outputs = {meta.name for meta in self.session.get_outputs()}
        for name, decl in spec["inputs"].items():
            if decl["input"] not in self.inputs:
                raise ModelFault(f"input {name!r} names {decl['input']!r}, which the file has not: {sorted(self.inputs)}")
        for name, decl in spec["outputs"].items():
            if decl["output"] not in self.outputs:
                raise ModelFault(f"output {name!r} names {decl['output']!r}, which the file has not: {sorted(self.outputs)}")
        undeclared = sorted(set(self.inputs) - {d["input"] for d in spec["inputs"].values()})
        if undeclared:
            raise ModelFault(f"the file's inputs {undeclared} are not declared")
        self.queue: "queue.Queue[_Job]" = queue.Queue()
        threading.Thread(target=self._batches, name=f"batch:{tag}", daemon=True).start()

    def prepare(self, inputs: Any) -> dict[str, Any]:
        """One example: each model input as an array without its batch dimension."""
        if not isinstance(inputs, dict):
            raise BadInput("`inputs` is an object of the model's input names to their values")
        unknown = sorted(set(inputs) - set(self.spec["inputs"]))
        if unknown:
            raise BadInput(f"inputs this model does not take: {unknown[:10]}")
        prepared = {}
        for name, decl in self.spec["inputs"].items():
            if name not in inputs:
                raise BadInput(f"input {name!r} is missing")
            meta = self.inputs[decl["input"]]
            dtype = _DTYPES.get(meta.type)
            if dtype is None:
                raise ModelFault(f"input {decl['input']!r} is of type {meta.type}, which this server does not prepare")
            rank = len(meta.shape) if meta.shape is not None else None
            if decl["as"] == "image":
                array = _picture(inputs[name], decl, dtype)
            elif decl["as"] == "features":
                array = _features(inputs[name], decl, dtype)
            else:
                array = _tensor(inputs[name], dtype, rank)
            if rank is not None and array.ndim != rank - 1:
                raise ModelFault(f"input {decl['input']!r} takes {rank} dimensions with its batch; "
                                 f"what is declared makes {array.ndim + 1}")
            prepared[decl["input"]] = array
        return prepared

    def read(self, results: dict[str, Any], row: int) -> dict[str, Any]:
        np = _np()
        answer: dict[str, Any] = {}
        for name, decl in self.spec["outputs"].items():
            value = results[decl["output"]][row]
            kind = decl["as"]
            if kind == "scores":
                scores = np.asarray(value, dtype="float64").reshape(-1)
                if decl.get("softmax"):
                    shifted = np.exp(scores - scores.max())
                    scores = shifted / shifted.sum()
                labels = decl.get("labels")
                if labels is not None and len(labels) != scores.size:
                    raise ModelFault(f"output {name!r} has {scores.size} scores and {len(labels)} labels")
                order = np.argsort(-scores, kind="stable")[: decl.get("top_k") or scores.size]
                answer[name] = [{"label": labels[i] if labels else int(i), "score": float(scores[i])} for i in order]
            elif kind == "vector":
                answer[name] = np.asarray(value, dtype="float64").reshape(-1).tolist()
            elif kind == "value":
                flat = np.asarray(value).reshape(-1).tolist()
                answer[name] = flat[0] if len(flat) == 1 else flat
            else:
                answer[name] = np.asarray(value).tolist()
        return answer

    def run(self, prepared: dict[str, Any]) -> dict[str, Any]:
        """One example, batched with any others that arrive within the window."""
        job = _Job(prepared)
        self.queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.answer

    def _batches(self) -> None:
        np = _np()
        while True:
            first = self.queue.get()
            jobs = [first]
            deadline = time.monotonic() + BATCH_WINDOW_S
            while len(jobs) < self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    jobs.append(self.queue.get(timeout=left))
                except queue.Empty:
                    break
            # Only examples of one shape stack; any other runs on its own.
            groups: dict[tuple, list[_Job]] = {}
            for job in jobs:
                key = tuple(sorted((name, array.shape) for name, array in job.prepared.items()))
                groups.setdefault(key, []).append(job)
            for group in groups.values():
                try:
                    feeds = {name: np.stack([job.prepared[name] for job in group]) for name in group[0].prepared}
                    names = sorted({d["output"] for d in self.spec["outputs"].values()})
                    produced = dict(zip(names, self.session.run(names, feeds), strict=True))
                    for row, job in enumerate(group):
                        job.answer = self.read(produced, row)
                except Exception as exc:  # noqa: BLE001 — one batch's failure is its requests', never the server's
                    for job in group:
                        job.error = exc if isinstance(exc, (BadInput, ModelFault)) else ModelFault(
                            f"the model failed to run: {exc.__class__.__name__}: {str(exc)[:300]}")
                for job in group:
                    job.done.set()


class _Job:
    def __init__(self, prepared: dict[str, Any]):
        self.prepared = prepared
        self.answer: dict[str, Any] = {}
        self.error: Optional[Exception] = None
        self.done = threading.Event()


# --- the server ---


class Builds:
    """Every planned build: loaded, or why not."""

    def __init__(self, plan: dict[str, dict[str, Any]], providers: list[str]):
        self.loaded: dict[str, Build] = {}
        self.failed: dict[str, str] = {}
        for tag, entry in plan.items():
            try:
                self.loaded[tag] = Build(tag, entry["path"], entry["spec"], entry.get("max_batch") or 1, providers)
            except Exception as exc:  # noqa: BLE001 — one build that will not load is that build's failure
                self.failed[tag] = f"{exc.__class__.__name__}: {str(exc)[:500]}"
                log.error("build %s did not load: %s", tag, self.failed[tag])


def _handler(builds: Builds) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "gpm-onnx-server"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # requests are the pool's to log
            return

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True})
            elif self.path == "/v1/models":
                rows = [{"id": tag, "loaded": True} for tag in builds.loaded]
                rows += [{"id": tag, "loaded": False, "error": why} for tag, why in builds.failed.items()]
                self._send(200, {"object": "list", "data": rows})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/predict":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self._send(413, {"error": {"message": "the request is too large"}})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"null")
            except ValueError:
                self._send(400, {"error": {"message": "the body is not JSON"}})
                return
            if not isinstance(body, dict) or not isinstance(body.get("model"), str):
                self._send(400, {"error": {"message": "send `model` and `inputs`"}})
                return
            tag = body["model"]
            build = builds.loaded.get(tag)
            if build is None:
                why = builds.failed.get(tag)
                self._send(503 if why else 404, {"error": {"message": f"{tag} did not load: {why}" if why else f"{tag} is not served here"}})
                return
            try:
                answer = build.run(build.prepare(body.get("inputs")))
            except BadInput as exc:
                self._send(400, {"error": {"message": str(exc), "type": "bad_input"}})
                return
            except ModelFault as exc:
                self._send(500, {"error": {"message": str(exc), "type": "model_fault"}})
                return
            self._send(200, {"model": tag, "outputs": answer})

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    #: The standard library keeps 5 connections waiting; requests arriving together — what batching
    #: is for — would be reset.
    request_queue_size = 256


def serve(plan: dict[str, dict[str, Any]], port: int, providers: list[str]) -> ThreadingHTTPServer:
    """The server on loopback, its builds loaded; not yet serving (call `serve_forever`)."""
    return _Server(("127.0.0.1", port), _handler(Builds(plan, providers)))


def providers_here() -> list[str]:
    """What runs the graph on this machine, best first: a card where there is one, else the processor."""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or ["CPUExecutionProvider"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="gpm-onnx-server", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--plan", required=True, help="the agent's plan: each build's file, declarations and batch")
    parser.add_argument("--processor-only", action="store_true", help="never use a card")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)
    providers = ["CPUExecutionProvider"] if args.processor_only else providers_here()
    server = serve(plan, args.port, providers)
    log.info("serving %d build(s) on 127.0.0.1:%d with %s", len(plan), args.port, providers)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
