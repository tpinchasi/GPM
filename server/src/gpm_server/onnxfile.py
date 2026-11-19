"""What an ONNX file takes in and gives back, read without loading it (S15, D179).

An ONNX file is a protocol buffer: `ModelProto`, whose `graph` holds the nodes, the weights
(`initializer`) and the graph's `input` and `output`, each a `ValueInfoProto` with a name, an
element type and a shape. This walks the file's fields and skips everything else by its length —
weights of hundreds of megabytes cost a seek — so the supervisor needs no ONNX package and runs
nothing from the file. Inputs that are weights (older files list them as inputs too) are left out.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Optional

#: ONNX's element types (`TensorProto.DataType`), as the console names them.
ELEMENT_TYPES = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64", 8: "string",
    9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16",
}
#: Fields read, by message: ModelProto.graph = 7; GraphProto.initializer = 5, .input = 11,
#: .output = 12; TensorProto.name = 8.
_MODEL_GRAPH, _GRAPH_INIT, _GRAPH_INPUT, _GRAPH_OUTPUT, _TENSOR_NAME = 7, 5, 11, 12, 8
#: A value-info is small; one larger than this is not one.
_MAX_VALUE_INFO = 1 << 20


class NotOnnx(ValueError):
    """The file is not an ONNX model this can read."""


def _varint(read: Any) -> Optional[int]:
    shift, value = 0, 0
    while True:
        byte = read(1)
        if not byte:
            if shift == 0:
                return None
            raise NotOnnx("the file ends inside a number")
        value |= (byte[0] & 0x7F) << shift
        if not byte[0] & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise NotOnnx("a number in the file is too long")


def _fields(handle: BinaryIO, end: int):
    """Each field in [here, end): (number, wire type, value or (start, length) for bytes)."""
    while handle.tell() < end:
        key = _varint(handle.read)
        if key is None:
            return
        number, wire = key >> 3, key & 7
        if wire == 0:
            yield number, wire, _varint(handle.read)
        elif wire == 1:
            handle.seek(8, 1)
        elif wire == 5:
            handle.seek(4, 1)
        elif wire == 2:
            length = _varint(handle.read)
            start = handle.tell()
            if length is None or start + length > end:
                raise NotOnnx("a field runs past the end of the file")
            yield number, wire, (start, length)
            handle.seek(start + length)
        else:
            raise NotOnnx(f"an unknown kind of field ({wire})")


def _bytes_fields(data: bytes):
    import io

    handle = io.BytesIO(data)
    for number, wire, value in _fields(handle, len(data)):
        if wire == 2:
            start, length = value
            yield number, data[start:start + length]
        elif wire == 0:
            yield number, value


def _value_info(data: bytes) -> dict[str, Any]:
    name, element, shape = "", None, None
    for number, value in _bytes_fields(data):
        if number == 1 and isinstance(value, bytes):
            name = value.decode("utf-8", "replace")
        elif number == 2 and isinstance(value, bytes):  # TypeProto
            for t_number, t_value in _bytes_fields(value):
                if t_number == 1 and isinstance(t_value, bytes):  # tensor_type
                    for k, v in _bytes_fields(t_value):
                        if k == 1 and isinstance(v, int):
                            element = ELEMENT_TYPES.get(v, f"type {v}")
                        elif k == 2 and isinstance(v, bytes):  # shape
                            shape = []
                            for d_number, dim in _bytes_fields(v):
                                if d_number != 1 or not isinstance(dim, bytes):
                                    continue
                                size: Any = None
                                for dk, dv in _bytes_fields(dim):
                                    if dk == 1 and isinstance(dv, int):
                                        size = dv
                                    elif dk == 2 and isinstance(dv, bytes):
                                        size = dv.decode("utf-8", "replace")
                                shape.append(size)
    return {"name": name, "type": element, "shape": shape}


def read(handle: BinaryIO) -> dict[str, Any]:
    """The graph's inputs and outputs: each `{name, type, shape}`, a shape's unknown sizes as their
    names (`"N"`, `"batch"`) or None."""
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(0)
    graph = None
    for number, wire, value in _fields(handle, size):
        if number == _MODEL_GRAPH and wire == 2:
            graph = value
    if graph is None:
        raise NotOnnx("the file holds no graph: it is not an ONNX model")
    start, length = graph
    handle.seek(start)
    inputs, outputs, weights = [], [], set()
    for number, wire, value in _fields(handle, start + length):
        if wire != 2 or number not in (_GRAPH_INIT, _GRAPH_INPUT, _GRAPH_OUTPUT):
            continue
        at, count = value
        if number == _GRAPH_INIT:
            handle.seek(at)
            for t_number, t_wire, t_value in _fields(handle, at + count):
                if t_number == _TENSOR_NAME and t_wire == 2:
                    t_at, t_count = t_value
                    handle.seek(t_at)
                    weights.add(handle.read(t_count).decode("utf-8", "replace"))
            continue
        if count > _MAX_VALUE_INFO:
            raise NotOnnx("an input or output is described at an unlikely length")
        handle.seek(at)
        info = _value_info(handle.read(count))
        (inputs if number == _GRAPH_INPUT else outputs).append(info)
    if not outputs:
        raise NotOnnx("the graph has no outputs: it is not an ONNX model this pool can serve")
    return {"inputs": [i for i in inputs if i["name"] not in weights], "outputs": outputs}


def problems(found: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Where a build's declarations and its file disagree, in words — none where they agree."""
    said: list[str] = []
    inputs = {i["name"]: i for i in found["inputs"]}
    outputs = {o["name"]: o for o in found["outputs"]}
    declared_inputs = {d["input"] for d in spec.get("inputs", {}).values()}
    for missing in sorted(set(inputs) - declared_inputs):
        said.append(f"the file's input {missing!r} is not declared")
    for name, decl in spec.get("inputs", {}).items():
        info = inputs.get(decl["input"])
        if info is None:
            said.append(f"input {name!r} names {decl['input']!r}, which the file has not ({sorted(inputs)})")
            continue
        shape = info["shape"]
        if decl["as"] == "image":
            if shape is not None and len(shape) != 4:
                said.append(f"input {name!r} is a picture, and {decl['input']!r} has {len(shape)} dimensions, not 4")
            elif shape is not None:
                channels = shape[1] if decl.get("layout") == "nchw" else shape[3]
                want = 1 if decl.get("channels") == "gray" else 3
                if isinstance(channels, int) and channels != want:
                    said.append(f"input {name!r}: the file takes {channels} channels, the declaration {want}")
                height, width = (shape[2], shape[3]) if decl.get("layout") == "nchw" else (shape[1], shape[2])
                size = decl.get("size") or [None, None]
                if isinstance(width, int) and isinstance(height, int) and [width, height] != list(size):
                    said.append(f"input {name!r}: the file takes {width}×{height} pictures, the declaration {size[0]}×{size[1]}")
        elif decl["as"] == "features" and shape is not None:
            if len(shape) != 2:
                said.append(f"input {name!r} is features, and {decl['input']!r} has {len(shape)} dimensions, not 2")
            elif isinstance(shape[1], int) and shape[1] != len(decl.get("names") or []):
                said.append(f"input {name!r}: the file takes {shape[1]} features, the declaration names {len(decl.get('names') or [])}")
    for name, decl in spec.get("outputs", {}).items():
        info = outputs.get(decl["output"])
        if info is None:
            said.append(f"output {name!r} names {decl['output']!r}, which the file has not ({sorted(outputs)})")
            continue
        labels, shape = decl.get("labels"), info["shape"]
        if decl["as"] == "scores" and labels and shape and isinstance(shape[-1], int) and shape[-1] != len(labels):
            said.append(f"output {name!r}: the file gives {shape[-1]} scores, the declaration {len(labels)} labels")
    return said
