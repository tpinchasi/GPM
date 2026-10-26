"""`multipart/form-data` uploads, read strictly and rebuilt (D144).

An upload — a recording for `/v1/audio/transcriptions` — names its model in a form field. The
pool must read that field, rewrite it, and forward a body the engine reads *exactly* as the pool
did: the rule D141 set for JSON. A form can be framed in ways parsers disagree about — a second
boundary parameter, a folded header, a bare line feed, a part named `MODEL` — and each is a way
for the pool to check one model while the engine serves another. So a form is parsed strictly,
anything a parser would have to guess about is refused (`BadUpload`), and the form forwarded is
**rebuilt**: the pool's own boundary, canonical part headers, every file's bytes untouched.

Everything in this module is on the router's request path: synchronous, no I/O, and linear in
the size of the body — the body is scanned with `bytes.find`, never with a pattern.
"""

from __future__ import annotations

import dataclasses
import secrets
from typing import Mapping, Optional


class BadUpload(ValueError):
    """A refused upload. `why` is a fixed, short English reason: it never repeats the client's
    bytes, field values or file names, so it is safe to log and to answer with."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


@dataclasses.dataclass(frozen=True)
class Part:
    name: str
    data: bytes
    #: Set for a file, None for a text field. An empty file name — what a browser sends for a
    #: file input left empty — is still a file.
    filename: Optional[str] = None
    content_type: Optional[str] = None


#: The form field the pool reads and rewrites.
MODEL = "model"

#: RFC 2046's limit, and a part's header section — two short lines in any real client.
_MAX_BOUNDARY = 70
_MAX_PART_HEADERS = 4096

_TOKEN = frozenset(
    "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
#: RFC 2046 `bchars`; a space is allowed, though not last, only inside quotes.
_BCHARS = frozenset("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'()+_,-./:=? ")
_BOUNDARY_TOKEN = _BCHARS & _TOKEN
#: What may not appear anywhere in a part's header section: every control character but tab.
_HEADER_CONTROLS = bytes(c for c in range(0x20) if c != 0x09) + b"\x7f"
_TEXT_CONTROLS = frozenset(chr(c) for c in [*range(0x20), *range(0x7F, 0xA0)])
_PLAIN_TEXT = frozenset({"text/plain", "text/plain;charset=utf-8", 'text/plain;charset="utf-8"'})


def parse(content_type: str, body: bytes, *, max_parts: int = 16) -> list[Part]:
    """The form's parts in order, or `BadUpload`.

    Refused: a `Content-Type` that is not `multipart/form-data` with exactly one plain
    `boundary`; a preamble, an epilogue or a form left open; a delimiter, header or parameter
    another parser could read differently; a nested form; no parts, or more than `max_parts`;
    two parts whose names differ only in letter case; and a `model` field that is a file, is
    spelled other than `model`, or is not one line of UTF-8 text.
    """
    dash = b"--" + _boundary(content_type)
    if not body.startswith(dash + b"\r\n"):
        if body.startswith(dash + b"--"):
            raise BadUpload("the form has no parts")
        raise BadUpload("the form does not start with its boundary")
    delimiter = b"\r\n" + dash
    at = len(dash) + 2
    parts: list[Part] = []
    names: set[str] = set()
    while True:
        part, at = _part(body, at, delimiter)
        folded = part.name.casefold()
        if folded in names:
            raise BadUpload("two parts have the same name")
        names.add(folded)
        if folded == MODEL:
            _model_checks(part)
        parts.append(part)
        after = body[at:at + 2]
        if after == b"\r\n":
            if len(parts) == max_parts:
                raise BadUpload("the form has too many parts")
            at += 2
            continue
        if after == b"--":
            if body[at + 2:] not in (b"", b"\r\n"):
                raise BadUpload("bytes follow the form's closing delimiter")
            return parts
        if len(after) < 2:
            raise BadUpload("the form is not closed")
        raise BadUpload("a delimiter is followed by something other than a line break")


def field(parts: list[Part], name: str) -> Optional[str]:
    """The value of the text field `name`, or None if the form has none.

    `name` is the pool's own, spelled as the pool reads it. A part by that name in another
    letter case, a file by that name, or a value that is not UTF-8 plain text is `BadUpload`:
    an engine could read any of them differently from the pool (D141).
    """
    folded = name.casefold()
    for part in parts:
        if part.name.casefold() == folded:
            return _text(part, name)
    return None


def rebuild(parts: list[Part], *, replace: Mapping[str, str] = {}) -> tuple[str, bytes]:
    """The form written afresh: `(Content-Type header value, body)`.

    A boundary the pool draws, part headers in one canonical form, parts in their order, and
    every part's bytes as they were — except the text fields named in `replace`, whose value is
    set (UTF-8, and no `Content-Type`: the value is the pool's own). `replace` never adds a
    field: a name that is not a text field of the form is a `KeyError`.
    """
    values = dict(replace)
    written: list[tuple[bytes, bytes]] = []
    for part in parts:
        content_type = part.content_type
        data = part.data
        if part.filename is None and part.name in values:
            data = values.pop(part.name).encode("utf-8")
            content_type = None
        written.append((_part_headers(part.name, part.filename, content_type), data))
    if values:
        raise KeyError(next(iter(values)))
    while True:
        boundary = "gpm-" + secrets.token_hex(16)
        dash = b"--" + boundary.encode("ascii")
        if not any(dash in data for _, data in written):
            break
    # Joined once, so a large file is copied once.
    pieces: list[bytes] = []
    for headers, data in written:
        pieces += (dash, b"\r\n", headers, b"\r\n", data, b"\r\n")
    pieces += (dash, b"--\r\n")
    return f"multipart/form-data; boundary={boundary}", b"".join(pieces)


# --- reading ---


def _boundary(content_type: str) -> bytes:
    """The one `boundary` parameter of a `multipart/form-data` content type, as bytes."""
    kind, _, rest = content_type.partition(";")
    kind = kind.strip(" \t")
    if not kind.isascii() or kind.lower() != "multipart/form-data":
        raise BadUpload("the request is not multipart/form-data")
    found: Optional[str] = None
    for parameter in rest.split(";") if rest else []:
        key, equals, value = parameter.strip(" \t").partition("=")
        if not equals or not _is_token(key):
            raise BadUpload("a content-type parameter is malformed")
        if "*" in key:
            # RFC 2231's extended and continued forms: one parser joins them, another ignores.
            raise BadUpload("the boundary is given in an extended form")
        if key.lower() != "boundary":
            raise BadUpload("the content type has a parameter other than boundary")
        if found is not None:
            raise BadUpload("the boundary is given twice")
        found = value
    if found is None:
        raise BadUpload("the content type has no boundary")
    allowed = _BOUNDARY_TOKEN
    if found.startswith('"'):
        if len(found) < 2 or not found.endswith('"'):
            raise BadUpload("a content-type parameter is malformed")
        found = found[1:-1]
        allowed = _BCHARS
        if found.endswith(" "):
            raise BadUpload("the boundary has a character RFC 2046 does not allow")
    if not found:
        raise BadUpload("the boundary is empty")
    if len(found) > _MAX_BOUNDARY:
        raise BadUpload("the boundary is longer than 70 characters")
    if not set(found) <= allowed:
        raise BadUpload("the boundary has a character RFC 2046 does not allow")
    return found.encode("ascii")


def _part(body: bytes, at: int, delimiter: bytes) -> tuple[Part, int]:
    """The part whose headers start at `at`, and where its closing delimiter ends."""
    if body[at:at + 2] == b"\r\n":
        raise BadUpload("a part has no headers")
    end = body.find(b"\r\n\r\n", at, at + _MAX_PART_HEADERS)
    if end < 0:
        raise BadUpload("a part's headers are too long or not terminated")
    headers = _headers(body[at:end])
    if "content-disposition" not in headers:
        raise BadUpload("a part has no Content-Disposition")
    name, filename = _disposition(headers["content-disposition"])
    content_type = headers.get("content-type")
    if content_type is not None:
        _check_content_type(content_type)
    start = end + 4
    close = body.find(delimiter, start)
    if close < 0:
        raise BadUpload("the form is not closed")
    return Part(name, body[start:close], filename, content_type), close + len(delimiter)


def _headers(section: bytes) -> dict[str, str]:
    """A part's headers by lower-case name: only the two a form part has, each at most once."""
    found: dict[str, str] = {}
    for line in section.split(b"\r\n"):
        if b"\r" in line or b"\n" in line:
            raise BadUpload("a part's headers have a bare carriage return or line feed")
        if line[:1] in (b" ", b"\t"):
            raise BadUpload("a part header is folded")
        if len(line.translate(None, _HEADER_CONTROLS)) != len(line):
            raise BadUpload("a part header has a control character")
        raw_name, colon, raw_value = line.partition(b":")
        name = raw_name.decode("latin-1")
        if not colon or not _is_token(name):
            raise BadUpload("a part header is malformed")
        try:
            value = raw_value.decode("utf-8").strip(" \t")
        except UnicodeDecodeError:
            raise BadUpload("a part header is not UTF-8") from None
        name = name.lower()
        if name == "content-transfer-encoding":
            raise BadUpload("a part has a transfer encoding")
        if name not in ("content-disposition", "content-type"):
            raise BadUpload("a part has a header other than Content-Disposition and Content-Type")
        if name in found:
            raise BadUpload("a part header is given twice")
        found[name] = value
    return found


def _disposition(value: str) -> tuple[str, Optional[str]]:
    """`form-data; name="..."` with an optional `filename`: the name, and the file name or
    None. Quoted values take no escapes — a backslash is refused rather than interpreted,
    because parsers disagree on whether it escapes."""
    semicolon = value.find(";")
    kind = (value if semicolon < 0 else value[:semicolon]).strip(" \t")
    if not _is_token(kind) or kind.lower() != "form-data":
        raise BadUpload("a part's disposition is not form-data")
    found: dict[str, str] = {}
    at = len(value) if semicolon < 0 else semicolon
    while at < len(value):
        at = _skip_space(value, at + 1)
        equals = value.find("=", at)
        key = value[at:equals] if equals >= 0 else ""
        if not _is_token(key):
            raise BadUpload("a part's Content-Disposition is malformed")
        if "*" in key:
            raise BadUpload("a part's Content-Disposition has an extended parameter")
        at = equals + 1
        if value[at:at + 1] == '"':
            close = value.find('"', at + 1)
            if close < 0:
                raise BadUpload("a part's Content-Disposition is malformed")
            parameter = value[at + 1:close]
            if "\\" in parameter:
                raise BadUpload("a part's Content-Disposition has a backslash")
            at = close + 1
        else:
            semicolon = value.find(";", at)
            close = len(value) if semicolon < 0 else semicolon
            parameter = value[at:close].rstrip(" \t")
            if not _is_token(parameter):
                raise BadUpload("a part's Content-Disposition is malformed")
            at = close
        at = _skip_space(value, at)
        if at < len(value) and value[at] != ";":
            raise BadUpload("a part's Content-Disposition is malformed")
        key = key.lower()
        if key not in ("name", "filename"):
            raise BadUpload("a part's Content-Disposition has an unknown parameter")
        if key in found:
            raise BadUpload("a part's Content-Disposition parameter is given twice")
        found[key] = parameter
    if "name" not in found:
        raise BadUpload("a part has no name")
    if not found["name"]:
        raise BadUpload("a part's name is empty")
    return found["name"], found.get("filename")


def _check_content_type(value: str) -> None:
    kind, _, subtype = value.split(";", 1)[0].strip(" \t").partition("/")
    if not _is_token(kind) or not _is_token(subtype):
        raise BadUpload("a part's Content-Type is malformed")
    if kind.lower() == "multipart":
        raise BadUpload("a part is itself a multipart form")


def _model_checks(part: Part) -> None:
    """A model is one line of text: spelled `model`, not a file, UTF-8, no control characters."""
    if any(c in _TEXT_CONTROLS for c in _text(part, MODEL)):
        raise BadUpload("the model field has a control character")


def _text(part: Part, name: str) -> str:
    """A text field's value, refused where an engine could read it otherwise. `name` is the
    pool's own word for the field, never the client's."""
    if part.name != name:
        raise BadUpload(f"the {name} field is spelled in another letter case")
    if part.filename is not None:
        raise BadUpload(f"the {name} field is a file")
    if part.content_type is not None:
        if part.content_type.lower().replace(" ", "").replace("\t", "") not in _PLAIN_TEXT:
            raise BadUpload(f"the {name} field declares a type other than plain UTF-8 text")
    try:
        return part.data.decode("utf-8")
    except UnicodeDecodeError:
        raise BadUpload(f"the {name} field is not UTF-8") from None


def _is_token(text: str) -> bool:
    return bool(text) and set(text) <= _TOKEN


def _skip_space(text: str, at: int) -> int:
    while text[at:at + 1] in (" ", "\t"):
        at += 1
    return at


# --- writing ---


def _part_headers(name: str, filename: Optional[str], content_type: Optional[str]) -> bytes:
    """One part's header section, in the form every part is written. A value that could not be
    written without changing the framing — possible only for a part built by hand, not one
    `parse` returned — is a `ValueError`."""
    if not name:
        raise ValueError("a part's name is empty")
    disposition = f'Content-Disposition: form-data; name="{_quotable(name)}"'
    if filename is not None:
        disposition += f'; filename="{_quotable(filename)}"'
    lines = [disposition]
    if content_type is not None:
        if _has_control(content_type):
            raise ValueError("a part's content type has a control character")
        lines.append(f"Content-Type: {content_type}")
    return "".join(line + "\r\n" for line in lines).encode("utf-8")


def _quotable(value: str) -> str:
    if '"' in value or "\\" in value or _has_control(value):
        raise ValueError("a part's name or file name cannot be quoted")
    return value


def _has_control(value: str) -> bool:
    return any(c == "\x7f" or (c < " " and c != "\t") for c in value)
