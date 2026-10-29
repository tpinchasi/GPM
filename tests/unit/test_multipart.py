"""Uploads read strictly and rebuilt (D144).

A form the pool forwards must be read by the engine exactly as the pool read it, so anything a
form parser would have to guess about is refused, and what is forwarded is written afresh with
the pool's own boundary. Bodies here are built by hand, byte by byte, and by httpx's own encoder
— what the OpenAI SDK and most Python clients send.
"""

import random
import secrets

import httpx
import pytest
from gpm_server.engines.multipart import BadUpload, Part, field, parse, rebuild

B = "Xb0undary"
CT = f"multipart/form-data; boundary={B}"
DASH = b"--" + B.encode()
CLOSE = DASH + b"--\r\n"

#: What a refusal must never repeat back: the client's own values.
SECRET_MODEL = "private-model-name"
SECRET_FILE = "private-recording.wav"

MODEL_HEAD = 'Content-Disposition: form-data; name="model"'
FILE_HEAD = f'Content-Disposition: form-data; name="file"; filename="{SECRET_FILE}"'

#: A recording whose bytes look like framing: line breaks, dashes, the boundary itself where it
#: is not a delimiter (not after a line break), a cut-off delimiter, and header-like lines.
AUDIO = (
    b"RIFF\x00\x01\xff\xfe\r\n--\r\n\r\nx" + DASH + b"--\r\n-" + DASH + b"\r\n"
    b"\r\n--Xb0und\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n\x00\x80end"
)


def part(*lines: str | bytes, data: bytes = b"") -> bytes:
    raw = [line.encode() if isinstance(line, str) else line for line in lines]
    return b"\r\n".join(raw) + b"\r\n\r\n" + data


def model(value: bytes = SECRET_MODEL.encode(), *extra: str | bytes) -> bytes:
    return part(MODEL_HEAD, *extra, data=value)


def audio(*extra: str | bytes, data: bytes = AUDIO) -> bytes:
    return part(FILE_HEAD, "Content-Type: audio/wav", *extra, data=data)


def text(name: str, value: bytes) -> bytes:
    return part(f'Content-Disposition: form-data; name="{name}"', data=value)


def form(*parts: bytes, boundary: str = B) -> bytes:
    dash = b"--" + boundary.encode()
    return b"".join(dash + b"\r\n" + p + b"\r\n" for p in parts) + dash + b"--\r\n"


GOOD = form(model(), audio())


def refused(content_type: str, body: bytes, **kwargs) -> str:
    with pytest.raises(BadUpload) as caught:
        parse(content_type, body, **kwargs)
    why = caught.value.why
    assert why and str(caught.value) == why
    assert SECRET_MODEL not in why and SECRET_FILE not in why
    return why


def from_httpx(files, data=None) -> tuple[str, bytes]:
    request = httpx.Request("POST", "http://pool/v1/audio/transcriptions", files=files, data=data)
    return request.headers["content-type"], request.read()


# --- what real clients send is read ---


def test_an_upload_as_httpx_encodes_it_is_read():
    """httpx writes the plain fields first and the file after: the OpenAI SDK's own shape."""
    content_type, body = from_httpx(
        files={"file": (SECRET_FILE, AUDIO, "audio/wav")},
        data={"model": SECRET_MODEL, "language": "en"},
    )
    assert parse(content_type, body) == [
        Part("model", SECRET_MODEL.encode()),
        Part("language", b"en"),
        Part("file", AUDIO, SECRET_FILE, "audio/wav"),
    ]


def test_the_model_may_come_after_the_file():
    content_type, body = from_httpx(
        files=[("file", (SECRET_FILE, AUDIO, "audio/wav")), ("model", (None, SECRET_MODEL))],
    )
    parts = parse(content_type, body)
    assert [p.name for p in parts] == ["file", "model"]
    assert field(parts, "model") == SECRET_MODEL
    assert parts[0].data == AUDIO


def test_a_hand_built_form_is_read_byte_for_byte():
    """Every byte of the file survives, including ones that look like the form's own framing."""
    assert parse(CT, GOOD) == [
        Part("model", SECRET_MODEL.encode()),
        Part("file", AUDIO, SECRET_FILE, "audio/wav"),
    ]


def test_a_quoted_boundary_is_read():
    boundary = "gpm test:(1)+,./=?'_-"
    body = form(model(), audio(), boundary=boundary)
    assert parse(f'multipart/form-data; boundary="{boundary}"', body)[1].data == AUDIO


def test_the_content_type_is_read_ignoring_letter_case():
    assert len(parse(f"Multipart/Form-Data ;  Boundary={B}", GOOD)) == 2


def test_a_boundary_of_seventy_characters_is_read():
    boundary = "b" * 70
    body = form(model(), boundary=boundary)
    assert parse(f"multipart/form-data; boundary={boundary}", body) == [
        Part("model", SECRET_MODEL.encode())
    ]


def test_a_form_shaped_as_dotnet_writes_it_is_read():
    """.NET quotes its boundary, leaves names unquoted and labels text fields as UTF-8 text."""
    boundary = "0f4c9a2e-6b1d-4d7e-9a35-2c1e8f7b6a10"
    body = form(
        part("Content-Type: text/plain; charset=utf-8",
             "Content-Disposition: form-data; name=model", data=SECRET_MODEL.encode()),
        part("Content-Disposition: form-data; name=file; filename=a.wav", data=b"\x00\x01"),
        boundary=boundary,
    )
    parts = parse(f'multipart/form-data; boundary="{boundary}"', body)
    assert field(parts, "model") == SECRET_MODEL
    assert parts[1] == Part("file", b"\x00\x01", "a.wav")


def test_a_form_may_end_without_a_final_line_break():
    assert len(parse(CT, GOOD[:-2])) == 2


def test_empty_values_are_read():
    """An empty file — a browser's empty file input sends `filename=""` — is still a file."""
    body = form(model(b""), part('Content-Disposition: form-data; name="file"; filename=""'))
    assert parse(CT, body) == [Part("model", b""), Part("file", b"", "")]


def test_exactly_the_most_parts_allowed_are_read():
    body = form(model(), *(text(f"f{i}", b"x") for i in range(15)))
    assert len(parse(CT, body)) == 16
    assert len(parse(CT, form(model(), audio()), max_parts=2)) == 2


# --- refused: the content type ---


@pytest.mark.parametrize(("content_type", "reason"), [
    ("text/plain", "not multipart/form-data"),
    (f"multipart/mixed; boundary={B}", "not multipart/form-data"),
    ("multipart/form-data", "no boundary"),
    (f"{CT}; charset=utf-8", "other than boundary"),
    (f"multipart/form-data; charset=utf-8; boundary={B}", "other than boundary"),
    (f"{CT}; boundary={B}", "boundary is given twice"),
    (f"{CT}; BOUNDARY=other", "boundary is given twice"),
    (f"multipart/form-data; boundary*=utf-8''{B}", "extended form"),
    (f"multipart/form-data; boundary*0={B}", "extended form"),
    ("multipart/form-data; boundary=", "boundary is empty"),
    ('multipart/form-data; boundary=""', "boundary is empty"),
    ("multipart/form-data; boundary=" + "b" * 71, "longer than 70"),
    ("multipart/form-data; boundary=a:b", "RFC 2046"),
    ('multipart/form-data; boundary="a@b"', "RFC 2046"),
    ('multipart/form-data; boundary="ab "', "RFC 2046"),
    ("multipart/form-data; boundary=Xb0undäry", "RFC 2046"),
    (f"multipart/form-data; boundary = {B}", "parameter is malformed"),
    (f'multipart/form-data; boundary="{B}', "parameter is malformed"),
    (f"{CT};", "parameter is malformed"),
])
def test_a_content_type_that_could_be_read_two_ways_is_refused(content_type, reason):
    assert reason in refused(content_type, GOOD)


# --- refused: the framing ---


@pytest.mark.parametrize(("body", "reason"), [
    pytest.param(b"preamble\r\n" + GOOD, "does not start with its boundary", id="preamble"),
    pytest.param(b"\r\n" + GOOD, "does not start with its boundary", id="leading line break"),
    pytest.param(CLOSE, "no parts", id="no parts"),
    pytest.param(GOOD + b"epilogue", "bytes follow", id="epilogue"),
    pytest.param(GOOD + b"\r\n", "bytes follow", id="second line break after the close"),
    pytest.param(GOOD[:-len(CLOSE)], "not closed", id="no closing delimiter"),
    pytest.param(GOOD[:-len(b"--\r\n")], "not closed", id="cut after the last boundary"),
    pytest.param(GOOD[:-len(b"-\r\n")], "not closed", id="cut inside the closing dashes"),
    pytest.param(DASH + b"\r\n" + model() + b"\n" + CLOSE, "not closed", id="bare-LF delimiter"),
    pytest.param(
        DASH + b"\r\n" + model() + b"\r\n" + DASH + b" \r\n" + audio() + b"\r\n" + CLOSE,
        "other than a line break", id="padding after a delimiter",
    ),
    pytest.param(
        form(model(), audio(data=b"a\r\n" + DASH + b"-more")),
        "other than a line break", id="the boundary inside a file after a line break",
    ),
])
def test_a_form_framed_ambiguously_is_refused(body, reason):
    assert reason in refused(CT, body)


# --- refused: part headers ---


@pytest.mark.parametrize(("body", "reason"), [
    pytest.param(form(b"\r\n" + AUDIO, model()), "no headers", id="no headers"),
    pytest.param(
        DASH + b"\r\n" + MODEL_HEAD.encode() + b"\r\n", "too long or not terminated",
        id="headers never end",
    ),
    pytest.param(
        form(model(), part('Content-Disposition: form-data; name="file"; filename="'
                           + "a" * 5000 + '"', data=b"x")),
        "too long or not terminated", id="headers too long",
    ),
    pytest.param(
        form(model(SECRET_MODEL.encode(), b"\nContent-Type: text/plain"), audio()),
        "bare carriage return or line feed", id="bare LF",
    ),
    pytest.param(
        form(model(SECRET_MODEL.encode(), b"\rContent-Type: text/plain"), audio()),
        "bare carriage return or line feed", id="bare CR",
    ),
    pytest.param(
        form(part("Content-Disposition: form-data;", ' name="model"', data=b"m"), audio()),
        "folded", id="folded with a space",
    ),
    pytest.param(
        form(part("Content-Disposition: form-data;", '\tname="model"', data=b"m"), audio()),
        "folded", id="folded with a tab",
    ),
    pytest.param(
        form(model(), audio("Content-Transfer-Encoding: base64")), "transfer encoding",
        id="transfer encoding",
    ),
    pytest.param(
        form(model(), audio("X-Note: 1")), "other than Content-Disposition", id="other header",
    ),
    pytest.param(
        form(model(SECRET_MODEL.encode(), 'Content-Disposition: form-data; name="other"')),
        "header is given twice", id="two dispositions",
    ),
    pytest.param(
        form(model(), audio("content-type: audio/mpeg")), "header is given twice",
        id="two content types in different case",
    ),
    pytest.param(
        form(part("Content-Type: text/plain", data=b"x"), audio()), "no Content-Disposition",
        id="no disposition",
    ),
    pytest.param(form(model(SECRET_MODEL.encode(), "NoColon"), audio()), "header is malformed",
                 id="no colon"),
    pytest.param(
        form(part('Content-Disposition : form-data; name="model"', data=b"m")),
        "header is malformed", id="space before the colon",
    ),
    pytest.param(
        form(model(), part(b'Content-Disposition: form-data; name="file"; filename="a\x00.wav"')),
        "control character", id="NUL in a header",
    ),
    pytest.param(
        form(model(), part(b'Content-Disposition: form-data; name="file"; filename="a\x1b.wav"')),
        "control character", id="escape in a header",
    ),
    pytest.param(
        form(model(), part(b'Content-Disposition: form-data; name="file"; filename="\xe9.wav"')),
        "not UTF-8", id="header not UTF-8",
    ),
])
def test_part_headers_that_could_be_read_two_ways_are_refused(body, reason):
    assert reason in refused(CT, body)


@pytest.mark.parametrize(("disposition", "reason"), [
    ('attachment; name="model"', "not form-data"),
    ('name="model"', "not form-data"),
    ("form-data; name*=UTF-8''model", "extended parameter"),
    ("form-data; name=\"file\"; filename*=UTF-8''a.wav", "extended parameter"),
    ('form-data; name="model"; size=3', "unknown parameter"),
    ('form-data; name="model"; name="other"', "parameter is given twice"),
    ('form-data; name="model"; NAME="other"', "parameter is given twice"),
    (f'form-data; filename="{SECRET_FILE}"', "has no name"),
    ("form-data", "has no name"),
    ('form-data; name=""', "name is empty"),
    ("form-data; name=mo/del", "is malformed"),
    ('form-data; name="mo\\"del"', "backslash"),
    ('form-data; name="mo"del"', "is malformed"),
    ('form-data; name="model', "is malformed"),
    ('form-data; name ="model"', "is malformed"),
    ('form-data; name="model";', "is malformed"),
])
def test_a_disposition_that_could_be_read_two_ways_is_refused(disposition, reason):
    body = form(part(f"Content-Disposition: {disposition}", data=SECRET_MODEL.encode()), audio())
    assert reason in refused(CT, body)


@pytest.mark.parametrize(("content_type", "reason"), [
    ("multipart/mixed; boundary=inner", "itself a multipart form"),
    ("Multipart/Form-Data; boundary=inner", "itself a multipart form"),
    ("audio", "Content-Type is malformed"),
    ("audio/", "Content-Type is malformed"),
])
def test_a_file_part_of_the_wrong_kind_is_refused(content_type, reason):
    body = form(model(), part(FILE_HEAD, f"Content-Type: {content_type}", data=AUDIO))
    assert reason in refused(CT, body)


# --- refused: the fields ---


@pytest.mark.parametrize(("body", "reason"), [
    pytest.param(form(model(), audio(), text("model", b"other")), "same name", id="two models"),
    pytest.param(form(model(), text("MODEL", b"other"), audio()), "same name",
                 id="model and MODEL"),
    pytest.param(form(model(), text("language", b"en"), text("Language", b"fr")), "same name",
                 id="any field in two cases"),
    pytest.param(form(model(), audio(), audio()), "same name", id="two files"),
    pytest.param(form(text("MODEL", SECRET_MODEL.encode()), audio()), "another letter case",
                 id="MODEL alone"),
    pytest.param(
        form(part(f'{MODEL_HEAD}; filename="{SECRET_FILE}"', data=SECRET_MODEL.encode())),
        "model field is a file", id="model with a file name",
    ),
    pytest.param(form(part(f'{MODEL_HEAD}; filename=""', data=b"m")), "model field is a file",
                 id="model with an empty file name"),
    pytest.param(form(model(b"\xff\xfe"), audio()), "not UTF-8", id="model not UTF-8"),
    pytest.param(form(model(SECRET_MODEL.encode() + b"\r\nx"), audio()), "control character",
                 id="model with a line break"),
    pytest.param(form(model(b"m\x00"), audio()), "control character", id="model with NUL"),
    pytest.param(form(model("m\u0085".encode()), audio()), "control character",
                 id="model with a C1 control"),
    pytest.param(
        form(model(SECRET_MODEL.encode(), "Content-Type: application/octet-stream"), audio()),
        "plain UTF-8 text", id="model as bytes",
    ),
    pytest.param(
        form(model(SECRET_MODEL.encode(), "Content-Type: text/plain; charset=utf-16"), audio()),
        "plain UTF-8 text", id="model in another character set",
    ),
    pytest.param(form(model(), audio(), *(text(f"f{i}", b"x") for i in range(15))),
                 "too many parts", id="seventeen parts"),
])
def test_fields_that_could_be_read_two_ways_are_refused(body, reason):
    assert reason in refused(CT, body)


def test_the_limit_on_parts_is_the_callers():
    assert "too many parts" in refused(CT, form(model(), audio(), text("x", b"1")), max_parts=2)


def test_a_refusal_is_a_value_error_with_a_fixed_reason():
    assert issubclass(BadUpload, ValueError)
    assert BadUpload("the form is not closed").why == "the form is not closed"


# --- reading a field ---


def test_a_field_is_read_as_text():
    parts = parse(CT, form(model(), text("language", "français".encode()), audio()))
    assert field(parts, "model") == SECRET_MODEL
    assert field(parts, "language") == "français"
    assert field(parts, "prompt") is None


@pytest.mark.parametrize(("parts", "reason"), [
    ([Part("Stream", b"true")], "another letter case"),
    ([Part("stream", b"true", filename="s.txt")], "is a file"),
    ([Part("stream", b"\xff")], "not UTF-8"),
    ([Part("stream", b"true", content_type="text/plain; charset=utf-16")], "plain UTF-8 text"),
])
def test_a_field_the_pool_reads_is_refused_where_an_engine_could_read_it_otherwise(parts, reason):
    """A field the pool reads beside the model — whether to stream, say — holds to the same
    rule: one value, read the same way by both."""
    with pytest.raises(BadUpload) as caught:
        field(parts, "stream")
    assert reason in caught.value.why


# --- rebuilding ---


def test_a_form_is_rebuilt_in_one_canonical_shape(monkeypatch):
    monkeypatch.setattr(secrets, "token_hex", lambda n: "0" * (2 * n))
    body = form(
        part("content-disposition:FORM-DATA;NAME=model", data=b"m"),
        part("CONTENT-TYPE:   audio/wav  ",
             'Content-Disposition: form-data ; filename="a b.wav" ;name="file"', data=b"\x00"),
    )
    content_type, rebuilt = rebuild(parse(CT, body))
    boundary = "gpm-" + "0" * 32
    assert content_type == f"multipart/form-data; boundary={boundary}"
    assert rebuilt == (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n'
        "\r\n"
        "m\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a b.wav"\r\n'
        "Content-Type: audio/wav\r\n"
        "\r\n"
        "\x00\r\n"
        f"--{boundary}--\r\n"
    ).encode()


def test_rebuilding_rewrites_the_model_and_nothing_else():
    content_type, body = from_httpx(
        files=[("file", (SECRET_FILE, AUDIO, "audio/wav"))],
        data={"model": SECRET_MODEL, "language": "en", "prompt": "Grüße\r\n--"},
    )
    parts = parse(content_type, body)
    new_type, new_body = rebuild(parts, replace={"model": "served-build:tag"})
    assert new_type != content_type
    again = parse(new_type, new_body)
    assert field(again, "model") == "served-build:tag"
    assert [p for p in again if p.name != "model"] == [p for p in parts if p.name != "model"]
    assert [p.name for p in again] == [p.name for p in parts]


def test_a_rewritten_model_carries_no_content_type():
    body = form(model(SECRET_MODEL.encode(), "Content-Type: text/plain; charset=utf-8"))
    _, rebuilt = rebuild(parse(CT, body), replace={"model": "tag"})
    assert b"Content-Type" not in rebuilt


def test_rebuilding_never_adds_a_field():
    parts = parse(CT, form(audio()))
    with pytest.raises(KeyError):
        rebuild(parts, replace={"model": "tag"})
    with pytest.raises(KeyError):
        rebuild(parts, replace={"file": "not a file"})


def test_a_file_holding_the_old_boundary_round_trips_byte_for_byte():
    data = b"\r\n" + DASH + b"\r\n" + DASH + b"--\r\n--gpm-\r\n\r\n" + bytes(range(256))
    parts = [Part("model", b"m"), Part("file", data, "a.bin", "application/octet-stream")]
    content_type, body = rebuild(parts)
    assert parse(content_type, body) == parts


def test_a_boundary_found_in_a_file_is_drawn_again(monkeypatch):
    draws = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr(secrets, "token_hex", lambda n: next(draws))
    data = b"\r\n--gpm-" + b"a" * 32 + b"--\r\n"
    content_type, body = rebuild([Part("file", data, "x.bin")])
    assert content_type.endswith("gpm-" + "b" * 32)
    assert parse(content_type, body) == [Part("file", data, "x.bin")]


@pytest.mark.parametrize("bad", [
    Part('a"b', b""),
    Part("a\\b", b""),
    Part("", b""),
    Part("file", b"", filename="a\r\nb.wav"),
    Part("file", b"", content_type="audio/wav\r\nX-Injected: 1"),
])
def test_a_part_that_cannot_be_written_safely_is_not_written(bad):
    """Only a part built by hand can be one: `parse` never returns such a part."""
    with pytest.raises(ValueError):
        rebuild([bad])


def test_random_files_round_trip_byte_for_byte():
    pieces = [b"\r\n", b"\r", b"\n", b"--", b"\r\n--", DASH, b"--gpm-", b"\x00", b"\xff"]
    for seed in range(200):
        rng = random.Random(seed)
        size = rng.randrange(0, 4096)
        data = bytearray()
        while len(data) < size:
            framing = rng.random() < 0.3
            data += rng.choice(pieces) if framing else rng.randbytes(rng.randrange(1, 64))
        parts = [
            Part("model", b"m"),
            Part("file", bytes(data), "a.bin", "application/octet-stream"),
            Part("prompt", rng.randbytes(rng.randrange(0, 32))),
        ]
        rng.shuffle(parts)
        content_type, body = rebuild(parts)
        assert parse(content_type, body) == parts
