"""Relaying an uploaded file to the operator's private repository on the model hub (S15, D182).

Done once, when the operator chooses it: the supervisor commits the file to `owner/repo` with the
write token the operator typed (kept in the credential store, never in the file, a log or an
answer), and from then on the build is an ordinary hub file, `owner/repo/<file>.onnx`, that every
machine fetches as any build — the owner's machines with their own `HF_TOKEN`, a rented one with
the read-only token the operator gives for it.

The hub's own HTTP API, as its client library uses it: ask how the file is to be uploaded
(`preupload`), upload it by Git LFS where it says so (the batch API, then one PUT or a multipart
upload to the addresses it hands out), and commit it. An HTTP client, never a command line.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, AsyncIterator, Callable, Optional

import httpx

REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.onnx$")
_LFS = "application/vnd.git-lfs+json"


class RelayFailed(Exception):
    """The hub refused, said in its words — never with the token."""


def _said(response: httpx.Response) -> str:
    try:
        body = response.json()
        said = body.get("error") or body.get("message") or body
    except ValueError:
        said = response.text
    return f"{response.status_code}: {str(said)[:300]}"


async def whoami(client: httpx.AsyncClient, token: str) -> dict[str, Any]:
    """Who a token is, and whether it may write — asked of the hub, spending nothing."""
    response = await client.get("/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        raise RelayFailed(f"the hub refused the token ({_said(response)})")
    body = response.json()
    role = ((body.get("auth") or {}).get("accessToken") or {}).get("role")
    return {"user": body.get("name"), "role": role}


async def relay(
    client: httpx.AsyncClient, token: str, repo: str, file: str, digest: str, size: int,
    sample: bytes, chunks: Callable[[int, int], AsyncIterator[bytes]],
) -> str:
    """Commit the file to `repo` as `file`; the build's tag, `repo/file`. `chunks(start, length)`
    reads the file from the pool's store."""
    if not REPO.match(repo) or ".." in repo:
        raise RelayFailed(f"{repo!r} is not a hub repository: `owner/name`")
    if not FILE.match(file) or ".." in file:
        raise RelayFailed(f"{file!r} is not a file name at the repository's root ending in .onnx")
    auth = {"Authorization": f"Bearer {token}"}
    asked = await client.post(f"/api/models/{repo}/preupload/main", headers=auth, json={
        "files": [{"path": file, "sample": base64.b64encode(sample[:512]).decode(), "size": size}]})
    if asked.status_code != 200:
        raise RelayFailed(f"the hub would not take a file into {repo} ({_said(asked)})")
    mode = next((f.get("uploadMode") for f in asked.json().get("files", []) if f.get("path") == file), "lfs")
    if mode == "lfs":
        await _lfs_upload(client, auth, repo, digest, size, chunks)
        operation = {"key": "lfsFile", "value": {"path": file, "algo": "sha256", "oid": digest, "size": size}}
    else:
        content = b"".join([part async for part in chunks(0, size)])
        operation = {"key": "file", "value": {"path": file, "encoding": "base64", "content": base64.b64encode(content).decode()}}
    header = {"key": "header", "value": {"summary": f"Add {file} (uploaded to a GPM pool)", "description": ""}}
    lines = "\n".join(json.dumps(line) for line in (header, operation)) + "\n"
    committed = await client.post(f"/api/models/{repo}/commit/main", content=lines.encode(),
                                  headers={**auth, "Content-Type": "application/x-ndjson"})
    if committed.status_code not in (200, 201):
        raise RelayFailed(f"the hub did not commit {file} to {repo} ({_said(committed)})")
    return f"{repo}/{file}"


async def _lfs_upload(client: httpx.AsyncClient, auth: dict[str, str], repo: str, digest: str, size: int,
                      chunks: Callable[[int, int], AsyncIterator[bytes]]) -> None:
    batch = await client.post(f"/{repo}.git/info/lfs/objects/batch", headers={**auth, "Accept": _LFS, "Content-Type": _LFS},
                              json={"operation": "upload", "transfers": ["basic", "multipart"], "hash_algo": "sha256",
                                    "ref": {"name": "main"}, "objects": [{"oid": digest, "size": size}]})
    if batch.status_code != 200:
        raise RelayFailed(f"the hub's file store refused ({_said(batch)})")
    found = (batch.json().get("objects") or [{}])[0]
    if found.get("error"):
        raise RelayFailed(f"the hub's file store refused: {found['error']}")
    actions = found.get("actions") or {}
    upload: Optional[dict[str, Any]] = actions.get("upload")
    if upload is None:
        return  # the hub has these bytes already
    href, header = upload["href"], upload.get("header") or {}
    # The addresses the hub hands out carry their own signature: no token goes to them.
    if "chunk_size" in header:
        part_size = int(header["chunk_size"])
        urls = sorted((int(key), url) for key, url in header.items() if key.isdigit())
        parts = []
        for number, url in urls:
            start = (number - 1) * part_size
            body = b"".join([b async for b in chunks(start, min(part_size, size - start))])
            sent = await client.put(url, content=body)
            if sent.status_code not in (200, 201):
                raise RelayFailed(f"part {number} of the file was refused ({_said(sent)})")
            parts.append({"partNumber": number, "etag": sent.headers.get("etag", "")})
        done = await client.post(href, json={"oid": digest, "parts": parts})
    else:
        done = await client.put(href, content=chunks(0, size), headers={"Content-Length": str(size)})
    if done.status_code not in (200, 201):
        raise RelayFailed(f"the file was not stored by the hub ({_said(done)})")
    verify = actions.get("verify")
    if verify is not None:
        checked = await client.post(verify["href"], headers={**auth, **(verify.get("header") or {}), "Content-Type": _LFS},
                                    json={"oid": digest, "size": size})
        if checked.status_code not in (200, 201):
            raise RelayFailed(f"the hub could not verify the file ({_said(checked)})")
