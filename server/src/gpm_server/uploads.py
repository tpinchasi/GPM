"""Trained models' files uploaded to the pool (S15, D182).

A file is kept by its own sha256, as `<sha256>.onnx` in an owner-only directory beside the pool's
database, and a catalog build names it `upload/<sha256>`. Nothing else names a file here: a hash is
the only key, so no request can point the store anywhere else.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import AsyncIterator, Optional

DIGEST = re.compile(r"^[0-9a-f]{64}$")
TAG = re.compile(r"^upload/([0-9a-f]{64})$")
#: ONNX files are protocol buffers whose first field is the IR version: field 1, a varint.
_ONNX_FIRST_BYTE = 0x08


class UploadRefused(ValueError):
    """The file is not one the pool keeps: too large, empty, or not an ONNX file."""


def digest_of(tag: str) -> Optional[str]:
    found = TAG.match(tag) if isinstance(tag, str) else None
    return found.group(1) if found else None


class UploadStore:
    def __init__(self, directory: Path, max_bytes: int):
        self.directory = Path(directory)
        self.max_bytes = max_bytes

    def _ready(self) -> None:
        # On Windows the folder inherits the profile's permissions (D163); elsewhere it is made
        # owner-only below.
        self.directory.mkdir(parents=True, exist_ok=True)
        found = os.lstat(self.directory)
        mine = not hasattr(os, "getuid") or found.st_uid == os.getuid()
        if not stat.S_ISDIR(found.st_mode) or not mine:
            raise UploadRefused(f"{self.directory} is not a directory of this user's")
        if os.name != "nt":
            os.chmod(self.directory, 0o700)

    def path_of(self, digest: str) -> Optional[Path]:
        if not DIGEST.match(digest):
            return None
        path = self.directory / f"{digest}.onnx"
        return path if path.is_file() else None

    async def keep(self, chunks: AsyncIterator[bytes]) -> tuple[str, int]:
        """Write the file as it arrives, hashing it; keep it under its hash. Refused past the bound."""
        self._ready()
        incoming = self.directory / f".incoming-{os.getpid()}-{id(chunks)}"
        hasher, size, first = hashlib.sha256(), 0, None
        try:
            fd = os.open(incoming, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "wb") as out:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    if first is None:
                        first = chunk[0]
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise UploadRefused(f"the file is larger than models.max_upload_mb ({self.max_bytes // 2**20} MB)")
                    hasher.update(chunk)
                    out.write(chunk)
            if size == 0:
                raise UploadRefused("the file is empty")
            if first != _ONNX_FIRST_BYTE:
                raise UploadRefused("this is not an ONNX file (it does not start as one)")
            digest = hasher.hexdigest()
            os.replace(incoming, self.directory / f"{digest}.onnx")
            return digest, size
        finally:
            incoming.unlink(missing_ok=True)

    async def chunks(self, digest: str, size: int = 1 << 20) -> AsyncIterator[bytes]:
        path = self.path_of(digest)
        if path is None:
            raise UploadRefused(f"upload/{digest} is not in the pool's store")
        with open(path, "rb") as source:
            while True:
                chunk = source.read(size)
                if not chunk:
                    return
                yield chunk
