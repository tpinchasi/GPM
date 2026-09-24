"""A real browser, driven: headless Chrome over its DevTools protocol, with no library needed.

The console's Python tests prove the API behind a screen, and one browser test proves the page
parses. Neither proves a screen *works* — that choosing one thing redraws another, or that a
button sends what it should. The engine editor showed why that matters: its checkboxes updated
the draft without redrawing, so the field they should have revealed never appeared, and every
other test passed.

Only the standard library: the protocol is JSON over a websocket, and a websocket client is a
handshake and a frame format.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional


def a_browser() -> Optional[str]:
    found = shutil.which("google-chrome") or shutil.which("chromium")
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    return found or (mac if Path(mac).exists() else None)


class Page:
    """One tab: evaluate JavaScript in it, and wait for things to become true."""

    def __init__(self, url: str):
        host_port, path = url[len("ws://"):].split("/", 1)
        host, port = host_port.split(":")
        self._sock = socket.create_connection((host, int(port)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall(
            f"GET /{path} HTTP/1.1\r\nHost: {host_port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        head = b""
        while b"\r\n\r\n" not in head:
            head += self._sock.recv(4096)
        head, self._buffer = head.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise RuntimeError(f"the browser refused the connection: {head[:200]!r}")
        self._next = 0

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._sock.close()

    def _send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message).encode()
        header = bytearray([0x81])
        if len(data) < 126:
            header.append(0x80 | len(data))
        elif len(data) < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", len(data))
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", len(data))
        mask = os.urandom(4)
        self._sock.sendall(bytes(header) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _exact(self, n: int) -> bytes:
        while len(self._buffer) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise EOFError("the browser closed the connection")
            self._buffer += chunk
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def _receive(self) -> dict[str, Any]:
        message = b""
        while True:
            first, second = self._exact(2)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._exact(8))[0]
            payload = self._exact(length)
            if (first & 0x0F) in (0x9, 0xA):
                continue  # ping and pong carry nothing for us
            message += payload
            if first & 0x80:
                return json.loads(message)

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self._next += 1
        mine = self._next
        self._send({"id": mine, "method": method, "params": params})
        while True:
            message = self._receive()
            if message.get("id") == mine:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def js(self, expression: str) -> Any:
        """Evaluate in the page and return the value; a thrown error is raised here."""
        result = self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            raise RuntimeError((details.get("exception") or {}).get("description") or details.get("text"))
        return (result.get("result") or {}).get("value")

    def until(self, expression: str, *, within: float = 20.0, what: str = "") -> None:
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            if self.js(expression):
                return
            time.sleep(0.2)
        raise AssertionError(f"timed out after {within:g}s waiting for {what or expression}")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def open_page(url: str) -> Iterator[Page]:
    """Start a headless browser on a fresh profile, open `url`, and hand back the tab."""
    browser = a_browser()
    if browser is None:
        raise RuntimeError("no browser here")
    port = _free_port()
    with tempfile.TemporaryDirectory() as profile:
        process = subprocess.Popen(
            [browser, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
             f"--remote-debugging-port={port}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        page = None
        try:
            deadline = time.monotonic() + 60
            target = None
            while target is None and time.monotonic() < deadline:
                try:
                    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2))
                    target = next((t for t in targets if t.get("type") == "page"), None)
                except OSError:
                    time.sleep(0.2)
            if target is None:
                raise RuntimeError("the browser never offered a page to drive")
            page = Page(target["webSocketDebuggerUrl"])
            page.call("Page.enable")
            page.call("Page.navigate", url=url)
            yield page
        finally:
            if page is not None:
                page.close()
            process.kill()
            process.wait(timeout=10)
