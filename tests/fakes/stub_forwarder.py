"""A stand-in for `ssh -N -L`: forward a local port to a target, and stay up until killed.

It gives the tunnel supervisor the same shape to watch as the real thing — a child process
that holds a listening socket and dies when the link dies — without needing an SSH server.
Run as: python stub_forwarder.py <local_port> <target_host> <target_port>
"""

from __future__ import annotations

import asyncio
import sys


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


async def main() -> None:
    local_port, target_host, target_port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
        except OSError:
            writer.close()
            return
        await asyncio.gather(
            _pipe(reader, upstream_writer),
            _pipe(upstream_reader, writer),
            return_exceptions=True,
        )

    server = await asyncio.start_server(handle, "127.0.0.1", local_port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
