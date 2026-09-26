"""The `tunnel` transport: a supervised SSH local forward.

docs/spec/hosts-routing-capacity.md §1.3. The engine is never exposed — the router dials a
local port, and the pool keeps the forward alive. The host's key is recorded on first
connection and checked on every later one, against a known-hosts file the pool owns.

This drives the `ssh` client rather than speaking SSH itself: one ubiquitous, standardised
binary against a large dependency and a second implementation of the protocol. What it gives
up is structured errors — a failure is a process exit and a port that stops accepting, which is
what the supervisor watches. (The "never a CLI wrapper" rule is about *providers*, whose output
formats churn and whose errors arrive as prose; `ssh -N -L` produces no output to parse.)
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from pathlib import Path
from typing import Callable, Optional

from ..config import TransportConfig

log = logging.getLogger("gpm.tunnel")

Command = list[str]
CommandBuilder = Callable[[TransportConfig, int], Command]

_RESTART_BACKOFF_MAX_S = 30.0
#: How long a tunnel must hold before its restart backoff is forgiven. Coming up at all is not
#: enough: a tunnel that comes up and dies a second later would otherwise reconnect every
#: second for as long as the host lives. Seen live against a machine whose SSH was refusing —
#: sixteen reconnects in six minutes, which the provider answers by throttling authentication,
#: so the hammering is what keeps the tunnel down.
_STEADY_AFTER_S = 30.0


def allocate_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_ssh_command(transport: TransportConfig, local_port: int) -> Command:
    """`ssh -N -L` with every interactive path closed off.

    `BatchMode` so it never waits for a human, `ExitOnForwardFailure` so a forward that cannot
    bind is a failure rather than a silent no-op, `accept-new` so a first connection pins the
    host key and a changed key later is refused, and keepalives so a dead link exits instead of
    hanging a worker.
    """
    known_hosts = Path(transport.known_hosts).expanduser()
    command = [
        "ssh",
        "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "ConnectTimeout=10",
    ]
    if transport.ssh_key:
        command += ["-i", str(Path(transport.ssh_key).expanduser()), "-o", "IdentitiesOnly=yes"]
    command += ["-p", str(transport.ssh_port)]
    command += ["-L", f"127.0.0.1:{local_port}:{transport.remote_host}:{transport.remote_port}"]
    target = f"{transport.ssh_user}@{transport.ssh_host}" if transport.ssh_user else str(transport.ssh_host)
    command.append(target)
    return command


def build_ssh_exec_command(
    *,
    ssh_host: str,
    ssh_port: int,
    ssh_user: Optional[str],
    ssh_key: Optional[str],
    known_hosts: str,
    command: str,
) -> Command:
    """Run one command on a host and come back. Same guarantees as the forward: never
    interactive, host key pinned on first use, bounded."""
    base = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={Path(known_hosts).expanduser()}",
        "-o", "ConnectTimeout=10",
    ]
    if ssh_key:
        base += ["-i", str(Path(ssh_key).expanduser()), "-o", "IdentitiesOnly=yes"]
    base += ["-p", str(ssh_port)]
    base.append(f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host)
    base.append(command)
    return base


async def run_command(
    command: Command, timeout: float = 20.0, stdin: Optional[bytes] = None
) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(input=stdin), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return 124, "timed out"
    return process.returncode or 0, stdout.decode(errors="replace")


class SshTunnel:
    """Keeps one local forward up for the life of the router process.

    The local port is fixed for the host's lifetime, so the URL the router dials never changes
    — a reconnect replaces the process underneath the same address.
    """

    def __init__(
        self,
        host_id: str,
        transport: TransportConfig,
        command_builder: Optional[CommandBuilder] = None,
    ):
        self.host_id = host_id
        self.transport = transport
        self.local_port = transport.local_port or allocate_local_port()
        self.local_url = f"http://127.0.0.1:{self.local_port}"
        self.up = False
        self.last_error: Optional[str] = None
        self.restarts = 0
        self._command = (command_builder or build_ssh_command)(transport, self.local_port)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._stopping = False

    async def start(self, wait_s: float = 10.0) -> bool:
        """Bring the forward up, and keep bringing it back for as long as the pool runs."""
        Path(self.transport.known_hosts).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._supervise(), name=f"tunnel:{self.host_id}")
        return await self._wait_until_listening(wait_s)

    async def detach(self) -> None:
        """The supervisor is going. A forward it runs itself goes with it (D110: that is what
        the forwarder exists to change)."""
        await self.stop()

    async def stop(self) -> None:
        self._stopping = True
        await self._terminate()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.up = False

    async def _supervise(self) -> None:
        backoff = 1.0
        while not self._stopping:
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *self._command,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:  # no ssh binary, most likely
                self.last_error = str(exc)
                log.error("tunnel %s could not start: %s", self.host_id, exc)
                if self._stopping:
                    return
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RESTART_BACKOFF_MAX_S)
                continue

            came_up = await self._wait_until_listening(10.0)
            if came_up:
                self.last_error = None
                log.info("tunnel %s up on 127.0.0.1:%d", self.host_id, self.local_port)
            up_at = time.monotonic()

            stderr = await self._process.stderr.read() if self._process.stderr else b""
            await self._process.wait()
            if came_up and time.monotonic() - up_at >= _STEADY_AFTER_S:
                backoff = 1.0  # it worked for a while: this is a new problem, not the same one
            self.up = False
            if self._stopping:
                return
            self.last_error = stderr.decode(errors="replace").strip() or "ssh exited"
            self.restarts += 1
            log.warning("tunnel %s went down (%s); reconnecting", self.host_id, self.last_error)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RESTART_BACKOFF_MAX_S)

    async def _wait_until_listening(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stopping:
                return False  # stopped while coming up: nobody is waiting for this any more
            if self._process is not None and self._process.returncode is not None:
                return False
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", self.local_port)
            except OSError:
                await asyncio.sleep(0.05)
                continue
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            self.up = True
            return True
        return False

    async def _terminate(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
