"""A pool running for real: fake engines and the router, each on its own port.

Tests drive it with ordinary synchronous clients, so what they exercise is the same path a
real app takes — real sockets, real streaming, real disconnects.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
import uvicorn
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.router.app import create_app
from gpm_server.supervisor import Supervisor

from .fake_ollama import FakeOllama, unused_port

APP_KEY = "test-app-key"
STUB_FORWARDER = Path(__file__).parent / "stub_forwarder.py"


def stub_ssh_command(transport: Any, local_port: int) -> list[str]:
    """What the tunnel supervisor runs instead of `ssh -N -L` in tests."""
    return [
        sys.executable,
        str(STUB_FORWARDER),
        str(local_port),
        transport.remote_host,
        str(transport.remote_port),
    ]


class BackgroundLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        while not self.loop.is_running():
            time.sleep(0.005)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=30)

    def spawn(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


class ServerHandle:
    def __init__(self, app: Any, loop: BackgroundLoop, port: Optional[int] = None):
        self.port = port or unused_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._future = loop.spawn(self._server.serve())
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError(f"server on port {self.port} did not start")
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.should_exit = True
        with contextlib.suppress(Exception):
            self._future.result(timeout=10)


@dataclasses.dataclass
class EngineSpec:
    id: str
    resident: set[str]
    kind: str = "local"
    workers: int = 1
    capabilities: list[str] = dataclasses.field(default_factory=list)
    priority: Optional[int] = None
    chunk_delay_s: float = 0.0
    chunks: int = 3
    #: "http" or "tunnel" — how the router reaches this engine.
    transport: str = "http"
    #: Tags on this engine's disk but not loaded; `resident` tags are on disk as well.
    available: set[str] = dataclasses.field(default_factory=set)
    #: "pinned" or "on_demand" — the host's residency policy in the pool's configuration.
    residency: str = "pinned"
    #: Which engine this machine runs: "ollama", or "vllm" for one that serves nothing until it
    #: is started with models on disk (D97). Only for machines the pool rents.
    engine: str = "ollama"


@dataclasses.dataclass
class RunningEngine:
    spec: EngineSpec
    fake: Any
    server: ServerHandle

    @property
    def base_url(self) -> str:
        return self.server.base_url


class PoolHarness:
    def __init__(
        self,
        engines: list[EngineSpec],
        *,
        model_set: list[str],
        catalog: Optional[dict[str, Any]] = None,
        queue_timeout_s: float = 30.0,
        probe_interval_s: float = 3600.0,
        upstream_connect_timeout_s: float = 2.0,
        upstream_read_timeout_s: float = 300.0,
        extra_hosts: Optional[list[dict[str, Any]]] = None,
        host_overrides: Optional[dict[str, dict[str, Any]]] = None,
        rentable: Optional[list[EngineSpec]] = None,
        rented: Optional[dict[str, Any]] = None,
        pool_settings: Optional[dict[str, Any]] = None,
        delivery: Optional[dict[str, Any]] = None,
        extra_config: Optional[dict[str, Any]] = None,
    ):
        self.loop = BackgroundLoop()
        self._tmp = tempfile.TemporaryDirectory()
        self.engines: dict[str, RunningEngine] = {}

        host_entries = []
        for spec in engines:
            fake = FakeOllama(
                resident=spec.resident, available=spec.available,
                chunk_delay_s=spec.chunk_delay_s, chunks=spec.chunks,
            )
            server = ServerHandle(fake.app, self.loop)
            self.engines[spec.id] = RunningEngine(spec=spec, fake=fake, server=server)
            if spec.transport == "tunnel":
                dialled: dict[str, Any] = {
                    "type": "tunnel",
                    "ssh_host": "127.0.0.1",
                    "remote_host": "127.0.0.1",
                    "remote_port": server.port,
                    "known_hosts": str(Path(self._tmp.name) / "known_hosts"),
                }
            else:
                dialled = {"type": "http", "base_url": server.base_url}
            entry: dict[str, Any] = {
                "id": spec.id,
                "kind": spec.kind,
                "workers": spec.workers,
                "capabilities": spec.capabilities,
                "transport": dialled,
            }
            if spec.priority is not None:
                entry["priority"] = spec.priority
            if spec.residency != "pinned":
                entry["residency"] = spec.residency
            for key, value in (host_overrides or {}).get(spec.id, {}).items():
                if isinstance(value, dict) and isinstance(entry.get(key), dict):
                    entry[key].update(value)
                else:
                    entry[key] = value
            host_entries.append(entry)
        # Hosts the harness does not run — a real engine someone else started.
        host_entries.extend(extra_hosts or [])

        # Engines the pool can *rent*: started here, but not configured as hosts. The fake
        # provider hands one out when it creates an instance.
        self.rentable: dict[str, RunningEngine] = {}
        rentable_urls: list[str] = []
        for spec in rentable or []:
            if spec.engine == "vllm":
                from .fake_vllm import FakeVllm

                fake = FakeVllm()
            else:
                fake = FakeOllama(
                    resident=spec.resident, available=spec.available,
                    chunk_delay_s=spec.chunk_delay_s, chunks=spec.chunks,
                )
            server = ServerHandle(fake.app, self.loop)
            self.rentable[spec.id] = RunningEngine(spec=spec, fake=fake, server=server)
            rentable_urls.append(server.base_url)

        rented_section = dict(rented) if rented else None
        if rented_section is not None and "providers" in rented_section:
            connections = {name: dict(conn) for name, conn in rented_section["providers"].items()}
            for conn in connections.values():
                conn["settings"] = {"engine_urls": rentable_urls, **(conn.get("settings") or {})}
            rented_section["providers"] = connections
        elif rented_section is not None:
            settings = dict(rented_section.get("provider_settings") or {})
            settings.setdefault("engine_urls", rentable_urls)
            rented_section["provider_settings"] = settings

        self.config = PoolConfig.model_validate(
            {
                "pool": {
                    "name": "test",
                    "model_set": model_set,
                    "queue_timeout_s": queue_timeout_s,
                    "probe_interval_s": probe_interval_s,
                    "upstream_connect_timeout_s": upstream_connect_timeout_s,
                    "upstream_read_timeout_s": upstream_read_timeout_s,
                    **({"delivery": delivery} if delivery else {}),
                    **(pool_settings or {}),
                },
                "auth": {"app_keys": [APP_KEY]},
                "catalog": catalog or {},
                "hosts": host_entries,
                **({"rented": rented_section} if rented_section else {}),
                **(extra_config or {}),
            }
        )
        # Tests never touch real ssh: every forward the pool opens runs the stand-in, and
        # every command it would run on a rented host is recorded instead.
        import gpm_server.transports.tunnel as tunnel_module

        self._real_ssh_command = tunnel_module.build_ssh_command
        tunnel_module.build_ssh_command = stub_ssh_command
        self.commands_on_hosts: list[tuple[str, str]] = []

        # Two halves sharing one database, as they are in production. The supervisor runs as
        # a task here rather than a second process; what the router depends on is the table.
        self.database = Database(Path(self._tmp.name) / "gpm.sqlite3")
        self.supervisor = Supervisor(self.config, self.database)
        if self.supervisor.fleet is not None:
            async def record(host, command):
                self.commands_on_hosts.append((host.host_id, command))
                return 0, ""

            self.supervisor.fleet.run_on_host = record
        self.loop.run(self.supervisor.start())
        self._supervisor_task = self.loop.spawn(self.supervisor.run_forever())

        self.app = create_app(self.config, database=self.database)
        self.server = ServerHandle(self.app, self.loop)
        self.url = self.server.base_url
        self._await_host_table()

    def _await_host_table(self, timeout: float = 15.0) -> None:
        """The router has seen the supervisor's first publish, so a test never races it."""
        deadline = time.monotonic() + timeout
        while len(self.state.hosts) < len(self.config.hosts):
            if time.monotonic() > deadline:
                raise RuntimeError("the router never picked up the published host table")
            time.sleep(0.02)

    # --- control ---

    @property
    def state(self) -> Any:
        return self.app.state.pool

    def reprobe(self) -> None:
        """One supervisor pass, then wait for the router to read the result."""
        self.loop.run(self.supervisor.pass_once())
        self.loop.run(self.state.registry.refresh())

    @property
    def rentable_urls(self) -> list[str]:
        """Every engine the market can hand out, whether or not it has been handed out yet."""
        return [engine.server.base_url for engine in self.rentable.values()]

    def restart_supervisor(self) -> None:
        """What `gpm restart` does: the process is replaced, the hosts stay rented.

        The new supervisor adopts what the old one published, as a fresh process does — this
        is where a restart under load is actually exercised (D61).
        """
        provider = self.supervisor.fleet.provider if self.supervisor.fleet else None
        # A real restart ends the old process, which releases the single-instance lock and
        # leaves the rented hosts alone. Stopping the task without closing the supervisor
        # would leave the lock held, and the new one would refuse to start — correctly.
        self.stop_supervisor()
        self.loop.run(self.supervisor.aclose())
        self.supervisor = Supervisor(self.config, self.database, provider=provider)
        if self.supervisor.fleet is not None:
            async def record(host, command):
                self.commands_on_hosts.append((host.host_id, command))
                return 0, ""

            self.supervisor.fleet.run_on_host = record
            self.supervisor.fleet.provider.reuse_engine_urls = True
        self.loop.run(self.supervisor.start())
        self._supervisor_task = self.loop.spawn(self.supervisor.run_forever())

    def stop_supervisor(self) -> None:
        """Leave the router serving from the last table it saw."""
        self.supervisor._stopping = True
        self._supervisor_task.cancel()

    def stop_engine(self, engine_id: str) -> None:
        self.engines[engine_id].server.stop()

    def request_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return [dict(row) for row in self.state.log.rows(limit)]

    def wait_for_log(self, count: int = 1, timeout: float = 5.0) -> list[dict[str, Any]]:
        """The log is written off the request path, so it lands just after the response."""
        deadline = time.monotonic() + timeout
        while True:
            rows = self.request_log()
            if len(rows) >= count or time.monotonic() > deadline:
                return rows
            time.sleep(0.02)

    # --- clients ---

    def client(self, *, key: Optional[str] = APP_KEY, timeout: float = 30.0) -> httpx.Client:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def direct_client(self, engine_id: str, timeout: float = 30.0) -> httpx.Client:
        """A client straight to one fake engine — the other half of a fidelity comparison."""
        return httpx.Client(base_url=self.engines[engine_id].base_url, timeout=timeout)

    def close(self) -> None:
        import gpm_server.transports.tunnel as tunnel_module

        tunnel_module.build_ssh_command = self._real_ssh_command
        self.server.stop()
        for engine in self.rentable.values():
            engine.server.stop()
        self._supervisor_task.cancel()
        with contextlib.suppress(Exception):
            self.loop.run(self.supervisor.aclose())
        self.database.close()
        for engine in self.engines.values():
            engine.server.stop()
        self.loop.stop()
        self._tmp.cleanup()


@contextlib.contextmanager
def pool_harness(engines: list[EngineSpec], **kwargs: Any) -> Iterator[PoolHarness]:
    harness = PoolHarness(engines, **kwargs)
    try:
        yield harness
    finally:
        harness.close()
