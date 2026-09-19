"""The `tunnel` transport: a supervised local forward, with the engine never exposed.

The supervisor is exercised against a stand-in for `ssh -N -L` (tests/fakes/stub_forwarder.py),
so the whole path — allocate a port, run a child, wait for it to listen, dial the engine
through it, notice it die, bring it back — is covered without an SSH server. What the stand-in
cannot cover is the `ssh` command itself, so that is asserted option by option instead.
"""

import asyncio
import time

import httpx
import pytest
from fakes.fake_ollama import FakeOllama, unused_port
from fakes.harness import EngineSpec, ServerHandle, pool_harness, stub_ssh_command
from gpm_server.config import PoolConfig, TransportConfig
from gpm_server.transports.tunnel import SshTunnel, build_ssh_command

MODEL = "m1"


def tunnel_config(**overrides):
    base = {
        "type": "tunnel",
        "ssh_host": "gpu.example.net",
        "ssh_user": "ubuntu",
        "ssh_key": "~/.ssh/pool_key",
        "remote_port": 11434,
        "known_hosts": "/tmp/gpm-known-hosts",
    }
    return TransportConfig.model_validate({**base, **overrides})


# --- the command ---


def test_the_ssh_command_closes_every_interactive_path():
    command = build_ssh_command(tunnel_config(), 45999)
    joined = " ".join(command)

    assert command[0] == "ssh"
    assert "-N" in command  # no remote command: a forward and nothing else
    assert "BatchMode=yes" in joined  # never waits for a human
    assert "ExitOnForwardFailure=yes" in joined  # a forward that cannot bind is a failure
    assert "ServerAliveInterval=15" in joined  # a dead link exits instead of hanging
    assert command[-1] == "ubuntu@gpu.example.net"


def test_the_host_key_is_pinned_on_first_use_against_a_pool_owned_file():
    joined = " ".join(build_ssh_command(tunnel_config(), 45999))
    assert "StrictHostKeyChecking=accept-new" in joined
    assert "UserKnownHostsFile=/tmp/gpm-known-hosts" in joined


def test_the_forward_binds_loopback_only():
    command = build_ssh_command(tunnel_config(remote_host="10.0.0.9"), 45999)
    assert "-L" in command
    assert command[command.index("-L") + 1] == "127.0.0.1:45999:10.0.0.9:11434"


def test_an_identity_file_is_used_alone():
    command = build_ssh_command(tunnel_config(), 45999)
    assert "IdentitiesOnly=yes" in " ".join(command)
    assert command[command.index("-i") + 1].endswith("/.ssh/pool_key")


def test_the_user_may_be_left_to_ssh_config():
    command = build_ssh_command(tunnel_config(ssh_user=None), 45999)
    assert command[-1] == "gpu.example.net"


def test_a_non_default_ssh_port_is_passed():
    command = build_ssh_command(tunnel_config(ssh_port=2222), 45999)
    assert command[command.index("-p") + 1] == "2222"


# --- configuration ---


def test_a_tunnel_host_has_no_base_url():
    with pytest.raises(ValueError, match="no base_url"):
        tunnel_config(base_url="http://127.0.0.1:11434")


def test_a_tunnel_needs_to_be_told_where_to_go():
    with pytest.raises(ValueError, match="ssh_host"):
        TransportConfig.model_validate({"type": "tunnel", "remote_port": 11434})
    with pytest.raises(ValueError, match="remote_port"):
        TransportConfig.model_validate({"type": "tunnel", "ssh_host": "h"})


def test_a_tunnel_host_is_accepted_in_a_pool():
    config = PoolConfig.model_validate(
        {
            "pool": {"model_set": [MODEL]},
            "auth": {"app_keys": ["k"]},
            "hosts": [
                {
                    "id": "rented-1",
                    "kind": "fixed-remote",
                    "transport": {"type": "tunnel", "ssh_host": "h", "remote_port": 11434},
                }
            ],
        }
    )
    assert config.hosts[0].transport.type == "tunnel"


# --- the supervisor ---


@pytest.fixture
def engine():
    """A fake engine on its own port, standing in for one on the far side of the link."""
    from fakes.harness import BackgroundLoop

    loop = BackgroundLoop()
    fake = FakeOllama(resident={MODEL})
    server = ServerHandle(fake.app, loop)
    try:
        yield fake, server
    finally:
        server.stop()
        loop.stop()


async def test_the_forward_comes_up_and_carries_traffic(engine):
    fake, server = engine
    tunnel = SshTunnel(
        "rented-1",
        tunnel_config(remote_port=server.port, ssh_host="127.0.0.1"),
        command_builder=stub_ssh_command,
    )
    try:
        assert await tunnel.start(wait_s=10)
        assert tunnel.up
        assert tunnel.local_url == f"http://127.0.0.1:{tunnel.local_port}"

        async with httpx.AsyncClient(base_url=tunnel.local_url, timeout=10) as client:
            response = await client.get("/api/ps")
        assert response.status_code == 200
        assert response.json()["models"][0]["name"] == MODEL
    finally:
        await tunnel.stop()


async def test_a_dropped_link_is_re_established_on_the_same_local_port(engine):
    fake, server = engine
    tunnel = SshTunnel(
        "rented-1",
        tunnel_config(remote_port=server.port, ssh_host="127.0.0.1"),
        command_builder=stub_ssh_command,
    )
    try:
        assert await tunnel.start(wait_s=10)
        port_before = tunnel.local_port

        tunnel._process.kill()  # the link dies, as an outbid or a flaky network would kill it

        deadline = time.monotonic() + 15
        while tunnel.restarts == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert tunnel.restarts >= 1

        while not tunnel.up and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert tunnel.up
        assert tunnel.local_port == port_before  # the URL the router dials never moves

        async with httpx.AsyncClient(base_url=tunnel.local_url, timeout=10) as client:
            assert (await client.get("/api/ps")).status_code == 200
    finally:
        await tunnel.stop()


async def test_stopping_leaves_nothing_listening(engine):
    fake, server = engine
    tunnel = SshTunnel(
        "rented-1",
        tunnel_config(remote_port=server.port, ssh_host="127.0.0.1"),
        command_builder=stub_ssh_command,
    )
    assert await tunnel.start(wait_s=10)
    await tunnel.stop()

    assert not tunnel.up
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", tunnel.local_port)


async def test_a_forward_that_never_comes_up_is_reported_not_hung():
    tunnel = SshTunnel(
        "rented-1",
        tunnel_config(remote_port=unused_port(), ssh_host="127.0.0.1"),
        command_builder=lambda transport, port: ["false"],
    )
    try:
        assert not await tunnel.start(wait_s=2)
        assert not tunnel.up
    finally:
        await tunnel.stop()


# --- a pool over a tunnel host ---


def test_a_tunnel_host_serves_requests_like_any_other(monkeypatch):
    monkeypatch.setattr("gpm_server.transports.tunnel.build_ssh_command", stub_ssh_command)
    with pool_harness(
        [EngineSpec(id="rented-1", resident={MODEL}, kind="fixed-remote", transport="tunnel")],
        model_set=[MODEL],
    ) as pool:
        with pool.client() as client:
            status = client.get("/pool/status").json()
            response = client.post(
                "/api/chat",
                json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )

        host = status["hosts"][0]
        assert host["transport"] == "tunnel"
        assert host["state"] == "ready"
        assert response.status_code == 200
        assert response.headers["X-GPM-Host"] == "rented-1"

        # The tunnel belongs to the supervisor; the app-facing status says how a host is
        # reached, not how the pool keeps that link alive.
        assert pool.supervisor.tunnel_status("rented-1")["up"] is True


def test_a_tunnel_host_is_reached_over_loopback_only(monkeypatch):
    """The engine is never exposed: the router dials a local port, not the far machine."""
    monkeypatch.setattr("gpm_server.transports.tunnel.build_ssh_command", stub_ssh_command)
    with pool_harness(
        [EngineSpec(id="rented-1", resident={MODEL}, kind="fixed-remote", transport="tunnel")],
        model_set=[MODEL],
    ) as pool:
        dialled = str(pool.state.hosts[0].client.base_url)
        local_port = pool.supervisor.hosts["rented-1"].tunnel.local_port
        assert dialled == f"http://127.0.0.1:{local_port}"
