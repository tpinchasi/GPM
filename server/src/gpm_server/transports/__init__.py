from typing import Any, Callable

from .forwarded import ForwardedTunnel
from .http import build_client
from .tunnel import SshTunnel, build_ssh_command, build_ssh_exec_command, run_command


def tunnel_maker(config: Any, database: Any) -> Callable[..., Any]:
    """How this pool opens a forward: through the forwarder, when it has one (D110), or by
    running `ssh` itself, as it always did."""
    if config.forwarder.enabled:
        return lambda name, transport: ForwardedTunnel(name, transport, database)
    return SshTunnel


__all__ = [
    "ForwardedTunnel",
    "SshTunnel",
    "build_client",
    "build_ssh_command",
    "build_ssh_exec_command",
    "run_command",
    "tunnel_maker",
]
