"""The `http` and `https` transports: the router only ever sees a URL and optional headers.

docs/spec/hosts-routing-capacity.md §1.3. Failures are classified identically across
transports, so host states never depend on which one a host uses.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..config import PoolSettings, TransportConfig


def build_client(
    transport: TransportConfig, settings: PoolSettings, base_url: Optional[str] = None
) -> httpx.AsyncClient:
    """`base_url` overrides the configured one — a tunnel host is dialled on its local port."""
    return httpx.AsyncClient(
        base_url=base_url or transport.base_url or "",
        headers=transport.auth_headers(),
        verify=transport.verify,
        cert=transport.client_cert,
        timeout=httpx.Timeout(
            connect=settings.upstream_connect_timeout_s,
            read=settings.upstream_read_timeout_s,
            write=settings.upstream_read_timeout_s,
            pool=settings.upstream_connect_timeout_s,
        ),
    )
