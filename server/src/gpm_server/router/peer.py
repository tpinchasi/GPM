"""The client certificate a connection presented, handed to the application (D117).

Uvicorn verifies a client certificate in the TLS handshake (against the configured CA) but does
not put it in the request scope. It does copy a per-connection state dict into every request's
scope, which the application reads as `request.state`; so the protocol below puts the verified
certificate there when the connection is made. Checked against uvicorn 0.53 by
`tests/test_provisioning.py`, which serves real TLS with real certificates.

It leans on uvicorn's internals, so a later uvicorn may stop carrying the certificate. That fails
closed: a request with no certificate seen is refused (403) by a workload that requires one — the
test above catches it before a release.
"""

from __future__ import annotations

from typing import Any


def _with_peer(base: type) -> type:
    class PeerCertificateProtocol(base):  # type: ignore[misc, valid-type]
        def connection_made(self, transport: Any) -> None:  # type: ignore[override]
            super().connection_made(transport)
            ssl_object = transport.get_extra_info("ssl_object")
            der = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
            self.app_state = {**self.app_state, "peer_cert": der}

    PeerCertificateProtocol.__name__ = f"PeerCertificate{base.__name__}"
    return PeerCertificateProtocol


def peer_certificate_protocol() -> type:
    """The HTTP/1.1 protocol uvicorn would pick, carrying the peer certificate."""
    try:
        import httptools  # noqa: F401 - the faster parser, where installed
        from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

        return _with_peer(HttpToolsProtocol)
    except ImportError:
        from uvicorn.protocols.http.h11_impl import H11Protocol

        return _with_peer(H11Protocol)
