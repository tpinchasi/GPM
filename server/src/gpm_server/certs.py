"""The pool's client certificates (D117): a small CA the operator keeps, signing one certificate
per workload for a program that asked with a signing request.

What is signed is the pool's, not the program's: only the public key is taken from the request
(after checking the program holds its private half); the subject, the workload it names, client
authentication only, not a CA, and a validity bounded by the workload's hours are written here.
Signing what was asked could hand a program a CA. The router never parses a certificate: it
compares the SHA-256 fingerprint of the one the TLS handshake verified with the one stored here.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import stat
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


class CertRefused(Exception):
    """A signing request that will not be signed — said in words."""


def fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def _key_usage(ca: bool) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=not ca, key_cert_sign=ca, crl_sign=ca, content_commitment=False,
        key_encipherment=False, data_encipherment=False, key_agreement=False,
        encipher_only=False, decipher_only=False,
    )


def make_ca(directory: str | Path, name: str = "GPM pool client CA", years: int = 5) -> tuple[Path, Path]:
    """A new client CA: its certificate (public) and its key (owner-readable only)."""
    folder = Path(directory).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    cert = (
        x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365 * years))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_key_usage(True), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = folder / "client-ca.pem", folder / "client-ca.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    handle = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "wb") as out:
        out.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    return cert_path, key_path


class ClientCA:
    def __init__(self, certfile: str | Path, keyfile: str | Path):
        keyfile = Path(keyfile).expanduser()
        if keyfile.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            # Whoever reads it can sign a certificate for any workload (threat model T19).
            raise CertRefused(f"{keyfile} is readable by others; run `chmod 600 {keyfile}` before starting the pool")
        self.cert = x509.load_pem_x509_certificate(Path(certfile).expanduser().read_bytes())
        self.key = serialization.load_pem_private_key(keyfile.read_bytes(), password=None)
        self.pem = self.cert.public_bytes(serialization.Encoding.PEM).decode()

    @staticmethod
    def public_key_of(csr_pem: str) -> ec.EllipticCurvePublicKey:
        """The key a signing request carries, once it is shown to be one the pool signs. The
        request is untrusted input: whatever the library raises on it is a refusal, never an
        error that escapes (D117)."""
        try:
            csr = x509.load_pem_x509_csr(csr_pem.encode())
            valid = csr.is_signature_valid
            public = csr.public_key()
        except Exception as exc:  # noqa: BLE001 - untrusted input; any failure is a refusal
            raise CertRefused(f"not a signing request the pool can read: {type(exc).__name__}") from exc
        if not valid:
            raise CertRefused("the signing request is not signed by the key it carries")
        if not isinstance(public, ec.EllipticCurvePublicKey):
            raise CertRefused("only elliptic-curve keys are signed")
        return public

    def sign(self, csr_pem: str, workload: str, hours: float) -> tuple[str, str]:
        """(certificate PEM, its fingerprint) for the workload `workload`, from `csr_pem`."""
        public = self.public_key_of(csr_pem)
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"workload:{workload}")]))
            .issuer_name(self.cert.subject).public_key(public)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(hours=max(hours, 0.1) + 1))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(_key_usage(False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()), critical=False)
            .sign(self.key, hashes.SHA256())
        )
        return cert.public_bytes(serialization.Encoding.PEM).decode(), fingerprint(cert.public_bytes(serialization.Encoding.DER))


def load_ca(certfile: Optional[str], keyfile: Optional[str], listener_certfile: Optional[str] = None) -> Optional[ClientCA]:
    """The pool's client CA, where one is configured. Refused when its key is readable by others,
    or when the router's listener trusts a different CA — every certificate it signed would then
    be refused, in silence."""
    if not (certfile and keyfile):
        return None
    ca = ClientCA(certfile, keyfile)
    if listener_certfile:
        trusted = x509.load_pem_x509_certificate(Path(listener_certfile).expanduser().read_bytes())
        if trusted != ca.cert:
            raise CertRefused(f"listen.client_ca_certfile ({listener_certfile}) is not the CA the pool signs with "
                              f"({certfile}); name the same certificate in both")
    return ca
