"""A program's own workload (D117, docs/stories/S6).

    pool = WorkloadProvisioner(base_url, provisioning_key)
    with pool.workload(model="gemma4:31b", latency_s=20, parallel=16, hours=6, max_spend=25) as w:
        reply = w.client.chat("gemma4:31b", [...])

The workload's key is made here and only its hash is sent: the pool never holds the key. The
pool knows a create by that hash, so one whose answer was lost is sent again until it is heard —
it is the same request, and one workload. Leaving the block ends the workload; a program that is
killed leaves it to the pool's idle cutoff once it serves, or to its lease.

Client certificates (`certs=True`) need the `certs` extra (`pip install gpm-client[certs]`): a
key pair is made here, a signing request sent, and the certificate the pool signs comes back.
The private key never leaves this machine; it is written to a directory only this user can read,
for the TLS library to load, and removed when the workload ends (a killed program leaves it).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import ssl
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx

from .client import PoolClient
from .config import env_url
from .errors import PoolAuthError, PoolError, PoolRequestError
from .transport import pool_transport


class WorkloadRefused(PoolError):
    """The pool would not create, plan or end it — its words in `detail`."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


def _certificate_request() -> tuple[bytes, str]:
    """(the private key as PEM, a signing request as PEM). Only the public key is used by the pool."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PoolError("client certificates need the certs extra: pip install 'gpm-client[certs]'") from exc
    key = ec.generate_private_key(ec.SECP256R1())
    csr = (x509.CertificateSigningRequestBuilder()
           .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gpm workload")]))
           .sign(key, hashes.SHA256()))
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return pem, csr.public_bytes(serialization.Encoding.PEM).decode()


class Workload:
    """One workload this program made: its name, a client bound to its key, and `end()`."""

    def __init__(self, provisioner: "WorkloadProvisioner", name: str, key: str, answer: dict,
                 client: PoolClient, cert_dir: Optional[Path]):
        self._provisioner = provisioner
        self.name = name
        self.key = key
        self.answer = answer
        self.client = client
        self._cert_dir = cert_dir
        self.ended = False

    @property
    def base_url(self) -> str:
        return self._provisioner.base_url

    @property
    def models(self) -> list[str]:
        """The models it serves: send any of them through `client` (D118)."""
        return list(self.answer.get("models") or [self.answer.get("model")])

    def state(self) -> dict:
        return self._provisioner._get(f"/pool/provisioning/workloads/{self.name}")

    def wait_until_serving(self, timeout_s: float = 1800.0, poll_s: float = 5.0) -> None:
        """Until one of its own hosts is ready. Meanwhile it may already be served on shared
        hosts, where the pool lets it borrow."""
        deadline = time.monotonic() + timeout_s
        while True:
            seen = self.state()
            if seen.get("state") == "serving":
                return
            if seen.get("state") in ("ending", "ended"):
                raise WorkloadRefused(f"workload {self.name} is {seen.get('state')}")
            if time.monotonic() >= deadline:
                raise PoolError(f"workload {self.name} was not serving within {timeout_s:g}s")
            time.sleep(poll_s)

    def end(self) -> None:
        if self.ended:
            return
        self.ended = True
        try:
            self._provisioner._post(f"/pool/provisioning/workloads/{self.name}/end", {})
        finally:
            self.client.close()
            if self._cert_dir is not None:
                shutil.rmtree(self._cert_dir, ignore_errors=True)

    def __enter__(self) -> "Workload":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end()


class WorkloadProvisioner:
    """Asks a pool for workloads with a provisioning key (`gpmp_…`), within its grant."""

    def __init__(self, base_url: Optional[str] = None, provisioning_key: Optional[str] = None, *,
                 verify: Any = True, timeout: float = 30.0, poll_s: float = 1.0):
        self.base_url = (base_url or env_url()).rstrip("/")
        self._key = provisioning_key or os.environ.get("GPM_PROVISIONING_KEY")
        if not self._key:
            raise PoolAuthError("a provisioning key is needed: pass it, or set GPM_PROVISIONING_KEY")
        self._verify = verify
        self._poll_s = poll_s
        # A CA file is turned into a context here: httpx no longer takes the path itself.
        http_verify = ssl.create_default_context(cafile=verify) if isinstance(verify, str) else verify
        # No retry transport: `_until` tries again itself, only where that cannot make a second workload.
        self._http = httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {self._key}"},
                                  timeout=timeout, verify=http_verify)

    # --- the calls ---

    def _check(self, response: httpx.Response) -> dict:
        if response.status_code == 401:
            raise PoolAuthError("the provisioning key was refused")
        if response.status_code >= 400:
            try:
                data = response.json()
            except ValueError:
                data = {}
            raise PoolRequestError(response.status_code, data.get("reason"), data.get("detail"))
        return response.json()

    def _get(self, path: str) -> dict:
        return self._check(self._http.get(path))

    def _post(self, path: str, body: dict) -> dict:
        return self._check(self._http.post(path, json=body))

    def _until(self, deadline: float, call: Any) -> dict:
        """`call()` until it answers: a lost connection, or the key's previous request still
        waiting, is tried again until `deadline`. Safe for a create: the pool takes the same key
        hash as the same request, so a create whose answer was lost is never made twice."""
        while True:
            try:
                return call()
            except httpx.TransportError:
                pass
            except PoolRequestError as exc:
                if exc.status_code != 429:
                    raise
            if time.monotonic() >= deadline:
                raise PoolError(f"the pool could not be asked within the time allowed ({self.base_url})")
            time.sleep(self._poll_s)

    def _ask(self, body: dict, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        request_id = self._until(deadline, lambda: self._post("/pool/provisioning/requests", body))["request_id"]
        while True:
            answered = self._until(deadline, lambda: self._get(f"/pool/provisioning/requests/{request_id}"))
            if answered["state"] == "refused":
                raise WorkloadRefused((answered.get("answer") or {}).get("detail") or "refused")
            if answered["state"] == "done":
                return answered["answer"]
            if time.monotonic() >= deadline:
                # A create answered after this is still bounded: unused, it is ended at its idle cutoff.
                raise PoolError(f"the pool did not answer request {request_id} within {timeout_s:g}s")
            time.sleep(self._poll_s)

    @staticmethod
    @staticmethod
    def _body(model: Optional[str], latency_s: Optional[float], parallel: Optional[int], hours: float,
              max_spend: float, machines: str, idle_end_minutes: Optional[float],
              models: Optional[Mapping[str, Mapping[str, Any]]] = None, placement: str = "auto") -> dict:
        """One model — `model`, `latency_s`, `parallel` — or several, each with its own target:
        `models={"chat": {"latency_s": 20, "parallel": 16}, "embed": {...}}` (D118)."""
        if (model is None) == (models is None):
            raise ValueError("name one model, or several with models= — not both, not neither")
        if models is not None:
            listed = []
            for name, target in models.items():
                if not isinstance(target, Mapping) or "latency_s" not in target or "parallel" not in target:
                    raise ValueError(f"models[{name!r}] needs latency_s and parallel")
                listed.append({"model": name, "latency_s": target["latency_s"], "parallel": target["parallel"]})
            body: dict[str, Any] = {"models": listed, "placement": placement}
        else:
            if latency_s is None or parallel is None:
                raise ValueError("a model needs latency_s and parallel")
            body = {"model": model, "latency_s": latency_s, "parallel": parallel}
        body.update({"hours": hours, "max_spend": max_spend, "machines": machines})
        if idle_end_minutes is not None:
            body["idle_end_minutes"] = idle_end_minutes
        return body

    def plan_workload(self, model: Optional[str] = None, *, latency_s: Optional[float] = None,
                      parallel: Optional[int] = None, hours: float, max_spend: float,
                      models: Optional[Mapping[str, Mapping[str, Any]]] = None, placement: str = "auto",
                      machines: str = "roi", timeout_s: float = 120.0) -> dict:
        """What creating it would do — its start, its first host, its price. Spends nothing."""
        body = {"kind": "plan", **self._body(model, latency_s, parallel, hours, max_spend, machines, None,
                                             models, placement)}
        return self._ask(body, timeout_s)["plan"]

    def workload(self, model: Optional[str] = None, *, latency_s: Optional[float] = None,
                 parallel: Optional[int] = None, hours: float, max_spend: float,
                 models: Optional[Mapping[str, Mapping[str, Any]]] = None, placement: str = "auto",
                 machines: str = "roi", idle_end_minutes: Optional[float] = None, certs: bool = False,
                 wait_until: str = "created", timeout_s: float = 1800.0) -> Workload:
        """Create a workload and hand back a `Workload` bound to it — a context manager that ends
        it on leaving. One model, or several with `models=` — each with its own latency target and
        answers at once, placed on the pool's hosts as it prices cheaper (D118). `wait_until` is
        "created" (return at once; it may be served on shared hosts while its own come up) or
        "serving" (wait until every model has hosts of its own)."""
        if wait_until not in ("created", "serving"):
            raise ValueError("wait_until is 'created' or 'serving'")
        key = "gpmw_" + secrets.token_hex(32)
        body = {"kind": "create", "key_hash": hashlib.sha256(key.encode()).hexdigest(),
                **self._body(model, latency_s, parallel, hours, max_spend, machines, idle_end_minutes,
                             models, placement)}
        private_pem = None
        if certs:
            private_pem, body["csr"] = _certificate_request()
        answer = self._ask(body, timeout_s)
        name = answer["workload"]
        cert_dir = None
        verify: Any = ssl.create_default_context(cafile=self._verify) if isinstance(self._verify, str) else self._verify
        if certs:
            if not answer.get("certificate"):
                raise WorkloadRefused("the pool did not sign a certificate for this workload")
            cert_dir = Path(tempfile.mkdtemp(prefix="gpm-workload-"))
            os.chmod(cert_dir, 0o700)
            (cert_dir / "client.pem").write_text(answer["certificate"])
            key_path = cert_dir / "client.key"
            handle = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(handle, "wb") as out:
                out.write(private_pem or b"")
            context = (ssl.create_default_context(cafile=self._verify) if isinstance(self._verify, str)
                       else ssl.create_default_context())
            context.load_cert_chain(str(cert_dir / "client.pem"), str(key_path))
            verify = context
        client = PoolClient(self.base_url, api_key=key,
                            transport=pool_transport(transport=httpx.HTTPTransport(verify=verify)))
        workload = Workload(self, name, key, answer, client, cert_dir)
        if wait_until == "serving":
            try:
                workload.wait_until_serving(timeout_s=timeout_s)
            except BaseException:
                workload.end()
                raise
        return workload

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "WorkloadProvisioner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
