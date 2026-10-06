"""Programs creating their own workloads (D117, stories/S6): a provisioning key and its grant,
requests the router records and the supervisor answers, workload keys made in the program with
only their hash sent, idle-end, and client certificates signed by the pool's CA.

The real supervisor, router, control API and SDK over one database; the fake provider; fake
engines. The certificate tests serve the router over real TLS with certificates made here.
Nothing here spends money.
"""

import asyncio
import datetime
import hashlib
import secrets
import ssl
import time

import httpx
import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fakes.harness import APP_KEY, EngineSpec, ServerHandle, pool_harness, unused_port
from gpm_client import WorkloadProvisioner, WorkloadRefused
from gpm_server.certs import make_ca
from gpm_server.cli import listener_options
from gpm_server.config import PoolConfig
from gpm_server.providers import default_offer
from gpm_server.router.app import create_app
from gpm_server.supervisor.control import create_control_app

ADMIN = "gpmx_provisioning_admin"
SHARED, BIG = "m1", "big"
CATALOG = {BIG: {"variants": [{"tag": BIG, "size_gb": 4}]}}


def harness(tmp_path=None, listen=None, provisioning=None):
    extra = {
        "auth": {"app_keys": [APP_KEY], "admin_keys": [ADMIN]},
        "limits": {"max_rented_hosts": 4},
        "workloads": {"min_reliability": 0.0},
        "provisioning": {"enabled": True, "max_spend_per_day": 10.0, "request_poll_s": 0.05, **(provisioning or {})},
    }
    if listen:
        extra["listen"] = listen
    return pool_harness(
        [EngineSpec(id="laptop", resident={SHARED, BIG}, workers=4)],
        model_set=[SHARED, BIG], catalog=CATALOG,
        rentable=[EngineSpec(id="market-1", resident={BIG}, workers=4),
                  EngineSpec(id="market-2", resident={BIG}, workers=4)],
        rented={"provider": "fake", "workers": 2, "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 1.0},
                "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0}},
        extra_config=extra,
    )


@pytest.fixture
def pool():
    with harness() as h:
        h.supervisor.fleet.provider.offers = [default_offer("o-1", "m-1", min_bid_hourly=0.2),
                                              default_offer("o-2", "m-2", min_bid_hourly=0.2)]
        control = ServerHandle(create_control_app(h.supervisor, h.config), h.loop)
        h.control_url = control.base_url
        try:
            yield h
        finally:
            control.stop()


def admin(pool):
    return httpx.Client(base_url=pool.control_url, headers={"Authorization": f"Bearer {ADMIN}"}, timeout=30)


def grant(pool, name="evals", **overrides):
    body = {"name": name, "models": [BIG], "max_open": 2, "max_spend": 5.0, "max_spend_per_day": 8.0,
            "max_hours": 4, **overrides}
    with admin(pool) as control:
        answer = control.post("/pool/provisioners", json=body)
    assert answer.status_code == 201, answer.text
    return answer.json()["key"]


def stop_answering(pool):
    """Stop the supervisor's watcher, so a request stays pending for as long as a test needs."""
    watcher = pool.supervisor._request_watcher
    if watcher is not None:
        pool.loop.loop.call_soon_threadsafe(watcher.cancel)
        time.sleep(0.1)


def refresh(pool):
    pool.loop.run(pool.state.registry.refresh())


def provisioner(pool, key):
    refresh(pool)
    return WorkloadProvisioner(pool.url, key, poll_s=0.05)


SPEC = {"latency_s": 30, "parallel": 2, "hours": 1, "max_spend": 2.0}


def test_a_program_creates_uses_and_ends_its_own_workload(pool):
    key = grant(pool)
    with provisioner(pool, key) as pool_side:
        plan = pool_side.plan_workload(BIG, **SPEC)
        assert plan["hosts_at_start"] == 1 and plan["within_grant"]
        with pool_side.workload(BIG, **SPEC) as w:
            assert w.name.startswith("evals-") and w.key.startswith("gpmw_")
            refresh(pool)
            reply = w.client.chat(BIG, [{"role": "user", "content": "hi"}])
            assert reply.content
            made = pool.supervisor.workloads.get(w.name)
            assert made.provisioner == "evals" and made.idle_end_minutes == 15
            assert pool.supervisor.leases.get(made.lease_id).max_spend == 2.0
        refresh(pool)
        for _ in range(40):
            if pool.supervisor.workloads.get(w.name).state in ("ending", "ended"):
                break
            time.sleep(0.05)
        assert pool.supervisor.workloads.get(w.name).state in ("ending", "ended"), "leaving the block ends it"
    raw = (pool.database.path).read_bytes()
    assert w.key.encode() not in raw, "the program's key never reached the pool"


def test_a_create_sent_twice_is_one_workload(pool):
    key = grant(pool)
    refresh(pool)
    key_hash = hashlib.sha256(("gpmw_" + secrets.token_hex(32)).encode()).hexdigest()
    body = {"kind": "create", "key_hash": key_hash, "model": BIG, **SPEC}
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {key}"}) as http:
        first = http.post("/pool/provisioning/requests", json=body).json()
        again = http.post("/pool/provisioning/requests", json=body).json()
    assert first["request_id"] == again["request_id"] and again["new"] is False
    pool.loop.run(pool.supervisor.provisioning.answer_pending())
    made = [w for w in pool.supervisor.workloads.store.all() if w.provisioner == "evals"]
    assert len(made) == 1 and len(pool.supervisor.leases.open_leases()) == 1


def ask(pool, key, **body):
    refresh(pool)
    with WorkloadProvisioner(pool.url, key, poll_s=0.05) as side:
        return side.workload(BIG, **{**SPEC, **body})


@pytest.mark.parametrize("change, words", [
    ({"max_spend": 9.0}, "at most $5.00 per workload"),
    ({"hours": 9}, "at most 4 hours"),
    ({"machines": "nope"}, "may rent"),
    ({"idle_end_minutes": 500}, "idle cutoff of at most"),
])
def test_what_the_grant_does_not_allow_is_refused_in_words(pool, change, words):
    key = grant(pool)
    with pytest.raises(WorkloadRefused, match=words.replace("$", r"\$")):
        ask(pool, key, **change)
    assert pool.supervisor.leases.open_leases() == []


def test_a_model_outside_the_grant_is_refused(pool):
    key = grant(pool)
    refresh(pool)
    with WorkloadProvisioner(pool.url, key, poll_s=0.05) as side, pytest.raises(WorkloadRefused, match="may create workloads for"):
        side.workload(SHARED, **SPEC)


def test_the_day_counts_what_was_committed(pool):
    key = grant(pool, max_spend_per_day=5.0)
    ask(pool, key, max_spend=3.0)
    with pytest.raises(WorkloadRefused, match="committed \\$3.00 in the last 24 hours"):
        ask(pool, key, max_spend=3.0)


def test_an_ended_workload_counts_what_it_spent_not_its_budget(pool):
    key = grant(pool, max_spend_per_day=5.0)
    first = ask(pool, key, max_spend=3.0)
    made = pool.supervisor.workloads.get(first.name)
    pool.supervisor.fleet.spend.record(lease_id=made.lease_id, host_id="h-gone", source="estimate", amount=0.4)
    with pytest.raises(WorkloadRefused, match="committed \\$3.00"):
        ask(pool, key, max_spend=3.0)       # still open: its whole budget counts
    pool.database.execute("UPDATE workloads SET state = 'ended', ended_at = ? WHERE name = ?", (time.time(), first.name))
    assert pool.supervisor.provisioning.usage("evals")["committed_today"] == 0.4
    ask(pool, key, max_spend=3.0)           # $0.40 spent + $3.00 fits the $5.00 a day


def test_the_pool_wide_day_bounds_every_key_together(pool):
    first, second = grant(pool, name="one", max_spend_per_day=8.0), grant(pool, name="two", max_spend_per_day=8.0)
    ask(pool, first, max_spend=5.0)
    ask(pool, second, max_spend=4.0)
    with pytest.raises(WorkloadRefused, match="pool's \\$10.00 a day"):
        ask(pool, second, max_spend=2.0)


def test_a_provisioning_key_never_requests_a_completion_nor_reaches_the_control_api(pool):
    key = grant(pool)
    refresh(pool)
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {key}"}) as http:
        chat = http.post("/v1/chat/completions", json={"model": BIG, "messages": []})
        assert chat.status_code == 403 and chat.json()["reason"] == "provisioning_key"
        assert http.get("/pool/status").status_code == 403
    with httpx.Client(base_url=pool.control_url, headers={"Authorization": f"Bearer {key}"}) as http:
        assert http.get("/pool/leases").json()["error"] == "provisioning_key_refused"
    with admin(pool) as control:
        listed = control.get("/pool/provisioners").json()["provisioners"]
    assert key not in str(listed), "the key is shown once, at creation"


def test_a_key_sees_and_ends_only_what_it_made(pool):
    mine, theirs = grant(pool, name="mine"), grant(pool, name="theirs")
    w = ask(pool, mine)
    refresh(pool)
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {theirs}"}) as http:
        assert http.get(f"/pool/provisioning/workloads/{w.name}").status_code == 404
        assert http.post(f"/pool/provisioning/workloads/{w.name}/end").status_code == 404


class LosesFirstAnswer(httpx.HTTPTransport):
    """Delivers the first create, then loses its answer — as a dropped connection would."""

    def __init__(self):
        super().__init__()
        self.lost = 0

    def handle_request(self, request):
        response = super().handle_request(request)
        if request.method == "POST" and request.url.path.endswith("/requests") and not self.lost:
            self.lost += 1
            response.close()
            raise httpx.ReadError("connection lost", request=request)
        return response


def test_a_create_whose_answer_was_lost_is_sent_again_and_made_once(pool):
    key = grant(pool)
    with provisioner(pool, key) as side:
        lossy = LosesFirstAnswer()
        side._http = httpx.Client(base_url=side.base_url, headers={"Authorization": f"Bearer {key}"}, transport=lossy)
        w = side.workload(BIG, **SPEC)
        assert lossy.lost == 1
        made = [x for x in pool.supervisor.workloads.store.all() if x.provisioner == "evals"]
        assert [x.name for x in made] == [w.name], "one workload, and the program knows its name"


def test_one_request_waits_at_a_time_and_a_hash_belongs_to_its_key(pool):
    """Every request may cost a market search the whole pool waits on; a key hash another key
    sent is refused, never answered with that key's request (D117)."""
    mine, theirs = grant(pool), grant(pool, name="other")
    refresh(pool)
    stop_answering(pool)
    key_hash = hashlib.sha256(("gpmw_" + secrets.token_hex(32)).encode()).hexdigest()
    body = {"kind": "create", "key_hash": key_hash, "model": BIG, **SPEC}
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {mine}"}) as http:
        first = http.post("/pool/provisioning/requests", json=body)
        assert first.status_code == 202
        busy = http.post("/pool/provisioning/requests", json={"kind": "plan", "model": BIG, **SPEC})
        assert busy.status_code == 429 and busy.json()["reason"] == "request_pending"
        other_create = {**body, "key_hash": "0" * 64}
        assert http.post("/pool/provisioning/requests", json=other_create).status_code == 429
        assert http.post("/pool/provisioning/requests", json=body).json()["request_id"] == first.json()["request_id"], \
            "the same create sent again is still the same request"
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {theirs}"}) as http:
        taken = http.post("/pool/provisioning/requests", json=body)
    assert taken.status_code == 409 and taken.json()["reason"] == "key_hash_taken"


def test_an_unused_workload_is_ended(pool):
    key = grant(pool)
    w = ask(pool, key, idle_end_minutes=1)
    an_hour_ago = time.time() - 3600
    workloads = pool.supervisor.workloads
    # Still preparing, however long ago it was made: a program waiting on a slow start is not idle.
    pool.database.execute("UPDATE workloads SET state = 'preparing', created_at = ?, serving_at = NULL WHERE name = ?",
                          (an_hour_ago, w.name))
    assert workloads._idle_past(workloads.get(w.name), time.time()) is None
    # Serving for an hour with nothing asked: ended.
    pool.database.execute("UPDATE workloads SET state = 'serving', serving_at = ? WHERE name = ?", (an_hour_ago, w.name))
    pool.reprobe()
    assert pool.supervisor.workloads.get(w.name).state in ("ending", "ended")
    assert any(e["kind"] == "workload_ending" and "unused" in e["summary"] for e in pool.supervisor.events.recent(50))


def test_a_programs_workload_key_is_never_rotated_by_the_pool(pool):
    key = grant(pool)
    w = ask(pool, key)
    with admin(pool) as control:
        answer = control.post(f"/pool/workloads/{w.name}/keys")
    assert answer.status_code == 400 and "holds its only key" in answer.json()["detail"]


def test_revoking_a_key_can_end_what_it_made(pool):
    key = grant(pool)
    w = ask(pool, key)
    with admin(pool) as control:
        answer = control.delete("/pool/provisioners/evals?end_workloads=true").json()
    assert answer["ended"] == [w.name]
    refresh(pool)
    with pytest.raises(Exception):
        WorkloadProvisioner(pool.url, key, poll_s=0.05).plan_workload(BIG, **SPEC)


def test_a_grant_is_refused_in_words():
    """Checked when made, not when used."""
    with harness() as h:
        with pytest.raises(Exception, match="names the models"):
            from gpm_server.provisioning_store import Grant

            h.supervisor.provisioning.grant("empty", Grant(models=()))


def test_provisioning_needs_a_pool_wide_cap():
    with pytest.raises(ValueError, match="pool-wide daily cap"):
        PoolConfig.model_validate({"pool": {"name": "t", "model_set": ["a"]}, "auth": {"app_keys": ["k"]},
                                   "hosts": [{"id": "h", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
                                   "provisioning": {"enabled": True}})


# --- client certificates, over real TLS ---


def _server_certificate(directory):
    """A server CA and a certificate for 127.0.0.1, as a pool's listener would have."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test server CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
          .not_valid_after(now + datetime.timedelta(days=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=False, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False, key_agreement=False,
                                       encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
          .sign(key, hashes.SHA256()))
    server_key = ec.generate_private_key(ec.SECP256R1())
    import ipaddress

    server = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
              .issuer_name(ca_name).public_key(server_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=1))
              .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
              .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
              .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
              .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False)
              .sign(key, hashes.SHA256()))
    (directory / "server-ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (directory / "server.pem").write_bytes(server.public_bytes(serialization.Encoding.PEM))
    (directory / "server.key").write_bytes(server_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return directory / "server-ca.pem"


@pytest.mark.timeout(120)
def test_a_workload_that_requires_a_certificate_is_reached_only_with_its_own(tmp_path):
    server_ca = _server_certificate(tmp_path)
    client_ca, client_key = make_ca(tmp_path / "client-ca")
    listen = {"host": "127.0.0.1", "port": unused_port(), "tls_certfile": str(tmp_path / "server.pem"),
              "tls_keyfile": str(tmp_path / "server.key"), "client_ca_certfile": str(client_ca)}
    with harness(listen=listen, provisioning={"client_ca_certfile": str(client_ca), "client_ca_keyfile": str(client_key)}) as h:
        h.supervisor.fleet.provider.offers = [default_offer("o-1", "m-1", min_bid_hourly=0.2)]
        tls_app = create_app(h.config, database=h.database)
        server = uvicorn.Server(uvicorn.Config(tls_app, **listener_options(h.config), log_level="warning"))
        running = h.loop.spawn(server.serve())
        while not server.started:
            time.sleep(0.01)
        url = f"https://127.0.0.1:{listen['port']}"
        try:
            key = h.supervisor.provisioning.grant(
                "secure", __import__("gpm_server.provisioning_store", fromlist=["Grant"]).Grant(
                    models=(BIG,), max_spend=5.0, max_spend_per_day=8.0, certs="required"))
            h.loop.run(tls_app.state.pool.registry.refresh())
            with WorkloadProvisioner(url, key, verify=str(server_ca), poll_s=0.05) as side:
                with pytest.raises(WorkloadRefused, match="send csr"):
                    side.workload(BIG, **SPEC)
                with side.workload(BIG, certs=True, **SPEC) as w:
                    stored = h.supervisor.workloads.get(w.name).cert_fingerprint
                    assert stored and w.answer["certificate"].startswith("-----BEGIN CERTIFICATE")
                    signed = x509.load_pem_x509_certificate(w.answer["certificate"].encode())
                    assert signed.subject.rfc4514_string() == f"CN=workload:{w.name}"
                    assert not signed.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
                    h.loop.run(tls_app.state.pool.registry.refresh())
                    served = w.client.chat(BIG, [{"role": "user", "content": "hi"}])
                    assert served.content, "its own certificate reaches it"
                    with httpx.Client(base_url=url, verify=ssl.create_default_context(cafile=str(server_ca)),
                                      headers={"Authorization": f"Bearer {w.key}"}) as bare:
                        refused = bare.post("/v1/chat/completions", json={"model": BIG, "messages": []})
                    assert refused.status_code == 403 and refused.json()["reason"] == "client_certificate_required", \
                        "its key alone does not"
        finally:
            server.should_exit = True
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), h.loop.loop).result(timeout=5)
            running.result(timeout=10)


def test_the_pool_signs_only_the_public_key_it_is_sent(tmp_path):
    from gpm_server.certs import CertRefused, ClientCA

    ca = ClientCA(*make_ca(tmp_path))
    key = ec.generate_private_key(ec.SECP256R1())
    greedy = (x509.CertificateSigningRequestBuilder()
              .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "please make me a CA")]))
              .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
              .sign(key, hashes.SHA256()))
    pem, _ = ca.sign(greedy.public_bytes(serialization.Encoding.PEM).decode(), "w-1", 1)
    signed = x509.load_pem_x509_certificate(pem.encode())
    assert signed.subject.rfc4514_string() == "CN=workload:w-1"
    assert not signed.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert signed.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value == x509.ExtendedKeyUsage(
        [ExtendedKeyUsageOID.CLIENT_AUTH])
    with pytest.raises(CertRefused):
        ca.sign("not a request", "w-1", 1)


def test_a_client_ca_is_refused_when_others_can_read_its_key_or_the_router_trusts_another(tmp_path):
    import os

    from gpm_server.certs import CertRefused, load_ca

    cert, key = make_ca(tmp_path / "one")
    other, _ = make_ca(tmp_path / "two")
    assert load_ca(str(cert), str(key), str(cert)) is not None
    with pytest.raises(CertRefused, match="not the CA the pool signs with"):
        load_ca(str(cert), str(key), str(other))
    os.chmod(key, 0o644)
    with pytest.raises(CertRefused, match="readable by others"):
        load_ca(str(cert), str(key))


def test_without_a_dead_man_timer_workloads_are_short(pool):
    """A provider nothing on the host can stop: no grant, plan or extension past the short limit."""
    import dataclasses

    provider = pool.supervisor.fleet.provider
    provider.capabilities = dataclasses.replace(provider.capabilities, self_terminate=False)
    with admin(pool) as control:
        refused = control.post("/pool/provisioners", json={
            "name": "long", "models": [BIG], "max_open": 1, "max_spend": 5.0, "max_spend_per_day": 8.0, "max_hours": 4})
        assert refused.status_code == 400 and "no dead-man timer" in refused.json()["detail"]
        plan = control.post("/pool/workloads/plan", json={
            "name": "research", "model": BIG, "latency_s": 30, "parallel": 2, "hours": 2, "max_spend": 5}).json()["plan"]
        assert "no dead-man timer" in (plan["refused"] or "")


def test_hours_are_not_added_past_a_workloads_certificate(pool):
    from gpm_server.supervisor.workloads import WorkloadRefused as Refused

    w = ask(pool, grant(pool))
    pool.supervisor.workloads.store.set_cert_fingerprint(w.name, "ab" * 32)
    with pytest.raises(Refused, match="signed for its hours"):
        pool.supervisor.workloads.extend(w.name, hours=1, confirm_hours=True)


def test_a_signing_request_the_pool_cannot_read_opens_nothing(pool, tmp_path):
    from gpm_server.certs import ClientCA
    from gpm_server.supervisor.workloads import WorkloadRefused as Refused
    from gpm_server.supervisor.workloads import WorkloadRequest

    ca = ClientCA(*make_ca(tmp_path))
    req = WorkloadRequest(name="evals-000001", model=BIG, latency_s=30, parallel=2, hours=1, max_spend=2.0)
    for junk in ("not a request", "-----BEGIN CERTIFICATE REQUEST-----\nAAAA\n-----END CERTIFICATE REQUEST-----\n"):
        with pytest.raises(Refused):
            pool.loop.run(pool.supervisor.workloads.create(req, provisioner="evals", csr=junk, ca=ca))
    assert pool.supervisor.leases.open_leases() == [] and pool.supervisor.workloads.store.all() == []


def test_certificates_are_refused_where_the_listener_asks_for_none(pool):
    """An optional grant, a program asking for a certificate, a listener with no client CA: the
    workload could never be reached, so it is not made."""
    pool.supervisor.provisioning._ca, pool.supervisor.provisioning._ca_loaded = object(), True
    key = grant(pool)
    refresh(pool)
    body = {"kind": "create", "key_hash": "ab" * 32, "model": BIG, "csr": "x", **SPEC}
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {key}"}) as http:
        request_id = http.post("/pool/provisioning/requests", json=body).json()["request_id"]
    pool.loop.run(pool.supervisor.provisioning.answer_pending())
    answered = pool.supervisor.provisioning.store.get_request(request_id)
    assert answered.state == "refused" and "does not ask for client certificates" in answered.answer["detail"]
    assert pool.supervisor.leases.open_leases() == []


def test_a_workload_answering_a_long_request_is_not_idle(pool):
    from gpm_server.db import CounterRow

    w = ask(pool, grant(pool), idle_end_minutes=1)
    pool.reprobe()  # its host rented
    an_hour_ago = time.time() - 3600
    pool.database.execute("UPDATE workloads SET state = 'serving', serving_at = ? WHERE name = ?", (an_hour_ago, w.name))
    workloads = pool.supervisor.workloads
    (host,) = workloads.fleet.hosts_of(w.name)
    pool.loop.run(pool.supervisor.counters.publish([CounterRow(host.host_id, busy=1, total=4, requests_served=0,
                                                               failures=0, last_request_at=an_hour_ago)]))
    assert workloads._idle_past(workloads.get(w.name), time.time()) is None, "a request is being answered"
    pool.loop.run(pool.supervisor.counters.publish([CounterRow(host.host_id, busy=0, total=4, requests_served=1,
                                                               failures=0, last_request_at=an_hour_ago)]))
    assert workloads._idle_past(workloads.get(w.name), time.time()) is not None


def test_a_key_past_its_expiry_is_refused_at_the_request(pool):
    """Time passing moves no revision: the router reads the expiry at each request."""
    key = grant(pool, expires_hours=0.5 / 3600)
    refresh(pool)
    time.sleep(0.6)  # past its expiry; nothing changed in the table, so the router read nothing new
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {key}"}) as http:
        assert http.get("/pool/provisioning/workloads/nothing").status_code == 401


def test_ends_are_one_at_a_time_per_workload(pool):
    w = ask(pool, grant(pool, max_open=2))
    refresh(pool)
    stop_answering(pool)
    with httpx.Client(base_url=pool.url, headers={"Authorization": f"Bearer {w._provisioner._key}"}) as http:
        first = http.post(f"/pool/provisioning/workloads/{w.name}/end")
        assert first.status_code == 202, first.text
        first = first.json()["request_id"]
        again = http.post(f"/pool/provisioning/workloads/{w.name}/end").json()["request_id"]
    assert first == again, "an end already waiting is the same end"


def test_a_pool_that_takes_no_programs_says_so_at_once():
    with harness() as h:
        h.supervisor.fleet.provider.offers = [default_offer("o-1", "m-1", min_bid_hourly=0.2)]
        control = ServerHandle(create_control_app(h.supervisor, h.config), h.loop)
        try:
            h.control_url = control.base_url
            key = grant(h)
            refresh(h)
            h.state.config = h.state.config.model_copy(update={
                "provisioning": h.state.config.provisioning.model_copy(update={"enabled": False})})
            with httpx.Client(base_url=h.url, headers={"Authorization": f"Bearer {key}"}) as http:
                answer = http.post("/pool/provisioning/requests", json={"kind": "plan", "model": BIG, **SPEC})
            assert answer.status_code == 403 and answer.json()["reason"] == "provisioning_disabled"
        finally:
            control.stop()



def test_a_program_creates_a_workload_of_two_models(pool):
    """Several models through the SDK (D118): the grant checks every one."""
    key = grant(pool, models=[BIG, SHARED], max_spend=5.0, max_spend_per_day=8.0)
    with provisioner(pool, key) as side:
        with pytest.raises(WorkloadRefused, match="may create workloads for"):
            side.workload(models={BIG: {"latency_s": 30, "parallel": 1}, "not-granted": {"latency_s": 5, "parallel": 1}},
                          hours=1, max_spend=2.0)
        with side.workload(models={BIG: {"latency_s": 30, "parallel": 1}, SHARED: {"latency_s": 5, "parallel": 1}},
                           hours=1, max_spend=2.0) as w:
            assert set(w.models) == {BIG, SHARED}
            made = pool.supervisor.workloads.get(w.name)
            assert set(made.models) == {BIG, SHARED} and made.placement in ("together", "apart")
            refresh(pool)
    with pytest.raises(ValueError, match="not both"):
        WorkloadProvisioner.__dict__["_body"].__func__(BIG, 1, 1, 1, 1, "roi", None, {BIG: {}}, "auto")
