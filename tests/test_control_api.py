"""The control API: everything that spends money, behind the admin key.

docs/spec/console-and-control-api.md §1 and threat model T1/T3. The rule with teeth is that the
app key is refused here — an app that can request a completion must not be able to open a lease.
"""

import httpx
import pytest
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.keys import KeyFileUnsafe, KeyStore, fingerprint, mint, verify
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

APP_KEY = "gpma_test_app_key"
ADMIN_KEY = "gpmx_test_admin_key"
MODEL = "m1"


@pytest.fixture
def control(tmp_path):
    config = PoolConfig.model_validate(
        {
            "pool": {"name": "test", "model_set": [MODEL], "probe_interval_s": 3600},
            "auth": {"app_keys": [APP_KEY], "admin_keys": [ADMIN_KEY]},
            "hosts": [
                {
                    "id": "local-1",
                    "kind": "local",
                    "transport": {"type": "http", "base_url": "http://127.0.0.1:1"},
                }
            ],
            "rented": {
                "provider": "fake",
                "bidding": {"bid_ceiling": 0.60},
                "scale": {"scale_up_after_s": 0},
            },
        }
    )
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(config, database)
    server = ServerHandle(create_control_app(supervisor, config), loop)
    try:
        yield supervisor, server.base_url, loop
    finally:
        server.stop()
        loop.stop()
        database.close()


def client(url, key=ADMIN_KEY):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.Client(base_url=url, headers=headers, timeout=30)


# --- keys ---


def test_a_key_is_shown_once_and_stored_only_as_a_hash(tmp_path):
    store = KeyStore(tmp_path / "app.keys")
    key, record = store.create("app", label="batch driver")

    assert key.startswith("gpma_")
    assert record.hashed == fingerprint(key)
    assert key not in (tmp_path / "app.keys").read_text()  # the file cannot hand anyone a key


def test_two_keys_of_a_role_are_valid_at_once_so_rotation_needs_no_downtime(tmp_path):
    store = KeyStore(tmp_path / "app.keys")
    first, first_record = store.create("app")
    second, _ = store.create("app")

    assert verify(first, store.hashes("app"))
    assert verify(second, store.hashes("app"))

    store.revoke(first_record.key_id)
    assert not verify(first, store.hashes("app"))
    assert verify(second, store.hashes("app"))


def test_a_key_file_others_can_read_is_refused(tmp_path):
    path = tmp_path / "app.keys"
    store = KeyStore(path)
    store.create("app")
    path.chmod(0o644)

    with pytest.raises(KeyFileUnsafe, match="readable by others"):
        store.load()


def test_an_unknown_key_verifies_against_nothing():
    assert not verify(mint("app"), {fingerprint(mint("app"))})
    assert not verify(None, {fingerprint("x")})
    assert not verify("anything", set())


# --- the boundary ---


def test_the_control_api_needs_the_admin_key(control):
    _, url, _ = control
    with client(url, key=None) as http:
        assert http.get("/pool/status").status_code == 401
    with client(url, key="wrong") as http:
        assert http.get("/pool/status").status_code == 401


def test_the_app_key_is_refused_and_told_why(control):
    """An app that can request a completion must not thereby be able to spend."""
    _, url, _ = control
    with client(url, key=APP_KEY) as http:
        response = http.post("/pool/leases", json={"workers": 1, "max_spend": 1.0, "allow_rent": True})

    assert response.status_code == 403
    assert response.json()["error"] == "app_key_refused"
    assert "admin key" in response.json()["detail"]


def test_a_request_from_another_site_is_refused(control):
    """Threat model T1: a web page open in the operator's browser must not drive this."""
    _, url, _ = control
    with client(url) as http:
        response = http.get("/pool/status", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert response.json()["error"] == "bad_origin"


# --- leases ---


def test_opening_a_lease_states_the_worst_case(control):
    _, url, _ = control
    with client(url) as http:
        response = http.post(
            "/pool/leases",
            json={"workers": 4, "max_hours": 2, "max_spend": 3.0, "allow_rent": True},
        )

    assert response.status_code == 201
    worst = response.json()["worst_case"]
    assert worst["dollars"] == 3.0
    assert worst["hours"] == 2
    # No overall cap by default any more (D46): the bound is per host, and it is stated.
    assert worst["max_hourly_burn"] is None
    assert worst["worst_case_hourly"] == pytest.approx(
        worst["max_rented_hosts"] * 0.60  # this fixture's per-host bid ceiling
    )


def test_a_lease_that_can_rent_is_refused_without_a_dollar_cap(control):
    _, url, _ = control
    with client(url) as http:
        response = http.post("/pool/leases", json={"workers": 4, "allow_rent": True})

    assert response.status_code == 400
    assert "dollar cap" in response.json()["detail"]


def test_a_lease_is_tightened_freely_and_extended_only_when_confirmed(control):
    """Extending in flight is what an operator wants when a host is worth keeping; raising a
    limit is still loosening, so it is typed again (D49)."""
    supervisor, url, _ = control
    with client(url) as http:
        lease_id = http.post(
            "/pool/leases", json={"workers": 4, "max_hours": 1.0, "max_spend": 3.0, "allow_rent": True}
        ).json()["lease_id"]

        assert http.patch(f"/pool/leases/{lease_id}", json={"max_spend": 1.0}).status_code == 200

        unconfirmed = http.patch(f"/pool/leases/{lease_id}", json={"max_hours": 4.0})
        assert unconfirmed.status_code == 400 and unconfirmed.json()["error"] == "not_confirmed"

        wrong = http.patch(f"/pool/leases/{lease_id}", json={"max_hours": 4.0, "confirm": "3.0"})
        assert wrong.status_code == 400

        extended = http.patch(f"/pool/leases/{lease_id}", json={"max_hours": 4.0, "confirm": "4.0"})
        assert extended.status_code == 200 and extended.json()["max_hours"] == 4.0
        assert extended.json()["hours_left"] > 3.9
        # and the worker count is left exactly as it was: only what was asked for changes
        assert extended.json()["workers"] == 4

        missing = http.patch("/pool/leases/nobody", json={"max_hours": 9.0})
    assert missing.status_code == 404
    assert "lease_extended" in [e["kind"] for e in supervisor.events.recent(20)]


def test_leases_report_spend_against_their_cap(control):
    supervisor, url, loop = control
    with client(url) as http:
        http.post("/pool/leases", json={"workers": 4, "max_spend": 3.0, "allow_rent": True})
        loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))
        lease = http.get("/pool/leases").json()["leases"][0]

    assert lease["state"] == "open"
    assert lease["dollars_left"] <= 3.0 * 0.9  # the cap, less the safety margin
    assert "estimated_spend" in lease


# --- plan spends nothing ---


def test_plan_says_what_would_happen_and_creates_nothing(control):
    supervisor, url, _ = control
    with client(url) as http:
        http.post("/pool/leases", json={"workers": 8, "max_spend": 3.0, "allow_rent": True})
        plan = http.get("/pool/plan").json()["plan"]

    acquire = [step for step in plan if step["step"] == "acquire"][0]
    assert acquire["would_rent"] is True
    assert acquire["would_bid"]["bid"] == pytest.approx(0.12)
    assert acquire["reasons"]
    assert supervisor.fleet.provider.instances == {}  # nothing was created


def test_plan_shows_what_a_cap_would_refuse(control):
    supervisor, url, _ = control
    supervisor.config.limits.max_hourly_burn = 0.01
    with client(url) as http:
        http.post("/pool/leases", json={"workers": 8, "max_spend": 3.0, "allow_rent": True})
        plan = http.get("/pool/plan").json()["plan"]

    acquire = [step for step in plan if step["step"] == "acquire"][0]
    assert acquire["would_rent"] is False
    assert "burn" in acquire["refused_by_caps"]


# --- the panic button ---


def test_down_all_destroys_everything_rented_and_closes_the_leases(control):
    supervisor, url, loop = control
    with client(url) as http:
        http.post("/pool/leases", json={"workers": 8, "max_spend": 3.0, "allow_rent": True})
        loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))
        assert supervisor.fleet.hosts

        released = http.post("/pool/down").json()["released"]

    assert released
    assert supervisor.fleet.hosts == {}
    assert supervisor.fleet.provider.instances == {}
    assert supervisor.leases.open_leases() == []


def test_preparing_a_host_reports_what_it_bid(control):
    supervisor, url, _ = control
    with client(url) as http:
        response = http.post(
            "/pool/hosts/prepare", json={"max_spend": 1.0, "max_hours": 1, "when_ready": "park"}
        )

    assert response.status_code == 201
    assert response.json()["bid_hourly"] == pytest.approx(0.12)
    assert supervisor.fleet.hosts


def test_one_offer_from_the_market_can_be_rented_as_listed(control):
    """D55: the market lists both kinds, each row names its offer, and prepare takes that name."""
    from gpm_server.providers import default_offer

    supervisor, url, _ = control
    supervisor.fleet.provider.offers.append(default_offer(
        offer_id="od-1", machine_id="m-9", min_bid_hourly=0.50, all_in_hourly=0.50,
        on_demand_hourly=0.50, interruptible=False))
    with client(url) as http:
        rows = http.get("/pool/market/preview?kinds=both").json()["best"]
        fixed = next(row for row in rows if row["kind"] == "on_demand")
        response = http.post("/pool/hosts/prepare", json={
            "max_spend": 1.0, "max_hours": 1, "offer_id": fixed["offer_id"], "kind": fixed["kind"]})

    assert response.status_code == 201
    (host,) = supervisor.fleet.hosts.values()
    assert host.offer.offer_id == "od-1" and host.interruptible is False


def test_a_refused_prepare_says_the_real_reason(control):
    supervisor, url, _ = control
    with client(url) as http:
        response = http.post("/pool/hosts/prepare", json={
            "max_spend": 1.0, "max_hours": 1, "offer_id": "gone-already"})

    assert response.status_code >= 400
    assert "gone-already" in response.json()["detail"]
    assert "nothing was spent" in response.json()["detail"]
    assert supervisor.fleet.provider.instances == {}


def test_events_carry_the_numbers_behind_each_decision(control):
    supervisor, url, loop = control
    with client(url) as http:
        http.post("/pool/leases", json={"workers": 8, "max_spend": 3.0, "allow_rent": True})
        loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))
        events = http.get("/pool/events").json()["events"]

    rented = [event for event in events if event["kind"] == "rented"][0]
    assert rented["numbers"]["floor"] == 0.10
    assert rented["numbers"]["bid"] == pytest.approx(0.12)


# --- read-only market view ---


def test_the_market_preview_runs_the_real_filters_and_creates_nothing(control):
    """Spec §2.2: moving a ceiling and watching '4 pass' become '0 pass' is how an operator
    learns what a number means."""
    supervisor, url, _ = control
    with client(url) as http:
        preview = http.get("/pool/market/preview").json()

    assert preview["seen"] == 1
    assert preview["passed"] == 1
    assert preview["best"][0]["would_bid"] == pytest.approx(0.12)
    assert preview["policy"]["bid_ceiling"] == 0.60
    assert supervisor.fleet.provider.instances == {}  # nothing was created


def test_a_ceiling_that_bites_shows_up_as_a_rejection_reason(control):
    supervisor, url, _ = control
    supervisor.fleet.rented.offer_policy.min_gpu_memory_gb = 999
    with client(url) as http:
        preview = http.get("/pool/market/preview").json()

    assert preview["passed"] == 0
    assert preview["rejected"] == 1
    assert any("gpu memory" in reason for reason in preview["rejected_by_reason"])


def test_the_account_check_spends_nothing(control):
    supervisor, url, _ = control
    with client(url) as http:
        account = http.get("/pool/account").json()

    assert account["credential_valid"] is True
    assert supervisor.fleet.provider.instances == {}


# --- changing a running host's worker count (D56) ---


def rent_one(supervisor, loop):
    supervisor.fleet.run_on_host = lambda *a: (0, "")
    supervisor.fleet.open_lease(workers=2, max_hours=2, max_spend=2.0, allow_rent=True)
    host = loop.run(supervisor.fleet.rent_one(supervisor.leases.open_leases()[0], ["for the test"]))
    host.state = "ready"
    host.workers = 3  # something to lower from, and to be refused a raise above
    return host


def test_raising_a_hosts_workers_asks_for_the_host_id_again(control):
    """It relaunches that host's engine, and every restart is typed twice (D41)."""
    supervisor, url, loop = control
    host = rent_one(supervisor, loop)

    with client(url) as http:
        refused = http.post(f"/pool/hosts/{host.host_id}/resize", json={"workers": host.workers + 2})

    assert refused.status_code == 400
    assert refused.json()["error"] == "not_confirmed"
    assert host.workers == 3, "nothing changed on a refusal"


def test_lowering_a_hosts_workers_needs_no_retype_because_nothing_restarts(control):
    supervisor, url, loop = control
    host = rent_one(supervisor, loop)

    with client(url) as http:
        answered = http.post(f"/pool/hosts/{host.host_id}/resize", json={"workers": 1})

    assert answered.status_code == 200
    assert answered.json()["workers"] == 1 and host.workers == 1


def test_resizing_needs_the_admin_key_like_everything_that_changes_the_pool(control):
    supervisor, url, loop = control
    host = rent_one(supervisor, loop)

    with client(url, key=APP_KEY) as http:
        refused = http.post(f"/pool/hosts/{host.host_id}/resize", json={"workers": 1})

    assert refused.status_code in (401, 403)
    assert host.workers == 3


# --- turning dynamic allocation on from the console (D74) ---


def test_allocation_can_be_switched_on_from_the_rented_screen(control, tmp_path):
    """The file stays the source of truth: the change is written into it, validated, planned
    and applied — and a block the operator never wrote is added whole."""
    supervisor, url, loop = control
    written = tmp_path / "pool.yaml"
    written.write_text(
        "pool: { name: test, model_set: [m1] }\n"
        "auth: { app_keys: [k], admin_keys: [a] }\n"
        "hosts:\n"
        "  - id: local-1\n"
        "    kind: local\n"
        "    transport: { type: http, base_url: 'http://127.0.0.1:1' }\n"
        "rented:\n"
        "  provider: fake\n"
        "  bidding: { bid_ceiling: 0.60 }\n"
    )
    from gpm_server.configplan import ConfigStore

    supervisor.config_path = written
    supervisor.store = ConfigStore(written)

    with client(url) as http:
        answered = http.patch(
            "/pool/config/rented",
            json={"allocation": "dynamic", "dynamic": {"max_round": 3, "window_s": 60}},
        )

    assert answered.status_code == 200, answered.text
    text = written.read_text()
    assert "allocation:" in text and "dynamic:" in text
    from gpm_server.config import load_config

    reloaded = load_config(written)
    assert reloaded.rented.allocation == "dynamic"
    assert reloaded.rented.dynamic.max_round == 3
