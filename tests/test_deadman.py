"""The dead-man timer, actually run.

docs/spec/supervisor.md §7. The script is not merely asserted on — it is executed here with a
short window against a local stand-in for the provider's API, so "it fires when the pool goes
silent, and does not while work is happening" is checked rather than assumed.
"""

import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fakes.fake_ollama import unused_port
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.deadman import build_script, heartbeat_command, onstart_script
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, ProviderCapabilities
from gpm_server.supervisor.renting import Fleet


class Terminations:
    """A stand-in for the provider endpoint an instance calls to end itself."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.port = unused_port()
        collected = self.calls

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                collected.append((self.path, self.headers.get("Authorization", "")))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/terminate"

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture
def endpoint():
    made = Terminations()
    try:
        yield made
    finally:
        made.stop()


def run_timer(script_path: Path, seconds: float) -> subprocess.Popen:
    process = subprocess.Popen(["/bin/sh", str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.1)
    return process


# --- what the script says ---


def test_the_timer_carries_only_the_instance_scoped_credential():
    """Threat model T5: the account credential is never placed on a rented machine."""
    provider = FakeProvider()
    script = onstart_script(provider.self_terminate_command(), window_s=1200)

    referenced = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD))", script))
    assert referenced == {"CONTAINER_API_KEY"}  # the provider's per-instance key, and nothing else
    assert "CONTAINER_ID" in script


def test_both_conditions_are_required():
    script = build_script("true", window_s=1200)
    assert '"$SINCE_HEARTBEAT" -ge "$WINDOW" ] && [ "$SINCE_ACTIVITY" -ge "$WINDOW"' in script


def test_the_timer_is_armed_before_anything_else_in_the_start_up():
    script = onstart_script("true", window_s=60, extra="start-the-engine")
    assert script.index("deadman.sh") < script.index("start-the-engine")


def test_the_pools_public_key_is_installed_after_the_timer_is_armed():
    """So the supervisor can reach a host whatever keys the account has registered."""
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyMaterialOnly pool"
    script = onstart_script("true", window_s=60, public_key=key, extra="start-the-engine")
    assert script.index("deadman.sh") < script.index("authorized_keys") < script.index("start-the-engine")
    assert key in script
    assert "chmod 600" in script  # sshd refuses a world-readable authorized_keys


def test_a_pool_key_with_no_public_half_installs_nothing(tmp_path):
    from gpm_server.supervisor.renting import Fleet  # noqa: F401 - fleet_for below

    database, fleet = fleet_for(tmp_path)
    try:
        fleet.rented.ssh_key = str(tmp_path / "missing_key")
        assert fleet.pool_public_key() is None
        assert "authorized_keys" not in fleet.deadman_onstart()
    finally:
        database.close()


def test_the_public_key_beside_the_private_one_is_what_travels(tmp_path):
    database, fleet = fleet_for(tmp_path)
    try:
        (tmp_path / "pool_key").write_text("PRIVATE KEY MATERIAL")
        (tmp_path / "pool_key.pub").write_text("ssh-ed25519 AAAAPublic pool\n")
        fleet.rented.ssh_key = str(tmp_path / "pool_key")
        onstart = fleet.deadman_onstart()
        assert "ssh-ed25519 AAAAPublic pool" in onstart
        assert "PRIVATE KEY MATERIAL" not in onstart  # the private half never leaves this machine
    finally:
        database.close()


# --- what the script does ---


def test_it_fires_when_the_pool_goes_silent(tmp_path, endpoint, monkeypatch):
    """No heartbeat and no inference for the window: the instance ends itself."""
    state = tmp_path / "state"
    script = build_script(
        f'curl -sS -X POST -H "Authorization: Bearer $CONTAINER_API_KEY" '
        f'"{endpoint.url}?instance=$CONTAINER_ID"',
        window_s=1,
        poll_s=1,
        engine_port=unused_port(),
        state_dir=str(state),
    )
    path = tmp_path / "deadman.sh"
    path.write_text(script)

    monkeypatch.setenv("CONTAINER_API_KEY", "instance-scoped-key")
    monkeypatch.setenv("CONTAINER_ID", "i-42")
    try:
        process = run_timer(path, seconds=12)
    finally:
        if process.poll() is None:
            process.kill()

    assert endpoint.calls, "the timer never fired"
    path_called, authorization = endpoint.calls[0]
    assert "instance=i-42" in path_called
    assert authorization == "Bearer instance-scoped-key"


def test_inference_is_detected_from_the_raw_tcp_table_without_ss_or_netstat():
    """Seen live: the engine image ships neither `ss` nor `netstat`. /proc/net/tcp always
    exists on Linux, so the awk match on it is what actually keeps a busy host alive."""
    script = build_script("true", window_s=60)
    assert "/proc/net/tcp" in script
    assert "printf '%04X'" in script  # 11434 -> 2CAA, as the kernel prints it
    # The awk program: established (01) and the local port suffix matches.
    import re
    import subprocess
    program = re.search(r"awk -v p=\":\$HEXPORT\" '([^']+)'", script).group(1)
    sample = (
        "  sl  local_address rem_address   st\n"
        "   0: 00000000:2CAA 00000000:0000 0A\n"   # listening on 11434: not activity
        "   1: 0100007F:2CAA 0100007F:C350 01\n"   # established on 11434: activity
    )
    hit = subprocess.run(["awk", "-v", "p=:2CAA", program], input=sample, text=True, capture_output=True)
    assert hit.returncode == 0
    miss = subprocess.run(["awk", "-v", "p=:2CAA", program], input=sample.replace(" 01", " 06"), text=True, capture_output=True)
    assert miss.returncode == 1  # nothing established


def test_it_stays_quiet_while_the_supervisor_is_beating(tmp_path, endpoint):
    state = tmp_path / "state"
    script = build_script(
        f'curl -sS -X POST "{endpoint.url}"',
        window_s=3,
        poll_s=1,
        engine_port=unused_port(),
        state_dir=str(state),
    )
    path = tmp_path / "deadman.sh"
    path.write_text(script)

    process = subprocess.Popen(["/bin/sh", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # A supervisor doing its job, for longer than the window.
        for _ in range(8):
            time.sleep(0.5)
            subprocess.run(["/bin/sh", "-c", heartbeat_command(str(state))], check=True)
        time.sleep(1)
        assert not endpoint.calls, "the timer fired while the supervisor was alive"
    finally:
        process.kill()


# --- the supervisor's side ---


def fleet_for(tmp_path, **capabilities):
    database = Database(tmp_path / "gpm.sqlite3")
    config = PoolConfig.model_validate(
        {
            "pool": {"name": "test", "model_set": ["m1"]},
            "auth": {"app_keys": ["k"]},
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
    provider = FakeProvider(capabilities=ProviderCapabilities(**capabilities) if capabilities else None)
    return database, Fleet(
        config, config.rented, provider, LeaseStore(database), EventLog(database), SpendLedger(database)
    )


async def test_a_rented_host_is_created_with_its_timer_armed(tmp_path):
    database, fleet = fleet_for(tmp_path)
    try:
        fleet.leases.open(workers=4, max_hours=2, max_spend=1.0, allow_rent=True)
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

        instance = next(iter(fleet.provider.instances.values()))
        assert instance.spec.onstart is not None
        assert "deadman.sh" in instance.spec.onstart
        assert "CONTAINER_API_KEY" in instance.spec.onstart
    finally:
        database.close()


async def test_the_supervisor_beats_every_live_host_each_pass(tmp_path):
    database, fleet = fleet_for(tmp_path)
    beats: list[tuple[str, str]] = []

    async def record(host, command):
        beats.append((host.host_id, command))
        return 0, ""

    fleet.run_on_host = record
    try:
        fleet.leases.open(workers=4, max_hours=2, max_spend=1.0, allow_rent=True)
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})

        assert beats, "no heartbeat was sent"
        assert all("touch" in command and "heartbeat" in command for _, command in beats)
    finally:
        database.close()


async def test_a_provider_that_cannot_self_terminate_gets_no_timer_and_only_short_leases(tmp_path):
    database, fleet = fleet_for(tmp_path, interruptible=True, self_terminate=False)
    try:
        assert fleet.deadman_onstart() is None
        assert fleet.max_lease_hours() == 1.0
    finally:
        database.close()


async def test_a_heartbeat_that_cannot_be_delivered_does_not_stop_the_pass(tmp_path):
    """The timer is meant to fire when the host is unreachable — so failing to beat it is
    exactly the case that must not break anything else."""
    database, fleet = fleet_for(tmp_path)

    async def refuse(host, command):
        raise OSError("host unreachable")

    fleet.run_on_host = refuse
    try:
        fleet.leases.open(workers=4, max_hours=2, max_spend=1.0, allow_rent=True)
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
        await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
        assert len(fleet.hosts) == 1
    finally:
        database.close()
