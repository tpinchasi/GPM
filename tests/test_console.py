"""The console, through the same API it uses.

docs/spec/console-and-control-api.md. The console is one more client of the control API, so
these tests drive the endpoints the page calls — which is also what proves the roadmap's exit
criterion: every console action is possible from the CLI, a mistyped ceiling is caught by plan
before apply, and the app key is refused.
"""

import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import httpx
import pytest
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.configplan import ConfigStore, RentedNow, plan_changes
from gpm_server.db import Database
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_console_admin"
APP_KEY = "gpma_console_app"
MODEL = "m1"

BASE_YAML = textwrap.dedent(
    f"""
    pool:
      name: console
      model_set: [{MODEL}]
      probe_interval_s: 3600
    auth:
      app_keys: ["{APP_KEY}"]
      admin_keys: ["{ADMIN_KEY}"]
    hosts:
      - id: local-1
        kind: local
        workers: 2
        transport: {{type: http, base_url: "http://127.0.0.1:1"}}
    limits: {{max_rented_hosts: 1, max_hourly_burn: 1.00}}
    rented:
      provider: fake
      bidding: {{bid_ceiling: 0.60}}
      scale: {{scale_up_after_s: 0}}
    """
).strip()


@pytest.fixture
def console(tmp_path):
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(BASE_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        yield supervisor, server.base_url, config_path, loop
    finally:
        server.stop()
        loop.stop()
        database.close()


def client(url, key=ADMIN_KEY):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.Client(base_url=url, headers=headers, timeout=30)


def edited(text, old, new):
    assert old in text
    return text.replace(old, new)


# --- the page parses at all ---

STATIC = Path(__file__).resolve().parents[1] / "server/src/gpm_server/console/static"
CLOSERS = {")": "(", "]": "[", "}": "{"}


def unbalanced(source: str) -> str | None:
    """Where the brackets stop matching, ignoring comments and string literals.

    The console is written as deeply nested `el(...)` calls, so a dropped `)` is the mistake
    this style invites — and it is fatal: the whole script fails to parse, the page draws
    nothing, and no test that only reads the file as text notices. This is a cheap structural
    check that runs everywhere; `node --check` below is the real parser where one is installed.
    """
    stack: list[tuple[str, int]] = []
    index, line, length = 0, 1, len(source)
    while index < length:
        char = source[index]
        if char == "\n":
            line += 1
        elif char == "/" and source[index + 1: index + 2] == "/":
            index = source.find("\n", index)
            if index < 0:
                break
            continue
        elif char == "/" and source[index + 1: index + 2] == "*":
            end = source.index("*/", index)
            line += source.count("\n", index, end)
            index = end + 2
            continue
        elif char in "\"'`":
            quote, index = char, index + 1
            while index < length and source[index] != quote:
                if source[index] == "\\":
                    index += 1
                elif source[index] == "\n":
                    line += 1
                index += 1
        elif char in "([{":
            stack.append((char, line))
        elif char in CLOSERS:
            if not stack or stack[-1][0] != CLOSERS[char]:
                opened = f"{stack[-1][0]!r} opened on line {stack[-1][1]}" if stack else "nothing open"
                return f"line {line}: {char!r} closes nothing — {opened}"
            stack.pop()
        index += 1
    if stack:
        char, opened_on = stack[-1]
        return f"{char!r} opened on line {opened_on} is never closed"
    return None


@pytest.mark.parametrize("name", ["app.js"])
def test_the_console_script_parses(name):
    """It is served to a browser and never imported by Python, so nothing else would tell us."""
    source = (STATIC / name).read_text()
    assert unbalanced(source) is None, unbalanced(source)

    node = shutil.which("node")
    if node:  # a real parser wherever one is installed, and always in CI
        subprocess.run([node, "--check", str(STATIC / name)], check=True, capture_output=True)


def test_the_bracket_check_would_catch_a_dropped_closing_paren():
    """The check itself, on the exact mistake it exists for — and on what must not trip it."""
    assert unbalanced('el("a", el("b"));') is None
    assert "never closed" in unbalanced('el("a", el("b");')
    # A bracket inside a string, a comment or a template literal is not a bracket.
    assert unbalanced('const s = ")))"; // )))\nconst t = `${x} )`;') is None


def test_every_screen_the_navigation_offers_exists_in_the_script():
    """A nav link with no screen behind it silently falls back to the overview."""
    page = (STATIC / "index.html").read_text()
    script = (STATIC / "app.js").read_text()
    import re

    links = set(re.findall(r'<a href="#([a-z]+)"', page))
    assert links, page
    for name in links:
        assert f"screens.{name} =" in script, name


# --- the page ---


def test_the_page_is_served_and_holds_no_key_of_its_own(console):
    _, url, _, _ = console
    with httpx.Client(base_url=url, timeout=30) as anonymous:
        page = anonymous.get("/ui/")
        assert page.status_code == 200  # the page itself needs no key: it asks for one
        assert "gpm" in page.text.lower()
        script = anonymous.get("/ui/app.js").text

    # The key is entered into the page and kept in memory; it is never persisted.
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "document.cookie" not in script
    # ...and every call it makes carries it as a header.
    assert "Authorization: `Bearer ${ADMIN_KEY}`" in script


def test_the_console_is_never_served_from_a_stale_cache(console):
    """Found the hard way: after a fix to app.js the owner's browser kept running the old,
    broken script — a plain reload revalidates the page, not its scripts, unless told to."""
    _, url, _, _ = console
    with httpx.Client(base_url=url, timeout=30) as anonymous:
        for path in ("/ui/", "/ui/app.js", "/ui/console.css"):
            response = anonymous.get(path)
            assert response.headers.get("cache-control") == "no-cache", path
            assert "etag" in response.headers, path  # so "ask every time" is a cheap 304


def test_the_page_cannot_read_anything_without_the_key(console):
    _, url, _, _ = console
    with httpx.Client(base_url=url, timeout=30) as anonymous:
        assert anonymous.get("/pool/status").status_code == 401
        assert anonymous.get("/pool/config").status_code == 401


def test_every_screens_data_call_is_refused_to_the_app_key(console):
    """An app that can request a completion must not be able to read or spend here."""
    _, url, _, _ = console
    paths = ["/pool/status", "/pool/events", "/pool/leases", "/pool/plan", "/pool/config",
             "/pool/config/history", "/pool/market/preview", "/pool/account"]
    with client(url, key=APP_KEY) as http:
        for path in paths:
            assert http.get(path).status_code == 403, path


def test_status_carries_what_the_overview_draws(console):
    supervisor, url, _, loop = console
    supervisor.fleet.open_lease(workers=4, max_hours=2, max_spend=2.0, allow_rent=True)
    loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))

    with client(url) as http:
        status = http.get("/pool/status").json()

    host = status["hosts"][0]
    assert {"host_id", "kind", "transport", "priority", "state", "workers", "busy"} <= set(host)
    lease = status["open_leases"][0]
    assert {"lease_id", "max_spend", "spent_enforced_on", "dollars_left", "stops_at"} <= set(lease)
    rented = status["rented"][0]
    assert {"bid_hourly", "storage_hourly", "estimated_spend", "hours_held", "hardware"} <= set(rented)


def test_status_carries_the_model_set_and_catalog_the_models_screen_draws(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        status = http.get("/pool/status").json()
        assert status["model_set"] == [MODEL]

        # And it follows a reload, rather than reporting what the process started with.
        changed = edited(path.read_text(), f"model_set: [{MODEL}]", "model_set: [m2]")
        http.put("/pool/config", json={"text": changed, "version": status_version(http)})
        assert http.get("/pool/status").json()["model_set"] == ["m2"]


def test_the_models_screen_is_told_which_build_is_served_and_whether_that_tag_is_resident(console):
    """Found live: a host holding the Apple build of a model showed 'missing' because the
    screen looked for the logical name in the resident list. The pool resolves a variant per
    host; status must say which one, so the screen draws what the pool decided."""
    supervisor, url, path, _ = console
    with client(url) as http:
        with_catalog = edited(
            edited(path.read_text(), "    workers: 2\n", "    workers: 2\n    capabilities: [apple-silicon]\n"),
            "hosts:\n",
            "catalog:\n"
            f"  {MODEL}:\n"
            "    variants:\n"
            f"      - {{tag: {MODEL}-mlx, requires: [apple-silicon], runtime_class: apple-mlx, enforces_schema: false}}\n"
            f"      - {{tag: {MODEL}}}\n"
            "hosts:\n",
        )
        assert http.put("/pool/config", json={"text": with_catalog, "version": status_version(http)}).status_code == 200

        # The engine holds the build, never the logical name.
        supervisor.hosts["local-1"].resident = frozenset({f"{MODEL}-mlx"})
        host = http.get("/pool/status").json()["hosts"][0]

    served = host["served"][MODEL]
    assert served == {
        "tag": f"{MODEL}-mlx", "runtime_class": "apple-mlx", "enforces_schema": False,
        "resident": True, "available": False,  # never probed: the engine at :1 is unreachable
    }
    assert host["residency"] == "pinned"
    assert MODEL not in host["resident"]  # what the old screen was checking, and would still call missing


def status_version(http):
    return http.get("/pool/config").json()["version"]


def test_limits_shown_follow_a_reload_too(console):
    _, url, path, _ = console
    with client(url) as http:
        assert http.get("/pool/status").json()["limits"]["max_hourly_burn"] == 1.00
        changed = edited(path.read_text(), "max_hourly_burn: 1.00", "max_hourly_burn: 0.25")
        http.put("/pool/config", json={"text": changed, "version": status_version(http)})
        assert http.get("/pool/status").json()["limits"]["max_hourly_burn"] == 0.25


def test_the_live_stream_sends_decisions_and_status(console):
    supervisor, url, _, _ = console
    supervisor.events.record("test_event", "something happened", numbers={"n": 1})

    with client(url) as http:
        with http.stream("GET", "/pool/events/stream", timeout=20) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            kinds, chunk = set(), ""
            for text in response.iter_text():
                chunk += text
                for frame in chunk.split("\n\n"):
                    if "event: decision" in frame:
                        kinds.add("decision")
                        assert "something happened" in frame
                    if "event: status" in frame:
                        kinds.add("status")
                if {"decision", "status"} <= kinds:
                    break
    assert kinds == {"decision", "status"}


# --- plan: nothing is applied blind ---


def test_a_mistyped_ceiling_is_caught_by_plan_and_must_be_retyped(console):
    """The roadmap's exit criterion: 6.0 typed for 0.60 is not applied silently."""
    supervisor, url, path, loop = console
    supervisor.fleet.open_lease(workers=4, max_hours=2, max_spend=2.0, allow_rent=True)
    loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))

    typo = edited(path.read_text(), "bid_ceiling: 0.60", "bid_ceiling: 6.0")
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": typo}).json()

    raised = [c for c in plan["changes"] if c["kind"] == "bid_ceiling_raised"]
    assert raised, plan
    assert raised[0]["requires_retype"] is True
    assert raised[0]["value"] == "6.000"
    assert "$0.600 to $6.000" in raised[0]["detail"]
    # And nothing has changed until Apply.
    assert supervisor.config.rented.bidding.bid_ceiling == 0.60


def test_plan_says_which_running_host_a_lowered_ceiling_would_release(console):
    supervisor, url, path, loop = console
    supervisor.fleet.open_lease(workers=4, max_hours=2, max_spend=2.0, allow_rent=True)
    loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))
    host = next(iter(supervisor.fleet.hosts.values()))

    lowered = edited(path.read_text(), "bid_ceiling: 0.60", "bid_ceiling: 0.05")
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": lowered}).json()

    change = [c for c in plan["changes"] if c["kind"] == "bid_ceiling_lowered"][0]
    assert host.host_id in change["detail"]
    assert "drained and released" in change["detail"]
    assert change["requires_retype"] is False  # tightening needs no ceremony


def test_plan_refuses_a_file_that_would_not_load(console):
    _, url, path, _ = console
    broken = path.read_text() + "\n  this: is not valid yaml: at all:\n"
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": broken}).json()
    assert plan["errors"]
    assert plan["changes"] == []


def test_plan_spends_nothing_and_creates_nothing(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        http.post("/pool/config/plan", json={"text": path.read_text()})
    assert supervisor.fleet.provider.instances == {}


# --- apply, and what it does to a running pool ---


def test_applying_a_host_makes_the_running_pool_follow(console):
    supervisor, url, path, loop = console
    added = edited(
        path.read_text(),
        '    transport: {type: http, base_url: "http://127.0.0.1:1"}',
        '    transport: {type: http, base_url: "http://127.0.0.1:1"}\n'
        '  - id: local-2\n    kind: local\n    workers: 1\n'
        '    transport: {type: http, base_url: "http://127.0.0.1:2"}',
    )
    with client(url) as http:
        version = http.get("/pool/config").json()["version"]
        assert http.put("/pool/config", json={"text": added, "version": version}).status_code == 200

    assert "local-2" in supervisor.hosts
    assert len(supervisor.config.hosts) == 2


def test_removing_a_host_retires_it_from_the_published_table(console):
    supervisor, url, path, loop = console
    loop.run(supervisor.pass_once())
    assert any(row.host_id == "local-1" for row in supervisor.table.all())

    without = edited(
        path.read_text(),
        '  - id: local-1\n    kind: local\n    workers: 2\n'
        '    transport: {type: http, base_url: "http://127.0.0.1:1"}\n',
        "",
    ).replace("hosts:\n", "hosts: []\n")
    with client(url) as http:
        version = http.get("/pool/config").json()["version"]
        assert http.put("/pool/config", json={"text": without, "version": version}).status_code == 200

    assert supervisor.hosts == {}
    assert all(row.host_id != "local-1" for row in supervisor.table.all())


def test_a_write_based_on_a_stale_version_is_refused(console):
    _, url, path, _ = console
    with client(url) as http:
        stale = http.get("/pool/config").json()["version"]
        first = edited(path.read_text(), "max_hourly_burn: 1.00", "max_hourly_burn: 0.50")
        assert http.put("/pool/config", json={"text": first, "version": stale}).status_code == 200

        second = edited(path.read_text(), "max_rented_hosts: 1", "max_rented_hosts: 1 ")
        refused = http.put("/pool/config", json={"text": second, "version": stale})

    assert refused.status_code == 409
    assert "changed since it was read" in refused.json()["detail"]


def test_a_file_that_does_not_load_never_reaches_the_running_pool(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        version = http.get("/pool/config").json()["version"]
        refused = http.put("/pool/config", json={"text": "pool: {}", "version": version})
    assert refused.status_code == 400
    assert supervisor.config.pool.name == "console"  # unchanged
    assert "local-1" in supervisor.hosts


def test_editing_the_file_by_hand_is_noticed(console):
    supervisor, url, path, loop = console
    path.write_text(edited(path.read_text(), "max_hourly_burn: 1.00", "max_hourly_burn: 0.25"))
    # The file's mtime must differ from the one recorded at start.
    import os
    os.utime(path, (time.time() + 1, time.time() + 1))

    loop.run(supervisor.pass_once())

    assert supervisor.config.limits.max_hourly_burn == 0.25
    assert "config_applied" in [e["kind"] for e in supervisor.events.recent(20)]


def test_history_keeps_what_was_applied_and_rollback_restores_it(console):
    supervisor, url, path, _ = console
    with client(url) as http:
        original = http.get("/pool/config").json()
        changed = edited(original["text"], "max_hourly_burn: 1.00", "max_hourly_burn: 0.25")
        http.put("/pool/config", json={"text": changed, "version": original["version"]})
        assert supervisor.config.limits.max_hourly_burn == 0.25

        history = http.get("/pool/config/history").json()["versions"]
        assert history and history[0]["version"] == original["version"]

        http.post("/pool/config/rollback", json={"version": original["version"]})

    assert supervisor.config.limits.max_hourly_burn == 1.00


# --- the market preview with unsaved values ---


def test_the_market_preview_can_use_the_forms_unsaved_values(console):
    _, url, _, _ = console
    with client(url) as http:
        saved = http.get("/pool/market/preview").json()
        tightened = http.post(
            "/pool/market/preview", json={"offer_policy": {"min_gpu_memory_gb": 999}}
        ).json()

    assert saved["passed"] == 1
    assert tightened["passed"] == 0  # moving a ceiling and watching "1 pass" become "0 pass"
    assert "gpu memory" in " ".join(tightened["rejected_by_reason"])


def test_an_unsaved_policy_that_makes_no_sense_is_a_400_not_something_the_pool_acts_on(console):
    _, url, _, _ = console
    with client(url) as http:
        response = http.post("/pool/market/preview", json={"bidding": {"bid_ceiling": "not a number"}})
    assert response.status_code == 400


# --- the plan function itself, as a pure function ---


def parse(text):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pool.yaml"
        path.write_text(text)
        return load_config(path)


def test_listen_changes_are_reported_as_needing_a_restart():
    current = parse(BASE_YAML)
    candidate = parse(BASE_YAML + "\nlisten: {host: 127.0.0.1, port: 9999}\n")
    change = [c for c in plan_changes(current, candidate) if c.kind == "listen"][0]
    assert change.needs_restart
    assert "Rented hosts are unaffected" in change.detail


def test_switching_on_allow_insecure_must_be_retyped():
    current = parse(BASE_YAML)
    candidate = parse(BASE_YAML.replace(
        'transport: {type: http, base_url: "http://127.0.0.1:1"}',
        'transport: {type: http, base_url: "http://10.0.0.5:1", allow_insecure: true}',
    ))
    kinds = {c.kind: c for c in plan_changes(current, candidate)}
    assert kinds["allow_insecure"].requires_retype == "local-1"
    assert "clear text" in kinds["allow_insecure"].detail


def test_a_model_set_change_says_hosts_are_re_prepared():
    current = parse(BASE_YAML)
    candidate = parse(BASE_YAML.replace(f"model_set: [{MODEL}]", f"model_set: [{MODEL}, m2]"))
    change = [c for c in plan_changes(current, candidate) if c.kind == "model_set"][0]
    assert "re-prepared one at a time" in change.detail


def test_raising_the_host_count_must_be_retyped_but_lowering_need_not_be():
    current = parse(BASE_YAML)
    raised = parse(BASE_YAML.replace("max_rented_hosts: 1", "max_rented_hosts: 5"))
    lowered = parse(BASE_YAML.replace("max_rented_hosts: 1", "max_rented_hosts: 0"))

    up = [c for c in plan_changes(current, raised) if c.kind == "max_rented_hosts_raised"][0]
    assert up.requires_retype == "5"
    down = [c for c in plan_changes(current, lowered, [RentedNow("rented-a", 0.1)]) if c.kind == "max_rented_hosts_lowered"][0]
    assert down.requires_retype is None
    assert "1 host(s) beyond it will be drained" in down.detail


def test_no_change_is_no_changes():
    assert plan_changes(parse(BASE_YAML), parse(BASE_YAML)) == []


def test_a_worker_count_change_says_which_way_it_goes():
    current = parse(BASE_YAML)
    raised = parse(BASE_YAML.replace("workers: 2", "workers: 4"))
    lowered = parse(BASE_YAML.replace("workers: 2", "workers: 1"))
    assert "start idle at once" in [c for c in plan_changes(current, raised) if c.kind == "workers_raised"][0].detail
    assert "drain after their current request" in [c for c in plan_changes(current, lowered) if c.kind == "workers_lowered"][0].detail


# --- the store ---


def test_the_store_refuses_a_candidate_that_does_not_load(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(BASE_YAML)
    store = ConfigStore(path)
    assert store.validate(BASE_YAML) == []
    assert store.validate("pool: {}")
    assert path.read_text() == BASE_YAML  # validation never touches the file


def test_applying_is_atomic_and_keeps_the_outgoing_version(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(BASE_YAML)
    store = ConfigStore(path)

    changed = BASE_YAML.replace("max_hourly_burn: 1.00", "max_hourly_burn: 0.25")
    store.apply(changed)

    assert "0.25" in path.read_text()
    assert len(store.history()) == 1
    assert store.text_of(store.history()[0]["version"]) == BASE_YAML
    assert not list(path.parent.glob("*.new"))  # nothing half-written left behind


def test_history_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setattr("gpm_server.configplan.HISTORY_LIMIT", 3)
    path = tmp_path / "pool.yaml"
    path.write_text(BASE_YAML)
    store = ConfigStore(path)
    for burn in ("0.90", "0.80", "0.70", "0.60", "0.50"):
        store.apply(BASE_YAML.replace("max_hourly_burn: 1.00", f"max_hourly_burn: {burn}"))
        time.sleep(0.002)
    assert len(store.history()) == 3


# --- a plan is never silent about part of the file ---

FULL_YAML = edited(
    BASE_YAML,
    "  bidding: {bid_ceiling: 0.60}",
    "  bidding: {bid_ceiling: 0.60, premium: 0.02}\n"
    "  disk_gb: 20\n  model_set_gb: 0.4\n  workers: 2\n  capabilities: [cuda]\n"
    "  offer_policy: {min_gpu_memory_gb: 12, min_disk_gb: 20, max_all_in_hourly: 0.30,\n"
    "                 max_download_per_gb: 0.01, min_download_mbps: 100, min_reliability: 0.95}\n"
    "  teardown: {idle_minutes: 10, deadman_minutes: 20}",
)


def test_the_live_failure_a_resized_rented_block_is_now_in_the_plan():
    """Found applying a real change: disk, download estimate and the offer policy all moved,
    and the plan listed none of them."""
    assert "disk_gb: 20" in FULL_YAML
    current = parse(FULL_YAML)
    candidate = parse(
        FULL_YAML.replace("disk_gb: 20", "disk_gb: 40").replace("model_set_gb: 0.4", "model_set_gb: 29")
        .replace("min_gpu_memory_gb: 12", "min_gpu_memory_gb: 40")
    )
    said = " | ".join(c.detail for c in plan_changes(current, candidate))
    for expected in ("rented.disk_gb changes from 20.0 to 40.0", "rented.model_set_gb", "rented.offer_policy.min_gpu_memory_gb"):
        assert expected in said, said


def test_raising_or_removing_a_price_ceiling_is_loosening_and_must_be_retyped():
    current = parse(FULL_YAML)
    raised = parse(FULL_YAML.replace("max_all_in_hourly: 0.30", "max_all_in_hourly: 3.0"))
    removed = parse(FULL_YAML.replace("max_download_per_gb: 0.01,", ""))
    lowered = parse(FULL_YAML.replace("max_all_in_hourly: 0.30", "max_all_in_hourly: 0.10"))

    up = [c for c in plan_changes(current, raised) if c.kind == "max_all_in_hourly_raised"][0]
    assert up.requires_retype == "3.000" and "$0.300/h to $3.000/h" in up.detail
    gone = [c for c in plan_changes(current, removed) if c.kind == "max_download_per_gb_raised"][0]
    assert gone.requires_retype == "none" and "no limit" in gone.detail
    down = [c for c in plan_changes(current, lowered) if c.kind == "max_all_in_hourly_lowered"][0]
    assert down.requires_retype is None


def test_no_setting_anywhere_in_the_file_can_change_without_the_plan_saying_so():
    """By construction, not by enumeration: every leaf of a fully populated configuration is
    changed, one at a time, and the plan must never come back empty. A setting added to the
    configuration tomorrow is covered the day it is added."""
    from gpm_server.config import PoolConfig
    from gpm_server.configplan import _leaves

    current = parse(FULL_YAML)
    dumped = current.model_dump(mode="json")
    tried, silent = 0, []
    for path, value in _leaves({k: v for k, v in dumped.items() if k != "hosts"}).items():
        if isinstance(value, bool):
            changed = not value
        elif isinstance(value, (int, float)):
            changed = value + 1
        elif isinstance(value, str):
            changed = value + "x"
        else:
            continue  # None and empty containers: nothing to perturb
        mutated = PoolConfig.model_validate(current.model_dump(mode="json")).model_dump(mode="json")
        node = mutated
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = changed
        try:
            candidate = PoolConfig.model_validate(mutated)
        except ValueError:
            continue  # not a value this setting accepts; the loader refuses it before any plan
        tried += 1
        if not plan_changes(current, candidate):
            silent.append(path)
    assert tried >= 30, tried  # the test is only worth something if it really walked the file
    assert silent == [], f"the plan said nothing about: {silent}"


@pytest.mark.parametrize("serialise, passes, fix", [(True, False, "engine_settings"), (False, True, "")])
def test_the_concurrency_check_tells_a_serialising_engine_from_a_parallel_one(console, serialise, passes, fix):
    """The check the console's Test connection rests on, against an engine that really does
    run one request at a time — and one that does not. A failure names its fix in a form a
    client can act on: the pool can now apply it itself through a host's agent (D41)."""
    from fakes.fake_ollama import FakeOllama

    _, url, _, loop = console
    engine = FakeOllama(resident={MODEL})
    engine.generate_delay_s, engine.serialise = 0.25, serialise
    server = ServerHandle(engine.app, loop)
    try:
        with client(url) as http:
            result = http.post("/pool/hosts/test", json={"base_url": server.base_url, "workers": 4}).json()
    finally:
        server.stop()
    step = next(s for s in result["steps"] if s["name"] == "concurrency")
    assert step["ok"] is passes, step
    assert step["fix"] == fix
    assert ("serialising" in step["detail"]) is serialise
