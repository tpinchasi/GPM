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
      offer_policy: {{ min_disk_gb: 10, max_all_in_hourly: 0.60 }}
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


def _a_browser() -> str | None:
    found = shutil.which("google-chrome") or shutil.which("chromium")
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    return found or (mac if Path(mac).exists() else None)


@pytest.mark.timeout(180)
def test_a_browser_really_runs_the_page_to_its_last_line():
    """The bracket check is not a parser, and `node` is not on every machine. Twice in one day
    a syntax error reached a running console — once a dropped `)`, once an arrow eaten by an
    edit (`const feed = () =`) — and both times every Python test passed, because nothing
    executed the page. A browser does.

    Loading the real page and finding the key dialog open proves the script parsed *and* ran
    to its final statement, which is what opens it.
    """
    browser = _a_browser()
    if browser is None:
        pytest.skip("no browser here; `node --check` above and CI cover the parse")

    import tempfile

    with tempfile.TemporaryDirectory() as profile:
        # It prints the DOM and then does not exit, so take what it printed and end it.
        browsing = subprocess.Popen(
            [browser, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
             "--virtual-time-budget=4000", "--dump-dom", (STATIC / "index.html").as_uri()],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            # A cold headless browser on a loaded machine (which is when CI runs) takes far
            # longer than one on an idle laptop, so this waits longer than the 30 s that made
            # it flaky — and the test's own limit is raised above it, or the two would race.
            dom, complaints = browsing.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            browsing.kill()
            dom, complaints = browsing.communicate()

    assert 'id="key-dialog" open' in dom, (
        "the page did not run to its last line — the script failed to parse or threw:\n"
        + dom[-600:] + "\n" + complaints[-2000:]
    )


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

    typo = edited(path.read_text(), "max_all_in_hourly: 0.60", "max_all_in_hourly: 6.0")
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": typo}).json()

    raised = [c for c in plan["changes"] if c["kind"] == "max_all_in_hourly_raised"]
    assert raised, plan
    assert raised[0]["requires_retype"] is True
    assert raised[0]["value"] == "6.000"
    assert "$0.600/h to $6.000/h" in raised[0]["detail"]
    # And nothing has changed until Apply.
    assert supervisor.config.rented.max_all_in_hourly == 0.60


def test_plan_says_which_running_host_already_costs_more_than_a_lowered_ceiling(console):
    supervisor, url, path, loop = console
    supervisor.fleet.open_lease(workers=4, max_hours=2, max_spend=2.0, allow_rent=True)
    loop.run(supervisor.fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={}))
    host = next(iter(supervisor.fleet.hosts.values()))

    lowered = edited(path.read_text(), "max_all_in_hourly: 0.60", "max_all_in_hourly: 0.05")
    with client(url) as http:
        plan = http.post("/pool/config/plan", json={"text": lowered}).json()

    change = [c for c in plan["changes"] if c["kind"] == "max_all_in_hourly_lowered"][0]
    assert host.host_id in change["detail"]
    assert "already costs more" in change["detail"]
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
        response = http.post("/pool/market/preview", json={"offer_policy": {"max_all_in_hourly": "not a number"}})
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
    "  offer_policy: { min_disk_gb: 10, max_all_in_hourly: 0.60 }",
    "  bidding: {premium: 0.02}\n"
    "  workers: 2\n  capabilities: [cuda]\n"
    "  offer_policy: {min_gpu_memory_gb: 12, min_disk_gb: 20, max_all_in_hourly: 0.30,\n"
    "                 max_download_per_gb: 0.01, min_download_mbps: 100, min_reliability: 0.95}\n"
    "  teardown: {idle_minutes: 10, deadman_minutes: 20}",
)


def test_the_live_failure_a_resized_rented_block_is_now_in_the_plan():
    """Found applying a real change: disk, download estimate and the offer policy all moved,
    and the plan listed none of them."""
    assert "min_disk_gb: 20" in FULL_YAML
    current = parse(FULL_YAML)
    candidate = parse(
        FULL_YAML.replace("min_disk_gb: 20", "min_disk_gb: 40")
        .replace("min_gpu_memory_gb: 12", "min_gpu_memory_gb: 40")
    )
    said = " | ".join(c.detail for c in plan_changes(current, candidate))
    for expected in ("hosts are rented with 40 GB of disk, from 20 GB", "rented.offer_policy.min_gpu_memory_gb"):
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


async def test_a_market_that_could_not_be_asked_is_not_reported_as_an_empty_market(console):
    """Found live: the provider rate-limited the pool, the offer search returned [], and the
    console said "0 offers seen" — which reads as "the market has nothing in it". The two are
    entirely different problems and must not look the same (D44)."""
    from gpm_server.providers.base import ProviderRateLimited

    supervisor, url, _, loop = console

    async def refuse(_query):
        raise ProviderRateLimited("POST /api/v0/bundles: rate limited")

    supervisor.fleet.provider.search_offers = refuse
    preview = loop.run(supervisor.fleet.market_preview(hours=1))

    assert preview["seen"] == 0
    assert "rate limited" in preview["problem"]

    # And when the market really is empty, nothing is blamed on the provider.
    async def nothing(_query):
        return []

    supervisor.fleet.provider.search_offers = nothing
    supervisor.fleet._offer_retry_at = 0.0  # past the back-off the refusal above set
    assert loop.run(supervisor.fleet.market_preview(hours=1))["problem"] is None


async def test_a_rate_limited_market_is_backed_off_rather_than_asked_again_next_pass(console):
    """Asking again on the next pass is what earned the refusal in the first place (D44)."""
    from gpm_server.providers.base import ProviderRateLimited

    supervisor, url, _, loop = console
    asked = []

    async def refuse(_query):
        asked.append(1)
        raise ProviderRateLimited("rate limited")

    supervisor.fleet.provider.search_offers = refuse
    for _ in range(4):
        loop.run(supervisor.fleet.market_preview(hours=1))

    assert len(asked) == 1, "the provider was asked again while backing off"
    preview = loop.run(supervisor.fleet.market_preview(hours=1))
    assert "not asking again" in preview["problem"]

    # Refused again after the wait: a minute again, never longer (D119).
    for _ in range(3):
        supervisor.fleet._offer_retry_at = 0.0
        loop.run(supervisor.fleet.market_preview(hours=1))
        assert supervisor.fleet._offer_backoff_s == 60.0
    assert len(asked) == 4

    # Once the window passes and the provider answers, the backoff clears.
    supervisor.fleet._offer_retry_at = 0.0

    async def works(_query):
        return []

    supervisor.fleet.provider.search_offers = works
    assert loop.run(supervisor.fleet.market_preview(hours=1))["problem"] is None
    assert supervisor.fleet._offer_backoff_s == 0.0


# --- following one host while it is prepared (D48) ---


def test_the_decision_log_can_be_asked_for_one_hosts_slice(console):
    supervisor, url, _, _ = console
    supervisor.events.record("rented", "a", host_id="host-a")
    supervisor.events.record("rented", "b", host_id="host-b")
    supervisor.events.record("prepared", "a again", host_id="host-a")

    with client(url) as http:
        mine = http.get("/pool/events?host_id=host-a").json()["events"]
        everything = http.get("/pool/events").json()["events"]

    assert [e["summary"] for e in mine] == ["a again", "a"]
    assert len(everything) > len(mine)


def test_one_host_answers_with_its_state_its_engine_and_its_own_events(console):
    supervisor, url, _, _ = console
    supervisor.events.record("rented", "something about it", host_id="local-1")

    with client(url) as http:
        detail = http.get("/pool/hosts/local-1").json()
        unknown = http.get("/pool/hosts/nobody")

    assert detail["host_id"] == "local-1" and detail["kind"] == "local"
    assert detail["required_tags"] == [MODEL]
    # Nothing listens at the fixture's address, so the honest answer is "not answering".
    assert detail["engine"]["answers"] is False
    assert [e["summary"] for e in detail["events"]] == ["something about it"]
    assert "stage_detail" in detail
    assert unknown.status_code == 404


@pytest.mark.parametrize("detail, expected", [
    ({"state": "ready"}, "ready — serving requests"),
    ({"state": "parked"}, "parked"),
    ({"state": "preparing", "engine": {"answers": False}, "provider": {"detail": "Pulling from ollama/ollama"}},
     "the provider is still starting"),
    ({"state": "preparing", "engine": {"answers": False}}, "waiting for the engine to answer"),
    ({"state": "preparing", "engine": {"answers": True}, "stage": "downloading m1",
      "progress": {"m1": {"completed": 6_000_000_000, "total": 18_600_000_000}}},
     "downloading m1 — 6.0 of 18.6 GB"),
    ({"state": "preparing", "engine": {"answers": True, "missing_from_disk": ["m1"], "not_loaded": ["m1"]}},
     "models still to download: m1"),
    ({"state": "preparing", "engine": {"answers": True, "missing_from_disk": [], "not_loaded": ["m1"]}},
     "downloaded; loading into memory: m1"),
    ({"state": "preparing", "engine": {"answers": True, "missing_from_disk": [], "not_loaded": []}},
     "waiting for the next probe"),
])
def test_the_stage_is_said_in_words_and_derived_from_what_is_known(detail, expected):
    from gpm_server.supervisor.control import _stage_of

    assert expected in _stage_of(detail)


def test_each_model_has_one_state_from_everything_the_pool_knows():
    """The fifth vLLM rental, as the console showed it: "not answering: ReadError" beside a
    download bar, no state per model while they landed, and "not on disk" for three models
    the agent held — because every word came from the engine, which for vLLM is silent by
    design until the weights are there (D105)."""
    from gpm_server.supervisor.control import _model_states

    required = {"a/e4b", "a/embed", "a/26b"}
    agent = {"models": [
        {"tag": "a/e4b", "on_disk": True, "loaded": False, "pulling": None, "error": None},
        {"tag": "a/embed", "on_disk": False, "loaded": False, "pulling": {"tag": "a/embed", "completed_bytes": 5e8, "total_bytes": 1.9e9}, "error": None},
        {"tag": "a/26b", "on_disk": False, "loaded": False, "pulling": None, "error": None},
    ]}
    # While the agent fetches: one landed, one on the wire, one not begun; the engine silent.
    fetching = {m["tag"]: m for m in _model_states(
        required, progress={}, agent_models=agent, resident=frozenset(), available=frozenset(),
        loads_by_restart=True, restart_asked=False)}
    assert fetching["a/e4b"]["state"] == "on disk" and "starts once every model" in fetching["a/e4b"]["detail"]
    assert fetching["a/embed"]["state"] == "downloading" and fetching["a/embed"]["detail"] == "0.5 of 1.9 GB"
    assert fetching["a/26b"]["state"] == "not here yet"
    # The pool's own download record, with its measured rate, wins over the agent's.
    with_rate = _model_states(
        {"a/embed"}, progress={"a/embed": {"completed": 1e9, "total": 1.9e9, "mbps": 812}},
        agent_models=agent, resident=frozenset(), available=frozenset(), loads_by_restart=True, restart_asked=False)
    assert with_rate[0]["detail"] == "1.0 of 1.9 GB at 812 Mbps" and with_rate[0]["completed"] == 1e9
    # All landed and the restart asked: loading, not "still to download".
    landed = {"models": [{"tag": t, "on_disk": True, "loaded": False, "awaiting_restart": True} for t in required]}
    loading = _model_states(required, progress={}, agent_models=landed, resident=frozenset(), available=frozenset(),
                            loads_by_restart=True, restart_asked=True)
    assert {m["state"] for m in loading} == {"loading"}
    # The engine serving one, the agent reporting another dead: each says which.
    dead = {"models": [{"tag": "a/26b", "on_disk": True, "loaded": False, "error": "ValueError: batch too small"}]}
    mixed = {m["tag"]: m for m in _model_states(
        required, progress={}, agent_models=dead, resident=frozenset({"a/e4b"}), available=frozenset({"a/embed"}),
        loads_by_restart=True, restart_asked=True)}
    assert mixed["a/e4b"]["state"] == "loaded"
    assert mixed["a/26b"] == {"tag": "a/26b", "state": "failed", "detail": "ValueError: batch too small"}
    assert mixed["a/embed"]["state"] == "loading"
    # An engine that loads for itself: on disk means loading, no restart involved.
    own = _model_states({"m"}, progress={}, agent_models=None, resident=frozenset(), available=frozenset({"m"}),
                        loads_by_restart=False, restart_asked=False)
    assert own[0]["state"] == "loading" and own[0]["detail"] == "downloaded; loading into memory"


@pytest.mark.parametrize("detail, expected", [
    ({"state": "preparing", "engine": {"answers": False, "expected": True, "detail": "not started yet"},
      "models": [{"tag": "a", "state": "on disk"}, {"tag": "b", "state": "downloading", "detail": "6.0 of 18.6 GB"},
                 {"tag": "c", "state": "not here yet"}]},
     "fetching models: 1 of 3 on disk; downloading b — 6.0 of 18.6 GB"),
    ({"state": "preparing", "engine": {"answers": False, "expected": True},
      "models": [{"tag": "a", "state": "on disk"}, {"tag": "b", "state": "loading"}]},
     "every model is on disk; the engine is starting on them"),
    ({"state": "preparing", "engine": {"answers": True, "missing_from_disk": [], "not_loaded": ["b"]},
      "models": [{"tag": "a", "state": "loaded"}, {"tag": "b", "state": "failed", "detail": "ValueError: no"}]},
     "b failed to start: ValueError: no"),
])
def test_the_stage_says_what_the_machine_is_doing_while_its_engine_is_silent_on_purpose(detail, expected):
    from gpm_server.supervisor.control import _stage_of

    assert _stage_of(detail) == expected


def test_a_silent_engine_is_a_fault_only_when_it_is_not_the_plan():
    from types import SimpleNamespace

    from gpm_server.supervisor.control import _silent_engine

    fetching = SimpleNamespace(agent=object(), restart_asked_at=None)
    assert _silent_engine("ReadError", fetching, loads_by_restart=True)["expected"] is True
    starting = SimpleNamespace(agent=object(), restart_asked_at=time.time() - 30)
    said = _silent_engine("ReadError", starting, loads_by_restart=True)
    assert said["expected"] is True and "starting" in said["detail"] and "30s ago" in said["detail"]
    # No agent yet: the machine may still be booting, and that is what is said.
    booting = SimpleNamespace(agent=None, restart_asked_at=None)
    assert _silent_engine("ReadError", booting, loads_by_restart=True) == {"answers": False, "detail": "ReadError"}
    # An engine that loads for itself is expected to answer as soon as the machine is up.
    assert _silent_engine("ReadError", fetching, loads_by_restart=False) == {"answers": False, "detail": "ReadError"}


def test_the_console_shows_each_models_state_and_an_expected_silence_as_no_fault():
    """Structural: the panel draws the per-model states the supervisor derives, and does not
    paint an engine that is silent by design red."""
    source = (STATIC / "app.js").read_text()
    assert "d.models" in source and "m.state" in source
    assert "engine.expected" in source


def test_what_the_agent_holds_on_disk_counts_as_on_disk():
    """An engine launched with its models can only name what it serves, so while its processes
    load it says nothing is on disk. Found live: three fetched models read "still to download"
    in the console until the engine came up — or, that time, died. The agent's report is the
    fact the engine cannot give."""
    from gpm_server.supervisor.control import _held_on_disk

    assert _held_on_disk(None) == frozenset()
    assert _held_on_disk({"models": [
        {"tag": "a/one", "on_disk": True, "loaded": False},
        {"tag": "a/two", "on_disk": False, "pulling": {"tag": "a/two"}},
        {"tag": "a/three", "on_disk": True, "loaded": True},
        {"on_disk": True},
    ]}) == frozenset({"a/one", "a/three"})


def test_the_console_opens_a_host_and_follows_it():
    """Structural: the page must ask for one host and keep asking while the panel is open."""
    script = (STATIC / "app.js").read_text()
    assert "/pool/hosts/${encodeURIComponent(id)}" in script
    assert "hostLink" in script and "setInterval(drawHost" in script
    assert "clearInterval(hostWatch.timer)" in script  # and stop when it is closed


async def test_extending_a_lease_keeps_the_host_it_was_extended_for(console):
    """The point of extending in flight: the host must outlive the original hour too."""
    supervisor, url, _, loop = console
    host = loop.run(supervisor.fleet.prepare(max_spend=1.00, max_hours=1.0))
    was_held_until = host.hold_until
    lease_id = host.lease_id

    with client(url) as http:
        answer = http.patch(f"/pool/leases/{lease_id}", json={"max_hours": 5.0, "confirm": "5.0"})

    assert answer.status_code == 200
    assert answer.json()["hosts_held"] == [host.host_id]
    assert host.hold_until > was_held_until + 3 * 3600
    extended = next(e for e in supervisor.events.recent(20) if e["kind"] == "lease_extended")
    assert host.host_id in extended["summary"] and "max_hours 1.0 → 5.0" in extended["summary"]


def test_a_lease_can_be_acted_on_wherever_it_is_shown():
    """The overview and the Leases screen share one set of buttons, so they cannot drift into
    offering different things for the same lease."""
    script = (STATIC / "app.js").read_text()
    assert script.count("leaseActions(lease)") == 2  # the overview table, and the leases table
    actions = script[script.index("const leaseActions"):script.index("const leaseActions") + 900]
    for label in ("Tighten", "Extend", "Close"):
        assert f'"{label}")' in actions, label


# --- changing the offer search from the Rented capacity screen (D51) ---


def test_a_setting_is_changed_in_place_leaving_the_rest_of_the_file_alone():
    """Loading the YAML and dumping it back would delete every comment an operator wrote."""
    from gpm_server.configplan import set_values

    before = (
        "rented:\n"
        "  provider: vast            # the account credential is read from the environment\n"
        "  offer_policy:\n"
        "    min_gpu_memory_gb: 80\n"
        "    max_download_per_gb: 0.015   # half a cent per GB\n"
        "    verified_only: true\n"
        "  bidding: { premium: 0.02 }\n"
    )
    after = set_values(before, ("rented", "offer_policy"), {"min_gpu_memory_gb": 48, "avoid_machines": ["144381"]})
    after = set_values(after, ("rented", "bidding"), {"premium": 0.05, "attempts": 5})

    assert "# the account credential is read from the environment" in after
    assert "max_download_per_gb: 0.015   # half a cent per GB" in after  # untouched, comment kept
    assert "    min_gpu_memory_gb: 48\n" in after
    assert '    avoid_machines: ["144381"]\n' in after  # added under the right block
    assert "  bidding: { premium: 0.05, attempts: 5 }\n" in after  # flow style kept


def test_a_nested_one_line_mapping_is_edited_whole_not_cut_at_its_first_comma():
    """Found writing a search profile back (D108): `search_profiles: { cheap: { a: 1, b: 2 } }`
    was split at every comma, the new profile landed beside half of the old one, and the file
    no longer loaded. The owner's own file writes its profiles this way."""
    import yaml
    from gpm_server.configplan import set_values

    before = (
        "rented:\n"
        '  search_profiles: { cheap tests: { min_disk_gb: 20, exclude_hardware: ["CMP", "P106"] }, other: {min_disk_gb: 5} }\n'
    )
    after = set_values(before, ("rented", "search_profiles"), {"cheap tests": {"min_disk_gb": 40}})
    assert yaml.safe_load(after)["rented"]["search_profiles"] == {
        "cheap tests": {"min_disk_gb": 40}, "other": {"min_disk_gb": 5},
    }


def test_a_setting_the_pool_cannot_place_unambiguously_is_refused_not_guessed():
    from gpm_server.configplan import CannotEdit, set_values

    with pytest.raises(CannotEdit, match="not in the configuration"):
        set_values("rented:\n  provider: vast\n", ("rented", "offer_policy"), {"min_disk_gb": 1})
    twice = "rented:\n  offer_policy:\n    min_disk_gb: 10\n    min_disk_gb: 20\n"
    with pytest.raises(CannotEdit, match="more than once"):
        set_values(twice, ("rented", "offer_policy"), {"min_disk_gb": 30})


def test_the_search_can_be_changed_from_the_rented_screen(console):
    supervisor, url, path, _ = console
    path.write_text(edited(path.read_text(), "  offer_policy: { min_disk_gb: 10, max_all_in_hourly: 0.60 }",
                           "  offer_policy: {min_disk_gb: 10, max_all_in_hourly: 0.60, min_gpu_memory_gb: 24}"))
    supervisor.reload_config()

    with client(url) as http:
        tightened = http.patch("/pool/config/rented", json={"offer_policy": {"min_gpu_memory_gb": 48}})
        assert tightened.status_code == 200, tightened.text
        assert supervisor.config.rented.offer_policy.min_gpu_memory_gb == 48

        # Raising a price ceiling is loosening: refused, and the refusal carries the plan.
        loosened = http.patch("/pool/config/rented", json={"offer_policy": {"max_all_in_hourly": 5.0}})
        assert loosened.status_code == 400 and loosened.json()["error"] == "not_confirmed"
        retype = [c for c in loosened.json()["changes"] if c["requires_retype"]][0]
        assert supervisor.config.rented.max_all_in_hourly == 0.60  # nothing applied

        confirmed = http.patch("/pool/config/rented", json={"offer_policy": {"max_all_in_hourly": 5.0}, "confirm": retype["value"]})
        assert confirmed.status_code == 200
        assert supervisor.config.rented.max_all_in_hourly == 5.0

        nonsense = http.patch("/pool/config/rented", json={"offer_policy": {"min_reliability": "soon"}})
    assert nonsense.status_code == 400 and nonsense.json()["error"] == "invalid_config"


def test_with_a_search_profile_in_force_the_screens_edits_go_into_that_profile(console):
    """Found live: the Finding tab showed the profile in force and wrote its edits into the
    pool's own offer_policy, which was not in force — the owner's max all-in "kept showing
    0.85" while the new value sat in the file unused (D108)."""
    supervisor, url, path, _ = console
    path.write_text(edited(
        path.read_text(), "  offer_policy: { min_disk_gb: 10, max_all_in_hourly: 0.60 }",
        "  offer_policy: { min_disk_gb: 10, max_all_in_hourly: 0.60 }\n"
        "  search_profiles: { cheap: { min_disk_gb: 20, max_all_in_hourly: 0.50 } }\n"
        "  search_profile: cheap",
    ))
    supervisor.reload_config()

    with client(url) as http:
        answer = http.patch("/pool/config/rented", json={"offer_policy": {"min_disk_gb": 40}})
    assert answer.status_code == 200, answer.text
    rented = supervisor.config.rented
    assert rented.search_profiles["cheap"].min_disk_gb == 40, "the profile in force took the edit"
    assert rented.search_profiles["cheap"].max_all_in_hourly == 0.50, "and kept the rest of itself"
    assert rented.offer_policy.min_disk_gb == 10, "the unused policy is left alone"
    assert rented.disk_gb == 40


def test_raising_the_price_by_switching_to_a_dearer_profile_must_be_retyped(console):
    """The search in force is what the pool pays by, so switching profile can raise the price
    as surely as editing it (D108)."""
    supervisor, url, path, _ = console
    path.write_text(edited(
        path.read_text(), "  offer_policy: { min_disk_gb: 10, max_all_in_hourly: 0.60 }",
        "  offer_policy: { min_disk_gb: 10, max_all_in_hourly: 0.60 }\n"
        "  search_profiles: { dear: { min_disk_gb: 10, max_all_in_hourly: 4.00 } }",
    ))
    supervisor.reload_config()
    with client(url) as http:
        answer = http.patch("/pool/config/rented", json={"search_profile": "dear"})
    assert answer.status_code == 400 and answer.json()["error"] == "not_confirmed"
    assert any(c["kind"] == "max_all_in_hourly_raised" for c in answer.json()["changes"])


def test_the_preview_says_what_is_saved_so_the_form_is_the_pools_own(console):
    supervisor, url, _, loop = console
    preview = loop.run(supervisor.fleet.market_preview(hours=1))
    assert preview["saved"]["offer_policy"]["max_all_in_hourly"] == supervisor.config.rented.max_all_in_hourly
    assert "min_gpu_memory_gb" in preview["saved"]["offer_policy"]


# --- how capacity is decided, edited where renting is watched (D74) ---


def test_the_allocation_panel_is_built_from_what_the_pool_reports():
    """Like the offer search before it, the fields are the pool's own values — never a copy in
    the page that can drift from the file."""
    source = (STATIC / "app.js").read_text()

    assert "allocationSection" in source and "...allocationSection(market)" in source
    for field in ("target_utilisation", "ramp_factor", "max_round", "min_hosts"):
        assert field in source, f"{field} is not editable in the console"
    assert "workers_auto" in source
    # And it saves through the same path as the search: file, plan, retype.
    assert "api.setSearch" in source.split("function saveAllocation")[1].split("function ")[0]


async def test_the_worker_ceiling_can_be_raised_without_the_cli(console):
    """Found live (D80): a pool refusing 70% of its requests sat at "the load has cleared"
    because its lease allowed 5 workers and one rented host already supplied 6. Under dynamic
    allocation that number is what decides whether another host is ever rented, and it was the
    one field the console's Extend dialog could not change."""
    supervisor, url, _, loop = console
    host = loop.run(supervisor.fleet.prepare(max_spend=1.00, max_hours=1.0))
    lease_id = host.lease_id
    before = supervisor.leases.get(lease_id).workers

    with client(url) as http:
        refused = http.patch(f"/pool/leases/{lease_id}", json={"workers": before + 40})
        raised = http.patch(
            f"/pool/leases/{lease_id}", json={"workers": before + 40, "confirm": str(before + 40)}
        )

    assert refused.status_code == 400, "raising a ceiling went through unconfirmed"
    assert raised.status_code == 200
    assert supervisor.leases.get(lease_id).workers == before + 40
    extended = next(e for e in supervisor.events.recent(20) if e["kind"] == "lease_extended")
    assert f"workers {before} → {before + 40}" in extended["summary"]


def test_the_extend_dialog_asks_for_the_worker_ceiling():
    """Structural, like the rest here: the page is served to a browser, never imported, so
    nothing else would notice the field going missing again."""
    script = (STATIC / "app.js").read_text()
    dialog = script[script.index("async function extendLease"):]
    dialog = dialog[: dialog.index("\n}")]

    assert "lease.workers" in dialog, "the dialog does not offer the current ceiling"
    assert "Workers it may reach" in dialog, "the operator is never asked for it"
    assert "workers: Number(workers.input.value)" in dialog, "the chosen ceiling is never sent"
    assert '"workers", Number(workers.input.value)' in dialog, "raising it is not confirmed like the rest"
    # One form, not a chain of browser prompts (D86).
    assert "prompt(" not in dialog, "the dialog still asks one question at a time"


async def test_how_the_pool_rents_is_set_where_renting_is_watched(console):
    """Found live (D80): a pool on the default `interruptible` never asks the on-demand
    listing, so an operator watching a fixed-price host in the market could not learn why it
    was never rented — it was not rejected, it was never seen. The only way to change it was
    the raw configuration file."""
    supervisor, url, _, loop = console
    was = supervisor.config.rented.mode
    assert was != "cheaper", "this pool already rents both ways; the test proves nothing"

    with client(url) as http:
        before = http.get("/pool/market/preview?hours=1").json()
        answer = http.patch("/pool/config/rented", json={"mode": "cheaper"})
        after = http.get("/pool/market/preview?hours=1").json()

    assert before["saved"]["mode"] == was, "the screen did not show the mode in force"
    assert answer.status_code == 200, answer.text
    assert supervisor.config.rented.mode == "cheaper"
    assert after["saved"]["mode"] == "cheaper", "the screen would still show the old mode"


def test_the_rented_screen_offers_every_way_of_renting():
    script = (STATIC / "app.js").read_text()
    assert "const MODES" in script
    for mode in ("interruptible", "on_demand", "cheaper"):
        assert f'["{mode}"' in script, f"{mode} cannot be chosen in the console"
    assert "body.mode = mode.value" in script, "the chosen mode is never sent"


async def test_every_teardown_lever_is_editable_in_the_console(console):
    """D86: the whole `teardown` block was missing from the console — fifteen fields deciding
    how long a host that is not working still bills. Changing how long a stuck host is given
    before it is dropped meant editing the supervisor's own file."""
    supervisor, url, _, loop = console

    with client(url) as http:
        before = http.get("/pool/market/preview?hours=1").json()
        answer = http.patch("/pool/config/rented", json={"teardown": {"max_starting_minutes": 4}})
        after = http.get("/pool/market/preview?hours=1").json()

    assert "teardown" in before["saved"], "the console is never told the tear-down settings"
    assert answer.status_code == 200, answer.text
    assert supervisor.config.rented.teardown.max_starting_minutes == 4
    assert after["saved"]["teardown"]["max_starting_minutes"] == 4


def test_the_teardown_levers_are_offered_as_sliders_and_dropdowns():
    """The owner asked for sliders and dropdowns, not empty number boxes: a bare box does not
    say whether 30 is high or low for a field you have never set."""
    script = (STATIC / "app.js").read_text()
    from gpm_server.config import TeardownConfig

    for field in TeardownConfig.model_fields:
        assert f'"{field}"' in script, f"teardown.{field} cannot be changed in the console"
    assert "function slider(" in script, "no slider control exists"
    assert '"deadman_action", "choice"' in script, "an enum is not offered as a dropdown"
    assert '"park_when_idle", "checkbox"' in script


async def test_a_search_can_be_saved_under_a_name_and_chosen_again(console):
    """The owner: "I want to be able to save a profile of search, name it and select it from a
    drop down" (D87). One pool wants a different market on different days — cheap and slow for
    an overnight batch, fast and dear for a demo — and rewriting nine filters by hand each time
    is how a filter gets left behind."""
    supervisor, url, _, loop = console

    with client(url) as http:
        saved = http.patch("/pool/config/rented", json={
            "save_profile_as": "cheap-and-slow",
            "offer_policy": {"max_all_in_hourly": 0.40, "min_download_mbps": 10},
        })
        after = http.get("/pool/market/preview?hours=1").json()

    assert saved.status_code == 200, saved.text
    rented = supervisor.config.rented
    assert "cheap-and-slow" in rented.search_profiles
    assert rented.search_profile == "cheap-and-slow", "saving it did not select it"
    assert rented.policy_in_force.max_all_in_hourly == 0.40, "the pool is not searching with it"
    assert after["saved"]["search_profile"] == "cheap-and-slow"
    assert "cheap-and-slow" in after["saved"]["search_profiles"]


def test_a_profile_that_does_not_exist_is_refused_at_load():
    """Falling back to the default policy silently would have the operator watch the pool buy
    from a market they thought they had left."""
    import pytest as _pytest
    from gpm_server.config import RentedConfig

    with _pytest.raises(ValueError, match="not among the search_profiles"):
        RentedConfig(provider="fake", offer_policy={"min_disk_gb": 10, "max_all_in_hourly": 1.0},
                     search_profile="no-such-thing")


def test_the_search_profile_is_chosen_from_a_dropdown():
    script = (STATIC / "app.js").read_text()
    assert "Search profile" in script, "there is no way to pick one"
    assert "saved.search_profiles" in script, "the saved names are never listed"
    assert "save these as" in script, "there is no way to save one"
    assert "body.save_profile_as" in script
