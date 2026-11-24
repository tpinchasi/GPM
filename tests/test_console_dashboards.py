"""The Dashboards screen and named app keys on the console, in a real browser (D193, D197).

The API behind them is covered in test_app_keys.py and test_stats.py; this proves the page draws
its charts — titled, with a legend and a table of the same numbers — switches tabs and ranges,
fits a phone's width, and makes and revokes a named app key, showing the key once. Skipped where
no browser is installed.
"""

import json
import time

import pytest
from fakes.browser import a_browser, open_page
from gpm_server.app_keys import AppKeyStore
from gpm_server.db import RequestLog, RequestRecord
from test_host_removal import serve

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

ADMIN_KEY = "gpmx_console_dashboards"
POOL = f"""
pool: {{name: shop, model_set: [chatter, writer], probe_interval_s: 3600}}
auth: {{admin_keys: ["{ADMIN_KEY}"], app_keys: [gpma_console_dashboards]}}
hosts:
  - id: mac
    kind: local
    transport: {{type: http, base_url: "http://127.0.0.1:1"}}
"""

HELPERS = r"""
window.__text = () => document.getElementById('screen').textContent;
window.__button = (text) => [...document.querySelectorAll('#screen button')].find((b) => b.textContent.trim() === text);
window.__type = (node, value) => { node.value = value; node.dispatchEvent(new Event("input")); return true; };
true;
"""


@pytest.fixture
def served(tmp_path):
    supervisor, server, loop, database, path = serve(tmp_path, POOL)
    log = RequestLog(database)
    now = time.time()
    for i in range(30):
        log._insert(RequestRecord(request_id=f"a{i}", outcome="ok", status_code=200, ts=now - i * 600, host_id="mac",
                                  model_requested="chatter", kind="chat", latency_ms=300.0 + i * 10, tokens_out=40,
                                  app="support-bot"))
    for i in range(10):
        log._insert(RequestRecord(request_id=f"b{i}", outcome="failed" if i < 2 else "ok", status_code=502 if i < 2 else 200,
                                  ts=now - i * 3000, host_id="mac", model_requested="writer", kind="chat", latency_ms=900.0))
    try:
        yield supervisor, server.base_url, database
    finally:
        server.stop()
        loop.stop()
        database.close()


def sign_in(page):
    page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
    page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
            "document.getElementById('key-form').requestSubmit(); true")
    page.js(HELPERS)


@pytest.mark.timeout(240)
def test_dashboards_draw_each_tab_with_charts_legends_and_tables(served):
    _, url, _ = served
    with open_page(f"{url}/ui/#dashboards") as page:
        sign_in(page)
        page.until("document.querySelectorAll('svg.chart').length >= 2", within=30, what="the Apps charts")
        assert page.js("document.querySelector('#nav a.active').textContent") == "Dashboards"
        text = page.js("__text()")
        assert "support-bot" in text and "shared pool (configured app key)" in text
        # Every chart names itself to a screen reader, and its numbers are in a table beside it.
        assert page.js("[...document.querySelectorAll('svg.chart')].every((s) => s.getAttribute('role') === 'img' "
                       "&& s.querySelector('title').textContent.length > 0 && s.querySelector('desc'))")
        assert page.js("document.querySelectorAll('figure.chart-panel details.chart-table table').length") >= 2
        assert page.js("[...document.querySelectorAll('ul.legend li')].some((li) => li.textContent === 'support-bot')")
        # The table of apps: the named key's thirty, the shared key's ten with two failed.
        rows = page.js("[...document.querySelectorAll('table.dash-table tbody tr')].map((r) => r.textContent)")
        assert any("support-bot" in r and "30" in r for r in rows), rows
        assert any("shared pool" in r and "2 (20.0%)" in r for r in rows), rows
        # A tooltip per time slot names each series' value there.
        assert "support-bot:" in page.js("document.querySelector('svg.chart rect.slot title').textContent")

        page.js("location.hash = '#dashboards/models'; true")
        page.until("__text().includes('Each model') && __text().includes('writer')", within=30, what="the Models tab")
        assert page.js("document.querySelector('.tabs a.active').textContent") == "Models"

        page.js("__button('7 d').click(); true")
        page.until("document.querySelector('.chip.on') && document.querySelector('.chip.on').textContent === '7 d'",
                   within=30, what="the 7-day range")
        page.js("__button('30 d').click(); true")
        page.until("__text().includes('Requests per day')", within=30, what="daily buckets for 30 days")

        for tab, heading in (("hosts", "Each host"), ("providers", "Each provider")):
            page.js(f"location.hash = '#dashboards/{tab}'; true")
            page.until(f"__text().includes({json.dumps(heading)})", within=30, what=f"the {tab} tab")
        assert "own machines" in page.js("__text()")

        # At a phone's width, every chart fits the screen.
        page.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
        page.js("__button('Refresh').click(); true")
        page.until("document.querySelectorAll('svg.chart').length >= 1", within=30, what="charts redrawn")
        page.until("[...document.querySelectorAll('svg.chart')].every((s) => s.getBoundingClientRect().right <= window.innerWidth + 1)",
                   within=10, what="charts within a phone's width")


@pytest.mark.timeout(240)
def test_a_named_app_key_is_made_shown_once_and_revoked(served):
    _, url, database = served
    with open_page(f"{url}/ui/#workloads") as page:
        sign_in(page)
        page.until("__text().includes('Apps on the shared pool')", within=30, what="the app keys section")
        field = "[...document.querySelectorAll('.app-keys input')].find((i) => i.placeholder.startsWith('e.g.'))"
        page.js(f"__type({field}, 'Bad Name')")
        assert page.js("__button('Make key').disabled") is True, "a name that is not a name"
        page.js(f"__type({field}, 'support-bot')")
        page.js("__button('Make key').click(); true")
        page.until("!!document.querySelector('.app-keys .key-panel code.secret')", within=30, what="the key, shown once")
        key = page.js("document.querySelector('.app-keys .key-panel code.secret').textContent")
        assert key.startswith("gpma_")
        assert [k.app for k in AppKeyStore(database).all()] == ["support-bot"]

        # Put away: not shown again, the app listed with its key's id.
        page.js("__button('I have saved it').click(); true")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="the confirmation")
        page.js("document.getElementById('confirm-ok').click(); true")
        page.until("!document.querySelector('.app-keys .key-panel')", within=10, what="the key put away")
        assert key not in page.js("document.body.textContent")
        assert "appkey-" in page.js("document.querySelector('.app-keys table').textContent")

        page.js("[...document.querySelectorAll('.app-keys button')].find((b) => b.textContent === 'Revoke…').click(); true")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="the revoke confirmation")
        page.js("document.getElementById('confirm-ok').click(); true")
        page.until("document.querySelector('.app-keys table').textContent.includes('revoked')", within=30, what="revoked")
        assert AppKeyStore(database).usable_hashes() == {}
