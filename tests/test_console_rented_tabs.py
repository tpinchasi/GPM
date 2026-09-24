"""Rented capacity, in tabs (owner: "very hard to navigate and understand what belongs where").

One tab per question an operator comes with — what is rented now, what it runs, how machines are
found, how many and how much, when they are given up — each at its own address. Clicked through
in a real browser against a real supervisor over the fake provider. Two things are held: every
section is on exactly one tab, and only the tabs that need the market ask the provider for it —
asking is a real call, and the provider rate-limits (D44).
"""

from __future__ import annotations

import json

import pytest
from fakes.browser import a_browser, open_page
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.db import Database
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app
from test_engine_editor import ADMIN_KEY, POOL_YAML

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

#: Each tab, and the section headings that belong on it — and only on it.
TABS = {
    "hosts": ["Provider", "Now", "Prepare a host", "Rented and parked hosts"],
    "engine": ["Engine", "Engine and placement on rented hosts"],
    "finding": ["What the pool looks for", "Live market — the real offer pipeline, read-only"],
    "scaling": ["Limits", "How capacity is decided"],
    "teardown": ["When a host is given up"],
}

HELPERS = r"""
if (!window.__asked) {
  window.__asked = [];
  const real = window.fetch;
  window.fetch = (...args) => { window.__asked.push(String(args[0])); return real(...args); };
}
window.__headings = () => [...document.querySelectorAll("#screen h2")].map((h) => h.firstChild ? h.firstChild.textContent.trim() : h.textContent.trim());
window.__tabs = () => [...document.querySelectorAll(".tabs a")].map((a) => ({ text: a.textContent, active: a.classList.contains("active") }));
true;
"""


@pytest.fixture
def served(tmp_path):
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(POOL_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        yield server.base_url
    finally:
        server.stop()
        loop.stop()
        database.close()


@pytest.mark.timeout(240)
def test_each_section_is_on_one_tab_and_only_the_market_tabs_ask_the_market(served):
    with open_page(f"{served}/ui/#rented") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(HELPERS)
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.until("__headings().includes('Rented and parked hosts')", within=30, what="the Hosts tab")

        # With no tab named, it opens on Hosts — what is being paid for now.
        assert [t["text"] for t in page.js("__tabs()") if t["active"]] == ["Hosts"]

        everywhere = {heading for headings in TABS.values() for heading in headings}
        for tab, expected in TABS.items():
            page.js(f"window.__asked = []; location.hash = '#rented/{tab}'; true")
            page.until(f"__headings().includes({json.dumps(expected[-1])})", within=30, what=f"the {tab} tab")
            shown = set(page.js("__headings()"))
            assert set(expected) <= shown, (tab, shown)
            assert not (shown & (everywhere - set(expected))), f"{tab} shows another tab's sections"
            asked_market = any("/pool/market/preview" in url for url in page.js("window.__asked"))
            assert asked_market == (tab in ("finding", "scaling", "teardown")), tab

        # Back goes to the previous tab, not off the screen.
        page.js("history.back(); true")
        page.until("__headings().includes('How capacity is decided')", within=30, what="the previous tab")
