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
    "providers": ["Fake market"],
    "hosts": ["Provider", "Now", "Prepare a host", "Rented and parked hosts"],
    "engine": ["Engine", "Profiles — what each rented machine holds"],
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
            asked = [url for url in page.js("window.__asked") if "/pool/market/preview" in url]
            # The settings tabs read their settings; opening a tab never searches (D120).
            assert bool(asked) == (tab in ("finding", "scaling", "teardown")), tab
            assert all("search=false" in url for url in asked), f"{tab} searched the market by itself: {asked}"

        # A search happens when asked for — once — and is shown again without asking again.
        page.js("window.__asked = []; location.hash = '#rented/finding'; true")
        page.until("__headings().includes('Live market — the real offer pipeline, read-only')", within=30, what="Finding")
        assert "not searched yet" in page.js("document.getElementById('screen').textContent")
        page.js("[...document.querySelectorAll('button')].find(b => b.textContent === 'Search the market').click(); true")
        page.until("document.getElementById('screen').textContent.includes('searched at')", within=30, what="the search")
        searched = [u for u in page.js("window.__asked") if "/pool/market/preview" in u and "search=false" not in u]
        assert len(searched) == 1
        page.js("window.__asked = []; location.hash = '#rented/scaling'; true")
        page.until("__headings().includes('How capacity is decided')", within=30, what="Scaling")
        page.js("location.hash = '#rented/finding'; true")
        page.until("document.getElementById('screen').textContent.includes('searched at')", within=30, what="the last search")
        assert not [u for u in page.js("window.__asked") if "/pool/market/preview" in u and "search=false" not in u]
        page.js("location.hash = '#rented/scaling'; true")
        page.until("__headings().includes('How capacity is decided')", within=30, what="Scaling again")
        page.js("location.hash = '#rented/teardown'; true")
        page.until("__headings().includes('When a host is given up')", within=30, what="Tear-down")

        # Back goes to the previous tab, not off the screen.
        page.js("history.back(); true")
        page.until("__headings().includes('How capacity is decided')", within=30, what="the previous tab")


@pytest.mark.timeout(180)
def test_the_days_search_quota_is_shown_beside_the_market_and_the_provider(tmp_path):
    """D121: how much of the provider's daily search quota the pool has used, where an operator
    decides whether to search."""
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(POOL_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    provider = supervisor.fleet.provider
    provider.daily_search_rows = 20_000
    pending = {"rows": 0}
    provider.take_search_usage = lambda: {"rows": pending.pop("rows", 0), "refusal": None, "limit": 20_000}
    real = provider.search_offers

    async def search(query):
        pending["rows"] = 600
        return await real(query)

    provider.search_offers = search
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        with open_page(f"{server.base_url}/ui/#rented/finding") as page:
            page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
            page.js(HELPERS)
            page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                    "document.getElementById('key-form').requestSubmit(); true")
            page.until("__headings().includes('Live market — the real offer pipeline, read-only')", within=30, what="Finding")
            assert "search quota: 0 of 20,000 offers used today (0%)" in page.js("document.getElementById('screen').textContent")
            page.js("[...document.querySelectorAll('button')].find(b => b.textContent === 'Search the market').click(); true")
            page.until("document.getElementById('screen').textContent.includes('600 of 20,000 offers used today (3%)')",
                       within=30, what="the count after a search")
            page.js("location.hash = '#rented/hosts'; true")
            page.until("document.getElementById('screen').textContent.includes('600 of 20,000 offers used today')",
                       within=30, what="the provider panel, read fresh")
    finally:
        server.stop()
        loop.stop()
        database.close()


@pytest.mark.timeout(180)
def test_a_page_from_an_older_release_reloads_itself_unless_the_operator_is_busy(served, monkeypatch):
    """D122: a forgotten tab must not keep running an older release's code after a deploy."""
    from gpm_server.supervisor import control

    with open_page(f"{served}/ui/#overview") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.until("typeof state !== 'undefined' && state.loadedRelease != null", within=30, what="the page's release")
        page.js("window.__same_page = true; true")

        # Busy: typing into a field — told, not reloaded.
        page.js("const i = document.createElement('input'); i.id = '__typing'; document.body.append(i); i.focus(); true")
        monkeypatch.setattr(control, "_server_version", lambda: "99.0.0")
        page.until("!!document.getElementById('release-banner')", within=15, what="the banner")
        assert page.js("window.__same_page === true"), "not reloaded under the operator"
        assert "now runs 99.0.0" in page.js("document.getElementById('release-banner').textContent")

        # Not busy: it reloads, and opens again on the key kept for this tab (D137) — no asking.
        page.js("document.getElementById('__typing').remove(); document.activeElement.blur(); true")
        page.until("window.__same_page === undefined", within=15, what="the reload")
        page.until("(document.getElementById('key-dialog') || {}).open === false && !!document.querySelector('#screen h1')",
                   within=30, what="the console again, without the key asked for")
