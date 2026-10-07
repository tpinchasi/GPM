"""The console keeps the operator's place while it updates (D137, found by the owner).

The overview redraws on every decision and every status update; scrolled into the live feed,
the operator was sent back to its top each time.
"""

import json

import pytest
from fakes.browser import a_browser, open_page
from test_provider_credentials import ADMIN_KEY, pool  # noqa: F401 - the fixture

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")


@pytest.mark.timeout(180)
def test_the_live_feed_stays_where_it_was_scrolled_as_decisions_arrive(pool):  # noqa: F811
    supervisor, market, base, path, database = pool
    for n in range(45):
        supervisor.events.record("test_event", f"an earlier decision, number {n}, with enough words to take a line")
    with open_page(f"{base}/ui/#overview") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.until("!!document.querySelector('.feed .ev')", within=30, what="the live feed")
        page.js("const f = document.querySelector('.feed'); f.scrollTop = 300; window.__top = f.scrollTop; true")
        assert page.js("window.__top") > 100, "the feed scrolls"
        page.js("window.__redraws = 0; new MutationObserver(() => window.__redraws++)"
                ".observe(document.getElementById('screen'), {childList: true}); true")
        supervisor.events.record("test_event", "a decision arriving while the operator reads")
        page.until("window.__redraws >= 1", within=20, what="the overview redrawn by the new decision")
        assert abs(page.js("document.querySelector('.feed').scrollTop") - page.js("window.__top")) < 5
