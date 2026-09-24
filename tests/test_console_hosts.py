"""Taking a configured host out of the pool from the Hosts screen, in a real browser (D99).

The API behind these buttons is covered in test_host_removal.py; this clicks them, including
the confirmation that asks for the host's id to be typed before it is removed. Skipped where no
browser is installed.
"""

import json

import pytest
import yaml
from fakes.browser import a_browser, open_page
from test_host_removal import ADMIN_KEY, pool_file, serve

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

HELPERS = r"""
window.__row = (id) => [...document.querySelectorAll("tbody tr")].find((tr) => (tr.querySelector("td.mono") || {}).textContent === id);
window.__button = (id, text) => [...__row(id).querySelectorAll("button")].find((b) => b.textContent === text);
window.__state = (id) => __row(id) ? __row(id).children[4].textContent : null;
window.__confirm = (typed) => {
  const input = document.getElementById("confirm-input");
  if (typed !== undefined) { input.value = typed; input.dispatchEvent(new Event("input")); }
  document.getElementById("confirm-ok").click();
};
true;
"""


@pytest.fixture
def served(tmp_path):
    supervisor, server, loop, database, path = serve(tmp_path, pool_file())
    try:
        yield supervisor, server.base_url, path
    finally:
        server.stop()
        loop.stop()
        database.close()


def hosts_in(path):
    return {h["id"]: h for h in yaml.safe_load(path.read_text())["hosts"]}


@pytest.mark.timeout(240)
def test_the_laptop_is_taken_out_of_service_returned_and_another_host_removed(served):
    supervisor, url, path = served
    with open_page(f"{url}/ui/#hosts") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.until("!!__row('laptop') && !!__button('laptop', 'Take out of service')", within=30, what="the hosts table")

        # Out of service: a plain confirmation, then the file says so and the button turns round.
        page.js("__button('laptop', 'Take out of service').click(); true")
        page.until("document.getElementById('confirm-dialog').open", what="the confirmation")
        page.js("__confirm(); true")
        page.until("!!__row('laptop') && !!__button('laptop', 'Return to service')", within=30, what="the host out of service")
        assert hosts_in(path)["laptop"]["disabled"] is True
        assert "disabled" in page.js("__state('laptop')")

        # Back with one click.
        page.js("__button('laptop', 'Return to service').click(); true")
        page.until("!!__row('laptop') && !!__button('laptop', 'Take out of service')", within=30, what="the host back")
        assert hosts_in(path)["laptop"]["disabled"] is False

        # Removed: the plan is shown, and the host's id must be typed before Confirm will work.
        page.js("__button('desk', 'Remove…').click(); true")
        page.until("document.getElementById('confirm-dialog').open", what="the removal confirmation")
        told = page.js("document.getElementById('confirm-body').textContent")
        assert "drained" in told and "rolled back" in told, told
        assert page.js("document.getElementById('confirm-ok').disabled") is True, "nothing typed yet"
        page.js("document.getElementById('confirm-input').value = 'laptop';"
                "document.getElementById('confirm-input').dispatchEvent(new Event('input')); true")
        assert page.js("document.getElementById('confirm-ok').disabled") is True, "the wrong host's id"
        page.js("__confirm('desk'); true")
        page.until("!__row('desk')", within=30, what="the row to go")

    assert set(hosts_in(path)) == {"laptop"}
    assert [h.id for h in supervisor.config.hosts] == ["laptop"]
