"""The Workloads screen, clicked through in a real browser (D115).

The real supervisor, router and control API; the page served from them; headless Chrome driven
through its DevTools protocol. An operator plans a workload, creates it by typing the proposed
budget again, reads its key once, and ends it by typing its name. Skipped where no browser is
installed; the API behind it is covered in test_workloads_end_to_end.py.
"""

import base64
import json
import os
from pathlib import Path

import pytest
from fakes.browser import a_browser, open_page
from test_workloads_end_to_end import ADMIN, pool  # noqa: F401 - the fixture, used by name

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

HELPERS = r"""
window.__field = (label) => [...document.querySelectorAll(".field")].find((f) => f.firstElementChild.textContent === label)
  .querySelector("input, select");
window.__set = (label, value) => { const f = __field(label); f.value = value;
  f.dispatchEvent(new Event(f.tagName === "SELECT" ? "change" : "input")); return true; };
window.__click = (text, scope) => { const b = [...(scope || document).querySelectorAll("button")].find((b) => b.textContent.startsWith(text));
  if (!b) throw new Error("no button " + text); b.click(); return true; };
window.__text = () => document.getElementById("screen").textContent;
window.__confirm = (typed) => { const input = document.getElementById("confirm-input"); input.value = typed;
  input.dispatchEvent(new Event("input")); document.getElementById("confirm-ok").click(); return true; };
true;
"""


def shot(page, name):
    """A screenshot for the UX review, where one is asked for (GPM_SCREENSHOTS=<dir>)."""
    where = os.environ.get("GPM_SCREENSHOTS")
    if not where:
        return
    page.call("Emulation.setDeviceMetricsOverride", width=1280, height=1100, deviceScaleFactor=1, mobile=False)
    data = page.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)["data"]
    Path(where).mkdir(parents=True, exist_ok=True)
    (Path(where) / f"{name}.png").write_bytes(base64.b64decode(data))


@pytest.mark.timeout(180)
def test_a_workload_is_planned_created_and_ended_from_the_console(pool):  # noqa: F811
    with open_page(f"{pool.control_url}/ui/#workloads") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.until("__text().includes('New workload')", within=30, what="the workloads screen")
        assert "None. Apps with the pool's app key are served by the shared hosts." in page.js("__text()")
        shot(page, "01-empty")

        page.js("__set('Name', 'research')")
        page.js("__set('Model', 'big')")
        page.js("__set('Answers at once', '2')")
        page.js("__set('Hours', '2')")
        assert not page.js("[...document.querySelectorAll('button')].some(b => b.textContent.startsWith('Create'))"), \
            "nothing can be created before it is planned"
        page.js("__click('Plan')")
        page.until("__text().includes('starts on')", within=30, what="the plan")
        text = page.js("__text()")
        assert "1 host, 2 answers at once each" in text and "not measured on this card yet" in text
        assert "cost per hour" in text and "plus 25%" in text and "503 workload_ended" in text
        shot(page, "02-planned")
        assert pool.supervisor.leases.open_leases() == [], "a plan opens nothing"

        page.js("__click('Create — up to')")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="the confirmation")
        budget = pool.loop.run(pool.supervisor.workloads.plan(
            __import__("gpm_server.supervisor.workloads", fromlist=["WorkloadRequest"]).WorkloadRequest(
                name="research", model="big", latency_s=30, parallel=2, hours=2)))["max_spend"]
        assert page.js("document.getElementById('confirm-ok').disabled"), "not before the budget is typed again"
        assert page.js("document.getElementById('confirm-retype-label').textContent") == f"Type {budget:.2f} to accept this budget"
        shot(page, "03-confirm")
        page.js(f"__confirm({json.dumps(f'${budget:.2f}')})")  # typed with its dollar sign: the same budget
        page.until("__text().includes('The key is shown this once')", within=30, what="the key")
        key = page.js("document.querySelector('.key-panel code.secret').textContent")
        assert key.startswith("gpmw_")
        assert "model" in page.js("document.querySelector('.key-panel').textContent"), "the app owner needs the model too"
        shot(page, "04-key")
        (lease,) = pool.supervisor.leases.open_leases()
        assert lease.workload == "research"

        page.js("__click('I have saved it', document.querySelector('.key-panel'))")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="are you sure")
        page.js("__confirm('')")
        page.until("!__text().includes('The key is shown this once')", within=10, what="the key put away")
        assert key not in page.js("document.body.innerHTML"), "gone from the page once put away"
        page.until("__text().includes('Active (1)')", within=10, what="the active list")
        page.until("!__text().includes('A model at a latency')", within=10, what="the form folded away")
        assert page.js("[...document.querySelectorAll('button')].some(b => b.textContent === 'New workload…')")
        page.js("document.querySelector('table.workloads button.link').click()")
        page.until("__text().includes('answers so far')", within=10, what="its detail, opened in place")
        assert page.js("document.querySelector('table.workloads button.link').getAttribute('aria-expanded')") == "true"
        assert "null" not in page.js("__text()")
        shot(page, "05-detail")

        page.js("__click('End…', document.querySelector('table.workloads'))")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="the confirmation")
        page.js("__confirm('research')")
        page.until("__text().includes('ending') || __text().includes('Active (0)')", within=20, what="it ending")
        shot(page, "06-ending")
    assert pool.supervisor.workloads.get("research").state in ("ending", "ended")


@pytest.mark.timeout(120)
def test_escape_never_confirms_even_after_a_dialog_that_was_confirmed(pool):  # noqa: F811
    """A dialog keeps the answer it last closed with; Escape must not reuse it."""
    with open_page(f"{pool.control_url}/ui/#workloads") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.until("typeof confirmAction === 'function' && !document.getElementById('key-dialog').open", within=30)
        page.js("window.__first = confirmAction({ title: 'one', body: 'x' }); document.getElementById('confirm-ok').click(); true")
        assert page.js("window.__first") is True
        page.js("window.__second = confirmAction({ title: 'two', body: 'y', retype: 'research' });"
                "document.getElementById('confirm-dialog').close(); true")  # what Escape does
        assert page.js("window.__second") is False


@pytest.mark.timeout(180)
def test_a_provisioning_key_is_made_and_revoked_from_the_console():
    """With programs allowed (D117): a key made with its grant, shown once, then revoked."""
    from fakes.harness import ServerHandle
    from gpm_server.supervisor.control import create_control_app
    from test_provisioning import ADMIN as PROVISIONING_ADMIN
    from test_provisioning import harness as provisioning_harness

    with provisioning_harness() as h:
        control = ServerHandle(create_control_app(h.supervisor, h.config), h.loop)
        try:
            with open_page(f"{control.base_url}/ui/#workloads") as page:
                page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
                page.js(f"document.getElementById('key-input').value = {json.dumps(PROVISIONING_ADMIN)};"
                        "document.getElementById('key-form').requestSubmit(); true")
                page.js(HELPERS)
                page.until("__text().includes('None: only operators create workloads')", within=30, what="the programs section")
                page.js("__click('New provisioning key…')")
                page.until("__text().includes('Lets one application create')", within=10, what="the form")
                page.js("__set('Application', 'evals')")
                create_disabled = "[...document.querySelectorAll('.programs button')].find(b => b.textContent === 'Create key').disabled"
                assert page.js(create_disabled), "not before a model is picked"
                page.js("document.querySelector('.programs input[type=checkbox][value=big]').click(); true")
                assert not page.js(create_disabled)
                page.js("__click('Create key')")
                page.until("document.getElementById('confirm-dialog').open", within=10)
                page.js("__confirm('20.00')")
                page.until("__text().includes('The key is shown this once')", within=20, what="the key")
                key = page.js("document.querySelector('.programs code.secret').textContent")
                assert key.startswith("gpmp_")
                assert "GPM_URL" in page.js("document.querySelector('.programs .key-panel').textContent")
                assert page.js("document.activeElement.textContent").startswith("Provisioning key for evals")
                shot(page, "07-programs")
                assert h.supervisor.provisioning.store.get("evals").grant.models == ("big",)
                page.js("__click('I have saved it', document.querySelector('.programs'))")
                page.until("document.getElementById('confirm-dialog').open", within=10)
                page.js("__confirm('')")
                page.until("!__text().includes('The key is shown this once')", within=10)
                page.js("__click('Revoke', document.querySelector('.programs'))")
                page.until("document.getElementById('confirm-dialog').open", within=10)
                page.js("__confirm('')")
                page.until("__text().includes('revoked')", within=10, what="it revoked")
            assert not h.supervisor.provisioning.store.get("evals").usable()
        finally:
            control.stop()
