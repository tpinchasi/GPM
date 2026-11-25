"""The serving plan builder, clicked through in a real browser (S19 stage 7, D204).

The four toggles write the plan; the JSON tab edits it both ways; the code tabs are filled from a
template with the plan as it stands; the plan check's readback is shown under the plan; a
created workload's plan is changed with current and proposed side by side. Skipped where no
browser is installed. Nothing here spends money.
"""

import json

import pytest
from fakes.browser import a_browser, open_page
from test_console_workloads import HELPERS, shot
from test_workloads_end_to_end import ADMIN, pool  # noqa: F401 - the fixture, used by name

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

MORE = r"""
window.__toggle = (label) => { const box = [...document.querySelectorAll("fieldset.toggle legend input")]
  .find((i) => i.parentElement.textContent.trim() === label); box.click(); return true; };
window.__tab = (name) => { [...document.querySelectorAll(".serving-plan [role=tab]")].find((b) => b.textContent === name).click();
  return true; };
window.__json = () => JSON.parse(document.querySelector(".serving-plan textarea").value);
window.__setJson = (text) => { const a = document.querySelector(".serving-plan textarea"); a.value = text;
  a.dispatchEvent(new Event("input")); return true; };
true;
"""


def opened(pool, page, until="__text().includes('Serving plan')"):  # noqa: F811
    page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
    page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN)};"
            "document.getElementById('key-form').requestSubmit(); true")
    page.js(HELPERS)
    page.js(MORE)
    page.until(until, within=30, what="the workloads screen")


@pytest.mark.timeout(240)
def test_a_plan_is_built_with_toggles_edited_as_json_read_back_and_created(pool):  # noqa: F811
    with open_page(f"{pool.control_url}/ui/#workloads") as page:
        opened(pool, page)
        page.js("__set('Name', 'night')")
        page.js("__set('Model', 'big')")
        page.js("__set('Answers at once', '2')")
        page.js("__set('Hours', '2')")
        page.js("__set('Budget ($)', '5')")
        page.js("__toggle('Cover start-up and replacements with…')")
        page.js("__toggle('When every worker is busy, also use…')")
        page.js("__toggle('Keep requests that say nothing off hosted APIs')")
        page.js("__tab('JSON')")
        assert page.js("__json()") == {
            "until_ready": ["shared"],
            "when_full": [{"name": "hosted-burst", "hosted": {"after_s": 5, "budget": {"per_day": 20}}}],
            "defaults": {"keep_on": "no_hosted"},
        }
        shot(page, "07-builder-json")
        # Edited as JSON, read by the builder: one plan, two ways to write it.
        page.js("""__setJson('{"until_ready": ["shared"], "max_min": 30}')""")
        page.js("__tab('Builder')")
        assert page.js("[...document.querySelectorAll('fieldset.toggle legend input')].map((i) => i.checked)") == [
            True, False, False, False]
        page.js("document.querySelector('details.advanced').open = true; true")
        assert page.js("__field('Stop-gap at most (min)').value") == "30"
        page.js("__tab('JSON')")
        page.js("__setJson('{\"until_ready\": [')")
        assert "Not a plan yet" in page.js("__text()")
        assert page.js("[...document.querySelectorAll('button')].find((b) => b.textContent === 'Plan').disabled")
        page.js("""__setJson('{"until_ready": ["shared"], "max_min": 30}')""")
        page.js("__tab('Code')")
        code = page.js("__text()")
        assert 'serve={\n        "until_ready": [\n            "shared",' in code and '"max_min": 30,' in code
        assert "--serve - <<'EOF'" in code and "/pool/workloads" in code
        shot(page, "08-builder-code")

        page.js("__click('Plan')")
        page.until("__text().includes('What it will do')", within=60, what="the readback")
        text = page.js("__text()")
        assert "Until ready — while its machines start or are replaced after an eviction, for at most 30 min" in text
        assert "Most it can spend: $5.00 at most" in text and "Timeline (UTC)" in text
        shot(page, "09-readback")
        page.js("__click('Create — up to')")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="the confirmation")
        page.js("__confirm('')")
        page.until("__text().includes('The key is shown this once')", within=30, what="the key")
    workload = pool.supervisor.workloads.get("night")
    assert workload.policy["serve"]["until_ready"] == ["shared"] and workload.policy["serve"]["max_min"] == 30


@pytest.mark.timeout(240)
def test_a_running_workloads_plan_is_changed_side_by_side(pool):  # noqa: F811
    import httpx

    made = httpx.post(f"{pool.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"}, timeout=60,
                      json={"name": "day", "model": "big", "hours": 2, "latency_s": 30, "parallel": 2, "max_spend": 5,
                            "serve": {"until_ready": ["shared"]}})
    assert made.status_code == 201, made.text
    with open_page(f"{pool.control_url}/ui/#workloads") as page:
        opened(pool, page, until="__text().includes('Active (1)')")
        page.js("__click('Change serving plan…', document.querySelector('table.workloads'))")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="the change")
        body = page.js("document.getElementById('confirm-body').textContent")
        assert "Current" in body and "Proposed" in body and "today's spend still counts" in body
        shot(page, "10-change-plan")
        page.js("""const a = document.querySelector('#confirm-body textarea');
                   a.value = '{"until_ready": ["shared"], "max_min": 20}'; true""")
        page.js("__confirm('')")
        page.until("!document.getElementById('confirm-dialog').open", within=10, what="the dialog closed")
        page.until(f"fetch('/pool/workloads/day', {{headers: {{Authorization: 'Bearer {ADMIN}'}}}})"
                   ".then(r => r.json()).then(j => window.__max = j.workload.policy.serve.max_min) && window.__max === 20",
                   within=20, what="the plan changed")
    assert pool.supervisor.workloads.get("day").policy["serve"]["max_min"] == 20


@pytest.mark.timeout(240)
def test_the_builder_keeps_what_was_typed_and_writes_code_that_runs(pool):  # noqa: F811
    import ast

    with open_page(f"{pool.control_url}/ui/#workloads") as page:
        opened(pool, page)
        page.js("__set('Name', 'night')")
        page.js("__toggle('When every worker is busy, also use…')")
        page.js("__toggle('At certain hours, keep off hosted APIs')")
        page.js("const f = document.getElementById('sp-hours-from'); f.value = '21:00'; f.dispatchEvent(new Event('input'));"
                "const u = document.getElementById('sp-hours-until'); u.value = '07:00'; u.dispatchEvent(new Event('input')); true")
        page.js("__tab('JSON')")
        assert page.js("__json()")["rules"][0]["when"] == {"between": "21:00-07:00"}, "both ends kept"
        # Text that is not a plan yet is kept, with why, across a visit to the builder.
        page.js("__setJson('{\"until_ready\": [')")
        page.js("__tab('Builder')")
        assert "not a plan yet" in page.js("__text()")
        page.js("__tab('JSON')")
        assert page.js("document.querySelector('.serving-plan textarea').value") == '{"until_ready": ['
        # A plan of the wrong shape is drawn as far as it goes, never a broken tab.
        page.js("""__setJson('{"until_ready": "shared", "rules": {}}')""")
        page.js("__tab('Builder')")
        assert "Cover start-up and replacements with…" in page.js("__text()")
        assert page.js("document.activeElement.id") == "sp-tab-builder", "the focus stays on the tab"
        page.js("__tab('JSON')")
        page.js("""__setJson('{"until_ready": ["shared"]}')""")
        page.js("__tab('Code')")
        sdk = page.js("document.querySelector('.code-block pre').textContent")
        ast.parse(sdk)  # no budget typed: still Python that runs
        assert "max_spend=None,  # a program must say what it will spend" in sdk


@pytest.mark.timeout(240)
def test_a_workload_with_only_bursts_is_proposed_its_fields_as_a_plan(pool):  # noqa: F811
    import httpx

    made = httpx.post(f"{pool.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"}, timeout=60,
                      json={"name": "old", "model": "big", "hours": 2, "latency_s": 30, "parallel": 2, "max_spend": 5,
                            "bursts": {"rented": {"max_hosts": 1, "budget": {"per_day": 2}}}})
    assert made.status_code == 201, made.text
    with open_page(f"{pool.control_url}/ui/#workloads") as page:
        opened(pool, page, until="__text().includes('Active (1)')")
        page.js("__click('Change serving plan…', document.querySelector('table.workloads'))")
        page.until("document.getElementById('confirm-dialog').open", within=10, what="the change")
        proposed = json.loads(page.js("document.querySelector('#confirm-body textarea').value"))
        assert proposed["until_ready"] == ["shared"] and proposed["during"] == ["start"]
        assert proposed["when_full"][0]["rented"]["budget"] == {"per_day": 2}, "its burst kept, written as when_full"
        page.js("const a = document.querySelector('#confirm-body textarea'); a.value = '{nope'; true")
        page.js("__confirm('')")
        page.until("document.getElementById('confirm-dialog').open && "
                   "document.getElementById('confirm-body').textContent.includes('Not JSON yet')", within=10,
                   what="asked again")
        assert page.js("document.querySelector('#confirm-body textarea').value") == "{nope", "what was typed is kept"
        page.js("document.getElementById('confirm-dialog').close(); true")
    assert pool.supervisor.workloads.get("old").policy["rented"]["budget"] == {"per_day": 2}, "nothing changed"
