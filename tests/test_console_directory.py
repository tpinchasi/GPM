"""The model directory, clicked through in a real browser (D101).

Refresh it, find a model, tick one of its sizes, look its builds up on the hub, pick one, switch
tool calling on, add it — and the pool's file has it, with a build for each engine. Ollama's site
and the hub are stand-ins serving what the real ones served, recorded: nothing reaches the
internet, nothing rents.

Skipped where no browser is installed; the API behind it is covered in test_model_directory.py.
"""

from __future__ import annotations

import json

import pytest
import yaml
from fakes.browser import a_browser, open_page
from test_model_directory import ADMIN_KEY, NVFP4, RECORDED, Pool

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

HELPERS = r"""
window.__dir = () => {
  const h = [...document.querySelectorAll("h2")].find((e) => e.textContent === "Model directory");
  return h ? h.nextElementSibling : null;
};
window.__button = (text) => [...__dir().querySelectorAll("button")].find((b) => b.textContent === text);
window.__models = () => [...__dir().querySelectorAll(".dir-head strong")].map((s) => s.textContent.slice(2));
window.__filter = (text) => {
  const box = __dir().querySelector('input[type="search"]'); box.value = text; box.dispatchEvent(new Event("input"));
};
window.__open = (name) => [...__dir().querySelectorAll(".dir-head")].find((h) => h.querySelector("strong").textContent.slice(2) === name).click();
window.__tickTag = (name) => {
  const label = [...__dir().querySelectorAll("label.pick")].find((l) => l.textContent === name);
  const box = label.querySelector("input"); box.checked = true; box.dispatchEvent(new Event("change"));
};
window.__radios = (name) => [...__dir().querySelectorAll(`input[type="radio"][name="dir-${name}"]`)].map((r) => r.value);
window.__pick = (name, repo) => {
  const r = [...__dir().querySelectorAll(`input[type="radio"][name="dir-${name}"]`)].find((r) => r.value === repo);
  r.checked = true; r.dispatchEvent(new Event("change"));
};
window.__option = (text) => {
  const box = [...__dir().querySelectorAll("label.option")].find((l) => l.textContent.includes(text)).querySelector("input");
  box.checked = true; box.dispatchEvent(new Event("change"));
};
true;
"""

ADDED = "gemma4:31b"


@pytest.fixture
def pool(tmp_path, monkeypatch):
    made = Pool(tmp_path, monkeypatch)
    # The 31b is looked up under its own spellings; the stand-in hub answers with the recording.
    made.hub.search["gemma-4-31b"] = RECORDED
    try:
        yield made
    finally:
        made.close()


@pytest.mark.timeout(240)
def test_adding_a_model_from_the_directory(pool):
    with open_page(f"{pool.control.base_url}/ui/#models") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.until("!!__dir() && !!__button('Refresh now')", within=30, what="the directory")

        # Empty until it is refreshed: the pool asks nobody anything on its own.
        assert page.js("__models()") == []
        page.js("__button('Refresh now').click(); true")
        page.until("__models().length === 2", within=60, what="the refreshed directory")
        assert set(page.js("__models()")) == {"gemma4", "nomic-embed-text"}

        page.js("__filter('gemma'); true")
        assert page.js("__models()") == ["gemma4"]
        page.js("__open('gemma4'); true")
        page.until(f"[...__dir().querySelectorAll('label.pick')].some((l) => l.textContent === {json.dumps(ADDED)})",
                   what="the sizes of gemma4")

        page.js(f"__tickTag({json.dumps(ADDED)}); true")
        page.until("!!__button('Look up on the hub')", what="the lookup button")
        page.js("__button('Look up on the hub').click(); true")
        page.until(f"__radios({json.dumps(ADDED)}).length > 2", within=30, what="the builds on the hub")
        offered = page.js(f"__radios({json.dumps(ADDED)})")
        assert offered[0] == "", "choosing no vLLM build is a choice too"
        assert NVFP4 in offered

        page.js(f"__pick({json.dumps(ADDED)}, {json.dumps(NVFP4)}); __option('tool calling'); true")
        page.js("__button('Add to the pool').click(); true")
        page.until("!!__dir() && [...__dir().querySelectorAll('.pill')].some((p) => p.textContent === 'in the pool')"
                   " || document.body.textContent.includes('not added')", within=30, what="the answer")
        assert "not added" not in page.js("document.body.textContent")

    written = yaml.safe_load(pool.path.read_text())
    assert ADDED in written["pool"]["model_set"]
    assert written["catalog"][ADDED]["variants"] == [
        {"tag": ADDED, "engine": "ollama"}, {"tag": NVFP4, "engine": "vllm"},
    ]
    assert ADDED in written["rented"]["models"]
    assert written["rented"]["engine_options"] == ["tool_calling"]
