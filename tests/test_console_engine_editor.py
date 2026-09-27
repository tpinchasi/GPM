"""The engine and model-profile editor, clicked through in a real browser (D98, D100, D111).

A real supervisor and control API over a file shaped like a real pool's; the real page served
from them; headless Chrome driven through its DevTools protocol. It is the only test that proves
the screen *works*: the first time the old editor ran, ticking a model to rent for did not redraw
the section, so the field for that model's build never appeared — and every other test passed.

What the owner found wrong with the old one, and this drives: under one model per machine,
several could be ticked; under every model, a few could not be chosen; only models already in
the file could be picked; and the card and disk a machine needed were typed by hand.

Builds and searches are answered by a stand-in hub serving a search recorded from the real one,
so nothing reaches the internet. Skipped where no browser is installed; the API behind it is
covered in test_engine_editor.py and test_model_profiles.py.
"""

import json
from pathlib import Path

import pytest
import yaml
from fakes.browser import a_browser, open_page
from fakes.fake_vllm import FakeHub
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.config import load_config
from gpm_server.db import Database
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app
from test_engine_editor import ADMIN_KEY, BIG, BIG_REPO, EMBED, POOL_YAML, SMALL

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

RECORDED = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "directory" / "hub_search_gemma-4-26b.json").read_text()
)
ORIGINAL, FP8 = "google/gemma-4-26B-A4B-it", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"

# Small helpers installed in the page: find the editor and describe it the way a person sees it.
HELPERS = r"""
window.__editor = () => {
  const h = [...document.querySelectorAll("h2")].find((e) => e.textContent.startsWith("Engine and models"));
  return h ? h.nextElementSibling : null;
};
window.__click = (text, scope) => {
  const b = [...(scope || __editor()).querySelectorAll("button")].find((b) => b.textContent === text);
  if (!b) throw new Error("no button " + text);
  b.click(); return true;
};
window.__profiles = () => [...__editor().querySelectorAll(".profile")].map((p) => ({
  text: p.textContent, rented: p.querySelector('input[type="checkbox"]').checked, editing: p.classList.contains("editing"),
}));
window.__editing = () => __editor().querySelector(".profile.editing");
window.__held = () => [...__editing().querySelectorAll(".profile-editor > table > tbody > tr")]
  .map((tr) => [...tr.querySelectorAll("td")].slice(0, 2).map((td) => td.textContent));
window.__inPool = (model) => __click(model, __editing());
window.__radios = () => [...__editing().querySelectorAll('input[type="radio"][name^="profile-build-"]')]
  .map((r) => ({ repo: r.value, disabled: r.disabled }));
window.__pick = (repo) => {
  const radio = [...__editing().querySelectorAll('input[type="radio"]')].find((r) => r.value === repo);
  radio.checked = true; radio.dispatchEvent(new Event("change")); return true;
};
window.__shape = (label) => {
  const r = [...__editing().querySelectorAll('input[name="profile-shape"]')].find((r) => r.parentElement.textContent === label);
  r.checked = true; r.dispatchEvent(new Event("change")); return true;
};
window.__name = (value) => {
  const input = __editing().querySelector('.profile-editor input[type="text"]');
  input.value = value; input.dispatchEvent(new Event("input")); return true;
};
window.__search = (term) => {
  const input = __editing().querySelector('input[type="search"]');
  input.value = term; input.dispatchEvent(new Event("input")); return __click("Search", __editing());
};
window.__hubRows = () => {
  const h = [...__editing().querySelectorAll("h4")].find((h) => h.textContent.startsWith("On the model hub"));
  const t = h && h.nextElementSibling;
  return t && t.tagName === "TABLE" ? [...t.querySelectorAll(":scope > tbody > tr")].map((tr) => tr.querySelector("td .mono").textContent) : [];
};
window.__filter = (placeholder, value) => {
  const input = [...__editing().querySelectorAll(".filters input")].find((i) => i.placeholder === placeholder);
  input.value = value; input.dispatchEvent(new Event("input")); return true;
};
window.__chooseHub = (repo) => {
  const h = [...__editing().querySelectorAll("h4")].find((h) => h.textContent.startsWith("On the model hub"));
  const row = [...h.nextElementSibling.querySelectorAll(":scope > tbody > tr")].find((tr) => tr.querySelector("td .mono").textContent === repo);
  row.querySelector("button").click(); return true;
};
window.__engine = (value) => {
  const s = __editor().querySelector("select"); s.value = value; s.dispatchEvent(new Event("change")); return true;
};
window.__note = () => ([...__editor().querySelectorAll("button")].find((b) => b.textContent === "Save to configuration")
  .parentElement.querySelector(":scope > span.muted") || {}).textContent || "";
// Count answered saves, so each wait is for its own save and not a message left by the last.
if (!window.__counting) {
  window.__counting = true; window.__saves = 0;
  const real = window.fetch;
  window.fetch = async (...args) => {
    try { return await real(...args); } finally { if ((args[1] || {}).method === "PATCH") window.__saves += 1; }
  };
}
true;
"""


@pytest.fixture
def served(tmp_path, monkeypatch):
    config_path = tmp_path / "pool.yaml"
    original = POOL_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n"
    config_path.write_text(original)
    loop = BackgroundLoop()
    hub = ServerHandle(FakeHub({}, search={"gemma-4-26b": RECORDED, "gemma-4-26B-A4B-it": RECORDED}).app, loop)
    monkeypatch.setenv("HF_ENDPOINT", hub.base_url)
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        yield supervisor, server.base_url, config_path, original
    finally:
        server.stop()
        hub.stop()
        loop.stop()
        database.close()


@pytest.mark.timeout(240)
def test_a_profile_is_made_from_a_search_and_saved_with_the_engine(served):
    supervisor, url, config_path, original = served
    with open_page(f"{url}/ui/#rented/engine") as page:
        # The page opens its key dialog as its very last statement: open means it has run to
        # the end and attached its handlers.
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.until("!!__editor()", within=30, what="the engine editor")

        # A pool with no profiles opens on what it rents today, written as one: every model.
        (only,) = page.js("__profiles()")
        assert only["rented"] and "3 models on one machine" in only["text"]

        # Under vLLM those Ollama builds are no build at all, and the profile says so.
        page.js("__engine('vllm')")
        assert "no vllm build chosen" in page.js("__profiles()")[0]["text"]
        page.js("__click('Remove', __editor().querySelector('.profile'))")
        page.js("__click('New profile')")
        page.js("__name('chat')")

        # One of the pool's own models, its build picked from the hub's answer, sorted: the
        # original first, files for other engines left out (D100).
        page.js(f"__inPool({json.dumps(BIG)})")
        page.until("__radios().length > 0", within=30, what="the builds on the hub")
        offered = [r["repo"] for r in page.js("__radios()")]
        assert offered[0] == ORIGINAL and BIG_REPO in offered
        assert not any("GGUF" in r or "MLX" in r for r in offered)
        page.js(f"__pick({json.dumps(BIG_REPO)})")
        assert page.js("__held()") == [[BIG, BIG_REPO]]
        assert "needs a card of ≥" in page.js("__editing().textContent")

        # Any model on the hub, found by searching — originals only, and narrowed by the filters.
        page.js("__search('gemma-4-26b')")
        page.until("__hubRows().length > 0", within=30, what="the search's answer")
        found = page.js("__hubRows()")
        assert ORIGINAL in found
        assert BIG_REPO not in found, "a quantisation is a build, offered once its model is chosen"
        assert not any("GGUF" in r or "MLX" in r for r in found)
        page.js("__filter('max B', '10')")
        small = page.js("__hubRows()")
        assert ORIGINAL not in small, "a 26B model is not under 10B"
        assert sorted(small) == ["google/gemma-4-26B-A4B-it-assistant", "z-lab/gemma-4-26B-A4B-it-DFlash"]
        page.js("__filter('max B', '')")

        # One model per machine holds one: choosing another replaces it (the old screen let
        # several be ticked here).
        page.js(f"__chooseHub({json.dumps(ORIGINAL)})")
        page.until("__radios().length > 0", within=30, what="the chosen model's builds")
        page.js(f"__pick({json.dumps(FP8)})")
        assert page.js("__held()") == [["gemma-4-26b-a4b-it", FP8]]

        # Several models is a choice of a few, not the whole set.
        page.js("__shape('several models')")
        page.js(f"__inPool({json.dumps(BIG)})")
        page.until("__radios().length > 0", within=30, what="the builds on the hub")
        page.js(f"__pick({json.dumps(BIG_REPO)})")
        assert page.js("__held()") == [["gemma-4-26b-a4b-it", FP8], [BIG, BIG_REPO]]
        page.js("__shape('one model')")
        assert page.js("__held()") == [["gemma-4-26b-a4b-it", FP8]], "one model keeps the first"

        # A profile holding nothing is refused in words, and nothing is written.
        page.js("__click('Done'); __click('New profile'); __click('Save to configuration')")
        page.until("__note().includes('not saved')", within=10, what="the refusal")
        assert "holds no model" in page.js("__note()")
        assert config_path.read_text() == original
        page.js("__click('Remove', __editor().querySelectorAll('.profile')[1])")

        page.js("__click('Save to configuration')")
        page.until("window.__saves >= 1", within=30, what="the save's answer")
        page.until("!!__editor() && __note().includes(' saved ·')", within=30, what="the redraw")

    written = yaml.safe_load(config_path.read_text())
    new = "gemma-4-26b-a4b-it"
    assert written["rented"]["engine"] == "vllm"
    assert written["rented"]["model_profiles"] == {"chat": {new: FP8}}
    assert written["rented"]["rent_profiles"] == ["chat"]
    assert written["pool"]["model_set"] == [EMBED, SMALL, BIG, new]
    assert written["catalog"][new]["variants"][0]["tag"] == FP8
    # The laptop runs Ollama and cannot hold a model built only for vLLM: the set is spread, and
    # it keeps exactly what it held.
    assert written["pool"]["models_per_host"] == "declared"
    assert written["hosts"][0]["models"] == [EMBED, SMALL, BIG]
    assert supervisor.config.rented.rent_profiles == ["chat"]
