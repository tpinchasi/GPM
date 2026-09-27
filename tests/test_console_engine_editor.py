"""The Profiles tab, clicked through in a real browser (D98, D100, D111).

A real supervisor and control API over a file shaped like a real pool's; the real page served
from them; headless Chrome driven through its DevTools protocol. It is the only test that proves
the screen *works*: the first time the old editor ran, ticking a model to rent for did not redraw
the section, so the field for that model's build never appeared — and every other test passed.

What the owner found wrong, and this drives: a model and its variant were chosen in two steps on
two different screens, the search was hidden, and only the file's own models showed. Here every
row offered — the pool's own, or the hub's — is a model and one variant of it, added in one
click; one model per machine holds one; several is a chosen few.

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
from test_engine_editor import ADMIN_KEY, BIG, EMBED, POOL_YAML, SMALL

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

RECORDED = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "directory" / "hub_search_gemma-4-26b.json").read_text()
)
ORIGINAL, FP8 = "google/gemma-4-26B-A4B-it", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"

# Small helpers installed in the page: find the tab's parts and describe them as a person sees them.
HELPERS = r"""
window.__editor = () => {
  const h = [...document.querySelectorAll("h2")].find((e) => e.textContent.startsWith("Profiles"));
  return h ? h.nextElementSibling : null;
};
window.__click = (text, scope) => {
  const b = [...(scope || document).querySelectorAll("button")].find((b) => b.textContent === text);
  if (!b) throw new Error("no button " + text);
  b.click(); return true;
};
window.__cards = () => [...__editor().querySelectorAll(".profile")].map((p) => ({
  text: p.innerText, rented: p.querySelector('input[type="checkbox"]').checked, finding: p.classList.contains("editing"),
}));
window.__editing = () => __editor().querySelector(".profile.editing");
window.__held = (card) => [...(card || __editing()).querySelectorAll("table.held > tbody > tr")]
  .map((tr) => [...tr.querySelectorAll("td")].slice(0, 2).map((td) => td.textContent));
window.__name = (value) => {
  const input = __editing().querySelector("input.profile-name");
  input.value = value; input.dispatchEvent(new Event("input")); return true;
};
window.__shape = (label) => {
  const r = [...__editing().querySelectorAll('input[type="radio"]')].find((r) => r.parentElement.textContent === label);
  r.checked = true; r.dispatchEvent(new Event("change")); return true;
};
window.__search = (term) => {
  const input = __editing().querySelector("input.finder-search");
  input.value = term; input.dispatchEvent(new Event("input")); return __click("Search", __editing());
};
window.__section = (title) => {
  const h = [...__editing().querySelectorAll("h4")].find((h) => h.textContent.startsWith(title));
  return h;
};
window.__poolRows = () => {
  const t = __section("In the pool").nextElementSibling;
  return t.tagName === "TABLE" ? [...t.querySelectorAll(":scope > tbody > tr")]
    .map((tr) => [...tr.querySelectorAll("td")].slice(0, 2).map((td) => td.textContent)) : [];
};
window.__hubRows = () => [...__editing().querySelectorAll(".hub-group")].flatMap((g) =>
  [...g.querySelectorAll(":scope > table > tbody > tr")].map((tr) => ({
    model: g.querySelector("strong").textContent, variant: tr.querySelector("td").firstChild.textContent,
    precision: tr.querySelectorAll("td")[1].textContent })));
window.__useHub = (model, variant) => {
  const g = [...__editing().querySelectorAll(".hub-group")].find((g) => g.querySelector("strong").textContent === model);
  const tr = [...g.querySelectorAll(":scope > table > tbody > tr")].find((tr) => tr.querySelector("td").firstChild.textContent === variant);
  const buttons = tr.querySelectorAll("button"); buttons[buttons.length - 1].click(); return true;
};
window.__precision = (value) => {
  const s = __editing().querySelector(".filters select"); s.value = value; s.dispatchEvent(new Event("change")); return true;
};
window.__engine = (value) => {
  const s = __editor().querySelector("select"); s.value = value; s.dispatchEvent(new Event("change")); return true;
};
window.__note = () => ([...document.querySelectorAll("button")].find((b) => b.textContent === "Save to configuration")
  .parentElement.querySelector(":scope > span.muted") || {}).textContent || "";
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
    hub = ServerHandle(FakeHub({}, search={"gemma-4-26b": RECORDED}).app, loop)
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


NVFP4 = "nvidia/Gemma-4-26B-A4B-NVFP4"
ASSISTANT = "google/gemma-4-26B-A4B-it-assistant"


@pytest.mark.timeout(240)
def test_a_profile_is_made_from_model_and_variant_rows_in_one_step(served):
    supervisor, url, config_path, original = served
    with open_page(f"{url}/ui/#rented/engine") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.until("!!__editor()", within=30, what="the Profiles tab")

        # A pool with no profiles opens on what it rents today, written as one, each row a model
        # and the variant a CUDA machine would fetch — never the Apple-silicon build.
        (only,) = page.js("__cards()")
        assert only["rented"]
        held = page.js("__held(__editor().querySelector('.profile'))")
        assert held == [[EMBED, EMBED], [SMALL, SMALL], [BIG, BIG]]

        # Under vLLM those are no variant at all, and every row says so.
        page.js("__engine('vllm')")
        assert page.js("__cards()")[0]["text"].count("no vllm variant") == 3
        page.js("__click('Remove profile', __editor().querySelector('.profile'))")
        page.js("__click('+ New profile')")
        assert page.js("!!__editing().querySelector('input.finder-search')"), "a new profile opens on the search"
        page.js("__name('chat')")

        # The hub's answer is models, each with its variants side by side — one row each.
        page.js("__search('gemma-4-26b')")
        page.until("__hubRows().length > 0", within=30, what="the search's answer")
        rows = page.js("__hubRows()")
        mine = [r["variant"] for r in rows if r["model"] == ORIGINAL]
        assert mine[0] == ORIGINAL and NVFP4 in mine and FP8 in mine
        assert not any("GGUF" in r["variant"] or "MLX" in r["variant"] for r in rows)

        # Filters narrow the rows themselves: FP8 leaves only FP8 variants.
        page.js("__precision('FP8')")
        assert {r["precision"] for r in page.js("__hubRows()")} == {"FP8"}
        page.js("__precision('')")

        # One click puts a model and its variant in the profile. One model per machine holds one:
        # choosing another replaces it — the old screen let several be ticked here.
        page.js(f"__useHub({json.dumps(ORIGINAL)}, {json.dumps(FP8)})")
        assert page.js("__held()") == [["gemma-4-26b-a4b-it", FP8]]
        page.js(f"__useHub({json.dumps(ORIGINAL)}, {json.dumps(NVFP4)})")
        assert page.js("__held()") == [["gemma-4-26b-a4b-it", NVFP4]]

        # What was found is now in the pool, offered as model-and-variant rows with no search.
        assert ["gemma-4-26b-a4b-it", FP8] in page.js("__poolRows()")

        # Several models is a chosen few.
        page.js("__shape('several models')")
        page.js(f"__useHub({json.dumps(ASSISTANT)}, {json.dumps(ASSISTANT)})")
        assert page.js("__held()") == [["gemma-4-26b-a4b-it", NVFP4], ["gemma-4-26b-a4b-it-assistant", ASSISTANT]]
        page.js("__shape('one model')")
        assert page.js("__held()") == [["gemma-4-26b-a4b-it", NVFP4]], "one model keeps the first"

        # A profile holding nothing is refused in words, and nothing is written.
        page.js("__click('Close', __editing()); __click('+ New profile'); __click('Save to configuration')")
        page.until("__note().includes('not saved')", within=10, what="the refusal")
        assert "holds no model" in page.js("__note()")
        assert config_path.read_text() == original
        page.js("__click('Remove profile', __editor().querySelectorAll('.profile')[1])")

        page.js("__click('Save to configuration')")
        page.until("window.__saves >= 1", within=30, what="the save's answer")
        page.until("!!__editor() && __note().includes(' saved ·')", within=30, what="the redraw")

    written = yaml.safe_load(config_path.read_text())
    new = "gemma-4-26b-a4b-it"
    assert written["rented"]["engine"] == "vllm"
    assert written["rented"]["model_profiles"] == {"chat": {new: NVFP4}}
    assert written["rented"]["rent_profiles"] == ["chat"]
    assert written["pool"]["model_set"] == [EMBED, SMALL, BIG, new]
    # Only the variant a profile uses is written: the FP8 one tried and replaced is not.
    assert [v["tag"] for v in written["catalog"][new]["variants"]] == [NVFP4]
    # The laptop runs Ollama and cannot hold a model built only for vLLM: the set is spread, and
    # it keeps exactly what it held.
    assert written["pool"]["models_per_host"] == "declared"
    assert written["hosts"][0]["models"] == [EMBED, SMALL, BIG]
    assert supervisor.config.rented.rent_profiles == ["chat"]
