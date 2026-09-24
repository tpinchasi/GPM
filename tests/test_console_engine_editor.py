"""The engine and placement editor, clicked through in a real browser (D98, D100).

A real supervisor and control API over a file shaped like a real pool's; the real page served
from them; headless Chrome driven through its DevTools protocol. It is the only test that proves
the screen *works*: the first time it ran, ticking a model to rent for did not redraw the
section, so the field for that model's build never appeared — and every other test passed.

Its vLLM builds are picked from the hub's answer, sorted — here a stand-in hub serving a search
recorded from the real one, so nothing reaches the internet (D100).

Skipped where no browser is installed; the API behind it is covered in test_engine_editor.py.
"""

import json
import time
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

# Small helpers installed in the page: find the editor and describe it the way a person sees it.
HELPERS = r"""
window.__editor = () => {
  const h = [...document.querySelectorAll("h2")].find((e) => e.textContent.startsWith("Engine and placement"));
  return h ? h.nextElementSibling : null;
};
window.__saveRow = () => [...__editor().querySelectorAll("button")].find((b) => b.textContent === "Save to configuration").parentElement;
window.__describe = () => {
  const box = __editor();
  const selects = [...box.querySelectorAll("select")];
  return {
    engine: selects[0].value,
    placement: selects[1].value,
    disabled: [...selects[1].options].filter((o) => o.disabled).map((o) => o.value),
    rows: [...box.querySelectorAll("td.mono")].map((td) => td.textContent),
    ticked: [...box.querySelectorAll('input[type="checkbox"]')].filter((c) => c.checked).map((c) => c.parentElement.textContent),
    note: (__saveRow().querySelector(":scope > span.muted") || {}).textContent || "",
  };
};
window.__choose = (i, value) => {
  const s = __editor().querySelectorAll("select")[i];
  s.value = value; s.dispatchEvent(new Event("change"));
};
window.__tick = (name, on) => {
  const c = [...__editor().querySelectorAll('input[type="checkbox"]')].find((c) => c.parentElement.textContent === name);
  c.checked = on; c.dispatchEvent(new Event("change"));
};
window.__type = (label, value) => {
  const row = [...__editor().querySelectorAll("tr")].find((tr) => tr.querySelector("td.mono").textContent === label);
  const input = row.querySelector("input"); input.value = value; input.dispatchEvent(new Event("input"));
};
window.__pick = (repo) => {
  const radio = [...__editor().querySelectorAll('input[type="radio"]')].find((r) => r.value === repo);
  radio.checked = true; radio.dispatchEvent(new Event("change"));
};
window.__radios = (model) => [...__editor().querySelectorAll(`input[type="radio"][name="engine-${model}"]`)]
  .map((r) => ({ repo: r.value, disabled: r.disabled }));
window.__option = (text, on) => {
  const box = [...__editor().querySelectorAll("label.option")].find((l) => l.textContent.includes(text)).querySelector("input");
  box.checked = on; box.dispatchEvent(new Event("change"));
};
window.__save = () => [...__editor().querySelectorAll("button")].find((b) => b.textContent === "Save to configuration").click();
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


@pytest.mark.timeout(240)
def test_switching_to_vllm_from_the_console(served):
    supervisor, url, config_path, original = served
    with open_page(f"{url}/ui/#rented") as page:
        # The page opens its key dialog as its very last statement: open means it has run to
        # the end and attached its handlers.
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.until("!!__editor()", within=30, what="the engine editor")

        # It opens on what the pool runs now.
        now = page.js("__describe()")
        assert (now["engine"], now["placement"]) == ("ollama", "all")
        assert now["disabled"] == ["all_proxy"], "the router is only for vLLM"

        # Choosing vLLM rules out one process holding every model, and lands on a model to a host.
        page.js("__choose(0, 'vllm'); true")
        chosen = page.js("__describe()")
        assert chosen["disabled"] == ["all"]
        assert chosen["placement"] == "declared"
        assert "images, newest first" in chosen["rows"]

        # Ticking a model asks for its vLLM build — the redraw that was missing.
        page.js(f"__tick({json.dumps(BIG)}, true); __tick({json.dumps(SMALL)}, true); true")
        ticked = page.js("__describe()")
        assert f"vllm build of {BIG}" in ticked["rows"] and f"vllm build of {SMALL}" in ticked["rows"]
        assert f"vllm build of {EMBED}" not in ticked["rows"], "not rented for, so not asked"

        # Its builds are offered from the hub, sorted, to pick with a radio button — the original
        # first, files for other engines left out (D100).
        page.until(f"__radios({json.dumps(BIG)}).length > 0", within=30, what="the builds on the hub")
        offered = [r["repo"] for r in page.js(f"__radios({json.dumps(BIG)})")]
        assert offered[0] == "google/gemma-4-26B-A4B-it" and BIG_REPO in offered
        assert not any("GGUF" in r or "MLX" in r for r in offered)

        # A model rented for with no build is refused, in words, and nothing is written.
        page.js(f"__pick({json.dumps(BIG_REPO)}); __save(); true")
        page.until("window.__saves >= 1", what="the first save's answer")
        time.sleep(0.5)
        refused = page.js("__describe().note")
        assert "not saved" in refused and "no build of" in refused and SMALL in refused, refused
        assert "pydantic" not in refused and "[type=" not in refused, "an operator's sentence, not a dump"
        assert config_path.read_text() == original

        # Renting for the one with a build is saved, with tool calling on, and the running pool
        # takes it.
        page.js(f"__tick({json.dumps(SMALL)}, false); __option('tool calling', true); __save(); true")
        page.until("window.__saves >= 2", within=30, what="the second save's answer")
        page.until("!!__editor() && __describe().note.includes(' saved ·')", within=30, what="the redraw")

    written = yaml.safe_load(config_path.read_text())
    assert written["pool"]["models_per_host"] == "declared"
    assert written["rented"]["engine"] == "vllm" and written["rented"]["models"] == [BIG]
    assert written["rented"]["engine_start"] is None
    assert written["rented"]["engine_options"] == ["tool_calling"]
    assert written["hosts"][0]["models"] == [EMBED, SMALL, BIG], "the laptop keeps what it held"
    assert supervisor.config.rented_engine() == "vllm"
