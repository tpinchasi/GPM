"""Add a trained model on the console's Models screen, in a real browser (S15, D178–D182).

The API behind it is covered in test_trained_add.py; this reads a file, fills in what its output
means, tests it, adds it, and turns containers on through the confirmation that asks for the pool's
name. Skipped where no browser is installed.
"""

import json

import httpx
import pytest
import yaml
from fakes.browser import a_browser, open_page
from test_host_removal import serve
from test_trained_add import MODEL

pytestmark = pytest.mark.skipif(a_browser() is None, reason="no browser here")

ADMIN_KEY = "gpmx_console_trained"
POOL = f"""
pool: {{name: shop, model_set: [chatter], models_per_host: declared, probe_interval_s: 3600}}
auth: {{admin_keys: ["{ADMIN_KEY}"], app_keys: [gpma_console_trained]}}
hosts:
  - id: mac
    kind: local
    models: [chatter]
    transport: {{type: http, base_url: "http://127.0.0.1:1"}}
  - id: box
    kind: local
    engine: onnx
    transport: {{type: http, base_url: "http://127.0.0.1:8190"}}
    agent: {{url: "http://127.0.0.1:8096", bearer_env: BOX_AGENT_KEY}}
"""

HELPERS = r"""
window.__panel = () => [...document.querySelectorAll(".panel")].find((p) => (p.querySelector("h2") || {}).textContent === "Add a trained model");
window.__button = (text) => [...__panel().querySelectorAll("button")].find((b) => b.textContent === text);
window.__field = (label) => [...__panel().querySelectorAll("label.field")].find((l) => l.querySelector(".k").textContent === label);
window.__type = (node, value) => { node.value = value; node.dispatchEvent(new Event("input")); return true; };
window.__pick = (node, value) => { node.value = value; node.dispatchEvent(new Event("change")); return true; };
true;
"""


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("BOX_AGENT_KEY", "test-agent-key")
    supervisor, server, loop, database, path = serve(tmp_path, POOL)
    try:
        yield supervisor, server.base_url, path
    finally:
        server.stop()
        loop.stop()
        database.close()


@pytest.mark.timeout(240)
def test_a_model_is_read_described_tested_and_added_and_containers_turned_on(served):
    supervisor, url, path = served
    tag = httpx.post(f"{url}/pool/models/uploads", content=MODEL, headers={"Authorization": f"Bearer {ADMIN_KEY}"}).json()["tag"]
    with open_page(f"{url}/ui/#models") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN_KEY)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.until("!!__panel() && !!__button('Read the file')", within=30, what="the Add a trained model panel")

        # The file's own inputs and outputs, read from it.
        page.js(f"__type(__field('…or on the hub').querySelector('input'), {json.dumps(tag)})")
        page.js("__button('Read the file').click(); true")
        page.until("__panel().textContent.includes('pixel_values  float32 [N, 3, 4, 4]')", within=30, what="the inputs read from the file")
        assert "logits  float32 [N, 3]" in page.js("__panel().textContent")
        assert page.js("__field('arrives as').querySelector('select').value") == "image", "a 4-dimension input reads as a picture"

        # Three scores, labelled — tested against the file, then added.
        page.js("__type(__field('labels, one a line').querySelector('textarea'), 'red\\ngreen\\nblue')")
        page.js("__type(__field('Apps ask for it as').querySelector('input'), 'colours')")
        page.js("[...__panel().querySelectorAll('label')].find((l) => l.textContent.trim() === 'box').querySelector('input').click(); true")
        page.js("__button('Test').click(); true")
        page.until("__panel().textContent.includes('matches its file')", within=30, what="the test passing")
        page.js("__button('Add').click(); true")
        page.until("!__panel().textContent.includes('pixel_values')", within=30, what="the draft cleared once added")

        written = yaml.safe_load(path.read_text())
        variant = written["catalog"]["colours"]["variants"][0]
        assert variant["tag"] == tag and variant["engine"] == "onnx"
        assert variant["spec"]["outputs"]["logits"]["labels"] == ["red", "green", "blue"]
        assert next(h for h in written["hosts"] if h["id"] == "box")["models"] == ["colours"]

        # Containers: off until the pool's name is typed.
        page.js("__pick(__field('Comes as').querySelector('select'), 'container')")
        page.until("!!__button('Turn on…')", within=30, what="the containers switch")
        page.js("__button('Turn on…').click(); true")
        page.until("document.getElementById('confirm-dialog').open", what="the confirmation")
        assert page.js("document.getElementById('confirm-ok').disabled") is True, "nothing typed yet"
        page.js("const i = document.getElementById('confirm-input'); i.value = 'shop'; i.dispatchEvent(new Event('input'));"
                "document.getElementById('confirm-ok').click(); true")
        page.until("!!__button('Turn off')", within=30, what="containers on")

        # A private registry's login, for rented machines (D183): saved, listed, never shown again.
        page.js("__type(__field('Registry').querySelector('input'), 'ghcr.io');"
                "__type(__field('User').querySelector('input'), 'me');"
                "__type(__field('Pull-only token').querySelector('input'), 'not-a-real-password'); true")
        page.js("__button('Save login').click(); true")
        page.until("__panel().textContent.includes('ghcr.io') && !!__button('Remove')", within=30, what="the login listed")
        assert page.js("__field('Pull-only token').querySelector('input').value") == ""
        assert "not-a-real-password" not in page.js("document.body.innerHTML")
        page.js("__type(__field('Start on a rented machine').querySelector('textarea'), 'python\\n-m\\nserver'); true")
        assert page.js("JSON.stringify(trainedBuild().start)") == '["python","-m","server"]'
    assert supervisor.config.containers.enabled
    assert supervisor.registry_logins().listed() == [{"registry": "ghcr.io", "username": "me"}]
