"""The console's plan chat (S19 stage 8, D204; threat model T42).

A description goes to the control API, which writes it to the database; the router's watcher
claims it once, asks the chat's model through its own dispatch — `keep_on: owned`, the app
`console`, held to the plan's schema — and writes back a plan the pool accepts with its readback,
or the model's words with their refusals after two tries. The answer is read once; a request can
be cancelled; one at a time. The model must run on the pool's own machines. Nothing here spends.
"""

import contextlib
import json
import time

import httpx
import pytest
from fakes.harness import EngineSpec, ServerHandle, pool_harness
from gpm_server.supervisor.control import create_control_app
from pydantic import ValidationError

MODEL = "m1"
ADMIN = "gpmx_admin_chat"
PLAN = {"until_ready": ["shared"], "when_full": [{"hosted": {"after_s": 5, "budget": {"per_day": 3}}}]}


@contextlib.contextmanager
def with_control(console, chunk_delay_s=0.0):
    with pool_harness(
        [EngineSpec(id="laptop", resident={MODEL}, workers=2, chunk_delay_s=chunk_delay_s)], model_set=[MODEL],
        catalog={MODEL: {"variants": [{"tag": MODEL}]}},
        extra_config={"auth": {"app_keys": ["gpmx_app_chat"], "admin_keys": [ADMIN]}, "console": console},
    ) as harness:
        harness.reprobe()
        control = ServerHandle(create_control_app(harness.supervisor, harness.config), harness.loop)
        harness.control_url = control.base_url
        try:
            yield harness
        finally:
            control.stop()


@pytest.fixture
def pool():
    with with_control({"plan_assistant": {"model": MODEL, "timeout_s": 20}}) as harness:
        yield harness


def call(pool, method, path, body=None):
    return httpx.request(method, f"{pool.control_url}{path}", headers={"Authorization": f"Bearer {ADMIN}"},
                         json=body, timeout=30)


def answered(pool, request_id, within=15.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        seen = call(pool, "GET", f"/pool/workloads/plan-assistant/{request_id}")
        assert seen.status_code == 200, seen.text
        if seen.json()["state"] == "answered":
            return seen.json()
        time.sleep(0.2)
    raise AssertionError("no answer")


def test_a_description_becomes_a_plan_read_back_and_the_answer_is_read_once(pool):
    fake = pool.engines["laptop"].fake
    fake.scripted = [json.dumps(PLAN)]
    asked = call(pool, "POST", "/pool/workloads/plan-assistant",
                 {"description": "cover the start with shared machines; hosted APIs when full, $3 a day",
                  "plan": {"until_ready": ["local"]}})
    assert asked.status_code == 202 and asked.json()["model"] == MODEL, asked.text
    answer = answered(pool, asked.json()["id"])
    assert answer["plan"] == PLAN
    assert answer["readback"][2].startswith("When full — every worker of its own busy: hosted: hosted APIs after 5 s")
    assert call(pool, "GET", f"/pool/workloads/plan-assistant/{asked.json()['id']}").status_code == 404, "read once"
    # Through the router's own dispatch, as the app `console`, kept on the pool's own machines.
    row = [r for r in pool.wait_for_log(1) if r["app"] == "console"][-1]
    assert row["keep_on"] == "owned" and row["host_id"] == "laptop"
    sent = json.loads([body for path, body, _ in fake.received if path == "/v1/chat/completions"][-1])
    assert sent["response_format"]["json_schema"]["name"] == "serving_plan" and sent["model"] == MODEL
    assert '"until_ready": ["local"]' in sent["messages"][-1]["content"], "the builder's plan, to change"
    assert call(pool, "GET", "/pool/status").json()["plan_assistant"] == {"model": MODEL}


def test_an_answer_still_refused_after_two_tries_comes_back_as_words_with_their_refusals(pool):
    fake = pool.engines["laptop"].fake
    fake.scripted = ['{"steady": "hosted"}', "a plan with hosted APIs"]
    asked = call(pool, "POST", "/pool/workloads/plan-assistant", {"description": "anything"})
    answer = answered(pool, asked.json()["id"])
    assert answer["words"] == "a plan with hosted APIs" and answer["refusals"][0].startswith("not JSON")
    second = json.loads([body for path, body, _ in fake.received if path == "/v1/chat/completions"][-1])
    assert "The pool refused that plan: serve.steady" in second["messages"][-1]["content"], "told what was wrong"


def test_one_at_a_time_and_a_request_can_be_cancelled(pool, monkeypatch):
    from gpm_server import plan_assistant

    # The router's watcher claims nothing here, so the request stays waiting.
    monkeypatch.setattr(plan_assistant.AssistantRequests, "claim", lambda self, *args, **kwargs: None)
    first = call(pool, "POST", "/pool/workloads/plan-assistant", {"description": "one"})
    assert first.status_code == 202
    busy = call(pool, "POST", "/pool/workloads/plan-assistant", {"description": "two"})
    assert busy.status_code == 409 and "one" not in busy.json()["detail"]
    assert call(pool, "GET", f"/pool/workloads/plan-assistant/{first.json()['id']}").json()["state"] == "waiting"
    assert call(pool, "DELETE", f"/pool/workloads/plan-assistant/{first.json()['id']}").json()["cancelled"] is True
    assert call(pool, "POST", "/pool/workloads/plan-assistant", {"description": "two"}).status_code == 202
    for refused in ({"description": ""}, {"description": "x" * 4001}, {"description": "x", "plan": [1]}):
        assert call(pool, "POST", "/pool/workloads/plan-assistant", refused).status_code == 400


def test_the_chat_is_off_unless_a_model_of_the_pools_own_machines_is_named():
    with with_control({}) as off:
        refused = call(off, "POST", "/pool/workloads/plan-assistant", {"description": "x"})
        assert refused.status_code == 404 and "console.plan_assistant.model" in refused.json()["detail"]
        assert call(off, "GET", "/pool/status").json()["plan_assistant"] is None
    from gpm_server.config import PoolConfig

    base = {"pool": {"name": "t", "model_set": ["a", "api", "b"]}, "auth": {"app_keys": ["k"]},
            "hosts": [{"id": "h", "kind": "local", "workers": 1, "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}],
            "catalog": {"a": {"variants": [{"tag": "a", "enforces_schema": False}]},
                        "api": {"variants": [{"tag": "api-1", "engine": "openai-api",
                                              "price": {"input": 1.0, "output": 1.0}}]}}}
    for model, says in (("nope", "not in the pool's model set"), ("api", "served by hosted APIs alone"),
                        ("a", "known to ignore a schema"), ("b", "is not catalogued")):
        with pytest.raises(ValidationError, match=says):
            PoolConfig.model_validate({**base, "console": {"plan_assistant": {"model": model}}})


@pytest.mark.timeout(240)
def test_the_console_shows_the_chats_plan_as_a_change_with_undo(pool):
    from fakes.browser import a_browser, open_page
    from test_console_serving_plan import MORE
    from test_console_workloads import HELPERS

    if a_browser() is None:
        pytest.skip("no browser here")
    pool.engines["laptop"].fake.scripted = [json.dumps(PLAN)]
    with open_page(f"{pool.control_url}/ui/#workloads") as page:
        page.until("(document.getElementById('key-dialog') || {}).open === true", within=60, what="the page")
        page.js(f"document.getElementById('key-input').value = {json.dumps(ADMIN)};"
                "document.getElementById('key-form').requestSubmit(); true")
        page.js(HELPERS)
        page.js(MORE)
        page.until("__text().includes('Serving plan')", within=30, what="the builder")
        assert f"{MODEL} on this pool writes the plan" in page.js("__text()")
        page.js("const box = document.querySelector('.plan-chat textarea'); box.value = 'shared, then hosted when full';"
                "box.dispatchEvent(new Event('input')); __click('Propose a plan'); true")
        page.until("__text().includes('proposed a plan; it is in the builder now')", within=30, what="the proposal")
        page.js("__tab('JSON')")
        assert page.js("__json()") == PLAN
        page.js("__click('Undo')")
        page.js("__tab('JSON')")
        assert page.js("__json()") == {}, "undone: the builder's plan as it was"


def busy(pool):
    return sum(h.busy for h in pool.state.hosts if h.host_id == "laptop")


def until(check, what, within=20.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.parametrize("how", ["cancelled", "timed out"])
def test_a_chat_request_ended_early_gives_its_worker_back(how):
    with with_control({"plan_assistant": {"model": MODEL, "timeout_s": 2 if how == "timed out" else 60}},
                      chunk_delay_s=1.0) as pool:
        asked = call(pool, "POST", "/pool/workloads/plan-assistant", {"description": "slowly"})
        until(lambda: busy(pool) == 1, "the chat's request on a worker")
        if how == "cancelled":
            assert call(pool, "DELETE", f"/pool/workloads/plan-assistant/{asked.json()['id']}").json()["cancelled"]
        else:
            assert "did not answer within 2s" in answered(pool, asked.json()["id"])["error"]
        until(lambda: busy(pool) == 0, "its worker given back")
        app = httpx.post(f"{pool.url}/v1/chat/completions", headers={"Authorization": "Bearer gpmx_app_chat"},
                         json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False},
                         timeout=30)
        assert app.status_code == 200, "both workers serve apps again"


def test_a_request_no_router_took_does_not_hold_the_chat_and_a_busy_answer_names_it(pool, monkeypatch):
    from gpm_server import plan_assistant

    monkeypatch.setattr(plan_assistant.AssistantRequests, "claim", lambda self, *args, **kwargs: None)
    requests = plan_assistant.AssistantRequests(pool.database)
    stuck = requests.create("long ago", None, 20, now=time.time() - 3600)
    busy_answer = call(pool, "POST", "/pool/workloads/plan-assistant", {"description": "now"})
    assert busy_answer.status_code == 202, "the one nobody took is answered as such, and this one taken"
    assert "no router took it" in requests.get(stuck)["answer"]["error"]
    again = call(pool, "POST", "/pool/workloads/plan-assistant", {"description": "again"})
    assert again.status_code == 409 and again.json()["id"] == busy_answer.json()["id"], "names the one in flight"
    assert call(pool, "POST", "/pool/workloads/plan-assistant", None).status_code == 400


def test_the_routers_own_key_is_never_one_anyone_can_send(pool):
    refused = httpx.post(f"{pool.url}/v1/chat/completions", headers={"Authorization": "Bearer caf\xe9".encode("latin-1")},
                         json={"model": MODEL, "messages": []}, timeout=10)
    assert refused.status_code == 401
    named = httpx.post(f"{pool.control_url}/pool/apps", headers={"Authorization": f"Bearer {ADMIN}"},
                       json={"app": "console"}, timeout=10)
    assert named.status_code == 400 and "the console's plan chat" in named.text
