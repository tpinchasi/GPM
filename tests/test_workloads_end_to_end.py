"""A workload from creation to its end, through every process (D115, workloads.md).

The real supervisor, router and control API over one database; the fake provider's market; fake
engines for the machines it hands out. The operator plans and creates a workload through the
control API, gets its key once, and an app with that key is served — first on a shared host it
borrows while its own comes up, then on its own. The app key never reaches the workload's host,
the workload key never reaches the control API, and ending the workload takes its host and its
key away. Nothing here spends money.
"""

import time

import httpx
import pytest
from fakes.harness import APP_KEY, EngineSpec, ServerHandle, pool_harness
from gpm_server.providers import default_offer
from gpm_server.supervisor.control import create_control_app

ADMIN = "gpmx_workloads_admin"
SHARED, BIG, SOLO = "m1", "big", "solo"
CATALOG = {
    BIG: {"variants": [{"tag": BIG, "size_gb": 4}]},
    # Outside the pool's set: only a workload serves it, and no shared host fetches it (D115).
    SOLO: {"variants": [{"tag": SOLO, "size_gb": 4}], "workloads_only": True},
}


def rented():
    return {
        "provider": "fake", "workers": 2,
        "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 1.0},
        "bidding": {"premium": 0.02}, "scale": {"scale_up_after_s": 0},
        "teardown": {"idle_minutes": 30},
    }


@pytest.fixture
def pool():
    with pool_harness(
        [EngineSpec(id="laptop", resident={SHARED, BIG}, workers=4)],
        model_set=[SHARED, BIG], catalog=CATALOG,
        rentable=[EngineSpec(id="market-1", resident={BIG, SOLO}, workers=4)],
        rented=rented(),
        extra_config={
            "auth": {"app_keys": [APP_KEY], "admin_keys": [ADMIN]},
            "limits": {"max_rented_hosts": 2},
            "workloads": {"min_reliability": 0.0, "borrow_share": 0.5},
        },
    ) as harness:
        harness.supervisor.fleet.provider.offers = [default_offer("o-1", "m-1", min_bid_hourly=0.20)]
        control = ServerHandle(create_control_app(harness.supervisor, harness.config), harness.loop)
        harness.control_url = control.base_url
        try:
            yield harness
        finally:
            control.stop()


def admin(pool):
    return httpx.Client(base_url=pool.control_url, headers={"Authorization": f"Bearer {ADMIN}"}, timeout=30)


def ask(client):
    return client.post("/v1/chat/completions", json={"model": BIG, "messages": [{"role": "user", "content": "hi"}]})


def until(pool, check, what, within=20.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        pool.reprobe()
        if check():
            return
        time.sleep(0.05)
    raise AssertionError(f"not {what} within {within}s")


REQUEST = {"name": "research", "model": BIG, "latency_s": 30, "parallel": 2, "hours": 2}


def test_a_workload_from_plan_to_end(pool):
    with admin(pool) as control:
        plan = control.post("/pool/workloads/plan", json=REQUEST).json()["plan"]
        assert plan["refused"] is None, plan
        assert plan["hosts_at_start"] == 1 and plan["workers_per_host"] == 2 and not plan["workers_measured"]
        assert plan["budget_derived"] and plan["max_spend"] == plan["derived_budget"] > 0
        assert pool.supervisor.fleet.hosts == {}, "a plan rents nothing"

        refused = control.post("/pool/workloads", json=REQUEST)
        assert refused.status_code == 400 and "confirm_max_spend" in refused.json()["detail"]
        assert pool.supervisor.leases.open_leases() == [], "nothing opened without the budget confirmed"

        made = control.post("/pool/workloads", json={**REQUEST, "confirm_max_spend": plan["max_spend"]})
        assert made.status_code == 201, made.text
        body = made.json()
        key = body["connection"]["api_key"]
        assert key.startswith("gpmw_") and body["connection"]["shown_once"]
        assert body["connection"]["base_url"].endswith("/v1")
        (lease,) = pool.supervisor.leases.open_leases()
        assert lease.workload == "research" and lease.max_spend == plan["max_spend"]
        assert "api_key" not in str(control.get("/pool/workloads/research").json()["workload"]), "never shown again"

    pool.loop.run(pool.state.registry.refresh())
    # Before its own host is ready: served on the laptop, lent by the shared workload.
    with pool.client(key=key) as app:
        first = ask(app)
    assert first.status_code == 200 and first.headers["X-GPM-Host"] == "laptop"
    assert pool.wait_for_log(1)[0]["borrowed"] == 1

    until(pool, lambda: any(h.state == "ready" for h in pool.supervisor.fleet.hosts_of("research")), "its host ready")
    until(pool, lambda: pool.supervisor.workloads.get("research").state == "serving", "serving")
    (host,) = pool.supervisor.fleet.hosts_of("research")
    with pool.client(key=key) as app:
        own = ask(app)
    assert own.status_code == 200 and own.headers["X-GPM-Host"] == host.host_id
    with pool.client() as shared_app:
        for _ in range(3):
            assert ask(shared_app).headers["X-GPM-Host"] == "laptop", "the app key never reaches a workload's host"

    with httpx.Client(base_url=pool.control_url, headers={"Authorization": f"Bearer {key}"}) as sneaky:
        assert sneaky.get("/pool/leases").status_code == 403, "a workload key never reaches the control API"

    with admin(pool) as control:
        view = control.get("/pool/workloads/research").json()["workload"]
        assert view["state"] == "serving" and [h["host_id"] for h in view["hosts"]] == [host.host_id]
        assert view["answers"]["count"] >= 2 and view["answers"]["borrowed"] == 1
        ended = control.post("/pool/workloads/research/end").json()["workload"]
        assert ended["state"] == "ending"

    pool.loop.run(pool.state.registry.refresh())
    with pool.client(key=key) as app:
        assert ask(app).json()["error"] == "workload_ended"
    until(pool, lambda: pool.supervisor.workloads.get("research").state == "ended", "ended")
    assert host.instance.instance_id not in pool.supervisor.fleet.provider.instances


def test_a_typed_budget_needs_no_confirmation_and_names_are_never_reused(pool):
    with admin(pool) as control:
        made = control.post("/pool/workloads", json={**REQUEST, "max_spend": 3.0})
        assert made.status_code == 201, made.text
        again = control.post("/pool/workloads", json={**REQUEST, "max_spend": 3.0})
        assert again.status_code == 400 and "already exists" in again.json()["detail"]


def test_a_start_the_pool_has_no_room_for_is_refused_with_the_numbers(pool):
    with admin(pool) as control:
        answer = control.post("/pool/workloads/plan", json={**REQUEST, "parallel": 6})
        plan = answer.json()["plan"]
        assert plan["hosts_at_start"] == 3
        assert "room for 2" in plan["refused"]
        created = control.post("/pool/workloads", json={**REQUEST, "parallel": 6, "max_spend": 5})
        assert created.status_code == 400 and "room for" in created.json()["detail"]


def test_a_model_outside_the_pool_is_refused_in_words(pool):
    with admin(pool) as control:
        answer = control.post("/pool/workloads/plan", json={**REQUEST, "model": "nope"})
    assert answer.status_code == 400 and "neither in the pool's model set" in answer.json()["detail"]


def test_a_rotated_key_replaces_the_old_after_its_grace(pool):
    with admin(pool) as control:
        old = control.post("/pool/workloads", json={**REQUEST, "max_spend": 3.0}).json()["connection"]["api_key"]
        rotated = control.post("/pool/workloads/research/keys").json()
        new = rotated["connection"]["api_key"]
        assert new != old and rotated["old_keys_valid_minutes"] == 30
    pool.loop.run(pool.state.registry.refresh())
    with pool.client(key=old) as app:
        assert ask(app).status_code == 200, "the old key works through its grace"
    with pool.client(key=new) as app:
        assert ask(app).status_code == 200


def test_extending_raises_the_lease_only_when_typed_again(pool):
    with admin(pool) as control:
        control.post("/pool/workloads", json={**REQUEST, "max_spend": 3.0})
        refused = control.post("/pool/workloads/research/extend", json={"max_spend": 6.0})
        assert refused.status_code == 400 and "type the new value again" in refused.json()["detail"]
        half = control.post("/pool/workloads/research/extend", json={"max_spend": 6.0, "hours": 1, "confirm": 6.0})
        assert half.status_code == 400 and "type the hours again" in half.json()["detail"], "each raise on its own"
        extended = control.post("/pool/workloads/research/extend",
                                json={"max_spend": 6.0, "hours": 1, "confirm": 6.0, "confirm_hours": 1})
        assert extended.status_code == 200, extended.text
        lease = extended.json()["workload"]["lease"]
        assert lease["max_spend"] == 6.0 and lease["hours_left"] > 2.5


# --- the command line (gpm workload …) ---


def cli(pool, capsys, monkeypatch, *argv):
    from gpm_server.cli import main

    monkeypatch.setenv("GPM_ADMIN_KEY", ADMIN)
    monkeypatch.setenv("GPM_CONTROL_URL", pool.control_url)
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


CREATE = ["--model", BIG, "--latency", "30", "--parallel", "2", "--hours", "2"]


def test_the_command_line_plans_without_creating(pool, capsys, monkeypatch):
    code, out, _ = cli(pool, capsys, monkeypatch, "workload", "plan", "research", *CREATE)
    assert code == 0
    assert "starts on 1 host(s) at 2 at once each (not measured yet)" in out and "(derived)" in out
    assert pool.supervisor.workloads.store.all() == []


def test_the_command_line_will_not_accept_a_derived_budget_unasked(pool, capsys, monkeypatch):
    code, _, err = cli(pool, capsys, monkeypatch, "workload", "create", "research", *CREATE)
    assert code == 1 and "--confirm-max-spend" in err
    assert pool.supervisor.workloads.store.all() == []


def test_the_command_line_creates_and_shows_the_key_once(pool, capsys, monkeypatch):
    plan = pool.loop.run(pool.supervisor.workloads.plan(
        __import__("gpm_server.supervisor.workloads", fromlist=["WorkloadRequest"]).WorkloadRequest(
            name="research", model=BIG, latency_s=30, parallel=2, hours=2)))
    code, out, err = cli(pool, capsys, monkeypatch, "workload", "create", "research", *CREATE,
                         "--confirm-max-spend", str(plan["max_spend"]))
    assert code == 0, err
    assert "api_key:  gpmw_" in out and "base_url: http://" in out
    code, out, _ = cli(pool, capsys, monkeypatch, "workload", "show", "research")
    assert code == 0 and "gpmw_" not in out, "shown once, never again"
    code, out, _ = cli(pool, capsys, monkeypatch, "workload", "end", "research")
    assert code == 0 and '"state": "ending"' in out


def test_two_creations_at_once_cannot_both_take_the_room(pool):
    """A plan awaits the market; two creations in flight must not both pass the caps (the
    review's finding 1)."""
    import asyncio

    from gpm_server.supervisor.workloads import WorkloadRefused, WorkloadRequest

    workloads = pool.supervisor.workloads

    async def both():
        def req(name):
            return WorkloadRequest(name=name, model=BIG, latency_s=30, parallel=4, hours=2, max_spend=3.0)
        return await asyncio.gather(workloads.create(req("one")), workloads.create(req("two")), return_exceptions=True)

    first, second = pool.loop.run(both())
    assert not isinstance(first, Exception)
    assert isinstance(second, WorkloadRefused) and "room for 0" in str(second)
    assert [lease.workload for lease in pool.supervisor.leases.open_leases()] == ["one"]


def test_the_same_name_twice_at_once_leaves_no_stray_lease(pool):
    import asyncio

    from gpm_server.supervisor.workloads import WorkloadRefused, WorkloadRequest

    workloads = pool.supervisor.workloads
    req = WorkloadRequest(name="same", model=BIG, latency_s=30, parallel=2, hours=2, max_spend=3.0)

    async def both():
        return await asyncio.gather(workloads.create(req), workloads.create(req), return_exceptions=True)

    results = pool.loop.run(both())
    assert sum(isinstance(r, WorkloadRefused) for r in results) == 1
    assert len(pool.supervisor.leases.open_leases()) == 1


def test_a_workloads_lease_is_changed_through_its_workload_only(pool):
    with admin(pool) as control:
        control.post("/pool/workloads", json={**REQUEST, "max_spend": 3.0})
        (lease,) = pool.supervisor.leases.open_leases()
        patched = control.patch(f"/pool/leases/{lease.lease_id}", json={"max_hours": 9, "confirm": 9})
        closed = control.delete(f"/pool/leases/{lease.lease_id}")
    assert patched.status_code == 409 and "workloads/research/extend" in patched.json()["detail"]
    assert closed.status_code == 409
    assert pool.supervisor.leases.get(lease.lease_id).is_open


def test_a_name_shaped_like_a_host_id_is_refused(pool):
    with admin(pool) as control:
        answer = control.post("/pool/workloads/plan", json={**REQUEST, "name": "rented-a1b2c3"})
    assert answer.status_code == 400


def test_a_workloads_only_model_is_served_to_its_workload_and_to_nobody_else(pool):
    with admin(pool) as control:
        plan = control.post("/pool/workloads/plan", json={**REQUEST, "model": SOLO}).json()["plan"]
        assert plan["refused"] is None and not plan["borrow_while_starting"], "no shared host holds it"
        assert not any("fetch it too" in r for r in plan["reasons"])
        made = control.post("/pool/workloads", json={**REQUEST, "model": SOLO, "max_spend": 3.0})
        assert made.status_code == 201, made.text
        key = made.json()["connection"]["api_key"]
    until(pool, lambda: any(h.state == "ready" for h in pool.supervisor.fleet.hosts_of("research")), "its host ready")
    with pool.client(key=key) as app:
        answer = app.post("/v1/chat/completions", json={"model": SOLO, "messages": [{"role": "user", "content": "hi"}]})
    assert answer.status_code == 200, answer.text
    with pool.client() as shared_app:
        refused = shared_app.post("/v1/chat/completions", json={"model": SOLO, "messages": [{"role": "user", "content": "hi"}]})
    assert refused.status_code == 404, "an app key does not learn a workload's model exists"
    assert SOLO not in pool.supervisor.config.pool.model_set


def test_a_model_is_either_shared_or_workloads_only():
    from gpm_server.config import PoolConfig

    base = {"pool": {"name": "t", "model_set": ["a"]}, "auth": {"app_keys": ["k"]},
            "hosts": [{"id": "h", "kind": "local", "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}}]}
    PoolConfig.model_validate({**base, "catalog": {"w": {"variants": [{"tag": "w"}], "workloads_only": True}}})
    with pytest.raises(ValueError, match="workloads_only: true"):
        PoolConfig.model_validate({**base, "catalog": {"w": {"variants": [{"tag": "w"}]}}})
    with pytest.raises(ValueError, match="either the shared hosts'"):
        PoolConfig.model_validate({**base, "catalog": {"a": {"variants": [{"tag": "a"}], "workloads_only": True}}})
