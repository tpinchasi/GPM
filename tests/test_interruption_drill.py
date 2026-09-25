"""The interruption drill — phase 2's exit criterion, end to end.

docs/roadmap.md §2: a forced eviction recovers unattended inside a lease and does nothing
outside one; an idle rented host releases itself; a stray instance is caught by the sweep;
killing the supervisor leaves routing up and the dead-man timer removes the rented host; a lease
stops before its dollar cap when reported charges run ahead of the estimate.

**No test here needs a cloud account.** A whole pool runs: a local engine, a rentable one, the
supervisor renting against the fake provider, and the router serving real HTTP throughout.
"""

import time

import pytest
from fakes.harness import EngineSpec, pool_harness

MODEL = "m1"

RENTED = {
    "provider": "fake",
    "workers": 2,
    "offer_policy": {"min_disk_gb": 10, "max_all_in_hourly": 0.60}, "bidding": {"premium": 0.02},
    "scale": {"scale_up_after_s": 0},
    "teardown": {"idle_minutes": 10},
}


def chat():
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False}


def drill_pool(**overrides):
    rented = {**RENTED, **overrides.pop("rented", {})}
    return pool_harness(
        # The local host answers slowly enough that a second request really does overflow.
        # Without this the drill is a race: on a fast machine one worker can serve three quick
        # requests one after another, and the rented host — the thing under test — sees none.
        [EngineSpec(id="local-1", resident={MODEL}, kind="local", workers=1, chunk_delay_s=0.5)],
        rentable=[EngineSpec(id="market-1", resident={MODEL}, workers=2)],
        model_set=[MODEL],
        rented=rented,
        **overrides,
    )


def rented_hosts(pool):
    return {k: v for k, v in pool.supervisor.fleet.hosts.items() if not v.released}


def settle(pool, passes=2):
    """A few control-loop passes, with the router picking up each result."""
    for _ in range(passes):
        pool.reprobe()


def event_kinds(pool):
    return [event["kind"] for event in pool.supervisor.events.recent(limit=200)]


# --- nothing spends without a lease ---


def test_with_no_lease_nothing_is_rented_however_much_is_asked_for():
    with drill_pool() as pool:
        with pool.client() as client:
            for _ in range(4):
                client.post("/api/chat", json=chat())
        settle(pool)

        assert rented_hosts(pool) == {}
        assert pool.supervisor.fleet.provider.instances == {}


# --- a lease brings capacity up, and the router uses it ---


def test_a_lease_rents_a_host_that_then_serves_traffic():
    with drill_pool() as pool:
        pool.supervisor.fleet.open_lease(
            workers=3, max_hours=2, max_spend=2.00, allow_rent=True
        )
        settle(pool, passes=3)

        rented = rented_hosts(pool)
        assert len(rented) == 1
        host_id = next(iter(rented))
        assert rented[host_id].state == "ready"

        # The router can see it, at the rented tier, behind the local host.
        with pool.client() as client:
            status = client.get("/pool/status").json()
        tiers = {h["host_id"]: h["priority"] for h in status["hosts"]}
        assert tiers[host_id] == 20
        assert tiers["local-1"] == 0

        # Local is one worker; a second concurrent request overflows onto the rented host.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            def call():
                with pool.client(timeout=20) as client:
                    return client.post("/api/chat", json=chat())

            responses = [f.result() for f in [threads.submit(call) for _ in range(3)]]

        assert all(r.status_code == 200 for r in responses)
        assert pool.rentable["market-1"].fake.received, "the rented host served nothing"


# --- eviction ---


def test_an_eviction_inside_a_lease_recovers_unattended():
    with drill_pool() as pool:
        pool.supervisor.fleet.open_lease(workers=3, max_hours=2, max_spend=2.00, allow_rent=True)
        settle(pool, passes=3)
        host = next(iter(rented_hosts(pool).values()))

        pool.supervisor.fleet.provider.evict(host.instance.instance_id)
        settle(pool, passes=3)

        assert "eviction" in event_kinds(pool)
        # Either back on the same machine or replaced — never left stopped and billing.
        for rented in rented_hosts(pool).values():
            assert rented.state != "stopped"


def test_an_eviction_outside_a_lease_does_nothing():
    with drill_pool() as pool:
        lease = pool.supervisor.fleet.open_lease(
            workers=3, max_hours=2, max_spend=2.00, allow_rent=True
        )
        settle(pool, passes=3)
        host = next(iter(rented_hosts(pool).values()))

        pool.supervisor.leases.close(lease.lease_id)
        pool.supervisor.fleet.provider.evict(host.instance.instance_id)
        settle(pool, passes=3)

        assert rented_hosts(pool) == {}
        assert pool.supervisor.fleet.provider.instances == {}


# --- idleness and caps ---


def test_an_idle_rented_host_releases_itself():
    with drill_pool(rented={"teardown": {"idle_minutes": 10, "park_when_idle": False}}) as pool:
        pool.supervisor.fleet.open_lease(workers=3, max_hours=2, max_spend=2.00, allow_rent=True)
        settle(pool, passes=3)
        host = next(iter(rented_hosts(pool).values()))

        # Ready, and nothing routed to it, for longer than the idle limit. (Idleness counts
        # from when a host became ready — a slow boot is not idleness.)
        host.ready_at = time.time() - 30 * 60
        host.hold_until = None
        settle(pool, passes=2)

        # That host is gone and stopped billing. The lease is still open and still wants
        # capacity, so the pool may well rent another — releasing the idle one is the point,
        # not ending up with none.
        assert host.host_id not in rented_hosts(pool)
        assert host.instance.instance_id not in pool.supervisor.fleet.provider.instances
        released = [
            e for e in pool.supervisor.events.recent(200)
            if e["kind"] == "released" and e["host_id"] == host.host_id
        ]
        assert released and "idle" in released[0]["summary"]


def test_a_lease_stops_before_its_cap_when_the_provider_reports_more():
    with drill_pool() as pool:
        lease = pool.supervisor.fleet.open_lease(
            workers=3, max_hours=2, max_spend=1.00, allow_rent=True
        )
        settle(pool, passes=3)
        host = next(iter(rented_hosts(pool).values()))

        host.created_at = time.time() - 1800
        pool.supervisor.fleet.provider.instances[host.instance.instance_id].created_at = host.created_at
        pool.supervisor.fleet.provider.set_reported_charges(host.instance.instance_id, multiplier=20)
        settle(pool, passes=2)

        assert pool.supervisor.leases.get(lease.lease_id).state == "closed"
        assert rented_hosts(pool) == {}
        capped = [e for e in pool.supervisor.events.recent(200) if e["kind"] == "lease_capped"][0]
        assert capped["numbers"]["enforced_at"] == pytest.approx(0.90)
        assert capped["numbers"]["spent"] > capped["numbers"]["estimate"]


# --- what nobody intended ---


def test_a_stray_instance_is_caught_by_the_sweep():
    with drill_pool() as pool:
        pool.supervisor.fleet.provider.strand(label="gpm/test/rented-nobody-knows")
        settle(pool)

        assert pool.supervisor.fleet.provider.instances == {}
        assert "orphan_swept" in event_kinds(pool)


# --- the supervisor dying ---


def test_killing_the_supervisor_leaves_routing_up_and_the_timer_armed():
    with drill_pool() as pool:
        pool.supervisor.fleet.open_lease(workers=3, max_hours=2, max_spend=2.00, allow_rent=True)
        settle(pool, passes=3)
        host = next(iter(rented_hosts(pool).values()))

        # What is on the rented host is what will end it if nobody comes back.
        onstart = pool.supervisor.fleet.provider.instances[host.instance.instance_id].spec.onstart
        assert "deadman.sh" in onstart
        assert "CONTAINER_API_KEY" in onstart

        pool.stop_supervisor()
        time.sleep(0.5)

        with pool.client() as client:
            response = client.post("/api/chat", json=chat())
            status = client.get("/pool/status").json()

        assert response.status_code == 200
        assert status["capacity"]["hosts_ready"] >= 1


# --- the panic button ---


def test_down_all_leaves_nothing_billing():
    with drill_pool() as pool:
        pool.supervisor.fleet.open_lease(workers=3, max_hours=2, max_spend=2.00, allow_rent=True)
        settle(pool, passes=3)
        assert rented_hosts(pool)

        fleet = pool.supervisor.fleet
        released = pool.loop.run(fleet.down_all())

        assert released
        assert fleet.provider.instances == {}
        assert pool.supervisor.leases.open_leases() == []  # or it would rent straight back
        # And the router stops being told about it.
        settle(pool)
        with pool.client() as client:
            status = client.get("/pool/status").json()
        assert all(h["kind"] != "rented-interruptible" for h in status["hosts"])
        assert fleet.provider.instances == {}  # still nothing, passes later


# --- a freshly rented host holds nothing yet ---


def test_a_freshly_rented_host_is_prepared_by_the_supervisor_before_it_serves():
    """A host the pool creates is the pool's to configure: the set is pulled and pinned by the
    supervisor itself, and the host is not routed to until the whole set is resident."""
    with pool_harness(
        [EngineSpec(id="local-1", resident={MODEL}, kind="local", workers=1)],
        rentable=[EngineSpec(id="market-1", resident=set(), workers=2)],  # empty disk
        model_set=[MODEL],
        rented=RENTED,
    ) as pool:
        pool.supervisor.fleet.open_lease(workers=3, max_hours=2, max_spend=2.00, allow_rent=True)
        settle(pool, passes=2)
        host = next(iter(rented_hosts(pool).values()))
        assert host.state == "preparing"

        deadline = time.monotonic() + 15
        while host.state != "ready" and time.monotonic() < deadline:
            settle(pool, passes=1)
            time.sleep(0.2)

        assert host.state == "ready"
        assert MODEL in pool.rentable["market-1"].fake.resident  # pulled and pinned by the pool
        assert "prepared" in event_kinds(pool)
        with pool.client() as client:
            rows = {h["host_id"]: h["state"] for h in client.get("/pool/status").json()["hosts"]}
        assert rows[host.host_id] == "ready"
