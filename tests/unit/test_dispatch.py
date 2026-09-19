import asyncio
import time

import pytest
from gpm_server.catalog import ResolvedVariant
from gpm_server.models import Host, HostState, Worker
from gpm_server.router.dispatch import (
    Dispatcher,
    Need,
    NoEligibleHost,
    NoReadyHost,
    QueueTimeout,
)

PLAIN = ResolvedVariant(tag="m1", runtime_class="cuda-ollama", enforces_schema=False)
SCHEMA = ResolvedVariant(tag="m1-strict", runtime_class="cuda-strict", enforces_schema=True)


def make_host(host_id, priority, workers=1, variants=(PLAIN,), resident=("m1",), state=HostState.READY):
    return Host(
        host_id=host_id,
        kind="local" if priority == 0 else "fixed-remote",
        priority=priority,
        capabilities=frozenset(),
        client=None,  # dispatch never dials; the request path does
        workers=[Worker(worker_id=f"{host_id}/w{i}") for i in range(workers)],
        variants={"m1": tuple(variants)},
        literal_variants={v.tag: v for v in variants},
        state=state,
        resident=frozenset(resident),
    )


async def test_the_highest_priority_tier_wins():
    local = make_host("local", 0)
    remote = make_host("remote", 10)
    dispatcher = Dispatcher([remote, local])

    assignment = await dispatcher.acquire(Need("m1"), request_id="r1", deadline=time.monotonic() + 1)
    assert assignment.host.host_id == "local"


async def test_a_lower_tier_is_used_only_when_the_one_above_is_busy():
    local = make_host("local", 0, workers=1)
    remote = make_host("remote", 10, workers=1)
    dispatcher = Dispatcher([local, remote])

    first = await dispatcher.acquire(Need("m1"), request_id="r1", deadline=time.monotonic() + 1)
    second = await dispatcher.acquire(Need("m1"), request_id="r2", deadline=time.monotonic() + 1)
    assert first.host.host_id == "local"
    assert second.host.host_id == "remote"


async def test_within_a_tier_the_least_loaded_host_wins():
    busy = make_host("busy", 0, workers=2)
    idle = make_host("idle", 0, workers=2)
    dispatcher = Dispatcher([busy, idle])
    await dispatcher.acquire(Need("m1"), request_id="r0", deadline=time.monotonic() + 1)  # loads "busy"

    assignment = await dispatcher.acquire(Need("m1"), request_id="r1", deadline=time.monotonic() + 1)
    assert assignment.host.host_id == "idle"


async def test_a_request_queues_and_the_freed_worker_takes_the_oldest_one():
    host = make_host("only", 0, workers=1)
    dispatcher = Dispatcher([host])
    held = await dispatcher.acquire(Need("m1"), request_id="r0", deadline=time.monotonic() + 1)

    order: list[str] = []

    async def queued(name: str) -> None:
        assignment = await dispatcher.acquire(Need("m1"), request_id=name, deadline=time.monotonic() + 2)
        order.append(name)
        await dispatcher.release(assignment)

    first = asyncio.create_task(queued("first"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(queued("second"))
    await asyncio.sleep(0.05)

    await dispatcher.release(held)
    await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
    assert order == ["first", "second"]


async def test_queue_timeout_when_every_eligible_worker_stays_busy():
    host = make_host("only", 0, workers=1)
    dispatcher = Dispatcher([host])
    await dispatcher.acquire(Need("m1"), request_id="r0", deadline=time.monotonic() + 1)

    with pytest.raises(QueueTimeout):
        await dispatcher.acquire(Need("m1"), request_id="r1", deadline=time.monotonic() + 0.1)


async def test_nothing_ready_is_told_apart_from_nothing_eligible():
    down = make_host("down", 0, state=HostState.UNREACHABLE)
    dispatcher = Dispatcher([down])
    with pytest.raises(NoReadyHost):
        await dispatcher.acquire(Need("m1"), request_id="r1", deadline=time.monotonic() + 1)

    ready = make_host("ready", 0, variants=(PLAIN,), resident=("m1",))
    dispatcher = Dispatcher([ready])
    with pytest.raises(NoEligibleHost):
        await dispatcher.acquire(Need("m1", wants_schema=True), request_id="r1", deadline=time.monotonic() + 1)


async def test_a_host_can_be_excluded_after_it_failed():
    first = make_host("first", 0)
    second = make_host("second", 0)
    dispatcher = Dispatcher([first, second])

    assignment = await dispatcher.acquire(
        Need("m1"), request_id="r1", deadline=time.monotonic() + 1, exclude=frozenset({"first"})
    )
    assert assignment.host.host_id == "second"


async def test_releasing_returns_the_worker_and_counts_it():
    host = make_host("only", 0, workers=1)
    dispatcher = Dispatcher([host])
    assignment = await dispatcher.acquire(Need("m1"), request_id="r1", deadline=time.monotonic() + 1)
    assert host.busy == 1

    await dispatcher.release(assignment)
    assert host.busy == 0
    assert assignment.worker.served == 1


async def test_a_schema_request_picks_the_enforcing_build():
    host = make_host("only", 0, workers=2, variants=(PLAIN, SCHEMA), resident=("m1", "m1-strict"))
    dispatcher = Dispatcher([host])

    plain = await dispatcher.acquire(Need("m1"), request_id="r1", deadline=time.monotonic() + 1)
    assert plain.variant.tag == "m1"

    strict = await dispatcher.acquire(Need("m1", wants_schema=True), request_id="r2", deadline=time.monotonic() + 1)
    assert strict.variant.tag == "m1-strict"


def test_knows_model_covers_logical_names_and_explicit_tags():
    host = make_host("only", 0, variants=(PLAIN, SCHEMA))
    dispatcher = Dispatcher([host])
    assert dispatcher.knows_model("m1")
    assert dispatcher.knows_model("m1-strict")
    assert not dispatcher.knows_model("something-else")
