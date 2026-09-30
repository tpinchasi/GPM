"""Dispatch across workloads (D115, workloads.md §4): a workload's request reaches only its own
hosts; while it has none ready it may borrow the shared workload's — behind any queued shared
request, and within its share — and the moment one of its own is ready, borrowing stops."""

import asyncio
import time

import pytest
from gpm_server.models import HostState
from gpm_server.router.dispatch import Dispatcher, Need, NoReadyHost, QueueTimeout
from test_dispatch import make_host


def workload_host(host_id, workload, workers=1, state=HostState.READY):
    host = make_host(host_id, 20, workers=workers, state=state)
    host.workload = workload
    return host


def need(workload=None, may_borrow=False, share=0.25):
    return Need("m1", workload=workload, may_borrow=may_borrow, borrow_share=share)


def soon(seconds=0.2):
    return time.monotonic() + seconds


async def test_a_workloads_request_reaches_only_its_own_hosts():
    shared = make_host("laptop", 0)
    mine = workload_host("rented-a", "research")
    theirs = workload_host("rented-b", "other")
    dispatcher = Dispatcher([shared, theirs, mine])
    got = await dispatcher.acquire(need("research"), request_id="r1", deadline=soon())
    assert got.host.host_id == "rented-a" and not got.borrowed


async def test_a_shared_request_never_reaches_a_workloads_host():
    busy_shared = make_host("laptop", 0, workers=1)
    dispatcher = Dispatcher([busy_shared, workload_host("rented-a", "research", workers=4)])
    await dispatcher.acquire(need(), request_id="r1", deadline=soon())
    with pytest.raises(QueueTimeout):
        await dispatcher.acquire(need(), request_id="r2", deadline=soon(0.05))


async def test_another_workloads_hosts_are_never_borrowed():
    dispatcher = Dispatcher([workload_host("rented-b", "other", workers=4)])
    with pytest.raises(NoReadyHost):
        await dispatcher.acquire(need("research", may_borrow=True), request_id="r1", deadline=soon())


async def test_without_borrowing_a_preparing_workload_has_no_ready_host():
    dispatcher = Dispatcher([make_host("laptop", 0, workers=4), workload_host("rented-a", "research", state=HostState.PREPARING)])
    with pytest.raises(NoReadyHost):
        await dispatcher.acquire(need("research"), request_id="r1", deadline=soon())


async def test_a_preparing_workload_borrows_within_its_share():
    laptop = make_host("laptop", 0, workers=8)
    dispatcher = Dispatcher([laptop, workload_host("rented-a", "research", state=HostState.PREPARING)])
    first = await dispatcher.acquire(need("research", may_borrow=True, share=0.25), request_id="r1", deadline=soon())
    second = await dispatcher.acquire(need("research", may_borrow=True, share=0.25), request_id="r2", deadline=soon())
    assert first.borrowed and second.borrowed and first.host.host_id == "laptop"
    assert dispatcher.lent_workers() == 2 and dispatcher.borrow_limit(0.25) == 2
    with pytest.raises(QueueTimeout):
        await dispatcher.acquire(need("research", may_borrow=True, share=0.25), request_id="r3", deadline=soon(0.05))
    # The shared workload's own requests still get the rest.
    mine = await dispatcher.acquire(need(), request_id="s1", deadline=soon())
    assert mine.host.host_id == "laptop" and not mine.borrowed
    await dispatcher.release(first)
    assert dispatcher.lent_workers() == 1 and not first.worker.borrowed


async def test_a_share_that_rounds_to_nothing_still_lends_one():
    dispatcher = Dispatcher([make_host("laptop", 0, workers=2)])
    assert dispatcher.borrow_limit(0.25) == 1
    assert dispatcher.borrow_limit(0.0) == 0


async def test_borrowing_stops_once_a_host_of_its_own_is_ready():
    laptop = make_host("laptop", 0, workers=8)
    own = workload_host("rented-a", "research", state=HostState.PREPARING)
    dispatcher = Dispatcher([laptop, own])
    assert (await dispatcher.acquire(need("research", may_borrow=True), request_id="r1", deadline=soon())).borrowed
    own.state = HostState.READY
    got = await dispatcher.acquire(need("research", may_borrow=True), request_id="r2", deadline=soon())
    assert got.host.host_id == "rented-a" and not got.borrowed
    # Its own host busy now: it queues on it, and never goes back to the shared ones.
    with pytest.raises(QueueTimeout):
        await dispatcher.acquire(need("research", may_borrow=True), request_id="r3", deadline=soon(0.05))


async def test_a_queued_shared_request_is_served_before_a_borrower():
    """A shared request waits for the laptop's one worker; a borrower waiting too — even one that
    began waiting first — must let it have the worker when it frees."""
    laptop = make_host("laptop", 0, workers=1)
    dispatcher = Dispatcher([laptop, workload_host("rented-a", "research", state=HostState.PREPARING)])
    holding = await dispatcher.acquire(need(), request_id="s0", deadline=soon())
    borrower = asyncio.create_task(
        dispatcher.acquire(need("research", may_borrow=True, share=1.0), request_id="b1", deadline=soon(1.0))
    )
    await asyncio.sleep(0.01)
    shared = asyncio.create_task(dispatcher.acquire(need(), request_id="s1", deadline=soon(1.0)))
    await asyncio.sleep(0.01)
    await dispatcher.release(holding)
    first = await shared
    assert first.worker.request_id == "s1" and not borrower.done()
    await dispatcher.release(first)
    got = await borrower
    assert got.borrowed and got.worker.request_id == "b1"


async def test_a_queued_shared_request_for_another_model_does_not_block_a_borrower():
    """The borrower yields only to a shared request that host could serve."""
    laptop = make_host("laptop", 0, workers=2)
    other = make_host("other-box", 0, workers=1)
    other.variants = {"m2": other.variants["m1"]}
    dispatcher = Dispatcher([laptop, other, workload_host("rented-a", "research", state=HostState.PREPARING)])
    holding = await dispatcher.acquire(Need("m2"), request_id="s0", deadline=soon())
    waiting_m2 = asyncio.create_task(dispatcher.acquire(Need("m2"), request_id="s1", deadline=soon(1.0)))
    await asyncio.sleep(0.01)
    got = await dispatcher.acquire(need("research", may_borrow=True, share=1.0), request_id="b1", deadline=soon())
    assert got.borrowed and got.host.host_id == "laptop"
    await dispatcher.release(holding)
    await waiting_m2


async def test_a_shared_request_that_gives_up_wakes_the_borrower_it_held_back():
    laptop = make_host("laptop", 0, workers=2)
    dispatcher = Dispatcher([laptop, workload_host("rented-a", "research", state=HostState.PREPARING)])
    a = await dispatcher.acquire(need(), request_id="s0", deadline=soon())
    b = await dispatcher.acquire(need(), request_id="s1", deadline=soon())
    impatient = asyncio.create_task(dispatcher.acquire(need(), request_id="s2", deadline=soon(0.1)))
    await asyncio.sleep(0.01)
    borrower = asyncio.create_task(
        dispatcher.acquire(need("research", may_borrow=True, share=1.0), request_id="b1", deadline=soon(2.0))
    )
    await asyncio.sleep(0.01)
    await dispatcher.release(a)  # frees a worker while the shared request is still queued
    first = await impatient
    assert first.worker.request_id == "s2"
    await dispatcher.release(b)
    got = await asyncio.wait_for(borrower, 1.0)
    assert got.borrowed


async def test_concurrency_is_read_when_the_worker_is_taken():
    laptop = make_host("laptop", 0, workers=3)
    dispatcher = Dispatcher([laptop])
    first = await dispatcher.acquire(need(), request_id="r1", deadline=soon())
    second = await dispatcher.acquire(need(), request_id="r2", deadline=soon())
    assert (first.concurrency, second.concurrency) == (1, 2)


# --- several models on one host (D118) ---


def two_model_host(host_id, workload, workers, models=("chat", "embed"), state=HostState.READY):
    from test_dispatch import PLAIN

    host = workload_host(host_id, workload, workers=workers, state=state)
    host.variants = {m: (PLAIN,) for m in models}
    host.resident = frozenset(PLAIN.tag for _ in models)
    return host


def for_model(model, cap=None, workload="research", may_borrow=False):
    return Need(model, workload=workload, cap=cap, may_borrow=may_borrow, borrow_share=0.5)


async def test_a_model_past_its_share_of_a_host_waits_and_the_other_still_gets_its_own():
    """Caps of 6 chat and 2 embed on an 8-worker host: chat never takes embed's two."""
    host = two_model_host("rented-a", "research", workers=8)
    dispatcher = Dispatcher([host])
    chats = [await dispatcher.acquire(for_model("chat", cap=6), request_id=f"c{i}", deadline=soon()) for i in range(6)]
    assert chats[-1].model_concurrency == 6 and chats[-1].concurrency == 6
    with pytest.raises(QueueTimeout):
        await dispatcher.acquire(for_model("chat", cap=6), request_id="c7", deadline=soon(0.05))
    embeds = [await dispatcher.acquire(for_model("embed", cap=2), request_id=f"e{i}", deadline=soon()) for i in range(2)]
    assert embeds[-1].model_concurrency == 2 and embeds[-1].concurrency == 8


async def test_a_capped_request_is_served_the_moment_its_model_frees_a_worker():
    host = two_model_host("rented-a", "research", workers=3)
    dispatcher = Dispatcher([host])
    first = await dispatcher.acquire(for_model("chat", cap=1), request_id="c1", deadline=soon())
    waiting = asyncio.create_task(dispatcher.acquire(for_model("chat", cap=1), request_id="c2", deadline=soon(2)))
    await asyncio.sleep(0.05)
    assert not waiting.done(), "one chat at a time on this host, though two workers are idle"
    await dispatcher.release(first)
    second = await asyncio.wait_for(waiting, 1)
    assert second.worker.model == "chat" and dispatcher.busy_with(host, "chat") == 1


async def test_no_cap_on_a_one_model_workload_or_the_shared_workload():
    host = two_model_host("rented-a", "research", workers=3)
    dispatcher = Dispatcher([host])
    for i in range(3):
        await dispatcher.acquire(for_model("chat"), request_id=f"c{i}", deadline=soon())


async def test_a_model_whose_hosts_are_still_coming_up_is_preparing_not_ineligible():
    """Apart: the chat group serves, the embed group prepares. An embed request is waiting on its
    own hosts — it borrows where it may, and is `no ready host` where it may not."""
    chat = two_model_host("rented-chat", "research", workers=4, models=("chat",))
    embed = two_model_host("rented-embed", "research", workers=4, models=("embed",), state=HostState.PREPARING)
    shared = make_host("laptop", 0, workers=4)
    shared.variants = {"embed": shared.variants["m1"]}
    dispatcher = Dispatcher([chat, embed, shared])
    with pytest.raises(NoReadyHost):
        await dispatcher.acquire(for_model("embed"), request_id="e1", deadline=soon(0.05))
    lent = await dispatcher.acquire(for_model("embed", may_borrow=True), request_id="e2", deadline=soon())
    assert lent.borrowed and lent.host.host_id == "laptop"
    own = await dispatcher.acquire(for_model("chat", may_borrow=True), request_id="c1", deadline=soon())
    assert not own.borrowed and own.host.host_id == "rented-chat", "a model with its own ready host never borrows"


async def test_a_ready_host_that_cannot_meet_the_request_is_ineligible_not_unready():
    """Found in review: a schema the workload's build cannot enforce read as `workload_preparing`."""
    from gpm_server.router.dispatch import NoEligibleHost

    host = workload_host("rented-a", "research", workers=2)
    dispatcher = Dispatcher([host])
    with pytest.raises(NoEligibleHost):
        await dispatcher.acquire(Need("m1", workload="research", wants_schema=True), request_id="r1", deadline=soon())


async def test_under_a_storm_of_requests_no_host_ever_exceeds_a_models_share():
    """Stress: 400 requests of two models, three split hosts (6 chat + 2 embed each), random hold
    times, clients giving up. At every moment each host holds at most its share of each model and
    at most its workers; every request is served or times out; nothing is left busy."""
    import random

    rng = random.Random(118)
    hosts = [two_model_host(f"rented-{i}", "research", workers=8) for i in range(3)]
    dispatcher = Dispatcher(hosts)
    caps = {"chat": 6, "embed": 2}
    worst = {"chat": 0, "embed": 0, "total": 0}
    outcomes = {"served": 0, "timed_out": 0}

    def check():
        for host in hosts:
            for model, cap in caps.items():
                held = dispatcher.busy_with(host, model)
                assert held <= cap, f"{host.host_id} holds {held} {model}, over its share {cap}"
                worst[model] = max(worst[model], held)
            assert host.busy <= len(host.workers)
            worst["total"] = max(worst["total"], host.busy)

    async def one(i):
        model = "chat" if rng.random() < 0.75 else "embed"
        await asyncio.sleep(rng.random() * 0.2)
        try:
            got = await dispatcher.acquire(for_model(model, cap=caps[model]), request_id=f"r{i}",
                                           deadline=time.monotonic() + rng.choice([0.05, 0.5, 2.0]))
        except QueueTimeout:
            outcomes["timed_out"] += 1
            return
        check()
        await asyncio.sleep(rng.random() * 0.02)
        check()
        await dispatcher.release(got)
        outcomes["served"] += 1

    await asyncio.gather(*(one(i) for i in range(400)))
    assert outcomes["served"] + outcomes["timed_out"] == 400 and outcomes["served"] > 300
    assert worst["chat"] == 6 and worst["embed"] == 2, "the shares were reached, and never passed"
    assert all(h.busy == 0 for h in hosts), "every worker handed back"
