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
