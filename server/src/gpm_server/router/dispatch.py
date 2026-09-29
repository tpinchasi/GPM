"""Dispatch: eligibility, strict priority tiers, and the first-come-first-served queue.

docs/spec/hosts-routing-capacity.md §5. Dispatch is a pull: a request goes to an idle worker on
the highest-priority eligible tier; if none is idle it queues, and the next worker to free up
takes the oldest request it is eligible for. Priority and queueing are one mechanism.

**A request's workload comes first** (docs/spec/workloads.md §4, D115): it may only reach that
workload's hosts. While a workload has no ready host of its own it may borrow the shared
workload's — never while a shared request is queued, and never more than its share of the shared
ready workers at once. The moment one of its own hosts is ready, borrowing stops.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from typing import Callable, Optional

from ..catalog import ResolvedVariant, select_variant
from ..models import Host, HostState, Worker, WorkerState


@dataclasses.dataclass(frozen=True)
class Need:
    model: str
    wants_schema: bool = False
    runtime_class_pin: Optional[str] = None
    #: Engines that can serve the path this request arrived on (D93). Empty means "any" — which
    #: is every pool running one engine, and is what this meant before pools could run two.
    #: A request on an engine's own API must never reach a host running a different engine: the
    #: path is not there, and the host would answer 404 to a request the pool called eligible.
    engines: frozenset[str] = frozenset()
    #: The workload the request is for (D115); None is the shared workload.
    workload: Optional[str] = None
    #: Whether it may be served on a shared host while its workload has none ready, and the
    #: share of the shared workload's ready workers all borrowers together may hold.
    may_borrow: bool = False
    borrow_share: float = 0.0


@dataclasses.dataclass(frozen=True)
class Assignment:
    host: Host
    worker: Worker
    variant: ResolvedVariant
    #: Served on a host the shared workload lent (D115).
    borrowed: bool = False
    #: How many answers the host was serving once this one was given it — the concurrency its
    #: latency is measured at, read under the dispatch lock where it cannot move (D115).
    concurrency: int = 1


class NoReadyHost(Exception):
    """Nothing in the pool is answering."""


class NoEligibleHost(Exception):
    """Hosts are ready, but none can serve this particular request."""


class QueueTimeout(Exception):
    """Eligible workers exist but stayed busy past the queue limit."""


class Dispatcher:
    def __init__(self, hosts: list[Host]):
        self.hosts = hosts
        self._condition = asyncio.Condition()
        #: Requests waiting for a worker, by request id: what each needs and which hosts it has
        #: already failed on. A borrower never takes a shared worker a queued shared request could
        #: use — the shared workload's own requests come first (D115). An entry is removed when
        #: its request **leaves** `acquire`, never merely when it is woken: a woken waiter has not
        #: yet re-taken the lock, and a request arriving meanwhile must still see it queued.
        self._waiting: dict[str, tuple[Need, frozenset[str]]] = {}

    def variant_for(self, host: Host, need: Need) -> Optional[ResolvedVariant]:
        variants = host.variants.get(need.model)
        if variants is None:
            literal = host.literal_variants.get(need.model)
            if literal is None:
                return None
            variants = (literal,)
        return select_variant(
            variants,
            host.servable,
            wants_schema=need.wants_schema,
            runtime_class_pin=need.runtime_class_pin,
        )

    def eligible(self, need: Need, exclude: frozenset[str] = frozenset()) -> list[tuple[Host, ResolvedVariant]]:
        own: list[tuple[Host, ResolvedVariant]] = []
        shared: list[tuple[Host, ResolvedVariant]] = []
        for host in self.hosts:
            if host.state is not HostState.READY or host.host_id in exclude:
                continue
            if need.engines and host.engine not in need.engines:
                continue
            if host.workload != need.workload and not (need.workload is not None and host.workload is None):
                continue  # another workload's host: never
            variant = self.variant_for(host, need)
            if variant is None:
                continue
            (own if host.workload == need.workload else shared).append((host, variant))
        if need.workload is not None and need.may_borrow and not self.has_ready_host(need.workload):
            return own + shared
        return own

    def has_ready_host(self, workload: Optional[str]) -> bool:
        return any(h.workload == workload and h.state is HostState.READY for h in self.hosts)

    def lent_workers(self) -> int:
        """Shared workers serving a borrower right now."""
        return sum(
            1 for h in self.hosts if h.workload is None
            for w in h.workers if w.borrowed and w.state is WorkerState.BUSY
        )

    def borrow_limit(self, share: float) -> int:
        """How many shared workers borrowers may hold at once: the share of the shared ready
        workers, rounded down — at least one, where there is any."""
        ready = sum(h.total_workers for h in self.hosts if h.workload is None and h.state is HostState.READY)
        return max(1, math.floor(share * ready)) if ready and share > 0 else 0

    def _shared_request_wants(self, host: Host) -> bool:
        """Is a queued shared request one this host could serve?"""
        for waiting, exclude in self._waiting.values():
            if waiting.workload is not None or host.host_id in exclude:
                continue
            if waiting.engines and host.engine not in waiting.engines:
                continue
            if self.variant_for(host, waiting) is not None:
                return True
        return False

    def _may_take(self, host: Host, need: Need) -> bool:
        if host.workload == need.workload:
            return True
        # Borrowing: after the shared workload's own requests, and within the share — which is
        # for every borrower together, not each workload's own (D115).
        if self._shared_request_wants(host):
            return False
        return self.lent_workers() < self.borrow_limit(need.borrow_share)

    def knows_model(self, name: str, workload: Optional[str] = None) -> bool:
        """Does any host of this workload know the name? A workload's hosts say nothing to the
        shared workload's requests: an app key must not learn another workload's model exists."""
        return any(
            name in host.variants or name in host.literal_variants
            for host in self.hosts if host.workload == workload
        )

    async def acquire(
        self,
        need: Need,
        *,
        request_id: str,
        deadline: float,
        exclude: frozenset[str] = frozenset(),
    ) -> Assignment:
        async with self._condition:
            try:
                while True:
                    candidates = self.eligible(need, exclude)
                    if not candidates:
                        # Judged within the request's own workload: a workload with no ready host
                        # is not "ready but ineligible" because the shared hosts are up, and the
                        # other way round.
                        if self.has_ready_host(need.workload):
                            raise NoEligibleHost()
                        raise NoReadyHost()

                    assignment = self._take_idle(candidates, request_id, need)
                    if assignment is not None:
                        return assignment

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise QueueTimeout()
                    self._waiting[request_id] = (need, exclude)
                    try:
                        # Waiters re-acquire the condition lock in the order they began waiting,
                        # so the oldest eligible request takes the freed worker.
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise QueueTimeout() from None
            finally:
                # Leaving — served, refused, timed out or cancelled with its client. A shared
                # request leaving may be the one a borrower was yielding to: wake them to look.
                left = self._waiting.pop(request_id, None)
                if left is not None and left[0].workload is None:
                    self._condition.notify_all()

    def _take_idle(
        self, candidates: list[tuple[Host, ResolvedVariant]], request_id: str, need: Optional[Need] = None
    ) -> Optional[Assignment]:
        need = need or Need(model="")
        for priority in sorted({host.priority for host, _ in candidates}):
            tier = [
                (h, v) for h, v in candidates
                if h.priority == priority and h.idle_worker() is not None and self._may_take(h, need)
            ]
            if not tier:
                continue
            host, variant = min(tier, key=lambda pair: (pair[0].load, pair[0].host_id))
            worker = host.idle_worker()
            assert worker is not None
            worker.state = WorkerState.BUSY
            worker.request_id = request_id
            worker.borrowed = host.workload != need.workload
            return Assignment(
                host=host, worker=worker, variant=variant, borrowed=worker.borrowed, concurrency=host.busy,
            )
        return None

    async def release(self, assignment: Assignment) -> None:
        async with self._condition:
            worker = assignment.worker
            worker.state = WorkerState.IDLE
            worker.request_id = None
            worker.borrowed = False
            worker.served += 1
            self._condition.notify_all()

    async def wake(self) -> None:
        """Called when host state changes, so queued requests re-check eligibility."""
        async with self._condition:
            self._condition.notify_all()

    async def update(self, mutate: Callable[[], None]) -> None:
        """Change the host list under the dispatch lock, then wake anything queued.

        The published table can add, remove or re-state a host at any moment; doing it here
        means a request is never choosing between hosts while the list is half-rewritten.
        """
        async with self._condition:
            mutate()
            self._condition.notify_all()
