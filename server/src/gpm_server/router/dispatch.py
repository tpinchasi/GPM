"""Dispatch: eligibility, strict priority tiers, and the first-come-first-served queue.

docs/spec/hosts-routing-capacity.md §5. Dispatch is a pull: a request goes to an idle worker on
the highest-priority eligible tier; if none is idle it queues, and the next worker to free up
takes the oldest request it is eligible for. Priority and queueing are one mechanism.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Callable, Optional

from ..catalog import ResolvedVariant, select_variant
from ..models import Host, HostState, Worker, WorkerState


@dataclasses.dataclass(frozen=True)
class Need:
    model: str
    wants_schema: bool = False
    runtime_class_pin: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Assignment:
    host: Host
    worker: Worker
    variant: ResolvedVariant


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

    def variant_for(self, host: Host, need: Need) -> Optional[ResolvedVariant]:
        variants = host.variants.get(need.model)
        if variants is None:
            literal = host.literal_variants.get(need.model)
            if literal is None:
                return None
            variants = (literal,)
        return select_variant(
            variants,
            host.resident,
            wants_schema=need.wants_schema,
            runtime_class_pin=need.runtime_class_pin,
        )

    def eligible(self, need: Need, exclude: frozenset[str] = frozenset()) -> list[tuple[Host, ResolvedVariant]]:
        found = []
        for host in self.hosts:
            if host.state is not HostState.READY or host.host_id in exclude:
                continue
            variant = self.variant_for(host, need)
            if variant is not None:
                found.append((host, variant))
        return found

    def knows_model(self, name: str) -> bool:
        return any(name in host.variants or name in host.literal_variants for host in self.hosts)

    async def acquire(
        self,
        need: Need,
        *,
        request_id: str,
        deadline: float,
        exclude: frozenset[str] = frozenset(),
    ) -> Assignment:
        async with self._condition:
            while True:
                candidates = self.eligible(need, exclude)
                if not candidates:
                    if any(host.state is HostState.READY for host in self.hosts):
                        raise NoEligibleHost()
                    raise NoReadyHost()

                assignment = self._take_idle(candidates, request_id)
                if assignment is not None:
                    return assignment

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueueTimeout()
                try:
                    # Waiters re-acquire the condition lock in the order they began waiting,
                    # so the oldest eligible request takes the freed worker.
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise QueueTimeout() from None

    def _take_idle(self, candidates: list[tuple[Host, ResolvedVariant]], request_id: str) -> Optional[Assignment]:
        for priority in sorted({host.priority for host, _ in candidates}):
            tier = [(h, v) for h, v in candidates if h.priority == priority and h.idle_worker() is not None]
            if not tier:
                continue
            host, variant = min(tier, key=lambda pair: (pair[0].load, pair[0].host_id))
            worker = host.idle_worker()
            assert worker is not None
            worker.state = WorkerState.BUSY
            worker.request_id = request_id
            return Assignment(host=host, worker=worker, variant=variant)
        return None

    async def release(self, assignment: Assignment) -> None:
        async with self._condition:
            worker = assignment.worker
            worker.state = WorkerState.IDLE
            worker.request_id = None
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
