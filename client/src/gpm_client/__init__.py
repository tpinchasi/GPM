"""Client SDK for a GPM pool.

An app knows one URL, one API key, and the contract in docs/spec/app-contract.md. Waiting for
capacity is the default behaviour — an app opts *out*, not in.
"""

from .client import AsyncPoolClient, PoolClient, Ranked, Reply, Transcript, image_part
from .errors import (
    PoolAuthError,
    PoolError,
    PoolRequestError,
    PoolStreamInterrupted,
    PoolUnavailable,
)
from .retry import RetryPolicy
from .transport import AsyncPoolTransport, PoolTransport, async_pool_transport, pool_transport
from .workloads import Workload, WorkloadProvisioner, WorkloadRefused

__all__ = [
    "AsyncPoolClient",
    "AsyncPoolTransport",
    "PoolAuthError",
    "PoolClient",
    "PoolError",
    "PoolRequestError",
    "PoolStreamInterrupted",
    "PoolTransport",
    "PoolUnavailable",
    "Ranked",
    "Reply",
    "RetryPolicy",
    "Transcript",
    "Workload",
    "WorkloadProvisioner",
    "WorkloadRefused",
    "async_pool_transport",
    "image_part",
    "pool_transport",
]

CONTRACT_VERSION = "1"
