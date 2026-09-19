"""The default retry policy. See docs/spec/app-contract.md §4."""

from __future__ import annotations

import dataclasses
import os
import random
from typing import Callable, Optional


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


@dataclasses.dataclass
class RetryPolicy:
    """Every field is overridable per call, per client, or by environment variable.

    `max_wait_s=None` waits indefinitely, which is what batch drivers usually want.
    """

    max_wait_s: Optional[float] = dataclasses.field(
        default_factory=lambda: _env_float("GPM_MAX_WAIT_S", 1800.0)
    )
    backoff_initial_s: float = 2.0
    backoff_max_s: float = 60.0
    wait_without_lease: bool = False
    on_wait: Optional[Callable[[str, float, float], None]] = None

    def backoff_s(self, attempt: int) -> float:
        """Exponential back-off with jitter, capped at `backoff_max_s`."""
        base = min(self.backoff_initial_s * (2**attempt), self.backoff_max_s)
        return base * (0.5 + random.random())
