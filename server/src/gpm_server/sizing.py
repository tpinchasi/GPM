"""What a machine must have to hold a set of models, from the sizes of their builds (D111).

A model profile knows what it holds and each build's size, so it knows the least card and disk
a machine rented for it needs. That number is the search's minimum — the operator's own minimum
still applies where it is higher — so no machine is bid on that cannot hold what it is bought for.

The memory rule is the one the machine's launcher applies before starting anything
(`gpm_agent.vllm_launch.memory_plan`): each model is given its weights with a tenth over for
loading them, plus a cache reserve, all within the launcher's share of the card. Kept in step by
a test rather than an import: the agent is a separate package that ships to rented machines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

#: Each model's weights, with a tenth over for what loading them costs.
WEIGHT_OVERHEAD = 1.10
#: What each process needs beyond its weights to batch at all.
CACHE_RESERVE_GB = 3 * 1024**3 / 1e9
#: The share of the card the launcher gives the engines; the rest is the driver's.
TOTAL_MEMORY_SHARE = 0.90

#: Disk beyond the weights: the download's own overhead, and room for the engine's image,
#: logs and caches beside them. An estimate, recorded as one (decisions.md, unverified).
DISK_OVERHEAD = 1.10
DISK_HEADROOM_GB = 10.0


@dataclass
class Needs:
    """The least card memory (per card), cards and disk a machine needs for these models."""

    card_memory_gb: float
    disk_gb: float
    #: The weights, summed over the builds whose size is known.
    weights_gb: float
    #: Models whose build has not been measured, and so are not counted.
    unknown: list[str] = field(default_factory=list)
    #: How many cards each copy of the models spans (D114): the machine needs at least this
    #: many, in whole groups of it.
    cards_per_copy: int = 1

    def as_dict(self) -> dict:
        return {
            "card_memory_gb": self.card_memory_gb,
            "disk_gb": self.disk_gb,
            "weights_gb": self.weights_gb,
            "unknown": list(self.unknown),
            "cards_per_copy": self.cards_per_copy,
        }


def needs_for(sizes: Mapping[str, Optional[float]], cards_per_copy: int = 1) -> Needs:
    """From each model's build size in GB (None where unknown), what a machine must have.

    Per card, because a machine with several cards runs a copy of the set on each card (D107),
    or on each group of `cards_per_copy` cards, every card holding that share of each model's
    weights and a cache reserve of its own (D114). Rounded up to a whole gigabyte: this becomes
    a search filter, and a filter a hair under what is needed lets through the one machine that
    cannot start. The disk holds each model once, however it is split.
    """
    known = {model: float(size) for model, size in sizes.items() if size}
    unknown = sorted(model for model, size in sizes.items() if not size)
    if not known:
        return Needs(card_memory_gb=0.0, disk_gb=0.0, weights_gb=0.0, unknown=unknown,
                     cards_per_copy=cards_per_copy)
    weights = sum(known.values())
    memory = (weights * WEIGHT_OVERHEAD / cards_per_copy + CACHE_RESERVE_GB * len(known)) / TOTAL_MEMORY_SHARE
    disk = weights * DISK_OVERHEAD + DISK_HEADROOM_GB
    return Needs(
        card_memory_gb=float(math.ceil(memory)),
        disk_gb=float(math.ceil(disk)),
        weights_gb=round(weights, 3),
        unknown=unknown,
        cards_per_copy=cards_per_copy,
    )
