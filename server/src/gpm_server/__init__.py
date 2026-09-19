"""GPM — GPU Hosts Pool Management.

Phase 1 (Route): the router over statically configured hosts. The supervisor, leases and the
operator console arrive in later phases; see docs/roadmap.md.
"""

from .contract import CONTRACT_VERSION

__all__ = ["CONTRACT_VERSION"]
