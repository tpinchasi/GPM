"""Model resolution: logical name → the build this host serves.

docs/spec/hosts-routing-capacity.md §4. Only names listed in the catalog are ever resolved;
anything else is passed through exactly as requested, never guessed and never pulled.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping, Optional, Sequence

from .config import CatalogEntry, Variant


@dataclasses.dataclass(frozen=True)
class ResolvedVariant:
    tag: str
    runtime_class: str
    enforces_schema: Optional[bool]


# The platform half of a derived runtime class. Runtime class is always derived from the
# variant and the host's platform, never typed in by hand (spec §4.3).
_PLATFORM_ORDER = ("apple-silicon", "cuda", "rocm")


def derive_runtime_class(variant: Variant, capabilities: Iterable[str], engine_name: str) -> str:
    if variant.runtime_class and variant.runtime_class != "by-platform":
        return variant.runtime_class
    caps = set(capabilities)
    platform = next((c for c in _PLATFORM_ORDER if c in caps), "unknown")
    return f"{platform}-{engine_name}"


def variants_for_host(
    model_set: Sequence[str],
    catalog: Mapping[str, CatalogEntry],
    capabilities: Iterable[str],
    engine_name: str,
) -> dict[str, tuple[ResolvedVariant, ...]]:
    """Per logical model, the variants this host could serve, in catalog preference order.

    A model with no catalog entry has exactly one variant: its own name, passed through.
    """
    caps = set(capabilities)
    resolved: dict[str, tuple[ResolvedVariant, ...]] = {}
    for name in model_set:
        entry = catalog.get(name)
        declared = entry.variants if entry else [Variant(tag=name)]
        resolved[name] = tuple(
            ResolvedVariant(
                tag=variant.tag,
                runtime_class=derive_runtime_class(variant, caps, engine_name),
                enforces_schema=variant.enforces_schema,
            )
            for variant in declared
            if set(variant.requires) <= caps
        )
    return resolved


def literal_variants_for_host(
    model_set: Sequence[str],
    catalog: Mapping[str, CatalogEntry],
    capabilities: Iterable[str],
    engine_name: str,
) -> dict[str, ResolvedVariant]:
    """Every tag this pool knows by name, for requests that name a build explicitly.

    An explicit variant tag is taken literally and is eligible only where that tag is
    resident — residency, not the `requires` list, is the test (spec §4.1).
    """
    caps = set(capabilities)
    literal: dict[str, ResolvedVariant] = {}
    for name in model_set:
        entry = catalog.get(name)
        declared = entry.variants if entry else [Variant(tag=name)]
        for variant in declared:
            literal[variant.tag] = ResolvedVariant(
                tag=variant.tag,
                runtime_class=derive_runtime_class(variant, caps, engine_name),
                enforces_schema=variant.enforces_schema,
            )
    return literal


def select_variant(
    variants: Sequence[ResolvedVariant],
    resident: frozenset[str],
    *,
    wants_schema: bool,
    runtime_class_pin: Optional[str] = None,
) -> Optional[ResolvedVariant]:
    """The build this host would serve for this request, or None if it cannot serve it.

    A request carrying a structured-output schema is never routed to a build the operator
    said does not enforce one: silently routing it to a back-end that accepts the schema and
    ignores it turns guaranteed-valid output into best-effort output. A build nobody has made
    a statement about is still usable, but only after every build known to enforce.
    """
    usable = [
        variant
        for variant in variants
        if variant.tag in resident
        and (runtime_class_pin is None or variant.runtime_class == runtime_class_pin)
    ]
    if not wants_schema:
        return usable[0] if usable else None
    known = [v for v in usable if v.enforces_schema is True]
    unstated = [v for v in usable if v.enforces_schema is None]
    ranked = known + unstated
    return ranked[0] if ranked else None
