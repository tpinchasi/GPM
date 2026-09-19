from gpm_server.catalog import (
    ResolvedVariant,
    literal_variants_for_host,
    select_variant,
    variants_for_host,
)
from gpm_server.config import CatalogEntry

CATALOG = {
    "m1": CatalogEntry.model_validate(
        {
            "variants": [
                {"tag": "m1-mlx", "requires": ["apple-silicon"], "runtime_class": "apple-mlx", "enforces_schema": False},
                {"tag": "m1-fp8", "requires": ["cuda", "fp8"], "runtime_class": "cuda-fp8", "enforces_schema": True},
                {"tag": "m1", "runtime_class": "by-platform", "enforces_schema": True},
            ]
        }
    )
}


def test_only_variants_the_host_can_run_are_offered_in_preference_order():
    apple = variants_for_host(["m1"], CATALOG, {"apple-silicon"}, "ollama")
    assert [v.tag for v in apple["m1"]] == ["m1-mlx", "m1"]

    cuda = variants_for_host(["m1"], CATALOG, {"cuda", "fp8"}, "ollama")
    assert [v.tag for v in cuda["m1"]] == ["m1-fp8", "m1"]

    plain = variants_for_host(["m1"], CATALOG, set(), "ollama")
    assert [v.tag for v in plain["m1"]] == ["m1"]


def test_a_model_with_no_catalog_entry_is_passed_through_by_its_own_name():
    resolved = variants_for_host(["other"], CATALOG, {"cuda"}, "ollama")
    assert [v.tag for v in resolved["other"]] == ["other"]


def test_runtime_class_is_derived_from_the_platform_when_not_declared():
    apple = variants_for_host(["m1"], CATALOG, {"apple-silicon"}, "ollama")["m1"]
    assert apple[1].runtime_class == "apple-silicon-ollama"

    cuda = variants_for_host(["m1"], CATALOG, {"cuda"}, "ollama")["m1"]
    assert cuda[0].runtime_class == "cuda-ollama"

    unknown = variants_for_host(["m1"], CATALOG, set(), "ollama")["m1"]
    assert unknown[0].runtime_class == "unknown-ollama"


def test_declared_runtime_class_wins():
    apple = variants_for_host(["m1"], CATALOG, {"apple-silicon"}, "ollama")["m1"]
    assert apple[0].runtime_class == "apple-mlx"


def test_literal_tags_are_known_regardless_of_the_hosts_capabilities():
    literal = literal_variants_for_host(["m1"], CATALOG, set(), "ollama")
    assert set(literal) == {"m1-mlx", "m1-fp8", "m1"}


def test_a_variant_that_is_not_resident_is_not_served():
    variants = variants_for_host(["m1"], CATALOG, {"apple-silicon"}, "ollama")["m1"]
    assert select_variant(variants, frozenset({"m1"}), wants_schema=False).tag == "m1"
    assert select_variant(variants, frozenset(), wants_schema=False) is None


def test_a_schema_request_never_lands_on_a_build_said_not_to_enforce_one():
    variants = variants_for_host(["m1"], CATALOG, {"apple-silicon"}, "ollama")["m1"]
    # m1-mlx is preferred and resident, but the operator said it does not enforce schemas.
    assert select_variant(variants, frozenset({"m1-mlx", "m1"}), wants_schema=True).tag == "m1"
    # With only that build resident, the host cannot take the request at all.
    assert select_variant(variants, frozenset({"m1-mlx"}), wants_schema=True) is None


def test_a_build_nobody_has_made_a_statement_about_is_still_usable():
    # Nothing is known about a model with no catalog entry, so it is passed through as it
    # would be to a bare engine rather than refused.
    variants = variants_for_host(["other"], CATALOG, set(), "ollama")["other"]
    assert select_variant(variants, frozenset({"other"}), wants_schema=True).tag == "other"


def test_a_build_known_to_enforce_is_preferred_over_one_nobody_has_stated():
    catalog = {
        "m2": CatalogEntry.model_validate(
            {"variants": [{"tag": "m2-unstated"}, {"tag": "m2-strict", "enforces_schema": True}]}
        )
    }
    variants = variants_for_host(["m2"], catalog, set(), "ollama")["m2"]
    resident = frozenset({"m2-unstated", "m2-strict"})
    assert select_variant(variants, resident, wants_schema=False).tag == "m2-unstated"
    assert select_variant(variants, resident, wants_schema=True).tag == "m2-strict"


def test_a_pinned_runtime_class_filters_before_preference():
    variants = variants_for_host(["m1"], CATALOG, {"apple-silicon"}, "ollama")["m1"]
    resident = frozenset({"m1-mlx", "m1"})
    assert select_variant(variants, resident, wants_schema=False, runtime_class_pin="apple-silicon-ollama").tag == "m1"
    assert select_variant(variants, resident, wants_schema=False, runtime_class_pin="cuda-fp8") is None


def test_resolved_variant_is_hashable_and_frozen():
    variant = ResolvedVariant(tag="m1", runtime_class="cuda-ollama", enforces_schema=True)
    assert {variant}
