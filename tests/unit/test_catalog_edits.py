"""What the console's saves write into the catalog (D111, D143, D147): a build that names no
engine is marked with the one it has been serving only when a build for another engine joins
it, and saying what a build serves or takes changes nothing else about it."""

import textwrap

import yaml
from gpm_server.config import PoolConfig
from gpm_server.supervisor.control import _engine_of_unmarked, _with_builds, _with_variants

RENT_ONLY = textwrap.dedent("""
    pool: { name: p, model_set: [vision] }
    auth: { app_keys: [k] }
    catalog:
      vision:
        variants:
          - { tag: "org/Vision-8B" }
    rented:
      engine: vllm
      provider: fake
      offer_policy: { min_disk_gb: 10, max_all_in_hourly: 0.60 }
""").lstrip()


def load(text):
    return PoolConfig.model_validate(yaml.safe_load(text))


def test_in_a_pool_that_only_rents_an_unmarked_build_is_the_rented_engines():
    assert _engine_of_unmarked(load(RENT_ONLY)) == "vllm"


def test_saying_what_a_build_takes_does_not_pin_its_engine():
    config = load(RENT_ONLY)
    text = _with_variants(RENT_ONLY, config, "vision", [], {"org/Vision-8B": {"inputs": ["image"]}})
    (variant,) = yaml.safe_load(text)["catalog"]["vision"]["variants"]
    assert variant == {"tag": "org/Vision-8B", "inputs": ["image"]}
    load(text)  # and the file still loads


def test_a_build_sent_again_keeps_what_was_written_about_it():
    text = RENT_ONLY.replace('{ tag: "org/Vision-8B" }', '{ tag: "org/Vision-8B", engine: vllm, serves: [chat], size_gb: 16.4 }')
    config = load(text)
    written = _with_builds(text, config, "vision", {"vllm": "org/Vision-8B"})
    (variant,) = yaml.safe_load(written)["catalog"]["vision"]["variants"]
    assert variant == {"tag": "org/Vision-8B", "engine": "vllm", "serves": ["chat"], "size_gb": 16.4}
