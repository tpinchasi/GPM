"""Sorting a model hub's answer into the builds an engine can use (D100).

Against a search recorded from the real hub (`gemma-4-26b`, sorted by downloads, as the pool
asks): fifty repositories — the original, quantisations for every generation of card, GGUF and
MLX files for other engines, and fine-tunes under similar names. The operator is offered the
first two kinds and told how many of the rest were left out.
"""

import json
from pathlib import Path

import pytest
from gpm_server import hubbuilds
from gpm_server.engines.vllm import VllmEngine

RECORDED = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "directory" / "hub_search_gemma-4-26b.json").read_text()
)
ORIGINAL = "google/gemma-4-26B-A4B-it"


@pytest.fixture
def found():
    return hubbuilds.sort_builds("gemma4:26b", ["gemma-4-26b"], RECORDED)


def by_repo(found):
    return {b.repo: b for b in found.builds}


# --- what is offered ---


def test_the_original_is_the_model_its_builds_name(found):
    assert found.original == ORIGINAL
    assert found.builds[0].repo == ORIGINAL and found.builds[0].relation == "original"


def test_the_well_known_builds_are_offered_with_their_precision(found):
    builds = by_repo(found)
    assert builds["nvidia/Gemma-4-26B-A4B-NVFP4"].precision == "NVFP4"
    assert builds["RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"].precision == "FP8"
    assert builds["cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"].precision == "INT4"
    assert builds[ORIGINAL].precision == "BF16"
    assert builds["nvidia/Gemma-4-26B-A4B-NVFP4"].of == ORIGINAL


def test_the_same_publishers_other_originals_are_offered_each_with_its_builds(found):
    """Google's quantisation-aware release is the same model, and has builds of its own."""
    qat = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
    assert found.originals[0] == ORIGINAL and qat in found.originals
    assert by_repo(found)["cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"].of == qat
    assert all(o.startswith("google/") for o in found.originals)


def test_a_draft_model_beside_the_real_one_is_not_an_original(found):
    """Same publisher, same name, a different family: a small assistant model, not this one."""
    assert "google/gemma-4-26B-A4B-it-assistant" not in found.originals


def test_the_original_must_carry_the_name_searched_for():
    """Found live: a search for qwen3-30b also finds the coder model's builds, which outnumber
    the model's own — and the most-linked rule alone chose the coder."""
    def quant(repo, of, downloads):
        return {"id": repo, "downloads": downloads, "tags": [f"base_model:quantized:{of}"],
                "safetensors": {"parameters": {"I32": 1}}, "config": {"model_type": "qwen3_moe"}}
    coder, plain = "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3-30B-A3B"
    entries = [quant(f"x{i}/coder-awq", coder, 100) for i in range(5)] + [quant("y/Qwen3-30B-A3B-AWQ", plain, 50)]
    assert hubbuilds.choose_originals(entries, ["qwen3-30b"])[0] == plain


def test_every_offered_build_has_its_precision_read(found):
    """Including a mixed build, and one published with its settings cut short."""
    assert [b.repo for b in found.builds if b.precision is None] == []


def test_the_cards_follow_from_the_precision(found):
    nvfp4 = by_repo(found)["nvidia/Gemma-4-26B-A4B-NVFP4"]
    assert (nvfp4.runs_on, nvfp4.full_speed_on) == ("Ampere and newer", "Blackwell")
    fp8 = by_repo(found)["RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"]
    assert fp8.full_speed_on == "Ada and newer"


def test_the_family_is_read_from_the_models_own_configuration(found):
    assert {b.family for b in found.builds} == {"gemma4"}


def test_each_originals_builds_follow_it_most_downloaded_first(found):
    for original in found.originals:
        group = [b for b in found.builds if b.of == original]
        if not group:
            continue
        assert group[0].repo == original
        downloads = [b.downloads for b in group[1:]]
        assert downloads == sorted(downloads, reverse=True)


# --- what is not, and why ---


def test_files_for_other_engines_are_left_out_and_counted(found):
    offered = {b.repo for b in found.builds}
    assert not any("GGUF" in r or "gguf" in r or "MLX" in r or "mlx" in r for r in offered)
    assert found.hidden["GGUF files, for llama.cpp and Ollama"] > 0
    assert found.hidden["MLX files, for Apple silicon"] > 0


def test_a_fine_tune_is_a_different_model_and_is_not_offered(found):
    offered = {b.repo for b in found.builds}
    assert "Gryphe/Gemma-4-26B-A4B-StyleTune-V2" not in offered
    assert found.hidden["a different model — fine-tuned, merged, or built from another"] > 0


def test_a_gated_build_says_a_rented_host_cannot_fetch_it():
    entry = {"id": "someone/gated-build", "gated": "manual", "downloads": 5,
             "tags": [f"base_model:quantized:{ORIGINAL}"], "safetensors": {"parameters": {"BF16": 10}},
             "config": {"model_type": "gemma4"}}
    found = hubbuilds.sort_builds("gemma4:26b", [], RECORDED + [entry])
    gated = by_repo(found).get("someone/gated-build")
    assert gated is not None and gated.gated and "rented host" in gated.why_not


def test_nothing_linked_falls_back_to_the_most_downloaded_plain_repository():
    entries = [
        {"id": "a/model", "downloads": 10, "tags": [], "safetensors": {"parameters": {"BF16": 1}}, "config": {}},
        {"id": "b/model", "downloads": 99, "tags": [], "safetensors": {"parameters": {"BF16": 1}}, "config": {}},
    ]
    assert hubbuilds.choose_originals(entries)[0] == "b/model"


# --- reading names and settings ---


@pytest.mark.parametrize("name, terms", [
    ("gemma4:26b", ["gemma4-26b", "gemma-4-26b"]),
    ("qwen3:30b", ["qwen3-30b", "qwen-3-30b"]),
    ("llama3.1:8b-instruct-q4_K_M", ["llama3.1-8b-instruct", "llama-3.1-8b-instruct"]),
    ("nomic-embed-text:latest", ["nomic-embed-text"]),
    ("gemma4:e4b-it-bf16", ["gemma4-e4b-it", "gemma-4-e4b-it"]),
])
def test_a_pool_name_becomes_the_spellings_a_publisher_uses(name, terms):
    assert hubbuilds.search_terms(name) == terms


@pytest.mark.parametrize("quant, tally, precision", [
    (None, {"BF16": 10, "F32": 1}, "BF16"),
    ({"quant_method": "awq", "bits": 4}, {}, "INT4"),
    ({"quant_method": "gptq", "bits": 8}, {}, "INT8"),
    ({"quant_method": "fp8"}, {}, "FP8"),
    ({"quant_method": "mxfp4"}, {}, "MXFP4"),
    ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, {}, "NVFP4"),
    ({"quant_method": "modelopt"}, {"U8": 10, "F8_E4M3": 1}, "NVFP4"),
    ({"quant_method": "modelopt"}, {"U8": 0, "F8_E4M3": 9}, "FP8"),
    ({"quant_method": "compressed-tensors", "format": "int-quantized"}, {}, "INT8"),
    ({"quant_method": "compressed-tensors", "format": "mixed-precision",
      "config_groups": {"a": {"format": "float-quantized"}, "b": {"format": "nvfp4-pack-quantized"}}}, {}, "NVFP4"),
])
def test_precision_is_read_from_the_quantisation_settings(quant, tally, precision):
    assert hubbuilds.precision_of(quant, tally) == precision


def test_a_search_term_is_a_name_and_nothing_else():
    assert hubbuilds.valid_search("gemma-4-26b")
    for bad in ("", "gemma 4", "a&b=c", "x" * 81, "-rf", "q?x=1"):
        assert not hubbuilds.valid_search(bad), bad


def test_the_weights_listing_counts_only_weights():
    listing = [{"path": "model-00001.safetensors", "size": 10_000_000_000},
               {"path": "model-00002.safetensors", "size": 6_400_000_000},
               {"path": "tokenizer.json", "size": 30_000_000}]
    assert hubbuilds.weights_size_gb(listing) == 16.4


# --- the options and the machine's table say the same thing ---


def test_the_options_offered_are_exactly_the_ones_the_launcher_turns_into_flags():
    """The operator is shown which families an option applies to; the machine decides. The two
    tables are in different packages, so this is what keeps them from drifting apart."""
    from gpm_agent.vllm_launch import OPTIONS

    assert set(VllmEngine.options) == set(OPTIONS)
    for name, option in VllmEngine.options.items():
        assert set(option.families) == set(OPTIONS[name]), name
