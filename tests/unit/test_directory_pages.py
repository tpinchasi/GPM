"""Reading Ollama's library from its pages (D101).

Ollama publishes its library only as pages, so the directory reads pages. These are the real
pages' markup, recorded and trimmed to a few models: the parser is tested against what the site
serves. When the site changes, a refresh fails and says so — which is what the last test is for.
"""

from pathlib import Path

import pytest
from gpm_server.directory import PageChanged, parse_library, parse_tags, size_tags

RECORDED = Path(__file__).resolve().parents[1] / "fixtures" / "directory"


@pytest.fixture(scope="module")
def library():
    return {m.name: m for m in parse_library((RECORDED / "ollama_library.html").read_text())}


@pytest.fixture(scope="module")
def gemma_tags():
    return {t.tag: t for t in parse_tags("gemma4", (RECORDED / "ollama_tags_gemma4.html").read_text())}


def test_every_model_on_the_page_is_read(library):
    assert set(library) == {"gemma4", "nomic-embed-text", "glm-5.1", "llama3"}


def test_a_model_carries_its_sizes_capabilities_and_description(library):
    gemma = library["gemma4"]
    assert gemma.sizes == ["e2b", "e4b", "12b", "26b", "31b"]
    assert {"tools", "vision", "thinking"} <= set(gemma.capabilities)
    assert gemma.description.startswith("Gemma 4 models are designed")
    assert gemma.pulls and gemma.tag_count == 50 and gemma.local


def test_a_model_only_on_ollamas_cloud_has_nothing_to_download(library):
    glm = library["glm-5.1"]
    assert glm.cloud and not glm.local and glm.sizes == []


def test_a_model_listing_no_sizes_is_still_one_to_download(library):
    """An embedding model has no size badges; it was first read as cloud-only and left out."""
    nomic = library["nomic-embed-text"]
    assert nomic.sizes == [] and nomic.local


def test_each_tag_carries_its_download_size_context_and_inputs(gemma_tags):
    big = gemma_tags["26b"]
    assert big.name == "gemma4:26b" and big.size_gb == 19.0
    assert big.context == "256K" and big.inputs == ["Text", "Image"]
    assert big.runtime is None and big.digest == "08ae7ec1744b"


def test_the_tag_latest_points_to_is_marked(gemma_tags):
    assert gemma_tags["e4b"].is_latest and not gemma_tags["26b"].is_latest


def test_apple_only_and_cloud_only_tags_are_marked(gemma_tags):
    assert gemma_tags["26b-mlx"].runtime == "mlx"
    assert gemma_tags["31b-cloud"].runtime == "cloud" and gemma_tags["31b-cloud"].size_gb is None


def test_megabytes_are_read_as_gigabytes():
    tags = parse_tags("nomic-embed-text", (RECORDED / "ollama_tags_nomic-embed-text.html").read_text())
    assert {t.tag: t.size_gb for t in tags}["latest"] == 0.274


def test_a_model_with_no_sizes_is_looked_up_under_latest(library):
    tags = parse_tags("nomic-embed-text", (RECORDED / "ollama_tags_nomic-embed-text.html").read_text())
    assert [t.name for t in size_tags(library["nomic-embed-text"], tags)] == ["nomic-embed-text:latest"]


def test_the_sizes_are_the_tags_a_hub_build_is_looked_up_for(library, gemma_tags):
    assert [t.tag for t in size_tags(library["gemma4"], list(gemma_tags.values()))] == [
        "e2b", "e4b", "12b", "26b", "31b",
    ]


def test_a_page_in_a_layout_this_does_not_read_is_refused_rather_than_read_as_empty():
    """An empty library would otherwise replace a good one."""
    with pytest.raises(PageChanged):
        parse_library("<html><body><p>We moved things around.</p></body></html>")
    with pytest.raises(PageChanged):
        parse_tags("gemma4", "<html></html>")
