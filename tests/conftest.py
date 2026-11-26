import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest


@pytest.fixture(autouse=True)
def _no_registry(monkeypatch):
    """Every test runs without the network (CLAUDE.md): a pool that tracks its engine's releases
    (D177) finds the registry away and keeps the shipped image. A test of the tracker hands it a
    fake fetch of its own."""
    async def away(url: str) -> bytes:
        raise ConnectionError("no network in tests")

    monkeypatch.setattr("gpm_server.engine_images._fetch", away)


@pytest.fixture
def allow_size_lookups():
    """Asked for by a test of the supervisor's own size lookups (D207), which points them at
    stand-ins for Ollama's site and the hub."""


@pytest.fixture(autouse=True)
def _no_size_lookups(request, monkeypatch):
    """Every test runs without the network (CLAUDE.md): the supervisor's lookup of its builds'
    published sizes (D207) is on by default and would read Ollama's site or the hub. A test of
    it asks for `allow_size_lookups` and gives it stand-ins."""
    if "allow_size_lookups" not in request.fixturenames:
        monkeypatch.setattr("gpm_server.directory.Directory.maybe_look_up_sizes", lambda self: False)
    # A supervisor a test starts as its own process does not see the patch above; it inherits
    # these, so whatever it reads is refused on this machine. A test that serves a stand-in
    # site sets its own address over them.
    monkeypatch.setenv("GPM_OLLAMA_SITE", "http://127.0.0.1:9")
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
