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
