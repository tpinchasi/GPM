"""Environment-variable configuration, per docs/spec/app-contract.md §4."""

from __future__ import annotations

import os


def env_url(default: str = "http://127.0.0.1:8080") -> str:
    return os.environ.get("GPM_URL", default)


def env_api_key() -> str | None:
    return os.environ.get("GPM_API_KEY")
