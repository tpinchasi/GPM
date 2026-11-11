"""Environment-variable configuration, per docs/spec/app-contract.md §4."""

from __future__ import annotations

import os


def env_url(default: str = "http://127.0.0.1:8080") -> str:
    return os.environ.get("GPM_URL", default)


def pool_root(url: str) -> str:
    """The pool's root, which the SDK adds its paths to: the address as given, less a trailing
    `/v1`. A pool hands out its OpenAI-style address, `…/v1`, for any OpenAI-compatible client;
    the SDK adds `/v1` itself, so that address would otherwise become `…/v1/v1/…` (found live)."""
    root = url.strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def env_api_key() -> str | None:
    return os.environ.get("GPM_API_KEY")
