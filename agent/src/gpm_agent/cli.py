"""`gpm-agent init` and `gpm-agent serve`."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from . import proxy as proxy_module
from . import vllm_launch, whisper_cpp
from .settings import DEFAULT_PATH, Settings, SettingsError, fingerprint, load, mint_key, save


def _init(args: argparse.Namespace) -> int:
    path = Path(args.config).expanduser()
    if path.exists() and not args.rotate:
        print(f"{path} already exists. Use --rotate to mint a new key and keep the other settings.", file=sys.stderr)
        return 1
    key = mint_key()
    if path.exists():
        settings = load(path)
        settings.key_hash = fingerprint(key)
    else:
        settings = Settings(
            key_hash=fingerprint(key), host=args.host, port=args.port,
            engine=args.engine,
            engine_url=args.engine_url or DEFAULT_ENGINE_URL.get(args.engine, "http://127.0.0.1:11434"),
            heartbeat_file=args.heartbeat_file,
            # Where this engine's weights live — measured for free disk, and for an engine the
            # agent fetches for, where the fetch puts them and the engine is started from.
            models_path=args.models_path or DEFAULT_MODELS_PATH.get(args.engine, "~/.ollama/models"),
            restart_command=([args.restart_command] if args.restart_command else None),
            engine_env_file=args.engine_env_file,
            engine_binary=args.engine_binary,
        )
    settings.check()
    save(settings, path)
    print(f"Agent key (shown once; only its hash is stored in {path}):\n\n  {key}\n")
    print("Give it to the pool as an environment variable on the pool's machine, and name that")
    print("variable in the host's `agent.bearer_env`. Then: gpm-agent serve")
    if settings.engine == "whisper-cpp" and not settings.engine_binary and shutil.which(whisper_cpp.PROGRAM) is None:
        # Not a refusal: the owner may install it next. Said now rather than on the first pass.
        print(f"\nNote: {whisper_cpp.PROGRAM} is not on this PATH. Install whisper.cpp, or set")
        print(f"engine_binary in {path} to its program, before the pool asks this machine to serve.")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app

    settings = load(args.config)
    # Beside the settings, owner-only: the pins this agent set, so it can release them later.
    state_path = Path(args.config).expanduser().with_suffix(".state.json")
    uvicorn.run(
        create_app(settings, state_path=state_path),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        ssl_certfile=settings.tls_certfile,
        ssl_keyfile=settings.tls_keyfile,
    )
    return 0


#: Where each engine keeps its models when nobody says otherwise.
DEFAULT_MODELS_PATH = {
    "ollama": "~/.ollama/models", "vllm": "~/gpm-models", "whisper-cpp": whisper_cpp.DEFAULT_MODELS_PATH,
}

#: Where each engine answers when nobody says otherwise. whisper.cpp's server is started by the
#: agent on the port named here — the pool's default for that engine, not the server's own 8080.
DEFAULT_ENGINE_URL = {"whisper-cpp": "http://127.0.0.1:8178"}


def _proxy(args: Any) -> int:
    proxy_module.serve(args.upstreams, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpm-agent", description="The GPM host agent.")
    parser.add_argument("-c", "--config", default=DEFAULT_PATH)
    verbs = parser.add_subparsers(dest="verb", required=True)

    init = verbs.add_parser("init", help="mint the agent key and write this machine's settings")
    init.add_argument("--host", default="127.0.0.1")
    init.add_argument("--port", type=int, default=8095)
    init.add_argument("--engine", default="ollama", help="which engine this machine runs")
    init.add_argument("--engine-url", default=None,
                      help="where the engine answers; the engine's usual address if unset")
    init.add_argument("--models-path", default=None,
                      help="where the engine's models live; the engine's usual place if unset")
    init.add_argument("--heartbeat-file", default=None, help="the dead-man timer's file, where the pool created this host")
    init.add_argument("--restart-command", default=None, help="how the engine is restarted here (a program to run)")
    init.add_argument("--engine-env-file", default=None, help="where the engine's start-up environment is written")
    init.add_argument("--engine-binary", default=None,
                      help="the engine's program, for an engine this agent starts (whisper-cpp); PATH if unset")
    init.add_argument("--rotate", action="store_true", help="mint a new key, keep everything else")
    init.set_defaults(run=_init)

    serve = verbs.add_parser("serve", help="answer the pool")
    serve.add_argument("--log-level", default="warning")
    serve.set_defaults(run=_serve)

    # A separate process from the agent, carrying inference traffic and nothing else (D96).
    # It ships here because this archive is already on every host the pool creates; it is not
    # the agent, and the agent is still never on the request path.
    proxy = verbs.add_parser(
        "proxy", help="one endpoint in front of several engine processes on this machine"
    )
    proxy_module.add_arguments(proxy)
    proxy.set_defaults(run=_proxy)

    # Run by the machine's own restart script, never over the agent's protocol (D97).
    start = verbs.add_parser(
        "vllm-start", help="start vLLM for the models on disk here, stopping what ran before"
    )
    vllm_launch.add_arguments(start)
    start.set_defaults(run=vllm_launch.main)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except SettingsError as problem:
        print(f"gpm-agent: {problem}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
