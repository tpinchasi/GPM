"""The `pool` command.

`gpm serve` starts both halves — the router in this process, the supervisor in its own — and
each can also be run alone, because they share only a database (docs/spec/supervisor.md §1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
from typing import Optional

from .config import ConfigError, load_config


def _supervisor_child(config_path: str, log_level: str) -> subprocess.Popen:
    child = subprocess.Popen(
        [sys.executable, "-m", "gpm_server.cli", "supervise", "-c", config_path, "--log-level", log_level]
    )

    def watch() -> None:
        # Reap it, and say so plainly. Routing carries on either way, which is the point of
        # the split — but an operator should not have to discover this from a stale host table.
        code = child.wait()
        if code != 0:
            logging.getLogger("gpm").error(
                "the supervisor exited (code %s). Routing continues on the last published "
                "host table; nothing will be rented, recovered or released until it is back.",
                code,
            )

    threading.Thread(target=watch, daemon=True).start()
    return child


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .router.app import create_app

    config = load_config(args.config)
    child: Optional[subprocess.Popen] = None
    if not args.router_only:
        child = _supervisor_child(args.config, args.log_level)

    try:
        app = create_app(config)
        uvicorn.run(
            app,
            host=config.listen.host,
            port=config.listen.port,
            ssl_certfile=config.listen.tls_certfile,
            ssl_keyfile=config.listen.tls_keyfile,
            log_level=args.log_level,
        )
    finally:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
    return 0


def _supervise(args: argparse.Namespace) -> int:
    from .db import SupervisorBusy
    from .state import open_database
    from .supervisor import run
    from .supervisor.service import ProviderCredentialMissing

    config = load_config(args.config)
    database = open_database(config)

    async def supervised() -> None:
        # SIGTERM is how `gpm serve` (and any service manager) stops us. Left to Python's
        # default it ends the process without running a single `finally` — and the lock,
        # tunnels and clients would all be left behind. Turned into a cancellation, the
        # ordinary shutdown path runs instead.
        import signal

        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: task.cancel() if task else None)
        await run(config, database, config_path=args.config)

    try:
        asyncio.run(supervised())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except SupervisorBusy as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ProviderCredentialMissing as exc:
        print(str(exc), file=sys.stderr)
        return 4
    finally:
        database.close()
    return 0


def _forwarder(args: argparse.Namespace) -> int:
    """Keep the pool's SSH forwards up, apart from the supervisor (D110)."""
    import signal
    from pathlib import Path

    from .db import SupervisorBusy
    from .forwarder import Forwarder
    from .state import open_database

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if not config.forwarder.enabled:
        print("this pool's forwarder is off (forwarder.enabled); the supervisor keeps its own forwards",
              file=sys.stderr)
        return 2
    database = open_database(config)
    forwarder = Forwarder(
        database, config.pool.name, config.forwarder.on_restart,
        Path(config.request_log).expanduser().resolve().parent,
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: forwarder.stop())
    try:
        forwarder.run()
    except SupervisorBusy as exc:
        print(str(exc).replace("supervisor", "forwarder"), file=sys.stderr)
        return 3
    finally:
        database.close()
    return 0


def _control(args: argparse.Namespace, method: str, path: str, body: Optional[dict] = None) -> int:
    """Every control verb is the same call with the admin key. Nothing is console-only, and
    nothing here is CLI-only either."""
    import httpx

    key = os.environ.get("GPM_ADMIN_KEY")
    if not key:
        print("GPM_ADMIN_KEY is not set", file=sys.stderr)
        return 2
    url = getattr(args, "url", None) or os.environ.get("GPM_CONTROL_URL", "http://127.0.0.1:8081")
    try:
        response = httpx.request(
            method,
            f"{url}{path}",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        print(f"could not reach the control API at {url}: {exc}", file=sys.stderr)
        return 1
    try:
        printed = json.dumps(response.json(), indent=2)
    except ValueError:
        printed = response.text
    print(printed, file=sys.stdout if response.status_code < 400 else sys.stderr)
    return 0 if response.status_code < 400 else 1


def _key(args: argparse.Namespace) -> int:
    from .config import load_config
    from .keys import KeyStore

    config = load_config(args.config)
    path = config.auth.app_keys_file if args.role == "app" else config.auth.admin_keys_file
    if not path:
        print(
            f"configuration has no auth.{args.role}_keys_file to write to", file=sys.stderr
        )
        return 2
    store = KeyStore(path)

    if args.action == "create":
        key, record = store.create(args.role, label=args.label)
        print(f"{record.key_id}  {record.role}")
        print(key)
        print("\nThis is the only time the key is shown: only its hash is stored.", file=sys.stderr)
        return 0
    if args.action == "list":
        for record in store.load():
            print(f"{record.key_id}  {record.role}  {record.label or ''}")
        return 0
    if args.action == "revoke":
        return 0 if store.revoke(args.key_id) else 1
    return 2


def _lease(args: argparse.Namespace) -> int:
    if args.action == "open":
        return _control(
            args,
            "POST",
            "/pool/leases",
            {
                "workers": args.workers if args.workers is not None else 1,
                "max_hours": args.max_hours if args.max_hours is not None else 4.0,
                "max_spend": args.max_spend,
                "allow_rent": args.allow_rent,
                "max_all_in_hourly": args.max_all_in_hourly,
            },
        )
    if args.action == "close":
        return _control(args, "DELETE", f"/pool/leases/{args.lease_id}")
    if args.action in ("tighten", "extend"):
        # One call either way: the pool decides which it is by comparing with the lease, and
        # only a raise needs --confirm.
        body = {k: v for k, v in (
            ("max_spend", args.max_spend), ("max_hours", args.max_hours), ("workers", args.workers),
        ) if v is not None}
        return _control(args, "PATCH", f"/pool/leases/{args.lease_id}", {**body, "confirm": args.confirm})
    return _control(args, "GET", "/pool/leases")


def _host(args: argparse.Namespace) -> int:
    if args.action == "prepare":
        return _control(
            args,
            "POST",
            "/pool/hosts/prepare",
            {
                "max_spend": args.max_spend,
                "max_hours": args.max_hours,
                "max_all_in_hourly": args.max_all_in_hourly,
                "when_ready": args.when_ready,
                "offer_id": args.offer_id,
                "kind": args.kind,
            },
        )
    if args.action == "show":
        return _control(args, "GET", f"/pool/hosts/{args.host_id}")
    if args.action == "resize":
        if args.workers is None:
            print("resize needs --workers", file=sys.stderr)
            return 2
        body = {"workers": args.workers}
        if args.confirm:
            body["confirm"] = args.confirm
        answer = _control(args, "POST", f"/pool/hosts/{args.host_id}/resize", body)
        print(answer.get("detail") or answer)
        return 0

    if args.action == "restart-engine":
        return _control(
            args, "POST", f"/pool/hosts/{args.host_id}/engine/restart",
            {"confirm": args.confirm, "apply_settings": args.apply_settings},
        )
    if args.action == "delete-model":
        # The same call the console's button makes. The tag is given twice on purpose.
        return _control(
            args, "POST", f"/pool/hosts/{args.host_id}/models/delete",
            {"tag": args.tag, "confirm": args.confirm},
        )
    return _control(args, "POST", f"/pool/hosts/{args.host_id}/{args.action}")


def _config(args: argparse.Namespace) -> int:
    """Everything the Configuration screen does, from the command line."""
    if args.action == "get":
        return _control(args, "GET", "/pool/config")
    if args.action == "history":
        return _control(args, "GET", "/pool/config/history")
    if args.action == "rollback":
        if not args.version:
            print("give --version, from `gpm config history`", file=sys.stderr)
            return 2
        return _control(args, "POST", "/pool/config/rollback", {"version": args.version})

    if not args.file:
        print(f"give --file: the candidate configuration to {args.action}", file=sys.stderr)
        return 2
    text = pathlib.Path(args.file).read_text()
    if args.action == "validate":
        return _control(args, "POST", "/pool/config/validate", {"text": text})
    if args.action == "plan":
        return _control(args, "POST", "/pool/config/plan", {"text": text})
    if args.action == "apply":
        # Same rule as the console: nothing is applied blind, and a change that loosens a
        # limit has to be meant.
        import httpx

        key = os.environ.get("GPM_ADMIN_KEY")
        url = getattr(args, "url", None) or os.environ.get("GPM_CONTROL_URL", "http://127.0.0.1:8081")
        headers = {"Authorization": f"Bearer {key}"}
        current = httpx.get(f"{url}/pool/config", headers=headers, timeout=30).json()
        plan = httpx.post(f"{url}/pool/config/plan", headers=headers, json={"text": text}, timeout=60).json()
        if plan.get("errors"):
            print("\n".join(plan["errors"]), file=sys.stderr)
            return 1
        loosening = [c for c in plan.get("changes", []) if c.get("requires_retype")]
        print(json.dumps(plan, indent=2))
        if loosening and not args.yes:
            print(
                "\nThis loosens a limit. Re-run with --yes to confirm:\n  "
                + "\n  ".join(c["detail"] for c in loosening),
                file=sys.stderr,
            )
            return 1
        return _control(args, "PUT", "/pool/config", {"text": text, "version": current.get("version")})
    return 2


def _host_test(args: argparse.Namespace) -> int:
    body: dict = {"workers": args.workers}
    if args.capability:
        body["capabilities"] = args.capability
    body["base_url"] = args.base_url
    return _control(args, "POST", "/pool/hosts/test", body)


def _status(args: argparse.Namespace) -> int:
    import httpx

    key = os.environ.get("GPM_API_KEY")
    if not key:
        print("GPM_API_KEY is not set", file=sys.stderr)
        return 2
    url = args.url or os.environ.get("GPM_URL", "http://127.0.0.1:8080")
    response = httpx.get(f"{url}/pool/status", headers={"Authorization": f"Bearer {key}"}, timeout=10.0)
    if response.status_code != 200:
        print(f"{response.status_code}: {response.text}", file=sys.stderr)
        return 1
    print(json.dumps(response.json(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpm", description="GPM — GPU Hosts Pool Management")
    parser.add_argument(
        "--version",
        action="store_true",
        help="what this install is: its release tag, or that it is a development tree (D79)",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    serve = subparsers.add_parser("serve", help="run the router, and the supervisor beside it")
    serve.add_argument("--config", "-c", default="pool.yaml")
    serve.add_argument("--log-level", default="info")
    serve.add_argument(
        "--router-only",
        action="store_true",
        help="do not start a supervisor; serve from whatever host table already exists",
    )
    serve.set_defaults(func=_serve)

    supervise = subparsers.add_parser("supervise", help="run the supervisor alone")
    supervise.add_argument("--config", "-c", default="pool.yaml")
    supervise.add_argument("--log-level", default="info")
    supervise.set_defaults(func=_supervise)

    forwarder = subparsers.add_parser(
        "forwarder", help="keep the pool's SSH forwards up, apart from the supervisor (forwarder.enabled)"
    )
    forwarder.add_argument("--config", "-c", default="pool.yaml")
    forwarder.add_argument("--log-level", default="info")
    forwarder.set_defaults(func=_forwarder)

    status = subparsers.add_parser("status", help="print the pool's status")
    status.add_argument("--url", default=None)
    status.set_defaults(func=_status)

    key = subparsers.add_parser("key", help="create, list and revoke keys")
    key.add_argument("action", choices=["create", "list", "revoke"])
    key.add_argument("--role", choices=["app", "admin"], default="app")
    key.add_argument("--label", default=None)
    key.add_argument("--key-id", default=None)
    key.add_argument("--config", "-c", default="pool.yaml")
    key.set_defaults(func=_key)

    lease = subparsers.add_parser("lease", help="open, close and list leases")
    lease.add_argument("action", choices=["open", "close", "list", "tighten", "extend"])
    lease.add_argument("lease_id", nargs="?", default=None)
    lease.add_argument("--workers", type=int, default=None)
    lease.add_argument("--max-hours", type=float, default=None)
    #: No default on purpose: a lease that can rent must state its dollars (D32).
    lease.add_argument("--max-spend", type=float, default=None)
    lease.add_argument("--allow-rent", action="store_true")
    lease.add_argument("--max-all-in-hourly", type=float, default=None,
                       help="tighten the pool's all-in maximum per host-hour for this lease")
    lease.add_argument("--confirm", default=None,
                       help="extend: the new value again, since raising a limit is loosening")
    lease.add_argument("--url", default=None)
    lease.set_defaults(func=_lease)

    config_cmd = subparsers.add_parser("config", help="read, check and apply the pool's configuration")
    config_cmd.add_argument("action", choices=["get", "validate", "plan", "apply", "history", "rollback"])
    config_cmd.add_argument("--file", "-f", default=None, help="the candidate configuration")
    config_cmd.add_argument("--version", default=None, help="for rollback, from `config history`")
    config_cmd.add_argument("--yes", action="store_true", help="confirm a change that loosens a limit")
    config_cmd.add_argument("--url", default=None)
    config_cmd.set_defaults(func=_config)

    host_test = subparsers.add_parser("host-test", help="test an unsaved host definition; saves nothing")
    host_test.add_argument("base_url")
    host_test.add_argument("--workers", type=int, default=1)
    host_test.add_argument("--capability", action="append", default=[])
    host_test.add_argument("--url", default=None)
    host_test.set_defaults(func=_host_test)

    host = subparsers.add_parser("host", help="prepare, drain, release or park a rented host")
    host.add_argument("action", choices=["prepare", "drain", "release", "park", "delete-model", "restart-engine", "resize", "show"])
    host.add_argument("--apply-settings", action="store_true",
                      help="restart-engine: first write the parallelism and models-held the pool needs")
    host.add_argument("--tag", default=None, help="delete-model: the model tag to delete from the host's disk")
    host.add_argument("--confirm", default=None, help="delete-model: the same tag again. restart-engine: the host id again")
    host.add_argument("host_id", nargs="?", default=None)
    host.add_argument("--workers", type=int, default=None, help="resize: how many requests this host takes at once")
    host.add_argument("--max-spend", type=float, default=None)
    host.add_argument("--max-hours", type=float, default=1.0)
    host.add_argument("--max-all-in-hourly", type=float, default=None,
                      help="prepare: tighten the pool's all-in maximum per host-hour")
    host.add_argument("--when-ready", choices=["join", "park", "destroy"], default="join")
    host.add_argument("--offer-id", default=None, help="prepare: rent exactly this offer from `gpm market`")
    host.add_argument("--kind", choices=["interruptible", "on_demand"], default=None,
                      help="prepare: bid for it, or pay the listed price so it cannot be outbid")
    host.add_argument("--url", default=None)
    host.set_defaults(func=_host)

    machines = subparsers.add_parser(
        "machines", help="what each machine has done for this pool; spends nothing"
    )
    machines.add_argument("--url", default=None)
    machines.set_defaults(func=lambda a: _control(a, "GET", "/pool/machines"))

    plan = subparsers.add_parser("plan", help="what the supervisor would do; spends nothing")
    plan.add_argument("--url", default=None)
    plan.set_defaults(func=lambda a: _control(a, "GET", "/pool/plan"))

    events = subparsers.add_parser("events", help="the decision log, with the numbers behind it")
    events.add_argument("--limit", type=int, default=50)
    events.add_argument("--url", default=None)
    events.set_defaults(func=lambda a: _control(a, "GET", f"/pool/events?limit={a.limit}"))

    market = subparsers.add_parser("market", help="the live market through your policy; spends nothing")
    market.add_argument("--hours", type=float, default=4.0)
    market.add_argument("--url", default=None)
    market.set_defaults(func=lambda a: _control(a, "GET", f"/pool/market/preview?hours={a.hours}"))

    account = subparsers.add_parser("account", help="is the provider credential valid; spends nothing")
    account.add_argument("--url", default=None)
    account.set_defaults(func=lambda a: _control(a, "GET", "/pool/account"))

    down = subparsers.add_parser("down", help="destroy every rented host now, verified")
    down.add_argument("--all", action="store_true", required=True)
    down.add_argument("--url", default=None)
    down.set_defaults(func=lambda a: _control(a, "POST", "/pool/down"))

    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from .version import running

        print(running().describe())
        return 0
    if getattr(args, "func", None) is None:
        parser.print_help(sys.stderr)
        return 2
    # Only the long-running verbs take --log-level; the rest are one call and an answer.
    level_name = getattr(args, "log_level", "warning")
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        return args.func(args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
