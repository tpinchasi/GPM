"""The agent's own configuration — owned by whoever owns this machine, never by the pool.

docs/spec/host-agent.md §4–5. One JSON file, owner-readable only. It holds the *hash* of the
agent key, never the key; and the two things the pool can ask for but never define: whether
models may be deleted here, and the command that restarts the engine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

DEFAULT_PATH = "~/.config/gpm-agent/agent.json"
KEY_PREFIX = "gpmg"
#: The pool's other two roles. Refused here by name, so a pasted wrong key says what it is.
FOREIGN_PREFIXES = {"gpma": "an app key", "gpmx": "the admin key"}


class SettingsError(Exception):
    """The file is missing, unsafe or does not make sense. The message is for an operator."""


def mint_key() -> str:
    # 32 random bytes: beyond any dictionary, so a plain digest is enough to store.
    return f"{KEY_PREFIX}_{secrets.token_hex(32)}"


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclasses.dataclass
class Settings:
    key_hash: str
    host: str = "127.0.0.1"
    port: int = 8095
    engine: str = "ollama"
    engine_url: str = "http://127.0.0.1:11434"
    #: Where the engine keeps its models; free disk is measured here.
    models_path: str = "~/.ollama/models"
    tls_certfile: Optional[str] = None
    tls_keyfile: Optional[str] = None
    #: Plain HTTP off loopback sends the agent key in clear text. Off unless typed here.
    allow_insecure: bool = False
    #: Pulls stop before free disk falls below this. The owner's number; the pool cannot lower it.
    min_free_disk_gb: float = 10.0
    #: The machine owner's last word on deletion. The pool cannot change it.
    allow_delete: bool = True
    #: How the engine is restarted here, as an argument list. The pool may ask for it to be
    #: run; it can never say what it is. Absent means the agent cannot restart the engine.
    restart_command: Optional[list[str]] = None
    restart_timeout_s: int = 120
    #: Where the engine's start-up environment is written, for the owner's service definition to
    #: read (a systemd `EnvironmentFile=`, a launch script). The pool supplies numbers; the
    #: owner supplies the path. Absent means the pool cannot change engine settings here.
    engine_env_file: Optional[str] = None

    def check(self) -> None:
        if not is_loopback(self.host) and not (self.tls_certfile and self.tls_keyfile):
            if not self.allow_insecure:
                raise SettingsError(
                    f"listening on {self.host} without TLS would send the agent key in clear "
                    "text. Set tls_certfile and tls_keyfile, keep the agent on loopback behind "
                    "an SSH tunnel, or set allow_insecure: true if the network is yours."
                )
        if self.restart_command is not None and (
            not isinstance(self.restart_command, list)
            or not all(isinstance(part, str) for part in self.restart_command)
        ):
            raise SettingsError("restart_command must be a list of strings: an argument list, never a shell line")

    def accepts(self, key: str) -> bool:
        return hmac.compare_digest(fingerprint(key), self.key_hash)


def load(path: str | Path = DEFAULT_PATH) -> Settings:
    path = Path(path).expanduser()
    if not path.exists():
        raise SettingsError(f"{path} does not exist. Run `gpm-agent init` on this machine first.")
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        raise SettingsError(f"{path} is readable or writable by others; `chmod 600` it.")
    try:
        body = json.loads(path.read_text())
    except ValueError as exc:
        raise SettingsError(f"{path} is not valid JSON: {exc}") from exc
    known = {field.name for field in dataclasses.fields(Settings)}
    unknown = set(body) - known
    if unknown:
        raise SettingsError(f"{path} has settings this agent does not know: {sorted(unknown)}")
    if "key_hash" not in body:
        raise SettingsError(f"{path} has no key_hash. Run `gpm-agent init`.")
    settings = Settings(**body)
    settings.check()
    return settings


def save(settings: Settings, path: str | Path = DEFAULT_PATH) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created owner-only from the first byte, not tightened afterwards.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(dataclasses.asdict(settings), handle, indent=2)
        handle.write("\n")
    os.chmod(path, 0o600)
    return path
