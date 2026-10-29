"""The agent's own configuration — owned by whoever owns this machine, never by the pool.

docs/spec/host-agent.md §4–5. One JSON file, owner-readable only — on Windows, kept under the
user's profile (`outside_profile`). It holds the *hash* of the agent key, never the key; and the
things the pool can ask for but never define: whether models
may be deleted here, the command that restarts the engine, and whether the agent may install and
run the engine itself (§2.3).
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
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

DEFAULT_PATH = "~/.config/gpm-agent/agent.json"
KEY_PREFIX = "gpmg"
#: The pool's other two roles. Refused here by name, so a pasted wrong key says what it is.
FOREIGN_PREFIXES = {"gpma": "an app key", "gpmx": "the admin key"}


class SettingsError(Exception):
    """The file is missing, unsafe or does not make sense. The message is for an operator."""


#: Whether this is Windows, where a file's permission bits mean nothing: Python reports every
#: writable file there as readable and writable by all, so a check of them would refuse every
#: file. There a file is trusted by where it lives instead (`outside_profile`).
WINDOWS = os.name == "nt"

#: The places Windows gives one user alone: the profile, and the local application data in it
#: (named apart, as it can be moved).
PROFILE_VARIABLES = ("USERPROFILE", "LOCALAPPDATA")


def outside_profile(path: str, environ: Mapping[str, str], paths: Any = os.path) -> Optional[str]:
    """Why a file at `path` is not to be trusted on Windows, or None: it lies under this user's
    profile.

    A folder under the profile is the user's, the system's and administrators' alone, by the
    rules Windows gives it. One elsewhere usually is not: a folder made at the root of a drive
    inherits a rule that lets every signed-in user change what is in it — and this agent's
    settings, and the programs they name, decide what runs as its owner. Who may change a folder
    is in its access control list, which only Windows' security API reads and the standard
    library does not reach; so the place decides. Links are followed first, so a junction out of
    the profile counts as where it leads. `paths` is the path module the rules are read with."""
    roots = []
    for name in PROFILE_VARIABLES:
        value = environ.get(name) or ""
        if value and paths.isabs(value):
            roots.append(paths.normcase(paths.realpath(value)))
    if not roots:
        return (
            "cannot be checked: on Windows this agent takes its settings, and the programs it runs, "
            "only from under your profile, and USERPROFILE does not say where that is"
        )
    where = paths.normcase(paths.realpath(path))
    for root in roots:
        try:
            if paths.commonpath([root, where]) == root:
                return None
        except ValueError:  # another drive
            continue
    profile = environ.get("USERPROFILE") or environ.get("LOCALAPPDATA")
    return (
        f"is outside your profile ({profile}). On Windows this agent takes its settings, and the "
        "programs it runs, only from under your profile: a folder elsewhere — one made at the root "
        "of a drive, say — usually lets every user of this machine change what is in it, and so "
        f"what this agent runs as you. Move it under {profile}"
    )


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


#: Engines whose server this agent starts itself (D148), and so binds where it says.
AGENT_STARTED_ENGINES = frozenset({"whisper-cpp"})

#: Engines this agent can install and run itself, where the machine's owner allows it (D150) —
#: those the installer's table names (`installs.RELEASES`).
INSTALLABLE_ENGINES = frozenset({"ollama", "whisper-cpp"})

#: The addresses such a server is started on: 127.0.0.1, as the agent passes it, or a name for it.
_SERVER_HOSTS = ("127.0.0.1", "localhost")


def loopback_engine_url_problem(url: str) -> Optional[str]:
    """What is wrong with `url` as the address of a server this agent starts, or None.

    Plain http, on 127.0.0.1, with its port written out — the port is the one the server is
    started on, so it cannot be left to a scheme's default."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "is not a URL"
    if parts.scheme != "http":
        return "must be plain http: the server it names speaks nothing else"
    if parts.hostname not in _SERVER_HOSTS:
        return "is not this machine's loopback address"
    if port is None:
        return "names no port"
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return "must be the server's address and nothing more"
    return None


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
    #: The dead-man timer's heartbeat file, on a host the pool created (D63). The pool writes
    #: this when it installs the agent; on a machine somebody else owns there is no timer, and
    #: the heartbeat verb is refused. Touching it can only ever *postpone* a shutdown the pool
    #: could equally cause by going silent, which is why it needs no other bound.
    heartbeat_file: Optional[str] = None
    #: The engine's program, for an engine this agent starts itself — whisper.cpp's
    #: `whisper-server` (D148). The owner's choice, an absolute path; unset means the one on this
    #: agent's PATH. Not with `install_engines`, where the program is the one the agent installed.
    #: The pool never names a program.
    engine_binary: Optional[str] = None
    #: Whether this agent may install the engine, run it, and switch its version when the pool
    #: asks (D150, D151). The machine owner's word, written by `gpm-agent init --install-engines`;
    #: the pool can never change it. Off unless said.
    install_engines: bool = False
    #: Where installed engines live, an absolute path. Unset: the system's per-user place for an
    #: application's data (`platform.engines_dir`).
    engines_path: Optional[str] = None

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
        if self.engine_binary is not None and (
            not isinstance(self.engine_binary, str)
            or not Path(self.engine_binary).expanduser().is_absolute()
        ):
            raise SettingsError(
                "engine_binary must be the program's absolute path; leave it unset to use the one "
                "on this agent's PATH"
            )
        if not isinstance(self.install_engines, bool):
            raise SettingsError("install_engines must be true or false")
        if self.engines_path is not None and (
            not isinstance(self.engines_path, str) or not Path(self.engines_path).expanduser().is_absolute()
        ):
            raise SettingsError("engines_path must be an absolute path; leave it unset for the system's own place")
        if WINDOWS:
            # What this agent runs: a program the owner named, and the engines it installs.
            for name, value in (("engine_binary", self.engine_binary), ("engines_path", self.engines_path)):
                problem = outside_profile(os.path.expanduser(value), os.environ) if value else None
                if problem:
                    raise SettingsError(f"{name} {value!r} {problem}")
        if self.install_engines:
            if self.engine not in INSTALLABLE_ENGINES:
                raise SettingsError(
                    f"install_engines: this agent cannot install {self.engine}; it installs "
                    f"{', '.join(sorted(INSTALLABLE_ENGINES))}"
                )
            if self.restart_command is not None or self.engine_env_file is not None:
                raise SettingsError(
                    "install_engines: the agent runs the engine itself, and restarts it with the "
                    "pool's numbers; remove restart_command and engine_env_file, which are for an "
                    "engine the machine's owner runs (D150)"
                )
            if self.engine_binary is not None:
                raise SettingsError(
                    "install_engines: the agent runs the version of the engine it installed, and "
                    "never another program; remove engine_binary, which names the owner's own (D150)"
                )
            problem = loopback_engine_url_problem(self.engine_url)
            if problem:
                raise SettingsError(
                    f"engine_url {self.engine_url!r} {problem}. With install_engines this agent "
                    f"starts {self.engine} itself, on 127.0.0.1, and that server has no "
                    "authentication of its own, so it is never put on a network (D150). "
                    "Use http://127.0.0.1:<port>."
                )
        if self.engine in AGENT_STARTED_ENGINES:
            problem = loopback_engine_url_problem(self.engine_url)
            if problem:
                raise SettingsError(
                    f"engine_url {self.engine_url!r} {problem}. This agent starts {self.engine}'s "
                    "server itself, on 127.0.0.1, and that server has no authentication of its "
                    "own, so it is never put on a network: the pool reaches it on this machine or "
                    "through a tunnel (D148). Use http://127.0.0.1:<port>."
                )

    def accepts(self, key: str) -> bool:
        return hmac.compare_digest(fingerprint(key), self.key_hash)


def load(path: str | Path = DEFAULT_PATH) -> Settings:
    path = Path(path).expanduser()
    if not path.exists():
        raise SettingsError(f"{path} does not exist. Run `gpm-agent init` on this machine first.")
    _check_place(path)
    mode = path.stat().st_mode
    if not WINDOWS and mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
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


def _check_place(path: Path) -> None:
    """On Windows, refuse settings kept where other users may change them (`outside_profile`)."""
    if WINDOWS:
        problem = outside_profile(str(path), os.environ)
        if problem:
            raise SettingsError(f"{path} {problem}.")


def save(settings: Settings, path: str | Path = DEFAULT_PATH) -> Path:
    path = Path(path).expanduser()
    _check_place(path)  # before anything is written, so `init` leaves nothing there
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created owner-only from the first byte, not tightened afterwards; binary on Windows, where
    # a descriptor is otherwise in text mode and doubles the stream's own line endings.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(dataclasses.asdict(settings), handle, indent=2)
        handle.write("\n")
    os.chmod(path, 0o600)
    return path
