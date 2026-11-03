"""The pool's own processes on every system it runs on: macOS, Linux and Windows.

- **Is a process alive?** `os.kill(pid, 0)` asks that on macOS and Linux. On Windows it does
  not: `os.kill` there sends signal 0 as a console Ctrl-C, and ends the process for any other
  number. So Windows is asked itself (`OpenProcess`), as the host agent does.
- **What is it?** Its command line: `ps` on macOS and Linux, Windows' own record of the process
  (`Win32_Process`, through PowerShell) on Windows.
- **Starting the pool's own programs** — the supervisor beside the router, the forwarder beside
  the supervisor — runs this same interpreter, handing it this process's `sys.path`: a Python
  whose path was extended as it ran (an embeddable Python with packages installed by
  `pip --target`, which ignores `PYTHONPATH`; any launcher that inserts into `sys.path`) would
  otherwise start a child that cannot import `gpm_server` at all.
- **Detached**: a session of its own on macOS and Linux; on Windows a process group of its own
  and no console window, since a program under `pythonw` has none to share.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Optional, Sequence

#: Read where it is used, so that a test can play Windows on another system.
WINDOWS = os.name == "nt"

#: Windows' process-creation flags, by value: the names exist in `subprocess` only on Windows.
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

#: What a child of the pool runs first: this process's import path, then the pool's command line.
#: The path comes as the argument after `-c`, so nothing in it is ever code.
BOOTSTRAP = (
    "import json, runpy, sys; "
    "sys.path[:0] = [p for p in json.loads(sys.argv.pop(1)) if p not in sys.path]; "
    "runpy.run_module('gpm_server.cli', run_name='__main__', alter_sys=True)"
)


def pool_command(*args: str, python: Optional[str] = None, path: Optional[Sequence[str]] = None) -> list[str]:
    """`gpm <args>` as a child of this process: this interpreter, this import path."""
    return [python or sys.executable, "-c", BOOTSTRAP, json.dumps(list(sys.path if path is None else path)), *args]


def detached() -> dict[str, Any]:
    """`subprocess.Popen` options that let a child outlive this process."""
    if WINDOWS:
        return {"creationflags": CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW}
    return {"start_new_session": True}


def windowless() -> dict[str, Any]:
    """`subprocess.Popen` options for a child that stays this process's: no console window of
    its own on Windows, where a program under `pythonw` has none."""
    return {"creationflags": CREATE_NO_WINDOW} if WINDOWS else {}


def has_console() -> bool:
    """Can a child write where this process does? Not under `pythonw`, nor a service: there is
    no standard error to inherit, and what the child says would be lost."""
    try:
        return sys.stderr is not None and sys.stderr.fileno() >= 0
    except (AttributeError, OSError, ValueError):
        return False


def elevated() -> bool:
    """Is this process running with an administrator's rights on Windows? Never asked elsewhere."""
    if not WINDOWS:
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def warn_if_elevated(what: str) -> None:
    """Say so where an operator will read it: what an elevated pool writes belongs to the
    Administrators group, and the same user, not elevated, may not read it afterwards."""
    if elevated():
        import logging

        logging.getLogger("gpm").warning(
            "the %s runs with administrator rights. Run it as yourself, not elevated: what it writes "
            "now — its database, keys and logs — may not be readable by your own account later", what)


def _windows_alive(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    query_limited_information, still_active, access_denied = 0x1000, 259, 5
    handle = kernel32.OpenProcess(query_limited_information, False, int(pid))
    if not handle:
        # There, and not this user's to open — which the command-line check then settles.
        return ctypes.get_last_error() == access_denied  # type: ignore[attr-defined]
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def alive(pid: int) -> bool:
    """Is process `pid` running? Says nothing about which program it is: `command_line`."""
    if pid <= 0:
        return False
    if WINDOWS:
        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # there, and not this user's — which the command-line check then settles
    except OSError:
        return False
    return True


def command_line(pid: int) -> Optional[str]:
    """A process's command line as the system reports it, or None where it cannot be read."""
    if WINDOWS:
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                f'(Get-CimInstance Win32_Process -Filter "ProcessId={int(pid)}").CommandLine']
    else:
        argv = ["ps", "-o", "args=", "-p", str(int(pid))]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
                             **windowless())
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip()
    return line or None
