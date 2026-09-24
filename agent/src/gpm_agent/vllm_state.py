"""What the vLLM launcher started, read back — and which of it has since died (D104).

The launcher (`vllm_launch`) starts processes, so nothing that answers the pool may import it
(D63). This module only reads: the launcher's record of what it started, each process's
liveness, and the last error line of a process's own log. The agent reports a process that has
exited without ever serving its model as that model's failure, with that line as the reason —
so the pool gives the host up saying why, instead of watching it "prepare" until its hold runs
out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

#: Written by the launcher on every start: process, port and log per model, or why nothing was
#: started.
STARTED_FILE = ".gpm-vllm.started.json"


def read_record(models_dir: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads((models_dir / STARTED_FILE).read_text())
    except (OSError, ValueError):
        return None


def last_error_line(log_path: str, tail_bytes: int = 16384) -> Optional[str]:
    """The last line of an engine's log that names an error — what the operator would read."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if "Error" in line or "error:" in line.lower():
            # vLLM prefixes every line with the process that wrote it; that is not the reason.
            cleaned = line.split(") ", 1)[1] if line.startswith("(") and ") " in line else line
            return cleaned.strip()[:300]
    return None


def failed_engines(models_dir: Path, served: Sequence[str],
                   alive: Optional[Callable[[int], bool]] = None) -> dict[str, str]:
    """Models whose engine process is gone without ever serving them, and why, by the log.

    A process that is still up is loading, however long it takes; only one that has exited
    without its model being served has failed. The launcher's refusal counts for every model.
    """
    record = read_record(models_dir)
    if not record:
        return {}
    if record.get("refused"):
        return {name: record["refused"] for name in (record.get("plan") or {}).get("models", {})}

    def is_alive(pid: int) -> bool:
        if alive is not None:
            return alive(pid)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    failed: dict[str, str] = {}
    for engine in record.get("engines") or []:
        name, pid = engine.get("model"), engine.get("pid")
        if not name or name in served or not isinstance(pid, int):
            continue
        if is_alive(pid):
            continue
        reason = last_error_line(engine.get("log") or "") or "no reason in its log"
        failed[name] = f"its vLLM process exited before serving it: {reason}"
    return failed
