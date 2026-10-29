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
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import platform as host_platform

#: Written by the launcher on every start: process, port and log per model, or why nothing was
#: started.
STARTED_FILE = ".gpm-vllm.started.json"


def state_dir_for(models_dir: str | Path) -> Path:
    """Where the launcher keeps what it started — process ids, its record, logs, the router's
    map: beside the models directory, never in it (D139). What arrives in the models directory
    from elsewhere — a provider's copy from a sibling, a volume another machine wrote — must not
    bring that machine's process ids with it: the next start would signal those numbers here, and
    read every model as failed because they are not running."""
    models_dir = Path(models_dir).expanduser().resolve()
    return models_dir.parent / f"{models_dir.name or 'models'}.gpm-state"


def read_record(models_dir: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads((state_dir_for(models_dir) / STARTED_FILE).read_text())
    except (OSError, ValueError):
        return None


#: An exception as Python prints it, closing a traceback: `ValueError: …`, `torch.OutOfMemoryError: …`.
_EXCEPTION = re.compile(r"(?:^|\s)((?:[A-Za-z_]\w*\.)*\w*(?:Error|Exception)): (\S.*)$")

#: One with nothing after its name — `AssertionError`, common in vLLM — alone on its line once the
#: process's and the logger's prefixes are taken off, so that a sentence ending in "Error" is not one.
_BARE_EXCEPTION = re.compile(r"(?:[A-Za-z_]\w*\.)*\w*(?:Error|Exception)")
_PREFIXES = re.compile(r"^(?:\([^)]*\)\s*)?(?:[A-Z]+\s+\d\d-\d\d\s+[\d:.]+\s+\[[^\]]*\]\s*)?")

#: What vLLM's API server says when its engine process failed: true, and never the reason — that
#: is the engine process's own exception, further up the same log.
_POINTS_ELSEWHERE = re.compile(r"See root cause above|Engine core initialization failed|Failed core proc")

#: Written by the launcher into an engine's log before each start: the log is kept across starts,
#: and a reason from an earlier one is not this one's.
RUN_MARKER = "=== gpm-agent: vLLM started here ==="


#: How the helper that starts a card's models in turn is known by its command line (D155).
IN_TURN_VERB = b"vllm-in-turn"


def in_turn_helper(pid: int, command_line: Callable[[int], Optional[bytes]]) -> bool:
    """Is `pid` still the helper that starts models in turn? A number in a file is only a number:
    once the helper has gone it may be any process's. Where the command line cannot be read, the
    number is taken at its word."""
    said = command_line(pid)
    return said is None or IN_TURN_VERB in said


def last_error_line(log_path: str, tail_bytes: int = 65536) -> Optional[str]:
    """Why an engine stopped, from its log — the line the operator would look for.

    The last exception of this start that is a reason, not a pointer to one. Found live: a vLLM
    process ended on `RuntimeError: Engine core initialization failed. See root cause above.`,
    which was all the pool heard, and the machine was gone before anyone could read what was
    above. A traceback's own source lines (`raise RuntimeError(`) are not exceptions; a pointer
    is kept only for want of anything better, and a line that merely mentions an error after
    that. Only what follows the last start's marker is read."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    starts = [i for i, line in enumerate(lines) if RUN_MARKER in line]
    if starts:
        lines = lines[starts[-1] + 1:]
    pointer: Optional[str] = None
    for line in reversed(lines):
        found = _EXCEPTION.search(line)
        if found:
            reason = f"{found.group(1)}: {found.group(2).strip()}"[:300]
        elif _BARE_EXCEPTION.fullmatch(_PREFIXES.sub("", line).strip()):
            reason = _PREFIXES.sub("", line).strip()[:300]
        else:
            continue
        if _POINTS_ELSEWHERE.search(reason):
            pointer = pointer or reason
            continue
        return reason
    if pointer:
        return pointer
    for line in reversed(lines):
        if "Error" in line or "error:" in line.lower():
            # vLLM prefixes every line with the process that wrote it; that is not the reason.
            cleaned = line.split(") ", 1)[1] if line.startswith("(") and ") " in line else line
            return cleaned.strip()[:300]
    return None


def failed_engines(models_dir: Path, served: Sequence[str],
                   alive: Optional[Callable[[int], bool]] = None,
                   command_line: Callable[[int], Optional[bytes]] = host_platform.command_line) -> dict[str, str]:
    """Models whose engine process is gone without ever serving them, and why, by the log.

    A process that is still up is loading, however long it takes; only one that has exited
    without its model being served has failed. The launcher's refusal counts for every model.
    A model with a copy on each card (D107) is served only once every copy answers, so one dead
    copy fails it, named with its card.
    """
    record = read_record(models_dir)
    if not record:
        return {}
    if record.get("refused"):
        return {name: record["refused"] for name in (record.get("plan") or {}).get("models", {})}

    def is_alive(pid: int) -> bool:
        if alive is not None:
            return alive(pid)
        # Asked of the system the way it answers safely: on Windows `os.kill(pid, 0)` would end
        # the process rather than look at it (`platform.alive`).
        return host_platform.alive(pid)

    failed: dict[str, str] = {}
    turns = (record.get("in_turn") or {}).get("pid")
    for engine in record.get("engines") or []:
        name, pid = engine.get("model"), engine.get("pid")
        if not name or name in served:
            continue
        if pid is None and engine.get("waiting"):
            # Started in turn, after the model before it on its card (D155): waiting is not
            # failing while what starts it is still there to do so.
            if isinstance(turns, int) and is_alive(turns) and in_turn_helper(turns, command_line):
                continue
            card = engine.get("card")
            on = "" if card is None else f" on card {card}"
            failed[name] = (f"its vLLM process{on} was never started: what starts the models on a "
                            "card one after another stopped before its turn")
            continue
        if not isinstance(pid, int):
            continue
        if is_alive(pid):
            continue
        reason = last_error_line(engine.get("log") or "") or "no reason in its log"
        card = engine.get("card")
        on = "" if card is None else f" on cards {card}" if "," in str(card) else f" on card {card}"
        failed[name] = f"its vLLM process{on} exited before serving it: {reason}"
    return failed


def capped_lengths(models_dir: Path) -> dict[str, dict[str, int]]:
    """Models the launcher started at a shorter `--max-model-len` than the pool asked for — their
    own maximum (D146) — with both numbers, by model. Nothing from a start it refused: nothing
    was started at any length."""
    record = read_record(models_dir)
    if not record or record.get("refused"):
        return {}
    models = (record.get("plan") or {}).get("models") or {}
    return {
        name: dict(planned["max_model_len_capped"])
        for name, planned in models.items()
        if isinstance(planned, dict) and isinstance(planned.get("max_model_len_capped"), dict)
    }
