"""What the vLLM launcher started, read back — and which of it has since died (D104).

The launcher (`vllm_launch`) starts processes, so nothing that answers the pool may import it
(D63). This module only reads: the launcher's record of what it started, each process's
liveness, and the last error line of a process's own log. The agent reports a process that has
exited without ever serving its model as that model's failure, with that line as the reason —
so the pool gives the host up saying why, instead of watching it "prepare" until its hold runs
out. And how each process the launcher starts is known again by its command line, and by the
boot of the machine it was started in — read here, so the launcher that signals and the agent
that only looks judge a number alike.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Collection, Optional, Sequence

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

#: The router's map of model to engine, in the launcher's state directory — where an older
#: launcher kept it, in the models directory itself.
UPSTREAMS_FILE = ".gpm-upstreams.json"

#: Where Linux says which boot this is: a new value at every start of the machine.
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"


def boot_id(path: str = BOOT_ID_PATH) -> Optional[str]:
    """This boot of the machine, where the system says (Linux) — None elsewhere. A process number
    recorded under another boot names nothing that runs now, whatever has that number since."""
    try:
        with open(path, encoding="ascii") as said:
            found = said.read(64).strip()
    except (OSError, ValueError):
        return None
    return found or None


def another_boot(record: Optional[dict[str, Any]], current: Optional[str]) -> bool:
    """Was `record` written under another boot of this machine? Only where both are known: a record
    from before boots were recorded, or a system that does not say, is judged by command lines."""
    recorded = (record or {}).get("boot_id")
    return isinstance(recorded, str) and current is not None and recorded != current


# --- what the launcher's processes look like (R5) ---
#
# A number in a file is only a number: after the machine restarts it may be any program's, the
# owner's own among them. Each kind of process the launcher starts is known by the exact shape of
# the argument list it starts it with — never by a word found somewhere in a command line.


def _forms(directory: str | Path) -> set[str]:
    """A directory as a command line may name it: as given, and with its links resolved."""
    given = os.path.normpath(os.path.expanduser(str(directory)))
    return {given, os.path.realpath(given)}


def _arguments(said: bytes) -> Optional[list[str]]:
    """A command line as its arguments, where the system gave them one by one (/proc, separated by
    NULs) — else None, and only the line joined by spaces is known (`ps`)."""
    if b"\0" not in said:
        return None
    listed = said.decode(errors="surrogateescape").split("\0")
    return listed[:-1] if listed and listed[-1] == "" else listed


def _model_path_in(path: str, models: set[str]) -> bool:
    """Is `path` a model directory directly inside the models directory?"""
    name = os.path.basename(os.path.normpath(path))
    if not name or name.startswith("."):
        return False
    parent = os.path.dirname(os.path.normpath(path))
    return parent in models or os.path.realpath(parent) in models


def is_engine(said: bytes, models_dir: str | Path, program: Optional[str] = None) -> bool:
    """Is this command line an engine the launcher started — `<vllm> serve <a model directory in
    this models directory> …` — where `<vllm>` is `program` when named, else any program named
    `vllm`?"""
    models = _forms(models_dir)
    listed = _arguments(said)
    if listed is not None:
        for at in range(len(listed) - 2):
            first = listed[at]
            runs = first == program if program is not None else os.path.basename(first) == "vllm"
            if runs and listed[at + 1] == "serve" and _model_path_in(listed[at + 2], models):
                return True
        return False
    line = f" {said.decode(errors='replace').strip()} "
    runs = re.escape(program) if program is not None else r"(?:\S*/)?vllm"
    return any(
        re.search(rf"\s{runs} serve {re.escape(models)}/[A-Za-z0-9][A-Za-z0-9._-]*\s", line)
        for models in models
    )


def is_router(said: bytes, models_dir: str | Path, port: Optional[int] = None) -> bool:
    """Is this command line the launcher's router — `… proxy --upstreams <its map in this models
    directory's state directory> …` (or, from an older launcher, in the models directory itself),
    on `port` where named?"""
    maps = {os.path.join(form, UPSTREAMS_FILE) for form in _forms(state_dir_for(models_dir)) | _forms(models_dir)}
    listed = _arguments(said)
    if listed is not None:
        found = any(
            listed[at:at + 2] == ["proxy", "--upstreams"] and listed[at + 2] in maps
            for at in range(len(listed) - 2)
        )
        return found and (port is None or listed[-2:] == ["--port", str(port)])
    line = f" {said.decode(errors='replace').strip()} "
    found = any(f" proxy --upstreams {path} " in line for path in maps)
    return found and (port is None or line.endswith(f" --port {port} "))


def is_helper(said: bytes, models_dir: str | Path) -> bool:
    """Is this command line the helper that starts a card's models in turn for this models
    directory — `… vllm-in-turn --models-dir <this models directory>`?"""
    models = _forms(models_dir)
    listed = _arguments(said)
    if listed is not None:
        return (len(listed) >= 3 and listed[-3:-1] == ["vllm-in-turn", "--models-dir"]
                and os.path.normpath(listed[-1]) in models)
    line = said.decode(errors="replace").strip()
    return any(line.endswith(f" vllm-in-turn --models-dir {form}") for form in models)


def in_turn_helper(pid: int, command_line: Callable[[int], Optional[bytes]], models_dir: str | Path) -> bool:
    """Is `pid` still the helper that starts models in turn for this models directory? A number in
    a file is only a number: once the helper has gone it may be any process's. Where the command
    line cannot be read, the number is taken at its word."""
    said = command_line(pid)
    return said is None or is_helper(said, models_dir)


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


def _liveness(alive: Optional[Callable[[int], bool]]) -> Callable[[int], bool]:
    def is_alive(pid: int) -> bool:
        if alive is not None:
            return alive(pid)
        # Asked of the system the way it answers safely: on Windows `os.kill(pid, 0)` would end
        # the process rather than look at it (`platform.alive`).
        return host_platform.alive(pid)

    return is_alive


def failed_engines(models_dir: Path, served: Sequence[str],
                   alive: Optional[Callable[[int], bool]] = None,
                   command_line: Callable[[int], Optional[bytes]] = host_platform.command_line,
                   served_before: Collection[str] = ()) -> dict[str, str]:
    """Models whose engine process is gone without serving them now, and why, by the log.

    A process that is still up is loading, however long it takes; only one that has exited
    without its model being served has failed. The launcher's refusal counts for every model.
    A model with a copy on each card (D107) is served only once every copy answers, so one dead
    copy fails it, named with its card. A model in `served_before` — the agent saw this start
    serve it — exited after serving, and is said so. What was started before the machine last
    restarted has not failed: it is simply not running (`another_boot`).
    """
    record = read_record(models_dir)
    if not record or another_boot(record, boot_id()):
        return {}
    if record.get("refused"):
        return {name: record["refused"] for name in (record.get("plan") or {}).get("models", {})}

    is_alive = _liveness(alive)
    failed: dict[str, str] = {}
    turns = (record.get("in_turn") or {}).get("pid")
    for engine in record.get("engines") or []:
        name, pid = engine.get("model"), engine.get("pid")
        if not name or name in served:
            continue
        if pid is None and engine.get("waiting"):
            # Started in turn, after the model before it on its card (D155): waiting is not
            # failing while what starts it is still there to do so.
            if isinstance(turns, int) and is_alive(turns) and in_turn_helper(turns, command_line, models_dir):
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
        when = "after serving it" if name in served_before else "before serving it"
        failed[name] = f"its vLLM process{on} exited {when}: {reason}"
    return failed


def loading_engines(models_dir: Path, served: Sequence[str],
                    alive: Optional[Callable[[int], bool]] = None,
                    command_line: Callable[[int], Optional[bytes]] = host_platform.command_line) -> set[str]:
    """Models the launcher started that are not served yet and are still on their way: every
    process started for one is still running, or it waits its turn on its card while what starts
    it in turn still runs (D155). Such a model is loading, not waiting for a start: asked to start
    again, a start that takes minutes would only begin again."""
    record = read_record(models_dir)
    if not record or record.get("refused") or another_boot(record, boot_id()):
        return set()
    is_alive = _liveness(alive)
    turns = (record.get("in_turn") or {}).get("pid")
    helper = isinstance(turns, int) and is_alive(turns) and in_turn_helper(turns, command_line, models_dir)
    going: dict[str, bool] = {}
    for engine in record.get("engines") or []:
        name, pid = engine.get("model"), engine.get("pid")
        if not name or name in served:
            continue
        if pid is None and engine.get("waiting"):
            on_its_way = helper
        else:
            on_its_way = isinstance(pid, int) and is_alive(pid)
        # Every copy of it (D107) must be on its way: one that died fails it (`failed_engines`).
        going[name] = going.get(name, True) and on_its_way
    return {name for name, on_its_way in going.items() if on_its_way}


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
