"""Start vLLM on this machine for whatever models are on disk (D97).

vLLM serves the model it was started with, and nothing else. So on a machine the pool rented,
"loading a model" means: the agent fetched its weights into the models directory, and the engine
is started again to pick them up. This module is that start. It runs on the machine, from the
agent's archive, when the machine's own restart script calls it — never over the agent's
protocol, and never with anything the pool said other than bounded numbers in the environment.

Two shapes, matching the pool's placement (D94, D96):

- **One model**: one vLLM process on the engine port, given nine tenths of the accelerator.
- **Several, behind the router**: one vLLM process per model on the ports above the engine port,
  the accelerator's memory split between them by the size of each model's weights (with a floor,
  so a small model still has room to run), and the router on the engine port in front of them.

It stops whatever it started last time before starting anything, so calling it twice is a
restart and not a second copy fighting the first for the same memory.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

# Written by the hub fetch when every file of a model has landed. A directory without it is a
# download in progress, and serving half a model is worse than serving none.
from .modelhub import COMPLETE_MARKER

log = logging.getLogger("gpm.vllm-start")


#: What this launcher started, so the next start can stop it first.
PIDS_FILE = ".gpm-vllm.pids"

#: The router's map of model to engine, when there is more than one model.
UPSTREAMS_FILE = ".gpm-upstreams.json"

#: The share of the accelerator one process may take. vLLM's own recommendation for a single
#: model; the rest is headroom the driver and the runtime need.
TOTAL_MEMORY_SHARE = 0.90

#: The least any one model is given when several share a card and the card's size is not
#: known. Proportional-to-weights alone would hand a 0.3 GB embedding model under 2% beside a
#: 19 GB one — too little to start at all.
MEMORY_FLOOR = 0.10

#: What a process needs beyond its weights, when the card's size is known: room for the cache
#: it batches in and for its own workspace. Found live: splitting a 48 GB card by weights alone
#: gave a 15 GB model 17.9 GB, and it refused to start for want of 2.1 GiB of cache. A model
#: is given its weights (with a tenth over for what loading them costs) plus this reserve; a set
#: that does not fit is refused before anything starts, rather than started and watched die.
WEIGHT_OVERHEAD = 1.10
CACHE_RESERVE_BYTES = 3 * 1024**3

# What this launcher started, by model — process, port, log — is read back by the agent to tell
# a process that has since died from one still loading. The reading lives apart from this
# module: the agent must never import anything that can start a process (D63).
from .vllm_state import (  # noqa: E402,F401
    STARTED_FILE,
    failed_engines,
    last_error_line,
    read_record,
)

#: The pool's numbers, under the names the agent writes them in (D41). Only those present are
#: passed on: an absent one means "the engine's own default", not zero.
NUMBER_FLAGS = (
    ("GPM_VLLM_MAX_NUM_SEQS", "--max-num-seqs"),
    ("GPM_VLLM_MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens"),
    ("GPM_VLLM_MAX_MODEL_LEN", "--max-model-len"),
)

#: The named options the pool may ask for (D100), and what each means for each model family —
#: the family read from the model's own `config.json` (`model_type`), on this machine. The pool
#: sends only a name from this table; the flags are written here and nowhere else, so nothing the
#: pool says becomes part of a command (D41). A family missing from an option's row starts
#: without it, and says so. Parser names checked against vLLM v0.29.0's registries.
OPTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    # Lets an app send `tools` and get tool calls back as structured `tool_calls`.
    "tool_calling": {
        "gemma4": ("--enable-auto-tool-choice", "--tool-call-parser", "gemma4"),
        "qwen2": ("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        "qwen3": ("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        "qwen3_moe": ("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        "gpt_oss": ("--enable-auto-tool-choice", "--tool-call-parser", "openai"),
    },
    # Returns a thinking model's reasoning apart from its answer, as `reasoning_content`.
    "reasoning": {
        "gemma4": ("--reasoning-parser", "gemma4"),
        "qwen3": ("--reasoning-parser", "qwen3"),
        "qwen3_moe": ("--reasoning-parser", "qwen3"),
    },
}


@dataclass
class Started:
    """What one call started, for the log and for the next call to stop."""

    engines: list[dict[str, Any]] = field(default_factory=list)
    proxy: Optional[dict[str, Any]] = None
    skipped: list[str] = field(default_factory=list)
    #: Options asked for that a model's family does not have, said out loud.
    not_applied: list[str] = field(default_factory=list)
    #: Why nothing was started, when the set could not have run on this card.
    refused: Optional[str] = None
    #: Each model's share of the card and what it was sized for, for the record.
    plan: dict[str, Any] = field(default_factory=dict)

    def pids(self) -> list[int]:
        found = [e["pid"] for e in self.engines if e.get("pid")]
        if self.proxy and self.proxy.get("pid"):
            found.append(self.proxy["pid"])
        return found


def served_name(directory: Path) -> str:
    """The name a model is served under: its repository, from the directory the fetch made.

    The fetch flattened `owner/name` to `owner__name` so that nothing a pool says can climb out
    of the models directory; this undoes exactly that and nothing more.
    """
    return directory.name.replace("__", "/", 1)


def complete_models(models_dir: Path) -> list[Path]:
    """Model directories whose every file has landed, in a stable order."""
    if not models_dir.is_dir():
        return []
    return sorted(
        path for path in models_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / COMPLETE_MARKER).exists()
    )


def size_of(directory: Path) -> int:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def memory_shares(sizes: Sequence[int]) -> list[float]:
    """Each model's share of the accelerator: proportional to its weights, floored, rescaled.

    Weights are a fair first proxy for what a model needs, and the floor keeps a small model
    runnable. It is still a split fixed at launch, which is the cost of this shape: the largest
    model gets a fraction of the cache it would have had to itself (D96).
    """
    if not sizes:
        return []
    if len(sizes) == 1:
        return [TOTAL_MEMORY_SHARE]
    total = sum(sizes) or len(sizes)
    floored = [max((size or 1) / total, MEMORY_FLOOR) for size in sizes]
    scale = sum(floored)
    return [round(TOTAL_MEMORY_SHARE * share / scale, 3) for share in floored]


def card_memory_bytes(run: Callable[..., Any] = subprocess.run) -> Optional[int]:
    """The accelerator's memory, from the driver — or None where it cannot be asked."""
    try:
        out = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    sizes = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    return min(sizes) * 1024 * 1024 if sizes else None


def memory_plan(sizes: Sequence[int], card_bytes: Optional[int]) -> tuple[list[float], Optional[str]]:
    """Each model's share of the card, or why the set cannot run on it.

    With the card's size known, each model is given its weights plus a cache reserve, and what
    is left of the launcher's share of the card is spread by weight — so the largest model gets
    the most cache. A set whose needs exceed that share is refused, with the arithmetic. Without
    the card's size, the old split by weights with a floor, which is a guess and says so.
    """
    if not sizes:
        return [], None
    if card_bytes is None:
        return memory_shares(sizes), None
    needs = [int(size * WEIGHT_OVERHEAD) + CACHE_RESERVE_BYTES for size in sizes]
    usable = TOTAL_MEMORY_SHARE * card_bytes
    if sum(needs) > usable:
        return [], (
            f"the models need {sum(needs) / 1e9:.1f} GB together (each its weights plus a "
            f"{CACHE_RESERVE_BYTES / 1024**3:.0f} GiB cache reserve) and this card gives "
            f"{usable / 1e9:.1f} GB; fewer models on this host, or a larger card"
        )
    spare = usable - sum(needs)
    total_weight = sum(sizes) or len(sizes)
    shares = [
        (need + spare * ((size or 1) / total_weight)) / card_bytes
        for need, size in zip(needs, sizes, strict=True)
    ]
    return [round(share, 3) for share in shares], None


def number_flags(env: Mapping[str, str]) -> list[str]:
    flags: list[str] = []
    for variable, flag in NUMBER_FLAGS:
        value = env.get(variable, "").strip()
        if value.isdigit():
            flags += [flag, value]
    return flags


def family_of(directory: Path) -> Optional[str]:
    """The model's family, as its own configuration names it — or None if it does not."""
    try:
        found = json.loads((directory / "config.json").read_text()).get("model_type")
    except (OSError, ValueError, AttributeError):
        return None
    return found if isinstance(found, str) else None


def option_flags(options: Sequence[str], family: Optional[str]) -> tuple[list[str], list[str]]:
    """The flags for `options` on a model of `family`, and the options that do not apply to it.

    A name not in the table is refused, not skipped: the command line's own choices already
    stop one arriving, so reaching here with one is a bug worth hearing about.
    """
    flags: list[str] = []
    missing: list[str] = []
    for option in options:
        if option not in OPTIONS:
            raise ValueError(f"unknown option {option!r}; known: {sorted(OPTIONS)}")
        row = OPTIONS[option].get(family or "")
        if row is None:
            missing.append(option)
            continue
        flags += row
    return flags, missing


def stop_previous(models_dir: Path, *, kill: Callable[[int, int], None] = os.kill,
                  alive: Optional[Callable[[int], bool]] = None, grace_s: float = 30.0) -> list[int]:
    """Stop what the last call started. A process already gone is not an error."""
    pids_file = models_dir / PIDS_FILE
    try:
        pids = [int(p) for p in json.loads(pids_file.read_text())]
    except (OSError, ValueError, TypeError):
        return []

    def is_alive(pid: int) -> bool:
        if alive is not None:
            return alive(pid)
        try:
            kill(pid, 0)
            return True
        except OSError:
            return False

    stopped = []
    for pid in pids:
        if not is_alive(pid):
            continue
        try:
            kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except OSError:
            continue
    deadline = time.monotonic() + grace_s
    while any(is_alive(pid) for pid in stopped) and time.monotonic() < deadline:
        time.sleep(0.5)
    for pid in stopped:
        if is_alive(pid):
            try:
                kill(pid, signal.SIGKILL)
            except OSError:
                pass
    pids_file.unlink(missing_ok=True)
    return stopped


def launch(
    models_dir: str | Path,
    port: int,
    *,
    proxy: bool = False,
    options: Sequence[str] = (),
    vllm: str = "vllm",
    agent: Optional[str] = None,
    python: str = sys.executable,
    env: Optional[Mapping[str, str]] = None,
    popen: Callable[..., Any] = subprocess.Popen,
    kill: Callable[[int, int], None] = os.kill,
    alive: Optional[Callable[[int], bool]] = None,
    card_bytes: Optional[int] = None,
    probe_card: Callable[[], Optional[int]] = card_memory_bytes,
) -> Started:
    """Stop what ran before, then start vLLM for every complete model on disk.

    Nothing on disk yet is the ordinary state of a machine that has just booted, and is not an
    error: this returns having started nothing, and is called again once the fetch has finished.
    """
    models_dir = Path(models_dir).expanduser()
    env = dict(os.environ if env is None else env)
    stop_previous(models_dir, kill=kill, alive=alive)
    (models_dir / STARTED_FILE).unlink(missing_ok=True)

    started = Started()
    found = complete_models(models_dir)
    if not found:
        return started

    if not proxy and len(found) > 1:
        # One model to a host without the router: the first, and the rest said out loud rather
        # than left to fight over the same memory.
        started.skipped = [served_name(d) for d in found[1:]]
        log.warning("several models on disk and no router; serving %s only", served_name(found[0]))
        found = found[:1]

    logs = models_dir / ".gpm-logs"
    logs.mkdir(parents=True, exist_ok=True)
    sizes = [size_of(d) for d in found]
    card = card_bytes if card_bytes is not None else probe_card()
    shares, refused = memory_plan(sizes, card)
    started.plan = {
        "card_bytes": card,
        "models": {served_name(d): {"weights_bytes": s} for d, s in zip(found, sizes, strict=True)},
    }
    if refused:
        # Said where the agent reads it, so the pool hears the reason instead of watching
        # processes die one by one.
        started.refused = refused
        log.error("not starting vLLM: %s", refused)
        _record(models_dir, started)
        return started
    upstreams: dict[str, str] = {}

    for index, (directory, share) in enumerate(zip(found, shares, strict=True)):
        name = served_name(directory)
        engine_port = port if not proxy else port + 1 + index
        family = family_of(directory)
        extra, missing = option_flags(options, family)
        for option in missing:
            started.not_applied.append(f"{option} for {name} (family {family or 'unknown'})")
            log.warning("%s has no %s in its family (%s); started without it", name, option, family)
        argv = [
            vllm, "serve", str(directory),
            "--served-model-name", name,
            # Loopback, always: the pool reaches the machine through a forward, and binding
            # anything wider only exposes the engine (D77).
            "--host", "127.0.0.1",
            "--port", str(engine_port),
            "--gpu-memory-utilization", str(share),
            *number_flags(env),
            *extra,
        ]
        log_path = logs / f"{directory.name}.log"
        out = open(log_path, "ab")
        process = popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        started.engines.append({"model": name, "port": engine_port, "memory_share": share,
                                "pid": getattr(process, "pid", None), "argv": argv, "log": str(log_path)})
        upstreams[name] = f"http://127.0.0.1:{engine_port}"

    if proxy:
        map_file = models_dir / UPSTREAMS_FILE
        draft = map_file.with_suffix(".new")
        draft.write_text(json.dumps(upstreams, indent=2, sort_keys=True))
        draft.replace(map_file)
        argv = [python, agent or sys.argv[0], "proxy", "--upstreams", str(map_file),
                "--host", "127.0.0.1", "--port", str(port)]
        out = open(logs / "proxy.log", "ab")
        process = popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        started.proxy = {"port": port, "pid": getattr(process, "pid", None), "argv": argv}

    (models_dir / PIDS_FILE).write_text(json.dumps(started.pids()))
    _record(models_dir, started)
    return started


def _record(models_dir: Path, started: Started) -> None:
    """What was started (or why nothing was), for the agent to read back."""
    record = {
        "at": time.time(),
        "refused": started.refused,
        "plan": started.plan,
        "engines": [{k: v for k, v in e.items() if k != "argv"} for e in started.engines],
        "proxy": {k: v for k, v in started.proxy.items() if k != "argv"} if started.proxy else None,
    }
    draft = (models_dir / STARTED_FILE).with_suffix(".new")
    draft.write_text(json.dumps(record, indent=2))
    draft.replace(models_dir / STARTED_FILE)


def add_arguments(parser: Any) -> None:
    parser.add_argument("--models-dir", required=True, help="where the agent fetched the models")
    parser.add_argument("--port", type=int, required=True, help="the port the pool dials")
    parser.add_argument("--proxy", action="store_true",
                        help="one engine per model, with the router on --port in front of them")
    parser.add_argument("--option", action="append", default=[], choices=sorted(OPTIONS),
                        help="a named option; each model gets it if its family has it (D100)")
    # Deliberately no way to name another program: this runs `vllm` and the agent's own router,
    # as fixed argument lists, and nothing on its command line can change which.


def main(args: Any) -> int:
    started = launch(args.models_dir, args.port, proxy=args.proxy, options=args.option)
    if started.refused:
        print(f"gpm-agent vllm-start: not started — {started.refused}")
        return 1
    if not started.engines:
        print("gpm-agent vllm-start: no complete model on disk yet; nothing started")
        return 0
    for engine in started.engines:
        print(f"started {engine['model']} on {engine['port']} (memory {engine['memory_share']})")
    if started.proxy:
        print(f"started the router on {started.proxy['port']}")
    for name in started.skipped:
        print(f"not started: {name} (several models on disk and no router)")
    for what in started.not_applied:
        print(f"not applied: {what}")
    return 0
