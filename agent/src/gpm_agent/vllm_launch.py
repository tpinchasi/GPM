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

#: The least any one model is given when several share a card. Proportional-to-weights alone
#: would hand a 0.3 GB embedding model under 2% beside a 19 GB one — too little to start at all.
MEMORY_FLOOR = 0.10

#: The pool's numbers, under the names the agent writes them in (D41). Only those present are
#: passed on: an absent one means "the engine's own default", not zero.
NUMBER_FLAGS = (
    ("GPM_VLLM_MAX_NUM_SEQS", "--max-num-seqs"),
    ("GPM_VLLM_MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens"),
    ("GPM_VLLM_MAX_MODEL_LEN", "--max-model-len"),
)


@dataclass
class Started:
    """What one call started, for the log and for the next call to stop."""

    engines: list[dict[str, Any]] = field(default_factory=list)
    proxy: Optional[dict[str, Any]] = None
    skipped: list[str] = field(default_factory=list)

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


def number_flags(env: Mapping[str, str]) -> list[str]:
    flags: list[str] = []
    for variable, flag in NUMBER_FLAGS:
        value = env.get(variable, "").strip()
        if value.isdigit():
            flags += [flag, value]
    return flags


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
    vllm: str = "vllm",
    agent: Optional[str] = None,
    python: str = sys.executable,
    env: Optional[Mapping[str, str]] = None,
    popen: Callable[..., Any] = subprocess.Popen,
    kill: Callable[[int, int], None] = os.kill,
    alive: Optional[Callable[[int], bool]] = None,
) -> Started:
    """Stop what ran before, then start vLLM for every complete model on disk.

    Nothing on disk yet is the ordinary state of a machine that has just booted, and is not an
    error: this returns having started nothing, and is called again once the fetch has finished.
    """
    models_dir = Path(models_dir).expanduser()
    env = dict(os.environ if env is None else env)
    stop_previous(models_dir, kill=kill, alive=alive)

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
    shares = memory_shares([size_of(d) for d in found])
    upstreams: dict[str, str] = {}

    for index, (directory, share) in enumerate(zip(found, shares, strict=True)):
        name = served_name(directory)
        engine_port = port if not proxy else port + 1 + index
        argv = [
            vllm, "serve", str(directory),
            "--served-model-name", name,
            # Loopback, always: the pool reaches the machine through a forward, and binding
            # anything wider only exposes the engine (D77).
            "--host", "127.0.0.1",
            "--port", str(engine_port),
            "--gpu-memory-utilization", str(share),
            *number_flags(env),
        ]
        out = open(logs / f"{directory.name}.log", "ab")
        process = popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        started.engines.append({"model": name, "port": engine_port, "memory_share": share,
                                "pid": getattr(process, "pid", None), "argv": argv})
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
    return started


def add_arguments(parser: Any) -> None:
    parser.add_argument("--models-dir", required=True, help="where the agent fetched the models")
    parser.add_argument("--port", type=int, required=True, help="the port the pool dials")
    parser.add_argument("--proxy", action="store_true",
                        help="one engine per model, with the router on --port in front of them")
    # Deliberately no way to name another program: this runs `vllm` and the agent's own router,
    # as fixed argument lists, and nothing on its command line can change which.


def main(args: Any) -> int:
    started = launch(args.models_dir, args.port, proxy=args.proxy)
    if not started.engines:
        print("gpm-agent vllm-start: no complete model on disk yet; nothing started")
        return 0
    for engine in started.engines:
        print(f"started {engine['model']} on {engine['port']} (memory {engine['memory_share']})")
    if started.proxy:
        print(f"started the router on {started.proxy['port']}")
    for name in started.skipped:
        print(f"not started: {name} (several models on disk and no router)")
    return 0
