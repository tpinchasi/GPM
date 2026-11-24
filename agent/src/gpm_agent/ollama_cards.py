"""One Ollama server per card (D190–D192).

Ollama keeps one loaded instance of a model, on one card when it fits. A machine with several
cards is given work for all of them (D107), so with mostly one model the extra work would queue on
one card while the others idle. vLLM runs a copy of each model on each card behind the agent's
router; this does the same for Ollama:

- **A server per card.** `ollama serve` once per NVIDIA card, each pinned to its card with
  `CUDA_VISIBLE_DEVICES`, each on its own loopback port — the ports just above the engine's — and
  the router (`proxy --engine ollama`) on the engine's port in front of them. Each is given its
  share of the host's workers (`OLLAMA_NUM_PARALLEL` divided by the servers, rounded up).
- **One store.** Every server reads the same models directory (`OLLAMA_MODELS`), so a model is
  pulled once — through the first server — and loaded by each. Only the first prunes the store as
  it starts (`OLLAMA_NOPRUNE` on the rest), so a server started again while the first pulls never
  takes away the download's unfinished layers.
- **Only where the models fit one card** (D191). The models held — as far as they are on disk —
  must fit the smallest card, by the vLLM launcher's own arithmetic (D104): weights and a tenth,
  and 3 GiB for each model's cache, within nine tenths of the card. Otherwise one server spans every
  card, as Ollama always ran: it spreads a model too large for one card across them itself.
- **Unchanged elsewhere.** A machine with one card, or none Ollama runs on through NVIDIA's driver —
  an Apple-silicon Mac, an AMD card, the processor alone — runs the one server it always did.

Two run it. On a machine whose owner lets the agent install engines, the agent's own runner
(`OllamaCards`, the installer's server) starts them and starts them again. On a machine the pool
rents, the pool's restart script runs `gpm-agent ollama-start` from the agent's archive (the same
class, from the command line), and the agent there only reads what it started. Either writes the
servers it started to `copies.json`, beside the models directory, never in it — which the router
reads, and the agent's model work reads to pull through the first and hold on each (D192).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import httpx

from . import facts as machine_facts
from . import platform as host_platform
from .ollama_server import RECORD_FILE, OllamaServer
from .vllm_state import state_dir_for

#: What was started, in the order it was: the cards, a URL each, and why this layout.
COPIES_FILE = "copies.json"
#: The router's record and log, and each card's server's, under the state directory.
ROUTER_DIR = "router"
CARD_DIR = "card-"
#: The one server a launcher started where the cards are not to be split, on a machine the pool
#: rents (the agent's own runner keeps its record where it always did).
SINGLE_DIR = "single"

#: The vLLM launcher's arithmetic (D104) and the pool's search (D111): weights and
#: a tenth, a cache reserve per model, within nine tenths of the card.
WEIGHT_OVERHEAD = 1.10
CACHE_RESERVE_BYTES = 3 * 1024**3
TOTAL_MEMORY_SHARE = 0.90

#: How long a look at the cards is kept. They do not change while a machine runs; `nvidia-smi` is
#: not free, and `wants_restart` looks every pass. A look that found none is asked again sooner:
#: a driver still bringing its cards up just after the machine started is no answer.
CARDS_KEPT_S = 300.0
NO_CARDS_KEPT_S = 60.0

#: Ollama's own names for a model that names no registry or namespace.
DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

#: How long the launcher gives each try to answer — the servers per card and the router, then the
#: one server if they did not start. Both, after the restart script's 30 s for the engine before
#: to stop, fit inside the 120 s the agent gives the pool's restart command (`restart_timeout_s`):
#: an Ollama server answers in seconds.
START_TIMEOUT_S = 30.0

#: The most cards given a server each; a machine with more runs the one server, saying so.
MOST_CARDS = 16

#: Never given to an engine the launcher starts (D184): the hub's read token the agent may carry.
WITHHELD = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT")

#: A server on this machine: `http://127.0.0.1:<port>`, nothing more.
_LOOPBACK = re.compile(r"http://127\.0\.0\.1:[0-9]{1,5}")


def state_dir(models_dir: str | Path, port: int) -> Path:
    """Where the servers of one engine port are recorded: beside the models directory, never in it
    (as vLLM's, D139), and by port, so two agents' Ollamas on one machine keep two."""
    return state_dir_for(models_dir) / f"ollama-{int(port)}"


def servers_path(models_dir: str | Path, port: int) -> Path:
    return state_dir(models_dir, port) / COPIES_FILE


def read_layout(models_dir: str | Path, port: int, *, look: bool = True) -> Optional[dict[str, Any]]:
    """What whoever runs this port's Ollama last wrote — `{cards, servers: [{card, url}], why}` —
    or None. Only loopback URLs are taken: every server runs on this machine.

    With `look`, also `running`: for servers per card, whether the router in front of them is
    running now — its recorded process alive and running exactly its recorded command line. The
    file is on the machine's disk and can outlive what it describes (a machine started again by its
    provider, D106); servers per card are taken to be there only while their router is."""
    try:
        raw = json.loads(servers_path(models_dir, port).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("servers"), list):
        return None
    servers = []
    for entry in raw["servers"]:
        if not (isinstance(entry, dict) and isinstance(entry.get("card"), str)
                and isinstance(entry.get("url"), str) and _LOOPBACK.fullmatch(entry["url"])):
            return None
        servers.append({"card": entry["card"], "url": entry["url"]})
    cards = raw.get("cards") if isinstance(raw.get("cards"), list) else []
    why = raw.get("why") if isinstance(raw.get("why"), str) else ""
    layout: dict[str, Any] = {"cards": [card for card in cards if isinstance(card, str)], "servers": servers,
                              "why": why}
    if look:
        layout["running"] = bool(servers) and router_running(models_dir, port)
    return layout


def router_running(models_dir: str | Path, port: int, *, os_name: Optional[str] = None) -> bool:
    """Whether the router this port's record names is running now: its process alive, on this
    port, running exactly the argument list recorded for it (on Windows, created when recorded)."""
    try:
        record = json.loads((state_dir(models_dir, port) / ROUTER_DIR / RECORD_FILE).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    pid, argv, created = record.get("pid"), record.get("argv"), record.get("created")
    if (not isinstance(pid, int) or record.get("port") != int(port) or not isinstance(argv, list)
            or not argv or not all(isinstance(part, str) for part in argv)):
        return False
    if not host_platform.alive(pid, os_name=os_name):
        return False
    return host_platform.recognised(pid, argv, created=created if isinstance(created, int) else None,
                                    os_name=os_name) is True


def copy_urls(models_dir: str | Path, port: int) -> list[str]:
    """The per-card servers' URLs, in order — empty where one server serves the port, or where
    the router in front of them is not running (the record is stale, or they are down)."""
    layout = read_layout(models_dir, port)
    return [entry["url"] for entry in layout["servers"]] if layout and layout["running"] else []


# --- the machine ---


#: Runs a fixed argument list and gives its output, or None: the facts' own (`facts._run`), which
#: never runs a shell, or a test's stand-in.
Runner = Callable[[Sequence[str]], Optional[str]]


def _smi(field: str, run: Runner) -> list[str]:
    out = run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"])
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


def nvidia_cards(env: Mapping[str, str], run: Runner = machine_facts._run) -> list[str]:
    """The NVIDIA cards to put a server on: those the machine's environment limits it to
    (`CUDA_VISIBLE_DEVICES`), or every card the driver lists — and none where the driver lists
    none, so a machine without an NVIDIA card (an AMD card, a Mac) is unchanged (D190)."""
    listed = [index for index in _smi("index", run) if index.isdigit()]
    if not listed:
        return []
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [device.strip() for device in visible.split(",") if device.strip()]
    return listed


def smallest_card_bytes(run: Runner = machine_facts._run) -> Optional[int]:
    """The smallest card's memory, from the driver — or None where it cannot be asked."""
    sizes = [int(line) for line in _smi("memory.total", run) if line.isdigit()]
    return min(sizes) * 1024 * 1024 if sizes else None


# --- what is on disk ---


def manifest_path(models_dir: str | Path, tag: str) -> Optional[Path]:
    """Where Ollama keeps `tag`'s manifest in its store: `manifests/<registry>/<namespace>/<name>/<tag>`,
    with Ollama's defaults for the parts a tag leaves out. None for a tag that is not plain."""
    name, _, version = tag.rpartition(":") if ":" in tag.rsplit("/", 1)[-1] else (tag, "", "latest")
    parts = name.split("/")
    if len(parts) == 1:
        parts = [DEFAULT_REGISTRY, DEFAULT_NAMESPACE, *parts]
    elif len(parts) == 2:
        parts = [DEFAULT_REGISTRY, *parts]
    if len(parts) != 3 or not all(_PART.fullmatch(part) and ".." not in part for part in [*parts, version]):
        return None
    return Path(models_dir).expanduser() / "manifests" / Path(*parts) / version


def size_on_disk(models_dir: str | Path, tag: str) -> Optional[int]:
    """`tag`'s size as its manifest gives it — every layer and its configuration — or None where it
    is not on disk (or its manifest cannot be read)."""
    path = manifest_path(models_dir, tag)
    if path is None:
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("layers"), list):
        return None
    total = 0
    for item in [*manifest["layers"], manifest.get("config")]:
        size = item.get("size") if isinstance(item, dict) else None
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            total += size
    return total or None


def tags_on_disk(models_dir: str | Path) -> list[str]:
    """Every model in the store, by tag as Ollama names it — what a machine the pool rents holds,
    since nothing but the pool's models is put there."""
    root = Path(models_dir).expanduser() / "manifests"
    found = []
    try:
        for path in root.glob("*/*/*/*"):
            if not path.is_file():
                continue
            registry, namespace, name, version = path.relative_to(root).parts
            if registry == DEFAULT_REGISTRY and namespace == DEFAULT_NAMESPACE:
                found.append(f"{name}:{version}")
            elif registry == DEFAULT_REGISTRY:
                found.append(f"{namespace}/{name}:{version}")
            else:
                found.append(f"{registry}/{namespace}/{name}:{version}")
    except OSError:
        return []
    return sorted(found)


# --- the layout ---


@dataclass(frozen=True)
class Layout:
    #: A server on each of these cards, behind the router; empty for one server across every card.
    cards: tuple[str, ...]
    why: str


def plan(cards: Sequence[str], sizes: Mapping[str, Optional[int]], card_bytes: Optional[int]) -> Layout:
    """A server per card where the machine has more than one NVIDIA card and the models held fit
    the smallest (D191); otherwise one server across every card. A model not on disk yet is counted
    once it is: the layout is looked at again as models land."""
    if len(cards) < 2:
        return Layout((), "one card" if cards else "no NVIDIA card")
    if len(cards) > MOST_CARDS:
        return Layout((), f"{len(cards)} cards, more than the {MOST_CARDS} given a server each, so one server spans them")
    if card_bytes is None:
        return Layout((), "the cards' memory could not be read, so one server spans every card")
    known = {tag: size for tag, size in sizes.items() if size}
    need = sum(int(size * WEIGHT_OVERHEAD) + CACHE_RESERVE_BYTES for size in known.values())
    usable = int(TOTAL_MEMORY_SHARE * card_bytes)
    if need > usable:
        return Layout((), (
            f"the models held ({', '.join(sorted(known))}) need {need / 1024**3:.1f} GiB of a card, with a "
            f"3 GiB cache reserve each — more than nine tenths of the smallest card's "
            f"{card_bytes / 1024**3:.1f} GiB — so one server spans every card"
        ))
    return Layout(tuple(cards), f"{len(cards)} cards, an Ollama server on each")


def agent_command() -> list[str]:
    """How the router runs: this agent again, as this process was run — `python -m gpm_agent.cli`
    under the same guard where it runs as a module, else its archive or program, with this
    interpreter."""
    main_module = sys.modules.get("__main__")
    spec = getattr(main_module, "__spec__", None)
    guard = ["-I"] if sys.flags.isolated else ["-P"]
    if spec is not None and spec.name == "gpm_agent.cli":
        return [sys.executable, *guard, "-m", "gpm_agent.cli"]
    if sys.argv and sys.argv[0].endswith(".pyz"):
        return [sys.executable, sys.argv[0]]
    return [sys.executable, *guard, "-m", "gpm_agent.cli"]


class OllamaCards:
    """Ollama's server as the installer drives it (`installs.Server`) — one per card behind the
    router where the machine's cards and the models held allow (D190, D191), and otherwise the one
    server it always was, whose record stays where it always was (`single_dir`)."""

    def __init__(
        self,
        single_dir: Path,
        *,
        engine_url: str,
        models_path: str,
        client: httpx.AsyncClient,
        os_name: Optional[str] = None,
        environment: Optional[Callable[[], dict[str, str]]] = None,
        cards: Optional[Callable[[], list[str]]] = None,
        card_bytes: Optional[Callable[[], Optional[int]]] = None,
        command: Optional[Sequence[str]] = None,
        copy_ports: Optional[Callable[[int], list[int]]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.single = OllamaServer(single_dir, engine_url=engine_url, models_path=models_path, client=client,
                                   os_name=os_name, environment=environment)
        self.port = self.single.port
        self.engine_url = engine_url
        self.models_path = models_path
        self.client = client
        self.os_name = os_name
        self.state = state_dir(models_path, self.port)
        self.foreign_reason = self.single.foreign_reason
        self._cards = cards or (lambda: nvidia_cards(os.environ))
        self._card_bytes = card_bytes or smallest_card_bytes
        self._command = list(command) if command else agent_command()
        self._copy_ports = copy_ports or (lambda count: [self.port + 1 + i for i in range(count)])
        self.clock = clock
        #: The models the pool told this machine to hold — None until it has said (`app.create_app`
        #: reads them from the model work; the launcher, every model on disk).
        self.held: Callable[[], Optional[Sequence[str]]] = lambda: None
        #: Why the last start ran the layout it did.
        self.why: Optional[str] = None
        self._looked: Optional[tuple[float, list[str], Optional[int]]] = None
        #: The plan whose servers did not start: not tried again until the plan changes (a card, a
        #: model), so a machine where they cannot start is not restarted every pass.
        self._failed: Optional[tuple[Any, ...]] = None
        #: Each card's server and the router, by the directory of its record (`_member`).
        self._members: dict[Path, OllamaServer] = {}

    # --- the pool's numbers: the one server's, as always ---

    def pool_environment(self) -> Optional[dict[str, str]]:
        return self.single.pool_environment()

    def set_pool_environment(self, environment: dict[str, str]) -> bool:
        return self.single.set_pool_environment(environment)

    def environment(self) -> dict[str, str]:
        return self.single.environment()

    # --- the machine, and the plan ---

    def machine(self) -> tuple[list[str], Optional[int]]:
        """The cards and the smallest card's memory, looked at again once the last look is old."""
        now = self.clock()
        if self._looked is not None:
            at, cards, size = self._looked
            if now - at < (CARDS_KEPT_S if cards else NO_CARDS_KEPT_S):
                return cards, size
        cards = list(self._cards())
        size = self._card_bytes() if len(cards) > 1 else None
        self._looked = (now, cards, size)
        return cards, size

    def _plan(self) -> tuple[Layout, tuple[Any, ...]]:
        """The plan for the models held — or, before the pool has said which (an agent just
        started), for every model in the store, as a machine the pool rents is planned; and with
        nothing in the store either, one server until it is told: a set that does not fit one card
        must never be split on a guess and pinned partly on the processor (D191)."""
        cards, size = self.machine()
        held = self.held()
        told = held is not None
        tags = list(held) if told else tags_on_disk(self.models_path)
        sizes = {tag: size_on_disk(self.models_path, tag) for tag in tags}
        key = (tuple(cards), told, tuple(sorted((tag, value or 0) for tag, value in sizes.items())))
        if not told and not tags and len(cards) > 1:
            return Layout((), "the models this machine holds are not known yet, so one server spans every card "
                              "until the pool says which"), key
        return plan(cards, sizes, size), key

    def plan(self) -> Layout:
        return self._plan()[0]

    # --- what runs ---

    def recorded(self) -> Optional[dict[str, Any]]:
        return read_layout(self.models_path, self.port, look=False)

    def _member(self, directory: Path, *, port: Optional[int] = None, environment=None, argv=None,
                what: str = "ollama serve") -> OllamaServer:
        """The server whose record is in `directory` — the same object each time, while its port is
        the same: one this agent started is its child, and is reaped only by asking the process it
        holds (a child nobody reaps stays a zombie, which no one can signal on some systems)."""
        known = self._members.get(directory)
        if known is not None and known.port == (self.port if port is None else int(port)):
            if environment is not None:
                known._environment = environment
            return known
        member = OllamaServer(directory, engine_url=self.engine_url, models_path=self.models_path, client=self.client,
                              os_name=self.os_name, port=port, environment=environment, argv=argv, what=what)
        self._members[directory] = member
        return member

    def _router(self) -> OllamaServer:
        upstreams = str(self.state / COPIES_FILE)

        def argv(program: Path) -> list[str]:
            return [*self._command, "proxy", "--engine", "ollama", "--upstreams", upstreams,
                    "--port", str(self.port)]

        return self._member(self.state / ROUTER_DIR, argv=argv, what="the router in front of the Ollama servers",
                            environment=lambda: host_platform.program_environment({}, os_name=self.os_name))

    def _copies(self) -> list[OllamaServer]:
        """Each card's server this machine has a record of, on the port it was started on."""
        found = []
        for directory in sorted(self.state.glob(f"{CARD_DIR}*")) if self.state.is_dir() else []:
            try:
                port = json.loads((directory / "server.json").read_text()).get("port")
            except (OSError, ValueError, AttributeError):
                continue
            if isinstance(port, int):
                found.append(self._member(directory, port=port))
        return found

    def ours(self, *, fresh: bool = False) -> bool:
        """Is what this machine's layout runs all running, and this agent's: every card's server
        and the router — or the one server?"""
        layout = self.recorded()
        if layout and layout["servers"]:
            copies = self._copies()
            return (len(copies) == len(layout["servers"]) and all(copy.ours(fresh=fresh) for copy in copies)
                    and self._router().ours(fresh=fresh))
        return self.single.ours(fresh=fresh)

    def running_version(self) -> Optional[str]:
        layout = self.recorded()
        if layout and layout["servers"]:
            return self._router().running_version()
        return self.single.running_version()

    def why_stopped(self) -> Optional[str]:
        return self.single.why_stopped()

    async def answers(self) -> Optional[str]:
        return await self.single.answers()

    async def listening(self) -> bool:
        return await self.single.listening()

    async def foreign(self) -> bool:
        """Something this agent did not start answers on the engine's port: neither the one server
        nor the router is its own."""
        if self.single.ours() or self._router().ours():
            return False
        return await self.single.listening()

    def _port_taken(self, port: int) -> bool:
        """Whether a card's server could not listen on `port`: asked by binding it, as the server
        will — not by dialling it, which a connection that happens to leave from that very port
        answers itself. Released at once; the server binds it a moment later."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if (self.os_name or host_platform.system()) != host_platform.WINDOWS:
                # As servers bind on POSIX: a port with only closed connections lingering is free.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", int(port)))
            except OSError:
                return True
        return False

    def _write(self, layout: Layout, ports: Sequence[int] = ()) -> None:
        host_platform.private_dir(self.state, parents=True, exist_ok=True)
        body = {
            "cards": list(layout.cards),
            "servers": [{"card": card, "url": f"http://127.0.0.1:{port}"} for card, port in zip(layout.cards, ports, strict=False)],
            "why": layout.why,
        }
        target = self.state / COPIES_FILE
        draft = target.with_name(target.name + ".new")
        draft.write_text(json.dumps(body, indent=2))
        os.replace(draft, target)

    def _forget(self) -> None:
        (self.state / COPIES_FILE).unlink(missing_ok=True)

    def _copy_environment(self, base: Mapping[str, str], index: int, card: str, port: int, count: int) -> dict[str, str]:
        environment = dict(base)
        environment["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        environment["CUDA_VISIBLE_DEVICES"] = card
        workers = environment.get("OLLAMA_NUM_PARALLEL", "")
        if workers.isdigit() and int(workers) > 0:
            # The host's workers, shared between its servers — rounded up, as vLLM's copies (D107).
            environment["OLLAMA_NUM_PARALLEL"] = str(math.ceil(int(workers) / count))
        if index:
            # Only the first prunes the shared store as it starts: the others never take away the
            # unfinished layers of a pull the first is making.
            environment["OLLAMA_NOPRUNE"] = "1"
        return environment

    # --- starting and stopping ---

    async def start(self, program: Path, version: Optional[str], timeout_s: float) -> Optional[str]:
        """Start the layout the plan says, and wait for it to answer as `version` (any, where None).
        None once it does; otherwise why not. Servers per card that do not start are stopped and
        one server is started instead, saying why — a machine is never left without an engine
        for want of its cards. Each try has `timeout_s` of its own: a card's server that hung to
        the end of its wait does not leave the one server a second to start in."""
        deadline = self.clock() + timeout_s
        try:
            await self.stop()  # whatever of a layout still runs: a card's server that died leaves the rest
        except OSError as exc:
            return f"what ran before could not be stopped: {exc}"
        layout, key = self._plan()
        several = len(self.machine()[0]) > 1  # a choice was made against them, and is said
        if layout.cards and self._failed != key:
            problem = await self._start_copies(program, version, layout, deadline)
            if problem is None:
                self._failed, self.why = None, layout.why
                return None
            self._failed = key
            layout = Layout((), f"a server on each of {len(layout.cards)} cards did not start ({problem}); "
                                "one server spans every card")
        elif layout.cards:
            layout = Layout((), "a server on each card did not start before, and nothing has changed since; "
                                "one server spans every card")
        if several:
            self._write(layout)
        else:
            self._forget()
        self.why = layout.why
        return await self.single.start(program, version, timeout_s)

    async def _start_copies(self, program: Path, version: Optional[str], layout: Layout, deadline: float) -> Optional[str]:
        ports = self._copy_ports(len(layout.cards))
        for port in ports:
            if self._port_taken(port):
                return f"port {port}, for a card's server, is taken"
        base = self.single.environment()
        copies = [
            self._member(self.state / f"{CARD_DIR}{index}", port=port, what=f"ollama serve on card {card}",
                         environment=(lambda e=self._copy_environment(base, index, card, port, len(ports)): e))
            for index, (card, port) in enumerate(zip(layout.cards, ports, strict=True))
        ]
        try:
            self._write(layout, ports)
        except OSError as exc:
            return f"could not record the servers in {self.state}: {exc}"
        remaining = max(1.0, deadline - self.clock())
        said = await asyncio.gather(*(copy.start(program, version, remaining) for copy in copies))
        problems = [f"card {card}: {problem}" for card, problem in zip(layout.cards, said, strict=True) if problem]
        if not problems:
            router = self._router()
            problem = await router.start(Path(self._command[0]), version, max(1.0, deadline - self.clock()))
            if problem is None:
                return None
            problems = [f"the router: {problem}"]
        try:
            await self.stop()
        except OSError as exc:
            problems.append(f"and what started could not be stopped: {exc}")
        return "; ".join(problems)

    async def stop(self) -> None:
        """Stop everything this machine's layout runs — the router first, then each card's server,
        then the one server — and forget the layout. An error from the system is raised once all
        have been tried, as `OllamaServer.stop` raises one."""
        errors: list[BaseException] = []
        for group in ([self._router()], self._copies(), [self.single]):
            for outcome in await asyncio.gather(*(member.stop() for member in group), return_exceptions=True):
                if isinstance(outcome, BaseException):
                    errors.append(outcome)
        if errors:
            raise OSError("; ".join(str(error) or type(error).__name__ for error in errors))
        layout = self.recorded()
        if layout and layout["servers"]:
            self._forget()

    def wants_restart(self) -> bool:
        """Whether what runs is not the layout the plan says now — a model that has landed and does
        not fit one card, or one gone that let the set fit (D191). Never for a plan whose servers
        did not start, until it changes; never while nothing of this agent's runs; and never before
        the pool has said what this machine holds — an agent just restarted would otherwise start
        its servers again on a guess, and again once told."""
        if self.held() is None or not self.ours():
            return False
        layout = self.recorded()
        running = tuple(layout["cards"]) if layout and layout["servers"] else ()
        planned, key = self._plan()
        if planned.cards == running:
            return False
        return not (planned.cards and self._failed == key)


# --- the launcher, on a machine the pool rents ---


def launcher_environment(environ: Mapping[str, str], models_dir: str, port: int, *, pinned: bool) -> dict[str, str]:
    """What the servers the launcher starts run with: the start-up's own environment — as the
    image's Ollama always did, the pool's numbers among it (`engine.env`) — less the hub's token
    (D184); the models directory every server shares; never Ollama's cloud (D168); and, on a
    machine whose models are held for good, nothing ever unloaded for being idle."""
    environment = {name: value for name, value in environ.items() if name not in WITHHELD}
    environment["OLLAMA_MODELS"] = models_dir
    environment["OLLAMA_NO_CLOUD"] = "1"
    environment["OLLAMA_HOST"] = f"127.0.0.1:{int(port)}"
    if pinned:
        environment.setdefault("OLLAMA_KEEP_ALIVE", "-1")
    return environment


async def _launch(verb: str, args: argparse.Namespace) -> int:
    models = str(Path(args.models_dir).expanduser())
    environment = launcher_environment(os.environ, models, args.port, pinned=args.pinned)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=10) as client:
        cards = OllamaCards(state_dir(models, args.port) / SINGLE_DIR, engine_url=f"http://127.0.0.1:{args.port}",
                            models_path=models, client=client, environment=lambda: dict(environment))
        cards.held = lambda: tags_on_disk(models)
        try:
            await cards.stop()
        except OSError as exc:
            print(f"gpm-agent {verb}: what ran before could not be stopped: {exc}", file=sys.stderr)
            return 1
        if verb == "ollama-stop":
            print(f"gpm-agent {verb}: stopped")
            return 0
        found = shutil.which("ollama", path=environment.get("PATH"))
        if found is None:
            print(f"gpm-agent {verb}: ollama is not on the PATH", file=sys.stderr)
            return 1
        if await cards.listening():
            # Never over something it did not start: the image's own server, say. It goes on serving.
            if len(cards.machine()[0]) > 1:
                cards._write(Layout((), f"port {args.port} is held by an Ollama this launcher did not start, "
                                        "so one server spans every card"))
            print(f"gpm-agent {verb}: {cards.foreign_reason}", file=sys.stderr)
            return 1
        problem = await cards.start(Path(found), None, args.timeout)
        print(f"gpm-agent {verb}: {cards.why}")
        if problem:
            # Whatever it started and did not answer in time is stopped: the restart script's one
            # server, started next, must not race it for the port.
            try:
                await cards.stop()
            except OSError as exc:
                problem = f"{problem}; and what it started could not be stopped: {exc}"
            print(f"gpm-agent {verb}: {problem}", file=sys.stderr)
            return 1
        return 0


def run_verb(verb: str, argv: Sequence[str]) -> int:
    """`gpm-agent ollama-start` and `ollama-stop`: what the pool's restart script runs on a machine
    it rents with more than one card (D190). Never from the protocol: the script is the pool's own
    start-up material, fixed at installation (D63)."""
    parser = argparse.ArgumentParser(prog=f"gpm-agent {verb}")
    parser.add_argument("--port", type=int, default=11434, help="the engine's port, where the router listens")
    parser.add_argument("--models-dir", default="~/.ollama/models",
                        help="the models directory every server shares (the agent's own default)")
    parser.add_argument("--pinned", action="store_true",
                        help="the models are held for good: no server unloads one for being idle")
    parser.add_argument("--timeout", type=float, default=START_TIMEOUT_S)
    args = parser.parse_args(list(argv))
    if not 0 < args.port < 65536:
        parser.error("--port must be a port")
    return asyncio.run(_launch(verb, args))
