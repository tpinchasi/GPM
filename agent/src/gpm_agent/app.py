"""The agent's HTTP surface: a closed list of verbs, every one behind the agent key.

docs/spec/host-agent.md §4, §7. Six verbs: report facts, hold this set of model tags, delete
this model tag, restart the engine, postpone the dead-man timer, and set the engine's version.
None takes a command, a path or a URL from the caller, and none ever will: the restart runs the
owner's command or the agent's own server, its settings are bounded integers, and a version is a
version — the agent's own table says where it comes from (D151).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Optional, Sequence
from urllib.parse import urlsplit

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import PROTOCOL_VERSION, __version__, installs, vllm_install, vllm_server, whisper_cpp
from . import facts as facts_module
from . import platform as host_platform
from .engine_control import EngineControl
from .engines import VllmFacts, engine_facts
from .facts import Probes, disk, gather
from .models import ModelWork, Refused
from .ollama_server import ANSWER_TIMEOUT_S as OLLAMA_ANSWER_TIMEOUT_S
from .ollama_server import OllamaServer
from .settings import AUTO, AUTO_ENGINES, FOREIGN_PREFIXES, INSTALLABLE_ENGINES, Settings
from .vllm_server import START_TIMEOUT_S as VLLM_START_TIMEOUT_S
from .vllm_server import VllmServer
from .whisper_cpp import InstalledServer, WhisperCppFacts

log = logging.getLogger("gpm_agent")

#: How often the agent looks at a server it runs itself, to start it again if it has stopped —
#: between the pool's passes, which do the same, so a machine the pool is not asking still serves.
KEEP_EVERY_S = 30.0


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)


def engine_dir_for(engines: Path, settings: Settings) -> Path:
    """Where this agent keeps its engine's versions and records: `<engines>/<engine>/<port>`. By
    port, so two agents on one machine — two Ollamas, or two whisper.cpp servers, one build each —
    each keep their own record of what they run, and neither removes a version the other runs."""
    return engines / settings.engine / str(urlsplit(settings.engine_url).port)


def engines_dir_of(settings: Settings) -> Path:
    """The engines directory: where the owner moved it (`engines_path`), else the system's place."""
    return Path(settings.engines_path).expanduser() if settings.engines_path else host_platform.engines_dir()


def server_for(
    engine_name: str, settings: Settings, engine_client: httpx.AsyncClient, engine: Any = None,
) -> Optional[installs.Server]:
    """The server this agent runs for `engine_name`, as the installer drives it — known by its own
    record in the engine's directory, so it is found again after the agent restarts. For
    whisper.cpp, `engine`'s own server, which then takes its program from the installer."""
    engines = engines_dir_of(settings)
    engine_dir = engine_dir_for(engines, dataclasses.replace(settings, engine=engine_name))
    if engine_name == "whisper-cpp":
        if not isinstance(engine, WhisperCppFacts):
            return None  # nothing to drive: its server is the engine's own
        return InstalledServer(engine, engine_client)
    if engine_name == "vllm":
        return VllmServer(
            engine_dir, engine_url=settings.engine_url, models_path=settings.models_path, client=engine_client,
            engines_dir=engines,
        )
    return OllamaServer(
        engine_dir, engine_url=settings.engine_url, models_path=settings.models_path, client=engine_client,
    )


def default_installer(
    settings: Settings, engine_client: httpx.AsyncClient, engine: Any = None
) -> Optional[installs.Installer]:
    """The installer for this machine's engine, where its owner allows installs (D150) — else None.
    For whisper.cpp it drives `engine`'s own server, which then takes its program from it."""
    if not settings.install_engines or settings.engine not in INSTALLABLE_ENGINES:
        return None
    engines = engines_dir_of(settings)
    source = installs.source_for(settings.engine, installs.this_machine(), tools_dir=engines / installs.TOOLS)
    if source is None:
        return None
    server = server_for(settings.engine, settings, engine_client, engine)
    if server is None:
        return None
    # vLLM serves once its models are loaded, which takes minutes, not Ollama's second.
    timeout_s = VLLM_START_TIMEOUT_S if settings.engine == "vllm" else installs.SWITCH_TIMEOUT_S
    installer = installs.Installer(
        source, engine_dir_for(engines, settings), server, min_free_disk_gb=settings.min_free_disk_gb,
        switch_timeout_s=timeout_s, switch_bound_s=switch_bound_s(settings.engine, timeout_s),
    )
    if isinstance(engine, WhisperCppFacts):
        engine.use_installer(installer)
    return installer


def switch_bound_s(engine: str, switch_timeout_s: float, os_name: Optional[str] = None) -> float:
    """The longest one switch of `engine`'s version — or one start or restart of it — takes this
    agent from its beginning until it has an outcome, as its facts say it (`engine.install`): what
    the pool waits for before it judges a switch, a rollback or an operator's restart, which for
    vLLM is far longer than for the others.

    Sized for a switch that fails, whose whole path is one switch: the server stopped, the port
    looked at, the version started and waited for up to `switch_timeout_s` — and, as it did not
    answer, stopped again and the one before started and waited for as long. Ollama's and
    whisper.cpp's servers stop by `platform.stop`: asked, given their grace, made to, and waited for
    as long again — on Windows after `taskkill`, which may take a command's whole time;
    whisper.cpp's start first stops one a pass started meanwhile. vLLM's own is
    `vllm_server.switch_bound_s`."""
    if engine == "vllm":
        return vllm_server.switch_bound_s(switch_timeout_s)
    grace = whisper_cpp.STOP_GRACE_S if engine == "whisper-cpp" else host_platform.STOP_GRACE_S
    stop = 2 * grace + (host_platform.COMMAND_TIMEOUT_S if (os_name or host_platform.system()) == host_platform.WINDOWS else 0)
    look = max(OLLAMA_ANSWER_TIMEOUT_S, whisper_cpp.ANSWER_TIMEOUT_S)
    start = stop + look + switch_timeout_s + look
    return 2 * (stop + start) + look


# --- an engine of `auto` (D161) ---

#: Where an agent of `auto` records the engine it chose, beside the engines' own directories.
CHOICE_FILE = "choice.json"


def this_python() -> str:
    """The agent's own Python minor version — the one a vLLM version's environment is made with,
    as the installer names it (`vllm_install.this_python`), so `auto` never chooses vLLM for an
    interpreter the installer would find no set for."""
    return vllm_install.this_python()


def choose_engine(
    machine: Mapping[str, Any], *, python: Optional[str] = None,
    sets: Optional[Sequence[vllm_install.LockedSet]] = None,
    glibc: Optional[Callable[[], Optional[str]]] = None,
) -> dict[str, Any]:
    """`engine.choice` for an agent of `auto` (D161): vLLM where the machine runs the builds
    catalogs carry for it — Linux on x86_64 or aarch64, every NVIDIA card Ampere or newer, a
    driver at the newest locked set's floor, a set for this Python at all, and the glibc that set's
    wheels need — and Ollama anywhere else, with every reason against vLLM. From the machine alone:
    its facts (`facts.gather`), as the pool reads them, and its C library (`glibc`, else
    `vllm_install.machine_glibc`) — every reason the installer would refuse the set for."""
    os_name, arch = machine.get("os"), machine.get("arch")
    python = python or this_python()
    mine = vllm_install.sets_for(python, arch or "", sets)
    floor = mine[0].driver_floor if mine else None
    reasons = vllm_install.machine_can_run(
        machine.get("accelerators") or [], os_name=os_name or "", arch=arch or "",
        minimum=vllm_install.AUTO_MIN_CAPABILITY, driver_floor=floor,
    )
    if os_name == "Linux" and arch in vllm_install.PLATFORMS:
        if not mine:
            # Nothing to install vLLM from here: the sets are generated per Python and processor.
            reasons.append(f"the agent carries no vLLM set for Python {python} on {arch}")
        elif mine[0].glibc is not None:
            # The newest set's wheels are built for a manylinux level: an older glibc (RHEL 8's
            # 2.28) would have every install refused, for ever.
            need, have = mine[0].glibc, (glibc or vllm_install.machine_glibc)()
            if have is None:
                reasons.append(f"vLLM's wheels need glibc {need} or later, and this machine's C library is not glibc")
            elif (vllm_install.driver_key(have) or ()) < (vllm_install.driver_key(need) or ()):
                reasons.append(f"vLLM's wheels need glibc {need} or later, and this machine has {have}")
    return {"engine": "ollama" if reasons else "vllm", "reasons": reasons}


#: How long `nvidia-smi` is given, in all, when an agent of `auto` asks again at its start: a
#: driver bringing its cards up — just after the machine started — can take far longer than the
#: look the facts take on every pass (5 s), and that look failing is no answer about the cards.
PATIENT_PROBE_S = 60.0

#: What says an NVIDIA driver is installed even where `nvidia-smi` did not answer.
NVIDIA_DRIVER_FILE = "/proc/driver/nvidia/version"


def _cards_unread(probes: Probes, machine: Mapping[str, Any], which: Callable[[str], Optional[str]]) -> bool:
    """On Linux, is an NVIDIA driver there — `nvidia-smi` on the PATH, or the kernel's own file
    for it — while its cards could not be read: none listed, or one without its compute capability
    or driver's version?"""
    if probes.system() != "Linux":
        return False
    if which("nvidia-smi") is None and probes.read(NVIDIA_DRIVER_FILE) is None:
        return False
    cards = [card for card in machine.get("accelerators") or [] if card.get("kind") == "cuda"]
    return not cards or any(
        vllm_install.capability(card.get("compute_capability")) is None
        or vllm_install.driver_key(card.get("driver")) is None
        for card in cards
    )


def _patient(probes: Probes) -> Probes:
    """The same probes, given `PATIENT_PROBE_S` in all for the commands they run — or a test's own
    probes as they are, which run nothing."""
    if probes.run is not facts_module._run:
        return probes
    deadline = time.monotonic() + PATIENT_PROBE_S

    def patiently(command: Sequence[str]) -> Optional[str]:
        return facts_module._run(command, timeout_s=max(1.0, deadline - time.monotonic()))

    return dataclasses.replace(probes, run=patiently)


def recorded_choice(path: Path) -> Optional[str]:
    """The engine an agent of `auto` recorded choosing the last time it started, if any."""
    try:
        found = json.loads(path.read_text()).get("engine")
    except (OSError, ValueError, AttributeError):
        return None
    return found if found in AUTO_ENGINES else None


def choose_at_start(
    probes: Probes, models_path: str, *, recorded: Optional[str] = None,
    sets: Optional[Sequence[vllm_install.LockedSet]] = None, patient: Optional[Probes] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> dict[str, Any]:
    """The choice of an agent of `auto` as it starts (`choose_engine`) — never on the strength of
    an `nvidia-smi` that failed or was slow. Where an NVIDIA driver is installed and its cards could
    not be read, they are asked again, more patiently (`PATIENT_PROBE_S`); still unread, the engine
    chosen the last time it started is kept (`recorded`), and with none, Ollama — each saying why.
    Taking such a look for "no card" would hand the port to Ollama and fetch every model again."""
    machine = gather(probes, models_path)
    if not _cards_unread(probes, machine, which):
        return choose_engine(machine, sets=sets)
    log.warning("the NVIDIA cards could not be read as this agent started; asking nvidia-smi again, for up to %.0fs",
                PATIENT_PROBE_S)
    patient = patient or _patient(probes)
    machine = gather(patient, models_path)
    if not _cards_unread(patient, machine, which):
        return choose_engine(machine, sets=sets)
    unread = (f"an NVIDIA driver is installed and its cards could not be read, though nvidia-smi was asked "
              f"again for up to {PATIENT_PROBE_S:.0f}s")
    if recorded is not None:
        return {"engine": recorded,
                "reasons": [f"{unread}; {recorded}, which this agent chose when it last started, is kept"]}
    return {"engine": "ollama", "reasons": [f"{unread}, and this agent has chosen no engine before"]}


class HandOver:
    """An agent of `auto` that chose another engine than the last time it started — a card
    changed, a driver was upgraded — finds the engine it started then still running on the port:
    engines outlive the agent (D148). Before the chosen one is started, that one is stopped, by its
    own record, so only what this agent started is ever stopped; then the choice is recorded.
    Tried on every look until it is done: a stop that fails says so, and is asked again."""

    def __init__(self, record: Path, chosen: str, other: Optional[installs.Server]) -> None:
        self.record, self.chosen, self.other = record, chosen, other
        self.done = False

    def recorded(self) -> Optional[str]:
        return recorded_choice(self.record)

    async def settle(self) -> bool:
        """True once the engine chosen may be started."""
        if self.done:
            return True
        if self.recorded() != self.chosen:
            if self.other is not None:
                try:
                    await self.other.stop()
                except Exception as exc:  # noqa: BLE001 - said, and asked again on the next look
                    log.warning("could not stop the engine this agent ran before choosing %s: %s", self.chosen, exc)
                    return False
            try:
                self.record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                draft = self.record.with_name(self.record.name + ".new")
                draft.write_text(json.dumps({"engine": self.chosen}))
                os.replace(draft, self.record)
            except OSError as exc:
                log.warning("could not record the engine chosen in %s: %s", self.record, exc)
        self.done = True
        return True


def create_app(
    settings: Settings,
    *,
    probes: Optional[Probes] = None,
    engine_transport: Optional[httpx.AsyncBaseTransport] = None,
    state_path: Optional[Path] = None,
    installer: Optional[installs.Installer] = None,
    vllm_sets: Optional[Sequence[vllm_install.LockedSet]] = None,
    patient_probes: Optional[Probes] = None,
) -> Starlette:
    """The agent. `installer` replaces the one its settings would make, `vllm_sets` the locked
    sets an agent of `auto` chooses by, and `patient_probes` what it asks again where the cards
    could not be read (`choose_at_start`) — for a test, which points them at a fake repository,
    fake programs and a machine of its own."""
    probes = probes or Probes()
    # Made here rather than at start-up so the app answers the same however it is served.
    engine_client = httpx.AsyncClient(base_url=settings.engine_url, timeout=10, transport=engine_transport)

    choice: Optional[dict[str, Any]] = None
    hand_over: Optional[HandOver] = None
    if settings.engine == AUTO:
        # Chosen once, as the agent starts, from the machine alone (D161); from then on this agent
        # is exactly one configured with the engine it chose — its directory, facts and installer.
        chosen_before = engine_dir_for(engines_dir_of(settings), settings) / CHOICE_FILE
        choice = choose_at_start(probes, settings.models_path, recorded=recorded_choice(chosen_before),
                                 sets=vllm_sets, patient=patient_probes)
        settings = dataclasses.replace(settings, engine=choice["engine"])
        other = next(name for name in AUTO_ENGINES if name != choice["engine"])
        hand_over = HandOver(chosen_before, choice["engine"], server_for(other, settings, engine_client))
        log.info("engine: auto chose %s%s", choice["engine"],
                 f" ({'; '.join(choice['reasons'])})" if choice["reasons"] else "")

    engine = engine_facts(settings.engine, settings)
    if installer is not None and getattr(installer.server, "engine", None) is not None:
        # A server the installer drives through the engine itself (whisper.cpp's): the same
        # object, so the model work and the installer see one record of what runs.
        engine = installer.server.engine

    if engine is not None and installer is None:
        installer = default_installer(settings, engine_client, engine)
    if isinstance(engine, VllmFacts) and installer is not None:
        engine.use_installer(installer)
    runs_vllm = installer is not None and isinstance(installer.server, VllmServer)

    async def keep() -> None:
        """One look at the engine this agent runs: the engine an `auto` agent ran before handed
        over first, then started if it is not running — and vLLM started again when the models it
        holds on disk have changed (D159)."""
        try:
            if hand_over is not None and not await hand_over.settle():
                return
            await installer.ensure_running()
            if runs_vllm and installer.server.wants_restart():
                with contextlib.suppress(Refused):  # busy, or not installed yet: the next look
                    await installer.restart()
        except Exception:  # noqa: BLE001 - a bad look must never end the looking
            log.exception("starting the engine again failed")

    #: The look in progress, if any: one at a time, whether the agent's own clock or a pass asked.
    looking: Optional[asyncio.Task[None]] = None

    def look() -> asyncio.Task[None]:
        nonlocal looking
        if looking is None or looking.done():
            looking = asyncio.create_task(keep(), name="gpm-agent:keep")
        return looking

    async def before_pass() -> None:
        """At the start of each of the pool's model passes. Ollama's and whisper.cpp's pass waits
        for the look, to find an engine to work with; vLLM's does not — its fetch needs no engine,
        and a start that takes minutes would hold up the fetch of every model meanwhile."""
        task = look()
        if not runs_vllm:
            await asyncio.shield(task)

    work = (
        ModelWork(
            engine, engine_client,
            free_disk_bytes=lambda: disk(probes, settings.models_path)["free_bytes"],
            min_free_disk_gb=settings.min_free_disk_gb,
            allow_delete=settings.allow_delete,
            state_path=state_path,
            before_pass=before_pass if installer is not None else None,
        )
        if engine is not None else None
    )
    if runs_vllm and work is not None:
        # What the pool told this machine to hold, once it has said: the model work starts blank,
        # and the pool's first pass replaces that with its set.
        untold = work.desired
        installer.server.held = lambda: None if work.desired is untold else work.desired.tags

    control = EngineControl(settings, engine, engine_client, installer) if engine is not None else None

    async def keep_running() -> None:
        """Start the engine this agent runs when the agent starts, and again whenever it stops."""
        while True:
            with contextlib.suppress(Exception):  # said by `keep`; a bad look never ends the looking
                await asyncio.shield(look())
            await asyncio.sleep(KEEP_EVERY_S)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        keeper = asyncio.create_task(keep_running(), name="gpm-agent:keeping") if installer is not None else None
        try:
            yield
        finally:
            if keeper is not None:
                keeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keeper
            if looking is not None and not looking.done():
                looking.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await looking
            if installer is not None:
                # The server is left running: it outlives the agent, which knows it again on return.
                await installer.aclose()
            if work is not None:
                await work.aclose()
            await engine_client.aclose()

    def refusal(request: Request) -> Optional[JSONResponse]:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            return _error(401, "unauthorized", "the agent requires its agent key")
        foreign = FOREIGN_PREFIXES.get(token.split("_", 1)[0])
        if foreign and not settings.accepts(token):
            # Said plainly: a pool key pasted here is a mistake worth naming, and neither
            # of the pool's keys may ever drive a host.
            return _error(403, "wrong_key_role", f"that is {foreign}; the agent accepts only its own agent key")
        if not settings.accepts(token):
            return _error(401, "unauthorized", "unknown agent key")
        return None

    async def facts(request: Request) -> JSONResponse:
        refused = refusal(request)
        if refused is not None:
            return refused
        described = (
            await engine.describe(engine_client)
            if engine is not None
            else {"name": settings.engine, "answers": False, "detail": "this agent does not know that engine"}
        )
        if engine is not None and settings.engine in INSTALLABLE_ENGINES:
            # Where the engine's versions stand (§2.3) — read from memory, so facts stay cheap.
            described["install"] = installer.state() if installer is not None else installs.not_allowed(settings.engine)
        if installer is not None:
            # Something answering on the engine's port that this agent did not start: reported,
            # and left alone (D150).
            described["foreign"] = bool(described.get("answers")) and not installer.server.ours()
        if choice is not None:
            # Which engine an agent of `auto` chose, and every reason against vLLM (D161): the
            # pool publishes the host's engine from this, and holds nothing there until it is said.
            described["choice"] = {"engine": choice["engine"], "reasons": list(choice["reasons"])}
        applied = control.applied() if control is not None else None
        body: dict[str, Any] = {
            "agent": {"version": __version__, "protocol": PROTOCOL_VERSION},
            **gather(probes, settings.models_path),
            "engine": described,
            # What the machine's owner allows, so the console can say so before anyone asks.
            "allows": {
                "delete": settings.allow_delete,
                "restart": bool(control and control.can_restart),
                "engine_settings": bool(control and control.can_apply_settings),
            },
            # What the engine was last started with, as written here — None if never.
            "engine_environment": applied,
            "engine_settings": engine.settings_from_environment(applied) if engine else None,
            "min_free_disk_bytes": int(settings.min_free_disk_gb * 1e9),
        }
        return JSONResponse(body)

    async def _models(request: Request, act: str) -> JSONResponse:
        refused = refusal(request)
        if refused is not None:
            return refused
        if work is None:
            return _error(501, "unknown_engine", f"this agent does not know the engine {settings.engine!r}")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        try:
            return JSONResponse(await (work.apply(body) if act == "apply" else work.delete(body)))
        except Refused as no:
            return _error(no.status, no.error, no.detail)

    async def engine_restart(request: Request) -> JSONResponse:
        refused = refusal(request)
        if refused is not None:
            return refused
        if control is None:
            return _error(501, "unknown_engine", f"this agent does not know the engine {settings.engine!r}")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        try:
            return JSONResponse(await control.act(body))
        except Refused as no:
            return _error(no.status, no.error, no.detail)

    async def engine_version(request: Request) -> JSONResponse:
        """Work toward an engine version, and switch to it when asked (D151). Answers at once."""
        refused = refusal(request)
        if refused is not None:
            return refused
        if engine is None:
            return _error(501, "unknown_engine", f"this agent does not know the engine {settings.engine!r}")
        if installer is None:
            return _error(
                403, "installs_not_allowed",
                "this machine's owner has not set install_engines: true in the agent's settings "
                "(`gpm-agent init --install-engines`); the pool cannot change that",
            )
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "bad_request", "the body must be JSON")
        try:
            version, switch = installs.validated(body)
            return JSONResponse(await installer.want(version, switch))
        except Refused as no:
            return _error(no.status, no.error, no.detail)

    async def hold_models(request: Request) -> JSONResponse:
        return await _models(request, "apply")

    async def delete_model(request: Request) -> JSONResponse:
        return await _models(request, "delete")

    async def heartbeat(request: Request) -> JSONResponse:
        """Postpone the dead-man timer, on a host the pool created (D63).

        No body, and nothing to get wrong: it touches one file. A machine nobody rented has no
        timer, so there is nothing here to beat and the verb says so.
        """
        refused = refusal(request)
        if refused is not None:
            return refused
        if not settings.heartbeat_file:
            return _error(409, "no_timer", "this machine carries no dead-man timer")
        path = Path(settings.heartbeat_file).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        except OSError as exc:
            return _error(500, "heartbeat_failed", f"the timer's file could not be touched: {exc}")
        return JSONResponse({"beat": True, "at": time.time()})

    base = f"/agent/v{PROTOCOL_VERSION}"
    app = Starlette(
        routes=[
            Route(f"{base}/facts", facts, methods=["GET"]),
            # The tag travels in the body, never the path: tags contain ':' and '/'.
            Route(f"{base}/models", hold_models, methods=["PUT"]),
            Route(f"{base}/models", delete_model, methods=["DELETE"]),
            # Restart the engine with the owner's command; optionally after writing settings.
            Route(f"{base}/engine", engine_restart, methods=["POST"]),
            # Postpones a shutdown the pool could equally cause by going silent (D63).
            Route(f"{base}/heartbeat", heartbeat, methods=["POST"]),
            # A version from the agent's own table, never a URL or a path (D151).
            Route(f"{base}/engine/version", engine_version, methods=["PUT"]),
        ],
        lifespan=lifespan,
    )
    app.state.work = work
    app.state.installer = installer
    #: One look at the engine this agent runs, as its own clock takes them — for a test, which
    #: serves the app without its lifespan.
    app.state.look = look if installer is not None else None
    return app
