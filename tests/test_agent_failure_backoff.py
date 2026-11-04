"""An agent that fails to fetch, hold or start a model keeps trying, ever less often, and serves
again on its own once the cause is put right — no restart of the agent or the pool (D163).

Found on a Windows install: an earlier run as administrator left the whisper agent's state folder
to administrators only, every start of whisper-server was refused "Access is denied", and the
host stayed `preparing` with a single event in the decision log, until the whole stack was
restarted after the folder was repaired.
"""

import os
import pathlib
import signal

import httpx
import pytest
from gpm_agent import platform as host_platform
from gpm_agent import whisper_cpp
from gpm_agent.engines import EngineRefused
from gpm_agent.models import ModelWork
from test_agent_whisper_install import C1, TAG, V1, rig  # noqa: F401

MODEL = "chat:7b"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FlakyEngine:
    """An engine whose download fails until told otherwise."""

    name = "ollama"

    def __init__(self) -> None:
        self.broken = "pull model manifest: connection reset by peer"
        self.on_disk: set[str] = set()
        self.loaded: set[str] = set()
        self.pulls = 0

    async def describe(self, client):
        return {"answers": True, "models_on_disk": [{"tag": t, "size_bytes": 1} for t in self.on_disk],
                "models_loaded": sorted(self.loaded)}

    async def pull(self, client, tag):
        self.pulls += 1
        if self.broken:
            raise EngineRefused(self.broken)
        self.on_disk.add(tag)
        yield 1, 1

    async def hold(self, client, tag, *, pinned):
        (self.loaded.add if pinned else self.loaded.discard)(tag)


@pytest.fixture
def clock():
    return Clock()


async def a_pass(work: ModelWork, tags=(MODEL,)) -> dict:
    await work.apply({"tags": list(tags), "residency": "pinned"})
    await work._task
    return {m["tag"]: m for m in (await work.snapshot())["models"]}


async def test_a_failure_is_tried_again_ever_less_often_up_to_ten_minutes(clock):
    engine = FlakyEngine()
    async with httpx.AsyncClient() as client:
        work = ModelWork(engine, client, free_disk_bytes=lambda: 10**12, min_free_disk_gb=0, allow_delete=False, clock=clock)
        waits = []
        for _ in range(8):
            model = (await a_pass(work))[MODEL]
            waits.append(model["retry_in_s"])
            pulls = engine.pulls
            clock.now += model["retry_in_s"] - 1
            await a_pass(work)
            assert engine.pulls == pulls, "not tried again before its wait is over"
            clock.now += 1
        assert waits == [30, 60, 120, 240, 480, 600, 600, 600]
        assert model["attempts"] == 8 and model["error"] == "pull model manifest: connection reset by peer"


async def test_once_the_cause_is_gone_the_next_attempt_serves_and_the_failure_is_forgotten(clock):
    engine = FlakyEngine()
    async with httpx.AsyncClient() as client:
        work = ModelWork(engine, client, free_disk_bytes=lambda: 10**12, min_free_disk_gb=0, allow_delete=False, clock=clock)
        for _ in range(3):
            model = (await a_pass(work))[MODEL]
            clock.now += model["retry_in_s"]
        engine.broken = None  # the network is back
        model = (await a_pass(work))[MODEL]
        assert model["on_disk"] and model["loaded"] and model["error"] is None
        assert model["attempts"] == 0 and model["retry_in_s"] is None
        engine.broken = "disk full"
        engine.on_disk.clear()
        engine.loaded.clear()
        assert (await a_pass(work))[MODEL]["retry_in_s"] == 30, "a new run of failures starts from the first wait"


@pytest.mark.skipif(os.name == "nt", reason="runs its fake servers as POSIX scripts")
async def test_a_whisper_server_its_state_folder_refused_serves_once_the_folder_is_repaired(rig, clock, monkeypatch):  # noqa: F811
    """The Windows case, as the agent that installs whisper.cpp runs it: the agent starts after an
    elevated run, the server's record and its state folder cannot be read, and an administrator
    repairs the folder later. The same agent then serves, at its next attempt."""
    rig.publish(V1, C1)
    await rig.want("latest")
    await rig.hold(TAG)
    os.kill(rig.engine._current()["pid"], signal.SIGKILL)
    rig.engine._process.wait(timeout=10)
    state = rig.engine._state_dir
    denied = {"on": True}
    real_read, real_prepare = pathlib.Path.read_text, whisper_cpp.WhisperCppFacts._prepare_state

    def read_text(self, *args, **kwargs):
        if denied["on"] and str(self).startswith(str(state)):
            raise PermissionError(13, "Access is denied", str(self))
        return real_read(self, *args, **kwargs)

    def prepare(self):
        if denied["on"]:
            with whisper_cpp._refusing(f"could not prepare this agent's state directory {state}", state):
                raise PermissionError(13, "Access is denied", str(state / "tmp"))
        return real_prepare(self)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    monkeypatch.setattr(whisper_cpp.WhisperCppFacts, "_prepare_state", prepare)
    rig.engine, rig.installer = rig.new_agent()  # the next ordinary start
    async with httpx.AsyncClient(base_url=rig.url, timeout=10) as client:
        work = ModelWork(rig.engine, client, free_disk_bytes=lambda: 10**12, min_free_disk_gb=0, allow_delete=False,
                         clock=clock)
        for attempt in (1, 2):
            model = (await a_pass(work, (TAG,)))[TAG]
            assert not model["loaded"] and model["attempts"] == attempt
            assert "Access is denied" in model["error"]
            clock.now += model["retry_in_s"]
        denied["on"] = False  # repaired by an administrator; nothing restarted
        model = (await a_pass(work, (TAG,)))[TAG]
    assert model["loaded"] and model["error"] is None and model["attempts"] == 0


def test_a_record_that_cannot_be_read_now_is_read_again_later(tmp_path, monkeypatch):
    """Remembered as "nothing was started" for the agent's whole life, a record unreadable at its
    start outlived the repair of its folder."""
    engine = whisper_cpp.WhisperCppFacts(type("S", (), {"models_path": str(tmp_path / "models")})())
    state = engine._state_dir
    state.mkdir(parents=True)
    (state / whisper_cpp.RECORD_FILE).write_text('{"pid": 4242, "tag": "x"}')
    real = pathlib.Path.read_text
    denied = {"on": True}
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(
        PermissionError(13, "Access is denied")) if denied["on"] else real(self, *a, **k))
    assert engine._current() is None
    denied["on"] = False
    assert engine._current() == {"pid": 4242, "tag": "x"}


def test_on_windows_a_folder_refused_says_how_to_give_it_back():
    error = PermissionError(13, "Access is denied", r"C:\Users\shop\gpm-whisper-models.gpm-state\tmp")
    folder = pathlib.PureWindowsPath(r"C:\Users\shop\gpm-whisper-models.gpm-state")
    hint = host_platform.access_hint(error, folder, os_name=host_platform.WINDOWS)
    assert 'icacls "C:\\Users\\shop\\gpm-whisper-models.gpm-state" /reset /T' in hint
    assert "run as administrator" in hint and "tries again by itself" in hint
    assert host_platform.access_hint(error, folder, os_name=host_platform.LINUX) == ""
    assert host_platform.access_hint(FileNotFoundError(2, "gone"), folder, os_name=host_platform.WINDOWS) == ""
