"""Starting vLLM for what the agent downloaded (D97).

The launcher runs on the machine, from the agent's archive, when the machine's own restart
script calls it. Against a fake process starter: nothing here starts a real process, needs a
GPU, or installs vLLM.
"""

import json
import signal

import httpx
import pytest
from gpm_agent import modelhub, vllm_launch

BIG = "nvidia/Gemma-4-26B-A4B-NVFP4"
EMBED = "nomic-ai/nomic-embed-text-v1.5"


def downloaded(models_dir, repo, size):
    into = modelhub.directory_for(models_dir, repo)
    into.mkdir(parents=True, exist_ok=True)
    (into / "model.safetensors").write_bytes(b"w" * size)
    (into / modelhub.COMPLETE_MARKER).write_text(f"{size}\n")
    return into


class Processes:
    """Records what would have been started, and hands out pids."""

    def __init__(self):
        self.started: list[list[str]] = []
        self.killed: list[tuple[int, int]] = []
        self.alive: set[int] = set()
        self._next = 1000

    def popen(self, argv, **kwargs):
        assert "shell" not in kwargs, "never through a shell"
        assert isinstance(argv, list), "always an argument list"
        self.started.append(argv)
        self._next += 1
        self.alive.add(self._next)
        return type("Process", (), {"pid": self._next})()

    def kill(self, pid, sig):
        self.killed.append((pid, sig))
        if sig in (signal.SIGTERM, signal.SIGKILL):
            self.alive.discard(pid)

    def launch(self, models_dir, **kwargs):
        return vllm_launch.launch(
            models_dir, 8000, env={}, popen=self.popen, kill=self.kill,
            alive=lambda pid: pid in self.alive, agent="/var/run/gpm/gpm-agent.pyz",
            python="python3", **kwargs,
        )


# --- a machine that has just booted ---


def test_nothing_on_disk_starts_nothing_and_is_not_an_error(tmp_path):
    """The ordinary state of a machine at boot: the models are not fetched yet."""
    processes = Processes()
    started = processes.launch(tmp_path)
    assert started.engines == [] and processes.started == []


def test_a_download_still_in_progress_is_not_started(tmp_path):
    """An engine started on half a model fails in ways that look like the model's fault."""
    partial = modelhub.directory_for(tmp_path, BIG)
    partial.mkdir(parents=True)
    (partial / "model.safetensors").write_bytes(b"w" * 100)

    processes = Processes()
    assert processes.launch(tmp_path).engines == []


# --- one model ---


def test_one_model_gets_one_engine_on_the_pools_port(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    started = processes.launch(tmp_path)

    (argv,) = processes.started
    assert argv[:3] == ["vllm", "serve", str(modelhub.directory_for(tmp_path, BIG))]
    assert argv[argv.index("--served-model-name") + 1] == BIG, "served under its repository"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--host") + 1] == "127.0.0.1", "loopback only (D77)"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"
    assert started.proxy is None


def test_the_pools_numbers_are_passed_on_and_only_those_present(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    vllm_launch.launch(
        tmp_path, 8000, popen=processes.popen, kill=processes.kill, alive=lambda pid: False,
        env={"GPM_VLLM_MAX_NUM_SEQS": "84", "GPM_VLLM_MAX_MODEL_LEN": "32768"},
    )
    (argv,) = processes.started
    assert argv[argv.index("--max-num-seqs") + 1] == "84"
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert "--max-num-batched-tokens" not in argv, "absent means the engine's own default"


def test_several_models_without_the_router_serve_the_first_and_say_so(tmp_path):
    """Two engines fighting for one card's memory is worse than one, and silence about the
    second would look like the pool dropped it."""
    downloaded(tmp_path, BIG, 1000)
    downloaded(tmp_path, EMBED, 10)
    processes = Processes()
    started = processes.launch(tmp_path)
    assert len(processes.started) == 1
    assert started.skipped == [BIG] or started.skipped == [EMBED]


# --- several models behind the router ---


def test_each_model_gets_its_own_engine_and_the_router_takes_the_pools_port(tmp_path):
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    started = processes.launch(tmp_path, proxy=True)

    engines = {e["model"]: e for e in started.engines}
    assert set(engines) == {BIG, EMBED}
    assert {e["port"] for e in engines.values()} == {8001, 8002}
    assert started.proxy["port"] == 8000
    router = processes.started[-1]
    assert router[:3] == ["python3", "/var/run/gpm/gpm-agent.pyz", "proxy"]

    upstreams = json.loads((tmp_path / vllm_launch.UPSTREAMS_FILE).read_text())
    assert upstreams == {name: f"http://127.0.0.1:{e['port']}" for name, e in engines.items()}


def test_memory_is_split_by_weights_with_a_floor_for_the_small_model(tmp_path):
    """Proportional alone would give a 0.3 GB model under 2% beside a 19 GB one — too little to
    start. The floor keeps it runnable; the total stays within what one process would take."""
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    started = processes.launch(tmp_path, proxy=True)

    shares = {e["model"]: e["memory_share"] for e in started.engines}
    assert shares[BIG] > shares[EMBED] >= 0.08
    assert sum(shares.values()) == pytest.approx(0.90, abs=0.01)


# --- starting again is a restart, not a second copy ---


def test_starting_again_stops_what_the_last_start_began(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    first = processes.launch(tmp_path)
    (first_pid,) = first.pids()

    processes.launch(tmp_path)

    assert (first_pid, signal.SIGTERM) in processes.killed
    assert len(processes.started) == 2


def test_a_process_already_gone_is_not_an_error(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    processes.launch(tmp_path)
    processes.alive.clear()
    processes.launch(tmp_path)
    assert processes.killed == [], "nothing alive to stop"


# --- the fetch's completion marker ---


class Hub:
    def __init__(self, files, cut_after=None):
        self.files, self.cut_after = files, cut_after

    def client(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.startswith("/api/models/"):
                return httpx.Response(200, json=[
                    {"type": "file", "path": n, "size": len(b)} for n, b in self.files.items()
                ])
            name = request.url.path.split("/resolve/main/", 1)[-1]
            body = self.files[name]
            asked = request.headers.get("Range")
            start = int(asked.removeprefix("bytes=").rstrip("-")) if asked else 0
            body = body[start:]
            if self.cut_after is not None:
                body = body[: self.cut_after]
            return httpx.Response(206 if asked else 200, content=body)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_the_marker_is_written_only_once_every_file_has_landed(tmp_path):
    hub = Hub({"config.json": b"{}", "model.safetensors": b"w" * 4096}, cut_after=1000)
    async with hub.client() as client:
        with pytest.raises(modelhub.HubRefused, match="resume"):
            async for _ in modelhub.fetch(BIG, tmp_path, client=client):
                pass
    into = modelhub.directory_for(tmp_path, BIG)
    assert not (into / modelhub.COMPLETE_MARKER).exists(), "a cut download is not complete"

    hub.cut_after = None
    async with hub.client() as client:
        async for _ in modelhub.fetch(BIG, tmp_path, client=client):
            pass
    assert (into / modelhub.COMPLETE_MARKER).exists()


async def test_a_fetch_that_starts_again_is_not_complete_until_it_finishes(tmp_path):
    """Whatever an earlier fetch left behind, this one says when it is done."""
    into = downloaded(tmp_path, BIG, 10)
    hub = Hub({"model.safetensors": b"w" * 4096}, cut_after=100)
    async with hub.client() as client:
        with pytest.raises(modelhub.HubRefused):
            async for _ in modelhub.fetch(BIG, tmp_path, client=client):
                pass
    assert not (into / modelhub.COMPLETE_MARKER).exists()


# --- named options (D100) ---


def with_family(directory, family):
    (directory / "config.json").write_text(json.dumps({"model_type": family}))
    return directory


def test_tool_calling_on_gemma_4_starts_its_parser(tmp_path):
    with_family(downloaded(tmp_path, BIG, 100), "gemma4")
    processes = Processes()
    started = processes.launch(tmp_path, options=["tool_calling", "reasoning"])
    argv = processes.started[0]
    assert argv[-5:] == ["--enable-auto-tool-choice", "--tool-call-parser", "gemma4",
                         "--reasoning-parser", "gemma4"]
    assert started.not_applied == []


def test_each_model_gets_its_own_familys_flags_behind_the_router(tmp_path):
    with_family(downloaded(tmp_path, BIG, 100), "gemma4")
    with_family(downloaded(tmp_path, "Qwen/Qwen3-8B", 100), "qwen3")
    processes = Processes()
    processes.launch(tmp_path, proxy=True, options=["tool_calling"])
    by_model = {argv[4]: argv for argv in processes.started if "serve" in argv}
    assert by_model["Qwen/Qwen3-8B"][-2:] == ["--tool-call-parser", "hermes"]
    assert by_model[BIG][-2:] == ["--tool-call-parser", "gemma4"]


def test_a_family_without_the_option_starts_without_it_and_says_so(tmp_path):
    """An embedding model has no tool calling; refusing to start it would take down the rest."""
    with_family(downloaded(tmp_path, EMBED, 100), "nomic_bert")
    processes = Processes()
    started = processes.launch(tmp_path, options=["tool_calling"])
    assert "--enable-auto-tool-choice" not in processes.started[0]
    assert started.not_applied == [f"tool_calling for {EMBED} (family nomic_bert)"]


def test_a_model_with_no_readable_family_starts_without_options(tmp_path):
    downloaded(tmp_path, BIG, 100)  # no config.json
    processes = Processes()
    started = processes.launch(tmp_path, options=["reasoning"])
    assert "--reasoning-parser" not in processes.started[0]
    assert "family unknown" in started.not_applied[0]


def test_an_option_the_launcher_does_not_know_is_refused(tmp_path):
    with_family(downloaded(tmp_path, BIG, 100), "gemma4")
    with pytest.raises(ValueError, match="unknown option"):
        Processes().launch(tmp_path, options=["--trust-remote-code"])


def test_the_command_line_accepts_only_the_named_options():
    """What the pool writes into the machine's start is a name from a closed list, never a flag."""
    import argparse

    parser = argparse.ArgumentParser()
    vllm_launch.add_arguments(parser)
    ok = parser.parse_args(["--models-dir", "/m", "--port", "8000", "--option", "tool_calling"])
    assert ok.option == ["tool_calling"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--models-dir", "/m", "--port", "8000", "--option", "--chat-template=/etc/x"])
