"""Starting vLLM for what the agent downloaded (D97).

The launcher runs on the machine, from the agent's archive, when the machine's own restart
script calls it. Against a fake process starter: nothing here starts a real process, needs a
GPU, or installs vLLM.
"""

import json
import signal
import subprocess
import time
from pathlib import Path

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


class Pipe:
    """A process's standard input, as the launcher writes to it."""

    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, data):
        assert not self.closed
        self.data += data

    def close(self):
        self.closed = True


class Processes:
    """Records what would have been started, and hands out pids."""

    def __init__(self):
        self.started: list[list[str]] = []
        self.envs: list[dict] = []
        self.killed: list[tuple[int, int]] = []
        self.alive: set[int] = set()
        self.inputs: dict[int, Pipe] = {}
        self.env_of: dict[int, dict] = {}
        self._next = 1000

    def popen(self, argv, **kwargs):
        assert "shell" not in kwargs, "never through a shell"
        assert isinstance(argv, list), "always an argument list"
        self.started.append(argv)
        self.envs.append(kwargs.get("env") or {})
        self._next += 1
        self.env_of[self._next] = kwargs.get("env") or {}
        self.alive.add(self._next)
        process = type("Process", (), {"pid": self._next})()
        if kwargs.get("stdin") == subprocess.PIPE:
            process.stdin = self.inputs[self._next] = Pipe()
        return process

    def engines(self):
        """The `vllm serve` processes started, in order."""
        return [argv for argv in self.started if argv[1:2] == ["serve"]]

    def handed_to_in_turn(self):
        """The plan the launcher gave what starts the rest of each card's models."""
        (pid,) = list(self.inputs)
        assert self.inputs[pid].closed
        return json.loads(self.inputs[pid].data)

    def take_turns(self, models_dir, ready=lambda port: True):
        """Run what starts the rest of each card's models, as the launcher started it (D155)."""
        if self.inputs:
            vllm_launch.run_in_turn(
                models_dir, self.handed_to_in_turn(), env={}, popen=self.popen, ready=ready,
                alive=lambda pid: pid in self.alive, sleep=lambda s: None, say=lambda line: None,
            )

    def kill(self, pid, sig):
        self.killed.append((pid, sig))
        if sig in (signal.SIGTERM, signal.SIGKILL):
            self.alive.discard(pid)

    def launch(self, models_dir, **kwargs):
        kwargs.setdefault("probe_card", lambda: None)  # no driver to ask on the test machine
        kwargs.setdefault("probe_devices", lambda env: [])
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
        probe_card=lambda: None, probe_devices=lambda env: [],
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

    upstreams = json.loads((vllm_launch.state_dir_for(tmp_path) / vllm_launch.UPSTREAMS_FILE).read_text())
    assert upstreams == {name: [f"http://127.0.0.1:{e['port']}"] for name, e in engines.items()}


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


def test_models_that_arrive_with_another_machines_state_bring_none_of_it(tmp_path):
    # A sibling's copy or a reused volume brings whatever was in its models directory (D139).
    # Another machine's process ids must never be signalled here, nor its record read as ours.
    models = tmp_path / "models"
    downloaded(models, BIG, 1000)
    elsewhere = tmp_path / "elsewhere"
    downloaded(elsewhere, BIG, 1000)
    processes = Processes()
    theirs = processes.launch(elsewhere)
    for name in vllm_launch.LEGACY_STATE[:2]:
        source = vllm_launch.state_dir_for(elsewhere) / name
        (models / name).write_text(source.read_text())
    (their_pid,) = theirs.pids()
    processes.killed.clear()

    assert vllm_launch.read_record(models) is None, "their record is not this machine's"
    assert vllm_launch.failed_engines(models, served=[]) == {}
    processes.launch(models)

    assert (their_pid, signal.SIGTERM) not in processes.killed
    assert not any((models / name).exists() for name in vllm_launch.LEGACY_STATE), "removed, unread"
    assert not any(p.name.startswith(".gpm") for p in models.iterdir()), "nothing of ours in the models"
    assert vllm_launch.read_record(models)["engines"], "ours is kept beside them"


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
    processes.take_turns(tmp_path)
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


# --- placing several models on one card (D104) ---

GB = 1024**3


def test_each_model_gets_its_weights_plus_a_cache_reserve_and_the_spare_goes_by_weight():
    """Found live: a 48 GB card split by weights alone gave a 15 GB model 17.9 GB, and it
    refused to start for want of 2.1 GiB of cache."""
    shares, refused = vllm_launch.memory_plan([16 * GB, 2 * GB], 48 * GB)
    assert refused is None
    big, small = shares
    assert big * 48 * GB >= 16 * GB * 1.10 + 3 * GB, "weights, loading overhead, and the reserve"
    assert small * 48 * GB >= 2 * GB * 1.10 + 3 * GB
    assert big > small and abs(big + small - 0.90) < 0.01, "everything the launcher may use, the larger model getting more"


def test_a_set_that_does_not_fit_is_refused_with_the_arithmetic():
    """The three the owner rented for: on a 48 GB card they need more than the launcher may use."""
    shares, refused = vllm_launch.memory_plan([16 * GB, 18.8 * GB, 1.9 * GB], 48 * GB)
    assert shares == [] and refused is not None
    assert "need" in refused and "this card gives" in refused and "fewer models" in refused


def test_a_card_of_unknown_size_falls_back_to_the_split_by_weights():
    shares, refused = vllm_launch.memory_plan([16 * GB, 2 * GB], None)
    assert refused is None and shares == vllm_launch.memory_shares([16 * GB, 2 * GB])


def test_a_refused_set_starts_nothing_and_says_why_where_the_agent_reads_it(tmp_path):
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 2_000)
    processes = Processes()
    started = processes.launch(tmp_path, proxy=True, card_bytes=20_000)  # far too small
    assert started.refused and processes.started == []
    record = vllm_launch.read_record(tmp_path)
    assert record["refused"] == started.refused
    assert set(record["plan"]["models"]) == {BIG, EMBED}
    assert vllm_launch.failed_engines(tmp_path, served=[]) == {BIG: started.refused, EMBED: started.refused}


def test_what_was_started_is_recorded_with_its_process_and_log(tmp_path):
    downloaded(tmp_path, BIG, 100)
    processes = Processes()
    processes.launch(tmp_path, card_bytes=48 * GB)
    (engine,) = vllm_launch.read_record(tmp_path)["engines"]
    assert engine["model"] == BIG and engine["pid"] in processes.alive
    assert engine["log"].endswith("nvidia__Gemma-4-26B-A4B-NVFP4.log")


# --- a process that dies is a failure, not a model still loading (D104) ---


def test_a_process_that_exited_without_serving_is_reported_with_its_logs_reason(tmp_path):
    downloaded(tmp_path, BIG, 100)
    processes = Processes()
    processes.launch(tmp_path, card_bytes=48 * GB)
    (engine,) = vllm_launch.read_record(tmp_path)["engines"]
    Path(engine["log"]).write_text(
        "(APIServer pid=1657) INFO loading\n"
        "(APIServer pid=1657) ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (1536).\n"
        "(APIServer pid=1657) INFO shutting down\n"
    )
    processes.alive.discard(engine["pid"])
    failed = vllm_launch.failed_engines(tmp_path, served=[], alive=lambda pid: pid in processes.alive)
    assert set(failed) == {BIG}
    assert failed[BIG].startswith("its vLLM process exited before serving it: ValueError: Chunked MM input")
    assert "(APIServer" not in failed[BIG], "the process prefix is not the reason"


def test_a_process_still_up_is_loading_however_long_it_takes(tmp_path):
    downloaded(tmp_path, BIG, 100)
    processes = Processes()
    processes.launch(tmp_path, card_bytes=48 * GB)
    assert vllm_launch.failed_engines(tmp_path, served=[], alive=lambda pid: pid in processes.alive) == {}


def test_a_model_being_served_is_never_failed_whatever_its_pid_says(tmp_path):
    downloaded(tmp_path, BIG, 100)
    processes = Processes()
    processes.launch(tmp_path, card_bytes=48 * GB)
    assert vllm_launch.failed_engines(tmp_path, served=[BIG], alive=lambda pid: False) == {}


def test_a_new_launch_forgets_the_last_launchs_record(tmp_path):
    downloaded(tmp_path, BIG, 100)
    processes = Processes()
    processes.launch(tmp_path, card_bytes=20_000)  # refused
    assert vllm_launch.read_record(tmp_path)["refused"]
    processes.launch(tmp_path, card_bytes=48 * GB)  # fits
    assert vllm_launch.read_record(tmp_path)["refused"] is None


# --- a machine with several cards: a copy of every model on each (D107) ---

SMALL = "google/gemma-4-E4B-it"


def test_every_model_runs_once_on_each_card_pinned_to_it(tmp_path):
    """Found live: a 2x H100 host ran all three models on card 0, and card 1 sat at 4 MiB."""
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, SMALL, 15_000)
    downloaded(tmp_path, EMBED, 2_000)
    processes = Processes()
    started = processes.launch(tmp_path, proxy=True, devices=["0", "1"])
    processes.take_turns(tmp_path)

    engines = vllm_launch.read_record(tmp_path)["engines"]
    assert len(engines) == 6, "three models, two cards"
    for card in ("0", "1"):
        on_card = [e for e in engines if e["card"] == card]
        assert {e["model"] for e in on_card} == {BIG, SMALL, EMBED}
    for engine in engines:
        assert processes.env_of[engine["pid"]]["CUDA_VISIBLE_DEVICES"] == engine["card"], "each copy sees only its card"
    assert sorted(e["port"] for e in engines) == list(range(8001, 8007))
    # The memory plan is per card: each copy of a model gets the same share of its own card.
    shares = {(e["model"], e["card"]): e["memory_share"] for e in engines}
    assert all(shares[(m, "0")] == shares[(m, "1")] for m in (BIG, SMALL, EMBED))

    upstreams = json.loads((vllm_launch.state_dir_for(tmp_path) / vllm_launch.UPSTREAMS_FILE).read_text())
    for model in (BIG, SMALL, EMBED):
        assert len(upstreams[model]) == 2, "the router is told about both copies"
    assert started.proxy["port"] == 8000


def test_the_hosts_workers_are_split_between_its_copies(tmp_path):
    """The pool gives a host its workers for all its cards; vLLM's limit is per process, so a
    copy per card at the whole number would admit twice what the pool asked for."""
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    vllm_launch.launch(
        tmp_path, 8000, popen=processes.popen, kill=processes.kill, alive=lambda pid: False,
        env={"GPM_VLLM_MAX_NUM_SEQS": "13", "GPM_VLLM_MAX_NUM_BATCHED_TOKENS": "8192"},
        devices=["0", "1"], probe_card=lambda: None,
    )
    engines = [argv for argv in processes.started if argv[0] == "vllm"]
    assert [argv[argv.index("--max-num-seqs") + 1] for argv in engines] == ["7", "7"], "rounded up"
    assert all(argv[argv.index("--max-num-batched-tokens") + 1] == "8192" for argv in engines), \
        "a batch is per process already, and is not split"


def test_one_model_on_two_cards_still_gets_the_router(tmp_path):
    """The pool placed one model, and still dials one port: two copies need something in front."""
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    started = processes.launch(tmp_path, devices=["0", "1"])
    assert [e["port"] for e in started.engines] == [8001, 8002]
    assert started.proxy and started.proxy["port"] == 8000
    assert processes.started[-1][2] == "proxy"


def test_one_card_is_exactly_as_before(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    started = processes.launch(tmp_path, devices=["0"])
    (engine,) = started.engines
    assert engine["port"] == 8000 and engine["card"] is None and started.proxy is None
    assert "CUDA_VISIBLE_DEVICES" not in processes.envs[0], "nothing pinned where there is no choice"


def test_the_cards_are_those_the_environment_allows_or_the_driver_lists():
    def driver(argv, **kwargs):
        return type("Done", (), {"stdout": "0\n1\n"})()

    def no_driver(argv, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert vllm_launch.card_devices({}, run=driver) == ["0", "1"]
    assert vllm_launch.card_devices({"CUDA_VISIBLE_DEVICES": "2,3"}, run=driver) == ["2", "3"], \
        "a machine limited to some cards is held to them"
    assert vllm_launch.card_devices({}, run=no_driver) == []


def test_a_copy_that_died_is_reported_with_its_card(tmp_path):
    """A model is served once every copy answers, so one dead copy fails it — and the operator
    needs to know which card to look at."""
    downloaded(tmp_path, BIG, 100)
    processes = Processes()
    processes.launch(tmp_path, devices=["0", "1"])
    record = vllm_launch.read_record(tmp_path)
    dead = next(e for e in record["engines"] if e["card"] == "1")
    Path(dead["log"]).write_text("(APIServer pid=9) torch.OutOfMemoryError: CUDA out of memory\n")
    processes.alive.discard(dead["pid"])
    failed = vllm_launch.failed_engines(tmp_path, served=[], alive=lambda pid: pid in processes.alive)
    assert failed[BIG].startswith("its vLLM process on card 1 exited before serving it: torch.OutOfMemoryError")
    assert dead["log"].endswith(".card1.log"), "each copy writes its own log"


# --- a model split across a group of cards (D114) ---


def split_launch(processes, models_dir, cards, per_copy, **kwargs):
    return vllm_launch.launch(
        models_dir, 8000, popen=processes.popen, kill=processes.kill, alive=lambda pid: False,
        env={"GPM_VLLM_CARDS_PER_COPY": str(per_copy), "GPM_VLLM_MAX_NUM_SEQS": "12"},
        devices=cards, probe_card=lambda: None, agent="/var/run/gpm/gpm-agent.pyz", python="python3",
        **kwargs,
    )


def with_heads(directory, heads, *, under_text_config=False):
    config = {"model_type": "gemma4", "num_attention_heads": heads}
    if under_text_config:
        config = {"model_type": "gemma4", "text_config": {"num_attention_heads": heads}}
    (directory / "config.json").write_text(json.dumps(config))


def test_a_model_split_across_two_cards_runs_once_per_pair(tmp_path):
    """Four cards, two to a copy: two copies, each one vLLM process across its pair."""
    with_heads(downloaded(tmp_path, BIG, 1000), 32)
    processes = Processes()
    started = split_launch(processes, tmp_path, ["0", "1", "2", "3"], 2)

    assert started.refused is None
    assert [e["card"] for e in started.engines] == ["0,1", "2,3"]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in processes.envs[:2]] == ["0,1", "2,3"]
    engines = [argv for argv in processes.started if argv[0] == "vllm"]
    assert all(argv[argv.index("--tensor-parallel-size") + 1] == "2" for argv in engines)
    assert [argv[argv.index("--max-num-seqs") + 1] for argv in engines] == ["6", "6"], \
        "the host's workers are split between its two copies, not its four cards"
    assert started.proxy and started.proxy["port"] == 8000, "two copies, one port: the router"
    assert started.engines[0]["log"].endswith(".card0-1.log")
    assert started.plan["cards_per_copy"] == 2


def test_one_group_of_cards_is_one_copy_pinned_to_it(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    started = split_launch(processes, tmp_path, ["0", "1"], 2)
    (engine,) = started.engines
    assert engine["card"] == "0,1" and engine["port"] == 8000 and started.proxy is None


def test_a_machine_whose_cards_do_not_make_whole_groups_is_refused(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    started = split_launch(processes, tmp_path, ["0", "1", "2"], 2)
    assert processes.started == []
    assert "has 3, which is not a whole number of groups of 2" in started.refused
    assert vllm_launch.read_record(tmp_path)["refused"] == started.refused, "said where the agent reads it"


def test_a_split_with_no_cards_listed_is_refused(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    started = split_launch(processes, tmp_path, [], 2)
    assert processes.started == [] and "the driver lists none" in started.refused


def test_a_model_whose_heads_do_not_divide_is_refused_before_anything_starts(tmp_path):
    """vLLM would refuse at start, after the machine was rented and the weights fetched."""
    with_heads(downloaded(tmp_path, BIG, 1000), 12)
    processes = Processes()
    started = split_launch(processes, tmp_path, ["0", "1", "2", "3", "4", "5", "6", "7"], 8)
    assert processes.started == []
    assert started.refused == (
        f"{BIG} has 12 attention heads, which do not divide between 8 cards; split it across fewer cards"
    )


def test_the_heads_of_a_model_built_around_a_text_model_are_read_from_it(tmp_path):
    with_heads(downloaded(tmp_path, BIG, 1000), 6, under_text_config=True)
    assert vllm_launch.attention_heads(modelhub.directory_for(tmp_path, BIG)) == 6
    assert "6 attention heads" in vllm_launch.split_refusal([modelhub.directory_for(tmp_path, BIG)], 4)


def test_a_model_that_does_not_state_its_heads_is_left_to_the_engine(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    assert vllm_launch.split_refusal([modelhub.directory_for(tmp_path, BIG)], 2) is None


def test_a_split_puts_a_share_of_the_weights_on_each_card():
    """A set refused on one card fits two: each card holds half of every model."""
    card = 24 * 10**9
    _, alone = vllm_launch.memory_plan([30 * 10**9], card)
    assert alone is not None
    shares, split = vllm_launch.memory_plan([30 * 10**9], card, 2)
    assert split is None and shares == [0.9]
    _, four_ways = vllm_launch.memory_plan([30 * 10**9, 30 * 10**9], card, 2)
    assert "split across 2 cards" in four_ways and "or the models split across more cards" in four_ways


def test_a_split_that_is_not_a_power_of_two_is_refused():
    placements, why = vllm_launch.card_groups(["0", "1", "2"], 3)
    assert placements == [] and "not 3" in why


def test_one_card_per_copy_is_d107_exactly():
    assert vllm_launch.card_groups(["0", "1"], 1) == (["0", "1"], None)
    assert vllm_launch.card_groups(["0"], 1) == ([None], None)
    assert vllm_launch.cards_per_copy({}) == 1
    assert vllm_launch.cards_per_copy({"GPM_VLLM_CARDS_PER_COPY": "x"}) == 1


def test_a_group_that_died_is_reported_with_its_cards(tmp_path):
    downloaded(tmp_path, BIG, 100)
    processes = Processes()
    split_launch(processes, tmp_path, ["0", "1", "2", "3"], 2)
    record = vllm_launch.read_record(tmp_path)
    dead = next(e for e in record["engines"] if e["card"] == "2,3")
    Path(dead["log"]).write_text("(APIServer pid=9) RuntimeError: NCCL error\n")
    failed = vllm_launch.failed_engines(tmp_path, served=[], alive=lambda pid: pid != dead["pid"])
    assert failed[BIG].startswith("its vLLM process on cards 2,3 exited before serving it: RuntimeError: NCCL")


# --- a model that arrived by a copy (D116) ---


def test_a_marker_that_arrived_before_the_weights_is_not_a_complete_model(tmp_path):
    """A provider's copy may bring the complete marker before the files it vouches for."""
    into = modelhub.directory_for(tmp_path, BIG)
    into.mkdir(parents=True)
    (into / modelhub.COMPLETE_MARKER).write_text("1000\n")
    (into / "model.safetensors").write_bytes(b"w" * 400)
    assert not modelhub.is_complete(into)
    assert vllm_launch.complete_models(tmp_path) == []
    (into / "model.safetensors").write_bytes(b"w" * 1000)
    assert modelhub.is_complete(into) and vllm_launch.complete_models(tmp_path) == [into]


def test_a_marker_that_cannot_be_read_vouches_for_nothing(tmp_path):
    """Every marker the agent writes carries a size; an empty one is a copy cut short."""
    into = modelhub.directory_for(tmp_path, BIG)
    into.mkdir(parents=True)
    (into / "model.safetensors").write_bytes(b"w" * 1000)
    for unreadable in ("", "not a size"):
        (into / modelhub.COMPLETE_MARKER).write_text(unreadable)
        assert not modelhub.is_complete(into)


def test_an_upgraded_host_stops_its_older_launchers_engines_and_nothing_else(tmp_path):
    # On a host upgraded in place, the older launcher's engines still hold the cards; its list is
    # in the models directory. Only processes that are an engine by their own command line are
    # stopped — the list may as well be another machine's (the code review's finding).
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    processes.alive.update({4242, 4343})
    (tmp_path / vllm_launch.PIDS_FILE).write_text("[4242, 4343]")
    lines = {4242: b"python3\x00-m\x00vllm\x00serve", 4343: b"/usr/sbin/sshd"}
    processes.launch(tmp_path, cmdline=lines.get)
    assert (4242, signal.SIGTERM) in processes.killed and not any(pid == 4343 for pid, _ in processes.killed)
    assert not (tmp_path / vllm_launch.PIDS_FILE).exists()


# --- each model no longer than it goes (D146) ---

WHISPER = "openai/whisper-large-v3-turbo"
RERANKER = "BAAI/bge-reranker-base"


def with_files(directory, **files):
    """A model's own JSON files, by name (`config`, `tokenizer_config`, …)."""
    for name, body in files.items():
        (directory / f"{name}.json").write_text(json.dumps(body))
    return directory


# Whisper's own configuration states its decoder's length and no `max_position_embeddings`.
WHISPER_CONFIG = {"model_type": "whisper", "max_source_positions": 1500, "max_target_positions": 448}


def launch_asking(processes, models_dir, length, **kwargs):
    env = {"GPM_VLLM_MAX_NUM_SEQS": "8"}
    if length is not None:
        env["GPM_VLLM_MAX_MODEL_LEN"] = str(length)
    return vllm_launch.launch(
        models_dir, 8000, popen=processes.popen, kill=processes.kill, alive=lambda pid: False, env=env,
        probe_card=lambda: None, probe_devices=lambda env: [], agent="/var/run/gpm/gpm-agent.pyz",
        python="python3", **kwargs,
    )


def length_flag(argv):
    return argv[argv.index("--max-model-len") + 1] if "--max-model-len" in argv else None


def test_whisper_is_started_at_its_decoders_length_and_says_so(tmp_path):
    """vLLM refuses to start a model with a length above the one its configuration states, and
    the pool sends a text model's context to every process: Whisper would never start."""
    with_files(downloaded(tmp_path, WHISPER, 100), config=WHISPER_CONFIG)
    processes = Processes()
    started = launch_asking(processes, tmp_path, 8192)

    assert length_flag(processes.started[0]) == "448"
    assert started.plan["models"][WHISPER]["max_model_len_capped"] == {"asked": 8192, "applied": 448}
    assert vllm_launch.capped_lengths(tmp_path) == {WHISPER: {"asked": 8192, "applied": 448}}, \
        "recorded where the agent reads it"


def test_a_model_that_goes_further_than_the_pool_asked_gets_the_pools_number(tmp_path):
    with_files(downloaded(tmp_path, BIG, 100), config={"model_type": "qwen3", "max_position_embeddings": 131072})
    processes = Processes()
    started = launch_asking(processes, tmp_path, 8192)
    assert length_flag(processes.started[0]) == "8192"
    assert "max_model_len_capped" not in started.plan["models"][BIG]
    assert vllm_launch.capped_lengths(tmp_path) == {}


def test_a_model_built_around_a_text_model_is_held_to_the_text_models_length(tmp_path):
    with_files(downloaded(tmp_path, BIG, 100), config={
        "model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "max_position_embeddings": 4096},
        "vision_config": {"max_position_embeddings": 999_999},
    })
    processes = Processes()
    launch_asking(processes, tmp_path, 8192)
    assert length_flag(processes.started[0]) == "4096"


def test_a_model_that_states_no_length_gets_the_pools_number(tmp_path):
    """vLLM then takes whatever it is given, so the launcher has nothing to hold it to."""
    downloaded(tmp_path, BIG, 100)  # no config.json at all
    with_files(downloaded(tmp_path, EMBED, 100), config={"model_type": "nomic_bert"})
    processes = Processes()
    launch_asking(processes, tmp_path, 8192, proxy=True)
    processes.take_turns(tmp_path)
    assert [length_flag(argv) for argv in processes.started if "serve" in argv] == ["8192", "8192"]
    assert vllm_launch.capped_lengths(tmp_path) == {}


def test_without_the_pools_number_no_length_is_passed_at_all(tmp_path):
    """Absent means the engine's own default, as for every number the pool sends."""
    with_files(downloaded(tmp_path, WHISPER, 100), config=WHISPER_CONFIG)
    processes = Processes()
    started = launch_asking(processes, tmp_path, None)
    assert length_flag(processes.started[0]) is None
    assert "max_model_len_capped" not in started.plan["models"][WHISPER]


def test_behind_the_router_each_model_gets_its_own_length_on_every_copy(tmp_path):
    with_files(downloaded(tmp_path, WHISPER, 100), config=WHISPER_CONFIG)
    with_files(downloaded(tmp_path, BIG, 100), config={"max_position_embeddings": 131072})
    processes = Processes()
    launch_asking(processes, tmp_path, 8192, proxy=True, devices=["0", "1"])
    processes.take_turns(tmp_path)
    lengths = {(argv[4], length_flag(argv)) for argv in processes.started if "serve" in argv}
    assert lengths == {(WHISPER, "448"), (BIG, "8192")}
    assert len([argv for argv in processes.started if "serve" in argv]) == 4, "a copy of each on each card"


def test_a_refused_start_reports_no_lengths(tmp_path):
    with_files(downloaded(tmp_path, WHISPER, 19_000), config=WHISPER_CONFIG)
    processes = Processes()
    started = launch_asking(processes, tmp_path, 8192, card_bytes=20_000)
    assert started.refused and vllm_launch.capped_lengths(tmp_path) == {}


@pytest.mark.parametrize(("files", "expected"), [
    # The smallest stated length, wherever vLLM looks for one.
    ({"config": {"max_position_embeddings": 4096, "n_positions": 2048}}, 2048),
    ({"config": {"seq_length": 32768}}, 32768),
    # `model_max_length` stands as it is, even where another key says less (Command-R).
    ({"config": {"max_position_embeddings": 8192, "model_max_length": 131072}}, 131072),
    # Lengths that are not numbers are not lengths.
    ({"config": {"max_position_embeddings": "long", "max_seq_len": True}}, None),
    # A reranker's encoder is held to its tokenizer's own limit: 514 positions, 512 tokens.
    ({"config": {"architectures": ["XLMRobertaForSequenceClassification"],
                 "position_embedding_type": "absolute", "max_position_embeddings": 514},
      "tokenizer_config": {"model_max_length": 512}}, 512),
    # … and so is an embedding encoder …
    ({"config": {"architectures": ["BertModel"], "position_embedding_type": "absolute", "max_position_embeddings": 514},
      "tokenizer_config": {"model_max_length": 512}}, 512),
    # … and a sentence-transformers model, whatever its architecture …
    ({"config": {"architectures": ["SomeForCausalLM"], "position_embedding_type": "absolute",
                 "max_position_embeddings": 514},
      "tokenizer_config": {"model_max_length": 512},
      "modules": [{"idx": 0, "type": "sentence_transformers.models.Transformer", "path": ""},
                  {"idx": 1, "type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}]}, 512),
    # … but only a model vLLM runs for pooling: one that generates is not held to it, absolute
    # positions or not — `…LMHeadModel` included, though it ends in `Model` …
    ({"config": {"architectures": ["SomeForCausalLM"], "position_embedding_type": "absolute",
                 "max_position_embeddings": 514},
      "tokenizer_config": {"model_max_length": 512}}, 514),
    ({"config": {"architectures": ["SomeLMHeadModel"], "position_embedding_type": "absolute",
                 "max_position_embeddings": 514},
      "tokenizer_config": {"model_max_length": 512}}, 514),
    ({"config": {"position_embedding_type": "absolute", "max_position_embeddings": 514},
      "tokenizer_config": {"model_max_length": 512}}, 514),
    # … and only an encoder with absolute positions, as vLLM does.
    ({"config": {"architectures": ["XLMRobertaForSequenceClassification"], "max_position_embeddings": 514},
      "tokenizer_config": {"model_max_length": 512}}, 514),
    # A sentence-transformers model's own `max_seq_length` replaces the rest outright.
    ({"config": {"max_position_embeddings": 512},
      "sentence_bert_config": {"max_seq_length": 256, "do_lower_case": False}}, 256),
    ({"config": {"max_position_embeddings": 512}, "sentence_bert_config": {"max_seq_length": 256}}, 512),
    # Rotary scaling, as vLLM counts it: multiplied by its factor …
    ({"config": {"max_position_embeddings": 4096,
                 "rope_scaling": {"type": "linear", "factor": 2.0}}}, 8192),
    # … from the length it was trained at, for yarn …
    ({"config": {"max_position_embeddings": 131072,
                 "rope_scaling": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}}},
     65536),
    # … and not at all where the stated length is already the scaled one.
    ({"config": {"max_position_embeddings": 131072, "rope_scaling": {"rope_type": "llama3", "factor": 8.0}}}, 131072),
    ({"config": {"model_type": "gemma3_text", "max_position_embeddings": 131072,
                 "rope_scaling": {"rope_type": "linear", "factor": 8.0}}}, 131072),
    # One scaling per kind of attention layer.
    ({"config": {"max_position_embeddings": 4096, "rope_parameters": {
        "full_attention": {"rope_type": "linear", "factor": 4.0},
        "sliding_attention": {"rope_type": "default"}}}}, 16384),
    ({"config": ["not", "a", "configuration"]}, None),
])
def test_a_models_own_length_is_read_as_vllm_reads_it(tmp_path, files, expected):
    directory = tmp_path / "model"
    directory.mkdir()
    with_files(directory, **files)
    assert vllm_launch.max_model_len_of(directory) == expected


# --- a card's models start one after another (D155) ---


def test_the_models_on_one_card_start_one_after_another(tmp_path):
    """vLLM sizes its cache from the card's free memory across its own start, so a neighbour
    loading meanwhile is counted as its own use. Only the first starts with the launcher."""
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    started = processes.launch(tmp_path, proxy=True)

    assert len(processes.engines()) == 1, "one engine now, the other in turn"
    first, second = vllm_launch.read_record(tmp_path)["engines"]
    assert first["pid"] in processes.alive and second["pid"] is None and second["waiting"]
    helper = [argv for argv in processes.started if "vllm-in-turn" in argv]
    assert helper == [["python3", "/var/run/gpm/gpm-agent.pyz", "vllm-in-turn", "--models-dir", str(tmp_path)]]
    assert started.in_turn["pid"] in started.pids(), "the next start stops it too"
    (queue,) = processes.handed_to_in_turn()
    assert [e["model"] for e in queue] == [first["model"], second["model"]]
    assert queue[0]["pid"] == first["pid"]

    answers = iter([False, False, True])
    processes.take_turns(tmp_path, ready=lambda port: next(answers))
    assert len(processes.engines()) == 2
    record = vllm_launch.read_record(tmp_path)["engines"]
    assert record[1]["pid"] in processes.alive and "waiting" not in record[1]
    pids = json.loads((vllm_launch.state_dir_for(tmp_path) / vllm_launch.PIDS_FILE).read_text())
    assert record[1]["pid"] in pids, "a process started in turn is stopped by the next start"


def test_the_next_model_waits_for_the_one_before_it_to_answer(tmp_path):
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    processes.launch(tmp_path, proxy=True)
    asked: list[int] = []
    engines_when_asked: list[int] = []

    def ready(port):
        asked.append(port)
        engines_when_asked.append(len(processes.engines()))
        return len(asked) >= 4

    processes.take_turns(tmp_path, ready=ready)
    first_port = vllm_launch.read_record(tmp_path)["engines"][0]["port"]
    assert asked == [first_port] * 4
    assert engines_when_asked == [1, 1, 1, 1], "nothing more started while the first was loading"
    assert len(processes.engines()) == 2


def test_a_model_after_one_that_died_still_starts(tmp_path):
    """Its failure is reported from its own log; the rest of the card may still serve."""
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    processes.launch(tmp_path, proxy=True)
    processes.alive.discard(vllm_launch.read_record(tmp_path)["engines"][0]["pid"])
    processes.take_turns(tmp_path, ready=lambda port: False)
    assert len(processes.engines()) == 2


def test_separate_cards_start_together_and_each_card_in_turn(tmp_path):
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    processes.launch(tmp_path, proxy=True, devices=["0", "1"])
    now = vllm_launch.read_record(tmp_path)["engines"]
    assert sorted(e["card"] for e in now if e["pid"]) == ["0", "1"], "the first model on each card at once"
    queues = processes.handed_to_in_turn()
    assert [[e["card"] for e in queue] for queue in queues] == [["0", "0"], ["1", "1"]]


def test_one_model_on_a_card_needs_nothing_started_in_turn(tmp_path):
    downloaded(tmp_path, BIG, 1000)
    processes = Processes()
    started = processes.launch(tmp_path)
    assert started.in_turn is None and not processes.inputs
    assert vllm_launch.read_record(tmp_path)["in_turn"] is None


def test_a_model_waiting_its_turn_is_not_failed_while_what_starts_it_runs(tmp_path):
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    processes.launch(tmp_path, proxy=True)
    alive = lambda pid: pid in processes.alive  # noqa: E731
    assert vllm_launch.failed_engines(tmp_path, served=[], alive=alive) == {}

    processes.alive.discard(vllm_launch.read_record(tmp_path)["in_turn"]["pid"])
    waiting = vllm_launch.read_record(tmp_path)["engines"][1]["model"]
    failed = vllm_launch.failed_engines(tmp_path, served=[], alive=alive)
    assert set(failed) == {waiting}
    assert "was never started" in failed[waiting]


def test_starting_again_stops_what_starts_models_in_turn_before_anything_else(tmp_path):
    """Until it has gone it may start one more engine, after the list of what to stop was read."""
    downloaded(tmp_path, BIG, 19_000)
    downloaded(tmp_path, EMBED, 300)
    processes = Processes()
    first = processes.launch(tmp_path, proxy=True)
    processes.launch(tmp_path, proxy=True)
    assert processes.killed[0] == (first.in_turn["pid"], signal.SIGTERM)
    assert {pid for pid, _ in processes.killed} >= set(first.pids())


def test_what_starts_models_in_turn_starts_only_the_launchers_vllm(tmp_path):
    processes = Processes()
    planted = [[{"model": "a", "port": 8001, "card": None, "pid": 5, "argv": ["vllm", "serve", "a"], "log": str(tmp_path / "a.log")},
                {"model": "b", "port": 8002, "card": None, "pid": None, "argv": ["sh", "-c", "anything"], "log": str(tmp_path / "b.log")}]]
    with pytest.raises(ValueError):
        vllm_launch.run_in_turn(tmp_path, planted, env={}, popen=processes.popen, ready=lambda port: True,
                                alive=lambda pid: True, sleep=lambda s: None, say=lambda line: None)
    assert processes.started == []


# --- why an engine stopped, from its log ---


def test_the_reason_is_the_engines_own_exception_not_the_line_pointing_at_it(tmp_path):
    """Found live: all the pool heard was "Engine core initialization failed. See root cause
    above.", and the machine was gone before anyone could read what was above."""
    log = tmp_path / "engine.log"
    log.write_text(
        "(EngineCore_DP0 pid=95) ERROR 10-29 20:24:58 [core.py:946] EngineCore failed to start.\n"
        "(EngineCore_DP0 pid=95) ERROR 10-29 20:24:58 [core.py:946] Traceback (most recent call last):\n"
        '(EngineCore_DP0 pid=95) ERROR 10-29 20:24:58 [core.py:946]     raise ValueError(\n'
        "(EngineCore_DP0 pid=95) ERROR 10-29 20:24:58 [core.py:946] ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.\n"
        "(APIServer pid=1) Traceback (most recent call last):\n"
        '(APIServer pid=1)   File "/usr/lib/python3/vllm/v1/engine/utils.py", line 900, in wait_for_engine_startup\n'
        "(APIServer pid=1)     raise RuntimeError(\n"
        "(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}\n"
    )
    assert vllm_launch.last_error_line(str(log)) == (
        "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` "
        "when initializing the engine."
    )


def test_a_pointer_is_reported_when_nothing_better_is_in_the_log(tmp_path):
    log = tmp_path / "engine.log"
    log.write_text("(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}\n")
    assert vllm_launch.last_error_line(str(log)).startswith("RuntimeError: Engine core initialization failed")


def test_a_reason_far_above_a_long_closing_traceback_is_still_found(tmp_path):
    log = tmp_path / "engine.log"
    log.write_text(
        "(EngineCore_DP0 pid=95) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
        + "(APIServer pid=1)   File \"/usr/lib/python3/somewhere.py\", line 1, in f\n" * 300
        + "(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above.\n"
    )
    assert vllm_launch.last_error_line(str(log)).startswith("torch.OutOfMemoryError: CUDA out of memory")


def test_the_helper_runs_as_its_own_process_from_the_plan_on_its_input(tmp_path):
    """The command the launcher starts, run for real: it reads the plan from its input, starts the
    next model once the one before has gone, writes it down, and ends."""
    import os
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vllm = bin_dir / "vllm"
    vllm.write_text(f"#!/bin/sh\necho \"$@\" > {tmp_path / 'started'}\n")
    vllm.chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    state = vllm_launch.state_dir_for(models)
    state.mkdir()
    gone = 2**22 + 11  # no such process: the first model has exited
    engines = [
        {"model": "a", "port": 8001, "card": None, "pid": gone, "log": str(state / "a.log")},
        {"model": "b", "port": 8002, "card": None, "pid": None, "waiting": True, "log": str(state / "b.log")},
    ]
    (state / vllm_launch.STARTED_FILE).write_text(json.dumps({"engines": engines, "in_turn": {"pid": 1}}))
    (state / vllm_launch.PIDS_FILE).write_text(json.dumps([gone]))
    plan = [[{**engines[0], "argv": [str(vllm), "serve", "a"]}, {**engines[1], "argv": [str(vllm), "serve", "b"]}]]

    agent_src = Path(vllm_launch.__file__).parents[1]
    done = subprocess.run(
        [sys.executable, "-m", "gpm_agent.cli", "vllm-in-turn", "--models-dir", str(models)],
        input=json.dumps(plan).encode(), capture_output=True, timeout=30,
        env={**os.environ, "PYTHONPATH": str(agent_src)},
    )
    assert done.returncode == 0, done.stderr.decode()
    assert b"started b on 8002, once a exited" in done.stdout
    for _ in range(50):  # the fake engine runs on its own
        if (tmp_path / "started").exists():
            break
        time.sleep(0.1)
    assert (tmp_path / "started").read_text().split() == ["serve", "b"]
    second = vllm_launch.read_record(models)["engines"][1]
    assert isinstance(second["pid"], int) and "waiting" not in second
    assert second["pid"] in json.loads((state / vllm_launch.PIDS_FILE).read_text())
