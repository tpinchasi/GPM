"""Starting vLLM for what the agent downloaded (D97).

The launcher runs on the machine, from the agent's archive, when the machine's own restart
script calls it. Against a fake process starter: nothing here starts a real process, needs a
GPU, or installs vLLM.
"""

import json
import signal
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


class Processes:
    """Records what would have been started, and hands out pids."""

    def __init__(self):
        self.started: list[list[str]] = []
        self.envs: list[dict] = []
        self.killed: list[tuple[int, int]] = []
        self.alive: set[int] = set()
        self._next = 1000

    def popen(self, argv, **kwargs):
        assert "shell" not in kwargs, "never through a shell"
        assert isinstance(argv, list), "always an argument list"
        self.started.append(argv)
        self.envs.append(kwargs.get("env") or {})
        self._next += 1
        self.alive.add(self._next)
        return type("Process", (), {"pid": self._next})()

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

    upstreams = json.loads((tmp_path / vllm_launch.UPSTREAMS_FILE).read_text())
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

    engines = started.engines
    assert len(engines) == 6, "three models, two cards"
    for card in ("0", "1"):
        on_card = [e for e in engines if e["card"] == card]
        assert {e["model"] for e in on_card} == {BIG, SMALL, EMBED}
    for engine, env in zip(engines, processes.envs, strict=False):
        assert env["CUDA_VISIBLE_DEVICES"] == engine["card"], "each copy sees only its card"
    assert sorted(e["port"] for e in engines) == list(range(8001, 8007))
    # The memory plan is per card: each copy of a model gets the same share of its own card.
    shares = {(e["model"], e["card"]): e["memory_share"] for e in engines}
    assert all(shares[(m, "0")] == shares[(m, "1")] for m in (BIG, SMALL, EMBED))

    upstreams = json.loads((tmp_path / vllm_launch.UPSTREAMS_FILE).read_text())
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
