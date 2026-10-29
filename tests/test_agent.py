"""The host agent, stage 1: facts (docs/spec/host-agent.md, D40).

Everything here runs without the hardware it describes: the agent's probes of the outside
world are injected, and the supervisor reaches a real agent app in-process.
"""

import ast
import asyncio
import collections
import pathlib
import stat
import time
from pathlib import Path

import gpm_agent
import httpx
import pytest
from fakes.harness import EngineSpec, pool_harness
from gpm_agent import facts as agent_facts
from gpm_agent.app import create_app
from gpm_agent.settings import Settings, SettingsError, fingerprint, load, mint_key, save
from gpm_server.config import ConfigError, load_config
from gpm_server.supervisor import agents

MODEL = "m1"
AGENT_KEY = "gpmg_" + "a" * 64
Usage = collections.namedtuple("Usage", "total used free")


def probes(system, machine, outputs=None, files=None, free_gb=100):
    """A machine described by what its commands and files would say."""
    outputs, files = outputs or {}, files or {}
    return agent_facts.Probes(
        system=lambda: system,
        machine=lambda: machine,
        run=lambda command: outputs.get(command[0] if command[0] != "sysctl" else " ".join(command)),
        read=lambda path: files.get(path),
        disk_usage=lambda path: Usage(500 * 10**9, (500 - free_gb) * 10**9, free_gb * 10**9),
    )


MAC = probes("Darwin", "arm64", {
    "sysctl -n hw.memsize": "51539607552\n",
    "sysctl -n machdep.cpu.brand_string": "Apple M4 Max\n",
    "vm_stat": "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
               "Pages free: 1000.\nPages inactive: 2000.\nPages purgeable: 500.\n",
})
CUDA_BOX = probes("Linux", "x86_64", {"nvidia-smi": "NVIDIA RTX A6000, 49140\nNVIDIA RTX A6000, 49140\n"},
                  {"/proc/meminfo": "MemTotal: 131072000 kB\nMemAvailable: 100000000 kB\n"})
PLAIN_LINUX = probes("Linux", "x86_64", files={"/proc/meminfo": "MemTotal: 8000000 kB\nMemAvailable: 4000000 kB\n"})
UNKNOWN = probes("Plan9", "mips")


# --- facts ---


def test_an_apple_silicon_machine_is_recognised_with_its_unified_memory():
    found = agent_facts.gather(MAC, "~/.ollama/models")
    assert found["capabilities"] == ["apple-silicon"]
    (chip,) = found["accelerators"]
    assert chip["name"] == "Apple M4 Max" and chip["unified"] and chip["memory_bytes"] == 51539607552
    assert found["memory"] == {"total_bytes": 51539607552, "available_bytes": 3500 * 16384}
    assert found["disk"]["free_bytes"] == 100 * 10**9


def test_every_cuda_card_is_listed_with_its_own_memory():
    found = agent_facts.gather(CUDA_BOX, "/models")
    assert found["capabilities"] == ["cuda"]
    assert [a["memory_bytes"] for a in found["accelerators"]] == [49140 * 1024 * 1024] * 2
    assert found["memory"]["total_bytes"] == 131072000 * 1024


def nvidia_box(details=None, driver=None):
    """A Linux machine with two NVIDIA cards, by what each of `nvidia-smi`'s queries prints: None
    for a query its driver refuses, as an old one refuses a field it does not know."""
    outputs = {
        " ".join(agent_facts.NVIDIA_CARDS): "NVIDIA RTX A6000, 49140\nNVIDIA GeForce RTX 2080 Ti, 11264\n",
        " ".join(agent_facts.NVIDIA_DETAILS): details,
        " ".join(agent_facts.NVIDIA_DRIVER): driver,
    }
    asked = []

    def run(command):
        asked.append(" ".join(command))
        return outputs.get(" ".join(command))

    return agent_facts.Probes(system=lambda: "Linux", machine=lambda: "x86_64", run=run, read=lambda path: None), asked


def test_an_nvidia_card_says_its_compute_capability_and_driver():
    """What vLLM's floor (7.5) and `engine: auto`'s (8.0) are judged by, and the locked set's
    driver (D157, D161)."""
    probes, _ = nvidia_box(details="8.6, 575.51.03\n7.5, 575.51.03\n")
    cards = agent_facts.accelerators(probes)
    assert [(c["name"], c["compute_capability"], c["driver"]) for c in cards] == [
        ("NVIDIA RTX A6000", "8.6", "575.51.03"), ("NVIDIA GeForce RTX 2080 Ti", "7.5", "575.51.03"),
    ]
    assert cards[0]["memory_bytes"] == 49140 * 1024 * 1024


def test_a_driver_that_does_not_know_compute_cap_keeps_the_cards_name_memory_and_its_version():
    """An old driver refuses the whole query a field it does not know is in: asked apart, so the
    card is still listed, and its driver's version — what the owner must change — is asked alone."""
    probes, asked = nvidia_box(details=None, driver="470.82.01\n470.82.01\n")
    cards = agent_facts.accelerators(probes)
    assert [c["name"] for c in cards] == ["NVIDIA RTX A6000", "NVIDIA GeForce RTX 2080 Ti"]
    assert [c["memory_bytes"] for c in cards] == [49140 * 1024 * 1024, 11264 * 1024 * 1024]
    assert all("compute_capability" not in c and c["driver"] == "470.82.01" for c in cards)
    assert asked[:3] == [" ".join(q) for q in (agent_facts.NVIDIA_CARDS, agent_facts.NVIDIA_DETAILS, agent_facts.NVIDIA_DRIVER)]

    probes, _ = nvidia_box(details=None, driver=None)
    assert all(set(c) == {"kind", "name", "memory_bytes", "unified"} for c in agent_facts.accelerators(probes))


@pytest.mark.parametrize("details", [
    "8.6, 575.51.03\n",                      # one line for two cards: which is whose would be a guess
    "[N/A], 575.51.03\nwhat, ever\n",       # not a capability, not a version
])
def test_details_that_do_not_read_as_each_cards_own_are_left_out_never_guessed(details):
    probes, _ = nvidia_box(details=details)
    cards = agent_facts.accelerators(probes)
    assert all("compute_capability" not in c for c in cards)
    if details.count("\n") == 2:
        assert cards[0]["driver"] == "575.51.03" and "driver" not in cards[1]
    else:
        assert all("driver" not in c for c in cards)


def test_no_accelerator_and_could_not_tell_are_different_answers():
    """The pool derives nothing from silence, so the agent must not turn one into the other."""
    assert agent_facts.gather(PLAIN_LINUX, "/models")["capabilities"] == []
    assert agent_facts.gather(UNKNOWN, "/models")["capabilities"] is None


def test_the_agent_never_runs_a_shell_or_a_command_it_was_given():
    """Structural: every command is a literal argument list, run without a shell, and only a few
    modules can start a process at all — each checked below for what it starts."""
    root = pathlib.Path(gpm_agent.__file__).parent
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "shell":
                raise AssertionError(f"{path.name} passes shell= to a subprocess")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {alias.name for alias in node.names} | {getattr(node, "module", None)}
                if "subprocess" in names or "os.system" in names:
                    # facts.py runs literal argument lists to learn what the machine is. The vLLM
                    # launcher (D97) starts `vllm` and the agent's own router, as fixed argument
                    # lists — and is a command-line entry for the machine's restart script,
                    # reachable from nothing on the agent's protocol (checked below). platform.py
                    # starts the programs the agent runs itself — whisper.cpp's server (D148) and
                    # an engine it installed (D150) — runs its own literal commands to recognise
                    # and stop them and to check signatures (D152, D154), and runs a caller's fixed
                    # argument list and waits for it — whisper.cpp's build (D152): checked below.
                    # startup.py runs the owner's own `init` and `startup` commands (D154), from
                    # the command line only: checked in test_agent_startup.py.
                    assert path.name in (
                        "facts.py", "vllm_launch.py", "platform.py", "startup.py",
                    ), f"{path.name} can start a process"
        text = path.read_text()
        assert "os.system" not in text and "os.popen" not in text and "create_subprocess_shell" not in text
        if "create_subprocess_exec" in text:
            # The one other place a process starts: the owner's restart command, and only that.
            assert path.name == "engine_control.py", f"{path.name} can start a process"
            assert text.count("create_subprocess_exec(") == 1
            assert "create_subprocess_exec(\n                *self.settings.restart_command," in text

    # The launcher can start processes, so nothing that answers the pool may reach it: only the
    # command line imports it, and the machine's restart script is what runs that.
    for name in (
        "app.py", "models.py", "engine_control.py", "engines.py", "proxy.py",
        "installs.py", "ollama_server.py", "whisper_cpp.py", "whisper_build.py", "platform.py",
    ):
        text = (root / name).read_text()
        assert "vllm_launch" not in text, f"{name} reaches the launcher, which can start processes"
    launcher = (root / "vllm_launch.py").read_text()
    assert "shell=" not in launcher and "os.system" not in launcher

    # platform.py: a program starts in `start`, on the caller's `argv`, and nowhere else; its own
    # commands run in `_run`, each a literal list naming one of four programs.
    platform_tree = ast.parse((root / "platform.py").read_text())
    functions = {node.name: node for node in ast.walk(platform_tree) if isinstance(node, ast.FunctionDef)}
    subprocess_calls = [
        node for node in ast.walk(platform_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
    ]
    # `list2cmdline` runs nothing: it quotes a recorded argument list as Windows reads a command
    # line back, to recognise a server by it (`started_with`), and only there.
    quoting = [call for call in subprocess_calls if call.func.attr == "list2cmdline"]
    assert len(quoting) == 1 and quoting[0] in list(ast.walk(functions["started_with"]))
    subprocess_calls = [call for call in subprocess_calls if call not in quoting]
    assert [call.func.attr for call in subprocess_calls] == ["run"]
    assert subprocess_calls[0] in list(ast.walk(functions["_run"]))
    assert isinstance(subprocess_calls[0].args[0], ast.Name) and subprocess_calls[0].args[0].id == "argv"
    launches = [
        node for node in ast.walk(platform_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "launch"
    ]
    # Two places start a program: `start`, for a server the agent runs; `run_command`, for a
    # command it waits for. Each on its caller's `argv` — which `run_command` may only put behind
    # POSIX's `nice`, to lower its priority.
    run_command = next(node for node in ast.walk(platform_tree)
                       if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_command")
    assert len(launches) == 2
    assert launches[0] in list(ast.walk(functions["start"])) and launches[1] in list(ast.walk(run_command))
    assert all(isinstance(call.args[0], ast.Name) and call.args[0].id == "argv" for call in launches)
    rebound = [node for node in ast.walk(run_command) if isinstance(node, ast.Assign)
               and any(getattr(target, "id", None) == "argv" for target in node.targets)]
    assert [ast.unparse(node.value) for node in rebound] == ["[nice, '-n', str(LOW_PRIORITY), *argv]"]

    def runs_a_command(call):
        func = call.func
        names = [func] if isinstance(func, ast.Name) else list(func.values) if isinstance(func, ast.BoolOp) else []
        return bool(names) and all(isinstance(n, ast.Name) and n.id in ("run", "_run") for n in names)

    # `_run_briefly` only passes its caller's command on, with a shorter time limit.
    passed_on = list(ast.walk(functions["_run_briefly"]))
    commands = [
        node for node in ast.walk(platform_tree)
        if isinstance(node, ast.Call) and runs_a_command(node) and node not in passed_on
    ]
    # A part is a literal, a number or a path made text here, or one of the module's own string
    # constants — PowerShell's fixed script — never anything a caller passes as text.
    constants = {
        target.id for node in platform_tree.body if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert commands, "the platform's own commands are found"
    for call in commands:
        command = call.args[0]
        assert isinstance(command, ast.List) and isinstance(command.elts[0], ast.Constant)
        assert command.elts[0].value in ("ps", "powershell", "taskkill", "codesign")
        for part in command.elts:
            assert isinstance(part, (ast.Constant, ast.Call, ast.JoinedStr)) or (
                isinstance(part, ast.Name) and part.id in constants
            ), ast.dump(part)

    # What the agent starts through it: whisper.cpp's server (D148) — the owner's program with
    # the argument list built in `server_argv` from the agent's own constants, a file under its
    # models directory and its port; and an installed Ollama (D150) — the program in a version
    # directory the agent unpacked, and `serve`, built in `serve_argv`.
    for name, builder in (("whisper_cpp.py", "server_argv"), ("ollama_server.py", "serve_argv")):
        tree = ast.parse((root / name).read_text())
        starts = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "start"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "host_platform"
        ]
        assert len(starts) == 1, name
        assert isinstance(starts[0].args[0], ast.Name) and starts[0].args[0].id == "argv"
        assigned = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "argv" for t in node.targets)
        ]
        assert len(assigned) == 1 and isinstance(assigned[0], ast.Call) and assigned[0].func.id == builder, name
    for name in ("app.py", "models.py", "engines.py", "installs.py", "proxy.py", "engine_control.py", "whisper_build.py",
                 "vllm_server.py"):
        assert "host_platform.start(" not in (root / name).read_text(), f"{name} starts a program"

    # vLLM run by the agent (D159): only the agent's own launcher and its stop, as programs, each
    # command built in one place — the agent's interpreter, its guard, its own module and a verb,
    # then the agent's settings and the installed program — and run as built.
    tree = ast.parse((root / "vllm_server.py").read_text())
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for builder, verb in (("launcher_argv", "vllm-start"), ("stop_argv", "vllm-stop")):
        lists = [node for node in ast.walk(methods[builder]) if isinstance(node, ast.List)]
        head = ast.unparse(lists[0])
        assert head.startswith("[self.python, *interpreter_guard(), *AGENT_MODULE, ") and f"'{verb}'" in head, head
    module = next(node.value for node in tree.body
                  if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "AGENT_MODULE")
    assert ast.unparse(module) == "('-m', 'gpm_agent.cli')"
    runs = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run" and getattr(node.func.value, "id", None) == "self"]
    assert len(runs) == 2
    for call in runs:
        command = ast.unparse(call.args[0])
        assert command in ("argv", "self.stop_argv()"), command
    assigned = [ast.unparse(node.value) for node in ast.walk(methods["start"])
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "argv" for t in node.targets)]
    assert assigned == ["self.launcher_argv(program, models)"]

    # What runs through `run_command`, and from where: whisper.cpp's build and checks (D152), and
    # vLLM's environment, pip and its check (D157, D158). Each command is a literal list — a fixed
    # program, or a path the agent found, then constants — or comes from one of the builders that
    # write those argument lists; nothing else.
    builders = {"configure_argv", "build_argv", "venv_argv", "pip_argv", "help_argv", "check_argv"}
    for name in sorted(path.name for path in root.glob("*.py")):
        text = (root / name).read_text()
        if "host_platform.run_command" not in text and "platform import run_command" not in text:
            continue  # startup.py has a `run_command` of its own, for the owner's commands
        if name == "vllm_server.py":
            continue  # its launcher and stop, above
        assert name in ("whisper_build.py", "whisper_cpp.py", "vllm_install.py"), f"{name} runs commands"
        tree = ast.parse(text)
        runs = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "run")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "run"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "self")
            )
        ]
        assert runs, name
        for call in runs:
            command = call.args[0]
            if isinstance(command, ast.Call):
                assert isinstance(command.func, ast.Name) and command.func.id in builders, ast.unparse(command)
            else:
                assert isinstance(command, ast.List), ast.unparse(command)
                first, *rest = command.elts
                assert isinstance(first, ast.Constant) or (
                    isinstance(first, ast.Call) and getattr(first.func, "id", None) == "str"
                ), ast.unparse(command)
                assert all(isinstance(part, ast.Constant) for part in rest), ast.unparse(command)


# --- the agent's own settings ---


def test_settings_are_written_owner_only_and_hold_no_key(tmp_path):
    key = mint_key()
    path = save(Settings(key_hash=fingerprint(key)), tmp_path / "agent.json")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert key not in path.read_text()
    assert load(path).accepts(key) and not load(path).accepts(mint_key())


def test_settings_others_can_read_are_refused(tmp_path):
    path = save(Settings(key_hash="x"), tmp_path / "agent.json")
    path.chmod(0o644)
    with pytest.raises(SettingsError, match="readable"):
        load(path)


def test_listening_off_loopback_without_tls_is_refused_unless_said_deliberately():
    with pytest.raises(SettingsError, match="clear text"):
        Settings(key_hash="x", host="0.0.0.0").check()
    Settings(key_hash="x", host="0.0.0.0", allow_insecure=True).check()
    Settings(key_hash="x", host="0.0.0.0", tls_certfile="c", tls_keyfile="k").check()


def test_a_restart_command_is_an_argument_list_never_a_shell_line():
    with pytest.raises(SettingsError, match="argument list"):
        Settings(key_hash="x", restart_command="systemctl restart ollama").check()


# --- the HTTP surface ---


def agent_app(machine=MAC, engine_app=None, **settings):
    transport = httpx.ASGITransport(app=engine_app) if engine_app is not None else None
    return create_app(Settings(key_hash=fingerprint(AGENT_KEY), **settings), probes=machine, engine_transport=transport)


async def get_facts(app, key):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
        return await http.get("/agent/v1/facts", headers=headers)


async def test_the_agent_answers_only_its_own_key_and_names_a_pool_key_for_what_it_is():
    app = agent_app()
    assert (await get_facts(app, None)).status_code == 401
    assert (await get_facts(app, "gpmg_" + "b" * 64)).status_code == 401
    for pool_key, called in (("gpma_" + "c" * 64, "an app key"), ("gpmx_" + "d" * 64, "the admin key")):
        refused = await get_facts(app, pool_key)
        assert refused.status_code == 403 and called in refused.json()["detail"]
    assert (await get_facts(app, AGENT_KEY)).status_code == 200


async def test_facts_include_what_the_engine_holds_and_what_the_owner_allows():
    with pool_harness([EngineSpec(id="e", resident={MODEL}, available={"on-disk-only"})], model_set=[MODEL]) as pool:
        app = agent_app(engine_app=pool.engines["e"].fake.app, allow_delete=False)
        body = (await get_facts(app, AGENT_KEY)).json()
    assert body["agent"]["protocol"] == "1"
    assert body["engine"]["answers"] and body["engine"]["models_loaded"] == [MODEL]
    assert [m["tag"] for m in body["engine"]["models_on_disk"]] == [MODEL, "on-disk-only"]
    assert body["allows"] == {"delete": False, "restart": False, "engine_settings": False}


def test_the_verbs_are_a_closed_list():
    """The security property of §4. Adding a route must be a deliberate edit to this test."""
    routes = {(route.path, tuple(sorted(route.methods - {"HEAD"}))) for route in agent_app().routes}
    assert routes == {
        ("/agent/v1/facts", ("GET",)),      # stage 1
        ("/agent/v1/models", ("PUT",)),     # stage 2: hold this set of tags
        ("/agent/v1/models", ("DELETE",)),  # stage 2: delete this tag (D40's three bounds)
        ("/agent/v1/engine", ("POST",)),    # stage 3: restart, with the owner's command (D41)
        ("/agent/v1/heartbeat", ("POST",)),  # stage 5a: postpone the timer on a rented host (D63)
        ("/agent/v1/engine/version", ("PUT",)),  # S9: set the engine's version, from the agent's table (D151)
    }


# --- the pool's side ---


def agent_block(monkeypatch, url="http://127.0.0.1:8095"):
    monkeypatch.setenv("TEST_AGENT_KEY", AGENT_KEY)
    return {"agent": {"url": url, "bearer_env": "TEST_AGENT_KEY"}}


CATALOG = {MODEL: {"variants": [
    {"tag": f"{MODEL}-mlx", "requires": ["apple-silicon"], "runtime_class": "apple-mlx"},
    {"tag": MODEL},
]}}


def test_a_platform_nobody_typed_is_learned_from_the_agent_and_the_right_build_served(monkeypatch):
    with pool_harness(
        [EngineSpec(id="mac", resident={MODEL, f"{MODEL}-mlx"})],
        model_set=[MODEL], catalog=CATALOG, host_overrides={"mac": agent_block(monkeypatch)},
    ) as pool:
        host = pool.supervisor.hosts["mac"]
        assert host.required_tags == {MODEL}  # before the agent is heard from: the fallback build

        pool.supervisor._agent_transport = httpx.ASGITransport(app=agent_app(MAC))
        pool.reprobe()

        assert host.capabilities == {"apple-silicon"} and host.required_tags == {f"{MODEL}-mlx"}
        assert "capabilities_derived" in [e["kind"] for e in pool.supervisor.events.recent(10)]
        with pool.client() as http:
            served = http.post("/api/chat", json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False})
        assert served.headers["X-GPM-Served-Model"] == f"{MODEL}-mlx"
        assert served.headers["X-GPM-Runtime-Class"] == "apple-mlx"


def test_a_typed_platform_the_machine_contradicts_is_reported_and_not_overridden(monkeypatch):
    with pool_harness(
        [EngineSpec(id="box", resident={MODEL}, capabilities=["cuda"])],
        model_set=[MODEL], host_overrides={"box": agent_block(monkeypatch)},
    ) as pool:
        pool.supervisor._agent_transport = httpx.ASGITransport(app=agent_app(MAC))
        pool.reprobe()
        host = pool.supervisor.hosts["box"]
        assert host.capabilities == {"cuda"}  # as written, until the operator corrects it
        conflict = agents.capability_conflict(host.config.capabilities, host.agent)
        assert "cuda" in conflict and "apple-silicon" in conflict


def test_a_silent_agent_changes_nothing_and_the_host_keeps_serving(monkeypatch):
    with pool_harness(
        [EngineSpec(id="box", resident={MODEL})],
        model_set=[MODEL], host_overrides={"box": agent_block(monkeypatch, "http://127.0.0.1:1")},
    ) as pool:
        host = pool.supervisor.hosts["box"]
        assert host.state.value == "ready"
        assert host.agent.reachable is False and "unreachable" in host.agent.detail
        assert host.capabilities == frozenset()  # nothing inferred from the silence


def test_a_missing_agent_key_is_said_plainly_and_does_not_stop_the_pool(monkeypatch):
    overrides = agent_block(monkeypatch)
    monkeypatch.delenv("TEST_AGENT_KEY")
    with pool_harness([EngineSpec(id="box", resident={MODEL})], model_set=[MODEL], host_overrides={"box": overrides}) as pool:
        host = pool.supervisor.hosts["box"]
        assert host.state.value == "ready"
        assert "TEST_AGENT_KEY is not set" in host.agent.detail


HOST_YAML = """
pool: {{name: p, model_set: [m1]}}
auth: {{app_keys: ["gpma_x"], admin_keys: ["gpmx_x"]}}
hosts:
  - id: box
    kind: fixed-remote
    transport: {{type: https, base_url: "https://box.example:11434"}}
    agent: {agent}
"""


def test_an_agent_key_never_crosses_a_network_in_clear_text_by_accident(tmp_path):
    def parse(agent):
        path = tmp_path / "pool.yaml"
        path.write_text(HOST_YAML.format(agent=agent))
        return load_config(path)

    with pytest.raises(ConfigError, match="clear text"):
        parse('{url: "http://box.example:8095", bearer_env: K}')
    parse('{url: "https://box.example:8095", bearer_env: K}')
    parse('{url: "http://box.example:8095", bearer_env: K, allow_insecure: true}')
    with pytest.raises(ConfigError):
        parse('{url: "https://box.example:8095"}')  # an agent with no key named is not an agent


# --- what the console is told ---


def control(pool):
    from fakes.harness import APP_KEY, ServerHandle
    from gpm_server.config import AuthConfig
    from gpm_server.supervisor.control import create_control_app

    config = pool.supervisor.config.model_copy(
        update={"auth": AuthConfig(app_keys=[APP_KEY], admin_keys=["gpmx_console"])}
    )
    return ServerHandle(create_control_app(pool.supervisor, config), pool.loop)


def test_status_reports_the_agent_its_facts_and_a_conflict_but_never_its_key(monkeypatch):
    with pool_harness(
        [EngineSpec(id="box", resident={MODEL}, capabilities=["cuda"])],
        model_set=[MODEL], host_overrides={"box": agent_block(monkeypatch)},
    ) as pool:
        pool.supervisor._agent_transport = httpx.ASGITransport(app=agent_app(MAC))
        pool.reprobe()
        server = control(pool)
        try:
            with httpx.Client(base_url=server.base_url, headers={"Authorization": "Bearer gpmx_console"}) as http:
                raw = http.get("/pool/status")
        finally:
            server.stop()

    agent = raw.json()["hosts"][0]["agent"]
    assert agent["reachable"] and agent["key"] == "set"
    assert agent["facts"]["accelerators"][0]["name"] == "Apple M4 Max"
    assert "apple-silicon" in agent["capability_conflict"]
    assert AGENT_KEY not in raw.text


def test_an_agent_key_admits_nothing_to_the_pool():
    """The third role is as separate as the other two: it opens one host's agent, and neither
    the control API nor the router."""
    with pool_harness([EngineSpec(id="box", resident={MODEL})], model_set=[MODEL]) as pool:
        server = control(pool)
        try:
            refused = httpx.get(f"{server.base_url}/pool/status", headers={"Authorization": f"Bearer {AGENT_KEY}"})
        finally:
            server.stop()
        assert refused.status_code == 403 and refused.json()["error"] == "agent_key_refused"
        with pool.client(key=AGENT_KEY) as http:
            assert http.get("/pool/status").status_code == 401


# =====================================================================================
# Stage 2: the machine holds the model set (docs/spec/host-agent.md §4)
# =====================================================================================

from fakes.fake_ollama import FakeOllama  # noqa: E402
from gpm_server.configplan import MachineNow, plan_changes  # noqa: E402


class Machine:
    """An agent app over a fake engine, called the way the pool calls it."""

    def __init__(self, *, resident=(), available=(), free_gb=100, engine=None, state_path=None, **settings):
        self.engine = engine or FakeOllama(resident=set(resident), available=set(available))
        self.app = create_app(
            Settings(key_hash=fingerprint(AGENT_KEY), **settings), probes=probes("Linux", "x86_64", free_gb=free_gb),
            engine_transport=httpx.ASGITransport(app=self.engine.app), state_path=state_path,
        )
        self.transport = httpx.ASGITransport(app=self.app)

    async def call(self, method, body):
        async with httpx.AsyncClient(transport=self.transport, base_url="http://agent") as http:
            return await http.request(method, "/agent/v1/models", json=body, headers={"Authorization": f"Bearer {AGENT_KEY}"})

    async def hold(self, tags, residency="on_demand"):
        response = await self.call("PUT", {"tags": tags, "residency": residency})
        assert response.status_code == 200, response.text
        await self.app.state.work._task  # let the background work finish
        return (await self.call("PUT", {"tags": tags, "residency": residency})).json()

    def model(self, state, tag):
        return next(m for m in state["models"] if m["tag"] == tag)


async def test_a_missing_model_is_pulled_onto_disk_and_under_on_demand_left_unloaded():
    machine = Machine(available={"have"})
    state = await machine.hold(["have", "want"])
    assert machine.model(state, "want")["on_disk"] and not machine.model(state, "want")["loaded"]
    assert [path for path, _, _ in machine.engine.received if path == "/api/pull"] == ["/api/pull"]  # only the missing one
    assert machine.engine.pinned == set()


async def test_pinned_loads_and_pins_the_whole_set():
    machine = Machine(available={"a", "b"})
    state = await machine.hold(["a", "b"], "pinned")
    assert all(m["loaded"] and m["pinned_by_agent"] for m in state["models"])
    assert machine.engine.pinned == {"a", "b"}


async def test_going_on_demand_releases_what_the_agent_pinned_and_nothing_the_owner_did():
    machine = Machine(available={"pool-model", "owners-own"})
    machine.engine.resident.add("owners-own")
    machine.engine.pinned.add("owners-own")  # pinned by whoever owns this machine, for their reasons
    await machine.hold(["pool-model"], "pinned")
    assert machine.engine.pinned == {"pool-model", "owners-own"}

    await machine.hold(["pool-model"], "on_demand")

    assert machine.engine.pinned == {"owners-own"}


@pytest.mark.parametrize("tag", ["http://evil.example/model", "../../etc/passwd", "a b", "x;rm -rf /", "", "-flag", 7])
async def test_anything_that_is_not_a_plain_model_tag_never_reaches_the_engine(tag):
    machine = Machine()
    refused = await machine.call("PUT", {"tags": [tag], "residency": "on_demand"})
    assert refused.status_code == 400 and refused.json()["error"] == "bad_tag"
    assert inference_seen(machine.engine) == []


async def test_registry_style_tags_are_plain_tags():
    machine = Machine()
    state = await machine.hold(["hf.co/someone/some-model:Q4_K_M", "library/model:7b"])
    assert all(m["on_disk"] for m in state["models"])


async def test_nothing_is_pulled_when_the_disk_is_already_under_the_owners_floor():
    machine = Machine(free_gb=5, min_free_disk_gb=10)
    state = await machine.hold(["want"])
    assert not machine.model(state, "want")["on_disk"]
    assert "floor" in machine.model(state, "want")["error"]
    assert "/api/pull" not in [path for path, _, _ in machine.engine.received]


async def test_a_pull_that_would_cross_the_floor_is_stopped_and_installs_nothing():
    """Over a real socket, because stopping a pull *is* closing its connection: an in-process
    transport hands over a finished response and there is nothing left to stop."""
    from fakes.harness import BackgroundLoop, ServerHandle

    loop = BackgroundLoop()
    engine = FakeOllama()
    engine.pull_bytes = 25 * 10**9  # 30 GB free - 25 = 5 GB left: under the 10 GB floor
    engine.pull_delay_s = 0.3
    server = ServerHandle(engine.app, loop)
    try:
        app = create_app(
            Settings(key_hash=fingerprint(AGENT_KEY), engine_url=server.base_url, min_free_disk_gb=10),
            probes=probes("Linux", "x86_64", free_gb=30),
        )
        machine = Machine.__new__(Machine)
        machine.engine, machine.app, machine.transport = engine, app, httpx.ASGITransport(app=app)
        state = await machine.hold(["big"])
        await asyncio.sleep(1.0)  # longer than the pull would have taken to finish
    finally:
        server.stop()
        loop.stop()
    assert "big" not in engine.available
    assert not machine.model(state, "big")["on_disk"]
    assert "25.0 GB" in machine.model(state, "big")["error"] and "floor" in machine.model(state, "big")["error"]


async def test_a_failed_pull_is_left_alone_for_a_while_not_hammered_every_pass():
    machine = Machine()
    machine.engine.refuse_pull = True
    await machine.hold(["nope"])
    await machine.hold(["nope"])
    await machine.hold(["nope"])
    assert [path for path, _, _ in machine.engine.received].count("/api/pull") == 1


async def test_a_model_that_leaves_the_set_while_it_loads_is_released_in_the_same_pass():
    """The pass loading it is still waiting when the pool's next pass empties the set. The pin
    that load sets is released — not dropped as stale by what the pass saw before the load, with
    the engine left holding the model for good."""
    machine = Machine(available={"a"})
    machine.engine.load_delay_s = 0.5
    assert (await machine.call("PUT", {"tags": ["a"], "residency": "pinned"})).status_code == 200
    loading = machine.app.state.work._task
    await asyncio.sleep(0.1)
    assert (await machine.call("PUT", {"tags": [], "residency": "pinned"})).status_code == 200
    await loading
    assert machine.engine.pinned == set()
    assert machine.app.state.work.pinned_by_agent == set()


async def test_an_error_from_the_system_while_holding_is_reported_and_left_alone_for_a_while():
    """An engine that starts a program writes files and signals processes: the system can say no
    — a directory this user may not write, a process it may not signal. That is a refusal like
    any other: in the snapshot against the model, and not tried again on every pass."""
    from gpm_agent.models import ModelWork

    class Engine:
        name = "fake"
        fetches_without_engine = True
        loads_by_restart = False
        holds = 0

        async def describe(self, client):
            return {"answers": False, "models_on_disk": [{"tag": "m", "size_bytes": 1}], "models_loaded": []}

        async def hold(self, client, tag, *, pinned):
            Engine.holds += 1
            raise PermissionError(13, "Permission denied", "/Volumes/models.gpm-state")

    work = ModelWork(Engine(), None, free_disk_bytes=lambda: None, min_free_disk_gb=0, allow_delete=False)
    for _ in range(3):
        await work.apply({"tags": ["m"], "residency": "pinned"})
        await work._task
    [held] = (await work.snapshot())["models"]
    assert "Permission denied" in held["error"] and not held["loaded"]
    assert Engine.holds == 1


# --- deletion: the owner's choice, inside three bounds (D40) ---


async def test_a_surplus_model_is_listed_and_can_be_deleted():
    machine = Machine(available={"needed", "left-over"})
    state = await machine.hold(["needed"])
    assert [m["tag"] for m in state["surplus"]] == ["left-over"]
    deleted = await machine.call("DELETE", {"tag": "left-over"})
    assert deleted.status_code == 200 and deleted.json()["surplus"] == []
    assert "left-over" not in machine.engine.available


async def test_a_model_the_pool_requires_here_is_never_deleted():
    machine = Machine(available={"needed"})
    await machine.hold(["needed"])
    refused = await machine.call("DELETE", {"tag": "needed"})
    assert refused.status_code == 409 and refused.json()["error"] == "model_required"
    assert "needed" in machine.engine.available


async def test_the_machines_owner_can_forbid_deletion_and_the_pool_cannot_change_that():
    machine = Machine(available={"left-over"}, allow_delete=False)
    refused = await machine.call("DELETE", {"tag": "left-over"})
    assert refused.status_code == 403 and refused.json()["error"] == "owner_forbids_delete"
    assert "left-over" in machine.engine.available
    # And there is no verb, field or header by which the pool could switch it on.
    await machine.call("PUT", {"tags": [], "residency": "on_demand", "allow_delete": True})
    assert (await machine.call("DELETE", {"tag": "left-over"})).status_code == 403


def test_the_supervisors_own_pass_can_never_delete():
    """Bound one: deletion is an operator's act. The control loop does not contain the call."""
    from gpm_server.supervisor import service

    assert "delete_model" not in pathlib.Path(service.__file__).read_text()


# --- the pool's side ---


def test_a_delegated_host_missing_a_model_gets_it_from_its_agent_and_joins(monkeypatch):
    with pool_harness(
        [EngineSpec(id="box", resident=set(), residency="on_demand")],
        model_set=[MODEL], host_overrides={"box": agent_block(monkeypatch)},
    ) as pool:
        host = pool.supervisor.hosts["box"]
        assert host.state.value == "preparing" and "not on disk" in host.last_error
        fake = pool.engines["box"].fake
        app = agent_app(probes("Linux", "x86_64"), engine_app=fake.app)
        pool.supervisor._agent_transport = httpx.ASGITransport(app=app)

        pool.reprobe()                        # the pool says what it wants; the agent starts pulling
        pool.loop.run(_finish(app))
        pool.reprobe()                        # the next pass finds it on disk

        assert host.state.value == "ready", host.last_error
        assert MODEL in fake.available and MODEL not in fake.pinned
        kinds = [e["kind"] for e in pool.supervisor.events.recent(20)]
        assert "agent_model_on_disk" in kinds


async def _finish(app):
    await app.state.work._task


def test_an_agent_told_not_to_manage_models_is_only_ever_asked_for_facts(monkeypatch):
    block = agent_block(monkeypatch)
    block["agent"]["manage_models"] = False
    with pool_harness([EngineSpec(id="box", resident=set(), residency="on_demand")], model_set=[MODEL], host_overrides={"box": block}) as pool:
        fake = pool.engines["box"].fake
        pool.supervisor._agent_transport = httpx.ASGITransport(app=agent_app(probes("Linux", "x86_64"), engine_app=fake.app))
        pool.reprobe()
        pool.reprobe()
        assert "/api/pull" not in [path for path, _, _ in fake.received]
        assert pool.supervisor.hosts["box"].state.value == "preparing"


# --- plan says what a machine would be made to do, before it is ---

PLAN_YAML = """
pool: {{name: p, model_set: [m1]}}
auth: {{app_keys: ["gpma_x"], admin_keys: ["gpmx_x"]}}
catalog:
  m1:
    variants:
      - {{tag: m1-cuda, requires: [cuda], size_gb: 19}}
      - {{tag: m1}}
hosts:
  - id: box
    kind: local
    capabilities: {capabilities}
    transport: {{type: http, base_url: "http://127.0.0.1:1"}}
    agent: {{url: "http://127.0.0.1:8095", bearer_env: K}}
"""


def plan_for(tmp_path, capabilities, machine):
    path = tmp_path / "pool.yaml"
    path.write_text(PLAN_YAML.format(capabilities=capabilities))
    config = load_config(path)
    return {c.kind: c.detail for c in plan_changes(config, config, machines={"box": machine})}


def test_plan_states_the_download_in_gigabytes_and_whether_it_fits(tmp_path):
    roomy = plan_for(tmp_path, "[cuda]", MachineNow(frozenset({"cuda"}), frozenset(), 200 * 10**9))
    assert "['m1-cuda']" in roomy["agent_download"] and "about 19.0 GB" in roomy["agent_download"] and "200 GB free" in roomy["agent_download"]
    tight = plan_for(tmp_path, "[cuda]", MachineNow(frozenset({"cuda"}), frozenset(), 8 * 10**9))
    assert "does not fit" in tight["agent_download"]
    assert "agent_download" not in plan_for(tmp_path, "[cuda]", MachineNow(frozenset({"cuda"}), frozenset({"m1-cuda"}), 200 * 10**9))


def test_plan_uses_the_platform_the_agent_found_when_none_is_typed_and_reports_a_contradiction(tmp_path):
    learned = plan_for(tmp_path, "[]", MachineNow(frozenset({"cuda"}), frozenset(), None))
    assert "m1-cuda" in learned["agent_download"]
    wrong = plan_for(tmp_path, "[cuda]", MachineNow(frozenset({"apple-silicon"}), frozenset(), None))
    assert "apple-silicon" in wrong["capability_conflict"] and "as written" in wrong["capability_conflict"]


DECLARED_YAML = """
pool: {name: p, model_set: [m1, m2], models_per_host: declared}
auth: {app_keys: ["gpma_x"], admin_keys: ["gpmx_x"]}
catalog:
  m1:
    variants:
      - {tag: m1-vllm, engine: vllm}
      - {tag: m1, engine: ollama}
hosts:
  - id: box
    kind: local
    engine: vllm
    models: [m1]
    transport: {type: http, base_url: "http://127.0.0.1:1"}
    agent: {url: "http://127.0.0.1:8095", bearer_env: K}
  - id: other
    kind: local
    models: [m2]
    transport: {type: http, base_url: "http://127.0.0.1:2"}
"""


def test_plan_downloads_only_the_hosts_own_share_in_its_own_engines_build(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(DECLARED_YAML)
    config = load_config(path)

    def plan(on_disk):
        machine = MachineNow(frozenset(), frozenset(on_disk), None)
        return {c.kind: c.detail for c in plan_changes(config, config, machines={"box": machine})}

    assert "['m1-vllm']" in plan(set())["agent_download"]
    assert "agent_download" not in plan({"m1-vllm"})   # m2 is another host's, and m1 is the engine's build


# --- the console's delete button, and its CLI twin ---


def test_deleting_from_the_console_needs_the_tag_twice_and_never_touches_a_catalogued_build(monkeypatch):
    with pool_harness(
        [EngineSpec(id="box", resident={MODEL}, available={"left-over"}, residency="on_demand")],
        model_set=[MODEL], host_overrides={"box": agent_block(monkeypatch)},
    ) as pool:
        fake = pool.engines["box"].fake
        pool.supervisor._agent_transport = httpx.ASGITransport(app=agent_app(probes("Linux", "x86_64"), engine_app=fake.app))
        pool.reprobe()
        server = control(pool)
        try:
            with httpx.Client(base_url=server.base_url, headers={"Authorization": "Bearer gpmx_console"}, timeout=30) as http:
                url = "/pool/hosts/box/models/delete"
                assert http.post(url, json={"tag": "left-over"}).json()["error"] == "not_confirmed"
                assert http.post(url, json={"tag": "left-over", "confirm": "left-ovr"}).status_code == 400
                assert http.post(url, json={"tag": MODEL, "confirm": MODEL}).json()["error"] == "model_required"
                assert "left-over" in fake.available and MODEL in fake.available

                done = http.post(url, json={"tag": "left-over", "confirm": "left-over"})
                assert done.status_code == 200 and "left-over" not in fake.available
                assert "agent_model_deleted" in [e["kind"] for e in pool.supervisor.events.recent(10)]
                assert http.post("/pool/hosts/nobody/models/delete", json={"tag": "x", "confirm": "x"}).status_code == 404
        finally:
            server.stop()


# =====================================================================================
# Stage 3: restarting the engine, and what it starts with (D41)
# =====================================================================================

import sys  # noqa: E402

from fakes.fake_ollama import inference_seen  # noqa: E402


def restartable(tmp_path, *, env_file=True, command=None, **more):
    """A machine whose owner wrote a restart command: here, one that leaves a mark."""
    marker = tmp_path / "restarted"
    command = command or [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('yes'); print('engine restarted')", str(marker)]
    machine = Machine(available={"m"}, restart_command=command,
                      engine_env_file=str(tmp_path / "engine.env") if env_file else None, **more)
    machine.marker, machine.env_file = marker, tmp_path / "engine.env"
    return machine


async def post_engine(machine, body):
    async with httpx.AsyncClient(transport=machine.transport, base_url="http://agent", timeout=60) as http:
        return await http.post("/agent/v1/engine", json=body, headers={"Authorization": f"Bearer {AGENT_KEY}"})


async def test_the_owners_command_runs_and_the_pools_numbers_become_the_engines_environment(tmp_path):
    machine = restartable(tmp_path)
    answer = await post_engine(machine, {"settings": {"workers": 3, "models_held": 2}})
    body = answer.json()
    assert answer.status_code == 200 and machine.marker.read_text() == "yes"
    assert body["restart_exit_code"] == 0 and "engine restarted" in body["restart_output"] and body["engine_answers"]
    assert body["settings_written"] and body["applied"] == {"OLLAMA_NUM_PARALLEL": "3", "OLLAMA_MAX_LOADED_MODELS": "2"}
    assert "OLLAMA_NUM_PARALLEL=3" in machine.env_file.read_text()
    facts = (await get_facts(machine.app, AGENT_KEY)).json()
    assert facts["engine_environment"]["OLLAMA_NUM_PARALLEL"] == "3"
    assert facts["allows"] == {"delete": True, "restart": True, "engine_settings": True}
    # Asked again with the same numbers: nothing is rewritten.
    assert (await post_engine(machine, {"settings": {"workers": 3, "models_held": 2}})).json()["settings_written"] is False


async def test_a_plain_restart_writes_nothing(tmp_path):
    machine = restartable(tmp_path)
    assert (await post_engine(machine, {"settings": None})).status_code == 200
    assert machine.marker.exists() and not machine.env_file.exists()


@pytest.mark.parametrize("body", [
    {"settings": {"workers": "3; rm -rf /", "models_held": 1}},
    {"settings": {"workers": 3, "models_held": 1, "OLLAMA_HOST": "0.0.0.0"}},
    {"settings": {"workers": 3, "models_held": 1, "context": "$(id)"}},
    {"settings": {"workers": 0, "models_held": 1}},
    {"settings": {"workers": 9999, "models_held": 1}},
    {"settings": {"workers": True, "models_held": 1}},
    {"settings": {"workers": 3}},
    {"settings": {"workers": 3, "models_held": 1, "cards_per_copy": 3}},
    {"settings": {"workers": 3, "models_held": 1, "cards_per_copy": 16}},
    {"settings": None, "command": ["sh", "-c", "id"]},
    {"settings": None, "restart_command": ["id"]},
    {"settings": None, "engine_env_file": "/etc/passwd"},
])
async def test_the_pool_can_say_numbers_and_nothing_else(tmp_path, body):
    """No name, no text, no command and no path from a request reaches a file or a process."""
    machine = restartable(tmp_path)
    assert (await post_engine(machine, body)).status_code == 400
    assert not machine.marker.exists() and not machine.env_file.exists()


async def test_an_engine_that_cannot_split_a_model_is_not_asked_to(tmp_path):
    """Ollama places a model on the cards itself (D114); the pool refuses this at load, and the
    agent refuses it again rather than restart the engine as though it had been done."""
    machine = restartable(tmp_path)
    refused = await post_engine(machine, {"settings": {"workers": 2, "models_held": 1, "cards_per_copy": 2}})
    assert refused.status_code == 409 and refused.json()["error"] == "engine_cannot_split"
    assert not machine.marker.exists() and not machine.env_file.exists()


def test_a_split_reaches_vllms_launcher_as_a_number_and_is_read_back():
    from gpm_agent.engines import VllmFacts

    engine = VllmFacts()
    written = engine.launch_environment(8, 1, cards_per_copy=2)
    assert written["GPM_VLLM_CARDS_PER_COPY"] == "2"
    assert engine.settings_from_environment(written)["cards_per_copy"] == 2
    unsplit = engine.launch_environment(8, 1)
    assert "GPM_VLLM_CARDS_PER_COPY" not in unsplit, "a host that splits nothing keeps the file it had"
    assert "cards_per_copy" not in engine.settings_from_environment(unsplit)


async def test_without_the_owners_command_there_is_no_restart_and_the_pool_cannot_supply_one(tmp_path):
    machine = Machine(available={"m"})
    refused = await post_engine(machine, {"settings": None})
    assert refused.status_code == 409 and refused.json()["error"] == "owner_has_not_enabled_restart"
    no_file = restartable(tmp_path, env_file=False)
    refused = await post_engine(no_file, {"settings": {"workers": 2, "models_held": 1}})
    assert refused.json()["error"] == "owner_has_not_enabled_settings" and not no_file.marker.exists()


async def test_a_restart_command_that_fails_or_does_not_exist_is_reported_not_raised(tmp_path):
    failing = restartable(tmp_path, command=[sys.executable, "-c", "import sys; print('no such service'); sys.exit(3)"])
    body = (await post_engine(failing, {"settings": None})).json()
    assert body["restart_exit_code"] == 3 and "no such service" in body["restart_output"]
    missing = restartable(tmp_path, command=["/nonexistent/restart-the-engine"])
    body = (await post_engine(missing, {"settings": None})).json()
    assert body["restart_exit_code"] is None and "could not start" in body["restart_output"]


def test_the_pool_restarts_only_when_an_operator_says_so_with_the_host_named_twice(monkeypatch, tmp_path):
    # The pass restarts no engine that serves its set (D41); its one restart is of an engine that
    # serves none of it yet, once its models are on disk (D160, tested in test_engine_auto.py).
    with pool_harness([EngineSpec(id="box", resident={MODEL}, workers=3)], model_set=[MODEL], host_overrides={"box": agent_block(monkeypatch)}) as pool:
        machine = restartable(tmp_path)
        pool.supervisor._agent_transport = machine.transport
        pool.reprobe()
        pool.reprobe()
        assert not machine.marker.exists(), "never from the pass"
        server = control(pool)
        try:
            with httpx.Client(base_url=server.base_url, headers={"Authorization": "Bearer gpmx_console"}, timeout=60) as http:
                status = http.get("/pool/status").json()["hosts"][0]["agent"]
                assert status["wanted_engine_settings"] == {"workers": 3, "models_held": 1}
                url = "/pool/hosts/box/engine/restart"
                assert http.post(url, json={"apply_settings": True}).json()["error"] == "not_confirmed"
                assert not machine.marker.exists()
                done = http.post(url, json={"apply_settings": True, "confirm": "box"})
                assert done.status_code == 200 and done.json()["applied"]["OLLAMA_NUM_PARALLEL"] == "3"
        finally:
            server.stop()
        event = next(e for e in pool.supervisor.events.recent(10) if e["kind"] == "agent_engine_restarted")
        assert "with the pool's settings" in event["summary"]


# =====================================================================================
# Stage 4: an agent on the far side of an SSH tunnel
# =====================================================================================


def test_an_agent_behind_the_hosts_ssh_tunnel_is_reached_through_a_second_forward(monkeypatch):
    """The agent stays on loopback over there. Real sockets end to end: a real agent app on a
    port, and the same stand-in for `ssh -N -L` the engine's tunnel is tested with."""
    from fakes.harness import BackgroundLoop, ServerHandle

    far_side = BackgroundLoop()
    agent_server = ServerHandle(agent_app(MAC), far_side)
    monkeypatch.setenv("TEST_AGENT_KEY", AGENT_KEY)
    try:
        with pool_harness(
            [EngineSpec(id="remote", resident={MODEL}, kind="fixed-remote", transport="tunnel")],
            model_set=[MODEL],
            host_overrides={"remote": {"agent": {"remote_port": agent_server.port, "bearer_env": "TEST_AGENT_KEY", "manage_models": False}}},
        ) as pool:
            host = pool.supervisor.hosts["remote"]
            assert host.agent_tunnel is not None and host.agent_tunnel is not host.tunnel
            assert host.agent_tunnel.local_port != host.tunnel.local_port
            for _ in range(20):  # the forward comes up beside the pass, not inside it
                pool.reprobe()
                if host.agent and host.agent.reachable:
                    break
                time.sleep(0.25)
            assert host.agent.reachable, host.agent.detail
            assert host.agent.facts["accelerators"][0]["name"] == "Apple M4 Max"
            assert host.state.value == "ready"  # and the engine's own tunnel is undisturbed
    finally:
        agent_server.stop()
        far_side.stop()


def test_adding_an_agent_to_a_tunnelled_host_does_not_touch_the_engines_tunnel(monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", AGENT_KEY)
    with pool_harness([EngineSpec(id="remote", resident={MODEL}, kind="fixed-remote", transport="tunnel")], model_set=[MODEL]) as pool:
        host = pool.supervisor.hosts["remote"]
        engine_tunnel, client = host.tunnel, host.client
        changed = pool.config.model_copy(deep=True)
        from gpm_server.config import AgentConfig
        changed.hosts[0].agent = AgentConfig(remote_port=8095, bearer_env="TEST_AGENT_KEY")

        pool.loop.run(_apply(pool.supervisor, changed))

        assert pool.supervisor.hosts["remote"] is host and host.tunnel is engine_tunnel and host.client is client
        assert host.agent_tunnel is not None
        changed_again = changed.model_copy(deep=True)
        changed_again.hosts[0].agent = None
        pool.loop.run(_apply(pool.supervisor, changed_again))
        assert host.agent_tunnel is None and host.tunnel is engine_tunnel


async def _apply(supervisor, config):
    supervisor.apply_config(config)


def test_an_agent_port_needs_a_tunnel_to_travel_through_and_an_agent_needs_one_address(tmp_path):
    def parse(transport, agent):
        path = tmp_path / "pool.yaml"
        path.write_text(
            'pool: {name: p, model_set: [m1]}\nauth: {app_keys: ["gpma_x"], admin_keys: ["gpmx_x"]}\n'
            f"hosts:\n  - id: box\n    kind: fixed-remote\n    transport: {transport}\n    agent: {agent}\n"
        )
        return load_config(path)

    tunnel = "{type: tunnel, ssh_host: box.example, remote_port: 11434}"
    parse(tunnel, "{remote_port: 8095, bearer_env: K}")
    with pytest.raises(ConfigError, match="SSH tunnel"):
        parse('{type: https, base_url: "https://box.example:11434"}', "{remote_port: 8095, bearer_env: K}")
    with pytest.raises(ConfigError, match="exactly one"):
        parse(tunnel, '{remote_port: 8095, url: "https://box.example:8095", bearer_env: K}')
    with pytest.raises(ConfigError, match="exactly one"):
        parse(tunnel, "{bearer_env: K}")


# =====================================================================================
# Loose ends
# =====================================================================================


async def test_an_agent_that_restarts_still_releases_the_pins_it_set_before(tmp_path):
    """It used to forget: after its own restart a later switch to on_demand released nothing,
    and the pool's models stayed pinned on a machine whose owner wanted the memory back."""
    state = tmp_path / "agent.state.json"
    first = Machine(available={"pool-model", "owners-own"}, state_path=state)
    first.engine.resident.add("owners-own")
    first.engine.pinned.add("owners-own")
    await first.hold(["pool-model"], "pinned")
    assert stat.S_IMODE(state.stat().st_mode) == 0o600

    restarted = Machine(engine=first.engine, state_path=state)  # a new agent process, same machine
    await restarted.hold(["pool-model"], "on_demand")

    assert first.engine.pinned == {"owners-own"}  # its own pin released; the owner's left alone
    assert "pool-model" not in state.read_text()


async def test_a_remembered_pin_the_engine_has_since_dropped_is_forgotten_not_reasserted(tmp_path):
    state = tmp_path / "agent.state.json"
    state.write_text('{"pinned_by_agent": ["gone-model", "../not-a-tag"]}')
    machine = Machine(available={"gone-model"}, state_path=state)  # on disk, not loaded: the engine restarted
    await machine.hold([], "on_demand")
    assert machine.engine.resident == set()  # releasing would have *loaded* it; it must not
    assert machine.app.state.work.pinned_by_agent == set()


# --- the heartbeat verb, on a host the pool created (D63) ---


async def post(app, path, key=AGENT_KEY, **kwargs):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
        return await http.post(path, headers=headers, **kwargs)


async def test_the_heartbeat_postpones_the_timer_on_a_rented_host(tmp_path):
    """It touches one file — and can only ever postpone a shutdown the pool could equally
    cause by going silent, which is why it needs no other bound."""
    beat_file = tmp_path / "state" / "heartbeat"
    app = agent_app(heartbeat_file=str(beat_file))

    answered = await post(app, "/agent/v1/heartbeat")

    assert answered.status_code == 200 and answered.json()["beat"] is True
    assert beat_file.exists(), "the timer's file is what the verb exists to touch"

    before = beat_file.stat().st_mtime_ns
    time.sleep(0.01)
    await post(app, "/agent/v1/heartbeat")
    assert beat_file.stat().st_mtime_ns > before


async def test_a_machine_nobody_rented_has_no_timer_to_beat():
    answered = await post(agent_app(), "/agent/v1/heartbeat")
    assert answered.status_code == 409
    assert answered.json()["error"] == "no_timer"


async def test_the_heartbeat_needs_the_agent_key_like_everything_else(tmp_path):
    app = agent_app(heartbeat_file=str(tmp_path / "heartbeat"))
    assert (await post(app, "/agent/v1/heartbeat", key=None)).status_code == 401
    assert not (tmp_path / "heartbeat").exists()


# --- each model loaded as its own download finishes (D57) ---


async def test_a_model_is_loaded_as_soon_as_its_own_download_finishes():
    """Not pull-everything-then-load-everything: that left the accelerator idle through the
    whole last phase, on a host that is billing (measured live: 78 seconds)."""
    with pool_harness([EngineSpec(id="e", resident=set(), available=set())], model_set=["a", "b"]) as pool:
        engine = pool.engines["e"].fake
        # Both slow enough to be seen: an instant download and an instant load overlap
        # trivially and prove nothing about the order they were asked for in.
        engine.pull_delay_s = 0.05
        engine.load_delay_s = 0.15
        app = agent_app(engine_app=engine.app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
            await http.put(
                "/agent/v1/models",
                headers={"Authorization": f"Bearer {AGENT_KEY}"},
                json={"tags": ["a", "b"], "residency": "pinned"},
            )
            for _ in range(200):
                await asyncio.sleep(0.05)
                if {"a", "b"} <= engine.resident:
                    break

    ordered = [path for path, _body, _headers in engine.received if path in ("/api/pull", "/api/generate", "/api/show", "/api/embed")]
    pulls = [i for i, path in enumerate(ordered) if path == "/api/pull"]
    loads = [i for i, path in enumerate(ordered) if path != "/api/pull"]
    assert len(pulls) == 2 and loads, "both models pulled, and loading happened"
    # Measured inside the engine rather than inferred from the order requests arrived in. That
    # order was the old proxy for this, and it cannot tell "loaded while the next downloads"
    # (D83) from "loaded after every download" — both put the loads last. Overlap can.
    assert engine.both_at_once, "no load ran while a download was still going"


async def test_the_agent_holds_an_embedding_model_through_the_endpoint_it_serves():
    """Found live, at the cost of a healthy host: the agent pinned every model through
    `generate`, the engine refused the embedding model with a 400, and the pool destroyed the
    host for "could not hold the model set" with the other two models already loaded."""
    tags = ["a-chat-model", "an-embed-model"]
    with pool_harness([EngineSpec(id="e", resident=set(), available=set())], model_set=tags) as pool:
        engine = pool.engines["e"].fake
        app = agent_app(engine_app=engine.app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
            report = None
            for _ in range(60):
                answer = await http.put(
                    "/agent/v1/models",
                    headers={"Authorization": f"Bearer {AGENT_KEY}"},
                    json={"tags": tags, "residency": "pinned"},
                )
                report = answer.json()
                if set(tags) <= engine.resident:
                    break
                await asyncio.sleep(0.05)

    assert set(tags) <= engine.resident, report
    assert not [m for m in report["models"] if m.get("error")], report


async def test_a_load_that_never_finishes_is_a_failure_rather_than_a_silent_stall():
    """Found live: a machine whose engine had fallen back to the processor took a 26B model
    past sixteen minutes. The agent sat in one `post` with no deadline, and because it works
    one tag at a time the rest of the model set never started — while it answered every poll
    with no error at all. A load now has a finite deadline, and exceeding it is a fact."""
    from gpm_agent import engines

    assert engines.LOAD_TIMEOUT_S and engines.LOAD_TIMEOUT_S < 3600, "a load must have a deadline"
    source = (Path(engines.__file__)).read_text()
    hold = source[source.index("async def hold"):]
    hold = hold[: hold.index("\n    async def ")]
    # The code, not the prose about it: the comment above the call names the old behaviour.
    code = "\n".join(line for line in hold.splitlines() if not line.strip().startswith("#"))
    assert "timeout=None" not in code, "the load can still wait forever"
    assert "timeout=LOAD_TIMEOUT_S" in code


async def test_a_load_that_times_out_is_recorded_against_that_model():
    """The deadline is only worth having if the pool is told which model failed and why."""
    with pool_harness([EngineSpec(id="e", resident=set(), available={"a"})], model_set=["a"]) as pool:
        engine = pool.engines["e"].fake
        app = agent_app(engine_app=engine.app)
        from gpm_agent import engines as agent_engines

        async def times_out(self, client, tag, *, pinned):
            raise httpx.ReadTimeout("the engine never finished loading")

        original = agent_engines.OllamaFacts.hold
        agent_engines.OllamaFacts.hold = times_out
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
                report = None
                for _ in range(40):
                    answer = await http.put(
                        "/agent/v1/models",
                        headers={"Authorization": f"Bearer {AGENT_KEY}"},
                        json={"tags": ["a"], "residency": "pinned"},
                    )
                    report = answer.json()
                    failed = [m for m in report["models"] if m.get("error")]
                    if failed:
                        break
                    await asyncio.sleep(0.05)
        finally:
            agent_engines.OllamaFacts.hold = original

    assert failed, report
    assert "never finished loading" in failed[0]["error"] or "timeout" in failed[0]["error"].lower()


async def test_a_model_loads_while_the_next_one_downloads():
    """D57 only ever half-delivered, and the owner spotted the other half live: a model was
    loaded as soon as its own download finished, but the load then blocked the next download,
    so the network idled through every load — about a minute per large model on a billing host.

    Proven by the clock, not by reading the code: with a load slower than a pull, a sequential
    agent cannot finish in less than the sum of them.
    """
    tags = ["a", "b", "c"]
    with pool_harness([EngineSpec(id="e", resident=set(), available=set())], model_set=tags) as pool:
        engine = pool.engines["e"].fake
        # A download is the long pole on a real host; a load is a matter of a minute. The
        # fake sleeps per chunk, so these are small.
        engine.pull_delay_s = 0.05
        engine.load_delay_s = 0.15
        app = agent_app(engine_app=engine.app)

        deadline = time.monotonic() + 30
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as http:
            while time.monotonic() < deadline:
                await http.put(
                    "/agent/v1/models",
                    headers={"Authorization": f"Bearer {AGENT_KEY}"},
                    json={"tags": tags, "residency": "pinned"},
                )
                if set(tags) <= engine.resident:
                    break
                await asyncio.sleep(0.02)

    assert set(tags) <= engine.resident, "the model set never landed"
    # The claim itself, watched from inside the engine: a download and a load in flight at the
    # same moment. Timing alone is a benchmark, and benchmarks are flaky on a loaded machine.
    assert engine.both_at_once, "every load still had the downloads waiting on it"
    # The fake sleeps per chunk, so the absolute numbers are its business, not a benchmark.
    # What is asserted is the overlap itself, above.


async def test_only_one_model_loads_at_a_time():
    """Overlapping a load with a *download* costs nothing but disk. Overlapping two loads would
    have them contend for accelerator memory, which is the one way this could cost more than it
    saves."""
    from gpm_agent import models as agent_models

    source = Path(agent_models.__file__).read_text()
    work = source[source.index("async def _work"):]
    work = work[: work.index("\n    def ")]
    assert "self._loading is None or self._loading.done()" in work, "a second load can start"
    assert "asyncio.create_task(" in work


async def test_a_slow_capability_check_does_not_cost_a_host(monkeypatch):
    """Exactly what destroyed a live host at 19:02. `/api/show` inherited the agent client's
    ten-second default while an 18.6GB download ran beside it (D83 put the load there), the
    agent recorded `gemma4:26b: ReadTimeout`, and the pool read that as "could not hold the
    model set" and destroyed a host that was perfectly healthy."""
    from gpm_agent import engines

    calls = []

    class SlowShow:
        async def post(self, path, json=None, timeout=None):
            calls.append((path, timeout))
            if path == "/api/show":
                raise httpx.ReadTimeout("the engine is busy downloading")
            return httpx.Response(200, json={}, request=httpx.Request("POST", "http://e" + path))

    await engines.OllamaFacts().hold(SlowShow(), "a-model", pinned=True)

    assert ("/api/show", engines.SHOW_TIMEOUT_S) in calls, "the check still uses the short default"
    assert any(p == "/api/generate" for p, _ in calls), "the model was never loaded at all"


async def test_an_embedding_model_is_still_held_when_the_engine_will_not_say():
    """The check being optional must not undo the fix it was added for: with no answer, both
    endpoints are tried, so an embedding model is still loaded."""
    from gpm_agent import engines

    tried = []

    class Unhelpful:
        async def post(self, path, json=None, timeout=None):
            tried.append(path)
            request = httpx.Request("POST", "http://e" + path)
            if path == "/api/show":
                return httpx.Response(500, json={}, request=request)
            if path == "/api/generate":   # what a real engine answers for an embedding model
                return httpx.Response(400, json={"error": "does not support generate"}, request=request)
            return httpx.Response(200, json={}, request=request)

    await engines.OllamaFacts().hold(Unhelpful(), "an-embed-model", pinned=True)

    assert tried == ["/api/show", "/api/generate", "/api/embed"]


async def test_a_refusal_that_is_not_about_the_endpoint_is_not_retried_elsewhere():
    """A 500 is about the model, not the endpoint; trying the other one would hide it."""
    from gpm_agent import engines

    tried = []

    class Broken:
        async def post(self, path, json=None, timeout=None):
            tried.append(path)
            request = httpx.Request("POST", "http://e" + path)
            return httpx.Response(200 if path == "/api/show" else 500, json={}, request=request)

    with pytest.raises(engines.EngineRefused, match="500"):
        await engines.OllamaFacts().hold(Broken(), "a-model", pinned=True)

    assert tried.count("/api/embed") == 0, "a real failure was retried on the wrong endpoint"


def test_an_answer_not_in_the_shape_of_facts_is_no_answer_never_an_exception(monkeypatch):
    # A rented host is not trusted: its agent's JSON of the wrong shape must not reach the pass
    # (the security review's finding).
    import asyncio

    import httpx
    from gpm_server.config import AgentConfig

    monkeypatch.setenv("GPM_TEST_AGENT_KEY", "agent-key-for-tests")
    agent = AgentConfig(url="https://agent.invalid", bearer_env="GPM_TEST_AGENT_KEY")
    for body in ([1, 2], {"agent": "x"}, {"agent": {"protocol": "1"}, "engine": "x"}):
        transport = httpx.MockTransport(lambda request, body=body: httpx.Response(200, json=body))
        view = asyncio.run(agents.ask(agent, transport=transport))
        assert not view.reachable and "shape" in view.detail


def test_the_version_the_agent_reports_is_its_packages():
    import tomllib
    from pathlib import Path

    import gpm_agent

    pyproject = Path(__file__).resolve().parents[1] / "agent" / "pyproject.toml"
    assert gpm_agent.__version__ == tomllib.loads(pyproject.read_text())["project"]["version"]
