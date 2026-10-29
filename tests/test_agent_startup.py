"""The agent starts with the machine, as its owner, on macOS, Linux and Windows (D154).

docs/spec/host-agent.md §2.3. Each system is described by a `Machine` — its paths, its user, what
its commands answer — and its files are a dictionary, so the exact launchd agent, systemd unit and
Windows task, and every command run to load them, are checked here on any machine. Nothing here
runs launchctl, systemctl, loginctl, schtasks or sudo.
"""

from __future__ import annotations

import ast
import json
import ntpath
import os
import pathlib
import plistlib
import socket
import stat
import subprocess
import sys
import xml.etree.ElementTree as ElementTree

import gpm_agent
import pytest
from gpm_agent import cli, startup
from gpm_agent.settings import Settings, fingerprint, save
from gpm_agent.startup import Machine, Ran

MAC_HOME = "/Users/owner"
MAC_SETTINGS = f"{MAC_HOME}/.config/gpm-agent/agent.json"
MAC_PYTHON = f"{MAC_HOME}/.local/share/uv/tools/gpm-agent/bin/python"
LINUX_HOME = "/home/owner"
LINUX_SETTINGS = f"{LINUX_HOME}/.config/gpm-agent/agent.json"
LINUX_PYTHON = f"{LINUX_HOME}/.local/share/uv/tools/gpm-agent/bin/python"
WIN_HOME = "C:\\Users\\owner"
WIN_SETTINGS = WIN_HOME + "\\.config\\gpm-agent\\agent.json"
WIN_SCRIPTS = WIN_HOME + "\\AppData\\Roaming\\uv\\tools\\gpm-agent\\Scripts"
SCHTASKS = "C:\\Windows\\System32\\schtasks.exe"


class Files(startup.Files):
    """A file system that is a dictionary of path to bytes, in the described system's own paths."""

    def __init__(self, content=None):
        self.content = dict(content or {})
        self.private = set()

    def exists(self, path):
        return path in self.content

    def read(self, path):
        return self.content.get(path)

    def write(self, path, content):
        self.content[path] = content

    def create_private(self, path):
        self.content.setdefault(path, b"")
        self.private.add(path)

    def delete(self, path):
        self.content.pop(path, None)


class Runner:
    """Stands in for `startup.run_command`: records each command, and answers from a table —
    a `Ran`, or a list of them given out in turn — or with success."""

    def __init__(self, answers=None):
        self.answers = {tuple(argv): answer for argv, answer in (answers or {}).items()}
        self.calls = []

    def __call__(self, argv, *, on_terminal=False):
        self.calls.append((list(argv), on_terminal))
        answer = self.answers.get(tuple(argv), Ran(0))
        return answer.pop(0) if isinstance(answer, list) else answer

    @property
    def commands(self):
        return [argv for argv, _ in self.calls]


def mac(runner=None, files=None, **changes):
    machine = Machine(
        system="Darwin", home=MAC_HOME, user="owner", uid=501, executable=MAC_PYTHON,
        prefix=f"{MAC_HOME}/.local/share/uv/tools/gpm-agent",
        path="/opt/homebrew/bin:.:/usr/bin:/bin:bin:/opt/homebrew/bin", environ={"HF_TOKEN": "hf_notreal"},
        interactive=True, run=runner or Runner(), which=lambda name: f"/usr/bin/{name}",
        answers=lambda host, port: False, files=files or Files(), sleep=lambda seconds: None,
    )
    for name, value in changes.items():
        setattr(machine, name, value)
    return machine


def linux(runner=None, files=None, **changes):
    machine = mac(runner, files, system="Linux", home=LINUX_HOME, uid=1000, executable=LINUX_PYTHON,
                  prefix=f"{LINUX_HOME}/.local/share/uv/tools/gpm-agent", path="/usr/local/bin:/usr/bin:/bin",
                  environ={})
    for name, value in changes.items():
        setattr(machine, name, value)
    return machine


def windows(runner=None, files=None, **changes):
    files = files or Files({WIN_SCRIPTS + "\\pythonw.exe": b""})
    machine = mac(runner, files, system="Windows", home=WIN_HOME, user="DESKTOP\\owner", uid=None,
                  executable=WIN_SCRIPTS + "\\python.exe", prefix=WIN_HOME + "\\AppData\\Roaming\\uv\\tools\\gpm-agent",
                  path="C:\\Windows\\System32;C:\\Windows", environ={"SystemRoot": "C:\\Windows"})
    for name, value in changes.items():
        setattr(machine, name, value)
    return machine


MAC_PLIST = f"{MAC_HOME}/Library/LaunchAgents/gpm-agent.plist"
MAC_QUERY = ("launchctl", "print", "gui/501/gpm-agent")
LINUX_UNIT = f"{LINUX_HOME}/.config/systemd/user/gpm-agent.service"
LINUX_INSTALLED = ("systemctl", "--user", "show", "-p", "FragmentPath", "gpm-agent.service")
WIN_DEFINITION = (SCHTASKS, "/Query", "/TN", "gpm-agent", "/XML")
NO_TASK = Ran(1, "ERROR: The system cannot find the file specified.")


def loaded_from(plist, state="running"):
    """`launchctl print` of a job loaded from `plist`, as launchd prints it."""
    return Ran(0, f"gui/501/gpm-agent = {{\n\tactive count = 1\n\tpath = {plist}\n\ttype = LaunchAgent\n\tstate = {state}\n}}")


def registered_as(entry):
    """`schtasks /Query /XML` of a task registered from `entry`'s definition."""
    return Ran(0, entry.content.decode("utf-16"))


# --- what each system is given: exactly ---


def test_the_name_follows_the_settings_file_so_two_agents_live_side_by_side():
    assert startup.name_for(MAC_SETTINGS) == "gpm-agent"
    assert startup.name_for(f"{MAC_HOME}/.config/gpm-agent/whisper.json") == "gpm-agent-whisper"
    assert startup.name_for("/srv/GPM Agents/Whisper_Large.json") == "gpm-agent-whisper-large"
    assert startup.name_for(WIN_SETTINGS, "Windows") == "gpm-agent"
    odd = startup.name_for("/srv/日本.json")
    assert odd.startswith("gpm-agent-") and len(odd) == len("gpm-agent-") + 8


def test_the_command_is_this_agents_own_interpreter_isolated_serving_one_settings_file():
    assert startup.serve_command(mac(), MAC_SETTINGS) == [
        MAC_PYTHON, "-I", "-m", "gpm_agent.cli", "--config", MAC_SETTINGS, "serve",
    ]
    assert startup.serve_command(linux(), LINUX_SETTINGS)[0] == LINUX_PYTHON


def test_on_windows_the_windowless_interpreter_runs_and_writes_its_output_beside_its_settings():
    log = WIN_HOME + "\\.config\\gpm-agent\\gpm-agent.log"
    assert startup.serve_command(windows(), WIN_SETTINGS) == [
        WIN_SCRIPTS + "\\pythonw.exe", "-I", "-m", "gpm_agent.cli", "--config", WIN_SETTINGS, "serve",
        "--log-file", log,
    ]
    # No pythonw.exe beside it: the console interpreter, which still runs, with a window.
    assert startup.serve_command(windows(files=Files()), WIN_SETTINGS)[0] == WIN_SCRIPTS + "\\python.exe"


def test_macos_gets_a_launchd_agent_that_starts_at_login_and_again_on_failure():
    entry = startup.entry_for(mac(), MAC_SETTINGS)
    assert entry.file == f"{MAC_HOME}/Library/LaunchAgents/gpm-agent.plist"
    log = f"{MAC_HOME}/.config/gpm-agent/gpm-agent.log"
    assert plistlib.loads(entry.content) == {
        "Label": "gpm-agent",
        "ProgramArguments": [MAC_PYTHON, "-I", "-m", "gpm_agent.cli", "--config", MAC_SETTINGS, "serve"],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Interactive",
        "WorkingDirectory": f"{MAC_HOME}/.config/gpm-agent",
        # The owner's PATH, without relative entries or repeats — and no other variable: the
        # owner's HF_TOKEN is never written into a file.
        "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/bin:/bin"},
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }
    assert b"hf_notreal" not in entry.content


def test_linux_gets_a_systemd_user_unit_restarted_on_failure():
    entry = startup.entry_for(linux(), LINUX_SETTINGS)
    assert entry.file == f"{LINUX_HOME}/.config/systemd/user/gpm-agent.service"
    assert entry.name == "gpm-agent.service"
    assert entry.content.decode() == "\n".join([
        "# Written by `gpm-agent startup add`; `gpm-agent startup remove` takes it away.",
        f"# Settings: {LINUX_SETTINGS}",
        "[Unit]",
        f"Description=GPM host agent, {LINUX_SETTINGS}",
        "",
        "[Service]",
        "Type=simple",
        f"ExecStart={LINUX_PYTHON} -I -m gpm_agent.cli --config {LINUX_SETTINGS} serve",
        f"WorkingDirectory={LINUX_HOME}/.config/gpm-agent",
        'Environment="PATH=/usr/local/bin:/usr/bin:/bin"',
        "Restart=on-failure",
        "RestartSec=5",
        "KillMode=process",  # the engines it started outlive a restart of the agent
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])
    assert startup.unit_directory(linux(environ={"XDG_CONFIG_HOME": "/srv/conf"})) == "/srv/conf/systemd/user"


def test_a_unit_says_an_odd_path_literally():
    """Spaces, quotes, `%` (systemd's specifiers) and `$` (its variables) all reach the agent as
    written."""
    settings = '/home/owner/my "agents"/100% $HOME.json'
    unit = startup.entry_for(linux(), settings).content.decode()
    (exec_start,) = [line for line in unit.splitlines() if line.startswith("ExecStart=")]
    assert exec_start.endswith(r' --config "/home/owner/my \"agents\"/100%% $$HOME.json" serve')
    assert 'WorkingDirectory=/home/owner/my "agents"' in unit


def test_windows_gets_a_task_at_logon_as_the_owner_with_no_password_and_no_time_limit():
    entry = startup.entry_for(windows(), WIN_SETTINGS)
    assert entry.file == WIN_HOME + "\\.config\\gpm-agent\\gpm-agent.task.xml"
    assert entry.content.startswith(b"\xff\xfe")  # UTF-16 with its mark, as Task Scheduler writes
    task = ElementTree.fromstring(entry.content.decode("utf-16").replace('encoding="UTF-16"', ""))
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}

    def text(path):
        return task.find(path, ns).text

    # The settings file it serves, and a mark of it that survives Task Scheduler's code page.
    assert text("t:RegistrationInfo/t:Description") == (
        f"GPM host agent, {WIN_SETTINGS} (id {startup.task_mark(WIN_SETTINGS)})"
    )
    assert startup.task_mark(WIN_SETTINGS) == startup.task_mark(WIN_SETTINGS.upper())  # paths ignore case there
    assert text("t:Triggers/t:LogonTrigger/t:UserId") == "DESKTOP\\owner"
    assert text("t:Principals/t:Principal/t:UserId") == "DESKTOP\\owner"
    assert text("t:Principals/t:Principal/t:LogonType") == "InteractiveToken"  # no password stored
    assert text("t:Principals/t:Principal/t:RunLevel") == "LeastPrivilege"
    # Task Scheduler's defaults, each wrong for an agent, all overridden.
    assert text("t:Settings/t:ExecutionTimeLimit") == "PT0S"
    assert text("t:Settings/t:DisallowStartIfOnBatteries") == "false"
    assert text("t:Settings/t:StopIfGoingOnBatteries") == "false"
    assert text("t:Settings/t:Priority") == "5"
    assert text("t:Actions/t:Exec/t:Command") == WIN_SCRIPTS + "\\pythonw.exe"
    assert text("t:Actions/t:Exec/t:Arguments") == subprocess.list2cmdline(entry.command[1:])
    assert text("t:Actions/t:Exec/t:WorkingDirectory") == WIN_HOME + "\\.config\\gpm-agent"


def test_a_task_quotes_and_escapes_an_odd_path():
    settings = "C:\\Users\\owner\\GPM & Co\\agent <one>.json"
    entry = startup.entry_for(windows(), settings)
    body = entry.content.decode("utf-16")
    assert "GPM &amp; Co" in body and "&lt;one&gt;" in body and "& Co" not in body.replace("&amp; Co", "")
    task = ElementTree.fromstring(body.replace('encoding="UTF-16"', ""))
    arguments = task.find(".//{*}Arguments").text
    assert f'--config "{settings}" serve' in arguments


# --- setting it up, per system ---


def test_macos_writes_the_agent_and_loads_it_into_the_owners_session():
    runner = Runner({("launchctl", "print", "gui/501/gpm-agent"): Ran(113, "not found")})
    machine = mac(runner)
    assert startup.add(machine, MAC_SETTINGS) == 0
    plist = f"{MAC_HOME}/Library/LaunchAgents/gpm-agent.plist"
    assert runner.commands == [
        ["launchctl", "print", "gui/501/gpm-agent"],  # before anything is written
        ["launchctl", "print", "gui/501/gpm-agent"],
        ["launchctl", "bootstrap", "gui/501", plist],
    ]
    assert plist in machine.files.content
    assert machine.files.private == {f"{MAC_HOME}/.config/gpm-agent/gpm-agent.log"}


def test_macos_reloads_one_already_loaded_so_new_settings_take_and_waits_for_launchd():
    start = ("launchctl", "bootstrap", "gui/501", MAC_PLIST)
    runner = Runner({MAC_QUERY: loaded_from(MAC_PLIST), start: [Ran(5, "Bootstrap failed: 5: Input/output error"), Ran(0)]})
    assert startup.add(mac(runner), MAC_SETTINGS) == 0
    assert runner.commands == [
        list(MAC_QUERY),  # before anything is written: whose job holds the label
        list(MAC_QUERY),
        ["launchctl", "bootout", "gui/501/gpm-agent"],
        list(start), list(start),
    ]


def test_macos_never_boots_out_another_programs_job_with_the_same_label(capsys):
    """A label is a name anyone may give a job: only one loaded from this agent's own file is
    booted out. Another's is said, and nothing is written or stopped."""
    for answer in (loaded_from("/Library/LaunchAgents/gpm-agent.plist"), Ran(0, "gui/501/gpm-agent = {\n}")):
        runner = Runner({MAC_QUERY: answer})
        machine = mac(runner)
        assert startup.add(machine, MAC_SETTINGS) == 1
        assert runner.commands == [list(MAC_QUERY)]
        assert machine.files.content == {}
    err = capsys.readouterr().err
    assert "loaded in your session from /Library/LaunchAgents/gpm-agent.plist" in err
    assert "a file launchctl does not name" in err

    # Ours on disk, but the label loaded from elsewhere (launchd refused ours): left running.
    machine = mac(Runner({MAC_QUERY: NO_TASK}))
    startup.add(machine, MAC_SETTINGS)
    machine.run = runner = Runner({MAC_QUERY: loaded_from("/Library/LaunchAgents/gpm-agent.plist")})
    assert startup.remove(machine, MAC_SETTINGS) == 0
    assert ["launchctl", "bootout", "gui/501/gpm-agent"] not in runner.commands
    assert "left running" in capsys.readouterr().out


def test_linux_never_writes_over_a_unit_of_the_same_name_installed_elsewhere(capsys):
    """One in ~/.config/systemd/user would take the place of a program's own in /etc or /usr."""
    runner = Runner({LINUX_INSTALLED: Ran(0, "FragmentPath=/usr/lib/systemd/user/gpm-agent.service")})
    machine = linux(runner)
    assert startup.add(machine, LINUX_SETTINGS) == 1
    assert runner.commands == [list(LINUX_INSTALLED)] and machine.files.content == {}
    assert "installed at /usr/lib/systemd/user/gpm-agent.service" in capsys.readouterr().err

    # A user manager that will not answer: said before anything is written.
    runner = Runner({LINUX_INSTALLED: Ran(1, "Failed to connect to bus: No medium found")})
    machine = linux(runner)
    assert startup.add(machine, LINUX_SETTINGS) == 1
    assert machine.files.content == {} and "Failed to connect to bus" in capsys.readouterr().err

    # Its own, from before: replaced, as `add` does to take changed settings.
    runner = Runner({LINUX_INSTALLED: Ran(0, f"FragmentPath={LINUX_UNIT}"),
                     ("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=yes")})
    machine = linux(runner, Files({LINUX_UNIT: startup.entry_for(linux(), LINUX_SETTINGS).content}))
    assert startup.add(machine, LINUX_SETTINGS) == 0


def test_macos_without_a_login_session_keeps_the_agent_for_the_next_login_and_says_so(capsys):
    start = ("launchctl", "bootstrap", "gui/501", f"{MAC_HOME}/Library/LaunchAgents/gpm-agent.plist")
    runner = Runner({("launchctl", "print", "gui/501/gpm-agent"): Ran(113),
                     start: Ran(125, "Domain does not support specified action")})
    machine = mac(runner)
    assert startup.add(machine, MAC_SETTINGS) == 1
    assert runner.commands.count(list(start)) == 3
    assert f"{MAC_HOME}/Library/LaunchAgents/gpm-agent.plist" in machine.files.content
    assert "next login" in capsys.readouterr().out


def test_linux_enables_and_starts_the_unit_and_prints_the_lingering_command_without_admin(capsys):
    runner = Runner({("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=no")})
    machine = linux(runner)
    assert startup.add(machine, LINUX_SETTINGS) == 0
    assert runner.commands == [
        list(LINUX_INSTALLED),  # before anything is written: no unit by this name elsewhere
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "gpm-agent.service"],
        ["systemctl", "--user", "restart", "gpm-agent.service"],
        ["loginctl", "show-user", "owner", "--property=Linger"],
    ]
    assert f"{LINUX_HOME}/.config/systemd/user/gpm-agent.service" in machine.files.content
    assert "sudo loginctl enable-linger owner" in capsys.readouterr().out


def test_linux_with_admin_at_a_terminal_switches_lingering_on_through_sudo():
    query = ("loginctl", "show-user", "owner", "--property=Linger")
    runner = Runner({query: [Ran(0, "Linger=no"), Ran(0, "Linger=yes")]})
    assert startup.add(linux(runner), LINUX_SETTINGS, admin=True) == 0
    assert (["sudo", "loginctl", "enable-linger", "owner"], True) in runner.calls  # on the terminal
    assert runner.commands[-1] == list(query)


def test_linux_never_runs_sudo_without_admin_or_without_a_terminal(capsys):
    for admin, interactive in ((False, True), (True, False)):
        runner = Runner({("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=no")})
        assert startup.add(linux(runner, interactive=interactive), LINUX_SETTINGS, admin=admin) == 0
        assert not [argv for argv in runner.commands if argv[0] == "sudo"]
    assert "needs a terminal" in capsys.readouterr().out


def test_linux_with_lingering_on_asks_nothing_more():
    runner = Runner({("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=yes")})
    assert startup.add(linux(runner), LINUX_SETTINGS, admin=True) == 0
    assert not [argv for argv in runner.commands if argv[0] == "sudo"]


def test_windows_registers_the_task_from_its_definition_and_starts_it_now(capsys):
    runner = Runner({WIN_DEFINITION: NO_TASK})
    machine = windows(runner)
    assert startup.add(machine, WIN_SETTINGS) == 0
    definition = WIN_HOME + "\\.config\\gpm-agent\\gpm-agent.task.xml"
    assert runner.commands == [
        list(WIN_DEFINITION),
        [SCHTASKS, "/Create", "/TN", "gpm-agent", "/XML", definition, "/F"],
        [SCHTASKS, "/End", "/TN", "gpm-agent"],
        [SCHTASKS, "/Run", "/TN", "gpm-agent"],
    ]
    assert machine.files.content[definition].startswith(b"\xff\xfe")
    assert WIN_HOME + "\\.config\\gpm-agent\\gpm-agent.log" in machine.files.private
    assert "automatic sign-in" in capsys.readouterr().out


def test_add_says_it_started_the_agent_not_that_it_answers_and_warns_of_one_on_its_port(capsys):
    """Started is all that is known when `add` ends. A `gpm-agent serve` started by hand keeps the
    port, and the agent started with the machine would fail there, again and again: said first."""
    machine = mac(Runner({MAC_QUERY: NO_TASK}), answers=lambda host, port: (host, port) == ("127.0.0.1", 8095))
    assert startup.add(machine, MAC_SETTINGS, listen=("127.0.0.1", 8095)) == 0
    out = capsys.readouterr().out
    assert "Something already answers on 127.0.0.1:8095" in out and "you started yourself" in out
    assert "running now" not in out and "`gpm-agent startup status` shows whether it answers" in out
    # Set up already: what answers is the agent started at login, which is started again now.
    machine.run = Runner({MAC_QUERY: loaded_from(MAC_PLIST)})
    assert startup.add(machine, MAC_SETTINGS, listen=("127.0.0.1", 8095)) == 0
    assert "Something already answers" not in capsys.readouterr().out
    assert startup.status(machine, MAC_SETTINGS, listen=("127.0.0.1", 8095)) == 0
    assert "  answers on 127.0.0.1:8095" in capsys.readouterr().out
    machine.answers = lambda host, port: False
    assert startup.status(machine, MAC_SETTINGS, listen=("127.0.0.1", 8095)) == 0
    assert "  nothing answers on 127.0.0.1:8095" in capsys.readouterr().out


def test_the_port_is_looked_at_on_this_machine():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert startup.port_answers("127.0.0.1", port)
        assert startup.port_answers("0.0.0.0", port)  # every address: looked at on loopback
    assert not startup.port_answers("127.0.0.1", port)


def test_restart_starts_the_agent_again_and_leaves_its_definition_as_it_is():
    machine = linux(Runner({("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=yes")}))
    startup.add(machine, LINUX_SETTINGS)
    machine.run = runner = Runner()
    assert startup.restart(machine, LINUX_SETTINGS) == 0
    assert runner.commands == [["systemctl", "--user", "restart", "gpm-agent.service"]]

    machine = windows(Runner({WIN_DEFINITION: NO_TASK}))
    startup.add(machine, WIN_SETTINGS)
    machine.run = runner = Runner()
    assert startup.restart(machine, WIN_SETTINGS) == 0
    assert runner.commands == [[SCHTASKS, "/End", "/TN", "gpm-agent"], [SCHTASKS, "/Run", "/TN", "gpm-agent"]]


def test_two_agents_on_one_machine_get_two_definitions():
    machine = mac(Runner({("launchctl", "print", "gui/501/gpm-agent"): Ran(113),
                          ("launchctl", "print", "gui/501/gpm-agent-whisper"): Ran(113)}))
    whisper = f"{MAC_HOME}/.config/gpm-agent/whisper.json"
    assert startup.add(machine, MAC_SETTINGS) == 0
    assert startup.add(machine, whisper) == 0
    agents = f"{MAC_HOME}/Library/LaunchAgents"
    assert plistlib.loads(machine.files.content[f"{agents}/gpm-agent-whisper.plist"])["ProgramArguments"][5] == whisper
    assert plistlib.loads(machine.files.content[f"{agents}/gpm-agent.plist"])["ProgramArguments"][5] == MAC_SETTINGS


# --- refusals, said before anything is written ---


#: What uv writes at the root of its cache — and of every environment it makes (uv 0.12.5).
TAG = b"Signature: 8a477f597d28d172789f06886806bc55"


def environment_files(prefix, home="/home/owner/.local/share/uv/python/cpython-3.14"):
    """The files uv writes at the root of an environment it makes: `uv tool install`'s, `uv venv`'s
    and uvx's alike carry their own `CACHEDIR.TAG`, beside `pyvenv.cfg`."""
    return {f"{prefix}/CACHEDIR.TAG": TAG, f"{prefix}/pyvenv.cfg": f"home = {home}/bin\n".encode(),
            f"{prefix}/.gitignore": b"*"}


def test_an_agent_run_by_uvx_is_not_set_to_start_from_a_cache_that_can_vanish():
    cache = f"{MAC_HOME}/.cache/uv"
    environment = f"{cache}/archive-v0/0GCSPIExjqRK8SU_"  # where uvx makes its environments
    files = Files({f"{cache}/CACHEDIR.TAG": TAG, **environment_files(environment)})
    runner = Runner()
    machine = mac(runner, files, prefix=environment, executable=f"{environment}/bin/python")
    problem = startup.problem_with_adding(machine, MAC_SETTINGS)
    assert problem and "uv tool install gpm-agent" in problem and f"environment in {cache} " in problem
    assert startup.add(machine, MAC_SETTINGS) == 1
    assert runner.commands == [] and not [path for path in files.content if path.startswith(MAC_HOME + "/Library")]
    # Reached through the link uv keeps under environments-v2: the same cache.
    linked = f"{cache}/environments-v2/33d64f84006932da/5fe5d1e0308cc937"
    machine = mac(files=Files({f"{cache}/CACHEDIR.TAG": TAG, **environment_files(linked)}), prefix=linked)
    assert startup.temporary_environment(machine) == cache


def test_an_agent_installed_with_uv_tool_install_is_set_to_start_although_uv_tags_its_environment():
    """`uv tool install` writes `CACHEDIR.TAG` into the environment it makes, as into every one:
    the tag of the environment's own directory is not a cache's (checked against uv 0.12.5)."""
    tools = f"{MAC_HOME}/.local/share/uv/tools"
    files = Files({**environment_files(f"{tools}/gpm-agent"), f"{tools}/.gitignore": b"*", f"{tools}/.lock": b""})
    machine = mac(Runner({MAC_QUERY: NO_TASK}), files, prefix=f"{tools}/gpm-agent")
    assert startup.temporary_environment(machine) is None
    assert startup.problem_with_adding(machine, MAC_SETTINGS) is None
    assert startup.add(machine, MAC_SETTINGS) == 0
    # A project's own environment (`uv venv`, `uv run`) likewise.
    project = Files(environment_files("/home/owner/src/gpm/.venv"))
    assert startup.temporary_environment(linux(files=project, prefix="/home/owner/src/gpm/.venv")) is None


def test_a_definition_serving_other_settings_is_never_replaced_or_removed():
    plist = f"{MAC_HOME}/Library/LaunchAgents/gpm-agent.plist"
    theirs = startup.entry_for(mac(), "/elsewhere/agent.json").content
    runner = Runner()
    machine = mac(runner, Files({plist: theirs}))
    problem = startup.problem_with_adding(machine, MAC_SETTINGS)
    assert "/elsewhere/agent.json" in problem
    assert startup.add(machine, MAC_SETTINGS) == 1
    assert startup.remove(machine, MAC_SETTINGS) == 1
    assert machine.files.content[plist] == theirs and runner.commands == []

    hand_written = Files({plist: b"<plist>not ours</plist>"})
    assert "not written by this agent" in startup.problem_with_adding(mac(files=hand_written), MAC_SETTINGS)

    unit = f"{LINUX_HOME}/.config/systemd/user/gpm-agent.service"
    assert startup.problem_with_adding(linux(files=Files({unit: b"[Service]\nExecStart=/bin/true\n"})), LINUX_SETTINGS)

    # On Windows: a task by this name whose own definition says nothing this agent writes.
    registered = Runner({WIN_DEFINITION: Ran(0, '<?xml version="1.0" encoding="UTF-16"?>\r\n<Task></Task>')})
    assert "not written by this agent" in startup.problem_with_adding(windows(registered), WIN_SETTINGS)


def test_on_windows_whose_task_it_is_is_asked_of_task_scheduler_not_of_a_file_beside_the_settings(capsys):
    """Another agent's task by the same name — registered from other settings, with the same file
    name — is never deleted or replaced, though a definition of ours lies beside these settings."""
    elsewhere = "D:\\Agents\\gpm-agent\\agent.json"
    theirs = startup.entry_for(windows(), elsewhere)
    ours = startup.entry_for(windows(), WIN_SETTINGS)
    runner = Runner({WIN_DEFINITION: registered_as(theirs)})
    machine = windows(runner, Files({ours.file: ours.content, WIN_SCRIPTS + "\\pythonw.exe": b""}))
    assert startup.remove(machine, WIN_SETTINGS) == 1
    assert elsewhere in capsys.readouterr().err
    assert startup.add(machine, WIN_SETTINGS) == 1
    assert runner.commands == [list(WIN_DEFINITION)] * 2  # asked, and nothing ended, deleted or made

    # Ours, read back in the console's code page — "José" as "Jos‚" — is still known as ours.
    accented = "C:\\Users\\José\\.config\\gpm-agent\\agent.json"
    garbled = startup.entry_for(windows(), accented).content.decode("utf-16").replace("é", "‚")
    runner = Runner({WIN_DEFINITION: Ran(0, garbled)})
    assert startup.remove(windows(runner), accented) == 0
    assert [SCHTASKS, "/Delete", "/TN", "gpm-agent", "/F"] in runner.commands


def test_a_system_without_a_way_is_told_the_command_to_run_instead():
    problem = startup.problem_with_adding(mac(system="FreeBSD"), "/home/owner/agent.json")
    assert "-I -m gpm_agent.cli --config /home/owner/agent.json serve" in problem
    no_systemd = linux(which=lambda name: None if name == "systemctl" else f"/usr/bin/{name}")
    assert "no systemd" in startup.problem_with_adding(no_systemd, LINUX_SETTINGS)
    assert "%" in startup.problem_with_adding(windows(), "C:\\Users\\owner\\100%\\agent.json")
    assert "control character" in startup.problem_with_adding(mac(), "/home/owner/a\nb.json")


# --- taking it away, and asking how it stands ---


def test_remove_stops_the_agent_and_takes_its_definition_away_on_each_system(capsys):
    machine = mac(Runner({("launchctl", "print", "gui/501/gpm-agent"): Ran(113)}))
    startup.add(machine, MAC_SETTINGS)
    machine.run = runner = Runner({MAC_QUERY: loaded_from(MAC_PLIST)})
    assert startup.remove(machine, MAC_SETTINGS) == 0
    assert runner.commands == [list(MAC_QUERY), ["launchctl", "bootout", "gui/501/gpm-agent"]]
    assert MAC_PLIST not in machine.files.content
    # The engines it started outlive it (D148), and the owner is told so, and how to end them.
    out = capsys.readouterr().out
    assert "goes on serving until this machine\nrestarts" in out and "Activity Monitor" in out

    machine = linux(Runner({("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=yes")}))
    startup.add(machine, LINUX_SETTINGS)
    machine.run = runner = Runner()
    assert startup.remove(machine, LINUX_SETTINGS) == 0
    assert runner.commands == [
        ["systemctl", "--user", "disable", "--now", "gpm-agent.service"],
        ["systemctl", "--user", "daemon-reload"],
    ]
    assert not [path for path in machine.files.content if path.endswith(".service")]

    machine = windows(Runner({WIN_DEFINITION: NO_TASK}))
    startup.add(machine, WIN_SETTINGS)
    machine.run = runner = Runner({WIN_DEFINITION: registered_as(startup.entry_for(machine, WIN_SETTINGS))})
    assert startup.remove(machine, WIN_SETTINGS) == 0
    assert runner.commands == [
        list(WIN_DEFINITION),
        [SCHTASKS, "/End", "/TN", "gpm-agent"],
        [SCHTASKS, "/Delete", "/TN", "gpm-agent", "/F"],
    ]
    assert not [path for path in machine.files.content if path.endswith(".task.xml")]
    assert "Task Manager" in capsys.readouterr().out


def test_nothing_set_up_is_nothing_to_remove():
    runner = Runner()
    assert startup.remove(mac(runner), MAC_SETTINGS) == 0
    assert runner.commands == []


def test_status_says_whether_it_is_set_up_and_running(capsys):
    machine = mac(Runner({("launchctl", "print", "gui/501/gpm-agent"): Ran(113)}))
    assert startup.status(machine, MAC_SETTINGS) == 1
    startup.add(machine, MAC_SETTINGS)
    machine.run = Runner({("launchctl", "print", "gui/501/gpm-agent"): Ran(0, "gui/501/gpm-agent = {\n\tstate = running\n\tpid = 4242\n}")})
    assert startup.status(machine, MAC_SETTINGS) == 0
    out = capsys.readouterr().out
    assert "running, process 4242" in out

    machine = linux(Runner({("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=no")}))
    startup.add(machine, LINUX_SETTINGS)
    machine.run = Runner({
        ("systemctl", "--user", "is-enabled", "gpm-agent.service"): Ran(0, "enabled"),
        ("systemctl", "--user", "is-active", "gpm-agent.service"): Ran(0, "active"),
        ("loginctl", "show-user", "owner", "--property=Linger"): Ran(0, "Linger=no"),
    })
    assert startup.status(machine, LINUX_SETTINGS) == 0
    out = capsys.readouterr().out
    assert "enabled; active" in out and "off (starts at login)" in out


# --- the command line ---


@pytest.fixture
def settings_file(tmp_path):
    return save(Settings(key_hash=fingerprint("gpmg_notreal")), tmp_path / "agent.json")


def test_startup_verbs_reach_this_module_with_the_absolute_settings_path(monkeypatch, settings_file):
    seen = []
    machine = mac()
    monkeypatch.setattr(startup.Machine, "this", classmethod(lambda cls: machine))
    monkeypatch.setattr(startup, "add", lambda m, path, admin, listen: seen.append(("add", path, admin, listen)) or 0)
    monkeypatch.setattr(startup, "remove", lambda m, path: seen.append(("remove", path)) or 0)
    monkeypatch.setattr(startup, "status", lambda m, path, listen=None: seen.append(("status", path, listen)) or 0)
    for action in ("add", "remove", "status"):
        assert cli.main(["-c", str(settings_file), "startup", action]) == 0
    assert cli.main(["-c", str(settings_file), "startup", "add", "--admin"]) == 0
    path, listen = str(settings_file), ("127.0.0.1", 8095)  # where its settings have it listen
    assert seen == [("add", path, False, listen), ("remove", path), ("status", path, listen), ("add", path, True, listen)]


def test_init_with_start_at_login_writes_the_settings_then_sets_the_agent_to_start(monkeypatch, tmp_path, capsys):
    machine = mac(Runner({("launchctl", "print", "gui/501/gpm-agent"): Ran(113)}), home=str(tmp_path))
    monkeypatch.setattr(startup.Machine, "this", classmethod(lambda cls: machine))
    path = tmp_path / "agent.json"
    assert cli.main(["-c", str(path), "init", "--start-at-login"]) == 0
    assert path.exists()
    assert plistlib.loads(machine.files.content[f"{tmp_path}/Library/LaunchAgents/gpm-agent.plist"])["ProgramArguments"][5] == str(path)
    assert "no need to run `gpm-agent serve`" in capsys.readouterr().out


def test_a_key_rotated_under_an_agent_that_starts_with_the_machine_is_taken_up_at_once(monkeypatch, tmp_path, capsys):
    """The agent reads its key when it starts: rotated under one started at login, that one is
    started again — never left on the old key with `gpm-agent serve` suggested beside it."""
    machine = mac(Runner({MAC_QUERY: NO_TASK}), home=str(tmp_path))
    monkeypatch.setattr(startup.Machine, "this", classmethod(lambda cls: machine))
    path = tmp_path / "agent.json"
    assert cli.main(["-c", str(path), "init", "--start-at-login"]) == 0
    plist = f"{tmp_path}/Library/LaunchAgents/gpm-agent.plist"
    written, old = machine.files.content[plist], json.loads(path.read_text())["key_hash"]
    capsys.readouterr()
    machine.run = runner = Runner({MAC_QUERY: loaded_from(plist)})
    assert cli.main(["-c", str(path), "init", "--rotate"]) == 0
    assert json.loads(path.read_text())["key_hash"] != old
    assert runner.commands == [
        list(MAC_QUERY), ["launchctl", "bootout", "gui/501/gpm-agent"], ["launchctl", "bootstrap", "gui/501", plist],
    ]
    assert machine.files.content[plist] == written  # what it runs is left as it is
    out = capsys.readouterr().out
    assert "again, with these settings" in out and "Then" not in out and "gpm-agent serve" not in out

    # None set up: the owner is told to start theirs again, or to set one up.
    other = tmp_path / "other" / "agent.json"
    assert cli.main(["-c", str(other), "init"]) == 0
    assert "Then: gpm-agent serve — or `gpm-agent startup add`" in capsys.readouterr().out
    machine.run = runner = Runner()
    assert cli.main(["-c", str(other), "init", "--rotate"]) == 0
    out = capsys.readouterr().out
    assert runner.commands == [] and "start `gpm-agent serve` again" in out and "`gpm-agent startup add`" in out


def test_init_that_cannot_set_up_a_start_at_login_mints_nothing(monkeypatch, tmp_path):
    files = Files({f"{MAC_HOME}/.cache/uv/CACHEDIR.TAG": b""})
    machine = mac(files=files, prefix=f"{MAC_HOME}/.cache/uv/archive-v0/Xyz")
    monkeypatch.setattr(startup.Machine, "this", classmethod(lambda cls: machine))
    path = tmp_path / "agent.json"
    assert cli.main(["-c", str(path), "init", "--start-at-login"]) == 1
    assert not path.exists()


def test_serve_can_write_its_output_to_a_file_of_its_own(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    log = tmp_path / "agent.log"
    startup.send_output_to(str(log))
    print("to the log")
    print("and its errors", file=sys.stderr)
    sys.stdout.flush()
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert log.read_text() == "to the log\nand its errors\n"


# --- the real file system and the real runner ---


def test_definitions_are_replaced_whole_and_logs_are_created_owner_only(tmp_path):
    files = startup.Files()
    definition = str(tmp_path / "LaunchAgents" / "gpm-agent.plist")
    files.write(definition, b"one")
    files.write(definition, b"two")
    assert files.read(definition) == b"two" and stat.S_IMODE((tmp_path / "LaunchAgents/gpm-agent.plist").stat().st_mode) == 0o644
    assert [p.name for p in (tmp_path / "LaunchAgents").iterdir()] == ["gpm-agent.plist"]
    log = str(tmp_path / "logs" / "gpm-agent.log")
    files.create_private(log)
    pathlib.Path(log).write_text("kept")
    files.create_private(log)
    assert pathlib.Path(log).read_text() == "kept" and stat.S_IMODE(pathlib.Path(log).stat().st_mode) == 0o600
    files.delete(definition)
    files.delete(definition)
    assert not files.exists(definition) and files.read(definition) is None


def test_files_are_written_in_binary_mode_so_windows_keeps_every_byte(tmp_path, monkeypatch):
    """Windows opens a descriptor in text mode unless told `O_BINARY`, and turns each 0x0A
    written into 0x0D 0x0A: inside a UTF-16 task definition, half of a character. Here a stand-in
    for Windows' flag is required of every file this setup writes."""
    binary = 0x8000  # Windows' O_BINARY
    monkeypatch.setattr(os, "O_BINARY", binary, raising=False)
    real_open, flags_seen = os.open, []

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        flags_seen.append((pathlib.Path(path).name, flags))
        return real_open(path, flags & ~binary, mode, *args, **kwargs)  # the flag means another thing here

    monkeypatch.setattr(os, "open", recording_open)
    content = startup.entry_for(windows(), WIN_SETTINGS).content
    definition = tmp_path / "gpm-agent.task.xml"
    startup.Files().write(str(definition), content)
    save(Settings(key_hash=fingerprint("gpmg_notreal")), tmp_path / "agent.json")
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    startup.send_output_to(str(tmp_path / "gpm-agent.log"))
    sys.stdout.close()
    assert [name for name, _ in flags_seen] == ["gpm-agent.task.xml.new", "agent.json", "gpm-agent.log"]
    assert all(flags & binary for _, flags in flags_seen)
    assert definition.read_bytes() == content and b"\r\x00\n\x00" in content


def test_the_runner_runs_an_argument_list_and_reports_a_missing_program():
    ran = startup.run_command([sys.executable, "-c", "print('answered')"])
    assert ran.ok and ran.output == "answered"
    assert startup.run_command([sys.executable, "-c", "raise SystemExit(3)"]).returncode == 3
    assert startup.run_command(["gpm-no-such-program-anywhere"]).returncode == 127


def test_only_the_owners_command_line_reaches_this_module_and_it_starts_processes_in_one_place():
    """Structural (D40, D154): one `subprocess.run`, of an argument list, never a shell; the
    prerequisites go through it; and nothing that answers the pool imports either module."""
    root = pathlib.Path(gpm_agent.__file__).parent
    tree = ast.parse((root / "startup.py").read_text())
    starts = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
        and node.func.attr in ("run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput")
    ]
    assert len(starts) == 1 and starts[0].func.attr == "run"
    (argument,) = starts[0].args
    assert isinstance(argument, ast.Call) and argument.func.id == "list" and argument.args[0].id == "argv"
    assert not any(keyword.arg == "shell" for keyword in starts[0].keywords)
    assert "subprocess" not in (root / "prerequisites.py").read_text()
    for path in root.glob("*.py"):
        if path.name in ("cli.py", "startup.py", "prerequisites.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                named = {node.module or ""} | {alias.name for alias in node.names}
                assert not named & {"startup", "prerequisites"}, f"{path.name} reaches the owner's setup"


def test_windows_paths_are_built_with_windows_rules_whatever_runs_the_test():
    assert startup.schtasks(windows(environ={"SystemRoot": "D:\\Win"})) == "D:\\Win\\System32\\schtasks.exe"
    assert startup.schtasks(windows(environ={})) == SCHTASKS
    assert ntpath.dirname(startup.entry_for(windows(), WIN_SETTINGS).file) == WIN_HOME + "\\.config\\gpm-agent"
