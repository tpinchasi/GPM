"""The pool puts its own agent on the hosts it rents (D63, D72).

The pool created the host, so the pool configures it: the packed agent goes over the SSH
connection the dead-man timer already needs, the agent mints a key for that host alone, and the
pool reaches it through a second forward.

The rule that shapes every test here: **no agent is never fatal.** A host that cannot take one
is prepared the way it always was and joins the pool.
"""

import zipfile

import pytest
from fakes.harness import BackgroundLoop
from gpm_server import agentpkg
from gpm_server.supervisor import hostagent

MINTED = "gpmg_" + "ab12" * 16


class FakeHost:
    """One rented host as the installer sees it: a shell to run in, and a file to write."""

    def __init__(self, *, python=True, minted=MINTED, start_ok=True):
        self.python = python
        self.minted = minted
        self.start_ok = start_ok
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}

    async def run(self, command: str):
        self.commands.append(command)
        if "command -v python3" in command:
            return (0, "/usr/bin/python3\n") if self.python else (0, "")
        if " init " in command:
            return 0, f"Agent key (shown once):\n\n  {self.minted}\n"
        if "serve" in command:
            return (0, "started\n") if self.start_ok else (1, "python3: cannot execute")
        return 0, ""

    async def push(self, data: bytes, path: str):
        self.files[path] = data
        return 0, ""


def install(host, archive):
    loop = BackgroundLoop()
    try:
        return loop.run(
            hostagent.install(run=host.run, push=host.push, archive=archive, engine_port=11434)
        )
    finally:
        loop.stop()


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "gpm-agent.pyz"
    path.write_bytes(b"not really a zipapp, but the installer only copies it")
    return path


# --- packing it ---


def test_the_agent_packs_into_one_file_that_runs(tmp_path):
    """Built from what is installed beside the supervisor: no network, no package index."""
    built = agentpkg.build(tmp_path / "gpm-agent.pyz")

    assert built.exists() and built.stat().st_size > 100_000
    inside = {name.split("/")[0] for name in zipfile.ZipFile(built).namelist()}
    assert "gpm_agent" in inside
    assert {"httpx", "starlette", "uvicorn"} <= inside, "its dependencies travel with it"
    assert agentpkg.digest(built) == agentpkg.digest(built)  # the same file names itself the same


def test_a_pool_that_cannot_pack_the_agent_says_so_instead_of_failing(tmp_path):
    assert agentpkg.cached(tmp_path, root="a-distribution-that-is-not-installed") is None


# --- putting it on a host ---


def test_the_agent_is_copied_started_and_mints_its_own_key(archive):
    host = FakeHost()

    key = install(host, archive)

    assert key == MINTED
    assert host.files[hostagent.ARCHIVE] == archive.read_bytes()
    ordered = " | ".join(host.commands)
    assert ordered.index("command -v python3") < ordered.index(" init ")
    assert ordered.index(" init ") < ordered.index("serve")
    assert "--host 127.0.0.1" in ordered, "the agent listens on loopback only"


def test_a_host_without_an_interpreter_gets_no_agent(archive):
    """The rule that keeps this off the list of ways to lose money."""
    host = FakeHost(python=False)

    with pytest.raises(hostagent.AgentInstallFailed, match="no python3"):
        install(host, archive)

    assert hostagent.ARCHIVE not in host.files, "nothing is copied to a host that cannot run it"


def test_an_agent_that_will_not_start_is_not_pretended_to_be_there(archive):
    with pytest.raises(hostagent.AgentInstallFailed, match="did not start"):
        install(FakeHost(start_ok=False), archive)


def test_a_key_the_pool_cannot_read_is_a_failed_install(archive):
    with pytest.raises(hostagent.AgentInstallFailed, match="did not mint a key"):
        install(FakeHost(minted="(nothing key-shaped here)"), archive)


# --- how the pool then dials it ---


def test_the_installed_agent_is_not_configuration(archive):
    """A key minted minutes ago on a machine the pool created lives in memory, never in the
    operator's file — where a key may only ever be named as an environment variable."""
    agent = hostagent.RentedAgent(url="http://127.0.0.1:1/", secret=MINTED)

    assert agent.key() == MINTED
    assert not hasattr(agent, "model_dump"), "it is deliberately not an AgentConfig"


def test_the_file_copy_carries_no_shell(archive):
    command = hostagent.push_command("/var/run/gpm/x; rm -rf /")
    assert "'/var/run/gpm/x; rm -rf /'" in command  # quoted whole, never split into shell


# --- changing the worker count while the host runs (D56, stage 5) ---


class StubAgent:
    """An agent that answers a restart the way a real one does."""

    def __init__(self, status=200, engine_answers=True):
        self.status = status
        self.engine_answers = engine_answers
        self.asked: list[dict] = []
        self.manage_models = True

    def key(self):
        return MINTED

    async def restart(self, settings):
        self.asked.append(settings)
        return self.status, {"engine_answers": self.engine_answers, "settings": settings}
