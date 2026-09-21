"""A deployed pool runs a tag, and says which one (D79).

Three times in one week a fix was on disk and the running supervisor was not running it, and
each time the only thing that noticed was a rented host failing. The cause under all three was
that the live pool ran from the development tree: whatever was being edited was also what was
deployed. These are the two halves of the answer — code that says what it is, and a release
built from a tag that the working tree cannot reach into.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from gpm_server.version import Running, running

REPO = Path(__file__).resolve().parent.parent
DEPLOY = REPO / "deploy" / "gpm-deploy"


def load_deploy_tool():
    """It is a script, not a package: it must not need installing to be run or read."""
    spec = importlib.util.spec_from_loader(
        "gpm_deploy_tool", importlib.machinery.SourceFileLoader("gpm_deploy_tool", str(DEPLOY))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- what this process is running ---


def test_a_development_tree_says_so_rather_than_looking_like_a_release():
    """The live pool ran this way all week and nothing ever said it was unusual."""
    tree = Running(version="0.6.0", tag=None, commit=None, editable=True, location="/w/GPM/server/src")

    assert not tree.is_release
    said = tree.describe()
    assert "DEVELOPMENT TREE" in said and "gpm-deploy" in said, said


def test_a_release_names_its_tag():
    release = Running(version="0.6.1", tag="v0.6.1", commit="abc123", editable=False, location="/r/v0.6.1")

    assert release.is_release
    assert "release v0.6.1" in release.describe()


def test_an_install_from_no_tag_is_not_called_a_release():
    """A plain `pip install .` is not a deploy: it can be anything, and nothing can say what."""
    loose = Running(version="0.6.1", tag=None, commit=None, editable=False, location="/somewhere")

    assert not loose.is_release
    assert "not deployed from a tag" in loose.describe()


def test_this_working_tree_reports_itself_as_editable():
    """Guards the detection itself: if this ever comes back False here, it is broken."""
    assert running().editable is True


# --- building one ---


def test_only_a_tag_can_be_deployed(tmp_path):
    """A branch moves. A release that moves is the problem this exists to remove."""
    tool = load_deploy_tool()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "a.txt").write_text("one\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "one"], check=True)

    with pytest.raises(SystemExit, match="no tags yet"):
        tool.resolve_tag(repo, "v1.0.0")

    subprocess.run(["git", "-C", str(repo), "tag", "v1.0.0"], check=True)
    assert tool.resolve_tag(repo, "v1.0.0")

    branch = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"], check=True, text=True, capture_output=True
    ).stdout.strip()
    with pytest.raises(SystemExit, match="is not a tag"):
        tool.resolve_tag(repo, branch)


def test_a_release_holds_the_tag_and_not_the_working_tree(tmp_path):
    """The fault in one line: what is deployed must not be what is being edited."""
    tool = load_deploy_tool()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "code.py").write_text("VERSION = 'tagged'\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "one"], check=True)
    subprocess.run(["git", "-C", str(repo), "tag", "v1.0.0"], check=True)

    # Edited after the tag, and never committed — exactly the state a pool was run from.
    (repo / "code.py").write_text("VERSION = 'edited, uncommitted'\n")
    (repo / "extra.py").write_text("# not in the tag at all\n")

    into = tmp_path / "release"
    into.mkdir()
    tool.extract(repo, "v1.0.0", into)

    assert (into / "code.py").read_text() == "VERSION = 'tagged'\n"
    assert not (into / "extra.py").exists(), "the working tree reached into the release"


def test_activating_is_one_swap_and_the_old_release_is_still_there(tmp_path):
    tool = load_deploy_tool()
    first, second = tmp_path / "v1", tmp_path / "v2"
    first.mkdir()
    second.mkdir()

    tool.activate(tmp_path, first)
    assert (tmp_path / "current").resolve() == first.resolve()

    tool.activate(tmp_path, second)
    assert (tmp_path / "current").resolve() == second.resolve()
    assert first.exists(), "rolling back means activating what is still on disk"

    tool.activate(tmp_path, first)  # and back again
    assert (tmp_path / "current").resolve() == first.resolve()


def test_a_release_is_built_with_the_interpreter_it_declares_not_the_one_running(tmp_path):
    """Found while building this: run by the system python3.9, it tried to build a release that
    needs 3.11 and failed. The floor is the release's to state."""
    tool = load_deploy_tool()
    release = tmp_path / "r"
    (release / "server").mkdir(parents=True)
    (release / "server" / "pyproject.toml").write_text(
        '[project]\nname = "gpm-server"\nversion = "9.9.9"\nrequires-python = ">=3.11"\n'
    )

    assert tool.requires_python(release) == ">=3.11"
    assert tool.declared_version(release) == "9.9.9"


def test_a_half_built_release_is_never_left_where_it_could_be_activated(tmp_path, monkeypatch):
    tool = load_deploy_tool()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "server").mkdir()
    (repo / "server" / "pyproject.toml").write_text(
        '[project]\nname = "gpm-server"\nversion = "9.9.9"\nrequires-python = ">=3.11"\n'
    )
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "one"], check=True)
    subprocess.run(["git", "-C", str(repo), "tag", "v9.9.9"], check=True)

    def fails(release, python):
        raise SystemExit("the install failed")

    monkeypatch.setattr(tool, "build", fails)
    root = tmp_path / "releases"
    with pytest.raises(SystemExit, match="nothing was changed"):
        tool.main(["v9.9.9", "--repo", str(repo), "--root", str(root)])

    assert not (root / "v9.9.9").exists()
    assert not (root / "current").exists()


def test_the_release_file_is_what_the_running_code_reads(tmp_path):
    """The two halves have to agree on one file, or a release cannot name its own tag."""
    from gpm_server import version as version_module

    # The tool writes this name; the running code looks for that same name.
    assert 'RELEASE.json' in DEPLOY.read_text(), "the tool no longer writes the file"
    (tmp_path / version_module.RELEASE_FILE).write_text(
        json.dumps({"tag": "v1.2.3", "commit": "abc", "version": "1.2.3"})
    )

    assert version_module._release_beside(tmp_path)["tag"] == "v1.2.3"
    # And it is found from a venv nested inside the release, which is where it is read from.
    nested = tmp_path / "venv" / "bin"
    nested.mkdir(parents=True)
    assert version_module._release_beside(nested)["tag"] == "v1.2.3"
