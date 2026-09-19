"""The `pool` command, driven through `main()` — every verb, not only the two that serve.

A verb that crashes before making its call is a verb nobody can use, and unit tests of the
functions behind it would never notice.
"""

import json

import httpx
import pytest
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_server.cli import main
from gpm_server.config import PoolConfig
from gpm_server.db import Database
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app

ADMIN_KEY = "gpmx_cli_admin"
APP_KEY = "gpma_cli_app"


@pytest.fixture
def control(tmp_path, monkeypatch):
    config = PoolConfig.model_validate(
        {
            "pool": {"name": "cli", "model_set": ["m1"], "probe_interval_s": 3600},
            "auth": {"app_keys": [APP_KEY], "admin_keys": [ADMIN_KEY]},
            "hosts": [],
            "rented": {
                "provider": "fake",
                "bidding": {"bid_ceiling": 0.60},
                "scale": {"scale_up_after_s": 0},
            },
        }
    )
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(config, database)
    server = ServerHandle(create_control_app(supervisor, config), loop)
    monkeypatch.setenv("GPM_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("GPM_CONTROL_URL", server.base_url)
    try:
        yield supervisor
    finally:
        server.stop()
        loop.stop()
        database.close()


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_every_read_only_verb_answers(control, capsys):
    for verb in (["account"], ["market"], ["plan"], ["events"], ["lease", "list"]):
        code, out, err = run(capsys, *verb)
        assert code == 0, f"{verb}: {err}"
        json.loads(out)  # the answer is the API's JSON, printed


def test_a_lease_can_be_opened_and_closed_from_the_command_line(control, capsys):
    code, out, _ = run(capsys, "lease", "open", "--workers", "2", "--max-spend", "1.5", "--allow-rent")
    assert code == 0
    lease_id = json.loads(out)["lease_id"]
    assert json.loads(out)["worst_case"]["dollars"] == 1.5

    code, out, _ = run(capsys, "lease", "close", lease_id)
    assert code == 0
    assert control.leases.get(lease_id).state == "closed"


def test_a_lease_that_can_rent_needs_its_dollars_stated(control, capsys):
    code, out, err = run(capsys, "lease", "open", "--workers", "2", "--allow-rent")
    assert code == 1
    assert "dollar cap" in err


def test_the_app_key_is_refused_by_every_control_verb(control, capsys, monkeypatch):
    monkeypatch.setenv("GPM_ADMIN_KEY", APP_KEY)
    code, out, err = run(capsys, "account")
    assert code == 1
    assert "app key" in err


def test_a_missing_admin_key_is_said_plainly(control, capsys, monkeypatch):
    monkeypatch.delenv("GPM_ADMIN_KEY")
    code, out, err = run(capsys, "market")
    assert code == 2
    assert "GPM_ADMIN_KEY" in err


def test_an_unreachable_control_api_is_an_error_not_a_traceback(capsys, monkeypatch):
    monkeypatch.setenv("GPM_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("GPM_CONTROL_URL", "http://127.0.0.1:9")
    code, out, err = run(capsys, "plan")
    assert code == 1
    assert "could not reach" in err


def test_down_all_requires_saying_all(capsys):
    with pytest.raises(SystemExit):
        main(["down"])
