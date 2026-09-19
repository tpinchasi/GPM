import stat

import pytest
from gpm_server.db import Database, RequestLog, RequestRecord


@pytest.fixture
def database(tmp_path):
    db = Database(tmp_path / "gpm.sqlite3")
    try:
        yield db
    finally:
        db.close()


async def test_a_request_is_recorded_with_what_the_console_and_a_join_need(database):
    log = RequestLog(database)
    await log.record(
        RequestRecord(
            request_id="r1",
            outcome="ok",
            session_id="s1",
            host_id="local-1",
            worker_id="local-1/w0",
            model_requested="m1",
            model_served="m1-mlx",
            runtime_class="apple-mlx",
            queue_wait_ms=12.0,
            latency_ms=340.0,
            status_code=200,
        )
    )
    rows = log.rows()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["session_id"] == "s1"
    assert row["model_requested"] == "m1"
    assert row["model_served"] == "m1-mlx"
    assert row["runtime_class"] == "apple-mlx"
    assert row["outcome"] == "ok"


async def test_the_log_has_nowhere_to_put_prompt_or_completion_text(database):
    log = RequestLog(database)
    await log.record(RequestRecord(request_id="r1", outcome="ok"))
    columns = set(dict(log.rows()[0]))
    assert not columns & {"prompt", "completion", "messages", "response", "content"}


def test_wal_mode_is_on(database):
    assert database.query("PRAGMA journal_mode")[0][0].lower() == "wal"


def test_the_database_is_not_readable_by_anyone_else(tmp_path):
    """Threat model T19: another local user must not be able to read pool state."""
    path = tmp_path / "gpm.sqlite3"
    db = Database(path)
    try:
        mode = path.stat().st_mode
        assert not mode & (stat.S_IRGRP | stat.S_IROTH)
    finally:
        db.close()


def test_a_world_readable_database_is_tightened(tmp_path):
    path = tmp_path / "gpm.sqlite3"
    path.touch(mode=0o644)
    db = Database(path)
    try:
        assert not path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH)
    finally:
        db.close()
