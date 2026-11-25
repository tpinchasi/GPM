"""A plan read back before it spends (S19 stage 6, D203; workloads.md §14.6).

The plan check answers with the plan in words, a timeline in the plan's time zone, the conditions
that can change it, the most it can spend, a warning per request option that reaches different
sources, and the hosted APIs serving its models. The plan's JSON Schema is served and shipped in
both packages; the CLI takes `--serve`, `--json` and `--serve-example`; the SDK gives each
budget; a workload's view carries the Now line. Nothing here spends money.
"""

import datetime
import io
import json
from pathlib import Path

import gpm_client
import httpx
import pytest
from gpm_server.serving_plan import (
    _TOP,
    EXAMPLES,
    LATER,
    SOURCE_TYPES,
    THEN,
    WHEN,
    conditions,
    json_schema,
    most_hosted,
    option_warnings,
    parse_serve,
    readback,
    timeline,
)
from test_serving_plan_routing import ADMIN, BIG, SPEC, grant, programs, provider, provisioner  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


# --- the plan, in words ---


def test_the_readback_names_every_role_and_every_empty_one():
    said = readback(parse_serve({"when_full": [{"hosted": {"budget": {"per_day": 5}}}]}), kind="on_demand")
    assert said[0] == "Steady: machines rented for it (on-demand)."
    assert said[1] == "Until ready: none — while its machines start, its requests wait for them."
    assert said[2].startswith("When full — every worker of its own busy: hosted: hosted APIs after 5 s, within $5 a day")
    assert said[-1] == "Rules: none."
    owned = readback(parse_serve(EXAMPLES["operators-machines"]))
    assert "nothing is rented" in owned[0] and owned[1].startswith("Until ready: not needed")
    assert "keep_on owned (the plan's default)" in owned[3]
    careful = readback(parse_serve(EXAMPLES["careful-spending"]))
    assert ("Rule 'careful spending': when hosted-burst has spent 80% of its month's budget — never use hosted-burst."
            in careful)


def test_each_request_option_that_reaches_other_sources_is_warned_of():
    plan = parse_serve({"until_ready": ["shared", {"hosted": {"budget": {"per_day": 10}}}],
                        "defaults": {"keep_on": "no_hosted"}})
    warned = option_warnings(plan)
    assert warned[0] == ("requests with no options keep off hosted APIs here: while the rented machines start they use "
                         "shared, then wait; when full they wait for a worker (the plan's defaults: keep_on no_hosted)")
    assert warned[1] == "requests with keep_on any lift the plan's default and may use hosted"
    assert not any(w.startswith("requests with priority batch") for w in warned), "kept off no more than by default"
    # As the router decides: a stop-gap of the operator's machines can still reach them during a
    # replacement, so they wait; with none, nothing ever can, and they are refused at once.
    assert option_warnings(parse_serve({"until_ready": ["shared"]})) == [
        "requests with keep_on owned keep off hosted APIs and rented machines here: while the rented machines start "
        "they use shared (its owned machines only), then wait; once it serves on its rented machines they are served "
        "only while a stop-gap covers (a replacement); otherwise they wait, until the queue's timeout"]
    assert option_warnings(parse_serve({"until_ready": [{"hosted": {"budget": {"per_day": 1}}}]}))[-1] == (
        "requests with keep_on owned keep off hosted APIs and rented machines here: no source of this plan is left to "
        "them, and they are refused at once (no_allowed_source)")
    ruled = option_warnings(parse_serve(EXAMPLES["careful-spending"]))
    assert ruled[-1] == "requests where a request's priority is interactive: rule 'interactive at once' — use hosted-burst after 0 s"


def test_conditions_say_what_changes_the_plan_and_rules_add_theirs():
    said = conditions(parse_serve(EXAMPLES["careful-spending"]))
    assert said[0] == {"condition": "a machine of its own is evicted",
                       "effect": "until_ready covers while it is replaced: shared, then hosted-cover", "rule": None}
    assert {"condition": "every worker of its own is busy", "effect": "when_full: hosted-burst", "rule": None} in said
    assert said[-1]["rule"] == "careful spending"


def test_the_most_it_can_spend_counts_every_day_and_month_the_hours_touch():
    plan = parse_serve({"until_ready": [{"name": "cover", "hosted": {"budget": {"per_day": 10, "per_month": 15}}}]})
    late = datetime.datetime(2026, 1, 31, 23, 0, tzinfo=datetime.timezone.utc).timestamp()
    # Two hours from 23:00 on the 31st: two days and two months — $20 by the day, $30 by the month.
    assert most_hosted(plan, late, 2) == [{"budget": "cover", "name": "cover", "most": 20.0}]
    assert most_hosted(plan, late - 12 * 3600, 2) == [{"budget": "cover", "name": "cover", "most": 10.0}]


def test_the_timeline_is_in_clock_time_and_splits_where_a_window_opens_or_closes():
    plan = parse_serve({"until_ready": ["shared"], "tz": "America/New_York", "rules": [
        {"name": "nights", "when": {"between": "22:00-06:00"}, "then": {"never_use": ["shared"]}},
        {"name": "evicted", "when": {"evictions": {"count": 2, "within_h": 1}}, "then": {"never_use": ["shared"]}}]})
    start = datetime.datetime(2026, 3, 3, 20, 0, tzinfo=datetime.timezone.utc).timestamp()  # 15:00 in New York
    phases = timeline(plan, start=start, hours=24, ready_min={"median": 10, "p90": 20, "measured": False},
                      hosts=2, workers=8, hourly=1.5)
    assert [(p["phase"], p["from"], p["to"], p["rules"]) for p in phases] == [
        ("preparing", "Tue 15:00", "Tue 15:10", []),
        ("serving", "Tue 15:10", "Tue 22:00", []),
        ("serving", "Tue 22:00", "Wed 06:00", ["nights"]),
        ("serving", "Wed 06:00", "Wed 15:00", []),
    ], "a condition that is not a time (evictions) is not on the timeline"
    assert phases[0]["said"] == "until ready: shared" and phases[1]["workers"] == 8
    short = timeline(parse_serve({"until_ready": ["shared"], "max_min": 5}), start=start, hours=1,
                     ready_min={"median": 10, "p90": 20, "measured": True}, hosts=1, workers=4, hourly=1.0)
    assert [(p["from"], p["to"], p["said"]) for p in short[:2]] == [
        ("Tue 20:00", "Tue 20:05", "until ready: shared"),
        ("Tue 20:05", "Tue 20:10", "the stop-gap has ended: its requests wait for its machines")]
    # A window opening at a time DST skips opens at the first minute after it, as the rules read it.
    spring = parse_serve({"until_ready": ["shared"], "tz": "America/New_York", "rules": [
        {"name": "early", "when": {"between": "02:30-05:00"}, "then": {"never_use": ["shared"]}}]})
    night = datetime.datetime(2026, 3, 8, 5, 0, tzinfo=datetime.timezone.utc).timestamp()
    opened = [p for p in timeline(spring, start=night, hours=6, ready_min=None, hosts=1, workers=4, hourly=1.0)
              if p["rules"]][0]
    assert opened["from_ts"] == datetime.datetime(2026, 3, 8, 7, 0, tzinfo=datetime.timezone.utc).timestamp()
    owned = timeline(parse_serve(EXAMPLES["operators-machines"]), start=start, hours=2, ready_min=None, hosts=0,
                     workers=4, hourly=0.0)
    assert [p["phase"] for p in owned] == ["serving"] and owned[0]["hourly_usd"] == 0.0


def test_every_starting_plan_is_one_the_check_accepts():
    for name, serve in EXAMPLES.items():
        assert parse_serve(serve).as_dict(), name


def valid(schema, value):
    """The JSON Schema keywords the plan's schema uses, checked — enough to hold a plan to it."""
    kinds = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "null": type(None)}
    if "anyOf" in schema and not any(valid(s, value) for s in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(valid(s, value) for s in schema["oneOf"]) != 1:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(isinstance(value, kinds[t]) and not (t in ("integer", "number") and isinstance(value, bool))
                   for t in types):
            return False
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        for k, v in value.items():
            if k in props:
                if not valid(props[k], v):
                    return False
            elif schema.get("additionalProperties") is False:
                return False
            elif isinstance(schema.get("additionalProperties"), dict) and not valid(schema["additionalProperties"], v):
                return False
        if len(value) < schema.get("minProperties", 0) or len(value) > schema.get("maxProperties", 10 ** 9):
            return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 10 ** 9):
            return False
        if "items" in schema and not all(valid(schema["items"], v) for v in value):
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            return False
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False
    return True


def test_every_starting_plan_and_bare_word_passes_the_schema():
    schema = json_schema()
    for name, serve in EXAMPLES.items():
        assert valid(schema, serve), name
    assert valid(schema, {"until_ready": ["shared", "local"], "when_full": ["hosted-cover"]})
    assert not valid(schema, {"until_ready": [{"rented": {"max_hosts": 1, "budget": {"per_day": 1}}}]})
    assert not valid(schema, {"steady": "hosted"}) and not valid(schema, {"surprise": 1})


# --- the schema ---


def test_the_schema_covers_every_field_and_both_packages_ship_the_same():
    schema = json_schema()
    assert set(schema["properties"]) == set(_TOP) - set(LATER)
    rules = schema["properties"]["rules"]["items"]["properties"]
    assert set(rules["when"]["properties"]) == set(WHEN) and set(rules["then"]["properties"]) == set(THEN)
    named = {t for item in schema["properties"]["until_ready"]["items"]["anyOf"][2:] for t in item["required"]}
    assert named | {t for item in schema["properties"]["when_full"]["items"]["anyOf"][2:] for t in item["required"]} \
        == set(SOURCE_TYPES)
    shipped = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    for path in ("server/src/gpm_server/schemas/serving-plan.schema.json",
                 "client/src/gpm_client/serving-plan.schema.json"):
        assert (ROOT / path).read_text() == shipped, f"{path} is out of date: regenerate it from json_schema()"
    assert gpm_client.serving_plan_schema() == json.loads(shipped)


# --- the plan check, the CLI, the SDK and the Now line ---


def check(pool, **body):
    return httpx.post(f"{pool.control_url}/pool/workloads/plan", headers={"Authorization": f"Bearer {ADMIN}"},
                      json={"name": "night", "model": BIG, "hours": 2, "latency_s": 30, "parallel": 2, "max_spend": 5.0,
                            **body}, timeout=60)


def test_the_plan_check_reads_a_plan_back_with_its_timeline_and_its_limit(programs):  # noqa: F811
    serve = {"until_ready": ["fixed", {"name": "cover", "hosted": {"budget": {"per_day": 3}}}], "max_min": 1,
             "tz": "America/New_York"}
    answer = check(programs, serve=serve)
    assert answer.status_code == 200, answer.text
    serving = answer.json()["plan"]["serving"]
    assert serving["readback"][1].startswith("Until ready — while its machines start or are replaced after an eviction, "
                                             "for at most 1 min each time: the operator's fixed machines elsewhere")
    # max_min 1: the stop-gap ends before the machines are ready, and the timeline says so.
    assert [p["phase"] for p in serving["timeline"]][:3] == ["preparing", "preparing", "serving"]
    assert serving["timeline"][1]["said"] == "the stop-gap has ended: its requests wait for its machines"
    assert serving["timeline"][0]["minutes"]["measured"] is False, "no machine of that card rented yet: estimated"
    most = serving["most_it_can_spend"]
    assert most["lease_cap"] == 5.0 and most["total"] == 5.0 + most["hosted"][0]["most"]
    assert any("fixed: no fixed machine of this pool holds big-model" in w for w in serving["warnings"])
    assert any(w.startswith("max_min 1 min is shorter than the 90th-percentile time to ready") for w in serving["warnings"])
    assert serving["providers"] == ["provider"]
    shorthand = check(programs, bursts={"hosted": {"budget": {"per_day": 2}}}).json()["plan"]["serving"]
    assert shorthand["plan"] is None and "Rules: none." in shorthand["readback"], "today's fields read back as a plan"
    schema = httpx.get(f"{programs.control_url}/pool/schema/serving-plan", headers={"Authorization": f"Bearer {ADMIN}"})
    assert schema.status_code == 200 and schema.json() == json_schema()
    assert httpx.get(f"{programs.control_url}/pool/schema/serving-plan").status_code == 401


def cli(pool, capsys, monkeypatch, *argv, stdin=None):
    from gpm_server.cli import main

    monkeypatch.setenv("GPM_ADMIN_KEY", ADMIN)
    monkeypatch.setenv("GPM_CONTROL_URL", pool.control_url)
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_the_cli_takes_a_plan_from_standard_input_and_says_it_back(programs, capsys, monkeypatch):  # noqa: F811
    args = ["workload", "plan", "night", "--model", BIG, "--latency", "30", "--parallel", "2", "--hours", "2",
            "--max-spend", "5"]
    code, out, err = cli(programs, capsys, monkeypatch, *args, "--serve", "-",
                         stdin=json.dumps(EXAMPLES["careful-spending"]))
    assert code == 0, err
    assert "serving plan:" in out and "Rule 'careful spending'" in out and "most it can spend: $" in out
    assert "if every worker of its own is busy: when_full: hosted-burst" in out
    code, out, err = cli(programs, capsys, monkeypatch, *args, "--serve", "-", "--json", stdin='{"until_ready": ["shared"]}')
    assert code == 0 and json.loads(out)["plan"]["serving"]["readback"][0].startswith("Steady:"), err
    code, out, _ = cli(programs, capsys, monkeypatch, "workload", "plan", "--serve-example", "stop-gap")
    assert code == 0 and json.loads(out) == EXAMPLES["stop-gap"]
    code, _, err = cli(programs, capsys, monkeypatch, *args, "--serve", "-", "--hosted-per-day", "2", stdin="{}")
    assert code == 2 and "not both" in err
    with pytest.raises(SystemExit, match="not JSON"):
        cli(programs, capsys, monkeypatch, *args, "--serve", "-", stdin="steady: rented")


def test_a_workload_says_now_where_it_serves_and_a_program_reads_each_budget(programs):  # noqa: F811
    key = grant(programs, hosted_per_day=5.0)
    serve = {"until_ready": ["shared", {"name": "cover", "hosted": {"budget": {"per_day": 2.0}}}],
             "when_full": [{"name": "burst", "hosted": {"after_s": 5, "budget": {"per_day": 1.0}}}],
             "rules": [{"name": "batch waits", "when": {"priority": "batch"}, "then": {"never_use": ["burst"]}}]}
    with provisioner(programs, key) as pool_side, pool_side.workload(BIG, **SPEC, serve=serve) as w:
        programs.loop.run(programs.state.registry.refresh())
        assert set(w.budgets()) == {"cover", "burst"} and w.budgets()["cover"]["per_day_usd"] == 2.0
        assert w.hosted_budget()["per_day_usd"] == 3.0
        view = httpx.get(f"{programs.control_url}/pool/workloads/{w.name}",
                         headers={"Authorization": f"Bearer {ADMIN}"}).json()["workload"]
        now = view["now"]
        assert now["role"] == "until_ready" and now["next"] == "until_ready/shared", now
        assert now["said"].startswith("Covering until ready: 0 of 1 rented machine(s) ready")
        assert "cover $0.00 of $2 today; burst $0.00 of $1 today." in now["said"]
        assert now["said"].endswith("Rules in force: batch waits (by a request's options: batch waits).")


def test_the_now_line_says_where_it_runs_what_it_spent_and_when_the_stop_gap_ended(programs):  # noqa: F811
    def made(name, **body):
        answer = httpx.post(f"{programs.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"},
                            json={"name": name, "model": BIG, "hours": 2, "latency_s": 30, "parallel": 2,
                                  **({} if "reserve" in str(body) else {"max_spend": 5.0}), **body}, timeout=60)
        assert answer.status_code == 201, answer.text

    def now(name):
        return httpx.get(f"{programs.control_url}/pool/workloads/{name}",
                         headers={"Authorization": f"Bearer {ADMIN}"}).json()["workload"]["now"]

    made("owned", serve={"steady": "local", "reserve": {"local": 2}})
    assert now("owned")["said"].startswith("Steady on the operator's local machines. Next request: steady/local.")
    made("legacy", bursts={"hosted": {"budget": {"per_day": 2}}})
    programs.database.execute(
        "INSERT INTO request_log (ts, request_id, outcome, workload, cost_usd, budget) VALUES (?, 'r1', 'ok', 'legacy', "
        "0.5, NULL)", (__import__("time").time(),))
    assert "hosted $0.50 of $2 today." in now("legacy")["said"], "today's fields: its answers carry no budget id"
    made("short", serve={"until_ready": ["shared"], "max_min": 1})
    assert now("short")["said"].startswith("Covering until ready")
    programs.database.execute("UPDATE workloads SET created_at = created_at - 120 WHERE name = 'short'")
    said = now("short")["said"]
    assert "its stop-gap has ended; requests wait." in said and "Next request: waits for its machines." in said
    listed = httpx.get(f"{programs.control_url}/pool/workloads", headers={"Authorization": f"Bearer {ADMIN}"}).json()
    assert all(w["now"]["said"] for w in listed["workloads"])
