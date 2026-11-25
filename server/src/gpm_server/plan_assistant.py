"""The console's plan chat (S19 stage 8, D204; console-and-control-api.md).

An operator describes what a workload should do; a model the pool serves on its own machines
turns the description into a serving plan's JSON, held to the plan's schema by structured output.
The chat only proposes: the plan comes back as data, is checked by `parse_serve` and read back in
fixed words, and the console shows it as a change to the builder, with Undo. Creating is the
builder's button.

The request travels through the database, never from one process to the other: the control API
(in the supervisor) writes it to `assistant_requests`; one router's watcher claims it, sends it
through that router's own dispatch with `keep_on: owned` under the app name `console`, and writes
the answer back; the row is deleted once the answer is read, or when the request is cancelled.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Mapping, Optional

from .serving_plan import EXAMPLES, PlanRefused, json_schema, parse_serve, readback

#: The longest description the chat takes.
MAX_DESCRIPTION = 4000
#: The model is asked at most this many times for one description (D204: "after two tries").
TRIES = 2
#: An answer nobody read is dropped after this long; a claim whose router went away, after its
#: timeout and this much more.
FORGOTTEN_S = 3600.0
CLAIM_GRACE_S = 30.0
#: The name its requests carry in the request log and the stats.
APP = "console"


class AssistantBusy(Exception):
    """One request at a time (D204): `request_id` is the one in flight, so it can be waited on or
    cancelled — by a console that reloaded, too."""

    def __init__(self, message: str, request_id: str):
        super().__init__(message)
        self.request_id = request_id


class AssistantRequests:
    """The `assistant_requests` table, as both processes use it."""

    def __init__(self, db: Any):
        self.db = db

    def _changed(self, sql: str, args: tuple) -> int:
        with self.db.transaction() as conn:
            return conn.execute(sql, args).rowcount

    def create(self, description: str, plan: Optional[Mapping[str, Any]], timeout_s: float,
               now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        request_id = uuid.uuid4().hex
        abandoned = now - timeout_s - CLAIM_GRACE_S
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM assistant_requests WHERE state = 'answered' AND created_at < ?",
                         (now - FORGOTTEN_S,))
            # One no router took, or whose router stopped, does not hold the chat for ever.
            conn.execute(
                "UPDATE assistant_requests SET state = 'answered', answer = ? WHERE state = 'waiting' AND created_at < ?",
                (json.dumps({"error": "no router took it: is a router running with console.plan_assistant set?"}),
                 abandoned))
            conn.execute(
                "UPDATE assistant_requests SET state = 'answered', answer = ? WHERE state = 'claimed' AND claimed_at < ?",
                (json.dumps({"error": "the router answering it stopped; ask again"}), abandoned))
            busy = conn.execute("SELECT id FROM assistant_requests WHERE state != 'answered' LIMIT 1").fetchall()
            if busy:
                raise AssistantBusy("the chat is answering another request; wait for it, or cancel it", busy[0]["id"])
            conn.execute("INSERT INTO assistant_requests (id, created_at, description, plan) VALUES (?, ?, ?, ?)",
                         (request_id, now, description, json.dumps(plan) if plan is not None else None))
        return request_id

    def get(self, request_id: str) -> Optional[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM assistant_requests WHERE id = ?", (request_id,))
        if not rows:
            return None
        row = dict(rows[0])
        row["plan"] = json.loads(row["plan"]) if row.get("plan") else None
        row["answer"] = json.loads(row["answer"]) if row.get("answer") else None
        return row

    def delete(self, request_id: str) -> bool:
        return self._changed("DELETE FROM assistant_requests WHERE id = ?", (request_id,)) > 0

    def claim(self, router: str, timeout_s: float, now: Optional[float] = None) -> Optional[dict[str, Any]]:
        """The oldest waiting request, claimed by this router — once: two routers asking at the
        same moment, during a drain, cannot both have it. A claim whose router stopped answers
        with that, so the operator can ask again."""
        now = time.time() if now is None else now
        stale = now - timeout_s - CLAIM_GRACE_S
        for row in self.db.query("SELECT id FROM assistant_requests WHERE state = 'claimed' AND claimed_at < ?",
                                 (stale,)):
            self.answer(row["id"], None, {"error": "the router answering it stopped; ask again"})
        for row in self.db.query("SELECT id FROM assistant_requests WHERE state = 'waiting' ORDER BY created_at"):
            taken = self._changed(
                "UPDATE assistant_requests SET state = 'claimed', claimed_by = ?, claimed_at = ? "
                "WHERE id = ? AND state = 'waiting'", (router, now, row["id"]))
            if taken:
                return self.get(row["id"])
        return None

    def answer(self, request_id: str, router: Optional[str], answer: Mapping[str, Any]) -> bool:
        """Write the answer; False where the request was cancelled meanwhile (its row is gone)."""
        if router is None:
            sql, args = "UPDATE assistant_requests SET state = 'answered', answer = ? WHERE id = ?", (
                json.dumps(answer), request_id)
        else:
            sql, args = ("UPDATE assistant_requests SET state = 'answered', answer = ? WHERE id = ? AND claimed_by = ?",
                         (json.dumps(answer), request_id, router))
        return self._changed(sql, args) > 0

    def exists(self, request_id: str) -> bool:
        return bool(self.db.query("SELECT 1 FROM assistant_requests WHERE id = ?", (request_id,)))


def _schema_for_the_model() -> dict[str, Any]:
    schema = json_schema()
    return {k: v for k, v in schema.items() if k not in ("$schema", "$id", "examples")}


def response_format() -> dict[str, Any]:
    """The OpenAI-compatible structured output that holds the answer to the plan's schema."""
    return {"type": "json_schema", "json_schema": {"name": "serving_plan", "schema": _schema_for_the_model()}}


SYSTEM = """You write serving plans for GPM, a pool that serves AI models to applications.
A serving plan (`serve`) says where a workload's requests go and when, as JSON:
- steady: where it runs — "rented" (machines rented for it, the default), "local" or "fixed" (the operator's own machines, with `reserve`).
- until_ready: a stop-gap used only while its rented machines are prepared (at the start, and while an evicted one is replaced): "shared", "local", "fixed" machines, then {"hosted": {"budget": {...}}} hosted APIs. `max_min` limits it.
- when_full: what helps when every worker is busy: {"hosted": {"after_s": seconds, "budget": {...}}}, {"rented": {"after_min": minutes, "max_hosts": n, "budget": {...}}}.
- A budget is {"per_day": dollars} and/or {"per_month": dollars}. A source may have a "name".
- defaults: {"keep_on": "no_hosted"} keeps requests that send no option off hosted APIs.
- rules: [{"name", "when", "then"}] — when: priority, keep_on, between "HH:MM-HH:MM" (with days), spent_over, evictions; then: use_only, never_use, after_s, steady. `tz` is an IANA time zone for windows.
Answer with the plan's JSON only: no prose, no field the schema does not have. Leave out what the description does not ask for. Never invent a budget larger than the description allows; where it names none, choose a small one."""


def messages(description: str, plan: Optional[Mapping[str, Any]]) -> list[dict[str, str]]:
    """What the model is asked: the plan's format, two examples, the plan the builder holds (to
    change) and the description."""
    examples = "\n".join(f"{name}: {json.dumps(EXAMPLES[name])}" for name in ("stop-gap", "careful-spending"))
    asked = description.strip()
    if plan:
        asked = f"The plan now:\n{json.dumps(plan)}\n\nChange it as follows:\n{asked}"
    return [
        {"role": "system", "content": f"{SYSTEM}\n\nExamples:\n{examples}"},
        {"role": "user", "content": asked},
    ]


def read_answer(content: str) -> tuple[Optional[dict[str, Any]], list[str]]:
    """The model's answer as a plan the pool accepts, or None and why not, in the pool's words."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        raw = json.loads(text)
    except ValueError as exc:
        return None, [f"not JSON: {exc}"]
    if not isinstance(raw, dict):
        return None, ["a serving plan is a JSON object"]
    try:
        plan = parse_serve(raw)
    except PlanRefused as exc:
        return None, [str(exc)]
    return raw, [] if plan is not None else ["empty"]


def answer_for(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A plan the pool accepted, as the console receives it: the plan and its readback."""
    return {"plan": dict(raw), "readback": readback(parse_serve(raw))}
