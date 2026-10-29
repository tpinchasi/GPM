"""Upgrading an engine the host's agent installed (S9, D153; docs/spec/supervisor.md §3.1).

The agent stages a new version while its host serves. Switching is the pool's call, because only
the pool knows when nothing is being asked of the host: it waits for the host to have served
nothing for a while — or for an operator's *Upgrade now* — takes it out of routing, waits for
its requests in flight, and only then asks the agent to switch. The new version must then answer
and hold what the host held before; otherwise the pool asks for the previous one back, keeps the
host out of routing until it is, and does not take that version on that host again.

A host whose `engine_version` names a version is held to it: that version is the only one the
pool switches to, once the agent has it on disk, and *Roll back* is refused there — the pin is
the operator's word. On `latest`, the staged version is the target.

This module decides; the supervisor carries it out. Each step is a plain function of what the
agent said, what the host holds and the clock, so the whole sequence is tested without a machine.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

#: How long the agent has to finish a switch it started in the background before the pool
#: stops waiting: the agent's own wait for the engine to answer is bounded below this.
SWITCH_TIMEOUT_S = 300.0

#: The fields of `engine.install` the steps read, each a version string or nothing.
_VERSION_FIELDS = ("running", "staged", "previous")


@dataclasses.dataclass
class Upgrade:
    """One host's switch in progress."""

    to: str
    #: What ran before, to go back to; None on a machine where nothing ran yet.
    came_from: Optional[str]
    #: What the host held before, of its own model set — its resident set when pinned, what was
    #: on its disk when on demand — which the new version must hold again.
    held_before: frozenset[str]
    #: "draining" → "switching" → "checking", or "rolling_back" after a check failed.
    phase: str = "draining"
    started_at: float = dataclasses.field(default_factory=time.time)
    until: float = 0.0
    #: Asked by an operator, rather than chosen at an idle moment.
    by_operator: bool = False
    #: Going back to the previous version, on an operator's instruction.
    rollback: bool = False

    def reason(self, engine: str) -> str:
        if self.phase == "rolling_back":
            return f"rolling {engine} back to {self.came_from}: {self.to} did not hold what the host held"
        if self.rollback:
            return f"rolling {engine} back to {self.to}"
        return f"upgrading {engine} {self.came_from or '(none)'} → {self.to}"


@dataclasses.dataclass
class Step:
    """What the supervisor should do now for one host, and the event to record."""

    action: str  # "start" | "switch" | "done" | "rollback" | "abandon" | "wait"
    summary: str = ""
    event: Optional[str] = None
    #: The version to ask the agent for, for "switch" and "rollback".
    version: Optional[str] = None


def well_formed(install: Any) -> bool:
    """Whether an agent's `engine.install` is one the steps can read: an object whose version
    fields are each a string or nothing. Anything else is treated as no answer."""
    return isinstance(install, dict) and all(
        install.get(name) is None or isinstance(install.get(name), str) for name in _VERSION_FIELDS
    ) and (install.get("working") is None or isinstance(install.get("working"), dict))


class EngineUpgrades:
    """The switches in progress, the versions each host will not take, and what operators asked."""

    def __init__(self) -> None:
        self.active: dict[str, Upgrade] = {}
        #: Versions a host switched to and gave back, by host: not tried there again. A newer
        #: release is a new version and is tried as usual.
        self.refused: dict[str, set[str]] = {}
        #: "upgrade" or "rollback", by host, from the console — taken by the next pass.
        self.asked: dict[str, str] = {}
        #: The staged version and the agent's last error seen for each host, so each change is
        #: said once as an event rather than on every pass.
        self.seen: dict[str, tuple[Optional[str], Optional[str]]] = {}

    def ask(self, host_id: str, what: str) -> None:
        self.asked[host_id] = what

    def refuse(self, host_id: str, version: Optional[str]) -> None:
        if version:
            self.refused.setdefault(host_id, set()).add(version)

    def forget(self, host_id: str) -> Optional[Upgrade]:
        """A host gone from the pool, disabled, or without its agent: whatever was in progress
        for it is dropped, so it holds no other host's upgrade back."""
        self.asked.pop(host_id, None)
        self.seen.pop(host_id, None)
        return self.active.pop(host_id, None)

    def news(self, host_id: str, install: Optional[dict[str, Any]]) -> list[Step]:
        """What the agent did on its own since the last pass — a version staged, an error —
        each said once."""
        if not well_formed(install) or not install.get("allowed"):
            return []
        staged, error = install.get("staged"), install.get("error")
        before = self.seen.get(host_id, (None, None))
        self.seen[host_id] = (staged, error)
        engine = str(install.get("engine") or "the engine")
        found = []
        if staged and staged != before[0] and staged != install.get("running"):
            found.append(Step("wait", f"{engine} {staged} is staged beside {install.get('running') or 'nothing'}",
                              event="engine_staged", version=staged))
        if error and error != before[1]:
            found.append(Step("wait", f"{engine}: {str(error)[:300]}", event="engine_install_error"))
        return found

    def target(self, host_id: str, install: dict[str, Any], pinned: Optional[str]) -> tuple[Optional[str], bool, bool]:
        """The version this host should switch to now, if any: (version, by operator, rollback)."""
        running, staged, previous = install.get("running"), install.get("staged"), install.get("previous")
        asked = self.asked.get(host_id)
        if pinned is not None:
            # The pin is the only version the pool switches to, once it is on the machine.
            on_disk = pinned in (staged, previous)
            return (pinned if pinned != running and on_disk else None), asked == "upgrade", False
        if asked == "rollback":
            return (previous, True, True) if previous and previous != running else (None, True, True)
        if not staged or staged == running:
            return None, asked == "upgrade", False
        if staged in self.refused.get(host_id, set()) and asked != "upgrade":
            return None, False, False
        return staged, asked == "upgrade", False

    def next_step(
        self,
        host_id: str,
        install: Optional[dict[str, Any]],
        *,
        busy: int,
        idle_s: Optional[float],
        held: frozenset[str],
        engine_answers: bool,
        idle_needed_s: float,
        drain_s: float,
        check_s: float,
        another_upgrading: bool,
        pinned: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Step:
        """Advance one host's upgrade by one pass. `idle_s` is how long it has served nothing
        (None: never served); `held` what it holds now, of its own model set; `busy` its requests
        in flight; `pinned` the version its configuration holds it to, or None for `latest`."""
        now = time.time() if now is None else now
        upgrade = self.active.get(host_id)
        if not well_formed(install) or not install.get("allowed"):
            return self._without_an_answer(host_id, upgrade, now)
        engine = str(install.get("engine") or "the engine")

        if upgrade is None:
            return self._maybe_start(host_id, install, engine, busy=busy, idle_s=idle_s, held=held,
                                     idle_needed_s=idle_needed_s, drain_s=drain_s,
                                     another_upgrading=another_upgrading, pinned=pinned, now=now)

        if upgrade.phase == "draining":
            if busy <= 0:
                upgrade.phase = "switching"
                upgrade.until = now + SWITCH_TIMEOUT_S
                return Step("switch", f"{upgrade.reason(engine)}: its requests have finished; switching",
                            event="engine_upgrade_switching", version=upgrade.to)
            if now >= upgrade.until:
                del self.active[host_id]
                return Step("abandon", f"{upgrade.reason(engine)} put off: {busy} request(s) were still running "
                            f"after {drain_s:g}s; tried again at its next quiet moment", event="engine_upgrade_postponed")
            return Step("wait")

        working = install.get("working") or {}
        if upgrade.phase == "switching":
            if working.get("step") == "switching" and now < upgrade.until:
                return Step("wait")
            if install.get("running") == upgrade.to:
                upgrade.phase = "checking"
                upgrade.until = now + check_s
                return Step("wait", f"{engine} {upgrade.to} started; checking it holds what the host held",
                            event="engine_switched")
            if now < upgrade.until and not install.get("error"):
                return Step("wait")
            if not upgrade.rollback:
                self.refuse(host_id, upgrade.to)
            if install.get("running") == upgrade.came_from:
                del self.active[host_id]
                return Step("done", f"{engine} {upgrade.to} did not start, and the agent went back to "
                            f"{upgrade.came_from}: {install.get('error') or 'no answer'}", event="engine_rolled_back")
            return self._roll_back(upgrade, engine, now, f"{engine} {upgrade.to} did not start in time: "
                                   f"{install.get('error') or 'no answer'}")

        if upgrade.phase == "checking":
            if engine_answers and upgrade.held_before <= held:
                del self.active[host_id]
                if upgrade.rollback:
                    # Rolled away from on an operator's word: not taken again automatically.
                    self.refuse(host_id, upgrade.came_from)
                return Step("done", f"{engine} {upgrade.to} " + ("is back" if upgrade.rollback else "serves")
                            + f" and holds what the host held ({len(upgrade.held_before)} model(s))",
                            event="engine_rolled_back" if upgrade.rollback else "engine_upgraded")
            if now < upgrade.until:
                return Step("wait")
            missing = sorted(upgrade.held_before - held)
            if upgrade.rollback or not upgrade.came_from:
                del self.active[host_id]
                return Step("done", f"{engine} {upgrade.to} did not hold {missing} after {check_s:g}s, and there "
                            "is nothing to go back to", event="engine_upgrade_failed")
            self.refuse(host_id, upgrade.to)
            return self._roll_back(upgrade, engine, now, f"{engine} {upgrade.to} "
                                   + (f"did not hold {missing}" if engine_answers else "did not answer")
                                   + f" after {check_s:g}s; not taking {upgrade.to} on this host again")

        if upgrade.phase == "rolling_back":
            # Kept out of routing until the old version is back: asked again each pass, as an
            # agent busy with something else answers 409 and a request can be lost.
            if install.get("running") == upgrade.came_from and working.get("step") != "switching":
                del self.active[host_id]
                return Step("done", f"{engine} is back on {upgrade.came_from}", event="engine_rolled_back")
            if now >= upgrade.until:
                del self.active[host_id]
                return Step("done", f"{engine} could not be brought back to {upgrade.came_from} in "
                            f"{SWITCH_TIMEOUT_S:g}s; it runs {install.get('running') or 'nothing'} — roll it back "
                            "from the console or the machine", event="engine_rollback_failed")
            if working.get("step") == "switching":
                return Step("wait")
            return Step("rollback", f"asking again for {upgrade.came_from}", version=upgrade.came_from)
        return Step("wait")

    def _maybe_start(self, host_id: str, install: dict[str, Any], engine: str, *, busy: int,
                     idle_s: Optional[float], held: frozenset[str], idle_needed_s: float, drain_s: float,
                     another_upgrading: bool, pinned: Optional[str], now: float) -> Step:
        to, by_operator, rollback = self.target(host_id, install, pinned)
        if to is None:
            if host_id in self.asked:
                self.asked.pop(host_id)
                return Step("wait", f"nothing to switch to: {engine} runs {install.get('running') or 'nothing'}"
                            + (", and there is no previous version" if rollback else ", and nothing newer is staged"),
                            event="engine_upgrade_not_possible")
            return Step("wait")
        running = install.get("running")
        if not running and not rollback:
            # Nothing runs yet: the agent starts the first version it installs by itself,
            # as there is nothing to drain and nothing held to keep (D153).
            return Step("wait")
        if another_upgrading and not by_operator:
            return Step("wait")
        quiet = idle_s is None or idle_s >= idle_needed_s
        if not by_operator and not (busy == 0 and quiet):
            return Step("wait")
        self.asked.pop(host_id, None)
        upgrade = Upgrade(to=to, came_from=running, held_before=held,
                          by_operator=by_operator, rollback=rollback, until=now + drain_s)
        self.active[host_id] = upgrade
        return Step("start", f"{upgrade.reason(engine)}: "
                    + ("an operator asked" if by_operator else
                       ("its configuration holds it to that version" if pinned else
                        f"it has served nothing for {idle_needed_s:g}s"))
                    + f"; no new requests, and up to {drain_s:g}s for the {busy} it has", event="engine_upgrade_draining")

    def _roll_back(self, upgrade: Upgrade, engine: str, now: float, why: str) -> Step:
        upgrade.phase = "rolling_back"
        upgrade.until = now + SWITCH_TIMEOUT_S
        return Step("rollback", f"{why}; going back to {upgrade.came_from}", event="engine_rolling_back",
                    version=upgrade.came_from)

    def _without_an_answer(self, host_id: str, upgrade: Optional[Upgrade], now: float) -> Step:
        """The agent says nothing readable, or its owner turned installs off: nothing can be
        switched. An ask is dropped, said so; a drain ends at once, as nothing was switched; a
        switch, check or rollback in progress is given up once its window passes, so a host that
        serves is never kept out of routing for good."""
        dropped = self.asked.pop(host_id, None)
        if upgrade is None:
            if dropped:
                return Step("wait", f"the operator's {dropped} was dropped: the host's agent gave no answer "
                            "about its engine versions", event="engine_upgrade_not_possible")
            return Step("wait")
        if upgrade.phase == "draining" or now >= upgrade.until:
            del self.active[host_id]
            return Step("abandon", f"{upgrade.reason('the engine')} given up: the host's agent gave no answer "
                        f"about its engine versions while {upgrade.phase}", event="engine_upgrade_abandoned")
        return Step("wait")
