"""Upgrading an engine the host's agent installed (S9, D153; docs/spec/supervisor.md §3.1).

The agent stages a new version while its host serves. Switching is the pool's call, because only
the pool knows when nothing is being asked of the host: it waits for the host to have served
nothing for a while — or for an operator's *Upgrade now* — takes it out of routing, waits for
its requests in flight, and only then asks the agent to switch. The new version must then answer
and hold what the host held before; otherwise the pool asks for the previous one back, and does
not take that version on that host again.

This module decides; the supervisor carries it out. Each step is a plain function of what the
agent said, what the host holds and the clock, so the whole sequence is tested without a machine.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

#: How long the agent has to finish a switch it started in the background before the pool
#: stops waiting and asks for the previous version: the agent's own wait for the engine to
#: answer is bounded below this.
SWITCH_TIMEOUT_S = 300.0


@dataclasses.dataclass
class Upgrade:
    """One host's switch in progress."""

    to: str
    #: What ran before, to go back to; None on a machine where nothing ran yet.
    came_from: Optional[str]
    #: What the host held before — its resident set when pinned, what was on its disk when on
    #: demand — which the new version must hold again.
    held_before: frozenset[str]
    #: "draining" → "switching" → "checking".
    phase: str = "draining"
    started_at: float = dataclasses.field(default_factory=time.time)
    until: float = 0.0
    #: Asked by an operator, rather than chosen at an idle moment.
    by_operator: bool = False
    #: Going back to the previous version, on an operator's instruction.
    rollback: bool = False

    def reason(self, engine: str) -> str:
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


class EngineUpgrades:
    """The switches in progress, the versions each host will not take, and what operators asked."""

    def __init__(self) -> None:
        self.active: dict[str, Upgrade] = {}
        #: Versions a host switched to and gave back, by host: not tried there again. A newer
        #: release is a new version and is tried as usual.
        self.refused: dict[str, set[str]] = {}
        #: "upgrade" or "rollback", by host, from the console — taken by the next pass.
        self.asked: dict[str, str] = {}

    def ask(self, host_id: str, what: str) -> None:
        self.asked[host_id] = what

    def refuse(self, host_id: str, version: Optional[str]) -> None:
        if version:
            self.refused.setdefault(host_id, set()).add(version)

    def target(self, host_id: str, install: dict[str, Any]) -> tuple[Optional[str], bool, bool]:
        """The version this host should switch to now, if any: (version, by operator, rollback)."""
        running, staged, previous = install.get("running"), install.get("staged"), install.get("previous")
        asked = self.asked.get(host_id)
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
        now: Optional[float] = None,
    ) -> Step:
        """Advance one host's upgrade by one pass. `idle_s` is how long it has served nothing
        (None: never served); `held` what it holds now; `busy` its requests in flight."""
        now = time.time() if now is None else now
        if not install or not install.get("allowed"):
            self.asked.pop(host_id, None)
            return Step("wait")
        engine = str(install.get("engine") or "the engine")
        upgrade = self.active.get(host_id)

        if upgrade is None:
            to, by_operator, rollback = self.target(host_id, install)
            if to is None:
                if host_id in self.asked:
                    self.asked.pop(host_id)
                    return Step("wait", f"nothing to switch to: {engine} runs {install.get('running') or 'nothing'}"
                                + (", and there is no previous version" if rollback else ", and nothing newer is staged"),
                                event="engine_upgrade_not_possible")
                return Step("wait")
            if another_upgrading and not by_operator:
                return Step("wait")
            running = install.get("running")
            quiet = idle_s is None or idle_s >= idle_needed_s
            if running and not by_operator and not (busy == 0 and quiet):
                return Step("wait")
            self.asked.pop(host_id, None)
            upgrade = Upgrade(to=to, came_from=running, held_before=held if running else frozenset(),
                              by_operator=by_operator, rollback=rollback, until=now + drain_s)
            self.active[host_id] = upgrade
            if not running:
                # Nothing runs yet: nothing to drain, and nothing held to keep.
                upgrade.phase = "switching"
                upgrade.until = now + SWITCH_TIMEOUT_S
                return Step("switch", f"installing {engine} {to}: nothing ran before", event="engine_upgrade_switching",
                            version=to)
            return Step("start", f"{upgrade.reason(engine)}: "
                        + ("an operator asked" if by_operator else f"it has served nothing for {idle_needed_s:g}s")
                        + f"; no new requests, and up to {drain_s:g}s for the {busy} it has", event="engine_upgrade_draining")

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

        if upgrade.phase == "switching":
            working = install.get("working") or {}
            if working.get("step") == "switching" and now < upgrade.until:
                return Step("wait")
            if install.get("running") == upgrade.to:
                upgrade.phase = "checking"
                upgrade.until = now + check_s
                return Step("wait", f"{engine} {upgrade.to} started; checking it holds what the host held",
                            event="engine_switched")
            if now < upgrade.until and not install.get("error"):
                return Step("wait")
            del self.active[host_id]
            if not upgrade.rollback:
                self.refuse(host_id, upgrade.to)
            if install.get("running") == upgrade.came_from:
                return Step("done", f"{engine} {upgrade.to} did not start, and the agent went back to "
                            f"{upgrade.came_from}: {install.get('error') or 'no answer'}", event="engine_rolled_back")
            return Step("rollback", f"{engine} {upgrade.to} did not start in time: "
                        f"{install.get('error') or 'no answer'}; asking for {upgrade.came_from}", event="engine_rolled_back",
                        version=upgrade.came_from)

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
            del self.active[host_id]
            if upgrade.rollback or not upgrade.came_from:
                return Step("done", f"{engine} {upgrade.to} did not hold {sorted(upgrade.held_before - held)} "
                            f"after {check_s:g}s, and there is nothing to go back to", event="engine_upgrade_failed")
            self.refuse(host_id, upgrade.to)
            missing = sorted(upgrade.held_before - held)
            return Step("rollback", f"{engine} {upgrade.to} " + (f"did not hold {missing}" if engine_answers else "did not answer")
                        + f" after {check_s:g}s; going back to {upgrade.came_from}, and not taking {upgrade.to} on "
                        "this host again", event="engine_rolled_back", version=upgrade.came_from)
        return Step("wait")
