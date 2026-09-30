"""Programs creating their own workloads (D117, docs/stories/S6), from the supervisor's side.

The router records what a provisioning key asked; this answers it. Every request is untrusted
input — the router checked a key, nothing else — so every field is checked again here, against
the key's grant and a pool-wide daily cap, before the one path that spends is taken: the same
`Workloads.create` an operator uses, with every rule of D115 still in force.
"""

from __future__ import annotations

import math
import secrets
import time
from typing import TYPE_CHECKING, Any, Optional

from ..certs import load_ca
from ..provisioning_store import KEY_HASH, NAME, Grant, ProvisioningStore
from .workloads import KINDS, WorkloadRefused, WorkloadRequest, parse_targets

if TYPE_CHECKING:
    from .service import Supervisor

DAY_S = 24 * 3600


def _number(body: dict, name: str, *, minimum: float = 0.0, integer: bool = False, required: bool = True) -> Any:
    value = body.get(name)
    if value is None:
        if required:
            raise WorkloadRefused(f"{name} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise WorkloadRefused(f"{name} must be a number")
    if integer and int(value) != value:
        raise WorkloadRefused(f"{name} must be a whole number")
    if value <= minimum:
        raise WorkloadRefused(f"{name} must be above {minimum:g}")
    return int(value) if integer else float(value)


class Provisioning:
    def __init__(self, supervisor: "Supervisor"):
        self.supervisor = supervisor
        self.store = ProvisioningStore(supervisor.db)
        self._plans: dict[tuple, tuple[float, dict]] = {}
        self._ca: Any = None
        self._ca_loaded = False

    @property
    def config(self) -> Any:
        return self.supervisor.config

    @property
    def ca(self) -> Any:
        if not self._ca_loaded:
            cfg = self.config.provisioning
            self._ca = load_ca(cfg.client_ca_certfile, cfg.client_ca_keyfile, self.config.listen.client_ca_certfile)
            self._ca_loaded = True
        return self._ca

    # --- the operator's side ---

    def grant(self, name: str, grant: Grant, expires_hours: Optional[float] = None) -> str:
        """A new provisioning key; returned once."""
        if not NAME.match(name) or name.startswith("rented"):
            raise WorkloadRefused("a provisioner's name is lower-case letters, digits and '-', up to 24, not 'rented…'")
        if self.store.get(name) is not None:
            raise WorkloadRefused(f"provisioner {name!r} exists; names are never reused")
        if not grant.models:
            raise WorkloadRefused("a grant names the models it allows")
        unknown = [m for m in grant.models if m not in self.config.pool.model_set
                   and not (self.config.catalog.get(m) and self.config.catalog[m].workloads_only)]
        if unknown:
            raise WorkloadRefused(f"not models of this pool: {unknown}")
        bad_kinds = [k for k in grant.kinds if k not in KINDS]
        if bad_kinds:
            raise WorkloadRefused(f"kinds are {', '.join(KINDS)}; not {bad_kinds}")
        if grant.certs not in ("optional", "required"):
            raise WorkloadRefused("certs is optional or required")
        if grant.certs == "required" and (self.ca is None or not self.config.listen.client_ca_certfile):
            raise WorkloadRefused("certificates need provisioning.client_ca_certfile/keyfile and listen.client_ca_certfile")
        if grant.max_spend > grant.max_spend_per_day:
            raise WorkloadRefused("one workload's budget cannot be above the day's")
        fleet = self.supervisor.workloads.fleet
        longest = fleet.max_lease_hours() if fleet is not None else None
        if longest is not None and grant.max_hours > longest:
            raise WorkloadRefused(f"{fleet.provider.name} has no dead-man timer, so a workload there runs at most "
                                  f"{longest:g}h; the grant's most hours cannot be {grant.max_hours:g}")
        expires = time.time() + expires_hours * 3600 if expires_hours else None
        key = self.store.create(name, grant, expires)
        self.supervisor.events.record(
            "provisioner_granted", f"provisioning key {name}: {grant.as_dict()}",
            numbers={"provisioner": name, "grant": grant.as_dict(), "expires_at": expires},
        )
        return key

    def revoke(self, name: str, end_workloads: bool = False) -> list[str]:
        if self.store.get(name) is None:
            raise WorkloadRefused(f"no provisioner {name!r}")
        self.store.revoke(name)
        ended = []
        if end_workloads:
            for workload in self.supervisor.workloads.store.active():
                if workload.provisioner == name and workload.state in ("preparing", "serving"):
                    self.supervisor.workloads.end(workload.name, "its provisioning key was revoked")
                    ended.append(workload.name)
        self.supervisor.events.record("provisioner_revoked", f"provisioning key {name} revoked",
                                      numbers={"provisioner": name, "ended": ended})
        return ended

    # --- answering requests ---

    async def answer_pending(self, limit: Optional[int] = None) -> int:
        """Answer pending requests, oldest first — at most `limit`. Returns how many were answered."""
        answered = 0
        for request in self.store.pending()[:limit]:
            try:
                state, answer, workload = await self._answer(request)
            except WorkloadRefused as exc:
                state, answer, workload = "refused", {"detail": str(exc)}, None
            except Exception as exc:  # noqa: BLE001 - a request never takes the supervisor down
                state, answer, workload = "refused", {"detail": f"could not be answered: {type(exc).__name__}"}, None
            self.store.answer(request.request_id, state, answer, workload)
            self.supervisor.events.record(
                "provisioning_" + state, f"{request.provisioner} asked to {request.kind}: {state}"
                + (f" — {answer.get('detail')}" if state == "refused" else ""),
                numbers={"provisioner": request.provisioner, "request": request.request_id, "workload": workload},
            )
            answered += 1
        return answered

    async def _answer(self, request: Any) -> tuple[str, dict, Optional[str]]:
        provisioner = self.store.get(request.provisioner)
        if provisioner is None or not provisioner.usable():
            raise WorkloadRefused("this provisioning key is revoked or expired")
        if not self.config.provisioning.enabled:
            raise WorkloadRefused("this pool does not take workloads from programs")
        if request.kind == "end":
            return self._end(provisioner, request)
        body = request.body if isinstance(request.body, dict) else {}
        if request.kind == "plan":
            return "done", {"plan": await self._plan(provisioner, body)}, None
        if request.kind == "create":
            return await self._create(provisioner, request, body)
        raise WorkloadRefused(f"unknown request kind {request.kind!r}")

    def _end(self, provisioner: Any, request: Any) -> tuple[str, dict, Optional[str]]:
        workload = self.supervisor.workloads.store.get(request.workload or "")
        if workload is None or workload.provisioner != provisioner.name:
            raise WorkloadRefused("no such workload of this key")
        self.supervisor.workloads.end(workload.name, "ended by the program that made it")
        return "done", {"state": self.supervisor.workloads.get(workload.name).state}, workload.name

    def _request(self, provisioner: Any, body: dict, name: str) -> tuple[WorkloadRequest, float, bool]:
        """The request, checked against the grant: every field is untrusted until here."""
        grant: Grant = provisioner.grant
        targets, placement = parse_targets(body)
        refused = [t.model for t in targets if t.model not in grant.models]
        if refused:
            raise WorkloadRefused(f"this key may create workloads for {list(grant.models)}")
        kind = body.get("machines", "roi")
        if kind not in grant.kinds:
            raise WorkloadRefused(f"this key may rent {list(grant.kinds)}")
        hours = _number(body, "hours")
        if hours > grant.max_hours:
            raise WorkloadRefused(f"at most {grant.max_hours:g} hours")
        spend = _number(body, "max_spend")
        if spend > grant.max_spend:
            raise WorkloadRefused(f"at most ${grant.max_spend:.2f} per workload")
        idle = _number(body, "idle_end_minutes", required=False)
        idle = grant.idle_end_minutes if idle is None else idle
        if idle > grant.max_idle_end_minutes:
            raise WorkloadRefused(f"an idle cutoff of at most {grant.max_idle_end_minutes:g} minutes")
        req = WorkloadRequest.of(name, targets, hours, max_spend=spend, kind=kind, placement=placement)
        return req, idle, grant.may_borrow

    async def _plan(self, provisioner: Any, body: dict) -> dict:
        key = (provisioner.name, repr(sorted((k, v) for k, v in body.items() if k not in ("key_hash", "csr"))))
        cached = self._plans.get(key)
        if cached is not None and time.monotonic() - cached[0] < self.config.provisioning.plan_cache_s:
            return cached[1]
        req, _, _ = self._request(provisioner, body, f"{provisioner.name}-plan")
        plan = await self.supervisor.workloads.plan(req)
        plan.pop("name", None)
        plan["within_grant"] = self._budget_refusal(provisioner, req.max_spend) is None
        self._plans[key] = (time.monotonic(), plan)
        return plan

    def _committed(self, provisioner: Optional[str], since: float) -> float:
        """Dollars committed by workloads made by programs since `since`, or still open: their
        budgets, not what has been recorded yet — a program could open several before any spend
        showed (D117). One made 23 hours ago and still running still counts: otherwise a day
        could see the cap plus every workload still spending from the day before."""
        total = 0.0
        for workload in self.supervisor.workloads.store.all():
            if workload.provisioner is None or (workload.created_at < since and workload.state == "ended"):
                continue
            if provisioner is not None and workload.provisioner != provisioner:
                continue
            lease = self.supervisor.leases.get(workload.lease_id)
            total += lease.max_spend if lease is not None else 0.0
        return total

    def usage(self, name: str) -> dict[str, Any]:
        """What one key has open and has committed in the last 24 hours: what an operator watches
        against its grant."""
        open_now = [w for w in self.supervisor.workloads.store.active()
                    if w.provisioner == name and w.state in ("preparing", "serving", "ending")]
        return {"open_now": len(open_now), "committed_today": round(self._committed(name, time.time() - DAY_S), 2)}

    def _budget_refusal(self, provisioner: Any, spend: float) -> Optional[str]:
        since = time.time() - DAY_S
        mine = self._committed(provisioner.name, since)
        if mine + spend > provisioner.grant.max_spend_per_day:
            return (f"this key committed ${mine:.2f} in the last 24 hours; ${spend:.2f} more would pass its "
                    f"${provisioner.grant.max_spend_per_day:.2f} a day")
        everyone = self._committed(None, since)
        cap = self.config.provisioning.max_spend_per_day or 0.0
        if everyone + spend > cap:
            return f"programs committed ${everyone:.2f} in the last 24 hours; ${spend:.2f} more would pass the pool's ${cap:.2f} a day"
        return None

    async def _create(self, provisioner: Any, request: Any, body: dict) -> tuple[str, dict, Optional[str]]:
        key_hash = request.key_hash
        if not isinstance(key_hash, str) or not KEY_HASH.match(key_hash):
            raise WorkloadRefused("key_hash is the sha256 of the workload key, in hex")
        workloads = self.supervisor.workloads
        existing = workloads.store.workload_of_key_hash(key_hash)
        if existing is not None:
            made = workloads.get(existing)
            if made.provisioner != provisioner.name:
                raise WorkloadRefused("this key_hash is in use; make a new workload key")
            # Already made for this key: the same request, answered again — never a second lease.
            return "done", self._made(made, None), existing
        at_once = [w for w in workloads.store.active()
                   if w.provisioner == provisioner.name and w.state in ("preparing", "serving", "ending")]
        if len(at_once) >= provisioner.grant.max_open:
            raise WorkloadRefused(f"this key has {len(at_once)} workload(s), at its limit of {provisioner.grant.max_open}")
        # The cheap refusals before the market search, which the whole pool waits on.
        full = workloads.full()
        if full is not None:
            raise WorkloadRefused(full)
        name = f"{provisioner.name}-{secrets.token_hex(3)}"
        req, idle, may_borrow = self._request(provisioner, body, name)
        refusal = self._budget_refusal(provisioner, req.max_spend)
        if refusal:
            raise WorkloadRefused(refusal)
        csr = body.get("csr")
        if provisioner.grant.certs == "required" and not csr:
            raise WorkloadRefused("this key's workloads are reached with a client certificate: send csr")
        if csr is not None and not isinstance(csr, str):
            raise WorkloadRefused("csr is a PEM signing request")
        if csr and not self.config.listen.client_ca_certfile:
            # The listener would never ask for the certificate, and the workload could not be reached.
            raise WorkloadRefused("this pool's listener does not ask for client certificates "
                                  "(listen.client_ca_certfile); create without certs")
        workload, _, _, certificate = await workloads.create(
            req, key_hash=key_hash, provisioner=provisioner.name, idle_end_minutes=idle,
            may_borrow=may_borrow, csr=csr, ca=self.ca if csr else None,
        )
        return "done", self._made(workload, certificate), workload.name

    def _made(self, workload: Any, certificate: Optional[str]) -> dict:
        lease = self.supervisor.leases.get(workload.lease_id)
        return {
            "workload": workload.name, "state": workload.state, "model": workload.model, "models": list(workload.models),
            "ends_at": workload.ends_at, "max_spend": lease.max_spend if lease else None,
            "idle_end_minutes": workload.idle_end_minutes, "path_prefix": f"/w/{workload.name}/v1",
            "certificate": certificate, "ca": self.ca.pem if (certificate and self.ca) else None,
        }
