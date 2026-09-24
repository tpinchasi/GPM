"""Pool configuration: the one source of truth, loaded and validated before anything listens.

Schema and rules come from docs/spec/hosts-routing-capacity.md (§1.3 transports, §3 model set,
§4.1 catalog) and docs/spec/app-contract.md §5 (the time budget invariant). Loading performs no
network calls: validation here is about the file, readiness is the probe's job.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigError(Exception):
    """The configuration file is not usable. Raised at load, never mid-request."""


class SecretMissing(ConfigError):
    """A secret the configuration refers to is not present in the environment."""


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str
    requires: list[str] = Field(default_factory=list)
    #: The engine this build is for, when a pool runs more than one (D93). Weights are named
    #: differently by different engines — the same model is `gemma4:26b` to one and a model-hub
    #: repository to another — so a variant that does not say is usable by any engine, and one
    #: that does is offered only to hosts running it.
    engine: Optional[str] = None
    runtime_class: str = "by-platform"
    #: Three-valued on purpose: True when this build is known to enforce a structured-output
    #: schema, False when it is known to accept one and ignore it, and unset when nobody has
    #: said. A schema-carrying request is never routed to a False.
    enforces_schema: Optional[bool] = None
    #: Optional: the download size, so a plan can say in gigabytes what an agent would fetch.
    #: An engine cannot tell the size of a model it does not have yet.
    size_gb: Optional[float] = Field(default=None, gt=0)


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variants: list[Variant] = Field(min_length=1)


class TransportConfig(BaseModel):
    """How the router dials a host. Independent of the host's kind (spec §1.3)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["http", "https", "tunnel"]
    #: `http` / `https` only. A tunnel's URL is derived: the engine is never exposed, so the
    #: router dials a local port the pool itself forwards.
    base_url: Optional[str] = None
    bearer_env: Optional[str] = None
    basic_env: Optional[str] = None
    verify: bool | str = True
    client_cert: Optional[str] = None
    allow_insecure: bool = False

    # --- tunnel ---
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    ssh_user: Optional[str] = None
    ssh_key: Optional[str] = None
    #: Where the engine listens *on the far side*. Not guessed: state it.
    remote_host: str = "127.0.0.1"
    remote_port: Optional[int] = None
    #: A pool-owned known-hosts file. The host's key is recorded on first connection and
    #: checked on every later one.
    known_hosts: str = "~/.config/gpm/known_hosts"
    #: Fixed local port for the forward. Allocated automatically when not set.
    local_port: Optional[int] = None

    @model_validator(mode="after")
    def _check(self) -> "TransportConfig":
        if self.type == "tunnel":
            return self._check_tunnel()
        if not self.base_url:
            raise ValueError(f"transport {self.type!r} needs a base_url")
        scheme = urlparse(self.base_url).scheme
        if scheme != self.type:
            raise ValueError(f"transport type {self.type!r} does not match base_url scheme {scheme!r}")
        if self.type == "http" and not self.allow_insecure:
            if not _is_loopback(self.base_url):
                raise ValueError(
                    "a plain-http host off loopback carries its bearer key in clear text. "
                    "Use https, or set allow_insecure: true on this host to say so deliberately."
                )
        return self

    def _check_tunnel(self) -> "TransportConfig":
        if self.base_url:
            raise ValueError(
                "a tunnel host has no base_url: the pool forwards a local port and dials that. "
                "Give ssh_host, remote_port and the rest instead."
            )
        missing = [
            name for name in ("ssh_host", "remote_port") if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"transport 'tunnel' needs {', '.join(missing)}")
        return self

    def auth_headers(self) -> dict[str, str]:
        """Secrets are read from the environment at load, never written in the file."""
        if self.bearer_env:
            value = os.environ.get(self.bearer_env)
            if not value:
                raise SecretMissing(f"environment variable {self.bearer_env} is not set")
            return {"Authorization": f"Bearer {value}"}
        if self.basic_env:
            value = os.environ.get(self.basic_env)
            if not value:
                raise SecretMissing(f"environment variable {self.basic_env} is not set")
            return {"Authorization": f"Basic {value}"}
        return {}


def _is_loopback(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class AgentConfig(BaseModel):
    """Where this host's agent answers (docs/spec/host-agent.md). The pool dials it, under the
    same rule as the engine: a bearer key never crosses a network in clear text by accident.

    Give `url` for an agent reached over `http` or `https`. On a host reached by SSH tunnel,
    give `remote_port` instead: the agent stays on loopback over there, exposed to nothing, and
    the pool forwards a second local port to it with the host's own SSH settings and pinned key.
    """

    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = None
    #: Tunnel hosts only: the port the agent listens on, on the far side's loopback.
    remote_port: Optional[int] = Field(default=None, ge=1, le=65535)
    #: Names the environment variable holding the agent key. Never the key itself.
    bearer_env: str
    verify: bool | str = True
    allow_insecure: bool = False
    #: With an agent, the host is delegated: the agent pulls what the model set needs and holds
    #: it as `residency` says. False keeps the agent to reporting what the machine is.
    manage_models: bool = True

    @model_validator(mode="after")
    def _check(self) -> "AgentConfig":
        if (self.url is None) == (self.remote_port is None):
            raise ValueError("an agent needs exactly one of `url` (http/https) or `remote_port` (on a tunnel host)")
        if self.url is None:
            return self
        scheme = urlparse(self.url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"an agent url must be http or https, not {scheme!r}")
        if scheme == "http" and not self.allow_insecure and not _is_loopback(self.url):
            raise ValueError(
                "a plain-http agent off loopback carries its agent key in clear text. Use https, "
                "or set allow_insecure: true on this agent to say so deliberately."
            )
        return self

    def key(self) -> Optional[str]:
        """Read when the agent is dialled, not at load: a missing agent key must not stop the
        pool from serving. The host works without its agent; the console says it is missing."""
        return os.environ.get(self.bearer_env) or None


_DEFAULT_PRIORITY = {"local": 0, "fixed-remote": 10, "rented-interruptible": 20, "rented-on-demand": 20}


class HostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["local", "fixed-remote"]
    transport: TransportConfig
    workers: int = Field(default=1, ge=1)
    capabilities: list[str] = Field(default_factory=list)
    priority: Optional[int] = None
    disabled: bool = False
    #: `pinned`: the host is routable only while the whole model set is loaded, and a model
    #: found evicted takes it out of routing — for a machine dedicated to serving. `on_demand`:
    #: routable once the set is *on disk*; the engine loads a model on first use and may evict
    #: it when memory is wanted elsewhere — for a laptop that is also used for other work.
    #: Either way nothing is ever downloaded because a request asked for it (spec §3).
    #: The engine on this machine, when it is not the pool's (D93). A laptop may run one engine
    #: and rented hosts another: what the machine runs is a fact about the machine, and a pool
    #: that can only hold one engine cannot use both at once.
    engine: Optional[str] = None
    residency: Literal["pinned", "on_demand"] = "pinned"
    #: Which of the pool's models this host holds, when the pool spreads its set across hosts
    #: (`pool.models_per_host: declared`, D89). Absent there means the first model in the set this
    #: host can serve — deterministic, and reported, rather than left to chance. Meaningless
    #: when every host holds everything, and refused there so it cannot read as a restriction
    #: the pool is quietly ignoring.
    models: Optional[list[str]] = None
    #: Optional. With an agent the pool learns what the machine is, rather than being told.
    agent: Optional[AgentConfig] = None

    @model_validator(mode="after")
    def _agent_matches_transport(self) -> "HostConfig":
        if self.agent is not None and self.agent.remote_port is not None and self.transport.type != "tunnel":
            raise ValueError(
                f"host {self.id!r}: an agent `remote_port` is forwarded over the host's SSH tunnel, "
                f"and this host's transport is {self.transport.type!r}. Give the agent a `url` instead."
            )
        return self

    @property
    def routing_priority(self) -> int:
        return self.priority if self.priority is not None else _DEFAULT_PRIORITY[self.kind]


class AuthConfig(BaseModel):
    """Two roles, never interchangeable. Keys live hashed in a file; the literal and
    environment forms exist for development and tests."""

    model_config = ConfigDict(extra="forbid")

    app_keys: list[str] = Field(default_factory=list)
    app_keys_env: Optional[str] = None
    app_keys_file: Optional[str] = None
    admin_keys: list[str] = Field(default_factory=list)
    admin_keys_env: Optional[str] = None
    admin_keys_file: Optional[str] = None

    def _literal(self, keys: list[str], env_name: Optional[str]) -> set[str]:
        found = set(keys)
        if env_name:
            raw = os.environ.get(env_name)
            if not raw:
                raise SecretMissing(f"environment variable {env_name} is not set")
            found |= {k.strip() for k in raw.split(",") if k.strip()}
        return found

    def app_hashes(self) -> set[str]:
        from .keys import KeyStore, fingerprint

        hashes = {fingerprint(k) for k in self._literal(self.app_keys, self.app_keys_env)}
        if self.app_keys_file:
            hashes |= KeyStore(self.app_keys_file).hashes("app")
        if not hashes:
            raise ConfigError(
                "the pool has no app key. The key is required on loopback too — any web page "
                "open in a browser on this machine can post to 127.0.0.1. "
                "Create one with `gpm key create --role app`."
            )
        return hashes

    def admin_hashes(self) -> set[str]:
        from .keys import KeyStore, fingerprint

        hashes = {fingerprint(k) for k in self._literal(self.admin_keys, self.admin_keys_env)}
        if self.admin_keys_file:
            hashes |= KeyStore(self.admin_keys_file).hashes("admin")
        return hashes



class ListenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8080
    tls_certfile: Optional[str] = None
    tls_keyfile: Optional[str] = None

    @model_validator(mode="after")
    def _tls_off_loopback(self) -> "ListenConfig":
        off_loopback = not _is_loopback(f"http://{self.host}")
        if off_loopback and not (self.tls_certfile and self.tls_keyfile):
            raise ValueError(
                "listening off loopback requires TLS: a bearer key over plain HTTP is a "
                "published key. Set listen.tls_certfile and listen.tls_keyfile."
            )
        return self


class ControlConfig(BaseModel):
    """Where the control API listens. It is served only when an admin key exists — an
    unauthenticated control API would be worse than none."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8081
    tls_certfile: Optional[str] = None
    tls_keyfile: Optional[str] = None

    @model_validator(mode="after")
    def _tls_off_loopback(self) -> "ControlConfig":
        if not _is_loopback(f"http://{self.host}") and not (self.tls_certfile and self.tls_keyfile):
            raise ValueError(
                "the control API off loopback requires TLS: it carries the admin key, which "
                "can spend money"
            )
        return self


class DeliveryConfig(BaseModel):
    """How a response reaches the app: as it is generated, or once it is whole (D62).

    A host that can be taken away mid-generation hands the app tokens it cannot take back, so
    its responses are held until complete; the pool can then re-run a lost request itself.
    """

    model_config = ConfigDict(extra="forbid")

    rented_interruptible: Literal["buffered", "stream"] = "buffered"
    rented_on_demand: Literal["buffered", "stream"] = "stream"
    local: Literal["buffered", "stream"] = "stream"
    fixed_remote: Literal["buffered", "stream"] = "stream"
    #: May an app ask for tokens as they come, and accept that the stream may break?
    allow_request_override: bool = True
    #: How many further hosts a request lost mid-buffer may be tried on. Each attempt can cost
    #: a whole generation time, so it is not a number to raise lightly.
    max_redispatch: int = Field(default=1, ge=0, le=3)
    #: A response larger than this is flushed and streamed from there on, never failed.
    max_buffer_mb: float = Field(default=16.0, gt=0)

    def for_kind(self, kind: str) -> str:
        return {
            "rented-interruptible": self.rented_interruptible,
            "rented-on-demand": self.rented_on_demand,
            "local": self.local,
            "fixed-remote": self.fixed_remote,
        }.get(kind, "stream")


class PoolSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    model_set: list[str] = Field(min_length=1)
    #: How the pool's model set is spread over its hosts (D89).
    #:
    #: `all` is the original rule (D23): every host holds the whole set, loaded permanently, so
    #: any ready host can serve any request and nothing is ever swapped. `declared` keeps the second
    #: half of that promise and drops the first — a host holds a single model, permanently, and
    #: the *pool* covers the set rather than each machine. Nothing swaps under either.
    #:
    #: `declared` exists because some engines serve exactly one model per process, and because a
    #: 0.3 GB embedding model does not need to sit on the card that was rented for a 26B one.
    models_per_host: Literal["all", "declared"] = "all"
    queue_timeout_s: float = 30.0
    probe_interval_s: float = 10.0
    #: How often the router re-reads the host table the supervisor publishes.
    host_table_poll_s: float = 2.0
    upstream_connect_timeout_s: float = 5.0
    upstream_read_timeout_s: float = 300.0
    #: What the SDK allows for time-to-first-byte. Only used to enforce the invariant below.
    client_time_to_first_byte_s: float = 300.0
    delivery: DeliveryConfig = Field(default_factory=DeliveryConfig)

    @field_validator("model_set")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("model_set contains duplicates")
        return value

    @model_validator(mode="after")
    def _time_budget_invariant(self) -> "PoolSettings":
        # docs/spec/app-contract.md §5: the router must always answer a queued request — with a
        # worker or a 503 — before an SDK client would give up on it.
        if self.queue_timeout_s >= self.client_time_to_first_byte_s:
            raise ValueError(
                f"queue_timeout_s ({self.queue_timeout_s}) must be below "
                f"client_time_to_first_byte_s ({self.client_time_to_first_byte_s})"
            )
        return self


class OfferPolicy(BaseModel):
    """Hard filters on the market. Never relaxed unattended (supervisor.md §6.1)."""

    model_config = ConfigDict(extra="forbid")

    min_gpu_memory_gb: float = 0.0
    min_disk_gb: float = 0.0
    max_all_in_hourly: Optional[float] = None
    #: The same ceiling, per accelerator (D85). A machine-level cap refuses every
    #: multi-GPU offer on its total price, however good its value: a 2-card machine at
    #: $2.80 is cheaper per card than a single at $1.47, and the machine cap cannot see
    #: that. Both may be set; an offer must pass whichever are set.
    max_all_in_per_gpu: Optional[float] = None
    max_download_per_gb: Optional[float] = None
    min_download_mbps: float = 0.0
    min_reliability: float = 0.0
    #: The accelerator driver an engine image needs, as "major" or "major.minor"
    #: (D81). Below it the engine finds the card unusable and runs on the CPU — at the
    #: accelerator's price. Seen live: an A100-80GB on driver 535 under an image wanting
    #: 550, which loaded a 26B model at 100% CPU and was given up 30 minutes later.
    min_driver_version: Optional[str] = None
    verified_only: bool = True
    avoid_machines: list[str] = Field(default_factory=list)
    #: Hardware names to refuse by substring, case-insensitive — a mining card passes every
    #: numeric filter and will not run the engine.
    exclude_hardware: list[str] = Field(default_factory=list)


class ScaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_up_after_s: float = 120.0
    scale_down_after_s: float = 600.0
    min_useful_hours: float = 1.0
    #: A bad market yields one failed bid, not five.
    one_at_a_time: bool = True


class BiddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["floor_plus_premium"] = "floor_plus_premium"
    premium: float = 0.02
    #: Mandatory: nothing bids without a stated ceiling.
    bid_ceiling: float
    on_demand_crossover: float = 0.8
    attempts: int = 3


class SpendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Caps are enforced on the higher of estimated and reported spend, less this margin, so a
    #: lease stops *before* its limit.
    cap_safety_margin: float = 0.10
    drift_alert: float = 0.15
    #: Widened automatically when a provider cannot report charges.
    margin_without_reported_charges: float = 0.25


class TeardownConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: A rented host with nothing routed to it for this long is paused: parked where the
    #: provider can park, left running where it cannot (D58, D64).
    idle_minutes: float = 2.0
    #: …and one still unused this long after its last request is destroyed. Between the two,
    #: measured load brings a parked host straight back (D64).
    #: Unset, it follows the idle window at the shipped ratio — 2 minutes gives 5.
    destroy_idle_minutes: Optional[float] = Field(default=None, gt=0)
    drain_timeout_s: float = 300.0
    deadman_minutes: float = 20.0
    deadman_action: Literal["destroy", "stop"] = "destroy"
    max_park_hours: float = 72.0
    park_when_idle: bool = True
    #: A host that is still not ready after this is destroyed: something is wrong with it,
    #: and it has been billing the whole time.
    max_preparing_minutes: float = 30.0
    #: The longest lease allowed on a provider with no instance-scoped credential, where
    #: nothing on the host can stop it billing.
    max_hours_without_deadman: float = 1.0

    @model_validator(mode="after")
    def _idle_order(self) -> "TeardownConfig":
        if self.destroy_idle_minutes is not None and self.destroy_idle_minutes < self.idle_minutes:
            raise ValueError(
                f"destroy_idle_minutes ({self.destroy_idle_minutes:g}) must not be below "
                f"idle_minutes ({self.idle_minutes:g}): a host is paused first, destroyed after"
            )
        return self

    @property
    def destroy_after_minutes(self) -> float:
        """When an unused host is destroyed, counted from its last request."""
        if self.destroy_idle_minutes is not None:
            return self.destroy_idle_minutes
        return self.idle_minutes * 2.5
    #: A host whose engine has still never answered after this is given up: the provider is
    #: stuck scheduling or starting it, and it has been billing all the while. Much shorter
    #: than `max_preparing_minutes`, which has to allow for downloading the model set.
    max_starting_minutes: float = Field(default=10.0, gt=0)
    #: A model download slower than this, sustained for `slow_pull_grace_s`, gives the host up
    #: — the offer's advertised speed was not what the machine delivers. 0 switches it off.
    min_pull_mbps: float = Field(default=50.0, ge=0)
    slow_pull_grace_s: float = Field(default=120.0, ge=0)
    #: A machine that failed to start or to download is not bid on again for this long, or the
    #: best-ranked offer — the same machine — is simply rented again.
    avoid_failed_machine_minutes: float = Field(default=60.0, ge=0)
    #: How many times one model's download is tried before the host is given up. A cut
    #: download resumes from what arrived, so a retry is cheap; a host is not.
    pull_attempts: int = Field(default=4, ge=1, le=10)
    #: The wait before the second attempt; it doubles for each one after, up to two minutes.
    pull_retry_after_s: float = Field(default=10.0, ge=0.0)


class CapacityMatch(BaseModel):
    """What a capacity profile applies to. Every field given must hold; none given matches all."""

    model_config = ConfigDict(extra="forbid")

    #: The hardware as the market lists it, e.g. "1x RTX PRO 6000 Max-Q" — compared whole and
    #: case-insensitively, so "2x …" of the same card is a different profile, as it should be.
    hardware: Optional[str] = None
    #: The **card**, whatever the machine holds of it: "RTX PRO 6000 WS" matches both
    #: "1x RTX PRO 6000 WS" and "2x …" (D88). What a card can run at once is a fact about the
    #: card, and writing it as a whole-string match means one profile per possible count — and
    #: a profile that silently never fires if the count is wrong, which is how a pool ran six
    #: workers on a card its operator had given nine.
    #:
    #: Matched this way, `max_workers` is **per card** and multiplied by how many the machine
    #: has: two cards run twice the work, and paying for the second one to sit idle is the
    #: whole reason a multi-GPU machine was worth renting.
    gpu: Optional[str] = None
    min_gpu_memory_gb: Optional[float] = None
    capability: Optional[str] = None


class DynamicAllocationConfig(BaseModel):
    """Adding hosts from measured load, gradually (D66).

    Not one at a time, which took half an hour to reach six live; and not the whole gap at
    once, which would buy a fleet of downloads for a short spike. One, then a multiple, then a
    multiple again, each round waiting for the last to land.
    """

    model_config = ConfigDict(extra="forbid")

    #: Rent before saturation, not at it: a pool held at exactly full is a pool that queues.
    target_utilisation: float = Field(default=0.75, gt=0, le=1)
    #: How long load must hold before the first round, and the window measurements cover.
    #: Zero acts on the first pass that sees load — honest, and aggressive: a spike shorter
    #: than a model download then costs a host that arrives after it is over.
    window_s: float = Field(default=120.0, ge=0)
    #: Each round asks for this many times the last.
    ramp_factor: float = Field(default=2.0, ge=1)
    #: And waits this long after the previous round has landed before growing.
    ramp_backoff_s: float = Field(default=300.0, ge=0)
    #: The most one round may add, however long the load lasts.
    max_round: int = Field(default=8, ge=1)
    #: A warm floor kept while a lease is open. Zero spends nothing without traffic.
    min_hosts: int = Field(default=0, ge=0)


class MachineHistoryConfig(BaseModel):
    """What a machine's record with this pool does to its offer's score (D69).

    Built and judged on its own before any model advises on it: a bounded, deterministic
    adjustment the advisor has to beat before it is worth its non-determinism.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: A machine nobody has tried is not a bad machine. Below this, its record says nothing.
    min_rentals: int = Field(default=2, ge=1)
    #: How often a rental of it must have ended up serving.
    min_reliability: float = Field(default=0.5, ge=0, le=1)
    #: What a good or poor record multiplies the score by.
    bonus: float = Field(default=1.25, ge=1)
    penalty: float = Field(default=0.5, gt=0, le=1)
    #: Measured throughput that counts as good, and as poor. Unset, only reliability is read.
    good_tokens_per_s: Optional[float] = Field(default=None, gt=0)
    poor_tokens_per_s: float = Field(default=0.0, ge=0)


class WorkersAutoConfig(BaseModel):
    """Each rented host finding its own worker count while it serves (D67, D68).

    Off by default. On, a host starts at its capacity profile's number — six where none
    matches — and moves from there on what it measures, never above what its engine was
    launched to run.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: How long a host's behaviour is measured before it is judged on it.
    window_s: float = Field(default=120.0, gt=0)
    #: A step up has to have raised throughput by this much to count as having paid.
    min_gain: float = Field(default=0.10, ge=0)
    #: Service time this far over the pool's median for the same model steps a host down.
    slow_host_factor: float = Field(default=2.0, gt=1)
    #: The most any rented host's engine is launched to run at once. Six is where a host
    #: *starts* (D68); this is the ceiling it may climb to, bounded by what memory allows.
    max: int = Field(default=16, ge=1, le=64)


class CapacityProfile(BaseModel):
    """How many workers a class of hardware runs (docs/spec/hosts-routing-capacity.md §2.1).

    On hosts the pool creates, the engine is launched with the same parallelism, so the number
    is real and not just a count of queue slots. First matching profile wins.
    """

    model_config = ConfigDict(extra="forbid")

    match: CapacityMatch
    max_workers: int = Field(ge=1, le=64)
    #: Why this number: the evidence it rests on. Shown beside it wherever it is used.
    note: Optional[str] = None


class LimitsConfig(BaseModel):
    """Pool-wide rate caps. A lease may tighten these, never loosen them."""

    model_config = ConfigDict(extra="forbid")

    max_rented_hosts: int = Field(default=1, ge=0)
    #: Optional, and unset by default (D46): the cap that matters is per host — every bid is
    #: clamped to `rented.bidding.bid_ceiling` and every offer filtered by `max_all_in_hourly` —
    #: so the pool's hourly spend is already bounded by `max_rented_hosts × bid_ceiling`, and a
    #: second total would only block the host count the operator asked for. Set it to impose an
    #: overall budget below that bound.
    max_hourly_burn: Optional[float] = Field(default=None, gt=0)


class EngineImage(BaseModel):
    """One build of the engine, and the accelerator driver it needs (D92)."""

    model_config = ConfigDict(extra="forbid")

    #: Pinned, never a floating tag (threat model T10).
    image: str
    #: The driver this build needs, as "major" or "major.minor". Compared part by part, so
    #: "580" admits "580.65" and refuses "550.144".
    min_driver: str
    #: Optional: what this build is for, shown beside the choice the pool made.
    note: Optional[str] = None


class RentedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Whether the machines this pool rents put a router in front of several engine processes
    #: (D96). With it, an engine serving one model per process can still hold a whole set on one
    #: machine: each model gets its own process and the router chooses between them by name, so
    #: the pool still dials one URL. Off by default, because it is the machine's start-up that
    #: has to run it — the pool never sends a command.
    #:
    #: It costs what it sounds like it costs: the accelerator's memory is split between the
    #: processes at launch, so the largest model gets a fraction of the cache it would have had
    #: to itself, and cache is where a batching engine's throughput comes from.
    engine_proxy: bool = False
    #: The engine on hosts the pool rents, when it is not the pool's (D93). This is the one that
    #: matters in practice: the machines worth renting run a different engine from the machine
    #: on the operator's desk.
    engine: Optional[str] = None
    #: Which of the pool's models a rented host holds, when the pool spreads its set across
    #: hosts (`pool.models_per_host: declared`, D89). This is where the money decision lives: a
    #: 0.3 GB embedding model does not need the card that was rented for a 26 B one, so a pool
    #: can rent only for the models that justify the price and serve the rest from machines it
    #: already has. Absent there means rented hosts may hold any model the pool still needs.
    models: Optional[list[str]] = None

    @model_validator(mode="after")
    def _profile_exists(self) -> "RentedConfig":
        """A named profile that is not there would silently fall back to the default policy —
        and the operator would watch the pool buy from a market they thought they had left."""
        if self.search_profile and self.search_profile not in self.search_profiles:
            known = ", ".join(sorted(self.search_profiles)) or "none are defined"
            raise ValueError(
                f"search_profile {self.search_profile!r} is not among the search_profiles ({known})"
            )
        return self

    @property
    def policy_in_force(self) -> OfferPolicy:
        """The offer policy this pool is actually searching with (D87)."""
        if self.search_profile:
            return self.search_profiles[self.search_profile]
        return self.offer_policy

    provider: str
    provider_settings: dict[str, Any] = Field(default_factory=dict)
    #: How hosts are rented (D52). `interruptible`: bid, cheaper, can be outbid at any moment.
    #: `on_demand`: pay the listed price, and nobody can take the host away. `cheaper`: look at
    #: both and let ranking decide, which weighs the price against the download it would waste.
    mode: Literal["interruptible", "on_demand", "cheaper"] = "interruptible"
    #: Pinned, never a floating tag (threat model T10).
    image: str = "ollama/ollama:0.34.2"
    #: Builds of the same engine for different accelerator generations, **chosen per machine**
    #: (D92). An engine is commonly published once per CUDA line — a newer one is smaller and
    #: faster but needs a newer driver — and a pool with a single image must either refuse every
    #: older machine or fail on it after paying for it. Given these, the pool takes the **first
    #: whose driver floor the machine meets**, so list them newest first; a machine that meets
    #: none is refused before it is bid on, with the reason.
    #:
    #: This replaces `image` when present. `offer_policy.min_driver_version` still applies and
    #: is the floor below which no machine is wanted at all, whatever image would run on it.
    images: list["EngineImage"] = Field(default_factory=list)
    disk_gb: float = 60.0
    #: Workers **per card** for a machine no capacity profile matches (D107): two cards run
    #: twice the work, as under a profile that names the card (D88), held at 64 per host.
    workers: int = Field(default=1, ge=1)
    capabilities: list[str] = Field(default_factory=list)
    model_set_gb: float = 0.0
    label_prefix: Optional[str] = None
    ssh_key: Optional[str] = None
    ssh_user: str = "root"
    known_hosts: str = "~/.config/gpm/known_hosts"
    #: Where the engine listens inside the instance. Unset means the engine's own default —
    #: 11434 for Ollama, 8000 for vLLM — so a pool need not state a port it cannot choose.
    engine_port: Optional[int] = None
    #: Context length the engine is launched with on hosts the pool creates.
    context_length: int = 8192
    #: Runs after the dead-man timer is armed, for images whose entrypoint the provider's
    #: launch mode does not run. Image-specific, so it lives next to `image`.
    engine_start: Optional[str] = None
    #: Named options of the engine's own start (D100) — `tool_calling`, `reasoning` for vLLM —
    #: from the closed list the engine declares. Names, never flags: the machine's launcher turns
    #: each into the engine's flags for each model by its family. Ignored under `engine_start`,
    #: which replaces the engine's own start, so naming both is refused.
    engine_options: list[str] = Field(default_factory=list)
    #: Put the pool's own agent on hosts it rents (D63): it reports what the machine is really
    #: doing, and later manages its models and worker count. A host with no interpreter gets no
    #: agent and joins without one, so this is safe to leave on.
    agent_on_rented_hosts: bool = True
    #: Where the pool's demand comes from: the lease's worker count, as it always has, or what
    #: the traffic is actually asking for (D66). A dynamic pool still rents nothing without an
    #: open lease and its dollar cap — the lease stops being the demand and becomes the ceiling.
    allocation: Literal["lease", "dynamic"] = "lease"
    dynamic: DynamicAllocationConfig = Field(default_factory=DynamicAllocationConfig)
    #: How many times the pool tries to put its agent on a host before leaving it without
    #: one. SSH answers before a machine has settled, so a first failure is not the last word;
    #: a machine with no interpreter, though, is not going to grow one.
    agent_attempts: int = Field(default=3, ge=1)
    #: What each machine's record with this pool does to its offer's score (D69).
    history: MachineHistoryConfig = Field(default_factory=MachineHistoryConfig)
    #: Each host finds its own worker count while it serves (D67, D68). Off by default.
    workers_auto: WorkersAutoConfig = Field(default_factory=WorkersAutoConfig)
    offer_policy: OfferPolicy = Field(default_factory=OfferPolicy)
    #: Named alternatives to `offer_policy`, chosen by name (D87). One pool wants a
    #: different market on different days — cheap and slow for an overnight batch, fast
    #: and dear for a demo — and rewriting nine filters by hand each time is how a filter
    #: gets left behind. `search_profile` names the one in force; unset means the
    #: `offer_policy` above, so a pool that never names one behaves exactly as before.
    search_profiles: dict[str, OfferPolicy] = Field(default_factory=dict)
    search_profile: Optional[str] = None
    scale: ScaleConfig = Field(default_factory=ScaleConfig)
    bidding: BiddingConfig
    spend: SpendConfig = Field(default_factory=SpendConfig)
    teardown: TeardownConfig = Field(default_factory=TeardownConfig)


class DirectoryConfig(BaseModel):
    """The model directory: what the pool could serve, cached from where models are published
    (D101). Reading it spends nothing and changes nothing; adding a model from it is an edit to
    this file like any other."""

    model_config = ConfigDict(extra="forbid")

    #: How often the supervisor refreshes it on its own. 0: only when the operator asks, from
    #: the console — the pool makes no outbound request nobody asked for.
    refresh_hours: float = Field(default=0.0, ge=0)
    #: Read Ollama's library (its pages; there is no listing to ask for).
    ollama_library: bool = True
    #: Whose vLLM builds a refresh looks up on the model hub: the pool's own models, every size
    #: in Ollama's library, or none. Any model can still be looked up on its own, at any time.
    hub_builds: Literal["pool", "all", "none"] = "pool"
    #: The pace of requests to the hub, a refresh's and the operator's together. One model's
    #: lookup is about fifteen: a search per spelling, then each shown build's file listing for
    #: its size. It is someone else's service.
    hub_requests_per_minute: int = Field(default=60, ge=1, le=6000)
    #: How long a model's looked-up builds are served from the cache before being asked again.
    hub_max_age_hours: float = Field(default=168.0, gt=0)


class PoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool: PoolSettings
    listen: ListenConfig = Field(default_factory=ListenConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    auth: AuthConfig
    engine: str = "ollama"
    catalog: dict[str, CatalogEntry] = Field(default_factory=dict)
    capacity_profiles: list[CapacityProfile] = Field(default_factory=list)
    hosts: list[HostConfig] = Field(default_factory=list)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    #: Absent means the pool cannot rent at all — there is nothing to spend with.
    rented: Optional[RentedConfig] = None
    request_log: str = "gpm.sqlite3"
    directory: DirectoryConfig = Field(default_factory=DirectoryConfig)

    @model_validator(mode="after")
    def _coherent(self) -> "PoolConfig":
        if not self.hosts and self.rented is None:
            raise ValueError(
                "this pool has no hosts and no rented capacity, so it can serve nothing"
            )
        ids = [h.id for h in self.hosts]
        if len(set(ids)) != len(ids):
            raise ValueError("host ids must be unique")
        unknown = set(self.catalog) - set(self.pool.model_set)
        if unknown:
            raise ValueError(f"catalog names not in the pool's model set: {sorted(unknown)}")
        self._engine_can_hold_what_the_pool_asks()
        self._rented_hosts_have_a_build_of_what_they_rent_for()
        self._images_are_built_for_this_engine()
        self._engine_options_are_the_engines()
        self._hosts_can_serve_what_they_are_asked_for()
        self._declared_models_are_in_the_set()
        self._every_model_is_held_by_somebody()
        return self

    def models_held_by(self, host: "HostConfig") -> list[str]:
        """Which of the pool's models this configured host holds (D89).

        With `all`, the whole set — that is what the setting means. With `declared`, what the host
        declared, or the first model in the set it can serve. "First it can serve" is chosen so
        the answer is stable across restarts and explainable in one sentence; a pool that wants
        a different split says so per host.
        """
        if self.pool.models_per_host == "all":
            return list(self.pool.model_set)
        if host.models:
            return list(host.models)
        servable = self._servable_by(host)
        return servable[:1]

    def _servable_by(self, host: "HostConfig") -> list[str]:
        """Models this host could serve: a build its platform can run **and its engine can
        read** (D93). A build named for another engine is not one this host has."""
        caps = set(host.capabilities)
        engine = self.engine_of(host)
        servable = []
        for name in self.pool.model_set:
            entry = self.catalog.get(name)
            if entry is None or any(
                set(v.requires) <= caps and (v.engine is None or v.engine == engine)
                for v in entry.variants
            ):
                servable.append(name)
        return servable

    def _declared_models_are_in_the_set(self) -> None:
        """A declared model the pool does not serve is a typo, and a silent one: the host would
        simply never be eligible for anything, with nothing said about why."""
        known = set(self.pool.model_set)
        spread = self.pool.models_per_host == "declared"
        for host in self.hosts:
            if host.models is None:
                continue
            if not spread:
                raise ValueError(
                    f"host {host.id!r} names the models it holds, but pool.models_per_host is "
                    f"'all', which means every host holds the whole set. Remove the host's "
                    f"`models`, or set pool.models_per_host to 'declared'."
                )
            if not host.models:
                raise ValueError(f"host {host.id!r} declares an empty `models`: it could serve nothing")
            unknown = set(host.models) - known
            if unknown:
                raise ValueError(
                    f"host {host.id!r} names models that are not in the pool's set: {sorted(unknown)}"
                )
            cannot = set(host.models) - set(self._servable_by(host))
            if cannot:
                raise ValueError(
                    f"host {host.id!r} is asked to hold {sorted(cannot)}, but its capabilities "
                    f"{sorted(set(host.capabilities))} meet no variant's requirements for them"
                )
        if self.rented is not None and self.rented.models is not None:
            if not spread:
                raise ValueError(
                    "rented.models names the models rented hosts hold, but pool.models_per_host "
                    "is 'all', which means every host holds the whole set"
                )
            unknown = set(self.rented.models) - known
            if unknown:
                raise ValueError(f"rented.models names models not in the pool's set: {sorted(unknown)}")

    def _every_model_is_held_by_somebody(self) -> None:
        """With the set spread across hosts, a model nobody holds can never be served — and the
        only symptom would be a 503 for that model alone, long after the pool looked healthy.

        Renting counts as holding only when the pool may actually rent for that model: a lease
        is what decides whether renting happens at all, but a configuration that could never
        cover a model however many hosts it bought is wrong on its face.
        """
        if self.pool.models_per_host != "declared":
            return
        covered: set[str] = set()
        for host in self.hosts:
            if not host.disabled:
                covered.update(self.models_held_by(host))
        if self.rented is not None:
            covered.update(self.rented.models if self.rented.models is not None else self.pool.model_set)
        missing = [name for name in self.pool.model_set if name not in covered]
        if missing:
            raise ValueError(
                f"with pool.models_per_host 'declared', the pool's hosts must between them hold every "
                f"model in its set, and {missing} would be held by none. Name them on a host's "
                f"`models`, add a host that holds them, or let rented hosts hold them."
            )

    def _engine_can_hold_what_the_pool_asks(self) -> None:
        """No host may be asked for more models than its engine can hold at once (D89, D94).

        An engine that serves one model per process holds exactly one. What each host is *asked*
        for depends on the placement: with `all` that is the pool's whole set, with `declared` it
        is what that host declares — and a host declaring two is over the line just as surely.

        Refused here, at load, rather than after a machine has been rented that can never reach
        `ready` and is paid for until the give-up window closes.
        """
        from .engines import EngineNotFound, get_engine

        def holds_one(name: str, *, behind_proxy: bool = False) -> bool:
            if behind_proxy:
                # Several processes behind a router on the machine: one URL, many models (D96).
                return False
            try:
                return bool(getattr(get_engine(name), "serves_one_model", False))
            except EngineNotFound:
                return False  # naming an uninstalled engine is its own error, reported elsewhere

        for host in self.hosts:
            if host.disabled:
                continue
            asked = self.models_held_by(host)
            engine = self.engine_of(host)
            if len(asked) > 1 and holds_one(engine):
                raise ValueError(
                    f"host {host.id!r} runs {engine!r}, which serves one model per process, but "
                    f"is asked to hold {len(asked)}: {asked}. Give it a single model in its "
                    f"`models`, or run an engine that holds several at once."
                )

        # Rented hosts differ from configured ones: with the set spread across hosts,
        # `rented.models` is the set the pool may rent **for**, and each host it buys is given
        # one of them (D94). Only `all` asks a single rented machine for the lot.
        if self.rented is not None and self.pool.models_per_host == "all":
            asked = list(self.pool.model_set)
            engine = self.rented_engine()
            if len(asked) > 1 and holds_one(engine, behind_proxy=self.rented.engine_proxy):
                raise ValueError(
                    f"rented hosts run {engine!r}, which serves one model per process, but with "
                    f"pool.models_per_host 'all' every one of them is asked to hold "
                    f"{len(asked)} models ({asked}). Either set pool.models_per_host to "
                    f"'declared', and the pool will buy a host per model; or set "
                    f"rented.engine_proxy and have the machine's own start-up run one engine "
                    f"per model behind the router the agent ships (D96) — which splits the "
                    f"accelerator's memory between them."
                )

    def _rented_hosts_have_a_build_of_what_they_rent_for(self) -> None:
        """Every model the pool may rent a host for needs a build that host's engine can serve.

        Otherwise the host is prepared for nothing and called ready holding nothing — no error,
        no event, just a model that is never served however many machines are bought for it.
        The same model is a plain tag to one engine and a model-hub repository to another, so a
        pool that changes its rented engine has to say what the new one should fetch (D98).
        """
        if self.rented is None:
            return
        engine = self.rented_engine()
        caps = set(self.rented.capabilities)
        missing = []
        for name in self._rented_models():
            entry = self.catalog.get(name)
            if entry is None:
                continue  # served under its own name, by any engine
            if not any(
                set(v.requires) <= caps and (v.engine is None or v.engine == engine)
                for v in entry.variants
            ):
                missing.append(name)
        if missing:
            raise ValueError(
                f"rented hosts run {engine!r}, and the catalog has no build of {missing} for it "
                f"(with capabilities {sorted(caps)}). Add a variant with `engine: {engine}` — for "
                f"vLLM, the model's repository on the hub — or stop renting for those models."
            )

    def _rented_models(self) -> list[str]:
        """What a rented host is asked to hold: the pool's whole set, or what `rented.models`
        names where the set is spread across hosts."""
        if self.pool.models_per_host == "all" or self.rented is None or self.rented.models is None:
            return list(self.pool.model_set)
        return list(self.rented.models)

    def _engines_and_where(self) -> list[tuple[str, str]]:
        """Each engine this pool runs, with something an operator can go and look at."""
        found = [(self.engine, "this pool")]
        for host in self.hosts:
            if host.engine:
                found.append((host.engine, f"host {host.id!r}"))
        if self.rented is not None and self.rented.engine:
            found.append((self.rented.engine, "rented hosts"))
        return found

    def engine_of(self, host: "HostConfig") -> str:
        """Which engine this host runs (D93): its own, or the pool's where it says nothing."""
        return host.engine or self.engine

    def rented_engine(self) -> str:
        """Which engine hosts the pool rents run (D93)."""
        if self.rented is not None and self.rented.engine:
            return self.rented.engine
        return self.engine

    def engines_in_use(self) -> list[str]:
        """Every engine this pool runs, the pool's default first.

        The router needs all of them: it must know which paths its hosts serve between them, and
        which host can serve the path a request actually arrived on.
        """
        found = [self.engine]
        for host in self.hosts:
            if host.engine and host.engine not in found:
                found.append(host.engine)
        rented = self.rented_engine()
        if rented not in found:
            found.append(rented)
        return found

    def engine_port(self) -> int:
        """Where the engine listens on a host the pool creates (D92).

        Stated wins; otherwise the engine's own default. A pool that changed engine and kept the
        previous engine's port would dial a closed door on every host it rented.
        """
        if self.rented is not None and self.rented.engine_port is not None:
            return self.rented.engine_port
        from .engines import EngineNotFound, get_engine

        try:
            return get_engine(self.rented_engine()).default_port or 11434
        except EngineNotFound:
            return 11434

    def _engine_options_are_the_engines(self) -> None:
        """Only names the rented engine's own start offers (D100), and only with that start.

        An unknown name would reach the machine and stop its start; one written beside an
        `engine_start` would be silently ignored, and the operator would believe tool calling
        was on when nothing had asked for it.
        """
        if self.rented is None or not self.rented.engine_options:
            return
        from .engines import EngineNotFound, get_engine

        try:
            offered = get_engine(self.rented_engine()).options
        except EngineNotFound:
            return
        unknown = [o for o in self.rented.engine_options if o not in offered]
        if unknown:
            raise ValueError(
                f"rented.engine_options names {unknown}, which {self.rented_engine()!r} does not "
                f"offer; it offers {sorted(offered) or 'none'}"
            )
        if self.rented.engine_start:
            raise ValueError(
                "rented.engine_options apply to the engine's own start, and rented.engine_start "
                "replaces it; remove one of them"
            )

    def _images_are_built_for_this_engine(self) -> None:
        """Refuse an image built for a *different* engine than the one configured (D92).

        A pool set to vLLM with an Ollama image rents a machine, starts the wrong server, never
        reaches `ready`, and pays until the give-up window closes — and nothing before this said
        anything was wrong. Only a clash is refused: an image matching no known engine is left
        alone, because a private build may be called anything.
        """
        if self.rented is None:
            return
        from .engines import EngineNotFound, available_engines, get_engine

        try:
            mine = get_engine(self.rented_engine())
        except EngineNotFound:
            return
        others = {}
        for name in available_engines():
            if name == self.rented_engine():
                continue
            try:
                others[name] = get_engine(name).image_words
            except EngineNotFound:
                continue
        # A start command written for another engine is the same mistake by another route: it
        # runs the wrong server inside the right image (D97). A pool that switched its rented
        # hosts to vLLM and kept `nohup ollama serve` would do exactly that.
        start = (self.rented.engine_start or "").lower()
        if start and not any(word in start for word in mine.image_words):
            for other, words in others.items():
                if any(word in start for word in words):
                    raise ValueError(
                        f"rented hosts run {self.rented_engine()!r}, but rented.engine_start "
                        f"starts {other!r}. Remove engine_start to use {self.rented_engine()!r}'s "
                        f"own start, or write one for it."
                    )

        # `images` replaces `image`, so only what would actually be used is checked — the
        # unused default must not refuse a configuration that never names it.
        named = [i.image for i in self.rented.images] or [self.rented.image]
        for image in named:
            lowered = image.lower()
            if any(word in lowered for word in mine.image_words):
                continue
            for other, words in others.items():
                if any(word in lowered for word in words):
                    raise ValueError(
                        f"rented hosts run {self.rented_engine()!r} but the image {image!r} is built for "
                        f"{other!r}. A host rented from it would start the wrong server, never "
                        f"answer, and be paid for until it was given up. Name an image for "
                        f"{self.rented_engine()!r}, or change the engine."
                    )

    def _hosts_can_serve_what_they_are_asked_for(self) -> None:
        """Every configured host must be able to serve something the pool needs (spec §3).

        With `models_per_host: all` that means a usable variant of *every* model, because the
        host is asked to hold the whole set. With `declared` it means a usable variant of *at least
        one* model — a host that can serve nothing in the set is still a mistake worth
        refusing, but one that can serve only the embedding model is now perfectly good.
        """
        whole_set = self.pool.models_per_host == "all"
        for host in self.hosts:
            caps = set(host.capabilities)
            servable_here = set(self._servable_by(host))
            servable = []
            for name in self.pool.model_set:
                if name in servable_here:
                    servable.append(name)
                elif whole_set:
                    raise ValueError(
                        f"host {host.id!r} has no usable variant of {name!r}: it runs "
                        f"{self.engine_of(host)!r} with capabilities {sorted(caps)}, and no "
                        f"variant matches both. Give it the capability, add a variant for that "
                        f"engine, add a fallback with no requirements, or remove the host."
                    )
            if not servable:
                raise ValueError(
                    f"host {host.id!r} has no usable variant of any model in the pool's set: its "
                    f"capabilities {sorted(caps)} meet no variant's requirements. Give it the "
                    f"capability, add a fallback variant with no requirements, or remove the host."
                )


def load_config(path: str | Path) -> PoolConfig:
    """Read, parse and validate. Raises ConfigError with a message meant for an operator."""
    path = Path(path)
    try:
        raw: Any = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"no configuration file at {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    try:
        return PoolConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError, or a ValueError from a validator
        problems = "\n".join(f"  - {line}" for line in operator_lines(exc))
        raise ConfigError(f"{path} is not a usable pool configuration:\n{problems}") from exc


def operator_lines(exc: Exception) -> list[str]:
    """Each problem as one sentence an operator can act on, and where in the file it is.

    Pydantic's own text is written for the programmer: an error count, a type code, the whole
    offending input echoed back, and a link to its documentation — with the one sentence that
    matters somewhere in the middle. Seen in the console, where a refused engine switch showed
    exactly that.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return [str(exc)]
    lines = []
    for error in errors():
        message = str(error.get("msg", "")).removeprefix("Value error, ")
        where = ".".join(str(part) for part in error.get("loc", ()) if part != "__root__")
        lines.append(f"{where}: {message}" if where else message)
    return lines or [str(exc)]
