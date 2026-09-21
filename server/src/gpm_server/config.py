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
    residency: Literal["pinned", "on_demand"] = "pinned"
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
    max_download_per_gb: Optional[float] = None
    min_download_mbps: float = 0.0
    min_reliability: float = 0.0
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
    min_gpu_memory_gb: Optional[float] = None
    capability: Optional[str] = None


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


class RentedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    provider_settings: dict[str, Any] = Field(default_factory=dict)
    #: How hosts are rented (D52). `interruptible`: bid, cheaper, can be outbid at any moment.
    #: `on_demand`: pay the listed price, and nobody can take the host away. `cheaper`: look at
    #: both and let ranking decide, which weighs the price against the download it would waste.
    mode: Literal["interruptible", "on_demand", "cheaper"] = "interruptible"
    #: Pinned, never a floating tag (threat model T10).
    image: str = "ollama/ollama:0.34.2"
    disk_gb: float = 60.0
    workers: int = Field(default=1, ge=1)
    capabilities: list[str] = Field(default_factory=list)
    model_set_gb: float = 0.0
    label_prefix: Optional[str] = None
    ssh_key: Optional[str] = None
    ssh_user: str = "root"
    known_hosts: str = "~/.config/gpm/known_hosts"
    #: Where the engine listens inside the instance. Not guessed: state it.
    engine_port: int = 11434
    #: Context length the engine is launched with on hosts the pool creates.
    context_length: int = 8192
    #: Runs after the dead-man timer is armed, for images whose entrypoint the provider's
    #: launch mode does not run. Image-specific, so it lives next to `image`.
    engine_start: Optional[str] = None
    #: Put the pool's own agent on hosts it rents (D63): it reports what the machine is really
    #: doing, and later manages its models and worker count. A host with no interpreter gets no
    #: agent and joins without one, so this is safe to leave on.
    agent_on_rented_hosts: bool = True
    #: Each host finds its own worker count while it serves (D67, D68). Off by default.
    workers_auto: WorkersAutoConfig = Field(default_factory=WorkersAutoConfig)
    offer_policy: OfferPolicy = Field(default_factory=OfferPolicy)
    scale: ScaleConfig = Field(default_factory=ScaleConfig)
    bidding: BiddingConfig
    spend: SpendConfig = Field(default_factory=SpendConfig)
    teardown: TeardownConfig = Field(default_factory=TeardownConfig)


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
        # Every host serves the pool's whole model set, so every host must have at least one
        # usable variant of every model in it (spec §3).
        for host in self.hosts:
            caps = set(host.capabilities)
            for name in self.pool.model_set:
                entry = self.catalog.get(name)
                if entry is None:
                    continue
                if not any(set(v.requires) <= caps for v in entry.variants):
                    raise ValueError(
                        f"host {host.id!r} has no usable variant of {name!r}: its capabilities "
                        f"{sorted(caps)} meet no variant's requirements. Give it the capability, "
                        f"add a fallback variant with no requirements, or remove the host."
                    )
        return self


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
        raise ConfigError(f"{path} is not a usable pool configuration:\n{exc}") from exc
