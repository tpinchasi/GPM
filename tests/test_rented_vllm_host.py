"""Rented vLLM hosts, from an empty disk to serving, with every step in between (D97).

Every piece of this had its own tests, and the path as a whole had none — which is how five
faults survived to the point where the owner asked how to switch his pool to vLLM:

1. the agent would not fetch anything while the engine was not answering, and vLLM cannot
   answer until the weights are there;
2. the pool would not install the agent until the engine answered — the same wait, from the
   other side — and would have given the machine up as never having started;
3. a model downloaded and waiting for its engine was reported as a failure, and the pool
   destroyed the host for it;
4. nothing ever started the engine again once the weights were down;
5. with a model to a host, a host was only called ready when it held *every* rented model — so
   none ever would be.

Each one alone meant a rented machine downloading its model, never serving, and being paid for
until it was given up. So these drive the whole sequence: the real supervisor and router, the
real agent application, the real hub fetch over real HTTP, the real launcher choosing what to
start — with fake vLLM engines behind it and a fake hub in front of it. Two things are replaced:
the SSH install (each host is handed the agent it would have installed) and the starting of an
operating-system process (the launcher's process starter starts the fake engine instead).

Both of the owner's placements are driven: a model to a host, and every model on one host
behind the router.
"""

import time
from pathlib import Path

import httpx
from fakes.fake_vllm import FakeHub
from fakes.harness import EngineSpec, ServerHandle, pool_harness
from gpm_agent import engine_control, vllm_launch
from gpm_agent.app import create_app
from gpm_agent.settings import Settings, fingerprint
from gpm_server.providers import default_offer
from gpm_server.supervisor import hostagent

BIG = "gemma4:26b"
EMBED = "an-embed-model"
BIG_REPO = "nvidia/Gemma-4-26B-A4B-NVFP4"
EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5"
AGENT_KEY = "gpmg_" + "b" * 64

HUB = {
    BIG_REPO: {
        "config.json": b'{"architectures": ["Gemma4ForCausalLM"]}',
        "model-00001-of-00002.safetensors": b"w" * 40_000,
        "model-00002-of-00002.safetensors": b"w" * 30_000,
        "tokenizer.json": b"{}",
        "README.md": b"not weights",
    },
    EMBED_REPO: {
        "config.json": b'{"architectures": ["NomicBertModel"]}',
        "model.safetensors": b"e" * 3_000,
    },
}

CATALOG = {
    BIG: {"variants": [{"tag": "gemma4:26b", "engine": "ollama"}, {"tag": BIG_REPO, "engine": "vllm"}]},
    EMBED: {"variants": [{"tag": EMBED, "engine": "ollama"}, {"tag": EMBED_REPO, "engine": "vllm"}]},
}


def rented(**overrides):
    base = {
        "provider": "fake", "engine": "vllm", "image": "vastai/vllm:v0.29.0-cuda-12.9",
        "workers": 2, "model_set_gb": 1.0,
        "bidding": {"bid_ceiling": 0.60, "premium": 0.02},
        "scale": {"scale_up_after_s": 0},
        "teardown": {"idle_minutes": 10},
    }
    base.update(overrides)
    return base


class Machines:
    """The rented machines' side: one agent per machine, each on its own disk, each in front of
    its own fake vLLM — reached by the pool as it would reach them, one agent URL per host."""

    def __init__(self, pool, tmp_path: Path, monkeypatch):
        self.pool = pool
        self.fleet = pool.supervisor.fleet
        self.fleet.agent_hold_retry_s = 0.01
        self.agents: dict[str, object] = {}          # agent hostname -> agent app
        self.disks: dict[str, Path] = {}             # agent hostname -> its models directory
        self.engines: dict[str, object] = {}         # agent hostname -> its fake vLLM
        self.launches: list[tuple[str, list[str], list[str], bool]] = []
        by_url = {engine.server.base_url.rstrip("/"): engine for engine in pool.rentable.values()}

        async def by_hostname(scope, receive, send):
            name = dict(scope["headers"]).get(b"host", b"").decode().split(":")[0]
            await self.agents[name](scope, receive, send)

        self.fleet._agent_transport = httpx.ASGITransport(app=by_hostname)

        async def installed(host):
            if host.agent is not None:
                return
            engine = by_url[host.dial_url.rstrip("/")]
            name = f"agent-{engine.spec.id}"
            disk = tmp_path / engine.spec.id / "models"
            self.disks[name], self.engines[name] = disk, engine.fake
            self.agents[name] = create_app(
                # What the pool tells a vLLM host's agent: its engine, where its models go, and
                # the restart script the machine was booted with.
                Settings(
                    key_hash=fingerprint(AGENT_KEY), engine="vllm", engine_url="http://engine",
                    models_path=str(disk), restart_command=["/var/run/gpm/restart-engine.sh"],
                    engine_env_file=str(tmp_path / engine.spec.id / "engine.env"),
                    restart_timeout_s=10,
                ),
                engine_transport=httpx.ASGITransport(app=engine.fake.app),
            )
            host.agent = hostagent.RentedAgent(url=f"http://{name}", secret=AGENT_KEY)

        monkeypatch.setattr(self.fleet, "install_agent", installed)

        machines = self

        # The machine's restart script runs the agent's real launcher, with the `--proxy` the
        # pool's own start command carries; only the operating-system processes are replaced.
        async def restart(control):
            name = next(n for n, disk in machines.disks.items() if str(disk) == control.settings.models_path)
            disk, engine = machines.disks[name], machines.engines[name]
            proxy = "--proxy" in (machines.fleet.engine_start_command() or "")
            on_disk = sorted(p.name for p in vllm_launch.complete_models(disk))
            names: list[str] = []

            def popen(argv, **kwargs):
                if "--served-model-name" in argv:
                    names.append(argv[argv.index("--served-model-name") + 1])
                return type("Process", (), {"pid": None})()

            vllm_launch.launch(
                disk, 8000, proxy=proxy, popen=popen, alive=lambda pid: False,
                env=engine_control.read_applied(control.settings.engine_env_file) or {},
                agent="/var/run/gpm/gpm-agent.pyz",
            )
            # One port in front of whatever was started — the router's job when there are several.
            engine.start(names)
            machines.launches.append((name, on_disk, names, proxy))
            return 0, "started"

        monkeypatch.setattr(engine_control.EngineControl, "_restart", restart)

        self.raised: list[str] = []
        real_pass = pool.supervisor.pass_once

        async def watched():
            try:
                await real_pass()
            except Exception as exc:  # noqa: BLE001 - the point is to see it, not to hide it
                self.raised.append(repr(exc))
                raise

        monkeypatch.setattr(pool.supervisor, "pass_once", watched)

    def until_ready(self, count: int, within_s: float = 40.0):
        deadline = time.monotonic() + within_s
        while time.monotonic() < deadline:
            self.pool.reprobe()
            live = [h for h in self.fleet.hosts.values() if not h.released]
            if len(live) >= count and all(h.state == "ready" for h in live):
                return live
            time.sleep(0.1)
        states = {h.host_id: (h.state, h.models) for h in self.fleet.hosts.values()}
        recent = [e["summary"] for e in self.pool.supervisor.events.recent(20)]
        raise AssertionError(f"not {count} ready within {within_s}s: {states}; recent: {recent}")

    def check_no_failures(self):
        kinds = [e["kind"] for e in self.pool.supervisor.events.recent(limit=300)]
        assert not self.raised, f"a control-loop pass raised: {self.raised[:2]}"
        assert "prepare_failed" not in kinds, [
            e["summary"] for e in self.pool.supervisor.events.recent(50) if e["kind"] == "prepare_failed"
        ]
        return kinds


# --- a model to a host (D94) ---


def test_a_model_to_a_host_downloads_starts_and_serves(monkeypatch, tmp_path):
    hub = FakeHub(HUB)
    with pool_harness(
        [EngineSpec(id="laptop", resident={EMBED}, kind="local", workers=1)],
        host_overrides={"laptop": {"models": [EMBED]}},
        rentable=[EngineSpec(id="market-1", resident=set(), workers=2, engine="vllm")],
        model_set=[BIG, EMBED], catalog=CATALOG,
        rented=rented(models=[BIG]),
        pool_settings={"models_per_host": "declared"},
    ) as pool:
        hub_server = ServerHandle(hub.app, pool.loop)
        monkeypatch.setenv("HF_ENDPOINT", hub_server.base_url)
        try:
            machines = Machines(pool, tmp_path, monkeypatch)
            machines.fleet.open_lease(workers=2, max_hours=2, max_spend=2.00, allow_rent=True)
            (host,) = machines.until_ready(1)
            kinds = machines.check_no_failures()

            # Bought for the one model; fetched from the hub while the engine was not running.
            assert host.models == (BIG,)
            assert f"list {BIG_REPO}" in hub.requests
            assert f"get {BIG_REPO}/README.md" not in hub.requests, "a readme is not weights"

            # Into the directory the engine's start reads, complete — and started once, after the
            # download had finished, on exactly that model, with the pool saying why.
            disk = machines.disks["agent-market-1"]
            assert (disk / "nvidia__Gemma-4-26B-A4B-NVFP4" / "model-00001-of-00002.safetensors").stat().st_size == 40_000
            assert machines.launches == [("agent-market-1", ["nvidia__Gemma-4-26B-A4B-NVFP4"], [BIG_REPO], False)]
            vllm = pool.rentable["market-1"].fake
            assert vllm.starts == [{BIG_REPO}]
            assert kinds.count("engine_restart") == 1

            # More passes do not start it again: asking every pass would only interrupt it.
            for _ in range(3):
                pool.reprobe()
            assert len(vllm.starts) == 1

            with pool.client() as client:
                # It serves, under the build that engine knows the pool's model by.
                answer = client.post("/v1/chat/completions", json={
                    "model": BIG, "messages": [{"role": "user", "content": "hello"}],
                })
                assert answer.status_code == 200, answer.text
                assert answer.headers["X-GPM-Host"] == host.host_id
                assert answer.json()["choices"][0]["message"]["content"] == "served by vllm"
                assert vllm.received[-1][1]["model"] == BIG_REPO

                # The laptop still serves what it holds, on the same pool, through the same API.
                embedded = client.post("/v1/embeddings", json={"model": EMBED, "input": ["x"]})
                assert embedded.status_code == 200, embedded.text
                assert embedded.headers["X-GPM-Host"] == "laptop"

                # Ollama's own API for the big model has nowhere to go: the laptop does not hold
                # it and the vLLM host does not serve that path (D93). Refused, not misrouted.
                native = client.post("/api/chat", json={
                    "model": BIG, "messages": [{"role": "user", "content": "hi"}], "stream": False,
                })
                assert native.status_code == 503, native.text
                assert all(path.startswith("/v1/") for path, _ in vllm.received)
        finally:
            hub_server.stop()


def test_two_models_are_bought_a_host_each_and_each_host_is_ready_on_its_own(monkeypatch, tmp_path):
    """With a model to a host, no host holds the whole rented set. Judging each by it — as the
    readiness check did — would leave every one of them preparing for ever."""
    hub = FakeHub(HUB)
    with pool_harness(
        [EngineSpec(id="laptop", resident={EMBED}, kind="local", workers=1)],
        host_overrides={"laptop": {"models": [EMBED]}},
        rentable=[
            EngineSpec(id="market-1", resident=set(), workers=2, engine="vllm"),
            EngineSpec(id="market-2", resident=set(), workers=2, engine="vllm"),
        ],
        model_set=[BIG, EMBED], catalog=CATALOG,
        rented=rented(models=[BIG, EMBED]),
        pool_settings={"models_per_host": "declared"},
        extra_config={"limits": {"max_rented_hosts": 2}},
    ) as pool:
        hub_server = ServerHandle(hub.app, pool.loop)
        monkeypatch.setenv("HF_ENDPOINT", hub_server.base_url)
        try:
            machines = Machines(pool, tmp_path, monkeypatch)
            # Two machines on the market: the pool never bids again on one it already rents (D59).
            machines.fleet.provider.offers = [default_offer("o-1", "m-1"), default_offer("o-2", "m-2")]
            machines.fleet.open_lease(workers=4, max_hours=2, max_spend=4.00, allow_rent=True)
            hosts = machines.until_ready(2, within_s=60)
            machines.check_no_failures()

            assert sorted(h.models for h in hosts) == sorted([(BIG,), (EMBED,)]), "a different model each"
            served = sorted(tuple(sorted(e.fake.starts[-1])) for e in pool.rentable.values())
            assert served == sorted([(BIG_REPO,), (EMBED_REPO,)]), "each engine started on its own model"
            # And each fetched only its own: nothing downloaded twice, nothing downloaded for nothing.
            for name, disk in machines.disks.items():
                assert len(vllm_launch.complete_models(disk)) == 1, name
        finally:
            hub_server.stop()


# --- every model on one host, behind the router (D96) ---


def test_every_model_on_one_host_starts_an_engine_each_behind_the_router(monkeypatch, tmp_path):
    hub = FakeHub(HUB)
    with pool_harness(
        [EngineSpec(id="laptop", resident={"gemma4:26b", EMBED}, kind="local", workers=1)],
        rentable=[EngineSpec(id="market-1", resident=set(), workers=2, engine="vllm")],
        model_set=[BIG, EMBED], catalog=CATALOG,
        rented=rented(engine_proxy=True),
    ) as pool:
        hub_server = ServerHandle(hub.app, pool.loop)
        monkeypatch.setenv("HF_ENDPOINT", hub_server.base_url)
        try:
            machines = Machines(pool, tmp_path, monkeypatch)
            machines.fleet.open_lease(workers=2, max_hours=2, max_spend=2.00, allow_rent=True)
            (host,) = machines.until_ready(1)
            machines.check_no_failures()

            # Both fetched, then one start — after both had landed — with the router, one engine
            # per model, memory split between them.
            ((name, on_disk, started, proxy),) = machines.launches
            assert proxy is True
            assert on_disk == sorted(["nvidia__Gemma-4-26B-A4B-NVFP4", "nomic-ai__nomic-embed-text-v1.5"])
            assert sorted(started) == sorted([BIG_REPO, EMBED_REPO])
            assert (machines.disks[name] / vllm_launch.UPSTREAMS_FILE).exists(), "the router's map"

            with pool.client() as client:
                for model in (BIG, EMBED):
                    path = "/v1/embeddings" if model == EMBED else "/v1/chat/completions"
                    body = {"model": model, "input": ["x"]} if model == EMBED else {
                        "model": model, "messages": [{"role": "user", "content": "hi"}]}
                    answer = client.post(path, json=body, headers={"X-GPM-Runtime-Class": "unknown-vllm"})
                    assert answer.status_code == 200, (model, answer.text)
                    assert answer.headers["X-GPM-Host"] == host.host_id
        finally:
            hub_server.stop()
