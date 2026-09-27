"""Model profiles: what one rented machine holds, the build of each model, and the least card
and disk that needs (D111).

The owner found the old selector wrong in three ways: under "one model per host" several models
could be ticked (they were the set the pool might rent *for*, which the screen never said); under
"every model" there was no choosing a few; and the card and disk a machine needed were typed by
hand, beside sizes the pool already knew. A profile is a named set of models with the build of
each; the pool rents hosts *as* profiles; and what a profile holds sets the search's minimums.

Against the fake provider and the fake hub; nothing here spends money.
"""

import math
import textwrap

import httpx
import pytest
import yaml
from fakes.harness import BackgroundLoop, ServerHandle
from gpm_agent import vllm_launch
from gpm_server import sizing
from gpm_server.config import PoolConfig, load_config
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor import Supervisor
from gpm_server.supervisor.control import create_control_app
from gpm_server.supervisor.renting import Fleet

BIG, SMALL, EMBED = "big:26b", "small:4b", "embed:v1"
BIG_FP4, BIG_FP8 = "org/Big-26B-NVFP4", "org/Big-26B-FP8"
SMALL_REPO, EMBED_REPO = "org/Small-4B", "org/Embed-v1"

CATALOG = {
    BIG: {"variants": [
        {"tag": "big:26b", "engine": "ollama"},
        {"tag": BIG_FP4, "engine": "vllm", "size_gb": 17.0},
        {"tag": BIG_FP8, "engine": "vllm", "size_gb": 28.0},
    ]},
    SMALL: {"variants": [{"tag": "small:4b", "engine": "ollama"}, {"tag": SMALL_REPO, "engine": "vllm", "size_gb": 8.0}]},
    EMBED: {"variants": [{"tag": "embed:v1", "engine": "ollama"}, {"tag": EMBED_REPO, "engine": "vllm", "size_gb": 0.5}]},
}

PROFILES = {
    "chat": {BIG: BIG_FP4},
    "chat-fp8": {BIG: BIG_FP8},
    "small-and-embed": {SMALL: SMALL_REPO, EMBED: EMBED_REPO},
}


def config(rent=("chat",), profiles=None, hosts=None, **rented_overrides):
    rented = {
        "provider": "fake", "engine": "vllm", "image": "vastai/vllm:v0.29.0-cuda-12.9", "workers": 2,
        "bidding": {"premium": 0.02},
        "offer_policy": {"min_disk_gb": 20, "max_all_in_hourly": 0.60},
        "scale": {"scale_up_after_s": 0},
        "model_profiles": PROFILES if profiles is None else profiles,
        "rent_profiles": list(rent),
    }
    rented.update(rented_overrides)
    return PoolConfig.model_validate({
        "pool": {"name": "test", "model_set": [BIG, SMALL, EMBED], "models_per_host": "declared"},
        "auth": {"app_keys": ["k"]},
        "catalog": CATALOG,
        "hosts": hosts if hosts is not None else [
            {"id": "laptop", "kind": "local", "workers": 1, "models": [SMALL, EMBED],
             "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}},
        ],
        "rented": rented,
    })


@pytest.fixture
def make_fleet(tmp_path):
    databases = []

    def build(cfg, offers=None):
        database = Database(tmp_path / f"gpm{len(databases)}.sqlite3")
        databases.append(database)
        provider = FakeProvider(offers=offers) if offers is not None else FakeProvider()
        return Fleet(cfg, cfg.rented, provider, LeaseStore(database), EventLog(database), SpendLedger(database))

    yield build
    for database in databases:
        database.close()


def open_lease(fleet):
    return fleet.leases.open(workers=2, max_hours=4, max_spend=5.0, allow_rent=True,
                             pool_max_all_in_hourly=fleet.rented.max_all_in_hourly)


def two_machines():
    return [default_offer("o-1", "m-1"), default_offer("o-2", "m-2")]


def created(fleet):
    return list(fleet.provider.instances.values())


# --- the file's rules ---


def test_a_rented_profile_that_is_not_defined_is_refused():
    with pytest.raises(ValueError, match="not among the model_profiles"):
        config(rent=("nope",))


def test_a_build_the_catalog_does_not_list_is_refused():
    with pytest.raises(ValueError, match="does not list for it"):
        config(profiles={"chat": {BIG: "someone/Else-26B"}})


def test_a_build_rented_hosts_cannot_run_is_refused():
    with pytest.raises(ValueError, match="rented hosts cannot serve"):
        config(profiles={"chat": {BIG: "big:26b"}})  # Ollama's tag, on a vLLM host


def test_a_profile_naming_a_model_outside_the_set_is_refused():
    with pytest.raises(ValueError, match="not in the pool's set"):
        config(profiles={"chat": {"other:1b": "other:1b"}})


def test_profiles_and_rented_models_together_are_refused():
    with pytest.raises(ValueError, match="rent_profiles and remove models"):
        config(models=[BIG])


def test_a_model_no_host_and_no_rented_profile_holds_is_refused():
    with pytest.raises(ValueError, match=r"\['big:26b'\] would be held by none"):
        config(rent=("small-and-embed",), hosts=[
            {"id": "laptop", "kind": "local", "workers": 1, "models": [SMALL],
             "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}},
        ])


def test_a_profile_of_several_models_needs_no_pool_wide_switch():
    """Under vLLM a machine holding two models runs the router in front of two processes (D96);
    a profile of two says so by being two, rather than by a switch for the whole pool."""
    cfg = config(rent=("chat", "small-and-embed"))
    assert cfg.rented.engine_proxy is False


# --- which profile the next machine is bought as ---


async def test_each_profile_is_bought_once_before_any_is_bought_twice(make_fleet):
    fleet = make_fleet(config(rent=("chat", "small-and-embed"), hosts=[]), offers=two_machines())
    lease = open_lease(fleet)
    assert fleet.profile_for_new_host() == "chat"
    assert await fleet.rent_one(lease, []) is not None
    # The small and embedding models are served by nobody yet: availability first.
    assert fleet.profile_for_new_host() == "small-and-embed"
    assert await fleet.rent_one(lease, []) is not None
    hosts = sorted(fleet.hosts.values(), key=lambda h: h.created_at)
    assert [h.profile for h in hosts] == ["chat", "small-and-embed"]
    assert hosts[1].models == (SMALL, EMBED)


async def test_waiting_requests_decide_between_covered_profiles(make_fleet):
    fleet = make_fleet(config(rent=("chat", "small-and-embed"), hosts=[]), offers=two_machines())
    lease = open_lease(fleet)
    for _ in range(2):
        await fleet.rent_one(lease, [])
    fleet._waiting_by_model = {EMBED: 3}
    assert fleet.profile_for_new_host() == "small-and-embed"


# --- the build a profile names is the build the machine holds ---


async def test_a_host_bought_as_a_profile_holds_exactly_its_builds(make_fleet):
    """The FP8 build is the catalog's second for vLLM: resolved by catalog order, the machine
    would have fetched the FP4 one."""
    fleet = make_fleet(config(rent=("chat-fp8",)))
    lease = open_lease(fleet)
    host = await fleet.rent_one(lease, [])
    assert host.profile == "chat-fp8"
    assert host.builds == {BIG: BIG_FP8}
    assert fleet.tags_for(host) == frozenset({BIG_FP8})


async def test_what_a_host_was_bought_as_survives_a_supervisor_restart(make_fleet):
    """The record a successor reads back once left out the models, so after a restart a host
    bought for one model was asked for the whole rented set."""
    fleet = make_fleet(config(rent=("small-and-embed",), hosts=[
        {"id": "laptop", "kind": "local", "workers": 1, "models": [BIG],
         "transport": {"type": "http", "base_url": "http://127.0.0.1:1"}},
    ]))
    lease = open_lease(fleet)
    host = await fleet.rent_one(lease, [])
    ref = fleet.published_ref(host)
    assert ref["models"] == [SMALL, EMBED]
    assert ref["profile"] == "small-and-embed"
    assert ref["builds"] == {SMALL: SMALL_REPO, EMBED: EMBED_REPO}
    assert ref["disk_gb"] == host.disk_gb


# --- the least card and disk, from the sizes ---


def test_the_sizing_rule_is_the_launchers():
    """The search asks for what the machine's own launcher will refuse to start without."""
    assert sizing.WEIGHT_OVERHEAD == vllm_launch.WEIGHT_OVERHEAD
    assert sizing.TOTAL_MEMORY_SHARE == vllm_launch.TOTAL_MEMORY_SHARE
    assert sizing.CACHE_RESERVE_GB == pytest.approx(vllm_launch.CACHE_RESERVE_BYTES / 1e9)

    needs = sizing.needs_for({SMALL: 8.0, EMBED: 0.5})
    card = int(needs.card_memory_gb * 1e9)
    _, refused = vllm_launch.memory_plan([int(8.0e9), int(0.5e9)], card)
    assert refused is None, "a card of exactly the searched size starts the set"
    _, refused = vllm_launch.memory_plan([int(8.0e9), int(0.5e9)], card - int(2e9))
    assert refused is not None


def test_an_unmeasured_build_is_named_not_counted():
    needs = sizing.needs_for({BIG: 17.0, EMBED: None})
    assert needs.unknown == [EMBED]
    assert needs.weights_gb == 17.0


async def test_the_search_is_raised_to_what_the_profile_needs(make_fleet):
    fleet = make_fleet(config(rent=("chat-fp8",)))
    policy = fleet.next_host_policy()
    assert policy.min_gpu_memory_gb == math.ceil((28.0 * 1.1 + sizing.CACHE_RESERVE_GB) / 0.9)
    assert policy.min_disk_gb == math.ceil(28.0 * 1.1 + 10)
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=0, idle_seconds={})
    (instance,) = created(fleet)
    assert instance.spec.disk_gb == policy.min_disk_gb, "the disk searched for is the disk rented"


async def test_an_operators_higher_minimum_stands(make_fleet):
    fleet = make_fleet(config(rent=("chat",), offer_policy={
        "min_disk_gb": 200, "min_gpu_memory_gb": 80, "max_all_in_hourly": 0.60,
    }))
    policy = fleet.next_host_policy()
    assert (policy.min_gpu_memory_gb, policy.min_disk_gb) == (80, 200)


async def test_the_preview_shows_the_profile_its_needs_and_what_was_searched(make_fleet):
    fleet = make_fleet(config(rent=("chat",)))
    preview = await fleet.market_preview(hours=1)
    nxt = preview["next_host"]
    assert nxt["profile"] == "chat" and nxt["builds"] == {BIG: BIG_FP4}
    assert nxt["typed"]["min_disk_gb"] == 20
    assert nxt["searched"]["min_disk_gb"] == nxt["needs"]["disk_gb"] == math.ceil(17 * 1.1 + 10)


# --- a machine holding several models runs the router, one holding one does not ---


async def test_the_router_follows_what_the_machine_holds(make_fleet):
    fleet = make_fleet(config(rent=("chat", "small-and-embed")))
    assert "--proxy" not in fleet.engine_start_command([BIG])
    assert "--proxy" in fleet.engine_start_command([SMALL, EMBED])


# --- saved from the console ---

ADMIN_KEY = "gpmx_profiles_admin"

POOL_YAML = textwrap.dedent(
    f"""
    pool:
      name: profiles
      model_set: ["{EMBED}", "{BIG}"]
      probe_interval_s: 3600

    auth:
      admin_keys: ["{ADMIN_KEY}"]
      app_keys: ["gpma_profiles_app"]

    engine: ollama

    # The laptop serves Ollama's builds; rented hosts, the hub's.
    catalog:
      {BIG}:
        variants:
          - {{ tag: "{BIG}", engine: ollama }}
          - {{ tag: "{BIG_FP4}", engine: vllm, size_gb: 17.0 }}
      {EMBED}:
        variants:
          - {{ tag: "{EMBED}", engine: ollama }}
          - {{ tag: "{EMBED_REPO}", engine: vllm, size_gb: 0.5 }}

    hosts:
      - id: laptop
        kind: local
        workers: 3
        transport: {{ type: http, base_url: "http://127.0.0.1:1" }}

    rented:
      provider: fake
      engine: vllm
      image: vastai/vllm:v0.29.0-cuda-12.9
      engine_proxy: true   # every model on one machine
      capabilities: [cuda]
      offer_policy: {{ min_disk_gb: 10, max_all_in_hourly: 0.60 }}
    """
).strip()


@pytest.fixture
def pool(tmp_path):
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(POOL_YAML + f"\nrequest_log: {tmp_path / 'gpm.sqlite3'}\n")
    loop = BackgroundLoop()
    database = Database(tmp_path / "gpm.sqlite3")
    supervisor = Supervisor(load_config(config_path), database, config_path=str(config_path))
    server = ServerHandle(create_control_app(supervisor, supervisor.config), loop)
    try:
        yield supervisor, server.base_url, config_path
    finally:
        server.stop()
        loop.stop()
        database.close()


def put(url, body):
    with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {ADMIN_KEY}"}, timeout=30) as http:
        return http.put("/pool/config/profiles", json=body)


def test_profiles_are_saved_in_place_with_comments_kept(pool):
    supervisor, url, path = pool
    answer = put(url, {"profiles": {"chat": {BIG: BIG_FP4}, "embed": {EMBED: EMBED_REPO}}, "rent": ["chat", "embed"]})
    assert answer.status_code == 200, answer.text
    text = path.read_text()
    assert "# The laptop serves Ollama's builds" in text
    written = yaml.safe_load(text)
    assert written["rented"]["model_profiles"] == {"chat": {BIG: BIG_FP4}, "embed": {EMBED: EMBED_REPO}}
    assert written["rented"]["rent_profiles"] == ["chat", "embed"]
    assert written["rented"]["engine_proxy"] is False, "each profile's size decides the router now"
    assert supervisor.config.rented.rent_profiles == ["chat", "embed"]

    status = httpx.get(f"{url}/pool/status", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).json()
    chat = next(p for p in status["engine"]["profiles"] if p["name"] == "chat")
    assert chat["rented"] and chat["models"] == [{"model": BIG, "build": BIG_FP4, "size_gb": 17.0}]
    assert chat["needs"]["card_memory_gb"] == math.ceil((17.0 * 1.1 + sizing.CACHE_RESERVE_GB) / 0.9)


def test_a_model_found_on_the_hub_joins_the_set_and_the_laptop_keeps_what_it_held(pool):
    """A model built only for vLLM cannot be held by an Ollama laptop told to hold the whole set:
    the set is spread across hosts, and the laptop keeps exactly what it had."""
    supervisor, url, path = pool
    new, repo = "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B"
    answer = put(url, {
        "profiles": {"qwen": {new: repo}, "chat": {BIG: BIG_FP4}},
        "rent": ["qwen", "chat"],
        "add": [{"name": new, "builds": [{"tag": repo, "engine": "vllm", "size_gb": 61.1}]}],
    })
    assert answer.status_code == 200, answer.text
    written = yaml.safe_load(path.read_text())
    assert written["pool"]["model_set"] == [EMBED, BIG, new]
    assert written["pool"]["models_per_host"] == "declared"
    (laptop,) = written["hosts"]
    assert laptop["models"] == [EMBED, BIG]
    assert written["catalog"][new]["variants"] == [{"tag": repo, "engine": "vllm", "size_gb": 61.1}]


def test_a_second_build_of_a_model_goes_after_the_first(pool):
    """What every other host resolves the model to does not move."""
    supervisor, url, path = pool
    answer = put(url, {
        "profiles": {"chat-fp8": {BIG: BIG_FP8}}, "rent": ["chat-fp8"],
        "add": [{"name": BIG, "builds": [{"tag": BIG_FP8, "engine": "vllm", "size_gb": 28.0}]}],
    })
    assert answer.status_code == 200, answer.text
    variants = [v["tag"] for v in yaml.safe_load(path.read_text())["catalog"][BIG]["variants"]]
    assert variants == [BIG, BIG_FP4, BIG_FP8]


def test_a_build_nobody_added_is_refused_in_words(pool):
    supervisor, url, path = pool
    before = path.read_text()
    answer = put(url, {"profiles": {"chat": {BIG: "someone/Else"}}, "rent": ["chat"]})
    assert answer.status_code == 400
    assert "does not list for it" in answer.text
    assert path.read_text() == before


def test_renting_as_no_profile_is_refused(pool):
    supervisor, url, path = pool
    answer = put(url, {"profiles": {"chat": {BIG: BIG_FP4}}, "rent": []})
    assert answer.status_code == 400 and "at least one profile" in answer.text
