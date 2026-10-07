"""The RunPod client, against stubbed HTTP. No network, no credentials, no spending.

The catalog fixture is a real answer of the provider's public GPU catalog (prices and stock,
nothing of any account's). Everything else is shaped after the provider's published v2 OpenAPI
document.
"""

import json
import pathlib

import httpx
import pytest
from gpm_server.providers import (
    Instance,
    InstanceSpec,
    InstanceState,
    OfferGone,
    OfferQuery,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderUnavailable,
    RunPodProvider,
)

CATALOG = json.loads((pathlib.Path(__file__).parent.parent / "fixtures" / "runpod_catalog.json").read_text())
BASE = "https://api.runpod.io/v2"
#: Obviously not a real key, and shorter than any.
KEY = "rp_test_key"

ON_DEMAND = OfferQuery(on_demand=True, interruptible=False)


def row(gpu_id="NVIDIA Test 1", name="Test 1", memory=24, availability="LOW", secure=True, community=True,
        price=(0.5, 0.3), max_count=(8, 4)):
    return {"id": gpu_id, "name": name, "memory": memory, "availability": availability, "secure": secure,
            "community": community, "manufacturer": "NVIDIA", "pool": None,
            "price": {"secure": price[0], "community": price[1]},
            "maxCount": {"secure": max_count[0], "community": max_count[1]}}


def catalog(*rows, seen=None):
    """A handler answering the catalog — the real one unless rows are given."""

    def handler(request):
        if seen is not None:
            seen.append(request)
        assert request.url.path == "/v2/catalog/gpus", request.url.path
        return httpx.Response(200, json={"gpus": list(rows)} if rows else CATALOG)

    return handler


def provider(handler, **settings) -> RunPodProvider:
    client = httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(handler),
                               headers={"Authorization": f"Bearer {KEY}"})
    p = RunPodProvider(client=client, **settings)
    p.set_credential(KEY)
    p._client = client  # set_credential drops a client; keep the stubbed one
    return p


def pod(pod_id="pod1", name="p/rented-1", status="RUNNING", **more):
    entry = {
        "id": pod_id, "name": name, "status": status, "image": "img", "args": "arm-the-timer", "disk": 20,
        "ports": ["22/tcp"], "env": {}, "registry": None, "actions": [], "cloud": "SECURE",
        "dataCenterId": "US-KS-2", "cudaVersion": "12.8", "template": None, "cost": 0.74, "locked": False,
        "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1, "vcpuCount": 16, "memory": 64},
        "mounts": {"persistent": {"size": 130, "path": "/opt/gpm/models"}},
        "ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22, "username": "x-1", "command": "ssh x"},
                "direct": {"host": "195.26.233.3", "port": 34446, "username": "root", "command": "ssh y"}},
        "runtime": None, "createdAt": "2026-06-01T12:00:00Z", "startedAt": None, "globalNetworking": {"enabled": False},
    }
    entry.update(more)
    return entry


def problem(status, detail, **headers):
    return httpx.Response(status, headers=headers,
                          json={"title": "x", "status": status, "detail": detail})


# --- offers ---


async def test_offers_come_from_the_real_catalog_per_cloud_and_only_what_is_available():
    seen = []
    offers = await provider(catalog(seen=seen)).search_offers(ON_DEMAND)

    # One read per cloud: the catalog's availability is computed for one cloud and card count.
    assert [(r.url.params["cloud"], r.url.params["count"]) for r in seen] == [("SECURE", "1"), ("COMMUNITY", "1")]
    assert all(r.url.params["include"] == "AVAILABILITY" and r.url.params["product"] == "POD" for r in seen)

    rows = {r["id"]: r for r in CATALOG["gpus"]}
    assert offers, "the fixture holds available rows"
    for offer in offers:
        cloud, gpu_id, gpus = offer.offer_id.split(":")
        entry = rows[gpu_id]
        assert entry["availability"] != "NONE"
        assert entry[cloud.lower()] is True and entry["maxCount"][cloud.lower()] >= 1
        assert offer.raw is entry or offer.raw == entry
    ids = {o.offer_id for o in offers}
    assert "SECURE:NVIDIA GeForce RTX 4090:1" in ids and "COMMUNITY:NVIDIA GeForce RTX 4090:1" in ids
    # Stock NONE: never offered.
    assert not any("NVIDIA GeForce RTX 3090:" in i for i in ids)
    # Not offered in community (`community: false`), whatever its price there says.
    assert "COMMUNITY:NVIDIA A40:1" not in ids and "SECURE:NVIDIA A40:1" in ids
    # `secure: true` but no secure machine of it (`maxCount.secure` 0).
    assert "SECURE:NVIDIA RTX PRO 5000 Blackwell:1" not in ids
    # Sorted cheapest first.
    assert [o.all_in_hourly for o in offers] == sorted(o.all_in_hourly for o in offers)


async def test_an_offer_maps_on_to_what_ranking_and_the_ceilings_need():
    query = OfferQuery(on_demand=True, interruptible=False, min_gpus=2, min_disk_gb=150)
    offers = await provider(catalog(row(price=(0.5, 0.3)))).search_offers(query)
    secure = next(o for o in offers if o.offer_id.startswith("SECURE:"))

    assert secure.offer_id == secure.machine_id == "SECURE:NVIDIA Test 1:2"
    assert secure.hardware == "2x Test 1"  # Vast's "N x card" form
    assert secure.gpus == 2 and secure.gpu_memory_gb == 24  # per card
    storage = 150 * 0.10 / 720
    assert secure.storage_hourly == pytest.approx(storage)
    assert secure.all_in_hourly == pytest.approx(0.5 * 2 + storage)
    assert secure.min_bid_hourly == secure.on_demand_hourly == secure.all_in_hourly
    assert secure.interruptible is False and secure.bidding is False
    assert secure.verified is True and secure.disk_gb == 1000.0
    assert secure.download_per_gb == 0.0, "the provider charges nothing for transfer"
    community = next(o for o in offers if o.offer_id.startswith("COMMUNITY:"))
    assert community.verified is False and community.all_in_hourly == pytest.approx(0.3 * 2 + storage)


async def test_an_offer_is_repriced_for_the_disk_the_pool_rents():
    (offer, *_) = await provider(catalog(row(community=False))).search_offers(
        OfferQuery(on_demand=True, interruptible=False, min_disk_gb=10))
    priced = offer.priced_for(200)
    assert priced.all_in_hourly == pytest.approx(0.5 + 200 * 0.10 / 720)
    assert priced.min_bid_hourly == priced.all_in_hourly, "a fixed price has no floor apart from itself"


async def test_what_the_provider_does_not_report_is_said_to_be_assumed():
    (offer, *_) = await provider(catalog(row()), assumed_download_mbps=500.0,
                                 assumed_reliability=0.95).search_offers(ON_DEMAND)
    assert set(offer.assumed) == {"download_mbps", "reliability", "verified"}
    assert offer.download_mbps == 500.0 and offer.reliability == 0.95
    assert "download_per_gb" not in offer.assumed, "documented: no transfer fees"


async def test_cards_memory_and_stock_filter_the_catalog():
    rows = [
        row("A", "A", memory=16), row("B", "B", memory=48),
        row("C", "C", max_count=(1, 1)), row("D", "D", availability="NONE"), row("E", "E", availability=None),
    ]
    query = OfferQuery(on_demand=True, interruptible=False, min_gpus=2, min_gpu_memory_gb=24)
    ids = {o.offer_id for o in await provider(catalog(*rows)).search_offers(query)}
    assert ids == {"SECURE:B:2", "COMMUNITY:B:2"}


async def test_the_querys_own_limits_are_applied():
    rows = [row("A", "A", price=(0.5, 0.3)), row("B", "B", price=(2.0, 1.5)), row("C", "Excluded")]
    api = provider(catalog(*rows))

    cheap = await api.search_offers(OfferQuery(on_demand=True, interruptible=False, max_all_in_hourly=1.0))
    assert {o.offer_id for o in cheap} == {"SECURE:A:1", "COMMUNITY:A:1", "SECURE:C:1", "COMMUNITY:C:1"}
    excluded = await api.search_offers(OfferQuery(on_demand=True, interruptible=False, exclude_hardware=("Excluded",)))
    assert not any(o.offer_id.endswith(":C:1") for o in excluded)
    verified = await api.search_offers(OfferQuery(on_demand=True, interruptible=False, verified_only=True))
    assert verified and all(o.verified for o in verified)
    limited = await api.search_offers(OfferQuery(on_demand=True, interruptible=False, limit=2))
    assert [o.offer_id for o in limited] == ["COMMUNITY:A:1", "COMMUNITY:C:1"]
    avoided = await api.search_offers(OfferQuery(on_demand=True, interruptible=False, avoid_machines=("SECURE:A:1",)))
    assert "SECURE:A:1" not in {o.offer_id for o in avoided}


async def test_a_search_for_bids_only_finds_nothing_and_asks_nothing():
    seen = []
    assert await provider(catalog(seen=seen)).search_offers(OfferQuery(interruptible=True, on_demand=False)) == []
    assert seen == []


async def test_only_the_clouds_configured_are_asked():
    seen = []
    offers = await provider(catalog(row(), seen=seen), clouds=["secure"]).search_offers(ON_DEMAND)
    assert [r.url.params["cloud"] for r in seen] == ["SECURE"]
    assert {o.offer_id for o in offers} == {"SECURE:NVIDIA Test 1:1"}


async def test_a_catalog_without_its_list_is_refused_not_read_as_empty():
    with pytest.raises(ProviderUnavailable, match="no `gpus` list"):
        await provider(lambda r: httpx.Response(200, json={})).search_offers(ON_DEMAND)


# --- creating ---


async def an_offer(**query):
    (offer, *_) = await provider(catalog(row(community=False))).search_offers(
        OfferQuery(on_demand=True, interruptible=False, **query))
    return offer


SPEC = InstanceSpec(label="p/rented-1", image="ollama/ollama:0.12.0", disk_gb=150,
                    env={"OLLAMA_NUM_PARALLEL": "2"}, onstart="arm-the-timer", ports=(11434,))


async def test_creating_asks_for_the_pod_the_offer_and_the_spec_describe():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(201, json=pod(status="PROVISIONING"))

    offer = await an_offer(min_gpus=2)
    instance = await provider(handler).create(offer, SPEC, None)

    assert instance.instance_id == "pod1" and instance.label == "p/rented-1"
    assert instance.machine_id == "SECURE:NVIDIA Test 1:2"
    (request,) = seen
    assert request.method == "POST" and request.url.path == "/v2/pods"
    body = json.loads(request.read())
    assert body["name"] == "p/rented-1"
    assert body["image"] == "ollama/ollama:0.12.0"
    assert body["cloud"] == "SECURE"
    assert body["gpu"] == {"id": "NVIDIA Test 1", "count": 2}
    # The container disk is wiped at every stop; the rest of the disk is the volume a park keeps.
    assert body["disk"] == 20
    assert body["mounts"] == {"persistent": {"size": 130, "path": "/opt/gpm/models"}}
    assert body["ports"] == ["11434/tcp", "22/tcp"]
    assert body["env"] == {"OLLAMA_NUM_PARALLEL": "2"}
    assert body["entrypoint"] == ["/bin/sh", "-c"]
    assert body["cmd"][0].startswith("arm-the-timer\n") and "while :; do sleep 3600; done" in body["cmd"][0]
    assert "price" not in json.dumps(body) and "bid" not in body


async def test_a_small_disk_still_gets_the_providers_smallest_volume():
    seen = []

    def handler(request):
        seen.append(json.loads(request.read()))
        return httpx.Response(201, json=pod())

    await provider(handler).create(await an_offer(), InstanceSpec(label="l", image="i", disk_gb=15), None)
    assert seen[0]["mounts"]["persistent"]["size"] == 10 and seen[0]["disk"] == 5
    assert seen[0]["ports"] == ["22/tcp"] and "env" not in seen[0] and "cmd" not in seen[0]


async def test_without_a_volume_path_everything_is_container_disk_and_nothing_parks():
    seen = []

    def handler(request):
        seen.append(json.loads(request.read()))
        return httpx.Response(201, json=pod())

    api = provider(handler, volume_path=None)
    await api.create(await an_offer(), SPEC, None)
    assert seen[0]["disk"] == 150 and "mounts" not in seen[0]
    assert api.capabilities.parkable is False
    assert RunPodProvider.capabilities.parkable is True, "the class's own declaration is untouched"


async def test_a_bid_is_refused_before_anything_is_asked():
    seen = []
    with pytest.raises(ProviderError, match="on demand only"):
        await provider(lambda r: seen.append(r)).create(await an_offer(), SPEC, 0.4)
    assert seen == []


async def test_no_capacity_is_a_gone_offer_and_a_broken_rule_is_not():
    no_capacity = problem(400, "There are no longer any instances available with the requested specifications.")
    with pytest.raises(OfferGone, match="no capacity"):
        await provider(lambda r: no_capacity).create(await an_offer(), SPEC, None)

    rule = problem(400, "minCudaVersion and allowedCudaVersions are mutually exclusive")
    with pytest.raises(ProviderError, match="refused the request") as refused:
        await provider(lambda r: rule).create(await an_offer(), SPEC, None)
    assert not isinstance(refused.value, OfferGone)


async def test_insufficient_balance_stops_and_says_so():
    with pytest.raises(ProviderError, match="insufficient balance") as stopped:
        await provider(lambda r: problem(402, "insufficient balance")).create(await an_offer(), SPEC, None)
    assert not isinstance(stopped.value, OfferGone)


async def test_a_refused_credential_on_create_is_typed():
    with pytest.raises(ProviderAuthError):
        await provider(lambda r: problem(401, "invalid bearer token")).create(await an_offer(), SPEC, None)


async def test_a_gpu_pool_the_account_cannot_use_is_skipped_not_fatal():
    with pytest.raises(OfferGone, match="may not create it"):
        await provider(lambda r: problem(403, "access denied")).create(await an_offer(), SPEC, None)


async def test_an_unanswered_create_leaves_nothing_behind():
    """A 5xx is not a refusal: the pod may exist. Whatever the label names is ended."""
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            return problem(502, "upstream failure")
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"pods": [pod("pod9", "p/rented-1"), pod("pod8", "p/rented-10")],
                                         "pagination": {"nextCursor": None, "hasNextPage": False}})

    with pytest.raises(ProviderUnavailable, match="created pod9, which has been destroyed"):
        await provider(handler).create(await an_offer(), SPEC, None)
    assert ("DELETE", "/v2/pods/pod9") in seen
    assert ("DELETE", "/v2/pods/pod8") not in seen, "only the exact label, never a longer one"


async def test_a_create_whose_aftermath_cannot_be_checked_says_so():
    def handler(request):
        if request.method == "POST":
            raise httpx.ConnectError("reset")
        return problem(500, "down")

    with pytest.raises(ProviderUnavailable, match="could not be checked"):
        await provider(handler).create(await an_offer(), SPEC, None)


async def test_an_answer_naming_no_pod_is_checked_by_label():
    seen = []

    def handler(request):
        seen.append(request.method)
        if request.method == "POST":
            return httpx.Response(201, json={})
        return httpx.Response(200, json={"pods": [], "pagination": {"nextCursor": None, "hasNextPage": False}})

    with pytest.raises(ProviderUnavailable, match="names no pod"):
        await provider(handler).create(await an_offer(), SPEC, None)
    assert seen == ["POST", "GET"]


async def test_a_pod_created_in_error_is_destroyed_rather_than_handed_back():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(201, json=pod("pod5", status="ERROR"))
        return httpx.Response(204)

    with pytest.raises(ProviderError, match="created error, and has been destroyed"):
        await provider(handler).create(await an_offer(), SPEC, None)
    assert ("DELETE", "/v2/pods/pod5") in seen


async def test_the_credential_never_appears_in_an_error():
    echo = problem(400, f"bad token {KEY} rejected")
    with pytest.raises(ProviderError) as refused:
        await provider(lambda r: echo).create(await an_offer(), SPEC, None)
    assert KEY not in str(refused.value) and "[credential]" in str(refused.value)

    with pytest.raises(ProviderError) as failed:
        await provider(lambda r: problem(500, f"oops {KEY}")).list_instances("p/")
    assert KEY not in str(failed.value)


# --- the listing ---


async def test_every_page_is_read_and_only_this_pools_label_kept():
    pages = {
        None: {"pods": [pod("1", "p/a"), pod("2", "someone-else/x")],
               "pagination": {"nextCursor": "c2", "hasNextPage": True}},
        "c2": {"pods": [pod("3", "p/b", status="EXITED"), pod("4", "p/c", status="TERMINATED")],
               "pagination": {"nextCursor": None, "hasNextPage": False}},
    }
    cursors = []

    def handler(request):
        assert request.url.path == "/v2/pods"
        cursors.append(request.url.params.get("cursor"))
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    found = await provider(handler).list_instances("p/")
    assert [i.instance_id for i in found] == ["1", "3"], "stopped kept; terminated is gone"
    assert cursors == [None, "c2"]
    assert found[0].machine_id == "SECURE:NVIDIA GeForce RTX 4090:1" and found[0].label == "p/a"


async def test_a_listing_that_never_ends_is_refused_rather_than_half_believed():
    def handler(request):
        return httpx.Response(200, json={"pods": [pod()], "pagination": {"nextCursor": "more", "hasNextPage": True}})

    api = provider(handler)
    api.max_instance_pages = 3
    with pytest.raises(ProviderUnavailable, match="partial list"):
        await api.list_instances("p/")


async def test_a_listing_with_no_pods_list_or_no_cursor_is_refused():
    with pytest.raises(ProviderUnavailable, match="no `pods` list"):
        await provider(lambda r: httpx.Response(200, json={"message": "moved"})).list_instances("p/")
    broken = {"pods": [], "pagination": {"nextCursor": None, "hasNextPage": True}}
    with pytest.raises(ProviderUnavailable, match="no cursor"):
        await provider(lambda r: httpx.Response(200, json=broken)).list_instances("p/")


async def test_a_redirect_is_never_read_as_the_answer():
    with pytest.raises(ProviderUnavailable, match="redirect"):
        await provider(lambda r: httpx.Response(302, headers={"location": "/x"}, json={"pods": []})).list_instances("p/")


# --- state ---


@pytest.mark.parametrize("said, state", [
    ("RUNNING", InstanceState.RUNNING), ("PROVISIONING", InstanceState.SCHEDULING),
    ("STARTING", InstanceState.SCHEDULING), ("EXITED", InstanceState.STOPPED),
    ("ERROR", InstanceState.STOPPED), ("TERMINATED", InstanceState.GONE),
])
async def test_status_maps_the_providers_states(said, state):
    status = await provider(lambda r: httpx.Response(200, json=pod(status=said))).status(Instance("pod1"))
    assert status.state == state
    assert status.detail == f"status {said}"


async def test_a_pod_the_provider_no_longer_knows_is_gone():
    status = await provider(lambda r: problem(404, "pod not found")).status(Instance("pod1"))
    assert status.state == InstanceState.GONE


async def test_status_carries_the_rate_and_the_start_up_material():
    status = await provider(lambda r: httpx.Response(200, json=pod(cost=0.74))).status(Instance("pod1"))
    assert status.bid_hourly == 0.74 and status.startup_material is True and status.stopped_by_provider is None
    lost = await provider(lambda r: httpx.Response(200, json=pod(args="", cost=0))).status(Instance("pod1"))
    assert lost.startup_material is False and lost.bid_hourly is None


async def test_the_connection_is_ssh_direct_never_the_proxy():
    info = await provider(lambda r: httpx.Response(200, json=pod())).connection(Instance("pod1"))
    assert (info.ssh_host, info.ssh_port, info.ssh_user) == ("195.26.233.3", 34446, "root")
    assert info.public_url is None


async def test_the_connection_falls_back_to_the_runtime_port_mapping():
    entry = pod(ssh={"proxy": None, "direct": None},
                runtime={"ports": [{"private": 11434, "public": 4000, "ip": "1.2.3.4", "type": "tcp"},
                                   {"private": 22, "public": 40022, "ip": "1.2.3.4", "type": "tcp"}]})
    info = await provider(lambda r: httpx.Response(200, json=entry)).connection(Instance("pod1"))
    assert (info.ssh_host, info.ssh_port) == ("1.2.3.4", 40022)
    none = await provider(lambda r: httpx.Response(200, json=pod(ssh={"proxy": None, "direct": None}))).connection(
        Instance("pod1"))
    assert none.ssh_host is None


async def test_status_and_connection_share_one_fetch():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=pod())

    api = provider(handler)
    await api.status(Instance("pod1"))
    await api.connection(Instance("pod1"))
    assert calls == ["/v2/pods/pod1"]


# --- park, resume, destroy ---


async def test_stop_and_start_are_actions_on_the_pod():
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.read())))
        return httpx.Response(200, json=pod())

    api = provider(handler)
    await api.stop(Instance("pod1"))
    await api.start(Instance("pod1"))
    assert seen == [("/v2/pods/pod1/action", {"action": "stop"}), ("/v2/pods/pod1/action", {"action": "start"})]


async def test_a_start_whose_gpu_was_taken_raises():
    with pytest.raises(ProviderError, match="could not be started"):
        await provider(lambda r: problem(400, "no GPUs free on this machine")).start(Instance("pod1"))


async def test_stopping_a_stopped_pod_is_not_an_error():
    def handler(request):
        if request.method == "POST":
            return problem(409, "action not valid for current status")
        return httpx.Response(200, json=pod(status="EXITED"))

    await provider(handler).stop(Instance("pod1"))


async def test_destroy_is_idempotent():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return problem(404, "pod not found")

    await provider(handler).destroy(Instance("pod1"))
    assert seen == [("DELETE", "/v2/pods/pod1")]
    with pytest.raises(ProviderError):
        await provider(lambda r: problem(409, "cluster member")).destroy(Instance("pod1"))


async def test_destroying_forgets_what_was_remembered():
    def handler(request):
        if request.method == "DELETE":
            return httpx.Response(204)
        return problem(404, "pod not found")

    api = provider(handler)
    api._instances["pod1"] = (10**12, pod())
    await api.destroy(Instance("pod1"))
    assert (await api.status(Instance("pod1"))).state == InstanceState.GONE


async def test_set_bid_is_refused():
    with pytest.raises(ProviderError):
        await provider(lambda r: httpx.Response(200)).set_bid(Instance("pod1"), 0.5)


# --- money ---

BILLING = {"records": [
    {"startTime": "2026-06-01T00:00:00Z", "endTime": "2026-06-02T00:00:00Z", "podId": "pod1",
     "totalAmount": 1.25, "gpuAmount": 1.0, "cpuAmount": 0, "diskAmount": 0.25},
    {"startTime": "2026-06-02T00:00:00Z", "endTime": "2026-06-03T00:00:00Z", "podId": "pod1",
     "totalAmount": 0.5, "gpuAmount": 0.4, "cpuAmount": 0, "diskAmount": 0.1},
    {"startTime": "2026-06-02T00:00:00Z", "endTime": "2026-06-03T00:00:00Z", "podId": "pod2",
     "totalAmount": 9.0, "gpuAmount": 9.0, "cpuAmount": 0, "diskAmount": 0},
], "metadata": {}}


async def test_charges_are_the_pods_billing_records_summed_and_fetched_once_per_pass():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BILLING)

    api = provider(handler)
    charges = await api.reported_charges(Instance("pod1"))
    assert charges.total == pytest.approx(1.75) and charges.currency == "USD"
    assert (await api.reported_charges(Instance("pod2"))).total == 9.0
    assert await api.reported_charges(Instance("pod3")) is None, "no record is None, not zero"
    assert len(seen) == 1 and seen[0].url.path == "/v2/billing/pods"
    assert seen[0].url.params["bucketSize"] == "day" and seen[0].url.params["startTime"].endswith("T00:00:00Z")


async def test_billing_without_its_records_is_refused():
    with pytest.raises(ProviderUnavailable):
        await provider(lambda r: httpx.Response(200, json={})).reported_charges(Instance("pod1"))


def account_handler(graphql, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        if request.url.path == "/graphql":
            return graphql(request) if callable(graphql) else graphql
        assert request.url.path == "/v2/pods"
        return httpx.Response(200, json={"pods": [], "pagination": {"nextCursor": None, "hasNextPage": False}})
    return handler


async def test_the_balance_comes_from_graphql_and_only_the_balance():
    seen = []
    balance = httpx.Response(200, json={"data": {"myself": {"clientBalance": 12.5}}})
    status = await provider(account_handler(balance, seen)).account()
    assert status.credential_valid is True and status.credit_remaining == 12.5
    graphql = [r for r in seen if r.url.path == "/graphql"]
    assert len(graphql) == 1 and str(graphql[0].url) == "https://api.runpod.io/graphql"
    assert json.loads(graphql[0].read()) == {"query": "query { myself { clientBalance } }"}
    assert graphql[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("graphql", [
    httpx.Response(500, json={}),
    httpx.Response(401, json={}),
    httpx.Response(200, json={"errors": [{"message": "nope"}]}),
    httpx.Response(200, text="not json"),
])
async def test_a_balance_that_cannot_be_read_leaves_the_credential_valid(graphql):
    status = await provider(account_handler(graphql)).account()
    assert status.credential_valid is True and status.credit_remaining is None
    assert status.detail and "balance not read" in status.detail


async def test_a_balance_read_that_fails_in_transit_is_unknown_too():
    def graphql(request):
        raise httpx.ConnectError("unreachable")

    status = await provider(account_handler(graphql)).account()
    assert status.credential_valid is True and status.credit_remaining is None


async def test_a_refused_credential_is_typed():
    with pytest.raises(ProviderAuthError):
        await provider(lambda r: problem(401, "invalid bearer token")).account()


async def test_rate_limiting_is_typed_and_its_retry_after_honoured():
    calls = []

    def handler(request):
        calls.append(1)
        return problem(429, "rate limit exceeded for the minute window", **{"Retry-After": "12"})

    api = provider(handler)
    with pytest.raises(ProviderRateLimited, match="12s") as limited:
        await api.list_instances("p/")
    assert limited.value.retry_after_s == 12
    with pytest.raises(ProviderRateLimited, match="asked for"):
        await api.list_instances("p/")
    assert len(calls) == 1, "no call before the provider's Retry-After has passed"


# --- logs ---


async def test_boot_logs_are_the_tail_of_the_event_streams_data_lines():
    seen = []
    stream = (
        "id: 2026-06-01T12:02:01Z/1\n"
        'data: {"ts":"2026-06-01T12:02:01Z","source":"system","line":"pulling image"}\n\n'
        "id: 2026-06-01T12:02:02Z/2\n"
        'data: {"ts":"2026-06-01T12:02:02Z","source":"container","line":"starting engine"}\n\n'
        ": keep-alive\n\n"
        'data: {"ts":"2026-06-01T12:02:03Z","source":"container","line":"listening"}\n\n'
    )

    def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream)

    logs = await provider(handler).instance_logs(Instance("pod1"), tail=2)
    assert logs == "starting engine\nlistening"
    assert seen[0].url.path == "/v2/pods/pod1/logs" and seen[0].url.params["tail"] == "2"
    everything = await provider(handler).instance_logs(Instance("pod1"))
    assert everything.splitlines()[0] == "[system] pulling image"


async def test_boot_logs_that_cannot_be_read_are_none():
    assert await provider(lambda r: problem(404, "pod not found")).instance_logs(Instance("pod1")) is None
    assert await provider(lambda r: httpx.Response(200, text="")).instance_logs(Instance("pod1")) is None

    def broken(request):
        raise httpx.ReadTimeout("slow")

    assert await provider(broken).instance_logs(Instance("pod1")) is None


# --- the credential ---


async def test_a_handed_credential_is_the_one_sent_and_a_new_one_is_used_at_once(monkeypatch):
    """D134: the supervisor hands the plug-in its credential; the environment is not read."""
    monkeypatch.setenv("RUNPOD_API_KEY", "from-the-environment")
    seen = []

    def answer(request):
        seen.append(request.headers["authorization"])
        if request.url.path == "/graphql":
            return httpx.Response(200, json={"data": {"myself": {"clientBalance": 1.0}}})
        return httpx.Response(200, json={"pods": [], "pagination": {"nextCursor": None, "hasNextPage": False}})

    real = httpx.AsyncClient
    monkeypatch.setattr("gpm_server.providers.runpod.httpx.AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(answer), **kw))
    api = RunPodProvider()
    assert api.credential_env == "RUNPOD_API_KEY"
    api.set_credential("handed-first")
    await api.account()
    api.set_credential("handed-second")
    await api.account()
    assert seen == ["Bearer handed-first"] * 2 + ["Bearer handed-second"] * 2
    assert "from-the-environment" not in " ".join(seen)
    api.set_credential(None)
    with pytest.raises(ProviderAuthError, match="no credential is set"):
        await api.account()


async def test_the_account_credential_is_never_read_from_configuration(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="no credential is set"):
        _ = RunPodProvider(api_key="in-the-settings").client


async def test_the_api_key_variable_is_a_setting(monkeypatch):
    monkeypatch.setenv("OTHER_RUNPOD_KEY", "x" * 8)
    api = RunPodProvider(api_key_env="OTHER_RUNPOD_KEY")
    assert api.credential_env == "OTHER_RUNPOD_KEY"
    assert api.client.headers["authorization"] == "Bearer " + "x" * 8
    await api.aclose()


def test_how_it_presents_itself():
    from gpm_server.providers import presentation

    shown = presentation(RunPodProvider, "runpod")
    assert shown["display_name"] == "RunPod" and shown["interface_version"] == "2"
    assert shown["icon_url"] and shown["icon_url"].startswith("https://")
    assert shown["endpoint_settings"] == ["base_url", "graphql_url"]
    assert shown["offered"] is True and shown["takes_credential"] is True
    caps = shown["capabilities"]
    assert caps["interruptible"] is False and caps["parkable"] is True and caps["self_terminate"] is False
    assert caps["reports_charges"] and caps["direct_port_mapping"] and caps["reports_instance_logs"]


def test_it_is_installed_under_its_name():
    from gpm_server.providers.base import available_providers

    assert available_providers()["runpod"].load() is RunPodProvider


# --- what goes on the host ---


def test_the_self_terminate_request_uses_only_the_pods_own_key_and_id():
    request = RunPodProvider().self_terminate_request("destroy")
    assert request.method == "DELETE"
    assert request.url == "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID"
    assert request.headers == {"Authorization": "Bearer $RUNPOD_API_KEY"}
    assert request.body is None


def test_the_self_terminate_request_never_carries_a_credential(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "account-key-in-env")
    api = RunPodProvider()
    api.set_credential("handed-account-key")
    for action in ("destroy", "stop"):
        text = repr(api.self_terminate_request(action))
        assert "account-key-in-env" not in text and "handed-account-key" not in text
    stop = api.self_terminate_request("stop")
    assert stop.method == "POST" and stop.url.endswith("/pods/$RUNPOD_POD_ID/action")
    assert json.loads(stop.body) == {"action": "stop"}
