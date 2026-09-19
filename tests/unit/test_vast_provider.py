"""The Vast.ai client, against stubbed HTTP. No network, no credentials, no spending.

What is checked here is the mapping and the error typing — the parts that must be right before
a single real call is made.
"""

import json

import httpx
import pytest
from gpm_server.providers import (
    BidLost,
    Instance,
    InstanceSpec,
    InstanceState,
    OfferGone,
    OfferQuery,
    ProviderAuthError,
    ProviderRateLimited,
    VastProvider,
)

OFFER = {
    "id": 12345,
    "machine_id": 999,
    "gpu_name": "RTX 4090",
    "num_gpus": 2,
    "gpu_ram": 24576,  # MB
    "disk_space": 120.0,
    "min_bid": 0.11,
    "dph_total": 0.19,
    "dph_base": 0.44,
    "storage_cost": 0.15,  # $/GB/month
    "inet_down_cost": 0.008,
    "inet_down": 850.0,
    "reliability": 0.987,
    "verification": "verified",
    "dlperf": 42.5,
}

#: The same machine in the on-demand listing. Its `dph_total` is the on-demand price.
ON_DEMAND = {"id": 54321, "machine_id": 999, "dph_total": 0.44, "dph_base": 0.43}


def market(bid_offers=(OFFER,), on_demand_offers=(ON_DEMAND,)):
    """A handler answering both listings — which one by the `type` in the request body."""

    def handler(request):
        if request.url.path != "/api/v0/bundles":
            return httpx.Response(404, json={})
        body = json.loads(request.read())
        if body.get("type") == "on-demand":
            return httpx.Response(200, json={"offers": list(on_demand_offers)})
        return httpx.Response(200, json={"offers": list(bid_offers)})

    return handler


def provider(handler) -> VastProvider:
    client = httpx.AsyncClient(
        base_url="https://console.vast.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test"},
    )
    return VastProvider(client=client)


# --- offers ---


async def test_offers_map_on_to_what_ranking_and_the_ceilings_need():
    seen = []

    def handler(request):
        assert request.url.path == "/api/v0/bundles"
        body = json.loads(request.read())
        seen.append(body)
        if body["type"] == "on-demand":
            return httpx.Response(200, json={"offers": [ON_DEMAND]})
        assert body["rentable"] == {"eq": True}
        assert body["gpu_ram"] == {"gte": 16 * 1024}
        return httpx.Response(200, json={"offers": [OFFER]})

    offers = await provider(handler).search_offers(OfferQuery(min_gpu_memory_gb=16, limit=10))

    assert [b["type"] for b in seen] == ["bid", "on-demand"]
    assert seen[1]["machine_id"] == {"in": [999]}  # exactly the machines in hand

    assert len(offers) == 1
    offer = offers[0]
    assert offer.offer_id == "12345"
    assert offer.machine_id == "999"
    assert offer.gpu_memory_gb == pytest.approx(24.0)  # MB in the API, GB in the pool
    assert offer.min_bid_hourly == 0.11  # the floor a bid is built from
    assert offer.all_in_hourly == 0.19
    # From the on-demand listing, not the bid row's dph_base (which is the floor again).
    assert offer.on_demand_hourly == 0.44
    assert offer.download_per_gb == 0.008
    assert offer.verified is True
    assert offer.throughput_proxy == 42.5


async def test_a_machine_with_no_on_demand_listing_has_no_crossover_to_clamp_against():
    offers = await provider(market(on_demand_offers=())).search_offers(OfferQuery())
    assert offers[0].on_demand_hourly is None


async def test_the_bid_rows_dph_base_is_never_mistaken_for_the_on_demand_price():
    """Verified on the live market: on a bid offer dph_base == min_bid on every row."""
    trap = {**OFFER, "dph_base": OFFER["min_bid"]}
    offers = await provider(market(bid_offers=(trap,), on_demand_offers=())).search_offers(OfferQuery())
    assert offers[0].on_demand_hourly is None  # not 0.11


async def test_storage_is_converted_from_per_month_to_per_hour():
    """The provider quotes $/GB/month; every cost the pool reasons about is hourly."""
    offers = await provider(market()).search_offers(OfferQuery())
    # 0.15 $/GB/month x 120GB / 730h
    assert offers[0].storage_hourly == pytest.approx(0.15 * 120 / 730)


async def test_an_unverified_machine_is_reported_as_such():
    entry = {**OFFER, "verification": "unverified"}
    offers = await provider(market(bid_offers=(entry,))).search_offers(OfferQuery())
    assert offers[0].verified is False


# --- creating ---


async def test_creating_passes_the_bid_the_label_and_the_start_up_script():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"success": True, "new_contract": 777})

    instance = await provider(handler).create(
        (await provider(market()).search_offers(OfferQuery()))[0],
        InstanceSpec(label="pool/rented-1", image="ollama/ollama:0.12.0", disk_gb=60, onstart="arm-the-timer"),
        bid=0.13,
    )

    assert instance.instance_id == "777"
    assert seen["path"] == "/api/v0/asks/12345/"
    assert seen["body"]["price"] == 0.13
    assert seen["body"]["label"] == "pool/rented-1"
    assert seen["body"]["onstart"] == "arm-the-timer"


async def test_a_lost_bid_raises_rather_than_returning_something_half_made():
    def handler(request):
        return httpx.Response(200, json={"success": False, "msg": "outbid"})

    offer = (await provider(market()).search_offers(OfferQuery()))[0]

    with pytest.raises(BidLost, match="outbid"):
        await provider(handler).create(offer, InstanceSpec(label="l", image="i", disk_gb=10), bid=0.1)


async def test_an_offer_that_went_between_search_and_create_is_typed():
    def handler(request):
        return httpx.Response(410, json={"error": "no_such_ask"})

    offer = (await provider(market()).search_offers(OfferQuery()))[0]

    with pytest.raises(OfferGone):
        await provider(handler).create(offer, InstanceSpec(label="l", image="i", disk_gb=10), bid=0.1)


# --- state ---


async def test_the_orphan_sweep_only_sees_this_pools_label():
    payload = {
        "instances": [
            {"id": 1, "label": "mypool/rented-a", "machine_id": 5},
            {"id": 2, "label": "someone-else/thing", "machine_id": 6},
            {"id": 3, "label": None, "machine_id": 7},
        ]
    }
    instances = await provider(lambda r: httpx.Response(200, json=payload)).list_instances("mypool/")

    assert [i.instance_id for i in instances] == ["1"]


async def test_stopped_while_we_wanted_running_reads_as_an_eviction():
    payload = {"instances": {"actual_status": "exited", "intended_status": "running"}}
    status = await provider(lambda r: httpx.Response(200, json=payload)).status(Instance("1"))

    assert status.state == InstanceState.STOPPED
    assert status.stopped_by_provider is True


async def test_a_running_instance_is_running():
    payload = {"instances": {"actual_status": "running", "intended_status": "running"}}
    status = await provider(lambda r: httpx.Response(200, json=payload)).status(Instance("1"))
    assert status.state == InstanceState.RUNNING


async def test_an_instance_the_provider_no_longer_knows_is_gone():
    status = await provider(lambda r: httpx.Response(200, json={})).status(Instance("1"))
    assert status.state == InstanceState.GONE


CHARGES = {
    "success": True,
    "count": 3,
    "results": [
        {"source": "instance-1", "type": "instance", "amount": 0.007, "start": 1, "end": 1, "items": []},
        {"source": "instance-1", "type": "instance", "amount": 0.030, "start": 0, "end": 0, "items": []},
        {"source": "instance-2", "type": "instance", "amount": 0.042, "start": 1, "end": 1,
         "metadata": {"label": "someone-else"}},
    ],
}


async def test_charges_come_from_the_charges_endpoint_summed_per_instance_over_its_days():
    """Verified live: the instance payload has no accumulated charge; /api/v0/charges/ has one
    row per instance per day."""
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=CHARGES)

    charges = await provider(handler).reported_charges(Instance("1"))
    assert charges.total == pytest.approx(0.037)
    assert seen[0].url.path == "/api/v0/charges/"
    filters = json.loads(seen[0].url.params["select_filters"])
    assert filters["type"] == {"in": ["instance"]}
    assert "day" in filters


async def test_an_instance_with_no_charge_row_yet_is_none_not_zero():
    """None keeps the cap margin wide; a zero would narrow it on a figure that never came."""
    charges = await provider(lambda r: httpx.Response(200, json=CHARGES)).reported_charges(Instance("99"))
    assert charges is None


async def test_the_charge_rows_are_fetched_once_and_shared_by_every_instance():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=CHARGES)

    p = provider(handler)
    await p.reported_charges(Instance("1"))
    await p.reported_charges(Instance("2"))
    await p.reported_charges(Instance("1"))
    assert len(calls) == 1


async def test_charge_pages_are_followed():
    pages = [
        {"results": [{"source": "instance-1", "amount": 0.5}], "next_token": "t2"},
        {"results": [{"source": "instance-1", "amount": 0.25}]},
    ]
    tokens = []

    def handler(request):
        tokens.append(request.url.params.get("after_token"))
        return httpx.Response(200, json=pages[len(tokens) - 1])

    charges = await provider(handler).reported_charges(Instance("1"))
    assert charges.total == 0.75
    assert tokens == [None, "t2"]


# --- errors are typed, never bare HTTP ---


async def test_a_refused_credential_is_typed():
    with pytest.raises(ProviderAuthError):
        await provider(lambda r: httpx.Response(401, json={})).account()


async def test_rate_limiting_is_typed():
    with pytest.raises(ProviderRateLimited):
        await provider(lambda r: httpx.Response(429, json={})).account()


async def test_the_account_credential_is_never_read_from_configuration(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="never from configuration"):
        _ = VastProvider().client


# --- what goes on the host ---


def test_the_self_terminate_command_uses_only_the_instance_scoped_key():
    command = VastProvider().self_terminate_command("destroy")
    assert "$CONTAINER_API_KEY" in command
    assert "$CONTAINER_ID" in command
    assert "VAST_API_KEY" not in command
    assert command.startswith("curl -sS -X DELETE")


def test_stopping_instead_of_destroying_is_available():
    command = VastProvider().self_terminate_command("stop")
    assert '"state": "stopped"' in command


# --- one fetch per instance per pass ---


async def test_status_and_connection_share_one_fetch():
    """Seen live: the provider rate-limits repeated fetches of the same instance, and the
    pool was making several per pass."""
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"instances": {"id": 1, "actual_status": "running", "intended_status": "running", "ssh_host": "h", "ssh_port": 22}})

    p = provider(handler)
    await p.status(Instance("1"))
    await p.connection(Instance("1"))
    await p.status(Instance("1"))
    assert calls == ["/api/v0/instances/1/"]


async def test_the_listing_feeds_the_cache_so_the_sweep_pass_costs_one_call():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/v0/instances":
            return httpx.Response(200, json={"instances": [{"id": 1, "label": "p/a", "machine_id": 5, "actual_status": "running", "intended_status": "running"}]})
        return httpx.Response(200, json={"instances": {}})

    p = provider(handler)
    await p.list_instances("p/")
    status = await p.status(Instance("1"))
    assert status.state == InstanceState.RUNNING
    assert calls == ["/api/v0/instances"]


async def test_a_stale_entry_is_fetched_again():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"instances": {"id": 1, "actual_status": "running"}})

    p = provider(handler)
    p.instance_cache_s = 0.0
    await p.status(Instance("1"))
    await p.status(Instance("1"))
    assert len(calls) == 2


async def test_destroying_forgets_what_was_remembered():
    def handler(request):
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"instances": {}})

    p = provider(handler)
    p._instances["1"] = (10**12, {"id": 1, "actual_status": "running"})
    await p.destroy(Instance("1"))
    assert (await p.status(Instance("1"))).state == InstanceState.GONE
