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
    """The provider quotes $/GB/month; every cost the pool reasons about is hourly — over the
    provider's own 720-hour month, which is what its `dph_total` is computed with."""
    offers = await provider(market()).search_offers(OfferQuery())
    assert offers[0].storage_per_gb_hourly == pytest.approx(0.15 / 720)


async def test_an_offer_is_priced_for_the_disk_the_pool_rents_not_the_providers_default():
    """Found live (D108): `dph_total` carries storage for a few GB of the provider's choosing,
    so a 150 GB host was billed more than the price the search compared. Checked against the
    live market: $0.20/GB-month at 150 GB is $0.041667/h inside `dph_total`."""
    entry = {**OFFER, "min_bid": 1.952, "dph_total": 1.954222, "storage_cost": 0.2,
             "storage_total_cost": 0.002222, "disk_space": 202.5}
    (offer,) = await provider(market(bid_offers=(entry,))).search_offers(OfferQuery())
    priced = offer.priced_for(150)
    assert priced.storage_hourly == pytest.approx(0.041667, abs=1e-6)
    assert priced.all_in_hourly == pytest.approx(1.993667, abs=1e-5), "the provider's own figure for 150 GB"
    assert priced.min_bid_hourly == offer.min_bid_hourly, "the floor is the machine's, whatever the disk"


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


async def test_a_refusal_keeps_the_providers_whole_answer():
    """Found live: three refusals in a row whose only recorded word was "refused" — the reason
    (a listing with less disk than was asked for) had to be worked out from the market after."""
    answer = {"success": False, "error": "invalid_args", "msg": None, "detail": "disk 150 > 101.25"}

    def handler(request):
        if request.method == "PUT":
            return httpx.Response(200, json=answer)
        return httpx.Response(200, json={"instances": []})

    offer = (await provider(market()).search_offers(OfferQuery()))[0]
    with pytest.raises(BidLost, match="invalid_args") as lost:
        await provider(handler).create(offer, InstanceSpec(label="l", image="i", disk_gb=150), bid=0.1)
    assert lost.value.response == answer


async def test_a_refusal_with_no_words_of_its_own_is_quoted_whole():
    def handler(request):
        if request.method == "PUT":
            return httpx.Response(200, json={"success": False, "new_contract": None})
        return httpx.Response(200, json={"instances": []})

    offer = (await provider(market()).search_offers(OfferQuery()))[0]
    with pytest.raises(BidLost, match='refused, answering {"new_contract":null,"success":false}'):
        await provider(handler).create(offer, InstanceSpec(label="l", image="i", disk_gb=10), bid=0.1)


async def test_an_offer_that_went_between_search_and_create_is_typed():
    def handler(request):
        return httpx.Response(410, json={"error": "no_such_ask"})

    offer = (await provider(market()).search_offers(OfferQuery()))[0]

    with pytest.raises(OfferGone):
        await provider(handler).create(offer, InstanceSpec(label="l", image="i", disk_gb=10), bid=0.1)


# --- state ---


# --- D43: a redirect must never be read as the answer ---


async def test_the_real_apis_redirect_on_the_bare_list_path_is_not_silently_read_as_empty():
    """Reproduces the live bug exactly: the real API 301s a GET without the trailing slash to
    the same URL with one, and the redirect's own body happens to parse as valid JSON with no
    "instances" key — which is why this read as "nothing exists" for as long as it did."""

    def handler(request):
        if request.url.path == "/api/v0/instances":  # the bare path: a 301 whose body is JSON
            return httpx.Response(
                301,
                headers={"location": "/api/v0/instances/?owner=me"},
                json={"message": "The resource has been moved...", "code": "301 Moved Permanently"},
            )
        if request.url.path == "/api/v0/instances/":  # and behind it, a 410
            return httpx.Response(410, json={"success": False, "error": "deprecated_endpoint"})
        assert request.url.path == "/api/v1/instances/", request.url.path
        return httpx.Response(200, json={"instances": [{"id": 1, "label": "p/a", "machine_id": 5}]})

    instances = await provider(handler).list_instances("p/")
    assert [i.instance_id for i in instances] == ["1"]


async def test_any_unexpected_redirect_raises_rather_than_being_read_as_the_payload():
    """Defense in depth beyond the one path above: whatever the reason, a 3xx is never quietly
    treated as a 2xx-shaped answer just because its body happens to parse as JSON."""
    from gpm_server.providers import ProviderUnavailable

    def handler(request):
        return httpx.Response(302, headers={"location": "/elsewhere"}, json={"ok": True})

    with pytest.raises(ProviderUnavailable, match="redirect"):
        await provider(handler).list_instances("p/")


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


async def test_outbid_while_its_image_is_still_loading_reads_as_stopped():
    """Found live (D109), the provider's own answer for rented-9023b1 four minutes in: still
    loading its image, and already decided not to run it. Read as "still starting", the pool
    neither re-bid nor released it."""
    payload = {"instances": {
        "actual_status": "loading", "intended_status": "stopped", "cur_state": "stopped",
        "next_state": "stopped", "status_msg": "\n#5 [2/6] RUN mkdir -p /tmp; chmod 1777 /tmp; exit 0;\n",
    }}
    status = await provider(lambda r: httpx.Response(200, json=payload)).status(Instance("1"))
    assert status.state == InstanceState.STOPPED


async def test_a_machines_own_price_is_asked_of_the_machine_rentable_or_not():
    """A machine the pool was just outbid on is held by whoever outbid it, so the ordinary
    search, which asks for rentable machines only, cannot see it (D109)."""
    seen = []

    def handler(request):
        body = json.loads(request.read())
        seen.append(body)
        if body.get("type") == "on-demand":
            return httpx.Response(200, json={"offers": []})
        return httpx.Response(200, json={"offers": [
            {**OFFER, "id": 40176930, "machine_id": 138919, "num_gpus": 1, "min_bid": 1.2, "rentable": False},
            {**OFFER, "id": 40176932, "machine_id": 138919, "num_gpus": 2, "min_bid": 2.4, "rentable": False},
        ]})

    offer = await provider(handler).offer_for_machine("138919", gpus=1)
    assert offer is not None and offer.offer_id == "40176930" and offer.min_bid_hourly == 1.2
    assert seen[0]["machine_id"] == {"eq": 138919} and "rentable" not in seen[0]
    assert await provider(handler).offer_for_machine("138919", gpus=4) is None


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


def test_the_self_terminate_request_uses_only_the_instance_scoped_key():
    request = VastProvider().self_terminate_request("destroy")
    assert request.method == "DELETE"
    assert request.headers["Authorization"] == "Bearer $CONTAINER_API_KEY"
    assert "$CONTAINER_ID" in request.url
    assert "VAST_API_KEY" not in str(request)


def test_stopping_instead_of_destroying_is_available():
    request = VastProvider().self_terminate_request("stop")
    assert request.method == "PUT"
    assert request.body is not None and '"state": "stopped"' in request.body


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
        if request.url.path == "/api/v1/instances/":
            return httpx.Response(200, json={"instances": [{"id": 1, "label": "p/a", "machine_id": 5, "actual_status": "running", "intended_status": "running"}]})
        return httpx.Response(200, json={"instances": {}})

    p = provider(handler)
    await p.list_instances("p/")
    status = await p.status(Instance("1"))
    assert status.state == InstanceState.RUNNING
    assert calls == ["/api/v1/instances/"]


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


async def test_every_page_of_the_instance_listing_is_followed():
    """A half-read listing is a missed orphan. v1 paginates with `next_token`."""
    pages = {
        None: {"instances": [{"id": 1, "label": "p/a", "machine_id": 5}], "next_token": "t2"},
        "t2": {"instances": [{"id": 2, "label": "p/b", "machine_id": 6}], "next_token": None},
    }

    def handler(request):
        return httpx.Response(200, json=pages[request.url.params.get("start_token")])

    assert [i.instance_id for i in await provider(handler).list_instances("p/")] == ["1", "2"]


async def test_a_listing_that_never_ends_is_refused_rather_than_half_believed():
    from gpm_server.providers import ProviderUnavailable

    def handler(request):
        return httpx.Response(200, json={"instances": [{"id": 1, "label": "p/a"}], "next_token": "more"})

    p = provider(handler)
    p.max_instance_pages = 3
    with pytest.raises(ProviderUnavailable, match="partial list"):
        await p.list_instances("p/")


async def test_a_bid_reported_lost_that_actually_created_an_instance_destroys_it():
    """Seen live: this API answered `success: false` and created the instance anyway. Three
    of those in one pass is how one lease came to be paying for three machines."""
    destroyed = []

    def handler(request):
        if request.method == "PUT":
            return httpx.Response(200, json={"success": False})
        if request.method == "DELETE":
            destroyed.append(request.url.path)
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"instances": [{"id": 77, "label": "p/rented-x", "machine_id": 5}]})

    offer = (await provider(market()).search_offers(OfferQuery()))[0]
    spec = InstanceSpec(label="p/rented-x", image="img", disk_gb=10, onstart="")
    with pytest.raises(BidLost, match="created 77, which has been destroyed"):
        await provider(handler).create(offer, spec, 0.2)
    assert destroyed == ["/api/v0/instances/77/"]


async def test_a_lost_bid_that_cannot_be_checked_raises_a_provider_error_not_a_lost_bid():
    """The caller treats these differently on purpose: BidLost means try the next offer,
    ProviderError means stop. Not knowing what is running must never mean "try the next"."""
    from gpm_server.providers import ProviderUnavailable

    def handler(request):
        if request.method == "PUT":
            return httpx.Response(200, json={"success": False})
        return httpx.Response(500, json={})

    offer = (await provider(market()).search_offers(OfferQuery()))[0]
    spec = InstanceSpec(label="p/rented-x", image="img", disk_gb=10, onstart="")
    with pytest.raises(ProviderUnavailable):
        await provider(handler).create(offer, spec, 0.2)


async def test_a_gone_offer_and_a_gone_endpoint_are_not_the_same_error():
    """410 on an offer is a market event — try the next one. 410 anywhere else means this
    client is calling something that no longer exists, and must not read as a lost offer."""
    from gpm_server.providers import ProviderUnavailable

    gone = httpx.Response(410, json={"success": False, "error": "deprecated_endpoint"})
    with pytest.raises(OfferGone):
        await provider(lambda r: gone).create(
            (await provider(market()).search_offers(OfferQuery()))[0],
            InstanceSpec(label="p/x", image="i", disk_gb=10, onstart=""),
            bid=0.2,
        )
    with pytest.raises(ProviderUnavailable, match="endpoint is gone"):
        await provider(lambda r: gone).list_instances("p/")
