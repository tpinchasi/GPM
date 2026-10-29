"""The Verda client, against stubbed HTTP. No network, no credentials, no spending.

The catalog fixture is a few rows of the provider's real public instance-type list (prices and
specifications, nothing of any account's). Everything else is shaped after its published OpenAPI
document for `https://api.verda.com/v1`.
"""

import json
import pathlib
import subprocess

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
    VerdaProvider,
)

TYPES = json.loads((pathlib.Path(__file__).parent.parent / "fixtures" / "verda_instance_types.json").read_text())
BASE = "https://api.verda.com/v1"
#: Obviously not a real client, and shorter than any.
CREDENTIAL = "vc_test_id:vc_test_secret"

SPOT = OfferQuery(interruptible=True, on_demand=False)
ON_DEMAND = OfferQuery(interruptible=False, on_demand=True)
AVAILABLE = [{"location_code": "FIN-01", "availabilities": ["1H100.80S.30V", "1A6000.10V", "CPU-TURIN.4V.16G"]},
             {"location_code": "ICE-01", "availabilities": ["2H100.80S.80V"]}]


class Api:
    """A stub of the provider: answers by method and path, records every request."""

    def __init__(self, **answers):
        self.answers = {"POST /v1/oauth2/token": {"access_token": "tok-1", "expires_in": 3600,
                                                  "refresh_token": "r", "scope": "cloud-api-v1", "token_type": "Bearer"},
                        "GET /v1/instance-types": TYPES,
                        "GET /v1/instance-availability": AVAILABLE,
                        "GET /v1/ssh-keys": [{"id": "key-1", "name": "operator", "key": "ssh-ed25519 AAAA op"}],
                        "POST /v1/scripts": "script-1",
                        "DELETE /v1/scripts/script-1": None,
                        **answers}
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        answer = self.answers.get(f"{request.method} {request.url.path}")
        if callable(answer):
            answer = answer(request)
        if isinstance(answer, httpx.Response):
            return answer
        if answer is None and f"{request.method} {request.url.path}" not in self.answers:
            return httpx.Response(404, json={"code": "not_found", "message": "no such route in the stub"})
        return httpx.Response(200, json=answer)

    def bodies(self, method, path):
        return [json.loads(r.content) for r in self.seen if r.method == method and r.url.path == path and r.content]

    def paths(self):
        return [f"{r.method} {r.url.path}" for r in self.seen]


def provider(api, **kwargs):
    made = VerdaProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(api), base_url=BASE), **kwargs)
    made.set_credential(CREDENTIAL)
    made._client = httpx.AsyncClient(transport=httpx.MockTransport(api), base_url=BASE)
    return made


def spec(**overrides):
    return InstanceSpec(**{"label": "gpm-test/rented-abc123", "image": "vllm/vllm-openai:latest", "disk_gb": 120,
                           "env": {"GPM_WORKERS": "6"}, "onstart": "echo engine", **overrides})


# --- the credential ---


async def test_the_client_id_and_secret_are_exchanged_for_a_token_used_on_every_call():
    api = Api()
    verda = provider(api)
    await verda.search_offers(SPOT)
    await verda.search_offers(SPOT)
    (exchange,) = api.bodies("POST", "/v1/oauth2/token")
    assert exchange == {"grant_type": "client_credentials", "client_id": "vc_test_id", "client_secret": "vc_test_secret"}
    calls = [r for r in api.seen if r.url.path != "/v1/oauth2/token"]
    assert calls and all(r.headers["authorization"] == "Bearer tok-1" for r in calls)


async def test_a_token_refused_is_made_again_once():
    tokens = iter(["tok-old", "tok-new"])
    refused = {"n": 0}

    def balance(request):
        if request.headers["authorization"] == "Bearer tok-old":
            refused["n"] += 1
            return httpx.Response(401, json={"code": "unauthorized_request", "message": "expired"})
        return {"amount": 12.5, "currency": "usd"}

    api = Api(**{"POST /v1/oauth2/token": lambda r: {"access_token": next(tokens), "expires_in": 3600},
                 "GET /v1/balance": balance})
    status = await provider(api).account()
    assert status.credential_valid and status.credit_remaining == 12.5 and refused["n"] == 1


async def test_a_credential_not_in_two_parts_is_refused_with_its_form():
    verda = provider(Api())
    verda.set_credential("only-one-part")
    with pytest.raises(ProviderAuthError, match="CLIENT_ID:CLIENT_SECRET"):
        await verda.search_offers(SPOT)


async def test_a_refused_credential_is_said_and_never_echoed():
    api = Api(**{"POST /v1/oauth2/token": httpx.Response(
        401, json={"code": "unauthorized_request", "message": "bad client vc_test_secret"})})
    with pytest.raises(ProviderAuthError) as caught:
        await provider(api).account()
    assert "vc_test_secret" not in str(caught.value) and "[credential]" in str(caught.value)


async def test_no_credential_says_where_one_goes(monkeypatch):
    monkeypatch.delenv("VERDA_CREDENTIAL", raising=False)
    verda = VerdaProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(Api()), base_url=BASE))
    with pytest.raises(ProviderAuthError, match="Providers screen"):
        await verda.search_offers(SPOT)


async def test_a_rate_limit_is_honoured_before_the_next_call():
    api = Api(**{"GET /v1/instance-availability": httpx.Response(429, headers={"retry-after": "30"},
                                                                  json={"code": "rate", "message": "slow down"})})
    verda = provider(api)
    with pytest.raises(ProviderRateLimited):
        await verda.search_offers(SPOT)
    before = len(api.seen)
    with pytest.raises(ProviderRateLimited, match="asked for"):
        await verda.search_offers(SPOT)
    assert not [r for r in api.seen[before:] if r.url.path == "/v1/instance-availability"], "not asked again"
    # Kept per operation: a search refused does not hold up a listing, nor a delete.
    api.answers["GET /v1/instances"] = []
    assert await verda.list_instances("gpm-") == []


# --- offers ---


async def test_spot_offers_are_the_providers_price_and_never_bid_for():
    offers = await provider(Api()).search_offers(OfferQuery(interruptible=True, min_disk_gb=360))
    h100 = next(o for o in offers if o.machine_id == "FIN-01:1H100.80S.30V")
    assert h100.interruptible and not h100.bidding
    # The spot price is the machine's alone, as a bid would be; the pool adds the storage (D132).
    assert h100.min_bid_hourly == pytest.approx(1.945)
    assert h100.all_in_hourly == pytest.approx(1.945 + 0.1) and h100.storage_hourly == pytest.approx(0.1)
    assert h100.on_demand_hourly == pytest.approx(3.889 + 0.1)
    assert h100.hardware == "1x H100 SXM5 80GB" and h100.gpu_memory_gb == 80 and h100.gpus == 1
    assert "download_mbps" in h100.assumed


async def test_spot_and_on_demand_are_asked_apart_and_offered_apart():
    api = Api()
    offers = await provider(api).search_offers(OfferQuery(interruptible=True, on_demand=True))
    asked = sorted(r.url.params["is_spot"] for r in api.seen if r.url.path == "/v1/instance-availability")
    assert asked == ["false", "true"]
    kinds = {(o.machine_id, o.interruptible) for o in offers}
    assert ("FIN-01:1H100.80S.30V", True) in kinds and ("FIN-01:1H100.80S.30V", False) in kinds


async def test_the_disk_is_priced_into_every_offer():
    offers = await provider(Api()).search_offers(OfferQuery(interruptible=False, on_demand=True, min_disk_gb=360))
    h100 = next(o for o in offers if o.machine_id == "FIN-01:1H100.80S.30V")
    assert h100.storage_hourly == pytest.approx(360 * 0.20 / 720)
    assert h100.all_in_hourly == pytest.approx(3.889 + 0.1)


async def test_card_memory_is_per_card_and_cpu_types_are_never_offered():
    offers = await provider(Api()).search_offers(OfferQuery(interruptible=True, min_gpu_memory_gb=60))
    assert {o.machine_id for o in offers} == {"FIN-01:1H100.80S.30V", "ICE-01:2H100.80S.80V"}
    two = next(o for o in offers if o.gpus == 2)
    assert two.gpu_memory_gb == 80


async def test_the_pools_filters_are_applied():
    verda = provider(Api())
    assert {o.machine_id for o in await verda.search_offers(OfferQuery(interruptible=True, min_gpus=2))} == {
        "ICE-01:2H100.80S.80V"}
    cheap = await verda.search_offers(OfferQuery(interruptible=True, max_all_in_hourly=1.0))
    assert {o.machine_id for o in cheap} == {"FIN-01:1A6000.10V"}
    avoided = await verda.search_offers(OfferQuery(interruptible=True, avoid_machines=("FIN-01:1H100.80S.30V",)))
    assert "FIN-01:1H100.80S.30V" not in {o.machine_id for o in avoided}


async def test_an_availability_answer_that_is_not_a_list_is_never_read_as_none():
    with pytest.raises(ProviderUnavailable, match="not a list"):
        await provider(Api(**{"GET /v1/instance-availability": {"error": "?"}})).search_offers(SPOT)


# --- creating ---


async def offer(kind=SPOT, machine="FIN-01:1H100.80S.30V"):
    return next(o for o in await provider(Api()).search_offers(kind) if o.machine_id == machine)


async def test_a_spot_instance_is_created_with_its_disk_deleted_with_it():
    api = Api(**{"POST /v1/instances": "inst-1"})
    made = await provider(api).create(await offer(), spec(), bid=2.5)
    assert made.instance_id == "inst-1" and made.location == "FIN-01" and made.machine_id == "FIN-01:1H100.80S.30V"
    (body,) = api.bodies("POST", "/v1/instances")
    assert body["instance_type"] == "1H100.80S.30V" and body["location_code"] == "FIN-01"
    assert body["is_spot"] is True and body["contract"] == "SPOT"
    assert body["os_volume"]["size"] == 120 and body["os_volume"]["on_spot_discontinue"] == "delete_permanently"
    assert body["description"] == "gpm-test/rented-abc123" and body["tags"] == [{"key": "gpm"}]
    assert body["hostname"] == "gpm-test-rented-abc123" and body["ssh_key_ids"] == ["key-1"]
    assert body["startup_script_id"] == "script-1"
    assert "2.5" not in json.dumps(body), "the provider takes no maximum price: none is sent"


async def test_an_on_demand_instance_is_pay_as_you_go():
    api = Api(**{"POST /v1/instances": "inst-2"})
    await provider(api).create(await offer(ON_DEMAND), spec(), bid=None)
    (body,) = api.bodies("POST", "/v1/instances")
    assert body["is_spot"] is False and body["contract"] == "PAY_AS_YOU_GO"


async def test_no_capacity_is_an_offer_gone_and_its_script_is_deleted():
    api = Api(**{"POST /v1/instances": httpx.Response(503, json={"code": "service_unavailable", "message": "none left"}),
                 "GET /v1/instances": []})
    with pytest.raises(OfferGone):
        await provider(api).create(await offer(), spec(), bid=None)
    assert "DELETE /v1/scripts/script-1" in api.paths()


async def test_a_quota_refusal_says_so():
    api = Api(**{"POST /v1/instances": httpx.Response(403, json={"code": "forbidden_action", "message": "quota exceeded"})})
    with pytest.raises(ProviderError, match="quota"):
        await provider(api).create(await offer(), spec(), bid=None)


async def test_an_account_without_an_ssh_key_is_told_what_to_do():
    api = Api(**{"GET /v1/ssh-keys": []})
    with pytest.raises(ProviderError, match="add any key"):
        await provider(api).create(await offer(), spec(), bid=None)
    assert "POST /v1/instances" not in api.paths() and "POST /v1/scripts" not in api.paths()


async def test_a_create_with_no_answer_deletes_whatever_the_label_names():
    def create(request):
        raise httpx.ReadTimeout("no answer", request=request)

    listed = [{"id": "inst-9", "description": "gpm-test/rented-abc123", "status": "provisioning",
               "location": "FIN-01", "instance_type": "1H100.80S.30V", "os_volume_id": "vol-9",
               "startup_script_id": "script-1"}]
    api = Api(**{"POST /v1/instances": create, "GET /v1/instances": listed,
                 "GET /v1/instances/inst-9": listed[0],
                 "PUT /v1/instances": [{"instanceId": "inst-9", "action": "delete", "status": "success"}]})
    with pytest.raises(ProviderUnavailable, match="inst-9, which has been deleted"):
        await provider(api).create(await offer(), spec(), bid=None)
    (delete,) = api.bodies("PUT", "/v1/instances")
    assert delete == {"action": "delete", "id": "inst-9", "delete_permanently": True, "volume_ids": ["vol-9"]}


async def test_a_volume_is_refused_until_supported():
    with pytest.raises(ProviderError, match="not supported"):
        from gpm_server.providers.base import VolumeSpec
        await provider(Api()).create(await offer(), spec(volume=VolumeSpec(mount="/m", label="v", size_gb=10)), None)


# --- the machine's start-up ---


def test_the_machine_runs_the_image_as_the_host_and_publishes_only_its_ssh_daemon():
    script = VerdaProvider().vm_script(spec(env={"GPM_WORKERS": "6", "ODD": "it's $HOME"}))
    run = next(line for line in script.splitlines() if line.startswith("docker run"))
    assert run.count(" -p ") == 1 and "-p 2222:22" in run, "only the container's SSH daemon is published"
    assert "--gpus all" in run and "vllm/vllm-openai:latest" in run
    assert "-e GPM_WORKERS=6" in run and """-e 'ODD=it'"'"'s $HOME'""" in run
    import base64

    encoded = next(line for line in script.splitlines() if "base64 -d" in line).split("'")[1]
    inside = base64.b64decode(encoded).decode()
    assert "echo engine" in inside and "sshd" in inside
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


# --- state ---


async def test_only_this_pools_live_instances_are_listed():
    rows = [
        {"id": "a", "description": "gpm-test/rented-1", "status": "running", "location": "FIN-01", "instance_type": "1H100.80S.30V"},
        {"id": "b", "description": "gpm-other/rented-2", "status": "running", "location": "FIN-01", "instance_type": "1H100.80S.30V"},
        {"id": "c", "description": "gpm-test/rented-3", "status": "discontinued", "location": "FIN-01", "instance_type": "1H100.80S.30V"},
        {"id": "d", "description": "gpm-test/rented-4", "status": "offline", "location": "ICE-01", "instance_type": "2H100.80S.80V"},
    ]
    api = Api(**{"GET /v1/instances": rows})
    found = await provider(api).list_instances("gpm-test/")
    assert [(i.instance_id, i.machine_id) for i in found] == [("a", "FIN-01:1H100.80S.30V"), ("d", "ICE-01:2H100.80S.80V")]
    # Not asked by tag: an instance whose tag did not stick still bills, and must be seen.
    assert "tag" not in next(r for r in api.seen if r.url.path == "/v1/instances").url.params


@pytest.mark.parametrize("said, state", [
    ("running", InstanceState.RUNNING), ("provisioning", InstanceState.SCHEDULING),
    ("ordered", InstanceState.SCHEDULING), ("offline", InstanceState.STOPPED),
    ("installation_failed", InstanceState.STOPPED), ("discontinued", InstanceState.GONE),
])
async def test_status(said, state):
    api = Api(**{"GET /v1/instances/x": {"id": "x", "status": said, "price_per_hour": 1.945, "startup_script_id": "s"}})
    status = await provider(api).status(Instance("x"))
    assert status.state == state and status.bid_hourly == 1.945
    assert status.startup_material is None, "unverified: a False would end the host (D65)"
    assert status.stopped_by_provider is (True if said == "discontinued" else None)


async def test_an_instance_the_provider_does_not_know_is_gone():
    assert (await provider(Api()).status(Instance("nope"))).state == InstanceState.GONE


async def test_the_pool_reaches_the_host_container_on_its_published_port():
    api = Api(**{"GET /v1/instances/x": {"id": "x", "status": "running", "ip": "203.0.113.7"}})
    info = await provider(api).connection(Instance("x"))
    assert (info.ssh_host, info.ssh_port, info.ssh_user, info.public_url) == ("203.0.113.7", 2222, "root", None)


async def test_destroy_deletes_the_disk_for_good_and_the_script():
    api = Api(**{"GET /v1/instances/x": {"id": "x", "status": "running", "os_volume_id": "vol-1", "startup_script_id": "script-1"},
                 "PUT /v1/instances": [{"instanceId": "x", "action": "delete", "status": "success"}]})
    await provider(api).destroy(Instance("x"))
    (body,) = api.bodies("PUT", "/v1/instances")
    assert body == {"action": "delete", "id": "x", "delete_permanently": True, "volume_ids": ["vol-1"]}
    assert "DELETE /v1/scripts/script-1" in api.paths()


async def test_destroy_is_idempotent_and_a_failed_delete_is_said():
    await provider(Api()).destroy(Instance("gone"))
    api = Api(**{"GET /v1/instances/x": {"id": "x", "status": "running"},
                 "PUT /v1/instances": httpx.Response(207, json=[{"instanceId": "x", "action": "delete", "status": "error",
                                                                 "error": "locked", "statusCode": 409}])})
    with pytest.raises(ProviderError, match="not deleted"):
        await provider(api).destroy(Instance("x"))


async def test_it_is_never_parked():
    verda = provider(Api())
    assert not verda.capabilities.parkable
    with pytest.raises(ProviderError, match="keeps billing"):
        await verda.stop(Instance("x"))


# --- money ---


async def test_a_balance_in_another_currency_is_not_read_as_dollars():
    api = Api(**{"GET /v1/balance": {"amount": 40.0, "currency": "eur"}})
    status = await provider(api).account()
    assert status.credential_valid and status.credit_remaining is None and "EUR" in status.detail


def test_it_declares_what_it_cannot_do():
    caps = VerdaProvider.capabilities
    assert caps.interruptible and not caps.self_terminate and not caps.parkable
    assert not caps.direct_port_mapping and not caps.machine_ids_are_machines and not caps.volumes


def test_the_console_is_told_what_the_credential_is():
    from gpm_server.providers import plugin_presentation

    shown = plugin_presentation("verda")
    assert "CLIENT_ID:CLIENT_SECRET" in shown["credential_format"]
    assert plugin_presentation("runpod")["credential_format"] is None


# --- after the review ---


async def test_a_gateways_503_is_an_answer_not_known_and_nothing_is_left():
    listed = [{"id": "inst-7", "description": "gpm-test/rented-abc123", "status": "provisioning",
               "location": "FIN-01", "instance_type": "1H100.80S.30V", "os_volume_id": "vol-7"}]
    api = Api(**{"POST /v1/instances": httpx.Response(503, text="Bad gateway"), "GET /v1/instances": listed,
                 "GET /v1/instances/inst-7": listed[0],
                 "PUT /v1/instances": [{"instanceId": "inst-7", "action": "delete", "status": "success"}]})
    with pytest.raises(ProviderUnavailable, match="inst-7, which has been deleted"):
        await provider(api).create(await offer(), spec(), bid=None)


async def test_an_answer_that_is_not_an_instance_is_never_taken_for_gone():
    api = Api(**{"GET /v1/instances/x": {"unexpected": True}})
    with pytest.raises(ProviderUnavailable, match="not an instance"):
        await provider(api).status(Instance("x"))


async def test_a_new_credential_drops_the_old_accounts_token_and_keys():
    api = Api(**{"POST /v1/instances": "inst-1"})
    verda = provider(api)
    await verda.create(await offer(), spec(), bid=None)
    verda.set_credential("vc_other_id:vc_other_secret")
    verda._client = httpx.AsyncClient(transport=httpx.MockTransport(api), base_url=BASE)
    assert verda._token is None and verda._ssh_keys is None
    await verda.create(await offer(), spec(), bid=None)
    exchanges = api.bodies("POST", "/v1/oauth2/token")
    assert [e["client_id"] for e in exchanges] == ["vc_test_id", "vc_other_id"]
    assert [r.url.path for r in api.seen].count("/v1/ssh-keys") == 2


async def test_a_token_asked_for_under_an_old_credential_is_not_kept():
    verda = provider(Api())

    async def exchange(request):
        verda.set_credential("vc_other_id:vc_other_secret")  # replaced while the token was asked for
        return httpx.Response(200, json={"access_token": "tok-old-account", "expires_in": 3600})

    verda._client = httpx.AsyncClient(transport=httpx.MockTransport(_async_answer(exchange)), base_url=BASE)
    await verda._access_token()
    assert verda._token is None


def _async_answer(handler):
    async def transport(request):
        return await handler(request)
    return transport


async def test_prices_in_another_currency_are_never_read_as_dollars():
    euro = [{**row, "currency": "eur"} for row in TYPES]
    offers = await provider(Api(**{"GET /v1/instance-types": euro})).search_offers(SPOT)
    assert offers == []


async def test_a_machine_already_being_deleted_is_not_asked_again():
    api = Api(**{"GET /v1/instances/x": {"id": "x", "status": "deleting"}})
    await provider(api).destroy(Instance("x"))
    assert "PUT /v1/instances" not in api.paths()


def test_no_line_of_the_operators_start_up_can_run_on_the_machine_itself():
    import base64

    script = VerdaProvider().vm_script(spec(onstart="echo one\nGPM_HOST_EOF\nrm -rf /important"))
    assert "rm -rf /important" not in script, "only base64 of it is on the machine"
    encoded = next(line for line in script.splitlines() if "base64 -d" in line).split("'")[1]
    assert "rm -rf /important" in base64.b64decode(encoded).decode()
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_an_image_name_that_reads_as_an_option_is_refused():
    with pytest.raises(ProviderError, match="not an image name"):
        VerdaProvider().vm_script(spec(image="--privileged"))


def test_the_host_keys_are_made_once_and_kept_across_a_restart():
    from gpm_server.providers.hostscript import SSH_DAEMON

    assert ".gpm-host-keys" in SSH_DAEMON


OLD = "2026-10-01T00:00:00Z"
NEW = "2999-01-01T00:00:00Z"


async def test_what_gone_machines_left_behind_is_deleted_and_nothing_else():
    scripts = [{"id": "s-old", "name": "gpm-test/rented-1", "created_at": OLD, "script": ""},
               {"id": "s-live", "name": "gpm-test/rented-2", "created_at": OLD, "script": ""},
               {"id": "s-new", "name": "gpm-test/rented-3", "created_at": NEW, "script": ""},
               {"id": "s-other", "name": "someone-elses", "created_at": OLD, "script": ""}]
    volumes = [{"id": "v-old", "name": "gpm-test-rented-1-os", "status": "detached", "is_os_volume": True, "created_at": OLD},
               {"id": "v-data", "name": "gpm-test-data", "status": "detached", "is_os_volume": False, "created_at": OLD},
               {"id": "v-other", "name": "mine-os", "status": "detached", "is_os_volume": True, "created_at": OLD}]
    live = [{"id": "a", "description": "gpm-test/rented-2", "status": "running", "startup_script_id": "s-live"}]
    api = Api(**{"GET /v1/instances": live, "GET /v1/scripts": scripts, "GET /v1/volumes": volumes,
                 "DELETE /v1/scripts/s-old": None, "DELETE /v1/volumes/v-old": None})
    verda = provider(api)
    await verda.list_instances("gpm-test/")
    deleted = [p for p in api.paths() if p.startswith("DELETE")]
    assert deleted == ["DELETE /v1/scripts/s-old", "DELETE /v1/volumes/v-old"]
    assert api.bodies("DELETE", "/v1/volumes/v-old") == [{"is_permanent": True}]
    await verda.list_instances("gpm-test/")
    assert [p for p in api.paths() if p.startswith("DELETE")] == deleted, "at most every tidy_interval_s"


async def test_a_failed_tidy_never_fails_the_listing():
    api = Api(**{"GET /v1/instances": [], "GET /v1/scripts": httpx.Response(500, json={"code": "server_error", "message": "x"})})
    assert await provider(api).list_instances("gpm-test/") == []


async def test_the_first_tidy_runs_on_a_machine_booted_moments_ago(monkeypatch):
    """The monotonic clock counts from boot: on a machine up for a minute, a tidy timed from
    zero would wait out its interval before ever running once."""
    import gpm_server.providers.verda as verda_module

    monkeypatch.setattr(verda_module.time, "monotonic", lambda: 60.0)
    scripts = [{"id": "s-old", "name": "gpm-test/rented-1", "created_at": OLD, "script": ""}]
    api = Api(**{"GET /v1/instances": [], "GET /v1/scripts": scripts, "GET /v1/volumes": [],
                 "DELETE /v1/scripts/s-old": None})
    await provider(api).list_instances("gpm-test/")
    assert [p for p in api.paths() if p.startswith("DELETE")] == ["DELETE /v1/scripts/s-old"]
