"""A model too large for one card, split across a group of cards on one machine (D114).

A profile names how many cards each copy of its models spans. The search asks for that many
cards, in whole groups, and for the share of each model a card holds; the host remembers it for
every relaunch; the agent passes it to the launcher as a number. One card per copy is D107.

Against the fake provider; nothing here spends money or needs a GPU.
"""

import math

import httpx
import pytest
import test_model_profiles
import yaml
from gpm_agent import vllm_launch
from gpm_server import sizing
from gpm_server.providers import default_offer
from gpm_server.strategies import reject_reasons
from gpm_server.supervisor import agents
from test_model_profiles import ADMIN_KEY, BIG, BIG_FP4, EMBED, EMBED_REPO, config, open_lease, put

# The profile tests' fixtures: a fleet on the fake provider, and a pool served from a file.
make_fleet = test_model_profiles.make_fleet
pool = test_model_profiles.pool

SPLIT = {"chat-fp8": 2}


def four_card_offer(offer_id="o-4", machine_id="m-4", **more):
    return default_offer(offer_id, machine_id, hardware="4x FakeGPU 24GB", gpus=4, gpu_memory_gb=24.0, **more)


# --- the file's rules ---


def test_a_split_of_a_profile_that_is_not_defined_is_refused():
    with pytest.raises(ValueError, match="split_across_cards names 'nope'"):
        config(split_across_cards={"nope": 2})


@pytest.mark.parametrize("cards", [0, 3, 16])
def test_a_split_that_is_not_1_2_4_or_8_is_refused(cards):
    with pytest.raises(ValueError, match="1, 2, 4 or 8"):
        config(split_across_cards={"chat": cards})


def test_a_split_on_an_engine_that_cannot_is_refused():
    ollama_profiles = {"chat": {BIG: "big:26b"}}
    with pytest.raises(ValueError, match="'ollama', which cannot be told to split"):
        config(engine="ollama", image="ollama/ollama:latest", profiles=ollama_profiles, split_across_cards={"chat": 2})


def test_a_split_with_the_operators_own_start_is_refused():
    with pytest.raises(ValueError, match="rented.engine_start replaces it"):
        config(split_across_cards={"chat": 2}, engine_start="vllm serve /models")


def test_a_split_of_one_is_allowed_anywhere():
    cfg = config(engine="ollama", image="ollama/ollama:latest", profiles={"chat": {BIG: "big:26b"}},
                 split_across_cards={"chat": 1})
    assert cfg.rented.cards_per_copy("chat") == 1


# --- what the search asks for ---


def test_the_sizing_rule_with_a_split_is_the_launchers():
    """A card of exactly the searched size starts the split set; a smaller one is refused."""
    needs = sizing.needs_for({BIG: 28.0}, 2)
    assert needs.cards_per_copy == 2
    card = int(needs.card_memory_gb * 1e9)
    assert vllm_launch.memory_plan([int(28.0e9)], card, 2)[1] is None
    assert vllm_launch.memory_plan([int(28.0e9)], card - int(2e9), 2)[1] is not None
    assert needs.disk_gb == sizing.needs_for({BIG: 28.0}).disk_gb, "the disk holds each model once"


async def test_the_search_asks_for_the_share_a_card_holds_and_whole_groups_of_cards(make_fleet):
    fleet = make_fleet(config(rent=("chat-fp8",), split_across_cards=SPLIT))
    policy = fleet.next_host_policy()
    assert policy.gpus_multiple_of == 2
    assert policy.min_gpu_memory_gb == math.ceil((28.0 * 1.1 / 2 + sizing.CACHE_RESERVE_GB) / 0.9)
    assert policy.min_disk_gb == math.ceil(28.0 * 1.1 + 10)


async def test_an_unsplit_profile_searches_as_before(make_fleet):
    fleet = make_fleet(config(rent=("chat-fp8",)))
    assert fleet.next_host_policy().gpus_multiple_of == 1


def test_a_machine_whose_cards_do_not_make_whole_groups_is_rejected_saying_so():
    cfg = config(rent=("chat-fp8",), split_across_cards=SPLIT)
    policy = cfg.rented.policy_in_force.model_copy(update={"gpus_multiple_of": 2})
    one, three, four = (default_offer(gpus=n) for n in (1, 3, 4))
    assert any(r.startswith("cards: 1 is not a whole number of groups of 2") for r in reject_reasons(one, policy))
    assert any(r.startswith("cards: 3 is not") for r in reject_reasons(three, policy))
    assert not any(r.startswith("cards:") for r in reject_reasons(four, policy))


async def test_the_provider_is_asked_only_for_machines_with_enough_cards(make_fleet):
    fleet = make_fleet(
        config(rent=("chat-fp8",), split_across_cards=SPLIT),
        offers=[default_offer("o-1", "m-1"), four_card_offer()],
    )
    offers = await fleet._offers(fleet.next_host_policy())
    assert [o.offer_id for o in offers] == ["o-4"]


async def test_the_preview_says_what_was_searched_and_why(make_fleet):
    fleet = make_fleet(config(rent=("chat-fp8",), split_across_cards=SPLIT), offers=[four_card_offer()])
    preview = await fleet.market_preview(hours=1)
    nxt = preview["next_host"]
    assert nxt["needs"]["cards_per_copy"] == 2
    assert nxt["searched"]["gpus_multiple_of"] == 2 and nxt["typed"]["gpus_multiple_of"] == 1
    assert preview["best"][0]["gpus"] == 4


# --- the host remembers it ---


async def test_a_host_bought_as_a_split_profile_keeps_its_split_across_a_restart(make_fleet):
    fleet = make_fleet(config(rent=("chat-fp8",), split_across_cards=SPLIT), offers=[four_card_offer()])
    host = await fleet.rent_one(open_lease(fleet), [])
    assert host.cards_per_copy == 2
    ref = fleet.published_ref(host)
    assert ref["cards_per_copy"] == 2


def test_the_agent_is_asked_for_a_split_only_when_there_is_one():
    """An agent from before splitting refuses a name it does not know; one card per copy is
    what every agent already does, so nothing is sent for it."""
    assert agents.wanted_engine_settings(8, 1) == {"workers": 8, "models_held": 1}
    assert agents.wanted_engine_settings(8, 1, 1) == {"workers": 8, "models_held": 1}
    assert agents.wanted_engine_settings(8, 1, 2) == {"workers": 8, "models_held": 1, "cards_per_copy": 2}


# --- saved from the console ---


def test_a_split_is_saved_with_the_profiles_and_shown_with_their_needs(pool):
    supervisor, url, path = pool
    answer = put(url, {"profiles": {"chat": {BIG: BIG_FP4}, "embed": {EMBED: EMBED_REPO}},
                       "rent": ["chat", "embed"], "split": {"chat": 2, "embed": 1}})
    assert answer.status_code == 200, answer.text
    written = yaml.safe_load(path.read_text())
    assert written["rented"]["split_across_cards"] == {"chat": 2}, "one card per copy is not written"
    assert supervisor.config.rented.cards_per_copy("chat") == 2

    status = httpx.get(f"{url}/pool/status", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).json()
    chat = next(p for p in status["engine"]["profiles"] if p["name"] == "chat")
    assert chat["cards_per_copy"] == 2 and chat["needs"]["cards_per_copy"] == 2
    assert chat["needs"]["card_memory_gb"] == math.ceil((17.0 * 1.1 / 2 + sizing.CACHE_RESERVE_GB) / 0.9)
    assert status["engine"]["offers"]["vllm"]["splits_across_cards"] is True
    assert status["engine"]["offers"]["ollama"]["splits_across_cards"] is False


def test_profiles_saved_without_a_split_keep_it_and_a_removed_profile_takes_its_split(pool):
    """A console from before splitting sends none; the file's split stays for the profiles it
    still names, and does not outlive a profile removed — which the file would refuse."""
    supervisor, url, path = pool
    assert put(url, {"profiles": {"chat": {BIG: BIG_FP4}, "embed": {EMBED: EMBED_REPO}},
                     "rent": ["chat", "embed"], "split": {"chat": 2}}).status_code == 200
    assert put(url, {"profiles": {"chat": {BIG: BIG_FP4}, "embed": {EMBED: EMBED_REPO}},
                     "rent": ["chat", "embed"]}).status_code == 200
    assert yaml.safe_load(path.read_text())["rented"]["split_across_cards"] == {"chat": 2}
    answer = put(url, {"profiles": {"embed": {EMBED: EMBED_REPO}, "big": {BIG: BIG_FP4}}, "rent": ["embed", "big"]})
    assert answer.status_code == 200, answer.text
    assert yaml.safe_load(path.read_text())["rented"]["split_across_cards"] == {}


@pytest.mark.parametrize("split", [{"chat": 3}, {"chat": True}, {"other": 2}, ["chat"]])
def test_a_split_that_cannot_be_is_refused_and_nothing_is_written(pool, split):
    supervisor, url, path = pool
    before = path.read_text()
    answer = put(url, {"profiles": {"chat": {BIG: BIG_FP4}, "embed": {EMBED: EMBED_REPO}},
                       "rent": ["chat", "embed"], "split": split})
    assert answer.status_code == 400, answer.text
    assert path.read_text() == before
