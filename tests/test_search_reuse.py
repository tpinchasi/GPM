"""One search's answer reused for the same question for a short while (D206).

Found live 2026-11-25: creating one workload asked the same market about four times in thirty
seconds — the plan shown, the plan made, the rental — and the provider billed every row of each
against its daily quota. Against the fake provider; nothing here needs a cloud account.
"""

import pytest
from gpm_server.config import OfferPolicy
from gpm_server.db import Database
from gpm_server.ledger import EventLog, LeaseStore, SpendLedger
from gpm_server.providers import FakeProvider, default_offer
from gpm_server.supervisor.renting import OFFER_REUSE_S, Fleet
from test_renting import make_config, open_lease


@pytest.fixture
def fleet(tmp_path):
    database = Database(tmp_path / "gpm.sqlite3")
    config = make_config()
    made = Fleet(config, config.rented, FakeProvider(), LeaseStore(database), EventLog(database),
                 SpendLedger(database))
    try:
        yield made
    finally:
        database.close()


def searches(fleet) -> int:
    return fleet.provider.calls.count("search_offers")


async def test_the_same_question_within_the_window_asks_the_provider_once(fleet):
    first = await fleet._offers()
    again = await fleet._offers()
    assert searches(fleet) == 1
    assert [o.offer_id for o in again] == [o.offer_id for o in first]


async def test_a_different_question_is_asked(fleet):
    await fleet._offers()
    await fleet._offers(OfferPolicy(min_gpu_memory_gb=48, min_disk_gb=10, max_all_in_hourly=0.60))
    assert searches(fleet) == 2


async def test_an_answer_older_than_the_window_is_asked_again(fleet):
    await fleet._offers()
    at, asked, answer = fleet.searching[fleet.connection].kept
    fleet.searching[fleet.connection].kept = (at - OFFER_REUSE_S - 1, asked, answer)
    await fleet._offers()
    assert searches(fleet) == 2


async def test_a_fresh_search_always_asks(fleet):
    """A price about to be bid on — winning back a machine just outbid on — is the market's now."""
    await fleet._offers()
    fleet.provider.offers = [default_offer(offer_id="newer", min_bid_hourly=0.30)]
    found = await fleet._offers(fresh=True)
    assert searches(fleet) == 2
    assert [o.offer_id for o in found] == ["newer"]


async def test_a_rental_tried_forgets_what_was_found_there(fleet):
    """The offer just taken, or found gone, is not offered again from memory."""
    open_lease(fleet)
    await fleet.pass_once(ready_workers_higher_tiers=1, idle_seconds={})
    assert len(fleet.hosts) == 1
    assert fleet.searching[fleet.connection].kept is None
    before = searches(fleet)
    await fleet._offers()
    assert searches(fleet) == before + 1


async def test_a_failed_search_is_not_kept_and_the_next_one_asks(fleet):
    fleet.provider.unavailable = True
    assert await fleet._offers() == []
    assert fleet.searching[fleet.connection].kept is None
    fleet.provider.unavailable = False
    assert await fleet._offers()
    assert searches(fleet) == 2


async def test_a_fresh_search_that_fails_leaves_nothing_kept_beside_its_error(fleet):
    await fleet._offers()
    fleet.provider.unavailable = True
    assert await fleet._offers(fresh=True) == []
    assert await fleet._offers() == [], "the provider was asked again, not answered from before its failure"
    assert fleet.last_offer_error


async def test_reuse_set_to_zero_asks_every_time(fleet):
    fleet.offer_reuse_s = 0
    await fleet._offers()
    await fleet._offers()
    assert searches(fleet) == 2
