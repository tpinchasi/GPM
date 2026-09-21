"""Every scenario, as a check — the gate a version passes before it is merged.

Slow on purpose: each one runs the real router and supervisor against a market that moves, for
long enough that windows elapse and hosts are paused and given up. Opt-in, like the integration
tests, and run in CI as its own job.

    uv run pytest -m simulation
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalogue import CATALOGUE  # noqa: E402
from scenarios import check_invariants, run  # noqa: E402

# Minutes by design: phases have to be long enough for windows to elapse and hosts to be
# paused, woken and given up.
pytestmark = [pytest.mark.simulation, pytest.mark.timeout(600)]


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_scenario(name: str) -> None:
    scenario = CATALOGUE[name]
    summary = run(scenario, verbose=False)
    try:
        check_invariants(summary)
        if scenario.expects is not None:
            scenario.expects(summary)
    except AssertionError:
        # The timeline is the evidence: without it a failure says what broke but never when.
        print(f"\n=== {name}: {scenario.what_it_shows}")
        print(summary["report"].render())
        raise
