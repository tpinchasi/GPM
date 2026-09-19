"""The threat model, checked against the implementation.

docs/threat-model.md, and the release checklist's "threat model re-read line by line against
the implementation". Most threats are already covered by the tests named in that document's
table; this file holds the ones best expressed as **structural invariants** — properties that
would be silently lost by an innocent-looking refactor, and that no behavioural test would
notice.
"""

import ast
import pathlib

import pytest
from gpm_server import deadman
from gpm_server.config import PoolConfig
from gpm_server.providers import FakeProvider, VastProvider
from gpm_server.router import app as router_app

SERVER = pathlib.Path(router_app.__file__).resolve().parent.parent


def source_of(module) -> str:
    return pathlib.Path(module.__file__).read_text()


# --- T5: the account credential never reaches a rented host ---


@pytest.mark.parametrize("provider", [FakeProvider(), VastProvider()])
def test_nothing_a_host_receives_can_carry_the_account_credential(provider):
    """Everything the pool writes onto an instance, scanned for anything secret-shaped."""
    onstart = deadman.onstart_script(
        provider.self_terminate_command("destroy"),
        window_s=1200,
        public_key="ssh-ed25519 AAAAPublicHalfOnly pool",
    )
    import re

    referenced = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", onstart))
    secret_shaped = {
        name for name in referenced
        if any(word in name for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
    }
    # Only the provider's own per-instance credential, which can act on that instance alone.
    assert secret_shaped <= {"CONTAINER_API_KEY"}, secret_shaped
    assert "VAST_API_KEY" not in onstart


def test_the_provider_reads_its_credential_from_the_environment_only():
    """Not from configuration, so it cannot end up in a file someone commits."""
    text = source_of(__import__("gpm_server.providers.vast", fromlist=["x"]))
    assert "os.environ.get(self.api_key_env)" in text
    # The pool's config models have no field that could hold it.
    assert "api_key" not in PoolConfig.model_json_schema()["$defs"]["RentedConfig"]["properties"]


# --- T6: the pool's own keys are never exposed to a host ---


def test_the_app_key_is_stripped_before_a_request_is_forwarded():
    assert "authorization" in router_app._DROP_UPSTREAM
    # And the whole pool dialect goes too, so nothing of ours leaks into the engine's API.
    text = source_of(router_app)
    assert 'key.lower().startswith("x-gpm-")' in text


# --- T10: no request can trigger a pull or a model load ---


def test_the_request_path_cannot_reach_a_pull_or_a_load():
    """Structural: the router module never names the engine operations that fetch or load a
    model. Those belong to the supervisor, at prepare time, for tags named in configuration."""
    tree = ast.parse(source_of(router_app))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"pull", "load_and_pin"}


def test_only_the_supervisor_pulls_and_only_for_configured_tags():
    from gpm_server.supervisor import renting

    text = source_of(renting)
    assert "engine.pull(client, tag)" in text
    # The tags come from the pool's own model set resolved through the catalog, never from a
    # request and never from a guessed name.
    assert "tags = sorted(self.required_tags)" in text


# --- T11: a strategy cannot spend past the limits ---


def test_every_bid_a_strategy_returns_passes_through_the_supervisors_clamp():
    """Strategies are advisory; the caps are not. Every place a bid becomes an action must go
    through `_cap_bid`, which clamps to both ceilings after the strategy has had its say."""
    from gpm_server.supervisor import renting

    tree = ast.parse(source_of(renting))
    priced, capped = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "price_bid":
                priced += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("_cap_bid", "_refuse_bid_for_burn"):
                capped += 1
    assert priced >= 3  # renting, re-bidding after eviction, restarting a parked host
    assert capped >= priced  # each of them clamped, and the burn cap re-checked too


def test_the_strategies_module_performs_no_io():
    """Pure functions of their arguments: no clock, no network, no database. That is what
    makes them testable, replayable and safe to let an operator swap."""
    from gpm_server import strategies

    tree = ast.parse(source_of(strategies))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"httpx", "asyncio", "subprocess", "sqlite3", "socket", "requests", "random", "time"}
    assert not imported & forbidden, imported & forbidden


# --- T14: only plug-ins named in configuration load, by one mechanism for everyone ---


def test_first_party_plugins_load_through_the_same_entry_points_as_anyone_elses():
    """No privileged path: the shipped provider and engine are discovered exactly the way a
    third party's would be, so the mechanism others depend on is the one we use ourselves."""
    from gpm_server.engines.base import available_engines
    from gpm_server.providers.base import available_providers

    assert {"fake", "vast"} <= set(available_providers())
    assert "ollama" in available_engines()
    # Discovered, not hardcoded: every one arrives as a real entry point.
    for point in list(available_providers().values()) + list(available_engines().values()):
        assert point.group in ("gpm.providers", "gpm.engines")


def test_a_plugin_configuration_does_not_name_is_never_loaded():
    from gpm_server.engines.base import EngineNotFound, get_engine
    from gpm_server.providers.base import ProviderNotFound, get_provider

    with pytest.raises(ProviderNotFound, match="installed"):
        get_provider("not-installed", {})
    with pytest.raises(EngineNotFound, match="installed"):
        get_engine("not-installed")


# --- T16: no prompt or completion text is ever recorded ---


def test_the_request_log_has_no_field_that_could_hold_text_from_a_request():
    from gpm_server.db import RequestRecord

    fields = set(RequestRecord.__dataclass_fields__)
    assert not fields & {"prompt", "completion", "messages", "response", "content", "body"}


def test_the_decision_log_records_numbers_and_identifiers_not_bodies():
    from gpm_server.supervisor import renting

    text = source_of(renting)
    # Every recorded event passes `numbers=`; none passes a request or response body.
    assert "numbers=" in text
    assert "body=" not in text


# --- T19: another local user cannot read the pool's state ---


def test_everything_written_is_owner_only():
    from gpm_server import db, keys

    assert "0o600" in source_of(db)
    assert "0o600" in source_of(keys)
    assert "S_IRGRP | stat.S_IROTH" in source_of(db)


# --- the non-goals are stated where a user will see them, not only in the threat model ---


def test_the_deliberate_non_goals_are_in_user_facing_documentation():
    """Spec: 'Each is stated in user-facing documentation, not only here.' The one that matters
    most is that a rented host's operator can read everything sent to it."""
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("SECURITY.md", "docs/quickstart-renting.md"):
        text = (root / name).read_text().lower()
        assert "read everything sent to it" in text, name
    assert "not sandboxed" in (root / "SECURITY.md").read_text().lower()
