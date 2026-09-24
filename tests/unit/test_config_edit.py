"""Editing the pool's file in place, keeping every comment the operator wrote (D51).

Two operations the engine and placement editor needs beyond changing one line (D98): replacing a
value that was a whole block — a model's list of builds — and changing one item of a list — the
host whose id is `laptop`. Loading the YAML and writing it back would do both in one line of code
and delete every comment in the file, which is the price D51 refuses.
"""

import pytest
import yaml
from gpm_server.configplan import CannotEdit, set_in_list_item, set_values

FILE = """\
pool:
  name: test   # the pool's own name
  model_set: [a, b]

# Logical names, and the builds that serve them.
catalog:
  a:
    variants:
      - { tag: "a-mlx", requires: [apple-silicon] }   # the laptop's
      - { tag: "a" }
  b:
    variants:
      - { tag: "b" }

hosts:
  - id: laptop
    kind: local
    workers: 3   # measured
    transport: { type: http, base_url: "http://127.0.0.1:11434" }
  - id: desk
    kind: local
    transport: { type: http, base_url: "http://127.0.0.1:11435" }

rented:
  engine_start: "nohup ollama serve &"   # the image does not start it
"""


def changed_lines(before: str, after: str) -> list[str]:
    import difflib

    return [
        line for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]


# --- replacing a value that was a whole block ---


def test_a_block_is_replaced_whole_and_nothing_else_moves():
    after = set_values(FILE, ("catalog", "a"), {"variants": [{"tag": "a", "engine": "ollama"}]})

    parsed = yaml.safe_load(after)
    assert parsed["catalog"]["a"]["variants"] == [{"tag": "a", "engine": "ollama"}]
    assert parsed["catalog"]["b"]["variants"] == [{"tag": "b"}], "the next model is untouched"
    assert len(changed_lines(FILE, after)) == 4, "three old lines out, one new line in"


def test_replacing_a_block_used_to_leave_the_old_one_behind():
    """Only the key's own line was rewritten, so the old list sat under the new value — a file
    that no longer parsed, or parsed as something else."""
    after = set_values(FILE, ("catalog", "a"), {"variants": []})
    assert yaml.safe_load(after)["catalog"]["a"] == {"variants": []}
    assert '"a-mlx"' not in after


def test_the_comments_around_a_replaced_block_are_kept():
    after = set_values(FILE, ("catalog", "a"), {"variants": [{"tag": "x"}]})
    for comment in ("# the pool's own name", "# Logical names", "# measured", "# the image does not start it"):
        assert comment in after, comment


def test_a_one_line_value_is_still_changed_in_place_with_its_comment():
    after = set_values(FILE, ("rented",), {"engine_start": None})
    assert "  engine_start: null   # the image does not start it" in after
    assert yaml.safe_load(after)["rented"]["engine_start"] is None


def test_a_value_written_after_a_key_that_held_a_block_is_separated_by_a_space():
    """`variants:[…]` is not YAML; a key that held a block ends at its colon."""
    after = set_values(FILE, ("catalog", "b"), {"variants": [{"tag": "b2"}]})
    assert '    variants: [{ tag: "b2" }]' in after


# --- one item of a list ---


def test_one_host_is_changed_and_the_other_is_not():
    after = set_in_list_item(FILE, "hosts", "id", "laptop", {"models": ["a", "b"]})
    hosts = yaml.safe_load(after)["hosts"]
    assert hosts[0]["models"] == ["a", "b"]
    assert "models" not in hosts[1]
    assert changed_lines(FILE, after) == ['+    models: ["a", "b"]']


def test_a_key_the_host_already_has_is_changed_in_place():
    after = set_in_list_item(FILE, "hosts", "id", "laptop", {"workers": 4})
    assert "    workers: 4   # measured" in after
    assert yaml.safe_load(after)["hosts"][0]["workers"] == 4


def test_the_key_on_the_items_first_line_is_found_too():
    after = set_in_list_item(FILE, "hosts", "id", "desk", {"kind": "fixed-remote"})
    assert yaml.safe_load(after)["hosts"][1]["kind"] == "fixed-remote"


def test_a_value_can_be_cleared():
    once = set_in_list_item(FILE, "hosts", "id", "laptop", {"models": ["a"]})
    twice = set_in_list_item(once, "hosts", "id", "laptop", {"models": None})
    assert yaml.safe_load(twice)["hosts"][0]["models"] is None


@pytest.mark.parametrize("match", ["nobody", ""])
def test_a_host_that_is_not_there_is_refused_rather_than_guessed(match):
    with pytest.raises(CannotEdit):
        set_in_list_item(FILE, "hosts", "id", match, {"models": ["a"]})


def test_an_item_written_on_one_line_is_refused_rather_than_rewritten():
    one_line = FILE.replace(
        "  - id: desk\n    kind: local\n    transport: { type: http, base_url: \"http://127.0.0.1:11435\" }\n",
        "  - { id: desk, kind: local }\n",
    )
    with pytest.raises(CannotEdit, match="one line"):
        set_in_list_item(one_line, "hosts", "id", "desk", {"models": ["a"]})
