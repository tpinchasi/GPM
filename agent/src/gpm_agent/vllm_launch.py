"""Start vLLM on this machine for whatever models are on disk (D97).

vLLM serves the model it was started with, and nothing else. So on a machine the pool rented,
"loading a model" means: the agent fetched its weights into the models directory, and the engine
is started again to pick them up. This module is that start. It runs on the machine, from the
agent's archive, when the machine's own restart script calls it — never over the agent's
protocol, and never with anything the pool said other than bounded numbers in the environment.

Two shapes, matching the pool's placement (D94, D96):

- **One model**: one vLLM process on the engine port, given nine tenths of the accelerator.
- **Several, behind the router**: one vLLM process per model on the ports above the engine port,
  the accelerator's memory split between them by the size of each model's weights (with a floor,
  so a small model still has room to run), and the router on the engine port in front of them.

**On a machine with several cards, every model runs once on each** (D107): a copy per card,
pinned to it, each given its share of that card and its share of the host's workers, with the
router in front choosing the least busy copy. The router is started for copies even where the
pool placed one model, because the pool still dials one port. The copies share nothing, so a
model of any kind — dense, mixture-of-experts, embedding — runs the same way on any number of
cards, and the set must fit on one card, as it already had to.

**A model too large for one card is split across a group of them** (D114): with
`GPM_VLLM_CARDS_PER_COPY` set to N, every model runs once per group of N cards, as one vLLM
process with `--tensor-parallel-size N`, and each card holds a 1/N share of every model. One card
per copy is the case above. A machine whose cards do not divide into groups of N, or a model whose
attention heads do not divide by N, is refused before anything starts, as a set that does not fit.

**Each model is started no longer than it goes** (D146): the pool sends one longest sequence for
every process, and a model whose own files state a shorter one — Whisper's decoder, a reranker —
is started at its own, which is recorded so the agent can say so.

It stops whatever it started last time before starting anything, so calling it twice is a
restart and not a second copy fighting the first for the same memory.

**On a machine whose agent installs vLLM** (D159) the agent runs this launcher itself, as a
program — `python -m gpm_agent.cli vllm-start` — with `--vllm` naming the installed version's own
program. That option takes only a file named `vllm` inside the agent's engines directory, so still
nothing the pool sends becomes part of a command (D41); and the agent stops what this started with
`gpm-agent vllm-stop`, never by importing this module (D63).
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

# Written by the hub fetch when every file of a model has landed. A directory without it is a
# download in progress, and serving half a model is worse than serving none.
from .modelhub import (  # noqa: F401 - the marker's name is part of this module's API
    COMPLETE_MARKER,
    is_complete,
)

log = logging.getLogger("gpm.vllm-start")


#: What this launcher started, so the next start can stop it first.
PIDS_FILE = ".gpm-vllm.pids"

#: The router's map of model to engine, when there is more than one model.
UPSTREAMS_FILE = ".gpm-upstreams.json"

#: Where an older launcher kept its state, in the models directory itself. Found there now, they
#: came with the models from another machine, or from before: removed, never acted on (D139).
LEGACY_STATE = (PIDS_FILE, ".gpm-vllm.started.json", UPSTREAMS_FILE)

#: The share of the accelerator one process may take. vLLM's own recommendation for a single
#: model; the rest is headroom the driver and the runtime need.
TOTAL_MEMORY_SHARE = 0.90

#: The least any one model is given when several share a card and the card's size is not
#: known. Proportional-to-weights alone would hand a 0.3 GB embedding model under 2% beside a
#: 19 GB one — too little to start at all.
MEMORY_FLOOR = 0.10

#: What a process needs beyond its weights, when the card's size is known: room for the cache
#: it batches in and for its own workspace. Found live: splitting a 48 GB card by weights alone
#: gave a 15 GB model 17.9 GB, and it refused to start for want of 2.1 GiB of cache. A model
#: is given its weights (with a tenth over for what loading them costs) plus this reserve; a set
#: that does not fit is refused before anything starts, rather than started and watched die.
WEIGHT_OVERHEAD = 1.10
CACHE_RESERVE_BYTES = 3 * 1024**3

# What this launcher started, by model — process, port, log — is read back by the agent to tell
# a process that has since died from one still loading. The reading lives apart from this
# module: the agent never imports this launcher, which starts what the machine's start-up asks
# (D63). The programs the agent starts itself are whisper.cpp's server (D148) and, where the
# machine's owner lets it install engines, Ollama's (D150) — each with fixed arguments.
from . import platform as host_platform  # noqa: E402
from .vllm_state import (  # noqa: E402,F401
    IN_TURN_VERB,
    RUN_MARKER,
    STARTED_FILE,
    capped_lengths,
    failed_engines,
    in_turn_helper,
    last_error_line,
    read_record,
    state_dir_for,
)

#: The pool's one number for every process's longest sequence, which each model is held to its
#: own maximum of (D146).
MAX_MODEL_LEN = "GPM_VLLM_MAX_MODEL_LEN"

#: The pool's numbers, under the names the agent writes them in (D41). Only those present are
#: passed on: an absent one means "the engine's own default", not zero.
NUMBER_FLAGS = (
    ("GPM_VLLM_MAX_NUM_SEQS", "--max-num-seqs"),
    ("GPM_VLLM_MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens"),
    (MAX_MODEL_LEN, "--max-model-len"),
)

#: Where a model's configuration states its longest sequence, in the order vLLM v0.29.0 reads
#: them (`derive_max_model_len_and_key`, vllm/transformers_utils/model_arch_config_convertor.py).
#: The smallest one present is the model's maximum — except `model_max_length`, which, where set,
#: is taken as it stands.
LENGTH_KEYS = (
    "max_position_embeddings",  # most models
    "n_positions",  # GPT-2
    "max_seq_len",  # MPT
    "seq_length",  # ChatGLM2
    "model_max_length",  # Command-R
    "max_target_positions",  # Whisper's decoder
    "max_sequence_length",
    "max_seq_length",
    "seq_len",
)

#: Where a model built from a text model and another (vision, audio) keeps the text model's own
#: configuration: the names transformers' `get_text_config()` looks under, and `llm_config`,
#: which vLLM renames to `text_config` before it looks.
TEXT_SECTIONS = ("text_encoder", "decoder", "generator", "text_config", "llm_config")

#: Scalings vLLM does not multiply the length by: their `max_position_embeddings` is already the
#: scaled length.
UNSCALED_ROPE = ("su", "longrope", "llama3")

#: A sentence-transformers model's own tokenizer settings. vLLM takes `max_seq_length` from the
#: first of these it finds as the model's maximum outright (`get_sentence_transformer_tokenizer_config`).
SENTENCE_TRANSFORMER_FILES = (
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
)

#: What an architecture's name ends in, and whether vLLM v0.29.0 runs a model so named as a
#: pooling model (embedding, classifying, ranking) rather than generating text — its own table
#: for a model it does not decide otherwise (`_SUFFIX_TO_DEFAULTS`, vllm/config/model.py), checked
#: in order, so a generating `…ChatModel` or `…LMHeadModel` is not taken for an encoder's `…Model`.
POOLING_BY_SUFFIX = (
    ("ForCausalLM", False),
    ("ForConditionalGeneration", False),
    ("ChatModel", False),
    ("LMHeadModel", False),
    ("ForTextEncoding", True),
    ("EmbeddingModel", True),
    ("ForSequenceClassification", True),
    ("ForAudioClassification", True),
    ("ForImageClassification", True),
    ("ForVideoClassification", True),
    ("ClassificationModel", True),
    ("ForRewardModeling", True),
    ("RewardModel", True),
    ("Model", True),
)

#: A sentence-transformers model lists its modules here; one naming a pooling module is run as a
#: pooling model whatever its architecture (`get_pooling_config`).
SENTENCE_TRANSFORMER_MODULES = "modules.json"
SENTENCE_TRANSFORMER_POOLING = "sentence_transformers.models.Pooling"

#: How many cards each copy of a model spans (D114). Absent means one: a copy on each card.
CARDS_PER_COPY = "GPM_VLLM_CARDS_PER_COPY"
#: What vLLM can split a model across: its attention heads are divided between the cards.
SPLITS = (1, 2, 4, 8)

#: The named options the pool may ask for (D100), and what each means for each model family —
#: the family read from the model's own `config.json` (`model_type`), on this machine. The pool
#: sends only a name from this table; the flags are written here and nowhere else, so nothing the
#: pool says becomes part of a command (D41). A family missing from an option's row starts
#: without it, and says so. Parser names checked against vLLM v0.29.0's registries.
OPTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    # Lets an app send `tools` and get tool calls back as structured `tool_calls`.
    "tool_calling": {
        "gemma4": ("--enable-auto-tool-choice", "--tool-call-parser", "gemma4"),
        "qwen2": ("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        "qwen3": ("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        "qwen3_moe": ("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        "gpt_oss": ("--enable-auto-tool-choice", "--tool-call-parser", "openai"),
    },
    # Returns a thinking model's reasoning apart from its answer, as `reasoning_content`.
    "reasoning": {
        "gemma4": ("--reasoning-parser", "gemma4"),
        "qwen3": ("--reasoning-parser", "qwen3"),
        "qwen3_moe": ("--reasoning-parser", "qwen3"),
    },
}


@dataclass
class Started:
    """What one call started, for the log and for the next call to stop."""

    engines: list[dict[str, Any]] = field(default_factory=list)
    proxy: Optional[dict[str, Any]] = None
    skipped: list[str] = field(default_factory=list)
    #: Options asked for that a model's family does not have, said out loud.
    not_applied: list[str] = field(default_factory=list)
    #: Why nothing was started, when the set could not have run on this card.
    refused: Optional[str] = None
    #: Each model's share of the card and what it was sized for, for the record.
    plan: dict[str, Any] = field(default_factory=dict)
    #: What starts the rest of a card's models, one after another (D155) — when there are any.
    in_turn: Optional[dict[str, Any]] = None

    def pids(self) -> list[int]:
        found = [e["pid"] for e in self.engines if e.get("pid")]
        for helper in (self.proxy, self.in_turn):
            if helper and helper.get("pid"):
                found.append(helper["pid"])
        return found


def served_name(directory: Path) -> str:
    """The name a model is served under: its repository, from the directory the fetch made.

    The fetch flattened `owner/name` to `owner__name` so that nothing a pool says can climb out
    of the models directory; this undoes exactly that and nothing more.
    """
    return directory.name.replace("__", "/", 1)


def complete_models(models_dir: Path) -> list[Path]:
    """Model directories whose every file has landed, in a stable order."""
    if not models_dir.is_dir():
        return []
    return sorted(
        path for path in models_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and is_complete(path)
    )


def size_of(directory: Path) -> int:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def memory_shares(sizes: Sequence[int]) -> list[float]:
    """Each model's share of the accelerator: proportional to its weights, floored, rescaled.

    Weights are a fair first proxy for what a model needs, and the floor keeps a small model
    runnable. It is still a split fixed at launch, which is the cost of this shape: the largest
    model gets a fraction of the cache it would have had to itself (D96).
    """
    if not sizes:
        return []
    if len(sizes) == 1:
        return [TOTAL_MEMORY_SHARE]
    total = sum(sizes) or len(sizes)
    floored = [max((size or 1) / total, MEMORY_FLOOR) for size in sizes]
    scale = sum(floored)
    return [round(TOTAL_MEMORY_SHARE * share / scale, 3) for share in floored]


def card_memory_bytes(run: Callable[..., Any] = subprocess.run) -> Optional[int]:
    """The accelerator's memory, from the driver — or None where it cannot be asked."""
    try:
        out = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    sizes = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    return min(sizes) * 1024 * 1024 if sizes else None


def card_devices(env: Mapping[str, str], run: Callable[..., Any] = subprocess.run) -> list[str]:
    """The cards to put a copy on: those the machine's own environment limits it to, or every
    card the driver lists — or none, where neither says, and one copy runs where CUDA puts it."""
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [d.strip() for d in visible.split(",") if d.strip()]
    try:
        out = run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                  capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip().isdigit()]


def cards_per_copy(env: Mapping[str, str]) -> int:
    """How many cards each copy of a model spans, as the pool's number says; one without it."""
    value = env.get(CARDS_PER_COPY, "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else 1


def card_groups(cards: Sequence[str], per_copy: int) -> tuple[list[Optional[str]], Optional[str]]:
    """Where each copy runs — a card, a group of cards, or None for "wherever CUDA puts it" —
    or why this machine's cards cannot be grouped this way.

    One card per copy keeps D107 exactly: several cards, a copy pinned to each; one or none
    known, a single unpinned copy. Several per copy needs the cards listed, and a whole number
    of groups: a card left over would be paid for and never used.
    """
    if per_copy == 1:
        return (list(cards) if len(cards) > 1 else [None]), None
    if per_copy not in SPLITS:
        return [], f"a model can be split across {', '.join(map(str, SPLITS))} cards, not {per_copy}"
    if not cards:
        return [], f"the models are to be split across {per_copy} cards and the driver lists none here"
    if len(cards) % per_copy:
        return [], (
            f"the models are to be split across {per_copy} cards and this machine has {len(cards)}, "
            f"which is not a whole number of groups of {per_copy}"
        )
    return [",".join(cards[i:i + per_copy]) for i in range(0, len(cards), per_copy)], None


def attention_heads(directory: Path) -> Optional[int]:
    """The model's attention heads, from its own configuration — or None if it does not say.
    A model built from a text model and another (vision, audio) keeps them under `text_config`."""
    try:
        config = json.loads((directory / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    for section in (config, config.get("text_config")):
        if isinstance(section, dict) and isinstance(section.get("num_attention_heads"), int):
            return section["num_attention_heads"]
    return None


def split_refusal(directories: Sequence[Path], per_copy: int) -> Optional[str]:
    """Why these models cannot be split across `per_copy` cards, if one of them cannot.

    vLLM gives each card an equal share of a model's attention heads, and refuses to start when
    they do not divide — after the machine has been rented and the weights fetched. Said here,
    before anything starts, the way a set that does not fit is. A model that does not state its
    heads is left to vLLM, which will say.
    """
    if per_copy == 1:
        return None
    for directory in directories:
        heads = attention_heads(directory)
        if heads is not None and heads % per_copy:
            return (
                f"{served_name(directory)} has {heads} attention heads, which do not divide "
                f"between {per_copy} cards; split it across fewer cards"
            )
    return None


def memory_plan(
    sizes: Sequence[int], card_bytes: Optional[int], per_copy: int = 1
) -> tuple[list[float], Optional[str]]:
    """Each model's share of every card it runs on, or why the set cannot run on them.

    With the card's size known, each model is given its weights plus a cache reserve, and what
    is left of the launcher's share of the card is spread by weight — so the largest model gets
    the most cache. A model split across `per_copy` cards puts a 1/`per_copy` share of its
    weights on each, and still needs the reserve on each (D114). A set whose needs exceed that
    share is refused, with the arithmetic. Without the card's size, the old split by weights
    with a floor, which is a guess and says so.
    """
    if not sizes:
        return [], None
    if card_bytes is None:
        return memory_shares(sizes), None
    needs = [int(size * WEIGHT_OVERHEAD / per_copy) + CACHE_RESERVE_BYTES for size in sizes]
    usable = TOTAL_MEMORY_SHARE * card_bytes
    if sum(needs) > usable:
        split = f", split across {per_copy} cards," if per_copy > 1 else ""
        return [], (
            f"the models{split} need {sum(needs) / 1e9:.1f} GB together on each card (each its "
            f"share of the weights plus a {CACHE_RESERVE_BYTES / 1024**3:.0f} GiB cache reserve) "
            f"and this card gives {usable / 1e9:.1f} GB; fewer models on this host, a larger "
            f"card, or the models split across more cards"
        )
    spare = usable - sum(needs)
    total_weight = sum(sizes) or len(sizes)
    shares = [
        (need + spare * ((size or 1) / total_weight)) / card_bytes
        for need, size in zip(needs, sizes, strict=True)
    ]
    return [round(share, 3) for share in shares], None


#: The pool's numbers that are the host's whole, to be split between its copies: the workers
#: the pool sends a host are for all its cards, and vLLM's limit is per process (D107).
SPLIT_BETWEEN_COPIES = {"GPM_VLLM_MAX_NUM_SEQS"}


def number_flags(env: Mapping[str, str], copies: int = 1, max_model_len: Optional[int] = None) -> list[str]:
    """The pool's numbers as flags. `max_model_len` is the model's own maximum, where it states
    one: the pool's length is held to it, never raised to it."""
    flags: list[str] = []
    for variable, flag in NUMBER_FLAGS:
        value = env.get(variable, "").strip()
        if value.isdigit():
            if variable in SPLIT_BETWEEN_COPIES and copies > 1:
                # Rounded up: a host of 13 on two cards runs 7 and 7, never 6 and 6.
                value = str(max(1, math.ceil(int(value) / copies)))
            if variable == MAX_MODEL_LEN and max_model_len is not None:
                value = str(min(int(value), max_model_len))
            flags += [flag, value]
    return flags


def asked_max_model_len(env: Mapping[str, str]) -> Optional[int]:
    """The longest sequence the pool asked every process for, or None where it asked for none."""
    value = env.get(MAX_MODEL_LEN, "").strip()
    return int(value) if value.isdigit() else None


def _json_object(path: Path) -> Optional[dict[str, Any]]:
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def _length(value: Any) -> Optional[float]:
    """A stated length, or None for anything that is not a positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return value


def text_section(config: dict[str, Any]) -> dict[str, Any]:
    """The part of a model's configuration that describes its text model: the configuration
    itself, or the one section nested under a name vLLM looks under."""
    nested = [config[name] for name in TEXT_SECTIONS if isinstance(config.get(name), dict)]
    return nested[0] if len(nested) == 1 else config


def _rope_scaled(length: float, text: dict[str, Any]) -> float:
    """The length as vLLM scales it by the model's rotary-position scaling, if any.

    The same rule as vLLM's `_get_and_verify_max_len`: multiplied by the scaling's `factor`,
    except for scalings already counted in the stated length and for Gemma 3, whose stated
    length is already scaled; a `yarn` scaling multiplies the length it was trained at instead.
    """
    rope = text.get("rope_parameters") or text.get("rope_scaling")
    if not isinstance(rope, dict) or not rope or "gemma3" in str(text.get("model_type") or ""):
        return length
    # One scaling, or one per kind of attention layer.
    each = list(rope.values()) if all(isinstance(v, dict) for v in rope.values()) else [rope]
    factor: Any = 1.0
    for entry in each:
        kind = entry.get("rope_type") or entry.get("type") or "default"
        if kind in UNSCALED_ROPE:
            continue
        factor = entry.get("factor", factor)
        if kind == "yarn":
            trained = (_length(text.get("original_max_position_embeddings"))
                       or _length(entry.get("original_max_position_embeddings"))
                       or _length(text.get("max_position_embeddings")))
            length = trained or length
    return length * (_length(factor) or 1.0)


def runs_as_pooling(directory: Path, config: dict[str, Any]) -> bool:
    """Does vLLM v0.29.0 run this model as a pooling model — the only kind it holds to its
    tokenizer's length?

    The rule vLLM applies when the launcher names no runner, which it never does: a
    sentence-transformers model whose `modules.json` names a pooling module is one, whatever its
    architecture; otherwise the first architecture in `config.json` whose name ends in a suffix
    of `POOLING_BY_SUFFIX` decides — `…ForSequenceClassification` and an encoder's bare `…Model`
    pool, `…ForCausalLM` generates. A model naming none generates, as vLLM's default runner does.
    vLLM consults its registry of architectures before that table; for the encoders and
    generators it registers, the two agree.
    """
    try:
        modules = json.loads((directory / SENTENCE_TRANSFORMER_MODULES).read_text())
    except (OSError, ValueError):
        modules = None
    if isinstance(modules, list) and any(
        isinstance(module, dict) and module.get("type") == SENTENCE_TRANSFORMER_POOLING for module in modules
    ):
        return True
    architectures = config.get("architectures")
    for architecture in architectures if isinstance(architectures, list) else []:
        if not isinstance(architecture, str):
            continue
        for suffix, pooling in POOLING_BY_SUFFIX:
            if architecture.endswith(suffix):
                return pooling
    return False


def max_model_len_of(directory: Path) -> Optional[int]:
    """The longest sequence vLLM v0.29.0 will start this model with, from the model's own files —
    or None where they do not say, and vLLM takes whatever it is given.

    vLLM refuses to start a model with a `--max-model-len` above this, and the pool sends one
    number for every process: a text model's context, which Whisper's decoder (448) and many
    rerankers (512) are far below (D146). Read the way vLLM derives it (`_get_and_verify_max_len`
    in vllm/config/model.py): the smallest of the stated lengths in the text model's section,
    held to the tokenizer's own limit for a pooling model with absolute positions
    (`runs_as_pooling`) — never for one that generates — scaled as the model's rotary positions
    are, and replaced outright by a sentence-transformers model's own `max_seq_length`.

    It reads the files, not transformers' classes, so a length a class supplies only by default
    goes unseen. Missing a stated length makes the answer longer, which costs nothing: vLLM
    refuses a length too long exactly as it would have refused the pool's own number.
    """
    config = _json_object(directory / "config.json")
    if config is None:
        return None
    text = text_section(config)
    stated = [found for key in LENGTH_KEYS if (found := _length(text.get(key))) is not None]
    length: Optional[float] = min(stated) if stated else None
    if (named := _length(text.get("model_max_length"))) is not None:
        length = named
    if config.get("position_embedding_type") == "absolute" and runs_as_pooling(directory, config):
        tokenizer = _json_object(directory / "tokenizer_config.json") or {}
        if (limit := _length(tokenizer.get("model_max_length"))) is not None:
            length = limit if length is None else min(length, limit)
    if length is None:
        return None
    length = _rope_scaled(length, text)
    for name in SENTENCE_TRANSFORMER_FILES:
        encoder = _json_object(directory / name)
        if encoder:
            if "max_seq_length" in encoder and "do_lower_case" in encoder:
                length = _length(encoder["max_seq_length"]) or length
            break
    return int(length)



def family_of(directory: Path) -> Optional[str]:
    """The model's family, as its own configuration names it — or None if it does not."""
    try:
        found = json.loads((directory / "config.json").read_text()).get("model_type")
    except (OSError, ValueError, AttributeError):
        return None
    return found if isinstance(found, str) else None


def option_flags(options: Sequence[str], family: Optional[str]) -> tuple[list[str], list[str]]:
    """The flags for `options` on a model of `family`, and the options that do not apply to it.

    A name not in the table is refused, not skipped: the command line's own choices already
    stop one arriving, so reaching here with one is a bug worth hearing about.
    """
    flags: list[str] = []
    missing: list[str] = []
    for option in options:
        if option not in OPTIONS:
            raise ValueError(f"unknown option {option!r}; known: {sorted(OPTIONS)}")
        row = OPTIONS[option].get(family or "")
        if row is None:
            missing.append(option)
            continue
        flags += row
    return flags, missing


def _cmdline(pid: int) -> Optional[bytes]:
    """A process's command line: from /proc, where vLLM runs; elsewhere as the system's `ps` says
    it, so that what a test machine starts is recognised too (`platform.command_line`)."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        pass
    return host_platform.command_line(pid)


def _ours(pid: int, cmdline: Callable[[int], Optional[bytes]]) -> bool:
    """Is this process one an older launcher on this machine started — an engine or the router? A
    number from a file is only a number: it may be another machine's, brought with its models."""
    said = cmdline(pid) or b""
    return b"vllm" in said or (b"proxy" in said and b"--upstreams" in said)


def stop_previous(models_dir: Path, *, kill: Callable[[int, int], None] = os.kill,
                  alive: Optional[Callable[[int], bool]] = None, grace_s: float = 30.0,
                  cmdline: Callable[[int], Optional[bytes]] = _cmdline) -> list[int]:
    """Stop what the last call started. A process already gone is not an error.

    Only a process that is still an engine, the router or the helper by its own command line is
    signalled: a number in a file is only a number. On a machine its owner uses for other things
    too (D159), one restarted since has handed those numbers to the owner's own programs, and the
    next start — the agent's own, as it starts — would otherwise end them.

    What an older launcher started is listed in the models directory, where it kept its state
    then: on a host upgraded in place those engines still hold the cards and ports. They are
    stopped too, by the same rule, since that file may equally have come from another machine
    with the models (D139)."""
    pids_file = state_dir_for(models_dir) / PIDS_FILE
    # What starts a card's models in turn goes first, and is waited for: until it has gone it
    # may start one more, after the list below was read — an engine nothing would then stop.
    record = read_record(Path(models_dir)) or {}
    turns = (record.get("in_turn") or {}).get("pid")
    stopped = (_stop([turns], kill=kill, alive=alive, grace_s=grace_s)
               if isinstance(turns, int) and in_turn_helper(turns, cmdline) else [])
    try:
        pids = [int(p) for p in json.loads(pids_file.read_text())]
    except (OSError, ValueError, TypeError):
        pids = []
    try:
        older = [int(p) for p in json.loads((Path(models_dir) / PIDS_FILE).read_text())]
    except (OSError, ValueError, TypeError):
        older = []
    pids += [p for p in older if p not in pids]
    # The helper's number is in the list too, and was dealt with above — or is no longer its.
    listed = [p for p in pids if p not in stopped and p != turns]
    stopped += _stop([p for p in listed if _ours(p, cmdline)], kill=kill, alive=alive, grace_s=grace_s)
    pids_file.unlink(missing_ok=True)
    return stopped


def _alive_by_signal(pid: int, kill: Callable[[int, int], None] = os.kill) -> bool:
    try:
        kill(pid, 0)
        return True
    except OSError:
        return False


def _stop(pids: Sequence[int], *, kill: Callable[[int, int], None], alive: Optional[Callable[[int], bool]],
          grace_s: float) -> list[int]:
    """Ask each process to end, give them `grace_s` together, then end those still there."""
    def is_alive(pid: int) -> bool:
        if alive is not None:
            return alive(pid)
        # The system's own answer, which knows an exited process nobody reaped is not running.
        return host_platform.alive(pid) if kill is os.kill else _alive_by_signal(pid, kill)

    stopped = []
    for pid in pids:
        if not is_alive(pid):
            continue
        try:
            kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except OSError:
            continue
    deadline = time.monotonic() + grace_s
    while any(is_alive(pid) for pid in stopped) and time.monotonic() < deadline:
        time.sleep(0.5)
    for pid in stopped:
        if is_alive(pid):
            try:
                kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return stopped


def agent_command() -> list[str]:
    """How this launcher's helpers — the router, and what starts a card's models in turn — run the
    agent again: as this process was run. From the agent's archive or its `gpm-agent` program,
    that file; as a module (`python -m gpm_agent.cli`, as the agent that runs vLLM itself starts
    this, D159), the module again, under the same guard on where it imports from — its file alone
    is part of a package, and would not run."""
    main_module = sys.modules.get("__main__")
    spec = getattr(main_module, "__spec__", None)
    if spec is not None and spec.name == "gpm_agent.cli":
        guard = ["-I"] if sys.flags.isolated else ["-P"] if getattr(sys.flags, "safe_path", False) else []
        return [*guard, "-m", "gpm_agent.cli"]
    return [sys.argv[0]]


#: The program `--vllm` may name: a file of this name, inside the agent's engines directory.
VLLM_PROGRAM = "vllm"


def vllm_refusal(vllm: str, engines_dir: Path) -> Optional[str]:
    """Why `vllm` may not be run as vLLM here, or None. Only the program of a version the agent
    installed (D157): an absolute path to a file named `vllm` that is, links followed, inside the
    agent's engines directory. The pool never sends a path (D41); this holds even if something
    else on the machine one day did."""
    path = Path(vllm)
    if not path.is_absolute():
        return f"--vllm must be an absolute path, and {vllm!r} is not"
    if path.name != VLLM_PROGRAM:
        return f"--vllm names {path.name!r}; only a program named {VLLM_PROGRAM!r} is run as vLLM"
    root = os.path.realpath(engines_dir)
    real = os.path.realpath(path)
    if os.path.commonpath([root, real]) != root:
        return (
            f"--vllm {vllm} is not inside this agent's engines directory ({engines_dir}): only a "
            "version of vLLM the agent installed is run this way"
        )
    if not os.path.isfile(real):
        return f"--vllm {vllm} is not a file"
    return None


def launch(
    models_dir: str | Path,
    port: int,
    *,
    proxy: bool = False,
    options: Sequence[str] = (),
    vllm: str = VLLM_PROGRAM,
    agent: Optional[str] = None,
    python: str = sys.executable,
    env: Optional[Mapping[str, str]] = None,
    popen: Callable[..., Any] = subprocess.Popen,
    kill: Callable[[int, int], None] = os.kill,
    alive: Optional[Callable[[int], bool]] = None,
    card_bytes: Optional[int] = None,
    probe_card: Callable[[], Optional[int]] = card_memory_bytes,
    devices: Optional[Sequence[str]] = None,
    probe_devices: Callable[[Mapping[str, str]], list[str]] = card_devices,
    cmdline: Callable[[int], Optional[bytes]] = _cmdline,
) -> Started:
    """Stop what ran before, then start vLLM for every complete model on disk.

    Nothing on disk yet is the ordinary state of a machine that has just booted, and is not an
    error: this returns having started nothing, and is called again once the fetch has finished.

    `vllm` is the program each engine runs: `vllm` on the PATH, or a version the agent installed,
    which `main` has checked (`vllm_refusal`). `agent` is how the helpers run the agent again
    (`agent_command`, where unnamed).
    """
    models_dir = Path(models_dir).expanduser()
    state = state_dir_for(models_dir)
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = dict(os.environ if env is None else env)
    stop_previous(models_dir, kill=kill, alive=alive, cmdline=cmdline)
    for name in LEGACY_STATE:
        (models_dir / name).unlink(missing_ok=True)
    (state / STARTED_FILE).unlink(missing_ok=True)

    started = Started()
    found = complete_models(models_dir)
    if not found:
        return started

    if not proxy and len(found) > 1:
        # One model to a host without the router: the first, and the rest said out loud rather
        # than left to fight over the same memory.
        started.skipped = [served_name(d) for d in found[1:]]
        log.warning("several models on disk and no router; serving %s only", served_name(found[0]))
        found = found[:1]

    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sizes = [size_of(d) for d in found]
    per_copy = cards_per_copy(env)
    # The smallest card, when they differ: every copy is planned for the card it might get.
    card = card_bytes if card_bytes is not None else probe_card()
    shares, refused = memory_plan(sizes, card, per_copy)
    cards = list(devices) if devices is not None else probe_devices(env)
    # A copy on each card — or on each group of cards, for a model split across several (D114).
    placements, not_grouped = card_groups(cards, per_copy)
    refused = not_grouped or split_refusal(found, per_copy) or refused
    copies = len(placements)
    router = proxy or copies > 1
    # Each model's own longest sequence, which the pool's one number is held to (D146).
    maxima = [max_model_len_of(d) for d in found]
    asked = asked_max_model_len(env)
    started.plan = {
        "card_bytes": card,
        "cards": cards,
        "cards_per_copy": per_copy,
        "models": {served_name(d): {"weights_bytes": s} for d, s in zip(found, sizes, strict=True)},
    }
    for directory, own in zip(found, maxima, strict=True):
        if asked is not None and own is not None and own < asked:
            # Recorded for the agent to report: the host serves this model shorter than the pool
            # asked, because the model goes no further.
            started.plan["models"][served_name(directory)]["max_model_len_capped"] = {
                "asked": asked, "applied": own,
            }
            log.info("%s is started at --max-model-len %d, its own maximum; the pool asked for %d",
                     served_name(directory), own, asked)
    if refused:
        # Said where the agent reads it, so the pool hears the reason instead of watching
        # processes die one by one.
        started.refused = refused
        log.error("not starting vLLM: %s", refused)
        _record(models_dir, started)
        return started
    upstreams: dict[str, list[str]] = {}

    index = 0
    for device in placements:
        for directory, share, own in zip(found, shares, maxima, strict=True):
            name = served_name(directory)
            engine_port = port if not router else port + 1 + index
            index += 1
            family = family_of(directory)
            extra, missing = option_flags(options, family)
            if device is None or device == placements[0]:
                for option in missing:
                    started.not_applied.append(f"{option} for {name} (family {family or 'unknown'})")
                    log.warning("%s has no %s in its family (%s); started without it", name, option, family)
            argv = [
                vllm, "serve", str(directory),
                "--served-model-name", name,
                # Loopback, always: the pool reaches the machine through a forward, and binding
                # anything wider only exposes the engine (D77).
                "--host", "127.0.0.1",
                "--port", str(engine_port),
                "--gpu-memory-utilization", str(share),
                *(["--tensor-parallel-size", str(per_copy)] if per_copy > 1 else []),
                *number_flags(env, copies, own),
                *extra,
            ]
            log_path = logs / (f"{directory.name}.log" if device is None
                               else f"{directory.name}.card{device.replace(',', '-')}.log")
            started.engines.append({"model": name, "port": engine_port, "memory_share": share,
                                    "card": device, "pid": None, "argv": argv, "log": str(log_path)})
            upstreams.setdefault(name, []).append(f"http://127.0.0.1:{engine_port}")

    # The models on one card start one after another, each once the one before it answers or
    # has exited (D155). vLLM sizes its cache from the card's free memory measured across its own
    # start, assuming nothing else on the card changes meanwhile, so a neighbour loading at the
    # same moment is counted as its own use. Seen live: Whisper beside Gemma 4 E4B on one card
    # ended with "Engine core initialization failed" — which this explains; the lines that would
    # have said for certain were lost with the machine. Separate cards still start together.
    turns: dict[Optional[str], list[dict[str, Any]]] = {}
    for engine in started.engines:
        turns.setdefault(engine["card"], []).append(engine)
    for queue in turns.values():
        start_engine(queue[0], env, popen)
        for later in queue[1:]:
            later["waiting"] = True
    waiting = [queue for queue in turns.values() if len(queue) > 1]
    helper = None
    itself = [agent] if agent else agent_command()
    if waiting:
        argv = [python, *itself, "vllm-in-turn", "--models-dir", str(models_dir)]
        out = open(logs / "in-turn.log", "ab")
        helper = popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT, env=env,
                       start_new_session=True)
        started.in_turn = {"pid": getattr(helper, "pid", None), "argv": argv}

    if router:
        map_file = state / UPSTREAMS_FILE
        draft = map_file.with_suffix(".new")
        draft.write_text(json.dumps(upstreams, indent=2, sort_keys=True))
        draft.replace(map_file)
        argv = [python, *itself, "proxy", "--upstreams", str(map_file),
                "--host", "127.0.0.1", "--port", str(port)]
        out = open(logs / "proxy.log", "ab")
        process = popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        started.proxy = {"port": port, "pid": getattr(process, "pid", None), "argv": argv}

    (state / PIDS_FILE).write_text(json.dumps(started.pids()))
    _record(models_dir, started)
    if helper is not None:
        # Handed over only now that the record it updates is written; through its standard
        # input, so no list of commands is ever left on disk for something else to edit.
        turns_plan = [[{k: e.get(k) for k in ("model", "port", "card", "pid", "argv", "log")} for e in queue]
                      for queue in waiting]
        try:
            helper.stdin.write(json.dumps(turns_plan).encode())
            helper.stdin.close()
        except OSError as exc:
            # Gone before it read its plan: the models waiting for it are reported as never
            # started, from the record, and the pool gives the host up saying so.
            log.warning("what starts the rest of each card's models did not take its plan (%s)", exc)
    return started


def start_engine(engine: dict[str, Any], env: Mapping[str, str], popen: Callable[..., Any]) -> Any:
    """Start one planned engine — on its own card, if it has one — and note its process. Its log
    is kept across starts, so each start is marked in it first: the reason a start failed is read
    from after its own mark."""
    card = engine.get("card")
    process_env = dict(env) if card is None else {**env, "CUDA_VISIBLE_DEVICES": card}
    with open(engine["log"], "ab") as out:  # the process keeps its own copy
        out.write(f"\n{RUN_MARKER} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        out.flush()
        process = popen(engine["argv"], stdout=out, stderr=subprocess.STDOUT, env=process_env,
                        start_new_session=True)
    engine["pid"] = getattr(process, "pid", None)
    engine.pop("waiting", None)
    return process


# --- the rest of a card's models, one after another (D155) ---


def engine_answers(port: int, timeout_s: float = 2.0) -> bool:
    """Does the engine on this loopback port say it is up? vLLM's `/health` answers 200 once its
    model is loaded and its cache allocated — when its use of the card has stopped growing."""
    # Never through a proxy the machine's environment names: this is the engine beside it.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{int(port)}/health", timeout=timeout_s) as answer:
            return answer.status == 200
    except (OSError, ValueError):
        return False


def _is_vllm_serve(argv: Any) -> bool:
    """Only what the launcher plans is started here: `vllm serve …`, as an argument list."""
    return (isinstance(argv, list) and len(argv) >= 3 and all(isinstance(a, str) for a in argv)
            and Path(argv[0]).name == "vllm" and argv[1] == "serve")


@contextmanager
def _not_interrupted() -> Iterator[None]:
    """Hold off a stop while a process is started and written down, so that one stopping this
    helper never leaves an engine running that no list names; the stop is acted on after.

    By a handler that only notes it, never by blocking the signal: a blocked signal stays blocked
    in every process started meanwhile, through `exec` — found in review, an engine started so
    ignored the next restart's request to stop, and its engine process kept the card. A caught
    signal goes back to its default in a program that is started."""
    if threading.current_thread() is not threading.main_thread():
        yield  # only the main thread may set a handler; the helper runs on its own
        return
    asked: list[int] = []
    before = signal.signal(signal.SIGTERM, lambda signum, frame: asked.append(signum))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, before)
        if asked:
            os.kill(os.getpid(), signal.SIGTERM)


def _write_json(path: Path, value: Any) -> None:
    draft = path.with_suffix(path.suffix + ".new")
    draft.write_text(json.dumps(value, indent=2))
    draft.replace(path)


def _note_started(models_dir: Path, engine: dict[str, Any]) -> None:
    """Add a process started in turn to the record the agent reads and to the list the next
    start stops."""
    state = state_dir_for(models_dir)
    record = read_record(models_dir)
    if record is None:
        # Never written over with less than it held: the process is still listed below, to stop.
        log.warning("the launcher's record could not be read; %s is listed to stop, not recorded", engine["model"])
    else:
        for planned in record.get("engines") or []:
            if (planned.get("model"), planned.get("card"), planned.get("port")) == (
                    engine["model"], engine.get("card"), engine["port"]):
                planned["pid"] = engine["pid"]
                planned.pop("waiting", None)
        _write_json(state / STARTED_FILE, record)
    try:
        pids = [int(p) for p in json.loads((state / PIDS_FILE).read_text())]
    except (OSError, ValueError, TypeError):
        pids = []
    if engine["pid"] is not None and engine["pid"] not in pids:
        pids.append(engine["pid"])
    _write_json(state / PIDS_FILE, pids)


def run_in_turn(
    models_dir: str | Path,
    turns: Sequence[Sequence[dict[str, Any]]],
    *,
    env: Optional[Mapping[str, str]] = None,
    popen: Callable[..., Any] = subprocess.Popen,
    ready: Callable[[int], bool] = engine_answers,
    alive: Optional[Callable[[int], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 2.0,
    say: Callable[[str], None] = lambda line: print(line, flush=True),
    own_pid: Optional[int] = None,
) -> None:
    """Start each card's waiting models, each once the one before it answers or has exited.

    `turns` is the launcher's plan, a list per card whose first engine it started itself. One
    that exits rather than answering does not hold the rest back: its failure is reported from
    its own log, and the models after it may still serve. Nothing here waits on a clock — a
    large model loads for as long as it takes, and the pool decides when a host has taken too
    long. Once every model is started it takes its own number (`own_pid`) out of the record and
    the list the next start stops: that number will soon be some other process's."""
    models_dir = Path(models_dir).expanduser()
    env = dict(os.environ if env is None else env)
    queues = [[dict(e) for e in queue] for queue in turns if queue]
    for queue in queues:
        for engine in queue:
            if not _is_vllm_serve(engine.get("argv")):
                raise ValueError("vllm-in-turn starts only the launcher's own `vllm serve` processes")

    # The engines started here are this process's children: one that has exited is known by its
    # process object, which also reaps it. Found in review: left unreaped, an engine that died at
    # start still answered a signal, read as loading for ever, and the model after it never
    # started. The engines the launcher started are asked of the system, which knows the same.
    children: dict[int, Any] = {}

    def is_alive(pid: int) -> bool:
        child = children.get(pid)
        if child is not None and callable(getattr(child, "poll", None)):
            return child.poll() is None
        return alive(pid) if alive is not None else host_platform.alive(pid)

    at = [0] * len(queues)
    while any(at[i] < len(queue) - 1 for i, queue in enumerate(queues)):
        moved = False
        for i, queue in enumerate(queues):
            if at[i] >= len(queue) - 1:
                continue
            current = queue[at[i]]
            pid = current.get("pid")
            if isinstance(pid, int) and is_alive(pid) and not ready(current["port"]):
                continue  # still loading
            following = queue[at[i] + 1]
            with _not_interrupted():
                process = start_engine(following, env, popen)
                if isinstance(following.get("pid"), int):
                    children[following["pid"]] = process
                _note_started(models_dir, following)
            after = "answered" if isinstance(pid, int) and is_alive(pid) else "exited"
            say(f"started {following['model']} on {following['port']}, once {current['model']} {after}")
            at[i] += 1
            moved = True
        if not moved:
            sleep(poll_s)
    if own_pid is not None:
        _note_finished(models_dir, own_pid)


def _note_finished(models_dir: Path, own_pid: int) -> None:
    """Every model started: this helper is no longer anything to stop or wait for."""
    state = state_dir_for(models_dir)
    with _not_interrupted():
        record = read_record(models_dir)
        if record is not None and (record.get("in_turn") or {}).get("pid") == own_pid:
            record["in_turn"] = None
            _write_json(state / STARTED_FILE, record)
        try:
            pids = [int(p) for p in json.loads((state / PIDS_FILE).read_text())]
        except (OSError, ValueError, TypeError):
            return
        if own_pid in pids:
            _write_json(state / PIDS_FILE, [p for p in pids if p != own_pid])


def stop(models_dir: str | Path, **kwargs: Any) -> list[int]:
    """Stop what the launcher last started here, and forget its record of it: nothing then runs,
    and nothing it started is a model that failed. For the agent that runs vLLM itself (D159), to
    switch versions or hand the port to another engine — run as `gpm-agent vllm-stop`, never in
    the agent's own process (D63)."""
    models_dir = Path(models_dir).expanduser()
    stopped = stop_previous(models_dir, **kwargs)
    (state_dir_for(models_dir) / STARTED_FILE).unlink(missing_ok=True)
    return stopped


def main_stop(args: Any) -> int:
    """`gpm-agent vllm-stop`: what the last `vllm-start` began, stopped."""
    stopped = stop(args.models_dir)
    print(f"gpm-agent vllm-stop: stopped {len(stopped)} process{'' if len(stopped) == 1 else 'es'}")
    return 0


def main_in_turn(args: Any) -> int:
    """`gpm-agent vllm-in-turn`: started by the launcher, which hands it the plan on its input."""
    turns = json.loads(sys.stdin.read() or "[]")
    run_in_turn(args.models_dir, turns, own_pid=os.getpid())
    return 0


def _record(models_dir: Path, started: Started) -> None:
    """What was started (or why nothing was), for the agent to read back."""
    record = {
        "at": time.time(),
        "refused": started.refused,
        "plan": started.plan,
        "engines": [{k: v for k, v in e.items() if k != "argv"} for e in started.engines],
        "proxy": {k: v for k, v in started.proxy.items() if k != "argv"} if started.proxy else None,
        "in_turn": {k: v for k, v in started.in_turn.items() if k != "argv"} if started.in_turn else None,
    }
    target = state_dir_for(models_dir) / STARTED_FILE
    draft = target.with_suffix(".new")
    draft.write_text(json.dumps(record, indent=2))
    draft.replace(target)


def add_arguments(parser: Any) -> None:
    parser.add_argument("--models-dir", required=True, help="where the agent fetched the models")
    parser.add_argument("--port", type=int, required=True, help="the port the pool dials")
    parser.add_argument("--proxy", action="store_true",
                        help="one engine per model, with the router on --port in front of them")
    parser.add_argument("--option", action="append", default=[], choices=sorted(OPTIONS),
                        help="a named option; each model gets it if its family has it (D100)")
    # The one way to name another program: a version of vLLM the agent installed, by its own
    # program, inside the agent's engines directory and nowhere else (D159). The agent passes it;
    # the pool never can, since nothing it says reaches this command line (D41).
    parser.add_argument("--vllm", default=None,
                        help="a version of vLLM this agent installed: its `vllm` program, inside the "
                             "agent's engines directory (`vllm` on the PATH if unset)")
    parser.add_argument("--engines-dir", default=None,
                        help="the agent's engines directory, where its owner moved it (engines_path); "
                             "the system's own place if unset")


def add_stop_arguments(parser: Any) -> None:
    parser.add_argument("--models-dir", required=True, help="where the agent fetched the models")


def main(args: Any) -> int:
    vllm = VLLM_PROGRAM
    if args.vllm is not None:
        engines = Path(args.engines_dir).expanduser() if args.engines_dir else host_platform.engines_dir()
        refused = vllm_refusal(args.vllm, engines)
        if refused:
            print(f"gpm-agent vllm-start: not started — {refused}")
            return 2
        vllm = args.vllm
    started = launch(args.models_dir, args.port, proxy=args.proxy, options=args.option, vllm=vllm)
    if started.refused:
        print(f"gpm-agent vllm-start: not started — {started.refused}")
        return 1
    if not started.engines:
        print("gpm-agent vllm-start: no complete model on disk yet; nothing started")
        return 0
    for engine in started.engines:
        on = f", card {engine['card']}" if engine.get("card") is not None else ""
        verb = "to start in turn" if engine.get("waiting") else "started"
        print(f"{verb} {engine['model']} on {engine['port']} (memory {engine['memory_share']}{on})")
    if started.in_turn:
        print("the rest of each card's models start one after another, each once the one before it answers")
    if started.proxy:
        print(f"started the router on {started.proxy['port']}")
    for name in started.skipped:
        print(f"not started: {name} (several models on disk and no router)")
    for what in started.not_applied:
        print(f"not applied: {what}")
    for name, planned in started.plan.get("models", {}).items():
        if capped := planned.get("max_model_len_capped"):
            print(f"capped: {name} at --max-model-len {capped['applied']}, its own maximum "
                  f"(the pool asked for {capped['asked']})")
    return 0
