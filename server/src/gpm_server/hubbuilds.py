"""Finding an engine's builds of a model on a model hub, sorted for an operator (D100).

An engine whose builds are hub repositories (vLLM) needs one named for each model — and the
hub answers a search for one model with dozens: the original, quantisations for different cards,
files for other engines, fine-tunes that are a different model under a similar name. Asking an
operator to type the right one is asking them to know all of that. This reads what the hub
already publishes about each repository and sorts it:

- **what it is**: the original, a build of it (the hub's own `base_model:quantized:` link), or a
  different model — fine-tuned, merged, or built from something else — which is not offered;
- **whether this engine loads it at all**: safetensors weights, not GGUF or MLX;
- **its precision and size**, from its quantisation settings and its weight files' sizes;
- **which cards run it**, and which run it at full speed, from the precision;
- **whether a rented host can fetch it**: a gated repository needs an account, and the pool
  never puts one on a machine it rents.

It only reads. The hub is asked without any credential — what an anonymous machine can fetch is
exactly what a rented host can — and nothing it says reaches a command: the operator's choice is
saved as a repository name, which the agent fetches under its own rules (D97).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional

import httpx

#: The public hub, or a mirror — the same variable the agent's fetch reads (D97).
DEFAULT_HUB = "https://huggingface.co"

#: What one search asks for, and how many builds are shown after sorting.
SEARCH_LIMIT = 50
SHOWN = 12

#: What each search result is asked to carry, so one request per search term is enough.
EXPAND = ("safetensors", "config", "gated", "tags", "downloads", "library_name")

#: A search term: a model's name, as the operator or the pool writes it. Nothing else reaches
#: the hub's query string.
_SEARCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")

#: An Ollama tag's quantisation suffix, which no hub repository is named after.
_ENGINE_SUFFIX = re.compile(r"-(q\d\w*|fp16|bf16|fp8|int4|int8)$", re.I)

#: Bytes per weight, by the type the hub's tally names.
_BYTES = {
    "F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1,
}

#: Accelerator generations, oldest first.
GENERATIONS = ("Turing", "Ampere", "Ada", "Hopper", "Blackwell")

#: Where each precision runs, and where it runs at full speed: the first generation of each.
#: "Runs" is the engine's own floor for the method (vLLM v0.29.0, `get_min_capability`), taken
#: conservatively; "full speed" is where the card computes in that precision itself rather than
#: unpacking it first — FP8 from Ada, FP4 only on Blackwell.
_CARDS: dict[str, tuple[str, Optional[str]]] = {
    "BF16": ("Ampere", "Ampere"),
    "FP16": ("Turing", "Turing"),
    "FP32": ("Turing", "Turing"),
    "FP8": ("Ampere", "Ada"),
    "NVFP4": ("Ampere", "Blackwell"),
    "MXFP4": ("Ampere", "Blackwell"),
    "INT4": ("Turing", "Ampere"),
    "INT8": ("Turing", "Ampere"),
    "bitsandbytes": ("Turing", None),
}


class HubUnavailable(RuntimeError):
    """The hub did not answer, or answered with something that is not a list of models."""


@dataclass
class Build:
    repo: str
    publisher: str
    #: "original", or "build" — a quantisation of the original, by the hub's own link.
    relation: str
    precision: Optional[str]
    #: What a host downloads: the weights' files, summed from the repository's own listing —
    #: or, when that could not be read, estimated from the hub's tally and marked so.
    size_gb: Optional[float]
    runs_on: Optional[str]
    full_speed_on: Optional[str]
    #: The model's family (`model_type`), which decides which engine options apply to it.
    family: Optional[str]
    gated: bool
    downloads: int
    #: Why a rented host cannot use it, when it cannot.
    why_not: Optional[str] = None
    size_estimated: bool = True
    #: The original this is a build of — itself, for an original.
    of: Optional[str] = None


@dataclass
class Found:
    model: str
    searched: list[str]
    original: Optional[str]
    builds: list[Build] = field(default_factory=list)
    #: Every original offered: the most likely first, then the same publisher's others.
    originals: list[str] = field(default_factory=list)
    #: What was left out, by reason, so an empty or short list explains itself.
    hidden: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def valid_search(term: str) -> bool:
    return bool(_SEARCH.match(term or ""))


def search_terms(name: str) -> list[str]:
    """What to ask the hub for, from a model's name in the pool.

    Pool names follow the first engine's habits — `gemma4:26b`, `qwen3:30b`, `llama3.1:8b` —
    and hub repositories are named by their publishers, some with a hyphen before the version
    (`gemma-4-26B`) and some without (`Qwen3-30B`). The hub's search matches a substring, so
    both spellings are asked for and the answers merged.
    """
    base, _, tag = name.partition(":")
    tag = _ENGINE_SUFFIX.sub("", "" if tag in ("", "latest") else tag)
    base = base.lower().split("/")[-1]
    spellings = dict.fromkeys([base, re.sub(r"(?<=[a-z])(?=\d)", "-", base)])
    return [f"{b}-{tag}" if tag else b for b in spellings]


def weight_bits(quant: dict[str, Any]) -> Optional[int]:
    for group in (quant.get("config_groups") or {}).values():
        bits = ((group or {}).get("weights") or {}).get("num_bits")
        if isinstance(bits, int):
            return bits
    for key in ("bits", "w_bit", "num_bits"):
        if isinstance(quant.get(key), int):
            return quant[key]
    return None


def precision_of(quant: Optional[dict[str, Any]], tally: dict[str, int]) -> Optional[str]:
    """The build's precision, as the card will see it — or None when it cannot be told."""
    if not quant:
        top = max(tally, key=tally.get) if tally else None
        return {"BF16": "BF16", "F16": "FP16", "F32": "FP32"}.get(top or "")
    method = str(quant.get("quant_method") or "").lower()
    # A mixed build names a format per group of layers; the most demanding of them decides
    # which cards can run it, so all of them are read.
    form = " ".join(
        [str(quant.get("format") or "")]
        + [str((g or {}).get("format") or "") for g in (quant.get("config_groups") or {}).values()]
    ).lower()
    algo = str(quant.get("quant_algo") or "").upper()
    bits = weight_bits(quant)
    if method == "modelopt" and bits is None and not algo:
        # Published with its settings cut short: packed four-bit weights are stored as bytes
        # beside eight-bit scales, so more bytes than eight-bit floats means FP4.
        bits = 4 if tally.get("U8", 0) > tally.get("F8_E4M3", 0) else 8
    if "nvfp4" in form or algo == "NVFP4" or (method == "modelopt" and bits == 4):
        return "NVFP4"
    if method == "mxfp4" or "mxfp4" in form:
        return "MXFP4"
    if "float-quantized" in form or method in ("fp8", "fbgemm_fp8") or algo == "FP8" or (
        method == "modelopt" and bits == 8
    ):
        return "FP8"
    if method in ("awq", "gptq") or "pack-quantized" in form:
        return "INT8" if bits == 8 else "INT4"
    if "int-quantized" in form:
        return "INT8"
    if method == "bitsandbytes":
        return "bitsandbytes"
    return None


def size_gb(tally: dict[str, int]) -> Optional[float]:
    """An estimate only: for packed formats the hub counts logical weights, not stored bytes,
    so this can be off by half. The repository's own file listing is what is shown when it can
    be read."""
    if not tally:
        return None
    return round(sum(count * _BYTES.get(kind, 2) for kind, count in tally.items()) / 1e9, 1)


def cards_for(precision: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    runs, full = _CARDS.get(precision or "", (None, None))

    def onward(first: Optional[str]) -> Optional[str]:
        if first is None:
            return None
        return first if first == GENERATIONS[-1] else f"{first} and newer"

    return onward(runs), onward(full)


def links(entry: dict[str, Any]) -> dict[str, set[str]]:
    """The hub's own record of where a repository came from, by kind (`quantized`, `finetune`,
    `merge`, `adapter`) — read from its tags, which is where the hub publishes it."""
    found: dict[str, set[str]] = {}
    for tag in entry.get("tags") or []:
        parts = tag.split(":", 2)
        if len(parts) == 3 and parts[0] == "base_model":
            found.setdefault(parts[1], set()).add(parts[2])
    return found


def not_loadable(entry: dict[str, Any]) -> Optional[str]:
    """Why the engine cannot load this repository's files, or None if it can."""
    tags = {t.lower() for t in entry.get("tags") or []}
    library = str(entry.get("library_name") or "").lower()
    quant = (entry.get("config") or {}).get("quantization_config") or {}
    if "gguf" in tags or library in ("gguf", "llama.cpp"):
        return "GGUF files, for llama.cpp and Ollama"
    # MLX's own quantisation names bits and no method; nothing else the engine loads does.
    if "mlx" in tags or library == "mlx" or ("bits" in quant and not quant.get("quant_method")):
        return "MLX files, for Apple silicon"
    if not entry.get("safetensors"):
        return "no safetensors weights"
    return None


def choose_originals(entries: list[dict[str, Any]], terms: list[str] = ()) -> list[str]:
    """The models the others are builds of, the most likely first.

    A search for one model returns its builds, each linking back to what it quantised — so the
    most-linked repository is usually the model itself, whether or not the search found it too.
    Usually: a search for `qwen3-30b` also finds the *coder* model's many builds, and the most
    links then point at the wrong model. So an original must carry the name searched for when
    any candidate does; the most-linked of those leads; and the same publisher's other
    originals under that name — an instruct revision, a quantisation-aware one — are offered
    beside it, each with its own builds. Another publisher's model under a similar name is a
    fine-tune until it says otherwise.
    """
    counts = Counter(t for e in entries for t in links(e).get("quantized", ()))
    plain = {
        e["id"] for e in entries
        if not not_loadable(e) and not ({"quantized", "merge", "adapter"} & set(links(e)))
    }
    candidates = set(counts) | plain
    wanted = [t.lower() for t in terms if t]
    named = [c for c in candidates if any(t in c.split("/")[-1].lower() for t in wanted)] or list(candidates)
    downloads = {e["id"]: e.get("downloads") or 0 for e in entries}
    ranked = sorted(named, key=lambda c: (counts[c] > 0, counts[c], downloads.get(c, 0)), reverse=True)
    if not ranked:
        return []
    families = {
        e["id"]: (e.get("config") or {}).get("model_type") for e in entries if isinstance(e.get("config"), dict)
    }
    lead = ranked[0]
    publisher, family = lead.split("/")[0], families.get(lead)

    def alike(candidate: str) -> bool:
        # A small draft model published beside the real one shares its name and publisher and
        # not its family; where both families are known, they must match.
        theirs = families.get(candidate)
        return candidate.split("/")[0] == publisher and (not family or not theirs or theirs == family)

    return [lead] + [c for c in ranked[1:] if alike(c)]


def describe(entry: dict[str, Any], relation: str, *, of: Optional[str] = None) -> Build:
    config = entry.get("config") or {}
    tally = (entry.get("safetensors") or {}).get("parameters") or {}
    precision = precision_of(config.get("quantization_config"), tally)
    runs, full = cards_for(precision)
    gated = bool(entry.get("gated"))
    return Build(
        repo=entry["id"],
        publisher=entry["id"].split("/")[0],
        relation=relation,
        precision=precision,
        size_gb=size_gb(tally),
        runs_on=runs,
        full_speed_on=full,
        family=config.get("model_type") if isinstance(config.get("model_type"), str) else None,
        gated=gated,
        downloads=int(entry.get("downloads") or 0),
        why_not=(
            "gated: the hub asks for an account, and a rented host fetches without one"
            if gated else None
        ),
        of=of,
    )


def sort_builds(model: str, searched: list[str], entries: list[dict[str, Any]],
                originals: Optional[list[dict[str, Any]]] = None) -> Found:
    """Everything above, over one search's results. Pure: the tests drive it with the hub's
    own answers, recorded. `originals` are ones asked for by name because no search found them."""
    by_id = {e["id"]: e for e in entries}
    chosen = choose_originals(entries, searched)
    for entry in originals or []:
        if entry.get("id") in chosen:
            by_id.setdefault(entry["id"], entry)
    found = Found(model=model, searched=searched, original=chosen[0] if chosen else None, originals=chosen)
    hidden: Counter[str] = Counter()
    builds: list[Build] = []
    for entry in by_id.values():
        reason = not_loadable(entry)
        if reason:
            hidden[reason] += 1
            continue
        if entry["id"] in chosen:
            builds.append(describe(entry, "original", of=entry["id"]))
            continue
        of = next((o for o in chosen if o in links(entry).get("quantized", set())), None)
        if of:
            builds.append(describe(entry, "build", of=of))
        else:
            hidden["a different model — fine-tuned, merged, or built from another"] += 1
    # Each original, then its builds most downloaded first; the most likely original leads.
    builds.sort(key=lambda b: (chosen.index(b.of), b.relation != "original", -b.downloads))
    if len(builds) > SHOWN:
        hidden["less used builds"] += len(builds) - SHOWN
        builds = builds[:SHOWN]
    found.builds = builds
    found.hidden = dict(hidden)
    return found


async def _unpaced() -> None:
    return None


#: What a search result is asked to carry: enough to filter by size, precision and task without
#: a request per result.
MODEL_EXPAND = EXPAND + ("pipeline_tag",)

#: The hub's task names, as an operator would say them.
_TASKS = {
    "text-generation": "chat",
    "image-text-to-text": "vision",
    "any-to-any": "vision",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
}


@dataclass
class HubGroup:
    """One model found by a free search, with every variant of it the search turned up — the
    original and its quantisations side by side, so a variant is chosen in one step (D111)."""

    #: The original the variants are builds of — the hub's own `base_model:quantized:` link.
    model: str
    #: Billions of parameters, from the original's weights; None where the search did not
    #: return the original (a quantisation's tally counts packed bytes, not weights).
    params_b: Optional[float]
    task: Optional[str]
    family: Optional[str]
    #: "fine-tune of …" or "merge of …", where the original says so.
    made_from: Optional[str]
    variants: list[Build] = field(default_factory=list)
    downloads: int = 0


def params_b(entry: dict[str, Any]) -> Optional[float]:
    tally = (entry.get("safetensors") or {})
    total = tally.get("total") or sum((tally.get("parameters") or {}).values())
    return round(total / 1e9, 2) if total else None


def group_search(entries: list[dict[str, Any]]) -> list[HubGroup]:
    """A search's answer as models, each with its variants. Pure: the tests drive it with the
    hub's own answers, recorded.

    Files the engine cannot load (GGUF, MLX, no safetensors) and adapters are left out — they
    are not variants a rented host can run. A quantisation joins the group of the model it is a
    quantisation of, whether or not the search returned that model itself."""
    groups: dict[str, HubGroup] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not_loadable(entry):
            continue
        linked = links(entry)
        if linked.get("adapter"):
            continue
        quantised = sorted(linked.get("quantized", ()))
        base = quantised[0] if quantised else entry["id"]
        group = groups.get(base)
        if group is None:
            group = groups[base] = HubGroup(model=base, params_b=None, task=None, family=None, made_from=None)
        if not quantised:
            config = entry.get("config") or {}
            group.params_b = params_b(entry)
            group.task = _TASKS.get(str(entry.get("pipeline_tag") or ""))
            group.family = config.get("model_type") if isinstance(config.get("model_type"), str) else None
            for kind in ("finetune", "merge"):
                if linked.get(kind):
                    group.made_from = f"{'fine-tune' if kind == 'finetune' else 'merge'} of {sorted(linked[kind])[0]}"
                    break
        group.task = group.task or _TASKS.get(str(entry.get("pipeline_tag") or ""))
        variant = describe(entry, "build" if quantised else "original", of=base)
        if (entry.get("config") or {}).get("quantization_config"):
            # The hub's tally counts packed weights by their container type, so a 4-bit build
            # can read as larger than its original. Not shown until measured (`exact_size_gb`).
            variant.size_gb = None
        group.variants.append(variant)
        group.downloads = max(group.downloads, int(entry.get("downloads") or 0))
    for group in groups.values():
        group.variants.sort(key=lambda v: (v.relation != "original", -v.downloads))
    return sorted(groups.values(), key=lambda g: -g.downloads)


async def search_variants(term: str, *, client: Optional[httpx.AsyncClient] = None,
                          pace: Callable[[], Awaitable[None]] = _unpaced,
                          limit: int = SEARCH_LIMIT) -> list[HubGroup]:
    """Every loadable repository on the hub whose name holds `term`, most downloaded first, as
    models with their variants (D111). One request, no credential — what an anonymous machine
    sees is what a rented host can fetch. Sizes are the hub's tally, an estimate; the exact size
    of the one chosen is read afterwards (`exact_size_gb`)."""
    if not valid_search(term):
        raise ValueError(f"not a model name to search for: {term!r}")
    owned = client is None
    client = client or httpx.AsyncClient(base_url=hub_url(), timeout=20.0, headers={})
    try:
        await pace()
        answer = await client.get("/api/models", params=[
            ("search", term), ("limit", str(max(1, min(limit, 100)))), ("sort", "downloads"), ("direction", "-1"),
        ] + [("expand[]", e) for e in MODEL_EXPAND])
        answer.raise_for_status()
        listed = answer.json()
        if not isinstance(listed, list):
            raise HubUnavailable("the hub's search did not return a list of models")
    except httpx.HTTPError as exc:
        raise HubUnavailable(f"the model hub did not answer: {exc or type(exc).__name__}") from exc
    finally:
        if owned:
            await client.aclose()
    return group_search(listed)


async def exact_size_gb(repo: str, *, client: Optional[httpx.AsyncClient] = None,
                        pace: Callable[[], Awaitable[None]] = _unpaced) -> Optional[float]:
    """The weights one repository holds, from its own file listing — the size a profile's
    minimums are worked out from, where the search only had an estimate."""
    if not valid_search(repo):
        raise ValueError(f"not a repository name: {repo!r}")
    owned = client is None
    client = client or httpx.AsyncClient(base_url=hub_url(), timeout=20.0, headers={})
    try:
        await pace()
        listing = await client.get(f"/api/models/{repo}/tree/main")
        return weights_size_gb(listing.json()) if listing.status_code == 200 else None
    except httpx.HTTPError as exc:
        raise HubUnavailable(f"the model hub did not answer: {exc or type(exc).__name__}") from exc
    finally:
        if owned:
            await client.aclose()


def hub_url() -> str:
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_HUB).rstrip("/")


def weights_size_gb(listing: Any) -> Optional[float]:
    """The weights' files in one repository listing, summed."""
    if not isinstance(listing, list):
        return None
    total = sum(
        int(f.get("size") or 0) for f in listing
        if isinstance(f, dict) and str(f.get("path", "")).endswith(".safetensors")
    )
    return round(total / 1e9, 1) if total else None


async def find_builds(model: str, search: Optional[str] = None, *,
                      client: Optional[httpx.AsyncClient] = None,
                      pace: Callable[[], Awaitable[None]] = _unpaced) -> Found:
    """Ask the hub, then sort what it said. `search` replaces the terms guessed from the name.

    `pace` is awaited before every request, so whoever calls this decides how hard the hub is
    asked: one search per spelling, the original by name if no search found it, then each shown
    build's file listing for its size."""
    terms = [search] if search else search_terms(model)
    for term in terms:
        if not valid_search(term):
            raise ValueError(f"not a model name to search for: {term!r}")
    # Most downloaded first: the hub's default order is not, and a well-used build past the
    # first page is a build nobody is offered.
    params_common = [("limit", str(SEARCH_LIMIT)), ("sort", "downloads"), ("direction", "-1")] + [
        ("expand[]", e) for e in EXPAND
    ]
    owned = client is None
    # No credential, deliberately: what an anonymous request can see is what a rented host can
    # fetch, and a token here would show gated builds as usable.
    client = client or httpx.AsyncClient(base_url=hub_url(), timeout=20.0, headers={})
    try:
        entries: dict[str, dict[str, Any]] = {}
        for term in terms:
            await pace()
            answer = await client.get("/api/models", params=[("search", term)] + params_common)
            answer.raise_for_status()
            listed = answer.json()
            if not isinstance(listed, list):
                raise HubUnavailable("the hub's search did not return a list of models")
            for entry in listed:
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    entries.setdefault(entry["id"], entry)
        missing = []
        for original_id in choose_originals(list(entries.values()), terms):
            if original_id in entries:
                continue
            # Its builds were found and it was not — ask for it by name, so it is offered too.
            await pace()
            answer = await client.get(f"/api/models/{original_id}", params=[("expand[]", e) for e in EXPAND])
            if answer.status_code == 200 and isinstance(answer.json(), dict):
                missing.append(answer.json() | {"id": original_id})
        found = sort_builds(model, terms, list(entries.values()), missing)
        for build in found.builds:
            await pace()
            listing = await client.get(f"/api/models/{build.repo}/tree/main")
            exact = weights_size_gb(listing.json()) if listing.status_code == 200 else None
            if exact is not None:
                build.size_gb, build.size_estimated = exact, False
        return found
    except httpx.HTTPError as exc:
        raise HubUnavailable(f"the model hub did not answer: {exc or type(exc).__name__}") from exc
    finally:
        if owned:
            await client.aclose()
