# S12 — Both API formats: OpenAI's and Anthropic's, to any model

> Status: **built** (2026-11-07, D169–D170; gpm-server 0.39.0) and **checked live against Anthropic**
> (2026-11-08) — the owner's answers below; other providers' live checks wait on their keys.
> Part of the [feature list](README.md).

## The story

An app written against OpenAI's Chat Completions and an app written against Anthropic's Messages
API both point at the pool — one URL, one key — and each reaches any model the pool serves: a
model on the pool's own machines, Claude, GPT, an open model at a hosted provider. A request goes
to its host **untouched** wherever the host speaks the request's format, and is **translated**
only where it does not. Where a provider bills less for a repeated prompt, the pool makes that
happen by default.

## Why

- **The owner asked** (2026-11-07): apps that use the OpenAI format today and want to use
  anything — Claude among it — and apps written with Anthropic's SDK. "We need to support both."
- **Claude through GPM is never cached today**, and the owner spent through a budget on exactly
  that (S11's known limit). Anthropic's OpenAI-compatible endpoint, which S11 uses, does not cache
  at all: "Prompt caching is not supported, but it is supported in the Anthropic SDKs." Cache reads
  cost 0.1× the input price on Anthropic's own API — 0.05× on Opus 5.5 and Sonnet 5.5 — so a
  long prompt sent again pays 10–20× what it would.
- **The Anthropic format is now widely spoken.** Ollama (from 0.14.0) and vLLM (from 0.11.1) serve
  `/v1/messages` themselves, so tools written for Claude — Claude Code among them — run on local
  models; Ollama's cloud, Together, OpenRouter and DeepSeek serve it too. Much of this needs no
  translation at all.

## What the providers say (research, 2026-11-07, each from its own documentation)

| Host | Speaks Anthropic's format | Speaks OpenAI's | Caching | Cached tokens reported as |
|---|---|---|---|---|
| Anthropic | yes — its own API | only through a layer it calls not production-ready, **uncached** | **opt-in**: `cache_control` (one top-level field caches automatically); writes 1.25× (5 min) or 2× (1 h), reads 0.1× (0.05×, 0.025× on the newest models); from 512 tokens on current models | `cache_creation_input_tokens`, `cache_read_input_tokens` |
| vLLM | yes, from 0.11.1 (`count_tokens` from 0.17.0); tools with the chat parsers | yes | its own prefix cache; nothing billed | — |
| Ollama (on a machine) | yes, from 0.14.0; no `tool_choice`, PDFs, citations or `cache_control`; counts approximate | yes | — | — |
| Ollama's cloud | yes (`https://ollama.com/v1/messages`); Bearer only, never `x-api-key` alone | yes | a cached-input price exists | not confirmed |
| OpenAI | **no** | yes | **automatic**; explicit breakpoints opt-in on GPT-5.6 and later, whose writes cost 1.25× (30 min) or 2.5× (12 h); reads 0.1× (0.05× on GPT-6.1) | `prompt_tokens_details.cached_tokens`, `cache_write_tokens` |
| Together | yes, but for GLM-5.2/5.3 only ("may return malformed output" on others) | yes | automatic; `cache_control` has no effect | `cache_read_input_tokens` / `prompt_tokens_details.cached_tokens` |
| Groq | **no** | yes | automatic, gpt-oss models only; 50% | `prompt_tokens_details.cached_tokens` |
| OpenRouter | yes (`/api/v1/messages`) | yes | passes `cache_control` to Claude; others automatic | Anthropic's fields / OpenAI's |
| DeepSeek | yes (`https://api.deepseek.com/anthropic`) | yes | automatic; a hit costs about 2% of a miss | `cache_read_input_tokens` / `prompt_cache_hit_tokens` |

**Anthropic's own SDK sends its key as `Authorization: Bearer`** — `api_key` and `auth_token`
alike; `x-api-key` is the API's "legacy fallback". So an app built on Anthropic's SDK reaches the
pool with the pool's key as it stands: `Anthropic(base_url=<pool>, api_key=<pool key>)`. The router
accepts `x-api-key` too, for older clients.

## A design

**Two front doors, one pool.** The router serves `/v1/messages` (and `/v1/messages/count_tokens`)
beside `/v1/chat/completions`, with the same keys, workloads, budgets, priorities, queue and
spill. The path still decides how a request is read (plugin-interfaces §2).

**Each host says which formats it speaks.** vLLM and Ollama hosts speak both (at the versions
above; an older engine speaks OpenAI's alone, read from the version it reports). A hosted host
names its format: Anthropic's API is `anthropic`, OpenAI and Groq are `openai`, a provider that
speaks both says so.

**Untouched wherever possible, translated only where not.** A request goes to a host that speaks
its format as it arrived (the model name and key swapped, as today). Where the model is served
only by hosts that do not — Claude for an OpenAI-format app, GPT for an Anthropic-format one — the
router translates the request on the way in and the answer, stream included, on the way out. Two
translators, each a closed, tested mapping: OpenAI → Anthropic and Anthropic → OpenAI. Hardware
first still holds; within a tier, a host needing no translation is chosen before one that does.

**Nothing is dropped without a word.** What a translation cannot carry — Anthropic's thinking
blocks, PDF documents, citations and server-side tools to an OpenAI-format host; an OpenAI field
with no Anthropic meaning — is refused `400` naming it, as a picture sent to a text-only build is
(D143). (Ollama's and Anthropic's own layers ignore such fields silently; the pool does not.)

**Caching on by default where it is opt-in.** A request reaching Anthropic's API that carries no
`cache_control` gets Anthropic's one top-level `"cache_control": {"type": "ephemeral"}` — the
5-minute cache, renewed free on every use — whether it came in Anthropic's format or was
translated from OpenAI's. A named rewrite, like D166's usage request, and a host setting can turn
it off. Where caching is automatic (OpenAI, Together, Groq, DeepSeek) the pool adds nothing.

**Metering learns both shapes.** Anthropic's usage — `input_tokens` counts only what follows the
last cache mark, so the whole input is it plus the cache reads and writes — and OpenAI's cache
writes on its newest models. The catalog's price gains a cache-write rate; the worst case held
before sending counts writes at that rate.

**Claude is reached through Anthropic's own API only.** S11's use of its OpenAI-compatible layer
is retired: uncached, and in Anthropic's own words not production-ready.

**Supersedes** "never translates between APIs" (D164, plugin-interfaces §2, overview): the pool
still passes through wherever a host speaks the request's format, translates only where none
does, and refuses rather than loses.

## What it gives up

- **Translation is new code on the request path**, and where tool calls and structured output lose
  fidelity — the reason it was refused before. Bounded by preferring untouched hosts, refusing what
  cannot be carried, and tests built from each provider's own examples.
- **A prompt sent once to Claude pays 25% more on its input** for a cache write it never reads.
  For chat and agents, which resend the same beginning, the reads repay it many times over.
- **Two formats to keep current**, as the providers change them.

## Questions for the owner

1. **Caching on by default for Claude**, 5-minute cache (recommended), with a host setting to turn
   it off — or off unless the app asks?
2. **What cannot be translated is refused in words** (recommended) — or carried as far as it can be
   and the rest dropped, as Ollama's and OpenRouter's layers do?
3. **Within a tier, a host that needs no translation first** (recommended) — or by load alone?
4. **Retire Anthropic's OpenAI-compatible layer** (recommended), so Claude is always reached
   through its own API?
5. **Build order** (recommended): (a) Anthropic-format apps to Claude, untouched, cached — fixes the
   owner's spend first; (b) OpenAI-format apps to Claude, translated, cached; (c) Anthropic-format
   apps to the pool's machines, untouched to Ollama and vLLM; (d) Anthropic-format apps to
   OpenAI-format-only providers, translated; then (e) the live check, S11's included, with the
   owner's provider keys.

## The owner's answers (2026-11-07)

1. **Caching:** on by default, Anthropic's 5-minute cache.
2. **What cannot be translated:** refused with an unsupported error, unless the app selects the
   silent option — `X-GPM-Unsupported: drop`, per request; what was dropped is still named in
   the answer's `X-GPM-Dropped`.
3. **Which host:** by load alone — a host needing translation is not put after one that does not.
4. **Anthropic's OpenAI-compatible layer:** retired — "it wouldn't be cost effective".
5. **Build order:** as recommended, end to end.

## What was built

Every stage of question 5 but the live check, in one change: the `/v1/messages` front door and
`count_tokens`; the `anthropic-api` engine; vLLM and Ollama serving `/v1/messages` untouched; the
two translators, streams and errors included, refusing what they cannot carry; the cache mark;
metering of cache reads and writes, with `price.cache_write`; the pool's refusals in Anthropic's
error shape; `x-api-key` read as the key and never forwarded (it would have been, before). A
provider speaking both formats is configured as two hosts, one per engine — the simpler mechanism;
what it gives up is one worker count shared between the two.

## The live check (2026-11-08)

Against Anthropic's own API, Claude Haiku 5.5, through a temporary loopback pool — nine requests,
$0.0006 by the pool's own count. Passed: an Anthropic-format request untouched, its prompt written
to the cache (3,328 tokens) and read back on the next; a stream; `count_tokens`, free; `x-api-key`
as the pool's key; an OpenAI-format request translated to Claude, reading the same cache; an
OpenAI-format stream and tool call translated; `n` refused before anything was sent. The pool's
cost agreed with Anthropic's counts at the catalog's prices (`X-GPM-Cost-USD` is rounded to a
millionth of a dollar; the request log keeps the exact figure). **Not yet live:** an
Anthropic-format app reaching an OpenAI-format provider (no such key in `env`), and Ollama's and
vLLM's own `/v1/messages`.

## Build stages

1. Decisions and specification, once the questions are answered.
2–5. As question 5 orders them, each against fake providers speaking each format's requests,
   streams, usage and errors.
6. **Live, asked first:** a few cents per provider the owner holds a key for — S11's check with it.
