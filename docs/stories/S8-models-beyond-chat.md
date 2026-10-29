# S8 — Models beyond chat: speech to text, pictures and audio in, rerank

> Status: **decided (D142–D149), specification written** (2026-10-26); the build is next. An
> architecture review (14 findings) and a security review (9) are folded in — see *Review* at the
> end — and one finding shipped on its own first (D141, gpm-server 0.34.1). The specification is
> `docs/spec/app-contract.md` §2, `plugin-interfaces.md` §2 (engine interface version 2),
> `hosts-routing-capacity.md` §4.4, `host-agent.md` §2.1 and the threat model (T7, T32–T34).
> Part of the [feature list](README.md). Text-to-speech and image generation are named only to keep
> the design open for them; they are not in this build. Installing the engines on local machines
> is [S9](S9-engines-installed-on-local-machines.md).

## The story

An app sends a recording to be transcribed, a picture or a sound with a chat message, or a query
and documents to be ranked — to the same URL, with the same key, without knowing where the GPU
is — and keeps every guarantee chat has: only catalogued models, nothing downloaded because a
request asked, workers and leases as for chat, the build served always reported.

## Why

- **Asked for by the owner** (2026-10-26): models that are not chat, Whisper-style speech to text
  first — and Hebrew, which general models handle poorly unless told the language.
- **A picture can already be sent, and nothing checks where it lands.** It travels inside an
  ordinary chat request (`image_url` parts on `/v1`, `images` on Ollama's `/api/chat`) and is
  passed through untouched. Whether it is seen depends on the build the catalog resolved to, and
  on the engine's version: the first adopter's Gemma 4 MLX builds are text-only under Ollama
  0.32.15 and take pictures under 0.40.11, from the same files.
- **The pool only reads JSON**, with no limit on a body's size: an upload's path is not in the
  allow-list (`unknown_path`), and if it were, the model could not be read from it
  (`model_missing`); the agent's proxy on vLLM hosts with several models reads JSON too.
- **The engines already serve these** (sources at the end): vLLM serves OpenAI's
  `/v1/audio/transcriptions` and `/translations` for Whisper and newer speech models, and
  `/v1/rerank` for pooling models; Ollama serves `/v1/audio/transcriptions` by giving the audio to
  an audio-capable chat model; whisper.cpp's server runs Whisper on a Mac's GPU.

## The owner's answers (2026-10-26)

1. **First build: the shared plumbing, speech to text, the input guard and rerank.**
   Text-to-speech and image generation come later, through vLLM-Omni as an engine of its own.
2. **The laptop runs the newest Ollama.** Ollama is the only engine that serves chat on a Mac,
   so it is kept current rather than pinned. Under 0.40.11 both Gemma 4 MLX builds take pictures;
   the cost, accepted, is that `gemma4:26b-mlx` and `gemma4:e4b-mlx` no longer stay loaded
   together and reload when alternated (§8).
3. **Non-chat requests count as requests and latency**, as embeddings do: each holds a worker
   and is measured for latency and demand; the measures built on output tokens (worker step-up
   S5, bad-value replacement D75, the machine history's speed) leave them out — by an explicit
   filter on the request's kind (§4), not by the accident of an empty token count.
4. **Pictures and recordings go to rented hosts as text does.** The risk is the one text already
   carries (threat model T7), accepted and written down for media as well; no new setting.
5. **Transcription is Whisper; for Hebrew, ivrit.ai's fine-tune, everywhere it can run**: its
   safetensors build on rented vLLM hosts, and its ggml build on the laptop through a new
   **whisper.cpp engine** (§5). **Gemma is not offered for transcription**: the owner's own
   measurements found it very poor at Hebrew, and a run here agreed (§8). Ollama therefore
   serves no `transcribe` at all.
6. **The duplicate model field was fixed first** (D141, 0.34.1), ahead of this story.
7. **Rented hosts run vLLM by default** — proposed by the owner, to be decided with this story
   (§6a). Rented machines are Linux with NVIDIA GPUs, where vLLM batches continuously and serves
   kinds Ollama cannot (Whisper, rerank); Ollama stays the default for local and fixed machines.
8. **The pool manages whisper.cpp** (owner, 2026-10-26): the operator installs the program; the
   host's agent fetches the model and runs the server (§5), as agents already do for Ollama's
   models and for vLLM's processes.

## Design

### 1. What a request asks for: its kind

A **kind** is what a request asks a model to do. Each engine plug-in declares the kinds it can serve and maps
the paths *it* serves to them — never through a shared list another engine would inherit, so
Ollama is never thought to serve rerank or transcription (D93) — and the router forwards only those exact paths: an allow-list, because several
servers in this space have routes that download or swap models (`POST /load` on whisper.cpp's
server, `POST /v1/models/{id}` on speaches).

| Kind | Paths | Engines |
|---|---|---|
| `chat` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`; Ollama's `/api/chat`, `/api/generate` | all but whisper.cpp |
| `embed` | `/v1/embeddings`; Ollama's `/api/embed`, `/api/embeddings` | Ollama, vLLM |
| `transcribe` | `/v1/audio/transcriptions`, `/v1/audio/translations` | vLLM, whisper.cpp — not Ollama (answer 5) |
| `rerank` | `/v1/rerank` only — vLLM's `/rerank` and `/v2/rerank` add shapes to parse with no stated need | vLLM |

### 2. What it carries: inputs, and media only inline

A request's **inputs** are what it carries besides text, read from a closed table: `image` — an
`image_url` or `input_image` part, Ollama's `images`; `audio` — an `input_audio` part, or the
upload of a `transcribe` request. They are read alike in a chat message, in what an embedding
request embeds (`messages`, or `input` and its items) and in a ranking's query and each document,
since vLLM takes pictures in all of them. The fields they are read from (`messages`, `input`, `query`,
`documents`, and a message's `content` and `images`) follow D141's rule: once, spelled exactly, or
`400 ambiguous_field` — an engine matching keys in any letter case would otherwise see a picture
the pool did not. **Media must travel inside the request**: an `image_url` must be
a `data:` URL, and any other part type (`audio_url`, `video_url`, `file`, `input_file`, or one
the table does not know) is refused `400 media_not_inline`. An engine given a URL fetches it from
the host it runs on — the operator's own network for a local or fixed host, the provider's for a
rented one — and engine errors come back to the app verbatim; one engine also accepts a local
file path. Refusing is the pool's guarantee — and the only one: vLLM v0.29.0 has no setting that
turns remote fetching off (checked in its source, stage 3).

### 3. What a build can do: declared, and observed where the engine can say

Each catalog variant gains two fields, read three ways like `enforces_schema` (catalog.py):

- **`serves`** — the kinds it answers. Unset means `[chat, embed]`, which is how every entry
  behaves today. `transcribe` implies audio input; nobody has to write both.
- **`inputs`** — what it accepts besides text. **Unset means not stated, and is usable**; only an
  explicit list refuses. A picture prefers builds that list `image`, falls back to builds that
  state nothing, and never goes to one whose list leaves it out.

**On an engine that can say, the pool asks.** What an Ollama build accepts depends on the
Ollama running it — the same files take pictures under 0.40.11 and not under 0.32.15 — so the
supervisor reads each resident tag's capabilities (`/api/show`: `vision`, `audio`) per host and
publishes them, and dispatch uses the declaration narrowed by what was observed. vLLM and
whisper.cpp cannot say, so their declaration stands.

The router reads `serves` and `inputs` **from its own copy of the configuration**, by tag, as it
already reloads it — not from the published host rows, whose fixed shape an older router would
fail to read during a staggered restart. Observed capabilities are published as a field of their
own, which an older router ignores.

**Two answers, defined at configuration load:**

- **`400 kind_not_served` / `400 input_not_accepted`** when no build of the model that any
  configured host, rented profile or rented engine could hold would ever take the request —
  every one explicitly excludes it. Refused at once, never retried. A name is read as dispatch
  reads it: a catalogued model first, then a build tag, judged by that build; a workload key by
  its workload's builds; an uncatalogued name states nothing. What each name is judged by is
  worked out when the configuration loads, so the request path never looks an engine up.
- **`503 no_eligible_host`**, as for a schema requirement today, when such a build exists but no
  ready host holds one, with the detail naming the kind or input. The exception is a picture or
  sound that is the only thing in the way: every configured host that could hold an accepting
  build is ready and either does not hold it or reports, through its engine, that it does not
  take it; no host the pool could rent would; and no host has failed this request already —
  `400 input_not_accepted`. A host that is not ready, or one the request failed on, may be the
  one that takes it, so either is a `503`.

### 4. Router: read once, bound everything

- **One reading per request.** The body's facts — model, stream, schema, kind, inputs — are read
  once into one object, off the event loop above about a megabyte; today three readers each parse
  the whole body on the loop, which a 30 MB picture turns into a stall for every stream in flight.
- **Bounded bodies.** `pool.max_request_mb` (default 24, under the 25 MB vLLM and Ollama accept
  for an upload): a declared length over it is refused `413 request_too_large` before the body
  is read, and a chunked body is counted as it arrives. A pool-wide budget for request bytes held
  — being read, queued, in flight and kept for a resend (D62) — answers `503` with a retry when
  full: a declared length is taken from it before the body is read, a chunked body as it arrives,
  and whatever was taken is given back exactly once, however the request ends. A
  request `Content-Encoding` other than `identity` is refused. The form parser that would spool
  uploads to the router's disk is never used.
- **Uploads, rebuilt.** A `transcribe` body is parsed strictly and **rebuilt**: a boundary the
  router chooses, canonical part headers, the `model` part rewritten, every file's bytes
  untouched, `Content-Type` set by the router. Refused `400 bad_upload` before any host is
  chosen: duplicate `Content-Type` headers or boundary parameters, `name*=`, folded headers,
  `Content-Transfer-Encoding`, nested multipart, bare line feeds, a preamble or epilogue, a
  missing closing delimiter, a second `model` part in any letter case, a `model` part with a file
  name, more parts than a small limit. The engine then reads exactly what the pool checked — the
  same rule D141 set for JSON, held by construction.
- **The served build named in a header.** The router sends `X-GPM-Model: <tag>` with every
  request (an app's own `X-GPM-*` headers are already stripped), so the agent's proxy routes an
  upload without parsing one; it falls back to the JSON body for an older router.
- **Long work is not a failure.** A timeout after the whole body was sent is `504
  generation_timeout`: no re-run elsewhere, no failure counted against the host, not retried by
  the SDK. Today it reads as a host that never answered — counted, re-run on a second host, and
  retried by the SDK for up to half an hour.
- **Measured by kind.** The request log gains `kind` — read from the path, so a request refused
  before its body is understood is logged as what it asked for — and `request_bytes`; the token-based
  measures and the latency curves (worker readings, D75, workload latency, the concurrency curve,
  machine history) filter by kind explicitly.
- **No new queue** (D23). A transcription waits first come, first served, and a long recording
  holds its worker for as long as it is transcribed.

### 5. Engines

- **vLLM.** The `transcribe` and `rerank` paths, and uploads. Three launch changes:
  - **Each model's length capped at its own maximum.** The pool sends `rented.context_length`
    (8192) as every process's `--max-model-len`; Whisper's decoder takes 448, a reranker its own.
    The launcher reads the maximum from the model's `config.json`, as it already reads its family.
  - **Audio decoding is a property of the image.** vLLM's own image leaves out the audio extra;
    `vastai/vllm:v0.29.0` adds it. `rented.images[].capabilities` says which image carries
    `audio-decoding`, and a machine's image is chosen so it carries what its builds `require` —
    known before renting, as the driver floor is. The agent only **verifies** it on the host (the
    `vllm` launcher's own interpreter, `python -I`, importing the extra) and fails the host with
    the reason if the image lied; without the extra vLLM starts and answers every transcription
    `400 Invalid or unsupported audio file`, saying why only in its log.
  - **Rerankers** are catalogued as sequence-classification models, which vLLM runs as pooling
    models. Nothing yet checks a reranker's `config.json`: a refusal at load for one needing
    per-model launch overrides is not built (review, 2026-10-29).
- **Ollama.** Pictures and sound inside chat on the hosts and tags where §3 observes them;
  never transcription or rerank.
- **whisper.cpp** — a new engine, `whisper-cpp`, for Whisper builds on a Mac's GPU (or any machine
  with a whisper.cpp build), **managed by the pool through the host's agent** (answer 8):
  - **The operator installs the program** — `whisper-server`, and `ffmpeg` for other audio formats
    — as they install Ollama: the pool never installs an engine on a machine it does not rent.
  - **The host carries an agent**, refused at load otherwise, as a vLLM host is (D90). An agent
    manages one engine, so a Mac running Ollama and whisper.cpp runs two agents, each on its own
    loopback port with its own key.
  - **The agent fetches the model** the configuration names, at preparation, never because a
    request asked: a whisper.cpp build's tag names **one file in a hub repository** —
    `ggerganov/whisper.cpp/ggml-large-v3-turbo-q5_0.bin` — because such repositories hold every
    size and quantisation side by side. The file is checked against the repository's listing; a
    name that climbs out of the models directory is refused, as for any hub fetch (D40).
  - **The agent runs the server**: `whisper-server -m <the file> --host 127.0.0.1 --port <its
    engine port> --inference-path /v1/audio/transcriptions --convert`, the program found on its
    `PATH` or at `engine_binary` in its own settings — never a command or a path from the pool.
    It restarts the server when the declared model changes and when it dies, and reports the
    program's version, `ffmpeg`, the model on disk and whether the server answers. No new verb:
    *put models*, *engine settings* and *facts* already cover it.
  - One model per server, so the host holds the one build it declares; one request at a time —
    the server holds a global lock — so the host has one worker; ready when `GET /` answers
    (1.9.2 has no health route).
  - The server has no authentication: it listens on loopback only, and the pool reaches it there
    or through a tunnel; plain `http` or `https` to another machine is refused at load. Its
    `/load`, which takes a path from the request, is never reachable through the pool.
  - It takes the OpenAI form fields it knows (`file`, `language`, `prompt`, `temperature`,
    `response_format`), ignores `model`, and with `--convert` turns other formats into WAV.
- **The agent's proxy** applies the same path allow-list, routes by `X-GPM-Model`, and answers
  with fixed error strings instead of listing the host's models and addresses.
- **The console's host check** fires a chat request at a chat build on the engine's own path;
  today it posts Ollama's `/api/generate` to the first model by name, wrong for vLLM already and
  for a speech or rerank build.

### 6a. Rented hosts run vLLM by default, and an engine that cannot serve a kind is caught early

`rented.engine` defaults to `vllm` instead of falling back to the pool's `engine` (amends D93;
the pool's own default stays `ollama`, for local and fixed machines). A pool that rents and set
neither would otherwise change engine on upgrade, so configuration load refuses it when its
rented models have no vLLM build, and says how to keep Ollama: *"rented hosts run vLLM by
default; gemma4:12b has no vLLM build — add one, or set `rented.engine: ollama`."*

An operator who wants Whisper on hosts that run Ollama is told **at the first point it is
knowable, with the reason and the fix**, from the kinds each engine declares (§1):

1. **Configuration check** (`gpm config plan`, the console's Apply), before anything is rented
   or written: a model whose builds serve `transcribe`, held only by hosts whose engine cannot,
   is refused — *"whisper-hebrew is a transcription model; the hosts that would hold it run
   ollama, which cannot serve transcription. Rent with vLLM, or hold it on a whisper.cpp host."*
   A build for an engine that claims a kind that engine does not serve is refused at load too.
2. **The Profiles tab**: a Whisper build is shown for a profile whose hosts run Ollama, greyed,
   with the same reason — as D139's keep-models checkbox is greyed with its reason.
3. **A workload created by a program** (D117): refused with the reason before anything is rented.
4. **A request that arrives anyway**: `400 kind_not_served`, the reason in the detail, decided
   from the configuration — answered at once, never retried.

### 6. SDK

Sync and async (`AsyncPoolClient` the same with `await`); transcription and rerank always on the
`/v1` paths, whichever request shape the client was made for.

- **`PoolClient.transcribe(model, file, *, language=None, prompt=None, response_format=None,
  temperature=None, session_id=None) -> Transcript`** — `file` a path, bytes or a binary file
  object. `Transcript` carries `text`, `segments` (when the build returns them, as
  `verbose_json` does), `language`, `served_model`, `host`, `wait_s` and `raw`. The upload is
  held in memory so a retry can resend it.
- **`PoolClient.rerank(model, query, documents, *, top_n=None, session_id=None) ->
  list[Ranked]`** — best first; `Ranked` carries `index` (into `documents`), `score` and
  `document`.
- **`image_part(source, mime=None)`** — a chat content part carrying a picture inline as a
  `data:` URL, from a path, bytes or a binary file object; the type is guessed from the file
  name when not given. Pictures must travel inline (§2), and this is the way not to get that
  wrong. Chat itself is unchanged: `chat()` already takes any message parts.
- **Errors**: the new refusals are `PoolRequestError`s with their reason (`kind_not_served`,
  `input_not_accepted`, `media_not_inline`, `bad_upload`, `request_too_large`,
  `generation_timeout`) and are never retried; `503 uploads_full` is waited out like any `503`.

The official OpenAI client works unchanged against the pool's `/v1` paths, and gets the waiting
by being given `httpx.Client(transport=pool_transport())` as its HTTP client.

The limits named in §4 are `pool.max_request_mb` (default 24) and `pool.max_held_request_mb`
(default 512), the second answered `503 uploads_full` when full.

### 7. Console

The **Profiles tab's** variant rows show and edit `serves` and `inputs` (the Models screen is
read-only since D112). The hub's tasks map `automatic-speech-recognition` to `transcribe`,
`text-ranking` to `rerank` and `image-text-to-text` to image input; `text-classification` is
ambiguous and maps to nothing.

### 8. The laptop (answers 2 and 5)

- **The newest Ollama, kept current.** 0.40.11 (notarized, checksum matched) runs since
  2026-10-26; 0.32.15 is kept beside it for a rollback. Both Gemma 4 MLX builds take pictures.
  Loading either of them stops the other — with 28.7 and 20.4 GiB of the 37.4 GiB Metal budget
  free; the log shows the Mac's free system memory just before each stop, a likely cause, not a
  verified one — so a pool alternating between them pays a reload of a few seconds; the pool's
  `on_demand` residency (D39) already lets the engine evict.
- **ivrit.ai's Whisper turbo through whisper.cpp**, as a second host on the same machine
  (`workers: 1`), with its own agent, outside Ollama's memory budget.
- **Measured, 2026-10-26**, on 30 recordings (about 5 minutes) of FLEURS's Hebrew validation
  set — read speech, not in ivrit.ai's training data — scored after removing vowel marks and
  punctuation:

  | System | Character errors | Word errors | Speed (× real time) |
  |---|---|---|---|
  | ivrit.ai Whisper turbo, ggml (official file), whisper.cpp | 10.1% | 21.9% | about 13× |
  | ivrit.ai Whisper turbo, ggml `q5_0`, whisper.cpp | 10.4% | 22.2% | about 13× |
  | Gemma 4 E2B, Ollama 0.32.15, `language=he` and a Hebrew instruction | 27.2% | 62.4% | about 1.6× |
  | Gemma 4 E2B, Ollama 0.32.15, `language=he` | 29.9% | 66.7% | about 2× |
  | Gemma 4 E4B (MLX), Ollama 0.40.11, `language=he` | 75.0% | 91.6% | about 4× |
  | Gemma 4 E4B (MLX), Ollama 0.40.11, `language=he` and a Hebrew instruction | 86.6% | 95.7% | about 1.7× |

  E2B sometimes answered in Latin transliteration, and E4B wrote the Hebrew in Arabic letters. Word error rates are high for every system
  because FLEURS writes some names in English and numbers as digits; the comparison between
  systems is what the table is for. The quantised file costs nothing measurable, at a third of
  the size.

## What it gives up

- **No transcription on Ollama.** A pool whose only hosts run Ollama cannot transcribe; it is
  told so when its configuration is checked (§6a).
- **On the laptop, 26b and E4B reload when alternated**, the price of a current Ollama (answer 2).
- **whisper.cpp is one request at a time** — one worker per server — and the machine runs a second
  agent for it.
- **Media must be inline.** An app that sends picture URLs inlines them.
- **Pictures and recordings reach rented hosts as text does** (answer 4): a pool that includes
  marketplace hosts must not carry media its owner would not show their operators.
- **Requests and latency only** (answer 3): rented hosts that mostly transcribe or rerank are not
  tuned (S5) or replaced (D75) on evidence.
- **A workload's latency target is per request**, and a transcription's latency grows with the
  recording; a target that suits short clips is missed by long ones. Noted, not solved.
- **Streams are the engine's own** — vLLM's transcription chunks are not OpenAI's events.
- **No text-to-speech or image generation** in this build (answer 1).

## Supersedes or amends

- **App contract §2**: the new paths; new answers `400 kind_not_served`, `400
  input_not_accepted`, `400 media_not_inline`, `400 bad_upload`, `413 request_too_large`, `504
  generation_timeout`, and a `503` for a full upload budget. **The contract version stays `1`**:
  existing requests keep their meaning — unset `inputs` states nothing — and every new answer
  refuses only what could not have worked or was never safe.
- **Plug-in interfaces**: an engine is told the request's content type, maps its own paths to
  kinds, reads inputs into the one facts object, and may report observed capabilities — a new
  engine interface version.
- **D93**: `rented.engine` defaults to `vllm` rather than the pool's engine (§6a).
- **Catalog and images**: `serves` and `inputs` on each variant; `capabilities` on each rented
  image; a per-model length cap in the vLLM launcher (D41 unchanged).
- **Threat model**: T7 names pictures and recordings; new entries for media fetched by URL
  (refused) and for upload parsing differences (rebuilt).
- **Request log**: `kind` and `request_bytes`. D23's single queue and D39's no-download rule
  stand unchanged.

## Unverified

| Assumption | How it is checked |
|---|---|
| vLLM v0.29.0 starts Whisper with `--max-model-len` capped at 448, and runs a sequence-classification reranker with no extra flag | Stage 3: one cheap rented host, minutes, capped, asked first |
| The Gemma 4 NVFP4 builds the catalog serves on vLLM keep the vision tower | Stage 3, same host |
| whisper.cpp's server queues a second request rather than refusing it | Stage 3: on the laptop |

**Checked, 2026-10-26.**
- *On the first adopter's laptop:* Ollama 0.40.11 gives both Gemma 4 MLX builds pictures and E4B
  audio, but will not keep the two loaded together on a 48 GB Mac; 0.32.15 does. The Hebrew
  measurements in §8. whisper.cpp 1.9.2's server answers OpenAI's transcription path with `--inference-path`,
  ignores `model`, has no `/health` (`GET /` is 200) and no `/v1/models`.
- *From the images' published contents:* `vastai/vllm:v0.29.0-cuda-13.0` and `-cuda-12.9` add
  vLLM's audio extra in a layer of their own (`install-vllm-audio-extras.sh`, github.com/vast-ai/
  base-image, which fails the build if any member does not import); `vllm/vllm-openai:v0.29.0`
  has none of it. At v0.29.0 the extra is `av`, `scipy`, `soundfile`, `soxr`,
  `mistral_common[audio]`.

Sources, read 2026-10-26: vLLM `docs/serving/online_serving/speech_to_text.md`,
`docs/models/supported_models.md#transcription`, `docs/models/pooling_models/README.md`,
`docs/getting_started/installation/gpu.cuda.inc.md`, `setup.py` at v0.29.0; Ollama
`server/routes.go`, `openai/openai.go` (`FromTranscriptionRequest`), release notes 0.20.0–0.40.11;
whisper.cpp `examples/server`; ivrit.ai's models on Hugging Face (`ivrit-ai/whisper-large-v3-turbo`
and its `-ggml` and `-ct2` builds, Apache-2.0).

## Review (2026-10-26)

| Finding | Where it went |
|---|---|
| Ollama reads a second, differently cased `model` key — a catalog bypass in released code | Fixed first: D141, 0.34.1 |
| `inputs` unset = none would break existing pools and force a contract major | §3: three-valued; contract stays `1` |
| What an Ollama build accepts depends on the host's Ollama | §3: observed per host |
| Whisper would not start: every vLLM process gets 8192 as its maximum length | §5: capped per model |
| An agent-learned audio fact arrives after the host is paid for | §5: declared per image, verified by the agent |
| Unbounded bodies; three parses on the event loop; queued uploads unbounded | §4 |
| The engine interface cannot see a multipart boundary; Ollama would inherit rerank | §1, *Supersedes* |
| Router, engine and proxy can frame the same upload differently | §4: strict parse, then rebuilt |
| Media URLs make the engine host fetch from its network | §2: inline only |
| A long transcription reads as a host failure and is re-run elsewhere | §4: `504`, no re-run |
| "No new code" for answer 3 was false: latency and worker measures do not filter by kind | §4: `kind` in the log, explicit filters |
| The host table's fixed shape breaks a staggered restart | §3: the router reads from its configuration |
| 400 vs 503 undefined when engines differ per host | §3: defined at load over reachable builds |
| The console fields belonged on the Profiles tab (D112) | §7 |
| The proxy forwards any path and lists the host's models in errors | §5 |
| The host check posts `/api/generate` to the first model by name | §5 |
| Rerankers needing per-model overrides | §5: a refusal at load, not built yet |
| Pictures and recordings on untrusted hosts (T7) | Answer 4: as text, documented |
| Decisions and spec were the last stage | Build stages: first |

## Build stages

0. **Checks before code** — done (§8 and *Checked*).
1. **Decisions and specification** — done 2026-10-26: D142–D149, written into the contract, the
   plug-in interface, the catalog, the agent, the threat model and `decisions.md`.
2. **Router and catalog** — done 2026-10-26: the facts object, the size limit and held-bytes budget, upload
   rebuilding, kinds and inputs, media inline, the two guards and their answers, `504`, the
   request log's columns and the measures' filters — against fakes: vLLM transcription (with a
   *missing audio extra* mode and a length check), a rerank endpoint, Ollama's `/api/show` with
   per-instance capabilities and its transcription, and whisper.cpp's server; uploads chunked,
   with a quoted boundary, with `model` after the file, and the refusal cases.
3. **Engines, agent and the laptop** — built and tested against fakes 2026-10-26; the laptop's
   whisper.cpp host and the live rented check remain: vLLM's paths, the length cap, image capabilities and the
   agent's check, the proxy's header routing and allow-list, Ollama's observed capabilities, the
   whisper.cpp engine and the agent's management of it (fetching one file, running and restarting
   the server, its facts), the host check. The laptop's whisper.cpp host. One live
   check on a rented host — ivrit.ai's Whisper turbo and a small reranker, minutes, capped, asked
   first.
4. **SDK and console** — done 2026-10-26: the two methods; the Profiles tab's fields; the hub's
   task mapping.

**Written ahead of the code, on this branch** (2026-10-26), and to be checked against it before
merging: how apps use it — `client/README.md` (*Beyond chat*) and the app contract §2 and §4 —
how operators set it up, `docs/models-beyond-chat.md`, and runnable examples in `examples/`
(a picture, a transcription, ranking, the OpenAI client, and the configuration they assume).
