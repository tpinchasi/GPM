# Beyond chat: speech to text, pictures and ranking

> For operators. Apps call these as described in the
> [client SDK's README](../client/README.md#beyond-chat-pictures-speech-to-text-ranking) and the
> [app contract](spec/app-contract.md#beyond-chat-what-a-request-asks-for-and-what-it-may-carry).
> Working examples are in [examples/](../examples/).

A pool serves more than chat: recordings turned into text, pictures inside a chat message, and
documents ranked against a query. Each is a model in the catalog like any other, with the same
guarantees: only catalogued models are served, nothing is downloaded because a request asked,
and the build that answered is reported on every response.

## What each engine can serve

| | Chat | Embeddings | Pictures in chat | Speech to text | Ranking |
|---|---|---|---|---|---|
| **vLLM** (rented hosts by default) | ✓ | ✓ | builds that take images | ✓ Whisper and newer speech models | ✓ |
| **Ollama** (Macs, local machines) | ✓ | ✓ | builds that take images | — | — |
| **whisper.cpp** (Macs) | — | — | — | ✓ Whisper | — |

If you configure something the engines you run cannot serve, the configuration check says so
before anything is rented or written (see *When a model cannot be served*).

## Speech to text on rented hosts

Rented hosts run vLLM unless `rented.engine` says otherwise. Name the model and where its weights
come from — for a vLLM build, the `tag` is its repository on the Hugging Face hub — and say it
transcribes:

```yaml
pool:
  model_set: ["gemma4:31b", "whisper-large-v3-turbo"]

catalog:
  whisper-large-v3-turbo:
    variants:
      - { tag: "openai/whisper-large-v3-turbo", engine: "vllm", serves: [transcribe] }

rented:
  models: ["gemma4:31b", "whisper-large-v3-turbo"]     # or a model profile, or a workload's models
  images:
    - { image: "vastai/vllm:v0.29.0-cuda-13.0", min_driver: "580", capabilities: [audio-decoding] }
```

- **The weights are fetched when a host is prepared**, by the pool's agent on the host, from the
  hub (or the mirror in the host's own `HF_ENDPOINT`) — never because a request asked.
- **The image must be able to decode audio.** vLLM needs its `audio` extra for speech, and
  vLLM's own image leaves it out; some provider builds add it. Mark the images that carry it with
  `capabilities: [audio-decoding]`: the pool chooses one of those for a host that will hold a
  speech model, and the host's agent checks the claim before the host serves. An image that
  lacks it would otherwise start, and answer every recording `400 Invalid or unsupported audio
  file`.
- **Nothing to set for the model's length.** The pool caps each model at the maximum its own
  configuration states — Whisper's is far below what chat models are given.
- **A language of its own?** Fine-tunes of Whisper for one language are catalogued the same way:
  for Hebrew, for example, `ivrit-ai/whisper-large-v3-turbo`. Apps should pass `language`.

## Speech to text on a Mac

Ollama cannot run Whisper. On a Mac, Whisper runs in **whisper.cpp**, on the Mac's GPU, as a host
of its own beside the machine's Ollama. You install the program; the pool's agent on the Mac
fetches the model and runs the server, as an agent already does for Ollama's models.

1. **Install whisper.cpp**, and `ffmpeg` so it can take any audio format:

   ```sh
   brew install whisper-cpp ffmpeg
   ```

2. **Start an agent for it.** An agent manages one engine, so a Mac that already runs one for its
   Ollama runs a second, on another loopback port:

   ```sh
   gpm-agent -c ~/.config/gpm/whisper-agent.json init --engine whisper-cpp \
     --engine-url http://127.0.0.1:8178 --port 8096 --models-path ~/.local/share/gpm/whisper
   gpm-agent -c ~/.config/gpm/whisper-agent.json serve
   ```

   `init` prints the agent's key once; give it to the supervisor's environment as
   `WHISPER_AGENT_KEY`, never to an app. The agent finds `whisper-server` on its `PATH`, or where
   `engine_binary` in its settings file says.

3. **Name the host and the model.** A whisper.cpp build's tag names **one file** in a Hugging
   Face repository, since those repositories hold every size side by side:

   ```yaml
   hosts:
     - id: mac-whisper
       kind: local
       engine: whisper-cpp
       workers: 1                       # the server transcribes one recording at a time
       transport: { type: http, base_url: "http://127.0.0.1:8178" }
       agent: { url: "http://127.0.0.1:8096", bearer_env: WHISPER_AGENT_KEY }
       models: ["whisper-large-v3-turbo"]

   catalog:
     whisper-large-v3-turbo:
       variants:
         - { tag: "openai/whisper-large-v3-turbo", engine: "vllm", serves: [transcribe] }
         - { tag: "ggerganov/whisper.cpp/ggml-large-v3-turbo-q5_0.bin", engine: "whisper-cpp", serves: [transcribe] }
   ```

   Once applied, the agent downloads the file and starts `whisper-server` with it on
   `127.0.0.1:8178`, answering on OpenAI's transcription path; naming another file restarts it
   with that one, and the agent restarts it if it stops. The `q5_0` file is a third of the full
   one's size: on Hebrew speech, the `q5_0` build of a Whisper turbo fine-tune made 10.4% character
   errors against the full file's 10.1%. A language's own fine-tune is named the same way — for
   Hebrew, `ivrit-ai/whisper-large-v3-turbo-ggml/ggml-model.bin`, published at full size only.

Apps ask for `whisper-large-v3-turbo` and get whichever host is free: the Mac first, a rented host
for the overflow.

- **Loopback only.** whisper.cpp's server has no authentication, so the agent starts it on
  `127.0.0.1`; a whisper.cpp host on another machine is reached through a `tunnel`, and the
  configuration check refuses plain `http` to one. The server's model-loading route, which takes a
  path from the request, is never reachable through the pool.
- **Check it through the pool:**

  ```sh
  curl -s "$GPM_URL/v1/audio/transcriptions" -H "Authorization: Bearer $GPM_API_KEY" \
    -F model=whisper-large-v3-turbo -F file=@sample.wav -F language=en
  ```

## Pictures in chat

A picture goes only to a build that accepts pictures.

- **On Ollama, the pool asks.** What an Ollama build accepts depends on the Ollama version
  running it, so the pool reads it from each host (`ollama show` lists `vision`). Keep Ollama
  current: newer releases add pictures to builds that had none.
- **On vLLM, say so** where you know it, from the model's card:

  ```yaml
  catalog:
    gemma4:26b:
      variants:
        - { tag: "gemma4:26b-mlx", requires: ["apple-silicon"], engine: "ollama" }
        - { tag: "google/gemma-4-26B-A4B-it", engine: "vllm", inputs: [image] }
  ```

  `inputs` unset means *not stated*: the build may still be sent a picture when no build that
  states `image` is free. Give an explicit list — `inputs: []` for text only — to refuse.

- **Inline only.** Apps send pictures inside the request; a link is refused, because the engine
  host would fetch it from its own network.

## Ranking

vLLM serves `/v1/rerank` for cross-encoder rerankers — models whose configuration names a
sequence-classification architecture:

```yaml
catalog:
  bge-reranker-v2-m3:
    variants:
      - { tag: "BAAI/bge-reranker-v2-m3", engine: "vllm", serves: [rerank] }
```

A reranker that needs model-specific launch overrides is refused at load rather than started
wrong.

## When a model cannot be served

You are told at the first point it is knowable, with the reason and what to change:

1. **The configuration check** (`gpm config plan`, or Apply in the console), before anything is
   rented or written:

   > whisper-large-v3-turbo is a transcription model; the hosts that would hold it run ollama,
   > which cannot serve transcription. Rent with vLLM, or hold it on a whisper.cpp host.

2. **The console's Profiles tab** shows such a build greyed, with the same reason.
3. **A workload a program creates** is refused, with the reason, before anything is rented.
4. **A request that arrives anyway** is answered `400 kind_not_served` at once.

**Rented hosts run vLLM by default.** A pool that rents and relies on that default needs a vLLM
build of every model its rented hosts hold; one that has none is refused at load, with a note on
keeping Ollama: set `rented.engine: ollama`. Ollama on rented hosts still serves chat and
embeddings — not speech or ranking.

## Limits and what to tell app developers

| Setting | Default | What it bounds |
|---|---|---|
| `pool.max_request_mb` | 24 | One request body — about 12 minutes of 16 kHz WAV, 20 or more of MP3 |
| `pool.max_held_request_mb` | 512 | Request bytes the router holds at once — queued, in flight, kept for a resend |

- A transcription holds a worker for as long as it runs; there is no separate queue for short
  calls.
- Long recordings should be split by the app. One that outlasts the pool's read time (`pool.upstream_read_timeout_s`, 300 s) is `504
  generation_timeout` and is not re-run.
- **Privacy:** pictures and recordings go wherever text goes. A pool that includes marketplace
  hosts must not carry media its owner would not show those hosts' operators
  ([threat model](threat-model.md) T7).
