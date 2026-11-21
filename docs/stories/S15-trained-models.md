# S15 — Trained models: say what the model takes and returns, and the pool does the rest

> Status: **built against fakes** (2026-11-19) — D178–D182; stages 1–6, with renting for ONNX builds only; container renting, the hub token on rented machines and ONNX on Windows were built in [S16](S16-trained-models-rented-and-windows.md). Part of the
> [feature list](README.md).

## The story

An operator has a model they trained themselves — an image classifier, an object detector, a
forecaster, a model over a table of features — that is neither a language model nor speech to
text. They add it to the pool by giving the model and saying what it takes in and what it gives
back. The pool does everything else, as it does for a chat model: puts it on the machines that
should hold it, installs what runs it, starts it, checks it answers, routes requests to it, waits
when every machine is busy, finds each machine's worker count, and rents a machine for it where the
operator allows. The app knows one URL, one key and one call: `POST /v1/predict`.

## Why

- **The owner asked** (2026-11-18): a custom trained model, not an LLM, an SLM or Whisper, served
  through GPM — "a custom plugin that will require the user to input those details. The GPM should
  abstract all the rest."
- **Today it cannot be done without code.** A request's kind is one of chat, embeddings,
  transcription and ranking (D142), a catalog build can serve only those, and the agent runs only
  Ollama, vLLM and whisper.cpp. An engine plug-in can be written (`gpm.engines`), but it leaves the
  operator to install, start and feed the server on every machine, and its requests are read as
  chat.
- **The pool already has the rest.** Hosts in priority order, catalog builds per engine, the agent
  fetching a build from a model hub and running a server on loopback (D90, D148), engines
  installed from a hash-locked set (D158), workers found per host (S5), renting under a lease.

## The owner's answers (2026-11-18)

1. **Format: ONNX, and container images too.** ONNX for a model that is a file; a container image
   for one that needs its own code.
2. **Source: a model hub, or uploaded on the console.**
3. **Inputs and outputs declared on Add model** — images, features, labelled scores — so the app
   sends a picture and gets labels back, from the first version.
4. **The owner's machines and rented ones together**, from the first version.
5. **An upload reaches the machines either way, the operator's choice** (2026-11-19): sent by the
   pool to each machine's agent, or relayed through a private hub repository of the operator's.
6. **Private container registries in the first version.**
7. **Containers are allowed by one pool-wide switch**, not by each machine's owner. The guard
   proposed — the switch set only by editing the pool's file by hand — was declined: "I don't like
   hands only edit". The switch is changeable from the console like any setting; a machine's owner
   can still say *never* in that machine's agent settings.

## Design

### Two ways to bring a model

| | **An ONNX file** | **A container image** |
|---|---|---|
| For | A model every framework can export: PyTorch `torch.onnx.export`, TensorFlow through `tf2onnx`, scikit-learn through `skl2onnx`, XGBoost and LightGBM through `onnxmltools` | A model that needs its own code: custom preparation, an operator ONNX lacks, another runtime |
| Runs on | **ONNX Runtime**, run by the pool's own server: NVIDIA cards (CUDA), Macs (CoreML), Windows (DirectML), any processor | The operator's image, which must answer the predict contract below |
| What runs on the machine | Only the runtime's built-in operators: the file is a graph, never code | **The image's code** — see "Containers run the operator's code" |
| Inputs and outputs | Prepared and read by the pool, as declared | Passed to the image as sent; the image prepares its own |
| From | A model hub, or uploaded on the console | A container registry, pinned by digest (`repo@sha256:…`) |

### What the operator says

On the console, **Add model** (beside Add provider), or in the catalog by hand:

| Detail | What it is | Filled in for them |
|---|---|---|
| **Name** | What apps ask for | — |
| **The model** | A hub file at a pinned revision (`owner/repo/file.onnx`, as whisper.cpp's builds are named, D148), a file uploaded now, or an image by digest | Size, from the hub's listing, the upload or the registry |
| **Inputs** (ONNX) | Each input's name, element type and shape, the batch dimension marked | **Read from the file itself** and shown; the operator confirms |
| **How each input arrives** (ONNX) | `tensor` (numbers as sent), `image` (decoded and resized to the input's size; channel order, scale, mean and std) or `features` (named values, in the order the input expects) | Shape and size from the file |
| **What each output means** (ONNX) | `tensor`, `scores` (with labels; softmax or not; top *k*), `vector` (an embedding), `value` (a regression) | Output names and shapes from the file |
| **An example input** (container) | One request body the image should answer, for the Test | — |
| **What it needs** | Memory, whether it needs a card, the largest batch | Memory proposed from the size |
| **Where it may run** | The pool's own machines, rented ones, or both | — |

Then **Test**: the pool puts the model on one of its own machines, runs one input — made from the
declared shapes for ONNX, the example for an image — and shows what came back, before anything is
saved. Testing on a rented machine would rent one, so it is offered only with a lease that may.

### The app's call

- **A kind, `predict`** (amends D142), on one path: `POST /v1/predict`, with a body
  `{"model": "...", "inputs": {...}}` and an answer `{"model": "...", "outputs": {...}}`. Images
  inline (base64), never a link (D143). The router reads the model field and nothing else, as for
  any body; the request log records kind, latency and size, with no tokens.
- **SDK**: `client.predict(model, inputs)` returning the outputs, with the usual waiting and
  retries; images passed as bytes.
- **An image's contract**: it serves `POST /v1/predict` as above, `GET /v1/models` (the model it
  holds) and `GET /health`, on the port the operator declares. That is all the pool asks of it.

### ONNX: the engine `onnx`

Serving only `predict`. Its server is part of the agent's own package: the agent installs ONNX
Runtime from a hash-locked set as it does vLLM (D158), puts the file on disk, and runs the server
on loopback. Decoding, resizing and normalising an image happen there, never on the router's
request path. **The model's declarations reach the agent with the tag**, in the existing *set
models* verb — enumerations, numbers and label strings, bounded and checked, never a path, a URL
or a command — a change to that verb's body, recorded as an amendment of D40. Requests that arrive
together run as one batch, up to the declared largest.

### Containers: the engine `container`

**Containers run the operator's code.** Today the agent's guarantee (host-agent §4) is that the
pool's admin key cannot make a machine execute code. A container named in the catalog can, so the
guarantee changes (answer 7):

- **One pool-wide switch**, `containers.enabled`, off by default, in the pool's file, changeable
  like any setting, the console included: turning it on through the control API asks for the
  pool's name typed again and is recorded as an event. The guarantee becomes: *with the switch on,
  whoever holds the admin key can run an image of their choosing on every machine whose owner has
  not said never* — recorded in D180 as what it gives up.
- **A machine's owner can still refuse**: `run_containers: never` in the agent's own settings on
  that machine, and the console says so on its card. Unset, the agent follows the pool.
- **Run narrowly**: pulled by digest only; through the Docker Engine API on the machine's own
  socket (an API client, never a command line); its port bound to loopback; no host network, no
  privileged mode, no host directory but the model directory read-only; a memory limit; a card only
  where declared.
- **Private registries** (answer 6): a registry's login is typed once on the console and kept by
  the supervisor in the credential store (D134), never in the file, a log or an answer. A provider
  is given it through its API to pull the image for a rented machine; it never goes on the rented
  machine itself. On the owner's machines the pool sends no secret: the agent pulls with the
  machine's own login (D181).
- **On a rented machine**, the operator's image is the machine's image, as `rented.image` is today,
  and the pool's start adds its agent and dead-man timer to it (S4). Renting, leases and caps are
  unchanged.
- On a Mac, a container gets no access to the GPU (Docker Desktop has none), so it runs on the
  processor; an ONNX file on the same Mac runs on the GPU through CoreML.

### Uploads

An uploaded file is sent to the supervisor through the control API (admin key), bounded by a size
the configuration sets, and kept content-addressed (`upload/<sha256>`) in an owner-only directory
of the pool's. **The operator chooses, per model, how it reaches the machines** (answer 5):

- **Sent by the pool.** The pool sends it to each machine's agent — the pool dials the agent, never
  the reverse — through a new verb that takes the file's bytes and its hash and nothing else; the
  agent writes it under its models directory by its hash and keeps it only if the hash matches. A
  new verb is a decision (amends D40). Every copy leaves through the pool's own line.
- **Relayed through a hub.** The supervisor puts it into a private hub repository the operator
  names, with a write token typed once and kept in the credential store; machines fetch it as any
  build, checked against the hub's listing. A machine needs a read token for that repository: on
  the owner's machines their own (`HF_TOKEN`, as today); on a rented machine **a read-only token
  the operator gives for this, placed on the machine** — the first secret a rented machine carries,
  scoped to that repository, never the write token.

Images are not uploaded: they come from a registry.

### Rented machines

An ONNX model: a plain image carrying Python, the agent installing the runtime for the card it
finds and fetching or receiving the file. A container: the operator's image, as above. Uploads
travel from the pool to each rented machine through its tunnel.

## Decisions

D178 (the kind `predict`), D179 (the engine `onnx`), D180 (the engine `container` and its switch),
D181 (private registries), D182 (uploads and their two deliveries).

## What it gives up

- An ONNX model whose preparation is custom Python (a bespoke tokenizer, a non-standard image
  pipeline) is served only as a container. Text inputs to an ONNX model need a tokenizer and are
  left for a later stage.
- What a container does with its input cannot be checked by the pool; its answers are passed back
  as the image sent them.
- No hosted endpoint for trained models (SageMaker, Vertex, Hugging Face Inference Endpoints).
- One model per request; no pipelines of models.

## Unverified

| Assumption | How it would be checked |
|---|---|
| ONNX Runtime's wheels can be locked per Python, processor and accelerator as vLLM's are | Generate the sets at release, as `tools/vllm-lock` does |
| CoreML and DirectML run common exported models without falling back to the processor for most of the graph | A small classifier and a detector on a Mac and a Windows machine |
| Reading inputs and outputs from an ONNX file needs only the `onnx` package, on the supervisor | Read the shapes of a few exported models |
| The pool's start can add its agent to an operator's image on Vast and RunPod: the image needs `python3` and `sh` | Start one image without Python and one with, on the fake provider, then once live with the owner's word |
| Vast and RunPod pull a private image with registry credentials passed through their APIs | Their API documentation, then a live check |
| A hub's tokens can be scoped read-only to one repository | The hub's documentation (Hugging Face's fine-grained tokens) |

## Build stages

1. Kind `predict`; engines `onnx` and `container` against fakes; catalog and configuration; the
   router and the request log. Tests with no GPU, no Docker, no hub.
2. The agent: the ONNX server and its locked set; containers through the Docker Engine API behind
   `run_containers`; fetching from the hub; the check with a made-up or example input.
3. ONNX declarations: image, features, scores, vector; batching.
4. Uploads: the control API, the store, the agent's new verb, the relay through a hub.
5. Add model on the console, with Test.
6. Rented machines for both; then the owner's live checks.

## What it touches

App contract (a new path, optional); D142 (a kind); D40 (the *set models* body, and a verb for
uploads); host-agent §4 (containers, behind a pool-wide switch); the credential store
(registry logins, a hub's tokens); a secret on a rented machine for the first time (a read-only hub
token, relayed uploads only); the catalog; the agent (an ONNX server, a locked set, a Docker
client); renting (an operator's image as the machine's); the console; the SDK.
