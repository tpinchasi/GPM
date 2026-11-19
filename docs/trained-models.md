# Trained models

A pool can serve models you trained yourself that are not language or speech models: an image
classifier, an object detector, a forecaster, a model over a table of features. You say what the
model takes in and what it gives back. The pool does the rest: puts it on the machines that should
hold it, installs what runs it, starts it, checks it answers, routes requests to it, and waits
when every machine is busy. Apps still know one URL and one key.

This guide is for the pool's operator. What an app sends is under
[What an app sends](#what-an-app-sends) below, and in [client/README.md](../client/README.md).

## Two ways to bring a model

| | An ONNX file | A container image |
|---|---|---|
| For | A model your framework can export to ONNX — PyTorch (`torch.onnx.export`), TensorFlow and Keras (`tf2onnx`), scikit-learn (`skl2onnx`), XGBoost and LightGBM (`onnxmltools`) | A model that needs its own code: custom preparation, an operation ONNX lacks, another runtime |
| Runs on | The pool's own server, on ONNX Runtime: an NVIDIA card (CUDA), a Mac's GPU (CoreML), or the processor | Your image |
| What runs on the machine | Only ONNX Runtime's own operators. The file is a graph, not a program | **Your image's code** |
| Inputs and outputs | Prepared and read by the pool, as you declare them | Passed to your image as the app sent them |
| From | Uploaded on the console, or a file on the model hub | A container registry, by digest |

## 1. A host to run it

A trained model runs on a host whose engine is `onnx` or `container`, with an agent the pool
reaches. One machine can carry both kinds of host beside its others, each with its own agent.

```yaml
hosts:
  - id: box-onnx
    kind: local
    engine: onnx
    transport: { type: http, base_url: "http://127.0.0.1:8190" }
    agent: { url: "http://127.0.0.1:8096", bearer_env: BOX_AGENT_KEY }
```

On the machine, the agent's own settings say how it runs the model:

- **ONNX:** `install_engines: true` lets the agent install ONNX Runtime from a locked set (Linux
  on x86_64 or aarch64, CUDA's set where an NVIDIA card's driver allows it; Apple-silicon Macs).
  On Windows, or to use a Python of your own instead, set `engine_binary` to a Python with
  `onnxruntime`, `numpy` and `pillow` installed.
- **Containers:** Docker on the machine. The agent reaches it through Docker's own socket
  (`docker_socket` if it is not in the usual place) and pulls a private image with the machine's
  own Docker login. `run_containers: never` makes the machine refuse every container.

A host on these engines may hold no model yet: it waits for the first one you add.

## 2. Add the model from the console

*Models → Add a trained model*:

1. **Comes as:** `onnx` or `container`.
2. **ONNX:** choose the file, or name a hub file (`owner/repo/model.onnx`), then **Read the file**.
   The pool reads the file's inputs and outputs from the file itself, without running it.
3. **Say how each input arrives:**
   - `image`: the app sends a picture (base64). The pool decodes it, resizes it to the size you
     give, orders its channels (`rgb`, `bgr` or `gray`) and layout (`nchw` or `nhwc`), divides by
     the scale (usually 255), then subtracts the mean and divides by the std, per channel.
   - `features`: the app sends named values. The pool puts them in the order you list.
   - `tensor`: the app sends the numbers, one example, without the batch dimension.
4. **Say what each output means:**
   - `scores`: labels, one per score, in the model's order; softmax applied or not; the best *k*.
   - `vector`: a list of numbers, such as an embedding.
   - `value`: one number, such as a regression.
   - `tensor`: the numbers as computed.
5. **Name it** — what apps ask for — and choose the hosts that hold it, its memory, whether it
   needs a card, and how many requests may run together.
6. **Test**, then **Add**. Test holds what you declared to the file: input names, channels,
   picture size, feature count, label count. It spends nothing. Each machine also runs one
   made-up input through the model as it loads it, and serves it only once that has answered.

The same can be written in the pool's file:

```yaml
catalog:
  pets:
    variants:
      - tag: "upload/<sha256>"            # or owner/repo/model.onnx on the hub
        engine: onnx
        max_batch: 16
        needs: { memory_gb: 1 }
        spec:
          inputs:
            image: { as: image, input: pixel_values, size: [224, 224], channels: rgb, layout: nchw,
                     scale: 255, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }
          outputs:
            label: { as: scores, output: logits, softmax: true, top_k: 3, labels: [cat, dog, bird] }
```

## 3. Uploads

A file uploaded on the console is kept by the pool under its own sha256, in an owner-only
directory beside its database, up to `models.max_upload_mb` (2048 by default). It reaches the
machines in one of two ways, your choice per model:

- **Sent by the pool:** the pool sends the file to each machine that holds the model, through the
  machine's agent, which keeps it only if its bytes match its hash. Every copy leaves through the
  pool's own connection.
- **Relayed through the hub:** the pool commits the file to a private repository of yours on the
  model hub (`POST /pool/models/uploads/<sha256>/relay`), with a write token you give it once
  (`PUT /pool/models/hub-credential`). From then on the model is an ordinary hub file. Your own
  machines read it with their own `HF_TOKEN`. A rented machine would be given a read-only token
  you provide; that is the one secret a rented machine would carry.

## 4. Containers

Container builds run only while **containers are on** (`containers: {enabled: true}`). They are
off by default.

> **What turning them on gives up.** With containers on, anyone who holds the pool's admin key can
> add an image of their choosing, and it runs on every machine the pool manages whose owner has
> not set `run_containers: never`. With containers off, the admin key cannot make any machine run
> code.

Turning them on from the console or the control API asks for the pool's name to be typed again,
and is recorded as an event. Your image:

- is named by digest (`registry.example/me/model@sha256:…`); a tag alone is refused, since it can move;
- serves `POST /v1/predict`, `GET /v1/models` and `GET /health` on the port you declare;
- runs one per host, with its port published on the machine's loopback only, not privileged, with
  no new privileges, no added capability and no directory of the host, its memory limited to what
  you declare, and a card only where you say it needs one.

## What an app sends

```
POST /v1/predict
{"model": "pets", "inputs": {"image": "<base64 of a JPEG or PNG>"}}

→ {"model": "...", "outputs": {"label": [{"label": "cat", "score": 0.93}, {"label": "dog", "score": 0.05}, ...]}}
```

With the SDK: `client.predict("pets", {"image": open("cat.jpg", "rb").read()})["label"]`. Bytes are
sent as base64. A picture is never a link.

## 5. Rented machines

A pool can rent machines for ONNX models: `rented.engine: onnx`, with the models in
`rented.models` or a profile. A rented machine starts from a plain Python image
(`python:3.12.16-slim-bookworm` unless `rented.image` names another with Python 3.11 or later), its
agent installs ONNX Runtime there from the locked set for the card it finds, takes each build from
the hub or from the pool, and the machine is ready once its server has answered a made-up input
for every build. Leases, caps and the dead-man timer are as for any rented machine. A pool rents
one engine, so a pool that rents ONNX machines does not also rent vLLM or Ollama ones.

## Not yet

- **Renting machines for container builds.** A provider starts a rented machine with its own
  command, not the image's, and the providers' APIs as the pool uses them take no registry login:
  container builds are held by your own machines.
- **The hub's read-only token on a rented machine**, for a relayed upload: rent for a model sent
  by the pool, or one in a public repository.
- **Text inputs** to an ONNX model, which need a tokenizer: use a container.
- **Hosted endpoints** for trained models (SageMaker, Vertex, Hugging Face Inference Endpoints).
