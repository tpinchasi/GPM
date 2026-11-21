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

- **ONNX:** `install_engines: true` lets the agent install ONNX Runtime from a locked set: Linux
  on x86_64 or aarch64 and Windows on x86_64, CUDA's set where an NVIDIA card's driver allows it
  (Windows: driver 580 or later), else the processor's; and Apple-silicon Macs. To use a Python of
  your own instead, set `engine_binary` to a Python with `onnxruntime`, `numpy` and `pillow`
  installed.
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
  machines read it with their own `HF_TOKEN`. A rented machine is given the **read-only** token you
  save beside the write token (`read_token`; one that can write is refused): copied over the SSH
  connection the pool installs the agent on, into an owner-only file the agent reads as
  `HF_TOKEN` (with the hub it was checked against as `HF_ENDPOINT`), only on a machine that holds a
  hub file, and gone with the machine. It is never in the provider's instance settings, a log or an
  event, but the owner of the rented machine could read it while it runs: make it a fine-grained
  token limited to your model repositories. A fine-grained token that may write is refused.

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

**ONNX.** A pool can rent machines for ONNX models: `rented.engine: onnx`, with the models in
`rented.models` or a profile. A rented machine starts from a plain Python image
(`python:3.12.16-slim-bookworm` unless `rented.image` names another with Python 3.11 or later), its
agent installs ONNX Runtime there from the locked set for the card it finds, takes each build from
the hub or from the pool, and the machine is ready once its server has answered a made-up input
for every build. Leases, caps and the dead-man timer are as for any rented machine. A pool rents
one engine, so a pool that rents ONNX machines does not also rent vLLM or Ollama ones.

**Containers.** With containers on, a pool can rent machines for one container build:
`rented.engine: container`, its model alone in `rented.models`. The rented machine is the image
itself. Because Vast and RunPod start a rented machine with their own command rather than the
image's, the build says how its server starts, as an argument list, one word each:

```yaml
catalog:
  boxes:
    variants:
      - tag: "ghcr.io/me/detector@sha256:…"
        engine: container
        port: 8080
        start: ["python", "-m", "detector", "--port", "8080"]
```

The pool runs `start` in the background from the machine's start-up script after the dead-man timer
is armed, as it starts any rented engine, and the machine is ready once `GET /health` answers on
`port`. The provider stores that script: put nothing secret in `start`. No agent is put on it. The
image needs `/bin/sh`; on RunPod, which starts no SSH server of its own, the pool installs one
with the image's package manager (`apt-get` or `apk`). Your own machines ignore `start` and run the
image's own entrypoint.

**A private image** is pulled with a login you save for its registry (Models → Add a trained
model → Container → Private registries, or `PUT /pool/containers/registry-logins` with `registry`,
`username` and `password`; `GET` lists them without passwords, `DELETE …/<registry>` removes one).
The pool keeps it in its credential store and hands it only to the provider's own create call:
Vast's `image_login`, or a registry credential RunPod keeps for it. It never goes on the machine's
disk by the pool's hand, in `pool.yaml`, a log or an event. A provider whose plug-in takes no login
is not rented from for an image that needs one.

> **Use a pull-only token.** On a marketplace the rented machine's own Docker daemon pulls the image,
> so whoever runs that machine can capture the login it was given. Give a token that can only pull
> the images you rent (a registry's read-only, repository-scoped token), never your account
> password. Removing a login removes the pool's copy; a credential RunPod saved for it stays in your
> RunPod account until you delete it there.

## Not yet

- **Text inputs** to an ONNX model, which need a tokenizer: use a container.
- **Hosted endpoints** for trained models (SageMaker, Vertex, Hugging Face Inference Endpoints).
