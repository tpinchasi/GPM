# Quick start — a pool over hosts you already have

This gets you a working pool over **one local engine and one remote machine you own**, with
nothing rented and nothing able to spend money. That is deliberate: the framework should be
useful before it is trusted with a credit card. Renting comes next, in
[quickstart-renting.md](quickstart-renting.md).

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and an
[Ollama](https://ollama.com) running somewhere you can reach.

## 1. Install

```sh
git clone <this repository> gpm && cd gpm
uv sync
```

Two packages get installed: `gpm-server` (the router, the supervisor and the `gpm` command) and
`gpm-client` (the SDK your application will depend on, whose only dependency is `httpx`).

## 2. Load the models on every host

A pool declares the set of models it serves, and **every host keeps all of them loaded, all the
time**. Nothing is loaded on demand, so nothing is ever swapped out mid-run. The pool *verifies*
this; it never configures someone else's engine, and no request can trigger a download.

On each machine that will serve:

```sh
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
# keep them resident rather than letting the engine unload them after a few minutes
curl http://127.0.0.1:11434/api/generate -d '{"model":"qwen2.5:7b-instruct","keep_alive":-1}'
curl http://127.0.0.1:11434/api/embed    -d '{"model":"nomic-embed-text","input":["warm"],"keep_alive":-1}'
```

A host that does not hold the whole set stays out of routing, and the console says which model
is missing. That is the intended behaviour, not a failure.

## 3. Write a configuration

Save this as `pool.yaml`:

```yaml
pool:
  name: default
  model_set:
    - qwen2.5:7b-instruct
    - nomic-embed-text

listen:  { host: 127.0.0.1, port: 8080 }   # the app-facing router
control: { host: 127.0.0.1, port: 8081 }   # the control API and the console

auth:
  app_keys_file:   ~/.config/gpm/default.app-keys
  admin_keys_file: ~/.config/gpm/default.admin-keys

engine: ollama

hosts:
  - id: laptop
    kind: local
    workers: 3
    transport: { type: http, base_url: "http://127.0.0.1:11434" }

  # A machine you own. `tunnel` keeps its engine off the network entirely: the pool holds an
  # SSH forward and dials its own loopback port.
  - id: workstation
    kind: fixed-remote
    workers: 4
    capabilities: [cuda]
    transport:
      type: tunnel
      ssh_host: workstation.local
      ssh_user: you
      ssh_key: ~/.ssh/id_ed25519
      remote_port: 11434
```

There is no `rented:` section, so **this pool has no way to spend money at all.**

## 4. Make two keys

```sh
uv run gpm key create --role app -c pool.yaml      # for your applications
uv run gpm key create --role admin -c pool.yaml    # for the console and the control API
```

Each key is printed once and stored only as a hash. The two roles are never interchangeable: an
application that can request a completion cannot open a lease or release a host. The app key is
required even on loopback, because any web page open in your browser can post to `127.0.0.1`.

## 5. Start it

```sh
uv run gpm serve -c pool.yaml
```

That runs the router in this process and the supervisor beside it in its own. They share a
SQLite file and never call each other, so if the supervisor stops, routing carries on.

```sh
export GPM_ADMIN_KEY=<the admin key>
uv run gpm status
```

Each host should reach `ready`. If one says `preparing`, it is reachable but does not hold the
whole model set yet — `uv run gpm host-test http://host:11434 --workers 4` will tell you exactly
which step fails, including whether the engine is really serving requests in parallel.

## 6. Open the console

**http://127.0.0.1:8081/ui** — paste the admin key when it asks. It is held in the tab's memory
and sent as a header; nothing is written to storage or a cookie.

Everything the console does is also a CLI verb, so anything you can click you can script.

## 7. Point an application at it

```python
from gpm_client import PoolClient

pool = PoolClient()            # GPM_URL and GPM_API_KEY from the environment
reply = pool.chat("qwen2.5:7b-instruct", [{"role": "user", "content": "hello"}])
print(reply.content, "served by", reply.host, "as", reply.served_model)
```

```sh
export GPM_URL=http://127.0.0.1:8080
export GPM_API_KEY=<the app key>
```

If your application already uses an HTTP client, inject the transport instead and change no
call sites at all — it will then wait and retry for capacity by default:

```python
from gpm_client import pool_transport
client = SomeEngineClient(base_url=GPM_URL, http_client_kwargs={"transport": pool_transport()})
```

Requests and responses are your engine's own API, passed through untouched. The pool adds only
a few headers: which build actually served the request, which host, and how long it waited for
a worker.

## What you have now

- One endpoint and one key in front of every GPU you own, with strict priority: local first,
  then your remote machines.
- Applications that pause and resume instead of failing when everything is busy.
- A record of every request — which host, which build, how long it queued — holding no prompt
  or completion text.
- Nothing that can spend money.

## Next

- [quickstart-renting.md](quickstart-renting.md) — adding a marketplace provider, written
  around leases and caps first.
- [spec/app-contract.md](spec/app-contract.md) — everything an application can rely on.
- [threat-model.md](threat-model.md) — in particular, that the operator of a rented marketplace
  host can read everything sent to it.
