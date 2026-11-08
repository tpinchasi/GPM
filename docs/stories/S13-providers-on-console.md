# S13 — Providers on the console: hosted model APIs added like a rented account

> Status: **built against fakes** (2026-11-08, D171; gpm-server 0.40.0). Part of the
> [feature list](README.md).

## The story

An operator adds a hosted model API — Anthropic, OpenAI, Groq, Together, Ollama's cloud,
OpenRouter, DeepSeek, Fireworks, DeepInfra, or any other API of either format — the way they add a
rented-machine account: on the console, by choosing it, typing its key once, testing it, and
choosing which of its models apps may ask for and what each costs. No file to edit, no environment
variable to set, no restart. And a pool that rents nothing yet can add its first rented account
from the same place.

## Why

The owner (2026-11-08), on a fresh pool with no rented section: "I can't open Rented capacity. …
how do I define hosted providers? I expect it to be like VastAI, etc." Two gaps:

- **Rented capacity was a dead end** on a pool that rents nothing: it said so and hid the very tab
  where a first provider would be added.
- **A hosted API could be added only by hand**: a `hosts` entry, catalog builds with prices, and a
  key in an environment variable that both processes had to be restarted to read.

## What was built

- **A Providers screen** (`#providers`) with two halves: **Hosted model APIs** and **Rented
  machines** (the existing accounts view). Rented capacity → Providers stays as it was.
- **Add hosted API**, a five-step dialog: *Provider* (a list of known APIs, or another of either
  format with its own address) → *Key* (a name, the address, the key, requests at once) → *Test*
  (the provider's `/models`, which spends nothing) → *Models* (tick each id the provider offers, the
  name apps ask for, its price per million tokens — the cache-write price required for Anthropic's
  format) → *Save* (the plan first, then the usual validate → plan → apply).
- **A card per hosted API**: status (serving, needs a key, key refused, out of money, cannot be
  reached, out of service), address, where its key comes from, spend today and this month, its
  models and prices; **Test**, **Type in / Replace key**, **Models and prices**, **Take out of
  service**, **Remove** (typed again; removes the models only it served).
- **The hosted budget** — the shared workload's `pool.bursts.hosted` — set from the same screen.
- **The key, typed once.** Kept by the supervisor in the credential store's `hosted/` directory,
  owner-only; read by the router, which sends the requests; used only while the host's `base_url`
  is the one it was saved for; never in the file, an answer, a log or the database; refused over
  plain HTTP from another machine. A new key reaches both processes without a restart.
- **Each model pinned to its API**: the build `requires: [hosted:<id>]`, the host has that
  capability — so a model id two providers share is sent and priced as each says.
- **A first rented account from a pool that rents nothing**: Add a provider account asks, besides
  the credential, for the two ceilings no machine is rented without — the disk it is rented with,
  the most it may cost an hour — the engine rented machines run, and which models they hold. It
  writes the `rented:` section; renting starts at the next start of the pool, as the plan says.

## Found on the way

- **A router serving a hosted API added while it ran raced the file** only in tests — but a real
  gap was nearby: no endpoint could add a host at all, and the file editor had no way to append to
  a list. `append_list_item` writes a new host as a block, key by key, so it can be edited later.

## What it gives up

- **The router now reads a secret from disk** — the hosted keys only, from the owner-only store
  the supervisor writes (T36). Unavoidable: the router sends the requests. The two processes still
  never call each other.
- **No price list is shipped.** Prices change without notice and are the operator's to answer for
  (D164); the console asks for them, per model, from the provider's own page.
- **A first rented account needs a restart** to start renting: the supervisor builds its renting
  machinery only at start. The console says so; nothing rents meanwhile.

## Not done

- One hosted API's own page (its request history and spend over time), as rented accounts lack too.
- Live checks against providers other than Anthropic (S12's) — the add flow uses only `/models`.
