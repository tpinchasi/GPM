# S11 — Hosted model APIs, paid per token from each app's budget

> Status: **decided** (2026-11-06) — D164–D168, specified; the build next. Part of the
> [feature list](README.md).

## The story

An operator adds a hosted model API to the pool — OpenAI, Anthropic, Together AI, Ollama's cloud,
or any other that speaks the OpenAI-shaped `/v1` API — with an API key. Each application is given
a budget in dollars. When the pool's own machines have no free worker for a request, or no machine
serves the model at all (Claude), the request goes to a hosted API, and its cost comes out of that
app's budget. The app knows none of it: one URL, one key, the same calls.

## Why

- **The owner asked** (2026-11-06): serverless providers reached with an API key, beside the
  machines.
- **Models the pool cannot run.** Claude, GPT and the largest open models are only, or only
  cheaply, reached as hosted APIs.
- **A burst without a rental.** Renting a machine takes minutes and costs by the hour; a hosted
  API answers at once and costs by the token — the right shape for a short peak, or for a model
  used now and then.
- **The pool already has most of what it needs.** Hosts in priority order, catalog variants per
  engine, one OpenAI-shaped wire API (D89), kinds of request (D142), credentials typed in once and
  kept owner-only (D134–D136), spend recorded per lease.

## The owner's answers (2026-11-06)

1. **An app has a budget, and may spend it** with its own requests. This is a third exception to
   "apps never trigger spending" (D1), beside the provisioning key's grant (D117): the operator
   gives the app the budget; the app's requests spend within it, and never beyond.
2. **Hardware first.** Where a request could be served by one of the pool's machines — local,
   fixed-remote or rented — with a free worker, it is; a hosted API takes it only when none can.
3. **Data may leave to the providers the operator configures.** A request sent to a hosted API is
   read by that provider; the threat model says so, and the operator's adding the provider is the
   consent.
4. **Anthropic through its OpenAI-compatible API**, so the router keeps passing requests through
   untouched — no translation between APIs (plugin-interfaces §2).

5. **The app decides its budget's shape** — per day, per month, per request, or several — rather
   than the pool fixing one.
6. **Spill after a short wait**: a request waits a few seconds for a machine's worker before it
   goes to a hosted API.
7. **The app decides the mix of renting and hosted APIs**: an ad-hoc app states a low renting
   maximum and leans on hosted APIs; a steady one rents, and uses hosted APIs for bursts.
8. **Tell the app what a hosted answer cost**, with `X-GPM-Cost-USD`.

**The owner's own example, which the design follows** (2026-11-06): *"I expect that my app will
need 2 hosts with 24 workers each steady state, but for short term bursts I would like to use
hosted and long term bursts rented. I should be able to define budget spend policies, as I know my
app's usage patterns best."* The app, not the pool, knows its pattern; so the app states its
**spending policy** — its steady capacity, how it meets a short burst and a long one, and a budget
for each — and the pool carries it out.

## Design

### A hosted API is a host

A hosted API joins the pool as a configured host of a new kind, `hosted`:

```yaml
hosts:
  - id: together
    kind: hosted
    engine: openai-api              # the OpenAI-shaped /v1 API, spoken by every provider below
    base_url: https://api.together.ai/v1
    credential: together            # typed in once, kept owner-only (D134); or an env variable
    max_concurrent: 8               # its rate limit, as requests in flight
    models: [llama-3.3-70b, embed-small]

catalog:
  llama-3.3-70b:
    variants:
      - { tag: "llama3.3:70b", engine: ollama }                               # the pool's own machines
      - { tag: "meta-llama/Llama-3.3-70B-Instruct-Turbo", engine: openai-api, # hosted
          price: { input: 0.88, output: 0.88 } }                              # $ per million tokens
```

- **One engine, `openai-api`, for every provider** that serves the OpenAI-shaped paths: OpenAI,
  Anthropic's compatibility endpoint, Together, Ollama's cloud (`https://ollama.com/v1`), Groq,
  Fireworks, DeepInfra, OpenRouter. No adapter per provider; a provider is a base URL, a key and a
  price list.
- **The router passes the request through untouched**, as for any engine: it rewrites the model
  field to the provider's name for the model (D29's `with_model`), replaces the app's key with the
  provider's, and forwards. The answer comes back as the provider sent it.
- **Kinds per model, as the catalog says** (D142): chat everywhere; embeddings at OpenAI and
  Together; transcription at OpenAI and Together; rerank at Together. Anthropic serves chat only.
- **The supervisor** asks a hosted host's `/v1/models` (with its key; no tokens spent) for health
  and which models it has. Nothing is pulled, loaded, rented or released.

### The app's spending policy (answers 1, 2, 5–8, and the owner's example)

An app states, for its workload (D115), how it is to be served:

```python
pool.workloads.create(
    "orders", model="llama-3.3-70b", latency_s=4,
    steady={"hosts": 2, "workers": 24},                  # the base, on machines, always
    bursts={
        "hosted": {                                      # a short burst: tokens, at once
            "after_s": 5,                                # waited for a machine's worker first
            "budget": {"per_day": 20, "per_month": 300}, # dollars; any of per_request / day / month
        },
        "rented": {                                      # a long burst: machines
            "after_min": 10,                             # once the burst has lasted this long
            "max_hosts": 4,
            "budget": {"per_day": 60},
        },
    },
)
```

What the pool does with it, as demand rises and falls:

1. **Steady:** the base is served on machines — the operator's own, or rented under the workload's
   lease (D115) — as today.
2. **A burst begins:** a request that finds no free worker waits `after_s` for one; then it goes
   to a hosted API, within the hosted budget. Hardware first (answer 2), but no request waits long
   for it.
3. **The burst lasts:** once the pressure has held for `rented.after_min`, the pool rents, as S1's
   allocation does from measured pressure — up to `max_hosts`, within the rented budget. The hosted
   API covers the minutes a machine takes to come up; once it serves, the spill falls away.
4. **The burst ends:** the rented machines are given up after their idle cutoff, as today; the
   hosted API simply stops being used.
5. **A budget spent** stops that kind of burst for the rest of its period: the other kind, or the
   queue, takes over — never a request beyond a budget.

**Some models can only be hosted** — the owner (2026-11-06): *"some LLMs cannot be rented, only
hosted."* Claude, GPT and other closed models have no build a machine runs. A model whose catalog
lists hosted builds only is hosted-only: the pool never rents, prepares or plans a machine for it,
a workload serving it must carry a hosted budget, and `steady` or `rented` for it is refused with
that reason. A workload of several models may mix both kinds.

**Every part is optional, and any mix is allowed** — the owner's words: *"I can also say, no
rented, just hosted, so we need full flexibility for the app to describe its needs."*

| The app says | What happens |
|---|---|
| `steady` + `hosted` + `rented` (the owner's example) | base on machines; short bursts to tokens; long bursts rent |
| `steady` + `hosted` | base on machines; every burst, short or long, to tokens; nothing rented |
| `steady` + `rented` | base on machines; bursts wait for rented machines, as today; no tokens |
| `hosted` alone (`steady` of 0 hosts) | everything to hosted APIs, within the budget; no machine at all — and the way to a model no machine serves (Claude) |
| `steady` alone | today's behaviour: the base, then the queue |

An app may change its policy while its workload runs: a new budget or a burst added or removed
takes effect at once, within its grant. Among hosted hosts, priority decides, as among machines; a
hosted host's workers are its `max_concurrent`.

**Who sets it:** the app, through the SDK, with its provisioning key (D117) — the key an operator
grants an application to create and end its own workloads. The grant already bounds what the app
may rent; it gains a ceiling on hosted dollars (per day and per month), and the pool-wide daily cap
bounds every app together. The policy is the app's; the grant is only the outer edge, set as wide
as the operator likes. This keeps the framework's rule that **nothing spends without a dollar cap**
— every policy carries one, and none can exceed what was granted. An app without a grant never
rents or reaches a hosted API: safe by default. The same policy can be written in the pool's
configuration, for a workload the operator runs.

### Metering a hosted budget

- **Set by the app in its spending policy** (above): any of per request, per day and per month,
  within its grant.
- **Metered by the router from the answer itself:** the token counts every provider returns
  (`usage`), times the catalog's price for that build. Providers' own spend limits cannot be the
  cap: OpenAI's are "not instantaneous", Groq's lag 10–15 minutes, Together and Ollama have none
  per key (research below). The operator should still set one at the provider, as a backstop.
- **Reserved before, settled after.** A request is sent only if the app's budget covers its worst
  case — its prompt's size and its `max_tokens` (or a default ceiling) at the build's prices — so
  requests in flight together can never take an app past its budget. The reservation is replaced
  by the real cost when the answer ends.
- **A stream reports its usage only when asked**: the router adds
  `stream_options: {include_usage: true}` to a streamed request to a hosted API, and removes the
  extra usage chunk from the stream if the app did not ask for it. This is a second, named rewrite
  beside the model field, to be recorded as a decision. A stream cut before its usage arrives is
  charged its reservation.
- **Recorded per request** in the request log — app, provider, model, tokens in and out, dollars —
  and summed per app and provider on the console.
- **A budget spent** makes hosted hosts ineligible for that app until the next period: its requests
  wait for the machines, as they would without hosted APIs, or are answered `503` with the usual
  `Retry-After`.

### Prices

No provider here publishes prices through an API one can rely on (OpenAI, Anthropic and Ollama
publish web pages only; Together needs a key; OpenRouter and DeepInfra publish lists anyone can
read). So **prices are the operator's, in the catalog**, per hosted build: input, cached input and
output per million tokens, or per minute for transcription. A hosted variant without a price is
refused by the configuration. Ollama's cloud charges less off-peak; the pool charges its budget at
the peak price, which is never less.

### Answers that are not a rate limit

A provider out of money says so differently — OpenAI 429 `project_spend_limit_exceeded`, Anthropic
429 with no `retry-after` or 400, Groq 400 `blocked_api_access`, OpenRouter 402. Read as a rate
limit, such an answer would be retried for ever. The engine reads each into one of: **busy** (wait
`retry-after`, try another host), **out of funds** (the host is taken out until an operator acts,
said on the console), or **refused** (the request's own fault, returned to the app as the provider
said it).

### Credentials

The router now holds secrets: each hosted API's key, read from the credential store the supervisor
already keeps (D134) or from the environment, and sent only to that host's base URL, never logged,
never shown. The provider *rental* credentials stay the supervisor's alone. Threat model: a new
entry for the router's keys, and T7 extended — hosted providers read what is sent to them.

### Closing a hole found while planning

A local Ollama signed in to Ollama's account serves its cloud models through `:cloud` tags, billed
to that account, with no key the pool sees. A catalog entry naming such a tag would spend money no
budget or lease counts. **The agent starts the Ollama it runs with `OLLAMA_NO_CLOUD=1`, and the
configuration refuses a `:cloud` or `-cloud` tag on a machine** — Ollama's cloud is reached as a
hosted host, metered like any other.

### What the app sees

Nothing new it must handle. `X-GPM-Served-Model` says which build answered, as today; a new
`X-GPM-Cost-USD` says what a hosted answer cost (answer 8 — amending "apps never know cost", D1,
since the app now spends its own budget). `GET /pool/status` lists hosted hosts like any other,
and an app may read its own budget and what is left of it.

## Open questions for the owner

None of substance. Defaults to be confirmed when the decisions are written: `hosted.after_s` 5,
`rented.after_min` 10, and a default `max_tokens` ceiling for a request that names none, used only
to reserve its worst case.

## Build stages

1. Decisions and specification, once the questions are answered.
2. The `openai-api` engine, the `hosted` host kind, the catalog's prices, the credential, the
   router's metering, reservation and spill — against a fake hosted API that speaks each
   provider's usage and error shapes.
3. The spending policy: the workload's `steady` and `bursts`, the SDK call and the configuration
   that set it, the grant's hosted ceilings, the rent-after-a-lasting-burst rule beside S1's
   allocation, the cost header; spend per app, burst kind and provider on the console.
4. The `:cloud` hole closed on agent-run Ollama.

**Stage 2 built (gpm-server 0.38.0):** the `openai-api` engine and the `hosted` host kind; prices
on hosted builds, required; hosted-only models kept off machines and out of renting; the router's
spill after `after_s`, the worst case held before sending, usage asked of streams and its chunk
removed again, the cost settled from the answer's counts (the whole hold for a stream cut short),
`X-GPM-Cost-USD` and, on streams, `X-GPM-Cost-Max-USD`; busy, out-of-money and the request's own
refusals told apart; spend in the request log, read back at start; what is left in
`GET /pool/status`. Covered only for the shared workload, from `pool.bursts.hosted` — a
workload's own policy is stage 3. Not yet run against a real provider (stage 5).

**Stage 3, first part built (same PR):** each workload's hosted burst — set at creation or while
it runs, by its operator (API, `gpm workload create|policy --hosted-…`) or by its program through
the SDK (`bursts=`, `set_bursts()`, `hosted_budget()`), within the grant's `hosted_per_day` /
`hosted_per_month` and the pool's `provisioning.max_hosted_per_day`, counted as committed (D125's
rule); the router serves a workload's burst from the pool's hosted hosts on the workload's own
budget, never as borrowing; replies carry `cost_usd` in the SDK.

**Stage 3, second part built (same PR):** `bursts.rented` — once the whole base serves and the
load has stayed above it for `after_min`, up to `max_hosts` hosts beyond it per group, within its
day/month budget, paid from the workload's lease (a derived budget carries it); burst hosts never
count as the base, go first when the load falls, and are drained when the budget is reached or the
burst taken away; `gpm workload create|policy --rent-…`. `steady` is the workload's `parallel` at
its `latency_s` (D165 amended). **Still to build in stage 3:** a workload with no machine at all
(hosted alone, and hosted-only models in a workload), and spend on the console.
5. **Live, asked first:** one small chat request to each provider the owner holds a key for, a few
   cents each, with a budget of $1.

## Research (2026-11-06)

Read from each provider's own documentation; the full notes, with a link per fact, are in the
session's research report. The facts this design rests on:

| Fact | Source |
|---|---|
| Anthropic's OpenAI-compatible endpoint (`https://api.anthropic.com/v1/`) serves streaming, tools and images, but is "primarily intended to test and compare model capabilities, and is not considered a long-term or production-ready solution for most use cases"; it reports no cached tokens; `response_format` takes only `json_schema` with `strict: true`; system messages are joined into one system prompt | platform.claude.com/docs/en/api/openai-sdk |
| OpenAI streams carry usage only with `stream_options.include_usage`, in an extra final chunk with empty `choices`, which an interrupted stream may never send | developers.openai.com — chat completions reference |
| Providers' spend limits lag (OpenAI "not instantaneous"; Groq 10–15 minutes) or do not exist per key (Together; Ollama below Enterprise) | developers.openai.com — spend limits; console.groq.com; docs.together.ai |
| Ollama's cloud bills per token, with off-peak rates, and is reached at `https://ollama.com/v1` with a key — or through a signed-in local Ollama with `:cloud` tags; `OLLAMA_NO_CLOUD=1` turns that off | ollama.com/pricing; docs.ollama.com |
| Running out of money: OpenAI 429 `project_spend_limit_exceeded`; Anthropic 429 without `retry-after`, or 400; Groq 400; OpenRouter 402 | each provider's limits page |

## Unverified

- Whether Together's streams carry usage without `stream_options`.
- What Ollama's cloud answers when its credit runs out, and whether its `/v1/embeddings` and
  transcription serve any cloud model.
- That each provider's `/v1/models` answers with a key and spends nothing.

## What it gives up

- **Anthropic through a layer Anthropic calls not production-ready**, with no cached-token
  pricing and stricter structured output (answer 4). Its native API would need a path of its own.
- **Prices kept by hand**: a provider's price change is wrong in the budget until the operator
  updates the catalog.
- **Data leaves the operator's machines** to each configured provider (answer 3).
- **The router holds API keys**, which until now it never did.

## What it touches

The router (spill, metering, reservation, the usage rewrite, the key), a new engine and host kind,
the catalog (prices), configuration and its validation, the credential store, the request log, the
console (budgets, spend), the control API, the agent (`OLLAMA_NO_CLOUD`), the app contract (hosted
answers; perhaps a cost header), the threat model, and decisions amending D1.
