# Hosted model APIs

A pool can send requests to hosted model APIs as well as to its own machines: Anthropic, OpenAI,
Groq, Together AI, Ollama's cloud, OpenRouter, DeepSeek, Fireworks, DeepInfra, or any other API
in OpenAI's or Anthropic's format. Apps still know one URL and one key. They ask for a model by
name, and the pool decides where the answer comes from.

This guide is for the pool's operator. What an app sees is in
[client/README.md](../client/README.md#hosted-model-apis-answers-paid-per-token).

## How the pool uses them

- **Its own machines first.** A request goes to a hosted API only after it has waited a set time
  for a machine (`after_s`), or at once for a model no machine runs, such as Claude.
- **Paid per token, from a budget.** Every request a hosted API takes is paid from a budget: the
  pool's shared one for requests made with an app key, or a workload's own. A request goes only
  if its worst-case cost fits what is left. Nothing is ever sent past a budget.
- **Without a budget, nothing goes to a hosted API.** Adding one does not by itself let any
  request reach it.
- **Each answer says what it cost** (`X-GPM-Cost-USD`). The console's **Spending** screen and
  `gpm spending` show the spend per workload and per provider, today and this month.

What an app sends reaches that provider. If some data must stay on your own machines, keep the
models it is sent to off hosted APIs.

## 1. Add one from the console

*Providers → Add provider → a hosted model API*:

1. **Choose the provider** from the list, or *Another OpenAI-format API* / *Another
   Anthropic-format API* with its address.
2. **Paste its API key.** The key is tested by listing the provider's models, which costs
   nothing. It is kept by the supervisor in an owner-only file, sent only to the address it was
   saved for, and never shown again. Replacing or removing it later takes effect at once.
3. **Choose the models**, each under the name apps will ask for. A model the pool already has
   (say `gemma4:31b`, on your machines) can be given the same name: the hosted API then becomes
   its overflow.
4. **Check the prices.** They are filled in from the provider, from the first source that has
   them. Each one shows which it came from:
   - the provider's own model list, where it gives prices (OpenRouter, Together AI);
   - the provider's pricing page, where the pool can read it (Ollama's cloud);
   - a public price list, LiteLLM's community-kept one by default (`pool.price_list_url`; set it
     empty to fetch none).

   A price none of these has is typed in. Nothing is saved until you save it, so check what was
   filled in. A provider you pay by subscription rather than per token is ticked **paid by
   plan**: its answers count against no budget, and the plan's own limit is the cap.
5. **Save.** It goes through the same check-and-plan as any configuration change, and serves at
   once.

Each hosted API's card has **Settings** (requests at once, prompt caching), **Models and
prices**, its key, **Take out of service** and **Remove**.

## 2. Give it a budget

For requests made with an app key, set the shared budget on the Providers screen, or in
`pool.yaml`:

```yaml
pool:
  bursts:
    hosted:
      after_s: 5               # how long a request waits for a machine first
      budget:
        per_request: 0.10      # optional: the most any one answer may cost
        per_day: 2.00          # per UTC day — per_day and/or per_month is required
        per_month: 20.00
```

A **workload** has its own budget, set when it is created and changed while it runs:

```sh
gpm workload policy <name> --hosted-per-day 5 --hosted-per-month 50 --hosted-after 5
gpm workload policy <name> --no-hosted
```

A program creating its own workloads sets them through the SDK, within its provisioning key's
grant: `gpm provisioner create … --hosted-per-day 2 --hosted-per-month 20`.

When a budget is spent, requests that find no free machine wait for one, and are answered
`503 hosted_budget_spent` if none frees in time. The budget's next day or month starts it again.

## 3. Or write it in `pool.yaml`

What the console writes is an ordinary host and catalog entries:

```yaml
hosts:
  - id: anthropic
    kind: hosted
    engine: anthropic-api          # openai-api for every OpenAI-format provider
    workers: 8                     # the provider's rate limit, as requests in flight
    capabilities: ["hosted:anthropic"]
    transport:
      type: https
      base_url: https://api.anthropic.com/v1
      # bearer_env: ANTHROPIC_API_KEY   # optional: read the key from this variable instead

catalog:
  claude:
    variants:
      - tag: <provider's model id>       # the provider's own name for the model
        engine: anthropic-api
        requires: ["hosted:anthropic"]   # this build is served only by that host
        # USD per 1M tokens, from the provider's price page. These figures are only an example.
        price: { input: 3.00, cached_input: 0.30, cache_write: 3.75, output: 15.00 }
```

The model's name goes in `pool.model_set` too; no machine is ever asked to hold a model only a
hosted API serves. A hosted build needs a price, or `price: { plan: true }`. Prices are per million tokens; a
transcription model's is per minute.

## When a provider refuses

- **Busy** (429, 503, 529): the pool waits as long as the provider says, and the request takes
  another worker or waits for this one.
- **Out of money**: the pool stops routing to the host and the console shows what the provider
  said. Top up the account, then tell the pool with **Money is back** on the host's card, or
  `gpm host funds-restored <id>`.
- **Anything else**, such as a request too long for the model, goes back to the app as the
  provider said it, and costs nothing.

## Worth knowing

- **Claude is reached through Anthropic's own API**, with its 5-minute prompt cache asked for on
  every request that does not mark its own. Turn that off with *Settings → prompt caching*, or
  `cache_prompts: false` on the host. Cache reads and writes are metered at their own prices.
- **Apps can use either API format with any model** (OpenAI's, or Anthropic's `/v1/messages`).
  The pool translates where the model is served in the other format.
- **Ollama's cloud is reached only as a hosted API**, priced and budgeted like any other. An
  Ollama on your machines never serves a `:cloud` model, so nothing reaches Ollama's cloud
  without a budget counting it.
