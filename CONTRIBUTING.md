# Contributing to GPM

Thank you for looking. GPM sits in the request path of other people's applications and spends
their money unattended, so the bar for changes is deliberately high — and the tests are built
to let you clear it without a GPU or a cloud account.

## Getting set up

```sh
uv sync                  # one workspace, two packages
uv run pytest            # the whole default suite: no GPU, no cloud account, ~60s
```

Two optional suites need something extra, and neither runs by default:

```sh
uv run pytest -m integration    # needs a local Ollama holding the models the tests name
```

## Sign your work

We use the [Developer Certificate of Origin](https://developercertificate.org/). It is a
one-line statement that you wrote the patch, or otherwise have the right to submit it under this
project's licence. Add it with `git commit -s`, which appends:

```
Signed-off-by: Your Name <you@example.com>
```

Pull requests without a sign-off cannot be merged. There is no separate CLA.

## What a good change looks like

- **Tests that would fail without your change.** Assert the behaviour, not the implementation.
- **Nothing that can spend money in a test.** Everything goes through the fake provider; the
  whole suite must stay runnable offline.
- **Decisions get written down.** If you change *why* something works the way it does, add a
  numbered entry to [docs/decisions.md](docs/decisions.md) — what, why, and what you rejected —
  and update the relevant file in `docs/spec/`. If your change contradicts an existing decision,
  say so and supersede it explicitly rather than editing it silently.
- **Comments explain why, not what.** The code says what it does.

## Things that will be asked of any change touching money or credentials

- A lease is the only thing that can spend. Nothing may rent without one.
- Every cap and ceiling is re-checked by the supervisor *after* a strategy returns. Strategies
  are advisory; the caps are not.
- The account credential never reaches a rented host, a log, an event, an API response or a
  browser.
- The app key and the admin key are never interchangeable.

## Plug-ins

Providers, engines and strategies are the three places GPM is meant to be extended without
forking — see [docs/spec/plugin-interfaces.md](docs/spec/plugin-interfaces.md). A new provider
should pass the same scripted scenarios the fake provider does. Note that a plug-in runs with
the supervisor's full authority; we will not merge one that needs more.

## Reporting a security problem

Not through an issue — see [SECURITY.md](SECURITY.md).
