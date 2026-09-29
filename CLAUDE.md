# GPM — Agent Instructions

## What this is

GPM (GPU Hosts Pool Management) — a framework that serves GPU inference to applications without
them knowing where the GPU runs: a pool of local, fixed-remote and rented-on-a-bidding-market
hosts behind one endpoint, with leases as the only spending authority. **It is to be released
publicly.** The codebase it was designed in (Aletheia) is its first adopter, not its scope.

Stack when code starts: Python; FastAPI + httpx for the router and control API; SQLite (WAL) for
state; a static HTML/JS console with no build step. The client SDK is a separate package with
`httpx` as its only dependency.

## Session workflow

1. **Read [STATUS.md](STATUS.md) first** — current state, next actions, open owner decisions.
2. Then [docs/overview.md](docs/overview.md) and [docs/decisions.md](docs/decisions.md). The
   specification is in `docs/spec/`; scope and build order in [docs/roadmap.md](docs/roadmap.md).
3. **Before finishing any session, update [STATUS.md](STATUS.md)**: current state, next actions,
   open decisions, and one session-log row. Do not skip this.

## Non-negotiable principles

- **Public-framework bar.** Never justify a shortcut with "one developer on one machine".
  Isolation, safety and contracts must hold by construction, not by discipline.
- **Generic core.** Nothing in `docs/spec/`, the code, or any identifier is named after or
  argued from a specific adopter, provider account or model. Adopter material lives only under
  `docs/adopters/`. Argue design from what *any* client, provider and engine need.
- **Safe by default.** Nothing rents without a lease; a lease that can rent must have a dollar
  cap; the app key is always required and can never reach the control API; no unauthenticated
  listener off loopback; only catalogued models are ever resolved; no request can trigger a
  model pull (a load into memory is allowed only on a host whose `residency` is `on_demand`,
  for a tag already on its disk — D39); the provider account credential never goes on a rented
  host, in a log, or to a browser.
- **The app boundary.** Apps know one URL, one key and the contract in
  `docs/spec/app-contract.md`. The pool never calls into an app; apps never trigger recovery, and
  never trigger spending **except through a provisioning key** (D117): a separate key an operator
  grants one application, bounded by its grant and a pool-wide daily cap, that can create and end
  its own workloads and can never request a completion or reach the control API.
- **Router and supervisor stay separate processes** sharing SQLite, never calling each other.
  Nothing slow or blocking goes on the router's request path.
- **The host agent's protocol is a closed list of verbs.** No operation takes a command, a path
  or a URL from the pool; the pool dials the agent, never the reverse; it is never on the request
  path. Adding a verb is a decision, recorded as one (D40, `docs/spec/host-agent.md`).
- **Providers are HTTP API clients, never CLI wrappers.** Strategies are pure functions that
  return their reasons; the supervisor re-checks every cap and ceiling after they return.
- **Every test runs without a cloud account or a GPU** — against the fake provider and a fake
  engine. Anything live is opt-in, manual, and hard-capped.

## Recording decisions

A design decision is recorded in **two** places: a numbered entry in
[docs/decisions.md](docs/decisions.md) (what, why, what was rejected) and the relevant spec file
(how things are, without the history). If a change contradicts an existing decision, say so and
supersede it explicitly rather than editing it silently. Unverified facts the design rests on go
in the "Unverified assumptions" table until checked; verified ones move to "Verified facts" with
the source.

## Working with the owner

- **Ask before anything that spends money, calls a paid API, rents hardware, or starts a
  long-running job.** Validate on the smallest case first.
- **Keep changes to the requested scope**; ask before widening.
- Commit or push only when asked.
- Use plain, descriptive names; no internal jargon or staging tags.
- The owner usually prefers the simpler mechanism when one exists — offer it first, and state
  plainly what it gives up.
- Before the repository is made public, the adopter-internal files listed in STATUS.md must be
  dealt with. Do not make anything public, and do not push, without being asked.
