# Release Checklist

> What has to be true before the first public commit and before the first public release.
> Items marked **owner** need a decision only the project owner can make; a recommendation is
> given for each, with the reasoning, so the decision is quick.

## 1. Before the first public commit

### 1.1 A clean repository — not a split of the current one

- [x] Create a **new repository with no shared history** — done 2026-09-17: `GPM`, private.
      Reasoning kept for the record: The repository this was designed in
      carries internal product material, exported data, and local credentials in prototype
      scripts and environment backups. `git filter-repo` style extraction is not safe enough:
      one missed path publishes it permanently.
- [ ] All of `docs/` was moved in on 2026-09-17, **including** `adopters/`, `archive/` and the
      review, because the repository is private. Decide what happens to those three before
      going public — see `STATUS.md`, "Before anything here becomes public". Nothing is ever
      moved from the adopter's application or prototype folders.
- [x] Run a secret scanner over the new repository before the first push, and add it to CI —
      done 2026-09-19: scanned before the first commit (the only matches are test fixtures: an
      example public key and a literal `super-secret-host-token`), and gitleaks runs in CI.
- [ ] The adopter guide stays in the adopter's repository and links to the public docs, not the
      reverse. The generic docs must read completely without it.

### 1.2 Name — **owner** — decided 2026-09-19 (D36)

The project and repository are **GPM** (decided 2026-09-17); the identifier set derived from it
was decided on 2026-09-19 and applied throughout: CLI `gpm`, packages `gpm-client` and
`gpm-server`, environment prefix `GPM_*`, header prefix `X-GPM-*`, configuration under
`~/.config/gpm/`, and rented instances labelled `gpm/<pool>/<host>`.

Criteria: not tied to the first adopter or to one provider or engine; available as a package
name and a repository name; short enough to type as a CLI; does not collide with an existing
well-known tool (`pool` alone is too generic for a CLI on a shared machine).

- [x] Choose the name — GPM.
- [x] Choose the identifier set and check the package name is free — `gpm` itself is **taken on
      PyPI** by an abandoned 2019 package, so the distributions are `gpm-client` and
      `gpm-server`, both free. `gpm` is also the Linux console mouse daemon (in `/usr/sbin`);
      accepted, since a pip install shadows it for a normal user.
- [x] Apply it consistently: package, CLI, environment-variable prefix, HTTP header prefix,
      label prefix, directories — done, with zero occurrences of the old identifiers left.
- [x] Nothing anywhere is named after the first adopter — outside `docs/adopters/`, which is
      still to be dealt with before going public.

### 1.3 Licence — **owner**

**Recommendation: Apache-2.0.** It is permissive, so it does not deter commercial adopters; it
carries an explicit patent grant, which matters for infrastructure software that other
companies build on; and it is the convention among comparable projects, so nobody needs a legal
review to try it. MIT is the simpler alternative if the patent grant is not a concern. A
copyleft licence (GPL or AGPL family) would protect against closed forks but will keep most
companies from adopting a component that sits in their serving path.

- [x] Choose the licence; add `LICENSE` and a `NOTICE` — **Apache-2.0** (D37), both files added.
- [x] Decide the copyright holder — **ClearViews** (D37), in `NOTICE`.
- [x] Decide whether contributions need a sign-off — **DCO**, stated in `CONTRIBUTING.md`.
- [x] Check every dependency's licence is compatible — all permissive: BSD-3-Clause, MIT,
      MPL-2.0, PSF-2.0.

## 2. Before the first public release (phase 4)

### 2.1 Packaging

- [x] **Two packages**, because apps must be able to depend on the client without installing
      the server: `gpm-client` (one runtime dependency, `httpx`) and `gpm-server` (router,
      supervisor, CLI, console, first-party plug-ins).
- [x] First-party provider and engine plug-ins load through the same entry-point mechanism as
      third-party ones — no privileged path. Groups: `gpm.providers`, `gpm.engines`.
- [x] Semantic versioning for both packages. The app contract (`X-GPM-Contract` and
      `GET /pool/status`), the control API and each plug-in interface (`interface_version`)
      carry their own version numbers, reported at runtime. Stated in `CHANGELOG.md`.
- [x] Pinned, reproducible builds (`uv.lock`, committed); `CHANGELOG.md` published.

### 2.2 Safe defaults, verified against the implementation

Each line was re-checked against the code on 2026-09-19; the test that holds it in place is in
[threat-model.md](threat-model.md) §6.

- [x] Fresh install binds loopback only; refuses a non-loopback listener without TLS — for the
      router and the control API alike (T4).
- [x] No request is served without the app key; no control call without the admin key; the two
      are never interchangeable, and the control API says so by name when given the app key
      (T2, T3).
- [x] Nothing rents without a lease; a lease that can rent cannot be created without a dollar
      cap (T11).
- [x] Plain-`http` public hosts without auth are refused unless explicitly allowed per host —
      and switching that on is a change the plan makes you retype (T17).
- [x] Only catalogued models are ever resolved; the request path cannot even reach `pull` or
      `load_and_pin`, which is enforced structurally (T9, T10).
- [x] Key files and the database are created owner-readable only, and a key file others can
      read is refused (T19).
- [x] The account credential appears in no log, no event, no API response and on no rented
      host; only the provider's instance-scoped credential goes there (T5, T6).

### 2.3 Tests

- [x] The whole suite runs with **no cloud account and no GPU** — 306 tests, ~60s.
- [x] Passthrough-fidelity tests: streaming, tool calls, structured output, cancel on
      disconnect, byte-compared against the engine directly.
- [x] The interruption, idle-release, orphan-sweep, supervisor-death and cap-overrun drills are
      automated in `tests/test_interruption_drill.py`.
- [x] An opt-in live suite: `pytest -m integration` against a real Ollama, and the capped
      Vast.ai run of 2026-09-19 ($0.008).

### 2.4 Documentation

- [x] A quick start that stands up a pool over one local and one remote host **without renting
      anything** — [quickstart.md](quickstart.md).
- [x] A second guide that adds a rented provider, written around leases and caps first —
      [quickstart-renting.md](quickstart-renting.md).
- [x] Plug-in author guides for provider, engine and strategy —
      [writing-a-plugin.md](writing-a-plugin.md), with the fake provider as the worked example.
- [x] The **deliberate non-goals** stated in user-facing docs — `SECURITY.md` and the renting
      guide, enforced by `test_threat_model.py::test_the_deliberate_non_goals_are_in_user_facing_documentation`.
- [x] [decisions.md](decisions.md) published: it answers most "why doesn't it just…" questions.

### 2.5 Project hygiene

- [x] `SECURITY.md` with a private reporting channel and a statement of what counts as critical.
- [x] `CONTRIBUTING.md` with the DCO, `CODE_OF_CONDUCT.md`, and issue and pull-request templates.
- [x] CI: tests on three Python versions, **ruff** (the rules that catch problems, not house
      style), a secret scan, a dependency audit, and a job that fails if anything able to reach
      a real provider enters the default suite. A type-check is still to add.
- [x] The threat model re-read line by line against the implementation, with each mitigation
      pointed at the code or test that provides it — [threat-model.md](threat-model.md) §6, and
      `tests/test_threat_model.py` for the invariants a refactor could silently lose.

## 3. Deliberately not promised at first release

Stated up front so expectations are set: more than one provider or engine; multi-pool
management; per-client limits or fairness inside a pool; data-trust levels per host; advanced
bidding strategies and market replay. See [roadmap.md](roadmap.md) §1 and §3.
