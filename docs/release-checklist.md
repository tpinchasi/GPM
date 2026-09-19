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
- [ ] First-party provider and engine plug-ins load through the same entry-point mechanism as
      third-party ones — no privileged path.
- [ ] Semantic versioning for both packages. The app contract, the control API and each plug-in
      interface carry their own version numbers, reported at runtime.
- [ ] Pinned, reproducible builds; a published changelog.

### 2.2 Safe defaults, verified against the implementation

- [ ] Fresh install binds loopback only; refuses a non-loopback listener without TLS.
- [ ] No request is served without the app key; no control call without the admin key; the two
      are never interchangeable.
- [ ] Nothing rents without a lease; a lease that can rent cannot be created without a dollar cap.
- [ ] Plain-`http` public hosts without auth are refused unless explicitly allowed per host.
- [ ] Only catalogued models are ever resolved; no request can trigger a pull or a model load.
- [ ] Key files, secret-bearing configuration and the database are owner-readable only, and the
      pool refuses to start otherwise.
- [ ] The account credential appears in no log, no event, no API response, no rented host.

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
- [ ] A second guide that adds a rented provider, written around leases and caps first.
- [ ] Plug-in author guides for provider, engine and strategy, each with the interface's
      guarantees and the fake provider as the worked example.
- [ ] The **deliberate non-goals** from [threat-model.md](threat-model.md) stated in user-facing
      docs — above all that a rented marketplace host's operator can read everything sent to it.
- [ ] [decisions.md](decisions.md) published: it answers most "why doesn't it just…" questions.

### 2.5 Project hygiene

- [x] `SECURITY.md` with a private reporting channel and a statement of what counts as critical.
- [x] `CONTRIBUTING.md` with the DCO. A code of conduct and issue/PR templates are still to do.
- [x] CI: tests on three Python versions, a secret scan, a dependency audit, and a job that
      fails if anything able to reach a real provider enters the default suite. Lint and
      type-check are still to add.
- [ ] The threat model re-read line by line against the implementation, with each mitigation
      pointed at the code or test that provides it.

## 3. Deliberately not promised at first release

Stated up front so expectations are set: more than one provider or engine; multi-pool
management; per-client limits or fairness inside a pool; data-trust levels per host; advanced
bidding strategies and market replay. See [roadmap.md](roadmap.md) §1 and §3.
