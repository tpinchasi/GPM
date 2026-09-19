# Changelog

Both packages follow [semantic versioning](https://semver.org). Three things carry their own
version numbers and are reported at runtime, because other people's code depends on them:

| Versioned separately | Where it is reported |
|---|---|
| The **app contract** | `X-GPM-Contract` on every response, and `GET /pool/status` |
| The **control API** | `GET /pool/status` |
| Each **plug-in interface** | `interface_version` on the provider, engine and strategy protocols |

## Unreleased

The first release is being prepared; see `docs/release-checklist.md` for what remains.

### Added

- **The router and the client SDK** over statically configured hosts, across three transports
  (`http`, `https`, and a supervised SSH `tunnel`). Requests pass through in the engine's own
  API: streaming, tool calls, structured output and cancel-on-disconnect behave the same
  through the pool as directly against the engine.
- **The supervisor**: leases with mandatory dollar caps, overflow-driven renting one host at a
  time, `floor_plus_premium` bidding under two ceilings, eviction recovery, idle release,
  parking with reuse, the orphan sweep, verified destroy, and spend reconciled against the
  provider's own reported charges.
- **A dead-man timer on every rented host**, armed before anything else in its start-up script
  and using the provider's instance-scoped credential, so a host ends itself if the pool goes
  silent.
- **The operator console** at `/ui`: hosts by tier, lease burn-down, the live market through
  your own offer policy, every decision with the numbers behind it, and configuration with
  validate → plan → apply, history and rollback.
- **Two provider plug-ins** (a scriptable fake and Vast.ai) and one engine plug-in (Ollama),
  loaded through the same entry-point mechanism a third party's would use.

### Known limits at this point

- One engine and one marketplace provider.
- No multi-pool management, no per-client limits inside a pool, and no data-trust levels per
  host — so **the operator of a rented marketplace host can read everything sent to it**.
- Advanced bidding strategies and market replay are deliberately not included; see
  `docs/roadmap.md` §1.
