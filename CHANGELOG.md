# Changelog

Both packages follow [semantic versioning](https://semver.org). Three things carry their own
version numbers and are reported at runtime, because other people's code depends on them:

| Versioned separately | Where it is reported |
|---|---|
| The **app contract** | `X-GPM-Contract` on every response, and `GET /pool/status` |
| The **control API** | `GET /pool/status` |
| Each **plug-in interface** | `interface_version` on the provider, engine and strategy protocols |

## Unreleased

Nothing is published yet, so these are the versions inside the repository. **Each package's
version is raised in the same change that alters it** — a consumer cannot tell what it has
otherwise — and CI refuses a pull request that edits a package's source without raising it.

| Package | Version | What it carries |
|---|---|---|
| `gpm-server` | 0.3.0 | Dynamic allocation (D66), buffered delivery (D62), the host agent on rented hosts (D63), per-host worker adjustment (D67, D68), the machine history (D69), and the spending-path fixes D58–D65, D71, D73 |
| `gpm-client` | 0.2.0 | The SDK widens its time budget to the figure the pool publishes, because a held response makes "first byte" the end of the generation (D62) |
| `gpm-agent` | 0.2.0 | The heartbeat verb, each model loaded as its own download finishes (D57), and engine settings written where the pool installed it (D63) |

The **app contract stays at 1**: every dialect item added is optional, and no status code
changed meaning, so an application built against the first version still works untouched.


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
