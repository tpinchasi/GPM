# Threat Model

> Written against the design, before any code. It must be re-checked against the implementation
> in the Release phase ([roadmap.md](roadmap.md) phase 4). Scope: one pool, one operator or a
> small team, the supervisor typically on a workstation or small server.

## 1. What is worth protecting

| Asset | Why it matters | Where it lives |
|---|---|---|
| **The provider account credential** | It can rent hardware without limit. The most damaging thing to lose | Supervisor's environment / a file on the supervisor's machine. **Never on a rented host, never in the browser, never in pool configuration** |
| **Money** | The pool spends unattended | Bounded by leases, caps and the dead-man timer |
| **The admin key** | Opens leases, raises ceilings, releases hosts | Operator's machine; held in console page memory for a tab's lifetime |
| **The app key** | Admits a client to the pool's compute | Each app's environment |
| **Prompts and completions** | May be sensitive to an adopter | In transit through the router; in memory on every host that serves them; **never written to the pool's request log** |
| **Availability** | Apps depend on the router | Router process |
| **Integrity of what is served** | An app asked for one model and must not silently get another | Catalog, variant resolution |

## 2. Who might attack, and from where

| Actor | Position |
|---|---|
| A **web page** open in the operator's browser | Can send requests to `127.0.0.1` and to private addresses |
| Another **user or process on the same machine** | Can reach loopback ports, read world-readable files |
| Someone on the **same network** as a non-loopback listener | Can reach the router or control API |
| The **operator of a rented host** | Controls the hardware: can read container memory, disk, environment, and all traffic once it arrives |
| A **compromised or malicious plug-in** | Runs inside the supervisor with its full authority |
| A **malicious model artifact or registry name** | Reaches hosts through a pull |
| A **holder of the app key** | Legitimate client behaving badly, or a leaked key |

## 3. Trust boundaries

```
 browser ──admin key──▶ control API ┐
                                    ├─ supervisor ──account credential──▶ provider API
 app ─────app key────▶ router ──────┘        │
                          │                  └─ SSH / HTTPS ──▶ rented host  ◀── its owner
                          └─ tunnel / http / https ──▶ local and fixed hosts
```

The rented host is **outside** the trust boundary even though the pool created it.

## 4. Threats and what the design does about them

| # | Threat | Mitigation in the design | Residual risk |
|---|---|---|---|
| **T1** | A web page drives the control API on loopback (CSRF, DNS rebinding) and spends money or destroys hosts | Admin key required on **every** control request, as a header, never a cookie; `Host` and `Origin` checked; no ambient authority | A page that can read the key from the console's memory — i.e. a browser or extension compromise |
| **T2** | A web page or local process uses the router on loopback for free compute, or to exhaust workers | App key required on every request, loopback included | Anyone who can read the app's environment has the key |
| **T3** | The app key is used to reach the control API | App and admin keys are separate and never interchangeable; the control API refuses the app key | — |
| **T4** | Keys captured on the network | Off loopback, the listener requires TLS; a bearer key over plain HTTP is refused at configuration load | Misconfigured TLS termination in front of the pool |
| **T5** | The account credential is stolen from a rented host | It is **never placed there**. The dead-man timer uses the provider's instance-scoped credential; a provider that has none gets no timer and only short leases. Nothing secret goes in instance-creation environment variables, which the host can read | The instance-scoped credential can stop or destroy that one instance — a nuisance, not a loss |
| **T6** | The pool's own app or admin key is exposed to a rented host | Neither is ever sent to a host; hosts are reached, they do not call back | — |
| **T7** | **The operator of a rented host reads prompts and completions** | **None in v1.** A tunnel or TLS protects the wire, not the machine. Host trust levels are a deferred requirement (D17) | **Accepted and documented:** a pool that includes marketplace hosts must not carry data its owner would not show the host's operator. The provider's "verified" flag raises the bar; it does not make a host trusted. Adopters with sensitive data should run a pool with no rented marketplace hosts until trust levels exist |
| **T8** | A rented host returns tampered output | None beyond corrupt-output detection, which targets faults, not adversaries | Accepted. An adopter needing integrity against a hostile host should not use marketplace hosts |
| **T9** | An app silently gets a different model than it asked for | Catalog-only resolution: only names an operator listed are ever substituted; the model actually served is reported on every response and logged | An operator's own catalog mistake |
| **T10** | A guessed or typo-squatted artifact is pulled onto a host | No pulls by convention or guess; pulls happen only for tags named in configuration, at prepare time, never triggered by a request. Engine images are pinned, never floating tags | A registry serving a different artifact under the same tag — outside the pool's control; pin by digest where the engine supports it |
| **T11** | Runaway spend — a bug, a bad strategy, a market spike | Nothing rents without a lease; leases carry mandatory dollar caps; caps enforced on the higher of estimated and provider-reported spend, less a margin; every bid clamped to two ceilings **after** the strategy returns; rate caps on hosts and hourly burn; one host at a time | Provider-side billing the provider does not report promptly |
| **T12** | The supervisor dies or its machine sleeps while hosts bill | Router keeps serving; dead-man timer on every rented host; lease expiry; orphan sweep at next start; parking is time-limited | A provider without an instance-scoped credential — mitigated by refusing long leases there |
| **T13** | Instances nobody knows about keep billing | Provider is the source of truth for existence; label-scoped listing; orphan sweep; every destroy verified by re-listing | Instances created outside the pool's label — out of scope |
| **T14** | A malicious plug-in exfiltrates the account credential or spends freely | Plug-ins load only if named in configuration. Strategies are pure functions with no I/O, and the supervisor re-checks every cap and ceiling after they return | **A provider or engine plug-in runs with the supervisor's full authority.** Installing one is a trust decision equal to installing the framework. Documented, not mitigated |
| **T15** | A holder of the app key exhausts the pool | Queue timeout, per-request deadlines, cancel on disconnect | No per-client limits inside a pool (deferred). Separate pools are the available control |
| **T16** | Sensitive text leaks through logs or the console | The request log records metadata only — never prompts or completions. Decision events carry numbers and identifiers. Secrets show in the console only as set / missing | Engine-side logs on hosts are the engine's and the host owner's |
| **T17** | An unauthenticated engine is exposed to the internet | Plain `http` to a public address without auth is refused at configuration load unless explicitly allowed per host; `tunnel` is the default for rented hosts and needs no exposed port | An operator who sets `allow_insecure` |
| **T18** | An SSH man-in-the-middle on first connection to a fresh rented host | Host keys are recorded on first connection and checked after; the address comes from the provider's authenticated API | Trust-on-first-use is weaker than a provider-attested host key, which providers generally do not offer |
| **T19** | Another local user reads keys or state | Key files, configuration referencing secrets, and the database are created owner-readable only; the pool refuses to start if they are group- or world-readable | A compromised operator account |

## 5. Deliberate non-goals

- Confidentiality or integrity against the owner of a host the pool does not control (T7, T8).
- Multi-tenant isolation inside one pool (T15).
- Sandboxing of plug-ins (T14).

Each is stated in user-facing documentation, not only here.

## 6. Reporting

The public repository carries a security policy with a private reporting channel
([release-checklist.md](release-checklist.md)). A finding that lets anyone spend an operator's
money, or obtain the account credential, is treated as critical regardless of how it is reached.
