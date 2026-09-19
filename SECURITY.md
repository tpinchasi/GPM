# Security policy

GPM spends money and holds credentials that can rent hardware. We treat findings in that area
as critical regardless of how they are reached.

## Reporting a vulnerability

**Please do not open a public issue.** Report privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository, or by email to the address in the repository's profile.

Tell us what you did, what happened, and what you expected. A proof of concept helps but is not
required. We will acknowledge within three working days and keep you informed until it is
resolved. If you would like credit in the advisory, say so.

## What counts as critical

Anything that lets someone other than the operator:

- **spend the operator's money** — rent hardware, raise a bid, bypass a lease, defeat a dollar
  cap or the cap safety margin;
- **obtain the provider account credential**, in any form: a log line, an event, an API
  response, a rented host, a browser;
- **reach the control API without the admin key**, or use an app key to do anything the control
  API permits;
- **serve inference without the app key**, on loopback included;
- **make a request trigger a model pull or a model load**, or be served a model the operator
  did not list in the catalog.

## What is out of scope, by design

These are stated as deliberate non-goals in [docs/threat-model.md](docs/threat-model.md), not
oversights:

- **The operator of a rented marketplace host can read everything sent to it.** A tunnel or TLS
  protects the wire, not the machine. A pool that includes marketplace hosts must not carry data
  its owner would not show that host's operator. Host trust levels are future work.
- **A rented host can return whatever it likes.** Corrupt-output detection targets faults, not
  adversaries.
- **Plug-ins are not sandboxed.** A provider or engine plug-in runs inside the supervisor with
  its full authority, including the account credential. Installing one is a trust decision equal
  to installing GPM itself. Only plug-ins named in configuration are ever loaded.
- **There is no isolation between apps sharing one pool's key.** Separate workloads use separate
  pools.

## Supported versions

Until 1.0, only the latest release receives fixes.
