---
name: Bug report
about: Something behaves differently than documented
labels: bug
---

**Please do not report security problems here** — see SECURITY.md.

## What happened, and what you expected

## To reproduce

Your `pool.yaml` with **secrets removed** (the pool only ever references them by environment
variable, so this should be safe), and the commands you ran.

## What the pool said

`gpm status`, and the relevant part of `gpm events` — the decision log carries the numbers
behind each decision, which is usually where the answer is.

## Versions

- `gpm --help` works from which version of `gpm-server`?
- Python, operating system, inference engine and version
- Provider, if renting was involved
