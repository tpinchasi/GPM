# Feature list

> The planned features, one story each: what it is for, the evidence behind it, a design, what it
> gives up, and the owner's answers. **Four of the five are decided, and the fifth in part** and written into
> [spec/](../spec/); their pages remain as the reasoning and the build plan. **None is built.**
> Keep this table current: a story's status changes here first.

| # | Feature | Status | Depends on | Touches |
|---|---|---|---|---|
| [S1](S1-dynamic-resource-allocation.md) | **Dynamic resource allocation** — hosts added and given up from measured queue pressure and worker utilisation; explicitly enabled; machines still chosen by the configured rules; hosts added in a ramp — 1, 2, 4 … | **Decided (D66)**, not built | — (better with S5) | Supersedes "demand is the lease" for pools that enable it |
| [S2](S2-reliable-responses.md) | **Reliable responses** — responses from interruptible hosts are held until whole, so a lost host means a re-run, never half an answer; other host kinds keep streaming | **Decided (D62)**, not built | — | Amends the app contract's failover rule; new dialect items |
| [S3](S3-advisor-model-for-offers.md) | **An advisor model that picks the machine** — a small model chooses among offers the rules already accepted, from machine history; toggleable; falls back to the rules | **Partly decided (D69)**: history first, bounded declining, evaluations gate it. Open: how it reaches a model | — (richer with S4, S5) | Names a non-pure stage beside the strategy interface |
| [S4](S4-agent-on-rented-hosts.md) | **The host agent on rented hosts** — model sync and load, worker count, heartbeat, the machine's own facts | **Decided (D63)**, not built | — | Supersedes the recommendation in host-agent §8; **delivers D56 and D57** |
| [S5](S5-automatic-worker-adjustment.md) | **Automatic worker adjustment per host** — each host finds its own number from load, profile and latency while it serves; starts at six and climbs on evidence, bounded by what the machine can hold | **Decided (D67, D68)**, not built | S4 (its slots-only core does not) | Supersedes "step down on evidence, never up" and D56's operator-only rule; replaces the one-off parallelism measurement |

## Already specified, waiting on a story

| Decision | What | Delivered by |
|---|---|---|
| D56 | Change a rented host's worker count while it runs | S4 (the mechanism), S5 (doing it automatically) |
| D57 | Load each model as its own download finishes | S4 |

## Suggested order

1. **S2** — independent, the smallest, and it fixes a failure apps see today.
2. **S4** — unblocks D56, D57 and S5, and supplies the facts S3 and S5 want.
3. **S1** — independent of the rest; its shrink ordering improves once S5 exists.
4. **S5** — its core can start before S4 finishes.
5. **S3** — last: its history table can begin any time, but the advisor is only worth judging
   once S4 and S5 are feeding that history.

## How the five fit together

```
S4 agent on rented hosts ──► facts from the machine ──┬──► S5 workers per host ──► measured throughput ──┐
                                                       │                                                   ├──► S1 which host to give up
                                                       └──► S3 machine history ◄───────────────────────────┘
S2 reliable responses — independent; changes what "time to first byte" means for S1 and S5
```

## What every story holds to

The project's standing rules apply unchanged: nothing rents without a lease and a dollar cap;
caps are re-checked after every strategy; router and supervisor stay separate; the agent's verbs
stay a closed list; every test runs with no GPU and no cloud account; every new behaviour is
explicitly enabled and explains itself in the decision log.
